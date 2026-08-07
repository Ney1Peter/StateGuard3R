"""Independent pinned recurrent loop for v5 current-pointmap consensus export."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from .recal3r_online_quarantine_v2 import OnlineQuarantineStateError, QuarantineWatchdog, StructuralPreState
from .recal3r_state_v3 import verify_pinned_lighter_source
from .recal3r_structural_witness_v3 import StructuralIdentityWitness, StructuralWitnessError


class CurrentPointmapRunnerError(RuntimeError):
    """Raised when v5's isolated recurrent execution is unsafe."""


class CurrentPointmapObserver(Protocol):
    def observe(self, frame_id: int, prediction: Mapping[str, Any], model: Any, frame_context: Any) -> Any: ...
    def alarm(self, observation: Any) -> bool: ...
    def finalize(self, observation: Any, *, quarantined: bool, prediction: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]: ...
    def timeline_evidence(self, observation: Any) -> Mapping[str, Any]: ...


def _reset_requested(value: Any) -> bool:
    if value is None:
        return False
    current = value.detach() if callable(getattr(value, "detach", None)) else value
    current = current.any() if callable(getattr(current, "any", None)) else current
    return bool(current.item() if callable(getattr(current, "item", None)) else current)


@dataclass(frozen=True)
class CurrentPointmapRunnerResult:
    predictions: list[Mapping[str, Any]]
    views: Sequence[Mapping[str, Any]]
    timeline: list[Mapping[str, Any]]
    source_provenance: Mapping[str, Any]
    recurrent_policy_runtime_seconds: float


def run_current_pointmap_recurrent_lighter(
    views: Sequence[Mapping[str, Any]], model: Any, device: Any, *, torch: Any,
    to_gpu: Callable[[Mapping[str, Any], Any], Mapping[str, Any]], to_cpu: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    canonicalize_model_update_type: Callable[[Any], str], before_frame: Callable[[int, Mapping[str, Any]], Any] | None = None,
    observer: CurrentPointmapObserver | None = None, watchdog_limit: int = 8, verify_source: bool = True,
    synchronize: Callable[[], None] | None = None,
) -> CurrentPointmapRunnerResult:
    """Run the pinned lighter body with v5 rollback and consensus-export hooks."""
    if not views:
        raise CurrentPointmapRunnerError("views must not be empty")
    source_provenance = verify_pinned_lighter_source() if verify_source else {}
    watchdog = QuarantineWatchdog(watchdog_limit) if observer is not None else None
    model.config.model_update_type = canonicalize_model_update_type(getattr(model.config, "model_update_type", "cut3r"))
    update_type = model.config.model_update_type
    predictions: list[Mapping[str, Any]] = []
    timeline: list[Mapping[str, Any]] = []
    elapsed = 0.0
    reset_mask: Any = False
    if model._uses_update_pressure_update() and hasattr(model, "update_pressure"):
        del model.update_pressure

    for i, raw_view in enumerate(views):
        if i and _reset_requested(raw_view.get("reset")):
            raise CurrentPointmapRunnerError("v5 recovery fails closed at a non-initial reset boundary")
        frame_context = before_frame(i, raw_view) if before_frame is not None else None
        # The public v1 baseline excludes common RGB-overlap work. Synchronize
        # here so the v5 recurrent/policy interval remains comparable.
        if synchronize is not None:
            synchronize()
        started = time.perf_counter()
        view = to_gpu(raw_view, device)
        device = view["img"].device
        batch_size = view["img"].shape[0]
        img_mask = view["img_mask"].reshape(-1, batch_size)
        ray_mask = view["ray_mask"].reshape(-1, batch_size)
        imgs = view["img"].unsqueeze(0).view(-1, *view["img"].shape[1:])
        ray_maps = view["ray_map"].unsqueeze(0)
        shapes = view["true_shape"].unsqueeze(0) if "true_shape" in view else torch.tensor(view["img"].shape[-2:], device=device).unsqueeze(0).repeat(batch_size, 1).unsqueeze(0)
        shapes = shapes.view(-1, 2).to(imgs.device)
        selected_imgs, selected_shapes = imgs[img_mask.view(-1)], shapes[img_mask.view(-1)]
        img_out, img_pos = (model._encode_image(selected_imgs, selected_shapes)[0:2] if selected_imgs.size(0) else (None, None))
        ray_maps = ray_maps.view(-1, *ray_maps.shape[2:]).permute(0, 3, 1, 2)
        selected_rays, selected_ray_shapes = ray_maps[ray_mask.view(-1)], shapes[ray_mask.view(-1)]
        ray_out, ray_pos = (model._encode_ray_map(selected_rays, selected_ray_shapes)[0:2] if selected_rays.size(0) else (None, None))
        if img_out is not None and ray_out is None:
            feat_i, pos_i = img_out[-1], img_pos
        elif img_out is None and ray_out is not None:
            feat_i, pos_i = ray_out[-1], ray_pos
        elif img_out is not None and ray_out is not None:
            feat_i, pos_i = img_out[-1] + ray_out[-1], img_pos
        else:
            raise CurrentPointmapRunnerError("ReCal3R frame has neither image nor raymap features")
        if i == 0:
            state_feat, state_pos = model._init_state(feat_i, pos_i)
            model._init_recal3r_reference_state(state_feat)
            mem = model.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
            init_state_feat, init_mem = state_feat.clone(), mem.clone()
        pre = StructuralPreState.capture(state_feat, state_pos, init_state_feat, mem, init_mem, reset_mask, model, torch=torch) if observer is not None and i else None
        witness = StructuralIdentityWitness.capture(pre, model) if pre is not None else None
        global_img_feat_i = model._get_img_level_feat(feat_i) if model.pose_head_flag else None
        if model.pose_head_flag:
            pose_feat_i = model.pose_token.expand(feat_i.shape[0], -1, -1) if i == 0 or reset_mask else model.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype)
        else:
            pose_feat_i, pose_pos_i = None, None
        new_state, dec, self_state, cross_state, self_img, cross_img = model._recurrent_rollout(state_feat, state_pos, feat_i, pos_i, pose_feat_i, pose_pos_i, init_state_feat, img_mask=view["img_mask"], reset_mask=view["reset"], update=view.get("update"), return_attn=True)
        del self_state, self_img, cross_img
        if len(dec) != model.dec_depth + 1:
            raise CurrentPointmapRunnerError("pinned ReCal3R decoder depth changed")
        new_mem = model.pose_retriever.update_mem(mem, global_img_feat_i, dec[-1][:, 0:1])
        head_input = [dec[0].float(), dec[model.dec_depth * 2 // 4][:, 1:].float(), dec[model.dec_depth * 3 // 4][:, 1:].float(), dec[model.dec_depth].float()]
        res = model._downstream_head(head_input, shapes, pos=pos_i)
        update = view.get("update")
        update_mask = (view["img_mask"] & update if update is not None else view["img_mask"])[:, None, None].float()
        previous_state = state_feat
        if i == 0 or reset_mask or update_type == "cut3r":
            update_mask1 = update_mask
        elif update_type == "ttt3r":
            update_mask1, _, _ = model._compute_state_update_mask(update_mask, cross_state)
        else:
            update_mask1 = model._compute_recal3r_update_mask(update_mask, cross_state, dec, prev_state_feat=state_feat)
        state_feat = new_state * update_mask1 + state_feat * (1 - update_mask1)
        mem = new_mem * update_mask + mem * (1 - update_mask)
        model._advance_recal3r_sequence_age(state_feat)
        model._maybe_record_u_calibration_step(i, previous_state, state_feat)
        reset_mask = view["reset"]
        if reset_mask is not None:
            model._reset_update_pressure_if_needed(reset_mask)
            model._reset_recal3r_reference_state_if_needed(reset_mask, init_state_feat)
            reset_mask = reset_mask[:, None, None].float()
            state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
            mem = init_mem * reset_mask + mem * (1 - reset_mask)
        observation = observer.observe(i, res, model, frame_context) if observer is not None else None
        quarantine = bool(observer.alarm(observation)) if observer is not None and i else False
        consecutive = watchdog.record(quarantine=quarantine) if watchdog is not None else 0
        action, reason, restore_evidence = "commit", ("always_commit_control" if observer is None else "initial_state_must_commit" if i == 0 else "current_online_detector_clear"), None
        if quarantine:
            if pre is None or witness is None:
                raise CurrentPointmapRunnerError("v5 quarantine has no pre-state witness")
            try:
                witness.reject_in_place_mutation(pre)
                restored = pre.restore(model, torch=torch)
                witness.verify_restored(restored, model)
                state_feat, state_pos, init_state_feat, mem, init_mem, reset_mask = restored
            except (OnlineQuarantineStateError, StructuralWitnessError) as error:
                raise CurrentPointmapRunnerError(f"v5 restore failed: {error}") from error
            action, reason, restore_evidence = "quarantine_current_rollback", "current_online_detector_alarm", witness.timeline_evidence()
        raw_prediction = to_cpu(res)
        if observer is None:
            exported, export_evidence = raw_prediction, {"export_action": "export_real_camera_pose", "anchor_frame_ids": None}
        else:
            exported, export_evidence = observer.finalize(observation, quarantined=quarantine, prediction=raw_prediction)
        predictions.append(exported)
        evidence = dict(observer.timeline_evidence(observation)) if observer is not None else {}
        timeline.append({"frame_id": i, "action": action, "reason": reason, "current_alarm": quarantine, "consecutive_rollbacks": consecutive, "pending_transaction_count": 0, "restore_witness": restore_evidence, **dict(export_evidence), **evidence})
        if synchronize is not None:
            synchronize()
        elapsed += time.perf_counter() - started
    return CurrentPointmapRunnerResult(predictions, views, timeline, source_provenance, elapsed)


__all__ = ["CurrentPointmapObserver", "CurrentPointmapRunnerError", "CurrentPointmapRunnerResult", "run_current_pointmap_recurrent_lighter"]
