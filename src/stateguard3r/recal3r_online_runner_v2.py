"""Hash-bound recurrent loop for Detector-v3 current-frame quarantine.

This is a minimal external derivative of pinned ReCal3R
``forward_recurrent_lighter``.  Unlike the v1 transaction wrapper it has no
pending/replay state and no post-frame snapshot on the normal path.  A caller
supplies a current-frame observer; only an observed alarm restores the
reference-only pre-frame closure captured immediately before that candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from .recal3r_online_quarantine_v2 import (
    OnlineQuarantineStateError,
    QuarantineWatchdog,
    StructuralPreState,
    current_state_digest,
)
from .recal3r_state_v3 import verify_pinned_lighter_source


class OnlineQuarantineRunnerError(RuntimeError):
    """Raised when the v2 external recurrent execution is invalid."""


class CandidateObserver(Protocol):
    def observe(self, frame_id: int, prediction: Mapping[str, Any], model: Any, frame_context: Any) -> Any: ...

    def alarm(self, observation: Any) -> bool: ...

    def finalize(self, observation: Any, *, quarantined: bool) -> None: ...

    def timeline_evidence(self, observation: Any) -> Mapping[str, Any]: ...


def _reset_requested(value: Any) -> bool:
    if value is None:
        return False
    detach = getattr(value, "detach", None)
    current = detach() if callable(detach) else value
    any_method = getattr(current, "any", None)
    if callable(any_method):
        current = any_method()
    item = getattr(current, "item", None)
    return bool(item() if callable(item) else current)


@dataclass(frozen=True)
class OnlineQuarantineRunnerResult:
    predictions: list[Mapping[str, Any]]
    views: Sequence[Mapping[str, Any]]
    timeline: list[Mapping[str, Any]]
    source_provenance: Mapping[str, Any]
    recurrent_policy_runtime_seconds: float


def run_online_quarantine_recurrent_lighter(
    views: Sequence[Mapping[str, Any]],
    model: Any,
    device: Any,
    *,
    torch: Any,
    to_gpu: Callable[[Mapping[str, Any], Any], Mapping[str, Any]],
    to_cpu: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    canonicalize_model_update_type: Callable[[Any], str],
    before_frame: Callable[[int, Mapping[str, Any]], Any] | None = None,
    observer: CandidateObserver | None = None,
    watchdog_limit: int = 8,
    verify_source: bool = True,
) -> OnlineQuarantineRunnerResult:
    """Execute the pinned lighter body with optional same-frame quarantine."""

    if not views:
        raise OnlineQuarantineRunnerError("views must not be empty")
    source_provenance = verify_pinned_lighter_source() if verify_source else {}
    watchdog = QuarantineWatchdog(watchdog_limit) if observer is not None else None
    model.config.model_update_type = canonicalize_model_update_type(getattr(model.config, "model_update_type", "cut3r"))
    model_update_type = model.config.model_update_type
    predictions: list[Mapping[str, Any]] = []
    timeline: list[Mapping[str, Any]] = []
    recurrent_policy_runtime_seconds = 0.0
    reset_mask: Any = False
    if model._uses_update_pressure_update() and hasattr(model, "update_pressure"):
        del model.update_pressure

    for i, raw_view in enumerate(views):
        if i > 0 and _reset_requested(raw_view.get("reset")):
            raise OnlineQuarantineRunnerError("online quarantine fails closed at a non-initial reset boundary")
        # RGB-overlap extraction is common detector instrumentation and was
        # precomputed outside ``runtime_seconds`` in the frozen baseline.  It
        # remains causal here, but is intentionally outside the comparable
        # recurrent/policy scope.  Current health scoring and rollback stay in
        # the timed scope below because they are candidate-specific policy work.
        frame_context = before_frame(i, raw_view) if before_frame is not None else None
        recurrent_started = time.perf_counter()
        view = to_gpu(raw_view, device)
        device = view["img"].device
        batch_size = view["img"].shape[0]
        img_mask = view["img_mask"].reshape(-1, batch_size)
        ray_mask = view["ray_mask"].reshape(-1, batch_size)
        imgs = view["img"].unsqueeze(0)
        ray_maps = view["ray_map"].unsqueeze(0)
        shapes = (
            view["true_shape"].unsqueeze(0)
            if "true_shape" in view
            else torch.tensor(view["img"].shape[-2:], device=device).unsqueeze(0).repeat(batch_size, 1).unsqueeze(0)
        )
        imgs = imgs.view(-1, *imgs.shape[2:])
        ray_maps = ray_maps.view(-1, *ray_maps.shape[2:])
        shapes = shapes.view(-1, 2).to(imgs.device)
        selected_imgs = imgs[img_mask.view(-1)]
        selected_shapes = shapes[img_mask.view(-1)]
        if selected_imgs.size(0) > 0:
            img_out, img_pos, _ = model._encode_image(selected_imgs, selected_shapes)
        else:
            img_out, img_pos = None, None
        ray_maps = ray_maps.permute(0, 3, 1, 2)
        selected_ray_maps = ray_maps[ray_mask.view(-1)]
        selected_shapes_ray = shapes[ray_mask.view(-1)]
        if selected_ray_maps.size(0) > 0:
            ray_out, ray_pos, _ = model._encode_ray_map(selected_ray_maps, selected_shapes_ray)
        else:
            ray_out, ray_pos = None, None
        if img_out is not None and ray_out is None:
            feat_i, pos_i = img_out[-1], img_pos
        elif img_out is None and ray_out is not None:
            feat_i, pos_i = ray_out[-1], ray_pos
        elif img_out is not None and ray_out is not None:
            feat_i, pos_i = img_out[-1] + ray_out[-1], img_pos
        else:
            raise OnlineQuarantineRunnerError("ReCal3R frame has neither image nor raymap features")

        if i == 0:
            state_feat, state_pos = model._init_state(feat_i, pos_i)
            model._init_recal3r_reference_state(state_feat)
            mem = model.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
            init_state_feat = state_feat.clone()
            init_mem = mem.clone()

        pre = (
            StructuralPreState.capture(state_feat, state_pos, init_state_feat, mem, init_mem, reset_mask, model, torch=torch)
            if observer is not None and i > 0
            else None
        )
        if model.pose_head_flag:
            global_img_feat_i = model._get_img_level_feat(feat_i)
            if i == 0 or reset_mask:
                pose_feat_i = model.pose_token.expand(feat_i.shape[0], -1, -1)
            else:
                pose_feat_i = model.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype)
        else:
            pose_feat_i, pose_pos_i = None, None
        new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = model._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            init_state_feat,
            img_mask=view["img_mask"],
            reset_mask=view["reset"],
            update=view.get("update", None),
            return_attn=True,
        )
        del self_attn_state, self_attn_img, cross_attn_img
        out_pose_feat_i = dec[-1][:, 0:1]
        new_mem = model.pose_retriever.update_mem(mem, global_img_feat_i, out_pose_feat_i)
        if len(dec) != model.dec_depth + 1:
            raise OnlineQuarantineRunnerError("pinned ReCal3R decoder depth changed")
        head_input = [
            dec[0].float(),
            dec[model.dec_depth * 2 // 4][:, 1:].float(),
            dec[model.dec_depth * 3 // 4][:, 1:].float(),
            dec[model.dec_depth].float(),
        ]
        res = model._downstream_head(head_input, shapes, pos=pos_i)
        update = view.get("update", None)
        update_mask = view["img_mask"] & update if update is not None else view["img_mask"]
        update_mask = update_mask[:, None, None].float()
        prev_state_feat = state_feat
        if i == 0 or reset_mask:
            update_mask1 = update_mask
        elif model_update_type == "cut3r":
            update_mask1 = update_mask
        elif model_update_type == "ttt3r":
            update_mask1, _, _ = model._compute_state_update_mask(update_mask, cross_attn_state)
        else:
            update_mask1 = model._compute_recal3r_update_mask(update_mask, cross_attn_state, dec, prev_state_feat=state_feat)
        state_feat = new_state_feat * update_mask1 + state_feat * (1 - update_mask1)
        mem = new_mem * update_mask + mem * (1 - update_mask)
        model._advance_recal3r_sequence_age(state_feat)
        model._maybe_record_u_calibration_step(i, prev_state_feat, state_feat)
        res_cpu = to_cpu(res)
        predictions.append(res_cpu)
        reset_mask = view["reset"]
        if reset_mask is not None:
            model._reset_update_pressure_if_needed(reset_mask)
            model._reset_recal3r_reference_state_if_needed(reset_mask, init_state_feat)
            reset_mask = reset_mask[:, None, None].float()
            state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
            mem = init_mem * reset_mask + mem * (1 - reset_mask)

        observation = observer.observe(i, res_cpu, model, frame_context) if observer is not None else None
        should_quarantine = bool(observer.alarm(observation)) if observer is not None and i > 0 else False
        if watchdog is not None:
            consecutive = watchdog.record(quarantine=should_quarantine)
        else:
            consecutive = 0
        action = "commit"
        reason = "always_commit_control" if observer is None else "initial_state_must_commit" if i == 0 else "current_online_detector_clear"
        proposed_digest = pre_digest = committed_digest = None
        if should_quarantine:
            if pre is None:
                raise OnlineQuarantineRunnerError("current-frame quarantine has no pre-state snapshot")
            try:
                proposed_digest = current_state_digest(state_feat, state_pos, init_state_feat, mem, init_mem, reset_mask, model)
                state_feat, state_pos, init_state_feat, mem, init_mem, reset_mask = pre.restore(model, torch=torch)
                pre_digest = current_state_digest(state_feat, state_pos, init_state_feat, mem, init_mem, reset_mask, model)
                committed_digest = current_state_digest(state_feat, state_pos, init_state_feat, mem, init_mem, reset_mask, model)
            except OnlineQuarantineStateError as error:
                raise OnlineQuarantineRunnerError(f"online quarantine restore failed: {error}") from error
            if pre_digest != committed_digest:
                raise OnlineQuarantineRunnerError("online quarantine restore digest differs from committed pre-state")
            action = "quarantine_current_rollback"
            reason = "current_online_detector_alarm"
        if observer is not None:
            observer.finalize(observation, quarantined=should_quarantine)
        evidence = dict(observer.timeline_evidence(observation)) if observer is not None else {}
        timeline.append(
            {
                "frame_id": i,
                "action": action,
                "reason": reason,
                "current_alarm": bool(should_quarantine),
                "consecutive_rollbacks": consecutive,
                "pending_transaction_count": 0,
                "proposal_digest_sha256": proposed_digest,
                "pre_state_digest_sha256": pre_digest,
                "proposed_state_digest_sha256": proposed_digest,
                "committed_state_digest_sha256": committed_digest,
                **evidence,
            }
        )
        recurrent_policy_runtime_seconds += time.perf_counter() - recurrent_started

    return OnlineQuarantineRunnerResult(
        predictions=predictions,
        views=views,
        timeline=timeline,
        source_provenance=source_provenance,
        recurrent_policy_runtime_seconds=recurrent_policy_runtime_seconds,
    )


__all__ = [
    "CandidateObserver",
    "OnlineQuarantineRunnerError",
    "OnlineQuarantineRunnerResult",
    "run_online_quarantine_recurrent_lighter",
]
