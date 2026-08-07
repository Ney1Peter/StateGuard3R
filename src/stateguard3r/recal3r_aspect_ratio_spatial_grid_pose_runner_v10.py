"""Deprecated, preserved v10 aspect-ratio spatial-grid runner.

PRESERVED BUT DISABLED FOR EXPERIMENTAL USE (DO NOT RUN): v10 only relaxed
v9's observed spatial-token count while retaining the same post-rollout
``dec[0]`` mean mechanism. The preimplementation NO-GO records why that is not
a mechanism-distinct successor. This module remains parseable and is kept
intact for audit and regression tests; no CLI or formal dispatcher may invoke
it. New recovery work must use the separately preregistered v11 route.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from .recal3r_online_quarantine_v2 import OnlineQuarantineStateError, QuarantineWatchdog, StructuralPreState
from .recal3r_state_v3 import verify_pinned_lighter_source
from .recal3r_structural_witness_v3 import StructuralIdentityWitness, StructuralWitnessError


class EarlySpatialGridPoseRunnerError(RuntimeError):
    """Raised when the preserved, non-authorized v10 runner is unsafe."""


class EarlySpatialGridPoseObserver(Protocol):
    def observe(self, frame_id: int, prediction: Mapping[str, Any], model: Any, frame_context: Any, early_spatial_grid_pose_token: Any) -> Any: ...
    def alarm(self, observation: Any) -> bool: ...
    def finalize(self, observation: Any, *, quarantined: bool, prediction: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]: ...
    def timeline_evidence(self, observation: Any) -> Mapping[str, Any]: ...


def _reset_requested(value: Any) -> bool:
    if value is None:
        return False
    current = value.detach() if callable(getattr(value, "detach", None)) else value
    current = current.any() if callable(getattr(current, "any", None)) else current
    return bool(current.item() if callable(getattr(current, "item", None)) else current)


def audit_v10_runner_contract(runner_path: Path) -> Mapping[str, Any]:
    """Fail closed unless v10 retains its exact candidate-only spatial mean boundary."""
    source = runner_path.read_text(encoding="utf-8")
    try:
        body = source[source.index("\ndef run_early_spatial_grid_pose_recurrent_lighter(") + 1:]
    except ValueError as error:
        raise EarlySpatialGridPoseRunnerError("v10 runner entry point is missing") from error
    rollout = "model._recurrent_rollout("
    capture = "early_spatial_grid_pose_token = model.decoder_embed(dec[0].mean(dim=1, keepdim=True)).detach().clone()"
    update = "model.pose_retriever.update_mem("
    finalize = "observer.finalize(observation, quarantined=quarantine, prediction=res)"
    candidate_transfer = "predictions.append(to_cpu(exported))"
    control_transfer = "predictions.append(to_cpu(res))"
    try:
        rollout_at = body.index(rollout)
        guard_at = body.index('tuple(getattr(dec[0], "shape", ())) != (1, 768, 1024)', rollout_at)
        candidate_at = body.index("else:\n            observation = observer.observe", guard_at)
        capture_at = body.index(capture, guard_at, candidate_at)
        update_at = body.index(update, capture_at)
        control_at = body.index("if observer is None:", update_at)
        candidate_at = body.index("else:\n            observation = observer.observe", control_at)
        finalize_at = body.index(finalize, candidate_at)
        candidate_transfer_at = body.index(candidate_transfer, finalize_at)
        control_transfer_at = body.index(control_transfer, control_at, candidate_at)
    except ValueError as error:
        raise EarlySpatialGridPoseRunnerError("v10 runner source contract changed") from error
    if not (rollout_at < guard_at < capture_at < update_at):
        raise EarlySpatialGridPoseRunnerError("v10 spatial capture is not after rollout before memory update")
    if not (candidate_at < finalize_at < candidate_transfer_at) or control_transfer_at < control_at:
        raise EarlySpatialGridPoseRunnerError("v10 export/control transfer order changed")
    control = body[control_at:candidate_at]
    if any(marker in control for marker in ("early_spatial_grid_pose_token =", "decoder_embed(", ".mean(", "observer.")):
        raise EarlySpatialGridPoseRunnerError("v10 always-control path is contaminated by candidate policy work")
    previous_recovery_markers = (
        "early_decoder_pose_runner_v8", "safe_anchor_runner_v3", "geometric_registration_runner_v4",
        "current_pointmap_runner_v5", "prerollout_pose_query_runner_v6", "anchor_export_v3",
        "registration_export_v4", "pointmap_consensus_export_v5", "pose_query_export_v6",
    )
    if any(marker in body for marker in previous_recovery_markers):
        raise EarlySpatialGridPoseRunnerError("v10 runner delegates to a previous recovery mechanism")
    return {
        "runner_path": str(runner_path.resolve(strict=True)),
        "runner_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "early_decoder_layer_index": 0,
        "preprojection_spatial_token_count": 768,
        "preprojection_width": 1024,
        "projected_width": 768,
        "aggregation": "mean",
        "frozen_decoder_projection": "model.decoder_embed",
        "capture_relation": "after_rollout_before_update_mem",
        "candidate_only_capture": True,
        "always_control_uses_legacy_raw_transfer": True,
        "candidate_finalize_before_cpu_transfer": True,
        "previous_recovery_delegation": False,
    }


@dataclass(frozen=True)
class EarlySpatialGridPoseRunnerResult:
    predictions: list[Mapping[str, Any]]
    views: Sequence[Mapping[str, Any]]
    timeline: list[Mapping[str, Any]]
    source_provenance: Mapping[str, Any]
    recurrent_policy_runtime_seconds: float


# PRESERVED / DISABLED (DO NOT RUN): historical v10 code only.  No launcher,
# Gate, or new experiment may call this count-only v9 variant; keep it only for
# source-integrity regression coverage and the retained NO-GO record.
def run_early_spatial_grid_pose_recurrent_lighter(
    views: Sequence[Mapping[str, Any]], model: Any, device: Any, *, torch: Any,
    to_gpu: Callable[[Mapping[str, Any], Any], Mapping[str, Any]], to_cpu: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    canonicalize_model_update_type: Callable[[Any], str], before_frame: Callable[[int, Mapping[str, Any]], Any] | None = None,
    observer: EarlySpatialGridPoseObserver | None = None, watchdog_limit: int = 8, verify_source: bool = True,
    synchronize: Callable[[], None] | None = None,
) -> EarlySpatialGridPoseRunnerResult:
    """Run the pinned lighter body with an isolated v10 post-rollout export hook."""
    if not views:
        raise EarlySpatialGridPoseRunnerError("views must not be empty")
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
            raise EarlySpatialGridPoseRunnerError("v10 recovery fails closed at a non-initial reset boundary")
        frame_context = before_frame(i, raw_view) if before_frame is not None else None
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
            raise EarlySpatialGridPoseRunnerError("ReCal3R frame has neither image nor raymap features")
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
            raise EarlySpatialGridPoseRunnerError("pinned ReCal3R decoder depth changed")
        # Candidate only: this shape guard and pool must never run in the
        # always-control, whose recurrent raw path remains byte-equivalent.
        early_spatial_grid_pose_token = None
        if observer is not None:
            if not bool(torch.is_tensor(dec[0])) or tuple(getattr(dec[0], "shape", ())) != (1, 768, 1024):
                raise EarlySpatialGridPoseRunnerError("pinned aspect-ratio spatial grid shape changed")
            if not bool(torch.is_floating_point(dec[0])) or not bool(torch.isfinite(dec[0]).all()):
                raise EarlySpatialGridPoseRunnerError("pinned aspect-ratio spatial grid is nonfinite")
            projection = getattr(model, "decoder_embed", None)
            if type(projection) is not torch.nn.Linear or projection.in_features != 1024 or projection.out_features != 768:
                raise EarlySpatialGridPoseRunnerError("pinned frozen decoder projection changed")
            if projection.weight.device != dec[0].device or projection.weight.dtype != dec[0].dtype:
                raise EarlySpatialGridPoseRunnerError("pinned frozen decoder projection device or dtype changed")
            try:
                early_spatial_grid_pose_token = model.decoder_embed(dec[0].mean(dim=1, keepdim=True)).detach().clone()
            except Exception as error:
                raise EarlySpatialGridPoseRunnerError(f"pinned early spatial projection failed: {error}") from error
            if not bool(torch.is_tensor(early_spatial_grid_pose_token)) or tuple(getattr(early_spatial_grid_pose_token, "shape", ())) != (1, 1, 768) or not bool(torch.is_floating_point(early_spatial_grid_pose_token)) or not bool(torch.isfinite(early_spatial_grid_pose_token).all()):
                raise EarlySpatialGridPoseRunnerError("pinned early spatial pooled token is unavailable")
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
        if observer is None:
            predictions.append(to_cpu(res))
            timeline.append({"frame_id": i, "action": "commit", "reason": "always_commit_control", "current_alarm": False, "consecutive_rollbacks": 0, "pending_transaction_count": 0, "restore_witness": None, "export_action": "export_real_camera_pose", "anchor_frame_ids": None})
        else:
            observation = observer.observe(i, res, model, frame_context, early_spatial_grid_pose_token)
            quarantine = bool(observer.alarm(observation)) if i else False
            consecutive = watchdog.record(quarantine=quarantine)
            action, reason, restore_evidence = "commit", ("initial_state_must_commit" if i == 0 else "current_online_detector_clear"), None
            if quarantine:
                if pre is None or witness is None:
                    raise EarlySpatialGridPoseRunnerError("v10 quarantine has no pre-state witness")
                try:
                    witness.reject_in_place_mutation(pre)
                    restored = pre.restore(model, torch=torch)
                    witness.verify_restored(restored, model)
                    state_feat, state_pos, init_state_feat, mem, init_mem, reset_mask = restored
                except (OnlineQuarantineStateError, StructuralWitnessError) as error:
                    raise EarlySpatialGridPoseRunnerError(f"v10 restore failed: {error}") from error
                action, reason, restore_evidence = "quarantine_current_rollback", "current_online_detector_alarm", witness.timeline_evidence()
            exported, export_evidence = observer.finalize(observation, quarantined=quarantine, prediction=res)
            predictions.append(to_cpu(exported))
            evidence = dict(observer.timeline_evidence(observation))
            timeline.append({"frame_id": i, "action": action, "reason": reason, "current_alarm": quarantine, "consecutive_rollbacks": consecutive, "pending_transaction_count": 0, "restore_witness": restore_evidence, **dict(export_evidence), **evidence})
        if synchronize is not None:
            synchronize()
        elapsed += time.perf_counter() - started
    return EarlySpatialGridPoseRunnerResult(predictions, views, timeline, source_provenance, elapsed)


__all__ = ["EarlySpatialGridPoseObserver", "EarlySpatialGridPoseRunnerError", "EarlySpatialGridPoseRunnerResult", "audit_v10_runner_contract", "run_early_spatial_grid_pose_recurrent_lighter"]
