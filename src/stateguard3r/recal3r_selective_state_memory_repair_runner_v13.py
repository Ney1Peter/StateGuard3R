"""Pinned recurrent loop with candidate-only v13 state/memory repair."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from .recal3r_online_quarantine_v2 import QuarantineWatchdog
from .recal3r_state_v3 import verify_pinned_lighter_source
from .selective_state_memory_repair_v13 import (
    SelectiveStateMemoryRepairError,
    capture_state_memory_prestate,
    require_unmutated_prestate,
    selective_state_memory_repair,
)


class SelectiveStateMemoryRepairRunnerError(RuntimeError):
    """The pinned v13 recurrent ordering or repair boundary is unsafe."""


class SelectiveStateMemoryRepairObserver(Protocol):
    def observe(self, frame_id: int, prediction: Mapping[str, Any], model: Any, frame_context: Any) -> Any: ...
    def alarm(self, observation: Any) -> bool: ...
    def finalize(self, observation: Any, *, repaired: bool, prediction: Mapping[str, Any], repair_evidence: Mapping[str, Any] | None) -> tuple[Mapping[str, Any], Mapping[str, Any]]: ...
    def timeline_evidence(self, observation: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class SelectiveStateMemoryRepairRunnerResult:
    predictions: list[Mapping[str, Any]]
    views: Sequence[Mapping[str, Any]]
    timeline: list[Mapping[str, Any]]
    source_provenance: Mapping[str, Any]
    recurrent_policy_runtime_seconds: float


def _reset_requested(value: Any) -> bool:
    if value is None:
        return False
    current = value.detach() if callable(getattr(value, "detach", None)) else value
    current = current.any() if callable(getattr(current, "any", None)) else current
    return bool(current.item() if callable(getattr(current, "item", None)) else current)


def audit_v13_runner_contract(runner_path: Path) -> Mapping[str, Any]:
    """Bind v13 to one post-update, candidate-only state/memory intervention."""
    source = runner_path.read_text(encoding="utf-8")
    try:
        body = source[source.index("\ndef run_selective_state_memory_repair_recurrent_lighter(") + 1:]
        pre_at = body.index("repair_prestate = capture_state_memory_prestate(state_feat, mem, torch=torch)")
        rollout_at = body.index("model._recurrent_rollout(", pre_at)
        state_update_at = body.index("state_feat = new_state * update_mask1 + state_feat * (1 - update_mask1)", rollout_at)
        memory_update_at = body.index("mem = new_mem * update_mask + mem * (1 - update_mask)", state_update_at)
        calibration_at = body.index("model._maybe_record_u_calibration_step(i, previous_state, state_feat)", memory_update_at)
        observer_at = body.index("if observer is None:", calibration_at)
        candidate_at = body.index("else:\n            observation = observer.observe", observer_at)
        repair_at = body.index("selective_state_memory_repair(", candidate_at)
        finalize_at = body.index("observer.finalize(observation, repaired=repair_evidence is not None", repair_at)
        candidate_transfer_at = body.index("predictions.append(to_cpu(exported))", finalize_at)
        control_transfer_at = body.index("predictions.append(to_cpu(res))", observer_at, candidate_at)
    except ValueError as error:
        raise SelectiveStateMemoryRepairRunnerError("v13 runner source contract is incomplete") from error
    if not (pre_at < rollout_at < state_update_at < memory_update_at < calibration_at < observer_at < candidate_at < repair_at < finalize_at < candidate_transfer_at):
        raise SelectiveStateMemoryRepairRunnerError("v13 state/memory repair ordering changed")
    if control_transfer_at < observer_at:
        raise SelectiveStateMemoryRepairRunnerError("v13 always-control transfer ordering changed")
    pre_region = body[pre_at:rollout_at]
    if any(marker in pre_region for marker in ("camera_pose", "prediction", "pointmap", "dec[", ".cpu(", ".numpy(", ".tolist(", "ground_truth", "future", "anchor", "history")):
        raise SelectiveStateMemoryRepairRunnerError("v13 pre-update repair capture contains a forbidden input")
    control = body[observer_at:candidate_at]
    if any(marker in control for marker in ("capture_state_memory_prestate(", "selective_state_memory_repair(", "observer.")):
        raise SelectiveStateMemoryRepairRunnerError("v13 always-control path is contaminated by candidate repair work")
    previous_recovery_markers = (
        "early_decoder_pose_runner_v8", "early_spatial_pooled_pose_runner_v9",
        "encoder_global_pooled_pose_runner_v11", "patch_embed_global_pooled_pose",
        "CurrentPointmapConsensus", "PreRolloutPoseQuery", "anchor_export_v3",
        "registration_export_v4", "pointmap_consensus_export_v5", "pose_query_export_v6",
    )
    if any(marker in body for marker in previous_recovery_markers):
        raise SelectiveStateMemoryRepairRunnerError("v13 runner delegates to an earlier recovery mechanism")
    forbidden_cpu_state_reads = ("get_u_calibration_last_state(", "_u_calibration_last_state")
    if any(marker in body for marker in forbidden_cpu_state_reads):
        raise SelectiveStateMemoryRepairRunnerError("v13 runner reads ReCal3R's CPU calibration-state mirror")
    return {
        "runner_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "candidate_only_selective_state_memory_repair": True,
        "repair_relation": "after_native_state_memory_update_and_calibration_before_cpu_transfer",
        "detector_before_repair": True,
        "always_control_uses_legacy_raw_transfer": True,
        "raw_current_prediction_is_not_replaced_by_runner": True,
        "cpu_calibration_state_mirror_is_not_read": True,
        "previous_recovery_delegation": False,
    }


def run_selective_state_memory_repair_recurrent_lighter(
    views: Sequence[Mapping[str, Any]], model: Any, device: Any, *, torch: Any,
    to_gpu: Callable[[Mapping[str, Any], Any], Mapping[str, Any]], to_cpu: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    canonicalize_model_update_type: Callable[[Any], str], before_frame: Callable[[int, Mapping[str, Any]], Any] | None = None,
    observer: SelectiveStateMemoryRepairObserver | None = None, watchdog_limit: int = 8,
    verify_source: bool = True, synchronize: Callable[[], None] | None = None,
    require_cuda_for_repair: bool = True,
) -> SelectiveStateMemoryRepairRunnerResult:
    """Run native ReCal3R, repairing only persistent rows after a raw alarm."""
    if not views:
        raise SelectiveStateMemoryRepairRunnerError("views must not be empty")
    if observer is not None and require_cuda_for_repair and getattr(device, "type", None) != "cuda":
        raise SelectiveStateMemoryRepairRunnerError("v13 candidate repair requires a CUDA execution device")
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
            raise SelectiveStateMemoryRepairRunnerError("v13 repair fails closed at a non-initial reset boundary")
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
            raise SelectiveStateMemoryRepairRunnerError("ReCal3R frame has neither image nor raymap features")
        if i == 0:
            state_feat, state_pos = model._init_state(feat_i, pos_i)
            model._init_recal3r_reference_state(state_feat)
            mem = model.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
            init_state_feat, init_mem = state_feat.clone(), mem.clone()
            if tuple(state_feat.shape) != (1, 768, 768) or tuple(mem.shape) != (1, 256, 1536):
                raise SelectiveStateMemoryRepairRunnerError("pinned ReCal3R state/memory shapes changed")
        repair_prestate = capture_state_memory_prestate(state_feat, mem, torch=torch) if observer is not None and i else None
        global_img_feat_i = model._get_img_level_feat(feat_i) if model.pose_head_flag else None
        if model.pose_head_flag:
            pose_feat_i = model.pose_token.expand(feat_i.shape[0], -1, -1) if i == 0 or reset_mask else model.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype)
        else:
            pose_feat_i, pose_pos_i = None, None
        new_state, dec, self_state, cross_state, self_img, cross_img = model._recurrent_rollout(state_feat, state_pos, feat_i, pos_i, pose_feat_i, pose_pos_i, init_state_feat, img_mask=view["img_mask"], reset_mask=view["reset"], update=view.get("update"), return_attn=True)
        del self_state, self_img, cross_img
        if len(dec) != model.dec_depth + 1:
            raise SelectiveStateMemoryRepairRunnerError("pinned ReCal3R decoder depth changed")
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
            timeline.append({"frame_id": i, "action": "commit", "reason": "always_commit_control", "current_alarm": False, "consecutive_repairs": 0, "pending_transaction_count": 0, "repair": None, "export_action": "export_real_camera_pose"})
        else:
            observation = observer.observe(i, res, model, frame_context)
            repair = bool(observer.alarm(observation)) if i else False
            consecutive = watchdog.record(quarantine=repair)
            repair_evidence = None
            if repair:
                if repair_prestate is None:
                    raise SelectiveStateMemoryRepairRunnerError("v13 state repair has no pre-update snapshot")
                if require_cuda_for_repair and (getattr(state_feat.device, "type", None) != "cuda" or getattr(mem.device, "type", None) != "cuda"):
                    raise SelectiveStateMemoryRepairRunnerError("v13 candidate repair requires CUDA-resident state and memory")
                try:
                    require_unmutated_prestate(repair_prestate)
                    repaired = selective_state_memory_repair(repair_prestate.state_feat, state_feat, repair_prestate.mem, mem, torch=torch)
                except SelectiveStateMemoryRepairError as error:
                    raise SelectiveStateMemoryRepairRunnerError(f"v13 selective state/memory repair failed closed: {error}") from error
                state_feat, mem, repair_evidence = repaired.state_feat, repaired.mem, repaired.evidence
            exported, export_evidence = observer.finalize(observation, repaired=repair_evidence is not None, prediction=res, repair_evidence=repair_evidence)
            predictions.append(to_cpu(exported))
            if repair_evidence is not None:
                state_evidence = repair_evidence["state_feat"]
                mem_evidence = repair_evidence["mem"]
                repair_evidence = {
                    **repair_evidence,
                    "state_feat": {key: value for key, value in state_evidence.items() if not key.endswith("_tensor_digest")} | {"pre_gpu_digest": state_evidence["pre_tensor_digest"], "proposed_gpu_digest": state_evidence["proposed_tensor_digest"], "committed_gpu_digest": state_evidence["committed_tensor_digest"]},
                    "mem": {key: value for key, value in mem_evidence.items() if not key.endswith("_tensor_digest")} | {"pre_gpu_digest": mem_evidence["pre_tensor_digest"], "proposed_gpu_digest": mem_evidence["proposed_tensor_digest"], "committed_gpu_digest": mem_evidence["committed_tensor_digest"]},
                    "cuda_resident": bool(require_cuda_for_repair),
                }
            timeline.append({"frame_id": i, "action": "selective_state_memory_repair" if repair else "commit", "reason": "current_online_detector_alarm" if repair else ("initial_state_must_commit" if i == 0 else "current_online_detector_clear"), "current_alarm": repair, "consecutive_repairs": consecutive, "pending_transaction_count": 0, "repair": repair_evidence, **dict(export_evidence), **dict(observer.timeline_evidence(observation))})
        if synchronize is not None:
            synchronize()
        elapsed += time.perf_counter() - started
    return SelectiveStateMemoryRepairRunnerResult(predictions, views, timeline, source_provenance, elapsed)


__all__ = [
    "SelectiveStateMemoryRepairObserver", "SelectiveStateMemoryRepairRunnerError",
    "SelectiveStateMemoryRepairRunnerResult", "audit_v13_runner_contract",
    "run_selective_state_memory_repair_recurrent_lighter",
]
