"""V20 official-loop state-mask hold with a closed post-commit scalar observer."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence


class OfficialNativeScalarHoldV20Error(RuntimeError):
    """The v20 post-commit scalar/official-loop contract was violated."""


class ScalarObserverV20(Protocol):
    def observe(self, row: Mapping[str, Any]) -> Any: ...
    def alarm(self, decision: Any) -> bool: ...
    def commit(self, row: Mapping[str, Any]) -> None: ...
    def evidence(self, decision: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class _Selection:
    armed_before: bool
    consumed: bool
    reset_cancelled: bool


def _scalar_bool(value: Any, *, label: str) -> bool:
    try:
        if int(value.numel()) != 1:
            raise OfficialNativeScalarHoldV20Error(f"{label} must have one item")
        return bool(value.item())
    except AttributeError as error:
        raise OfficialNativeScalarHoldV20Error(f"{label} must be a scalar tensor") from error


def _summary(value: Any, *, torch: Any, label: str) -> str:
    if not bool(torch.is_tensor(value)) or not bool(torch.is_floating_point(value)) or getattr(value.device, "type", None) != "cuda" or not bool(torch.isfinite(value).all().item()):
        raise OfficialNativeScalarHoldV20Error(f"{label} must be finite CUDA float")
    flat = value.detach().reshape(-1)
    values = (flat.sum(), flat.abs().sum(), flat.square().sum(), flat.min(), flat.max())
    return "v20-gpu-summary:" + hashlib.sha256("|".join((str(tuple(value.shape)), str(value.dtype), str(value.device)) + tuple(float(item.item()).hex() for item in values)).encode("ascii")).hexdigest()


def _median(torch: Any, value: Any) -> Any:
    ordered = torch.sort(value.reshape(-1)).values
    return ordered[len(ordered) // 2] if len(ordered) % 2 else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2


def _health_row(
    frame_id: int,
    prediction: Mapping[str, Any],
    native_trace: Mapping[str, Any],
    *,
    torch: Any,
    decode: Callable[[Any], Any],
    previous_camera: Any | None,
    state_before: Any,
    state_after: Any,
) -> tuple[dict[str, Any], Any]:
    """Convert current native values into a finite scalar detector row.

    ``_maybe_record_u_calibration_step`` receives the two live state values
    immediately after the official global-state commit.  Reducing their
    difference here preserves the required current-frame state-delta signal
    without enabling ReCal3R's ``oracle_window``/``final_state`` trace modes;
    those modes are unnecessary for v20 and would create an impermissible
    future/final-state route.  The observer receives only the resulting Python
    scalar, never either state tensor.
    """

    required = {"frame_u_mean", "frame_step"}
    if not isinstance(native_trace, Mapping) or not required <= set(native_trace):
        raise OfficialNativeScalarHoldV20Error("current native calibration trace differs")
    if frame_id == 0:
        uncertainty = delta = None
    else:
        if not native_trace["frame_u_mean"] or not native_trace["frame_step"] or int(native_trace["frame_step"][-1].item()) != frame_id:
            raise OfficialNativeScalarHoldV20Error("native current uncertainty summary is unavailable")
        uncertainty = float(native_trace["frame_u_mean"][-1].item())
        if not bool(torch.is_tensor(state_before)) or not bool(torch.is_tensor(state_after)):
            raise OfficialNativeScalarHoldV20Error("native current state delta lacks state tensors")
        if state_before.shape != state_after.shape:
            raise OfficialNativeScalarHoldV20Error("native current state delta shape differs")
        delta = float(torch.linalg.vector_norm(state_after - state_before, dim=-1).mean().item())
    camera = decode(prediction["camera_pose"])
    pose_jump = None
    if previous_camera is not None:
        relative = previous_camera[:, :3, :3].transpose(-1, -2) @ camera[:, :3, :3]
        cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
        translation = previous_camera[:, :3, :3].transpose(-1, -2) @ (camera[:, :3, 3] - previous_camera[:, :3, 3]).unsqueeze(-1)
        pose_jump = float(torch.hypot(torch.linalg.vector_norm(translation.squeeze(-1), dim=-1), torch.acos(cosine)).item())
    transformed = prediction["pts3d_in_self_view"] @ camera[:, :3, :3].transpose(-1, -2) + camera[:, None, None, :3, 3]
    residual = float((_median(torch, torch.linalg.vector_norm(transformed - prediction["pts3d_in_other_view"], dim=-1)) / torch.clamp(_median(torch, torch.linalg.vector_norm(prediction["pts3d_in_other_view"], dim=-1)), min=1e-8)).item())
    values = (uncertainty, delta, pose_jump, residual)
    if any(value is not None and (not math.isfinite(value) or value < 0.0) for value in values):
        raise OfficialNativeScalarHoldV20Error("current native scalar is nonfinite")
    return {
        "frame_id": frame_id,
        "uncertainty_u": uncertainty,
        "reliability": None if uncertainty is None else 1.0 - uncertainty,
        "global_state_delta": delta,
        "pose_jump": pose_jump,
        "geometric_residual": residual,
    }, camera.detach().clone()


@contextmanager
def official_native_scalar_hold_v20(
    model: Any,
    raw_views: Sequence[Mapping[str, Any]],
    *,
    torch: Any,
    decode: Callable[[Any], Any],
    observer: ScalarObserverV20 | None,
) -> Iterator[list[Mapping[str, Any]]]:
    """Enclose one official ``inference_recurrent_lighter`` call safely."""

    if not raw_views:
        raise OfficialNativeScalarHoldV20Error("v20 needs official raw views")
    originals = (model._downstream_head, model._compute_recal3r_update_mask, model._maybe_record_u_calibration_step)
    timeline: list[Mapping[str, Any]] = []
    armed, frame_id, prediction, selection, native_mask, previous_camera, previous_reset = False, -1, None, None, None, None, False

    def head(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        nonlocal frame_id, prediction, selection, native_mask
        frame_id += 1
        if frame_id >= len(raw_views):
            raise OfficialNativeScalarHoldV20Error("official head exceeded v20 stream")
        # The official loop makes native-mask eligibility depend on the reset
        # flag carried from the *preceding* frame.  Selection must follow the
        # same timing: consuming an arm where upstream has no native mask is
        # forbidden rather than silently moving the intervention.
        selection = _Selection(armed, bool(armed and not previous_reset), bool(armed and previous_reset))
        native_mask = None
        prediction = originals[0](*args, **kwargs)
        if not isinstance(prediction, Mapping):
            raise OfficialNativeScalarHoldV20Error("official prediction differs")
        return prediction

    def mask(*args: Any, **kwargs: Any) -> Any:
        nonlocal native_mask
        if selection is None or native_mask is not None:
            raise OfficialNativeScalarHoldV20Error("v20 native mask order differs")
        native_mask = originals[1](*args, **kwargs)
        if not selection.consumed:
            return native_mask
        held = native_mask.clone().zero_()
        if held is native_mask:
            raise OfficialNativeScalarHoldV20Error("held mask aliases native mask")
        return held

    def record(native_frame: int, state_before: Any, state_after: Any) -> None:
        nonlocal armed, prediction, previous_camera, previous_reset
        originals[2](native_frame, state_before, state_after)
        if native_frame != frame_id or prediction is None or selection is None:
            raise OfficialNativeScalarHoldV20Error("v20 calibration order differs")
        if frame_id > 0 and native_mask is None:
            raise OfficialNativeScalarHoldV20Error("native mask missing before current state commit")
        if selection.consumed:
            if not bool(torch.equal(state_before, state_after)):
                raise OfficialNativeScalarHoldV20Error("held global state is not identity")
        alarm, row, detector_evidence, alarm_witness = False, None, {}, {}
        if observer is not None:
            row, next_camera = _health_row(
                frame_id,
                prediction,
                model.get_u_calibration_trace(),
                torch=torch,
                decode=decode,
                previous_camera=previous_camera,
                state_before=state_before,
                state_after=state_after,
            )
            state_before_observer = _summary(state_after, torch=torch, label="alarm committed state")
            pose_before_observer = _summary(prediction["camera_pose"], torch=torch, label="alarm raw pose")
            decision = observer.observe(row)
            alarm = bool(observer.alarm(decision))
            observer.commit(row)
            alarm_witness = {
                "raw_current_state_gpu_fingerprint_unchanged": state_before_observer == _summary(state_after, torch=torch, label="alarm post-observer state"),
                "raw_current_pose_gpu_fingerprint_unchanged": pose_before_observer == _summary(prediction["camera_pose"], torch=torch, label="alarm post-observer pose"),
            }
            if not all(alarm_witness.values()):
                raise OfficialNativeScalarHoldV20Error("scalar detector altered the alarm frame")
            detector_evidence = dict(observer.evidence(decision))
            previous_camera = next_camera
        if alarm and selection.consumed:
            raise OfficialNativeScalarHoldV20Error("same v20 frame consumed old arm and raised new alarm")
        current_reset = _scalar_bool(raw_views[frame_id]["reset"], label="native reset")
        armed = bool(alarm and frame_id + 1 < len(raw_views))
        timeline.append({
            "frame_id": frame_id,
            "action": "one_shot_native_scalar_state_hold" if selection.consumed else ("armed_one_future_native_scalar_hold" if armed else "native_commit"),
            "current_alarm": alarm,
            "arm_pending": armed,
            "arm_consumed": selection.consumed,
            "arm_reset_cancelled": selection.reset_cancelled,
            "native_mask_computed": native_mask is not None,
            "final_state_mask_replaced_by_zero": selection.consumed,
            "held_global_state_gpu_identity": None if not selection.consumed else True,
            "held_global_state_exact_equal": None if not selection.consumed else True,
            "native_memory_update_rule_unchanged": True,
            "detector_row": row,
            **alarm_witness,
            **detector_evidence,
        })
        previous_reset = current_reset
        prediction = None

    model._downstream_head, model._compute_recal3r_update_mask, model._maybe_record_u_calibration_step = head, mask, record
    try:
        yield timeline
        if frame_id != len(raw_views) - 1 or len(timeline) != len(raw_views):
            raise OfficialNativeScalarHoldV20Error("official loop did not cover v20 stream")
    finally:
        model._downstream_head, model._compute_recal3r_update_mask, model._maybe_record_u_calibration_step = originals


__all__ = ["OfficialNativeScalarHoldV20Error", "ScalarObserverV20", "official_native_scalar_hold_v20"]
