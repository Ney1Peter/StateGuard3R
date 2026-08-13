"""Causal v19 wrapper around the official ReCal3R lightweight execution path.

This module never implements a recurrent model loop.  Its context manager
temporarily replaces three instance attributes used by the official loop:
the downstream head (to retain only the current raw prediction), the native
mask function (to return zero only after calling it), and the post-commit
calibration function (to invoke the scalar observer after native commits).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

from .native_mask_hold_v19 import (
    NativeMaskHoldSelectionV19,
    NativeMaskHoldV19Error,
    arm_one_native_state_mask_v19,
    select_native_mask_hold_v19,
)


class OfficialMaskHoldRuntimeV19Error(RuntimeError):
    """The official-loop v19 causal boundary was violated."""


class ScalarPostCommitObserverV19(Protocol):
    def observe(
        self,
        frame_id: int,
        prediction: Mapping[str, Any],
        native_trace: Any,
        frame_context: Mapping[str, Any] | None,
    ) -> Any: ...

    def alarm(self, observation: Any) -> bool: ...

    def commit(self, observation: Any) -> Mapping[str, Any]: ...

    def evidence(self, observation: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class _FrameSelection:
    frame_id: int
    selection: NativeMaskHoldSelectionV19
    raw_update_eligible: bool


def _plain_tensor_bool(value: Any, *, label: str) -> bool:
    try:
        if int(value.numel()) != 1:
            raise OfficialMaskHoldRuntimeV19Error(f"{label} must have one item")
        return bool(value.item())
    except AttributeError as error:
        raise OfficialMaskHoldRuntimeV19Error(f"{label} must be a scalar tensor") from error


def _cuda_digest(value: Any, *, torch: Any, label: str) -> str:
    if not bool(torch.is_tensor(value)) or not bool(torch.is_floating_point(value)):
        raise OfficialMaskHoldRuntimeV19Error(f"{label} is not a floating tensor")
    if getattr(value.device, "type", None) != "cuda" or not bool(torch.isfinite(value).all().item()):
        raise OfficialMaskHoldRuntimeV19Error(f"{label} must be finite CUDA resident")
    flat = value.detach().reshape(-1)
    reductions = (flat.sum(), flat.abs().sum(), flat.square().sum(), flat.min(), flat.max())
    text = "|".join((str(tuple(value.shape)), str(value.dtype), str(value.device)) + tuple(float(item.item()).hex() for item in reductions))
    return "v19-cuda-summary:" + hashlib.sha256(text.encode("ascii")).hexdigest()


@contextmanager
def official_native_mask_hold_v19(
    model: Any,
    raw_views: Sequence[Mapping[str, Any]],
    *,
    torch: Any,
    observer: ScalarPostCommitObserverV19 | None,
    before_frame: Callable[[int, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> Iterator[list[Mapping[str, Any]]]:
    """Apply one future state-mask hold while the official loop owns recurrence.

    The context expects to enclose one call to
    ``inference_recurrent_lighter(raw_views, model, device)``.  Its wrappers
    are restored whether the official call succeeds or fails.
    """

    if not raw_views:
        raise OfficialMaskHoldRuntimeV19Error("v19 needs official raw views")
    originals = {
        "head": model._downstream_head,
        "mask": model._compute_recal3r_update_mask,
        "record": model._maybe_record_u_calibration_step,
    }
    timeline: list[Mapping[str, Any]] = []
    armed = False
    frame_id = -1
    current_prediction: Mapping[str, Any] | None = None
    current_selection: _FrameSelection | None = None
    current_native_mask: Any = None
    current_state_after_mask: str | None = None

    def head(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        nonlocal frame_id, current_prediction, current_selection, current_native_mask, current_state_after_mask
        frame_id += 1
        if frame_id >= len(raw_views):
            raise OfficialMaskHoldRuntimeV19Error("official downstream head exceeded raw view count")
        raw = raw_views[frame_id]
        try:
            reset = _plain_tensor_bool(raw["reset"], label="native reset")
            eligible = _plain_tensor_bool(raw["img_mask"], label="native image eligibility") and _plain_tensor_bool(raw["update"], label="native update eligibility")
        except (KeyError, TypeError) as error:
            raise OfficialMaskHoldRuntimeV19Error("raw official view lacks v19 eligibility") from error
        selection = select_native_mask_hold_v19(armed, reset=reset)
        if selection.return_zero_mask and not eligible:
            raise OfficialMaskHoldRuntimeV19Error("v19 arm reached a noneligible official view")
        current_selection = _FrameSelection(frame_id, selection, eligible)
        current_native_mask = None
        current_state_after_mask = None
        prediction = originals["head"](*args, **kwargs)
        if not isinstance(prediction, Mapping):
            raise OfficialMaskHoldRuntimeV19Error("official downstream head output differs")
        current_prediction = prediction
        return prediction

    def mask(*args: Any, **kwargs: Any) -> Any:
        nonlocal current_native_mask
        if current_selection is None or current_selection.frame_id != frame_id:
            raise OfficialMaskHoldRuntimeV19Error("native mask call lacks current official frame")
        if current_native_mask is not None:
            raise OfficialMaskHoldRuntimeV19Error("official frame computed native mask more than once")
        native = originals["mask"](*args, **kwargs)
        current_native_mask = native
        if not current_selection.selection.return_zero_mask:
            return native
        try:
            held = native.clone().zero_()
        except AttributeError as error:
            raise OfficialMaskHoldRuntimeV19Error("official native mask is not cloneable") from error
        if held is native:
            raise OfficialMaskHoldRuntimeV19Error("v19 final-mask hold aliased native result")
        return held

    def record(native_frame_id: int, state_before: Any, state_after: Any) -> None:
        nonlocal armed, current_prediction, current_state_after_mask
        originals["record"](native_frame_id, state_before, state_after)
        if native_frame_id != frame_id or current_selection is None or current_selection.frame_id != frame_id or current_prediction is None:
            raise OfficialMaskHoldRuntimeV19Error("official post-commit order differs")
        selection = current_selection.selection
        if frame_id == 0 and current_native_mask is not None:
            raise OfficialMaskHoldRuntimeV19Error("first official frame unexpectedly called native ReCal3R mask")
        if frame_id > 0 and current_native_mask is None:
            raise OfficialMaskHoldRuntimeV19Error("official eligible frame omitted native ReCal3R mask")
        state_before_digest = _cuda_digest(state_before, torch=torch, label="native pre-state")
        state_after_digest = _cuda_digest(state_after, torch=torch, label="native post-state")
        if selection.return_zero_mask and state_before_digest != state_after_digest:
            raise OfficialMaskHoldRuntimeV19Error("held official global state is not an identity")
        current_state_after_mask = state_after_digest
        alarm = False
        observer_evidence: Mapping[str, Any] = {}
        alarm_witness: Mapping[str, Any] = {}
        if observer is not None:
            context = before_frame(frame_id, raw_views[frame_id]) if before_frame is not None else None
            state_before_observer = _cuda_digest(state_after, torch=torch, label="alarm-frame committed state")
            pose_before_observer = _cuda_digest(current_prediction["camera_pose"], torch=torch, label="alarm-frame raw pose")
            observation = observer.observe(frame_id, current_prediction, model.get_u_calibration_trace(), context)
            alarm = bool(observer.alarm(observation))
            commit_evidence = dict(observer.commit(observation))
            state_after_observer = _cuda_digest(state_after, torch=torch, label="alarm-frame post-observer state")
            pose_after_observer = _cuda_digest(current_prediction["camera_pose"], torch=torch, label="alarm-frame post-observer pose")
            alarm_witness = {
                "raw_current_state_gpu_fingerprint_unchanged": state_before_observer == state_after_observer,
                "raw_current_pose_gpu_fingerprint_unchanged": pose_before_observer == pose_after_observer,
            }
            if not all(alarm_witness.values()):
                raise OfficialMaskHoldRuntimeV19Error("scalar observer altered alarm-frame native output")
            observer_evidence = commit_evidence | dict(observer.evidence(observation))
        if alarm and selection.consumed:
            raise OfficialMaskHoldRuntimeV19Error("v19 same frame consumed an old arm and raised a new alarm")
        # A current selection always settles the old arm before the current
        # detector is allowed to create a new future one.
        armed = selection.armed_after
        armed = arm_one_native_state_mask_v19(armed, alarm=alarm, has_next_view=frame_id + 1 < len(raw_views))
        timeline.append({
            "frame_id": frame_id,
            "action": "one_shot_native_state_mask_hold" if selection.consumed else ("armed_one_future_native_state_hold" if alarm and armed else "native_commit"),
            "current_alarm": alarm,
            "arm_pending": armed,
            "arm_consumed": selection.consumed,
            "arm_reset_cancelled": selection.reset_cancelled,
            "native_mask_computed": current_native_mask is not None,
            "final_state_mask_replaced_by_zero": selection.return_zero_mask,
            "held_global_state_gpu_identity": None if not selection.consumed else state_before_digest == state_after_digest,
            "native_memory_update_rule_unchanged": True,
            **alarm_witness,
            **observer_evidence,
        })
        current_prediction = None

    model._downstream_head = head
    model._compute_recal3r_update_mask = mask
    model._maybe_record_u_calibration_step = record
    try:
        yield timeline
        if frame_id != len(raw_views) - 1 or len(timeline) != len(raw_views):
            raise OfficialMaskHoldRuntimeV19Error("official loop did not cover every raw view")
    finally:
        model._downstream_head = originals["head"]
        model._compute_recal3r_update_mask = originals["mask"]
        model._maybe_record_u_calibration_step = originals["record"]
        if (
            model._downstream_head is not originals["head"]
            or model._compute_recal3r_update_mask is not originals["mask"]
            or model._maybe_record_u_calibration_step is not originals["record"]
        ):
            raise OfficialMaskHoldRuntimeV19Error("v19 official method wrapper was not restored")


__all__ = [
    "OfficialMaskHoldRuntimeV19Error",
    "ScalarPostCommitObserverV19",
    "official_native_mask_hold_v19",
]
