from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from stateguard3r.native_scalar_detector_v20 import (
    IncrementalNativeScalarDetectorV20,
    NativeScalarDetectorV20Config,
    NativeScalarDetectorV20Error,
    canonical_native_scalar_row_v20,
)
from stateguard3r import official_native_scalar_hold_v20 as runtime


def _row(frame_id: int, value: float | None) -> dict[str, Any]:
    return {
        "frame_id": frame_id,
        "uncertainty_u": value,
        "reliability": None if value is None else 1.0 - value,
        "global_state_delta": value,
        "pose_jump": value,
        "geometric_residual": value,
    }


def test_v20_detector_accepts_only_current_scalar_rows_and_has_bounded_prefix() -> None:
    detector = IncrementalNativeScalarDetectorV20()
    for frame_id in range(3):
        decision = detector.observe(_row(frame_id, 0.1))
        assert decision.alarm is False
        detector.commit(_row(frame_id, 0.1))
    decision = detector.observe(_row(3, 0.9))
    assert decision.alarm is True
    assert decision.finite_component_count == 4
    detector.commit(_row(3, 0.9))
    with pytest.raises(NativeScalarDetectorV20Error, match="prefix"):
        detector.observe(_row(5, 0.1))


@pytest.mark.parametrize(
    "mutate",
    (
        lambda row: row.update({"model": object()}),
        lambda row: row.update({"uncertainty_u": float("nan")}),
        lambda row: row.update({"reliability": 0.1}),
        lambda row: row.update({"frame_id": True}),
    ),
)
def test_v20_detector_rejects_hidden_or_nonfinite_channels(mutate: Any) -> None:
    row = _row(0, 0.1)
    mutate(row)
    with pytest.raises(NativeScalarDetectorV20Error):
        canonical_native_scalar_row_v20(row)


class _Tensor:
    def __init__(self, value: float) -> None:
        self.value = value
        self.device = SimpleNamespace(type="cuda")
        self.dtype = "fake"

    def numel(self) -> int: return 1
    def item(self) -> float: return self.value
    def detach(self) -> "_Tensor": return self
    def reshape(self, _shape: int) -> "_Tensor": return self
    def sum(self) -> "_Tensor": return self
    def abs(self) -> "_Tensor": return self
    def square(self) -> "_Tensor": return self
    def min(self) -> "_Tensor": return self
    def max(self) -> "_Tensor": return self
    def clone(self) -> "_Tensor": return _Tensor(self.value)
    def zero_(self) -> "_Tensor": self.value = 0.0; return self
    @property
    def shape(self) -> tuple[int, ...]: return (1, 1, 1)


class _Torch:
    @staticmethod
    def is_tensor(value: Any) -> bool: return isinstance(value, _Tensor)
    @staticmethod
    def is_floating_point(value: Any) -> bool: return isinstance(value, _Tensor)
    @staticmethod
    def isfinite(value: _Tensor) -> SimpleNamespace: return SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True))


class _Observer:
    def __init__(self, alarms: set[int]) -> None:
        self.alarms, self.rows = alarms, []

    def observe(self, row: Mapping[str, Any]) -> int:
        assert set(row) == {"frame_id", "uncertainty_u", "reliability", "global_state_delta", "pose_jump", "geometric_residual"}
        self.rows.append(dict(row))
        return int(row["frame_id"])

    def alarm(self, decision: int) -> bool: return decision in self.alarms
    def commit(self, row: Mapping[str, Any]) -> None: assert set(row) == {"frame_id", "uncertainty_u", "reliability", "global_state_delta", "pose_jump", "geometric_residual"}
    def evidence(self, decision: int) -> Mapping[str, Any]: return {"detector_frame": decision}


def _views(count: int) -> list[dict[str, _Tensor]]:
    return [{"img_mask": _Tensor(1), "update": _Tensor(1), "reset": _Tensor(0)} for _ in range(count)]


def _model() -> Any:
    return SimpleNamespace(
        _downstream_head=lambda *_args, **_kwargs: {"camera_pose": _Tensor(0)},
        _compute_recal3r_update_mask=lambda *_args, **_kwargs: _Tensor(1),
        _maybe_record_u_calibration_step=lambda *_args, **_kwargs: None,
        get_u_calibration_trace=lambda: {"not": "exposed"},
    )


def _official_loop(model: Any, views: list[dict[str, _Tensor]]) -> None:
    state = _Tensor(1)
    for frame_id, _view in enumerate(views):
        prediction = model._downstream_head(None, None)
        prediction["camera_pose"].value = frame_id + 10
        before = state.clone()
        if frame_id:
            mask = model._compute_recal3r_update_mask(None, None, None, prev_state_feat=state)
            state = _Tensor(state.value * mask.value + state.value * (1.0 - mask.value))
        model._maybe_record_u_calibration_step(frame_id, before, state)


def test_v20_wrapper_only_passes_scalar_row_and_holds_next_official_state(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_health(frame_id: int, prediction: Mapping[str, Any], trace: Mapping[str, Any], **_kwargs: Any) -> tuple[dict[str, Any], _Tensor]:
        assert trace == {"not": "exposed"}
        assert prediction["camera_pose"].value == frame_id + 10
        return _row(frame_id, 0.1), _Tensor(frame_id + 10)

    monkeypatch.setattr(runtime, "_health_row", fake_health)
    model, views, observer = _model(), _views(4), _Observer({1})
    originals = (model._downstream_head, model._compute_recal3r_update_mask, model._maybe_record_u_calibration_step)
    with runtime.official_native_scalar_hold_v20(model, views, torch=_Torch(), decode=lambda value: value, observer=observer) as timeline:
        _official_loop(model, views)
    assert [row["action"] for row in timeline] == ["native_commit", "armed_one_future_native_scalar_hold", "one_shot_native_scalar_state_hold", "native_commit"]
    assert timeline[2]["native_mask_computed"] is True
    assert timeline[2]["final_state_mask_replaced_by_zero"] is True
    assert timeline[2]["held_global_state_gpu_identity"] is True
    assert all(set(row) == {"frame_id", "uncertainty_u", "reliability", "global_state_delta", "pose_jump", "geometric_residual"} for row in observer.rows)
    assert (model._downstream_head, model._compute_recal3r_update_mask, model._maybe_record_u_calibration_step) == originals


def test_v20_wrapper_rejects_same_frame_consume_and_alarm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "_health_row", lambda frame_id, *_args, **_kwargs: (_row(frame_id, 0.1), _Tensor(frame_id + 10)))
    model, views = _model(), _views(3)
    with pytest.raises(runtime.OfficialNativeScalarHoldV20Error, match="same v20"):
        with runtime.official_native_scalar_hold_v20(model, views, torch=_Torch(), decode=lambda value: value, observer=_Observer({0, 1})):
            _official_loop(model, views)
