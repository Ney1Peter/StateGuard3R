from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from stateguard3r.official_mask_hold_runtime_v19 import (
    OfficialMaskHoldRuntimeV19Error,
    official_native_mask_hold_v19,
)


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
        self.alarms = alarms
        self.committed: list[int] = []

    def observe(self, frame_id: int, prediction: Mapping[str, Any], native_trace: Any, frame_context: Mapping[str, Any] | None) -> int:
        assert prediction["camera_pose"].value == frame_id + 10
        assert native_trace == {"trace": True}
        return frame_id

    def alarm(self, observation: int) -> bool: return observation in self.alarms
    def commit(self, observation: int) -> Mapping[str, Any]: self.committed.append(observation); return {"committed": observation}
    def evidence(self, observation: int) -> Mapping[str, Any]: return {"detector_frame": observation}


def _views(count: int) -> list[dict[str, _Tensor]]:
    return [{"img_mask": _Tensor(1), "update": _Tensor(1), "reset": _Tensor(0)} for _ in range(count)]


def _model() -> Any:
    return SimpleNamespace(
        _downstream_head=lambda *_args, **_kwargs: {"camera_pose": _Tensor(0)},
        _compute_recal3r_update_mask=lambda *_args, **_kwargs: _Tensor(1),
        _maybe_record_u_calibration_step=lambda *_args, **_kwargs: None,
        get_u_calibration_trace=lambda: {"trace": True},
    )


def _fake_official_loop(model: Any, views: list[dict[str, _Tensor]]) -> None:
    state = _Tensor(1)
    for frame_id, _view in enumerate(views):
        prediction = model._downstream_head(None, None)
        prediction["camera_pose"].value = frame_id + 10
        if frame_id:
            state_before = state.clone()
            mask = model._compute_recal3r_update_mask(None, None, None, prev_state_feat=state)
            state = _Tensor(state.value * mask.value + state.value * (1.0 - mask.value))
        else:
            state_before = state.clone()
        model._maybe_record_u_calibration_step(frame_id, state_before, state)


def test_v19_official_wrapper_arm_then_holds_only_immediate_next_state_mask() -> None:
    model, views, observer = _model(), _views(4), _Observer({1})
    originals = (model._downstream_head, model._compute_recal3r_update_mask, model._maybe_record_u_calibration_step)
    with official_native_mask_hold_v19(model, views, torch=_Torch(), observer=observer) as timeline:
        _fake_official_loop(model, views)
    assert [row["action"] for row in timeline] == ["native_commit", "armed_one_future_native_state_hold", "one_shot_native_state_mask_hold", "native_commit"]
    assert timeline[2]["native_mask_computed"] is True
    assert timeline[2]["final_state_mask_replaced_by_zero"] is True
    assert timeline[2]["held_global_state_gpu_identity"] is True
    assert observer.committed == [0, 1, 2, 3]
    assert (model._downstream_head, model._compute_recal3r_update_mask, model._maybe_record_u_calibration_step) == originals


def test_v19_wrapper_reset_cancels_old_arm() -> None:
    model, views, observer = _model(), _views(3), _Observer({0})
    views[1]["reset"] = _Tensor(1)
    with official_native_mask_hold_v19(model, views, torch=_Torch(), observer=observer) as timeline:
        _fake_official_loop(model, views)
    assert timeline[1]["arm_reset_cancelled"] is True
    assert timeline[1]["final_state_mask_replaced_by_zero"] is False


def test_v19_wrapper_rejects_same_frame_old_consume_and_new_alarm_and_restores() -> None:
    model, views, observer = _model(), _views(3), _Observer({0, 1})
    originals = (model._downstream_head, model._compute_recal3r_update_mask, model._maybe_record_u_calibration_step)
    with pytest.raises(OfficialMaskHoldRuntimeV19Error, match="same frame"):
        with official_native_mask_hold_v19(model, views, torch=_Torch(), observer=observer):
            _fake_official_loop(model, views)
    assert (model._downstream_head, model._compute_recal3r_update_mask, model._maybe_record_u_calibration_step) == originals


def test_v19_wrapper_rejects_an_official_frame_missing_native_mask() -> None:
    model, views = _model(), _views(2)
    with pytest.raises(OfficialMaskHoldRuntimeV19Error, match="omitted"):
        with official_native_mask_hold_v19(model, views, torch=_Torch(), observer=_Observer(set())):
            prediction = model._downstream_head(None, None); prediction["camera_pose"].value = 10
            model._maybe_record_u_calibration_step(0, _Tensor(1), _Tensor(1))
            prediction = model._downstream_head(None, None); prediction["camera_pose"].value = 11
            model._maybe_record_u_calibration_step(1, _Tensor(1), _Tensor(1))
