from __future__ import annotations

from types import SimpleNamespace

import pytest

from stateguard3r.recal3r_online_quarantine_v2 import (
    OnlineQuarantineStateError,
    QuarantineWatchdog,
    StructuralPreState,
)


class _FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(model_update_type="recal3r")
        self.update_pressure = "pre-pressure"
        self.recal3r_state0 = "pre-reference"
        self._recal3r_sequence_age = "pre-age"
        self._u_calibration_trace = {"frame_step": [0], "frame_u_mean": [0.1]}
        self._u_calibration_pending_u = "pre-pending-u"
        self._u_calibration_pending_h = "pre-pending-h"
        self._u_calibration_last_state = "pre-last"
        self._u_calibration_final_state = None
        self._u_calibration_oracle_window = 1
        self._u_calibration_queue = ["pre-entry"]


def test_structural_snapshot_restores_rebound_state_and_trace_tails() -> None:
    model = _FakeModel()
    local = ("state", "pos", "init", "mem", "init-mem", "reset")
    snapshot = StructuralPreState.capture(*local[:-1], local[-1], model, torch=None)
    model.update_pressure = "candidate-pressure"
    model.recal3r_state0 = "candidate-reference"
    model._u_calibration_trace["frame_step"].append(1)
    model._u_calibration_trace["frame_u_mean"].append(0.2)
    model._u_calibration_queue.append("candidate-entry")
    model._u_calibration_pending_u = "candidate-pending"
    restored = snapshot.restore(model, torch=None)
    assert restored == local
    assert model.update_pressure == "pre-pressure"
    assert model.recal3r_state0 == "pre-reference"
    assert model._u_calibration_trace == {"frame_step": [0], "frame_u_mean": [0.1]}
    assert model._u_calibration_queue == ["pre-entry"]
    assert model._u_calibration_pending_u == "pre-pending-u"


def test_structural_snapshot_rejects_trace_schema_or_shrinking_mutation() -> None:
    model = _FakeModel()
    snapshot = StructuralPreState.capture("s", "p", "i", "m", "im", "r", model, torch=None)
    model._u_calibration_trace["extra"] = []
    with pytest.raises(OnlineQuarantineStateError, match="schema changed"):
        snapshot.restore(model, torch=None)


def test_watchdog_is_fixed_and_fails_closed_after_eight_rollbacks() -> None:
    watchdog = QuarantineWatchdog()
    assert [watchdog.record(quarantine=True) for _ in range(8)] == list(range(1, 9))
    with pytest.raises(OnlineQuarantineStateError, match="watchdog"):
        watchdog.record(quarantine=True)
    assert watchdog.record(quarantine=False) == 0
    with pytest.raises(OnlineQuarantineStateError, match="fixed"):
        QuarantineWatchdog(7)
