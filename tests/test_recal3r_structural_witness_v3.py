from __future__ import annotations

from types import SimpleNamespace

import pytest

from stateguard3r.recal3r_online_quarantine_v2 import StructuralPreState
from stateguard3r.recal3r_structural_witness_v3 import StructuralIdentityWitness, StructuralWitnessError


class _Tensor:
    def __init__(self, pointer: int, version: int = 0) -> None:
        self.pointer = pointer
        self._version = version

    def data_ptr(self) -> int:
        return self.pointer


class _Model:
    def __init__(self) -> None:
        self.config = SimpleNamespace(model_update_type="recal3r")
        self.update_pressure = _Tensor(1)
        self.recal3r_state0 = _Tensor(2)
        self._recal3r_sequence_age = _Tensor(3)
        self._u_calibration_trace = {"frame_step": [0], "frame_u_mean": [0.1]}
        self._u_calibration_pending_u = _Tensor(4)
        self._u_calibration_pending_h = _Tensor(5)
        self._u_calibration_last_state = _Tensor(6)
        self._u_calibration_final_state = None
        self._u_calibration_oracle_window = 1
        self._u_calibration_queue = ["safe"]


def _snapshot(model: _Model) -> StructuralPreState:
    return StructuralPreState.capture(_Tensor(10), _Tensor(11), _Tensor(12), _Tensor(13), _Tensor(14), False, model, torch=None)


def test_structural_witness_accepts_rebind_restore_and_records_lengths() -> None:
    model = _Model()
    snapshot = _snapshot(model)
    witness = StructuralIdentityWitness.capture(snapshot, model)
    model.update_pressure = _Tensor(99)
    model._u_calibration_trace["frame_step"].append(1)
    model._u_calibration_queue.append("candidate")
    witness.reject_in_place_mutation(snapshot)
    restored = snapshot.restore(model, torch=None)
    witness.verify_restored(restored, model)
    assert witness.timeline_evidence()["queue_length"] == 1


def test_structural_witness_rejects_in_place_pre_state_mutation() -> None:
    model = _Model()
    snapshot = _snapshot(model)
    witness = StructuralIdentityWitness.capture(snapshot, model)
    snapshot.local_state["state_feat"]._version += 1
    with pytest.raises(StructuralWitnessError, match="mutated in place"):
        witness.reject_in_place_mutation(snapshot)
