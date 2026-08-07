from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from stateguard3r.recal3r_state_v3 import (
    CausalHoldDiscardController,
    CausalHoldReplayController,
    StateClosure,
    StateTransaction,
    capture_model_state,
    verify_pinned_lighter_source,
)


class _FakeTorch:
    def __init__(self) -> None:
        self.cpu_rng = np.asarray([1, 2, 3], dtype=np.uint8)
        self.cuda = SimpleNamespace(is_available=lambda: False)

    def get_rng_state(self) -> np.ndarray:
        return self.cpu_rng.copy()

    def set_rng_state(self, value: np.ndarray) -> None:
        self.cpu_rng = value.copy()


def _model() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(model_update_type="recal3r"),
        update_pressure=np.asarray([0.2, 0.3]),
        recal3r_state0=np.asarray([[1.0, 2.0]]),
        _recal3r_sequence_age=np.asarray([4], dtype=np.int64),
        _u_calibration_trace={"frame_step": [np.asarray([4], dtype=np.int32)]},
        _u_calibration_pending_u=np.asarray([0.5]),
        _u_calibration_pending_h=np.asarray([0.6]),
        _u_calibration_last_state=np.asarray([[7.0]]),
        _u_calibration_final_state=None,
        _u_calibration_oracle_window=1,
        _u_calibration_queue=[{"frame_idx": 4, "state": np.asarray([9.0])}],
    )


def _closure(value: float) -> StateClosure:
    return StateClosure.capture(
        np.asarray([[value]], dtype=np.float32),
        np.asarray([[1, 2]], dtype=np.int64),
        np.asarray([[0.0]], dtype=np.float32),
        np.asarray([[value + 10]], dtype=np.float32),
        np.asarray([[10.0]], dtype=np.float32),
    )


def _transaction(frame_id: int) -> StateTransaction:
    model = _model()
    return StateTransaction(
        frame_id=frame_id,
        pre_state=_closure(float(frame_id)),
        post_state=_closure(float(frame_id + 1)),
        pre_model=capture_model_state(model),
        post_model=capture_model_state(model),
        pre_reset_mask=False,
        post_reset_mask=False,
    )


def test_model_snapshot_restores_full_mutable_closure_and_rng() -> None:
    model = _model()
    fake_torch = _FakeTorch()
    before = capture_model_state(model, torch=fake_torch)
    before_digest = before.digest()

    # This emulates the upstream ReCal3R path: update=False suppresses a
    # state blend but does not itself stop pressure/age/trace mutation.
    update = False
    assert update is False
    model.update_pressure += 1.0
    model._recal3r_sequence_age += 1
    model._u_calibration_trace["frame_step"].append(np.asarray([5], dtype=np.int32))
    model._u_calibration_queue[0]["state"][0] = -1.0
    model.config.model_update_type = "cut3r"
    fake_torch.cpu_rng[:] = 0
    del model._u_calibration_pending_h

    assert not np.array_equal(model.update_pressure, np.asarray([0.2, 0.3]))
    assert not np.array_equal(model._recal3r_sequence_age, np.asarray([4]))
    before.restore(model, torch=fake_torch)

    after = capture_model_state(model, torch=fake_torch)
    assert after.digest() == before_digest
    assert model.config.model_update_type == "recal3r"
    assert np.array_equal(fake_torch.cpu_rng, np.asarray([1, 2, 3], dtype=np.uint8))


def test_state_closure_is_alias_free_and_restorable() -> None:
    original = np.asarray([[2.0]], dtype=np.float32)
    closure = StateClosure.capture(
        original,
        np.asarray([[0, 1]], dtype=np.int64),
        np.asarray([[3.0]], dtype=np.float32),
        np.asarray([[4.0]], dtype=np.float32),
        np.asarray([[5.0]], dtype=np.float32),
    )
    original[0, 0] = 99.0
    restored = closure.restored()

    assert closure.state_feat[0, 0] == 2.0
    assert restored[0][0, 0] == 2.0
    restored[0][0, 0] = -1.0
    assert closure.state_feat[0, 0] == 2.0


def test_causal_hold_controller_holds_then_replays_only_after_next_clear_score() -> None:
    controller = CausalHoldReplayController([False, True, False, False], max_hold=3)
    first = _transaction(0)
    second = _transaction(1)
    held = _transaction(2)
    final = _transaction(3)

    assert controller.take_replay(0) is None
    assert controller.decide_candidate(first).action == "commit"
    assert controller.decide_candidate(second).action == "commit"
    decision = controller.decide_candidate(held)
    assert (decision.action, decision.prior_alarm, decision.reason) == (
        "hold",
        True,
        "prior_detector_alarm",
    )
    assert controller.take_replay(2) is None
    assert controller.take_replay(3) == held
    assert controller.decide_candidate(final).action == "commit"
    assert controller.drain() == []


def test_hold_controller_enforces_the_declared_three_update_bound() -> None:
    controller = CausalHoldReplayController([False, True, True, True, True, True], max_hold=3)
    assert controller.decide_candidate(_transaction(0)).action == "commit"
    assert controller.decide_candidate(_transaction(1)).action == "commit"
    assert controller.decide_candidate(_transaction(2)).action == "hold"
    assert controller.decide_candidate(_transaction(3)).action == "hold"
    assert controller.decide_candidate(_transaction(4)).action == "hold"
    bounded = controller.decide_candidate(_transaction(5))
    assert (bounded.action, bounded.reason) == ("commit", "hold_bound_reached")
    assert controller.drain() == []


def test_discard_controller_releases_a_held_candidate_without_replay() -> None:
    controller = CausalHoldDiscardController([False, True, False, False], max_hold=3)
    assert controller.decide_candidate(_transaction(0)).action == "commit"
    assert controller.decide_candidate(_transaction(1)).action == "commit"
    held = _transaction(2)
    assert controller.decide_candidate(held).action == "hold"
    assert controller.take_discard(2) is None
    assert controller.take_discard(3) == held
    assert controller.drain() == []


def test_pinned_lighter_source_guard_binds_exact_recal3r_method() -> None:
    provenance = verify_pinned_lighter_source()

    assert provenance["baseline_tracked_worktree_clean"] is True
    assert provenance["line_range"] == [1661, 1833]
    assert len(provenance["source_sha256"]) == 64
    assert len(provenance["line_sha256"]) == 64
