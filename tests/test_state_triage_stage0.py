from __future__ import annotations

import pytest

from stateguard3r.state_triage_stage0 import (
    DecisionThresholds,
    Stage0DecisionError,
    cause_metrics,
    decide_event,
    decide_frame,
    sequence_bootstrap_delta,
)


def _thresholds() -> DecisionThresholds:
    return DecisionThresholds(
        coverage_novelty_max=0.25,
        scalar_s0_native_geometric_residual_z_min=3.0,
        scalar_s1_normalized_residual_median_min=0.2,
        quality_sharpness_max=0.01,
        quality_low_light_fraction_min=0.8,
        quality_clipped_fraction_min=0.8,
        quality_confidence_disagreement_min=2.0,
        registration_pose_jump_z_min=3.0,
        registration_geometric_residual_z_min=3.0,
        registration_globality_min=0.5,
        transient_residual_high_fraction_min=0.2,
        transient_locality_min=0.7,
    )


def _row(**updates: float | None) -> dict[str, float | None]:
    value: dict[str, float | None] = {
        "coverage_ratio": 0.8,
        "native_geometric_residual_z": 0.0,
        "normalized_residual_median": 0.01,
        "sharpness": 0.1,
        "low_light_fraction": 0.0,
        "clipped_fraction": 0.0,
        "confidence_disagreement_mean": 0.1,
        "pose_jump_z": 0.0,
        "residual_globality": 0.3,
        "residual_high_fraction": 0.05,
        "residual_locality": 0.7,
    }
    value.update(updates)
    return value


def test_frozen_rule_is_coverage_first_then_quality_global_and_local() -> None:
    thresholds = _thresholds()
    assert decide_frame("T0", _row(coverage_ratio=0.2), thresholds).label == "normal_novelty"
    assert decide_frame("T0", _row(sharpness=0.005), thresholds).label == "bad_observation"
    assert decide_frame("T0", _row(native_geometric_residual_z=4.0, residual_globality=0.6), thresholds).label == "registration_or_order_fault"
    assert decide_frame("T0", _row(residual_high_fraction=0.3, residual_locality=0.8), thresholds).label == "transient_local_content"
    assert decide_frame("T0", _row(), thresholds).label == "unresolved"
    assert decide_frame("S0", _row(native_geometric_residual_z=4.0), thresholds).label == "registration_or_order_fault"
    assert decide_frame("S1", _row(normalized_residual_median=0.3), thresholds).label == "registration_or_order_fault"


def test_offline_upper_bound_is_explicit_and_never_used_by_t0() -> None:
    thresholds = _thresholds()
    rows = [_row(residual_high_fraction=0.3, residual_locality=0.8)] + [_row(residual_high_fraction=0.01) for _ in range(3)]
    causal = decide_event("T0", rows, start_frame=0, end_frame=0, thresholds=thresholds, future_k=2)
    offline = decide_event("T0+K", rows, start_frame=0, end_frame=0, thresholds=thresholds, future_k=2)
    assert causal["label"] == "transient_local_content"
    assert offline["label"] == "transient_local_content"
    assert offline["reason"] == "offline_future_window_clears"
    rows[1]["residual_high_fraction"] = 0.3
    assert decide_event("T0+K", rows, start_frame=0, end_frame=0, thresholds=thresholds, future_k=2)["label"] == "unresolved"


def test_metrics_keep_unresolved_and_unsafe_confusions_visible() -> None:
    events = [
        {"truth": "registration_or_order_fault", "prediction": "transient_local_content", "posterior": {"registration_or_order_fault": 0.1, "bad_observation": 0.1, "transient_local_content": 0.7, "normal_novelty": 0.1}, "detection_delay_frames": 1},
        {"truth": "bad_observation", "prediction": "bad_observation", "posterior": {"registration_or_order_fault": 0.1, "bad_observation": 0.7, "transient_local_content": 0.1, "normal_novelty": 0.1}, "detection_delay_frames": 0},
        {"truth": "transient_local_content", "prediction": "unresolved", "posterior": {"registration_or_order_fault": 0.25, "bad_observation": 0.25, "transient_local_content": 0.25, "normal_novelty": 0.25}, "detection_delay_frames": None},
        {"truth": "normal_novelty", "prediction": "registration_or_order_fault", "posterior": {"registration_or_order_fault": 0.7, "bad_observation": 0.1, "transient_local_content": 0.1, "normal_novelty": 0.1}, "detection_delay_frames": 0},
    ]
    metrics = cause_metrics(events)
    assert metrics["unresolved_rate"] == pytest.approx(0.25)
    assert metrics["unsafe_confusions"]["registration_as_transient"] == pytest.approx(1.0)
    assert metrics["unsafe_confusions"]["normal_novelty_false_reject"] == pytest.approx(1.0)
    assert metrics["per_cause"]["transient_local_content"]["recall"] == 0.0


def test_bootstrap_resamples_sequences_not_frames() -> None:
    truth = ("registration_or_order_fault", "bad_observation", "transient_local_content", "normal_novelty")
    typed, scalar = [], []
    for index, label in enumerate(truth):
        sequence = "sequence-a" if index < 2 else "sequence-b"
        base = {"event_id": str(index), "source_sequence": sequence, "truth": label, "prediction": label, "posterior": {cause: 0.7 if cause == label else 0.1 for cause in truth}, "detection_delay_frames": 0}
        typed.append(base)
        scalar.append({**base, "prediction": "unresolved", "posterior": {cause: 0.25 for cause in truth}})
    result = sequence_bootstrap_delta(typed, scalar, samples=100, seed=7)
    assert result["lower_95"] > 0.0
    with pytest.raises(Stage0DecisionError, match="different event IDs"):
        sequence_bootstrap_delta(typed, scalar[:-1], samples=1)
