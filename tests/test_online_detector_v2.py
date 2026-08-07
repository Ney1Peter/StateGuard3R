from __future__ import annotations

import pytest

from stateguard3r.health import HealthFrame
from stateguard3r.online_detector_v2 import (
    FrozenDetectorV3Config,
    OnlineDetectorError,
    OnlinePrefixDetector,
)


def _config() -> FrozenDetectorV3Config:
    return FrozenDetectorV3Config.from_mapping(
        {
            "continuous_scoring": {"window": 5, "min_history": 3, "max_z": 10.0, "master_seed": 0},
            "scale_floors": {
                "geometric_residual": 0.01,
                "pose_jump": 0.01,
                "update_magnitude": 0.01,
                "overlap": 0.01,
                "reliability": 0.01,
            },
            "thresholds": {"combined": 1.0, "random": 0.9, "reliability_only": 1.0, "update_magnitude_only": 1.0},
            "hybrid_rule": "continuous_score_gte_threshold_or_timestamp_order_violation",
        }
    )


def _captures(values: list[str]) -> tuple[list[dict[str, object]], dict[str, object]]:
    return (
        [
            {
                "frame_id": index,
                "rgb_capture_timestamp": value,
                "rgb_capture_timestamp_text": value,
                "rgb_txt_physical_line": index + 1,
                "rgb_path_sha256": f"{index:064x}",
            }
            for index, value in enumerate(values)
        ],
        {
            "parser_version": "tum-rgb-txt-decimal-strict.v1",
            "rgb_txt_path": "/readonly/rgb.txt",
            "rgb_txt_sha256": "a" * 64,
            "dataset_root": "/readonly",
            "input_contract": "ordered_rgb_paths_only_no_source_index_gt_label_or_event_metadata",
        },
    )


def _record(frame_id: int, *, overlap: float | None) -> HealthFrame:
    return HealthFrame(
        frame_id=frame_id,
        overlap=overlap,
        pose_jump=0.01 * frame_id if frame_id else None,
        geometric_residual=0.02,
        global_state_delta=0.01 * frame_id if frame_id else None,
        uncertainty_u=0.1,
    )


def test_prefix_decision_is_invariant_to_future_capture_records() -> None:
    first_captures, provenance = _captures(["1.0", "2.0", "3.0", "4.0"])
    changed_future, _ = _captures(["1.0", "2.0", "3.0", "0.1"])
    left = OnlinePrefixDetector(_config(), captures=first_captures, capture_provenance=provenance)
    right = OnlinePrefixDetector(_config(), captures=changed_future, capture_provenance=provenance)
    for frame_id in range(3):
        record = _record(frame_id, overlap=None if frame_id == 0 else 0.8)
        left_decision = left.observe(record)
        right_decision = right.observe(record)
        assert left_decision == right_decision
        left.commit(record)
        right.commit(record)


def test_timestamp_channel_is_current_frame_and_quarantine_history_is_sanitized() -> None:
    captures, provenance = _captures(["1.0", "2.0", "1.5"])
    detector = OnlinePrefixDetector(_config(), captures=captures, capture_provenance=provenance)
    first = _record(0, overlap=None)
    detector.commit(first) if detector.observe(first) else None
    second = _record(1, overlap=0.8)
    detector.observe(second)
    detector.quarantine(second)
    assert detector.history_frame_ids == (0, 1)
    third = _record(2, overlap=0.7)
    decision = detector.observe(third)
    assert decision.timestamp_order_alarm is True
    # The quarantined model health did not survive in detector history; only
    # the current RGB overlap is retained as a safe placeholder.
    assert detector._history[1] == {"frame_id": 1, "overlap": 0.8}  # type: ignore[attr-defined]


def test_rejects_forbidden_or_non_prefix_health_mapping() -> None:
    captures, provenance = _captures(["1.0", "2.0"])
    detector = OnlinePrefixDetector(_config(), captures=captures, capture_provenance=provenance)
    with pytest.raises(OnlineDetectorError, match="forbidden"):
        detector.observe({"frame_id": 0, "overlap": None, "groundtruth": {}})
    with pytest.raises(OnlineDetectorError, match="causal prefix"):
        detector.observe({"frame_id": 1, "overlap": 0.7})


def test_requires_finalization_before_next_observation() -> None:
    captures, provenance = _captures(["1.0", "2.0"])
    detector = OnlinePrefixDetector(_config(), captures=captures, capture_provenance=provenance)
    detector.observe(_record(0, overlap=None))
    with pytest.raises(OnlineDetectorError, match="not finalized"):
        detector.observe(_record(0, overlap=None))
