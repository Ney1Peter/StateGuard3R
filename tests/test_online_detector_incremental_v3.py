from __future__ import annotations

from dataclasses import replace

import pytest

from stateguard3r.health import HealthFrame
from stateguard3r.online_detector_incremental_v3 import IncrementalOnlinePrefixDetector
from stateguard3r.online_detector_v2 import FrozenDetectorV3Config, OnlinePrefixDetector


def _config() -> FrozenDetectorV3Config:
    return FrozenDetectorV3Config.from_mapping(
        {
            "continuous_scoring": {"window": 5, "min_history": 3, "max_z": 10.0, "master_seed": 0},
            "scale_floors": {"geometric_residual": 0.01, "pose_jump": 0.01, "update_magnitude": 0.01, "overlap": 0.01, "reliability": 0.01},
            "thresholds": {"combined": 1.0, "random": 0.9, "reliability_only": 1.0, "update_magnitude_only": 1.0},
            "hybrid_rule": "continuous_score_gte_threshold_or_timestamp_order_violation",
        }
    )


def _captures(length: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    timestamps = [str(index + 1) for index in range(length)]
    if length > 8:
        timestamps[8] = "3.5"  # a real current-only timestamp-order alarm
    return (
        [
            {"frame_id": index, "rgb_capture_timestamp": value, "rgb_capture_timestamp_text": value, "rgb_txt_physical_line": index + 1, "rgb_path_sha256": f"{index:064x}"}
            for index, value in enumerate(timestamps)
        ],
        {"parser_version": "tum-rgb-txt-decimal-strict.v1", "rgb_txt_path": "/readonly/rgb.txt", "rgb_txt_sha256": "a" * 64, "dataset_root": "/readonly", "input_contract": "ordered_rgb_paths_only_no_source_index_gt_label_or_event_metadata"},
    )


def _record(frame_id: int) -> HealthFrame:
    return HealthFrame(
        frame_id=frame_id,
        overlap=None if frame_id == 0 else 0.7 - 0.01 * (frame_id % 3),
        pose_jump=None if frame_id == 0 else 0.01 * frame_id,
        geometric_residual=0.02 + 0.001 * frame_id,
        global_state_delta=0.1 + 0.01 * frame_id,
        uncertainty_u=0.1 + 0.01 * (frame_id % 4),
    )


def _comparable(decision: object) -> tuple[object, ...]:
    return tuple(getattr(decision, name) for name in ("frame_id", "continuous_score", "hybrid_score", "continuous_alarm", "timestamp_order_alarm", "hybrid_alarm", "finite_component_count"))


def test_incremental_detector_matches_reference_on_each_finalized_prefix_and_is_bounded() -> None:
    captures, provenance = _captures(12)
    reference = OnlinePrefixDetector(_config(), captures=captures, capture_provenance=provenance)
    incremental = IncrementalOnlinePrefixDetector(_config(), captures=captures, capture_provenance=provenance)
    for frame_id in range(12):
        record = _record(frame_id)
        expected = reference.observe(record)
        actual = incremental.observe(record)
        assert _comparable(actual) == pytest.approx(_comparable(expected))
        if frame_id == 6:
            reference.quarantine(record)
            incremental.quarantine(record)
        else:
            reference.commit(record)
            incremental.commit(record)
        assert incremental.retained_row_count <= 5
    assert incremental.history_frame_ids == (7, 8, 9, 10, 11)


def test_quarantined_health_cannot_select_future_signal_or_escape_overlap_placeholder() -> None:
    captures, provenance = _captures(3)
    detector = IncrementalOnlinePrefixDetector(_config(), captures=captures, capture_provenance=provenance)
    first = replace(_record(0), global_state_delta=None)
    detector.commit(first) if detector.observe(first) else None
    candidate = replace(_record(1), global_state_delta=999.0)
    detector.observe(candidate)
    detector.quarantine(candidate)
    assert detector._history[-1] == {"frame_id": 1, "overlap": candidate.overlap}  # type: ignore[attr-defined]
    assert detector._seen_aliases["update_magnitude"]["global_state_delta"] is False  # type: ignore[attr-defined]


def test_future_capture_mutation_cannot_change_previous_incremental_decision() -> None:
    captures, provenance = _captures(6)
    changed = [dict(row) for row in captures]
    changed[-1]["rgb_capture_timestamp"] = "0.1"
    changed[-1]["rgb_capture_timestamp_text"] = "0.1"
    left = IncrementalOnlinePrefixDetector(_config(), captures=captures, capture_provenance=provenance)
    right = IncrementalOnlinePrefixDetector(_config(), captures=changed, capture_provenance=provenance)
    for frame_id in range(5):
        record = _record(frame_id)
        assert _comparable(left.observe(record)) == _comparable(right.observe(record))
        left.commit(record)
        right.commit(record)
