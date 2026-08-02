from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.derive_detector_v2_development import (
    DetectorV2DevelopmentError,
    EXPECTED_RUN_IDS,
    cross_validation_folds,
    merge_online_overlap,
)
from stateguard3r.health import HealthFrame


def _health_records() -> list[dict[str, object]]:
    return [
        HealthFrame(
            frame_id=frame_id,
            timestamp=float(frame_id),
            geometric_residual=0.1,
            pose_jump=None if frame_id == 0 else 0.1,
            global_state_delta=None if frame_id == 0 else 0.1,
            reliability=None if frame_id == 0 else 0.9,
        ).to_dict()
        for frame_id in range(30)
    ]


def test_merge_online_overlap_is_copying_causal_and_rejects_extra_online_fields() -> None:
    health = _health_records()
    original = deepcopy(health)
    online = [
        {"frame_id": frame_id, "overlap": None if frame_id == 0 else 0.75}
        for frame_id in range(30)
    ]

    merged = merge_online_overlap(health, online, run_id="example")

    assert health == original
    assert merged[0]["overlap"] is None
    assert [row["overlap"] for row in merged[1:]] == [0.75] * 29
    assert all(set(row) == set(health[0]) for row in merged)

    leaking = deepcopy(online)
    leaking[1]["label"] = 1
    with pytest.raises(DetectorV2DevelopmentError, match="frame_id/overlap"):
        merge_online_overlap(health, leaking, run_id="example")


def test_cross_validation_folds_hold_out_complete_type_or_original_window() -> None:
    regions = {
        run_id: {
            "dataset_split": "development"
            if run_id.startswith("development-")
            else "holdout",
            "corruption_type": (
                "dynamic_occlusion"
                if run_id.endswith("dynamic")
                else "low_overlap_jump"
                if run_id.endswith("low")
                else "wrong_order_segment"
            ),
        }
        for run_id in EXPECTED_RUN_IDS
    }

    folds = cross_validation_folds(regions)

    assert len(folds) == 5
    assert {fold["kind"] for fold in folds} == {
        "corruption_type",
        "original_v1_window",
    }
    for fold in folds:
        assert set(fold["train_run_ids"]).isdisjoint(fold["test_run_ids"])
        assert set(fold["train_run_ids"]) | set(fold["test_run_ids"]) == EXPECTED_RUN_IDS
        assert fold["train_run_ids"]
        assert fold["test_run_ids"]
