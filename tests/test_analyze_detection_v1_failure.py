from __future__ import annotations

from scripts.analyze_detection_v1_failure import _positions_for_method


def test_position_summary_keeps_startup_and_washout_separate() -> None:
    run = {
        "run_id": "tiny",
        "primary_labels": [0, 0, 1, 0, 0, 0],
        "evaluation_mask": [True, True, True, False, False, True],
        "washout_mask": [False, False, False, True, True, False],
        "event_start": 2,
        "event_end": 2,
    }
    method = {
        "scores": [0.9, 0.1, 0.8, 0.7, 0.6, 0.9],
        "metrics": {"threshold": 0.5},
    }

    actual = _positions_for_method(run, method)

    assert actual["startup_false_positive_positions"] == [0]
    assert actual["true_positive_positions"] == [2]
    assert actual["primary_false_positive_positions"] == [0, 5]
    assert actual["false_negative_positions"] == []
    assert actual["recovery_boundary_position"] == 3
    assert actual["recovery_boundary_alarm"] is True
    assert actual["washout_alarm_positions"] == [3, 4]
