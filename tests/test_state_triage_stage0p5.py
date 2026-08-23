from __future__ import annotations

import ast
from pathlib import Path

import pytest

from stateguard3r.state_triage_stage0p5 import Stage0p5DecisionError, cause_metrics, decide_event, decide_frame


def _row(**updates: float | None | str) -> dict[str, float | None | str]:
    value: dict[str, float | None | str] = {
        "coverage_ratio": 0.8,
        "sharpness": 0.2,
        "clipped_fraction": 0.0,
        "pose_jump_z": 0.0,
        "native_geometric_residual_z": 0.0,
    }
    value.update(updates)
    return value


def test_typed_priority_is_quality_then_coverage_then_local_then_registration() -> None:
    # The first assertion is the independently tested Stage 0.5 hypothesis:
    # a global collapse cannot be hidden by its induced coverage loss.
    assert decide_frame("T0.5", _row(sharpness=0.0, coverage_ratio=0.0)).label == "bad_observation"
    assert decide_frame("T0.5", _row(coverage_ratio=0.2)).label == "normal_novelty"
    assert decide_frame("T0.5", _row(clipped_fraction=0.2)).label == "transient_local_content"
    assert decide_frame("T0.5", _row(pose_jump_z=3.0)).label == "registration_or_order_fault"
    assert decide_frame("T0.5", _row(native_geometric_residual_z=3.0)).label == "registration_or_order_fault"
    assert decide_frame("T0.5", _row()).label == "unresolved"


def test_scalar_comparator_is_fixed_and_distinct() -> None:
    assert decide_frame("S0.5", _row(coverage_ratio=0.2)).label == "normal_novelty"
    assert decide_frame("S0.5", _row(native_geometric_residual_z=3.0)).label == "registration_or_order_fault"
    assert decide_frame("S0.5", _row(sharpness=0.0)).label == "unresolved"


def test_event_is_first_non_unresolved_and_never_reads_future_rows() -> None:
    rows = [_row(), _row(clipped_fraction=0.3), _row(sharpness=0.0)]
    event = decide_event(rows, method="T0.5", start_frame=0, end_frame=2)
    assert event["label"] == "transient_local_content"
    assert event["decision_frame"] == 1
    assert event["detection_delay_frames"] == 1
    assert decide_event([_row()], method="T0.5", start_frame=0, end_frame=0)["label"] == "unresolved"
    with pytest.raises(Stage0p5DecisionError, match="range"):
        decide_event(rows, method="T0.5", start_frame=0, end_frame=3)


def test_decision_does_not_consume_recipe_or_source_metadata() -> None:
    plain = _row(clipped_fraction=0.3)
    decorated = {**plain, "cause": "bad_observation", "source_sequence": "leak-attempt", "frame_path": "/not/read"}
    assert decide_frame("T0.5", plain).to_dict() == decide_frame("T0.5", decorated).to_dict()


def test_metrics_keep_unsafe_confusions_as_event_counts() -> None:
    events = [
        {"truth": "registration_or_order_fault", "prediction": "transient_local_content", "detection_delay_frames": 0},
        {"truth": "bad_observation", "prediction": "bad_observation", "detection_delay_frames": 0},
        {"truth": "transient_local_content", "prediction": "unresolved", "detection_delay_frames": None},
        {"truth": "normal_novelty", "prediction": "normal_novelty", "detection_delay_frames": 0},
    ]
    metrics = cause_metrics(events)
    assert metrics["unsafe_confusion_counts"]["registration_as_transient"] == 1
    assert metrics["normal_novelty_false_reject"] == 0.0
    assert metrics["per_cause"]["transient_local_content"]["recall"] == 0.0


def test_pure_decision_module_has_no_model_or_recovery_imports() -> None:
    path = Path(__file__).parents[1] / "src" / "stateguard3r" / "state_triage_stage0p5.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    forbidden = ("torch", "recal3r", "recovery", "rollback", "quarantine", "state_triage_stage0")
    assert not any(any(token in name.lower() for token in forbidden) for name in imported)
