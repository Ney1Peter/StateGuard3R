from __future__ import annotations

import ast
from pathlib import Path

from stateguard3r.state_triage_stage0p6 import decide_event, decide_frame


def _row(**updates: float | None | str) -> dict[str, float | None | str]:
    row: dict[str, float | None | str] = {"coverage_ratio": 0.8, "sharpness": 0.2, "clipped_fraction": 0.0, "pose_jump_z": 0.0, "native_geometric_residual_z": 0.0}
    row.update(updates)
    return row


def test_typed_priority_is_quality_then_local_then_coverage_then_registration() -> None:
    assert decide_frame("T0.6", _row(sharpness=0.0, clipped_fraction=0.9, coverage_ratio=0.0)).label == "bad_observation"
    # This is the new, independently tested Stage 0.6 hypothesis.
    assert decide_frame("T0.6", _row(clipped_fraction=0.2, coverage_ratio=0.0)).label == "transient_local_content"
    assert decide_frame("T0.6", _row(coverage_ratio=0.2)).label == "normal_novelty"
    assert decide_frame("T0.6", _row(pose_jump_z=3.0)).label == "registration_or_order_fault"
    assert decide_frame("T0.6", _row()).label == "unresolved"


def test_scalar_comparator_and_event_aggregation_remain_fixed() -> None:
    assert decide_frame("S0.6", _row(clipped_fraction=0.9, coverage_ratio=0.2)).label == "normal_novelty"
    assert decide_frame("S0.6", _row(native_geometric_residual_z=3.0)).label == "registration_or_order_fault"
    event = decide_event([_row(), _row(clipped_fraction=0.3), _row(coverage_ratio=0.2)], method="T0.6", start_frame=0, end_frame=2)
    assert (event["label"], event["decision_frame"], event["detection_delay_frames"]) == ("transient_local_content", 1, 1)


def test_decision_ignores_label_and_source_metadata() -> None:
    row = _row(clipped_fraction=0.3, coverage_ratio=0.1)
    assert decide_frame("T0.6", row).to_dict() == decide_frame("T0.6", {**row, "cause": "normal_novelty", "source_sequence": "attempted-leak", "path": "/not/read"}).to_dict()


def test_pure_decision_has_no_model_or_recovery_import() -> None:
    path = Path(__file__).parents[1] / "src" / "stateguard3r" / "state_triage_stage0p6.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported.update(node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module)
    assert not any(token in name.lower() for name in imported for token in ("torch", "recal3r", "recovery", "rollback", "quarantine"))
