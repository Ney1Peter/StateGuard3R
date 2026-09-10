from __future__ import annotations

import ast
from pathlib import Path

from stateguard3r.state_triage_stage0p8 import decide_event, decide_frame


def _row(**updates: float | None | str) -> dict[str, float | None | str]:
    value: dict[str, float | None | str] = {"coverage_ratio": 0.8, "sharpness": 0.2, "clipped_fraction": 0.0, "pose_jump_z": 0.0, "native_geometric_residual_z": 0.0}
    value.update(updates)
    return value


def test_primary_and_coverage_ablation_differ_only_at_predeclared_interaction() -> None:
    activated = _row(sharpness=0.2, clipped_fraction=0.75, coverage_ratio=0.1)
    assert decide_frame("T0.8", activated).label == "transient_local_content"
    assert decide_frame("C0.8", activated).label == "normal_novelty"
    assert decide_frame("S0.8", activated).label == "normal_novelty"
    assert decide_frame("T0.8", _row(sharpness=0.0, clipped_fraction=0.75, coverage_ratio=0.1)).label == "bad_observation"
    assert decide_frame("T0.8", _row(coverage_ratio=0.1)).label == "normal_novelty"
    assert decide_frame("T0.8", _row(pose_jump_z=3.0)).label == "registration_or_order_fault"


def test_event_uses_first_current_row_decision() -> None:
    rows = [_row(), _row(sharpness=0.2, clipped_fraction=0.75, coverage_ratio=0.1), _row(coverage_ratio=0.1)]
    typed = decide_event(rows, method="T0.8", start_frame=0, end_frame=2)
    coverage = decide_event(rows, method="C0.8", start_frame=0, end_frame=2)
    assert (typed["label"], typed["decision_frame"]) == ("transient_local_content", 1)
    assert (coverage["label"], coverage["decision_frame"]) == ("normal_novelty", 1)


def test_decision_ignores_recipe_and_source_metadata() -> None:
    row = _row(clipped_fraction=0.75, coverage_ratio=0.1)
    assert decide_frame("T0.8", row).to_dict() == decide_frame("T0.8", {**row, "cause": "normal_novelty", "source_path": "/not/read", "source_index": 0}).to_dict()


def test_pure_module_has_no_model_or_recovery_imports() -> None:
    path = Path(__file__).parents[1] / "src" / "stateguard3r" / "state_triage_stage0p8.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported.update(node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module)
    assert not any(token in name.lower() for name in imported for token in ("torch", "recal3r", "recovery", "rollback", "quarantine"))
