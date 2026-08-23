from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from scripts.run_state_triage_v2_stage0 import _observe


def _prediction() -> dict[str, np.ndarray]:
    points = np.zeros((1, 4, 4, 3), dtype=np.float64)
    points[..., 2] = 1.0
    return {"pts3d_in_self_view": points, "pts3d_in_other_view": points.copy(), "conf": np.ones((1, 4, 4)), "conf_self": np.ones((1, 4, 4))}


class _Health:
    def to_dict(self):
        return {"pose_jump": 0.0, "geometric_residual": 0.0, "uncertainty_u": 0.2}


def test_observe_uses_one_current_prediction_at_a_time() -> None:
    prediction = _prediction()
    views = [{"img": np.zeros((1, 3, 4, 4), dtype=np.float64)}, {"img": np.zeros((1, 3, 4, 4), dtype=np.float64)}]
    trajectory = [{"camera_to_reference": np.eye(4).tolist()}, {"camera_to_reference": np.eye(4).tolist()}]

    rows = _observe([prediction, prediction], views, trajectory, [_Health(), _Health()])

    assert [row["frame_id"] for row in rows] == [0, 1]
    assert rows[0]["reference_anchor_count_before"] == 0
    assert rows[1]["reference_anchor_count_before"] > 0


def test_runner_source_does_not_import_recovery_or_model_write_modules() -> None:
    path = Path(__file__).parents[1] / "scripts" / "run_state_triage_v2_stage0.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    forbidden = ("recovery", "quarantine", "transactional", "v14", "v15", "v16", "v17", "v18", "v19", "v20")
    assert not any(any(token in name for token in forbidden) for name in imported)
