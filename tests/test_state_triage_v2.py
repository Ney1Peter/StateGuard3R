from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from stateguard3r.state_triage_v2 import (
    EvidenceConfig,
    EvidenceObserver,
    StateTriageEvidenceError,
)


def _prediction(*, offset: float = 0.0, local_outlier: bool = False) -> dict[str, np.ndarray]:
    height, width = 8, 8
    y, x = np.meshgrid(np.arange(height, dtype=np.float64), np.arange(width, dtype=np.float64), indexing="ij")
    points = np.stack((x * 0.02 + offset, y * 0.02, np.ones_like(x)), axis=-1)[None]
    other = points.copy()
    if local_outlier:
        # The observer's 2x3 test grid samples the image corners and centre;
        # place one deterministic local disagreement on a sampled anchor.
        other[0, 0, 0, 0] += 0.4
    return {
        "pts3d_in_self_view": points,
        "pts3d_in_other_view": other,
        "conf": np.full((1, height, width), 2.0, dtype=np.float64),
        "conf_self": np.full((1, height, width), 1.5, dtype=np.float64),
    }


def _rgb(*, flat: bool = False) -> np.ndarray:
    height, width = 8, 8
    if flat:
        return np.zeros((1, 3, height, width), dtype=np.float64)
    image = np.indices((height, width)).sum(axis=0) % 2
    rgb = np.stack((image, 1.0 - image, image), axis=0).astype(np.float64)
    return rgb[None] * 2.0 - 1.0


def _camera() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def _observe(observer: EvidenceObserver, frame_id: int, *, offset: float = 0.0, local_outlier: bool = False, flat: bool = False):
    return observer.observe(
        frame_id=frame_id,
        prediction=_prediction(offset=offset, local_outlier=local_outlier),
        camera_to_reference=_camera(),
        model_ready_rgb=_rgb(flat=flat),
        native_health={"pose_jump": 0.01 + frame_id * 0.01, "geometric_residual": 0.02, "uncertainty_u": 0.2},
    )


def test_observer_uses_only_previous_anchors_and_enforces_bound() -> None:
    observer = EvidenceObserver(
        EvidenceConfig(grid_rows=2, grid_columns=3, max_anchor_history=7, max_conflict_anchor_history=4)
    )

    first = _observe(observer, 0)
    second = _observe(observer, 1)
    third = _observe(observer, 2, local_outlier=True)

    assert first.reference_anchor_count_before == 0
    assert first.coverage_ratio == 0.0
    assert second.reference_anchor_count_before == 6
    assert second.coverage_ratio == pytest.approx(1.0)
    assert third.reference_anchor_count_before == 7
    assert third.residual_locality > 0.0
    assert observer.anchor_count == 7
    assert observer.conflict_anchor_count <= 4


def test_observer_prefix_rows_are_invariant_to_different_suffixes() -> None:
    config = EvidenceConfig(grid_rows=2, grid_columns=3)
    left = EvidenceObserver(config)
    right = EvidenceObserver(config)

    left_prefix = [_observe(left, 0), _observe(left, 1)]
    right_prefix = [_observe(right, 0), _observe(right, 1)]
    _observe(left, 2, offset=0.0, local_outlier=True, flat=True)
    _observe(right, 2, offset=10.0, local_outlier=False, flat=False)

    left_json = [json.dumps(row.to_dict(), sort_keys=True, allow_nan=False) for row in left_prefix]
    right_json = [json.dumps(row.to_dict(), sort_keys=True, allow_nan=False) for row in right_prefix]
    assert left_json == right_json


def test_observer_rejects_future_like_frame_order_and_bad_inputs() -> None:
    observer = EvidenceObserver(EvidenceConfig(grid_rows=2, grid_columns=3))
    with pytest.raises(StateTriageEvidenceError, match="next consecutive"):
        _observe(observer, 1)

    bad = _prediction()
    bad["conf"] = np.full((1, 8, 8), np.nan)
    with pytest.raises(StateTriageEvidenceError, match="finite"):
        observer.observe(
            frame_id=0,
            prediction=bad,
            camera_to_reference=_camera(),
            model_ready_rgb=_rgb(),
        )


def test_observer_does_not_mutate_prediction_or_model_ready_rgb() -> None:
    observer = EvidenceObserver(EvidenceConfig(grid_rows=2, grid_columns=3))
    prediction = _prediction()
    rgb = _rgb()
    before_prediction = {name: value.copy() for name, value in prediction.items()}
    before_rgb = rgb.copy()

    observer.observe(
        frame_id=0,
        prediction=prediction,
        camera_to_reference=_camera(),
        model_ready_rgb=rgb,
    )

    for name in prediction:
        np.testing.assert_array_equal(prediction[name], before_prediction[name])
    np.testing.assert_array_equal(rgb, before_rgb)


def test_observer_module_has_no_model_or_recovery_imports() -> None:
    module_path = Path(__file__).parents[1] / "src" / "stateguard3r" / "state_triage_v2.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    forbidden = ("torch", "dust3r", "recal3r", "recovery", "quarantine")
    assert not any(any(token in name for token in forbidden) for name in imported)
