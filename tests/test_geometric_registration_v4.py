from __future__ import annotations

import numpy as np
import pytest

from stateguard3r.geometric_registration_v4 import GeometricRegistrationError, register_anchor_to_current


def _rotation() -> np.ndarray:
    return np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))


def test_fixed_ransac_recovers_proper_current_to_anchor_transform_with_outliers() -> None:
    rng = np.random.default_rng(4)
    current = rng.normal(size=(40, 3))
    rotation, translation = _rotation(), np.array((0.2, -0.4, 0.6))
    anchor = current @ rotation.T + translation
    anchor[-4:] += 50.0
    result = register_anchor_to_current(anchor, current)
    np.testing.assert_allclose(result.camera_to_reference[:3, :3], rotation, atol=1e-8)
    np.testing.assert_allclose(result.camera_to_reference[:3, 3], translation, atol=1e-8)
    assert result.inlier_mask.sum() == 36


def test_refined_inlier_cloud_must_have_centered_rank_three() -> None:
    grid = np.array([(x, y, 0.0) for x in range(5) for y in range(5)], dtype=np.float64)
    with pytest.raises(GeometricRegistrationError, match="rank 3"):
        register_anchor_to_current(grid, grid)


def test_registration_fails_closed_for_insufficient_or_degenerate_pairs() -> None:
    with pytest.raises(GeometricRegistrationError, match="fewer than 24"):
        register_anchor_to_current(np.zeros((23, 3)), np.zeros((23, 3)))
    line = np.column_stack((np.arange(30), np.zeros(30), np.zeros(30)))
    with pytest.raises(GeometricRegistrationError, match="inliers"):
        register_anchor_to_current(line, line)
