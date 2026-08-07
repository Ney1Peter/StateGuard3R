from __future__ import annotations

import numpy as np
import pytest

from stateguard3r.current_pointmap_consensus_v5 import CurrentPointmapConsensusError, current_self_cross_consensus


def _grid() -> np.ndarray:
    rows, cols = np.meshgrid(np.linspace(-1.0, 1.0, 32), np.linspace(-0.5, 0.5, 32), indexing="ij")
    return np.stack((cols, rows, 0.2 * np.sin(2.0 * rows) + 0.1 * cols), axis=-1)[None]


def test_fixed_lattice_recovers_proper_self_to_cross_transform() -> None:
    source = _grid()
    rotation = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    target = source @ rotation.T + np.array((0.2, -0.3, 0.4))
    confidence = np.full(source.shape[:3], 2.0)
    result = current_self_cross_consensus(source, target, confidence, confidence)
    np.testing.assert_allclose(result.camera_to_reference[:3, :3], rotation, atol=1e-8)
    np.testing.assert_allclose(result.camera_to_reference[:3, 3], (0.2, -0.3, 0.4), atol=1e-8)
    assert result.finite_positive_pairs == 256
    assert result.final_positive_weights == 256
    assert result.source_rank == result.target_rank == 3
    assert result.normalized_residual == pytest.approx(0.0)


def test_consensus_fails_closed_for_planar_or_nonpositive_lattice() -> None:
    source = np.zeros((1, 32, 32, 3), dtype=np.float64)
    source[..., 0] = np.linspace(0.0, 1.0, 32)[None, :, None]
    source[..., 1] = np.linspace(0.0, 1.0, 32)[None, None, :]
    confidence = np.ones((1, 32, 32), dtype=np.float64)
    with pytest.raises(CurrentPointmapConsensusError, match="rank 3"):
        current_self_cross_consensus(source, source, confidence, confidence)
    source = _grid()
    with pytest.raises(CurrentPointmapConsensusError, match="fewer than 192"):
        current_self_cross_consensus(source, source, np.zeros((1, 32, 32)), confidence)
