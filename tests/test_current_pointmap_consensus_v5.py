from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from stateguard3r.current_pointmap_consensus_v5 import (
    CurrentPointmapConsensusError,
    audit_pinned_cross_head,
    current_self_cross_consensus,
)


torch = pytest.importorskip("torch")


def _grid() -> object:
    rows, cols = torch.meshgrid(torch.linspace(-1.0, 1.0, 32, dtype=torch.float64), torch.linspace(-0.5, 0.5, 32, dtype=torch.float64), indexing="ij")
    return torch.stack((cols, rows, 0.2 * torch.sin(2.0 * rows) + 0.1 * cols), dim=-1).unsqueeze(0)


def _consensus(source: object, target: object, confidence: object) -> object:
    return current_self_cross_consensus(source, target, confidence, confidence, torch=torch)


def test_fixed_lattice_recovers_proper_gpu_tensor_self_to_cross_transform() -> None:
    source = _grid()
    rotation = torch.tensor(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=torch.float64)
    target = source @ rotation.T + torch.tensor((0.2, -0.3, 0.4), dtype=torch.float64)
    confidence = torch.full(source.shape[:3], 2.0, dtype=torch.float64)
    result = _consensus(source, target, confidence)
    torch.testing.assert_close(result.camera_to_reference[:3, :3], rotation, atol=1e-8, rtol=0.0)
    torch.testing.assert_close(result.camera_to_reference[:3, 3], torch.tensor((0.2, -0.3, 0.4), dtype=torch.float64), atol=1e-8, rtol=0.0)
    assert result.camera_to_reference.device == source.device
    assert result.finite_pairs == result.finite_positive_pairs == result.final_positive_weights == 256
    assert result.irls_positive_weights == (256, 256)
    assert result.source_rank == result.target_rank == 3
    assert result.normalized_residual == pytest.approx(0.0)
    assert int(result.lattice_rows.min()) == int(result.lattice_cols.min()) == 0
    assert int(result.lattice_rows.max()) == source.shape[1] - 1
    assert int(result.lattice_cols.max()) == source.shape[2] - 1


def test_two_fixed_irls_rounds_are_deterministic_and_reject_sparse_outliers() -> None:
    source = _grid()
    rotation = torch.tensor(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=torch.float64)
    target = source @ rotation.T + torch.tensor((0.2, -0.3, 0.4), dtype=torch.float64)
    target = target.clone()
    target[0, 0, 0] += 20.0
    target[0, 0, -1] -= 20.0
    target[0, -1, 0] += 15.0
    target[0, -1, -1] -= 15.0
    confidence = torch.full(source.shape[:3], 2.0, dtype=torch.float64)
    first, second = _consensus(source, target, confidence), _consensus(source, target, confidence)
    torch.testing.assert_close(first.camera_to_reference, second.camera_to_reference, atol=0.0, rtol=0.0)
    torch.testing.assert_close(first.camera_to_reference[:3, :3], rotation, atol=1e-8, rtol=0.0)
    torch.testing.assert_close(first.camera_to_reference[:3, 3], torch.tensor((0.2, -0.3, 0.4), dtype=torch.float64), atol=1e-8, rtol=0.0)
    assert first.irls_positive_weights[0] == first.irls_positive_weights[1] == 252


def test_consensus_fails_closed_for_shape_nonfinite_rank_and_insufficient_pairs() -> None:
    source = torch.zeros((1, 32, 32, 3), dtype=torch.float64)
    source[..., 0] = torch.linspace(0.0, 1.0, 32)[None, :, None]
    source[..., 1] = torch.linspace(0.0, 1.0, 32)[None, None, :]
    confidence = torch.ones((1, 32, 32), dtype=torch.float64)
    with pytest.raises(CurrentPointmapConsensusError, match="rank 3"):
        _consensus(source, source, confidence)
    source = _grid()
    with pytest.raises(CurrentPointmapConsensusError, match="fewer than 192"):
        _consensus(source, source, torch.zeros((1, 32, 32), dtype=torch.float64))
    nonfinite = source.clone()
    nonfinite[..., 0] = torch.nan
    with pytest.raises(CurrentPointmapConsensusError, match="fewer than 192"):
        _consensus(nonfinite, source, confidence)
    with pytest.raises(CurrentPointmapConsensusError, match="shapes differ"):
        current_self_cross_consensus(source, source[:, :-1], confidence, confidence, torch=torch)


def test_solver_signature_has_no_raw_pose_or_other_forbidden_numeric_input() -> None:
    source = inspect.getsource(current_self_cross_consensus)
    signature = inspect.signature(current_self_cross_consensus)
    assert tuple(signature.parameters) == ("self_points", "cross_points", "conf_self", "conf_cross", "torch")
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.FunctionDef)
    forbidden = {"camera_pose", "rgb", "anchor", "ground_truth", "future", "timestamp", "detector", "history", "pose"}
    names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
    assert not names & forbidden
    attributes = {node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute)}
    assert "cpu" not in attributes


def test_consensus_rejects_non_tensor_and_nonfloating_inputs_before_numeric_work() -> None:
    source = _grid()
    confidence = torch.ones((1, 32, 32), dtype=torch.float64)
    with pytest.raises(CurrentPointmapConsensusError, match="must be tensors"):
        current_self_cross_consensus(source.numpy(), source.numpy(), confidence.numpy(), confidence.numpy(), torch=torch)
    with pytest.raises(CurrentPointmapConsensusError, match="must be floating"):
        current_self_cross_consensus(source, source, torch.ones((1, 32, 32), dtype=torch.int64), torch.ones((1, 32, 32), dtype=torch.int64), torch=torch)


def test_cross_head_audit_binds_real_pinned_source_and_records_shared_latent_caveat() -> None:
    root = Path(__file__).resolve().parents[1]
    audit = audit_pinned_cross_head(root.parent / "baselines" / "ReCal3R" / "src" / "dust3r" / "heads" / "dpt_head.py")
    assert audit["camera_pose_numeric_input_to_cross_head"] is False
    assert audit["cross_head_uses_shared_pose_token_latent"] is True
    assert len(audit["sha256"]) == 64
