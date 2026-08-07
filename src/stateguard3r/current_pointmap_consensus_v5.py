"""Fixed GPU-resident current-frame self/cross pointmap consensus for v5.

The estimator accepts only current self/cross pointmaps and confidence maps.
It deliberately has no RGB, history, anchor, detector, timestamp, GT, or raw
camera-pose numeric input.  For row-vector points it estimates
``self @ R.T + t == cross`` from a fixed 16 by 16 lattice.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


LATTICE_SIDE = 16
LATTICE_PAIRS = LATTICE_SIDE * LATTICE_SIDE
MIN_FINITE_POSITIVE_PAIRS = 192
MIN_FINAL_POSITIVE_WEIGHTS = 128
IRLS_ITERATIONS = 2
TUKEY_CONSTANT = 4.685
MAD_MULTIPLIER = 1.4826
NORMALIZED_RESIDUAL_LIMIT = 0.05
SCALE_FLOOR = 1e-3
RELATIVE_MAD_FLOOR = 1e-6
ROTATION_ATOL = 1e-8


class CurrentPointmapConsensusError(RuntimeError):
    """The current-frame geometry cannot safely replace an alarm pose."""


@dataclass(frozen=True)
class CurrentPointmapConsensusResult:
    """A validated current-self to cross/reference transform and its evidence."""

    camera_to_reference: Any
    lattice_rows: Any
    lattice_cols: Any
    finite_pairs: int
    finite_positive_pairs: int
    irls_positive_weights: tuple[int, int]
    final_positive_weights: int
    source_rank: int
    target_rank: int
    mad_scales: tuple[float, float]
    normalized_residual: float
    rotation_determinant: float
    orthonormality_max_abs_error: float

    @property
    def transform_sha256(self) -> str:
        return matrix_sha256(self.camera_to_reference)


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(getattr(value, "shape", ()))


def _weighted_median(torch: Any, values: Any, weights: Any, *, label: str) -> Any:
    if values.ndim != 1 or weights.ndim != 1 or values.numel() != weights.numel() or not int(values.numel()):
        raise CurrentPointmapConsensusError(f"{label} weighted median inputs are malformed")
    if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
        raise CurrentPointmapConsensusError(f"{label} weighted median has nonfinite or nonpositive values")
    ordered, indices = torch.sort(values)
    ordered_weights = weights[indices]
    total = ordered_weights.sum()
    if not bool(torch.isfinite(total)) or not bool(total > 0):
        raise CurrentPointmapConsensusError(f"{label} weighted median lacks positive total weight")
    return ordered[torch.searchsorted(torch.cumsum(ordered_weights, dim=0), total * 0.5, right=False)]


def _weighted_rank(torch: Any, points: Any, weights: Any, *, label: str) -> int:
    positive = weights > 0
    if not bool(positive.any()):
        raise CurrentPointmapConsensusError(f"{label} has no positive robust weights")
    selected, selected_weights = points[positive], weights[positive]
    center = (selected * selected_weights[:, None]).sum(dim=0) / selected_weights.sum()
    rank = int(torch.linalg.matrix_rank((selected - center) * torch.sqrt(selected_weights)[:, None]).item())
    if rank < 3:
        raise CurrentPointmapConsensusError(f"{label} lacks weighted centered rank 3")
    return rank


def _weighted_kabsch(torch: Any, source: Any, target: Any, weights: Any) -> tuple[Any, int, int]:
    if source.ndim != 2 or source.shape != target.shape or source.shape[1:] != (3,) or weights.shape != source.shape[:1]:
        raise CurrentPointmapConsensusError("weighted Kabsch correspondence shapes are malformed")
    if not bool(torch.isfinite(source).all()) or not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(weights).all()) or not bool((weights > 0).any()):
        raise CurrentPointmapConsensusError("weighted Kabsch inputs are nonfinite or lack positive weight")
    source_rank = _weighted_rank(torch, source, weights, label="source")
    target_rank = _weighted_rank(torch, target, weights, label="target")
    total = weights.sum()
    source_center = (source * weights[:, None]).sum(dim=0) / total
    target_center = (target * weights[:, None]).sum(dim=0) / total
    covariance = (source - source_center).transpose(0, 1) @ ((target - target_center) * weights[:, None])
    try:
        left, _singular, right_t = torch.linalg.svd(covariance, full_matrices=False)
    except Exception as error:
        raise CurrentPointmapConsensusError(f"weighted Kabsch SVD failed: {error}") from error
    rotation = right_t.transpose(0, 1) @ left.transpose(0, 1)
    if float(torch.linalg.det(rotation).item()) < 0.0:
        right_t = right_t.clone()
        right_t[-1] *= -1.0
        rotation = right_t.transpose(0, 1) @ left.transpose(0, 1)
    transform = torch.eye(4, dtype=source.dtype, device=source.device)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform, source_rank, target_rank


def _residuals(torch: Any, source: Any, target: Any, transform: Any) -> Any:
    values = torch.linalg.vector_norm(source @ transform[:3, :3].transpose(0, 1) + transform[:3, 3] - target, dim=1)
    if not bool(torch.isfinite(values).all()):
        raise CurrentPointmapConsensusError("consensus residual is nonfinite")
    return values


def _next_irls_weights(torch: Any, residuals: Any, base_weights: Any, cross_norms: Any) -> tuple[Any, float]:
    median = _weighted_median(torch, residuals, base_weights, label="residual")
    mad = _weighted_median(torch, torch.abs(residuals - median), base_weights, label="residual MAD")
    cross_scale = _weighted_median(torch, cross_norms, base_weights, label="cross pointmap scale")
    floor = RELATIVE_MAD_FLOOR * torch.maximum(cross_scale, torch.as_tensor(SCALE_FLOOR, device=cross_scale.device, dtype=cross_scale.dtype))
    scale = torch.maximum(MAD_MULTIPLIER * mad, floor)
    if not bool(torch.isfinite(mad)) or not bool(torch.isfinite(scale)) or not bool(scale > 0):
        raise CurrentPointmapConsensusError("IRLS MAD scale is nonfinite or zero")
    ratio = residuals / (TUKEY_CONSTANT * scale)
    robust = torch.where(ratio < 1.0, base_weights * (1.0 - ratio.square()).square(), torch.zeros_like(base_weights))
    if not bool(torch.isfinite(robust).all()) or not bool((robust > 0).any()):
        raise CurrentPointmapConsensusError("IRLS has no positive robust weight")
    return robust, float(scale.item())


def fixed_lattice(torch: Any, *, height: int, width: int, device: Any) -> tuple[Any, Any]:
    """Fixed evenly spaced nearest-integer lattice with all four edges."""
    if height < LATTICE_SIDE or width < LATTICE_SIDE:
        raise CurrentPointmapConsensusError("pointmap is smaller than fixed 16x16 lattice")
    rows = torch.linspace(0, height - 1, LATTICE_SIDE, device=device).round().to(dtype=torch.long)
    cols = torch.linspace(0, width - 1, LATTICE_SIDE, device=device).round().to(dtype=torch.long)
    grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing="ij")
    return grid_rows.reshape(-1), grid_cols.reshape(-1)


def current_self_cross_consensus(self_points: Any, cross_points: Any, conf_self: Any, conf_cross: Any, *, torch: Any) -> CurrentPointmapConsensusResult:
    """Estimate fixed current self-to-cross SE(3) entirely on the tensor device.

    The caller must supply matching `(1,H,W,3)` pointmaps and `(1,H,W)`
    confidences.  No raw pose numerical value is accepted or read here.
    """
    shape = _shape(self_points)
    if len(shape) != 4 or shape[0] != 1 or shape[-1] != 3:
        raise CurrentPointmapConsensusError("self pointmap must have shape (1,H,W,3)")
    if _shape(cross_points) != shape:
        raise CurrentPointmapConsensusError("current self/cross pointmap shapes differ")
    height, width = shape[1:3]
    if _shape(conf_self) != (1, height, width) or _shape(conf_cross) != (1, height, width):
        raise CurrentPointmapConsensusError("current pointmap/confidence shapes differ")
    values = (self_points, cross_points, conf_self, conf_cross)
    if not all(bool(torch.is_tensor(value)) for value in values):
        raise CurrentPointmapConsensusError("pointmaps and confidences must be tensors")
    if any(value.device != self_points.device for value in (cross_points, conf_self, conf_cross)):
        raise CurrentPointmapConsensusError("pointmaps and confidences must share one device")
    if not all(bool(torch.is_floating_point(value)) for value in values):
        raise CurrentPointmapConsensusError("pointmaps and confidences must be floating tensors")
    rows, cols = fixed_lattice(torch, height=height, width=width, device=self_points.device)
    source = self_points[0, rows, cols].to(dtype=torch.float64)
    target = cross_points[0, rows, cols].to(dtype=torch.float64)
    source_conf = conf_self[0, rows, cols].to(dtype=torch.float64)
    target_conf = conf_cross[0, rows, cols].to(dtype=torch.float64)
    finite = torch.isfinite(source).all(dim=1) & torch.isfinite(target).all(dim=1) & torch.isfinite(source_conf) & torch.isfinite(target_conf)
    valid = finite & (source_conf > 0) & (target_conf > 0)
    finite_pairs, finite_positive_pairs = int(finite.sum().item()), int(valid.sum().item())
    if finite_positive_pairs < MIN_FINITE_POSITIVE_PAIRS:
        raise CurrentPointmapConsensusError("fewer than 192 finite positive-confidence lattice pairs")
    source, target = source[valid], target[valid]
    base_weights = torch.sqrt(torch.minimum(source_conf[valid], target_conf[valid]))
    if not bool(torch.isfinite(base_weights).all()) or not bool((base_weights > 0).all()):
        raise CurrentPointmapConsensusError("initial consensus weights are invalid")
    cross_norms = torch.linalg.vector_norm(target, dim=1)
    transform, source_rank, target_rank = _weighted_kabsch(torch, source, target, base_weights)
    scales: list[float] = []
    positive_counts: list[int] = []
    robust_weights = base_weights
    for _ in range(IRLS_ITERATIONS):
        robust_weights, scale = _next_irls_weights(torch, _residuals(torch, source, target, transform), base_weights, cross_norms)
        positive = int((robust_weights > 0).sum().item())
        if positive < MIN_FINAL_POSITIVE_WEIGHTS:
            raise CurrentPointmapConsensusError("IRLS has fewer than 128 positive robust weights")
        transform, source_rank, target_rank = _weighted_kabsch(torch, source, target, robust_weights)
        scales.append(scale)
        positive_counts.append(positive)
    final_residuals = _residuals(torch, source, target, transform)
    # Tukey deliberately assigns zero weight to rejected correspondences.  The
    # final robust statistics must therefore be evaluated on exactly the
    # positive final inliers, as were the final Kabsch solve and the v5
    # preregistered "final positive robust weight" contract.
    final_positive = robust_weights > 0
    if int(final_positive.sum().item()) < MIN_FINAL_POSITIVE_WEIGHTS:
        raise CurrentPointmapConsensusError("IRLS has fewer than 128 final positive robust weights")
    denominator = torch.maximum(
        _weighted_median(torch, cross_norms[final_positive], robust_weights[final_positive], label="final cross pointmap scale"),
        torch.as_tensor(SCALE_FLOOR, device=cross_norms.device, dtype=cross_norms.dtype),
    )
    normalized = float((
        _weighted_median(torch, final_residuals[final_positive], robust_weights[final_positive], label="final residual")
        / denominator
    ).item())
    rotation = transform[:3, :3]
    determinant = float(torch.linalg.det(rotation).item())
    orth_error = float(torch.max(torch.abs(rotation.transpose(0, 1) @ rotation - torch.eye(3, dtype=rotation.dtype, device=rotation.device))).item())
    if not bool(torch.isfinite(transform).all()) or not math.isfinite(determinant) or not math.isfinite(orth_error):
        raise CurrentPointmapConsensusError("consensus transform is nonfinite")
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=ROTATION_ATOL) or orth_error > ROTATION_ATOL:
        raise CurrentPointmapConsensusError("consensus transform is not proper SO(3)")
    if not math.isfinite(normalized) or normalized > NORMALIZED_RESIDUAL_LIMIT:
        raise CurrentPointmapConsensusError("consensus normalized residual exceeds fixed limit")
    return CurrentPointmapConsensusResult(transform, rows, cols, finite_pairs, finite_positive_pairs, (positive_counts[0], positive_counts[1]), positive_counts[-1], source_rank, target_rank, (scales[0], scales[1]), normalized, determinant, orth_error)


def matrix_sha256(value: Any) -> str:
    """Digest only a validated 4x4 transform for immutable external evidence."""
    tensor = value.detach() if callable(getattr(value, "detach", None)) else value
    tensor = tensor.cpu() if callable(getattr(tensor, "cpu", None)) else tensor
    array = np.asarray(tensor, dtype=np.float64)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise CurrentPointmapConsensusError("transform digest input must be a finite 4x4")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def audit_pinned_cross_head(path: Path) -> Mapping[str, Any]:
    """Bind v5 to the audited self/cross head topology and its causal caveat."""
    source = path.read_text(encoding="utf-8")
    required = (
        "pose = self.pose_head(pose_token)",
        "token_cross = blk(token_cross, pose_token, kwargs.get(\"pos\"))",
        "x_cross = x[:-1] + [token_cross]",
        "self.dpt_self,",
        "self.dpt_cross,",
        'final_output["camera_pose"] = pose',
        'final_output["pts3d_in_other_view"]',
    )
    if any(fragment not in source for fragment in required):
        raise CurrentPointmapConsensusError("pinned ReCal3R cross-head topology changed")
    if "dpt_cross,\n                    pose" in source or "x_cross = x[:-1] + [pose" in source:
        raise CurrentPointmapConsensusError("raw pose output directly conditions cross pointmap")
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "self_head_input": "x",
        "cross_head_input": "x_cross",
        "camera_pose_numeric_input_to_cross_head": False,
        "cross_head_uses_shared_pose_token_latent": True,
        "causal_interpretation": "pointmap-head consensus only; cross pointmap is not claimed latent-independent from pose head",
    }


__all__ = [
    "CurrentPointmapConsensusError", "CurrentPointmapConsensusResult", "IRLS_ITERATIONS", "LATTICE_PAIRS", "LATTICE_SIDE", "MIN_FINAL_POSITIVE_WEIGHTS", "MIN_FINITE_POSITIVE_PAIRS", "NORMALIZED_RESIDUAL_LIMIT", "ROTATION_ATOL", "audit_pinned_cross_head", "current_self_cross_consensus", "fixed_lattice", "matrix_sha256",
]
