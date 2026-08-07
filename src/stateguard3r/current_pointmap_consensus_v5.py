"""Fixed-grid robust current self/cross pointmap consensus for recovery v5."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np


LATTICE_SIDE = 16
LATTICE_PAIRS = LATTICE_SIDE * LATTICE_SIDE
MIN_FINITE_POSITIVE_PAIRS = 192
MIN_FINAL_POSITIVE_WEIGHTS = 128
IRLS_ITERATIONS = 2
TUKEY_CONSTANT = 4.685
NORMALIZED_RESIDUAL_LIMIT = 0.05
SCALE_FLOOR = 1e-3
RELATIVE_MAD_FLOOR = 1e-6


class CurrentPointmapConsensusError(RuntimeError):
    """The v5 current-frame geometry cannot safely replace an alarm pose."""


@dataclass(frozen=True)
class CurrentPointmapConsensusResult:
    camera_to_reference: np.ndarray
    lattice_rows: tuple[int, ...]
    lattice_cols: tuple[int, ...]
    finite_positive_pairs: int
    final_positive_weights: int
    source_rank: int
    target_rank: int
    mad_scales: tuple[float, ...]
    normalized_residual: float
    rotation_determinant: float
    orthonormality_max_abs_error: float

    @property
    def transform_sha256(self) -> str:
        return hashlib.sha256(self.camera_to_reference.tobytes(order="C")).hexdigest()


def _array(value: Any, *, label: str, channels: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim >= 1 and result.shape[0] == 1:
        result = result[0]
    required = 3 if channels is None else channels
    if result.ndim != required:
        raise CurrentPointmapConsensusError(f"{label} has malformed rank")
    if channels == 3 and result.shape[-1] != 3:
        raise CurrentPointmapConsensusError(f"{label} must end in xyz channels")
    return result


def _lattice(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    if height < LATTICE_SIDE or width < LATTICE_SIDE:
        raise CurrentPointmapConsensusError("pointmap is smaller than fixed 16x16 lattice")
    rows = np.rint(np.linspace(0, height - 1, LATTICE_SIDE)).astype(np.int64)
    cols = np.rint(np.linspace(0, width - 1, LATTICE_SIDE)).astype(np.int64)
    if len(np.unique(rows)) != LATTICE_SIDE or len(np.unique(cols)) != LATTICE_SIDE:
        raise CurrentPointmapConsensusError("fixed lattice did not yield unique coordinates")
    return rows, cols


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if values.ndim != 1 or weights.shape != values.shape or not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise CurrentPointmapConsensusError("weighted median inputs are malformed")
    if bool((weights < 0).any()) or not bool((weights > 0).any()):
        raise CurrentPointmapConsensusError("weighted median lacks positive finite weights")
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum() / 2.0, side="left")])


def _rank(points: np.ndarray, weights: np.ndarray) -> int:
    center = np.average(points, axis=0, weights=weights)
    singular = np.linalg.svd((points - center) * np.sqrt(weights)[:, None], compute_uv=False)
    if len(singular) != 3 or not np.isfinite(singular).all() or singular[0] <= 0:
        return 0
    return int((singular > singular[0] * 1e-10).sum())


def _kabsch(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise CurrentPointmapConsensusError("weighted Kabsch correspondence shape is malformed")
    if not np.isfinite(source).all() or not np.isfinite(target).all() or not np.isfinite(weights).all() or bool((weights <= 0).any()):
        raise CurrentPointmapConsensusError("weighted Kabsch inputs are nonfinite or nonpositive")
    source_center = np.average(source, axis=0, weights=weights)
    target_center = np.average(target, axis=0, weights=weights)
    covariance = ((source - source_center) * weights[:, None]).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance, full_matrices=False)
    rotation = vt.T @ u.T
    if float(np.linalg.det(rotation)) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    determinant = float(np.linalg.det(rotation))
    orth_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    if not math.isfinite(determinant) or not math.isfinite(orth_error) or not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1e-8) or orth_error > 1e-8:
        raise CurrentPointmapConsensusError("weighted Kabsch did not yield proper SO(3)")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = target_center - rotation @ source_center
    return result


def current_self_cross_consensus(self_points: Any, cross_points: Any, conf_self: Any, conf_cross: Any) -> CurrentPointmapConsensusResult:
    """Estimate self-to-reference SE(3) from fixed current-frame pointmap pairs.

    This deliberately accepts neither RGB, historical state, anchor, raw pose,
    GT, depth, timestamp, detector record nor future-frame capability.
    """
    source, target = _array(self_points, label="self pointmap", channels=3), _array(cross_points, label="cross pointmap", channels=3)
    source_conf, target_conf = _array(conf_self, label="self confidence", channels=2), _array(conf_cross, label="cross confidence", channels=2)
    if source.shape != target.shape or source.shape[:2] != source_conf.shape or source.shape[:2] != target_conf.shape:
        raise CurrentPointmapConsensusError("current pointmap/confidence shapes differ")
    rows, cols = _lattice(*source.shape[:2])
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
    source, target = source[row_grid.ravel(), col_grid.ravel()], target[row_grid.ravel(), col_grid.ravel()]
    source_conf, target_conf = source_conf[row_grid.ravel(), col_grid.ravel()], target_conf[row_grid.ravel(), col_grid.ravel()]
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1) & np.isfinite(source_conf) & np.isfinite(target_conf) & (source_conf > 0) & (target_conf > 0)
    source, target, source_conf, target_conf = source[valid], target[valid], source_conf[valid], target_conf[valid]
    if len(source) < MIN_FINITE_POSITIVE_PAIRS:
        raise CurrentPointmapConsensusError("fewer than 192 finite positive-confidence lattice pairs")
    weights = np.sqrt(np.minimum(source_conf, target_conf))
    if not np.isfinite(weights).all() or not bool((weights > 0).all()):
        raise CurrentPointmapConsensusError("initial consensus weights are invalid")
    source_rank, target_rank = _rank(source, weights), _rank(target, weights)
    if source_rank < 3 or target_rank < 3:
        raise CurrentPointmapConsensusError("consensus lattice lacks weighted centered rank 3")
    scales: list[float] = []
    for _ in range(IRLS_ITERATIONS):
        transform = _kabsch(source, target, weights)
        residual = np.linalg.norm(source @ transform[:3, :3].T + transform[:3, 3] - target, axis=1)
        median = _weighted_median(residual, weights)
        mad = _weighted_median(np.abs(residual - median), weights)
        scene_scale = max(_weighted_median(np.linalg.norm(target, axis=1), weights), SCALE_FLOOR)
        scale = max(1.4826 * mad, RELATIVE_MAD_FLOOR * scene_scale)
        if not math.isfinite(scale) or scale <= 0:
            raise CurrentPointmapConsensusError("IRLS MAD scale is invalid")
        ratio = residual / (TUKEY_CONSTANT * scale)
        robust = np.where(ratio < 1.0, (1.0 - ratio**2) ** 2, 0.0)
        weights = np.sqrt(np.minimum(source_conf, target_conf)) * robust
        scales.append(float(scale))
        if int((weights > 0).sum()) < MIN_FINAL_POSITIVE_WEIGHTS:
            raise CurrentPointmapConsensusError("IRLS has fewer than 128 positive robust weights")
    positive = weights > 0
    transform = _kabsch(source[positive], target[positive], weights[positive])
    residual = np.linalg.norm(source[positive] @ transform[:3, :3].T + transform[:3, 3] - target[positive], axis=1)
    normalized = _weighted_median(residual, weights[positive]) / max(_weighted_median(np.linalg.norm(target[positive], axis=1), weights[positive]), SCALE_FLOOR)
    if not math.isfinite(normalized) or normalized > NORMALIZED_RESIDUAL_LIMIT:
        raise CurrentPointmapConsensusError("consensus normalized residual exceeds fixed limit")
    rotation = transform[:3, :3]
    return CurrentPointmapConsensusResult(transform, tuple(int(value) for value in rows), tuple(int(value) for value in cols), int(valid.sum()), int(positive.sum()), source_rank, target_rank, tuple(scales), float(normalized), float(np.linalg.det(rotation)), float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))))


__all__ = ["CurrentPointmapConsensusError", "CurrentPointmapConsensusResult", "IRLS_ITERATIONS", "LATTICE_PAIRS", "LATTICE_SIDE", "MIN_FINAL_POSITIVE_WEIGHTS", "MIN_FINITE_POSITIVE_PAIRS", "NORMALIZED_RESIDUAL_LIMIT", "current_self_cross_consensus"]
