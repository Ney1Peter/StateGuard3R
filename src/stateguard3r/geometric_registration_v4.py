"""Fixed, fail-closed 3D--3D rigid registration for recovery v4."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


RANSAC_SEED = 0
RANSAC_HYPOTHESES = 256
SAMPLE_SIZE = 3
MIN_FINITE_PAIRS = 24
MIN_INLIERS = 16
MIN_INLIER_CENTERED_RANK = 3
NORMALIZED_INLIER_LIMIT = 0.05
SCALE_FLOOR = 1e-3


class GeometricRegistrationError(RuntimeError):
    """A registration cannot safely replace an alarmed camera pose."""


@dataclass(frozen=True)
class RegistrationResult:
    camera_to_reference: np.ndarray
    inlier_mask: np.ndarray
    scale: float
    normalized_residual: float


def _points(value: Any, *, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise GeometricRegistrationError(f"{label} must have shape (N, 3)")
    return result


def _proper_transform(current: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    if current.shape != anchor.shape or current.shape[0] < SAMPLE_SIZE:
        raise GeometricRegistrationError("3D correspondence sample is malformed")
    if np.linalg.matrix_rank(current - current.mean(axis=0), tol=1e-10) < 2 or np.linalg.matrix_rank(anchor - anchor.mean(axis=0), tol=1e-10) < 2:
        raise GeometricRegistrationError("3D correspondence sample is degenerate")
    current_center, anchor_center = current.mean(axis=0), anchor.mean(axis=0)
    left = current - current_center
    right = anchor - anchor_center
    u, _, vt = np.linalg.svd(left.T @ right, full_matrices=False)
    rotation = vt.T @ u.T
    if float(np.linalg.det(rotation)) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-8) or not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise GeometricRegistrationError("3D registration did not yield a proper rotation")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = anchor_center - rotation @ current_center
    return result


def _require_full_rank_inliers(current: np.ndarray, anchor: np.ndarray) -> None:
    """Require a volumetric refined cloud after a non-collinear 3-point draw.

    A centered set of exactly three points has rank at most two, so RANSAC
    hypotheses use the separate non-collinearity check in ``_proper_transform``.
    The predeclared rank-three constraint is applied to the final >=16-inlier
    cloud, where it is both meaningful and necessary for a stable 3D pose.
    """
    if np.linalg.matrix_rank(current - current.mean(axis=0), tol=1e-10) < MIN_INLIER_CENTERED_RANK:
        raise GeometricRegistrationError("refined current inlier cloud lacks centered rank 3")
    if np.linalg.matrix_rank(anchor - anchor.mean(axis=0), tol=1e-10) < MIN_INLIER_CENTERED_RANK:
        raise GeometricRegistrationError("refined anchor inlier cloud lacks centered rank 3")


def register_anchor_to_current(anchor_points: Any, current_points: Any) -> RegistrationResult:
    """Estimate the fixed current-self -> anchor-reference transform.

    Correspondences must already be causally established by the v4 RGB matcher;
    this routine intentionally accepts no image, pose, GT, timestamp or label.
    """
    anchor, current = _points(anchor_points, label="anchor points"), _points(current_points, label="current points")
    if anchor.shape != current.shape:
        raise GeometricRegistrationError("anchor/current correspondence count differs")
    finite = np.isfinite(anchor).all(axis=1) & np.isfinite(current).all(axis=1)
    anchor, current = anchor[finite], current[finite]
    if len(anchor) < MIN_FINITE_PAIRS:
        raise GeometricRegistrationError("fewer than 24 finite 3D pairs")
    scale = max(float(np.median(np.linalg.norm(anchor - np.median(anchor, axis=0), axis=1))), SCALE_FLOOR)
    limit = NORMALIZED_INLIER_LIMIT * scale
    rng = np.random.default_rng(RANSAC_SEED)
    best: tuple[int, float, np.ndarray, np.ndarray] | None = None
    for _ in range(RANSAC_HYPOTHESES):
        indices = rng.choice(len(anchor), size=SAMPLE_SIZE, replace=False)
        try:
            transform = _proper_transform(current[indices], anchor[indices])
        except GeometricRegistrationError:
            continue
        predicted = current @ transform[:3, :3].T + transform[:3, 3]
        residuals = np.linalg.norm(predicted - anchor, axis=1)
        mask = residuals <= limit
        candidate = (int(mask.sum()), float(residuals[mask].mean()) if bool(mask.any()) else float("inf"), transform, mask)
        if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
            best = candidate
    if best is None or best[0] < MIN_INLIERS:
        raise GeometricRegistrationError("3D registration has fewer than 16 inliers")
    _require_full_rank_inliers(current[best[3]], anchor[best[3]])
    transform = _proper_transform(current[best[3]], anchor[best[3]])
    residuals = np.linalg.norm(current @ transform[:3, :3].T + transform[:3, 3] - anchor, axis=1)
    mask = residuals <= limit
    if int(mask.sum()) < MIN_INLIERS:
        raise GeometricRegistrationError("refined registration lost required inliers")
    normalized = float(np.median(residuals[mask]) / scale)
    if not math.isfinite(normalized) or normalized > NORMALIZED_INLIER_LIMIT:
        raise GeometricRegistrationError("registration residual exceeds fixed normalized limit")
    return RegistrationResult(transform, mask, scale, normalized)


__all__ = ["GeometricRegistrationError", "MIN_FINITE_PAIRS", "MIN_INLIER_CENTERED_RANK", "MIN_INLIERS", "NORMALIZED_INLIER_LIMIT", "RANSAC_HYPOTHESES", "RANSAC_SEED", "RegistrationResult", "register_anchor_to_current"]
