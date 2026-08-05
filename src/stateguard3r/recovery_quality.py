"""Prefix-aligned, post-event trajectory metrics for recovery-quality studies.

This module is intentionally detector-free.  It accepts only trajectories and
post-hoc ground truth supplied by the quality evaluator, so GT cannot become an
online alarm or state-policy input by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


class RecoveryQualityError(ValueError):
    """Raised when a trajectory/GT comparison is ambiguous or degenerate."""


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SimilarityTransform:
    """A proper Sim(3) mapping predicted positions into GT coordinates."""

    scale: float
    rotation: FloatArray
    translation: FloatArray

    def apply_positions(self, positions: ArrayLike) -> FloatArray:
        values = _points(positions, name="positions")
        return self.scale * (values @ self.rotation.T) + self.translation

    def apply_rotations(self, rotations: ArrayLike) -> FloatArray:
        values = _rotations(rotations, name="rotations")
        return np.einsum("ij,njk->nik", self.rotation, values)


@dataclass(frozen=True)
class TailMetrics:
    """Metrics after one prefix-only alignment, with explicit frame support."""

    alignment: SimilarityTransform
    prefix_frame_ids: tuple[int, ...]
    tail_frame_ids: tuple[int, ...]
    ate_rmse_m: float
    rpe_translation_rmse_m: float
    rpe_rotation_rmse_deg: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "alignment": {
                "kind": "proper_umeyama_sim3",
                "scale": self.alignment.scale,
                "rotation": self.alignment.rotation.tolist(),
                "translation": self.alignment.translation.tolist(),
            },
            "prefix_frame_ids": list(self.prefix_frame_ids),
            "tail_frame_ids": list(self.tail_frame_ids),
            "ATE_RMSE_m": self.ate_rmse_m,
            "RPE_translation_RMSE_m": self.rpe_translation_rmse_m,
            "RPE_rotation_RMSE_deg": self.rpe_rotation_rmse_deg,
        }


def _points(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 1:
        raise RecoveryQualityError(f"{name} must have shape (N, 3) with N >= 1")
    if not np.all(np.isfinite(array)):
        raise RecoveryQualityError(f"{name} must be finite")
    return array


def _rotations(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (3, 3) or array.shape[0] < 1:
        raise RecoveryQualityError(f"{name} must have shape (N, 3, 3) with N >= 1")
    if not np.all(np.isfinite(array)):
        raise RecoveryQualityError(f"{name} must be finite")
    identity = np.eye(3, dtype=np.float64)
    if not np.allclose(array @ np.swapaxes(array, 1, 2), identity, atol=1e-5):
        raise RecoveryQualityError(f"{name} must contain orthonormal matrices")
    determinants = np.linalg.det(array)
    if not np.allclose(determinants, 1.0, atol=1e-5):
        raise RecoveryQualityError(f"{name} must contain proper rotations")
    return array


def proper_umeyama_sim3(predicted: ArrayLike, ground_truth: ArrayLike) -> SimilarityTransform:
    """Fit a non-reflective Sim(3) from predicted to GT positions.

    The function deliberately rejects zero-variance prefixes.  A camera that
    has not moved cannot define a scale, and silently accepting it would make a
    tail ATE claim arbitrary.
    """

    source = _points(predicted, name="predicted")
    target = _points(ground_truth, name="ground_truth")
    if source.shape != target.shape or source.shape[0] < 3:
        raise RecoveryQualityError("predicted and ground_truth must share N >= 3")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    source_variance = float(np.mean(np.sum(centered_source * centered_source, axis=1)))
    if not math.isfinite(source_variance) or source_variance <= 1e-14:
        raise RecoveryQualityError("prefix predicted positions have zero variance")
    covariance = centered_target.T @ centered_source / float(source.shape[0])
    try:
        left, singular_values, right_transpose = np.linalg.svd(covariance)
    except np.linalg.LinAlgError as error:
        raise RecoveryQualityError("SVD failed for prefix alignment") from error
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(left @ right_transpose) < 0.0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_transpose
    scale = float(np.trace(np.diag(singular_values) @ correction) / source_variance)
    if not math.isfinite(scale) or scale <= 0.0:
        raise RecoveryQualityError("prefix alignment produced a non-positive scale")
    translation = target_mean - scale * (rotation @ source_mean)
    return SimilarityTransform(scale=scale, rotation=rotation, translation=translation)


def quaternion_xyzw_to_rotation(values: ArrayLike) -> FloatArray:
    """Convert one finite, nonzero TUM ``qx qy qz qw`` quaternion to SO(3)."""

    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise RecoveryQualityError("quaternion_xyzw must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-14:
        raise RecoveryQualityError("quaternion_xyzw may not be zero")
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def trajectory_arrays(payload: Mapping[str, Any]) -> tuple[FloatArray, FloatArray]:
    """Read contiguous camera-to-reference matrices from a trajectory payload."""

    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RecoveryQualityError("trajectory frames must be a non-empty list")
    positions: list[FloatArray] = []
    rotations: list[FloatArray] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or frame.get("frame_id") != index:
            raise RecoveryQualityError("trajectory frame IDs must be contiguous and zero-based")
        matrix = np.asarray(frame.get("camera_to_reference"), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise RecoveryQualityError(f"trajectory frame {index} camera matrix is invalid")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
            raise RecoveryQualityError(f"trajectory frame {index} is not homogeneous")
        positions.append(matrix[:3, 3].copy())
        rotations.append(matrix[:3, :3].copy())
    return _points(np.vstack(positions), name="trajectory positions"), _rotations(
        np.stack(rotations), name="trajectory rotations"
    )


def evaluate_tail(
    predicted_positions: ArrayLike,
    predicted_rotations: ArrayLike,
    ground_truth_positions: ArrayLike,
    ground_truth_rotations: ArrayLike,
    *,
    prefix_frame_ids: Sequence[int],
    tail_frame_ids: Sequence[int],
) -> TailMetrics:
    """Align only the prefix and report ATE/RPE strictly on the supplied tail."""

    predicted_xyz = _points(predicted_positions, name="predicted_positions")
    predicted_rot = _rotations(predicted_rotations, name="predicted_rotations")
    target_xyz = _points(ground_truth_positions, name="ground_truth_positions")
    target_rot = _rotations(ground_truth_rotations, name="ground_truth_rotations")
    if not (
        predicted_xyz.shape == target_xyz.shape
        and predicted_rot.shape == target_rot.shape
        and predicted_xyz.shape[0] == predicted_rot.shape[0]
    ):
        raise RecoveryQualityError("trajectory/GT positions and rotations must have matching frame counts")
    count = predicted_xyz.shape[0]
    prefix = tuple(int(index) for index in prefix_frame_ids)
    tail = tuple(int(index) for index in tail_frame_ids)
    if len(prefix) < 3 or not tail:
        raise RecoveryQualityError("prefix requires at least three frames and tail may not be empty")
    if len(set(prefix)) != len(prefix) or len(set(tail)) != len(tail):
        raise RecoveryQualityError("prefix and tail frame IDs may not repeat")
    if set(prefix) & set(tail):
        raise RecoveryQualityError("prefix and tail frame IDs must be disjoint")
    if any(index < 0 or index >= count for index in (*prefix, *tail)):
        raise RecoveryQualityError("prefix/tail frame ID is outside trajectory")
    transform = proper_umeyama_sim3(predicted_xyz[list(prefix)], target_xyz[list(prefix)])
    aligned_positions = transform.apply_positions(predicted_xyz)
    aligned_rotations = transform.apply_rotations(predicted_rot)
    tail_indices = np.asarray(tail, dtype=np.int64)
    ate_errors = np.linalg.norm(aligned_positions[tail_indices] - target_xyz[tail_indices], axis=1)
    ate = float(np.sqrt(np.mean(ate_errors * ate_errors)))

    rpe_pairs = [(index - 1, index) for index in tail if index - 1 in tail]
    if not rpe_pairs:
        raise RecoveryQualityError("tail must contain at least one consecutive frame pair for RPE")
    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    for previous, current in rpe_pairs:
        predicted_delta = aligned_positions[current] - aligned_positions[previous]
        target_delta = target_xyz[current] - target_xyz[previous]
        translation_errors.append(float(np.linalg.norm(predicted_delta - target_delta)))
        predicted_relative = aligned_rotations[previous].T @ aligned_rotations[current]
        target_relative = target_rot[previous].T @ target_rot[current]
        residual = predicted_relative.T @ target_relative
        cosine = float((np.trace(residual) - 1.0) / 2.0)
        rotation_errors.append(float(np.arccos(np.clip(cosine, -1.0, 1.0))))
    translation = float(np.sqrt(np.mean(np.square(translation_errors))))
    rotation = float(math.degrees(math.sqrt(np.mean(np.square(rotation_errors)))))
    return TailMetrics(
        alignment=transform,
        prefix_frame_ids=prefix,
        tail_frame_ids=tail,
        ate_rmse_m=ate,
        rpe_translation_rmse_m=translation,
        rpe_rotation_rmse_deg=rotation,
    )


def positive_effect(baseline: float, policy: float, *, name: str) -> float:
    """Return ``1 - policy / baseline`` while rejecting a meaningless denominator."""

    if not math.isfinite(baseline) or not math.isfinite(policy) or baseline <= 0.0 or policy < 0.0:
        raise RecoveryQualityError(f"{name} baseline must be positive and both values finite/non-negative")
    return 1.0 - policy / baseline


def scene_stratified_bootstrap_lower_bound(
    effects: Mapping[str, Sequence[float]], *, seed: int, samples: int = 10_000
) -> float:
    """Return the 2.5th percentile of median effects after resampling scenes.

    Each sampled scene carries all of its corruption trials, preventing the
    correlated variants of one physical sequence from being treated as wholly
    independent samples.
    """

    if type(seed) is not int or type(samples) is not int or samples < 1:
        raise RecoveryQualityError("seed must be an int and samples must be positive")
    normalized: dict[str, FloatArray] = {}
    for scene, values in effects.items():
        if not isinstance(scene, str) or not scene:
            raise RecoveryQualityError("scene IDs must be non-empty strings")
        vector = np.asarray(values, dtype=np.float64)
        if vector.ndim != 1 or vector.size < 1 or not np.all(np.isfinite(vector)):
            raise RecoveryQualityError(f"scene {scene!r} effects must be a non-empty finite vector")
        normalized[scene] = vector
    if len(normalized) < 2:
        raise RecoveryQualityError("scene-stratified bootstrap requires at least two scenes")
    scenes = tuple(sorted(normalized))
    generator = np.random.default_rng(seed)
    medians = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        chosen = generator.integers(0, len(scenes), size=len(scenes))
        values = np.concatenate([normalized[scenes[int(scene_index)]] for scene_index in chosen])
        medians[index] = float(np.median(values))
    return float(np.quantile(medians, 0.025))


__all__ = [
    "RecoveryQualityError",
    "SimilarityTransform",
    "TailMetrics",
    "evaluate_tail",
    "positive_effect",
    "proper_umeyama_sim3",
    "quaternion_xyzw_to_rotation",
    "scene_stratified_bootstrap_lower_bound",
    "trajectory_arrays",
]
