"""Causal, non-GT visual correspondence coverage for Detector v2.

``online_visual_correspondence_coverage`` compares only the previous and
current model-ready RGB images.  It is deliberately not a physical overlap
percentage: the score is a conservative coverage proxy based on verified ORB
correspondences.  GT depth/poses are accepted only by the separately named
``gt_depth_reprojection_overlap`` helper for offline donor construction.

OpenCV is imported lazily because StateGuard3R's CPU test environment does not
need or provide it.  Production integration uses ReCal3R's pinned OpenCV
environment and records its provenance in the v2 run metadata.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


SCHEMA_VERSION = "stateguard3r.visual-overlap.v1"
DEFAULT_GRID_COLUMNS = 8
DEFAULT_GRID_ROWS = 6
DEFAULT_NFEATURES = 2000
DEFAULT_MINIMUM_MATCHES = 12
DEFAULT_RATIO_THRESHOLD = 0.70
DEFAULT_RANSAC_PIXEL_LIMIT = 1.0
DEFAULT_RNG_SEED = 0

FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]


class VisualOverlapError(ValueError):
    """Raised when a correspondence or offline reprojection input is invalid."""


@dataclass(frozen=True)
class VisualOverlapConfig:
    """Frozen online correspondence parameters for one v2 candidate."""

    nfeatures: int = DEFAULT_NFEATURES
    grid_columns: int = DEFAULT_GRID_COLUMNS
    grid_rows: int = DEFAULT_GRID_ROWS
    minimum_matches: int = DEFAULT_MINIMUM_MATCHES
    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD
    ransac_pixel_limit: float = DEFAULT_RANSAC_PIXEL_LIMIT
    rng_seed: int = DEFAULT_RNG_SEED

    def __post_init__(self) -> None:
        for field in ("nfeatures", "grid_columns", "grid_rows", "minimum_matches"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise VisualOverlapError(f"{field} must be a positive integer")
        if self.minimum_matches < 8:
            raise VisualOverlapError("minimum_matches must be at least eight for a fundamental matrix")
        if not math.isfinite(self.ratio_threshold) or not 0.0 < self.ratio_threshold < 1.0:
            raise VisualOverlapError("ratio_threshold must be finite and in (0, 1)")
        if not math.isfinite(self.ransac_pixel_limit) or self.ransac_pixel_limit <= 0:
            raise VisualOverlapError("ransac_pixel_limit must be positive and finite")
        if isinstance(self.rng_seed, bool) or not isinstance(self.rng_seed, int):
            raise VisualOverlapError("rng_seed must be an integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualOverlapResult:
    """One finite online correspondence result for a consecutive image pair."""

    score: float
    status: str
    keypoints_previous: int
    keypoints_current: int
    ratio_matches: int
    mutual_matches: int
    inliers: int
    inlier_ratio: float
    previous_grid_coverage: float
    current_grid_coverage: float
    reference_frame_id: int | None = None
    frame_id: int | None = None

    def __post_init__(self) -> None:
        for field in (
            "score",
            "inlier_ratio",
            "previous_grid_coverage",
            "current_grid_coverage",
        ):
            value = float(getattr(self, field))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise VisualOverlapError(f"{field} must be finite and in [0, 1]")
        for field in (
            "keypoints_previous",
            "keypoints_current",
            "ratio_matches",
            "mutual_matches",
            "inliers",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise VisualOverlapError(f"{field} must be a non-negative integer")
        if self.inliers > self.mutual_matches:
            raise VisualOverlapError("inliers may not exceed mutual_matches")
        if self.frame_id is not None and (isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int)):
            raise VisualOverlapError("frame_id must be an integer or null")
        if self.reference_frame_id is not None and (
            isinstance(self.reference_frame_id, bool) or not isinstance(self.reference_frame_id, int)
        ):
            raise VisualOverlapError("reference_frame_id must be an integer or null")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GTDepthReprojectionOverlap:
    """Offline geometric overlap for donor selection; never an online signal."""

    score: float
    reference_to_target: float
    target_to_reference: float
    reference_valid_samples: int
    target_valid_samples: int
    reference_consistent_samples: int
    target_consistent_samples: int

    def __post_init__(self) -> None:
        for field in ("score", "reference_to_target", "target_to_reference"):
            value = float(getattr(self, field))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise VisualOverlapError(f"{field} must be finite and in [0, 1]")
        for field in (
            "reference_valid_samples",
            "target_valid_samples",
            "reference_consistent_samples",
            "target_consistent_samples",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise VisualOverlapError(f"{field} must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalized_tensor_to_uint8_rgb(image: ArrayLike) -> UInt8Array:
    """Copy one normalized model image from ``[-1, 1]`` into RGB uint8.

    The conversion accepts ``3xHxW``, ``HxWx3``, or a batch-one variant.  It
    rejects non-finite/out-of-range input rather than silently clipping a
    malformed model tensor, and always returns an independent C-contiguous
    array so downstream OpenCV code cannot mutate model input storage.
    """

    array = np.asarray(image)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise VisualOverlapError("batched model-ready RGB must have batch size one")
        array = array[0]
    if array.ndim != 3:
        raise VisualOverlapError("model-ready RGB must have three dimensions")
    if array.shape[0] == 3:
        rgb = np.moveaxis(array, 0, -1)
    elif array.shape[-1] == 3:
        rgb = array
    else:
        raise VisualOverlapError("model-ready RGB must have exactly three channels")
    numeric = np.asarray(rgb, dtype=np.float64)
    if numeric.shape[0] < 1 or numeric.shape[1] < 1 or not np.all(np.isfinite(numeric)):
        raise VisualOverlapError("model-ready RGB must be finite and non-empty")
    if float(numeric.min()) < -1.0 - 1e-6 or float(numeric.max()) > 1.0 + 1e-6:
        raise VisualOverlapError("model-ready RGB must be normalized to [-1, 1]")
    converted = np.rint(np.clip((numeric + 1.0) * 127.5, 0.0, 255.0)).astype(np.uint8)
    return np.ascontiguousarray(converted).copy()


def grid_endpoint_coverage(
    points: ArrayLike,
    *,
    image_width: int,
    image_height: int,
    grid_columns: int = DEFAULT_GRID_COLUMNS,
    grid_rows: int = DEFAULT_GRID_ROWS,
) -> float:
    """Return the fraction of grid cells containing an in-bounds endpoint."""

    if image_width < 1 or image_height < 1 or grid_columns < 1 or grid_rows < 1:
        raise VisualOverlapError("image and grid dimensions must be positive")
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.size == 0:
        return 0.0
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise VisualOverlapError("points must have shape (N, 2)")
    finite = np.isfinite(coordinates).all(axis=1)
    in_bounds = (
        finite
        & (coordinates[:, 0] >= 0.0)
        & (coordinates[:, 0] < image_width)
        & (coordinates[:, 1] >= 0.0)
        & (coordinates[:, 1] < image_height)
    )
    if not np.any(in_bounds):
        return 0.0
    valid = coordinates[in_bounds]
    columns = np.minimum((valid[:, 0] * grid_columns / image_width).astype(int), grid_columns - 1)
    rows = np.minimum((valid[:, 1] * grid_rows / image_height).astype(int), grid_rows - 1)
    occupied = {(int(row), int(column)) for row, column in zip(rows, columns, strict=True)}
    return float(len(occupied) / (grid_columns * grid_rows))


def _gray_rgb(image: UInt8Array) -> UInt8Array:
    if image.ndim != 3 or image.shape[2] != 3:
        raise VisualOverlapError("RGB image must have shape HxWx3")
    return np.rint(
        image[..., 0].astype(np.float64) * 0.299
        + image[..., 1].astype(np.float64) * 0.587
        + image[..., 2].astype(np.float64) * 0.114
    ).astype(np.uint8)


def _load_cv2() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as error:
        raise VisualOverlapError(
            "OpenCV is required for real visual overlap; use ReCal3R's pinned environment"
        ) from error


def configure_opencv_determinism(cv2_module: Any, *, rng_seed: int = DEFAULT_RNG_SEED) -> None:
    """Set every OpenCV global relevant to the correspondence computation."""

    try:
        cv2_module.setNumThreads(1)
        cv2_module.setRNGSeed(rng_seed)
        cv2_module.ocl.setUseOpenCL(False)
    except (AttributeError, TypeError, ValueError) as error:
        raise VisualOverlapError("OpenCV module lacks deterministic runtime controls") from error


def opencv_runtime_provenance(cv2_module: Any | None = None) -> dict[str, Any]:
    """Return deterministic OpenCV version/build evidence for run metadata."""

    cv2_module = _load_cv2() if cv2_module is None else cv2_module
    configure_opencv_determinism(cv2_module)
    build = str(cv2_module.getBuildInformation())
    module_file = Path(str(cv2_module.__file__))
    binary_candidates = sorted(
        candidate
        for pattern in ("cv2*.so", "cv2*.pyd", "cv2*.dll", "cv2*.dylib")
        for candidate in module_file.parent.glob(pattern)
        if candidate.is_file()
    )
    binary_file = binary_candidates[0] if binary_candidates else module_file
    try:
        module_sha256 = hashlib.sha256(module_file.read_bytes()).hexdigest()
        binary_sha256 = hashlib.sha256(binary_file.read_bytes()).hexdigest()
    except OSError as error:
        raise VisualOverlapError(f"cannot hash OpenCV module binary: {error}") from error
    return {
        "version": str(cv2_module.__version__),
        "module_path": str(module_file.resolve(strict=True)),
        "module_sha256": module_sha256,
        "binary_path": str(binary_file.resolve(strict=True)),
        "binary_sha256": binary_sha256,
        "build_info_sha256": hashlib.sha256(build.encode("utf-8")).hexdigest(),
        "threads": int(cv2_module.getNumThreads()),
        "opencl_enabled": bool(cv2_module.ocl.useOpenCL()),
        "rng_seed": DEFAULT_RNG_SEED,
    }


def _ratio_matches(match_rows: Iterable[Sequence[Any]], ratio_threshold: float) -> dict[tuple[int, int], Any]:
    accepted: dict[tuple[int, int], Any] = {}
    for row in match_rows:
        if len(row) < 2:
            continue
        best, second = row[0], row[1]
        if not math.isfinite(float(best.distance)) or not math.isfinite(float(second.distance)):
            continue
        if float(best.distance) < ratio_threshold * float(second.distance):
            accepted[(int(best.queryIdx), int(best.trainIdx))] = best
    return accepted


def _empty_result(
    status: str,
    *,
    keypoints_previous: int,
    keypoints_current: int,
    ratio_matches: int = 0,
    mutual_matches: int = 0,
    reference_frame_id: int | None = None,
    frame_id: int | None = None,
) -> VisualOverlapResult:
    return VisualOverlapResult(
        score=0.0,
        status=status,
        keypoints_previous=keypoints_previous,
        keypoints_current=keypoints_current,
        ratio_matches=ratio_matches,
        mutual_matches=mutual_matches,
        inliers=0,
        inlier_ratio=0.0,
        previous_grid_coverage=0.0,
        current_grid_coverage=0.0,
        reference_frame_id=reference_frame_id,
        frame_id=frame_id,
    )


def online_visual_correspondence_coverage(
    previous_rgb: ArrayLike,
    current_rgb: ArrayLike,
    *,
    config: VisualOverlapConfig = VisualOverlapConfig(),
    cv2_module: Any | None = None,
    reference_frame_id: int | None = None,
    frame_id: int | None = None,
) -> VisualOverlapResult:
    """Compute finite ORB/RANSAC/grid coverage from a strictly causal pair.

    No depth, pose, labels, manifest interval, or future frame is accepted by
    this API.  Descriptor/match/geometric failures are explicit zero-valued
    results, which keeps a v2 ledger finite without pretending that coverage
    was observed.
    """

    previous = normalized_tensor_to_uint8_rgb(previous_rgb)
    current = normalized_tensor_to_uint8_rgb(current_rgb)
    if previous.shape[:2] != current.shape[:2]:
        raise VisualOverlapError("consecutive model-ready RGB images must share a shape")
    cv2_module = _load_cv2() if cv2_module is None else cv2_module
    configure_opencv_determinism(cv2_module, rng_seed=config.rng_seed)
    try:
        detector = cv2_module.ORB_create(nfeatures=config.nfeatures)
        previous_keypoints, previous_descriptors = detector.detectAndCompute(_gray_rgb(previous), None)
        current_keypoints, current_descriptors = detector.detectAndCompute(_gray_rgb(current), None)
    except (AttributeError, TypeError, ValueError) as error:
        raise VisualOverlapError("OpenCV ORB extraction failed") from error
    previous_keypoints = list(previous_keypoints or ())
    current_keypoints = list(current_keypoints or ())
    if previous_descriptors is None or current_descriptors is None:
        return _empty_result(
            "descriptor_insufficient",
            keypoints_previous=len(previous_keypoints),
            keypoints_current=len(current_keypoints),
            reference_frame_id=reference_frame_id,
            frame_id=frame_id,
        )
    try:
        matcher = cv2_module.BFMatcher(cv2_module.NORM_HAMMING, crossCheck=False)
        forward = _ratio_matches(matcher.knnMatch(previous_descriptors, current_descriptors, k=2), config.ratio_threshold)
        backward = _ratio_matches(matcher.knnMatch(current_descriptors, previous_descriptors, k=2), config.ratio_threshold)
    except (AttributeError, TypeError, ValueError) as error:
        raise VisualOverlapError("OpenCV Hamming matching failed") from error
    mutual = [match for (left, right), match in sorted(forward.items()) if (right, left) in backward]
    ratio_count = len(forward)
    if len(mutual) < config.minimum_matches:
        return _empty_result(
            "match_insufficient",
            keypoints_previous=len(previous_keypoints),
            keypoints_current=len(current_keypoints),
            ratio_matches=ratio_count,
            mutual_matches=len(mutual),
            reference_frame_id=reference_frame_id,
            frame_id=frame_id,
        )
    try:
        previous_points = np.asarray([previous_keypoints[match.queryIdx].pt for match in mutual], dtype=np.float64)
        current_points = np.asarray([current_keypoints[match.trainIdx].pt for match in mutual], dtype=np.float64)
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise VisualOverlapError("OpenCV keypoint/match indices are inconsistent") from error
    try:
        matrix, mask = cv2_module.findFundamentalMat(
            previous_points,
            current_points,
            cv2_module.FM_RANSAC,
            config.ransac_pixel_limit,
        )
    except (AttributeError, TypeError, ValueError) as error:
        return _empty_result(
            "geometric_verification_failed",
            keypoints_previous=len(previous_keypoints),
            keypoints_current=len(current_keypoints),
            ratio_matches=ratio_count,
            mutual_matches=len(mutual),
            reference_frame_id=reference_frame_id,
            frame_id=frame_id,
        )
    if matrix is None or mask is None:
        return _empty_result(
            "geometric_verification_failed",
            keypoints_previous=len(previous_keypoints),
            keypoints_current=len(current_keypoints),
            ratio_matches=ratio_count,
            mutual_matches=len(mutual),
            reference_frame_id=reference_frame_id,
            frame_id=frame_id,
        )
    inlier_mask = np.asarray(mask).reshape(-1)
    if inlier_mask.size != len(mutual):
        return _empty_result(
            "geometric_verification_failed",
            keypoints_previous=len(previous_keypoints),
            keypoints_current=len(current_keypoints),
            ratio_matches=ratio_count,
            mutual_matches=len(mutual),
            reference_frame_id=reference_frame_id,
            frame_id=frame_id,
        )
    inlier_mask = inlier_mask.astype(bool)
    inliers = int(inlier_mask.sum())
    if inliers < 8:
        return _empty_result(
            "geometric_verification_failed",
            keypoints_previous=len(previous_keypoints),
            keypoints_current=len(current_keypoints),
            ratio_matches=ratio_count,
            mutual_matches=len(mutual),
            reference_frame_id=reference_frame_id,
            frame_id=frame_id,
        )
    height, width = previous.shape[:2]
    previous_coverage = grid_endpoint_coverage(
        previous_points[inlier_mask],
        image_width=width,
        image_height=height,
        grid_columns=config.grid_columns,
        grid_rows=config.grid_rows,
    )
    current_coverage = grid_endpoint_coverage(
        current_points[inlier_mask],
        image_width=width,
        image_height=height,
        grid_columns=config.grid_columns,
        grid_rows=config.grid_rows,
    )
    inlier_ratio = float(inliers / len(mutual))
    score = float(np.clip(min(previous_coverage, current_coverage) * inlier_ratio, 0.0, 1.0))
    return VisualOverlapResult(
        score=score,
        status="ok",
        keypoints_previous=len(previous_keypoints),
        keypoints_current=len(current_keypoints),
        ratio_matches=ratio_count,
        mutual_matches=len(mutual),
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        previous_grid_coverage=previous_coverage,
        current_grid_coverage=current_coverage,
        reference_frame_id=reference_frame_id,
        frame_id=frame_id,
    )


def online_visual_correspondence_series(
    frames: Sequence[ArrayLike],
    *,
    config: VisualOverlapConfig = VisualOverlapConfig(),
    cv2_module: Any | None = None,
) -> list[VisualOverlapResult | None]:
    """Score each frame only against its immediate predecessor; frame zero is null."""

    if not frames:
        raise VisualOverlapError("visual correspondence series must contain at least one frame")
    results: list[VisualOverlapResult | None] = [None]
    for index in range(1, len(frames)):
        results.append(
            online_visual_correspondence_coverage(
                frames[index - 1],
                frames[index],
                config=config,
                cv2_module=cv2_module,
                reference_frame_id=index - 1,
                frame_id=index,
            )
        )
    return results


def _intrinsics_matrix(intrinsics: ArrayLike | Mapping[str, float]) -> FloatArray:
    if isinstance(intrinsics, Mapping):
        try:
            fx, fy, cx, cy = (float(intrinsics[name]) for name in ("fx", "fy", "cx", "cy"))
        except (KeyError, TypeError, ValueError) as error:
            raise VisualOverlapError("intrinsics mapping needs finite fx, fy, cx, cy") from error
        matrix = np.asarray(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)), dtype=np.float64)
    else:
        matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise VisualOverlapError("intrinsics must be a finite 3x3 camera matrix with positive focal lengths")
    return matrix


def _camera_to_world_matrix(pose: ArrayLike, *, name: str) -> FloatArray:
    matrix = np.asarray(pose, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise VisualOverlapError(f"{name} must be a finite 4x4 camera-to-world matrix")
    try:
        np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        raise VisualOverlapError(f"{name} must be invertible") from error
    return matrix


def _depth_metres(depth: ArrayLike, *, depth_scale: float, name: str) -> FloatArray:
    if not math.isfinite(depth_scale) or depth_scale <= 0:
        raise VisualOverlapError("depth_scale must be positive and finite")
    array = np.asarray(depth, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1 or not np.all(np.isfinite(array)):
        raise VisualOverlapError(f"{name} must be a finite non-empty depth image")
    return array / depth_scale


def _directional_reprojection_fraction(
    source_depth: FloatArray,
    target_depth: FloatArray,
    source_camera_to_world: FloatArray,
    target_camera_to_world: FloatArray,
    intrinsics: FloatArray,
    *,
    stride: int,
    max_relative_depth_error: float,
) -> tuple[float, int, int]:
    height, width = source_depth.shape
    target_height, target_width = target_depth.shape
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    source_z = source_depth[ys, xs].reshape(-1)
    source_x = xs.reshape(-1).astype(np.float64)
    source_y = ys.reshape(-1).astype(np.float64)
    valid = source_z > 0.0
    if not np.any(valid):
        return 0.0, 0, 0
    source_z = source_z[valid]
    source_x = source_x[valid]
    source_y = source_y[valid]
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    source_points = np.vstack(
        (
            (source_x - cx) * source_z / fx,
            (source_y - cy) * source_z / fy,
            source_z,
            np.ones_like(source_z),
        )
    )
    target_points = np.linalg.inv(target_camera_to_world) @ source_camera_to_world @ source_points
    target_z = target_points[2]
    projectable = target_z > 0.0
    projected_x = np.rint(fx * target_points[0] / np.where(projectable, target_z, 1.0) + cx).astype(int)
    projected_y = np.rint(fy * target_points[1] / np.where(projectable, target_z, 1.0) + cy).astype(int)
    in_bounds = projectable & (projected_x >= 0) & (projected_x < target_width) & (projected_y >= 0) & (projected_y < target_height)
    consistent = np.zeros(source_z.shape, dtype=bool)
    if np.any(in_bounds):
        observed = target_depth[projected_y[in_bounds], projected_x[in_bounds]]
        expected = target_z[in_bounds]
        consistent[in_bounds] = (observed > 0.0) & (np.abs(observed - expected) <= max_relative_depth_error * expected)
    valid_count = int(source_z.size)
    consistent_count = int(consistent.sum())
    return float(consistent_count / valid_count), valid_count, consistent_count


def gt_depth_reprojection_overlap(
    reference_depth: ArrayLike,
    target_depth: ArrayLike,
    reference_camera_to_world: ArrayLike,
    target_camera_to_world: ArrayLike,
    intrinsics: ArrayLike | Mapping[str, float],
    *,
    depth_scale: float = 5000.0,
    stride: int = 4,
    max_relative_depth_error: float = 0.05,
) -> GTDepthReprojectionOverlap:
    """Measure offline bidirectional depth-consistent reprojection overlap.

    The result is suitable only for source/donor selection and severity audit.
    It must never be copied to an online health ledger or a detector feature.
    ``camera_to_world`` is explicit to avoid silently assuming a pose convention.
    """

    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise VisualOverlapError("stride must be a positive integer")
    if not math.isfinite(max_relative_depth_error) or max_relative_depth_error <= 0:
        raise VisualOverlapError("max_relative_depth_error must be positive and finite")
    reference = _depth_metres(reference_depth, depth_scale=depth_scale, name="reference_depth")
    target = _depth_metres(target_depth, depth_scale=depth_scale, name="target_depth")
    matrix = _intrinsics_matrix(intrinsics)
    reference_pose = _camera_to_world_matrix(reference_camera_to_world, name="reference_camera_to_world")
    target_pose = _camera_to_world_matrix(target_camera_to_world, name="target_camera_to_world")
    forward, reference_valid, reference_consistent = _directional_reprojection_fraction(
        reference,
        target,
        reference_pose,
        target_pose,
        matrix,
        stride=stride,
        max_relative_depth_error=max_relative_depth_error,
    )
    backward, target_valid, target_consistent = _directional_reprojection_fraction(
        target,
        reference,
        target_pose,
        reference_pose,
        matrix,
        stride=stride,
        max_relative_depth_error=max_relative_depth_error,
    )
    return GTDepthReprojectionOverlap(
        score=min(forward, backward),
        reference_to_target=forward,
        target_to_reference=backward,
        reference_valid_samples=reference_valid,
        target_valid_samples=target_valid,
        reference_consistent_samples=reference_consistent,
        target_consistent_samples=target_consistent,
    )


__all__ = [
    "DEFAULT_GRID_COLUMNS",
    "DEFAULT_GRID_ROWS",
    "DEFAULT_MINIMUM_MATCHES",
    "DEFAULT_NFEATURES",
    "DEFAULT_RANSAC_PIXEL_LIMIT",
    "DEFAULT_RATIO_THRESHOLD",
    "GTDepthReprojectionOverlap",
    "SCHEMA_VERSION",
    "VisualOverlapConfig",
    "VisualOverlapError",
    "VisualOverlapResult",
    "configure_opencv_determinism",
    "grid_endpoint_coverage",
    "gt_depth_reprojection_overlap",
    "normalized_tensor_to_uint8_rgb",
    "online_visual_correspondence_coverage",
    "online_visual_correspondence_series",
    "opencv_runtime_provenance",
]
