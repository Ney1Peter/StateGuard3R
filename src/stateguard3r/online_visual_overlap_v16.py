"""Narrow causal previous/current RGB correspondence score for v16 only."""

from __future__ import annotations

import hashlib
import importlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


DEFAULT_GRID_COLUMNS = 8
DEFAULT_GRID_ROWS = 6
DEFAULT_NFEATURES = 2000
DEFAULT_MINIMUM_MATCHES = 12
DEFAULT_RATIO_THRESHOLD = 0.80
DEFAULT_RANSAC_PIXEL_LIMIT = 1.0
DEFAULT_RNG_SEED = 0
UInt8Array = NDArray[np.uint8]


class OnlineVisualOverlapV16Error(ValueError):
    """The v16 online RGB-only correspondence contract was violated."""


@dataclass(frozen=True)
class OnlineVisualOverlapConfigV16:
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
                raise OnlineVisualOverlapV16Error(f"{field} must be a positive integer")
        if self.minimum_matches < 8:
            raise OnlineVisualOverlapV16Error("minimum_matches must be at least eight")
        if not math.isfinite(self.ratio_threshold) or not 0.0 < self.ratio_threshold < 1.0:
            raise OnlineVisualOverlapV16Error("ratio_threshold must be finite and in (0, 1)")
        if not math.isfinite(self.ransac_pixel_limit) or self.ransac_pixel_limit <= 0.0:
            raise OnlineVisualOverlapV16Error("ransac_pixel_limit must be finite and positive")
        if isinstance(self.rng_seed, bool) or not isinstance(self.rng_seed, int):
            raise OnlineVisualOverlapV16Error("rng_seed must be an integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OnlineVisualOverlapResultV16:
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
        for field in ("score", "inlier_ratio", "previous_grid_coverage", "current_grid_coverage"):
            value = float(getattr(self, field))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise OnlineVisualOverlapV16Error(f"{field} must be finite and in [0, 1]")
        for field in ("keypoints_previous", "keypoints_current", "ratio_matches", "mutual_matches", "inliers"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise OnlineVisualOverlapV16Error(f"{field} must be a non-negative integer")
        if self.inliers > self.mutual_matches:
            raise OnlineVisualOverlapV16Error("inliers may not exceed mutual_matches")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalized_tensor_to_uint8_rgb_v16(image: ArrayLike) -> UInt8Array:
    """Copy a finite batch-one normalized RGB tensor into an owned uint8 array."""
    array = np.asarray(image)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise OnlineVisualOverlapV16Error("batched RGB must have batch size one")
        array = array[0]
    if array.ndim != 3:
        raise OnlineVisualOverlapV16Error("model-ready RGB must have three dimensions")
    if array.shape[0] == 3:
        rgb = np.moveaxis(array, 0, -1)
    elif array.shape[-1] == 3:
        rgb = array
    else:
        raise OnlineVisualOverlapV16Error("model-ready RGB must have exactly three channels")
    numeric = np.asarray(rgb, dtype=np.float64)
    if numeric.shape[0] < 1 or numeric.shape[1] < 1 or not np.all(np.isfinite(numeric)):
        raise OnlineVisualOverlapV16Error("model-ready RGB must be finite and nonempty")
    if float(numeric.min()) < -1.0 - 1e-6 or float(numeric.max()) > 1.0 + 1e-6:
        raise OnlineVisualOverlapV16Error("model-ready RGB must be normalized to [-1, 1]")
    return np.ascontiguousarray(np.rint(np.clip((numeric + 1.0) * 127.5, 0.0, 255.0)).astype(np.uint8)).copy()


def _gray_rgb(image: UInt8Array) -> UInt8Array:
    if image.ndim != 3 or image.shape[2] != 3:
        raise OnlineVisualOverlapV16Error("RGB image must have shape HxWx3")
    return np.rint(image[..., 0].astype(np.float64) * 0.299 + image[..., 1].astype(np.float64) * 0.587 + image[..., 2].astype(np.float64) * 0.114).astype(np.uint8)


def _load_cv2() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as error:
        raise OnlineVisualOverlapV16Error("OpenCV is required for v16 online RGB overlap") from error


def _configure_cv2(cv2_module: Any, *, rng_seed: int) -> None:
    try:
        cv2_module.setNumThreads(1)
        cv2_module.setRNGSeed(rng_seed)
        cv2_module.ocl.setUseOpenCL(False)
    except (AttributeError, TypeError, ValueError) as error:
        raise OnlineVisualOverlapV16Error("OpenCV lacks deterministic controls") from error


def online_visual_overlap_provenance_v16(cv2_module: Any | None = None) -> dict[str, Any]:
    cv2_module = _load_cv2() if cv2_module is None else cv2_module
    _configure_cv2(cv2_module, rng_seed=DEFAULT_RNG_SEED)
    module_file = Path(str(cv2_module.__file__))
    binaries = sorted(candidate for pattern in ("cv2*.so", "cv2*.pyd", "cv2*.dll", "cv2*.dylib") for candidate in module_file.parent.glob(pattern) if candidate.is_file())
    binary = binaries[0] if binaries else module_file
    try:
        module_hash = hashlib.sha256(module_file.read_bytes()).hexdigest()
        binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    except OSError as error:
        raise OnlineVisualOverlapV16Error("cannot hash OpenCV runtime") from error
    return {
        "version": str(cv2_module.__version__),
        "module_path": str(module_file.resolve(strict=True)),
        "module_sha256": module_hash,
        "binary_path": str(binary.resolve(strict=True)),
        "binary_sha256": binary_hash,
        "build_info_sha256": hashlib.sha256(str(cv2_module.getBuildInformation()).encode("utf-8")).hexdigest(),
        "threads": int(cv2_module.getNumThreads()),
        "opencl_enabled": bool(cv2_module.ocl.useOpenCL()),
        "rng_seed": DEFAULT_RNG_SEED,
    }


def _ratio_matches(rows: Iterable[Sequence[Any]], threshold: float) -> dict[tuple[int, int], Any]:
    accepted: dict[tuple[int, int], Any] = {}
    for row in rows:
        if len(row) < 2:
            continue
        best, second = row[0], row[1]
        if math.isfinite(float(best.distance)) and math.isfinite(float(second.distance)) and float(best.distance) < threshold * float(second.distance):
            accepted[(int(best.queryIdx), int(best.trainIdx))] = best
    return accepted


def _grid_coverage(points: ArrayLike, *, width: int, height: int, columns: int, rows: int) -> float:
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.size == 0:
        return 0.0
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise OnlineVisualOverlapV16Error("point coordinates must have shape (N, 2)")
    valid = np.isfinite(coordinates).all(axis=1) & (coordinates[:, 0] >= 0.0) & (coordinates[:, 0] < width) & (coordinates[:, 1] >= 0.0) & (coordinates[:, 1] < height)
    if not np.any(valid):
        return 0.0
    selected = coordinates[valid]
    xs = np.minimum((selected[:, 0] * columns / width).astype(int), columns - 1)
    ys = np.minimum((selected[:, 1] * rows / height).astype(int), rows - 1)
    return float(len({(int(y), int(x)) for x, y in zip(xs, ys, strict=True)}) / (columns * rows))


def _empty_result(status: str, *, previous: int, current: int, ratio: int = 0, mutual: int = 0, reference_frame_id: int | None, frame_id: int | None) -> OnlineVisualOverlapResultV16:
    return OnlineVisualOverlapResultV16(0.0, status, previous, current, ratio, mutual, 0, 0.0, 0.0, 0.0, reference_frame_id, frame_id)


def online_visual_correspondence_coverage_v16(previous_rgb: ArrayLike, current_rgb: ArrayLike, *, config: OnlineVisualOverlapConfigV16 = OnlineVisualOverlapConfigV16(), cv2_module: Any | None = None, reference_frame_id: int | None = None, frame_id: int | None = None) -> OnlineVisualOverlapResultV16:
    """Compute deterministic ORB coverage from exactly previous and current RGB."""
    previous, current = normalized_tensor_to_uint8_rgb_v16(previous_rgb), normalized_tensor_to_uint8_rgb_v16(current_rgb)
    if previous.shape[:2] != current.shape[:2]:
        raise OnlineVisualOverlapV16Error("consecutive RGB images must share a shape")
    cv2_module = _load_cv2() if cv2_module is None else cv2_module
    _configure_cv2(cv2_module, rng_seed=config.rng_seed)
    try:
        detector = cv2_module.ORB_create(nfeatures=config.nfeatures)
        prior_points, prior_descriptors = detector.detectAndCompute(_gray_rgb(previous), None)
        current_points, current_descriptors = detector.detectAndCompute(_gray_rgb(current), None)
    except (AttributeError, TypeError, ValueError) as error:
        raise OnlineVisualOverlapV16Error("OpenCV ORB extraction failed") from error
    prior_points, current_points = list(prior_points or ()), list(current_points or ())
    if prior_descriptors is None or current_descriptors is None:
        return _empty_result("descriptor_insufficient", previous=len(prior_points), current=len(current_points), reference_frame_id=reference_frame_id, frame_id=frame_id)
    try:
        matcher = cv2_module.BFMatcher(cv2_module.NORM_HAMMING, crossCheck=False)
        forward = _ratio_matches(matcher.knnMatch(prior_descriptors, current_descriptors, k=2), config.ratio_threshold)
        backward = _ratio_matches(matcher.knnMatch(current_descriptors, prior_descriptors, k=2), config.ratio_threshold)
    except (AttributeError, TypeError, ValueError) as error:
        raise OnlineVisualOverlapV16Error("OpenCV Hamming matching failed") from error
    mutual = [match for (left, right), match in sorted(forward.items()) if (right, left) in backward]
    if len(mutual) < config.minimum_matches:
        return _empty_result("match_insufficient", previous=len(prior_points), current=len(current_points), ratio=len(forward), mutual=len(mutual), reference_frame_id=reference_frame_id, frame_id=frame_id)
    try:
        prior_xy = np.asarray([prior_points[match.queryIdx].pt for match in mutual], dtype=np.float64)
        current_xy = np.asarray([current_points[match.trainIdx].pt for match in mutual], dtype=np.float64)
        matrix, mask = cv2_module.findFundamentalMat(prior_xy, current_xy, cv2_module.FM_RANSAC, config.ransac_pixel_limit)
    except (AttributeError, IndexError, TypeError, ValueError):
        matrix, mask = None, None
    if matrix is None or mask is None or np.asarray(mask).reshape(-1).size != len(mutual):
        return _empty_result("geometric_verification_failed", previous=len(prior_points), current=len(current_points), ratio=len(forward), mutual=len(mutual), reference_frame_id=reference_frame_id, frame_id=frame_id)
    inlier_mask = np.asarray(mask).reshape(-1).astype(bool)
    inliers = int(inlier_mask.sum())
    if inliers < 8:
        return _empty_result("geometric_verification_failed", previous=len(prior_points), current=len(current_points), ratio=len(forward), mutual=len(mutual), reference_frame_id=reference_frame_id, frame_id=frame_id)
    height, width = previous.shape[:2]
    previous_coverage = _grid_coverage(prior_xy[inlier_mask], width=width, height=height, columns=config.grid_columns, rows=config.grid_rows)
    current_coverage = _grid_coverage(current_xy[inlier_mask], width=width, height=height, columns=config.grid_columns, rows=config.grid_rows)
    ratio = float(inliers / len(mutual))
    return OnlineVisualOverlapResultV16(float(np.clip(min(previous_coverage, current_coverage) * ratio, 0.0, 1.0)), "ok", len(prior_points), len(current_points), len(forward), len(mutual), inliers, ratio, previous_coverage, current_coverage, reference_frame_id, frame_id)


__all__ = ["OnlineVisualOverlapConfigV16", "OnlineVisualOverlapResultV16", "OnlineVisualOverlapV16Error", "normalized_tensor_to_uint8_rgb_v16", "online_visual_correspondence_coverage_v16", "online_visual_overlap_provenance_v16"]
