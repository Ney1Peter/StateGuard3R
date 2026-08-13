"""v17's own causal RGB correspondence measurement.

Only a copied previous/current model-ready RGB pair reaches this module.  It
does not accept model output, pose, annotation, frame-selection data, or a
future image.  OpenCV is loaded lazily so CPU-only checks never initialize a
CUDA runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


class OnlineVisualOverlapV17Error(ValueError):
    """A v17 previous/current RGB measurement violates its narrow contract."""


@dataclass(frozen=True)
class OnlineVisualOverlapConfigV17:
    nfeatures: int = 2000
    grid_columns: int = 8
    grid_rows: int = 6
    minimum_matches: int = 12
    ratio_threshold: float = 0.80
    ransac_pixel_limit: float = 1.0
    rng_seed: int = 0

    def __post_init__(self) -> None:
        for name in ("nfeatures", "grid_columns", "grid_rows", "minimum_matches"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise OnlineVisualOverlapV17Error(f"{name} must be a positive integer")
        if self.minimum_matches < 8:
            raise OnlineVisualOverlapV17Error("minimum_matches must support a fundamental matrix")
        if not math.isfinite(self.ratio_threshold) or not 0.0 < self.ratio_threshold < 1.0:
            raise OnlineVisualOverlapV17Error("ratio_threshold must be finite in (0, 1)")
        if not math.isfinite(self.ransac_pixel_limit) or self.ransac_pixel_limit <= 0.0:
            raise OnlineVisualOverlapV17Error("ransac_pixel_limit must be finite and positive")
        if type(self.rng_seed) is not int:
            raise OnlineVisualOverlapV17Error("rng_seed must be an integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OnlineVisualOverlapResultV17:
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
    reference_frame_id: int | None
    frame_id: int | None

    def __post_init__(self) -> None:
        for name in ("score", "inlier_ratio", "previous_grid_coverage", "current_grid_coverage"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise OnlineVisualOverlapV17Error(f"{name} must be finite in [0, 1]")
        for name in ("keypoints_previous", "keypoints_current", "ratio_matches", "mutual_matches", "inliers"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise OnlineVisualOverlapV17Error(f"{name} must be a non-negative integer")
        if self.inliers > self.mutual_matches:
            raise OnlineVisualOverlapV17Error("inliers cannot exceed mutual matches")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalized_tensor_to_uint8_rgb_v17(image: Any) -> np.ndarray:
    """Copy one finite batch-one normalized RGB value into an owned array."""

    try:
        array = np.asarray(image)
    except Exception as error:
        raise OnlineVisualOverlapV17Error("RGB input cannot be projected to a CPU array") from error
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise OnlineVisualOverlapV17Error("batched RGB must have batch size one")
        array = array[0]
    if array.ndim != 3:
        raise OnlineVisualOverlapV17Error("RGB must have exactly three dimensions")
    if array.shape[0] == 3:
        rgb = np.moveaxis(array, 0, -1)
    elif array.shape[-1] == 3:
        rgb = array
    else:
        raise OnlineVisualOverlapV17Error("RGB must have exactly three channels")
    numeric = np.asarray(rgb, dtype=np.float64)
    if numeric.shape[0] < 1 or numeric.shape[1] < 1 or not bool(np.isfinite(numeric).all()):
        raise OnlineVisualOverlapV17Error("RGB must be nonempty and finite")
    if float(numeric.min()) < -1.000001 or float(numeric.max()) > 1.000001:
        raise OnlineVisualOverlapV17Error("RGB must be normalized to [-1, 1]")
    return np.ascontiguousarray(np.rint(np.clip((numeric + 1.0) * 127.5, 0.0, 255.0)).astype(np.uint8)).copy()


def _cv2() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as error:
        raise OnlineVisualOverlapV17Error("OpenCV is unavailable in this runtime") from error


def _configure(cv2_module: Any, *, seed: int) -> None:
    try:
        cv2_module.setNumThreads(1)
        cv2_module.setRNGSeed(seed)
        cv2_module.ocl.setUseOpenCL(False)
    except (AttributeError, TypeError, ValueError) as error:
        raise OnlineVisualOverlapV17Error("OpenCV deterministic controls are unavailable") from error


def online_visual_overlap_provenance_v17(cv2_module: Any | None = None) -> dict[str, Any]:
    cv2_module = _cv2() if cv2_module is None else cv2_module
    _configure(cv2_module, seed=0)
    module = Path(str(cv2_module.__file__)).resolve(strict=True)
    candidates = sorted(
        value
        for pattern in ("cv2*.so", "cv2*.pyd", "cv2*.dll", "cv2*.dylib")
        for value in module.parent.glob(pattern)
        if value.is_file()
    )
    binary = candidates[0] if candidates else module
    return {
        "version": str(cv2_module.__version__),
        "module_path": str(module),
        "module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        "binary_path": str(binary.resolve(strict=True)),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "build_info_sha256": hashlib.sha256(str(cv2_module.getBuildInformation()).encode("utf-8")).hexdigest(),
        "threads": int(cv2_module.getNumThreads()),
        "opencl_enabled": bool(cv2_module.ocl.useOpenCL()),
        "rng_seed": 0,
    }


def _gray(rgb: np.ndarray) -> np.ndarray:
    return np.rint(rgb[..., 0].astype(np.float64) * 0.299 + rgb[..., 1].astype(np.float64) * 0.587 + rgb[..., 2].astype(np.float64) * 0.114).astype(np.uint8)


def _ratio(rows: Iterable[Sequence[Any]], threshold: float) -> dict[tuple[int, int], Any]:
    accepted: dict[tuple[int, int], Any] = {}
    for row in rows:
        if len(row) < 2:
            continue
        first, second = row[0], row[1]
        if math.isfinite(float(first.distance)) and math.isfinite(float(second.distance)) and float(first.distance) < threshold * float(second.distance):
            accepted[(int(first.queryIdx), int(first.trainIdx))] = first
    return accepted


def _coverage(points: np.ndarray, *, width: int, height: int, columns: int, rows: int) -> float:
    if points.size == 0:
        return 0.0
    if points.ndim != 2 or points.shape[1] != 2:
        raise OnlineVisualOverlapV17Error("match endpoints must have shape (N, 2)")
    valid = np.isfinite(points).all(axis=1) & (points[:, 0] >= 0.0) & (points[:, 0] < width) & (points[:, 1] >= 0.0) & (points[:, 1] < height)
    if not bool(valid.any()):
        return 0.0
    selected = points[valid]
    x = np.minimum((selected[:, 0] * columns / width).astype(int), columns - 1)
    y = np.minimum((selected[:, 1] * rows / height).astype(int), rows - 1)
    return float(len({(int(a), int(b)) for a, b in zip(y, x, strict=True)}) / (columns * rows))


def _empty(status: str, *, previous: int, current: int, ratio: int, mutual: int, reference_frame_id: int | None, frame_id: int | None) -> OnlineVisualOverlapResultV17:
    return OnlineVisualOverlapResultV17(0.0, status, previous, current, ratio, mutual, 0, 0.0, 0.0, 0.0, reference_frame_id, frame_id)


def online_visual_correspondence_coverage_v17(previous_rgb: Any, current_rgb: Any, *, config: OnlineVisualOverlapConfigV17 = OnlineVisualOverlapConfigV17(), cv2_module: Any | None = None, reference_frame_id: int | None = None, frame_id: int | None = None) -> OnlineVisualOverlapResultV17:
    """Measure only a predecessor/current RGB pair and return a finite score."""

    previous, current = normalized_tensor_to_uint8_rgb_v17(previous_rgb), normalized_tensor_to_uint8_rgb_v17(current_rgb)
    if previous.shape[:2] != current.shape[:2]:
        raise OnlineVisualOverlapV17Error("consecutive RGB frames must share image shape")
    cv2_module = _cv2() if cv2_module is None else cv2_module
    _configure(cv2_module, seed=config.rng_seed)
    try:
        detector = cv2_module.ORB_create(nfeatures=config.nfeatures)
        prior_keys, prior_desc = detector.detectAndCompute(_gray(previous), None)
        current_keys, current_desc = detector.detectAndCompute(_gray(current), None)
    except (AttributeError, TypeError, ValueError) as error:
        raise OnlineVisualOverlapV17Error("OpenCV ORB extraction failed") from error
    prior_keys, current_keys = list(prior_keys or ()), list(current_keys or ())
    if prior_desc is None or current_desc is None:
        return _empty("descriptor_insufficient", previous=len(prior_keys), current=len(current_keys), ratio=0, mutual=0, reference_frame_id=reference_frame_id, frame_id=frame_id)
    try:
        matcher = cv2_module.BFMatcher(cv2_module.NORM_HAMMING, crossCheck=False)
        forward = _ratio(matcher.knnMatch(prior_desc, current_desc, k=2), config.ratio_threshold)
        backward = _ratio(matcher.knnMatch(current_desc, prior_desc, k=2), config.ratio_threshold)
    except (AttributeError, TypeError, ValueError) as error:
        raise OnlineVisualOverlapV17Error("OpenCV Hamming matching failed") from error
    mutual = [match for left_right, match in sorted(forward.items()) if (left_right[1], left_right[0]) in backward]
    if len(mutual) < config.minimum_matches:
        return _empty("match_insufficient", previous=len(prior_keys), current=len(current_keys), ratio=len(forward), mutual=len(mutual), reference_frame_id=reference_frame_id, frame_id=frame_id)
    try:
        prior_xy = np.asarray([prior_keys[value.queryIdx].pt for value in mutual], dtype=np.float64)
        current_xy = np.asarray([current_keys[value.trainIdx].pt for value in mutual], dtype=np.float64)
        matrix, mask = cv2_module.findFundamentalMat(prior_xy, current_xy, cv2_module.FM_RANSAC, config.ransac_pixel_limit)
    except (AttributeError, IndexError, TypeError, ValueError):
        matrix, mask = None, None
    if matrix is None or mask is None or np.asarray(mask).reshape(-1).size != len(mutual):
        return _empty("geometric_verification_failed", previous=len(prior_keys), current=len(current_keys), ratio=len(forward), mutual=len(mutual), reference_frame_id=reference_frame_id, frame_id=frame_id)
    keep = np.asarray(mask).reshape(-1).astype(bool)
    inliers = int(keep.sum())
    if inliers < 8:
        return _empty("geometric_verification_failed", previous=len(prior_keys), current=len(current_keys), ratio=len(forward), mutual=len(mutual), reference_frame_id=reference_frame_id, frame_id=frame_id)
    height, width = previous.shape[:2]
    before = _coverage(prior_xy[keep], width=width, height=height, columns=config.grid_columns, rows=config.grid_rows)
    after = _coverage(current_xy[keep], width=width, height=height, columns=config.grid_columns, rows=config.grid_rows)
    inlier_ratio = float(inliers / len(mutual))
    return OnlineVisualOverlapResultV17(float(np.clip(min(before, after) * inlier_ratio, 0.0, 1.0)), "ok", len(prior_keys), len(current_keys), len(forward), len(mutual), inliers, inlier_ratio, before, after, reference_frame_id, frame_id)


__all__ = ["OnlineVisualOverlapConfigV17", "OnlineVisualOverlapResultV17", "OnlineVisualOverlapV17Error", "normalized_tensor_to_uint8_rgb_v17", "online_visual_correspondence_coverage_v17", "online_visual_overlap_provenance_v17"]
