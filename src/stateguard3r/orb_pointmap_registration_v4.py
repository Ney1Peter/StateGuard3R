"""Causal ORB correspondence to a past safe pointmap for recovery v4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .geometric_registration_v4 import RegistrationResult, register_anchor_to_current
from .visual_overlap import (
    DEFAULT_NFEATURES,
    DEFAULT_RATIO_THRESHOLD,
    _gray_rgb,
    _load_cv2,
    _ratio_matches,
    configure_opencv_determinism,
    normalized_tensor_to_uint8_rgb,
)


class ORBPointmapRegistrationError(RuntimeError):
    """No causal, geometrically usable registration could be formed."""


@dataclass(frozen=True)
class ORBPointmapRegistrationEvidence:
    anchor_keypoints: int
    current_keypoints: int
    ratio_matches: int
    mutual_matches: int
    finite_3d_pairs: int
    registration: RegistrationResult


def _pointmap(value: Any, *, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim == 4 and points.shape[0] == 1:
        points = points[0]
    if points.ndim != 3 or points.shape[2] != 3:
        raise ORBPointmapRegistrationError(f"{label} must have shape (H, W, 3) or (1, H, W, 3)")
    return points


def _nearest_pairs(
    anchor_points: Any, current_points: Any, anchor_pixels: Iterable[Sequence[float]], current_pixels: Iterable[Sequence[float]]
) -> tuple[np.ndarray, np.ndarray]:
    anchor, current = _pointmap(anchor_points, label="anchor pointmap"), _pointmap(current_points, label="current pointmap")
    if anchor.shape != current.shape:
        raise ORBPointmapRegistrationError("anchor/current pointmap shapes differ")
    left, right = np.asarray(list(anchor_pixels), dtype=np.float64), np.asarray(list(current_pixels), dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 2:
        raise ORBPointmapRegistrationError("matched pixels must be aligned shape (N, 2)")
    left, right = np.rint(left).astype(np.int64), np.rint(right).astype(np.int64)
    height, width = anchor.shape[:2]
    valid = (
        (left[:, 0] >= 0) & (left[:, 0] < width) & (left[:, 1] >= 0) & (left[:, 1] < height)
        & (right[:, 0] >= 0) & (right[:, 0] < width) & (right[:, 1] >= 0) & (right[:, 1] < height)
    )
    anchor_3d, current_3d = anchor[left[valid, 1], left[valid, 0]], current[right[valid, 1], right[valid, 0]]
    finite = np.isfinite(anchor_3d).all(axis=1) & np.isfinite(current_3d).all(axis=1)
    return anchor_3d[finite], current_3d[finite]


def register_anchor_orb_3d3d(anchor_rgb: Any, current_rgb: Any, anchor_reference_points: Any, current_self_points: Any, *, cv2_module: Any | None = None) -> ORBPointmapRegistrationEvidence:
    """Return an alarm-frame current-self -> anchor-reference registration.

    Inputs are deliberately limited to a past safe RGB/pointmap and the current
    RGB/self-pointmap.  This signature contains no pose, model state, detector
    history, GT, timestamp, label, source index, or future frame capability.
    """
    anchor = normalized_tensor_to_uint8_rgb(anchor_rgb)
    current = normalized_tensor_to_uint8_rgb(current_rgb)
    if anchor.shape != current.shape:
        raise ORBPointmapRegistrationError("anchor/current RGB shapes differ")
    cv2 = _load_cv2() if cv2_module is None else cv2_module
    configure_opencv_determinism(cv2, rng_seed=0)
    try:
        detector = cv2.ORB_create(nfeatures=DEFAULT_NFEATURES)
        anchor_keys, anchor_desc = detector.detectAndCompute(_gray_rgb(anchor), None)
        current_keys, current_desc = detector.detectAndCompute(_gray_rgb(current), None)
    except (AttributeError, TypeError, ValueError) as error:
        raise ORBPointmapRegistrationError("ORB feature extraction failed") from error
    anchor_keys, current_keys = list(anchor_keys or ()), list(current_keys or ())
    if anchor_desc is None or current_desc is None:
        raise ORBPointmapRegistrationError("ORB descriptors are insufficient")
    try:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        forward = _ratio_matches(matcher.knnMatch(anchor_desc, current_desc, k=2), DEFAULT_RATIO_THRESHOLD)
        backward = _ratio_matches(matcher.knnMatch(current_desc, anchor_desc, k=2), DEFAULT_RATIO_THRESHOLD)
    except (AttributeError, TypeError, ValueError) as error:
        raise ORBPointmapRegistrationError("ORB Hamming matching failed") from error
    mutual = [match for (left, right), match in sorted(forward.items()) if (right, left) in backward]
    anchor_pixels = [anchor_keys[match.queryIdx].pt for match in mutual]
    current_pixels = [current_keys[match.trainIdx].pt for match in mutual]
    anchor_3d, current_3d = _nearest_pairs(anchor_reference_points, current_self_points, anchor_pixels, current_pixels)
    try:
        registration = register_anchor_to_current(anchor_3d, current_3d)
    except Exception as error:
        raise ORBPointmapRegistrationError(f"3D registration unavailable: {error}") from error
    return ORBPointmapRegistrationEvidence(len(anchor_keys), len(current_keys), len(forward), len(mutual), len(anchor_3d), registration)


__all__ = ["ORBPointmapRegistrationError", "ORBPointmapRegistrationEvidence", "register_anchor_orb_3d3d"]
