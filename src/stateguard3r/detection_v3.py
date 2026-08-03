"""Frozen-v2 continuous scores plus a causal capture-timestamp hard channel."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .timestamp_order_v3 import TimestampOrderError, validate_timestamp_order_sidecar


SCHEMA_VERSION = "stateguard3r.detection-v3.v1"
TIMESTAMP_SCORE_MARGIN = 1.0


class DetectorV3Error(ValueError):
    """Raised when a frozen continuous score cannot form a v3 hybrid."""


def hybridize_v2_continuous_scores(
    continuous_scores: ArrayLike,
    *,
    continuous_threshold: float,
    timestamp_sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the predeclared v3 hard order invariant to frozen v2 scores.

    A timestamp violation is an alarm regardless of continuous score.  For
    finite rank metrics it receives the fixed finite score
    ``max(continuous, threshold + 1.0)``.  The margin is an attribution
    separator, not a calibrated threshold or learned weight.
    """

    threshold = float(continuous_threshold)
    if not math.isfinite(threshold):
        raise DetectorV3Error("continuous_threshold must be finite")
    scores = np.asarray(continuous_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.all(np.isfinite(scores)):
        raise DetectorV3Error("continuous_scores must be a non-empty finite one-dimensional array")
    try:
        predicates = validate_timestamp_order_sidecar(timestamp_sidecar, require_available=True)
    except TimestampOrderError as error:
        raise DetectorV3Error(f"invalid timestamp sidecar: {error}") from error
    if len(predicates) != scores.size:
        raise DetectorV3Error("continuous scores and timestamp records have different lengths")
    timestamp_alarm = np.asarray([bool(value) if value is not None else False for value in predicates], dtype=bool)
    continuous_alarm = scores >= threshold
    hybrid_scores = scores.copy()
    hybrid_scores[timestamp_alarm] = np.maximum(hybrid_scores[timestamp_alarm], threshold + TIMESTAMP_SCORE_MARGIN)
    alarms = continuous_alarm | timestamp_alarm
    attribution = [
        {
            "frame_id": int(index),
            "continuous_alarm": bool(continuous_alarm[index]),
            "timestamp_order_alarm": bool(timestamp_alarm[index]),
            "hybrid_alarm": bool(alarms[index]),
        }
        for index in range(scores.size)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "continuous_threshold": threshold,
        "timestamp_score_margin": TIMESTAMP_SCORE_MARGIN,
        "decision_rule": "continuous_score_gte_threshold_or_timestamp_order_violation",
        "continuous_scores": scores,
        "hybrid_scores": hybrid_scores,
        "alarms": alarms,
        "attribution": attribution,
    }


__all__ = [
    "DetectorV3Error",
    "SCHEMA_VERSION",
    "TIMESTAMP_SCORE_MARGIN",
    "hybridize_v2_continuous_scores",
]
