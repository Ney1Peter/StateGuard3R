"""Training-free, causal detection metrics for StateGuard3R.

The rolling reference at frame 't' is computed from at most 'window'
finite observations strictly before 't'. Consequently, changing any future
frame cannot change a score already emitted by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


DEFAULT_WINDOW = 15
DEFAULT_THRESHOLD = 3.0
DEFAULT_SEED = 0
DEFAULT_EPSILON = 1e-6
DEFAULT_MAX_Z: float | None = None
CALIBRATION_SCHEMA_VERSION = "stateguard3r.detection-calibration.v0"

METHOD_NAMES: tuple[str, ...] = (
    "random",
    "update_magnitude_only",
    "reliability_only",
    "combined",
)
ROBUST_Z_SIGNALS = frozenset(
    ("geometric_residual", "pose_jump", "update_magnitude")
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SignalAlias:
    """One possible health-ledger field and its risk direction."""

    field: str
    direction: str


@dataclass(frozen=True)
class SignalSpec:
    """Canonical signal and ordered source-field aliases."""

    name: str
    aliases: tuple[SignalAlias, ...]


SIGNAL_SPECS: tuple[SignalSpec, ...] = (
    SignalSpec(
        "geometric_residual",
        (
            SignalAlias("geometric_residual", "high"),
            SignalAlias("geometric_or_pointmap_residual", "high"),
            SignalAlias("pointmap_residual", "high"),
            SignalAlias("relative_residual", "high"),
            SignalAlias("residual_score", "high"),
        ),
    ),
    SignalSpec(
        "pose_jump",
        (
            SignalAlias("pose_jump", "high"),
            SignalAlias("pose_jump_translation", "high"),
            SignalAlias("pose_translation_jump", "high"),
        ),
    ),
    SignalSpec(
        "update_magnitude",
        (
            SignalAlias("update_magnitude", "high"),
            SignalAlias("global_state_delta", "high"),
            SignalAlias("global_state_delta_mean", "high"),
            SignalAlias("state_delta_mean", "high"),
            SignalAlias("delta_norm_mean", "high"),
        ),
    ),
    SignalSpec(
        "overlap",
        (
            SignalAlias("overlap_score", "low"),
            SignalAlias("overlap", "low"),
        ),
    ),
    SignalSpec(
        "reliability",
        (
            SignalAlias("reliability", "low"),
            SignalAlias("reliability_mean", "low"),
            SignalAlias("state_reliability_mean", "low"),
            # ReCal3R's trace exposes U = 1 - calibrated reliability.
            SignalAlias("uncertainty_u", "high"),
            SignalAlias("uncertainty_u_mean", "high"),
            SignalAlias("unreliability_mean", "high"),
        ),
    ),
)


def _as_float_array(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"expected a one-dimensional series, got shape {array.shape}")
    return array


def trailing_rolling_median_mad(
    values: ArrayLike,
    window: int = DEFAULT_WINDOW,
) -> tuple[FloatArray, FloatArray]:
    """Return causal rolling median and MAD references.

    For index 't', only finite values in [max(0, t-window), t) are used.
    When no finite history exists, a finite current value is used as its own
    neutral reference, which makes its initial robust z-score zero. Missing
    current values remain missing.
    """

    if window < 1:
        raise ValueError("window must be at least 1")

    series = _as_float_array(values)
    medians = np.full(series.shape, np.nan, dtype=np.float64)
    mads = np.full(series.shape, np.nan, dtype=np.float64)

    for index, current in enumerate(series):
        history = series[max(0, index - window) : index]
        history = history[np.isfinite(history)]
        if history.size == 0:
            if np.isfinite(current):
                medians[index] = current
                mads[index] = 0.0
            continue

        median = float(np.median(history))
        medians[index] = median
        mads[index] = float(np.median(np.abs(history - median)))

    return medians, mads


def robust_z(
    values: ArrayLike,
    window: int = DEFAULT_WINDOW,
    epsilon: float = DEFAULT_EPSILON,
) -> FloatArray:
    """Compute absolute trailing median/MAD deviation without future leakage."""

    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a positive finite number")

    series = _as_float_array(values)
    medians, mads = trailing_rolling_median_mad(series, window=window)
    scales = mads + epsilon

    scores = np.full(series.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(series) & np.isfinite(medians) & np.isfinite(scales)
    scores[valid] = np.abs(series[valid] - medians[valid]) / scales[valid]
    return scores


def _coerce_numeric(value: Any) -> float:
    if value is None or isinstance(value, (dict, list, tuple)):
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _field_series(records: Sequence[Mapping[str, Any]], field: str) -> FloatArray:
    return np.asarray(
        [_coerce_numeric(record.get(field)) for record in records],
        dtype=np.float64,
    )


def _resolve_signal(
    records: Sequence[Mapping[str, Any]],
    spec: SignalSpec,
) -> dict[str, Any] | None:
    for alias in spec.aliases:
        raw = _field_series(records, alias.field)
        finite_frames = int(np.isfinite(raw).sum())
        if finite_frames == 0:
            continue
        if alias.direction not in ("high", "low"):
            # Guarded by the constant specification.
            raise ValueError(f"unknown risk direction: {alias.direction}")
        return {
            "signal": spec.name,
            "field": alias.field,
            "direction": alias.direction,
            "finite_frames": finite_frames,
            "values": raw,
        }
    return None


def _positive_component(
    values: FloatArray,
    *,
    window: int,
    epsilon: float,
    max_z: float | None,
) -> FloatArray:
    component = robust_z(values, window=window, epsilon=epsilon)
    finite = np.isfinite(component)
    if max_z is not None:
        component[finite] = np.minimum(component[finite], max_z)
    return component


def _component_for_signal(
    name: str,
    signal: Mapping[str, Any],
    *,
    window: int,
    epsilon: float,
    max_z: float | None,
) -> tuple[FloatArray, str]:
    values = signal["values"]
    direction = signal["direction"]

    if name in ROBUST_Z_SIGNALS:
        return (
            _positive_component(
                values,
                window=window,
                epsilon=epsilon,
                max_z=max_z,
            ),
            "absolute_trailing_robust_z",
        )

    component = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if name == "reliability" and direction == "low":
        component[finite] = -values[finite]
        transformation = "negated_value"
    elif name == "reliability":
        # ReCal3R exposes U = 1 - reliability, hence -reliability = U - 1.
        component[finite] = values[finite] - 1.0
        transformation = "value_minus_one"
    elif direction == "low":
        component[finite] = 1.0 - values[finite]
        transformation = "one_minus_value"
    else:
        component[finite] = values[finite]
        transformation = "direct_value"
    return component, transformation


def _aggregate_components(components: Sequence[FloatArray], length: int) -> FloatArray:
    if not components:
        return np.zeros(length, dtype=np.float64)

    matrix = np.vstack(components)
    finite = np.isfinite(matrix)
    return np.where(finite, matrix, 0.0).sum(axis=0)


def compute_baseline_scores(
    records: Sequence[Mapping[str, Any]],
    *,
    window: int = DEFAULT_WINDOW,
    seed: int = DEFAULT_SEED,
    epsilon: float = DEFAULT_EPSILON,
    max_z: float | None = DEFAULT_MAX_Z,
) -> dict[str, dict[str, Any]]:
    """Compute the four required detection-only baselines.

    Missing canonical signals are excluded. Partially missing signals
    participate only on frames where their value is finite. A method with no
    usable signal emits a neutral all-zero score and records no used signals.
    Residual, pose, and update signals use their absolute two-sided robust
    z-score; overlap uses ``1 - value`` and reliability uses ``-value``.
    """

    if window < 1:
        raise ValueError("window must be at least 1")
    if max_z is not None and (not math.isfinite(max_z) or max_z <= 0):
        raise ValueError("max_z must be a positive finite number or None")

    length = len(records)
    resolved = {
        spec.name: signal
        for spec in SIGNAL_SPECS
        if (signal := _resolve_signal(records, spec)) is not None
    }
    component_payloads = {
        name: _component_for_signal(
            name,
            signal,
            window=window,
            epsilon=epsilon,
            max_z=max_z,
        )
        for name, signal in resolved.items()
    }

    def method_payload(signal_names: Sequence[str]) -> dict[str, Any]:
        used = [name for name in signal_names if name in component_payloads]
        sources = [
            {
                "signal": name,
                "field": resolved[name]["field"],
                "direction": resolved[name]["direction"],
                "finite_frames": resolved[name]["finite_frames"],
                "transformation": component_payloads[name][1],
            }
            for name in used
        ]
        return {
            "scores": _aggregate_components(
                [component_payloads[name][0] for name in used],
                length,
            ),
            "used_signals": used,
            "signal_sources": sources,
        }

    rng = np.random.default_rng(seed)
    random_payload = {
        "scores": rng.random(length, dtype=np.float64),
        "used_signals": ["seeded_random"],
        "signal_sources": [
            {
                "signal": "seeded_random",
                "field": None,
                "direction": "random",
                "finite_frames": length,
                "transformation": "seeded_uniform_0_1",
            }
        ],
    }

    return {
        "random": random_payload,
        "update_magnitude_only": method_payload(("update_magnitude",)),
        "reliability_only": method_payload(("reliability",)),
        "combined": method_payload(tuple(spec.name for spec in SIGNAL_SPECS)),
    }


def _binary_labels(labels: ArrayLike) -> IntArray:
    array = np.asarray(labels)
    if array.ndim != 1:
        raise ValueError(f"expected one-dimensional labels, got shape {array.shape}")
    if not np.all(np.isin(array, (0, 1, False, True))):
        raise ValueError("labels must contain only 0 and 1")
    return array.astype(np.int64, copy=False)


def auroc(labels: ArrayLike, scores: ArrayLike) -> float | None:
    """Compute tie-aware AUROC using the Mann-Whitney rank statistic."""

    truth = _binary_labels(labels)
    risk = _as_float_array(scores)
    if truth.shape != risk.shape:
        raise ValueError("labels and scores must have the same length")

    valid = np.isfinite(risk)
    truth = truth[valid]
    risk = risk[valid]
    positive_count = int(truth.sum())
    negative_count = int(truth.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None

    order = np.argsort(risk, kind="mergesort")
    sorted_scores = risk[order]
    ranks = np.empty(risk.size, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        stop = start + 1
        while stop < sorted_scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        average_rank = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average_rank
        start = stop

    positive_rank_sum = float(ranks[truth == 1].sum())
    statistic = positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    return float(statistic / (positive_count * negative_count))


def threshold_metrics(
    labels: ArrayLike,
    scores: ArrayLike,
    threshold: float,
) -> dict[str, Any]:
    """Return confusion counts, F1, and false-positive rate."""

    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")

    truth = _binary_labels(labels)
    risk = _as_float_array(scores)
    if truth.shape != risk.shape:
        raise ValueError("labels and scores must have the same length")

    valid = np.isfinite(risk)
    evaluated_truth = truth[valid]
    predicted = risk[valid] >= threshold
    positive = evaluated_truth == 1
    negative = ~positive

    true_positive = int(np.sum(predicted & positive))
    false_positive = int(np.sum(predicted & negative))
    true_negative = int(np.sum(~predicted & negative))
    false_negative = int(np.sum(~predicted & positive))
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())

    precision_denominator = true_positive + false_positive
    precision = (
        float(true_positive / precision_denominator)
        if precision_denominator
        else None
    )
    recall = float(true_positive / positive_count) if positive_count else None
    if positive_count == 0:
        f1 = None
    else:
        f1_denominator = 2 * true_positive + false_positive + false_negative
        f1 = float(2 * true_positive / f1_denominator) if f1_denominator else 0.0
    false_positive_rate = (
        float(false_positive / negative_count) if negative_count else None
    )

    return {
        "threshold": float(threshold),
        "evaluated_frames": int(valid.sum()),
        "ignored_nonfinite_scores": int((~valid).sum()),
        "positive_frames": positive_count,
        "negative_frames": negative_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
    }


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _positive_runs(labels: IntArray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, label in enumerate(labels):
        if label and start is None:
            start = index
        if start is not None and (not label or index == labels.size - 1):
            end = index if label and index == labels.size - 1 else index - 1
            runs.append((start, end))
            start = None
    return runs


def detection_delay(
    labels: ArrayLike,
    scores: ArrayLike,
    threshold: float,
    *,
    frame_ids: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Measure first threshold crossing inside each positive interval.

    Delay is reported in observed frame positions, so non-contiguous numeric
    frame IDs do not silently turn a sampled sequence into a longer delay.
    """

    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")

    truth = _binary_labels(labels)
    risk = _as_float_array(scores)
    if truth.shape != risk.shape:
        raise ValueError("labels and scores must have the same length")
    if frame_ids is None:
        ids: list[Any] = list(range(truth.size))
    else:
        ids = list(frame_ids)
        if len(ids) != truth.size:
            raise ValueError("frame_ids and labels must have the same length")

    interval_results: list[dict[str, Any]] = []
    delays: list[int] = []
    for start, end in _positive_runs(truth):
        candidates = np.flatnonzero(
            np.isfinite(risk[start : end + 1])
            & (risk[start : end + 1] >= threshold)
        )
        if candidates.size:
            detected_position = start + int(candidates[0])
            delay = detected_position - start
            delays.append(delay)
            detected_frame_id = _json_scalar(ids[detected_position])
        else:
            detected_position = None
            delay = None
            detected_frame_id = None

        interval_results.append(
            {
                "start_position": start,
                "end_position": end,
                "start_frame_id": _json_scalar(ids[start]),
                "end_frame_id": _json_scalar(ids[end]),
                "detected_position": detected_position,
                "detected_frame_id": detected_frame_id,
                "delay_frames": delay,
            }
        )

    interval_count = len(interval_results)
    detected_count = len(delays)
    return {
        "interval_count": interval_count,
        "detected_intervals": detected_count,
        "missed_intervals": interval_count - detected_count,
        "mean_delay_frames": float(np.mean(delays)) if delays else None,
        "median_delay_frames": float(np.median(delays)) if delays else None,
        "intervals": interval_results,
    }


def evaluate_scores(
    labels: ArrayLike,
    scores: ArrayLike,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    frame_ids: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one risk series with all required detection metrics."""

    result = threshold_metrics(labels, scores, threshold)
    result["auroc"] = auroc(labels, scores)
    result["detection_delay"] = detection_delay(
        labels,
        scores,
        threshold,
        frame_ids=frame_ids,
    )
    return result


def _extract_corruption_intervals(
    corruption: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if isinstance(corruption, Mapping):
        if "corruptions" in corruption:
            raw = corruption["corruptions"]
        elif "intervals" in corruption:
            raw = corruption["intervals"]
        elif any(
            key in corruption
            for key in ("start_frame", "start", "start_index", "frame_ids", "frames")
        ):
            raw = [corruption]
        else:
            raw = []
    else:
        raw = corruption

    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("corruption intervals must be a JSON array")
    intervals = list(raw)
    if not all(isinstance(item, Mapping) for item in intervals):
        raise ValueError("every corruption interval must be a JSON object")
    return intervals


def _numeric_value(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _strict_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _mark_frame_range(
    labels: IntArray,
    frame_ids: Sequence[Any],
    start: Any,
    end: Any,
) -> None:
    start = _strict_integer(start, name="start_frame")
    end = _strict_integer(end, name="end_frame")
    numeric_ids = [_numeric_value(frame_id) for frame_id in frame_ids]
    numeric_start = _numeric_value(start)
    numeric_end = _numeric_value(end)
    if (
        numeric_start is not None
        and numeric_end is not None
        and all(value is not None for value in numeric_ids)
    ):
        if numeric_end < numeric_start:
            raise ValueError(f"corruption end {end!r} precedes start {start!r}")
        values = np.asarray(numeric_ids, dtype=np.float64)
        labels[(values >= numeric_start) & (values <= numeric_end)] = 1
        return

    try:
        start_position = list(frame_ids).index(start)
        end_position = list(frame_ids).index(end)
    except ValueError as error:
        raise ValueError(
            "non-numeric corruption boundaries must exactly match frame IDs"
        ) from error
    if end_position < start_position:
        raise ValueError(f"corruption end {end!r} precedes start {start!r}")
    labels[start_position : end_position + 1] = 1


def frame_labels_from_corruption(
    frame_ids: Sequence[Any],
    corruption: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> IntArray:
    """Expand inclusive corruption intervals to one binary label per frame."""

    ids = list(frame_ids)
    labels = np.zeros(len(ids), dtype=np.int64)
    for interval in _extract_corruption_intervals(corruption):
        if interval.get("enabled", True) is False:
            continue

        explicit_frames = interval.get("frame_ids", interval.get("frames"))
        if explicit_frames is not None:
            if not isinstance(explicit_frames, Sequence) or isinstance(
                explicit_frames, (str, bytes)
            ):
                raise ValueError("frame_ids/frames must be a JSON array")
            explicit = set(explicit_frames)
            for index, frame_id in enumerate(ids):
                if frame_id in explicit:
                    labels[index] = 1
            continue

        if any(key in interval for key in ("start_index", "end_index", "start", "end")):
            if "start_index" in interval or "end_index" in interval:
                if "start_index" not in interval or "end_index" not in interval:
                    raise ValueError(
                        "start_index and end_index must be provided together"
                    )
                start_index = _strict_integer(
                    interval["start_index"], name="start_index"
                )
                end_index = _strict_integer(interval["end_index"], name="end_index")
            else:
                if "start" not in interval or "end" not in interval:
                    raise ValueError("start and end must be provided together")
                # StateGuard3R corruption manifests define start/end as
                # zero-based inclusive positions in the derived stream.
                start_index = _strict_integer(interval["start"], name="start")
                end_index = _strict_integer(interval["end"], name="end")
            if start_index < 0 or end_index >= len(ids) or end_index < start_index:
                raise ValueError(
                    f"invalid inclusive index interval [{start_index}, {end_index}]"
                )
            labels[start_index : end_index + 1] = 1
            continue

        start = interval.get("start_frame")
        end = interval.get("end_frame")
        if start is None or end is None:
            raise ValueError(
                "each interval needs start_frame/end_frame, start/end, "
                "start_index/end_index, or frame_ids"
            )
        _mark_frame_range(labels, ids, start, end)

    return labels


def _labels_by_corruption_type(
    frame_ids: Sequence[Any],
    corruption: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, IntArray]:
    grouped: dict[str, IntArray] = {}
    for interval in _extract_corruption_intervals(corruption):
        if interval.get("enabled", True) is False:
            continue
        corruption_type = interval.get("type", "unspecified")
        if not isinstance(corruption_type, str) or not corruption_type.strip():
            raise ValueError("corruption type must be a non-empty string")
        labels = frame_labels_from_corruption(frame_ids, [interval])
        if corruption_type in grouped:
            grouped[corruption_type] = np.maximum(
                grouped[corruption_type], labels
            ).astype(np.int64, copy=False)
        else:
            grouped[corruption_type] = labels
    return grouped


# A descriptive alias for callers that work directly with interval manifests.
labels_from_corruption_intervals = frame_labels_from_corruption


def _serializable_scores(scores: FloatArray) -> list[float | None]:
    return [float(value) if math.isfinite(float(value)) else None for value in scores]


def _validated_threshold(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _resolve_method_thresholds(
    threshold: float,
    thresholds: Mapping[str, float] | None,
) -> dict[str, float]:
    fallback = _validated_threshold(threshold, name="threshold")
    resolved = {name: fallback for name in METHOD_NAMES}
    if thresholds is None:
        return resolved
    if not isinstance(thresholds, Mapping):
        raise ValueError("thresholds must map method names to finite numbers")

    unknown = sorted(
        (name for name in thresholds if name not in METHOD_NAMES),
        key=repr,
    )
    if unknown:
        formatted = ", ".join(repr(name) for name in unknown)
        raise ValueError(f"unknown detection method threshold(s): {formatted}")
    for name, value in thresholds.items():
        resolved[name] = _validated_threshold(
            value,
            name=f"thresholds[{name!r}]",
        )
    return resolved


def evaluate_detection(
    records: Sequence[Mapping[str, Any]],
    corruption: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
    thresholds: Mapping[str, float] | None = None,
    seed: int = DEFAULT_SEED,
    epsilon: float = DEFAULT_EPSILON,
    max_z: float | None = DEFAULT_MAX_Z,
) -> dict[str, Any]:
    """Evaluate all four baselines and return a JSON-serializable result."""

    method_thresholds = _resolve_method_thresholds(threshold, thresholds)
    frame_ids = [record.get("frame_id", index) for index, record in enumerate(records)]
    labels = frame_labels_from_corruption(frame_ids, corruption)
    labels_by_type = _labels_by_corruption_type(frame_ids, corruption)
    baselines = compute_baseline_scores(
        records,
        window=window,
        seed=seed,
        epsilon=epsilon,
        max_z=max_z,
    )

    methods: dict[str, Any] = {}
    for name, payload in baselines.items():
        scores = payload["scores"]
        methods[name] = {
            "used_signals": payload["used_signals"],
            "signal_sources": payload["signal_sources"],
            "scores": _serializable_scores(scores),
            "metrics": evaluate_scores(
                labels,
                scores,
                threshold=method_thresholds[name],
                frame_ids=frame_ids,
            ),
        }

    by_corruption_type: dict[str, Any] = {}
    for corruption_type, type_labels in labels_by_type.items():
        by_corruption_type[corruption_type] = {
            "labels": type_labels.tolist(),
            "positive_frames": int(type_labels.sum()),
            "negative_frames": int(type_labels.size - type_labels.sum()),
            "methods": {
                name: evaluate_scores(
                    type_labels,
                    payload["scores"],
                    threshold=method_thresholds[name],
                    frame_ids=frame_ids,
                )
                for name, payload in baselines.items()
            },
        }

    return {
        "schema_version": "detection-only-v0",
        "config": {
            "window": int(window),
            "threshold": float(threshold),
            "thresholds": method_thresholds,
            "threshold_policy": "frozen_per_method",
            "seed": int(seed),
            "epsilon": float(epsilon),
            "max_z": None if max_z is None else float(max_z),
            "rolling_reference": "strictly_preceding_frames",
            "robust_z_tail": "absolute_two_sided",
            "robust_z_scale": "MAD+epsilon",
            "component_aggregation": "sum",
            "reliability_risk": "-reliability (uncertainty_u alias uses value-1)",
            "corruption_type_grouping": "one_type_vs_all_other_frames",
            "interval_endpoints": "inclusive",
        },
        "frame_count": len(records),
        "positive_frames": int(labels.sum()),
        "negative_frames": int(labels.size - labels.sum()),
        "frame_ids": [_json_scalar(value) for value in frame_ids],
        "labels": labels.tolist(),
        "methods": methods,
        "by_corruption_type": by_corruption_type,
    }


def _parse_health_jsonl_lines(
    lines: Iterable[str], input_path: Path
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{input_path}:{line_number}: invalid JSON: {error.msg}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"{input_path}:{line_number}: expected a JSON object")
        records.append(value)
    if not records:
        raise ValueError(f"{input_path}: health ledger contains no records")
    return records


def load_health_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a health ledger, rejecting malformed or non-object JSON lines."""

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        return _parse_health_jsonl_lines(handle, input_path)


def load_corruption_json(path: str | Path) -> Any:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_method_threshold(value: str) -> tuple[str, float]:
    method, separator, raw_threshold = value.partition("=")
    if not separator or not method or not raw_threshold:
        raise argparse.ArgumentTypeError(
            "method threshold must use METHOD=VALUE syntax"
        )
    if method not in METHOD_NAMES:
        choices = ", ".join(METHOD_NAMES)
        raise argparse.ArgumentTypeError(
            f"unknown detection method {method!r}; choose one of: {choices}"
        )
    try:
        threshold = _validated_threshold(
            raw_threshold,
            name=f"threshold for {method}",
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return method, threshold


def _validated_frozen_detection_config(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if metadata.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError(
            "development calibration config schema_version must be "
            f"{CALIBRATION_SCHEMA_VERSION!r}"
        )
    if metadata.get("dataset_split") != "development":
        raise ValueError(
            "development calibration config dataset_split must be 'development'"
        )

    development_run_id = metadata.get("development_run_id")
    if not isinstance(development_run_id, str) or not development_run_id.strip():
        raise ValueError(
            "development calibration config must contain a non-empty "
            "development_run_id string"
        )

    frozen = metadata.get("frozen_detection_config")
    if not isinstance(frozen, Mapping):
        raise ValueError(
            "development calibration config must contain a "
            "frozen_detection_config JSON object"
        )
    required_fields = ("window", "seed", "epsilon", "max_z", "thresholds")
    missing_fields = [name for name in required_fields if name not in frozen]
    if missing_fields:
        raise ValueError(
            "frozen_detection_config is missing required field(s): "
            + ", ".join(missing_fields)
        )

    window = _strict_integer(frozen["window"], name="frozen_detection_config.window")
    if window < 1:
        raise ValueError("frozen_detection_config.window must be at least 1")
    seed = _strict_integer(frozen["seed"], name="frozen_detection_config.seed")
    epsilon = _validated_threshold(
        frozen["epsilon"], name="frozen_detection_config.epsilon"
    )
    if epsilon <= 0:
        raise ValueError(
            "frozen_detection_config.epsilon must be a positive finite number"
        )

    raw_max_z = frozen["max_z"]
    if raw_max_z is None:
        max_z = None
    else:
        max_z = _validated_threshold(
            raw_max_z, name="frozen_detection_config.max_z"
        )
        if max_z <= 0:
            raise ValueError(
                "frozen_detection_config.max_z must be null or a positive "
                "finite number"
            )

    raw_thresholds = frozen["thresholds"]
    if not isinstance(raw_thresholds, Mapping):
        raise ValueError("frozen_detection_config.thresholds must be a JSON object")
    missing_methods = [name for name in METHOD_NAMES if name not in raw_thresholds]
    unknown_methods = [name for name in raw_thresholds if name not in METHOD_NAMES]
    if missing_methods or unknown_methods:
        problems: list[str] = []
        if missing_methods:
            problems.append("missing method(s): " + ", ".join(missing_methods))
        if unknown_methods:
            problems.append("unknown method(s): " + ", ".join(unknown_methods))
        raise ValueError(
            "frozen_detection_config.thresholds must contain exactly the four "
            "detection methods; " + "; ".join(problems)
        )
    thresholds = {
        name: _validated_threshold(
            raw_thresholds[name],
            name=f"frozen_detection_config.thresholds[{name!r}]",
        )
        for name in METHOD_NAMES
    }
    return {
        "window": window,
        "seed": seed,
        "epsilon": epsilon,
        "max_z": max_z,
        "thresholds": thresholds,
    }


def _load_development_calibration_provenance(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load, validate, and fingerprint a formal development configuration."""

    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"cannot read development calibration config {path}: {error}"
        ) from error

    def reject_nonfinite_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} is not allowed")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r} is not allowed")
            result[key] = value
        return result

    try:
        metadata = json.loads(
            raw,
            parse_constant=reject_nonfinite_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"development calibration config {path} is not valid JSON: {error}"
        ) from error
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError(
            "development calibration config must be a non-empty JSON object"
        )
    frozen = _validated_frozen_detection_config(metadata)

    return (
        {
            "config_path": str(path.resolve()),
            "config_sha256": hashlib.sha256(raw).hexdigest(),
            "config": metadata,
        },
        frozen,
    )


def _verify_formal_config_matches_frozen(
    frozen: Mapping[str, Any],
    *,
    window: int,
    seed: int,
    epsilon: float,
    max_z: float | None,
    thresholds: Mapping[str, float],
) -> None:
    runtime_values = {
        "window": window,
        "seed": seed,
        "epsilon": epsilon,
        "max_z": max_z,
    }
    for name, runtime_value in runtime_values.items():
        frozen_value = frozen[name]
        if runtime_value != frozen_value:
            raise ValueError(
                f"formal CLI {name}={runtime_value!r} does not match "
                f"frozen_detection_config.{name}={frozen_value!r}"
            )
    for method in METHOD_NAMES:
        runtime_threshold = thresholds[method]
        frozen_threshold = frozen["thresholds"][method]
        if runtime_threshold != frozen_threshold:
            raise ValueError(
                f"formal CLI threshold for {method}={runtime_threshold!r} does "
                "not match frozen_detection_config.thresholds"
                f"[{method!r}]={frozen_threshold!r}"
            )


def _read_cli_input_snapshot(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    return (
        raw,
        {
            "absolute_path": str(path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        },
    )


def _parse_health_jsonl_snapshot(raw: bytes, path: Path) -> list[dict[str, Any]]:
    text = raw.decode("utf-8")
    return _parse_health_jsonl_lines(text.splitlines(keepends=True), path)


def _parse_corruption_json_snapshot(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8"))


def _write_metrics_json_atomic(path: str | Path, result: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                result,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate seeded-random, update-only, reliability-only, and "
            "combined causal health scores."
        )
    )
    parser.add_argument("health_jsonl", type=Path, help="per-frame health JSONL")
    parser.add_argument(
        "corruption_json",
        type=Path,
        help="corruption manifest with inclusive frame intervals",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="metrics JSON output path",
    )
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--method-threshold",
        action="append",
        type=_parse_method_threshold,
        default=[],
        metavar="METHOD=VALUE",
        help=(
            "override and freeze one method's threshold; may be repeated for "
            "different methods"
        ),
    )
    parser.add_argument(
        "--formal",
        action="store_true",
        help=(
            "require one explicit threshold for every method and bind the run "
            "to a development calibration config"
        ),
    )
    parser.add_argument(
        "--dev-calibration-config",
        type=Path,
        help=(
            "existing non-empty JSON object documenting development-set "
            "calibration; required with --formal"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument(
        "--max-z",
        type=float,
        default=DEFAULT_MAX_Z,
        help="optional positive robust-z cap; omitted by default",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    method_thresholds: dict[str, float] = {}
    duplicate_methods: list[str] = []
    for method, threshold in args.method_threshold:
        if method in method_thresholds and method not in duplicate_methods:
            duplicate_methods.append(method)
        method_thresholds[method] = threshold

    development_calibration: dict[str, Any] | None = None
    if args.formal:
        if duplicate_methods:
            parser.error(
                "--formal rejects duplicate --method-threshold method(s): "
                + ", ".join(duplicate_methods)
            )
        missing_methods = [
            method for method in METHOD_NAMES if method not in method_thresholds
        ]
        if missing_methods:
            parser.error(
                "--formal requires an explicit --method-threshold for every "
                "method; missing: " + ", ".join(missing_methods)
            )
        if args.dev_calibration_config is None:
            parser.error("--formal requires --dev-calibration-config")
        try:
            (
                development_calibration,
                frozen_detection_config,
            ) = _load_development_calibration_provenance(args.dev_calibration_config)
            _verify_formal_config_matches_frozen(
                frozen_detection_config,
                window=args.window,
                seed=args.seed,
                epsilon=args.epsilon,
                max_z=args.max_z,
                thresholds=method_thresholds,
            )
        except ValueError as error:
            parser.error(str(error))
    elif args.dev_calibration_config is not None:
        parser.error("--dev-calibration-config requires --formal")

    health_snapshot, health_provenance = _read_cli_input_snapshot(args.health_jsonl)
    records = _parse_health_jsonl_snapshot(health_snapshot, args.health_jsonl)
    corruption_snapshot, corruption_provenance = _read_cli_input_snapshot(
        args.corruption_json
    )
    corruption = _parse_corruption_json_snapshot(corruption_snapshot)
    result = evaluate_detection(
        records,
        corruption,
        window=args.window,
        threshold=args.threshold,
        thresholds=method_thresholds,
        seed=args.seed,
        epsilon=args.epsilon,
        max_z=args.max_z,
    )

    explicit_methods = [
        method for method in METHOD_NAMES if method in method_thresholds
    ]
    result["config"].update(
        {
            "execution_mode": "formal" if args.formal else "exploratory",
            "formal": bool(args.formal),
            "frozen_from_development": bool(args.formal),
            "frozen_config_match_verified": bool(args.formal),
            "explicit_method_thresholds": explicit_methods,
            "fallback_threshold_methods": [
                method for method in METHOD_NAMES if method not in method_thresholds
            ],
            "development_calibration_provenance": development_calibration,
        }
    )
    if args.formal:
        result["config"]["threshold_policy"] = "formal_frozen_per_method"
    result["input_provenance"] = {
        "snapshot_policy": "single_read_bytes_used_for_parse_and_sha256",
        "health_jsonl": health_provenance,
        "corruption_json": corruption_provenance,
    }

    _write_metrics_json_atomic(args.output, result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
