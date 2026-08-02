"""Causal, direction-aware Detector v2 scoring.

This module is intentionally separate from :mod:`stateguard3r.detection`.
Formal-v1 continues to use its original absolute-z/raw-reliability/sum score;
v2 instead uses development-derived scale floors, explicit warm-up, one-sided
anomaly directions, and finite-component mean aggregation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import detection


SCHEMA_VERSION = "stateguard3r.detection-v2.v1"
SCALE_FLOOR_QUANTILE = 0.10
MINIMUM_SCALE_FLOOR = 1e-6
CANONICAL_SIGNALS = tuple(spec.name for spec in detection.SIGNAL_SPECS)
METHOD_NAMES = detection.METHOD_NAMES

FloatArray = NDArray[np.float64]


class DetectorV2Error(ValueError):
    """Raised when v2 scoring, scale estimation, or calibration is invalid."""


@dataclass(frozen=True)
class V2ScoringConfig:
    """One predeclared causal Detector-v2 candidate configuration."""

    window: int = 5
    min_history: int = 3
    max_z: float = 5.0
    master_seed: int = 0

    def __post_init__(self) -> None:
        for name in ("window", "min_history"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise DetectorV2Error(f"{name} must be a positive integer")
        if self.min_history > self.window:
            raise DetectorV2Error("min_history may not exceed window")
        if not math.isfinite(self.max_z) or self.max_z <= 0:
            raise DetectorV2Error("max_z must be positive and finite")
        if isinstance(self.master_seed, bool) or not isinstance(self.master_seed, int) or self.master_seed < 0:
            raise DetectorV2Error("master_seed must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def candidate_grid() -> list[V2ScoringConfig]:
    """Return the eight protocol-limited v2 candidates in deterministic order."""

    return [
        V2ScoringConfig(window=window, min_history=min_history, max_z=max_z)
        for window in (5, 10)
        for min_history in (3, 5)
        for max_z in (5.0, 10.0)
    ]


def _coerce_numeric(value: Any) -> float:
    if value is None or isinstance(value, (dict, list, tuple)):
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _field_series(records: Sequence[Mapping[str, Any]], field: str) -> FloatArray:
    return np.asarray([_coerce_numeric(record.get(field)) for record in records], dtype=np.float64)


def _resolve_signal(records: Sequence[Mapping[str, Any]], spec: detection.SignalSpec) -> dict[str, Any] | None:
    for alias in spec.aliases:
        values = _field_series(records, alias.field)
        finite = int(np.isfinite(values).sum())
        if finite:
            return {
                "signal": spec.name,
                "field": alias.field,
                "direction": alias.direction,
                "finite_frames": finite,
                "values": values,
            }
    return None


def _validate_scale_floors(scale_floors: Mapping[str, float]) -> dict[str, float]:
    if set(scale_floors) != set(CANONICAL_SIGNALS):
        missing = sorted(set(CANONICAL_SIGNALS) - set(scale_floors))
        unknown = sorted(set(scale_floors) - set(CANONICAL_SIGNALS))
        raise DetectorV2Error(
            "scale_floors must contain exactly canonical signals"
            + (f"; missing={missing}" if missing else "")
            + (f"; unknown={unknown}" if unknown else "")
        )
    validated: dict[str, float] = {}
    for signal in CANONICAL_SIGNALS:
        value = _coerce_numeric(scale_floors[signal])
        if not math.isfinite(value) or value < MINIMUM_SCALE_FLOOR:
            raise DetectorV2Error(f"scale floor for {signal} must be finite and >= {MINIMUM_SCALE_FLOOR}")
        validated[signal] = value
    return validated


def derive_development_scale_floors(
    development_runs: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    clean_prefix_frames: int = 15,
) -> dict[str, float]:
    """Derive immutable v2 scale floors from disclosed development prefixes only.

    For each resolved canonical signal, this pools strictly positive absolute
    first differences of consecutive finite values from positions before the
    fixed event start.  It returns the 10th percentile or ``1e-6``, whichever
    is larger.  Missing overlap therefore has an explicit numerical fallback
    rather than an implicit epsilon-dependent scale.
    """

    if isinstance(clean_prefix_frames, bool) or not isinstance(clean_prefix_frames, int) or clean_prefix_frames < 2:
        raise DetectorV2Error("clean_prefix_frames must be an integer of at least two")
    if not development_runs:
        raise DetectorV2Error("development_runs must not be empty")
    positive_differences: dict[str, list[float]] = {signal: [] for signal in CANONICAL_SIGNALS}
    for run_id, records in sorted(development_runs.items()):
        if len(records) < clean_prefix_frames:
            raise DetectorV2Error(f"development run {run_id!r} is shorter than clean prefix")
        prefix = records[:clean_prefix_frames]
        for spec in detection.SIGNAL_SPECS:
            resolved = _resolve_signal(prefix, spec)
            if resolved is None:
                continue
            values = np.asarray(resolved["values"], dtype=np.float64)
            for previous, current in zip(values[:-1], values[1:], strict=True):
                if math.isfinite(float(previous)) and math.isfinite(float(current)):
                    difference = abs(float(current - previous))
                    if difference > 0.0 and math.isfinite(difference):
                        positive_differences[spec.name].append(difference)
    floors: dict[str, float] = {}
    for signal in CANONICAL_SIGNALS:
        values = positive_differences[signal]
        quantile = float(np.quantile(np.asarray(values, dtype=np.float64), SCALE_FLOOR_QUANTILE)) if values else MINIMUM_SCALE_FLOOR
        floors[signal] = max(quantile, MINIMUM_SCALE_FLOOR)
    return floors


def directional_causal_component(
    values: ArrayLike,
    *,
    direction: str,
    window: int,
    min_history: int,
    scale_floor: float,
    max_z: float,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Return a one-sided causal robust component and its warm-up mask.

    The reference at frame ``t`` sees only finite values earlier than ``t``.
    A finite frame without ``min_history`` prior finite observations emits the
    neutral score zero and is retained in the metric denominator.  A missing
    current value remains missing so finite-component aggregation can record
    exactly which signals were used.
    """

    if direction not in {"high", "low"}:
        raise DetectorV2Error("direction must be 'high' or 'low'")
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise DetectorV2Error("window must be a positive integer")
    if isinstance(min_history, bool) or not isinstance(min_history, int) or not 1 <= min_history <= window:
        raise DetectorV2Error("min_history must be in [1, window]")
    if not math.isfinite(scale_floor) or scale_floor < MINIMUM_SCALE_FLOOR:
        raise DetectorV2Error(f"scale_floor must be finite and >= {MINIMUM_SCALE_FLOOR}")
    if not math.isfinite(max_z) or max_z <= 0:
        raise DetectorV2Error("max_z must be positive and finite")
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1:
        raise DetectorV2Error("values must be one-dimensional")
    component = np.full(series.shape, np.nan, dtype=np.float64)
    warmup = np.zeros(series.shape, dtype=bool)
    for index, current in enumerate(series):
        if not math.isfinite(float(current)):
            continue
        history = series[max(0, index - window) : index]
        history = history[np.isfinite(history)]
        if history.size < min_history:
            component[index] = 0.0
            warmup[index] = True
            continue
        center = float(np.median(history))
        mad = float(np.median(np.abs(history - center)))
        scale = max(1.4826 * mad, scale_floor)
        raw = (float(current) - center) / scale if direction == "high" else (center - float(current)) / scale
        component[index] = min(max(raw, 0.0), max_z)
    return component, warmup


def _finite_component_mean(components: Sequence[FloatArray], length: int) -> tuple[FloatArray, NDArray[np.int64]]:
    if not components:
        return np.zeros(length, dtype=np.float64), np.zeros(length, dtype=np.int64)
    matrix = np.vstack(components)
    finite = np.isfinite(matrix)
    count = finite.sum(axis=0, dtype=np.int64)
    total = np.where(finite, matrix, 0.0).sum(axis=0)
    score = np.zeros(length, dtype=np.float64)
    nonempty = count > 0
    score[nonempty] = total[nonempty] / count[nonempty]
    return score, count


def validate_online_overlap(records: Sequence[Mapping[str, Any]]) -> None:
    """Enforce formal-v2 overlap availability without reading any GT field."""

    if not records:
        raise DetectorV2Error("records must not be empty")
    for index, record in enumerate(records):
        if record.get("frame_id") != index:
            raise DetectorV2Error("v2 records must have contiguous frame_id values")
        value = record.get("overlap")
        if index == 0:
            if value is not None:
                raise DetectorV2Error("v2 frame zero overlap must be null without a causal predecessor")
            continue
        number = _coerce_numeric(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise DetectorV2Error(f"v2 frame {index} requires finite overlap in [0, 1]")


def compute_v2_scores(
    records: Sequence[Mapping[str, Any]],
    *,
    config: V2ScoringConfig,
    scale_floors: Mapping[str, float],
    seed: int | None = None,
    require_online_overlap: bool = False,
) -> dict[str, Any]:
    """Compute random, update, reliability, and combined v2 score payloads.

    Only finite components participate in a frame's arithmetic mean.  In a
    formal-v2 caller, ``require_online_overlap=True`` rejects missing overlap
    after frame zero before any detector score is emitted.
    """

    if not records:
        raise DetectorV2Error("records must not be empty")
    floors = _validate_scale_floors(scale_floors)
    if require_online_overlap:
        validate_online_overlap(records)
    length = len(records)
    resolved = {
        spec.name: value
        for spec in detection.SIGNAL_SPECS
        if (value := _resolve_signal(records, spec)) is not None
    }
    components: dict[str, FloatArray] = {}
    warmups: dict[str, NDArray[np.bool_]] = {}
    for name, signal in resolved.items():
        component, warmup = directional_causal_component(
            signal["values"],
            direction=str(signal["direction"]),
            window=config.window,
            min_history=config.min_history,
            scale_floor=floors[name],
            max_z=config.max_z,
        )
        components[name] = component
        warmups[name] = warmup

    def payload(signal_names: Sequence[str]) -> dict[str, Any]:
        used = [name for name in signal_names if name in components]
        scores, counts = _finite_component_mean([components[name] for name in used], length)
        return {
            "scores": scores,
            "used_signals": used,
            "signal_sources": [
                {
                    "signal": name,
                    "field": resolved[name]["field"],
                    "direction": resolved[name]["direction"],
                    "finite_frames": resolved[name]["finite_frames"],
                    "scale_floor": floors[name],
                    "warmup_positions": [int(index) for index in np.flatnonzero(warmups[name])],
                    "transformation": "causal_direction_aware_robust_component",
                }
                for name in used
            ],
            "finite_component_count": counts,
        }

    random_seed = config.master_seed if seed is None else seed
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise DetectorV2Error("seed must be a non-negative integer")
    rng = np.random.default_rng(random_seed)
    return {
        "schema_version": SCHEMA_VERSION,
        "config": config.to_dict(),
        "scale_floors": floors,
        "methods": {
            "random": {
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
                "finite_component_count": np.ones(length, dtype=np.int64),
            },
            "update_magnitude_only": payload(("update_magnitude",)),
            "reliability_only": payload(("reliability",)),
            "combined": payload(CANONICAL_SIGNALS),
        },
    }


def _as_binary_labels(values: ArrayLike, *, name: str) -> NDArray[np.int64]:
    labels = np.asarray(values)
    if labels.ndim != 1 or not np.all(np.isin(labels, (0, 1, False, True))):
        raise DetectorV2Error(f"{name} must be one-dimensional binary labels")
    return labels.astype(np.int64, copy=False)


def _longest_primary_false_positive(labels: NDArray[np.int64], scores: FloatArray, mask: NDArray[np.bool_], threshold: float) -> int:
    active = mask & (labels == 0) & (scores >= threshold)
    longest = 0
    current = 0
    for value in active:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def select_constrained_threshold(
    labels_by_run: Mapping[str, ArrayLike],
    scores_by_run: Mapping[str, ArrayLike],
    evaluation_masks_by_run: Mapping[str, ArrayLike],
    *,
    max_pooled_fpr: float = 0.15,
    max_per_run_fp_streak: int = 2,
) -> dict[str, Any]:
    """Choose a development threshold subject to FPR and streak hard limits."""

    run_ids = sorted(labels_by_run)
    if not run_ids or set(run_ids) != set(scores_by_run) or set(run_ids) != set(evaluation_masks_by_run):
        raise DetectorV2Error("labels, scores, and masks must use the same non-empty run IDs")
    if not math.isfinite(max_pooled_fpr) or not 0.0 <= max_pooled_fpr <= 1.0:
        raise DetectorV2Error("max_pooled_fpr must be in [0, 1]")
    if isinstance(max_per_run_fp_streak, bool) or not isinstance(max_per_run_fp_streak, int) or max_per_run_fp_streak < 0:
        raise DetectorV2Error("max_per_run_fp_streak must be a non-negative integer")
    prepared: dict[str, tuple[NDArray[np.int64], FloatArray, NDArray[np.bool_]]] = {}
    values: list[FloatArray] = []
    for run_id in run_ids:
        labels = _as_binary_labels(labels_by_run[run_id], name=f"labels[{run_id}]")
        scores = np.asarray(scores_by_run[run_id], dtype=np.float64)
        mask = np.asarray(evaluation_masks_by_run[run_id], dtype=bool)
        if scores.ndim != 1 or scores.shape != labels.shape or mask.shape != labels.shape or not np.all(np.isfinite(scores)):
            raise DetectorV2Error(f"{run_id} labels/scores/mask are invalid")
        prepared[run_id] = labels, scores, mask
        values.append(scores[mask])
    candidates = np.unique(np.concatenate(values))
    if candidates.size == 0:
        raise DetectorV2Error("development masks select no score values")
    candidate_values = [float(value) for value in candidates] + [float(np.nextafter(candidates[-1], math.inf))]
    evaluations: list[dict[str, Any]] = []
    for threshold in candidate_values:
        per_run: dict[str, Any] = {}
        pooled_labels: list[NDArray[np.int64]] = []
        pooled_scores: list[FloatArray] = []
        delays: list[int] = []
        detected_events = 0
        max_streak = 0
        for run_id, (labels, scores, mask) in prepared.items():
            metrics = detection.threshold_metrics(labels[mask], scores[mask], threshold)
            delay = detection.detection_delay(labels, scores, threshold)
            detected_events += int(delay["detected_intervals"])
            delays.extend(int(row["delay_frames"]) for row in delay["intervals"] if row["delay_frames"] is not None)
            streak = _longest_primary_false_positive(labels, scores, mask, threshold)
            max_streak = max(max_streak, streak)
            per_run[run_id] = {"metrics": metrics, "delay": delay, "max_false_positive_streak": streak}
            pooled_labels.append(labels[mask])
            pooled_scores.append(scores[mask])
        pooled = detection.threshold_metrics(np.concatenate(pooled_labels), np.concatenate(pooled_scores), threshold)
        macro_f1_values = [row["metrics"]["f1"] for row in per_run.values()]
        macro_f1 = float(np.mean([float(value) for value in macro_f1_values if value is not None]))
        fpr = pooled["false_positive_rate"]
        feasible = fpr is not None and float(fpr) <= max_pooled_fpr and max_streak <= max_per_run_fp_streak
        evaluations.append(
            {
                "threshold": threshold,
                "feasible": feasible,
                "macro_f1": macro_f1,
                "pooled": pooled,
                "detected_events": detected_events,
                "mean_delay_frames": float(np.mean(delays)) if delays else None,
                "max_per_run_false_positive_streak": max_streak,
                "per_run": per_run,
            }
        )
    feasible = [row for row in evaluations if row["feasible"]]
    if not feasible:
        raise DetectorV2Error("no development threshold satisfies v2 FPR/streak constraints")

    def quality(row: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
        delay = row["mean_delay_frames"]
        return (
            float(row["macro_f1"]),
            float(row["detected_events"]),
            -float(delay) if delay is not None else -math.inf,
            -float(row["pooled"]["false_positive_rate"]),
            float(row["threshold"]),
        )

    selected = max(feasible, key=quality)
    return {
        "selection_rule": "constrain pooled FPR/streak, then maximize macro-F1, events, lower delay, lower FPR, higher threshold",
        "constraints": {"max_pooled_fpr": max_pooled_fpr, "max_per_run_fp_streak": max_per_run_fp_streak},
        "threshold_candidate_count": len(evaluations),
        "candidates": evaluations,
        "selected": selected,
    }


__all__ = [
    "CANONICAL_SIGNALS",
    "DetectorV2Error",
    "METHOD_NAMES",
    "MINIMUM_SCALE_FLOOR",
    "SCHEMA_VERSION",
    "SCALE_FLOOR_QUANTILE",
    "V2ScoringConfig",
    "candidate_grid",
    "compute_v2_scores",
    "derive_development_scale_floors",
    "directional_causal_component",
    "select_constrained_threshold",
    "validate_online_overlap",
]
