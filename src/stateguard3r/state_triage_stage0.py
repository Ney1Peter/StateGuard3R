"""Frozen, non-learning decisions and metrics for StateTriage3R Stage 0.

This module is deliberately post-hoc evaluation code.  It only accepts rows
already emitted by :mod:`state_triage_v2`; it never receives an image, model,
ground truth, source index, or future row when making the causal T0 decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


CAUSES = (
    "registration_or_order_fault",
    "bad_observation",
    "transient_local_content",
    "normal_novelty",
)
UNRESOLVED = "unresolved"
METHODS = ("S0", "S1", "T0", "T0+K")


class Stage0DecisionError(ValueError):
    """Raised when a frozen decision contract or evidence row is malformed."""


@dataclass(frozen=True)
class DecisionThresholds:
    """All constants must be written into the Gate-B protocol before events run."""

    coverage_novelty_max: float
    scalar_s0_native_geometric_residual_z_min: float
    scalar_s1_normalized_residual_median_min: float
    quality_sharpness_max: float
    quality_low_light_fraction_min: float
    quality_clipped_fraction_min: float
    quality_confidence_disagreement_min: float
    registration_pose_jump_z_min: float
    registration_geometric_residual_z_min: float
    registration_globality_min: float
    transient_residual_high_fraction_min: float
    transient_locality_min: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise Stage0DecisionError(f"{name} must be finite")
        for name in (
            "coverage_novelty_max",
            "quality_low_light_fraction_min",
            "quality_clipped_fraction_min",
            "registration_globality_min",
            "transient_residual_high_fraction_min",
            "transient_locality_min",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise Stage0DecisionError(f"{name} must be in [0, 1]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DecisionThresholds":
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise Stage0DecisionError("threshold keys differ from frozen schema")
        return cls(**{name: float(value[name]) for name in expected})

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}


@dataclass(frozen=True)
class FrameDecision:
    label: str
    posterior: Mapping[str, float]
    reason: str

    def __post_init__(self) -> None:
        if self.label not in (*CAUSES, UNRESOLVED):
            raise Stage0DecisionError("decision label is unsupported")
        if set(self.posterior) != set(CAUSES):
            raise Stage0DecisionError("posterior must contain exactly the four causes")
        values = [float(self.posterior[cause]) for cause in CAUSES]
        if any(not math.isfinite(value) or value < 0.0 for value in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise Stage0DecisionError("posterior must be finite, non-negative, and sum to one")

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "posterior": dict(self.posterior), "reason": self.reason}


def _finite(row: Mapping[str, Any], name: str, *, allow_none: bool = False) -> float | None:
    if name not in row:
        raise Stage0DecisionError(f"evidence row lacks {name}")
    value = row[name]
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise Stage0DecisionError(f"evidence row {name} must be finite")
    return float(value)


def _posterior(label: str) -> dict[str, float]:
    # Rules do not learn calibrated probability.  This fixed, explicitly
    # non-learned distribution makes NLL/ECE diagnostic values reproducible.
    if label == UNRESOLVED:
        return {cause: 0.25 for cause in CAUSES}
    return {cause: 0.70 if cause == label else 0.10 for cause in CAUSES}


def _decision(label: str, reason: str) -> FrameDecision:
    return FrameDecision(label=label, posterior=_posterior(label), reason=reason)


def decide_frame(method: str, row: Mapping[str, Any], thresholds: DecisionThresholds) -> FrameDecision:
    """Apply one fixed causal branch to one current evidence row.

    ``T0+K`` intentionally has the same per-frame branch as T0.  Its only
    future use is the separate offline refinement in :func:`decide_event`.
    """

    if method not in METHODS:
        raise Stage0DecisionError("method is unsupported")
    coverage = _finite(row, "coverage_ratio")
    if coverage < thresholds.coverage_novelty_max:
        return _decision("normal_novelty", "coverage_below_frozen_novelty_threshold")
    if method == "S0":
        scalar = _finite(row, "native_geometric_residual_z", allow_none=True)
        if scalar is not None and scalar >= thresholds.scalar_s0_native_geometric_residual_z_min:
            return _decision("registration_or_order_fault", "native_scalar_above_frozen_threshold")
        return _decision(UNRESOLVED, "native_scalar_below_or_missing")
    if method == "S1":
        scalar = _finite(row, "normalized_residual_median")
        if scalar >= thresholds.scalar_s1_normalized_residual_median_min:
            return _decision("registration_or_order_fault", "residual_scalar_above_frozen_threshold")
        return _decision(UNRESOLVED, "residual_scalar_below_threshold")

    sharpness = _finite(row, "sharpness")
    low_light = _finite(row, "low_light_fraction")
    clipped = _finite(row, "clipped_fraction")
    disagreement = _finite(row, "confidence_disagreement_mean")
    if (
        sharpness <= thresholds.quality_sharpness_max
        or low_light >= thresholds.quality_low_light_fraction_min
        or clipped >= thresholds.quality_clipped_fraction_min
        or disagreement >= thresholds.quality_confidence_disagreement_min
    ):
        return _decision("bad_observation", "frozen_observation_quality_branch")
    pose_z = _finite(row, "pose_jump_z", allow_none=True)
    geometry_z = _finite(row, "native_geometric_residual_z", allow_none=True)
    globality = _finite(row, "residual_globality")
    if (
        globality >= thresholds.registration_globality_min
        and (
            (pose_z is not None and pose_z >= thresholds.registration_pose_jump_z_min)
            or (geometry_z is not None and geometry_z >= thresholds.registration_geometric_residual_z_min)
        )
    ):
        return _decision("registration_or_order_fault", "frozen_global_registration_branch")
    high_fraction = _finite(row, "residual_high_fraction")
    locality = _finite(row, "residual_locality")
    if high_fraction >= thresholds.transient_residual_high_fraction_min and locality >= thresholds.transient_locality_min:
        return _decision("transient_local_content", "frozen_local_transient_branch")
    return _decision(UNRESOLVED, "no_frozen_typed_branch")


def decide_event(
    method: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    start_frame: int,
    end_frame: int,
    thresholds: DecisionThresholds,
    future_k: int,
) -> dict[str, Any]:
    """Aggregate causal frame decisions, optionally exposing an offline upper bound."""

    if start_frame < 0 or end_frame < start_frame or end_frame >= len(rows) or future_k < 0:
        raise Stage0DecisionError("event frame range or future K is invalid")
    causal_method = "T0" if method == "T0+K" else method
    selected: FrameDecision | None = None
    selected_frame: int | None = None
    for frame_id in range(start_frame, end_frame + 1):
        current = decide_frame(causal_method, rows[frame_id], thresholds)
        if current.label != UNRESOLVED:
            selected, selected_frame = current, frame_id
            break
    if selected is None:
        selected, selected_frame = _decision(UNRESOLVED, "no_causal_decision_in_event_window"), None
    if method == "T0+K" and selected.label == "transient_local_content":
        future = rows[end_frame + 1 : min(len(rows), end_frame + 1 + future_k)]
        if len(future) < future_k:
            selected = _decision(UNRESOLVED, "offline_future_window_incomplete")
        elif any(
            _finite(row, "residual_high_fraction") >= thresholds.transient_residual_high_fraction_min
            for row in future
        ):
            selected = _decision(UNRESOLVED, "offline_future_residual_persists")
        else:
            selected = _decision("transient_local_content", "offline_future_window_clears")
    return {
        "label": selected.label,
        "posterior": dict(selected.posterior),
        "reason": selected.reason,
        "decision_frame": selected_frame,
        "detection_delay_frames": None if selected_frame is None else selected_frame - start_frame,
    }


def cause_metrics(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute fixed-class metrics; unresolved remains visible rather than correct."""

    if not events:
        raise Stage0DecisionError("metrics require at least one event")
    matrix = {truth: {prediction: 0 for prediction in (*CAUSES, UNRESOLVED)} for truth in CAUSES}
    for event in events:
        truth, prediction = event.get("truth"), event.get("prediction")
        if truth not in CAUSES or prediction not in (*CAUSES, UNRESOLVED):
            raise Stage0DecisionError("event label is unsupported")
        matrix[truth][prediction] += 1
    per_cause: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for cause in CAUSES:
        tp = matrix[cause][cause]
        fp = sum(matrix[other][cause] for other in CAUSES if other != cause)
        fn = sum(matrix[cause][other] for other in (*CAUSES, UNRESOLVED) if other != cause)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_cause[cause] = {"precision": precision, "recall": recall, "f1": f1, "support": float(sum(matrix[cause].values()))}
        f1_values.append(f1)
    probabilities = [float(event["posterior"][event["truth"]]) for event in events]
    nll = float(-np.mean(np.log(np.maximum(np.asarray(probabilities, dtype=np.float64), 1e-12))))
    confidences = np.asarray([max(float(value) for value in event["posterior"].values()) for event in events], dtype=np.float64)
    correct = np.asarray([event["truth"] == event["prediction"] for event in events], dtype=np.float64)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidences >= lower) & (confidences < upper if upper < 1.0 else confidences <= upper)
        if mask.any():
            ece += float(mask.mean() * abs(confidences[mask].mean() - correct[mask].mean()))
    normal = matrix["normal_novelty"]
    unsafe = {
        "registration_as_transient": matrix["registration_or_order_fault"]["transient_local_content"] / max(1, sum(matrix["registration_or_order_fault"].values())),
        "bad_observation_as_novelty": matrix["bad_observation"]["normal_novelty"] / max(1, sum(matrix["bad_observation"].values())),
        "transient_as_registration": matrix["transient_local_content"]["registration_or_order_fault"] / max(1, sum(matrix["transient_local_content"].values())),
        "normal_novelty_false_reject": sum(normal[cause] for cause in CAUSES if cause != "normal_novelty") / max(1, sum(normal.values())),
    }
    delays = [event["detection_delay_frames"] for event in events if event.get("detection_delay_frames") is not None]
    return {
        "macro_f1": float(np.mean(f1_values)),
        "per_cause": per_cause,
        "confusion_matrix": matrix,
        "nll": nll,
        "ece": ece,
        "unsafe_confusions": unsafe,
        "unresolved_rate": sum(event["prediction"] == UNRESOLVED for event in events) / len(events),
        "event_detection_delay_frames": {"median": float(np.median(delays)) if delays else None, "max": max(delays) if delays else None},
    }


def sequence_bootstrap_delta(
    typed_events: Sequence[Mapping[str, Any]],
    scalar_events: Sequence[Mapping[str, Any]],
    *,
    samples: int = 10_000,
    seed: int = 20260823,
) -> dict[str, float]:
    """Sequence-resampled Macro-F1 difference, with no frame-level leakage."""

    if samples < 1:
        raise Stage0DecisionError("bootstrap sample count must be positive")
    typed_by_id = {str(event["event_id"]): event for event in typed_events}
    scalar_by_id = {str(event["event_id"]): event for event in scalar_events}
    if set(typed_by_id) != set(scalar_by_id):
        raise Stage0DecisionError("bootstrap methods have different event IDs")
    sequences = sorted({str(event["source_sequence"]) for event in typed_events})
    if len(sequences) < 2:
        raise Stage0DecisionError("sequence bootstrap requires at least two sequences")
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for draw in rng.integers(0, len(sequences), size=(samples, len(sequences))):
        chosen = [sequences[int(index)] for index in draw]
        typed_sample = [event for sequence in chosen for event in typed_events if str(event["source_sequence"]) == sequence]
        scalar_sample = [scalar_by_id[str(event["event_id"])] for event in typed_sample]
        deltas.append(cause_metrics(typed_sample)["macro_f1"] - cause_metrics(scalar_sample)["macro_f1"])
    values = np.asarray(deltas, dtype=np.float64)
    return {"samples": float(samples), "seed": float(seed), "mean_delta": float(values.mean()), "lower_95": float(np.quantile(values, 0.025)), "upper_95": float(np.quantile(values, 0.975))}


__all__ = [
    "CAUSES",
    "METHODS",
    "UNRESOLVED",
    "DecisionThresholds",
    "FrameDecision",
    "Stage0DecisionError",
    "cause_metrics",
    "decide_event",
    "decide_frame",
    "sequence_bootstrap_delta",
]
