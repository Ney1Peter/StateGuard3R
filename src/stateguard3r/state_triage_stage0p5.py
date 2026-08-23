"""Frozen causal decisions for the independent StateTriage3R Stage 0.5 pilot.

This is deliberately a small, pure post-forward evaluator.  It receives one
observer evidence row at a time and cannot read an image, capsule recipe,
ground-truth label, source index, model state, or a future observer row.  It
is *not* a revision of the frozen Stage 0 decision code.
"""

from __future__ import annotations

from dataclasses import dataclass
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
METHODS = ("S0.5", "T0.5")

# These values and their priority are pre-registered in the Stage 0.5 protocol.
# They are numerical/observer contracts, not values fitted on pilot responses.
DECISION_CONTRACT: dict[str, Any] = {
    "typed_method": "T0.5",
    "scalar_method": "S0.5",
    "typed_order": [
        "sharpness_lte_0.05_bad_observation",
        "coverage_lt_0.25_normal_novelty",
        "clipped_fraction_gte_0.20_transient_local_content",
        "pose_jump_z_or_native_geometric_residual_z_gte_3_registration_or_order_fault",
        "otherwise_unresolved",
    ],
    "scalar_order": [
        "coverage_lt_0.25_normal_novelty",
        "native_geometric_residual_z_gte_3_registration_or_order_fault",
        "otherwise_unresolved",
    ],
    "sharpness_bad_observation_max": 0.05,
    "coverage_novelty_max_exclusive": 0.25,
    "clipped_fraction_transient_min": 0.20,
    "registration_z_min": 3.0,
    "aggregation": "first_non_unresolved_current_row_in_predeclared_event_window",
}


class Stage0p5DecisionError(ValueError):
    """Raised when a fixed Stage 0.5 evidence/decision contract is malformed."""


@dataclass(frozen=True)
class FrameDecision:
    label: str
    posterior: Mapping[str, float]
    reason: str

    def __post_init__(self) -> None:
        if self.label not in (*CAUSES, UNRESOLVED):
            raise Stage0p5DecisionError("unsupported decision label")
        if set(self.posterior) != set(CAUSES):
            raise Stage0p5DecisionError("posterior must contain exactly the four causes")
        values = [float(self.posterior[cause]) for cause in CAUSES]
        if any(not math.isfinite(value) or value < 0.0 for value in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise Stage0p5DecisionError("posterior must be finite, non-negative, and sum to one")

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "posterior": dict(self.posterior), "reason": self.reason}


def _finite(row: Mapping[str, Any], name: str, *, allow_none: bool = False) -> float | None:
    if name not in row:
        raise Stage0p5DecisionError(f"evidence row lacks {name}")
    value = row[name]
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise Stage0p5DecisionError(f"evidence row {name} must be finite")
    return float(value)


def _posterior(label: str) -> dict[str, float]:
    # The pilot does not fit probabilities.  These fixed values only keep the
    # event record complete and intentionally make unresolved visible.
    if label == UNRESOLVED:
        return {cause: 0.25 for cause in CAUSES}
    return {cause: 0.70 if cause == label else 0.10 for cause in CAUSES}


def _decision(label: str, reason: str) -> FrameDecision:
    return FrameDecision(label=label, posterior=_posterior(label), reason=reason)


def decide_frame(method: str, row: Mapping[str, Any]) -> FrameDecision:
    """Apply the pre-registered current-row branch for one pilot method."""

    if method not in METHODS:
        raise Stage0p5DecisionError("unsupported method")
    coverage = _finite(row, "coverage_ratio")
    residual_z = _finite(row, "native_geometric_residual_z", allow_none=True)
    if method == "S0.5":
        if coverage < float(DECISION_CONTRACT["coverage_novelty_max_exclusive"]):
            return _decision("normal_novelty", "coverage_below_predeclared_novelty_threshold")
        if residual_z is not None and residual_z >= float(DECISION_CONTRACT["registration_z_min"]):
            return _decision("registration_or_order_fault", "native_geometric_residual_above_predeclared_threshold")
        return _decision(UNRESOLVED, "no_predeclared_scalar_branch")

    # T0.5's intentional change from Stage 0 is that global quality collapse
    # is evaluated before current geometric coverage can disappear.
    sharpness = _finite(row, "sharpness")
    if sharpness <= float(DECISION_CONTRACT["sharpness_bad_observation_max"]):
        return _decision("bad_observation", "global_information_collapse_before_coverage")
    if coverage < float(DECISION_CONTRACT["coverage_novelty_max_exclusive"]):
        return _decision("normal_novelty", "coverage_below_predeclared_novelty_threshold")
    clipped_fraction = _finite(row, "clipped_fraction")
    if clipped_fraction >= float(DECISION_CONTRACT["clipped_fraction_transient_min"]):
        return _decision("transient_local_content", "localized_clipped_content_after_coverage")
    pose_jump_z = _finite(row, "pose_jump_z", allow_none=True)
    if (pose_jump_z is not None and pose_jump_z >= float(DECISION_CONTRACT["registration_z_min"])) or (
        residual_z is not None and residual_z >= float(DECISION_CONTRACT["registration_z_min"])
    ):
        return _decision("registration_or_order_fault", "temporal_or_geometric_conflict_after_structural_cues")
    return _decision(UNRESOLVED, "no_predeclared_typed_branch")


def decide_event(rows: Sequence[Mapping[str, Any]], *, method: str, start_frame: int, end_frame: int) -> dict[str, Any]:
    """Use only each current event row and select the first non-unresolved call."""

    if start_frame < 0 or end_frame < start_frame or end_frame >= len(rows):
        raise Stage0p5DecisionError("event frame range is invalid")
    for frame_id in range(start_frame, end_frame + 1):
        decision = decide_frame(method, rows[frame_id])
        if decision.label != UNRESOLVED:
            return {
                **decision.to_dict(),
                "decision_frame": frame_id,
                "detection_delay_frames": frame_id - start_frame,
            }
    decision = _decision(UNRESOLVED, "no_causal_decision_in_event_window")
    return {**decision.to_dict(), "decision_frame": None, "detection_delay_frames": None}


def cause_metrics(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute fixed event-level cause metrics; unresolved is never correct."""

    if not events:
        raise Stage0p5DecisionError("metrics require at least one event")
    matrix = {truth: {prediction: 0 for prediction in (*CAUSES, UNRESOLVED)} for truth in CAUSES}
    for event in events:
        truth, prediction = event.get("truth"), event.get("prediction")
        if truth not in CAUSES or prediction not in (*CAUSES, UNRESOLVED):
            raise Stage0p5DecisionError("event label is unsupported")
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
    normal = matrix["normal_novelty"]
    unsafe_counts = {
        "registration_as_transient": matrix["registration_or_order_fault"]["transient_local_content"],
        "bad_observation_as_novelty": matrix["bad_observation"]["normal_novelty"],
        "transient_as_registration": matrix["transient_local_content"]["registration_or_order_fault"],
    }
    delays = [event["detection_delay_frames"] for event in events if event.get("detection_delay_frames") is not None]
    return {
        "macro_f1": float(np.mean(f1_values)),
        "per_cause": per_cause,
        "confusion_matrix": matrix,
        "normal_novelty_false_reject": sum(normal[cause] for cause in CAUSES if cause != "normal_novelty") / max(1, sum(normal.values())),
        "unsafe_confusion_counts": unsafe_counts,
        "unresolved_rate": sum(event["prediction"] == UNRESOLVED for event in events) / len(events),
        "event_detection_delay_frames": {"median": float(np.median(delays)) if delays else None, "max": max(delays) if delays else None},
    }


__all__ = [
    "CAUSES",
    "DECISION_CONTRACT",
    "METHODS",
    "UNRESOLVED",
    "FrameDecision",
    "Stage0p5DecisionError",
    "cause_metrics",
    "decide_event",
    "decide_frame",
]
