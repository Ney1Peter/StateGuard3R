"""Fixed causal ordering for the independent StateTriage3R Stage 0.6 pilot."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .state_triage_stage0p5 import CAUSES, UNRESOLVED, FrameDecision, Stage0p5DecisionError, cause_metrics


METHODS = ("S0.6", "T0.6")
DECISION_CONTRACT: dict[str, Any] = {
    "typed_method": "T0.6",
    "scalar_method": "S0.6",
    "typed_order": [
        "sharpness_lte_0.05_bad_observation",
        "clipped_fraction_gte_0.20_transient_local_content",
        "coverage_lt_0.25_normal_novelty",
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


class Stage0p6DecisionError(Stage0p5DecisionError):
    """Raised when the Stage 0.6 fixed decision contract is malformed."""


def _finite(row: Mapping[str, Any], name: str, *, allow_none: bool = False) -> float | None:
    import math

    if name not in row:
        raise Stage0p6DecisionError(f"evidence row lacks {name}")
    value = row[name]
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise Stage0p6DecisionError(f"evidence row {name} must be finite")
    return float(value)


def _decision(label: str, reason: str) -> FrameDecision:
    if label == UNRESOLVED:
        posterior = {cause: 0.25 for cause in CAUSES}
    else:
        posterior = {cause: 0.70 if cause == label else 0.10 for cause in CAUSES}
    return FrameDecision(label=label, posterior=posterior, reason=reason)


def decide_frame(method: str, row: Mapping[str, Any]) -> FrameDecision:
    """Make one causal current-row decision without recipe/source dependence."""

    if method not in METHODS:
        raise Stage0p6DecisionError("unsupported method")
    coverage = _finite(row, "coverage_ratio")
    residual_z = _finite(row, "native_geometric_residual_z", allow_none=True)
    if method == "S0.6":
        if coverage < float(DECISION_CONTRACT["coverage_novelty_max_exclusive"]):
            return _decision("normal_novelty", "coverage_below_predeclared_novelty_threshold")
        if residual_z is not None and residual_z >= float(DECISION_CONTRACT["registration_z_min"]):
            return _decision("registration_or_order_fault", "native_geometric_residual_above_predeclared_threshold")
        return _decision(UNRESOLVED, "no_predeclared_scalar_branch")
    if _finite(row, "sharpness") <= float(DECISION_CONTRACT["sharpness_bad_observation_max"]):
        return _decision("bad_observation", "global_information_collapse_before_local_and_coverage")
    if _finite(row, "clipped_fraction") >= float(DECISION_CONTRACT["clipped_fraction_transient_min"]):
        return _decision("transient_local_content", "local_clipped_content_before_coverage")
    if coverage < float(DECISION_CONTRACT["coverage_novelty_max_exclusive"]):
        return _decision("normal_novelty", "coverage_below_predeclared_novelty_threshold")
    pose_jump_z = _finite(row, "pose_jump_z", allow_none=True)
    if (pose_jump_z is not None and pose_jump_z >= float(DECISION_CONTRACT["registration_z_min"])) or (residual_z is not None and residual_z >= float(DECISION_CONTRACT["registration_z_min"])):
        return _decision("registration_or_order_fault", "temporal_or_geometric_conflict_after_structural_cues")
    return _decision(UNRESOLVED, "no_predeclared_typed_branch")


def decide_event(rows: Sequence[Mapping[str, Any]], *, method: str, start_frame: int, end_frame: int) -> dict[str, Any]:
    if start_frame < 0 or end_frame < start_frame or end_frame >= len(rows):
        raise Stage0p6DecisionError("event frame range is invalid")
    for frame_id in range(start_frame, end_frame + 1):
        decision = decide_frame(method, rows[frame_id])
        if decision.label != UNRESOLVED:
            return {**decision.to_dict(), "decision_frame": frame_id, "detection_delay_frames": frame_id - start_frame}
    decision = _decision(UNRESOLVED, "no_causal_decision_in_event_window")
    return {**decision.to_dict(), "decision_frame": None, "detection_delay_frames": None}


__all__ = ["CAUSES", "DECISION_CONTRACT", "METHODS", "UNRESOLVED", "Stage0p6DecisionError", "cause_metrics", "decide_event", "decide_frame"]
