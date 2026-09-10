"""Frozen decisions for the independent Stage 0.8 interaction replication."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .state_triage_stage0p5 import CAUSES, UNRESOLVED, FrameDecision, Stage0p5DecisionError, cause_metrics


METHODS = ("S0.8", "C0.8", "T0.8")
DECISION_CONTRACT: dict[str, Any] = {
    "typed_method": "T0.8",
    "coverage_ablation_method": "C0.8",
    "scalar_method": "S0.8",
    "typed_order": ["sharpness_lte_0.05_bad_observation", "clipped_fraction_gte_0.20_transient_local_content", "coverage_lt_0.25_normal_novelty", "pose_jump_z_or_native_geometric_residual_z_gte_3_registration_or_order_fault", "otherwise_unresolved"],
    "coverage_ablation_order": ["sharpness_lte_0.05_bad_observation", "coverage_lt_0.25_normal_novelty", "clipped_fraction_gte_0.20_transient_local_content", "pose_jump_z_or_native_geometric_residual_z_gte_3_registration_or_order_fault", "otherwise_unresolved"],
    "scalar_order": ["coverage_lt_0.25_normal_novelty", "native_geometric_residual_z_gte_3_registration_or_order_fault", "otherwise_unresolved"],
    "sharpness_bad_observation_max": 0.05,
    "coverage_novelty_max_exclusive": 0.25,
    "clipped_fraction_transient_min": 0.20,
    "registration_z_min": 3.0,
    "aggregation": "first_non_unresolved_current_row_in_predeclared_event_window",
}
INTERACTION_CONTRACT: dict[str, Any] = {
    "truth": "transient_local_content",
    "row": "first_predeclared_event_row",
    "sharpness_min_exclusive": 0.05,
    "clipped_fraction_min": 0.20,
    "coverage_max_exclusive": 0.25,
    "typed_expected_label": "transient_local_content",
    "coverage_ablation_expected_label": "normal_novelty",
}


class Stage0p8DecisionError(Stage0p5DecisionError):
    pass


def _finite(row: Mapping[str, Any], name: str, *, allow_none: bool = False) -> float | None:
    if name not in row:
        raise Stage0p8DecisionError(f"evidence row lacks {name}")
    value = row[name]
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise Stage0p8DecisionError(f"evidence row {name} must be finite")
    return float(value)


def _decision(label: str, reason: str) -> FrameDecision:
    posterior = {cause: 0.25 for cause in CAUSES} if label == UNRESOLVED else {cause: 0.70 if cause == label else 0.10 for cause in CAUSES}
    return FrameDecision(label=label, posterior=posterior, reason=reason)


def _registration(row: Mapping[str, Any], residual_z: float | None) -> FrameDecision:
    pose_z = _finite(row, "pose_jump_z", allow_none=True)
    if (pose_z is not None and pose_z >= float(DECISION_CONTRACT["registration_z_min"])) or (residual_z is not None and residual_z >= float(DECISION_CONTRACT["registration_z_min"])):
        return _decision("registration_or_order_fault", "temporal_or_geometric_conflict_after_structural_cues")
    return _decision(UNRESOLVED, "no_predeclared_typed_branch")


def decide_frame(method: str, row: Mapping[str, Any]) -> FrameDecision:
    if method not in METHODS:
        raise Stage0p8DecisionError("unsupported method")
    coverage, residual_z = _finite(row, "coverage_ratio"), _finite(row, "native_geometric_residual_z", allow_none=True)
    if method == "S0.8":
        if coverage < float(DECISION_CONTRACT["coverage_novelty_max_exclusive"]):
            return _decision("normal_novelty", "coverage_below_predeclared_novelty_threshold")
        if residual_z is not None and residual_z >= float(DECISION_CONTRACT["registration_z_min"]):
            return _decision("registration_or_order_fault", "native_geometric_residual_above_predeclared_threshold")
        return _decision(UNRESOLVED, "no_predeclared_scalar_branch")
    if _finite(row, "sharpness") <= float(DECISION_CONTRACT["sharpness_bad_observation_max"]):
        return _decision("bad_observation", "global_information_collapse_before_structural_branches")
    clipped = _finite(row, "clipped_fraction")
    if method == "T0.8" and clipped >= float(DECISION_CONTRACT["clipped_fraction_transient_min"]):
        return _decision("transient_local_content", "local_clipped_content_before_coverage")
    if coverage < float(DECISION_CONTRACT["coverage_novelty_max_exclusive"]):
        return _decision("normal_novelty", "coverage_below_predeclared_novelty_threshold")
    if method == "C0.8" and clipped >= float(DECISION_CONTRACT["clipped_fraction_transient_min"]):
        return _decision("transient_local_content", "local_clipped_content_after_coverage")
    return _registration(row, residual_z)


def decide_event(rows: Sequence[Mapping[str, Any]], *, method: str, start_frame: int, end_frame: int) -> dict[str, Any]:
    if start_frame < 0 or end_frame < start_frame or end_frame >= len(rows):
        raise Stage0p8DecisionError("event frame range is invalid")
    for frame_id in range(start_frame, end_frame + 1):
        decision = decide_frame(method, rows[frame_id])
        if decision.label != UNRESOLVED:
            return {**decision.to_dict(), "decision_frame": frame_id, "detection_delay_frames": frame_id - start_frame}
    decision = _decision(UNRESOLVED, "no_causal_decision_in_event_window")
    return {**decision.to_dict(), "decision_frame": None, "detection_delay_frames": None}


__all__ = ["CAUSES", "DECISION_CONTRACT", "INTERACTION_CONTRACT", "METHODS", "UNRESOLVED", "Stage0p8DecisionError", "cause_metrics", "decide_event", "decide_frame"]
