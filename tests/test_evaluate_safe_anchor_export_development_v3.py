from __future__ import annotations

from scripts.evaluate_safe_anchor_export_development_v3 import _selection


def _trial(ate: float, rpe: float, runtime: float) -> dict[str, object]:
    return {"candidate_vs_baseline_effect": {"ATE_RMSE_effect": ate, "RPE_translation_RMSE_effect": rpe}, "runtime": {"candidate_to_baseline_ratio": runtime}, "candidate_evidence": {"rollback_count": 1}, "trajectory_byte_different_from_baseline": True, "baseline_always_commit_byte_equivalence": {"checkpoint-load-audit.json": True, "health.jsonl": True, "predictions-summary.json": True, "trajectory.json": True}}


def test_v3_selection_is_ready_only_when_every_preregistered_gate_passes() -> None:
    report = _selection([_trial(0.08, 0.08, 1.1), _trial(0.06, 0.07, 1.0), _trial(-0.01, 0.06, 1.2)])
    assert report["outcome"] == "SAFE_ANCHOR_EXPORT_DEVELOPMENT_V3_CANDIDATE_READY"


def test_v3_selection_refuses_runtime_failure_despite_quality() -> None:
    report = _selection([_trial(0.2, 0.2, 1.3), _trial(0.1, 0.1, 1.4), _trial(0.15, 0.11, 1.5)])
    assert report["outcome"] == "SAFE_ANCHOR_EXPORT_DEVELOPMENT_V3_FEASIBILITY_NO_GO"
    assert report["gates"]["candidate_runtime_median_at_most_1_20"] is False
