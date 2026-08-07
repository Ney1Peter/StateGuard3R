from __future__ import annotations

from scripts.evaluate_online_quarantine_development_v2 import _selection


def _trial(*, ate: float, rpe: float, runtime: float, actions: int = 1, changed: bool = True) -> dict[str, object]:
    return {
        "candidate_vs_baseline_effect": {"ATE_RMSE_effect": ate, "RPE_translation_RMSE_effect": rpe},
        "runtime": {"candidate_to_baseline_ratio": runtime},
        "candidate_evidence": {"rollback_count": actions},
        "trajectory_byte_different_from_baseline": changed,
        "baseline_always_commit_byte_equivalence": {
            "checkpoint-load-audit.json": True,
            "health.jsonl": True,
            "predictions-summary.json": True,
            "trajectory.json": True,
        },
    }


def test_selection_reports_ready_only_when_every_preregistered_gate_passes() -> None:
    report = _selection(
        [
            _trial(ate=0.10, rpe=0.10, runtime=1.1),
            _trial(ate=0.08, rpe=0.07, runtime=1.0),
            _trial(ate=-0.01, rpe=0.06, runtime=1.2),
        ]
    )
    assert report["outcome"] == "ONLINE_QUARANTINE_DEVELOPMENT_CANDIDATE_READY"
    assert all(report["gates"].values())


def test_selection_emits_no_go_for_runtime_even_if_quality_improves() -> None:
    report = _selection(
        [
            _trial(ate=0.20, rpe=0.20, runtime=2.0),
            _trial(ate=0.10, rpe=0.10, runtime=2.5),
            _trial(ate=0.15, rpe=0.11, runtime=1.5),
        ]
    )
    assert report["outcome"] == "ONLINE_QUARANTINE_DEVELOPMENT_FEASIBILITY_NO_GO"
    assert report["gates"]["candidate_runtime_median_at_most_1_20"] is False
