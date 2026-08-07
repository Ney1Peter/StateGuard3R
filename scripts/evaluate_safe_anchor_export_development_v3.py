#!/usr/bin/env python3
"""CPU-only post-hoc audit of the frozen safe-anchor v3 development matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs"
SCHEMA_VERSION = "stateguard3r.safe-anchor-export-development-evaluation.v3"
EQUIVALENCE_FILES = ("checkpoint-load-audit.json", "health.jsonl", "predictions-summary.json", "trajectory.json")


class SafeAnchorEvaluationError(ValueError):
    pass


def _require(value: bool, message: str) -> None:
    if not value:
        raise SafeAnchorEvaluationError(message)


def _legacy() -> Any:
    if str(ROOT) not in os.sys.path:
        os.sys.path.insert(0, str(ROOT))
    from scripts import evaluate_development_recovery_v1

    return evaluate_development_recovery_v1


def _trajectory(directory: Path, *, frame_count: int) -> dict[str, Any]:
    legacy = _legacy()
    payload = legacy._json(legacy._regular(directory / "trajectory.json", label="trajectory"), label="trajectory")
    frames = payload.get("frames")
    _require(isinstance(frames, list) and len(frames) == frame_count, "trajectory frames differ")
    for frame_id, frame in enumerate(frames):
        _require(isinstance(frame, Mapping) and frame.get("frame_id") == frame_id, "trajectory frame IDs differ")
        matrix = frame.get("camera_to_reference")
        _require(isinstance(matrix, list) and len(matrix) == 4 and all(isinstance(row, list) and len(row) == 4 for row in matrix), "trajectory matrix differs")
    return dict(payload)


def _candidate_evidence(directory: Path, *, frame_count: int) -> dict[str, Any]:
    legacy = _legacy()
    timeline = legacy._json(legacy._regular(directory / "state-timeline.json", label="candidate timeline"), label="candidate timeline")
    _require(timeline.get("schema_version") == "stateguard3r.recal3r-safe-anchor-export-v3.v1", "candidate timeline schema differs")
    _require(timeline.get("policy") == "detector-v3-incremental-safe-anchor-se3-export", "candidate policy differs")
    _require(timeline.get("watchdog") == 8 and timeline.get("pending_transaction_count") == 0, "candidate watchdog/pending differs")
    rows = timeline.get("transactions")
    _require(isinstance(rows, list) and len(rows) == frame_count, "candidate timeline length differs")
    rollbacks: list[int] = []
    actions: list[str] = []
    clear_real_exports = 0
    for frame_id, row in enumerate(rows):
        _require(isinstance(row, Mapping) and row.get("frame_id") == frame_id, "candidate timeline frame differs")
        detector = row.get("online_detector")
        _require(isinstance(detector, Mapping) and detector.get("frame_id") == frame_id, "candidate detector evidence differs")
        alarm = bool(detector.get("hybrid_alarm"))
        _require(row.get("current_alarm") is alarm and row.get("pending_transaction_count") == 0, "candidate alarm/pending differs")
        if alarm:
            _require(row.get("action") == "quarantine_current_rollback", "alarm lacks rollback")
            _require(row.get("export_action") == "export_safe_anchor_se3_constant_velocity", "alarm lacks safe-anchor export")
            anchors = row.get("anchor_frame_ids")
            _require(isinstance(anchors, list) and len(anchors) == 2 and all(type(value) is int and value < frame_id for value in anchors), "safe-anchor inputs are not causal")
            _require(isinstance(row.get("restore_witness"), Mapping), "rollback lacks structural witness")
            rollbacks.append(frame_id)
        else:
            _require(row.get("action") == "commit" and row.get("export_action") == "export_real_camera_pose", "clear frame did not export real pose")
            clear_real_exports += 1
        actions.append(str(row.get("export_action")))
    _require(rollbacks, "candidate never quarantined")
    if str(ROOT / "src") not in os.sys.path:
        os.sys.path.insert(0, str(ROOT / "src"))
    from stateguard3r.health import read_health_jsonl

    ledger_path = legacy._regular(directory / "observed-health-ledger.jsonl", label="observed health ledger")
    ledger = read_health_jsonl(ledger_path)
    _require(len(ledger) == frame_count and [row.frame_id for row in ledger] == list(range(frame_count)), "observed ledger differs")
    _require([row.frame_id for row in ledger if row.decision == "quarantine_current_rollback"] == rollbacks, "ledger/rollback mismatch")
    return {"rollback_frame_ids": rollbacks, "rollback_count": len(rollbacks), "clear_real_export_count": clear_real_exports, "export_actions": actions, "observed_health_ledger": legacy._artifact(ledger_path)}


def _trial(name: str, manifest_path: Path, baseline_dir: Path, always_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    legacy = _legacy()
    manifest, _payload, source_manifest, source_payload = legacy._development_manifest(manifest_path)
    baseline_run = legacy._json(legacy._regular(baseline_dir / "run.json", label="baseline run"), label="baseline run")
    always_run = legacy._json(legacy._regular(always_dir / "run.json", label="always run"), label="always run")
    candidate_run = legacy._json(legacy._regular(candidate_dir / "run.json", label="candidate run"), label="candidate run")
    count = int(candidate_run.get("frame_count", 0))
    _require(count == 30 and int(baseline_run.get("frame_count", 0)) == count and int(always_run.get("frame_count", 0)) == count, "frame count differs")
    _require(always_run.get("state_policy", {}).get("name") == "always-commit", "always policy differs")
    _require(candidate_run.get("state_policy", {}).get("name") == "detector-v3-incremental-safe-anchor-se3-export", "candidate policy differs")
    equivalence = {name: legacy._sha256(legacy._regular(baseline_dir / name, label=f"baseline {name}")) == legacy._sha256(legacy._regular(always_dir / name, label=f"always {name}")) for name in EQUIVALENCE_FILES}
    _require(all(equivalence.values()), "baseline/always artifacts differ")
    baseline, always, candidate = _trajectory(baseline_dir, frame_count=count), _trajectory(always_dir, frame_count=count), _trajectory(candidate_dir, frame_count=count)
    gt_positions, gt_rotations = legacy._logical_base_gt(source_payload)
    metrics = {"baseline": legacy._metrics(baseline, gt_positions, gt_rotations), "always_commit": legacy._metrics(always, gt_positions, gt_rotations), "candidate": legacy._metrics(candidate, gt_positions, gt_rotations)}
    evidence = _candidate_evidence(candidate_dir, frame_count=count)
    return {"name": name, "input_manifest": legacy._artifact(manifest), "logical_base_source_manifest": legacy._artifact(source_manifest), "baseline_always_commit_byte_equivalence": equivalence, "runs": {"baseline": legacy._artifact(baseline_dir / "run.json"), "always_commit": legacy._artifact(always_dir / "run.json"), "candidate": legacy._artifact(candidate_dir / "run.json")}, "metrics": metrics, "candidate_vs_baseline_effect": legacy._effect(metrics["baseline"], metrics["candidate"]), "runtime": {"baseline_seconds": float(baseline_run["runtime_seconds"]), "always_seconds": float(always_run["runtime_seconds"]), "candidate_seconds": float(candidate_run["runtime_seconds"]), "always_to_baseline_ratio": float(always_run["runtime_seconds"]) / float(baseline_run["runtime_seconds"]), "candidate_to_baseline_ratio": float(candidate_run["runtime_seconds"]) / float(baseline_run["runtime_seconds"])}, "candidate_evidence": evidence, "trajectory_byte_different_from_baseline": legacy._sha256(legacy._regular(candidate_dir / "trajectory.json", label="candidate trajectory")) != legacy._sha256(legacy._regular(baseline_dir / "trajectory.json", label="baseline trajectory"))}


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    _require(bool(ordered), "empty median")
    return ordered[len(ordered) // 2] if len(ordered) % 2 else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2


def _selection(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ate = [float(item["candidate_vs_baseline_effect"]["ATE_RMSE_effect"]) for item in trials]
    rpe = [float(item["candidate_vs_baseline_effect"]["RPE_translation_RMSE_effect"]) for item in trials]
    runtime = [float(item["runtime"]["candidate_to_baseline_ratio"]) for item in trials]
    gates = {"all_controls_byte_equivalent": all(all(item["baseline_always_commit_byte_equivalence"].values()) for item in trials), "all_conditions_have_quarantine": all(int(item["candidate_evidence"]["rollback_count"]) > 0 for item in trials), "at_least_one_trajectory_changed": any(bool(item["trajectory_byte_different_from_baseline"]) for item in trials), "at_least_two_conditions_improve_both_metrics": sum(left > 0 and right > 0 for left, right in zip(ate, rpe, strict=True)) >= 2, "median_ate_effect_at_least_5_percent": _median(ate) >= 0.05, "median_translation_rpe_effect_at_least_5_percent": _median(rpe) >= 0.05, "candidate_runtime_median_at_most_1_20": _median(runtime) <= 1.20}
    return {"effects": {"ATE_RMSE": ate, "RPE_translation_RMSE": rpe, "candidate_runtime_ratios": runtime}, "medians": {"ATE_RMSE_effect": _median(ate), "RPE_translation_RMSE_effect": _median(rpe), "candidate_runtime_ratio": _median(runtime)}, "gates": gates, "outcome": "SAFE_ANCHOR_EXPORT_DEVELOPMENT_V3_CANDIDATE_READY" if all(gates.values()) else "SAFE_ANCHOR_EXPORT_DEVELOPMENT_V3_FEASIBILITY_NO_GO"}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def evaluate(trials: Sequence[tuple[str, Path, Path, Path, Path]], output_dir: Path) -> Path:
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "v3 evaluation requires CUDA_VISIBLE_DEVICES='' ")
    output = output_dir.resolve(strict=False)
    _require(output.parent == OUTPUT_ROOT and not output.exists(), "output must be a new direct outputs child")
    _require(len(trials) == 3 and {item[0] for item in trials} == {"dynamic", "wrong", "low"}, "v3 requires dynamic/wrong/low")
    results = [_trial(*trial) for trial in sorted(trials)]
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".staging", dir=output.parent))
    try:
        _write_json(staging / "evaluation.json", {"schema_version": SCHEMA_VERSION, "scope": "disclosed development-only post-hoc evaluation; not formal recovery evidence", "gt_binding": "source-manifest logical-base GT read only after all six v3 forwards froze", "trials": results, "selection": _selection(results)})
        for path in sorted(staging.rglob("*"), key=lambda item: len(item.parts), reverse=True): path.chmod(0o444)
        staging.chmod(0o555); os.replace(staging, output)
    except Exception:
        for path in sorted(staging.rglob("*"), key=lambda item: len(item.parts), reverse=True): path.unlink(missing_ok=True)
        staging.rmdir(); raise
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--trial", action="append", nargs=5, metavar=("NAME", "MANIFEST", "BASELINE", "ALWAYS", "CANDIDATE"), required=True); parser.add_argument("output_dir", type=Path); args = parser.parse_args(argv)
    output = evaluate([(name, Path(manifest), Path(baseline), Path(always), Path(candidate)) for name, manifest, baseline, always, candidate in args.trial], args.output_dir); print(json.dumps({"output_dir": str(output)}, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
