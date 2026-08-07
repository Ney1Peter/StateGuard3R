#!/usr/bin/env python3
"""CPU-only audit of the frozen online-quarantine development v2 matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs"
SCHEMA_VERSION = "stateguard3r.online-quarantine-development-evaluation.v2"
EQUIVALENCE_FILES = ("checkpoint-load-audit.json", "health.jsonl", "predictions-summary.json", "trajectory.json")


class OnlineQuarantineEvaluationError(ValueError):
    """Raised when a frozen v2 development matrix is malformed or unsafe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OnlineQuarantineEvaluationError(message)


def _legacy() -> Any:
    if str(ROOT) not in os.sys.path:
        os.sys.path.insert(0, str(ROOT))
    from scripts import evaluate_development_recovery_v1

    return evaluate_development_recovery_v1


def _candidate_evidence(directory: Path, *, frame_count: int) -> dict[str, Any]:
    legacy = _legacy()
    timeline = legacy._json(legacy._regular(directory / "state-timeline.json", label="candidate state timeline"), label="candidate state timeline")
    _require(timeline.get("schema_version") == "stateguard3r.recal3r-online-quarantine-v2.v1", "candidate timeline schema differs")
    _require(timeline.get("policy") == "detector-v3-online-current-quarantine", "candidate timeline policy differs")
    _require(timeline.get("watchdog") == 8 and timeline.get("pending_transaction_count") == 0, "candidate watchdog or pending contract differs")
    rows = timeline.get("transactions")
    _require(isinstance(rows, list) and len(rows) == frame_count, "candidate timeline length differs")
    rollback_ids: list[int] = []
    clear_after_episode: list[int] = []
    previous_was_rollback = False
    consecutive_rollbacks = 0
    proposed_digests: set[str] = set()
    for frame_id, row in enumerate(rows):
        _require(isinstance(row, Mapping) and row.get("frame_id") == frame_id, "candidate timeline frame IDs differ")
        _require(row.get("pending_transaction_count") == 0, "candidate timeline has a pending transaction")
        detector = row.get("online_detector")
        _require(isinstance(detector, Mapping), "candidate timeline lacks online detector evidence")
        _require(detector.get("frame_id") == frame_id and isinstance(detector.get("hybrid_alarm"), bool), "candidate detector evidence differs")
        action = row.get("action")
        if action == "quarantine_current_rollback":
            _require(detector["hybrid_alarm"] is True, "rollback lacks a current online alarm")
            for key in ("pre_state_digest_sha256", "proposed_state_digest_sha256", "committed_state_digest_sha256"):
                _require(isinstance(row.get(key), str) and len(row[key]) == 64, f"rollback {key} is invalid")
            _require(row["pre_state_digest_sha256"] == row["committed_state_digest_sha256"], "rollback committed state differs from pre-state")
            _require(row["proposed_state_digest_sha256"] != row["committed_state_digest_sha256"], "rollback did not change candidate post-state")
            consecutive_rollbacks += 1
            _require(row.get("consecutive_rollbacks") == consecutive_rollbacks and consecutive_rollbacks <= 8, "rollback watchdog evidence differs")
            rollback_ids.append(frame_id)
            proposed_digests.add(row["proposed_state_digest_sha256"])
            previous_was_rollback = True
        else:
            _require(action == "commit" and detector["hybrid_alarm"] is False, "candidate has an invalid non-rollback action")
            if previous_was_rollback:
                _require(row.get("consecutive_rollbacks") == 0, "first clear frame did not reset rollback count")
                clear_after_episode.append(frame_id)
            previous_was_rollback = False
            consecutive_rollbacks = 0
    _require(rollback_ids, "candidate never performed an actual quarantine")
    ledger_path = legacy._regular(directory / "observed-health-ledger.jsonl", label="candidate observed-health ledger")
    if str(ROOT / "src") not in os.sys.path:
        os.sys.path.insert(0, str(ROOT / "src"))
    from stateguard3r.health import read_health_jsonl

    ledger = read_health_jsonl(ledger_path)
    _require(len(ledger) == frame_count and [record.frame_id for record in ledger] == list(range(frame_count)), "observed health ledger is not one-to-one")
    ledger_rollbacks = [record.frame_id for record in ledger if record.decision == "quarantine_current_rollback"]
    _require(ledger_rollbacks == rollback_ids, "observed health ledger differs from rollback timeline")
    return {
        "rollback_frame_ids": rollback_ids,
        "rollback_count": len(rollback_ids),
        "first_clear_frames_after_episode": clear_after_episode,
        "observed_health_ledger": legacy._artifact(ledger_path),
        "proposed_digest_count": len(proposed_digests),
    }


def _trial(name: str, manifest_path: Path, baseline_dir: Path, always_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    legacy = _legacy()
    manifest, _manifest_payload, source_manifest, source_payload = legacy._development_manifest(manifest_path)
    baseline, baseline_trajectory = legacy._run_and_trajectory(baseline_dir, manifest, label=f"{name} baseline")
    always, always_trajectory = legacy._run_and_trajectory(always_dir, manifest, label=f"{name} always")
    candidate, candidate_trajectory = legacy._run_and_trajectory(candidate_dir, manifest, label=f"{name} candidate")
    _require(always.get("state_policy", {}).get("name") == "always-commit", f"{name} always policy differs")
    _require(candidate.get("state_policy", {}).get("name") == "detector-v3-online-current-quarantine", f"{name} candidate policy differs")
    equivalence = {
        filename: legacy._sha256(legacy._regular(Path(baseline_dir) / filename, label=f"{name} baseline {filename}"))
        == legacy._sha256(legacy._regular(Path(always_dir) / filename, label=f"{name} always {filename}"))
        for filename in EQUIVALENCE_FILES
    }
    _require(all(equivalence.values()), f"{name} baseline and always controls are not byte-equivalent")
    gt_positions, gt_rotations = legacy._logical_base_gt(source_payload)
    metrics = {
        "baseline": legacy._metrics(baseline_trajectory, gt_positions, gt_rotations),
        "always_commit": legacy._metrics(always_trajectory, gt_positions, gt_rotations),
        "candidate": legacy._metrics(candidate_trajectory, gt_positions, gt_rotations),
    }
    effects = legacy._effect(metrics["baseline"], metrics["candidate"])
    evidence = _candidate_evidence(Path(candidate_dir), frame_count=int(candidate.get("frame_count", 0)))
    return {
        "name": name,
        "input_manifest": legacy._artifact(manifest),
        "logical_base_source_manifest": legacy._artifact(source_manifest),
        "baseline_always_commit_byte_equivalence": equivalence,
        "runs": {
            "baseline": legacy._artifact(Path(baseline_dir) / "run.json"),
            "always_commit": legacy._artifact(Path(always_dir) / "run.json"),
            "candidate": legacy._artifact(Path(candidate_dir) / "run.json"),
        },
        "metrics": metrics,
        "candidate_vs_baseline_effect": effects,
        "runtime": {
            "baseline_seconds": float(baseline["runtime_seconds"]),
            "always_seconds": float(always["runtime_seconds"]),
            "candidate_seconds": float(candidate["runtime_seconds"]),
            "always_to_baseline_ratio": float(always["runtime_seconds"]) / float(baseline["runtime_seconds"]),
            "candidate_to_baseline_ratio": float(candidate["runtime_seconds"]) / float(baseline["runtime_seconds"]),
            "runtime_scope": candidate.get("runtime_scope"),
        },
        "candidate_evidence": evidence,
        "trajectory_byte_different_from_baseline": legacy._sha256(legacy._regular(Path(candidate_dir) / "trajectory.json", label=f"{name} candidate trajectory"))
        != legacy._sha256(legacy._regular(Path(baseline_dir) / "trajectory.json", label=f"{name} baseline trajectory")),
    }


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    length = len(ordered)
    _require(length > 0, "cannot take median of no values")
    return ordered[length // 2] if length % 2 else (ordered[length // 2 - 1] + ordered[length // 2]) / 2.0


def _selection(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    effects = [trial["candidate_vs_baseline_effect"] for trial in trials]
    ate = [float(effect["ATE_RMSE_effect"]) for effect in effects]
    rpe = [float(effect["RPE_translation_RMSE_effect"]) for effect in effects]
    ratios = [float(trial["runtime"]["candidate_to_baseline_ratio"]) for trial in trials]
    action_counts = [int(trial["candidate_evidence"]["rollback_count"]) for trial in trials]
    changed = [bool(trial["trajectory_byte_different_from_baseline"]) for trial in trials]
    gates = {
        "all_controls_byte_equivalent": all(all(trial["baseline_always_commit_byte_equivalence"].values()) for trial in trials),
        "all_conditions_have_quarantine": all(value > 0 for value in action_counts),
        "at_least_one_trajectory_changed": any(changed),
        "at_least_two_conditions_improve_both_metrics": sum(left > 0.0 and right > 0.0 for left, right in zip(ate, rpe, strict=True)) >= 2,
        "median_ate_effect_at_least_5_percent": _median(ate) >= 0.05,
        "median_translation_rpe_effect_at_least_5_percent": _median(rpe) >= 0.05,
        "candidate_runtime_median_at_most_1_20": _median(ratios) <= 1.20,
    }
    return {
        "effects": {"ATE_RMSE": ate, "RPE_translation_RMSE": rpe, "candidate_runtime_ratios": ratios},
        "medians": {"ATE_RMSE_effect": _median(ate), "RPE_translation_RMSE_effect": _median(rpe), "candidate_runtime_ratio": _median(ratios)},
        "gates": gates,
        "outcome": "ONLINE_QUARANTINE_DEVELOPMENT_CANDIDATE_READY" if all(gates.values()) else "ONLINE_QUARANTINE_DEVELOPMENT_FEASIBILITY_NO_GO",
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _freeze(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        _require(not path.is_symlink(), f"refusing to freeze symlink {path}")
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def evaluate(trials: Sequence[tuple[str, Path, Path, Path, Path]], output_dir: Path) -> Path:
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "online quarantine evaluation requires CUDA_VISIBLE_DEVICES='' ")
    output = output_dir.resolve(strict=False)
    _require(output.parent == OUTPUT_ROOT and not output.exists(), "output must be a new direct StateGuard3R/outputs child")
    _require(len(trials) == 3 and {trial[0] for trial in trials} == {"dynamic", "wrong", "low"}, "v2 evaluation requires exactly dynamic, wrong, and low trials")
    results = [_trial(*trial) for trial in sorted(trials)]
    selection = _selection(results)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scope": "disclosed development-only post-hoc evaluation; not formal recovery evidence",
        "gt_binding": "source-manifest rows 0--29 with source_pool_role=base; read only after all GPU forwards froze",
        "trials": results,
        "selection": selection,
    }
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".staging", dir=output.parent))
    try:
        _write_json(staging / "evaluation.json", payload)
        _freeze(staging)
        os.replace(staging, output)
    except Exception:
        for path in sorted(staging.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        staging.rmdir()
        raise
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", action="append", nargs=5, metavar=("NAME", "MANIFEST", "BASELINE", "ALWAYS", "CANDIDATE"), required=True)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    trials = [(name, Path(manifest), Path(baseline), Path(always), Path(candidate)) for name, manifest, baseline, always, candidate in args.trial]
    output = evaluate(trials, args.output_dir)
    print(json.dumps({"output_dir": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
