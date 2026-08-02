#!/usr/bin/env python3
"""One-shot exploratory ledger-level quarantine proxy after Detector v2 GO."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
PROTOCOL = ROOT / "docs" / "protocols" / "quarantine-formal-v2.md"
FORMAL_EVALUATION = OUTPUTS / "formal-v2-evaluation-0001"
FORMAL_RUNS = OUTPUTS / "formal-v2-runs-0002"
COMBINED_THRESHOLD = 2.0686323694270046
BUFFER_LENGTH = 3
RUN_FILES = {"health_jsonl": "health.jsonl", "run_json": "run.json"}
SCHEMA = "stateguard3r.quarantine-v2"


class QuarantineV2Error(ValueError):
    """Raised when the exploratory quarantine evidence chain is incomplete."""


@dataclass(frozen=True)
class Snapshot:
    path: Path
    payload: bytes

    def artifact(self, path_text: str) -> dict[str, Any]:
        return {"path": path_text, "sha256": hashlib.sha256(self.payload).hexdigest(), "size_bytes": len(self.payload)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QuarantineV2Error(message)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _strict_json(payload: bytes, *, name: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise QuarantineV2Error(f"{name} contains non-finite JSON {value!r}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise QuarantineV2Error(f"{name} has duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, parse_constant=reject_constant, object_pairs_hook=reject_duplicate)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuarantineV2Error(f"{name} is not strict JSON: {error}") from error
    _require(isinstance(value, Mapping), f"{name} must be a JSON object")
    return value


def _read(path: Path, *, name: str, mode: int | None = 0o444) -> Snapshot:
    metadata = os.lstat(path)
    _require(stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), f"{name} must be a regular non-symlink file")
    if mode is not None:
        _require(stat.S_IMODE(metadata.st_mode) == mode, f"{name} must be frozen {mode:04o}")
    return Snapshot(path, path.read_bytes())


def _new_output(path: Path) -> Path:
    output = path.resolve(strict=False)
    _require(output.parent == OUTPUTS and not output.exists(), "output must be a new direct child of outputs")
    return output


def _freeze(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _publish(output: Path, files: Mapping[str, bytes]) -> Path:
    output = _new_output(output)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".staging", dir=OUTPUTS))
    published = False
    try:
        for name, payload in files.items():
            target = staging / name
            _require(target.resolve(strict=False).is_relative_to(staging.resolve(strict=False)), "unsafe output path")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        _freeze(staging)
        os.replace(staging, output)
        published = True
        return output
    finally:
        if not published and staging.exists():
            for directory in sorted((path for path in staging.rglob("*") if path.is_dir()), key=lambda item: len(item.parts), reverse=True):
                directory.chmod(0o755)
            staging.chmod(0o755)
            shutil.rmtree(staging)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(["git", "-C", str(repository), *arguments], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return completed.stdout.strip()


def _tracked(path: Path, *, name: str) -> Snapshot:
    relative = str(path.relative_to(ROOT))
    _require(_git(ROOT, "ls-files", "--error-unmatch", relative) == relative, f"{name} is not tracked")
    # Shared-workspace umasks may make tracked source files group-writable.
    # Exact Git HEAD bytes, not a checkout permission bit, establish provenance.
    snapshot = _read(path, name=name, mode=None)
    committed = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{relative}"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    _require(snapshot.payload == committed, f"{name} differs from tracked HEAD")
    return snapshot


def _formal_artifacts(evaluation_dir: Path) -> tuple[Snapshot, Snapshot, Snapshot, Mapping[str, Any], Mapping[str, Any]]:
    manifest = _read(evaluation_dir / "evaluation-manifest.json", name="formal v2 evaluation manifest")
    gate = _read(evaluation_dir / "go-no-go.json", name="formal v2 gate")
    metrics = _read(evaluation_dir / "holdout-metrics.json", name="formal v2 holdout metrics")
    manifest_value = _strict_json(manifest.payload, name="formal v2 evaluation manifest")
    gate_value = _strict_json(gate.payload, name="formal v2 gate")
    _require(manifest_value.get("decision") == "GO" and gate_value.get("decision") == "GO", "quarantine pilot requires formal Detector v2 GO")
    return manifest, gate, metrics, manifest_value, _strict_json(metrics.payload, name="formal v2 holdout metrics")


def _policy(scores: Sequence[float], deltas: Sequence[float | None]) -> dict[str, Any]:
    """Use scores and online update deltas only; labels are intentionally absent."""

    _require(len(scores) == len(deltas) == 30, "quarantine policy requires 30 score/delta positions")
    numeric_scores = [float(value) for value in scores]
    _require(all(math.isfinite(value) for value in numeric_scores), "quarantine scores must be finite")
    alarm = next((index for index, value in enumerate(numeric_scores) if value >= COMBINED_THRESHOLD), None)
    if alarm is None:
        return {"triggered": False, "alarm_position": None, "buffer_positions": [], "decision": "not_triggered", "baseline_absolute_update_mass": 0.0, "counterfactual_committed_mass": 0.0, "withheld_update_mass_proxy": 0.0}
    positions = list(range(alarm, min(alarm + BUFFER_LENGTH, len(numeric_scores))))
    values: list[float] = []
    for position in positions:
        value = deltas[position]
        number = 0.0 if value is None else float(value)
        _require(math.isfinite(number), "global_state_delta must be finite or null")
        values.append(abs(number))
    tail = [numeric_scores[position] for position in positions[-2:]]
    release = len(tail) == 2 and all(value < COMBINED_THRESHOLD for value in tail)
    baseline = math.fsum(values)
    committed = baseline if release else 0.0
    return {"triggered": True, "alarm_position": alarm, "buffer_positions": positions, "decision": "release" if release else "drop", "baseline_absolute_update_mass": baseline, "counterfactual_committed_mass": committed, "withheld_update_mass_proxy": baseline - committed}


def _health_deltas(path: Path, *, run_id: str) -> list[float | None]:
    snapshot = _read(path, name=f"{run_id} health ledger")
    records = [_strict_json(line, name=f"{run_id} health line {index}") for index, line in enumerate(snapshot.payload.splitlines(), start=1) if line.strip()]
    _require(len(records) == 30, f"{run_id} health frame count")
    forbidden = {"groundtruth", "depth", "label", "corruption_type", "event_start"}
    _require(all(not (forbidden & set(record)) for record in records), f"{run_id} health contains forbidden policy input")
    deltas: list[float | None] = []
    for index, record in enumerate(records):
        _require(record.get("frame_id") == index, f"{run_id} health frame IDs")
        value = record.get("global_state_delta")
        if value is None:
            deltas.append(None)
        else:
            number = float(value)
            _require(math.isfinite(number), f"{run_id} global_state_delta non-finite")
            deltas.append(number)
    return deltas


def commit(evaluation_dir: Path, runs_root: Path, output_dir: Path, *, protocol: Path) -> dict[str, Any]:
    _new_output(output_dir)
    _require(_git(ROOT, "status", "--porcelain") == "", "StateGuard3R must be clean before quarantine commitment")
    protocol_snapshot = _tracked(protocol.resolve(strict=True), name="quarantine protocol")
    script_snapshot = _tracked(Path(__file__).resolve(), name="quarantine evaluator")
    evaluation_manifest, gate, metrics, manifest_value, _ = _formal_artifacts(evaluation_dir.resolve(strict=True))
    run_artifacts = manifest_value.get("run_artifacts")
    _require(isinstance(run_artifacts, Mapping) and set(run_artifacts) == {"development-dynamic", "development-wrong", "development-low", "development2-dynamic", "development2-wrong", "development2-low", "holdout-dynamic", "holdout-wrong", "holdout-low"}, "formal evaluation must bind all nine run outputs")
    holdout: dict[str, Any] = {}
    for run_id in ("holdout-dynamic", "holdout-wrong", "holdout-low"):
        directory = runs_root / run_id
        _require(directory.is_dir() and stat.S_IMODE(os.lstat(directory).st_mode) == 0o555, f"{run_id} output not frozen")
        health = _read(directory / "health.jsonl", name=f"{run_id} health ledger")
        run_json = _read(directory / "run.json", name=f"{run_id} run metadata")
        holdout[run_id] = {"health_jsonl": health.artifact(f"{run_id}/health.jsonl"), "run_json": run_json.artifact(f"{run_id}/run.json")}
    commitment = {"schema_version": f"{SCHEMA}.commitment.v1", "stage": "commit", "state_guard_commit": _git(ROOT, "rev-parse", "HEAD"), "tracked_sources": {"protocol": protocol_snapshot.artifact(str(protocol)), "evaluator": script_snapshot.artifact(str(Path(__file__).resolve()))}, "formal_v2": {"evaluation_manifest": evaluation_manifest.artifact("evaluation-manifest.json"), "gate": gate.artifact("go-no-go.json"), "holdout_metrics": metrics.artifact("holdout-metrics.json")}, "holdout_ledger_artifacts": holdout, "policy": {"combined_threshold": COMBINED_THRESHOLD, "maximum_buffer_length": BUFFER_LENGTH, "release_rule": "release only if final two buffered scores are strictly below threshold; otherwise drop", "runtime_inputs": ["fixed_combined_score", "global_state_delta"], "forbidden_runtime_inputs": ["groundtruth", "depth", "labels", "corruption_type", "future_outside_fixed_buffer"]}, "scope": "exploratory ledger-level proxy only; no ReCal3R recurrent state or output is modified; no rollback unlocked"}
    payload = _json_bytes(commitment)
    output = _publish(output_dir, {"commitment-manifest.json": payload})
    return {"stage": "commit", "output_dir": str(output), "commitment_sha256": hashlib.sha256(payload).hexdigest()}


def evaluate(evaluation_dir: Path, commit_dir: Path, runs_root: Path, output_dir: Path) -> dict[str, Any]:
    _new_output(output_dir)
    commitment = _read(commit_dir / "commitment-manifest.json", name="quarantine commitment")
    commitment_value = _strict_json(commitment.payload, name="quarantine commitment")
    _require(commitment_value.get("schema_version") == f"{SCHEMA}.commitment.v1", "quarantine commitment schema")
    evaluation_manifest, _, _, _, metrics = _formal_artifacts(evaluation_dir.resolve(strict=True))
    _require(commitment_value.get("formal_v2", {}).get("evaluation_manifest", {}).get("sha256") == hashlib.sha256(evaluation_manifest.payload).hexdigest(), "formal evaluation binding changed")
    started = time.perf_counter()
    rows: dict[str, Any] = {}
    positive_reductions = 0
    for run in metrics["runs"]:
        run_id = str(run["run_id"])
        combined = metrics["methods"]["combined"]["per_run"][run_id]
        action = _policy(combined["scores"], _health_deltas(runs_root / run_id / "health.jsonl", run_id=run_id))
        labels = run["primary_labels"]
        _require(isinstance(labels, list) and len(labels) == 30, f"{run_id} evaluation labels")
        positions = action["buffer_positions"]
        action["evaluation_only_event_positions"] = sum(int(labels[position]) for position in positions)
        action["evaluation_only_clean_positions"] = len(positions) - action["evaluation_only_event_positions"]
        positive_reductions += int(action["withheld_update_mass_proxy"] > 0.0)
        rows[run_id] = action
    runtime = time.perf_counter() - started
    metrics_output = {"schema_version": f"{SCHEMA}.metrics.v1", "scope": "exploratory ledger-level committed-update-magnitude proxy; no altered ReCal3R forward", "policy": commitment_value["policy"], "runs": rows, "summary": {"triggered_runs": sum(int(row["triggered"]) for row in rows.values()), "dropped_runs": sum(int(row["decision"] == "drop") for row in rows.values()), "released_runs": sum(int(row["decision"] == "release") for row in rows.values()), "total_withheld_update_mass_proxy": math.fsum(float(row["withheld_update_mass_proxy"]) for row in rows.values()), "clean_buffer_positions_evaluation_only": sum(int(row["evaluation_only_clean_positions"]) for row in rows.values()), "event_buffer_positions_evaluation_only": sum(int(row["evaluation_only_event_positions"]) for row in rows.values()), "cpu_runtime_seconds": runtime}}
    feasibility = positive_reductions >= 1 and metrics_output["summary"]["triggered_runs"] >= 1
    decision = {"schema_version": f"{SCHEMA}.decision.v1", "decision": "EXPLORATORY_PROXY_ONLY_NO_ROLLBACK", "feasibility_conditions_passed": feasibility, "conditions": {"formal_detector_go_bound": True, "policy_uses_only_fixed_scores_and_online_global_state_delta": True, "no_baseline_recal3r_output_modified": True, "at_least_one_positive_proxy_reduction": positive_reductions >= 1}, "limitation": "The proxy withholds ledger update magnitude only. It does not establish a different recurrent state, trajectory, geometry, or recovery gain."}
    timeline = {f"timelines/{run_id}.json": _json_bytes({"run_id": run_id, **row}) for run_id, row in rows.items()}
    metric_bytes = _json_bytes(metrics_output)
    decision_bytes = _json_bytes(decision)
    manifest = {"schema_version": f"{SCHEMA}.evaluation.v1", "stage": "evaluate", "commitment": commitment.artifact("commitment-manifest.json"), "formal_evaluation": evaluation_manifest.artifact("evaluation-manifest.json"), "artifacts": {"metrics": {"path": "quarantine-metrics.json", "sha256": hashlib.sha256(metric_bytes).hexdigest(), "size_bytes": len(metric_bytes)}, "decision": {"path": "quarantine-decision.json", "sha256": hashlib.sha256(decision_bytes).hexdigest(), "size_bytes": len(decision_bytes)}}, "decision": decision["decision"]}
    manifest_bytes = _json_bytes(manifest)
    output = _publish(output_dir, {"quarantine-metrics.json": metric_bytes, "quarantine-decision.json": decision_bytes, "evaluation-manifest.json": manifest_bytes, **timeline})
    return {"stage": "evaluate", "output_dir": str(output), "decision": decision["decision"], "evaluation_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    commit_parser = actions.add_parser("commit")
    commit_parser.add_argument("evaluation_dir", type=Path)
    commit_parser.add_argument("runs_root", type=Path)
    commit_parser.add_argument("output_dir", type=Path)
    commit_parser.add_argument("--protocol", type=Path, required=True)
    evaluate_parser = actions.add_parser("evaluate")
    evaluate_parser.add_argument("evaluation_dir", type=Path)
    evaluate_parser.add_argument("commit_dir", type=Path)
    evaluate_parser.add_argument("runs_root", type=Path)
    evaluate_parser.add_argument("output_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = commit(args.evaluation_dir, args.runs_root, args.output_dir, protocol=args.protocol) if args.command == "commit" else evaluate(args.evaluation_dir, args.commit_dir, args.runs_root, args.output_dir)
    except (QuarantineV2Error, OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
