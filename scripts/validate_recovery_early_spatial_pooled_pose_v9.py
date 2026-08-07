#!/usr/bin/env python3
"""Independently validate one immutable v9 terminal-evidence/control record."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = "GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250"
MIN_FREE_MIB = 12288
BASELINE = ROOT / "outputs" / "recovery-policy-development-v1-dynamic-baseline-0001"
PROTECTED = ("checkpoint-load-audit.json", "health.jsonl", "predictions-summary.json", "trajectory.json")


class V9ValidationError(RuntimeError):
    """The sole validator found incomplete, unordered, or failing v9 evidence."""


@dataclass(frozen=True)
class Paths:
    output: Path
    preflight: Path
    main: Path
    postflight: Path
    transcript: Path
    driver: Path
    result: Path
    report: Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    return parser


def _paths(run_id: str) -> Paths:
    if not run_id.startswith("recovery-early-spatial-pooled-pose-v9-"):
        raise V9ValidationError("not a canonical v9 run id")
    logs, tmp = ROOT / "logs", ROOT / "tmp"
    return Paths(
        ROOT / "outputs" / run_id, logs / f"{run_id}-preflight.json",
        logs / f"{run_id}-main.log", logs / f"{run_id}-postflight.json",
        logs / f"{run_id}-tmux-transcript.log", tmp / f"{run_id}-driver.sh",
        logs / f"{run_id}-result.json", logs / f"{run_id}-gate-b-control-validation.log",
    )


def _regular_frozen(path: Path, *, mode: int = 0o444) -> bytes:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != mode:
        raise V9ValidationError(f"not frozen regular {mode:04o}: {path}")
    data = path.read_bytes()
    if b"\0" in data:
        raise V9ValidationError(f"NUL-containing artifact: {path}")
    return data


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(_regular_frozen(path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V9ValidationError(f"cannot parse terminal JSON: {path}") from error
    if not isinstance(value, dict):
        raise V9ValidationError(f"terminal JSON is not object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(_regular_frozen(path)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V9ValidationError(message)


def _validate_permissions(paths: Paths) -> None:
    metadata = os.lstat(paths.output)
    _require(stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) == 0o555, "output root is not 0555")
    for path in paths.output.rglob("*"):
        metadata = os.lstat(path)
        _require(not stat.S_ISLNK(metadata.st_mode), f"output symlink: {path}")
        if stat.S_ISREG(metadata.st_mode):
            _regular_frozen(path)
        elif stat.S_ISDIR(metadata.st_mode):
            _require(stat.S_IMODE(metadata.st_mode) in (0o555, 0o755), f"unexpected output directory mode: {path}")
        else:
            raise V9ValidationError(f"unexpected output artifact type: {path}")


def _gpu_preflight(preflight: Mapping[str, Any]) -> None:
    snapshots = preflight.get("gpu_snapshots")
    _require(isinstance(snapshots, list) and len(snapshots) == 2, "preflight lacks exactly two GPU snapshots")
    for snapshot in snapshots:
        _require(isinstance(snapshot, dict), "GPU snapshot is malformed")
        selected = snapshot.get("selected")
        _require(isinstance(selected, dict) and selected.get("uuid") == GPU_UUID and type(selected.get("memory_free_mib")) is int and selected["memory_free_mib"] >= MIN_FREE_MIB, "GPU preflight UUID/free-memory gate failed")


def _ordered_markers(main: bytes, transcript: bytes, *, run_id: str, command_sha: str, pid: int) -> None:
    main_text, transcript_text = main.decode("utf-8"), transcript.decode("utf-8")
    ordered = (
        f"V9_DRIVER_START run_id={run_id}",
        f"V9_DRIVER_COMMAND_SHA256 {command_sha}",
        f"V9_DRIVER_DISPATCHED run_id={run_id}",
        f"V9_FORWARD_DISPATCHED run_id={run_id} command_sha256={command_sha} child_pid={pid}",
        f"V9_DRIVER_EXIT run_id={run_id} child_pid={pid} exit_code=0",
    )
    indexes = [main_text.find(marker) for marker in ordered]
    _require(all(index >= 0 for index in indexes) and indexes == sorted(indexes), "main log markers are missing or unordered")
    for marker in (ordered[2], ordered[3], ordered[4]):
        _require(marker in transcript_text, "live tmux transcript lacks original ordered driver marker")


def validate_control(run_id: str) -> dict[str, Any]:
    """Validate only the preregistered dynamic always-commit control."""
    paths = _paths(run_id)
    required_id = "recovery-early-spatial-pooled-pose-v9-dynamic-always-commit-0001"
    _require(run_id == required_id, "this validator authorizes only the one v9 dynamic control")
    _require(not paths.report.exists(), "validation report path is already consumed")
    for path in (paths.preflight, paths.main, paths.postflight, paths.transcript, paths.driver, paths.result):
        _regular_frozen(path)
    _validate_permissions(paths)
    preflight, postflight, result = _json(paths.preflight), _json(paths.postflight), _json(paths.result)
    _require(preflight.get("run_id") == run_id and result.get("run_id") == run_id and postflight.get("run_id") == run_id, "terminal run IDs disagree")
    command_sha, pid = preflight.get("command_sha256"), result.get("child_pid")
    _require(isinstance(command_sha, str) and len(command_sha) == 64 and type(pid) is int, "preflight/result command or PID is malformed")
    _require(result.get("command_sha256") == command_sha and result.get("exit_code") == 0, "result command hash or exit is invalid")
    _require(postflight.get("result") == result and postflight.get("child_pid_absent") is True, "postflight is not bound to exited original child")
    _gpu_preflight(preflight)
    _ordered_markers(_regular_frozen(paths.main), _regular_frozen(paths.transcript), run_id=run_id, command_sha=command_sha, pid=pid)
    run, timeline = _json(paths.output / "run.json"), _json(paths.output / "state-timeline.json")
    _require(run.get("status") == "succeeded" and run.get("pid") == pid, "output run JSON is not bound to successful child")
    _require(run.get("output_dir") == str(paths.output), "run JSON output path is not direct/final")
    _require(run.get("state_policy", {}).get("name") == "always-commit" and timeline.get("policy") == "always-commit", "control policy is not always-commit")
    transactions = timeline.get("transactions")
    _require(isinstance(transactions, list) and len(transactions) == 30 and timeline.get("pending_transaction_count") == 0, "control has wrong transaction/pending count")
    for index, transaction in enumerate(transactions):
        _require(isinstance(transaction, dict) and transaction.get("frame_id") == index and transaction.get("action") == "commit" and transaction.get("current_alarm") is False and transaction.get("export_action") == "export_real_camera_pose" and transaction.get("pending_transaction_count") == 0, "control timeline is not final direct always-commit")
        _require(transaction.get("restore_witness") is None and "early_spatial_pooled_pose_token" not in transaction, "control path has candidate recovery evidence")
    baseline_run = _json(BASELINE / "run.json")
    baseline_runtime, runtime = baseline_run.get("runtime_seconds"), run.get("runtime_seconds")
    _require(type(baseline_runtime) in (int, float) and type(runtime) in (int, float) and baseline_runtime > 0 and runtime > 0, "runtime evidence is malformed")
    ratio = float(runtime) / float(baseline_runtime)
    _require(ratio <= 1.20, "control runtime ratio exceeds 1.20")
    equal: dict[str, bool] = {}
    for name in PROTECTED:
        equal[name] = _regular_frozen(paths.output / name) == _regular_frozen(BASELINE / name)
    _require(all(equal.values()), "protected model/health files are not byte-identical to v1 baseline")
    return {
        "schema_version": "stateguard3r.v9-gate-b-control-validator.v1",
        "status": "PASS", "run_id": run_id, "command_sha256": command_sha,
        "child_pid": pid, "protected_file_byte_equivalence": equal,
        "runtime_seconds": runtime, "baseline_runtime_seconds": baseline_runtime,
        "runtime_ratio": ratio, "terminal_artifacts": {
            name: _sha(path) for name, path in paths.__dict__.items()
            if name not in ("output", "report")
        },
    }


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _regular_frozen_or_freeze(path)


def _regular_frozen_or_freeze(path: Path) -> None:
    if not path.is_file() or path.is_symlink() or b"\0" in path.read_bytes():
        raise V9ValidationError("validator could not freeze its report")
    os.chmod(path, 0o444)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = _paths(args.run_id)
    try:
        report = validate_control(args.run_id)
    except V9ValidationError as error:
        if not paths.report.exists():
            _write_report(paths.report, {"schema_version": "stateguard3r.v9-gate-b-control-validator.v1", "status": "FAIL", "run_id": args.run_id, "error": str(error)})
        raise
    _write_report(paths.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
