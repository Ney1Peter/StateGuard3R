#!/usr/bin/env python3
"""Immutable commit, launch, calibration, and one-shot evaluation for Detector v2.

The v1 formal orchestrator is deliberately not reused: its source schema,
scorer, split semantics, and evidence status are specific to the v1 pilot.
This program keeps v2's disclosed six-run development pool separate from its
three new-scene blind runs and makes the blind read lock structural.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from stateguard3r import detection, detection_suite
from stateguard3r.detection_v2 import (
    CANONICAL_SIGNALS,
    DetectorV2Error,
    V2ScoringConfig,
    compute_v2_scores,
    validate_online_overlap,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
LOGS = ROOT / "logs"
RECAL3R = ROOT.parent / "baselines" / "ReCal3R"
RECAL3R_PYTHON = RECAL3R / ".venv" / "bin" / "python"
RUNNER = ROOT / "scripts" / "run_recal3r_smoke.py"
VALIDATOR = ROOT / "scripts" / "validate_formal_pilot_v2_inputs.py"
PROTOCOL = ROOT / "docs" / "protocols" / "detection-formal-v2.md"
V1_INPUTS = OUTPUTS / "formal-v1-inputs-0001"
V2_INPUTS = OUTPUTS / "formal-v2-inputs-0001"
V2_SEARCH = OUTPUTS / "detector-v2-development-search-0001" / "search.json"
V2_READINESS = OUTPUTS / "detector-v2-readiness-0001" / "readiness.json"
CHECKPOINT = RECAL3R / "src" / "cut3r_512_dpt_4_64.pth"
CHECKPOINT_SHA256 = "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103"
EXPECTED_RECAL3R_COMMIT = "466c7cdf3acd2f589f1d82e5f6391966f19db9ff"
RUN_FILES = {
    "checkpoint_load_audit": "checkpoint-load-audit.json",
    "health_jsonl": "health.jsonl",
    "predictions_summary": "predictions-summary.json",
    "trajectory": "trajectory.json",
    "run_json": "run.json",
}
RUNS = (
    ("development-dynamic", "development", "dynamic_occlusion", V1_INPUTS / "development" / "development-dynamic"),
    ("development-wrong", "development", "wrong_order_segment", V1_INPUTS / "development" / "development-wrong"),
    ("development-low", "development", "low_overlap_jump", V1_INPUTS / "development" / "development-low"),
    ("development2-dynamic", "development", "dynamic_occlusion", V1_INPUTS / "holdout" / "holdout-dynamic"),
    ("development2-wrong", "development", "wrong_order_segment", V1_INPUTS / "holdout" / "holdout-wrong"),
    ("development2-low", "development", "low_overlap_jump", V1_INPUTS / "holdout" / "holdout-low"),
    ("holdout-dynamic", "holdout", "dynamic_occlusion", V2_INPUTS / "holdout-dynamic"),
    ("holdout-wrong", "holdout", "wrong_order_segment", V2_INPUTS / "holdout-wrong"),
    ("holdout-low", "holdout", "low_overlap_jump", V2_INPUTS / "holdout-low"),
)
EVENT_ENDS = {"dynamic_occlusion": 19, "wrong_order_segment": 18, "low_overlap_jump": 19}
SCORING = V2ScoringConfig(window=10, min_history=5, max_z=10.0, master_seed=0)
SCALE_FLOORS = {
    "geometric_residual": 4.948262292669166e-05,
    "overlap": 0.006875120234255727,
    "pose_jump": 0.000806030222234299,
    "reliability": 0.000527346134185791,
    "update_magnitude": 0.0074473662806365136,
}
THRESHOLDS = {
    "combined": 2.0686323694270046,
    "random": 0.8757517985798549,
    "reliability_only": 5.485329360937135,
    "update_magnitude_only": 1.2113849262902454,
}
SCHEMA = "stateguard3r.formal-detection-v2"


class FormalV2Error(ValueError):
    """A condition necessary for formal-v2 evidence has not been proven."""


@dataclass(frozen=True)
class Snapshot:
    path: Path
    payload: bytes

    def artifact(self, path_text: str) -> dict[str, Any]:
        return {"path": path_text, "sha256": _sha_bytes(self.payload), "size_bytes": len(self.payload)}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    split: str
    corruption: str
    input_root: Path
    source_manifest: Snapshot
    input_manifest: Snapshot


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalV2Error(message)


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FormalV2Error(f"cannot encode strict JSON: {error}") from error


def _strict_json(payload: bytes, *, name: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise FormalV2Error(f"{name} contains non-finite JSON {value!r}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FormalV2Error(f"{name} has duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, parse_constant=reject_constant, object_pairs_hook=reject_duplicate)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalV2Error(f"{name} is not strict UTF-8 JSON: {error}") from error
    _require(isinstance(value, Mapping), f"{name} must be a JSON object")
    return value


def _read(path: Path, *, name: str, mode: int | None = None) -> Snapshot:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise FormalV2Error(f"cannot inspect {name}: {error}") from error
    _require(stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), f"{name} must be a regular non-symlink file")
    if mode is not None:
        _require(stat.S_IMODE(metadata.st_mode) == mode, f"{name} must have mode {mode:04o}")
    try:
        return Snapshot(path, path.read_bytes())
    except OSError as error:
        raise FormalV2Error(f"cannot read {name}: {error}") from error


def _inside(parent: Path, child: Path) -> bool:
    try:
        child.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def _new_output(path: Path) -> Path:
    output = path.resolve(strict=False)
    _require(output.parent == OUTPUTS, "formal output must be a new direct child of outputs")
    _require(not output.exists(), f"formal output already exists: {output}")
    return output


def _freeze_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _publish(output: Path, files: Mapping[str, bytes]) -> Path:
    output = _new_output(output)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".staging", dir=OUTPUTS))
    published = False
    try:
        for relative, payload in files.items():
            target = staging / relative
            _require(_inside(staging, target), "unsafe formal artifact path")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        _freeze_tree(staging)
        os.replace(staging, output)
        published = True
        return output
    finally:
        if not published and staging.exists():
            for directory in sorted((item for item in staging.rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True):
                directory.chmod(0o755)
            staging.chmod(0o755)
            shutil.rmtree(staging)


def _git(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(["git", "-C", str(repository), *arguments], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise FormalV2Error(f"cannot query repository {repository}: {error}") from error
    return completed.stdout.strip()


def _clean_commit(repository: Path, *, expected: str | None = None) -> str:
    _require(_git(repository, "status", "--porcelain") == "", f"repository must be clean: {repository}")
    commit = _git(repository, "rev-parse", "HEAD")
    if expected is not None:
        _require(commit == expected, f"unexpected repository commit for {repository}: {commit}")
    return commit


def _tracked_snapshot(path: Path, *, repository: Path, name: str) -> Snapshot:
    relative = str(path.relative_to(repository))
    _require(_git(repository, "ls-files", "--error-unmatch", relative) == relative, f"{name} is not tracked")
    snapshot = _read(path, name=name)
    try:
        committed = subprocess.run(
            ["git", "-C", str(repository), "show", f"HEAD:{relative}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise FormalV2Error(f"cannot read tracked bytes for {name}: {error}") from error
    _require(snapshot.payload == committed, f"{name} bytes differ from tracked HEAD blob")
    return snapshot


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _opencv_provenance() -> Mapping[str, Any]:
    """Capture the production OpenCV build without importing a model or CUDA."""

    code = (
        "import json,sys; "
        f"sys.path.insert(0,{str(ROOT / 'src')!r}); "
        "from stateguard3r.visual_overlap import opencv_runtime_provenance; "
        "print(json.dumps(opencv_runtime_provenance(),sort_keys=True,allow_nan=False))"
    )
    try:
        completed = subprocess.run(
            [str(RECAL3R_PYTHON), "-c", code],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=ROOT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "TMPDIR": str(ROOT / "tmp")},
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FormalV2Error(f"cannot capture production OpenCV provenance: {error}") from error
    value = _strict_json(completed.stdout.encode("utf-8"), name="production OpenCV provenance")
    _require(value.get("threads") == 1 and value.get("opencl_enabled") is False and value.get("rng_seed") == 0, "production OpenCV determinism provenance")
    for name in ("version", "module_sha256", "binary_sha256", "build_info_sha256"):
        _require(isinstance(value.get(name), str) and value.get(name), f"production OpenCV {name} missing")
    return value


def _run_specs() -> tuple[RunSpec, ...]:
    specs: list[RunSpec] = []
    for run_id, split, corruption, directory in RUNS:
        _require(directory.is_dir() and not directory.is_symlink(), f"input directory missing: {directory}")
        _require(stat.S_IMODE(os.lstat(directory).st_mode) == 0o555, f"input directory not frozen: {directory}")
        source = _read(directory / "source-manifest.json", name=f"{run_id} source manifest", mode=0o444)
        manifest = _read(directory / "input-manifest.json", name=f"{run_id} input manifest", mode=0o444)
        source_value = _strict_json(source.payload, name=f"{run_id} source manifest")
        input_value = _strict_json(manifest.payload, name=f"{run_id} input manifest")
        _require(source_value.get("corruption_type") == corruption, f"{run_id} source corruption mismatch")
        _require(input_value.get("frame_count") == 30, f"{run_id} input is not a 30-frame manifest")
        corruptions = input_value.get("corruptions")
        _require(isinstance(corruptions, list) and len(corruptions) == 1 and isinstance(corruptions[0], Mapping), f"{run_id} corruption layout")
        _require(corruptions[0].get("type") == corruption and corruptions[0].get("start") == 15 and corruptions[0].get("end") == EVENT_ENDS[corruption], f"{run_id} event layout")
        specs.append(RunSpec(run_id, split, corruption, directory, source, manifest))
    return tuple(specs)


def _spec_by_id(specs: Sequence[RunSpec], run_id: str) -> RunSpec:
    matches = [spec for spec in specs if spec.run_id == run_id]
    _require(len(matches) == 1, f"unknown formal run ID {run_id!r}")
    return matches[0]


def _artifact(value: Snapshot, expected: Mapping[str, Any], *, name: str) -> None:
    _require(expected.get("sha256") == _sha_bytes(value.payload), f"{name} digest mismatch")
    _require(expected.get("size_bytes") == len(value.payload), f"{name} byte-size mismatch")


def _commit_value(commit_dir: Path, specs: Sequence[RunSpec]) -> tuple[Snapshot, Mapping[str, Any]]:
    manifest = _read(commit_dir / "commitment-manifest.json", name="commitment manifest", mode=0o444)
    value = _strict_json(manifest.payload, name="commitment manifest")
    _require(value.get("schema_version") == f"{SCHEMA}.commitment.v1" and value.get("stage") == "commit", "commitment schema/stage")
    rows = value.get("run_registry")
    _require(isinstance(rows, list) and [row.get("run_id") for row in rows if isinstance(row, Mapping)] == [spec.run_id for spec in specs], "commitment run order")
    for row, spec in zip(rows, specs, strict=True):
        _require(isinstance(row, Mapping), "invalid commitment run registry row")
        _require(row.get("split") == spec.split and row.get("corruption_type") == spec.corruption, f"commitment identity mismatch for {spec.run_id}")
        artifacts = row.get("input_artifacts")
        _require(isinstance(artifacts, Mapping), f"commitment input artifacts missing for {spec.run_id}")
        _artifact(spec.source_manifest, artifacts.get("source_manifest", {}), name=f"{spec.run_id} source")
        _artifact(spec.input_manifest, artifacts.get("input_manifest", {}), name=f"{spec.run_id} input")
    return manifest, value


def commit_inputs(input_root: Path, output_dir: Path, *, protocol: Path) -> dict[str, Any]:
    """Bind code, data, fixed configuration, and the blind read lock before forwards."""

    _require(input_root.resolve(strict=True) == V2_INPUTS.resolve(strict=True), "formal v2 must use frozen inputs 0001")
    _new_output(output_dir)
    state_commit = _clean_commit(ROOT)
    recal_commit = _clean_commit(RECAL3R, expected=EXPECTED_RECAL3R_COMMIT)
    protocol_snapshot = _tracked_snapshot(protocol.resolve(strict=True), repository=ROOT, name="formal-v2 protocol")
    runner_snapshot = _tracked_snapshot(RUNNER, repository=ROOT, name="ReCal3R runner")
    orchestrator_snapshot = _tracked_snapshot(Path(__file__).resolve(), repository=ROOT, name="formal-v2 orchestrator")
    validator_snapshot = _tracked_snapshot(VALIDATOR, repository=ROOT, name="formal-v2 input validator")
    specs = _run_specs()
    v2_registry = _read(V2_INPUTS / "formal-v2-manifest.json", name="v2 input registry", mode=0o444)
    registry_value = _strict_json(v2_registry.payload, name="v2 input registry")
    _require(registry_value.get("status") == "pre_forward_blind_holdout_inputs", "v2 inputs not in blind pre-forward state")
    archive_value = registry_value.get("archive")
    acquisition_value = registry_value.get("acquisition")
    _require(isinstance(archive_value, Mapping) and isinstance(acquisition_value, Mapping), "v2 input registry archive/acquisition provenance")
    archive = _read(Path(str(archive_value.get("path", ""))), name="new-scene archive", mode=0o444)
    acquisition = _read(Path(str(acquisition_value.get("path", ""))), name="new-scene acquisition", mode=0o444)
    _require(_sha_bytes(archive.payload) == archive_value.get("sha256"), "new-scene archive hash mismatch")
    _require(_sha_bytes(acquisition.payload) == acquisition_value.get("sha256"), "new-scene acquisition hash mismatch")
    _require(_sha_file(CHECKPOINT) == CHECKPOINT_SHA256, "checkpoint digest mismatch")
    opencv = _opencv_provenance()
    search = _read(V2_SEARCH, name="development search", mode=0o444)
    readiness = _read(V2_READINESS, name="development readiness", mode=0o444)
    _require(_sha_bytes(search.payload) == "45287f2eeeee33c8c0eee9cf7586f6996589d9963683775fcb8c93e61250c129", "development search hash mismatch")
    _require(_strict_json(readiness.payload, name="development readiness").get("status") == "PASS", "development readiness is not PASS")

    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.validator.", suffix=".staging", dir=OUTPUTS))
    try:
        validation_dir = staging / "cpu-validation"
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TMPDIR": str(ROOT / "tmp")}
        subprocess.run([str(ROOT / ".venv" / "bin" / "python"), str(VALIDATOR), str(V2_INPUTS), str(validation_dir)], cwd=ROOT, env=environment, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        report = _read(validation_dir / "cpu-validation.json", name="fresh v2 CPU validation", mode=0o444)
        report_value = _strict_json(report.payload, name="fresh v2 CPU validation")
        _require(report_value.get("status") == "PASS", "fresh v2 CPU validation did not pass")
        _require(report_value.get("checks", {}).get("cuda_hidden") is True, "fresh CPU validation initialized CUDA")
        _require(report_value.get("validator", {}).get("sha256") == _sha_bytes(validator_snapshot.payload), "CPU validation did not use committed validator bytes")
    except subprocess.CalledProcessError as error:
        raise FormalV2Error(f"fresh formal-v2 CPU validation failed: {error.stderr}") from error
    finally:
        if staging.exists():
            for path in sorted(staging.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                if path.is_dir():
                    path.chmod(0o755)
            staging.chmod(0o755)
            shutil.rmtree(staging)

    registry_rows = [
        {
            "run_id": spec.run_id,
            "split": spec.split,
            "corruption_type": spec.corruption,
            "input_root": str(spec.input_root),
            "input_artifacts": {
                "source_manifest": spec.source_manifest.artifact("source-manifest.json"),
                "input_manifest": spec.input_manifest.artifact("input-manifest.json"),
            },
        }
        for spec in specs
    ]
    holdout_rows = [row for row in registry_rows if row["split"] == "holdout"]
    run_registry = {
        "schema_version": f"{SCHEMA}.run-registry.v1",
        "execution_order": [spec.run_id for spec in specs],
        "runs": registry_rows,
        "blind_read_lock": "no holdout response, response hash, or runner main-log byte may be read before all three frozen holdout outputs pass structural preflight",
    }
    holdout_commitment = {
        "schema_version": f"{SCHEMA}.holdout-commitment.v1",
        "status": "pre_forward_blind_holdout",
        "runs": holdout_rows,
        "input_registry": v2_registry.artifact("formal-v2-manifest.json"),
        "read_lock": run_registry["blind_read_lock"],
    }
    commitment = {
        "schema_version": f"{SCHEMA}.commitment.v1",
        "stage": "commit",
        "state_guard_commit": state_commit,
        "recal3r_commit": recal_commit,
        "tracked_sources": {
            "protocol": protocol_snapshot.artifact(str(protocol.resolve(strict=True))),
            "runner": runner_snapshot.artifact(str(RUNNER)),
            "orchestrator": orchestrator_snapshot.artifact(str(Path(__file__).resolve())),
            "input_validator": validator_snapshot.artifact(str(VALIDATOR)),
        },
        "checkpoint": {"path": str(CHECKPOINT), "sha256": CHECKPOINT_SHA256, "size_bytes": CHECKPOINT.stat().st_size},
        "v2_fixed_configuration": {"scoring": SCORING.to_dict(), "scale_floors": SCALE_FLOORS, "thresholds": THRESHOLDS},
        "development_evidence": {"search": search.artifact(str(V2_SEARCH)), "readiness": readiness.artifact(str(V2_READINESS))},
        "new_scene_inputs": {
            "registry": v2_registry.artifact("formal-v2-manifest.json"),
            "archive": archive.artifact(str(archive.path)),
            "acquisition": acquisition.artifact(str(acquisition.path)),
            "cpu_validation": report.artifact("cpu-validation.json"),
        },
        "production_opencv": opencv,
        "run_registry": registry_rows,
        "execution_order": run_registry["execution_order"],
        "gpu_requirement": {"minimum_free_mib": 12288, "snapshots_before_launch": 2, "single_gpu": True, "serial": True},
        "holdout_read_lock": holdout_commitment["read_lock"],
        "execution_not_before": _utc(),
    }
    registry_bytes = _json_bytes(run_registry)
    holdout_bytes = _json_bytes(holdout_commitment)
    commitment["artifacts"] = {
        "run_registry": {"path": "run-registry.json", "sha256": _sha_bytes(registry_bytes), "size_bytes": len(registry_bytes)},
        "holdout_commitment": {"path": "holdout-commitment.json", "sha256": _sha_bytes(holdout_bytes), "size_bytes": len(holdout_bytes)},
        "cpu_validation": report.artifact("cpu-validation.json"),
    }
    commitment_bytes = _json_bytes(commitment)
    output = _publish(output_dir, {"commitment-manifest.json": commitment_bytes, "run-registry.json": registry_bytes, "holdout-commitment.json": holdout_bytes, "cpu-validation.json": report.payload})
    return {"stage": "commit", "output_dir": str(output), "commitment_sha256": _sha_bytes(commitment_bytes)}


def _frozen_run_output(runs_root: Path, spec: RunSpec) -> dict[str, Snapshot]:
    directory = runs_root / spec.run_id
    metadata = os.lstat(directory)
    _require(stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), f"{spec.run_id} output is missing")
    _require(stat.S_IMODE(metadata.st_mode) == 0o555, f"{spec.run_id} output directory must be frozen 0555")
    names = {entry.name for entry in os.scandir(directory)}
    _require(names == set(RUN_FILES.values()), f"{spec.run_id} output must contain exactly five runner artifacts")
    return {key: _read(directory / filename, name=f"{spec.run_id} {filename}", mode=0o444) for key, filename in RUN_FILES.items()}


def _snapshot_outputs(runs_root: Path, specs: Sequence[RunSpec]) -> dict[str, dict[str, Snapshot]]:
    _require(runs_root.is_dir() and not runs_root.is_symlink(), "formal runs root is missing")
    return {spec.run_id: _frozen_run_output(runs_root, spec) for spec in specs}


def _records_root(runs_root: Path) -> Path:
    return runs_root.with_name(runs_root.name + "-records")


def _frozen_launch_record(runs_root: Path, spec: RunSpec) -> Snapshot:
    record_root = _records_root(runs_root) / spec.run_id
    _require(record_root.is_dir() and not record_root.is_symlink(), f"{spec.run_id} launch record is missing")
    _require(stat.S_IMODE(os.lstat(record_root).st_mode) == 0o555, f"{spec.run_id} launch record is not frozen")
    _require({entry.name for entry in os.scandir(record_root)} == {"launch.json"}, f"{spec.run_id} launch record layout")
    snapshot = _read(record_root / "launch.json", name=f"{spec.run_id} launch record", mode=0o444)
    value = _strict_json(snapshot.payload, name=f"{spec.run_id} launch record")
    _require(value.get("run_id") == spec.run_id and value.get("split") == spec.split and value.get("exit_code") == 0, f"{spec.run_id} launch record identity/status")
    snapshots = value.get("prelaunch_snapshots")
    _require(isinstance(snapshots, list) and len(snapshots) == 2, f"{spec.run_id} lacks two prelaunch snapshots")
    for row in snapshots:
        _require(isinstance(row, Mapping) and isinstance(row.get("free_mib"), int) and row["free_mib"] >= 12288, f"{spec.run_id} launch GPU capacity record")
    log_path = Path(str(value.get("log_path", "")))
    _require(log_path.is_file() and not log_path.is_symlink() and stat.S_IMODE(os.lstat(log_path).st_mode) == 0o444, f"{spec.run_id} launch log is not frozen")
    return snapshot


def _launch_artifacts(runs_root: Path, specs: Sequence[RunSpec]) -> dict[str, Any]:
    return {spec.run_id: _frozen_launch_record(runs_root, spec).artifact(f"{spec.run_id}/launch.json") for spec in specs}


def _health(snapshot: Snapshot, *, run_id: str) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(snapshot.payload.splitlines(), start=1):
        if line.strip():
            value = _strict_json(line, name=f"{run_id} health line {line_number}")
            records.append(value)
    _require(len(records) == 30, f"{run_id} health ledger must contain 30 records")
    try:
        validate_online_overlap(records)
    except DetectorV2Error as error:
        raise FormalV2Error(f"{run_id} invalid online overlap: {error}") from error
    forbidden = {"groundtruth", "depth", "label", "event_start", "corruption_type"}
    for record in records:
        _require(not (forbidden & set(record)), f"{run_id} health ledger contains non-online provenance")
    return records


def _validate_run_json(snapshot: Snapshot, *, spec: RunSpec, commitment: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _strict_json(snapshot.payload, name=f"{spec.run_id} run.json")
    _require(value.get("status") == "succeeded" and value.get("health_profile") == "v2", f"{spec.run_id} is not a successful v2 run")
    _require(value.get("frame_count") == 30 and value.get("calibrated_update_calls") == 29, f"{spec.run_id} frame/update count")
    _require(value.get("checkpoint_sha256") == CHECKPOINT_SHA256, f"{spec.run_id} checkpoint mismatch")
    _require(value.get("baseline_commit") == EXPECTED_RECAL3R_COMMIT and value.get("baseline_tracked_worktree_clean") is True, f"{spec.run_id} baseline provenance")
    runner = value.get("runner")
    _require(isinstance(runner, Mapping), f"{spec.run_id} runner provenance missing")
    expected_runner = commitment["tracked_sources"]["runner"]
    _require(runner.get("commit") == commitment["state_guard_commit"] and runner.get("script_sha256") == expected_runner["sha256"], f"{spec.run_id} runner provenance mismatch")
    visual = value.get("online_visual_correspondence")
    _require(isinstance(visual, Mapping), f"{spec.run_id} visual provenance missing")
    _require(visual.get("input") == "post_official_loader_post_deferred_transform_normalized_rgb", f"{spec.run_id} visual input semantics")
    rows = visual.get("frame_results")
    _require(isinstance(rows, list) and len(rows) == 29, f"{spec.run_id} visual frame provenance")
    _require("groundtruth" not in visual and "depth" not in visual and "label" not in visual, f"{spec.run_id} visual provenance leaks GT/label")
    for name in ("runtime_seconds", "peak_memory_allocated_mib"):
        value_number = value.get(name)
        _require(isinstance(value_number, (int, float)) and not isinstance(value_number, bool) and math.isfinite(float(value_number)) and float(value_number) > 0, f"{spec.run_id} invalid {name}")
    return value


def _event_labels(spec: RunSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.zeros(30, dtype=np.int64)
    labels[15 : EVENT_ENDS[spec.corruption] + 1] = 1
    primary = np.ones(30, dtype=bool)
    primary[EVENT_ENDS[spec.corruption] + 1 : EVENT_ENDS[spec.corruption] + 6] = False
    prefix = np.zeros(30, dtype=bool)
    prefix[:15] = True
    return labels, primary, prefix


def _streak(labels: np.ndarray, scores: np.ndarray, mask: np.ndarray, threshold: float) -> int:
    active = mask & (labels == 0) & (scores >= threshold)
    current = longest = 0
    for item in active:
        if bool(item):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _score_split(specs: Sequence[RunSpec], outputs: Mapping[str, Mapping[str, Snapshot]]) -> dict[str, Any]:
    records_by_run = {spec.run_id: _health(outputs[spec.run_id]["health_jsonl"], run_id=spec.run_id) for spec in specs}
    labels = {spec.run_id: _event_labels(spec)[0] for spec in specs}
    masks = {spec.run_id: _event_labels(spec)[1] for spec in specs}
    prefixes = {spec.run_id: _event_labels(spec)[2] for spec in specs}
    scores: dict[str, dict[str, Any]] = {}
    for spec in specs:
        seed = detection_suite.derive_run_seed(SCORING.master_seed, spec.split, spec.run_id)
        scores[spec.run_id] = compute_v2_scores(records_by_run[spec.run_id], config=SCORING, scale_floors=SCALE_FLOORS, seed=seed, require_online_overlap=True)
    methods: dict[str, Any] = {}
    for method_name, threshold in THRESHOLDS.items():
        per_run: dict[str, Any] = {}
        pooled_labels: list[np.ndarray] = []
        pooled_scores: list[np.ndarray] = []
        prefix_labels: list[np.ndarray] = []
        prefix_scores: list[np.ndarray] = []
        aurocs: list[float] = []
        detected: set[str] = set()
        delays: list[int] = []
        max_streak = 0
        max_startup_streak = 0
        for spec in specs:
            payload = scores[spec.run_id]["methods"][method_name]
            values = np.asarray(payload["scores"], dtype=np.float64)
            _require(values.shape == (30,) and bool(np.isfinite(values).all()), f"{spec.run_id} {method_name} has non-finite score")
            metric = detection.threshold_metrics(labels[spec.run_id][masks[spec.run_id]], values[masks[spec.run_id]], threshold)
            delay = detection.detection_delay(labels[spec.run_id], values, threshold)
            auroc = detection.auroc(labels[spec.run_id][masks[spec.run_id]], values[masks[spec.run_id]])
            _require(auroc is not None, f"{spec.run_id} {method_name} AUROC unavailable")
            event_detected = int(delay["detected_intervals"]) > 0
            if event_detected:
                detected.add(spec.corruption)
            delays.extend(int(row["delay_frames"]) for row in delay["intervals"] if row["delay_frames"] is not None)
            run_streak = _streak(labels[spec.run_id], values, masks[spec.run_id], threshold)
            startup_streak = _streak(labels[spec.run_id], values, prefixes[spec.run_id], threshold)
            max_streak = max(max_streak, run_streak)
            max_startup_streak = max(max_startup_streak, startup_streak)
            per_run[spec.run_id] = {
                "scores": values.tolist(),
                "threshold": threshold,
                "primary_metrics": metric,
                "primary_auroc": auroc,
                "delay": delay,
                "detected": event_detected,
                "max_primary_false_positive_streak": run_streak,
                "max_startup_false_positive_streak": startup_streak,
                "used_signals": payload["used_signals"],
                "signal_sources": payload["signal_sources"],
                "finite_component_count": np.asarray(payload["finite_component_count"], dtype=np.int64).tolist(),
            }
            pooled_labels.append(labels[spec.run_id][masks[spec.run_id]])
            pooled_scores.append(values[masks[spec.run_id]])
            prefix_labels.append(labels[spec.run_id][prefixes[spec.run_id]])
            prefix_scores.append(values[prefixes[spec.run_id]])
            aurocs.append(float(auroc))
        methods[method_name] = {
            "threshold": threshold,
            "per_run": per_run,
            "macro_auroc": float(np.mean(aurocs)),
            "pooled": detection.threshold_metrics(np.concatenate(pooled_labels), np.concatenate(pooled_scores), threshold),
            "clean_prefix": detection.threshold_metrics(np.concatenate(prefix_labels), np.concatenate(prefix_scores), threshold),
            "events": {"detected_corruption_types": sorted(detected), "detected_events": len(detected), "mean_delay_frames": float(np.mean(delays)) if delays else None},
            "diagnostics": {"max_per_run_consecutive_false_positives": max_streak, "startup_max_streak": max_startup_streak},
        }
    return {
        "schema_version": f"{SCHEMA}.metrics.v1",
        "scoring": SCORING.to_dict(),
        "scale_floors": SCALE_FLOORS,
        "thresholds": THRESHOLDS,
        "runs": [{"run_id": spec.run_id, "split": spec.split, "corruption_type": spec.corruption, "primary_labels": _event_labels(spec)[0].tolist(), "primary_mask": _event_labels(spec)[1].tolist(), "clean_prefix_mask": _event_labels(spec)[2].tolist()} for spec in specs],
        "methods": methods,
    }


def _run_artifacts(outputs: Mapping[str, Mapping[str, Snapshot]]) -> dict[str, Any]:
    return {run_id: {kind: snapshot.artifact(f"{run_id}/{RUN_FILES[kind]}") for kind, snapshot in rows.items()} for run_id, rows in outputs.items()}


def _load_calibration(calibration_dir: Path, commit: Snapshot) -> tuple[Snapshot, Mapping[str, Any], Mapping[str, Any]]:
    manifest = _read(calibration_dir / "calibration-manifest.json", name="calibration manifest", mode=0o444)
    value = _strict_json(manifest.payload, name="calibration manifest")
    _require(value.get("schema_version") == f"{SCHEMA}.calibration.v1" and value.get("stage") == "calibrate", "calibration schema/stage")
    _require(value.get("commitment", {}).get("sha256") == _sha_bytes(commit.payload), "calibration commitment binding")
    metrics = _read(calibration_dir / "development-metrics.json", name="development metrics", mode=0o444)
    _artifact(metrics, value.get("artifacts", {}).get("development_metrics", {}), name="calibration metrics")
    unlock_dir = calibration_dir.with_name(calibration_dir.name + "-holdout-unlock")
    unlock = _read(unlock_dir / "holdout-unlock.json", name="holdout unlock", mode=0o444)
    unlock_value = _strict_json(unlock.payload, name="holdout unlock")
    _require(unlock_value.get("status") == "holdout-forwards-permitted", "holdout unlock status")
    _require(unlock_value.get("calibration_manifest", {}).get("sha256") == _sha_bytes(manifest.payload), "holdout unlock calibration binding")
    return manifest, value, _strict_json(metrics.payload, name="development metrics")


def calibrate_development(input_root: Path, commit_dir: Path, runs_root: Path, output_dir: Path) -> dict[str, Any]:
    """Read the six disclosed runs only, recompute fixed metrics, then unlock holdout."""

    _require(input_root.resolve(strict=True) == V2_INPUTS.resolve(strict=True), "calibration inputs mismatch")
    _new_output(output_dir)
    specs = _run_specs()
    commitment, commit_value = _commit_value(commit_dir.resolve(strict=True), specs)
    development = tuple(spec for spec in specs if spec.split == "development")
    # No holdout spec/path is passed to this snapshot function; calibration has no blind-output read capability.
    outputs = _snapshot_outputs(runs_root.resolve(strict=True), development)
    launch_artifacts = _launch_artifacts(runs_root.resolve(strict=True), development)
    run_json = {spec.run_id: _validate_run_json(outputs[spec.run_id]["run_json"], spec=spec, commitment=commit_value) for spec in development}
    metrics = _score_split(development, outputs)
    metric_bytes = _json_bytes(metrics)
    config = {"schema_version": f"{SCHEMA}.formal-config.v1", "scoring": SCORING.to_dict(), "scale_floors": SCALE_FLOORS, "thresholds": THRESHOLDS, "selection": "fixed before new-scene response; calibration recomputes only"}
    config_bytes = _json_bytes(config)
    manifest = {
        "schema_version": f"{SCHEMA}.calibration.v1",
        "stage": "calibrate",
        "commitment": commitment.artifact("commitment-manifest.json"),
        "development_run_artifacts": _run_artifacts(outputs),
        "development_launch_artifacts": launch_artifacts,
        "artifacts": {"development_metrics": {"path": "development-metrics.json", "sha256": _sha_bytes(metric_bytes), "size_bytes": len(metric_bytes)}, "formal_config": {"path": "formal-config.json", "sha256": _sha_bytes(config_bytes), "size_bytes": len(config_bytes)}},
        "runtime_provenance_checked": {run_id: {"started_at": value.get("started_at"), "finished_at": value.get("finished_at")} for run_id, value in run_json.items()},
    }
    manifest_bytes = _json_bytes(manifest)
    output = _publish(output_dir, {"development-metrics.json": metric_bytes, "formal-config.json": config_bytes, "calibration-manifest.json": manifest_bytes})
    unlock = {"schema_version": f"{SCHEMA}.holdout-unlock.v1", "calibration_manifest": {"path": str(output / "calibration-manifest.json"), "sha256": _sha_bytes(manifest_bytes), "size_bytes": len(manifest_bytes)}, "unlocked_at": _utc(), "status": "holdout-forwards-permitted"}
    # Atomic sibling publication gives a holdout launcher one immutable predicate.
    unlock_dir = _new_output(output.with_name(output.name + "-holdout-unlock"))
    _publish(unlock_dir, {"holdout-unlock.json": _json_bytes(unlock)})
    return {"stage": "calibrate", "output_dir": str(output), "holdout_unlock": str(unlock_dir / "holdout-unlock.json"), "calibration_manifest_sha256": _sha_bytes(manifest_bytes)}


def _preflight_holdout_outputs(runs_root: Path, specs: Sequence[RunSpec]) -> None:
    """Metadata-only blind preflight. Do not read a response or log byte here."""

    problems: list[str] = []
    for spec in specs:
        try:
            _metadata_only_frozen_run(runs_root, spec)
        except OSError as error:
            problems.append(f"{spec.run_id}: {error}")
        except FormalV2Error as error:
            problems.append(str(error))
    _require(not problems, "all three blind outputs must finish and freeze before any response/log read: " + " | ".join(problems))


def _metadata_only_frozen_run(runs_root: Path, spec: RunSpec) -> None:
    """Validate an output tree with lstat/scandir only; never open its bytes."""

    directory = runs_root / spec.run_id
    metadata = os.lstat(directory)
    _require(stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) == 0o555, f"{spec.run_id} output is not frozen")
    names = {entry.name for entry in os.scandir(directory)}
    _require(names == set(RUN_FILES.values()), f"{spec.run_id} output artifact layout")
    for filename in RUN_FILES.values():
        child = os.lstat(directory / filename)
        _require(stat.S_ISREG(child.st_mode) and not stat.S_ISLNK(child.st_mode) and stat.S_IMODE(child.st_mode) == 0o444, f"{spec.run_id} {filename} is not frozen")


def _go_no_go(development: Mapping[str, Any], holdout: Mapping[str, Any], *, provenance: bool) -> dict[str, Any]:
    methods = holdout["methods"]
    combined = float(methods["combined"]["macro_auroc"])
    random = float(methods["random"]["macro_auroc"])
    best_single = max(float(methods["update_magnitude_only"]["macro_auroc"]), float(methods["reliability_only"]["macro_auroc"]))
    development_detected = set(development["methods"]["combined"]["events"]["detected_corruption_types"])
    holdout_detected = set(methods["combined"]["events"]["detected_corruption_types"])
    combined_runs = methods["combined"]["per_run"]
    overlap_used = all("overlap" in row["used_signals"] and all(count >= 1 for count in row["finite_component_count"][1:]) for row in combined_runs.values())
    checks = [
        {"id": "combined_macro_auroc_strictly_above_0_75", "passed": combined > 0.75, "actual": combined, "operator": ">", "threshold": 0.75},
        {"id": "combined_minus_random_at_least_0_10", "passed": combined - random >= 0.10, "actual": combined - random, "operator": ">=", "threshold": 0.10},
        {"id": "combined_within_0_02_of_best_single_signal", "passed": combined >= best_single - 0.02, "actual": combined, "best_single": best_single},
        {"id": "at_least_two_shared_detected_types", "passed": len(development_detected & holdout_detected) >= 2, "development": sorted(development_detected), "holdout": sorted(holdout_detected), "shared": sorted(development_detected & holdout_detected)},
        {"id": "pooled_primary_fpr_at_most_0_20", "passed": float(methods["combined"]["pooled"]["false_positive_rate"]) <= 0.20, "actual": methods["combined"]["pooled"]["false_positive_rate"]},
        {"id": "max_primary_fp_streak_at_most_3", "passed": int(methods["combined"]["diagnostics"]["max_per_run_consecutive_false_positives"]) <= 3, "actual": methods["combined"]["diagnostics"]["max_per_run_consecutive_false_positives"]},
        {"id": "provenance_runtime_replay_and_real_signals", "passed": provenance, "actual": provenance},
        {"id": "finite_online_overlap_combined_usage_no_gt_label_leakage", "passed": overlap_used, "actual": overlap_used},
        {"id": "clean_prefix_fpr_at_most_0_15_and_startup_streak_at_most_2", "passed": float(methods["combined"]["clean_prefix"]["false_positive_rate"]) <= 0.15 and int(methods["combined"]["diagnostics"]["startup_max_streak"]) <= 2, "clean_prefix_fpr": methods["combined"]["clean_prefix"]["false_positive_rate"], "startup_max_streak": methods["combined"]["diagnostics"]["startup_max_streak"]},
    ]
    passed = all(item["passed"] is True for item in checks)
    return {"schema_version": f"{SCHEMA}.go-no-go.v1", "decision": "GO" if passed else "NO-GO", "passed": passed, "checks": checks, "next_step": "separately preregistered minimum quarantine pilot" if passed else "stop before quarantine or rollback and publish failure report"}


def evaluate_holdout(input_root: Path, commit_dir: Path, calibration_dir: Path, runs_root: Path, output_dir: Path) -> dict[str, Any]:
    """Perform the unique v2 blind evaluation after structural all-three preflight."""

    _require(input_root.resolve(strict=True) == V2_INPUTS.resolve(strict=True), "evaluation inputs mismatch")
    _new_output(output_dir)
    specs = _run_specs()
    holdout = tuple(spec for spec in specs if spec.split == "holdout")
    # This call uses lstat/scandir only. It must remain before every blind byte read.
    _preflight_holdout_outputs(runs_root.resolve(strict=True), holdout)
    commitment, commit_value = _commit_value(commit_dir.resolve(strict=True), specs)
    calibration_manifest, calibration_value, development_metrics = _load_calibration(calibration_dir.resolve(strict=True), commitment)
    holdout_outputs = _snapshot_outputs(runs_root.resolve(strict=True), holdout)
    development_specs = tuple(spec for spec in specs if spec.split == "development")
    development_outputs = _snapshot_outputs(runs_root.resolve(strict=True), development_specs)
    all_outputs = {**development_outputs, **holdout_outputs}
    launch_artifacts = _launch_artifacts(runs_root.resolve(strict=True), specs)
    _freeze_tree(_records_root(runs_root.resolve(strict=True)))
    for spec in specs:
        _validate_run_json(all_outputs[spec.run_id]["run_json"], spec=spec, commitment=commit_value)
    recomputed_development = _score_split(development_specs, development_outputs)
    _require(_json_bytes(recomputed_development) == _json_bytes(development_metrics), "calibration development metrics do not replay byte-identically")
    holdout_metrics = _score_split(holdout, holdout_outputs)
    provenance = True
    gate = _go_no_go(development_metrics, holdout_metrics, provenance=provenance)
    metric_bytes = _json_bytes(holdout_metrics)
    gate_bytes = _json_bytes(gate)
    timelines = {f"timelines/{spec.run_id}.json": _json_bytes({"run_id": spec.run_id, "corruption_type": spec.corruption, "scores": holdout_metrics["methods"]["combined"]["per_run"][spec.run_id]["scores"], "labels": _event_labels(spec)[0].tolist()}) for spec in holdout}
    manifest = {
        "schema_version": f"{SCHEMA}.evaluation.v1",
        "stage": "evaluate",
        "decision": gate["decision"],
        "inputs": {"commitment": commitment.artifact("commitment-manifest.json"), "calibration": calibration_manifest.artifact("calibration-manifest.json")},
        "run_artifacts": _run_artifacts(all_outputs),
        "launch_artifacts": launch_artifacts,
        "artifacts": {"holdout_metrics": {"path": "holdout-metrics.json", "sha256": _sha_bytes(metric_bytes), "size_bytes": len(metric_bytes)}, "go_no_go": {"path": "go-no-go.json", "sha256": _sha_bytes(gate_bytes), "size_bytes": len(gate_bytes)}},
        "blind_preflight": "all three frozen holdout layouts were inspected by metadata before any holdout output bytes were read",
        "provenance_checks_passed": provenance,
    }
    manifest_bytes = _json_bytes(manifest)
    output = _publish(output_dir, {"holdout-metrics.json": metric_bytes, "go-no-go.json": gate_bytes, "evaluation-manifest.json": manifest_bytes, **timelines})
    return {"stage": "evaluate", "output_dir": str(output), "decision": gate["decision"], "evaluation_manifest_sha256": _sha_bytes(manifest_bytes)}


def _gpu_snapshot(gpu_id: int) -> dict[str, Any]:
    query = subprocess.run(["nvidia-smi", f"--id={gpu_id}", "--query-gpu=index,uuid,memory.free,memory.total", "--format=csv,noheader,nounits"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pieces = [piece.strip() for piece in query.stdout.strip().split(",")]
    _require(len(pieces) == 4 and pieces[0] == str(gpu_id), f"cannot parse GPU {gpu_id} memory query")
    try:
        free_mib = int(pieces[2])
    except ValueError as error:
        raise FormalV2Error(f"cannot parse GPU free memory: {query.stdout!r}") from error
    full = subprocess.run(["nvidia-smi", f"--id={gpu_id}"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    process = subprocess.run(["ps", "-eo", "pid=,user=,args="], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    gpustat = shutil.which("gpustat")
    gpustat_output = None
    if gpustat is not None:
        gpustat_output = subprocess.run([gpustat], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
    return {"captured_at": _utc(), "gpu_id": gpu_id, "gpu_uuid": pieces[1], "free_mib": free_mib, "total_mib": int(pieces[3]), "nvidia_smi": full.stdout, "gpustat": gpustat_output, "processes": process.stdout}


def launch_forward(commit_dir: Path, calibration_dir: Path | None, runs_root: Path, records_root: Path, log_dir: Path, run_id: str, gpu_id: int) -> dict[str, Any]:
    """Launch one serial formal run after two safe shared-GPU snapshots."""

    specs = _run_specs()
    commitment, commit_value = _commit_value(commit_dir.resolve(strict=True), specs)
    spec = _spec_by_id(specs, run_id)
    if spec.split == "holdout":
        _require(calibration_dir is not None, "holdout launch requires calibration directory")
        _load_calibration(calibration_dir.resolve(strict=True), commitment)
    else:
        preceding = [item for item in specs if item.split == "development"]
        index = preceding.index(spec)
        for earlier in preceding[:index]:
            _frozen_run_output(runs_root, earlier)
    _require(runs_root.parent.resolve(strict=False) == OUTPUTS and records_root.resolve(strict=False) == _records_root(runs_root).resolve(strict=False), "runs and records roots must use the fixed direct-output layout")
    _require(log_dir.parent.resolve(strict=False) == LOGS, "log directory must be a direct logs child")
    runs_root.mkdir(exist_ok=True)
    records_root.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    _require(not (runs_root / run_id).exists() and not (records_root / run_id).exists() and not (log_dir / f"{run_id}.log").exists(), f"formal run {run_id} already has an output, record, or log")
    before_one = _gpu_snapshot(gpu_id)
    before_two = _gpu_snapshot(gpu_id)
    _require(before_one["free_mib"] >= 12288 and before_two["free_mib"] >= 12288, f"GPU {gpu_id} has insufficient free memory in pre-launch snapshots")
    command = [str(RECAL3R_PYTHON), str(RUNNER), "--baseline-root", str(RECAL3R), "--checkpoint", str(CHECKPOINT), "--checkpoint-sha256", CHECKPOINT_SHA256, "--input-manifest", str(spec.input_manifest.path), "--output-dir", str(runs_root / run_id), "--device", "cuda", "--size", "512", "--seed", "0", "--beta-base", "0.1", "--health-profile", "v2"]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id), "TMPDIR": str(ROOT / "tmp")}
    started = _utc()
    with (log_dir / f"{run_id}.log").open("xb") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, text=False)
        status = process.wait()
    finished = _utc()
    after = _gpu_snapshot(gpu_id)
    record = {"schema_version": f"{SCHEMA}.launch.v1", "run_id": run_id, "split": spec.split, "command": command, "pid": process.pid, "gpu": {"id": gpu_id, "uuid": before_two["gpu_uuid"]}, "prelaunch_snapshots": [before_one, before_two], "postexit_snapshot": after, "minimum_free_mib": 12288, "started_at": started, "finished_at": finished, "exit_code": status, "log_path": str(log_dir / f"{run_id}.log"), "output_dir": str(runs_root / run_id)}
    _require(status == 0, f"formal runner failed for {run_id}; preserved log/output at {log_dir / f'{run_id}.log'}")
    _freeze_tree(runs_root / run_id)
    if spec.split == "holdout":
        # The read lock forbids even hashing/parsing one blind response until all
        # three sibling runs are frozen.  Only metadata inspection is allowed.
        _metadata_only_frozen_run(runs_root, spec)
    else:
        _frozen_run_output(runs_root, spec)
    record_dir = records_root / run_id
    record_dir.mkdir()
    (record_dir / "launch.json").write_bytes(_json_bytes(record))
    _freeze_tree(record_dir)
    (log_dir / f"{run_id}.log").chmod(0o444)
    return {"stage": "launch", "run_id": run_id, "pid": process.pid, "output_dir": str(runs_root / run_id), "log_path": str(log_dir / f"{run_id}.log")}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    commit = actions.add_parser("commit")
    commit.add_argument("input_root", type=Path)
    commit.add_argument("output_dir", type=Path)
    commit.add_argument("--protocol", type=Path, required=True)
    calibrate = actions.add_parser("calibrate")
    calibrate.add_argument("input_root", type=Path)
    calibrate.add_argument("commit_dir", type=Path)
    calibrate.add_argument("runs_root", type=Path)
    calibrate.add_argument("output_dir", type=Path)
    evaluate = actions.add_parser("evaluate")
    evaluate.add_argument("input_root", type=Path)
    evaluate.add_argument("commit_dir", type=Path)
    evaluate.add_argument("calibration_dir", type=Path)
    evaluate.add_argument("runs_root", type=Path)
    evaluate.add_argument("output_dir", type=Path)
    launch = actions.add_parser("launch")
    launch.add_argument("commit_dir", type=Path)
    launch.add_argument("runs_root", type=Path)
    launch.add_argument("records_root", type=Path)
    launch.add_argument("log_dir", type=Path)
    launch.add_argument("run_id")
    launch.add_argument("--gpu-id", required=True, type=int)
    launch.add_argument("--calibration-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "commit":
            result = commit_inputs(args.input_root, args.output_dir, protocol=args.protocol)
        elif args.command == "calibrate":
            result = calibrate_development(args.input_root, args.commit_dir, args.runs_root, args.output_dir)
        elif args.command == "evaluate":
            result = evaluate_holdout(args.input_root, args.commit_dir, args.calibration_dir, args.runs_root, args.output_dir)
        else:
            result = launch_forward(args.commit_dir, args.calibration_dir, args.runs_root, args.records_root, args.log_dir, args.run_id, args.gpu_id)
    except (FormalV2Error, DetectorV2Error, OSError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
