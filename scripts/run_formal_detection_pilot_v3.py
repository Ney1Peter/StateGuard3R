#!/usr/bin/env python3
"""One-shot formal Detector v3 orchestration with a structural blind read lock.

This is intentionally a self-contained v3 control plane.  It does not import
the v2 orchestrator: v2's evidence, source schema, and formal decision remain
immutable.  The only v2 material consumed here is the disclosed, frozen
continuous-detector configuration bound below by byte hash.
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
    DetectorV2Error,
    V2ScoringConfig,
    compute_v2_scores,
    validate_online_overlap,
)
from stateguard3r.detection_v3 import DetectorV3Error, hybridize_v2_continuous_scores
from stateguard3r.timestamp_order_v3 import (
    TimestampOrderError,
    validate_timestamp_order_sidecar,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
LOGS = ROOT / "logs"
RECAL3R = ROOT.parent / "baselines" / "ReCal3R"
RECAL3R_PYTHON = RECAL3R / ".venv" / "bin" / "python"
RUNNER = ROOT / "scripts" / "run_recal3r_smoke.py"
VALIDATOR = ROOT / "scripts" / "validate_formal_detection_v3_inputs.py"
PROTOCOL = ROOT / "docs" / "protocols" / "detection-formal-v3.md"
TIMESTAMP_MODULE = ROOT / "src" / "stateguard3r" / "timestamp_order_v3.py"
HYBRID_MODULE = ROOT / "src" / "stateguard3r" / "detection_v3.py"

V1_INPUTS = OUTPUTS / "formal-v1-inputs-0001"
V3_INPUTS = OUTPUTS / "formal-v3-inputs-0001"
V2_CALIBRATION = OUTPUTS / "formal-v2-calibration-0002"
V3_DEVELOPMENT = OUTPUTS / "detector-v3-derived-development-0002" / "development.json"
V1_TIMESTAMP_ROOT = RECAL3R / "data" / "tum" / "rgbd_dataset_freiburg1_desk"
V3_TIMESTAMP_ROOT = RECAL3R / "data" / "tum" / "rgbd_dataset_freiburg2_desk"
CHECKPOINT = RECAL3R / "src" / "cut3r_512_dpt_4_64.pth"
CHECKPOINT_SHA256 = "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103"
EXPECTED_RECAL3R_COMMIT = "466c7cdf3acd2f589f1d82e5f6391966f19db9ff"
V2_FORMAL_CONFIG_SHA256 = "dc5ffe45549915b6b276856656222230845cfde3a576178117f5a3501f288480"

# The six v1 inputs are disclosed development data.  The three v3 input
# directories are the sole blind responses and deliberately use distinct IDs.
RUNS = (
    ("development-dynamic", "development", "dynamic_occlusion", V1_INPUTS / "development" / "development-dynamic", V1_TIMESTAMP_ROOT),
    ("development-wrong", "development", "wrong_order_segment", V1_INPUTS / "development" / "development-wrong", V1_TIMESTAMP_ROOT),
    ("development-low", "development", "low_overlap_jump", V1_INPUTS / "development" / "development-low", V1_TIMESTAMP_ROOT),
    ("development2-dynamic", "development", "dynamic_occlusion", V1_INPUTS / "holdout" / "holdout-dynamic", V1_TIMESTAMP_ROOT),
    ("development2-wrong", "development", "wrong_order_segment", V1_INPUTS / "holdout" / "holdout-wrong", V1_TIMESTAMP_ROOT),
    ("development2-low", "development", "low_overlap_jump", V1_INPUTS / "holdout" / "holdout-low", V1_TIMESTAMP_ROOT),
    ("blind-dynamic", "holdout", "dynamic_occlusion", V3_INPUTS / "blind-dynamic", V3_TIMESTAMP_ROOT),
    ("blind-wrong-order", "holdout", "wrong_order_segment", V3_INPUTS / "blind-wrong-order", V3_TIMESTAMP_ROOT),
    ("blind-low-overlap", "holdout", "low_overlap_jump", V3_INPUTS / "blind-low-overlap", V3_TIMESTAMP_ROOT),
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
RUN_FILES = {
    "checkpoint_load_audit": "checkpoint-load-audit.json",
    "health_jsonl": "health.jsonl",
    "predictions_summary": "predictions-summary.json",
    "trajectory": "trajectory.json",
    "timestamp_order": "timestamp-order.json",
    "run_json": "run.json",
}
SCHEMA = "stateguard3r.formal-detection-v3"
MINIMUM_FREE_MIB = 12288


class FormalV3Error(ValueError):
    """A requirement for valid Detector v3 formal evidence failed."""


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
    timestamp_listing: Snapshot
    timestamp_root: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalV3Error(message)


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
        raise FormalV3Error(f"cannot encode strict JSON: {error}") from error


def _strict_json(payload: bytes, *, name: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise FormalV3Error(f"{name} contains non-finite JSON {value!r}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FormalV3Error(f"{name} has duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, parse_constant=reject_constant, object_pairs_hook=reject_duplicate)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalV3Error(f"{name} is not strict UTF-8 JSON: {error}") from error
    _require(isinstance(value, Mapping), f"{name} must be a JSON object")
    return value


def _read(path: Path, *, name: str, mode: int | None = None) -> Snapshot:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise FormalV3Error(f"cannot inspect {name}: {error}") from error
    _require(stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), f"{name} must be a regular non-symlink file")
    if mode is not None:
        _require(stat.S_IMODE(metadata.st_mode) == mode, f"{name} must have mode {mode:04o}")
    try:
        return Snapshot(path, path.read_bytes())
    except OSError as error:
        raise FormalV3Error(f"cannot read {name}: {error}") from error


def _inside(parent: Path, child: Path) -> bool:
    try:
        child.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def _new_output(path: Path) -> Path:
    output = path.resolve(strict=False)
    _require(output.parent == OUTPUTS, "formal v3 output must be a new direct child of outputs")
    _require(not output.exists(), f"formal v3 output already exists: {output}")
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
            _require(_inside(staging, target), "unsafe formal v3 artifact path")
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
        raise FormalV3Error(f"cannot query repository {repository}: {error}") from error
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
        committed = subprocess.run(["git", "-C", str(repository), "show", f"HEAD:{relative}"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise FormalV3Error(f"cannot read tracked bytes for {name}: {error}") from error
    _require(snapshot.payload == committed, f"{name} bytes differ from tracked HEAD blob")
    return snapshot


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _aware_time(value: Any, *, name: str) -> datetime:
    _require(isinstance(value, str), f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise FormalV3Error(f"{name} is not ISO-8601: {value!r}") from error
    _require(parsed.tzinfo is not None and parsed.utcoffset() is not None, f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _opencv_provenance() -> Mapping[str, Any]:
    """Capture OpenCV provenance in a CUDA-hidden subprocess."""

    code = (
        "import json,sys; "
        f"sys.path.insert(0,{str(ROOT / 'src')!r}); "
        "from stateguard3r.visual_overlap import opencv_runtime_provenance; "
        "print(json.dumps(opencv_runtime_provenance(),sort_keys=True,allow_nan=False))"
    )
    try:
        completed = subprocess.run([str(RECAL3R_PYTHON), "-c", code], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "TMPDIR": str(ROOT / "tmp")})
    except (OSError, subprocess.CalledProcessError) as error:
        raise FormalV3Error(f"cannot capture production OpenCV provenance: {error}") from error
    value = _strict_json(completed.stdout.encode("utf-8"), name="production OpenCV provenance")
    _require(value.get("threads") == 1 and value.get("opencl_enabled") is False and value.get("rng_seed") == 0, "production OpenCV determinism provenance")
    for name in ("version", "module_sha256", "binary_sha256", "build_info_sha256"):
        _require(isinstance(value.get(name), str) and value.get(name), f"production OpenCV {name} missing")
    return value


def _run_specs() -> tuple[RunSpec, ...]:
    specs: list[RunSpec] = []
    for run_id, split, corruption, directory, timestamp_root in RUNS:
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
        listing = timestamp_root / "rgb.txt"
        listing_snapshot = _read(listing, name=f"{run_id} raw RGB timestamp listing")
        _require(stat.S_IMODE(os.lstat(listing).st_mode) & 0o222 == 0, f"{run_id} raw RGB timestamp listing is writable")
        _require(timestamp_root.is_dir() and not timestamp_root.is_symlink(), f"{run_id} timestamp dataset root")
        specs.append(RunSpec(run_id, split, corruption, directory, source, manifest, listing_snapshot, timestamp_root))
    return tuple(specs)


def _spec_by_id(specs: Sequence[RunSpec], run_id: str) -> RunSpec:
    matches = [spec for spec in specs if spec.run_id == run_id]
    _require(len(matches) == 1, f"unknown formal v3 run ID {run_id!r}")
    return matches[0]


def _artifact(value: Snapshot, expected: Mapping[str, Any], *, name: str) -> None:
    _require(expected.get("sha256") == _sha_bytes(value.payload), f"{name} digest mismatch")
    _require(expected.get("size_bytes") == len(value.payload), f"{name} byte-size mismatch")


def _v2_fixed_config() -> tuple[Snapshot, Snapshot, Mapping[str, Any]]:
    config = _read(V2_CALIBRATION / "formal-config.json", name="frozen Detector v2 formal configuration", mode=0o444)
    manifest = _read(V2_CALIBRATION / "calibration-manifest.json", name="frozen Detector v2 calibration manifest", mode=0o444)
    _require(_sha_bytes(config.payload) == V2_FORMAL_CONFIG_SHA256, "frozen Detector v2 formal configuration hash mismatch")
    value = _strict_json(config.payload, name="frozen Detector v2 formal configuration")
    _require(value.get("scoring") == SCORING.to_dict(), "v3 scoring differs from frozen v2 scoring")
    _require(value.get("scale_floors") == SCALE_FLOORS, "v3 scale floors differ from frozen v2 floors")
    _require(value.get("thresholds") == THRESHOLDS, "v3 thresholds differ from frozen v2 thresholds")
    return config, manifest, value


def _commit_value(commit_dir: Path, specs: Sequence[RunSpec]) -> tuple[Snapshot, Mapping[str, Any]]:
    manifest = _read(commit_dir / "commitment-manifest.json", name="v3 commitment manifest", mode=0o444)
    value = _strict_json(manifest.payload, name="v3 commitment manifest")
    _require(value.get("schema_version") == f"{SCHEMA}.commitment.v1" and value.get("stage") == "commit", "v3 commitment schema/stage")
    rows = value.get("run_registry")
    _require(isinstance(rows, list) and [row.get("run_id") for row in rows if isinstance(row, Mapping)] == [spec.run_id for spec in specs], "v3 commitment run order")
    for row, spec in zip(rows, specs, strict=True):
        _require(isinstance(row, Mapping), "invalid v3 commitment run registry row")
        _require(row.get("split") == spec.split and row.get("corruption_type") == spec.corruption, f"v3 commitment identity mismatch for {spec.run_id}")
        artifacts = row.get("input_artifacts")
        _require(isinstance(artifacts, Mapping), f"v3 commitment input artifacts missing for {spec.run_id}")
        _artifact(spec.source_manifest, artifacts.get("source_manifest", {}), name=f"{spec.run_id} source")
        _artifact(spec.input_manifest, artifacts.get("input_manifest", {}), name=f"{spec.run_id} input")
        _artifact(spec.timestamp_listing, artifacts.get("rgb_timestamp_listing", {}), name=f"{spec.run_id} RGB timestamp listing")
        _require(str(spec.timestamp_root) == row.get("timestamp_dataset_root"), f"{spec.run_id} timestamp root changed")
    fixed = value.get("v3_fixed_configuration")
    _require(isinstance(fixed, Mapping), "v3 fixed configuration is missing")
    continuous = fixed.get("continuous_v2")
    _require(isinstance(continuous, Mapping), "v3 frozen v2 continuous configuration is missing")
    _require(continuous.get("scoring") == SCORING.to_dict() and continuous.get("scale_floors") == SCALE_FLOORS and continuous.get("thresholds") == THRESHOLDS, "v3 commitment continuous configuration changed")
    artifact = continuous.get("formal_config")
    _require(isinstance(artifact, Mapping) and artifact.get("sha256") == V2_FORMAL_CONFIG_SHA256, "v3 commitment is not bound to the frozen v2 formal configuration")
    return manifest, value


def _verify_runtime_binding(commitment: Mapping[str, Any]) -> None:
    _clean_commit(ROOT, expected=str(commitment.get("state_guard_commit")))
    _clean_commit(RECAL3R, expected=EXPECTED_RECAL3R_COMMIT)
    expected = commitment.get("tracked_sources")
    _require(isinstance(expected, Mapping), "v3 commitment tracked sources missing")
    for name, path in {
        "protocol": PROTOCOL,
        "runner": RUNNER,
        "orchestrator": Path(__file__).resolve(),
        "input_validator": VALIDATOR,
        "timestamp_module": TIMESTAMP_MODULE,
        "hybrid_module": HYBRID_MODULE,
    }.items():
        current = _tracked_snapshot(path, repository=ROOT, name=f"v3 {name}")
        _artifact(current, expected.get(name, {}), name=f"v3 committed {name}")


def commit_inputs(
    input_root: Path,
    output_dir: Path,
    *,
    protocol: Path,
    validation_dir: Path,
) -> dict[str, Any]:
    """Freeze v3 sources, fixed v2 configuration, inputs, and CPU readiness."""

    _require(input_root.resolve(strict=True) == V3_INPUTS.resolve(strict=True), "formal v3 must use frozen inputs 0001")
    _new_output(output_dir)
    state_commit = _clean_commit(ROOT)
    recal_commit = _clean_commit(RECAL3R, expected=EXPECTED_RECAL3R_COMMIT)
    snapshots = {
        "protocol": _tracked_snapshot(protocol.resolve(strict=True), repository=ROOT, name="formal-v3 protocol"),
        "runner": _tracked_snapshot(RUNNER, repository=ROOT, name="ReCal3R v3 runner"),
        "orchestrator": _tracked_snapshot(Path(__file__).resolve(), repository=ROOT, name="formal-v3 orchestrator"),
        "input_validator": _tracked_snapshot(VALIDATOR, repository=ROOT, name="formal-v3 input validator"),
        "timestamp_module": _tracked_snapshot(TIMESTAMP_MODULE, repository=ROOT, name="timestamp order v3 module"),
        "hybrid_module": _tracked_snapshot(HYBRID_MODULE, repository=ROOT, name="Detector v3 hybrid module"),
    }
    specs = _run_specs()
    registry = _read(V3_INPUTS / "formal-v3-manifest.json", name="v3 input registry", mode=0o444)
    registry_value = _strict_json(registry.payload, name="v3 input registry")
    _require(registry_value.get("status") == "pre_forward_blind_formal_v3_inputs", "v3 inputs are not pre-forward blind inputs")
    archive_value = registry_value.get("archive")
    acquisition_value = registry_value.get("acquisition")
    _require(isinstance(archive_value, Mapping) and isinstance(acquisition_value, Mapping), "v3 input archive/acquisition provenance missing")
    archive = _read(Path(str(archive_value.get("path", ""))), name="v3 new-scene archive", mode=0o444)
    acquisition = _read(Path(str(acquisition_value.get("path", ""))), name="v3 new-scene acquisition", mode=0o444)
    _require(_sha_bytes(archive.payload) == archive_value.get("sha256"), "v3 archive hash mismatch")
    _require(_sha_bytes(acquisition.payload) == acquisition_value.get("sha256"), "v3 acquisition hash mismatch")
    _require(_sha_file(CHECKPOINT) == CHECKPOINT_SHA256, "checkpoint digest mismatch")
    v2_config, v2_calibration_manifest, _ = _v2_fixed_config()
    development = _read(V3_DEVELOPMENT, name="Detector v3 development readiness", mode=0o444)
    _require(_strict_json(development.payload, name="Detector v3 development readiness").get("status") == "PASS", "Detector v3 development readiness is not PASS")
    opencv = _opencv_provenance()

    validation_root = validation_dir.resolve(strict=True)
    _require(validation_root.parent == OUTPUTS and validation_root.is_dir() and not validation_root.is_symlink(), "v3 CPU validation must be a direct immutable outputs child")
    _require(stat.S_IMODE(os.lstat(validation_root).st_mode) == 0o555, "v3 CPU validation directory is not frozen")
    report = _read(validation_root / "cpu-validation.json", name="frozen v3 CPU validation", mode=0o444)
    report_value = _strict_json(report.payload, name="frozen v3 CPU validation")
    checks = report_value.get("checks")
    _require(report_value.get("status") == "PASS" and isinstance(checks, Mapping), "frozen v3 CPU validation did not pass")
    _require(checks.get("cuda_hidden") is True, "frozen v3 CPU validation did not hide CUDA")
    _require(report_value.get("validator", {}).get("sha256") == _sha_bytes(snapshots["input_validator"].payload), "v3 CPU validation did not use committed validator bytes")

    rows = [
        {
            "run_id": spec.run_id,
            "split": spec.split,
            "corruption_type": spec.corruption,
            "input_root": str(spec.input_root),
            "timestamp_dataset_root": str(spec.timestamp_root),
            "input_artifacts": {
                "source_manifest": spec.source_manifest.artifact("source-manifest.json"),
                "input_manifest": spec.input_manifest.artifact("input-manifest.json"),
                "rgb_timestamp_listing": spec.timestamp_listing.artifact(str(spec.timestamp_listing.path)),
            },
        }
        for spec in specs
    ]
    execution_order = [spec.run_id for spec in specs]
    read_lock = "no blind response, response hash, timestamp sidecar byte, or runner main-log byte may be read before all three frozen blind outputs pass lstat/scandir-only structural preflight"
    run_registry = {"schema_version": f"{SCHEMA}.run-registry.v1", "execution_order": execution_order, "runs": rows, "blind_read_lock": read_lock}
    blind_commitment = {"schema_version": f"{SCHEMA}.blind-commitment.v1", "status": "pre_forward_blind_holdout", "runs": [row for row in rows if row["split"] == "holdout"], "input_registry": registry.artifact("formal-v3-manifest.json"), "read_lock": read_lock}
    config = {
        "continuous_v2": {"formal_config": v2_config.artifact(str(v2_config.path)), "calibration_manifest": v2_calibration_manifest.artifact(str(v2_calibration_manifest.path)), "scoring": SCORING.to_dict(), "scale_floors": SCALE_FLOORS, "thresholds": THRESHOLDS},
        "timestamp_hard_channel": {"schema_version": "stateguard3r.timestamp-order-v3.v1", "predicate": "null_first_frame_else_current_capture_timestamp_lte_previous", "decision_rule": "continuous_score_gte_threshold_or_timestamp_order_violation", "timestamp_score_margin": 1.0},
    }
    registry_bytes = _json_bytes(run_registry)
    blind_bytes = _json_bytes(blind_commitment)
    commitment = {
        "schema_version": f"{SCHEMA}.commitment.v1",
        "stage": "commit",
        "state_guard_commit": state_commit,
        "recal3r_commit": recal_commit,
        "tracked_sources": {name: snapshot.artifact(str(snapshot.path)) for name, snapshot in snapshots.items()},
        "checkpoint": {"path": str(CHECKPOINT), "sha256": CHECKPOINT_SHA256, "size_bytes": CHECKPOINT.stat().st_size},
        "v3_fixed_configuration": config,
        "development_evidence": development.artifact(str(V3_DEVELOPMENT)),
        "new_scene_inputs": {"registry": registry.artifact("formal-v3-manifest.json"), "archive": archive.artifact(str(archive.path)), "acquisition": acquisition.artifact(str(acquisition.path)), "cpu_validation": report.artifact(str(validation_root / "cpu-validation.json"))},
        "production_opencv": opencv,
        "run_registry": rows,
        "execution_order": execution_order,
        "gpu_requirement": {"minimum_free_mib": MINIMUM_FREE_MIB, "snapshots_before_launch": 2, "single_gpu": True, "serial": True},
        "blind_read_lock": read_lock,
        "execution_not_before": _utc(),
    }
    commitment["artifacts"] = {
        "run_registry": {"path": "run-registry.json", "sha256": _sha_bytes(registry_bytes), "size_bytes": len(registry_bytes)},
        "blind_commitment": {"path": "blind-commitment.json", "sha256": _sha_bytes(blind_bytes), "size_bytes": len(blind_bytes)},
        "cpu_validation": report.artifact(str(validation_root / "cpu-validation.json")),
    }
    commitment_bytes = _json_bytes(commitment)
    output = _publish(output_dir, {"commitment-manifest.json": commitment_bytes, "run-registry.json": registry_bytes, "blind-commitment.json": blind_bytes})
    return {"stage": "commit", "output_dir": str(output), "commitment_sha256": _sha_bytes(commitment_bytes)}


def _frozen_run_output(runs_root: Path, spec: RunSpec) -> dict[str, Snapshot]:
    directory = runs_root / spec.run_id
    metadata = os.lstat(directory)
    _require(stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) == 0o555, f"{spec.run_id} output is not frozen")
    _require({entry.name for entry in os.scandir(directory)} == set(RUN_FILES.values()), f"{spec.run_id} output artifact layout")
    return {key: _read(directory / filename, name=f"{spec.run_id} {filename}", mode=0o444) for key, filename in RUN_FILES.items()}


def _snapshot_outputs(runs_root: Path, specs: Sequence[RunSpec]) -> dict[str, dict[str, Snapshot]]:
    _require(runs_root.is_dir() and not runs_root.is_symlink(), "formal v3 runs root is missing")
    return {spec.run_id: _frozen_run_output(runs_root, spec) for spec in specs}


def _records_root(runs_root: Path) -> Path:
    return runs_root.with_name(runs_root.name + "-records")


def _metadata_only_frozen_run(runs_root: Path, spec: RunSpec) -> None:
    """Use lstat/scandir only.  This function must never open blind bytes."""

    directory = runs_root / spec.run_id
    metadata = os.lstat(directory)
    _require(stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) == 0o555, f"{spec.run_id} output is not frozen")
    _require({entry.name for entry in os.scandir(directory)} == set(RUN_FILES.values()), f"{spec.run_id} output artifact layout")
    for filename in RUN_FILES.values():
        child = os.lstat(directory / filename)
        _require(stat.S_ISREG(child.st_mode) and not stat.S_ISLNK(child.st_mode) and stat.S_IMODE(child.st_mode) == 0o444, f"{spec.run_id} {filename} is not frozen")


def _preflight_holdout_outputs(runs_root: Path, specs: Sequence[RunSpec]) -> None:
    """All-three-before-read structural gate for blind runner artifacts."""

    problems: list[str] = []
    for spec in specs:
        try:
            _metadata_only_frozen_run(runs_root, spec)
        except (OSError, FormalV3Error) as error:
            problems.append(f"{spec.run_id}: {error}")
    _require(not problems, "all three blind outputs must finish and freeze before any response/log/sidecar byte read: " + " | ".join(problems))


def _frozen_launch_record(runs_root: Path, spec: RunSpec, *, command: Sequence[str], not_before: datetime, unlock_at: datetime | None, previous_finished: datetime | None) -> tuple[Snapshot, Mapping[str, Any], datetime]:
    record_root = _records_root(runs_root) / spec.run_id
    _require(record_root.is_dir() and not record_root.is_symlink() and stat.S_IMODE(os.lstat(record_root).st_mode) == 0o555, f"{spec.run_id} launch record is not frozen")
    _require({entry.name for entry in os.scandir(record_root)} == {"launch.json"}, f"{spec.run_id} launch record layout")
    snapshot = _read(record_root / "launch.json", name=f"{spec.run_id} launch record", mode=0o444)
    value = _strict_json(snapshot.payload, name=f"{spec.run_id} launch record")
    _require(value.get("run_id") == spec.run_id and value.get("split") == spec.split and value.get("exit_code") == 0, f"{spec.run_id} launch record identity/status")
    _require(value.get("command") == list(command), f"{spec.run_id} launch command differs from commitment")
    _require(value.get("environment") == {"CUDA_VISIBLE_DEVICES": str(value.get("gpu", {}).get("id")), "TMPDIR": str(ROOT / "tmp")}, f"{spec.run_id} launch environment")
    started = _aware_time(value.get("started_at"), name=f"{spec.run_id} launch started_at")
    finished = _aware_time(value.get("finished_at"), name=f"{spec.run_id} launch finished_at")
    _require(not_before <= started <= finished, f"{spec.run_id} launch time predates commitment or is reversed")
    if unlock_at is not None:
        _require(unlock_at <= started, f"{spec.run_id} blind launch predates holdout unlock")
    if previous_finished is not None:
        _require(previous_finished <= started, f"{spec.run_id} blind launch order overlaps a preceding run")
    snapshots = value.get("prelaunch_snapshots")
    _require(isinstance(snapshots, list) and len(snapshots) == 2, f"{spec.run_id} lacks two prelaunch GPU snapshots")
    gpu = value.get("gpu")
    _require(isinstance(gpu, Mapping) and type(gpu.get("id")) is int and isinstance(gpu.get("uuid"), str), f"{spec.run_id} GPU record")
    for index, row in enumerate(snapshots):
        _require(isinstance(row, Mapping) and row.get("gpu_uuid") == gpu["uuid"] and isinstance(row.get("free_mib"), int) and row["free_mib"] >= MINIMUM_FREE_MIB, f"{spec.run_id} launch GPU capacity record {index}")
        _require(_aware_time(row.get("captured_at"), name=f"{spec.run_id} prelaunch snapshot {index}") <= started, f"{spec.run_id} prelaunch snapshot time")
    postflight = value.get("postexit_snapshot")
    _require(isinstance(postflight, Mapping) and postflight.get("gpu_uuid") == gpu["uuid"], f"{spec.run_id} postflight GPU record")
    _require(_aware_time(postflight.get("captured_at"), name=f"{spec.run_id} postflight snapshot") >= finished, f"{spec.run_id} postflight snapshot time")
    log_path = Path(str(value.get("log_path", "")))
    _require(log_path.is_file() and not log_path.is_symlink() and stat.S_IMODE(os.lstat(log_path).st_mode) == 0o444, f"{spec.run_id} launch log is not frozen")
    return snapshot, value, finished


def _health(snapshot: Snapshot, *, run_id: str) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(snapshot.payload.splitlines(), start=1):
        if line.strip():
            records.append(_strict_json(line, name=f"{run_id} health line {line_number}"))
    _require(len(records) == 30, f"{run_id} health ledger must contain 30 records")
    try:
        validate_online_overlap(records)
    except DetectorV2Error as error:
        raise FormalV3Error(f"{run_id} invalid online overlap: {error}") from error
    forbidden = {"groundtruth", "depth", "label", "event_start", "corruption_type", "source_index"}
    _require(all(not (forbidden & set(record)) for record in records), f"{run_id} health ledger contains non-online provenance")
    return records


def _validate_timestamp_sidecar(snapshot: Snapshot, run_json: Mapping[str, Any], spec: RunSpec) -> list[bool | None]:
    sidecar = _strict_json(snapshot.payload, name=f"{spec.run_id} timestamp sidecar")
    try:
        predicates = validate_timestamp_order_sidecar(sidecar, require_available=True)
    except TimestampOrderError as error:
        raise FormalV3Error(f"{spec.run_id} timestamp sidecar is invalid: {error}") from error
    _require(len(predicates) == 30, f"{spec.run_id} timestamp sidecar frame count")
    _require(not any(value is True for value in predicates[:15]), f"{spec.run_id} timestamp sidecar has a clean-prefix violation")
    metadata = run_json.get("capture_timestamp_order")
    _require(isinstance(metadata, Mapping) and set(metadata) == {"path", "sha256", "schema_version", "purpose", "violation_positions"}, f"{spec.run_id} timestamp sidecar metadata")
    _require(Path(str(metadata["path"])).resolve(strict=False) == snapshot.path.resolve(strict=False), f"{spec.run_id} timestamp sidecar path")
    _require(metadata.get("sha256") == _sha_bytes(snapshot.payload), f"{spec.run_id} timestamp sidecar digest")
    _require(metadata.get("schema_version") == sidecar.get("schema_version") and metadata.get("purpose") == sidecar.get("purpose"), f"{spec.run_id} timestamp sidecar version/purpose")
    positions = [index for index, value in enumerate(predicates) if value is True]
    _require(metadata.get("violation_positions") == positions, f"{spec.run_id} timestamp violation positions")
    provenance = sidecar.get("provenance")
    _require(isinstance(provenance, Mapping) and set(provenance) == {"parser_version", "rgb_txt_path", "rgb_txt_sha256", "dataset_root", "input_contract"}, f"{spec.run_id} timestamp sidecar provenance")
    _require(Path(str(provenance["rgb_txt_path"])).resolve(strict=False) == spec.timestamp_listing.path.resolve(strict=False), f"{spec.run_id} timestamp raw listing path")
    _require(provenance.get("rgb_txt_sha256") == _sha_bytes(spec.timestamp_listing.payload), f"{spec.run_id} timestamp raw listing digest")
    _require(Path(str(provenance["dataset_root"])).resolve(strict=False) == spec.timestamp_root.resolve(strict=False), f"{spec.run_id} timestamp raw root")
    _require(provenance.get("input_contract") == "ordered_rgb_paths_only_no_source_index_gt_label_or_event_metadata", f"{spec.run_id} timestamp input contract")
    images = run_json.get("images")
    rows = sidecar.get("records")
    _require(isinstance(images, list) and isinstance(rows, list) and len(images) == len(rows) == 30, f"{spec.run_id} timestamp/model input binding")
    for frame_id, (image, row) in enumerate(zip(images, rows, strict=True)):
        _require(isinstance(image, Mapping) and isinstance(row, Mapping) and row.get("frame_id") == frame_id and image.get("sha256") == row.get("rgb_path_sha256"), f"{spec.run_id} timestamp sidecar RGB binding {frame_id}")
    return predicates


def _validate_run_json(outputs: Mapping[str, Snapshot], *, spec: RunSpec, commitment: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _strict_json(outputs["run_json"].payload, name=f"{spec.run_id} run.json")
    _require(value.get("status") == "succeeded" and value.get("health_profile") == "v3", f"{spec.run_id} is not a successful v3 run")
    _require(value.get("frame_count") == 30 and value.get("calibrated_update_calls") == 29, f"{spec.run_id} frame/update count")
    _require(value.get("checkpoint_sha256") == CHECKPOINT_SHA256, f"{spec.run_id} checkpoint mismatch")
    _require(value.get("baseline_commit") == EXPECTED_RECAL3R_COMMIT and value.get("baseline_tracked_worktree_clean") is True, f"{spec.run_id} baseline provenance")
    runner = value.get("runner")
    _require(isinstance(runner, Mapping), f"{spec.run_id} runner provenance missing")
    expected_runner = commitment["tracked_sources"]["runner"]
    _require(runner.get("commit") == commitment["state_guard_commit"] and runner.get("script_sha256") == expected_runner["sha256"] and runner.get("tracked_worktree_clean") is True, f"{spec.run_id} runner provenance mismatch")
    manifest = value.get("input_manifest")
    _require(isinstance(manifest, Mapping) and manifest.get("sha256") == _sha_bytes(spec.input_manifest.payload) and manifest.get("source_manifest_sha256") == _sha_bytes(spec.source_manifest.payload), f"{spec.run_id} input manifest provenance")
    visual = value.get("online_visual_correspondence")
    _require(isinstance(visual, Mapping) and visual.get("input") == "post_official_loader_post_deferred_transform_normalized_rgb", f"{spec.run_id} visual provenance")
    _require("groundtruth" not in visual and "depth" not in visual and "label" not in visual and "source_index" not in visual, f"{spec.run_id} visual provenance leaks forbidden fields")
    for name in ("runtime_seconds", "peak_memory_allocated_mib"):
        number = value.get(name)
        _require(isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(float(number)) and float(number) > 0, f"{spec.run_id} invalid {name}")
    _validate_timestamp_sidecar(outputs["timestamp_order"], value, spec)
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
    current = longest = 0
    for active in mask & (labels == 0) & (scores >= threshold):
        if bool(active):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _score_split(specs: Sequence[RunSpec], outputs: Mapping[str, Mapping[str, Snapshot]]) -> dict[str, Any]:
    records = {spec.run_id: _health(outputs[spec.run_id]["health_jsonl"], run_id=spec.run_id) for spec in specs}
    scores: dict[str, Mapping[str, Any]] = {}
    hybrids: dict[str, Mapping[str, Any]] = {}
    for spec in specs:
        seed = detection_suite.derive_run_seed(SCORING.master_seed, spec.split, spec.run_id)
        scores[spec.run_id] = compute_v2_scores(records[spec.run_id], config=SCORING, scale_floors=SCALE_FLOORS, seed=seed, require_online_overlap=True)
        sidecar = _strict_json(outputs[spec.run_id]["timestamp_order"].payload, name=f"{spec.run_id} timestamp sidecar")
        try:
            hybrids[spec.run_id] = hybridize_v2_continuous_scores(scores[spec.run_id]["methods"]["combined"]["scores"], continuous_threshold=THRESHOLDS["combined"], timestamp_sidecar=sidecar)
        except DetectorV3Error as error:
            raise FormalV3Error(f"{spec.run_id} hybrid score failed: {error}") from error

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
        max_streak = max_startup_streak = 0
        for spec in specs:
            labels, mask, prefix = _event_labels(spec)
            payload = scores[spec.run_id]["methods"][method_name]
            values = np.asarray(payload["scores"], dtype=np.float64)
            hybrid: Mapping[str, Any] | None = None
            if method_name == "combined":
                hybrid = hybrids[spec.run_id]
                values = np.asarray(hybrid["hybrid_scores"], dtype=np.float64)
                declared = np.asarray(hybrid["alarms"], dtype=bool)
                _require(np.array_equal(values >= threshold, declared), f"{spec.run_id} hybrid score/alarm semantics diverge")
            _require(values.shape == (30,) and bool(np.isfinite(values).all()), f"{spec.run_id} {method_name} non-finite score")
            metric = detection.threshold_metrics(labels[mask], values[mask], threshold)
            delay = detection.detection_delay(labels, values, threshold)
            auroc = detection.auroc(labels[mask], values[mask])
            _require(auroc is not None, f"{spec.run_id} {method_name} AUROC unavailable")
            if int(delay["detected_intervals"]) > 0:
                detected.add(spec.corruption)
            delays.extend(int(row["delay_frames"]) for row in delay["intervals"] if row["delay_frames"] is not None)
            run_streak = _streak(labels, values, mask, threshold)
            startup_streak = _streak(labels, values, prefix, threshold)
            max_streak = max(max_streak, run_streak)
            max_startup_streak = max(max_startup_streak, startup_streak)
            row: dict[str, Any] = {
                "scores": values.tolist(), "threshold": threshold, "primary_metrics": metric, "primary_auroc": float(auroc), "delay": delay,
                "detected": int(delay["detected_intervals"]) > 0, "max_primary_false_positive_streak": run_streak,
                "max_startup_false_positive_streak": startup_streak, "used_signals": payload["used_signals"],
                "signal_sources": payload["signal_sources"], "finite_component_count": np.asarray(payload["finite_component_count"], dtype=np.int64).tolist(),
            }
            if hybrid is not None:
                row.update({"continuous_scores": np.asarray(hybrid["continuous_scores"], dtype=np.float64).tolist(), "timestamp_attribution": hybrid["attribution"], "decision_rule": hybrid["decision_rule"]})
            per_run[spec.run_id] = row
            pooled_labels.append(labels[mask]); pooled_scores.append(values[mask]); prefix_labels.append(labels[prefix]); prefix_scores.append(values[prefix]); aurocs.append(float(auroc))
        methods[method_name] = {
            "threshold": threshold, "per_run": per_run, "macro_auroc": float(np.mean(aurocs)),
            "pooled": detection.threshold_metrics(np.concatenate(pooled_labels), np.concatenate(pooled_scores), threshold),
            "clean_prefix": detection.threshold_metrics(np.concatenate(prefix_labels), np.concatenate(prefix_scores), threshold),
            "events": {"detected_corruption_types": sorted(detected), "detected_events": len(detected), "mean_delay_frames": float(np.mean(delays)) if delays else None},
            "diagnostics": {"max_per_run_consecutive_false_positives": max_streak, "startup_max_streak": max_startup_streak},
        }
    return {"schema_version": f"{SCHEMA}.metrics.v1", "continuous_scoring": SCORING.to_dict(), "scale_floors": SCALE_FLOORS, "thresholds": THRESHOLDS, "hybrid_rule": "continuous_score_gte_threshold_or_timestamp_order_violation", "runs": [{"run_id": spec.run_id, "split": spec.split, "corruption_type": spec.corruption, "primary_labels": _event_labels(spec)[0].tolist(), "primary_mask": _event_labels(spec)[1].tolist(), "clean_prefix_mask": _event_labels(spec)[2].tolist()} for spec in specs], "methods": methods}


def _run_artifacts(outputs: Mapping[str, Mapping[str, Snapshot]]) -> dict[str, Any]:
    return {run_id: {kind: snapshot.artifact(f"{run_id}/{RUN_FILES[kind]}") for kind, snapshot in row.items()} for run_id, row in outputs.items()}


def _load_calibration(calibration_dir: Path, commitment: Snapshot) -> tuple[Snapshot, Mapping[str, Any], Mapping[str, Any], Snapshot, Mapping[str, Any]]:
    manifest = _read(calibration_dir / "calibration-manifest.json", name="v3 calibration manifest", mode=0o444)
    value = _strict_json(manifest.payload, name="v3 calibration manifest")
    _require(value.get("schema_version") == f"{SCHEMA}.calibration.v1" and value.get("stage") == "calibrate", "v3 calibration schema/stage")
    _require(value.get("commitment", {}).get("sha256") == _sha_bytes(commitment.payload), "v3 calibration commitment binding")
    metrics = _read(calibration_dir / "development-metrics.json", name="v3 development metrics", mode=0o444)
    _artifact(metrics, value.get("artifacts", {}).get("development_metrics", {}), name="v3 calibration metrics")
    unlock_dir = calibration_dir.with_name(calibration_dir.name + "-holdout-unlock")
    unlock = _read(unlock_dir / "holdout-unlock.json", name="v3 holdout unlock", mode=0o444)
    unlock_value = _strict_json(unlock.payload, name="v3 holdout unlock")
    _require(unlock_value.get("status") == "holdout-forwards-permitted" and unlock_value.get("calibration_manifest", {}).get("sha256") == _sha_bytes(manifest.payload), "v3 holdout unlock binding")
    return manifest, value, _strict_json(metrics.payload, name="v3 development metrics"), unlock, unlock_value


def calibrate_development(input_root: Path, commit_dir: Path, runs_root: Path, output_dir: Path) -> dict[str, Any]:
    """Score disclosed runs only, then atomically create the blind unlock."""

    _require(input_root.resolve(strict=True) == V3_INPUTS.resolve(strict=True), "v3 calibration input root mismatch")
    _new_output(output_dir)
    specs = _run_specs()
    commitment, value = _commit_value(commit_dir.resolve(strict=True), specs)
    _verify_runtime_binding(value)
    development = tuple(spec for spec in specs if spec.split == "development")
    # Keep this call capability-narrow: no blind spec/path can reach it.
    outputs = _snapshot_outputs(runs_root.resolve(strict=True), development)
    for spec in development:
        _validate_run_json(outputs[spec.run_id], spec=spec, commitment=value)
    metrics = _score_split(development, outputs)
    metric_bytes = _json_bytes(metrics)
    config = {"schema_version": f"{SCHEMA}.formal-config.v1", "continuous_scoring": SCORING.to_dict(), "scale_floors": SCALE_FLOORS, "thresholds": THRESHOLDS, "hybrid_rule": "continuous_score_gte_threshold_or_timestamp_order_violation", "selection": "all values frozen before blind response; calibration recomputes only"}
    config_bytes = _json_bytes(config)
    manifest = {"schema_version": f"{SCHEMA}.calibration.v1", "stage": "calibrate", "commitment": commitment.artifact("commitment-manifest.json"), "development_run_artifacts": _run_artifacts(outputs), "artifacts": {"development_metrics": {"path": "development-metrics.json", "sha256": _sha_bytes(metric_bytes), "size_bytes": len(metric_bytes)}, "formal_config": {"path": "formal-config.json", "sha256": _sha_bytes(config_bytes), "size_bytes": len(config_bytes)}}}
    manifest_bytes = _json_bytes(manifest)
    output = _publish(output_dir, {"development-metrics.json": metric_bytes, "formal-config.json": config_bytes, "calibration-manifest.json": manifest_bytes})
    unlock = {"schema_version": f"{SCHEMA}.holdout-unlock.v1", "calibration_manifest": {"path": str(output / "calibration-manifest.json"), "sha256": _sha_bytes(manifest_bytes), "size_bytes": len(manifest_bytes)}, "unlocked_at": _utc(), "status": "holdout-forwards-permitted"}
    unlock_dir = _new_output(output.with_name(output.name + "-holdout-unlock"))
    _publish(unlock_dir, {"holdout-unlock.json": _json_bytes(unlock)})
    return {"stage": "calibrate", "output_dir": str(output), "holdout_unlock": str(unlock_dir / "holdout-unlock.json"), "calibration_manifest_sha256": _sha_bytes(manifest_bytes)}


def _go_no_go(development: Mapping[str, Any], holdout: Mapping[str, Any], *, provenance: bool) -> dict[str, Any]:
    methods = holdout["methods"]
    combined = float(methods["combined"]["macro_auroc"])
    random = float(methods["random"]["macro_auroc"])
    best_single = max(float(methods["update_magnitude_only"]["macro_auroc"]), float(methods["reliability_only"]["macro_auroc"]))
    per_run = methods["combined"]["per_run"]
    wrong_delay = per_run["blind-wrong-order"]["delay"]["intervals"][0]["delay_frames"]
    detected = set(methods["combined"]["events"]["detected_corruption_types"])
    expected = {"dynamic_occlusion", "wrong_order_segment", "low_overlap_jump"}
    checks = [
        {"id": "combined_macro_auroc_strictly_above_0_75", "passed": combined > 0.75, "actual": combined},
        {"id": "combined_minus_seeded_random_at_least_0_10", "passed": combined - random >= 0.10, "actual": combined - random},
        {"id": "combined_within_0_02_of_best_real_single", "passed": combined >= best_single - 0.02, "actual": combined, "best_single": best_single},
        {"id": "all_three_corruption_types_detected", "passed": detected == expected, "actual": sorted(detected)},
        {"id": "wrong_order_delay_at_most_one", "passed": wrong_delay is not None and int(wrong_delay) <= 1, "actual": wrong_delay},
        {"id": "pooled_primary_fpr_at_most_0_20", "passed": float(methods["combined"]["pooled"]["false_positive_rate"]) <= 0.20, "actual": methods["combined"]["pooled"]["false_positive_rate"]},
        {"id": "max_primary_fp_streak_at_most_3", "passed": int(methods["combined"]["diagnostics"]["max_per_run_consecutive_false_positives"]) <= 3, "actual": methods["combined"]["diagnostics"]["max_per_run_consecutive_false_positives"]},
        {"id": "clean_prefix_fpr_at_most_0_15_and_startup_streak_at_most_2", "passed": float(methods["combined"]["clean_prefix"]["false_positive_rate"]) <= 0.15 and int(methods["combined"]["diagnostics"]["startup_max_streak"]) <= 2, "actual": {"fpr": methods["combined"]["clean_prefix"]["false_positive_rate"], "streak": methods["combined"]["diagnostics"]["startup_max_streak"]}},
        {"id": "provenance_timestamp_hybrid_runtime_integrity", "passed": provenance, "actual": provenance},
    ]
    passed = all(item["passed"] is True for item in checks)
    return {"schema_version": f"{SCHEMA}.go-no-go.v1", "decision": "DETECTOR_V3_GO" if passed else "DETECTOR_V3_NO_GO", "passed": passed, "checks": checks, "next_step": "separately preregistered exploratory state-policy feasibility" if passed else "publish formal v3 no-go and stop before state-policy evaluation"}


def _gpu_snapshot(gpu_id: int) -> dict[str, Any]:
    query = subprocess.run(["nvidia-smi", f"--id={gpu_id}", "--query-gpu=index,uuid,memory.free,memory.total", "--format=csv,noheader,nounits"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pieces = [piece.strip() for piece in query.stdout.strip().split(",")]
    _require(len(pieces) == 4 and pieces[0] == str(gpu_id), f"cannot parse GPU {gpu_id} memory query")
    try:
        free_mib, total_mib = int(pieces[2]), int(pieces[3])
    except ValueError as error:
        raise FormalV3Error(f"cannot parse GPU memory query: {query.stdout!r}") from error
    full = subprocess.run(["nvidia-smi", f"--id={gpu_id}"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    process = subprocess.run(["ps", "-eo", "pid=,user=,args="], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return {"captured_at": _utc(), "gpu_id": gpu_id, "gpu_uuid": pieces[1], "free_mib": free_mib, "total_mib": total_mib, "nvidia_smi": full.stdout, "processes": process.stdout}


def _runner_command(spec: RunSpec, output: Path) -> list[str]:
    return [str(RECAL3R_PYTHON), str(RUNNER), "--baseline-root", str(RECAL3R), "--checkpoint", str(CHECKPOINT), "--checkpoint-sha256", CHECKPOINT_SHA256, "--input-manifest", str(spec.input_manifest.path), "--output-dir", str(output), "--device", "cuda", "--size", "512", "--seed", "0", "--beta-base", "0.1", "--health-profile", "v3", "--rgb-timestamp-listing", str(spec.timestamp_listing.path), "--timestamp-dataset-root", str(spec.timestamp_root)]


def launch_forward(commit_dir: Path, calibration_dir: Path | None, runs_root: Path, records_root: Path, log_dir: Path, run_id: str, gpu_id: int) -> dict[str, Any]:
    """Launch one serial formal run without reading an individual blind result."""

    specs = _run_specs()
    commitment, value = _commit_value(commit_dir.resolve(strict=True), specs)
    _verify_runtime_binding(value)
    spec = _spec_by_id(specs, run_id)
    if spec.split == "development":
        prior = [item for item in specs if item.split == "development"]
        for earlier in prior[:prior.index(spec)]:
            _frozen_run_output(runs_root, earlier)
    else:
        _require(calibration_dir is not None, "blind launch requires calibration directory")
        _, _, _, _, unlock_value = _load_calibration(calibration_dir.resolve(strict=True), commitment)
        blind = [item for item in specs if item.split == "holdout"]
        # Enforce the frozen blind order with metadata-only checks; never open
        # a preceding blind response, timestamp sidecar, or main log here.
        for earlier in blind[:blind.index(spec)]:
            _metadata_only_frozen_run(runs_root, earlier)
        _aware_time(unlock_value.get("unlocked_at"), name="v3 holdout unlock timestamp")
    _require(runs_root.parent.resolve(strict=False) == OUTPUTS and records_root.resolve(strict=False) == _records_root(runs_root).resolve(strict=False), "runs and records roots must use fixed direct-output layout")
    _require(log_dir.parent.resolve(strict=False) == LOGS, "log directory must be a direct logs child")
    runs_root.mkdir(exist_ok=True); records_root.mkdir(exist_ok=True); log_dir.mkdir(exist_ok=True)
    output = runs_root / run_id
    record_dir = records_root / run_id
    log_path = log_dir / f"{run_id}.log"
    _require(not output.exists() and not record_dir.exists() and not log_path.exists(), f"formal v3 run {run_id} already has an output, record, or log")
    before_one = _gpu_snapshot(gpu_id); before_two = _gpu_snapshot(gpu_id)
    _require(before_one["free_mib"] >= MINIMUM_FREE_MIB and before_two["free_mib"] >= MINIMUM_FREE_MIB and before_one["gpu_uuid"] == before_two["gpu_uuid"], f"GPU {gpu_id} lacks stable formal v3 capacity")
    command = _runner_command(spec, output)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id), "TMPDIR": str(ROOT / "tmp")}
    started = _utc()
    with log_path.open("xb") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, text=False)
        status = process.wait()
    finished = _utc(); after = _gpu_snapshot(gpu_id)
    record = {"schema_version": f"{SCHEMA}.launch.v1", "run_id": run_id, "split": spec.split, "command": command, "environment": {"CUDA_VISIBLE_DEVICES": str(gpu_id), "TMPDIR": str(ROOT / "tmp")}, "pid": process.pid, "gpu": {"id": gpu_id, "uuid": before_two["gpu_uuid"]}, "prelaunch_snapshots": [before_one, before_two], "postexit_snapshot": after, "minimum_free_mib": MINIMUM_FREE_MIB, "started_at": started, "finished_at": finished, "exit_code": status, "log_path": str(log_path), "output_dir": str(output)}
    _require(status == 0, f"formal v3 runner failed for {run_id}; preserved log/output at {log_path}")
    _freeze_tree(output)
    if spec.split == "holdout":
        _metadata_only_frozen_run(runs_root, spec)
    else:
        _frozen_run_output(runs_root, spec)
    record_dir.mkdir(); (record_dir / "launch.json").write_bytes(_json_bytes(record)); _freeze_tree(record_dir); log_path.chmod(0o444)
    return {"stage": "launch", "run_id": run_id, "pid": process.pid, "output_dir": str(output), "log_path": str(log_path)}


def evaluate_holdout(input_root: Path, commit_dir: Path, calibration_dir: Path, runs_root: Path, output_dir: Path) -> dict[str, Any]:
    """Seal one evaluation attempt after structural all-three blind preflight."""

    _require(input_root.resolve(strict=True) == V3_INPUTS.resolve(strict=True), "v3 evaluation input root mismatch")
    _new_output(output_dir)
    attempt_dir = output_dir.with_name(output_dir.name + "-attempt")
    _new_output(attempt_dir)
    specs = _run_specs()
    blind = tuple(spec for spec in specs if spec.split == "holdout")
    # Do not move anything that opens a blind byte above this call.
    _preflight_holdout_outputs(runs_root.resolve(strict=True), blind)
    # The seal is irreversible.  A crash after it means the blind scene is
    # disclosed and a second evaluation attempt is rejected by no-overwrite.
    seal = {"schema_version": f"{SCHEMA}.evaluation-attempt.v1", "stage": "sealed_before_blind_byte_read", "created_at": _utc(), "commit_dir": str(commit_dir.resolve(strict=True)), "calibration_dir": str(calibration_dir.resolve(strict=True)), "runs_root": str(runs_root.resolve(strict=True)), "read_lock": "all three blind layouts passed lstat/scandir-only preflight"}
    _publish(attempt_dir, {"evaluation-attempt.json": _json_bytes(seal)})
    commitment, value = _commit_value(commit_dir.resolve(strict=True), specs)
    _verify_runtime_binding(value)
    calibration_manifest, _, development_metrics, unlock, unlock_value = _load_calibration(calibration_dir.resolve(strict=True), commitment)
    not_before = _aware_time(value.get("execution_not_before"), name="v3 commitment execution_not_before")
    unlock_at = _aware_time(unlock_value.get("unlocked_at"), name="v3 holdout unlock timestamp")
    holdout_outputs = _snapshot_outputs(runs_root.resolve(strict=True), blind)
    development = tuple(spec for spec in specs if spec.split == "development")
    development_outputs = _snapshot_outputs(runs_root.resolve(strict=True), development)
    all_outputs = {**development_outputs, **holdout_outputs}
    launch_artifacts: dict[str, Any] = {}
    previous: datetime | None = None
    for spec in specs:
        command = _runner_command(spec, runs_root / spec.run_id)
        launch_snapshot, launch, finished = _frozen_launch_record(runs_root.resolve(strict=True), spec, command=command, not_before=not_before, unlock_at=unlock_at if spec.split == "holdout" else None, previous_finished=previous if spec.split == "holdout" else None)
        run_value = _validate_run_json(all_outputs[spec.run_id], spec=spec, commitment=value)
        run_started = _aware_time(run_value.get("started_at"), name=f"{spec.run_id} runner started_at")
        run_finished = _aware_time(run_value.get("finished_at"), name=f"{spec.run_id} runner finished_at")
        _require(_aware_time(launch["started_at"], name=f"{spec.run_id} launch start") <= run_started <= run_finished <= _aware_time(launch["finished_at"], name=f"{spec.run_id} launch finish"), f"{spec.run_id} runner time is outside launcher interval")
        launch_artifacts[spec.run_id] = launch_snapshot.artifact(f"{spec.run_id}/launch.json")
        if spec.split == "holdout":
            previous = finished
    recomputed_development = _score_split(development, development_outputs)
    _require(_json_bytes(recomputed_development) == _json_bytes(development_metrics), "v3 calibration development metrics do not replay byte-identically")
    holdout_metrics = _score_split(blind, holdout_outputs)
    gate = _go_no_go(development_metrics, holdout_metrics, provenance=True)
    metric_bytes = _json_bytes(holdout_metrics); gate_bytes = _json_bytes(gate)
    timelines = {f"timelines/{spec.run_id}.json": _json_bytes({"run_id": spec.run_id, "corruption_type": spec.corruption, "scores": holdout_metrics["methods"]["combined"]["per_run"][spec.run_id]["scores"], "attribution": holdout_metrics["methods"]["combined"]["per_run"][spec.run_id]["timestamp_attribution"], "labels": _event_labels(spec)[0].tolist()}) for spec in blind}
    manifest = {"schema_version": f"{SCHEMA}.evaluation.v1", "stage": "evaluate", "decision": gate["decision"], "inputs": {"commitment": commitment.artifact("commitment-manifest.json"), "calibration": calibration_manifest.artifact("calibration-manifest.json"), "attempt": {"path": str(attempt_dir / "evaluation-attempt.json"), "sha256": _sha_file(attempt_dir / "evaluation-attempt.json"), "size_bytes": (attempt_dir / "evaluation-attempt.json").stat().st_size}}, "run_artifacts": _run_artifacts(all_outputs), "launch_artifacts": launch_artifacts, "artifacts": {"holdout_metrics": {"path": "holdout-metrics.json", "sha256": _sha_bytes(metric_bytes), "size_bytes": len(metric_bytes)}, "go_no_go": {"path": "go-no-go.json", "sha256": _sha_bytes(gate_bytes), "size_bytes": len(gate_bytes)}}, "blind_preflight": "all three frozen blind layouts, including timestamp-order.json, were inspected by metadata before any blind output bytes were read", "provenance_checks_passed": True}
    manifest_bytes = _json_bytes(manifest)
    output = _publish(output_dir, {"holdout-metrics.json": metric_bytes, "go-no-go.json": gate_bytes, "evaluation-manifest.json": manifest_bytes, **timelines})
    return {"stage": "evaluate", "output_dir": str(output), "attempt_dir": str(attempt_dir), "decision": gate["decision"], "evaluation_manifest_sha256": _sha_bytes(manifest_bytes)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    commit = actions.add_parser("commit"); commit.add_argument("input_root", type=Path); commit.add_argument("output_dir", type=Path); commit.add_argument("--protocol", type=Path, required=True); commit.add_argument("--validation-dir", type=Path, required=True)
    calibrate = actions.add_parser("calibrate"); calibrate.add_argument("input_root", type=Path); calibrate.add_argument("commit_dir", type=Path); calibrate.add_argument("runs_root", type=Path); calibrate.add_argument("output_dir", type=Path)
    evaluate = actions.add_parser("evaluate"); evaluate.add_argument("input_root", type=Path); evaluate.add_argument("commit_dir", type=Path); evaluate.add_argument("calibration_dir", type=Path); evaluate.add_argument("runs_root", type=Path); evaluate.add_argument("output_dir", type=Path)
    launch = actions.add_parser("launch"); launch.add_argument("commit_dir", type=Path); launch.add_argument("runs_root", type=Path); launch.add_argument("records_root", type=Path); launch.add_argument("log_dir", type=Path); launch.add_argument("run_id"); launch.add_argument("--gpu-id", required=True, type=int); launch.add_argument("--calibration-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser(); args = parser.parse_args(argv)
    try:
        if args.command == "commit":
            result = commit_inputs(args.input_root, args.output_dir, protocol=args.protocol, validation_dir=args.validation_dir)
        elif args.command == "calibrate":
            result = calibrate_development(args.input_root, args.commit_dir, args.runs_root, args.output_dir)
        elif args.command == "evaluate":
            result = evaluate_holdout(args.input_root, args.commit_dir, args.calibration_dir, args.runs_root, args.output_dir)
        else:
            result = launch_forward(args.commit_dir, args.calibration_dir, args.runs_root, args.records_root, args.log_dir, args.run_id, args.gpu_id)
    except (FormalV3Error, DetectorV2Error, DetectorV3Error, OSError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
