#!/usr/bin/env python3
"""Validate that Detector v3 timestamp instrumentation leaves ReCal3R unchanged.

This CPU-only validator compares one completed v2 run with a v3 run over the
same disclosed input.  It only publishes a new audit directory: neither
runner result is modified.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from stateguard3r.timestamp_order_v3 import (
    TimestampOrderError,
    validate_timestamp_order_sidecar,
)


RUN_ARTIFACTS = (
    "checkpoint-load-audit.json",
    "health.jsonl",
    "predictions-summary.json",
    "run.json",
    "trajectory.json",
)
BYTE_IDENTICAL_ARTIFACTS = (
    "checkpoint-load-audit.json",
    "health.jsonl",
    "predictions-summary.json",
    "trajectory.json",
)
RUN_INVARIANT_FIELDS = (
    "schema_version",
    "status",
    "baseline_root",
    "baseline_commit",
    "baseline_tracked_worktree_clean",
    "module_paths",
    "curope_kernel",
    "runner",
    "checkpoint",
    "checkpoint_sha256",
    "checkpoint_size_bytes",
    "input_mode",
    "input_manifest",
    "images",
    "input_frames",
    "frame_count",
    "size",
    "seed",
    "model_update_type",
    "loaded_model_interface",
    "checkpoint_state_dict",
    "beta_base",
    "recal3r_runtime_config",
    "recal3r_runtime_config_source",
    "model_eval",
    "calibrated_update_calls",
    "expected_calibrated_update_calls",
    "trace_frame_steps",
    "health_signal_semantics",
    "state_args_count",
    "runtime_scope",
    "peak_memory_scope",
    "device",
    "cuda_visible_devices",
    "gpu_name",
    "torch_version",
    "cuda_runtime",
)


class InstrumentationValidationError(RuntimeError):
    """Raised when completed v2/v3 runs violate the observational contract."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v2_run", type=Path)
    parser.add_argument("v3_run", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--v2-postflight-log", required=True, type=Path)
    parser.add_argument("--v3-postflight-log", required=True, type=Path)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InstrumentationValidationError(message)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InstrumentationValidationError(f"cannot read JSON {path}: {error}") from error


def _assert_regular_file(path: Path, *, label: str) -> None:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular non-symlink file: {path}")


def _artifact_hashes(run_dir: Path, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        path = run_dir / name
        _assert_regular_file(path, label=f"runner artifact {name}")
        result[name] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    return result


def _assert_run_invariants(v2_run: Mapping[str, Any], v3_run: Mapping[str, Any]) -> None:
    _require(v2_run.get("status") == "succeeded", "v2 run did not succeed")
    _require(v3_run.get("status") == "succeeded", "v3 run did not succeed")
    _require(v2_run.get("health_profile") == "v2", "v2 run does not declare profile v2")
    _require(v3_run.get("health_profile") == "v3", "v3 run does not declare profile v3")
    for key in RUN_INVARIANT_FIELDS:
        _require(key in v2_run and key in v3_run, f"run invariant is missing: {key}")
        _require(v2_run[key] == v3_run[key], f"run invariant differs: {key}")
    frame_count = v2_run["frame_count"]
    _require(
        isinstance(frame_count, int) and not isinstance(frame_count, bool) and frame_count >= 2,
        "frame_count must be an integer of at least two",
    )
    _require(
        v2_run["calibrated_update_calls"] == frame_count - 1
        and v3_run["calibrated_update_calls"] == frame_count - 1,
        "recurrent update count does not equal frame_count - 1",
    )


def _visual_metadata(run: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    metadata = run.get("online_visual_correspondence")
    _require(isinstance(metadata, dict), f"{label} visual correspondence metadata is missing")
    _require(
        metadata.get("schema_version") == "stateguard3r.visual-overlap.v1",
        f"{label} has an unexpected visual schema",
    )
    _require(
        metadata.get("input") == "post_official_loader_post_deferred_transform_normalized_rgb",
        f"{label} visual input is not model-ready RGB",
    )
    _require(
        metadata.get("causal_reference") == "immediately_previous_frame_only",
        f"{label} visual reference is not causal",
    )
    return copy.deepcopy(metadata)


def _assert_timestamp_sidecar(v3_dir: Path, v3_run: Mapping[str, Any]) -> dict[str, Any]:
    metadata = v3_run.get("capture_timestamp_order")
    _require(isinstance(metadata, dict), "v3 capture timestamp metadata is missing")
    expected_keys = {
        "path",
        "sha256",
        "schema_version",
        "purpose",
        "violation_positions",
    }
    _require(set(metadata) == expected_keys, "v3 capture timestamp metadata schema is invalid")
    sidecar_path = v3_dir / "timestamp-order.json"
    _require(Path(metadata["path"]).resolve(strict=False) == sidecar_path.resolve(), "v3 timestamp sidecar path is not its run-local artifact")
    _assert_regular_file(sidecar_path, label="v3 timestamp sidecar")
    _require(metadata["sha256"] == _sha256(sidecar_path), "v3 timestamp sidecar hash differs from run metadata")
    sidecar = _read_json(sidecar_path)
    try:
        predicates = validate_timestamp_order_sidecar(sidecar, require_available=True)
    except TimestampOrderError as error:
        raise InstrumentationValidationError(f"invalid v3 timestamp sidecar: {error}") from error
    _require(metadata["schema_version"] == sidecar.get("schema_version"), "v3 timestamp schema metadata differs")
    _require(metadata["purpose"] == sidecar.get("purpose"), "v3 timestamp purpose metadata differs")
    positions = [index for index, predicate in enumerate(predicates) if predicate is True]
    _require(metadata["violation_positions"] == positions, "v3 violation positions differ from sidecar")
    images = v3_run.get("images")
    records = sidecar.get("records") if isinstance(sidecar, Mapping) else None
    _require(isinstance(images, list) and isinstance(records, list), "v3 images or timestamp records are invalid")
    _require(len(images) == len(records), "v3 timestamp record count differs from model input count")
    for index, (image, record) in enumerate(zip(images, records)):
        _require(
            isinstance(image, Mapping) and isinstance(record, Mapping)
            and image.get("sha256") == record.get("rgb_path_sha256"),
            f"v3 timestamp record is not bound to model input frame {index}",
        )
    return {
        "schema_version": sidecar["schema_version"],
        "purpose": sidecar["purpose"],
        "sidecar_sha256": metadata["sha256"],
        "violation_positions": positions,
        "frame_count": len(records),
    }


def _assert_postflight(log_path: Path, *, run_pid: Any, label: str) -> dict[str, Any]:
    _assert_regular_file(log_path, label=f"{label} postflight log")
    _require(isinstance(run_pid, int) and not isinstance(run_pid, bool), f"{label} run PID is invalid")
    text = log_path.read_text(encoding="utf-8")
    _require(
        "NVIDIA-SMI" in text or "NVSMI LOG" in text,
        f"{label} postflight log has no nvidia-smi snapshot",
    )
    _require(str(run_pid) not in text, f"{label} run PID {run_pid} is still listed in its postflight log")
    return {
        "path": str(log_path.resolve()),
        "sha256": _sha256(log_path),
        "size_bytes": log_path.stat().st_size,
        "run_pid_absent": True,
    }


def validate_instrumentation(
    *,
    v2_dir: Path,
    v3_dir: Path,
    v2_postflight_log: Path,
    v3_postflight_log: Path,
) -> dict[str, Any]:
    """Read and validate completed v2/v3 artifacts without modifying them."""

    v2_dir = v2_dir.resolve()
    v3_dir = v3_dir.resolve()
    _require(v2_dir.is_dir(), f"v2 run directory does not exist: {v2_dir}")
    _require(v3_dir.is_dir(), f"v3 run directory does not exist: {v3_dir}")
    _require(v2_dir != v3_dir, "v2 and v3 run directories must be distinct")
    v2_artifacts = _artifact_hashes(v2_dir, RUN_ARTIFACTS)
    v3_artifacts = _artifact_hashes(v3_dir, (*RUN_ARTIFACTS, "timestamp-order.json"))
    for name in BYTE_IDENTICAL_ARTIFACTS:
        _require(v2_artifacts[name] == v3_artifacts[name], f"model or health artifact is not byte-identical: {name}")
    v2_run = _read_json(v2_dir / "run.json")
    v3_run = _read_json(v3_dir / "run.json")
    _require(isinstance(v2_run, dict) and isinstance(v3_run, dict), "run.json must be an object")
    _assert_run_invariants(v2_run, v3_run)
    v2_visual = _visual_metadata(v2_run, label="v2")
    v3_visual = _visual_metadata(v3_run, label="v3")
    v2_visual.pop("runtime_seconds", None)
    v3_visual.pop("runtime_seconds", None)
    _require(v2_visual == v3_visual, "visual instrumentation provenance differs outside runtime")
    timestamp = _assert_timestamp_sidecar(v3_dir, v3_run)
    return {
        "schema_version": "stateguard3r.detector-v3-instrumentation-equivalence.v1",
        "status": "PASS",
        "v2_run_dir": str(v2_dir),
        "v3_run_dir": str(v3_dir),
        "frame_count": v2_run["frame_count"],
        "recurrent_update_count": v2_run["calibrated_update_calls"],
        "checks": {
            "checkpoint_load_audit_byte_identical": True,
            "health_ledger_byte_identical": True,
            "prediction_summary_byte_identical": True,
            "trajectory_byte_identical": True,
            "run_model_invariants_identical": True,
            "visual_provenance_identical_except_runtime": True,
            "v3_timestamp_sidecar_valid_and_input_bound": True,
            "v2_postflight_pid_absent": True,
            "v3_postflight_pid_absent": True,
        },
        "runner_artifacts": {"v2": v2_artifacts, "v3": v3_artifacts},
        "timestamp_order": timestamp,
        "postflight": {
            "v2": _assert_postflight(v2_postflight_log, run_pid=v2_run.get("pid"), label="v2"),
            "v3": _assert_postflight(v3_postflight_log, run_pid=v3_run.get("pid"), label="v3"),
        },
    }


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(_canonical_json(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = args.output_dir.resolve(strict=False)
    _require(not output_dir.exists(), f"refusing to overwrite existing output directory: {output_dir}")
    report = validate_instrumentation(
        v2_dir=args.v2_run,
        v3_dir=args.v3_run,
        v2_postflight_log=args.v2_postflight_log,
        v3_postflight_log=args.v3_postflight_log,
    )
    script_path = Path(__file__).resolve()
    report["validator"] = {"path": str(script_path), "sha256": _sha256(script_path)}
    _write_atomic(output_dir / "instrumentation-equivalence.json", report)
    for path in output_dir.iterdir():
        if path.is_file():
            path.chmod(0o444)
    output_dir.chmod(0o555)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
