#!/usr/bin/env python3
"""Validate that the v2 visual hook left ReCal3R inference unchanged.

This validator is deliberately CPU-only.  It compares two completed smoke-run
directories made from one frozen input manifest: a legacy v1 profile run and a
v2 profile run.  Its only allowed output is a new, atomically-published audit
directory, so neither runner result can be altered during validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


RUN_ARTIFACTS = (
    "checkpoint-load-audit.json",
    "health.jsonl",
    "predictions-summary.json",
    "run.json",
    "trajectory.json",
)
BYTE_IDENTICAL_ARTIFACTS = (
    "checkpoint-load-audit.json",
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
    """Raised when the two run directories do not meet the v2 contract."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v1_run", type=Path)
    parser.add_argument("v2_run", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--v1-postflight-log", required=True, type=Path)
    parser.add_argument("--v2-postflight-log", required=True, type=Path)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InstrumentationValidationError(f"cannot read JSON {path}: {error}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise InstrumentationValidationError(f"cannot read JSONL {path}: {error}") from error
    if not lines:
        raise InstrumentationValidationError(f"health ledger is empty: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise InstrumentationValidationError(
                f"cannot parse {path}:{line_number}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise InstrumentationValidationError(
                f"health record {path}:{line_number} must be an object"
            )
        records.append(payload)
    return records


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InstrumentationValidationError(message)


def _assert_regular_file(path: Path, *, label: str) -> None:
    _require(path.is_file(), f"{label} is not a regular file: {path}")


def _artifact_hashes(run_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in RUN_ARTIFACTS:
        path = run_dir / name
        _assert_regular_file(path, label=f"runner artifact {name}")
        result[name] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    return result


def _assert_run_invariants(v1_run: Mapping[str, Any], v2_run: Mapping[str, Any]) -> None:
    _require(v1_run.get("status") == "succeeded", "v1 run did not succeed")
    _require(v2_run.get("status") == "succeeded", "v2 run did not succeed")
    _require(
        v1_run.get("health_profile", "v1") == "v1",
        "v1 run contains an unexpected health profile",
    )
    _require(v2_run.get("health_profile") == "v2", "v2 run does not declare profile v2")
    for key in RUN_INVARIANT_FIELDS:
        _require(key in v1_run and key in v2_run, f"run invariant is missing: {key}")
        _require(v1_run[key] == v2_run[key], f"run invariant differs: {key}")

    frame_count = v1_run["frame_count"]
    _require(
        isinstance(frame_count, int) and not isinstance(frame_count, bool) and frame_count >= 2,
        "frame_count must be an integer of at least two",
    )
    _require(
        v1_run["calibrated_update_calls"] == frame_count - 1,
        "v1 recurrent update count does not equal frame_count - 1",
    )
    _require(
        v2_run["calibrated_update_calls"] == frame_count - 1,
        "v2 recurrent update count does not equal frame_count - 1",
    )


def _assert_health_equivalence(
    v1_records: Sequence[Mapping[str, Any]], v2_records: Sequence[Mapping[str, Any]]
) -> None:
    _require(len(v1_records) == len(v2_records), "v1/v2 health ledger lengths differ")
    _require(len(v1_records) >= 2, "health ledger must contain at least two frames")
    for index, (v1_record, v2_record) in enumerate(zip(v1_records, v2_records)):
        _require(v1_record.get("frame_id") == index, f"v1 frame id mismatch at {index}")
        _require(v2_record.get("frame_id") == index, f"v2 frame id mismatch at {index}")
        _require(v1_record.get("overlap") is None, f"v1 overlap must be null at frame {index}")
        v1_without_overlap = {key: value for key, value in v1_record.items() if key != "overlap"}
        v2_without_overlap = {key: value for key, value in v2_record.items() if key != "overlap"}
        _require(
            v1_without_overlap == v2_without_overlap,
            f"health fields other than overlap differ at frame {index}",
        )
        overlap = v2_record.get("overlap")
        if index == 0:
            _require(overlap is None, "v2 frame 0 overlap must be null")
        else:
            _require(
                isinstance(overlap, (int, float))
                and not isinstance(overlap, bool)
                and math.isfinite(float(overlap))
                and 0.0 <= float(overlap) <= 1.0,
                f"v2 overlap must be finite in [0, 1] at frame {index}",
            )


def _assert_visual_provenance(
    v2_run: Mapping[str, Any], v2_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    metadata = v2_run.get("online_visual_correspondence")
    _require(isinstance(metadata, dict), "v2 visual correspondence metadata is missing")
    _require(
        metadata.get("schema_version") == "stateguard3r.visual-overlap.v1",
        "unexpected visual correspondence schema",
    )
    _require(
        metadata.get("input") == "post_official_loader_post_deferred_transform_normalized_rgb",
        "visual correspondence was not computed from post-loader model-ready RGB",
    )
    _require(
        metadata.get("causal_reference") == "immediately_previous_frame_only",
        "visual reference is not causal",
    )
    image_hashes = metadata.get("model_ready_uint8_rgb_sha256")
    _require(
        isinstance(image_hashes, list) and len(image_hashes) == len(v2_records),
        "v2 model-ready RGB hash count differs from frame count",
    )
    _require(
        all(isinstance(item, str) and len(item) == 64 for item in image_hashes),
        "v2 model-ready RGB hashes are malformed",
    )
    opencv = metadata.get("opencv")
    _require(isinstance(opencv, dict), "OpenCV provenance is missing")
    _require(opencv.get("threads") == 1, "OpenCV threads must be pinned to one")
    _require(opencv.get("opencl_enabled") is False, "OpenCL must be disabled")
    _require(opencv.get("rng_seed") == 0, "OpenCV RNG seed must be zero")
    for key in ("version", "module_sha256", "binary_sha256", "build_info_sha256"):
        _require(isinstance(opencv.get(key), str) and opencv[key], f"OpenCV provenance missing {key}")

    diagnostics = metadata.get("frame_results")
    _require(isinstance(diagnostics, list), "v2 visual frame diagnostics are missing")
    _require(
        len(diagnostics) == len(v2_records) - 1,
        "v2 visual diagnostic count must equal frame_count - 1",
    )
    for index, diagnostic in enumerate(diagnostics, start=1):
        _require(isinstance(diagnostic, dict), f"v2 visual diagnostic {index} is not an object")
        _require(diagnostic.get("frame_id") == index, f"v2 visual diagnostic frame id mismatch at {index}")
        _require(
            diagnostic.get("reference_frame_id") == index - 1,
            f"v2 visual diagnostic reference is not previous frame at {index}",
        )
        _require(
            diagnostic.get("score") == v2_records[index].get("overlap"),
            f"v2 visual diagnostic score differs from health overlap at {index}",
        )
    runtime = metadata.get("runtime_seconds")
    _require(
        isinstance(runtime, (int, float))
        and not isinstance(runtime, bool)
        and math.isfinite(float(runtime))
        and float(runtime) >= 0.0,
        "v2 visual overlap runtime is invalid",
    )
    return {
        "schema_version": metadata["schema_version"],
        "input": metadata["input"],
        "causal_reference": metadata["causal_reference"],
        "runtime_seconds": float(runtime),
        "opencv": {
            key: opencv[key]
            for key in (
                "version",
                "module_sha256",
                "binary_sha256",
                "build_info_sha256",
                "threads",
                "opencl_enabled",
                "rng_seed",
            )
        },
    }


def _assert_postflight(log_path: Path, *, run_pid: Any, label: str) -> dict[str, Any]:
    _assert_regular_file(log_path, label=f"{label} postflight log")
    _require(isinstance(run_pid, int) and not isinstance(run_pid, bool), f"{label} run PID is invalid")
    text = log_path.read_text(encoding="utf-8")
    _require("NVIDIA-SMI" in text, f"{label} postflight log has no nvidia-smi snapshot")
    _require(
        str(run_pid) not in text,
        f"{label} run PID {run_pid} is still listed in its postflight log",
    )
    return {
        "path": str(log_path.resolve()),
        "sha256": _sha256(log_path),
        "size_bytes": log_path.stat().st_size,
        "run_pid_absent": True,
    }


def validate_instrumentation(
    *,
    v1_dir: Path,
    v2_dir: Path,
    v1_postflight_log: Path,
    v2_postflight_log: Path,
) -> dict[str, Any]:
    """Read and validate completed v1/v2 artifacts without modifying them."""

    v1_dir = v1_dir.resolve()
    v2_dir = v2_dir.resolve()
    _require(v1_dir.is_dir(), f"v1 run directory does not exist: {v1_dir}")
    _require(v2_dir.is_dir(), f"v2 run directory does not exist: {v2_dir}")
    _require(v1_dir != v2_dir, "v1 and v2 run directories must be distinct")
    v1_artifacts = _artifact_hashes(v1_dir)
    v2_artifacts = _artifact_hashes(v2_dir)
    for name in BYTE_IDENTICAL_ARTIFACTS:
        _require(
            v1_artifacts[name] == v2_artifacts[name],
            f"model artifact is not byte-identical: {name}",
        )
    v1_run = _read_json(v1_dir / "run.json")
    v2_run = _read_json(v2_dir / "run.json")
    _require(isinstance(v1_run, dict) and isinstance(v2_run, dict), "run.json must be an object")
    _assert_run_invariants(v1_run, v2_run)
    v1_health = _read_jsonl(v1_dir / "health.jsonl")
    v2_health = _read_jsonl(v2_dir / "health.jsonl")
    _assert_health_equivalence(v1_health, v2_health)
    visual = _assert_visual_provenance(v2_run, v2_health)
    return {
        "schema_version": "stateguard3r.detector-v2-instrumentation-equivalence.v1",
        "status": "PASS",
        "v1_run_dir": str(v1_dir),
        "v2_run_dir": str(v2_dir),
        "frame_count": len(v1_health),
        "recurrent_update_count": v1_run["calibrated_update_calls"],
        "checks": {
            "checkpoint_load_audit_byte_identical": True,
            "prediction_summary_byte_identical": True,
            "trajectory_byte_identical": True,
            "run_model_invariants_identical": True,
            "health_fields_except_overlap_identical": True,
            "v1_overlap_all_null": True,
            "v2_overlap_null_then_finite": True,
            "v2_causal_visual_provenance_valid": True,
            "v1_postflight_pid_absent": True,
            "v2_postflight_pid_absent": True,
        },
        "runner_artifacts": {"v1": v1_artifacts, "v2": v2_artifacts},
        "visual": visual,
        "postflight": {
            "v1": _assert_postflight(
                v1_postflight_log, run_pid=v1_run.get("pid"), label="v1"
            ),
            "v2": _assert_postflight(
                v2_postflight_log, run_pid=v2_run.get("pid"), label="v2"
            ),
        },
    }


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
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
        v1_dir=args.v1_run,
        v2_dir=args.v2_run,
        v1_postflight_log=args.v1_postflight_log,
        v2_postflight_log=args.v2_postflight_log,
    )
    script_path = Path(__file__).resolve()
    report["validator"] = {
        "path": str(script_path),
        "sha256": _sha256(script_path),
    }
    _write_atomic(output_dir / "instrumentation-equivalence.json", report)
    for path in output_dir.iterdir():
        if path.is_file():
            path.chmod(0o444)
    output_dir.chmod(0o555)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
