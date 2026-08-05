#!/usr/bin/env python3
"""Evaluate one frozen baseline/control/policy triple using GT only post hoc."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs"
SCHEMA_VERSION = "stateguard3r.recovery-quality-evaluation.v1"
PREFIX = tuple(range(15))
TAIL = tuple(range(19, 30))
EQUIVALENCE_FILES = ("checkpoint-load-audit.json", "health.jsonl", "predictions-summary.json", "trajectory.json")


class RecoveryQualityEvaluationError(ValueError):
    """Raised when a formal quality triple is not exactly comparable."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryQualityEvaluationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, *, label: str, mode: int | None = None) -> Path:
    try:
        details = os.lstat(path)
    except OSError as error:
        raise RecoveryQualityEvaluationError(f"cannot stat {label}: {error}") from error
    _require(not stat.S_ISLNK(details.st_mode) and stat.S_ISREG(details.st_mode), f"{label} must be a regular non-symlink file")
    if mode is not None:
        _require(stat.S_IMODE(details.st_mode) == mode, f"{label} must have mode {mode:04o}")
    return path.resolve(strict=True)


def _directory(path: Path, *, label: str, mode: int = 0o555) -> Path:
    try:
        details = os.lstat(path)
    except OSError as error:
        raise RecoveryQualityEvaluationError(f"cannot stat {label}: {error}") from error
    _require(not stat.S_ISLNK(details.st_mode) and stat.S_ISDIR(details.st_mode), f"{label} must be a directory")
    _require(stat.S_IMODE(details.st_mode) == mode, f"{label} must have mode {mode:04o}")
    return path.resolve(strict=True)


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise RecoveryQualityEvaluationError(f"{label} contains non-finite JSON {value!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise RecoveryQualityEvaluationError(f"{label} repeats key {key!r}")
            output[key] = value
        return output

    try:
        payload = json.loads(path.read_bytes(), parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryQualityEvaluationError(f"cannot parse {label}: {error}") from error
    _require(isinstance(payload, dict), f"{label} must be an object")
    return payload


def _artifact(path: Path) -> dict[str, Any]:
    regular = _regular(path, label=str(path), mode=0o444)
    return {"path": str(regular), "sha256": _sha256(regular), "size_bytes": regular.stat().st_size, "mode_octal": "0444"}


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


def _run_bindings(run_dir: Path, manifest: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    run = _strict_json(_regular(run_dir / "run.json", label=f"{label} run", mode=0o444), label=f"{label} run")
    binding = run.get("input_manifest")
    _require(isinstance(binding, Mapping), f"{label} run lacks input manifest binding")
    _require(binding.get("path") == str(manifest) and binding.get("sha256") == _sha256(manifest), f"{label} run is not bound to this input manifest")
    trajectory_path = _regular(run_dir / "trajectory.json", label=f"{label} trajectory", mode=0o444)
    return run, _strict_json(trajectory_path, label=f"{label} trajectory")


def _gt_arrays(manifest_payload: Mapping[str, Any]) -> tuple[Any, Any]:
    bindings = manifest_payload.get("quality_evaluation_bindings")
    _require(isinstance(bindings, list) and len(bindings) == 30, "quality GT bindings must contain exactly 30 rows")
    positions: list[list[float]] = []
    rotations: list[Any] = []
    sys.path.insert(0, str(ROOT / "src"))
    from stateguard3r.recovery_quality import quaternion_xyzw_to_rotation

    for index, row in enumerate(bindings):
        _require(isinstance(row, Mapping) and row.get("frame_index") == index and row.get("base_source_index") == index, "quality GT binding frame identity differs")
        gt = row.get("groundtruth")
        _require(isinstance(gt, Mapping), "quality GT binding lacks groundtruth")
        position = gt.get("position_xyz")
        quaternion = gt.get("orientation_xyzw")
        _require(isinstance(position, list) and len(position) == 3 and isinstance(quaternion, list) and len(quaternion) == 4, "quality GT pose schema differs")
        positions.append([float(value) for value in position])
        rotations.append(quaternion_xyzw_to_rotation(quaternion))
    import numpy as np

    return np.asarray(positions, dtype=np.float64), np.stack(rotations)


def evaluate(input_manifest: Path, baseline_dir: Path, always_commit_dir: Path, policy_dir: Path, output_dir: Path) -> Path:
    """Write one immutable triple comparison after validating every binding."""

    manifest = _regular(input_manifest, label="input manifest", mode=0o444)
    baseline = _directory(baseline_dir, label="baseline output")
    always = _directory(always_commit_dir, label="always-commit output")
    policy = _directory(policy_dir, label="detector-policy output")
    output = output_dir.resolve(strict=False)
    _require(output.parent == OUTPUT_ROOT and not output.exists(), "output must be a new direct child of StateGuard3R/outputs")
    manifest_payload = _strict_json(manifest, label="input manifest")
    _require(manifest_payload.get("source_is_read_only") is True, "input source is not read-only")
    base_run, base_trajectory = _run_bindings(baseline, manifest, label="baseline")
    always_run, always_trajectory = _run_bindings(always, manifest, label="always-commit")
    policy_run, policy_trajectory = _run_bindings(policy, manifest, label="detector-policy")
    _require(always_run.get("state_policy", {}).get("name") == "always-commit", "always control has wrong policy")
    _require(policy_run.get("state_policy", {}).get("name") == "detector-v3-quality-prior-alarm", "policy run has wrong state policy")
    equivalence: dict[str, bool] = {}
    for name in EQUIVALENCE_FILES:
        left = _regular(baseline / name, label=f"baseline {name}", mode=0o444)
        right = _regular(always / name, label=f"always-commit {name}", mode=0o444)
        equivalence[name] = _sha256(left) == _sha256(right)
    _require(all(equivalence.values()), "baseline and always-commit are not byte-equivalent")
    gt_positions, gt_rotations = _gt_arrays(manifest_payload)
    sys.path.insert(0, str(ROOT / "src"))
    from stateguard3r.recovery_quality import evaluate_tail, positive_effect, trajectory_arrays

    def metrics(trajectory: Mapping[str, Any]) -> Any:
        predicted_positions, predicted_rotations = trajectory_arrays(trajectory)
        return evaluate_tail(predicted_positions, predicted_rotations, gt_positions, gt_rotations, prefix_frame_ids=PREFIX, tail_frame_ids=TAIL)

    base_metrics = metrics(base_trajectory)
    always_metrics = metrics(always_trajectory)
    policy_metrics = metrics(policy_trajectory)
    effects = {
        "ATE_RMSE_effect": positive_effect(base_metrics.ate_rmse_m, policy_metrics.ate_rmse_m, name="ATE"),
        "RPE_translation_RMSE_effect": positive_effect(base_metrics.rpe_translation_rmse_m, policy_metrics.rpe_translation_rmse_m, name="translation RPE"),
    }
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".staging", dir=output.parent))
    try:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "input_manifest": _artifact(manifest),
            "runs": {"baseline": _artifact(baseline / "run.json"), "always_commit": _artifact(always / "run.json"), "detector_policy": _artifact(policy / "run.json")},
            "baseline_always_commit_byte_equivalence": equivalence,
            "metrics": {"baseline": base_metrics.to_dict(), "always_commit": always_metrics.to_dict(), "detector_policy": policy_metrics.to_dict()},
            "effects": effects,
            "prefix_only_alignment": {"frames": list(PREFIX), "tail_frames": list(TAIL)},
        }
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
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--always-commit-dir", required=True, type=Path)
    parser.add_argument("--policy-dir", required=True, type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    output = evaluate(args.input_manifest, args.baseline_dir, args.always_commit_dir, args.policy_dir, args.output_dir)
    print(json.dumps({"output_dir": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
