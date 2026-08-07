#!/usr/bin/env python3
"""Evaluate disclosed legacy development recovery runs after inference.

This CPU-only evaluator is intentionally separate from detector/policy code.
For legacy corruption manifests, the final image at a low-overlap or reordered
position may come from another source frame.  Ground truth is therefore bound
to the corresponding logical base row (0--29) of the source manifest, never
to the final model-image donor/order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs"
SCHEMA_VERSION = "stateguard3r.development-recovery-evaluation.v1"
PREFIX = tuple(range(15))
TAIL = tuple(range(19, 30))
EQUIVALENCE_FILES = ("checkpoint-load-audit.json", "health.jsonl", "predictions-summary.json", "trajectory.json")
DEVELOPMENT_INPUT_ROOT = OUTPUT_ROOT / "formal-v1-inputs-0001" / "development"


class DevelopmentRecoveryEvaluationError(ValueError):
    """Raised when a development triple is not safely comparable."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DevelopmentRecoveryEvaluationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, *, label: str, mode: int = 0o444) -> Path:
    try:
        details = os.lstat(path)
    except OSError as error:
        raise DevelopmentRecoveryEvaluationError(f"cannot stat {label}: {error}") from error
    _require(not stat.S_ISLNK(details.st_mode) and stat.S_ISREG(details.st_mode), f"{label} must be a regular non-symlink file")
    _require(stat.S_IMODE(details.st_mode) == mode, f"{label} must have mode {mode:04o}")
    return path.resolve(strict=True)


def _directory(path: Path, *, label: str, mode: int = 0o555) -> Path:
    try:
        details = os.lstat(path)
    except OSError as error:
        raise DevelopmentRecoveryEvaluationError(f"cannot stat {label}: {error}") from error
    _require(not stat.S_ISLNK(details.st_mode) and stat.S_ISDIR(details.st_mode), f"{label} must be a directory")
    _require(stat.S_IMODE(details.st_mode) == mode, f"{label} must have mode {mode:04o}")
    return path.resolve(strict=True)


def _json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise DevelopmentRecoveryEvaluationError(f"{label} contains non-finite JSON {value!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, f"{label} repeats key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(path.read_bytes(), parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DevelopmentRecoveryEvaluationError(f"cannot parse {label}: {error}") from error
    _require(isinstance(payload, dict), f"{label} must be an object")
    return payload


def _artifact(path: Path) -> dict[str, Any]:
    regular = _regular(path, label=str(path))
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


def _development_manifest(path: Path) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    manifest = _regular(path, label="development input manifest")
    root = DEVELOPMENT_INPUT_ROOT.resolve(strict=True)
    _require(root in manifest.parents, f"development manifest must be under {root}")
    payload = _json(manifest, label="development input manifest")
    _require(payload.get("schema_version") == "stateguard3r.corruption.v1", "development manifest schema differs")
    _require(payload.get("source_is_read_only") is True, "development manifest source is not read-only")
    source_name = payload.get("source_manifest")
    _require(isinstance(source_name, str) and source_name, "development manifest lacks source manifest")
    source = _regular((manifest.parent / source_name).resolve(strict=True), label="development source manifest")
    source_payload = _json(source, label="development source manifest")
    return manifest, payload, source, source_payload


def _logical_base_gt(source_payload: Mapping[str, Any]) -> tuple[Any, Any]:
    rows = source_payload.get("frames")
    _require(isinstance(rows, list) and len(rows) >= 30, "source manifest lacks 30 logical-base frames")
    positions: list[list[float]] = []
    quaternions: list[list[float]] = []
    for index, row in enumerate(rows[:30]):
        _require(isinstance(row, Mapping), f"logical-base row {index} must be an object")
        _require(row.get("source_pool_index") == index and row.get("source_pool_role") == "base", f"logical-base row {index} is not the fixed base sequence")
        gt = row.get("groundtruth")
        _require(isinstance(gt, Mapping), f"logical-base row {index} lacks GT")
        position = gt.get("translation_xyz")
        quaternion = gt.get("quaternion_xyzw")
        _require(isinstance(position, list) and len(position) == 3, f"logical-base row {index} GT position differs")
        _require(isinstance(quaternion, list) and len(quaternion) == 4, f"logical-base row {index} GT quaternion differs")
        try:
            position_values = [float(value) for value in position]
            quaternion_values = [float(value) for value in quaternion]
        except (TypeError, ValueError) as error:
            raise DevelopmentRecoveryEvaluationError(f"logical-base row {index} GT is non-numeric") from error
        _require(all(math.isfinite(value) for value in (*position_values, *quaternion_values)), f"logical-base row {index} GT is non-finite")
        positions.append(position_values)
        quaternions.append(quaternion_values)
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    from stateguard3r.recovery_quality import quaternion_xyzw_to_rotation

    return np.asarray(positions, dtype=np.float64), np.stack([quaternion_xyzw_to_rotation(value) for value in quaternions])


def _run_and_trajectory(run_dir: Path, manifest: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = _directory(run_dir, label=f"{label} run directory")
    run_path = _regular(directory / "run.json", label=f"{label} run")
    run = _json(run_path, label=f"{label} run")
    binding = run.get("input_manifest")
    _require(isinstance(binding, Mapping), f"{label} run lacks input manifest binding")
    _require(binding.get("path") == str(manifest) and binding.get("sha256") == _sha256(manifest), f"{label} run is not bound to this input manifest")
    return run, _json(_regular(directory / "trajectory.json", label=f"{label} trajectory"), label=f"{label} trajectory")


def _metrics(trajectory: Mapping[str, Any], gt_positions: Any, gt_rotations: Any) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "src"))
    from stateguard3r.recovery_quality import evaluate_tail, trajectory_arrays

    positions, rotations = trajectory_arrays(trajectory)
    return evaluate_tail(positions, rotations, gt_positions, gt_rotations, prefix_frame_ids=PREFIX, tail_frame_ids=TAIL).to_dict()


def _effect(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    sys.path.insert(0, str(ROOT / "src"))
    from stateguard3r.recovery_quality import positive_effect

    return {
        "ATE_RMSE_effect": positive_effect(float(baseline["ATE_RMSE_m"]), float(candidate["ATE_RMSE_m"]), name="ATE"),
        "RPE_translation_RMSE_effect": positive_effect(float(baseline["RPE_translation_RMSE_m"]), float(candidate["RPE_translation_RMSE_m"]), name="translation RPE"),
    }


def _discard_evidence(directory: Path) -> dict[str, Any]:
    timeline = _json(_regular(directory / "state-timeline.json", label="discard state timeline"), label="discard state timeline")
    _require(timeline.get("release_mode") == "discard", "discard run does not record discard release mode")
    rows = timeline.get("transactions")
    _require(isinstance(rows, list), "discard timeline lacks transactions")
    releases: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("action") != "release_discard_then_commit":
            continue
        frame = row.get("frame_id")
        held = row.get("discarded_transaction_frame_id")
        _require(type(frame) is int and type(held) is int and 0 <= held < frame < len(rows), "discard release IDs are invalid")
        held_row = rows[held]
        _require(isinstance(held_row, Mapping) and held_row.get("action") == "hold", "discard release does not reference a held candidate")
        _require(row.get("pre_state_digest_sha256") != held_row.get("proposed_state_digest_sha256"), "discard clear frame reinstalled held post-state")
        releases.append({"release_frame_id": frame, "discarded_frame_id": held})
    discarded = timeline.get("discarded_transaction_frame_ids")
    _require(isinstance(discarded, list), "discard timeline lacks discarded IDs")
    _require(discarded == [entry["discarded_frame_id"] for entry in releases], "discarded ID list differs from release evidence")
    return {"release_count": len(releases), "releases": releases, "post_discard_state_differs_from_held_post_state": bool(releases)}


def evaluate(input_manifest: Path, baseline_dir: Path, always_dir: Path, replay_dir: Path, discard_dir: Path, output_dir: Path) -> Path:
    """Freeze one disclosed development comparison; GT is read only here."""

    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "development evaluation requires CUDA_VISIBLE_DEVICES='' ")
    manifest, _manifest_payload, source_manifest, source_payload = _development_manifest(input_manifest)
    output = output_dir.resolve(strict=False)
    _require(output.parent == OUTPUT_ROOT and not output.exists(), "output must be a new direct child of StateGuard3R/outputs")
    baseline, baseline_trajectory = _run_and_trajectory(baseline_dir, manifest, label="baseline")
    always, always_trajectory = _run_and_trajectory(always_dir, manifest, label="always-commit")
    replay, replay_trajectory = _run_and_trajectory(replay_dir, manifest, label="replay")
    discard, discard_trajectory = _run_and_trajectory(discard_dir, manifest, label="discard")
    _require(always.get("state_policy", {}).get("name") == "always-commit", "always control has wrong policy")
    _require(replay.get("state_policy", {}).get("name") == "detector-v3-quality-prior-alarm", "replay has wrong policy")
    _require(discard.get("state_policy", {}).get("name") == "detector-v3-quality-prior-alarm-discard", "discard has wrong policy")
    equivalence: dict[str, bool] = {}
    for name in EQUIVALENCE_FILES:
        equivalence[name] = _sha256(_regular(Path(baseline_dir) / name, label=f"baseline {name}")) == _sha256(_regular(Path(always_dir) / name, label=f"always {name}"))
    _require(all(equivalence.values()), "baseline and always-commit are not byte-equivalent")
    gt_positions, gt_rotations = _logical_base_gt(source_payload)
    metrics = {
        "baseline": _metrics(baseline_trajectory, gt_positions, gt_rotations),
        "always_commit": _metrics(always_trajectory, gt_positions, gt_rotations),
        "replay": _metrics(replay_trajectory, gt_positions, gt_rotations),
        "discard": _metrics(discard_trajectory, gt_positions, gt_rotations),
    }
    discard_evidence = _discard_evidence(Path(discard_dir))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scope": "disclosed development-only post-hoc evaluation; not formal recovery evidence",
        "input_manifest": _artifact(manifest),
        "logical_base_source_manifest": _artifact(source_manifest),
        "gt_binding": "source-manifest rows 0--29 with source_pool_role=base; never final donor/reordered frame metadata",
        "runs": {"baseline": _artifact(Path(baseline_dir) / "run.json"), "always_commit": _artifact(Path(always_dir) / "run.json"), "replay": _artifact(Path(replay_dir) / "run.json"), "discard": _artifact(Path(discard_dir) / "run.json")},
        "baseline_always_commit_byte_equivalence": equivalence,
        "metrics": metrics,
        "effects": {"replay_vs_baseline": _effect(metrics["baseline"], metrics["replay"]), "discard_vs_baseline": _effect(metrics["baseline"], metrics["discard"])},
        "runtime_ratios": {"replay_to_baseline": float(replay["runtime_seconds"]) / float(baseline["runtime_seconds"]), "discard_to_baseline": float(discard["runtime_seconds"]) / float(baseline["runtime_seconds"])},
        "discard_evidence": discard_evidence,
        "trajectory_byte_different_from_replay": _sha256(_regular(Path(discard_dir) / "trajectory.json", label="discard trajectory")) != _sha256(_regular(Path(replay_dir) / "trajectory.json", label="replay trajectory")),
        "prefix_only_alignment": {"frames": list(PREFIX), "tail_frames": list(TAIL)},
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
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--always-commit-dir", required=True, type=Path)
    parser.add_argument("--replay-dir", required=True, type=Path)
    parser.add_argument("--discard-dir", required=True, type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    output = evaluate(args.input_manifest, args.baseline_dir, args.always_commit_dir, args.replay_dir, args.discard_dir, args.output_dir)
    print(json.dumps({"output_dir": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
