#!/usr/bin/env python3
"""CPU-only validation and immutable read lock for formal-v2 holdout inputs.

This verifier intentionally reads only frozen manifests and their raw RGB/depth
sources.  It neither imports the ReCal3R model nor opens a formal run output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from stateguard3r.input_manifest import load_input_manifest


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
RAW = (
    ROOT.parent
    / "baselines"
    / "ReCal3R"
    / "data"
    / "tum"
    / "rgbd_dataset_freiburg3_walking_static"
)
RUNS = (
    ("holdout-dynamic", "dynamic_occlusion", 5),
    ("holdout-wrong", "wrong_order_segment", 4),
    ("holdout-low", "low_overlap_jump", 5),
)
SCHEMA_VERSION = "stateguard3r.formal-v2-input-cpu-validation.v2"


class V2InputValidationError(ValueError):
    """Raised when frozen formal-v2 input evidence is incomplete or altered."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V2InputValidationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise V2InputValidationError(f"{path} contains non-finite JSON {value!r}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V2InputValidationError(f"{path} has duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_bytes(),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V2InputValidationError(f"cannot parse strict JSON {path}: {error}") from error
    _require(isinstance(value, Mapping), f"{path} must contain a JSON object")
    return value


def _frozen_regular(path: Path, *, mode: int, label: str) -> None:
    metadata = os.lstat(path)
    _require(stat.S_ISREG(metadata.st_mode), f"{label} is not a regular file")
    _require(not stat.S_ISLNK(metadata.st_mode), f"{label} is a symlink")
    _require(stat.S_IMODE(metadata.st_mode) == mode, f"{label} has unexpected mode")


def _raw_file(path: Path, expected_sha256: object, *, label: str) -> None:
    _require(isinstance(expected_sha256, str) and len(expected_sha256) == 64, f"{label} digest is invalid")
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or symlinked")
    _require(RAW in path.resolve(strict=True).parents, f"{label} is outside raw root")
    _require(_sha256(path) == expected_sha256, f"{label} content digest mismatch")


def _validate_run(
    root: Path,
    row: Mapping[str, Any],
    *,
    run_id: str,
    corruption_type: str,
    event_length: int,
    seen_rgb: set[str],
    seen_depth: set[str],
) -> dict[str, Any]:
    _require(row.get("run_id") == run_id, f"{run_id} registry identity mismatch")
    _require(row.get("dataset_split") == "holdout", f"{run_id} is not holdout")
    _require(row.get("corruption_type") == corruption_type, f"{run_id} corruption mismatch")
    directory = root / run_id
    _require(directory.is_dir() and not directory.is_symlink(), f"{run_id} directory missing")
    _require(stat.S_IMODE(os.lstat(directory).st_mode) == 0o555, f"{run_id} directory is not frozen")
    source_path = directory / "source-manifest.json"
    manifest_path = directory / "input-manifest.json"
    _frozen_regular(source_path, mode=0o444, label=f"{run_id} source manifest")
    _frozen_regular(manifest_path, mode=0o444, label=f"{run_id} input manifest")
    artifacts = row.get("artifacts")
    _require(isinstance(artifacts, Mapping), f"{run_id} artifact registry missing")
    for name, path in (("source_manifest", source_path), ("input_manifest", manifest_path)):
        artifact = artifacts.get(name)
        _require(isinstance(artifact, Mapping), f"{run_id} {name} registry missing")
        _require(artifact.get("sha256") == _sha256(path), f"{run_id} {name} registry digest")
        _require(artifact.get("size_bytes") == path.stat().st_size, f"{run_id} {name} registry size")

    source = _strict_json(source_path)
    payload = _strict_json(manifest_path)
    _require(source.get("schema_version") == "stateguard3r.tum-formal-v2-source.v1", f"{run_id} source schema")
    _require(source.get("dataset") == RAW.name and source.get("source_is_read_only") is True, f"{run_id} source identity")
    _require(source.get("run_id") == run_id and source.get("corruption_type") == corruption_type, f"{run_id} source layout")
    _require(payload.get("source_manifest") == "source-manifest.json", f"{run_id} input source path")
    _require(payload.get("source_manifest_sha256") == _sha256(source_path), f"{run_id} input source digest")
    manifest = load_input_manifest(manifest_path)
    _require(len(manifest.frames) == 30 and payload.get("frame_count") == 30, f"{run_id} frame count")
    corruptions = payload.get("corruptions")
    _require(isinstance(corruptions, list) and len(corruptions) == 1, f"{run_id} corruption layout")
    corruption = corruptions[0]
    _require(isinstance(corruption, Mapping), f"{run_id} corruption object")
    _require(corruption.get("type") == corruption_type, f"{run_id} corruption type")
    _require(corruption.get("start") == 15 and corruption.get("end") == 14 + event_length, f"{run_id} event interval")

    source_frames = source.get("frames")
    _require(isinstance(source_frames, list) and len(source_frames) in {30, 35}, f"{run_id} source frame count")
    for frame in source_frames:
        _require(isinstance(frame, Mapping), f"{run_id} source frame object")
        rgb = Path(str(frame.get("path", "")))
        depth = frame.get("depth")
        _require(isinstance(depth, Mapping), f"{run_id} depth provenance missing")
        _raw_file(rgb, frame.get("rgb_sha256"), label=f"{run_id} RGB")
        _raw_file(Path(str(depth.get("path", ""))), depth.get("sha256"), label=f"{run_id} depth")
    for frame in manifest.frames:
        _raw_file(frame.path, frame.metadata.get("rgb_sha256"), label=f"{run_id} model RGB")
        _require(frame.metadata.get("rgb_sha256") not in seen_rgb, f"{run_id} reuses RGB across blind runs")
        seen_rgb.add(str(frame.metadata["rgb_sha256"]))
        depth = frame.metadata.get("depth")
        _require(isinstance(depth, Mapping), f"{run_id} model depth provenance missing")
        digest = depth.get("sha256")
        _require(isinstance(digest, str) and digest not in seen_depth, f"{run_id} reuses depth across blind runs")
        seen_depth.add(digest)
    return {"run_id": run_id, "frame_count": len(manifest.frames), "loader_replay_passed": True}


def validate(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    _require(root.is_dir() and not root.is_symlink(), "input root missing or symlinked")
    _require(stat.S_IMODE(os.lstat(root).st_mode) == 0o555, "input root is not frozen")
    registry_path = root / "formal-v2-manifest.json"
    _frozen_regular(registry_path, mode=0o444, label="input registry")
    registry = _strict_json(registry_path)
    _require(registry.get("schema_version") == "stateguard3r.formal-v2-inputs.v1", "input schema")
    _require(registry.get("status") == "pre_forward_blind_holdout_inputs", "input status")
    proof = registry.get("cross_scene_proof")
    _require(isinstance(proof, Mapping) and proof.get("disjoint") is True, "cross-scene proof")
    _require(proof.get("development_dataset") == "rgbd_dataset_freiburg1_desk", "development dataset")
    _require(proof.get("holdout_dataset") == RAW.name, "holdout dataset")
    rows = registry.get("runs")
    _require(isinstance(rows, list) and len(rows) == len(RUNS), "run registry layout")
    by_id = {row.get("run_id"): row for row in rows if isinstance(row, Mapping)}
    _require(len(by_id) == len(RUNS), "duplicate or non-object run registry row")
    seen_rgb: set[str] = set()
    seen_depth: set[str] = set()
    checked = [
        _validate_run(
            root,
            by_id[run_id],
            run_id=run_id,
            corruption_type=kind,
            event_length=length,
            seen_rgb=seen_rgb,
            seen_depth=seen_depth,
        )
        for run_id, kind, length in RUNS
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "input_root": str(root),
        "input_registry_sha256": _sha256(registry_path),
        "runs": checked,
        "checks": {
            "cuda_hidden": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "strict_loader_replay": True,
            "raw_rgb_and_depth_digest_replay": True,
            "cross_scene_disjoint": True,
            "no_holdout_model_output_read": True,
        },
    }


def _freeze(directory: Path) -> None:
    for path in sorted(directory.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    directory.chmod(0o555)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve(strict=False)
    try:
        _require(OUTPUTS == output.parent or OUTPUTS in output.parents, "validation output must stay under outputs")
        _require(not output.exists(), "validation output already exists")
        _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA_VISIBLE_DEVICES must be empty")
        report = validate(args.input_root)
        report["validator"] = {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        }
        output.mkdir(parents=False)
        (output / "cpu-validation.json").write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _freeze(output)
    except (OSError, V2InputValidationError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
