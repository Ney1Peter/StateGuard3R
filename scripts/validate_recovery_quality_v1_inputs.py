#!/usr/bin/env python3
"""Independently validate frozen recovery-quality v1 manifests before a GPU run."""

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
CONDITIONS = ("clean", "dynamic", "wrong-order", "low-overlap")
SCHEMA_VERSION = "stateguard3r.recovery-quality-input-validation.v1"


class RecoveryQualityInputValidationError(ValueError):
    """Raised when a pre-forward quality input lacks immutable provenance."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryQualityInputValidationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, *, label: str, mode: int = 0o444) -> Path:
    details = os.lstat(path)
    _require(not stat.S_ISLNK(details.st_mode) and stat.S_ISREG(details.st_mode), f"{label} must be a regular non-symlink file")
    _require(stat.S_IMODE(details.st_mode) == mode, f"{label} must have mode {mode:04o}")
    return path.resolve(strict=True)


def _directory(path: Path, *, label: str, mode: int = 0o555) -> Path:
    details = os.lstat(path)
    _require(not stat.S_ISLNK(details.st_mode) and stat.S_ISDIR(details.st_mode), f"{label} must be a directory")
    _require(stat.S_IMODE(details.st_mode) == mode, f"{label} must have mode {mode:04o}")
    return path.resolve(strict=True)


def _json(path: Path, *, label: str) -> dict[str, Any]:
    def no_constant(value: str) -> None:
        raise RecoveryQualityInputValidationError(f"{label} has non-finite JSON {value!r}")

    def no_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise RecoveryQualityInputValidationError(f"{label} repeats key {key!r}")
            output[key] = value
        return output

    value = json.loads(path.read_bytes(), parse_constant=no_constant, object_pairs_hook=no_duplicate)
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _artifact(artifact: object, path: Path, *, label: str) -> None:
    _require(isinstance(artifact, Mapping), f"{label} artifact is absent")
    expected = _regular(path, label=label)
    _require(artifact.get("path") == str(expected), f"{label} published path differs")
    _require(artifact.get("sha256") == _sha256(expected), f"{label} SHA-256 differs")
    _require(artifact.get("size_bytes") == expected.stat().st_size, f"{label} size differs")
    _require(artifact.get("mode_octal") == "0444", f"{label} mode differs")


def _timestamp_violations(manifest: Path, dataset_root: Path) -> list[int]:
    sys.path.insert(0, str(ROOT / "src"))
    from stateguard3r.input_manifest import load_input_manifest
    from stateguard3r.timestamp_order_v3 import capture_timestamp_records, timestamp_order_sidecar, validate_timestamp_order_sidecar

    loaded = load_input_manifest(manifest)
    captures, provenance = capture_timestamp_records(list(loaded.frame_paths), rgb_txt=dataset_root / "rgb.txt", dataset_root=dataset_root)
    predicates = validate_timestamp_order_sidecar(timestamp_order_sidecar(captures, provenance=provenance), require_available=True)
    return [index for index, value in enumerate(predicates) if value is True]


def _validate_run(root: Path, condition: str, dataset_root: Path) -> dict[str, Any]:
    run_dir = _directory(root / condition, label=f"{condition} directory")
    source_path = _regular(run_dir / "source-manifest.json", label=f"{condition} source")
    input_path = _regular(run_dir / "input-manifest.json", label=f"{condition} input")
    source = _json(source_path, label=f"{condition} source")
    payload = _json(input_path, label=f"{condition} input")
    _require(source.get("schema_version") == "stateguard3r.recovery-quality-source.v1", f"{condition} source schema differs")
    frames = source.get("frames")
    bindings = payload.get("quality_evaluation_bindings")
    _require(isinstance(frames, list) and len(frames) == 35, f"{condition} source pool must contain 35 frames")
    _require(isinstance(bindings, list) and len(bindings) == 30, f"{condition} logical GT bindings must contain 30 frames")
    for index, binding in enumerate(bindings):
        _require(isinstance(binding, Mapping) and binding.get("frame_index") == index and binding.get("base_source_index") == index, f"{condition} GT binding index differs")
        _require(binding.get("groundtruth") == frames[index].get("quality_source_groundtruth"), f"{condition} GT binding is not the logical base frame")
    corruptions = payload.get("corruptions")
    if condition == "clean":
        _require(corruptions == [] and payload.get("recovery_quality_clean_control") is True, "clean control is not an explicit no-transform manifest")
    else:
        _require(isinstance(corruptions, list) and len(corruptions) == 1, f"{condition} must have exactly one corruption")
        expected_type = {"dynamic": "dynamic_occlusion", "wrong-order": "wrong_order_segment", "low-overlap": "low_overlap_jump"}[condition]
        _require(corruptions[0].get("type") == expected_type, f"{condition} corruption type differs")
    sys.path.insert(0, str(ROOT / "src"))
    from stateguard3r.input_manifest import load_input_manifest

    loaded = load_input_manifest(input_path)
    _require(len(loaded.frames) == 30, f"{condition} strict input replay frame count differs")
    for frame in loaded.frames:
        try:
            frame.path.relative_to(dataset_root)
        except ValueError as error:
            raise RecoveryQualityInputValidationError(f"{condition} image escapes raw root") from error
        _regular(frame.path, label=f"{condition} raw RGB")
    violations = _timestamp_violations(input_path, dataset_root)
    _require(not [index for index in violations if index < 15], f"{condition} clean prefix has timestamp violation")
    if condition == "wrong-order":
        _require(violations, "wrong-order input has no capture-timestamp violation")
    return {"condition": condition, "input_manifest": {"path": str(input_path), "sha256": _sha256(input_path)}, "strict_replay": True, "timestamp_order_violation_positions": violations, "gt_binding": "logical_base_only_posthoc"}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate(input_root: Path, dataset_root: Path, output_dir: Path) -> Path:
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "input validation requires CUDA_VISIBLE_DEVICES='' ")
    root = _directory(input_root, label="quality input root")
    raw = _directory(dataset_root, label="raw dataset root")
    _regular(raw / "rgb.txt", label="raw rgb listing")
    registry_path = _regular(root / "recovery-quality-inputs.json", label="quality registry")
    registry = _json(registry_path, label="quality registry")
    _require(registry.get("status") == "pre_forward_recovery_quality_inputs" and registry.get("dataset_root") == str(raw), "quality registry identity differs")
    rows = registry.get("runs")
    _require(isinstance(rows, list) and [row.get("condition") for row in rows if isinstance(row, Mapping)] == list(CONDITIONS), "quality registry condition order differs")
    for row in rows:
        condition = str(row["condition"])
        _artifact(row.get("source_manifest"), root / condition / "source-manifest.json", label=f"{condition} source")
        _artifact(row.get("input_manifest"), root / condition / "input-manifest.json", label=f"{condition} input")
    output = output_dir.resolve(strict=False)
    _require(output.parent == OUTPUT_ROOT and not output.exists(), "validation output must be a new direct child of outputs")
    report = {"schema_version": SCHEMA_VERSION, "status": "PASS", "input_registry": {"path": str(registry_path), "sha256": _sha256(registry_path)}, "dataset_root": str(raw), "runs": [_validate_run(root, condition, raw) for condition in CONDITIONS]}
    output.mkdir()
    _write_json(output / "validation.json", report)
    (output / "validation.json").chmod(0o444)
    output.chmod(0o555)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    output = validate(args.input_root, args.dataset_root, args.output_dir)
    print(json.dumps({"output_dir": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
