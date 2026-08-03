#!/usr/bin/env python3
"""CPU-only validator for immutable formal-v3 blind input manifests.

It reads only a prepared input root and the read-only formal-v3 raw scene.  In
particular, it does not import torch/ReCal3R, construct visual overlap, open a
runner output, or inspect a model response.  The validator replays each final
frame's binding to the exact raw ``rgb.txt`` line, decimal capture timestamp,
and RGB content hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_formal_detection_v3_inputs as prepare


SCHEMA_VERSION = "stateguard3r.formal-v3-input-cpu-validation.v1"


class FormalV3InputValidationError(ValueError):
    """Raised when a blind v3 input, raw source, or provenance binding differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalV3InputValidationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise FormalV3InputValidationError(f"{label} contains non-finite JSON {value!r}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FormalV3InputValidationError(f"{label} has duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_bytes(),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalV3InputValidationError(f"cannot parse {label}: {error}") from error
    _require(isinstance(value, dict), f"{label} must contain a JSON object")
    return value


def _regular(path: Path, *, label: str, mode: int | None = None) -> Path:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise FormalV3InputValidationError(f"cannot stat {label}: {error}") from error
    _require(not stat.S_ISLNK(metadata.st_mode), f"{label} may not be a symlink")
    _require(stat.S_ISREG(metadata.st_mode), f"{label} must be a regular file")
    if mode is not None:
        _require(stat.S_IMODE(metadata.st_mode) == mode, f"{label} must have mode {mode:04o}")
    return path.resolve(strict=True)


def _directory(path: Path, *, label: str, mode: int | None = None) -> Path:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise FormalV3InputValidationError(f"cannot stat {label}: {error}") from error
    _require(not stat.S_ISLNK(metadata.st_mode), f"{label} may not be a symlink")
    _require(stat.S_ISDIR(metadata.st_mode), f"{label} must be a directory")
    if mode is not None:
        _require(stat.S_IMODE(metadata.st_mode) == mode, f"{label} must have mode {mode:04o}")
    return path.resolve(strict=True)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _verify_artifact(
    artifact: object,
    *,
    expected_path: Path,
    label: str,
    require_read_only: bool = True,
) -> dict[str, Any]:
    _require(isinstance(artifact, Mapping), f"{label} artifact is missing")
    required = {"path", "sha256", "size_bytes"}
    allowed = required | {"mode_octal"}
    _require(set(artifact) <= allowed and required <= set(artifact), f"{label} artifact schema is invalid")
    actual_path = _regular(expected_path, label=label, mode=0o444 if require_read_only else None)
    _require(artifact.get("path") == str(actual_path), f"{label} path binding differs")
    _require(artifact.get("sha256") == _sha256(actual_path), f"{label} SHA-256 binding differs")
    _require(artifact.get("size_bytes") == actual_path.stat().st_size, f"{label} size binding differs")
    if "mode_octal" in artifact:
        _require(artifact["mode_octal"] == "0444", f"{label} mode binding differs")
    return {"path": str(actual_path), "sha256": _sha256(actual_path), "size_bytes": actual_path.stat().st_size}


def _expected_source_indices(corruption_type: str) -> list[int]:
    return prepare._expected_source_indices(corruption_type)


def _replay_timestamp_sidecar(
    source: Mapping[str, Any],
    *,
    run_id: str,
) -> tuple[list[int], list[int]]:
    """Replay the online timestamp capability from ordered final RGB paths only."""

    state_src = ROOT / "src"
    if str(state_src) not in sys.path:
        sys.path.insert(0, str(state_src))
    from stateguard3r.timestamp_order_v3 import (
        capture_timestamp_records,
        timestamp_order_sidecar,
        validate_timestamp_order_sidecar,
    )

    bindings = source["final_frame_to_raw_rgb_bindings"]
    frames = source["frames"]
    paths = [Path(str(frames[binding["source_pool_index"]]["path"])) for binding in bindings]
    captures, provenance = capture_timestamp_records(
        paths,
        rgb_txt=prepare.RAW_ROOT / "rgb.txt",
        dataset_root=prepare.RAW_ROOT,
    )
    sidecar = timestamp_order_sidecar(captures, provenance=provenance)
    predicates = validate_timestamp_order_sidecar(sidecar)
    records = sidecar["records"]
    _require(len(records) == prepare.FRAME_COUNT, f"{run_id} timestamp sidecar frame count differs")
    _require(predicates[0] is None, f"{run_id} frame zero timestamp predicate differs")
    for frame_index, (record, binding) in enumerate(zip(records, bindings, strict=True)):
        _require(record["frame_id"] == frame_index, f"{run_id} timestamp sidecar frame index differs")
        _require(
            record["rgb_capture_timestamp"] == binding["rgb_capture_timestamp"],
            f"{run_id} timestamp sidecar decimal binding differs",
        )
        _require(
            record["rgb_capture_timestamp_text"] == binding["rgb_capture_timestamp_text"],
            f"{run_id} timestamp sidecar text binding differs",
        )
        _require(
            record["rgb_txt_physical_line"] == binding["rgb_txt_physical_line"],
            f"{run_id} timestamp sidecar physical line differs",
        )
        _require(
            record["rgb_path_sha256"] == binding["raw_rgb_sha256"],
            f"{run_id} timestamp sidecar RGB hash binding differs",
        )
    violations = [index for index, predicate in enumerate(predicates) if predicate is True]
    clean_prefix_violations = [index for index in violations if index < prepare.EVENT_START]
    _require(
        not clean_prefix_violations,
        f"{run_id} clean prefix has timestamp-order violations at {clean_prefix_violations}",
    )
    return violations, clean_prefix_violations


def _raw_rgb_index() -> dict[Path, prepare.TUMEntry]:
    entries = prepare.parse_tum_listing(
        prepare.RAW_ROOT / "rgb.txt", value_count=2, label="raw rgb.txt"
    )
    result: dict[Path, prepare.TUMEntry] = {}
    for entry in entries:
        path = prepare._raw_path(entry, label=f"raw RGB entry {entry.data_index}")
        _require(path not in result, "raw rgb.txt repeats a resolved RGB path")
        result[path] = entry
    return result


def _validate_rgb_capture_record(
    source: Mapping[str, Any],
    *,
    raw_index: Mapping[Path, prepare.TUMEntry],
    label: str,
) -> None:
    required = {
        "path",
        "source_pool_index",
        "source_pool_role",
        "raw_rgb_source_index",
        "rgb_sha256",
        "rgb_size_bytes",
        "rgb_capture_timestamp",
        "rgb_capture_timestamp_text",
        "rgb_txt_physical_line",
        "rgb_txt_physical_line_sha256",
        "rgb_txt_physical_line_size_bytes",
        "depth",
        "groundtruth",
        "association",
    }
    _require(set(source) == required, f"{label} source-frame schema is invalid")
    path = _regular(Path(str(source["path"])), label=f"{label} raw RGB", mode=0o444)
    _require(_inside(path, prepare.RAW_ROOT), f"{label} RGB is outside raw root")
    raw = raw_index.get(path)
    _require(raw is not None, f"{label} RGB is absent from raw rgb.txt")
    _require(source["raw_rgb_source_index"] == raw.data_index, f"{label} raw RGB index differs")
    _require(source["rgb_sha256"] == _sha256(path), f"{label} raw RGB SHA-256 differs")
    _require(source["rgb_size_bytes"] == path.stat().st_size, f"{label} raw RGB size differs")
    _require(source["rgb_capture_timestamp"] == str(raw.timestamp), f"{label} capture timestamp differs")
    _require(source["rgb_capture_timestamp_text"] == raw.timestamp_text, f"{label} capture timestamp text differs")
    _require(source["rgb_txt_physical_line"] == raw.physical_line, f"{label} rgb.txt physical line differs")
    _require(source["rgb_txt_physical_line_sha256"] == raw.physical_sha256, f"{label} rgb.txt physical line SHA-256 differs")
    _require(source["rgb_txt_physical_line_size_bytes"] == len(raw.physical_bytes), f"{label} rgb.txt physical line size differs")
    # Reparse both representations as Decimal; this rejects a string that merely
    # looks related to the raw token but would change the invariant.
    _require(
        prepare._parse_decimal(str(source["rgb_capture_timestamp"]), label=f"{label} capture timestamp")
        == prepare._parse_decimal(str(source["rgb_capture_timestamp_text"]), label=f"{label} capture timestamp text"),
        f"{label} capture timestamp representations differ",
    )


def _validate_source_manifest(
    source_path: Path,
    *,
    run_id: str,
    corruption_type: str,
    raw_index: Mapping[Path, prepare.TUMEntry],
) -> dict[str, Any]:
    source = _strict_json(source_path, label=f"{run_id} source manifest")
    required = {
        "schema_version",
        "dataset",
        "dataset_split",
        "run_id",
        "corruption_type",
        "source_is_read_only",
        "frame_count",
        "output_frame_count",
        "sequence",
        "rgb_capture_listing",
        "association_policy",
        "offline_gt_depth_reprojection",
        "frames",
        "final_frame_to_raw_rgb_bindings",
    }
    _require(set(source) == required, f"{run_id} source manifest schema is invalid")
    _require(source["schema_version"] == prepare.SOURCE_SCHEMA_VERSION, f"{run_id} source schema differs")
    _require(source["dataset"] == prepare.DATASET_NAME, f"{run_id} source dataset differs")
    _require(source["dataset_split"] == "blind_formal_v3", f"{run_id} source split differs")
    _require(source["run_id"] == run_id and source["corruption_type"] == corruption_type, f"{run_id} source identity differs")
    _require(source["source_is_read_only"] is True, f"{run_id} source is not read-only")
    expected_source_count = 35 if corruption_type == "low_overlap_jump" else 30
    _require(source["frame_count"] == expected_source_count, f"{run_id} source frame count differs")
    _require(source["output_frame_count"] == prepare.FRAME_COUNT, f"{run_id} output frame count differs")
    listing = source["rgb_capture_listing"]
    _require(isinstance(listing, Mapping), f"{run_id} RGB listing binding is missing")
    listing_required = {"parser_version", "path", "sha256", "size_bytes", "mode_octal"}
    _require(set(listing) == listing_required, f"{run_id} RGB listing schema is invalid")
    _require(listing["parser_version"] == prepare.RGB_LISTING_PARSER_VERSION, f"{run_id} RGB listing parser differs")
    _verify_artifact(
        {key: listing[key] for key in ("path", "sha256", "size_bytes", "mode_octal")},
        expected_path=prepare.RAW_ROOT / "rgb.txt",
        label=f"{run_id} RGB listing",
    )
    association = source["association_policy"]
    _require(
        isinstance(association, Mapping)
        and association == {
            "method": "unique_nearest_absolute_timestamp",
            "tie_policy": "reject",
            "max_absolute_delta_seconds": str(prepare.MAX_ASSOCIATION_DELTA),
        },
        f"{run_id} association policy differs",
    )
    frames = source["frames"]
    _require(isinstance(frames, list) and len(frames) == expected_source_count, f"{run_id} source frames differ")
    for index, frame in enumerate(frames):
        _require(isinstance(frame, Mapping), f"{run_id} source frame {index} is not an object")
        _require(frame.get("source_pool_index") == index, f"{run_id} source frame indexes differ")
        expected_role = "base" if index < prepare.FRAME_COUNT else "low_overlap_donor"
        _require(frame.get("source_pool_role") == expected_role, f"{run_id} source frame role differs")
        _validate_rgb_capture_record(frame, raw_index=raw_index, label=f"{run_id} source frame {index}")
    bindings = source["final_frame_to_raw_rgb_bindings"]
    _require(isinstance(bindings, list) and len(bindings) == prepare.FRAME_COUNT, f"{run_id} final binding count differs")
    expected_indices = _expected_source_indices(corruption_type)
    for frame_index, (binding, source_index) in enumerate(zip(bindings, expected_indices, strict=True)):
        _require(isinstance(binding, Mapping), f"{run_id} final binding {frame_index} is not an object")
        expected = frames[source_index]
        _require(
            binding
            == {
                "frame_index": frame_index,
                "source_pool_index": source_index,
                "raw_rgb_sha256": expected["rgb_sha256"],
                "rgb_capture_timestamp": expected["rgb_capture_timestamp"],
                "rgb_capture_timestamp_text": expected["rgb_capture_timestamp_text"],
                "rgb_txt_physical_line": expected["rgb_txt_physical_line"],
                "rgb_txt_physical_line_sha256": expected["rgb_txt_physical_line_sha256"],
            },
            f"{run_id} final binding {frame_index} differs from its raw RGB source",
        )
    return source


def _validate_input_manifest(
    input_path: Path,
    source_path: Path,
    source: Mapping[str, Any],
    *,
    run_id: str,
    corruption_type: str,
) -> dict[str, Any]:
    state_src = ROOT / "src"
    if str(state_src) not in sys.path:
        sys.path.insert(0, str(state_src))
    from stateguard3r.input_manifest import load_input_manifest

    payload = _strict_json(input_path, label=f"{run_id} input manifest")
    _require(payload.get("source_manifest") == "source-manifest.json", f"{run_id} input source path differs")
    _require(payload.get("source_manifest_sha256") == _sha256(source_path), f"{run_id} input source SHA-256 differs")
    manifest = load_input_manifest(input_path)
    _require(manifest.source_manifest_path == source_path, f"{run_id} strict loader source path differs")
    _require(len(manifest.frames) == prepare.FRAME_COUNT, f"{run_id} final frame count differs")
    corruption = payload.get("corruptions")
    _require(isinstance(corruption, list) and len(corruption) == 1, f"{run_id} corruption layout differs")
    interval = corruption[0]
    _require(isinstance(interval, Mapping) and interval.get("type") == corruption_type, f"{run_id} corruption type differs")
    expected_end = prepare.EVENT_START + (3 if corruption_type == "wrong_order_segment" else 4)
    _require(interval.get("start") == prepare.EVENT_START and interval.get("end") == expected_end, f"{run_id} corruption interval differs")
    bindings = source["final_frame_to_raw_rgb_bindings"]
    frames = source["frames"]
    for frame_index, (frame, binding) in enumerate(zip(manifest.frames, bindings, strict=True)):
        _require(frame.frame_index == frame_index, f"{run_id} final frame index differs")
        source_index = binding["source_pool_index"]
        expected_source = frames[source_index]
        _require(frame.source_index == source_index, f"{run_id} final frame source index differs")
        _require(frame.path == Path(expected_source["path"]), f"{run_id} final RGB path differs")
        _require(frame.metadata == {key: value for key, value in expected_source.items() if key != "path"}, f"{run_id} final RGB metadata differs")
        _require(frame.metadata.get("rgb_sha256") == binding["raw_rgb_sha256"], f"{run_id} final RGB hash is not raw-bound")
        _require(frame.metadata.get("rgb_capture_timestamp") == binding["rgb_capture_timestamp"], f"{run_id} final timestamp differs")
        _require(frame.metadata.get("rgb_capture_timestamp_text") == binding["rgb_capture_timestamp_text"], f"{run_id} final timestamp text differs")
        _require(frame.metadata.get("rgb_txt_physical_line") == binding["rgb_txt_physical_line"], f"{run_id} final RGB line differs")
        _require(frame.metadata.get("rgb_txt_physical_line_sha256") == binding["rgb_txt_physical_line_sha256"], f"{run_id} final RGB line hash differs")
        _require(_sha256(frame.path) == binding["raw_rgb_sha256"], f"{run_id} final RGB bytes differ from raw binding")
    return payload


def _validate_run(
    root: Path,
    row: Mapping[str, Any],
    *,
    run_id: str,
    corruption_type: str,
    raw_index: Mapping[Path, prepare.TUMEntry],
    seen_rgb: set[str],
) -> dict[str, Any]:
    _require(row.get("run_id") == run_id, f"{run_id} registry identity differs")
    _require(row.get("dataset_split") == "blind_formal_v3", f"{run_id} registry split differs")
    _require(row.get("corruption_type") == corruption_type, f"{run_id} registry corruption differs")
    run_dir = _directory(root / run_id, label=f"{run_id} input directory", mode=0o555)
    source_path = _regular(run_dir / "source-manifest.json", label=f"{run_id} source manifest", mode=0o444)
    input_path = _regular(run_dir / "input-manifest.json", label=f"{run_id} input manifest", mode=0o444)
    artifacts = row.get("artifacts")
    _require(isinstance(artifacts, Mapping) and set(artifacts) == {"source_manifest", "input_manifest"}, f"{run_id} registry artifacts differ")
    for name, path in (("source_manifest", source_path), ("input_manifest", input_path)):
        artifact = artifacts[name]
        _require(isinstance(artifact, Mapping), f"{run_id} {name} artifact is invalid")
        _require(
            artifact == {
                "path": f"{run_id}/{path.name}",
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            },
            f"{run_id} {name} registry binding differs",
        )
    source = _validate_source_manifest(
        source_path, run_id=run_id, corruption_type=corruption_type, raw_index=raw_index
    )
    _validate_input_manifest(
        input_path, source_path, source, run_id=run_id, corruption_type=corruption_type
    )
    for binding in source["final_frame_to_raw_rgb_bindings"]:
        digest = binding["raw_rgb_sha256"]
        _require(digest not in seen_rgb, f"{run_id} reuses a final RGB across blind runs")
        seen_rgb.add(digest)
    violations, clean_prefix_violations = _replay_timestamp_sidecar(source, run_id=run_id)
    return {
        "run_id": run_id,
        "frame_count": prepare.FRAME_COUNT,
        "timestamp_order_violation_positions": violations,
        "clean_prefix_timestamp_order_violation_positions": clean_prefix_violations,
        "strict_loader_replay_passed": True,
        "final_frame_raw_rgb_hash_binding_passed": True,
        "online_timestamp_sidecar_replay_passed": True,
    }


def validate(input_root: Path) -> dict[str, Any]:
    """Validate one frozen formal-v3 input root without model/output access."""

    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU validation requires CUDA_VISIBLE_DEVICES='' ")
    root = _directory(input_root.resolve(strict=True), label="formal-v3 input root", mode=0o555)
    _directory(prepare.RAW_ROOT, label="formal-v3 raw tree", mode=0o555)
    _regular(prepare.ARCHIVE, label="formal-v3 archive", mode=0o444)
    registry_path = _regular(root / "formal-v3-manifest.json", label="formal-v3 input registry", mode=0o444)
    registry = _strict_json(registry_path, label="formal-v3 input registry")
    required = {
        "schema_version",
        "status",
        "dataset",
        "frame_count",
        "event_start",
        "seed",
        "archive",
        "acquisition",
        "acquisition_status",
        "timestamp_capability",
        "offline_gt_depth_reprojection",
        "runs",
    }
    _require(set(registry) == required, "formal-v3 input registry schema is invalid")
    _require(registry["schema_version"] == prepare.REGISTRY_SCHEMA_VERSION, "formal-v3 registry schema differs")
    _require(registry["status"] == "pre_forward_blind_formal_v3_inputs", "formal-v3 registry status differs")
    _require(registry["dataset"] == prepare.DATASET_NAME, "formal-v3 registry dataset differs")
    _require(registry["frame_count"] == prepare.FRAME_COUNT and registry["event_start"] == prepare.EVENT_START and registry["seed"] == prepare.SEED, "formal-v3 registry core parameters differ")
    _verify_artifact(registry["archive"], expected_path=prepare.ARCHIVE, label="formal-v3 archive")
    _verify_artifact(registry["acquisition"], expected_path=prepare.ACQUISITION, label="formal-v3 acquisition")
    _require(registry["acquisition_status"] == "PASS", "formal-v3 acquisition status differs")
    acquisition, _ = prepare._read_acquisition()
    _require(acquisition["status"] == "PASS", "formal-v3 acquisition no longer passes")
    capability = registry["timestamp_capability"]
    _require(
        isinstance(capability, Mapping)
        and capability.get("scope") == "timestamp_preserving_reorder_only"
        and capability.get("online_input") == "final_rgb_path_bound_to_raw_rgb_txt_capture_timestamp"
        and capability.get("forbidden_online_inputs")
        == ["source_pool_index", "transform_metadata", "corruption_interval", "groundtruth", "depth", "future_frame"],
        "formal-v3 timestamp capability contract differs",
    )
    rows = registry["runs"]
    _require(isinstance(rows, list) and len(rows) == len(prepare.RUN_SPECS), "formal-v3 run registry layout differs")
    by_id = {row.get("run_id"): row for row in rows if isinstance(row, Mapping)}
    _require(len(by_id) == len(prepare.RUN_SPECS), "formal-v3 run registry has duplicate/non-object rows")
    raw_index = _raw_rgb_index()
    seen_rgb: set[str] = set()
    checked = [
        _validate_run(root, by_id[run_id], run_id=run_id, corruption_type=kind, raw_index=raw_index, seen_rgb=seen_rgb)
        for run_id, kind, _length in prepare.RUN_SPECS
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "input_root": str(root),
        "input_registry_sha256": _sha256(registry_path),
        "runs": checked,
        "checks": {
            "cuda_hidden": True,
            "archive_and_acquisition_bound": True,
            "raw_tree_read_only": True,
            "strict_rgb_decimal_and_physical_line_replay": True,
            "final_frame_to_raw_rgb_hash_binding": True,
            "strict_loader_replay": True,
            "online_timestamp_sidecar_replay": True,
            "blind_final_rgb_disjoint": True,
            "timestamp_scope_contract": True,
            "no_holdout_model_output_read": True,
        },
    }


def _freeze(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _publish(output_dir: Path, report: Mapping[str, Any]) -> Path:
    _require(output_dir.parent == OUTPUT_ROOT, "validator output must be a new direct child of outputs")
    _require(not output_dir.exists(), "validator output already exists")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".staging", dir=output_dir.parent))
    published = False
    try:
        payload = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        (staging / "cpu-validation.json").write_bytes(payload)
        _freeze(staging)
        os.replace(staging, output_dir)
        published = True
        return output_dir
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        report = validate(args.input_root)
        script = Path(__file__).resolve()
        report["validator"] = {"path": str(script), "sha256": _sha256(script)}
        _publish(args.output_dir.resolve(strict=False), report)
    except (FormalV3InputValidationError, prepare.FormalV3InputError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
