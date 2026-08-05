#!/usr/bin/env python3
"""Build one CPU-only, manifest-only recovery-quality input set from TUM RGB-D.

The emitted manifests preserve model-image provenance separately from the
logical-base GT bindings consumed only by the later evaluator.  This script
never imports the model, starts CUDA, or writes into the raw dataset.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
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
FRAME_COUNT = 30
EVENT_START = 15
MAX_ASSOCIATION_DELTA = Decimal("0.02")
SOURCE_SCHEMA = "stateguard3r.recovery-quality-source.v1"
REGISTRY_SCHEMA = "stateguard3r.recovery-quality-inputs.v1"
SEED = 20260806


class RecoveryQualityInputError(ValueError):
    """Raised when a formal-quality input cannot be bound to read-only TUM raw data."""


@dataclass(frozen=True)
class TUMRow:
    index: int
    physical_line: int
    timestamp: Decimal
    timestamp_text: str
    fields: tuple[str, ...]
    physical_bytes: bytes

    @property
    def physical_sha256(self) -> str:
        return _sha256_bytes(self.physical_bytes)


@dataclass(frozen=True)
class Associated:
    rgb: TUMRow
    depth: TUMRow
    groundtruth: TUMRow
    rgb_path: Path
    depth_path: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryQualityInputError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    regular = _regular(path, label=str(path), mode=0o444)
    return {
        "path": str(regular),
        "sha256": _sha256(regular),
        "size_bytes": regular.stat().st_size,
        "mode_octal": "0444",
    }


def _published_artifact(staged_path: Path, published_path: Path) -> dict[str, Any]:
    """Hash a staged immutable file while binding the path after atomic publish."""

    artifact = _artifact(staged_path)
    artifact["path"] = str(published_path)
    return artifact


def _regular(path: Path, *, label: str, mode: int | None = None) -> Path:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise RecoveryQualityInputError(f"cannot stat {label}: {error}") from error
    _require(not stat.S_ISLNK(metadata.st_mode) and stat.S_ISREG(metadata.st_mode), f"{label} must be a regular non-symlink file")
    if mode is not None:
        _require(stat.S_IMODE(metadata.st_mode) == mode, f"{label} must have mode {mode:04o}")
    return path.resolve(strict=True)


def _directory(path: Path, *, label: str, mode: int = 0o555) -> Path:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise RecoveryQualityInputError(f"cannot stat {label}: {error}") from error
    _require(not stat.S_ISLNK(metadata.st_mode) and stat.S_ISDIR(metadata.st_mode), f"{label} must be a directory")
    _require(stat.S_IMODE(metadata.st_mode) == mode, f"{label} must have mode {mode:04o}")
    return path.resolve(strict=True)


def parse_tum_listing(path: Path, *, value_count: int, label: str) -> list[TUMRow]:
    """Read a strict TUM listing while retaining exact timestamp-line evidence."""

    listing = _regular(path, label=label, mode=0o444)
    rows: list[TUMRow] = []
    for physical_line, raw in enumerate(listing.read_bytes().splitlines(keepends=True), start=1):
        body = raw.rstrip(b"\r\n")
        if not body.strip() or body.lstrip().startswith(b"#"):
            continue
        try:
            tokens = tuple(body.decode("ascii").split())
        except UnicodeDecodeError as error:
            raise RecoveryQualityInputError(f"{label} line {physical_line} is not ASCII") from error
        _require(len(tokens) == value_count + 1, f"{label} line {physical_line} has wrong field count")
        try:
            timestamp = Decimal(tokens[0])
        except InvalidOperation as error:
            raise RecoveryQualityInputError(f"{label} line {physical_line} timestamp is invalid") from error
        _require(timestamp.is_finite(), f"{label} line {physical_line} timestamp is not finite")
        rows.append(TUMRow(len(rows), physical_line, timestamp, tokens[0], tokens[1:], raw))
    _require(rows, f"{label} has no data rows")
    return rows


def _raw_path(row: TUMRow, *, dataset_root: Path, label: str) -> Path:
    _require(len(row.fields) == 1, f"{label} must contain one relative path")
    relative = Path(row.fields[0])
    _require(not relative.is_absolute() and ".." not in relative.parts, f"{label} path is unsafe")
    path = (dataset_root / relative).resolve(strict=True)
    try:
        path.relative_to(dataset_root)
    except ValueError as error:
        raise RecoveryQualityInputError(f"{label} path escapes dataset root") from error
    return _regular(path, label=label, mode=0o444)


def _nearest_unique(timestamp: Decimal, rows: Sequence[TUMRow], *, label: str) -> TUMRow | None:
    candidates = [(abs(row.timestamp - timestamp), row) for row in rows if abs(row.timestamp - timestamp) <= MAX_ASSOCIATION_DELTA]
    if not candidates:
        return None
    distance = min(item[0] for item in candidates)
    closest = [row for candidate_distance, row in candidates if candidate_distance == distance]
    # A non-unique nearest raw association invalidates this RGB row, not every
    # otherwise-valid block in the immutable sequence.
    if len(closest) != 1:
        return None
    return closest[0]


def associated_frames(dataset_root: Path) -> list[Associated | None]:
    rgb_rows = parse_tum_listing(dataset_root / "rgb.txt", value_count=1, label="rgb.txt")
    depth_rows = parse_tum_listing(dataset_root / "depth.txt", value_count=1, label="depth.txt")
    gt_rows = parse_tum_listing(dataset_root / "groundtruth.txt", value_count=7, label="groundtruth.txt")
    output: list[Associated | None] = []
    for rgb in rgb_rows:
        depth = _nearest_unique(rgb.timestamp, depth_rows, label=f"RGB row {rgb.index} depth")
        gt = _nearest_unique(rgb.timestamp, gt_rows, label=f"RGB row {rgb.index} GT")
        if depth is None or gt is None:
            output.append(None)
            continue
        output.append(
            Associated(
                rgb=rgb,
                depth=depth,
                groundtruth=gt,
                rgb_path=_raw_path(rgb, dataset_root=dataset_root, label=f"RGB row {rgb.index}"),
                depth_path=_raw_path(depth, dataset_root=dataset_root, label=f"depth row {depth.index}"),
            )
        )
    return output


def _valid_block(frames: Sequence[Associated | None], start: int, length: int) -> bool:
    block = frames[start : start + length]
    if len(block) != length or any(item is None for item in block):
        return False
    typed = [item for item in block if item is not None]
    return len({item.rgb.index for item in typed}) == length and len({item.depth.index for item in typed}) == length and len({item.groundtruth.index for item in typed}) == length


def select_base_and_donor(frames: Sequence[Associated | None]) -> tuple[int, int]:
    """Choose a median valid 30-frame block and a far, valid five-frame donor."""

    base_candidates = [index for index in range(len(frames) - FRAME_COUNT + 1) if _valid_block(frames, index, FRAME_COUNT)]
    _require(base_candidates, "dataset has no valid 30-frame RGB/depth/GT block")
    base_start = base_candidates[len(base_candidates) // 2]
    donor_candidates = [
        index
        for index in range(len(frames) - 5 + 1)
        if _valid_block(frames, index, 5)
        and (index + 4 < base_start or index > base_start + FRAME_COUNT - 1)
        and min(abs(index - base_start), abs(index - (base_start + FRAME_COUNT - 1))) >= 60
    ]
    _require(donor_candidates, "dataset has no non-overlapping donor at least 60 RGB rows away")
    donor_start = min(donor_candidates, key=lambda index: (-abs(index - base_start), index))
    return base_start, donor_start


def _gt_binding(row: TUMRow) -> dict[str, Any]:
    _require(len(row.fields) == 7, "ground-truth row must contain tx ty tz qx qy qz qw")
    try:
        numbers = [float(value) for value in row.fields]
    except ValueError as error:
        raise RecoveryQualityInputError("ground-truth pose contains a non-numeric value") from error
    _require(all(math.isfinite(value) for value in numbers), "ground-truth pose must be finite")
    return {
        "groundtruth_source_index": row.index,
        "groundtruth_timestamp": str(row.timestamp),
        "groundtruth_timestamp_text": row.timestamp_text,
        "groundtruth_physical_line": row.physical_line,
        "groundtruth_physical_line_sha256": row.physical_sha256,
        "position_xyz": numbers[:3],
        "orientation_xyzw": numbers[3:],
    }


def _source_record(frame: Associated, *, source_pool_index: int, role: str) -> dict[str, Any]:
    return {
        "path": str(frame.rgb_path),
        "source_pool_index": source_pool_index,
        "source_pool_role": role,
        "raw_rgb_source_index": frame.rgb.index,
        "rgb_sha256": _sha256(frame.rgb_path),
        "rgb_size_bytes": frame.rgb_path.stat().st_size,
        "rgb_capture_timestamp": str(frame.rgb.timestamp),
        "rgb_capture_timestamp_text": frame.rgb.timestamp_text,
        "rgb_txt_physical_line": frame.rgb.physical_line,
        "rgb_txt_physical_line_sha256": frame.rgb.physical_sha256,
        "depth_source_index": frame.depth.index,
        "depth_sha256": _sha256(frame.depth_path),
        "quality_source_groundtruth": _gt_binding(frame.groundtruth),
    }


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


def _quality_bindings(base: Sequence[Associated]) -> list[dict[str, Any]]:
    return [
        {"frame_index": index, "base_source_index": index, "groundtruth": _gt_binding(frame.groundtruth)}
        for index, frame in enumerate(base)
    ]


def _clean_manifest(source_path: Path, source: Mapping[str, Any], *, bindings: list[dict[str, Any]], sequence: str) -> dict[str, Any]:
    source_hash = _sha256(source_path)
    frames = []
    for index, record in enumerate(source["frames"][:FRAME_COUNT]):
        frames.append({"frame_index": index, "source_index": index, "path": record["path"], "metadata": {key: value for key, value in record.items() if key != "path"}, "transforms": []})
    return {
        "schema_version": "stateguard3r.corruption.v1",
        "sequence": sequence,
        "source_sequence": source["sequence"],
        "source_manifest": "source-manifest.json",
        "source_manifest_sha256": source_hash,
        "source_ground_truth_path": source["ground_truth_path"],
        "source_ground_truth_resolved_path": source["ground_truth_path"],
        "source_ground_truth_sha256": source["ground_truth_sha256"],
        "source_is_read_only": True,
        "source_frame_count": len(source["frames"]),
        "frame_count": FRAME_COUNT,
        "seed": SEED,
        "index_convention": "zero_based_inclusive",
        "materialization": "deferred_transforms_no_image_copy",
        "frames": frames,
        "corruptions": [],
        "recovery_quality_clean_control": True,
        "quality_evaluation_bindings": bindings,
    }


def _corruption_specs() -> tuple[tuple[str, list[dict[str, Any]]], ...]:
    return (
        ("dynamic", [{"type": "dynamic_occlusion", "start": 15, "end": 18, "parameters": {"coordinate_space": "normalized", "rectangle": {"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5}, "velocity": {"dx": 0.04, "dy": 0.025}, "fill": [255, 0, 0]}, "expected_effect": "transient foreground contamination"}]),
        ("wrong-order", [{"type": "wrong_order_segment", "start": 15, "end": 18, "parameters": {"mode": "reverse"}, "expected_effect": "timestamp-preserving temporal reorder"}]),
        ("low-overlap", [{"type": "low_overlap_jump", "start": 15, "end": 18, "parameters": {"source_start": FRAME_COUNT}, "expected_effect": "far disjoint donor substitution"}]),
    )


def prepare(dataset_root: Path, output_dir: Path, *, dataset_id: str) -> Path:
    """Publish a non-overwriting four-condition input tree after strict CPU checks."""

    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "input preparation requires CUDA_VISIBLE_DEVICES='' ")
    raw_root = _directory(dataset_root, label="TUM dataset root")
    archive = _regular(raw_root.with_suffix(".tgz"), label="TUM archive", mode=0o444)
    output = output_dir.resolve(strict=False)
    _require(output.parent == OUTPUT_ROOT and not output.exists(), "output must be a new direct child of StateGuard3R/outputs")
    _require(dataset_id == raw_root.name, "dataset_id must equal raw root basename")
    frames = associated_frames(raw_root)
    base_start, donor_start = select_base_and_donor(frames)
    base = [item for item in frames[base_start : base_start + FRAME_COUNT] if item is not None]
    donor = [item for item in frames[donor_start : donor_start + 5] if item is not None]
    _require(len(base) == FRAME_COUNT and len(donor) == 5, "selected source blocks are incomplete")
    source = {
        "schema_version": SOURCE_SCHEMA,
        "dataset": dataset_id,
        "dataset_split": "recovery-quality-formal-v1",
        "sequence": f"{dataset_id}-base-{base_start}",
        "source_is_read_only": True,
        "ground_truth_path": str(_regular(raw_root / "groundtruth.txt", label="groundtruth.txt", mode=0o444)),
        "ground_truth_sha256": _sha256(raw_root / "groundtruth.txt"),
        "output_frame_count": FRAME_COUNT,
        "frames": [
            *[_source_record(frame, source_pool_index=index, role="base") for index, frame in enumerate(base)],
            *[_source_record(frame, source_pool_index=FRAME_COUNT + index, role="low_overlap_donor") for index, frame in enumerate(donor)],
        ],
    }
    bindings = _quality_bindings(base)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".staging", dir=output.parent))
    try:
        records: list[dict[str, Any]] = []
        for condition, specs in (("clean", []), *_corruption_specs()):
            run_dir = staging / condition
            run_dir.mkdir()
            source_path = run_dir / "source-manifest.json"
            _write_json(source_path, source)
            source_path.chmod(0o444)
            if condition == "clean":
                manifest = _clean_manifest(source_path, source, bindings=bindings, sequence=f"{dataset_id}-{condition}")
            else:
                sys.path.insert(0, str(ROOT / "src"))
                from stateguard3r.corruption import build_corruption_manifest

                manifest = build_corruption_manifest(source, specs, seed=SEED, check_paths=True, source_base_dir=run_dir, source_manifest_path="source-manifest.json", source_manifest_sha256=_sha256(source_path), source_ground_truth_resolved_path=source["ground_truth_path"], source_ground_truth_sha256=source["ground_truth_sha256"], sequence=f"{dataset_id}-{condition}")
                manifest["quality_evaluation_bindings"] = bindings
            input_path = run_dir / "input-manifest.json"
            _write_json(input_path, manifest)
            input_path.chmod(0o444)
            from stateguard3r.input_manifest import load_input_manifest

            loaded = load_input_manifest(input_path)
            _require(len(loaded.frames) == FRAME_COUNT, f"{condition} strict input replay differs")
            records.append(
                {
                    "condition": condition,
                    "directory": condition,
                    "source_manifest": _published_artifact(
                        source_path, output / condition / "source-manifest.json"
                    ),
                    "input_manifest": _published_artifact(
                        input_path, output / condition / "input-manifest.json"
                    ),
                }
            )
        registry = {
            "schema_version": REGISTRY_SCHEMA,
            "status": "pre_forward_recovery_quality_inputs",
            "dataset": dataset_id,
            "dataset_root": str(raw_root),
            "archive": _artifact(archive),
            "association_policy": {
                "method": "unique_nearest_absolute_timestamp",
                "max_delta_seconds": str(MAX_ASSOCIATION_DELTA),
                "tie_policy": "reject_as_unusable_row",
                "rejected_rgb_rows": [index for index, frame in enumerate(frames) if frame is None],
            },
            "frame_count": FRAME_COUNT,
            "event_start": EVENT_START,
            "seed": SEED,
            "base_start_rgb_row": base_start,
            "donor_start_rgb_row": donor_start,
            "quality_evaluation": "logical_base_GT_only_posthoc",
            "runs": records,
        }
        _write_json(staging / "recovery-quality-inputs.json", registry)
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
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    result = prepare(args.dataset_root, args.output_dir, dataset_id=args.dataset_id)
    print(json.dumps({"output_dir": str(result)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
