#!/usr/bin/env python3
"""Prepare the one formal-v3 blind input set without a model invocation.

The only permitted fresh scene is TUM ``rgbd_dataset_freiburg2_desk``.  This
script never downloads or mutates raw data.  It first requires a frozen,
successful acquisition record and a read-only archive/tree, then creates three
new immutable manifest-only corruptions:

* ``blind-dynamic`` — dynamic occlusion;
* ``blind-wrong-order`` — timestamp-preserving reverse segment; and
* ``blind-low-overlap`` — GT-depth-selected donor substitution.

The v3 addition is deliberately narrow and auditable.  Every source frame
stores the exact raw ``rgb.txt`` timestamp token, parsed decimal value,
physical source-line hash, listing hash, and RGB byte hash.  The final input
manifest is tied to those source records before any ReCal3R execution.  No
online component may use source-pool indices, transforms, GT, or labels; the
runner's v3 sidecar independently consumes only final RGB paths and raw
``rgb.txt``.
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
import shutil
import stat
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
OUTPUT_ROOT = ROOT / "outputs"
DATASET_NAME = "rgbd_dataset_freiburg2_desk"
RAW_ROOT = PROJECT_ROOT / "baselines" / "ReCal3R" / "data" / "tum" / DATASET_NAME
ARCHIVE = RAW_ROOT.with_suffix(".tgz")
ACQUISITION = OUTPUT_ROOT / "formal-v3-data-0001" / "acquisition.json"

FRAME_COUNT = 30
EVENT_START = 15
SEED = 0
MAX_ASSOCIATION_DELTA = Decimal("0.02")
INTRINSICS = {"fx": 520.9, "fy": 521.0, "cx": 325.1, "cy": 249.7}

RUN_SPECS: tuple[tuple[str, str, int], ...] = (
    ("blind-dynamic", "dynamic_occlusion", 5),
    ("blind-wrong-order", "wrong_order_segment", 4),
    ("blind-low-overlap", "low_overlap_jump", 5),
)

SOURCE_SCHEMA_VERSION = "stateguard3r.tum-formal-v3-source.v1"
REGISTRY_SCHEMA_VERSION = "stateguard3r.formal-v3-inputs.v1"
RGB_LISTING_PARSER_VERSION = "tum-rgb-txt-decimal-line-bound.v1"


class FormalV3InputError(ValueError):
    """Raised when formal-v3 construction would be unsafe or non-reproducible."""


@dataclass(frozen=True)
class TUMEntry:
    """One strict physical line from a TUM association listing."""

    data_index: int
    physical_line: int
    timestamp: Decimal
    timestamp_text: str
    fields: tuple[str, ...]
    physical_bytes: bytes

    @property
    def physical_sha256(self) -> str:
        return _sha256_bytes(self.physical_bytes)


@dataclass(frozen=True)
class AssociatedFrame:
    """One RGB frame with deterministic depth/GT associations."""

    rgb: TUMEntry
    depth: TUMEntry
    groundtruth: TUMEntry
    rgb_path: Path
    depth_path: Path
    rgb_sha256: str
    depth_sha256: str
    rgb_size_bytes: int
    depth_size_bytes: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalV3InputError(message)


def _enable_stateguard_src() -> Path:
    """Expose the local StateGuard3R package to the reused ReCal3R CPU env."""
    source = ROOT / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return source


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise FormalV3InputError(f"{label} contains non-finite JSON {value!r}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FormalV3InputError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_bytes(),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalV3InputError(f"cannot parse {label}: {error}") from error
    _require(isinstance(payload, dict), f"{label} must contain a JSON object")
    return payload


def _regular_read_only(path: Path, *, label: str, mode: int | None = 0o444) -> Path:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise FormalV3InputError(f"cannot stat {label}: {error}") from error
    _require(not stat.S_ISLNK(metadata.st_mode), f"{label} may not be a symlink")
    _require(stat.S_ISREG(metadata.st_mode), f"{label} must be a regular file")
    if mode is not None:
        _require(stat.S_IMODE(metadata.st_mode) == mode, f"{label} must have mode {mode:04o}")
    return path.resolve(strict=True)


def _read_only_directory(path: Path, *, label: str) -> Path:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise FormalV3InputError(f"cannot stat {label}: {error}") from error
    _require(not stat.S_ISLNK(metadata.st_mode), f"{label} may not be a symlink")
    _require(stat.S_ISDIR(metadata.st_mode), f"{label} must be a directory")
    _require(stat.S_IMODE(metadata.st_mode) == 0o555, f"{label} must have mode 0555")
    return path.resolve(strict=True)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _artifact(path: Path) -> dict[str, Any]:
    regular = _regular_read_only(path, label=str(path))
    return {
        "path": str(regular),
        "sha256": _sha256_file(regular),
        "size_bytes": regular.stat().st_size,
        "mode_octal": f"{stat.S_IMODE(regular.stat().st_mode):04o}",
    }


def _parse_decimal(text: str, *, label: str) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise FormalV3InputError(f"{label} is not a decimal timestamp") from error
    _require(value.is_finite(), f"{label} must be finite")
    return value


def parse_tum_listing(
    path: Path,
    *,
    value_count: int,
    label: str,
) -> list[TUMEntry]:
    """Strictly parse a physical TUM listing while retaining line provenance."""

    listing = _regular_read_only(path, label=label)
    try:
        physical_lines = listing.read_bytes().splitlines(keepends=True)
    except OSError as error:
        raise FormalV3InputError(f"cannot read {label}: {error}") from error
    result: list[TUMEntry] = []
    previous: Decimal | None = None
    for physical_line, raw in enumerate(physical_lines, start=1):
        try:
            text = raw.decode("utf-8").strip()
        except UnicodeDecodeError as error:
            raise FormalV3InputError(f"{label} line {physical_line} is not UTF-8") from error
        if not text or text.startswith("#"):
            continue
        fields = tuple(text.split())
        _require(
            len(fields) == value_count,
            f"{label} line {physical_line} must contain exactly {value_count} fields",
        )
        timestamp = _parse_decimal(fields[0], label=f"{label} line {physical_line}")
        if previous is not None:
            _require(timestamp >= previous, f"{label} capture timestamps decrease at line {physical_line}")
        previous = timestamp
        result.append(
            TUMEntry(
                data_index=len(result),
                physical_line=physical_line,
                timestamp=timestamp,
                timestamp_text=fields[0],
                fields=fields,
                physical_bytes=raw,
            )
        )
    _require(result, f"{label} contains no data entries")
    return result


def _raw_path(entry: TUMEntry, *, label: str) -> Path:
    relative = Path(entry.fields[1])
    _require(not relative.is_absolute() and ".." not in relative.parts, f"{label} has unsafe raw path")
    candidate = RAW_ROOT / relative
    resolved = _regular_read_only(candidate, label=label)
    _require(_inside(resolved, RAW_ROOT), f"{label} escapes raw dataset root")
    return resolved


def _nearest_unique(
    reference: TUMEntry,
    candidates: Sequence[TUMEntry],
    *,
    label: str,
) -> TUMEntry | None:
    distances = [(abs(candidate.timestamp - reference.timestamp), candidate) for candidate in candidates]
    minimum = min(distance for distance, _ in distances)
    ties = [candidate for distance, candidate in distances if distance == minimum]
    if len(ties) != 1 or minimum > MAX_ASSOCIATION_DELTA:
        return None
    return ties[0]


def _associated_frames() -> list[AssociatedFrame | None]:
    rgb = parse_tum_listing(RAW_ROOT / "rgb.txt", value_count=2, label="raw rgb.txt")
    depth = parse_tum_listing(RAW_ROOT / "depth.txt", value_count=2, label="raw depth.txt")
    groundtruth = parse_tum_listing(
        RAW_ROOT / "groundtruth.txt", value_count=8, label="raw groundtruth.txt"
    )
    result: list[AssociatedFrame | None] = []
    for rgb_entry in rgb:
        depth_entry = _nearest_unique(rgb_entry, depth, label="depth")
        gt_entry = _nearest_unique(rgb_entry, groundtruth, label="groundtruth")
        if depth_entry is None or gt_entry is None:
            result.append(None)
            continue
        rgb_path = _raw_path(rgb_entry, label=f"raw RGB entry {rgb_entry.data_index}")
        depth_path = _raw_path(depth_entry, label=f"raw depth entry {depth_entry.data_index}")
        result.append(
            AssociatedFrame(
                rgb=rgb_entry,
                depth=depth_entry,
                groundtruth=gt_entry,
                rgb_path=rgb_path,
                depth_path=depth_path,
                rgb_sha256=_sha256_file(rgb_path),
                depth_sha256=_sha256_file(depth_path),
                rgb_size_bytes=rgb_path.stat().st_size,
                depth_size_bytes=depth_path.stat().st_size,
            )
        )
    return result


def _valid_block(frames: Sequence[AssociatedFrame | None], start: int, length: int) -> bool:
    block = frames[start : start + length]
    if len(block) != length or any(item is None for item in block):
        return False
    typed = [item for item in block if item is not None]
    return (
        len({item.rgb.data_index for item in typed}) == length
        and len({item.depth.data_index for item in typed}) == length
        and len({item.groundtruth.data_index for item in typed}) == length
    )


def _allocate_base_starts(frames: Sequence[AssociatedFrame | None]) -> list[int]:
    starts: list[int] = []
    used_rgb: set[int] = set()
    used_depth: set[int] = set()
    used_gt: set[int] = set()
    for start in range(0, len(frames) - FRAME_COUNT + 1):
        if not _valid_block(frames, start, FRAME_COUNT):
            continue
        block = [item for item in frames[start : start + FRAME_COUNT] if item is not None]
        rgb = {item.rgb.data_index for item in block}
        depth = {item.depth.data_index for item in block}
        gt = {item.groundtruth.data_index for item in block}
        if rgb & used_rgb or depth & used_depth or gt & used_gt:
            continue
        starts.append(start)
        used_rgb.update(rgb)
        used_depth.update(depth)
        used_gt.update(gt)
        if len(starts) == len(RUN_SPECS):
            return starts
    raise FormalV3InputError("cannot allocate three disjoint 30-frame base windows")


def _pose_matrix(entry: TUMEntry) -> np.ndarray:
    try:
        tx, ty, tz, qx, qy, qz, qw = (float(value) for value in entry.fields[1:])
    except ValueError as error:
        raise FormalV3InputError("groundtruth pose has non-numeric values") from error
    quaternion = np.asarray((qw, qx, qy, qz), dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    _require(math.isfinite(norm) and norm > 0.0, "groundtruth quaternion is invalid")
    w, x, y, z = quaternion / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    matrix[:3, 3] = (tx, ty, tz)
    return matrix


def _read_depth(path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as error:
        raise FormalV3InputError(
            "offline low-overlap donor selection requires the existing ReCal3R OpenCV environment"
        ) from error
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    _require(image is not None and image.shape == (480, 640), f"invalid TUM depth image: {path}")
    return image


def _select_low_overlap_donor(
    frames: Sequence[AssociatedFrame | None],
    *,
    low_base_start: int,
    base_starts: Sequence[int],
) -> tuple[int, list[float]]:
    """Use GT only to choose the fixed low-overlap donor, never online."""

    _enable_stateguard_src()
    from stateguard3r.visual_overlap import gt_depth_reprojection_overlap

    used_rgb = {
        item.rgb.data_index
        for start in base_starts
        for item in frames[start : start + FRAME_COUNT]
        if item is not None
    }
    ranked: list[tuple[float, int, list[float]]] = []
    for start in range(0, len(frames) - 5 + 1):
        if not _valid_block(frames, start, 5):
            continue
        donor = [item for item in frames[start : start + 5] if item is not None]
        if {item.rgb.data_index for item in donor} & used_rgb:
            continue
        scores: list[float] = []
        for offset, right in enumerate(donor):
            left = frames[low_base_start + EVENT_START + offset]
            _require(left is not None, "low-overlap base frame is unavailable")
            overlap = gt_depth_reprojection_overlap(
                _read_depth(left.depth_path),
                _read_depth(right.depth_path),
                _pose_matrix(left.groundtruth),
                _pose_matrix(right.groundtruth),
                INTRINSICS,
                stride=8,
            )
            scores.append(float(overlap.score))
        ranked.append((float(math.fsum(scores) / len(scores)), start, scores))
    _require(ranked, "cannot allocate a disjoint five-frame low-overlap donor")
    _, start, scores = min(ranked, key=lambda item: (item[0], item[1]))
    return start, scores


def _entry_provenance(entry: TUMEntry, *, path: Path | None = None, digest: str | None = None, size: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "source_entry_index": entry.data_index,
        "physical_line": entry.physical_line,
        "physical_line_sha256": entry.physical_sha256,
        "physical_line_size_bytes": len(entry.physical_bytes),
        "timestamp": str(entry.timestamp),
        "timestamp_text": entry.timestamp_text,
    }
    if path is not None:
        value.update({"path": str(path), "sha256": digest, "size_bytes": size})
    return value


def _source_frame(frame: AssociatedFrame, *, source_pool_index: int, role: str) -> dict[str, Any]:
    rgb = _entry_provenance(
        frame.rgb,
        path=frame.rgb_path,
        digest=frame.rgb_sha256,
        size=frame.rgb_size_bytes,
    )
    return {
        "path": str(frame.rgb_path),
        "source_pool_index": source_pool_index,
        "source_pool_role": role,
        "raw_rgb_source_index": frame.rgb.data_index,
        "rgb_sha256": frame.rgb_sha256,
        "rgb_size_bytes": frame.rgb_size_bytes,
        "rgb_capture_timestamp": rgb["timestamp"],
        "rgb_capture_timestamp_text": rgb["timestamp_text"],
        "rgb_txt_physical_line": rgb["physical_line"],
        "rgb_txt_physical_line_sha256": rgb["physical_line_sha256"],
        "rgb_txt_physical_line_size_bytes": rgb["physical_line_size_bytes"],
        "depth": _entry_provenance(
            frame.depth,
            path=frame.depth_path,
            digest=frame.depth_sha256,
            size=frame.depth_size_bytes,
        ),
        "groundtruth": _entry_provenance(frame.groundtruth),
        "association": {
            "depth_absolute_delta_seconds": str(abs(frame.depth.timestamp - frame.rgb.timestamp)),
            "groundtruth_absolute_delta_seconds": str(abs(frame.groundtruth.timestamp - frame.rgb.timestamp)),
        },
    }


def _corruption(kind: str) -> dict[str, Any]:
    if kind == "dynamic_occlusion":
        return {
            "type": kind,
            "start": EVENT_START,
            "end": EVENT_START + 4,
            "parameters": {
                "coordinate_space": "normalized",
                "rectangle": {"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5},
                "velocity": {"dx": 0.04, "dy": 0.025},
                "fill": [255, 0, 0],
            },
        }
    if kind == "wrong_order_segment":
        return {
            "type": kind,
            "start": EVENT_START,
            "end": EVENT_START + 3,
            "parameters": {"mode": "reverse"},
        }
    if kind == "low_overlap_jump":
        return {
            "type": kind,
            "start": EVENT_START,
            "end": EVENT_START + 4,
            "parameters": {"source_start": FRAME_COUNT},
        }
    raise FormalV3InputError(f"unsupported formal-v3 corruption {kind!r}")


def _expected_source_indices(kind: str) -> list[int]:
    if kind == "dynamic_occlusion":
        return list(range(FRAME_COUNT))
    if kind == "wrong_order_segment":
        return [*range(EVENT_START), 18, 17, 16, 15, *range(19, FRAME_COUNT)]
    if kind == "low_overlap_jump":
        return [*range(EVENT_START), *range(FRAME_COUNT, FRAME_COUNT + 5), *range(20, FRAME_COUNT)]
    raise FormalV3InputError(f"unsupported formal-v3 corruption {kind!r}")


def _final_bindings(kind: str, source_frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for frame_index, source_index in enumerate(_expected_source_indices(kind)):
        source = source_frames[source_index]
        bindings.append(
            {
                "frame_index": frame_index,
                "source_pool_index": source_index,
                "raw_rgb_sha256": source["rgb_sha256"],
                "rgb_capture_timestamp": source["rgb_capture_timestamp"],
                "rgb_capture_timestamp_text": source["rgb_capture_timestamp_text"],
                "rgb_txt_physical_line": source["rgb_txt_physical_line"],
                "rgb_txt_physical_line_sha256": source["rgb_txt_physical_line_sha256"],
            }
        )
    return bindings


def _freeze_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise FormalV3InputError(f"refusing to freeze symlink: {path}")
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _read_acquisition() -> tuple[dict[str, Any], dict[str, Any]]:
    acquisition_path = _regular_read_only(ACQUISITION, label="formal-v3 acquisition")
    acquisition = _strict_json(acquisition_path, label="formal-v3 acquisition")
    _require(acquisition.get("status") == "PASS", "formal-v3 acquisition is not PASS")
    _require(acquisition.get("dataset") == DATASET_NAME, "formal-v3 acquisition selected a different dataset")
    archive = acquisition.get("archive")
    _require(isinstance(archive, Mapping), "formal-v3 acquisition lacks archive binding")
    _require(archive.get("path") == str(ARCHIVE.resolve(strict=True)), "acquisition archive path differs")
    _require(archive.get("sha256") == _sha256_file(ARCHIVE), "acquisition archive SHA-256 differs")
    _require(archive.get("size_bytes") == ARCHIVE.stat().st_size, "acquisition archive size differs")
    _require(archive.get("mode_octal") == "0444", "acquisition archive mode binding differs")
    raw_tree = acquisition.get("raw_tree")
    _require(isinstance(raw_tree, Mapping), "formal-v3 acquisition lacks raw-tree evidence")
    tree_manifest = raw_tree.get("manifest")
    _require(isinstance(tree_manifest, Mapping) and tree_manifest.get("dataset") == DATASET_NAME, "raw-tree dataset binding differs")
    postconditions = acquisition.get("postconditions")
    _require(
        isinstance(postconditions, Mapping)
        and postconditions.get("no_model_or_online_overlap_execution") is True,
        "acquisition does not prove model/online-overlap abstention",
    )
    return acquisition, _artifact(acquisition_path)


def prepare(
    output_dir: Path,
    *,
    donor_selector: Callable[..., tuple[int, list[float]]] | None = None,
) -> Path:
    """Create one no-overwrite formal-v3 input root from read-only raw data."""

    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "input preparation requires CUDA_VISIBLE_DEVICES='' ")
    output = output_dir.resolve(strict=False)
    _require(output.parent == OUTPUT_ROOT, "output_dir must be a new direct child of outputs")
    _require(not output.exists(), "refusing to overwrite an existing formal-v3 input root")
    _regular_read_only(ARCHIVE, label="formal-v3 archive")
    _read_only_directory(RAW_ROOT, label="formal-v3 raw tree")
    acquisition, acquisition_artifact = _read_acquisition()

    frames = _associated_frames()
    base_starts = _allocate_base_starts(frames)
    selector = _select_low_overlap_donor if donor_selector is None else donor_selector
    donor_start, donor_scores = selector(
        frames,
        low_base_start=base_starts[2],
        base_starts=base_starts,
    )
    _require(_valid_block(frames, donor_start, 5), "selected low-overlap donor is invalid")

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".staging", dir=output.parent))
    published = False
    try:
        records: list[dict[str, Any]] = []
        rgb_listing = _artifact(RAW_ROOT / "rgb.txt")
        for (run_id, corruption_type, _event_length), base_start in zip(RUN_SPECS, base_starts, strict=True):
            base = [item for item in frames[base_start : base_start + FRAME_COUNT] if item is not None]
            _require(len(base) == FRAME_COUNT, f"{run_id} base allocation is incomplete")
            pool = list(base)
            if corruption_type == "low_overlap_jump":
                donor = [item for item in frames[donor_start : donor_start + 5] if item is not None]
                _require(len(donor) == 5, "selected low-overlap donor is incomplete")
                pool.extend(donor)
            source_frames = [
                _source_frame(
                    item,
                    source_pool_index=index,
                    role="base" if index < FRAME_COUNT else "low_overlap_donor",
                )
                for index, item in enumerate(pool)
            ]
            source = {
                "schema_version": SOURCE_SCHEMA_VERSION,
                "dataset": DATASET_NAME,
                "dataset_split": "blind_formal_v3",
                "run_id": run_id,
                "corruption_type": corruption_type,
                "source_is_read_only": True,
                "frame_count": len(source_frames),
                "output_frame_count": FRAME_COUNT,
                "sequence": f"{run_id}-source-pool",
                "rgb_capture_listing": {
                    "parser_version": RGB_LISTING_PARSER_VERSION,
                    **rgb_listing,
                },
                "association_policy": {
                    "method": "unique_nearest_absolute_timestamp",
                    "tie_policy": "reject",
                    "max_absolute_delta_seconds": str(MAX_ASSOCIATION_DELTA),
                },
                "offline_gt_depth_reprojection": {
                    "usage": "low_overlap_donor_selection_only_not_online_ledger",
                    "intrinsics": INTRINSICS,
                    "stride": 8,
                },
                "frames": source_frames,
                "final_frame_to_raw_rgb_bindings": _final_bindings(corruption_type, source_frames),
            }
            source_bytes = _canonical_json_bytes(source)
            manifest = _build_manifest(
                source,
                corruption=_corruption(corruption_type),
                source_sha256=_sha256_bytes(source_bytes),
                sequence=run_id,
            )
            input_bytes = _canonical_json_bytes(manifest)
            run_dir = staging / run_id
            run_dir.mkdir()
            (run_dir / "source-manifest.json").write_bytes(source_bytes)
            (run_dir / "input-manifest.json").write_bytes(input_bytes)
            records.append(
                {
                    "run_id": run_id,
                    "dataset_split": "blind_formal_v3",
                    "corruption_type": corruption_type,
                    "base_rgb_source_indices": [item.rgb.data_index for item in base],
                    "donor_rgb_source_indices": (
                        [item.rgb.data_index for item in pool[FRAME_COUNT:]]
                        if corruption_type == "low_overlap_jump"
                        else []
                    ),
                    "artifacts": {
                        "source_manifest": {
                            "path": f"{run_id}/source-manifest.json",
                            "sha256": _sha256_bytes(source_bytes),
                            "size_bytes": len(source_bytes),
                        },
                        "input_manifest": {
                            "path": f"{run_id}/input-manifest.json",
                            "sha256": _sha256_bytes(input_bytes),
                            "size_bytes": len(input_bytes),
                        },
                    },
                }
            )
        registry = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "status": "pre_forward_blind_formal_v3_inputs",
            "dataset": DATASET_NAME,
            "frame_count": FRAME_COUNT,
            "event_start": EVENT_START,
            "seed": SEED,
            "archive": _artifact(ARCHIVE),
            "acquisition": acquisition_artifact,
            "acquisition_status": acquisition["status"],
            "timestamp_capability": {
                "scope": "timestamp_preserving_reorder_only",
                "online_input": "final_rgb_path_bound_to_raw_rgb_txt_capture_timestamp",
                "forbidden_online_inputs": [
                    "source_pool_index",
                    "transform_metadata",
                    "corruption_interval",
                    "groundtruth",
                    "depth",
                    "future_frame",
                ],
            },
            "offline_gt_depth_reprojection": {
                "low_overlap_donor_start": donor_start,
                "per_pair_scores": donor_scores,
                "usage": "donor_selection_only_not_online_ledger",
            },
            "runs": records,
        }
        (staging / "formal-v3-manifest.json").write_bytes(_canonical_json_bytes(registry))
        _freeze_tree(staging)
        os.replace(staging, output)
        published = True
        return output
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _build_manifest(
    source: Mapping[str, Any],
    *,
    corruption: Mapping[str, Any],
    source_sha256: str,
    sequence: str,
) -> dict[str, Any]:
    _enable_stateguard_src()
    from stateguard3r.corruption import build_corruption_manifest

    return build_corruption_manifest(
        source,
        [corruption],
        seed=SEED,
        check_paths=True,
        source_base_dir=RAW_ROOT,
        source_manifest_path="source-manifest.json",
        source_manifest_sha256=source_sha256,
        sequence=sequence,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.output_dir)
    except (FormalV3InputError, OSError, ValueError) as error:
        parser.error(str(error))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
