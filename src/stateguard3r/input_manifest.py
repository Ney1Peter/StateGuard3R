"""Strict, read-only replay of StateGuard3R corruption v1 manifests.

Frame substitution and temporal reordering are already encoded in the final
``frames`` path order.  This module validates that provenance and only applies
deferred pixel transforms to cloned/copied in-memory views.  It deliberately
does not import PyTorch; tensors are supported through a small ``clone`` and
index-assignment protocol.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "stateguard3r.corruption.v1"
MATERIALIZATION_MODE = "deferred_transforms_no_image_copy"
INDEX_CONVENTION = "zero_based_inclusive"
RECTANGLE_COORDINATE_REFERENCE = "model_input_after_resize_and_center_crop"
FORMAL_SOURCE_SCHEMA_VERSION = "stateguard3r.tum-formal-pilot-source.v1"
FORMAL_RAW_MANIFEST_SCHEMA_VERSION = "stateguard3r.tum-raw-audit.v1"
RECOVERY_QUALITY_SOURCE_SCHEMA_VERSION = "stateguard3r.recovery-quality-source.v1"

CORRUPTION_TO_TRANSFORM = {
    "low_overlap_jump": "source_frame_substitution",
    "dynamic_occlusion": "rectangle_occlusion",
    "wrong_order_segment": "temporal_reorder",
}
SOURCE_CONTENT_SHA256_KEYS = ("rgb_sha256", "content_sha256", "sha256")
SOURCE_RAW_INDEX_KEYS = (
    "raw_rgb_source_index",
    "raw_source_index",
    "raw_index",
)


class InputManifestError(ValueError):
    """Raised when a persisted input manifest cannot be replayed safely."""


@dataclass(frozen=True, slots=True)
class ManifestFrame:
    """One validated frame in final replay order."""

    frame_index: int
    source_index: int
    path: Path
    metadata: Mapping[str, Any]
    transforms: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class InputManifest:
    """A validated corruption manifest and its ordered source paths."""

    path: Path
    source_manifest_path: Path
    frames: tuple[ManifestFrame, ...]

    @property
    def frame_paths(self) -> tuple[Path, ...]:
        return tuple(frame.path for frame in self.frames)


@dataclass(frozen=True, slots=True)
class _StableFileSnapshot:
    path: Path
    device: int
    inode: int
    mode: int
    link_count: int
    size_bytes: int
    mtime_ns: int
    sha256: str


def _plain_int(value: Any, *, name: str) -> int:
    if type(value) is not int:
        raise InputManifestError(f"{name} must be an integer")
    return value


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise InputManifestError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise InputManifestError(f"{name} must be a finite number")
    return number


def _nonempty_path(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputManifestError(f"{name} must be a non-empty path string")
    if "\x00" in value:
        raise InputManifestError(f"{name} contains a NUL byte")
    return value


def _load_json_object(path: Path, *, name: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload_bytes = path.read_bytes()
    except OSError as error:
        raise InputManifestError(f"cannot read {name} {path}: {error}") from error
    def reject_constant(value: str) -> None:
        raise InputManifestError(
            f"{name} contains non-finite JSON constant {value!r}: {path}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise InputManifestError(
                    f"{name} contains duplicate JSON key {key!r}: {path}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InputManifestError(f"{name} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(payload, dict):
        raise InputManifestError(f"{name} must be a JSON object")
    return payload, payload_bytes


def _stable_file_snapshot(path: Path, *, name: str) -> _StableFileSnapshot:
    """Hash one regular non-symlink file through one descriptor."""

    try:
        link_metadata = os.lstat(path)
    except OSError as error:
        raise InputManifestError(f"cannot inspect {name}: {path}: {error}") from error
    if stat.S_ISLNK(link_metadata.st_mode) or not stat.S_ISREG(link_metadata.st_mode):
        raise InputManifestError(
            f"{name} must be a regular non-symlink file: {path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise InputManifestError(f"cannot open {name}: {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if (
            (link_metadata.st_dev, link_metadata.st_ino)
            != (before.st_dev, before.st_ino)
            or not stat.S_ISREG(before.st_mode)
        ):
            raise InputManifestError(f"{name} changed between lstat and open: {path}")
        digest = hashlib.sha256()
        size_read = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size_read += len(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise InputManifestError(f"cannot hash {name}: {path}: {error}") from error
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or size_read != before.st_size:
        raise InputManifestError(f"{name} changed while being hashed: {path}")
    return _StableFileSnapshot(
        path=path,
        device=before.st_dev,
        inode=before.st_ino,
        mode=stat.S_IMODE(before.st_mode),
        link_count=before.st_nlink,
        size_bytes=before.st_size,
        mtime_ns=before.st_mtime_ns,
        sha256=digest.hexdigest(),
    )


def _source_frames(
    payload: Mapping[str, Any], *, base_dir: Path, reject_aliases: bool = False
) -> list[tuple[Path, dict[str, Any]]]:
    if "frames" in payload and "frame_paths" in payload:
        raise InputManifestError(
            "source manifest must use only one of frames or frame_paths"
        )
    key = "frames" if "frames" in payload else "frame_paths"
    raw_frames = payload.get(key)
    if not isinstance(raw_frames, list) or not raw_frames:
        raise InputManifestError(
            "source manifest must contain a non-empty frames or frame_paths list"
        )

    result: list[tuple[Path, dict[str, Any]]] = []
    for index, entry in enumerate(raw_frames):
        if isinstance(entry, str):
            raw_path = entry
            metadata: dict[str, Any] = {}
        elif isinstance(entry, Mapping):
            raw_path = entry.get("path")
            metadata = copy.deepcopy(
                {key: value for key, value in entry.items() if key != "path"}
            )
        else:
            raise InputManifestError(
                f"source frame {index} must be a path or an object"
            )
        path_text = _nonempty_path(raw_path, name=f"source frame {index} path")
        raw_path = Path(path_text)
        unresolved = raw_path if raw_path.is_absolute() else base_dir / raw_path
        absolute = Path(os.path.abspath(unresolved))
        try:
            path = unresolved.resolve(strict=True)
        except OSError as error:
            raise InputManifestError(
                f"source frame {index} path does not exist: {unresolved}"
            ) from error
        if reject_aliases and path != absolute:
            raise InputManifestError(
                f"source frame {index} path may not resolve through a symlink or alias"
            )
        if not path.is_file():
            raise InputManifestError(
                f"source frame {index} path is not a file: {path}"
            )
        result.append((path, metadata))
    return result


def _source_output_frame_count(
    payload: Mapping[str, Any], *, source_frame_count: int
) -> tuple[int, bool]:
    if "output_frame_count" not in payload:
        return source_frame_count, False
    frame_count = _plain_int(
        payload["output_frame_count"], name="source output_frame_count"
    )
    if frame_count < 1 or frame_count > source_frame_count:
        raise InputManifestError(
            "source output_frame_count must be between 1 and source frame count "
            f"({source_frame_count})"
        )
    return frame_count, True


def _source_pool_frame_identity(
    path: Path, metadata: Mapping[str, Any], *, frame_index: int
) -> dict[str, Any]:
    snapshot = _stable_file_snapshot(path, name=f"source frame {frame_index}")
    actual_content_sha256 = snapshot.sha256
    content_sha256 = {actual_content_sha256}
    for key in SOURCE_CONTENT_SHA256_KEYS:
        if key not in metadata:
            continue
        value = metadata[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in value)
        ):
            raise InputManifestError(
                f"source frame {frame_index} metadata {key} must be a SHA-256"
            )
        declared = value.lower()
        if declared != actual_content_sha256:
            raise InputManifestError(
                f"source frame {frame_index} metadata {key} does not match actual "
                "file SHA-256"
            )
        content_sha256.add(declared)

    raw_indices: set[int] = set()
    for key in SOURCE_RAW_INDEX_KEYS:
        if key not in metadata:
            continue
        raw_indices.add(
            _plain_int(
                metadata[key], name=f"source frame {frame_index} metadata {key}"
            )
        )
    if len(raw_indices) > 1:
        raise InputManifestError(
            f"source frame {frame_index} metadata raw-index fields must agree"
        )
    return {
        "resolved_path": str(path),
        "inode": (snapshot.device, snapshot.inode),
        "content_sha256": content_sha256,
        "raw_index": next(iter(raw_indices), None),
        "snapshot": snapshot,
    }


def _validate_source_pool_disjoint(
    source_frames: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    output_frame_count: int,
) -> None:
    identities = [
        _source_pool_frame_identity(path, metadata, frame_index=frame_index)
        for frame_index, (path, metadata) in enumerate(source_frames)
    ]
    for donor_index in range(output_frame_count, len(source_frames)):
        donor = identities[donor_index]
        for base_index in range(output_frame_count):
            base = identities[base_index]
            reasons: list[str] = []
            if donor["resolved_path"] == base["resolved_path"]:
                reasons.append("resolved path")
            if donor["inode"] == base["inode"]:
                reasons.append("inode")
            if donor["content_sha256"] & base["content_sha256"]:
                reasons.append("content SHA-256")
            if (
                donor["raw_index"] is not None
                and donor["raw_index"] == base["raw_index"]
            ):
                reasons.append("raw index")
            if reasons:
                raise InputManifestError(
                    f"source-pool donor frame {donor_index} overlaps output/base "
                    f"frame {base_index} by {', '.join(reasons)}"
                )


def _validate_declared_source_hashes(
    source_frames: Sequence[tuple[Path, Mapping[str, Any]]],
) -> None:
    for frame_index, (path, metadata) in enumerate(source_frames):
        if any(key in metadata for key in SOURCE_CONTENT_SHA256_KEYS):
            _source_pool_frame_identity(path, metadata, frame_index=frame_index)


_FORMAL_SNAPSHOT_KEYS = {
    "path",
    "sha256",
    "size_bytes",
    "mode_octal",
    "link_count",
    "device",
    "inode",
    "mtime_ns",
}


def _formal_read_only_directory(path: Path, *, name: str) -> Path:
    absolute = Path(os.path.abspath(path))
    try:
        metadata = os.lstat(absolute)
    except OSError as error:
        raise InputManifestError(f"cannot inspect {name}: {absolute}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise InputManifestError(f"{name} must be a real non-symlink directory")
    if stat.S_IMODE(metadata.st_mode) & 0o222:
        raise InputManifestError(f"{name} must be read-only (no write bits)")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise InputManifestError(f"cannot resolve {name}: {absolute}") from error
    if resolved != absolute:
        raise InputManifestError(f"{name} may not resolve through a symlink")
    return resolved


def _formal_path(
    value: Any,
    *,
    base_dir: Path,
    name: str,
    expected_root: Path | None = None,
) -> Path:
    path_text = _nonempty_path(value, name=name)
    raw = Path(path_text)
    unresolved = raw if raw.is_absolute() else base_dir / raw
    absolute = Path(os.path.abspath(unresolved))
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as error:
        raise InputManifestError(f"{name} does not exist: {unresolved}") from error
    if resolved != absolute:
        raise InputManifestError(f"{name} may not resolve through a symlink or alias")
    if expected_root is not None:
        try:
            resolved.relative_to(expected_root)
        except ValueError as error:
            raise InputManifestError(f"{name} resolves outside the formal dataset root") from error
    return resolved


def _formal_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise InputManifestError(f"{name} must be a lowercase SHA-256")
    return value


def _formal_nonnegative_int(value: Any, *, name: str) -> int:
    result = _plain_int(value, name=name)
    if result < 0:
        raise InputManifestError(f"{name} must be non-negative")
    return result


def _assert_formal_snapshot(
    snapshot: _StableFileSnapshot,
    declared: Mapping[str, Any],
    *,
    name: str,
    prefix: str = "",
) -> None:
    expected = {
        "sha256": _formal_sha256(
            declared.get(f"{prefix}sha256"), name=f"{name} sha256"
        ),
        "size_bytes": _formal_nonnegative_int(
            declared.get(f"{prefix}size_bytes"), name=f"{name} size_bytes"
        ),
        "mode_octal": declared.get(f"{prefix}mode_octal"),
        "link_count": _formal_nonnegative_int(
            declared.get(f"{prefix}link_count"), name=f"{name} link_count"
        ),
        "device": _formal_nonnegative_int(
            declared.get(f"{prefix}device"), name=f"{name} device"
        ),
        "inode": _formal_nonnegative_int(
            declared.get(f"{prefix}inode"), name=f"{name} inode"
        ),
        "mtime_ns": _formal_nonnegative_int(
            declared.get(f"{prefix}mtime_ns"), name=f"{name} mtime_ns"
        ),
    }
    if not isinstance(expected["mode_octal"], str):
        raise InputManifestError(f"{name} mode_octal must be an octal string")
    actual = {
        "sha256": snapshot.sha256,
        "size_bytes": snapshot.size_bytes,
        "mode_octal": f"{snapshot.mode:04o}",
        "link_count": snapshot.link_count,
        "device": snapshot.device,
        "inode": snapshot.inode,
        "mtime_ns": snapshot.mtime_ns,
    }
    if expected != actual:
        raise InputManifestError(
            f"{name} actual path/stat/SHA-256 differs from formal declaration"
        )
    if snapshot.mode & 0o222:
        raise InputManifestError(f"{name} must remain read-only")


def _validate_formal_artifact(
    value: Any, *, base_dir: Path, name: str
) -> tuple[Path, _StableFileSnapshot]:
    if not isinstance(value, Mapping) or set(value) != _FORMAL_SNAPSHOT_KEYS:
        raise InputManifestError(f"{name} has invalid formal artifact fields")
    path = _formal_path(value.get("path"), base_dir=base_dir, name=f"{name} path")
    snapshot = _stable_file_snapshot(path, name=name)
    _assert_formal_snapshot(snapshot, value, name=name)
    return path, snapshot


def _validate_formal_source_provenance(
    source_payload: Mapping[str, Any],
    source_frames: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    source_manifest_path: Path,
    input_payload: Mapping[str, Any],
) -> None:
    if source_payload.get("dataset") != "rgbd_dataset_freiburg1_desk":
        raise InputManifestError("formal source dataset identity mismatch")
    if (
        source_payload.get("file_path_semantics")
        != "relative_to_this_manifest_directory"
    ):
        raise InputManifestError("formal source file_path_semantics mismatch")
    if source_payload.get("source_is_read_only") is not True:
        raise InputManifestError("formal source manifest must mark source_is_read_only=true")
    lineage = source_payload.get("official_lineage")
    if not isinstance(lineage, Mapping) or set(lineage) != {
        "dataset_root",
        "archive",
        "raw_manifest",
        "raw_manifest_schema_version",
    }:
        raise InputManifestError("formal source manifest official_lineage is invalid")
    if lineage.get("raw_manifest_schema_version") != FORMAL_RAW_MANIFEST_SCHEMA_VERSION:
        raise InputManifestError("formal raw-manifest schema lineage mismatch")
    dataset_root_text = _nonempty_path(
        lineage.get("dataset_root"), name="formal lineage dataset_root"
    )
    if not Path(dataset_root_text).is_absolute():
        raise InputManifestError("formal lineage dataset_root must be absolute")
    dataset_root = _formal_read_only_directory(
        Path(dataset_root_text), name="formal dataset root"
    )
    rgb_root = _formal_read_only_directory(dataset_root / "rgb", name="formal RGB root")
    depth_root = _formal_read_only_directory(
        dataset_root / "depth", name="formal depth root"
    )
    _validate_formal_artifact(
        lineage.get("archive"), base_dir=source_manifest_path.parent, name="formal archive"
    )
    _validate_formal_artifact(
        lineage.get("raw_manifest"),
        base_dir=source_manifest_path.parent,
        name="formal raw manifest",
    )

    raw_indices: set[int] = set()
    depth_indices: set[int] = set()
    groundtruth_indices: set[int] = set()
    for frame_index, (rgb_path, metadata) in enumerate(source_frames):
        try:
            rgb_path.relative_to(rgb_root)
        except ValueError as error:
            raise InputManifestError(
                f"formal source frame {frame_index} is outside the RGB root"
            ) from error
        if metadata.get("source_pool_index") != frame_index:
            raise InputManifestError(
                f"formal source frame {frame_index} source_pool_index mismatch"
            )
        expected_role = (
            "base"
            if frame_index < int(source_payload.get("output_frame_count", -1))
            else "low_overlap_donor"
        )
        if metadata.get("source_pool_role") != expected_role:
            raise InputManifestError(
                f"formal source frame {frame_index} source_pool_role mismatch"
            )
        raw_index = _formal_nonnegative_int(
            metadata.get("raw_rgb_source_index"),
            name=f"formal source frame {frame_index} raw RGB index",
        )
        if metadata.get("source_entry_index") != raw_index:
            raise InputManifestError(
                f"formal source frame {frame_index} RGB entry aliases disagree"
            )
        if raw_index in raw_indices:
            raise InputManifestError("formal source pool contains duplicate raw RGB indices")
        raw_indices.add(raw_index)
        rgb_snapshot = _stable_file_snapshot(
            rgb_path, name=f"formal source RGB frame {frame_index}"
        )
        _assert_formal_snapshot(
            rgb_snapshot,
            metadata,
            name=f"formal source RGB frame {frame_index}",
            prefix="rgb_",
        )

        depth = metadata.get("depth")
        if not isinstance(depth, Mapping):
            raise InputManifestError(f"formal source frame {frame_index} lacks depth provenance")
        depth_path = _formal_path(
            depth.get("path"),
            base_dir=source_manifest_path.parent,
            name=f"formal source frame {frame_index} depth path",
            expected_root=dataset_root,
        )
        try:
            depth_path.relative_to(depth_root)
        except ValueError as error:
            raise InputManifestError(
                f"formal source frame {frame_index} depth is outside the depth root"
            ) from error
        depth_snapshot = _stable_file_snapshot(
            depth_path, name=f"formal source depth frame {frame_index}"
        )
        _assert_formal_snapshot(
            depth_snapshot, depth, name=f"formal source depth frame {frame_index}"
        )
        depth_index = _formal_nonnegative_int(
            depth.get("source_entry_index"),
            name=f"formal source frame {frame_index} depth source index",
        )
        if depth_index in depth_indices:
            raise InputManifestError("formal source pool contains duplicate depth associations")
        depth_indices.add(depth_index)

        groundtruth = metadata.get("groundtruth")
        if not isinstance(groundtruth, Mapping):
            raise InputManifestError(
                f"formal source frame {frame_index} lacks ground-truth provenance"
            )
        groundtruth_index = _formal_nonnegative_int(
            groundtruth.get("source_entry_index"),
            name=f"formal source frame {frame_index} ground-truth source index",
        )
        if groundtruth_index in groundtruth_indices:
            raise InputManifestError(
                "formal source pool contains duplicate ground-truth associations"
            )
        groundtruth_indices.add(groundtruth_index)

    index_files = source_payload.get("tum_index_files")
    if not isinstance(index_files, Mapping) or set(index_files) != {
        "rgb",
        "depth",
        "groundtruth",
    }:
        raise InputManifestError("formal source tum_index_files is invalid")
    index_snapshots: dict[str, tuple[Path, _StableFileSnapshot]] = {}
    for kind in ("rgb", "depth", "groundtruth"):
        declaration = index_files[kind]
        if not isinstance(declaration, Mapping) or set(declaration) != (
            _FORMAL_SNAPSHOT_KEYS | {"data_entry_count"}
        ):
            raise InputManifestError(f"formal {kind} index declaration is invalid")
        if _formal_nonnegative_int(
            declaration.get("data_entry_count"),
            name=f"formal {kind} index data_entry_count",
        ) < 1:
            raise InputManifestError(f"formal {kind} index must contain data entries")
        index_path = _formal_path(
            declaration.get("path"),
            base_dir=source_manifest_path.parent,
            name=f"formal {kind} index path",
            expected_root=dataset_root,
        )
        if index_path != dataset_root / f"{kind}.txt":
            raise InputManifestError(f"formal {kind} index path is not canonical")
        snapshot = _stable_file_snapshot(index_path, name=f"formal {kind} index")
        _assert_formal_snapshot(snapshot, declaration, name=f"formal {kind} index")
        index_snapshots[kind] = (index_path, snapshot)

    ground_truth_text = _nonempty_path(
        source_payload.get("ground_truth_path"), name="formal ground_truth_path"
    )
    ground_truth_path = _formal_path(
        ground_truth_text,
        base_dir=source_manifest_path.parent,
        name="formal ground_truth_path",
        expected_root=dataset_root,
    )
    groundtruth_index_path, groundtruth_snapshot = index_snapshots["groundtruth"]
    if ground_truth_path != groundtruth_index_path:
        raise InputManifestError("formal ground_truth_path differs from groundtruth index")
    if input_payload.get("source_ground_truth_path") != ground_truth_text:
        raise InputManifestError("input/source ground_truth_path declarations differ")
    if input_payload.get("source_ground_truth_resolved_path") != str(ground_truth_path):
        raise InputManifestError("source_ground_truth_resolved_path mismatch")
    if input_payload.get("source_ground_truth_sha256") != groundtruth_snapshot.sha256:
        raise InputManifestError("source_ground_truth_sha256 no longer matches")


def _corruption_expectations(
    payload: Mapping[str, Any],
    *,
    frame_count: int,
    source_frame_count: int,
    donor_pool_start: int | None,
    allow_empty_clean_control: bool = False,
) -> tuple[list[str | None], list[int | None]]:
    raw_corruptions = payload.get("corruptions")
    if not isinstance(raw_corruptions, list):
        raise InputManifestError("corruptions must be a non-empty JSON array")
    if not raw_corruptions:
        if not allow_empty_clean_control or payload.get("recovery_quality_clean_control") is not True:
            raise InputManifestError("corruptions must be a non-empty JSON array")
        return [None] * frame_count, [None] * frame_count
    expected: list[str | None] = [None] * frame_count
    expected_replacements: list[int | None] = [None] * frame_count
    for index, raw in enumerate(raw_corruptions):
        if not isinstance(raw, Mapping):
            raise InputManifestError(f"corruption {index} must be a JSON object")
        corruption_type = raw.get("type")
        if corruption_type not in CORRUPTION_TO_TRANSFORM:
            raise InputManifestError(
                f"corruption {index} has unsupported type {corruption_type!r}"
            )
        start = _plain_int(raw.get("start"), name=f"corruption {index} start")
        end = _plain_int(raw.get("end"), name=f"corruption {index} end")
        if raw.get("start_frame") != start or raw.get("end_frame") != end:
            raise InputManifestError(
                f"corruption {index} frame aliases must agree with start/end"
            )
        if start < 0 or end < start or end >= frame_count:
            raise InputManifestError(
                f"corruption {index} has invalid inclusive interval [{start}, {end}]"
            )
        if corruption_type == "dynamic_occlusion":
            parameters = raw.get("parameters")
            if not isinstance(parameters, Mapping):
                raise InputManifestError(
                    f"corruption {index} dynamic_occlusion parameters must be an object"
                )
            if parameters.get("coordinate_space") != "normalized":
                raise InputManifestError(
                    f"corruption {index} dynamic_occlusion must be normalized"
                )
            if (
                parameters.get("coordinate_reference")
                != RECTANGLE_COORDINATE_REFERENCE
            ):
                raise InputManifestError(
                    f"corruption {index} dynamic_occlusion has unsupported "
                    "coordinate reference"
                )
        elif corruption_type == "low_overlap_jump":
            parameters = raw.get("parameters")
            required = {"source_start", "source_end", "source_indices", "selection"}
            if not isinstance(parameters, Mapping) or set(parameters) != required:
                raise InputManifestError(
                    f"corruption {index} low_overlap_jump parameters have "
                    "unexpected fields"
                )
            source_start = _plain_int(
                parameters.get("source_start"),
                name=f"corruption {index} source_start",
            )
            source_end = _plain_int(
                parameters.get("source_end"), name=f"corruption {index} source_end"
            )
            raw_source_indices = parameters.get("source_indices")
            if not isinstance(raw_source_indices, list):
                raise InputManifestError(
                    f"corruption {index} source_indices must be a list"
                )
            source_indices = [
                _plain_int(
                    source_index,
                    name=f"corruption {index} source_indices[{offset}]",
                )
                for offset, source_index in enumerate(raw_source_indices)
            ]
            if (
                source_start < 0
                or source_end < source_start
                or source_end >= source_frame_count
            ):
                raise InputManifestError(
                    f"corruption {index} low-overlap donor range is out of bounds"
                )
            interval_length = end - start + 1
            if source_end - source_start + 1 != interval_length:
                raise InputManifestError(
                    f"corruption {index} low-overlap donor length disagrees "
                    "with interval"
                )
            expected_indices = list(range(source_start, source_end + 1))
            if source_indices != expected_indices:
                raise InputManifestError(
                    f"corruption {index} low-overlap source range and indices disagree"
                )
            if donor_pool_start is not None and source_start < donor_pool_start:
                raise InputManifestError(
                    f"corruption {index} low-overlap donor range is outside the "
                    "declared donor pool"
                )
            if not set(range(start, end + 1)).isdisjoint(source_indices):
                raise InputManifestError(
                    f"corruption {index} low-overlap donor and target ranges overlap"
                )
            if parameters.get("selection") not in {
                "explicit",
                "seeded_farthest_disjoint_index_proxy",
            }:
                raise InputManifestError(
                    f"corruption {index} low-overlap selection is unsupported"
                )
            for target_index, source_index in zip(
                range(start, end + 1), source_indices, strict=True
            ):
                expected_replacements[target_index] = source_index
        elif corruption_type == "wrong_order_segment":
            parameters = raw.get("parameters")
            required = {"mode", "permutation", "relative_permutation"}
            if not isinstance(parameters, Mapping) or set(parameters) != required:
                raise InputManifestError(
                    f"corruption {index} wrong_order_segment parameters have "
                    "unexpected fields"
                )
            if parameters.get("mode") not in {"reverse", "shuffle"}:
                raise InputManifestError(
                    f"corruption {index} wrong_order_segment mode is unsupported"
                )
            raw_permutation = parameters.get("permutation")
            raw_relative = parameters.get("relative_permutation")
            if not isinstance(raw_permutation, list) or not isinstance(
                raw_relative, list
            ):
                raise InputManifestError(
                    f"corruption {index} wrong-order permutations must be lists"
                )
            permutation = [
                _plain_int(
                    source_index,
                    name=f"corruption {index} permutation[{offset}]",
                )
                for offset, source_index in enumerate(raw_permutation)
            ]
            relative = [
                _plain_int(
                    source_index,
                    name=f"corruption {index} relative_permutation[{offset}]",
                )
                for offset, source_index in enumerate(raw_relative)
            ]
            original = list(range(start, end + 1))
            if sorted(permutation) != original or len(permutation) != len(original):
                raise InputManifestError(
                    f"corruption {index} wrong-order permutation is invalid"
                )
            if relative != [source_index - start for source_index in permutation]:
                raise InputManifestError(
                    f"corruption {index} wrong-order relative permutation disagrees"
                )
            if parameters.get("mode") == "reverse" and permutation != list(
                reversed(original)
            ):
                raise InputManifestError(
                    f"corruption {index} reverse permutation is not reversed"
                )
            for target_index, source_index in zip(
                original, permutation, strict=True
            ):
                expected_replacements[target_index] = source_index
        for frame_index in range(start, end + 1):
            if expected[frame_index] is not None:
                raise InputManifestError(
                    f"corruption intervals overlap at frame {frame_index}"
                )
            expected[frame_index] = corruption_type
    return expected, expected_replacements


def _rectangle(value: Any, *, name: str) -> dict[str, float]:
    required = {"x", "y", "width", "height"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise InputManifestError(
            f"{name} must contain exactly x, y, width, and height"
        )
    rectangle = {
        key: _finite_number(value[key], name=f"{name}.{key}") for key in required
    }
    x = rectangle["x"]
    y = rectangle["y"]
    width = rectangle["width"]
    height = rectangle["height"]
    if width <= 0.0 or height <= 0.0 or width > 1.0 or height > 1.0:
        raise InputManifestError(f"{name} width/height must be in (0, 1]")
    if x < 0.0 or y < 0.0 or x + width > 1.0 or y + height > 1.0:
        raise InputManifestError(f"{name} must fit inside normalized image bounds")
    return rectangle


def _fill(value: Any, *, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise InputManifestError(f"{name} must be an RGB list with three channels")
    channels = tuple(
        _finite_number(channel, name=f"{name}[{index}]")
        for index, channel in enumerate(value)
    )
    if any(channel < 0.0 or channel > 255.0 for channel in channels):
        raise InputManifestError(f"{name} RGB channels must be in [0, 255]")
    return channels  # type: ignore[return-value]


def _validated_transform(
    raw: Any,
    *,
    frame_index: int,
    source_index: int,
    source_frame_count: int,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise InputManifestError(f"frame {frame_index} transform must be an object")
    transform_type = raw.get("type")
    if transform_type == "rectangle_occlusion":
        if set(raw) != {
            "type",
            "coordinate_space",
            "coordinate_reference",
            "rectangle",
            "fill",
        }:
            raise InputManifestError(
                f"frame {frame_index} rectangle_occlusion has unexpected fields"
            )
        if raw.get("coordinate_space") != "normalized":
            raise InputManifestError(
                f"frame {frame_index} rectangle_occlusion must be normalized"
            )
        if raw.get("coordinate_reference") != RECTANGLE_COORDINATE_REFERENCE:
            raise InputManifestError(
                f"frame {frame_index} rectangle_occlusion has unsupported "
                "coordinate reference"
            )
        return {
            "type": transform_type,
            "coordinate_space": "normalized",
            "coordinate_reference": RECTANGLE_COORDINATE_REFERENCE,
            "rectangle": _rectangle(
                raw.get("rectangle"), name=f"frame {frame_index} rectangle"
            ),
            "fill": list(_fill(raw.get("fill"), name=f"frame {frame_index} fill")),
        }

    if transform_type in {"source_frame_substitution", "temporal_reorder"}:
        required = {
            "type",
            "corruption",
            "original_source_index",
            "replacement_source_index",
        }
        if set(raw) != required:
            raise InputManifestError(
                f"frame {frame_index} {transform_type} has unexpected fields"
            )
        original = _plain_int(
            raw.get("original_source_index"),
            name=f"frame {frame_index} original_source_index",
        )
        replacement = _plain_int(
            raw.get("replacement_source_index"),
            name=f"frame {frame_index} replacement_source_index",
        )
        if original != frame_index or replacement != source_index:
            raise InputManifestError(
                f"frame {frame_index} provenance indices disagree with final frame"
            )
        if replacement < 0 or replacement >= source_frame_count:
            raise InputManifestError(
                f"frame {frame_index} replacement_source_index is out of bounds"
            )
        expected_corruption = (
            "low_overlap_jump"
            if transform_type == "source_frame_substitution"
            else "wrong_order_segment"
        )
        if raw.get("corruption") != expected_corruption:
            raise InputManifestError(
                f"frame {frame_index} {transform_type} has wrong corruption tag"
            )
        return copy.deepcopy(dict(raw))

    raise InputManifestError(
        f"frame {frame_index} has unknown transform type {transform_type!r}"
    )


def load_input_manifest(path: str | os.PathLike[str]) -> InputManifest:
    """Load and fully validate a persisted corruption manifest."""

    manifest_path = Path(path)
    try:
        manifest_path = manifest_path.resolve(strict=True)
    except OSError as error:
        raise InputManifestError(f"input manifest does not exist: {path}") from error
    payload, _ = _load_json_object(manifest_path, name="input manifest")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise InputManifestError(
            f"unsupported input manifest schema {payload.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if payload.get("source_is_read_only") is not True:
        raise InputManifestError("input manifest must mark source_is_read_only=true")
    if payload.get("materialization") != MATERIALIZATION_MODE:
        raise InputManifestError(
            f"input manifest materialization must be {MATERIALIZATION_MODE!r}"
        )
    if payload.get("index_convention") != INDEX_CONVENTION:
        raise InputManifestError(
            f"input manifest index_convention must be {INDEX_CONVENTION!r}"
        )

    frame_count = _plain_int(payload.get("frame_count"), name="frame_count")
    source_frame_count = _plain_int(
        payload.get("source_frame_count"), name="source_frame_count"
    )
    if frame_count < 1 or source_frame_count < 1:
        raise InputManifestError("frame counts must be positive")
    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list) or len(raw_frames) != frame_count:
        raise InputManifestError(
            f"frames must contain exactly frame_count={frame_count} entries"
        )

    source_manifest_text = _nonempty_path(
        payload.get("source_manifest"), name="source_manifest"
    )
    source_manifest_path = Path(source_manifest_text)
    if not source_manifest_path.is_absolute():
        source_manifest_path = manifest_path.parent / source_manifest_path
    try:
        source_manifest_path = source_manifest_path.resolve(strict=True)
    except OSError as error:
        raise InputManifestError(
            f"source_manifest does not exist: {source_manifest_path}"
        ) from error
    source_payload, source_bytes = _load_json_object(
        source_manifest_path, name="source manifest"
    )
    expected_sha256 = payload.get("source_manifest_sha256")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise InputManifestError("source_manifest_sha256 must be lowercase SHA-256")
    actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise InputManifestError("source manifest SHA-256 no longer matches")

    is_formal_source = (
        source_payload.get("schema_version") == FORMAL_SOURCE_SCHEMA_VERSION
    )
    is_recovery_quality_source = (
        source_payload.get("schema_version") == RECOVERY_QUALITY_SOURCE_SCHEMA_VERSION
    )
    source_frames = _source_frames(
        source_payload,
        base_dir=source_manifest_path.parent,
        reject_aliases=is_formal_source,
    )
    if len(source_frames) != source_frame_count:
        raise InputManifestError(
            "source_frame_count does not match source manifest frame count"
        )
    source_output_frame_count, has_declared_source_pool = _source_output_frame_count(
        source_payload, source_frame_count=source_frame_count
    )
    if frame_count != source_output_frame_count:
        raise InputManifestError(
            "frame_count does not match source output_frame_count "
            f"({source_output_frame_count})"
        )
    if is_formal_source:
        _validate_formal_source_provenance(
            source_payload,
            source_frames,
            source_manifest_path=source_manifest_path,
            input_payload=payload,
        )
    if has_declared_source_pool:
        _validate_source_pool_disjoint(
            source_frames, output_frame_count=source_output_frame_count
        )
    else:
        _validate_declared_source_hashes(source_frames)
    expected_corruptions, expected_replacements = _corruption_expectations(
        payload,
        frame_count=frame_count,
        source_frame_count=source_frame_count,
        donor_pool_start=(
            source_output_frame_count if has_declared_source_pool else None
        ),
        allow_empty_clean_control=is_recovery_quality_source,
    )

    frames: list[ManifestFrame] = []
    for position, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, Mapping):
            raise InputManifestError(f"frame {position} must be a JSON object")
        frame_index = _plain_int(
            raw_frame.get("frame_index"), name=f"frame {position} frame_index"
        )
        source_index = _plain_int(
            raw_frame.get("source_index"), name=f"frame {position} source_index"
        )
        if frame_index != position:
            raise InputManifestError(
                f"frame_index {frame_index} is out of order at position {position}"
            )
        if source_index < 0 or source_index >= source_frame_count:
            raise InputManifestError(
                f"frame {position} source_index {source_index} is out of bounds"
            )

        path_text = _nonempty_path(
            raw_frame.get("path"), name=f"frame {position} path"
        )
        final_path = Path(path_text)
        if not final_path.is_absolute():
            final_path = source_manifest_path.parent / final_path
        try:
            final_path = final_path.resolve(strict=True)
        except OSError as error:
            raise InputManifestError(
                f"frame {position} path does not exist: {final_path}"
            ) from error
        if not final_path.is_file():
            raise InputManifestError(f"frame {position} path is not a file: {final_path}")
        source_path, source_metadata = source_frames[source_index]
        if final_path != source_path:
            raise InputManifestError(
                f"frame {position} path disagrees with source_index {source_index}"
            )
        metadata = raw_frame.get("metadata")
        if not isinstance(metadata, Mapping) or dict(metadata) != source_metadata:
            raise InputManifestError(
                f"frame {position} metadata disagrees with source_index {source_index}"
            )

        raw_transforms = raw_frame.get("transforms")
        if not isinstance(raw_transforms, list):
            raise InputManifestError(f"frame {position} transforms must be a list")
        transforms = tuple(
            _validated_transform(
                transform,
                frame_index=frame_index,
                source_index=source_index,
                source_frame_count=source_frame_count,
            )
            for transform in raw_transforms
        )
        expected_corruption = expected_corruptions[position]
        expected_transform = (
            CORRUPTION_TO_TRANSFORM[expected_corruption]
            if expected_corruption is not None
            else None
        )
        actual_types = [transform["type"] for transform in transforms]
        if actual_types != ([] if expected_transform is None else [expected_transform]):
            raise InputManifestError(
                f"frame {position} transforms do not match corruption labels"
            )
        expected_replacement = expected_replacements[position]
        if expected_replacement is not None and source_index != expected_replacement:
            raise InputManifestError(
                f"frame {position} replacement provenance disagrees with "
                "corruption label"
            )
        if expected_replacement is None and source_index != frame_index:
            raise InputManifestError(
                f"non-replacement frame {position} must retain its source_index"
            )
        frames.append(
            ManifestFrame(
                frame_index=frame_index,
                source_index=source_index,
                path=final_path,
                metadata=copy.deepcopy(dict(metadata)),
                transforms=transforms,
            )
        )
    return InputManifest(
        path=manifest_path,
        source_manifest_path=source_manifest_path,
        frames=tuple(frames),
    )


def _copy_image(image: Any, *, frame_index: int) -> Any:
    shape_value = getattr(image, "shape", None)
    try:
        shape = tuple(int(dimension) for dimension in shape_value)
    except (TypeError, ValueError) as error:
        raise InputManifestError(
            f"view {frame_index} img must expose a BxCxHxW shape"
        ) from error
    if len(shape) != 4 or shape[0] < 1 or shape[1] != 3 or min(shape[2:]) < 1:
        raise InputManifestError(
            f"view {frame_index} img must have shape Bx3xHxW, got {shape}"
        )

    if isinstance(image, np.ndarray):
        if not np.issubdtype(image.dtype, np.floating):
            raise InputManifestError(
                f"view {frame_index} NumPy img must have floating dtype"
            )
        return image.copy()
    clone = getattr(image, "clone", None)
    if not callable(clone):
        raise InputManifestError(
            f"view {frame_index} img must be a NumPy array or expose clone()"
        )
    floating_check = getattr(image, "is_floating_point", None)
    if callable(floating_check) and not bool(floating_check()):
        raise InputManifestError(
            f"view {frame_index} tensor-like img must have floating dtype"
        )
    copied = clone()
    if copied is image:
        raise InputManifestError(f"view {frame_index} clone() returned the source object")
    return copied


def _apply_rectangle(image: Any, transform: Mapping[str, Any]) -> None:
    _, _, height, width = (int(dimension) for dimension in image.shape)
    rectangle = transform["rectangle"]
    x_start = max(0, min(width, math.floor(rectangle["x"] * width)))
    x_end = max(
        x_start + 1,
        min(width, math.ceil((rectangle["x"] + rectangle["width"]) * width)),
    )
    y_start = max(0, min(height, math.floor(rectangle["y"] * height)))
    y_end = max(
        y_start + 1,
        min(height, math.ceil((rectangle["y"] + rectangle["height"]) * height)),
    )
    normalized_fill = [channel / 127.5 - 1.0 for channel in transform["fill"]]
    try:
        for channel, value in enumerate(normalized_fill):
            image[:, channel, y_start:y_end, x_start:x_end] = value
    except Exception as error:
        raise InputManifestError(
            "view image does not support in-place BxCxHxW channel assignment"
        ) from error


def apply_deferred_transforms(
    views: Sequence[Mapping[str, Any]], manifest: InputManifest
) -> list[dict[str, Any]]:
    """Clone every loaded view and apply only deferred pixel transforms."""

    if isinstance(views, (str, bytes)) or not isinstance(views, Sequence):
        raise InputManifestError("view loader must return a sequence of mappings")
    if len(views) != len(manifest.frames):
        raise InputManifestError(
            f"view loader returned {len(views)} views; expected {len(manifest.frames)}"
        )
    output: list[dict[str, Any]] = []
    for frame, view in zip(manifest.frames, views, strict=True):
        if not isinstance(view, Mapping) or "img" not in view:
            raise InputManifestError(
                f"view {frame.frame_index} must be a mapping with an img field"
            )
        copied_view = dict(view)
        copied_image = _copy_image(view["img"], frame_index=frame.frame_index)
        copied_view["img"] = copied_image
        for transform in frame.transforms:
            if transform["type"] == "rectangle_occlusion":
                _apply_rectangle(copied_image, transform)
            # Substitution/reordering already changed the ordered path list.
        output.append(copied_view)
    return output


def materialize_manifest_views(
    path: str | os.PathLike[str],
    loader: Callable[[list[str]], Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Load final ordered paths through ``loader`` and replay deferred transforms."""

    if not callable(loader):
        raise InputManifestError("loader must be callable")
    manifest = load_input_manifest(path)
    ordered_paths = [str(frame.path) for frame in manifest.frames]
    views = loader(ordered_paths)
    return apply_deferred_transforms(views, manifest)
