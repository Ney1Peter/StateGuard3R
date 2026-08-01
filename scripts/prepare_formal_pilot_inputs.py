#!/usr/bin/env python3
"""Prepare the frozen six-run TUM formal-pilot inputs without copying images.

The preparer is intentionally narrow.  It reads the three official TUM index
files and the selected RGB/depth files, constructs source and corruption JSON
manifests, proves that the *actually consumed* RGB/depth/ground-truth entries
are disjoint, and atomically publishes one directory.  It never materializes
an image transform, runs a model, or modifies a source file.

RGB indices in this module are zero-based data-entry indices: comments and
blank lines in ``rgb.txt`` do not count.  Physical source lines remain
one-based and are recorded separately in every manifest.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Mapping, Sequence

from stateguard3r.corruption import (
    RECTANGLE_COORDINATE_REFERENCE,
    build_corruption_manifest,
)
from stateguard3r.input_manifest import load_input_manifest
from stateguard3r.detection_suite import (
    DetectionSuiteInput,
    make_holdout_commitment,
    make_run_commitment,
    make_split_registry,
)


SCHEMA_VERSION = "stateguard3r.formal-pilot-inputs.v1"
SOURCE_SCHEMA_VERSION = "stateguard3r.tum-formal-pilot-source.v1"
RAW_MANIFEST_SCHEMA_VERSION = "stateguard3r.tum-raw-audit.v1"
DATASET_NAME = "rgbd_dataset_freiburg1_desk"
FRAME_COUNT = 30
EVENT_START = 15
SEED = 0
MAX_ASSOCIATION_DELTA = Decimal("0.02")
EXPLORATORY_RGB_INDICES = frozenset(range(17, 47))
INDEX_CONVENTION = (
    "zero_based_data_entry_index_excluding_comments_and_blank_lines"
)
PHYSICAL_LINE_CONVENTION = "one_based_physical_source_line"
REGISTRY_FILENAME = "formal-pilot-manifest.json"
SOURCE_MANIFEST_FILENAME = "source-manifest.json"
INPUT_MANIFEST_FILENAME = "input-manifest.json"
SPLIT_REGISTRY_FILENAME = "split-registry.json"
HOLDOUT_COMMITMENT_FILENAME = "holdout-commitment.json"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TUM_ROOT = (
    REPOSITORY_ROOT.parent
    / "baselines"
    / "ReCal3R"
    / "data"
    / "tum"
    / DATASET_NAME
)
EXPECTED_TUM_ARCHIVE = EXPECTED_TUM_ROOT.with_suffix(".tgz")
EXPECTED_TUM_ARCHIVE_SIZE_BYTES = 344_011_403
EXPECTED_TUM_ARCHIVE_SHA256 = (
    "e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c"
)
EXPECTED_RAW_MANIFEST = (
    REPOSITORY_ROOT.parent
    / "baselines"
    / "ReCal3R"
    / "logs"
    / "gate2-fr1-desk-raw-manifest.json"
)
EXPECTED_RAW_MANIFEST_SIZE_BYTES = 231_139
EXPECTED_RAW_MANIFEST_SHA256 = (
    "5908db0f357fd4a21b2c777220de38e651b48b80c7d53182123c6eb4e7163d87"
)
FORMAL_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs"

DYNAMIC_PARAMETERS = {
    "coordinate_space": "normalized",
    "rectangle": {"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5},
    "velocity": {"dx": 0.04, "dy": 0.025},
    "fill": [255, 0, 0],
}


class FormalPilotInputError(ValueError):
    """Raised when formal-pilot provenance cannot be proved safely."""


@dataclass(frozen=True, slots=True)
class _FormalSourcePolicy:
    """Exact immutable source identities accepted by the formal preparer.

    The production CLI always uses :data:`PRODUCTION_SOURCE_POLICY`.  Tests may
    inject an isolated policy through the private ``_source_policy`` argument;
    no CLI flag can weaken these identities.
    """

    dataset_root: Path
    archive_path: Path
    archive_size_bytes: int
    archive_sha256: str
    raw_manifest_path: Path
    raw_manifest_size_bytes: int
    raw_manifest_sha256: str
    allowed_output_root: Path


PRODUCTION_SOURCE_POLICY = _FormalSourcePolicy(
    dataset_root=EXPECTED_TUM_ROOT,
    archive_path=EXPECTED_TUM_ARCHIVE,
    archive_size_bytes=EXPECTED_TUM_ARCHIVE_SIZE_BYTES,
    archive_sha256=EXPECTED_TUM_ARCHIVE_SHA256,
    raw_manifest_path=EXPECTED_RAW_MANIFEST,
    raw_manifest_size_bytes=EXPECTED_RAW_MANIFEST_SIZE_BYTES,
    raw_manifest_sha256=EXPECTED_RAW_MANIFEST_SHA256,
    allowed_output_root=FORMAL_OUTPUT_ROOT,
)


@dataclass(frozen=True, slots=True)
class RunSpec:
    run_id: str
    dataset_split: str
    corruption_type: str
    base_start: int
    donor_start: int | None = None

    @property
    def base_indices(self) -> tuple[int, ...]:
        return tuple(range(self.base_start, self.base_start + FRAME_COUNT))

    @property
    def donor_indices(self) -> tuple[int, ...]:
        if self.donor_start is None:
            return ()
        return tuple(range(self.donor_start, self.donor_start + 5))

    @property
    def pool_indices(self) -> tuple[int, ...]:
        return self.base_indices + self.donor_indices


_FROZEN_RUN_SPECS = (
    RunSpec(
        "development-dynamic",
        "development",
        "dynamic_occlusion",
        248,
    ),
    RunSpec(
        "development-wrong",
        "development",
        "wrong_order_segment",
        320,
    ),
    RunSpec(
        "development-low",
        "development",
        "low_overlap_jump",
        380,
        donor_start=300,
    ),
    RunSpec(
        "holdout-dynamic",
        "holdout",
        "dynamic_occlusion",
        520,
    ),
    RunSpec(
        "holdout-wrong",
        "holdout",
        "wrong_order_segment",
        550,
    ),
    RunSpec(
        "holdout-low",
        "holdout",
        "low_overlap_jump",
        583,
        donor_start=500,
    ),
)
RUN_SPECS = _FROZEN_RUN_SPECS


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    path: Path
    device: int
    inode: int
    mode: int
    link_count: int
    size_bytes: int
    mtime_ns: int
    sha256: str

    def artifact(self, *, path_text: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "mode_octal": f"{self.mode:04o}",
            "link_count": self.link_count,
            "device": self.device,
            "inode": self.inode,
            "mtime_ns": self.mtime_ns,
        }
        if path_text is not None:
            result["path"] = path_text
        return result


@dataclass(frozen=True, slots=True)
class _TumEntry:
    source_entry_index: int
    source_line: int
    physical_line_sha256: str
    physical_line_size_bytes: int
    timestamp_text: str
    timestamp_decimal: Decimal
    timestamp: float
    path_text: str | None
    values_text: tuple[str, ...]
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _AssociatedFrame:
    rgb: _TumEntry
    depth: _TumEntry
    groundtruth: _TumEntry
    rgb_path: Path
    depth_path: Path
    rgb_snapshot: _FileSnapshot
    depth_snapshot: _FileSnapshot


def _strict_decimal(text: str, *, name: str) -> tuple[Decimal, float]:
    try:
        decimal_value = Decimal(text)
    except InvalidOperation as error:
        raise FormalPilotInputError(f"{name} is not a decimal timestamp") from error
    if not decimal_value.is_finite():
        raise FormalPilotInputError(f"{name} must be finite")
    try:
        float_value = float(decimal_value)
    except (OverflowError, ValueError) as error:
        raise FormalPilotInputError(f"{name} is outside the JSON number range") from error
    if not math.isfinite(float_value):
        raise FormalPilotInputError(f"{name} is outside the JSON number range")
    return decimal_value, float_value


def _strict_float(text: str, *, name: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise FormalPilotInputError(f"{name} is not a number") from error
    if not math.isfinite(value):
        raise FormalPilotInputError(f"{name} must be finite")
    return value


def _read_file_snapshot(path: Path) -> tuple[_FileSnapshot, bytes]:
    try:
        resolved = path.resolve(strict=True)
        with resolved.open("rb") as stream:
            before = os.fstat(stream.fileno())
            digest = hashlib.sha256()
            chunks: list[bytes] = []
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                chunks.append(chunk)
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise FormalPilotInputError(f"cannot read source file {path}: {error}") from error
    identity_before = (
        before.st_dev,
        before.st_ino,
        stat.S_IFMT(before.st_mode),
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        stat.S_IFMT(after.st_mode),
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise FormalPilotInputError(f"source file changed while hashing: {resolved}")
    if not stat.S_ISREG(before.st_mode):
        raise FormalPilotInputError(f"source path is not a regular file: {resolved}")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise FormalPilotInputError(f"source file size changed while reading: {resolved}")
    return (
        _FileSnapshot(
            path=resolved,
            device=before.st_dev,
            inode=before.st_ino,
            mode=stat.S_IMODE(before.st_mode),
            link_count=before.st_nlink,
            size_bytes=before.st_size,
            mtime_ns=before.st_mtime_ns,
            sha256=digest.hexdigest(),
        ),
        payload,
    )


def _snapshot_file(path: Path) -> _FileSnapshot:
    return _read_file_snapshot(path)[0]


def _strict_json_object(payload: bytes, *, name: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise FormalPilotInputError(
            f"{name} contains non-finite JSON constant {value!r}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FormalPilotInputError(
                    f"{name} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalPilotInputError(f"{name} is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise FormalPilotInputError(f"{name} must contain a JSON object")
    return value


def _lower_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FormalPilotInputError(f"{name} must be a lowercase SHA-256")
    return value


def _plain_nonnegative_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise FormalPilotInputError(f"{name} must be a non-negative integer")
    return value


def _require_real_read_only_directory(path: Path, *, role: str) -> Path:
    absolute = Path(os.path.abspath(path))
    try:
        metadata = os.lstat(absolute)
    except OSError as error:
        raise FormalPilotInputError(f"cannot inspect {role} directory: {absolute}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise FormalPilotInputError(
            f"{role} must be a real non-symlink directory: {absolute}"
        )
    if stat.S_IMODE(metadata.st_mode) & 0o222:
        raise FormalPilotInputError(
            f"{role} directory must be read-only (no write bits): {absolute}"
        )
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise FormalPilotInputError(f"cannot resolve {role} directory: {absolute}") from error
    if resolved != absolute:
        raise FormalPilotInputError(
            f"{role} directory may not resolve through a symlink: {absolute}"
        )
    return resolved


def _snapshot_policy_file(path: Path, *, role: str) -> tuple[_FileSnapshot, bytes]:
    absolute = Path(os.path.abspath(path))
    try:
        metadata = os.lstat(absolute)
    except OSError as error:
        raise FormalPilotInputError(f"cannot inspect {role}: {absolute}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise FormalPilotInputError(
            f"{role} must be a regular non-symlink file: {absolute}"
        )
    snapshot, payload = _read_file_snapshot(absolute)
    if snapshot.path != absolute:
        raise FormalPilotInputError(f"{role} may not resolve through a symlink: {absolute}")
    _require_read_only(snapshot, role=role)
    return snapshot, payload


def _require_read_only(snapshot: _FileSnapshot, *, role: str) -> None:
    if snapshot.mode & 0o222:
        raise FormalPilotInputError(
            f"{role} source must be read-only (no write bits): {snapshot.path}"
        )


def _assert_snapshot_unchanged(snapshot: _FileSnapshot) -> None:
    current = _snapshot_file(snapshot.path)
    if current != snapshot:
        raise FormalPilotInputError(
            f"source file changed during manifest preparation: {snapshot.path}"
        )


def _raw_manifest_file_entries(
    payload: bytes,
    *,
    policy: _FormalSourcePolicy,
    archive_snapshot: _FileSnapshot,
) -> tuple[Mapping[str, Any], dict[str, dict[str, Any]]]:
    raw_manifest = _strict_json_object(payload, name="frozen raw manifest")
    if raw_manifest.get("schema_version") != RAW_MANIFEST_SCHEMA_VERSION:
        raise FormalPilotInputError("frozen raw manifest schema_version mismatch")
    if raw_manifest.get("status") != "PASS":
        raise FormalPilotInputError("frozen raw manifest status must be PASS")
    if raw_manifest.get("dataset_name") != DATASET_NAME:
        raise FormalPilotInputError("frozen raw manifest dataset_name mismatch")
    if (
        raw_manifest.get("archive_sha256") != archive_snapshot.sha256
        or raw_manifest.get("archive_size_bytes") != archive_snapshot.size_bytes
    ):
        raise FormalPilotInputError(
            "frozen raw manifest archive identity does not match the actual TGZ"
        )
    if (
        archive_snapshot.sha256 != policy.archive_sha256
        or archive_snapshot.size_bytes != policy.archive_size_bytes
    ):
        raise FormalPilotInputError("official TUM TGZ exact size/SHA-256 mismatch")

    raw_files = raw_manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise FormalPilotInputError("frozen raw manifest files must be non-empty")
    entries: dict[str, dict[str, Any]] = {}
    for index, raw_entry in enumerate(raw_files):
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != {
            "path",
            "size_bytes",
            "sha256",
            "mode",
        }:
            raise FormalPilotInputError(
                f"frozen raw manifest files[{index}] has invalid fields"
            )
        path_text = raw_entry.get("path")
        if not isinstance(path_text, str) or not path_text or "\x00" in path_text:
            raise FormalPilotInputError(
                f"frozen raw manifest files[{index}].path is invalid"
            )
        relative = Path(path_text)
        if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
            raise FormalPilotInputError(
                f"frozen raw manifest files[{index}].path is unsafe"
            )
        canonical_path = relative.as_posix()
        if canonical_path != path_text or canonical_path in entries:
            raise FormalPilotInputError(
                f"frozen raw manifest contains duplicate/non-canonical path {path_text!r}"
            )
        mode = raw_entry.get("mode")
        if not isinstance(mode, str) or len(mode) != 4 or any(
            character not in "01234567" for character in mode
        ):
            raise FormalPilotInputError(
                f"frozen raw manifest files[{index}].mode is invalid"
            )
        if int(mode, 8) & 0o222:
            raise FormalPilotInputError(
                f"frozen raw manifest file is not read-only: {path_text}"
            )
        entries[canonical_path] = {
            "path": canonical_path,
            "size_bytes": _plain_nonnegative_int(
                raw_entry.get("size_bytes"),
                name=f"frozen raw manifest files[{index}].size_bytes",
            ),
            "sha256": _lower_sha256(
                raw_entry.get("sha256"),
                name=f"frozen raw manifest files[{index}].sha256",
            ),
            "mode": mode,
        }
    if raw_manifest.get("regular_file_count") != len(entries):
        raise FormalPilotInputError("frozen raw manifest regular_file_count mismatch")
    if raw_manifest.get("total_regular_bytes") != sum(
        entry["size_bytes"] for entry in entries.values()
    ):
        raise FormalPilotInputError("frozen raw manifest total_regular_bytes mismatch")
    return raw_manifest, entries


def _assert_raw_manifest_match(
    snapshot: _FileSnapshot,
    *,
    source_root: Path,
    entries: Mapping[str, Mapping[str, Any]],
    role: str,
) -> str:
    try:
        relative = snapshot.path.relative_to(source_root).as_posix()
    except ValueError as error:
        raise FormalPilotInputError(
            f"{role} resolves outside the frozen TUM root: {snapshot.path}"
        ) from error
    declared = entries.get(relative)
    if declared is None:
        raise FormalPilotInputError(
            f"{role} is absent from the frozen raw manifest: {relative}"
        )
    actual = {
        "path": relative,
        "size_bytes": snapshot.size_bytes,
        "sha256": snapshot.sha256,
        "mode": f"{snapshot.mode:04o}",
    }
    if actual != dict(declared):
        raise FormalPilotInputError(
            f"{role} path/size/SHA-256/mode differs from frozen raw manifest: {relative}"
        )
    return relative


def _parse_tum_index(
    snapshot: _FileSnapshot,
    payload: bytes,
    *,
    kind: str,
) -> tuple[_TumEntry, ...]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise FormalPilotInputError(f"{snapshot.path} is not UTF-8") from error
    raw_lines = payload.splitlines(keepends=True)
    text_lines = text.splitlines(keepends=True)
    if len(raw_lines) != len(text_lines):
        raise FormalPilotInputError(f"cannot preserve physical lines in {snapshot.path}")

    expected_tokens = 8 if kind == "groundtruth" else 2
    entries: list[_TumEntry] = []
    for source_line, (raw_line, text_line) in enumerate(
        zip(raw_lines, text_lines, strict=True), start=1
    ):
        stripped = text_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) != expected_tokens:
            raise FormalPilotInputError(
                f"{snapshot.path}:{source_line} must contain exactly "
                f"{expected_tokens} fields for {kind}"
            )
        entry_index = len(entries)
        timestamp_decimal, timestamp = _strict_decimal(
            tokens[0], name=f"{snapshot.path}:{source_line} timestamp"
        )
        if kind == "groundtruth":
            values_text = tuple(tokens[1:])
            values = tuple(
                _strict_float(
                    token,
                    name=f"{snapshot.path}:{source_line} pose field {index}",
                )
                for index, token in enumerate(values_text, start=1)
            )
            path_text = None
        else:
            raw_path = tokens[1]
            if not raw_path or "\x00" in raw_path or Path(raw_path).is_absolute():
                raise FormalPilotInputError(
                    f"{snapshot.path}:{source_line} must use a non-empty relative path"
                )
            path_text = raw_path
            values_text = ()
            values = ()
        entries.append(
            _TumEntry(
                source_entry_index=entry_index,
                source_line=source_line,
                physical_line_sha256=hashlib.sha256(raw_line).hexdigest(),
                physical_line_size_bytes=len(raw_line),
                timestamp_text=tokens[0],
                timestamp_decimal=timestamp_decimal,
                timestamp=timestamp,
                path_text=path_text,
                values_text=values_text,
                values=values,
            )
        )
    if not entries:
        raise FormalPilotInputError(f"{snapshot.path} contains no {kind} data entries")
    return tuple(entries)


def _resolve_dataset_file(root: Path, entry: _TumEntry, *, kind: str) -> Path:
    if entry.path_text is None:
        raise FormalPilotInputError(f"{kind} entry does not contain a file path")
    unresolved = root / entry.path_text
    try:
        link_metadata = os.lstat(unresolved)
        if stat.S_ISLNK(link_metadata.st_mode):
            raise FormalPilotInputError(
                f"{kind} entry {entry.source_entry_index} may not reference a symlink: "
                f"{entry.path_text}"
            )
        candidate = unresolved.resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, ValueError) as error:
        raise FormalPilotInputError(
            f"{kind} entry {entry.source_entry_index} escapes the TUM root or is missing: "
            f"{entry.path_text}"
        ) from error
    if not candidate.is_file():
        raise FormalPilotInputError(
            f"{kind} entry {entry.source_entry_index} is not a regular file: {candidate}"
        )
    if candidate != Path(os.path.abspath(unresolved)):
        raise FormalPilotInputError(
            f"{kind} entry {entry.source_entry_index} resolves through an alias: "
            f"{entry.path_text}"
        )
    return candidate


def _nearest_unique(
    rgb: _TumEntry,
    candidates: Sequence[_TumEntry],
    *,
    kind: str,
) -> _TumEntry:
    distances = [
        (abs(candidate.timestamp_decimal - rgb.timestamp_decimal), candidate)
        for candidate in candidates
    ]
    minimum = min(distance for distance, _ in distances)
    nearest = [candidate for distance, candidate in distances if distance == minimum]
    if len(nearest) != 1:
        tied = ", ".join(str(item.source_entry_index) for item in nearest)
        raise FormalPilotInputError(
            f"RGB entry {rgb.source_entry_index} has a non-unique nearest {kind} "
            f"association at entries {tied}"
        )
    if minimum > MAX_ASSOCIATION_DELTA:
        raise FormalPilotInputError(
            f"RGB entry {rgb.source_entry_index} nearest {kind} delta {minimum} "
            f"exceeds {MAX_ASSOCIATION_DELTA} seconds"
        )
    return nearest[0]


def _relative_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, start=base)).as_posix()


def _line_provenance(entry: _TumEntry) -> dict[str, Any]:
    return {
        "source_entry_index": entry.source_entry_index,
        "source_line": entry.source_line,
        "physical_line_sha256": entry.physical_line_sha256,
        "physical_line_size_bytes": entry.physical_line_size_bytes,
        "timestamp": entry.timestamp,
        "timestamp_text": entry.timestamp_text,
    }


def _association_metadata(
    associated: _AssociatedFrame,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    rgb = associated.rgb
    depth = associated.depth
    groundtruth = associated.groundtruth
    depth_delta = depth.timestamp_decimal - rgb.timestamp_decimal
    groundtruth_delta = groundtruth.timestamp_decimal - rgb.timestamp_decimal
    return {
        **_line_provenance(rgb),
        "raw_rgb_source_index": rgb.source_entry_index,
        "raw_rgb_source_line": rgb.source_line,
        "rgb_sha256": associated.rgb_snapshot.sha256,
        "rgb_size_bytes": associated.rgb_snapshot.size_bytes,
        "rgb_mode_octal": f"{associated.rgb_snapshot.mode:04o}",
        "rgb_link_count": associated.rgb_snapshot.link_count,
        "rgb_device": associated.rgb_snapshot.device,
        "rgb_inode": associated.rgb_snapshot.inode,
        "rgb_mtime_ns": associated.rgb_snapshot.mtime_ns,
        "depth": {
            **_line_provenance(depth),
            "path": _relative_path(associated.depth_path, run_dir),
            "sha256": associated.depth_snapshot.sha256,
            "size_bytes": associated.depth_snapshot.size_bytes,
            "mode_octal": f"{associated.depth_snapshot.mode:04o}",
            "link_count": associated.depth_snapshot.link_count,
            "device": associated.depth_snapshot.device,
            "inode": associated.depth_snapshot.inode,
            "mtime_ns": associated.depth_snapshot.mtime_ns,
            "delta_from_rgb_seconds": float(depth_delta),
            "absolute_delta_seconds": float(abs(depth_delta)),
        },
        "groundtruth": {
            **_line_provenance(groundtruth),
            "translation_xyz": list(groundtruth.values[:3]),
            "quaternion_xyzw": list(groundtruth.values[3:]),
            "translation_xyz_text": list(groundtruth.values_text[:3]),
            "quaternion_xyzw_text": list(groundtruth.values_text[3:]),
            "delta_from_rgb_seconds": float(groundtruth_delta),
            "absolute_delta_seconds": float(abs(groundtruth_delta)),
        },
    }


def _validate_frozen_specs(specs: Sequence[RunSpec]) -> None:
    run_ids = [spec.run_id for spec in specs]
    if len(run_ids) != 6 or len(set(run_ids)) != 6:
        raise FormalPilotInputError("formal pilot must contain exactly six unique run IDs")
    for split in ("development", "holdout"):
        split_specs = [spec for spec in specs if spec.dataset_split == split]
        if len(split_specs) != 3 or {
            spec.corruption_type for spec in split_specs
        } != {"dynamic_occlusion", "low_overlap_jump", "wrong_order_segment"}:
            raise FormalPilotInputError(
                f"{split} must contain each formal corruption type exactly once"
            )
    all_pool_indices = [index for spec in specs for index in spec.pool_indices]
    forbidden = sorted(set(all_pool_indices) & EXPLORATORY_RGB_INDICES)
    if forbidden:
        raise FormalPilotInputError(
            "formal pilot may not reuse exploratory RGB indices: "
            + ", ".join(str(index) for index in forbidden)
        )
    if len(all_pool_indices) != len(set(all_pool_indices)):
        raise FormalPilotInputError("formal source pools overlap in raw RGB entry indices")
    for spec in specs:
        if len(spec.base_indices) != FRAME_COUNT:
            raise FormalPilotInputError(f"{spec.run_id} base must contain 30 RGB entries")
        expected_donors = 5 if spec.corruption_type == "low_overlap_jump" else 0
        if len(spec.donor_indices) != expected_donors:
            raise FormalPilotInputError(
                f"{spec.run_id} has an invalid low-overlap donor pool"
            )
    if tuple(specs) != _FROZEN_RUN_SPECS:
        raise FormalPilotInputError(
            "formal-pilot run specification is not the frozen v1 layout"
        )


def _corruption_spec(spec: RunSpec) -> dict[str, Any]:
    if spec.corruption_type == "dynamic_occlusion":
        return {
            "type": spec.corruption_type,
            "start": EVENT_START,
            "end": EVENT_START + 4,
            "parameters": json.loads(json.dumps(DYNAMIC_PARAMETERS)),
        }
    if spec.corruption_type == "low_overlap_jump":
        return {
            "type": spec.corruption_type,
            "start": EVENT_START,
            "end": EVENT_START + 4,
            "parameters": {"source_start": FRAME_COUNT},
        }
    return {
        "type": spec.corruption_type,
        "start": EVENT_START,
        "end": EVENT_START + 3,
        "parameters": {"mode": "reverse"},
    }


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise FormalPilotInputError("pose-proxy summary requires finite values")
    return {
        "minimum": min(values),
        "mean": math.fsum(values) / len(values),
        "maximum": max(values),
    }


def _low_overlap_gt_pose_proxy(
    pool: Sequence[_AssociatedFrame],
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    translation_distances: list[float] = []
    rotation_angles: list[float] = []
    for offset in range(5):
        output_position = EVENT_START + offset
        base_pool_index = output_position
        donor_pool_index = FRAME_COUNT + offset
        base = pool[base_pool_index].groundtruth
        donor = pool[donor_pool_index].groundtruth
        translation_distance = math.sqrt(
            math.fsum(
                (left - right) ** 2
                for left, right in zip(base.values[:3], donor.values[:3], strict=True)
            )
        )
        base_quaternion = base.values[3:]
        donor_quaternion = donor.values[3:]
        base_norm = math.sqrt(math.fsum(value * value for value in base_quaternion))
        donor_norm = math.sqrt(math.fsum(value * value for value in donor_quaternion))
        if base_norm <= 0.0 or donor_norm <= 0.0:
            raise FormalPilotInputError(
                "low-overlap GT-pose proxy requires non-zero quaternions"
            )
        quaternion_dot = math.fsum(
            left * right
            for left, right in zip(
                base_quaternion, donor_quaternion, strict=True
            )
        ) / (base_norm * donor_norm)
        rotation_angle = 2.0 * math.acos(
            min(1.0, max(0.0, abs(quaternion_dot)))
        )
        if not math.isfinite(translation_distance) or not math.isfinite(
            rotation_angle
        ):
            raise FormalPilotInputError("low-overlap GT-pose proxy is non-finite")
        translation_distances.append(translation_distance)
        rotation_angles.append(rotation_angle)
        pairs.append(
            {
                "output_position": output_position,
                "base_source_pool_index": base_pool_index,
                "donor_source_pool_index": donor_pool_index,
                "base_rgb_source_entry_index": pool[
                    base_pool_index
                ].rgb.source_entry_index,
                "donor_rgb_source_entry_index": pool[
                    donor_pool_index
                ].rgb.source_entry_index,
                "translation_l2_meters": translation_distance,
                "rotation_angle_radians": rotation_angle,
            }
        )
    return {
        "status": "gt_pose_proxy_only_not_measured_image_overlap",
        "interpretation": (
            "larger pose separation is only a donor-selection proxy; image overlap "
            "was not measured and no overlap percentage is claimed"
        ),
        "rotation_formula": (
            "2*acos(clamp(abs(dot(normalize(q_base),normalize(q_donor))),0,1))"
        ),
        "pairs": pairs,
        "summary": {
            "translation_l2_meters": _summary(translation_distances),
            "rotation_angle_radians": _summary(rotation_angles),
        },
    }


def _json_bytes(payload: Any) -> bytes:
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise FormalPilotInputError(f"cannot serialize strict JSON: {error}") from error
    return (text + "\n").encode("utf-8")


def _bytes_artifact(payload: bytes, *, path: str) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _assert_unique_associations(
    frames: Sequence[_AssociatedFrame], *, run_id: str
) -> None:
    for kind, identities in (
        ("RGB", [frame.rgb.source_entry_index for frame in frames]),
        ("depth", [frame.depth.source_entry_index for frame in frames]),
        (
            "groundtruth",
            [frame.groundtruth.source_entry_index for frame in frames],
        ),
    ):
        if len(identities) != len(set(identities)):
            raise FormalPilotInputError(
                f"{run_id} reuses a nearest {kind} association inside its source pool"
            )


def _identity_tokens(frame: _AssociatedFrame, kind: str) -> tuple[Any, ...]:
    if kind == "rgb":
        snapshot = frame.rgb_snapshot
        return (
            frame.rgb.source_entry_index,
            str(frame.rgb_path),
            (snapshot.device, snapshot.inode),
            snapshot.sha256,
        )
    if kind == "depth":
        snapshot = frame.depth_snapshot
        return (
            frame.depth.source_entry_index,
            str(frame.depth_path),
            (snapshot.device, snapshot.inode),
            snapshot.sha256,
        )
    return (
        frame.groundtruth.source_entry_index,
        frame.groundtruth.source_line,
        frame.groundtruth.physical_line_sha256,
    )


def _assert_runs_disjoint(
    frames_by_run: Mapping[str, Sequence[_AssociatedFrame]],
    *,
    scope: str,
    expected_frame_counts: Mapping[str, int],
    count_field: str,
) -> dict[str, Any]:
    proof: dict[str, Any] = {
        "scope": scope,
        "development_holdout_disjoint": True,
        "pairwise_intersection_count": {},
        "identity_checks": {
            "rgb": ["source_entry_index", "resolved_path", "device_inode", "sha256"],
            "depth": [
                "source_entry_index",
                "resolved_path",
                "device_inode",
                "sha256",
            ],
            "groundtruth": [
                "source_entry_index",
                "physical_source_line",
                "physical_line_sha256",
            ],
        },
    }
    owners: dict[str, dict[Any, str]] = {
        kind: {} for kind in ("rgb", "depth", "groundtruth")
    }
    run_ids = list(frames_by_run)
    for run_id, frames in frames_by_run.items():
        if len(frames) != expected_frame_counts[run_id]:
            raise FormalPilotInputError(
                f"{run_id} has {len(frames)} frames in disjointness scope; "
                f"expected {expected_frame_counts[run_id]}"
            )
        for kind in owners:
            per_run: set[Any] = set()
            for frame in frames:
                tokens = _identity_tokens(frame, kind)
                for token_type, token in enumerate(tokens):
                    key = (token_type, token)
                    if key in per_run:
                        raise FormalPilotInputError(
                            f"{run_id} contains duplicate consumed {kind} identity: {token}"
                        )
                    per_run.add(key)
                    previous = owners[kind].get(key)
                    if previous is not None:
                        raise FormalPilotInputError(
                            f"consumed {kind} association overlap between {previous} "
                            f"and {run_id}: {token}"
                        )
                    owners[kind][key] = run_id

    for left_index, left_id in enumerate(run_ids):
        for right_id in run_ids[left_index + 1 :]:
            pair = f"{left_id}|{right_id}"
            proof["pairwise_intersection_count"][pair] = {
                kind: 0 for kind in owners
            }
    proof[count_field] = {
        kind: sum(len(frames) for frames in frames_by_run.values()) for kind in owners
    }
    return proof


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        path.chmod(0o444)
    except OSError as error:
        raise FormalPilotInputError(f"cannot stage manifest {path}: {error}") from error


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _freeze_staging_tree(staging_root: Path) -> None:
    directories = [
        path for path in staging_root.rglob("*") if path.is_dir()
    ]
    for directory in sorted(
        directories, key=lambda path: len(path.parts), reverse=True
    ):
        directory.chmod(0o555)
        _fsync_directory(directory)
    staging_root.chmod(0o555)
    _fsync_directory(staging_root)


def _remove_staging_tree(staging_root: Path) -> None:
    if not staging_root.exists():
        return
    for directory in [staging_root, *staging_root.rglob("*")]:
        if directory.is_dir() and not directory.is_symlink():
            directory.chmod(0o700)
    shutil.rmtree(staging_root)


def _validate_staging_tree(
    staging_root: Path,
    run_records: Sequence[Mapping[str, Any]],
) -> None:
    try:
        registry_payload = json.loads(
            (staging_root / REGISTRY_FILENAME).read_bytes().decode("utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalPilotInputError("cannot reload the staged pilot registry") from error
    if not isinstance(registry_payload, Mapping) or registry_payload.get(
        "runs"
    ) != list(run_records):
        raise FormalPilotInputError(
            "staged registry run records differ from the validated build records"
        )
    split_registry_path = staging_root / SPLIT_REGISTRY_FILENAME
    holdout_commitment_path = staging_root / HOLDOUT_COMMITMENT_FILENAME
    split_registry_bytes = split_registry_path.read_bytes()
    holdout_commitment_bytes = holdout_commitment_path.read_bytes()
    expected_commitment_artifacts = {
        "split_registry": _bytes_artifact(
            split_registry_bytes, path=SPLIT_REGISTRY_FILENAME
        ),
        "holdout_commitment": _bytes_artifact(
            holdout_commitment_bytes, path=HOLDOUT_COMMITMENT_FILENAME
        ),
    }
    if registry_payload.get(
        "formal_commitment_artifacts"
    ) != expected_commitment_artifacts:
        raise FormalPilotInputError(
            "staged canonical commitment artifact hashes differ from the registry"
        )
    development_commitments = [
        record["suite_run_commitment"]
        for record in run_records
        if record["dataset_split"] == "development"
    ]
    holdout_commitments = [
        record["suite_run_commitment"]
        for record in run_records
        if record["dataset_split"] == "holdout"
    ]
    if split_registry_bytes != make_split_registry(
        development_commitments, holdout_commitments
    ):
        raise FormalPilotInputError(
            "staged split registry is not the canonical run-commitment projection"
        )
    if holdout_commitment_bytes != make_holdout_commitment(holdout_commitments):
        raise FormalPilotInputError(
            "staged holdout commitment is not the canonical holdout projection"
        )

    expected_files = {staging_root / REGISTRY_FILENAME}
    expected_files.update(
        {
            staging_root / SPLIT_REGISTRY_FILENAME,
            staging_root / HOLDOUT_COMMITMENT_FILENAME,
        }
    )
    for record in run_records:
        run_dir = staging_root / record["dataset_split"] / record["run_id"]
        source_path = run_dir / SOURCE_MANIFEST_FILENAME
        input_path = run_dir / INPUT_MANIFEST_FILENAME
        expected_files.update(
            {
                source_path,
                input_path,
            }
        )
        source_bytes = source_path.read_bytes()
        input_bytes = input_path.read_bytes()
        expected_source_artifact = _bytes_artifact(
            source_bytes,
            path=source_path.relative_to(staging_root).as_posix(),
        )
        expected_input_artifact = _bytes_artifact(
            input_bytes,
            path=input_path.relative_to(staging_root).as_posix(),
        )
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, Mapping) or artifacts != {
            "source_manifest": expected_source_artifact,
            "input_manifest": expected_input_artifact,
            "corruption_json": expected_input_artifact,
        }:
            raise FormalPilotInputError(
                f"staged {record['run_id']} artifact commitment mismatch"
            )
        try:
            source_payload = json.loads(source_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FormalPilotInputError(
                f"staged {record['run_id']} source manifest is invalid JSON"
            ) from error
        if not isinstance(source_payload, Mapping):
            raise FormalPilotInputError(
                f"staged {record['run_id']} source manifest is not an object"
            )
        source_frames = source_payload.get("frames")
        if not isinstance(source_frames, list):
            raise FormalPilotInputError(
                f"staged {record['run_id']} source frames are missing"
            )
        derived_pool_indices: list[int] = []
        for pool_position, frame in enumerate(source_frames):
            if not isinstance(frame, Mapping):
                raise FormalPilotInputError(
                    f"staged {record['run_id']} source frame is not an object"
                )
            if frame.get("source_pool_index") != pool_position:
                raise FormalPilotInputError(
                    f"staged {record['run_id']} source-pool position mismatch"
                )
            source_entry_index = frame.get("source_entry_index")
            if type(source_entry_index) is not int:
                raise FormalPilotInputError(
                    f"staged {record['run_id']} source entry index is invalid"
                )
            derived_pool_indices.append(source_entry_index)
        if (
            derived_pool_indices != record["source_pool_rgb_indices"]
            or len(source_frames) != record["source_pool_frame_count"]
        ):
            raise FormalPilotInputError(
                f"staged {record['run_id']} source-pool registry mismatch"
            )

        loaded = load_input_manifest(input_path)
        if len(loaded.frames) != FRAME_COUNT:
            raise FormalPilotInputError(
                f"staged {record['run_id']} input manifest is not 30 frames"
            )
        derived_positions = [frame.source_index for frame in loaded.frames]
        derived_associations = {
            "rgb_source_entry_indices": [],
            "depth_source_entry_indices": [],
            "groundtruth_source_entry_indices": [],
        }
        derived_rgb_shas: list[str] = []
        for frame in loaded.frames:
            metadata = frame.metadata
            depth = metadata.get("depth")
            groundtruth = metadata.get("groundtruth")
            if not isinstance(depth, Mapping) or not isinstance(
                groundtruth, Mapping
            ):
                raise FormalPilotInputError(
                    f"staged {record['run_id']} association metadata is missing"
                )
            derived_associations["rgb_source_entry_indices"].append(
                metadata.get("source_entry_index")
            )
            derived_associations["depth_source_entry_indices"].append(
                depth.get("source_entry_index")
            )
            derived_associations["groundtruth_source_entry_indices"].append(
                groundtruth.get("source_entry_index")
            )
            derived_rgb_shas.append(metadata.get("rgb_sha256"))
        if derived_positions != record[
            "consumed_source_pool_indices_by_output_position"
        ]:
            raise FormalPilotInputError(
                f"staged {record['run_id']} consumed source positions mismatch"
            )
        if derived_associations != record["consumed_associations"]:
            raise FormalPilotInputError(
                f"staged {record['run_id']} consumed associations mismatch"
            )
        if sorted(derived_rgb_shas) != record["raw_frame_sha256s"]:
            raise FormalPilotInputError(
                f"staged {record['run_id']} consumed RGB SHA set mismatch"
            )
    actual_files = set(staging_root.rglob("*"))
    regular_files = {path for path in actual_files if path.is_file()}
    if regular_files != expected_files:
        unexpected = sorted(str(path) for path in regular_files ^ expected_files)
        raise FormalPilotInputError(
            "staging tree contains missing or unexpected files: " + ", ".join(unexpected)
        )
    for path in actual_files:
        if path.is_symlink():
            raise FormalPilotInputError(f"staging tree may not contain symlinks: {path}")
        if path.is_file():
            if path.suffix != ".json" or not stat.S_ISREG(path.stat().st_mode):
                raise FormalPilotInputError(
                    f"staging tree may contain JSON regular files only: {path}"
                )
            try:
                json.loads(path.read_bytes().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise FormalPilotInputError(f"staged manifest is invalid JSON: {path}") from error


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    try:
        rename_noreplace = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as error:
        raise FormalPilotInputError(
            "atomic no-replace directory publication is unavailable"
        ) from error
    rename_noreplace.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename_noreplace.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace_flag = 1
    result = rename_noreplace(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace_flag,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FormalPilotInputError(
                f"refusing to replace existing path during atomic publication: "
                f"{destination}"
            )
        raise FormalPilotInputError(
            "cannot atomically rename directory with no-replace "
            f"semantics: {os.strerror(error_number)}"
        )


def _publish_directory(staging_root: Path, output_root: Path) -> None:
    if os.path.lexists(output_root):
        raise FormalPilotInputError(
            f"refusing to replace existing formal-pilot output: {output_root}"
        )
    _rename_directory_noreplace(staging_root, output_root)
    try:
        _fsync_directory(output_root.parent)
    except OSError as error:
        try:
            _rename_directory_noreplace(output_root, staging_root)
        except (FormalPilotInputError, OSError) as rollback_error:
            raise FormalPilotInputError(
                "published formal-pilot directory but parent fsync failed and "
                f"atomic rollback also failed: {rollback_error}"
            ) from error
        try:
            _fsync_directory(output_root.parent)
        except OSError:
            pass
        raise FormalPilotInputError(
            "cannot fsync published formal-pilot parent; publication was "
            f"atomically rolled back: {error}"
        ) from error


def prepare_formal_pilot_inputs(
    tum_root: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    _source_policy: _FormalSourcePolicy | None = None,
) -> dict[str, Path]:
    """Build and atomically publish the immutable formal-pilot manifest tree."""

    _validate_frozen_specs(RUN_SPECS)
    policy = PRODUCTION_SOURCE_POLICY if _source_policy is None else _source_policy
    if not isinstance(policy, _FormalSourcePolicy):
        raise FormalPilotInputError("formal source policy is invalid")
    requested_source_root = Path(os.path.abspath(tum_root))
    expected_source_root = Path(os.path.abspath(policy.dataset_root))
    if requested_source_root != expected_source_root:
        raise FormalPilotInputError(
            "formal TUM root must equal the frozen official raw root: "
            f"{expected_source_root}"
        )
    source_root = _require_real_read_only_directory(
        requested_source_root, role="formal TUM root"
    )
    source_directories = (
        source_root,
        _require_real_read_only_directory(source_root / "rgb", role="TUM RGB"),
        _require_real_read_only_directory(source_root / "depth", role="TUM depth"),
    )

    archive_snapshot, _ = _snapshot_policy_file(
        policy.archive_path, role="official TUM TGZ"
    )
    if (
        archive_snapshot.size_bytes != policy.archive_size_bytes
        or archive_snapshot.sha256 != _lower_sha256(
            policy.archive_sha256, name="formal source policy archive_sha256"
        )
    ):
        raise FormalPilotInputError("official TUM TGZ exact size/SHA-256 mismatch")
    raw_manifest_snapshot, raw_manifest_bytes = _snapshot_policy_file(
        policy.raw_manifest_path, role="frozen TUM raw manifest"
    )
    if (
        raw_manifest_snapshot.size_bytes != policy.raw_manifest_size_bytes
        or raw_manifest_snapshot.sha256 != _lower_sha256(
            policy.raw_manifest_sha256,
            name="formal source policy raw_manifest_sha256",
        )
    ):
        raise FormalPilotInputError("frozen TUM raw manifest exact size/SHA-256 mismatch")
    raw_manifest, raw_manifest_entries = _raw_manifest_file_entries(
        raw_manifest_bytes,
        policy=policy,
        archive_snapshot=archive_snapshot,
    )

    output_path = Path(output_dir)
    if not output_path.name:
        raise FormalPilotInputError("output directory must have a final path component")
    output_root = output_path.resolve(strict=False)
    allowed_output_root = Path(os.path.abspath(policy.allowed_output_root))
    try:
        allowed_metadata = os.lstat(allowed_output_root)
    except OSError as error:
        raise FormalPilotInputError(
            f"formal output root does not exist: {allowed_output_root}"
        ) from error
    if stat.S_ISLNK(allowed_metadata.st_mode) or not stat.S_ISDIR(
        allowed_metadata.st_mode
    ):
        raise FormalPilotInputError(
            f"formal output root must be a real directory: {allowed_output_root}"
        )
    if allowed_output_root.resolve(strict=True) != allowed_output_root:
        raise FormalPilotInputError("formal output root may not resolve through a symlink")
    try:
        output_relative = output_root.relative_to(allowed_output_root)
    except ValueError as error:
        raise FormalPilotInputError(
            f"formal output directory must be inside {allowed_output_root}"
        ) from error
    if not output_relative.parts or output_root == allowed_output_root:
        raise FormalPilotInputError(
            "formal output directory must be a new child of the formal output root"
        )
    try:
        output_root.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise FormalPilotInputError("output directory may not be inside the read-only TUM root")
    try:
        source_root.relative_to(output_root)
    except ValueError:
        pass
    else:
        raise FormalPilotInputError("output directory may not contain the TUM source root")
    if os.path.lexists(output_root):
        raise FormalPilotInputError(
            f"refusing to replace existing formal-pilot output: {output_root}"
        )

    index_snapshots: dict[str, _FileSnapshot] = {}
    entries: dict[str, tuple[_TumEntry, ...]] = {}
    for kind, filename in (
        ("rgb", "rgb.txt"),
        ("depth", "depth.txt"),
        ("groundtruth", "groundtruth.txt"),
    ):
        snapshot, payload = _snapshot_policy_file(
            source_root / filename, role=f"TUM {filename}"
        )
        try:
            snapshot.path.relative_to(source_root)
        except ValueError as error:
            raise FormalPilotInputError(
                f"TUM {filename} resolves outside the dataset root: {snapshot.path}"
            ) from error
        index_snapshots[kind] = snapshot
        entries[kind] = _parse_tum_index(snapshot, payload, kind=kind)
        _assert_raw_manifest_match(
            snapshot,
            source_root=source_root,
            entries=raw_manifest_entries,
            role=f"TUM {filename}",
        )

    maximum_index = max(index for spec in RUN_SPECS for index in spec.pool_indices)
    if maximum_index >= len(entries["rgb"]):
        raise FormalPilotInputError(
            f"rgb.txt has {len(entries['rgb'])} data entries; frozen pilot needs "
            f"entry {maximum_index}"
        )

    snapshot_cache: dict[Path, _FileSnapshot] = {
        snapshot.path: snapshot for snapshot in index_snapshots.values()
    }
    snapshot_cache[archive_snapshot.path] = archive_snapshot
    snapshot_cache[raw_manifest_snapshot.path] = raw_manifest_snapshot

    def cached_snapshot(path: Path) -> _FileSnapshot:
        snapshot = snapshot_cache.get(path)
        if snapshot is None:
            snapshot = _snapshot_file(path)
            _require_read_only(snapshot, role="selected RGB/depth")
            snapshot_cache[path] = snapshot
        return snapshot

    associated_by_rgb_index: dict[int, _AssociatedFrame] = {}
    for rgb_index in sorted({index for spec in RUN_SPECS for index in spec.pool_indices}):
        rgb = entries["rgb"][rgb_index]
        depth = _nearest_unique(rgb, entries["depth"], kind="depth")
        groundtruth = _nearest_unique(
            rgb, entries["groundtruth"], kind="groundtruth"
        )
        rgb_path = _resolve_dataset_file(source_root, rgb, kind="RGB")
        depth_path = _resolve_dataset_file(source_root, depth, kind="depth")
        associated_by_rgb_index[rgb_index] = _AssociatedFrame(
            rgb=rgb,
            depth=depth,
            groundtruth=groundtruth,
            rgb_path=rgb_path,
            depth_path=depth_path,
            rgb_snapshot=cached_snapshot(rgb_path),
            depth_snapshot=cached_snapshot(depth_path),
        )

    for rgb_index, associated in associated_by_rgb_index.items():
        _assert_raw_manifest_match(
            associated.rgb_snapshot,
            source_root=source_root,
            entries=raw_manifest_entries,
            role=f"allocated RGB entry {rgb_index}",
        )
        _assert_raw_manifest_match(
            associated.depth_snapshot,
            source_root=source_root,
            entries=raw_manifest_entries,
            role=f"allocated depth association for RGB entry {rgb_index}",
        )

    source_pools: dict[str, tuple[_AssociatedFrame, ...]] = {}
    for spec in RUN_SPECS:
        pool = tuple(associated_by_rgb_index[index] for index in spec.pool_indices)
        _assert_unique_associations(pool, run_id=spec.run_id)
        source_pools[spec.run_id] = pool

    consumed: dict[str, tuple[_AssociatedFrame, ...]] = {}
    consumed_pool_positions: dict[str, tuple[int, ...]] = {}
    for spec in RUN_SPECS:
        positions = list(range(FRAME_COUNT))
        if spec.corruption_type == "low_overlap_jump":
            positions[EVENT_START : EVENT_START + 5] = list(
                range(FRAME_COUNT, FRAME_COUNT + 5)
            )
        elif spec.corruption_type == "wrong_order_segment":
            positions[EVENT_START : EVENT_START + 4] = reversed(
                positions[EVENT_START : EVENT_START + 4]
            )
        consumed_pool_positions[spec.run_id] = tuple(positions)
        consumed[spec.run_id] = tuple(source_pools[spec.run_id][index] for index in positions)
    allocation_disjointness_proof = _assert_runs_disjoint(
        source_pools,
        scope="all_six_complete_base_and_donor_source_pools",
        expected_frame_counts={
            spec.run_id: len(spec.pool_indices) for spec in RUN_SPECS
        },
        count_field="unique_allocated_count",
    )
    disjointness_proof = _assert_runs_disjoint(
        consumed,
        scope="all_six_runs_actual_model_inputs_including_low_overlap_donors",
        expected_frame_counts={spec.run_id: FRAME_COUNT for spec in RUN_SPECS},
        count_field="unique_consumed_count",
    )
    official_lineage = {
        "dataset_root": str(source_root),
        "archive": archive_snapshot.artifact(
            path_text=str(archive_snapshot.path)
        ),
        "raw_manifest": raw_manifest_snapshot.artifact(
            path_text=str(raw_manifest_snapshot.path)
        ),
        "raw_manifest_schema_version": raw_manifest["schema_version"],
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.",
            suffix=".staging",
            dir=output_root.parent,
        )
    )
    published = False
    try:
        run_records: list[dict[str, Any]] = []
        staged_payloads: list[tuple[Path, bytes]] = []
        for spec in RUN_SPECS:
            final_run_dir = output_root / spec.dataset_split / spec.run_id
            staged_run_dir = staging_root / spec.dataset_split / spec.run_id
            staged_run_dir.mkdir(parents=True)
            pool = source_pools[spec.run_id]
            source_frames = [
                {
                    "path": _relative_path(frame.rgb_path, final_run_dir),
                    **_association_metadata(frame, run_dir=final_run_dir),
                    "source_pool_index": pool_index,
                    "source_pool_role": (
                        "base" if pool_index < FRAME_COUNT else "low_overlap_donor"
                    ),
                }
                for pool_index, frame in enumerate(pool)
            ]
            source_manifest = {
                "schema_version": SOURCE_SCHEMA_VERSION,
                "sequence": f"{spec.run_id}-source-pool",
                "dataset": DATASET_NAME,
                "dataset_split": spec.dataset_split,
                "run_id": spec.run_id,
                "corruption_type": spec.corruption_type,
                "source_is_read_only": True,
                "official_lineage": official_lineage,
                "file_path_semantics": "relative_to_this_manifest_directory",
                "index_convention": INDEX_CONVENTION,
                "physical_line_convention": PHYSICAL_LINE_CONVENTION,
                "association_policy": {
                    "method": "unique_nearest_absolute_timestamp",
                    "tie_policy": "reject",
                    "reuse_policy": "reject_within_source_pool_and_across_consumed_runs",
                    "max_absolute_delta_seconds": float(MAX_ASSOCIATION_DELTA),
                },
                "frame_count": len(source_frames),
                "output_frame_count": FRAME_COUNT,
                "ground_truth_path": _relative_path(
                    index_snapshots["groundtruth"].path, final_run_dir
                ),
                "tum_index_files": {
                    kind: {
                        **index_snapshots[kind].artifact(
                            path_text=_relative_path(
                                index_snapshots[kind].path, final_run_dir
                            )
                        ),
                        "data_entry_count": len(entries[kind]),
                    }
                    for kind in ("rgb", "depth", "groundtruth")
                },
                "frames": source_frames,
            }
            source_bytes = _json_bytes(source_manifest)
            input_manifest = build_corruption_manifest(
                source_manifest,
                [_corruption_spec(spec)],
                seed=SEED,
                check_paths=True,
                source_base_dir=staged_run_dir,
                source_manifest_path=SOURCE_MANIFEST_FILENAME,
                source_manifest_sha256=hashlib.sha256(source_bytes).hexdigest(),
                source_ground_truth_resolved_path=str(
                    index_snapshots["groundtruth"].path
                ),
                source_ground_truth_sha256=index_snapshots["groundtruth"].sha256,
                sequence=spec.run_id,
            )
            input_bytes = _json_bytes(input_manifest)
            source_relative = (
                Path(spec.dataset_split) / spec.run_id / SOURCE_MANIFEST_FILENAME
            ).as_posix()
            input_relative = (
                Path(spec.dataset_split) / spec.run_id / INPUT_MANIFEST_FILENAME
            ).as_posix()
            consumed_frames = consumed[spec.run_id]
            record = {
                "run_id": spec.run_id,
                "dataset_split": spec.dataset_split,
                "corruption_type": spec.corruption_type,
                "execution_position_within_split": [
                    candidate
                    for candidate in RUN_SPECS
                    if candidate.dataset_split == spec.dataset_split
                ].index(spec),
                "seed": SEED,
                "base_rgb_source_indices": list(spec.base_indices),
                "donor_rgb_source_indices": list(spec.donor_indices),
                "source_pool_rgb_indices": list(spec.pool_indices),
                "source_pool_frame_count": len(pool),
                "output_frame_count": FRAME_COUNT,
                "consumed_source_pool_indices_by_output_position": list(
                    consumed_pool_positions[spec.run_id]
                ),
                "consumed_associations": {
                    "rgb_source_entry_indices": [
                        frame.rgb.source_entry_index for frame in consumed_frames
                    ],
                    "depth_source_entry_indices": [
                        frame.depth.source_entry_index for frame in consumed_frames
                    ],
                    "groundtruth_source_entry_indices": [
                        frame.groundtruth.source_entry_index for frame in consumed_frames
                    ],
                },
                "raw_frame_sha256s": sorted(
                    frame.rgb_snapshot.sha256 for frame in consumed_frames
                ),
                "artifacts": {
                    "source_manifest": _bytes_artifact(
                        source_bytes, path=source_relative
                    ),
                    "input_manifest": _bytes_artifact(
                        input_bytes, path=input_relative
                    ),
                    "corruption_json": _bytes_artifact(
                        input_bytes, path=input_relative
                    ),
                },
            }
            suite_commitment = make_run_commitment(
                DetectionSuiteInput(
                    run_id=spec.run_id,
                    dataset_split=spec.dataset_split,
                    corruption_type=spec.corruption_type,
                    raw_frame_sha256s=tuple(record["raw_frame_sha256s"]),
                    corruption_json=input_bytes,
                    input_manifest_json=input_bytes,
                    source_manifest_json=source_bytes,
                )
            )
            for artifact_name, suite_artifact in suite_commitment[
                "artifacts"
            ].items():
                registry_artifact = record["artifacts"][artifact_name]
                if {
                    "sha256": registry_artifact["sha256"],
                    "size_bytes": registry_artifact["size_bytes"],
                } != suite_artifact:
                    raise FormalPilotInputError(
                        f"{spec.run_id} suite/registry artifact adapter mismatch"
                    )
            record["suite_run_commitment"] = suite_commitment
            if spec.corruption_type == "low_overlap_jump":
                record["low_overlap_gt_pose_proxy"] = _low_overlap_gt_pose_proxy(
                    pool
                )
            if len(set(record["raw_frame_sha256s"])) != FRAME_COUNT:
                raise FormalPilotInputError(
                    f"{spec.run_id} does not have 30 unique consumed RGB SHA-256 values"
                )
            run_records.append(record)
            staged_payloads.extend(
                [
                    (staged_run_dir / SOURCE_MANIFEST_FILENAME, source_bytes),
                    (staged_run_dir / INPUT_MANIFEST_FILENAME, input_bytes),
                ]
            )

        development_commitments = [
            record["suite_run_commitment"]
            for record in run_records
            if record["dataset_split"] == "development"
        ]
        holdout_commitments = [
            record["suite_run_commitment"]
            for record in run_records
            if record["dataset_split"] == "holdout"
        ]
        split_registry_bytes = make_split_registry(
            development_commitments, holdout_commitments
        )
        holdout_commitment_bytes = make_holdout_commitment(holdout_commitments)
        registry = {
            "schema_version": SCHEMA_VERSION,
            "dataset": DATASET_NAME,
            "source_root": str(source_root),
            "source_is_read_only": True,
            "official_lineage": official_lineage,
            "materialization": "manifests_only_no_copy_no_symlink",
            "formal_status": "pre_forward_inputs_only",
            "seed": SEED,
            "frame_count": FRAME_COUNT,
            "event_start": EVENT_START,
            "index_convention": INDEX_CONVENTION,
            "physical_line_convention": PHYSICAL_LINE_CONVENTION,
            "forbidden_exploratory_rgb_range": {"start": 17, "end": 46},
            "association_policy": {
                "method": "unique_nearest_absolute_timestamp",
                "tie_policy": "reject",
                "max_absolute_delta_seconds": float(MAX_ASSOCIATION_DELTA),
            },
            "fixed_execution_order": {
                split: [
                    spec.run_id
                    for spec in RUN_SPECS
                    if spec.dataset_split == split
                ]
                for split in ("development", "holdout")
            },
            "formal_commitment_artifacts": {
                "split_registry": _bytes_artifact(
                    split_registry_bytes, path=SPLIT_REGISTRY_FILENAME
                ),
                "holdout_commitment": _bytes_artifact(
                    holdout_commitment_bytes,
                    path=HOLDOUT_COMMITMENT_FILENAME,
                ),
            },
            "allocation_disjointness_proof": allocation_disjointness_proof,
            "disjointness_proof": disjointness_proof,
            "runs": run_records,
        }
        registry_bytes = _json_bytes(registry)
        for path, payload in staged_payloads:
            _write_exclusive(path, payload)
        _write_exclusive(
            staging_root / SPLIT_REGISTRY_FILENAME, split_registry_bytes
        )
        _write_exclusive(
            staging_root / HOLDOUT_COMMITMENT_FILENAME,
            holdout_commitment_bytes,
        )
        _write_exclusive(staging_root / REGISTRY_FILENAME, registry_bytes)
        _validate_staging_tree(staging_root, run_records)
        for snapshot in snapshot_cache.values():
            _assert_snapshot_unchanged(snapshot)
        for directory in source_directories:
            _require_real_read_only_directory(directory, role="formal TUM source")
        _freeze_staging_tree(staging_root)
        _publish_directory(staging_root, output_root)
        try:
            for snapshot in snapshot_cache.values():
                _assert_snapshot_unchanged(snapshot)
            for directory in source_directories:
                _require_real_read_only_directory(directory, role="formal TUM source")
        except (FormalPilotInputError, OSError) as error:
            try:
                _rename_directory_noreplace(output_root, staging_root)
            except (FormalPilotInputError, OSError) as rollback_error:
                raise FormalPilotInputError(
                    "post-publication source verification failed and atomic "
                    f"rollback also failed: {rollback_error}"
                ) from error
            try:
                _fsync_directory(output_root.parent)
            except OSError:
                pass
            raise
        published = True
    finally:
        if not published and staging_root.exists():
            _remove_staging_tree(staging_root)

    result = {
        "registry": output_root / REGISTRY_FILENAME,
        "split_registry": output_root / SPLIT_REGISTRY_FILENAME,
        "holdout_commitment": output_root / HOLDOUT_COMMITMENT_FILENAME,
    }
    result.update(
        {
            spec.run_id: output_root
            / spec.dataset_split
            / spec.run_id
            / INPUT_MANIFEST_FILENAME
            for spec in RUN_SPECS
        }
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the frozen six-run TUM formal pilot as JSON manifests only; "
            "the destination must not already exist."
        )
    )
    parser.add_argument("tum_root", help="read-only TUM fr1_desk sequence root")
    parser.add_argument("output_dir", help="new directory for immutable JSON manifests")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        outputs = prepare_formal_pilot_inputs(args.tum_root, args.output_dir)
    except (FormalPilotInputError, OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"prepared six manifest-only formal-pilot runs at "
        f"{outputs['registry'].parent}; no images were copied or linked"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
