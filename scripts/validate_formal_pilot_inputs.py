#!/usr/bin/env python3
"""CPU-only, immutable validation of the six formal-pilot input manifests.

This program deliberately imports only the tracked smoke runner's image-loader
adapter.  It never accepts a checkpoint, constructs a model, or performs a
ReCal3R forward.  The resulting report is consumed by
``run_formal_detection_pilot.py commit``.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import errno
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


CPU_VALIDATION_SCHEMA_VERSION = "stateguard3r.formal-input-cpu-validation.v1"
INPUT_SCHEMA_VERSION = "stateguard3r.formal-pilot-inputs.v1"
SOURCE_SCHEMA_VERSION = "stateguard3r.tum-formal-pilot-source.v1"
INPUT_MANIFEST_SCHEMA_VERSION = "stateguard3r.corruption.v1"
INPUT_REGISTRY_FILENAME = "formal-pilot-manifest.json"
SOURCE_MANIFEST_FILENAME = "source-manifest.json"
INPUT_MANIFEST_FILENAME = "input-manifest.json"
SPLIT_REGISTRY_FILENAME = "split-registry.json"
HOLDOUT_COMMITMENT_FILENAME = "holdout-commitment.json"
FRAME_COUNT = 30
EVENT_START = 15
IMAGE_SIZE = 512
EXPECTED_SOURCE_ARTIFACT_COUNT = 385
EXPECTED_ALLOCATED_FRAME_COUNT = 190
EXPECTED_RECAL3R_COMMIT = "466c7cdf3acd2f589f1d82e5f6391966f19db9ff"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPOSITORY_ROOT / "outputs"
RECAL3R_ROOT = REPOSITORY_ROOT.parent / "baselines" / "ReCal3R"
EXPECTED_TUM_ROOT = (
    RECAL3R_ROOT / "data" / "tum" / "rgbd_dataset_freiburg1_desk"
)
RUNNER_RELATIVE_PATH = Path("scripts/run_recal3r_smoke.py")
GENERATOR_RELATIVE_PATH = Path("scripts/validate_formal_pilot_inputs.py")
OFFICIAL_LOADER_RELATIVE_PATH = Path("src/dust3r/utils/image.py")
PRE_FORWARD_VALIDATION_DIRNAME = "pre-forward-validation"
CPU_VALIDATION_FILENAME = "cpu-validation.json"
MAX_ASSOCIATION_DELTA = Decimal("0.02")
CPU_VALIDATION_CHECKS = frozenset(
    {
        "strict_manifest_validation",
        "cpu_pixel_replay",
        "raw_source_snapshot_unchanged",
        "input_tree_snapshot_unchanged",
        "official_tum_associations",
        "production_runtime_provenance",
        "cuda_hidden",
        "six_run_coverage",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DYNAMIC_RECTANGLE = {"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5}
_DYNAMIC_VELOCITY = {"dx": 0.04, "dy": 0.025}
_DYNAMIC_FILL = [255, 0, 0]
_DYNAMIC_COORDINATE_REFERENCE = "model_input_after_resize_and_center_crop"
_EXPECTED_CORRUPTION_EFFECTS = {
    "low_overlap_jump": "pose drift and local geometry inconsistency",
    "dynamic_occlusion": "static-state contamination and ghost geometry",
    "wrong_order_segment": (
        "temporal inconsistency and recurrent-state update instability"
    ),
}


class FormalPilotValidationError(ValueError):
    """Raised when CPU validation cannot prove the frozen formal inputs."""


@dataclass(frozen=True, slots=True)
class _FrozenRunSpec:
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

    @property
    def consumed_pool_positions(self) -> tuple[int, ...]:
        positions = list(range(FRAME_COUNT))
        if self.corruption_type == "low_overlap_jump":
            positions[EVENT_START : EVENT_START + 5] = range(30, 35)
        elif self.corruption_type == "wrong_order_segment":
            positions[EVENT_START : EVENT_START + 4] = (18, 17, 16, 15)
        return tuple(positions)


FROZEN_RUN_SPECS = (
    _FrozenRunSpec(
        "development-dynamic", "development", "dynamic_occlusion", 248
    ),
    _FrozenRunSpec(
        "development-wrong", "development", "wrong_order_segment", 320
    ),
    _FrozenRunSpec(
        "development-low", "development", "low_overlap_jump", 380, 300
    ),
    _FrozenRunSpec("holdout-dynamic", "holdout", "dynamic_occlusion", 520),
    _FrozenRunSpec("holdout-wrong", "holdout", "wrong_order_segment", 550),
    _FrozenRunSpec("holdout-low", "holdout", "low_overlap_jump", 583, 500),
)


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

    def artifact(self, path_text: str) -> dict[str, Any]:
        return {
            "path": path_text,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    def canonical_record(self) -> dict[str, Any]:
        return {
            "canonical_path": str(self.path),
            "device": self.device,
            "inode": self.inode,
            "mode_octal": f"{self.mode:04o}",
            "link_count": self.link_count,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class _TreeEntrySnapshot:
    relative_path: str
    kind: str
    device: int
    inode: int
    mode: int
    link_count: int
    size_bytes: int
    mtime_ns: int
    sha256: str | None

    def canonical_record(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "kind": self.kind,
            "device": self.device,
            "inode": self.inode,
            "mode_octal": f"{self.mode:04o}",
            "link_count": self.link_count,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "sha256": self.sha256,
        }


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
class _DeclaredSource:
    path: Path
    declaration: Mapping[str, Any]
    prefix: str
    role: str
    kind: str


@dataclass(frozen=True, slots=True)
class _PreparedRun:
    spec: _FrozenRunSpec
    record: Mapping[str, Any]
    run_root: Path
    source_path: Path
    input_path: Path
    source_snapshot: _FileSnapshot
    input_snapshot: _FileSnapshot
    source_payload: Mapping[str, Any]
    input_payload: Mapping[str, Any]


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FormalPilotValidationError(
            f"cannot encode canonical strict JSON: {error}"
        ) from error


def _strict_json_object(payload: bytes, *, name: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise FormalPilotValidationError(
            f"{name} contains non-finite JSON constant {value!r}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FormalPilotValidationError(
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
        raise FormalPilotValidationError(
            f"{name} is not strict UTF-8 JSON: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise FormalPilotValidationError(f"{name} must contain a JSON object")
    return value


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FormalPilotValidationError(f"{name} must be a lowercase SHA-256")
    return value


def _plain_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise FormalPilotValidationError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _git_output(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FormalPilotValidationError(
            f"cannot inspect Git provenance for {repository}: {error}"
        ) from error
    return completed.stdout.strip()


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FormalPilotValidationError(
            f"cannot inspect Git bytes for {repository}: {error}"
        ) from error
    return completed.stdout


def _canonical_directory(path: Path, *, name: str, read_only: bool = False) -> Path:
    absolute = Path(os.path.abspath(path))
    try:
        metadata = os.lstat(absolute)
    except OSError as error:
        raise FormalPilotValidationError(
            f"cannot inspect {name} directory {absolute}: {error}"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise FormalPilotValidationError(
            f"{name} must be a real non-symlink directory: {absolute}"
        )
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise FormalPilotValidationError(
            f"cannot resolve {name} directory: {absolute}"
        ) from error
    if resolved != absolute:
        raise FormalPilotValidationError(
            f"{name} directory may not resolve through a symlink: {absolute}"
        )
    if read_only and stat.S_IMODE(metadata.st_mode) & 0o222:
        raise FormalPilotValidationError(f"{name} directory must be read-only")
    return resolved


def _read_snapshot(
    path: Path, *, name: str, capture_payload: bool = False
) -> tuple[_FileSnapshot, bytes | None]:
    absolute = Path(os.path.abspath(path))
    try:
        link_metadata = os.lstat(absolute)
    except OSError as error:
        raise FormalPilotValidationError(f"cannot inspect {name}: {absolute}: {error}") from error
    if stat.S_ISLNK(link_metadata.st_mode) or not stat.S_ISREG(link_metadata.st_mode):
        raise FormalPilotValidationError(
            f"{name} must be a regular non-symlink file: {absolute}"
        )
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise FormalPilotValidationError(f"cannot resolve {name}: {absolute}") from error
    if resolved != absolute:
        raise FormalPilotValidationError(
            f"{name} may not resolve through a symlink or alias: {absolute}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise FormalPilotValidationError(f"cannot open {name}: {absolute}: {error}") from error
    chunks: list[bytes] | None = [] if capture_payload else None
    try:
        before = os.fstat(descriptor)
        if (
            (link_metadata.st_dev, link_metadata.st_ino)
            != (before.st_dev, before.st_ino)
            or not stat.S_ISREG(before.st_mode)
        ):
            raise FormalPilotValidationError(
                f"{name} changed between lstat and open: {absolute}"
            )
        digest = hashlib.sha256()
        bytes_read = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            bytes_read += len(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise FormalPilotValidationError(f"cannot hash {name}: {absolute}: {error}") from error
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or bytes_read != before.st_size:
        raise FormalPilotValidationError(f"{name} changed while being hashed: {absolute}")
    snapshot = _FileSnapshot(
        path=absolute,
        device=before.st_dev,
        inode=before.st_ino,
        mode=stat.S_IMODE(before.st_mode),
        link_count=before.st_nlink,
        size_bytes=before.st_size,
        mtime_ns=before.st_mtime_ns,
        sha256=digest.hexdigest(),
    )
    return snapshot, (b"".join(chunks) if chunks is not None else None)


def _require_read_only(snapshot: _FileSnapshot, *, name: str) -> None:
    if snapshot.mode & 0o222:
        raise FormalPilotValidationError(
            f"{name} must remain read-only: {snapshot.path}"
        )


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _snapshot_input_tree(root: Path) -> dict[str, _TreeEntrySnapshot]:
    """Capture every directory and file below the immutable formal-input root."""

    canonical_root = _canonical_directory(
        root, name="formal input tree root", read_only=True
    )
    snapshots: dict[str, _TreeEntrySnapshot] = {}
    inode_owners: dict[tuple[int, int], str] = {}

    def register(snapshot: _TreeEntrySnapshot) -> None:
        if snapshot.relative_path in snapshots:
            raise FormalPilotValidationError(
                f"formal input tree repeats path {snapshot.relative_path!r}"
            )
        identity = (snapshot.device, snapshot.inode)
        previous = inode_owners.get(identity)
        if previous is not None:
            raise FormalPilotValidationError(
                "formal input tree paths alias one inode: "
                f"{previous!r} and {snapshot.relative_path!r}"
            )
        inode_owners[identity] = snapshot.relative_path
        snapshots[snapshot.relative_path] = snapshot

    def visit(path: Path, relative: str) -> None:
        try:
            before = os.lstat(path)
        except OSError as error:
            raise FormalPilotValidationError(
                f"cannot inspect formal input tree entry {path}: {error}"
            ) from error
        if stat.S_ISLNK(before.st_mode):
            raise FormalPilotValidationError(
                f"formal input tree may not contain symlinks: {path}"
            )
        absolute = Path(os.path.abspath(path))
        try:
            resolved = absolute.resolve(strict=True)
        except OSError as error:
            raise FormalPilotValidationError(
                f"cannot resolve formal input tree entry {absolute}: {error}"
            ) from error
        if resolved != absolute:
            raise FormalPilotValidationError(
                f"formal input tree entry resolves through an alias: {absolute}"
            )
        mode = stat.S_IMODE(before.st_mode)
        if stat.S_ISREG(before.st_mode):
            file_snapshot, _ = _read_snapshot(
                absolute, name=f"formal input tree file {relative}"
            )
            if file_snapshot.mode != 0o444:
                raise FormalPilotValidationError(
                    f"formal input tree file must have exact mode 0444: {absolute}"
                )
            register(
                _TreeEntrySnapshot(
                    relative_path=relative,
                    kind="file",
                    device=file_snapshot.device,
                    inode=file_snapshot.inode,
                    mode=file_snapshot.mode,
                    link_count=file_snapshot.link_count,
                    size_bytes=file_snapshot.size_bytes,
                    mtime_ns=file_snapshot.mtime_ns,
                    sha256=file_snapshot.sha256,
                )
            )
            return
        if not stat.S_ISDIR(before.st_mode):
            raise FormalPilotValidationError(
                f"formal input tree contains a non-file/non-directory entry: {absolute}"
            )
        if mode != 0o555:
            raise FormalPilotValidationError(
                f"formal input tree directory must have exact mode 0555: {absolute}"
            )
        try:
            with os.scandir(absolute) as iterator:
                child_names = sorted(entry.name for entry in iterator)
        except OSError as error:
            raise FormalPilotValidationError(
                f"cannot enumerate formal input directory {absolute}: {error}"
            ) from error
        for child_name in child_names:
            child_relative = (
                child_name if relative == "." else f"{relative}/{child_name}"
            )
            visit(absolute / child_name, child_relative)
        try:
            after = os.lstat(absolute)
        except OSError as error:
            raise FormalPilotValidationError(
                f"formal input directory disappeared while snapshotted: {absolute}"
            ) from error
        if _directory_identity(before) != _directory_identity(after):
            raise FormalPilotValidationError(
                f"formal input directory changed while snapshotted: {absolute}"
            )
        register(
            _TreeEntrySnapshot(
                relative_path=relative,
                kind="directory",
                device=before.st_dev,
                inode=before.st_ino,
                mode=mode,
                link_count=before.st_nlink,
                size_bytes=before.st_size,
                mtime_ns=before.st_mtime_ns,
                sha256=None,
            )
        )

    visit(canonical_root, ".")
    return snapshots


def _expected_input_tree_layout() -> dict[str, str]:
    expected: dict[str, str] = {
        ".": "directory",
        INPUT_REGISTRY_FILENAME: "file",
        SPLIT_REGISTRY_FILENAME: "file",
        HOLDOUT_COMMITMENT_FILENAME: "file",
        "development": "directory",
        "holdout": "directory",
    }
    for spec in FROZEN_RUN_SPECS:
        run_relative = f"{spec.dataset_split}/{spec.run_id}"
        expected[run_relative] = "directory"
        expected[f"{run_relative}/{SOURCE_MANIFEST_FILENAME}"] = "file"
        expected[f"{run_relative}/{INPUT_MANIFEST_FILENAME}"] = "file"
    return expected


def _validate_input_tree_layout(
    snapshots: Mapping[str, _TreeEntrySnapshot],
) -> None:
    actual = {relative: snapshot.kind for relative, snapshot in snapshots.items()}
    expected = _expected_input_tree_layout()
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        mismatched = sorted(
            relative
            for relative in set(actual) & set(expected)
            if actual[relative] != expected[relative]
        )
        raise FormalPilotValidationError(
            "formal input tree layout is not the exact frozen layout; "
            f"missing={missing}, extra={extra}, kind_mismatch={mismatched}"
        )


def _input_tree_digest(snapshots: Mapping[str, _TreeEntrySnapshot]) -> str:
    records = [
        snapshots[relative].canonical_record() for relative in sorted(snapshots)
    ]
    return hashlib.sha256(_json_bytes(records)).hexdigest()


def _assert_input_tree_equal(
    expected: Mapping[str, _TreeEntrySnapshot],
    actual: Mapping[str, _TreeEntrySnapshot],
    *,
    name: str,
) -> None:
    if dict(expected) != dict(actual):
        changed = sorted(
            relative
            for relative in set(expected) | set(actual)
            if expected.get(relative) != actual.get(relative)
        )
        raise FormalPilotValidationError(
            f"{name} changed during CPU validation: " + ", ".join(changed)
        )


def _resolve_declared_path(value: Any, *, base: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise FormalPilotValidationError(f"{name} must be a non-empty path string")
    raw = Path(value)
    return Path(os.path.abspath(raw if raw.is_absolute() else base / raw))


def _validate_artifact(
    value: Any,
    *,
    expected_path: str,
    snapshot: _FileSnapshot,
    name: str,
) -> None:
    expected = {
        "path": expected_path,
        "sha256": snapshot.sha256,
        "size_bytes": snapshot.size_bytes,
    }
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise FormalPilotValidationError(f"{name} artifact does not match exact bytes/path")


def _validate_formal_declaration(
    snapshot: _FileSnapshot,
    declaration: Mapping[str, Any],
    *,
    prefix: str,
    name: str,
) -> None:
    expected = {
        "sha256": snapshot.sha256,
        "size_bytes": snapshot.size_bytes,
        "mode_octal": f"{snapshot.mode:04o}",
        "link_count": snapshot.link_count,
        "device": snapshot.device,
        "inode": snapshot.inode,
        "mtime_ns": snapshot.mtime_ns,
    }
    actual = {key: declaration.get(f"{prefix}{key}") for key in expected}
    if actual != expected:
        raise FormalPilotValidationError(
            f"{name} declaration differs from canonical path/stat/SHA snapshot"
        )
    _require_read_only(snapshot, name=name)


def _validate_corruption_payload(
    payload: Mapping[str, Any], spec: _FrozenRunSpec
) -> None:
    if (
        payload.get("schema_version") != INPUT_MANIFEST_SCHEMA_VERSION
        or payload.get("frame_count") != FRAME_COUNT
        or payload.get("source_frame_count") != len(spec.pool_indices)
        or payload.get("seed") != 0
    ):
        raise FormalPilotValidationError(
            f"{spec.run_id} input manifest does not retain frozen counts/seed/schema"
        )
    corruptions = payload.get("corruptions")
    if not isinstance(corruptions, list) or len(corruptions) != 1:
        raise FormalPilotValidationError(f"{spec.run_id} must contain one corruption")
    corruption = corruptions[0]
    if (
        not isinstance(corruption, Mapping)
        or set(corruption)
        != {
            "type",
            "start",
            "end",
            "start_frame",
            "end_frame",
            "expected_effect",
            "parameters",
        }
        or corruption.get("type") != spec.corruption_type
    ):
        raise FormalPilotValidationError(f"{spec.run_id} corruption type mismatch")
    expected_end = 18 if spec.corruption_type == "wrong_order_segment" else 19
    if (
        corruption.get("start") != EVENT_START
        or corruption.get("start_frame") != EVENT_START
        or corruption.get("end") != expected_end
        or corruption.get("end_frame") != expected_end
        or corruption.get("expected_effect")
        != _EXPECTED_CORRUPTION_EFFECTS[spec.corruption_type]
    ):
        raise FormalPilotValidationError(f"{spec.run_id} corruption interval mismatch")
    parameters = corruption.get("parameters")
    if not isinstance(parameters, Mapping):
        raise FormalPilotValidationError(f"{spec.run_id} corruption parameters are missing")
    if spec.corruption_type == "dynamic_occlusion":
        expected_rectangles = [
            {"x": x, "y": y, "width": 0.5, "height": 0.5}
            for x, y in (
                (0.25, 0.25),
                (0.29, 0.275),
                (0.33, 0.3),
                (0.37, 0.325),
                (0.41, 0.35),
            )
        ]
        expected_parameters = {
            "coordinate_space": "normalized",
            "coordinate_reference": _DYNAMIC_COORDINATE_REFERENCE,
            "initial_rectangle": _DYNAMIC_RECTANGLE,
            "velocity": _DYNAMIC_VELOCITY,
            "fill": _DYNAMIC_FILL,
            "frame_rectangles": expected_rectangles,
            "rectangle_generated_from_seed": False,
            "velocity_generated_from_seed": False,
        }
        if dict(parameters) != expected_parameters:
            raise FormalPilotValidationError(
                f"{spec.run_id} dynamic parameters are not the exact frozen v1 object"
            )
    elif spec.corruption_type == "wrong_order_segment":
        if dict(parameters) != {
            "mode": "reverse",
            "permutation": [18, 17, 16, 15],
            "relative_permutation": [3, 2, 1, 0],
        }:
            raise FormalPilotValidationError(
                f"{spec.run_id} wrong-order permutation mismatch"
            )
    elif dict(parameters) != {
        "source_start": 30,
        "source_end": 34,
        "source_indices": [30, 31, 32, 33, 34],
        "selection": "explicit",
    }:
        raise FormalPilotValidationError(f"{spec.run_id} donor range mismatch")


def _validate_run_static(run: _PreparedRun) -> None:
    spec = run.spec
    record = run.record
    expected_position = [
        candidate for candidate in FROZEN_RUN_SPECS if candidate.dataset_split == spec.dataset_split
    ].index(spec)
    expected_values = {
        "run_id": spec.run_id,
        "dataset_split": spec.dataset_split,
        "corruption_type": spec.corruption_type,
        "execution_position_within_split": expected_position,
        "seed": 0,
        "base_rgb_source_indices": list(spec.base_indices),
        "donor_rgb_source_indices": list(spec.donor_indices),
        "source_pool_rgb_indices": list(spec.pool_indices),
        "source_pool_frame_count": len(spec.pool_indices),
        "output_frame_count": FRAME_COUNT,
        "consumed_source_pool_indices_by_output_position": list(
            spec.consumed_pool_positions
        ),
    }
    for key, expected in expected_values.items():
        if record.get(key) != expected:
            raise FormalPilotValidationError(
                f"{spec.run_id} registry field {key!r} is not frozen v1"
            )
    source = run.source_payload
    if (
        source.get("schema_version") != SOURCE_SCHEMA_VERSION
        or source.get("run_id") != spec.run_id
        or source.get("dataset_split") != spec.dataset_split
        or source.get("corruption_type") != spec.corruption_type
        or source.get("frame_count") != len(spec.pool_indices)
        or source.get("output_frame_count") != FRAME_COUNT
    ):
        raise FormalPilotValidationError(f"{spec.run_id} source manifest layout mismatch")
    source_frames = source.get("frames")
    if not isinstance(source_frames, list) or len(source_frames) != len(spec.pool_indices):
        raise FormalPilotValidationError(f"{spec.run_id} source-pool length mismatch")
    for position, (frame, raw_index) in enumerate(
        zip(source_frames, spec.pool_indices, strict=True)
    ):
        if not isinstance(frame, Mapping):
            raise FormalPilotValidationError(
                f"{spec.run_id} source frame {position} must be an object"
            )
        expected_role = "base" if position < FRAME_COUNT else "low_overlap_donor"
        if (
            frame.get("source_pool_index") != position
            or frame.get("source_entry_index") != raw_index
            or frame.get("raw_rgb_source_index") != raw_index
            or frame.get("source_pool_role") != expected_role
        ):
            raise FormalPilotValidationError(
                f"{spec.run_id} source frame {position} lineage mismatch"
            )
    _validate_corruption_payload(run.input_payload, spec)
    input_frames = run.input_payload.get("frames")
    if not isinstance(input_frames, list) or len(input_frames) != FRAME_COUNT:
        raise FormalPilotValidationError(f"{spec.run_id} input frames mismatch")
    actual_positions = []
    dynamic_parameters: Mapping[str, Any] | None = None
    if spec.corruption_type == "dynamic_occlusion":
        corruption = run.input_payload["corruptions"][0]
        assert isinstance(corruption, Mapping)
        parameters = corruption["parameters"]
        assert isinstance(parameters, Mapping)
        dynamic_parameters = parameters
    for position, frame in enumerate(input_frames):
        if not isinstance(frame, Mapping) or frame.get("frame_index") != position:
            raise FormalPilotValidationError(
                f"{spec.run_id} input frame {position} is malformed"
            )
        actual_positions.append(frame.get("source_index"))
        transforms = frame.get("transforms")
        if not isinstance(transforms, list):
            raise FormalPilotValidationError(
                f"{spec.run_id} input frame {position} transforms are malformed"
            )
        if dynamic_parameters is not None and EVENT_START <= position <= EVENT_START + 4:
            rectangle = dynamic_parameters["frame_rectangles"][position - EVENT_START]
            expected_transform = {
                "type": "rectangle_occlusion",
                "coordinate_space": dynamic_parameters["coordinate_space"],
                "coordinate_reference": dynamic_parameters["coordinate_reference"],
                "rectangle": rectangle,
                "fill": dynamic_parameters["fill"],
            }
            if transforms != [expected_transform]:
                raise FormalPilotValidationError(
                    f"{spec.run_id} dynamic transform is not bound to frozen "
                    f"rectangle/fill at output {position}"
                )
    if actual_positions != list(spec.consumed_pool_positions):
        raise FormalPilotValidationError(f"{spec.run_id} final source order mismatch")
    associations = {key: [] for key in ("rgb_source_entry_indices", "depth_source_entry_indices", "groundtruth_source_entry_indices")}
    consumed_rgb_sha256s: list[str] = []
    for pool_position in spec.consumed_pool_positions:
        source_frame = source_frames[pool_position]
        depth = source_frame.get("depth")
        groundtruth = source_frame.get("groundtruth")
        if not isinstance(depth, Mapping) or not isinstance(groundtruth, Mapping):
            raise FormalPilotValidationError(f"{spec.run_id} lacks depth/GT lineage")
        associations["rgb_source_entry_indices"].append(source_frame.get("source_entry_index"))
        associations["depth_source_entry_indices"].append(depth.get("source_entry_index"))
        associations["groundtruth_source_entry_indices"].append(
            groundtruth.get("source_entry_index")
        )
        consumed_rgb_sha256s.append(
            _sha256(source_frame.get("rgb_sha256"), name=f"{spec.run_id} RGB SHA")
        )
    if record.get("consumed_associations") != associations:
        raise FormalPilotValidationError(
            f"{spec.run_id} consumed RGB/depth/GT associations mismatch"
        )
    if record.get("raw_frame_sha256s") != sorted(consumed_rgb_sha256s):
        raise FormalPilotValidationError(f"{spec.run_id} consumed RGB SHA set mismatch")


def _read_input_bundle(
    input_root: str | os.PathLike[str],
) -> tuple[Path, _FileSnapshot, tuple[_PreparedRun, ...], dict[Path, _FileSnapshot]]:
    root = _canonical_directory(Path(input_root), name="formal input root", read_only=True)
    registry_path = root / INPUT_REGISTRY_FILENAME
    registry_snapshot, registry_bytes = _read_snapshot(
        registry_path, name="formal input registry", capture_payload=True
    )
    assert registry_bytes is not None
    _require_read_only(registry_snapshot, name="formal input registry")
    registry = _strict_json_object(registry_bytes, name=INPUT_REGISTRY_FILENAME)
    if (
        registry.get("schema_version") != INPUT_SCHEMA_VERSION
        or registry.get("seed") != 0
        or registry.get("frame_count") != FRAME_COUNT
        or registry.get("event_start") != EVENT_START
        or registry.get("formal_status") != "pre_forward_inputs_only"
        or registry.get("materialization") != "manifests_only_no_copy_no_symlink"
    ):
        raise FormalPilotValidationError("formal input registry frozen identity mismatch")
    expected_order = {
        split: [spec.run_id for spec in FROZEN_RUN_SPECS if spec.dataset_split == split]
        for split in ("development", "holdout")
    }
    if registry.get("fixed_execution_order") != expected_order:
        raise FormalPilotValidationError("formal input execution order mismatch")
    records = registry.get("runs")
    if not isinstance(records, list) or len(records) != len(FROZEN_RUN_SPECS):
        raise FormalPilotValidationError("formal input registry must contain six runs")
    if [record.get("run_id") if isinstance(record, Mapping) else None for record in records] != [
        spec.run_id for spec in FROZEN_RUN_SPECS
    ]:
        raise FormalPilotValidationError("formal input run layout/order mismatch")

    controls: dict[Path, _FileSnapshot] = {registry_path: registry_snapshot}
    prepared: list[_PreparedRun] = []
    for spec, record in zip(FROZEN_RUN_SPECS, records, strict=True):
        if not isinstance(record, Mapping) or _SAFE_COMPONENT_RE.fullmatch(spec.run_id) is None:
            raise FormalPilotValidationError(f"invalid registry record for {spec.run_id}")
        run_root = _canonical_directory(
            root / spec.dataset_split / spec.run_id,
            name=f"{spec.run_id} input run",
            read_only=True,
        )
        source_path = run_root / SOURCE_MANIFEST_FILENAME
        input_path = run_root / INPUT_MANIFEST_FILENAME
        source_snapshot, source_bytes = _read_snapshot(
            source_path, name=f"{spec.run_id} source manifest", capture_payload=True
        )
        input_snapshot, input_bytes = _read_snapshot(
            input_path, name=f"{spec.run_id} input manifest", capture_payload=True
        )
        assert source_bytes is not None and input_bytes is not None
        _require_read_only(source_snapshot, name=f"{spec.run_id} source manifest")
        _require_read_only(input_snapshot, name=f"{spec.run_id} input manifest")
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {
            "source_manifest",
            "input_manifest",
            "corruption_json",
        }:
            raise FormalPilotValidationError(f"{spec.run_id} artifacts are invalid")
        source_relative = f"{spec.dataset_split}/{spec.run_id}/{SOURCE_MANIFEST_FILENAME}"
        input_relative = f"{spec.dataset_split}/{spec.run_id}/{INPUT_MANIFEST_FILENAME}"
        _validate_artifact(
            artifacts["source_manifest"],
            expected_path=source_relative,
            snapshot=source_snapshot,
            name=f"{spec.run_id} source manifest",
        )
        _validate_artifact(
            artifacts["input_manifest"],
            expected_path=input_relative,
            snapshot=input_snapshot,
            name=f"{spec.run_id} input manifest",
        )
        if artifacts["corruption_json"] != artifacts["input_manifest"]:
            raise FormalPilotValidationError(
                f"{spec.run_id} corruption/input artifacts must be identical"
            )
        run = _PreparedRun(
            spec=spec,
            record=record,
            run_root=run_root,
            source_path=source_path,
            input_path=input_path,
            source_snapshot=source_snapshot,
            input_snapshot=input_snapshot,
            source_payload=_strict_json_object(
                source_bytes, name=f"{spec.run_id} source manifest"
            ),
            input_payload=_strict_json_object(
                input_bytes, name=f"{spec.run_id} input manifest"
            ),
        )
        _validate_run_static(run)
        prepared.append(run)
        controls[source_path] = source_snapshot
        controls[input_path] = input_snapshot

    commitment_artifacts = registry.get("formal_commitment_artifacts")
    if not isinstance(commitment_artifacts, Mapping) or set(commitment_artifacts) != {
        "split_registry",
        "holdout_commitment",
    }:
        raise FormalPilotValidationError("formal commitment artifacts are invalid")
    for key, filename in (
        ("split_registry", SPLIT_REGISTRY_FILENAME),
        ("holdout_commitment", HOLDOUT_COMMITMENT_FILENAME),
    ):
        path = root / filename
        snapshot, payload = _read_snapshot(path, name=key, capture_payload=True)
        assert payload is not None
        _require_read_only(snapshot, name=key)
        _strict_json_object(payload, name=filename)
        _validate_artifact(
            commitment_artifacts[key], expected_path=filename, snapshot=snapshot, name=key
        )
        controls[path] = snapshot
    return root, registry_snapshot, tuple(prepared), controls


def _declared_sources(runs: Sequence[_PreparedRun]) -> tuple[_DeclaredSource, ...]:
    declared: list[_DeclaredSource] = []
    rgb_paths: set[Path] = set()
    depth_paths: set[Path] = set()
    index_paths: dict[str, set[Path]] = {kind: set() for kind in ("rgb", "depth", "groundtruth")}
    archive_paths: set[Path] = set()
    raw_manifest_paths: set[Path] = set()
    for run in runs:
        source = run.source_payload
        frames = source.get("frames")
        assert isinstance(frames, list)
        for position, frame in enumerate(frames):
            assert isinstance(frame, Mapping)
            rgb_path = _resolve_declared_path(
                frame.get("path"), base=run.run_root, name=f"{run.spec.run_id} RGB {position}"
            )
            depth = frame.get("depth")
            if not isinstance(depth, Mapping):
                raise FormalPilotValidationError(
                    f"{run.spec.run_id} source frame {position} lacks depth"
                )
            depth_path = _resolve_declared_path(
                depth.get("path"), base=run.run_root, name=f"{run.spec.run_id} depth {position}"
            )
            rgb_paths.add(rgb_path)
            depth_paths.add(depth_path)
            declared.append(
                _DeclaredSource(rgb_path, frame, "rgb_", f"{run.spec.run_id} RGB {position}", "rgb")
            )
            declared.append(
                _DeclaredSource(depth_path, depth, "", f"{run.spec.run_id} depth {position}", "depth")
            )
        index_files = source.get("tum_index_files")
        if not isinstance(index_files, Mapping) or set(index_files) != {
            "rgb",
            "depth",
            "groundtruth",
        }:
            raise FormalPilotValidationError(f"{run.spec.run_id} TUM index declarations invalid")
        for kind in ("rgb", "depth", "groundtruth"):
            declaration = index_files[kind]
            if not isinstance(declaration, Mapping):
                raise FormalPilotValidationError(f"{run.spec.run_id} {kind} index invalid")
            path = _resolve_declared_path(
                declaration.get("path"), base=run.run_root, name=f"{kind} index"
            )
            index_paths[kind].add(path)
            declared.append(_DeclaredSource(path, declaration, "", f"TUM {kind} index", "index"))
        lineage = source.get("official_lineage")
        if not isinstance(lineage, Mapping):
            raise FormalPilotValidationError(f"{run.spec.run_id} official lineage invalid")
        for key, kind, path_set in (
            ("archive", "archive", archive_paths),
            ("raw_manifest", "raw_manifest", raw_manifest_paths),
        ):
            declaration = lineage.get(key)
            if not isinstance(declaration, Mapping):
                raise FormalPilotValidationError(f"{run.spec.run_id} lineage {key} invalid")
            path = _resolve_declared_path(
                declaration.get("path"), base=run.run_root, name=f"official {key}"
            )
            path_set.add(path)
            declared.append(_DeclaredSource(path, declaration, "", f"official {key}", kind))
    if len(rgb_paths) != 190 or len(depth_paths) != 190:
        raise FormalPilotValidationError(
            "frozen source pools must cover 190 unique RGB and 190 unique depth files"
        )
    if any(len(paths) != 1 for paths in index_paths.values()):
        raise FormalPilotValidationError("all six runs must bind the same three TUM indexes")
    if len(archive_paths) != 1 or len(raw_manifest_paths) != 1:
        raise FormalPilotValidationError("all six runs must bind one TGZ and one raw manifest")
    unique_paths = rgb_paths | depth_paths | set().union(*index_paths.values()) | archive_paths | raw_manifest_paths
    if len(unique_paths) != EXPECTED_SOURCE_ARTIFACT_COUNT:
        raise FormalPilotValidationError(
            f"raw snapshot set must contain exactly {EXPECTED_SOURCE_ARTIFACT_COUNT} unique artifacts"
        )
    return tuple(declared)


def _snapshot_declared_sources(
    declarations: Sequence[_DeclaredSource],
) -> dict[Path, _FileSnapshot]:
    paths = sorted({declaration.path for declaration in declarations}, key=str)
    snapshots: dict[Path, _FileSnapshot] = {}
    seen_inodes: dict[tuple[int, int], Path] = {}
    for path in paths:
        snapshot, _ = _read_snapshot(path, name="formal raw source")
        identity = (snapshot.device, snapshot.inode)
        if identity in seen_inodes and seen_inodes[identity] != path:
            raise FormalPilotValidationError(
                f"raw snapshot paths alias one inode: {seen_inodes[identity]} and {path}"
            )
        seen_inodes[identity] = path
        snapshots[path] = snapshot
    for declaration in declarations:
        snapshot = snapshots[declaration.path]
        _validate_formal_declaration(
            snapshot,
            declaration.declaration,
            prefix=declaration.prefix,
            name=declaration.role,
        )
    return snapshots


def _parse_decimal_timestamp(text: str, *, name: str) -> tuple[Decimal, float]:
    try:
        decimal_value = Decimal(text)
    except InvalidOperation as error:
        raise FormalPilotValidationError(f"{name} is not a decimal timestamp") from error
    if not decimal_value.is_finite():
        raise FormalPilotValidationError(f"{name} must be finite")
    try:
        float_value = float(decimal_value)
    except (OverflowError, ValueError) as error:
        raise FormalPilotValidationError(
            f"{name} is outside the finite JSON number range"
        ) from error
    if not math.isfinite(float_value):
        raise FormalPilotValidationError(
            f"{name} is outside the finite JSON number range"
        )
    return decimal_value, float_value


def _parse_tum_index_independently(
    snapshot: _FileSnapshot,
    payload: bytes,
    *,
    kind: str,
) -> tuple[_TumEntry, ...]:
    """Parse a TUM index without importing or trusting the input preparer."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise FormalPilotValidationError(
            f"TUM {kind} index is not UTF-8: {snapshot.path}"
        ) from error
    raw_lines = payload.splitlines(keepends=True)
    text_lines = text.splitlines(keepends=True)
    if len(raw_lines) != len(text_lines):
        raise FormalPilotValidationError(
            f"cannot preserve physical lines in TUM {kind} index"
        )
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
            raise FormalPilotValidationError(
                f"{snapshot.path}:{source_line} must contain exactly "
                f"{expected_tokens} fields for {kind}"
            )
        timestamp_decimal, timestamp = _parse_decimal_timestamp(
            tokens[0], name=f"{snapshot.path}:{source_line} timestamp"
        )
        if kind == "groundtruth":
            values_text = tuple(tokens[1:])
            try:
                values = tuple(float(value) for value in values_text)
            except ValueError as error:
                raise FormalPilotValidationError(
                    f"{snapshot.path}:{source_line} contains a non-numeric pose"
                ) from error
            if any(not math.isfinite(value) for value in values):
                raise FormalPilotValidationError(
                    f"{snapshot.path}:{source_line} contains a non-finite pose"
                )
            path_text = None
        else:
            path_text = tokens[1]
            if (
                not path_text
                or "\x00" in path_text
                or Path(path_text).is_absolute()
            ):
                raise FormalPilotValidationError(
                    f"{snapshot.path}:{source_line} must use a relative dataset path"
                )
            values_text = ()
            values = ()
        entries.append(
            _TumEntry(
                source_entry_index=len(entries),
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
        raise FormalPilotValidationError(f"TUM {kind} index contains no entries")
    return tuple(entries)


def _nearest_tum_entry(
    rgb: _TumEntry,
    candidates: Sequence[_TumEntry],
    *,
    kind: str,
) -> _TumEntry:
    distances = tuple(
        (abs(candidate.timestamp_decimal - rgb.timestamp_decimal), candidate)
        for candidate in candidates
    )
    minimum = min(distance for distance, _ in distances)
    nearest = tuple(candidate for distance, candidate in distances if distance == minimum)
    if len(nearest) != 1:
        raise FormalPilotValidationError(
            f"RGB entry {rgb.source_entry_index} has a non-unique nearest {kind} association"
        )
    if minimum > MAX_ASSOCIATION_DELTA:
        raise FormalPilotValidationError(
            f"RGB entry {rgb.source_entry_index} nearest {kind} delta {minimum} "
            f"exceeds {MAX_ASSOCIATION_DELTA} seconds"
        )
    return nearest[0]


def _resolve_tum_entry_path(
    dataset_root: Path,
    entry: _TumEntry,
    *,
    kind: str,
) -> Path:
    if entry.path_text is None:
        raise FormalPilotValidationError(f"TUM {kind} entry lacks a path")
    unresolved = dataset_root / entry.path_text
    absolute = Path(os.path.abspath(unresolved))
    try:
        metadata = os.lstat(absolute)
        resolved = absolute.resolve(strict=True)
        resolved.relative_to(dataset_root)
    except (OSError, ValueError) as error:
        raise FormalPilotValidationError(
            f"TUM {kind} entry {entry.source_entry_index} is missing or escapes the dataset"
        ) from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or resolved != absolute
    ):
        raise FormalPilotValidationError(
            f"TUM {kind} entry {entry.source_entry_index} is not one canonical regular file"
        )
    expected_parent = dataset_root / kind
    try:
        resolved.relative_to(expected_parent)
    except ValueError as error:
        raise FormalPilotValidationError(
            f"TUM {kind} entry {entry.source_entry_index} is outside {expected_parent}"
        ) from error
    return resolved


def _line_provenance(entry: _TumEntry) -> dict[str, Any]:
    return {
        "source_entry_index": entry.source_entry_index,
        "source_line": entry.source_line,
        "physical_line_sha256": entry.physical_line_sha256,
        "physical_line_size_bytes": entry.physical_line_size_bytes,
        "timestamp": entry.timestamp,
        "timestamp_text": entry.timestamp_text,
    }


def _require_declared_line(
    declaration: Mapping[str, Any],
    entry: _TumEntry,
    *,
    name: str,
) -> None:
    expected = _line_provenance(entry)
    actual = {key: declaration.get(key) for key in expected}
    if actual != expected:
        raise FormalPilotValidationError(
            f"{name} does not match the independently parsed TUM physical line"
        )


def _validate_tum_associations(
    runs: Sequence[_PreparedRun],
    source_snapshots: Mapping[Path, _FileSnapshot],
) -> dict[str, Any]:
    expected_root = _canonical_directory(
        EXPECTED_TUM_ROOT, name="frozen TUM dataset root", read_only=True
    )
    for subdirectory in ("rgb", "depth"):
        _canonical_directory(
            expected_root / subdirectory,
            name=f"frozen TUM {subdirectory} root",
            read_only=True,
        )

    expected_policy = {
        "method": "unique_nearest_absolute_timestamp",
        "tie_policy": "reject",
        "reuse_policy": "reject_within_source_pool_and_across_consumed_runs",
        "max_absolute_delta_seconds": float(MAX_ASSOCIATION_DELTA),
    }
    index_paths: dict[str, Path] = {}
    for run in runs:
        lineage = run.source_payload.get("official_lineage")
        if not isinstance(lineage, Mapping) or lineage.get("dataset_root") != str(
            expected_root
        ):
            raise FormalPilotValidationError(
                f"{run.spec.run_id} does not bind the frozen canonical TUM root"
            )
        if run.source_payload.get("association_policy") != expected_policy:
            raise FormalPilotValidationError(
                f"{run.spec.run_id} TUM association policy is not frozen v1"
            )
        declarations = run.source_payload.get("tum_index_files")
        if not isinstance(declarations, Mapping):
            raise FormalPilotValidationError(
                f"{run.spec.run_id} lacks TUM index declarations"
            )
        for kind in ("rgb", "depth", "groundtruth"):
            declaration = declarations.get(kind)
            if not isinstance(declaration, Mapping):
                raise FormalPilotValidationError(
                    f"{run.spec.run_id} {kind} index declaration is malformed"
                )
            path = _resolve_declared_path(
                declaration.get("path"),
                base=run.run_root,
                name=f"{run.spec.run_id} {kind} index",
            )
            expected_path = expected_root / f"{kind}.txt"
            if path != expected_path:
                raise FormalPilotValidationError(
                    f"{run.spec.run_id} {kind} index is not the canonical TUM index"
                )
            previous = index_paths.setdefault(kind, path)
            if previous != path:
                raise FormalPilotValidationError(
                    f"formal runs disagree on the {kind} index path"
                )

    entries: dict[str, tuple[_TumEntry, ...]] = {}
    for kind in ("rgb", "depth", "groundtruth"):
        path = index_paths[kind]
        snapshot, payload = _read_snapshot(
            path, name=f"independent TUM {kind} index", capture_payload=True
        )
        assert payload is not None
        if source_snapshots.get(path) != snapshot:
            raise FormalPilotValidationError(
                f"TUM {kind} index changed before independent association replay"
            )
        entries[kind] = _parse_tum_index_independently(snapshot, payload, kind=kind)

    association_records: list[dict[str, Any]] = []
    global_depth: dict[int, str] = {}
    global_groundtruth: dict[int, str] = {}
    for run in runs:
        index_declarations = run.source_payload["tum_index_files"]
        assert isinstance(index_declarations, Mapping)
        for kind in ("rgb", "depth", "groundtruth"):
            declaration = index_declarations[kind]
            assert isinstance(declaration, Mapping)
            if declaration.get("data_entry_count") != len(entries[kind]):
                raise FormalPilotValidationError(
                    f"{run.spec.run_id} {kind} index entry count is not independently reproducible"
                )
        frames = run.source_payload["frames"]
        assert isinstance(frames, list)
        for pool_position, raw_rgb_index in enumerate(run.spec.pool_indices):
            if raw_rgb_index >= len(entries["rgb"]):
                raise FormalPilotValidationError(
                    f"{run.spec.run_id} RGB allocation exceeds rgb.txt"
                )
            frame = frames[pool_position]
            assert isinstance(frame, Mapping)
            rgb = entries["rgb"][raw_rgb_index]
            depth = _nearest_tum_entry(rgb, entries["depth"], kind="depth")
            groundtruth = _nearest_tum_entry(
                rgb, entries["groundtruth"], kind="groundtruth"
            )
            _require_declared_line(
                frame, rgb, name=f"{run.spec.run_id} RGB allocation {pool_position}"
            )
            if (
                frame.get("raw_rgb_source_index") != raw_rgb_index
                or frame.get("raw_rgb_source_line") != rgb.source_line
            ):
                raise FormalPilotValidationError(
                    f"{run.spec.run_id} RGB allocation aliases disagree at {pool_position}"
                )
            rgb_path = _resolve_tum_entry_path(expected_root, rgb, kind="rgb")
            declared_rgb_path = _resolve_declared_path(
                frame.get("path"),
                base=run.run_root,
                name=f"{run.spec.run_id} RGB allocation {pool_position}",
            )
            if declared_rgb_path != rgb_path:
                raise FormalPilotValidationError(
                    f"{run.spec.run_id} RGB allocation path differs from rgb.txt"
                )

            depth_declaration = frame.get("depth")
            groundtruth_declaration = frame.get("groundtruth")
            if not isinstance(depth_declaration, Mapping) or not isinstance(
                groundtruth_declaration, Mapping
            ):
                raise FormalPilotValidationError(
                    f"{run.spec.run_id} allocation {pool_position} lacks depth/GT"
                )
            _require_declared_line(
                depth_declaration,
                depth,
                name=f"{run.spec.run_id} depth association {pool_position}",
            )
            _require_declared_line(
                groundtruth_declaration,
                groundtruth,
                name=f"{run.spec.run_id} GT association {pool_position}",
            )
            depth_path = _resolve_tum_entry_path(expected_root, depth, kind="depth")
            declared_depth_path = _resolve_declared_path(
                depth_declaration.get("path"),
                base=run.run_root,
                name=f"{run.spec.run_id} depth association {pool_position}",
            )
            if declared_depth_path != depth_path:
                raise FormalPilotValidationError(
                    f"{run.spec.run_id} depth association path differs from depth.txt"
                )
            depth_delta = depth.timestamp_decimal - rgb.timestamp_decimal
            groundtruth_delta = groundtruth.timestamp_decimal - rgb.timestamp_decimal
            expected_depth_values = {
                "delta_from_rgb_seconds": float(depth_delta),
                "absolute_delta_seconds": float(abs(depth_delta)),
            }
            if {
                key: depth_declaration.get(key) for key in expected_depth_values
            } != expected_depth_values:
                raise FormalPilotValidationError(
                    f"{run.spec.run_id} depth timestamp delta is not reproducible"
                )
            expected_groundtruth_values = {
                "translation_xyz": list(groundtruth.values[:3]),
                "quaternion_xyzw": list(groundtruth.values[3:]),
                "translation_xyz_text": list(groundtruth.values_text[:3]),
                "quaternion_xyzw_text": list(groundtruth.values_text[3:]),
                "delta_from_rgb_seconds": float(groundtruth_delta),
                "absolute_delta_seconds": float(abs(groundtruth_delta)),
            }
            if {
                key: groundtruth_declaration.get(key)
                for key in expected_groundtruth_values
            } != expected_groundtruth_values:
                raise FormalPilotValidationError(
                    f"{run.spec.run_id} GT pose/timestamp is not reproducible"
                )

            owner = f"{run.spec.run_id}:{pool_position}"
            previous_depth = global_depth.setdefault(depth.source_entry_index, owner)
            if previous_depth != owner:
                raise FormalPilotValidationError(
                    "formal allocations reuse one independently associated depth entry: "
                    f"{previous_depth} and {owner}"
                )
            previous_gt = global_groundtruth.setdefault(
                groundtruth.source_entry_index, owner
            )
            if previous_gt != owner:
                raise FormalPilotValidationError(
                    "formal allocations reuse one independently associated GT entry: "
                    f"{previous_gt} and {owner}"
                )
            association_records.append(
                {
                    "run_id": run.spec.run_id,
                    "source_pool_index": pool_position,
                    "rgb_source_entry_index": rgb.source_entry_index,
                    "depth_source_entry_index": depth.source_entry_index,
                    "groundtruth_source_entry_index": groundtruth.source_entry_index,
                }
            )

    if (
        len(association_records) != EXPECTED_ALLOCATED_FRAME_COUNT
        or len(global_depth) != EXPECTED_ALLOCATED_FRAME_COUNT
        or len(global_groundtruth) != EXPECTED_ALLOCATED_FRAME_COUNT
    ):
        raise FormalPilotValidationError(
            "formal allocations must independently prove 190 globally unique depth/GT associations"
        )
    return {
        "method": "unique_nearest_absolute_timestamp",
        "tie_policy": "reject",
        "max_absolute_delta_seconds": float(MAX_ASSOCIATION_DELTA),
        "allocated_rgb_count": EXPECTED_ALLOCATED_FRAME_COUNT,
        "globally_unique_depth_count": len(global_depth),
        "globally_unique_groundtruth_count": len(global_groundtruth),
        "association_digest_sha256": hashlib.sha256(
            _json_bytes(association_records)
        ).hexdigest(),
    }


def _snapshot_digest(snapshots: Mapping[Path, _FileSnapshot]) -> str:
    records = [snapshots[path].canonical_record() for path in sorted(snapshots, key=str)]
    return hashlib.sha256(_json_bytes(records)).hexdigest()


def _assert_snapshot_maps_equal(
    before: Mapping[Path, _FileSnapshot],
    after: Mapping[Path, _FileSnapshot],
    *,
    name: str,
) -> None:
    if dict(before) != dict(after):
        changed = sorted(str(path) for path in set(before) | set(after) if before.get(path) != after.get(path))
        raise FormalPilotValidationError(
            f"{name} changed during CPU replay: " + ", ".join(changed)
        )


def _snapshot_loader_sources(
    paths: Sequence[Path],
    *,
    expected: Mapping[Path, _FileSnapshot],
    name: str,
) -> dict[Path, _FileSnapshot]:
    snapshots: dict[Path, _FileSnapshot] = {}
    for position, path in enumerate(paths):
        if path in snapshots:
            raise FormalPilotValidationError(
                f"{name} repeats source path at loader position {position}: {path}"
            )
        snapshot, _ = _read_snapshot(
            path, name=f"{name} source frame {position}"
        )
        if expected.get(path) != snapshot:
            raise FormalPilotValidationError(
                f"{name} source frame {position} differs from the frozen raw snapshot"
            )
        snapshots[path] = snapshot
    if len(snapshots) != FRAME_COUNT:
        raise FormalPilotValidationError(
            f"{name} must bind exactly {FRAME_COUNT} unique loader sources"
        )
    return snapshots


def _capture_generator_provenance() -> dict[str, Any]:
    commit_before = _git_output(REPOSITORY_ROOT, "rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", commit_before) is None:
        raise FormalPilotValidationError("StateGuard3R HEAD is not a full commit ID")
    tracked_changes = _git_output(
        REPOSITORY_ROOT, "status", "--porcelain", "--untracked-files=no"
    )
    if tracked_changes:
        raise FormalPilotValidationError(
            "StateGuard3R has tracked changes; CPU report requires a clean commit"
        )
    tracked = _git_output(
        REPOSITORY_ROOT,
        "ls-files",
        "--error-unmatch",
        "--",
        GENERATOR_RELATIVE_PATH.as_posix(),
    )
    if tracked != GENERATOR_RELATIVE_PATH.as_posix():
        raise FormalPilotValidationError("CPU validation generator is not uniquely tracked")
    script_snapshot, _ = _read_snapshot(Path(__file__), name="CPU validation generator")
    committed_bytes = _git_bytes(
        REPOSITORY_ROOT,
        "show",
        f"{commit_before}:{GENERATOR_RELATIVE_PATH.as_posix()}",
    )
    committed_sha256 = hashlib.sha256(committed_bytes).hexdigest()
    if (
        committed_sha256 != script_snapshot.sha256
        or len(committed_bytes) != script_snapshot.size_bytes
    ):
        raise FormalPilotValidationError(
            "CPU validation generator bytes differ from the tracked HEAD blob"
        )
    commit_after = _git_output(REPOSITORY_ROOT, "rev-parse", "HEAD")
    changes_after = _git_output(
        REPOSITORY_ROOT, "status", "--porcelain", "--untracked-files=no"
    )
    if commit_after != commit_before or changes_after:
        raise FormalPilotValidationError("StateGuard3R provenance changed while captured")
    return {
        "repository_commit": commit_before,
        "tracked_worktree_clean": True,
        "git_blob_sha256": committed_sha256,
        "script": script_snapshot.artifact(GENERATOR_RELATIVE_PATH.as_posix()),
    }


def _validate_generator_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != {
        "repository_commit",
        "tracked_worktree_clean",
        "git_blob_sha256",
        "script",
    }:
        raise FormalPilotValidationError("generator provenance fields are invalid")
    if (
        not isinstance(value.get("repository_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", str(value["repository_commit"])) is None
        or value.get("tracked_worktree_clean") is not True
    ):
        raise FormalPilotValidationError("generator must bind a clean full commit")
    snapshot, _ = _read_snapshot(Path(__file__), name="CPU validation generator")
    expected = snapshot.artifact(GENERATOR_RELATIVE_PATH.as_posix())
    if value.get("script") != expected:
        raise FormalPilotValidationError("generator script SHA/size/path mismatch")
    if value.get("git_blob_sha256") != snapshot.sha256:
        raise FormalPilotValidationError(
            "generator tracked blob SHA-256 differs from executing script"
        )
    return dict(value)


def _validate_baseline(baseline_root: Path) -> Path:
    expected = _canonical_directory(RECAL3R_ROOT, name="frozen ReCal3R root")
    requested = _canonical_directory(baseline_root, name="requested ReCal3R root")
    if requested != expected:
        raise FormalPilotValidationError(f"baseline root must be exactly {expected}")
    if _git_output(requested, "rev-parse", "HEAD") != EXPECTED_RECAL3R_COMMIT:
        raise FormalPilotValidationError("ReCal3R commit differs from frozen revision")
    if _git_output(requested, "status", "--porcelain", "--untracked-files=no"):
        raise FormalPilotValidationError("ReCal3R has tracked changes")
    expected_prefix = (requested / ".venv").resolve(strict=True)
    actual_prefix = Path(sys.prefix).resolve(strict=True)
    if actual_prefix != expected_prefix:
        raise FormalPilotValidationError(
            f"CPU validator must use isolated ReCal3R interpreter {expected_prefix}; got {actual_prefix}"
        )
    expected_executable = requested / ".venv" / "bin" / "python"
    actual_executable = Path(os.path.abspath(sys.executable))
    if actual_executable != expected_executable:
        raise FormalPilotValidationError(
            "CPU validator must be invoked through the fixed ReCal3R Python entry "
            f"{expected_executable}; got {actual_executable}"
        )
    try:
        executable_metadata = os.lstat(actual_executable)
        executable_realpath = actual_executable.resolve(strict=True)
    except OSError as error:
        raise FormalPilotValidationError(
            f"cannot resolve ReCal3R Python executable: {actual_executable}"
        ) from error
    if not (
        stat.S_ISREG(executable_metadata.st_mode)
        or stat.S_ISLNK(executable_metadata.st_mode)
    ) or not executable_realpath.is_file():
        raise FormalPilotValidationError(
            "ReCal3R Python executable is not a real file or canonical symlink"
        )
    return requested


def _capture_production_runtime_provenance(
    baseline_root: Path,
    *,
    require_loader_imported: bool,
) -> dict[str, Any]:
    baseline = _validate_baseline(baseline_root)
    loader_path = baseline / OFFICIAL_LOADER_RELATIVE_PATH
    loader_snapshot, _ = _read_snapshot(
        loader_path, name="official ReCal3R CPU image loader"
    )
    if require_loader_imported:
        module = sys.modules.get("dust3r.utils.image")
        module_path = getattr(module, "__file__", None)
        if not isinstance(module_path, str):
            raise FormalPilotValidationError(
                "official ReCal3R CPU image loader was not imported"
            )
        try:
            imported = Path(module_path).resolve(strict=True)
        except OSError as error:
            raise FormalPilotValidationError(
                f"imported ReCal3R image-loader path is missing: {module_path}"
            ) from error
        if imported != loader_path:
            raise FormalPilotValidationError(
                "image loader did not originate from the fixed ReCal3R module: "
                f"{imported}"
            )
        imported_snapshot, _ = _read_snapshot(
            imported, name="imported official ReCal3R CPU image loader"
        )
        if imported_snapshot != loader_snapshot:
            raise FormalPilotValidationError(
                "official ReCal3R loader changed between path and module validation"
            )
    executable = Path(os.path.abspath(sys.executable))
    return {
        "recal3r_repository": {
            "path": str(baseline),
            "commit": EXPECTED_RECAL3R_COMMIT,
            "tracked_worktree_clean": True,
        },
        "python": {
            "executable": str(executable),
            "executable_realpath": str(executable.resolve(strict=True)),
            "prefix": str(Path(sys.prefix).resolve(strict=True)),
        },
        "official_loader": loader_snapshot.artifact(
            OFFICIAL_LOADER_RELATIVE_PATH.as_posix()
        ),
    }


def _load_production_runtime(baseline_root: Path) -> tuple[Any, Any]:
    baseline = _validate_baseline(baseline_root)
    runner_path = REPOSITORY_ROOT / RUNNER_RELATIVE_PATH
    tracked = _git_output(
        REPOSITORY_ROOT,
        "ls-files",
        "--error-unmatch",
        "--",
        RUNNER_RELATIVE_PATH.as_posix(),
    )
    if tracked != RUNNER_RELATIVE_PATH.as_posix():
        raise FormalPilotValidationError("ReCal3R smoke runner is not uniquely tracked")
    baseline_src = baseline / "src"
    if str(baseline_src) not in sys.path:
        sys.path.insert(0, str(baseline_src))
    try:
        import torch
    except ImportError as error:
        raise FormalPilotValidationError("isolated ReCal3R torch is unavailable") from error
    module_name = "_stateguard3r_tracked_recal3r_smoke_for_cpu_validation"
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    if spec is None or spec.loader is None:
        raise FormalPilotValidationError("cannot import tracked ReCal3R smoke runner")
    runner = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = runner
    try:
        spec.loader.exec_module(runner)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if not callable(getattr(runner, "_prepare_views", None)):
        raise FormalPilotValidationError("tracked smoke runner lacks _prepare_views")
    return torch, runner


def _cuda_initialized(torch_module: Any) -> bool:
    cuda = getattr(torch_module, "cuda", None)
    check = getattr(cuda, "is_initialized", None)
    if not callable(check):
        raise FormalPilotValidationError("torch.cuda.is_initialized is unavailable")
    value = check()
    if type(value) is not bool:
        raise FormalPilotValidationError("torch.cuda.is_initialized returned a non-boolean")
    return value


def _image_array(image: Any, *, name: str) -> np.ndarray:
    if isinstance(image, np.ndarray):
        array = image
    else:
        device = getattr(image, "device", None)
        if getattr(device, "type", None) != "cpu":
            raise FormalPilotValidationError(f"{name} must remain on CPU")
        detach = getattr(image, "detach", None)
        if not callable(detach):
            raise FormalPilotValidationError(f"{name} is not a tensor/NumPy array")
        try:
            array = detach().contiguous().numpy()
        except Exception as error:
            raise FormalPilotValidationError(f"cannot snapshot {name} tensor") from error
    if array.ndim != 4 or array.shape[0] < 1 or array.shape[1] != 3 or min(array.shape[2:]) < 1:
        raise FormalPilotValidationError(f"{name} must have Bx3xHxW shape")
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise FormalPilotValidationError(f"{name} must contain finite floating CPU pixels")
    return np.array(array, copy=True, order="C")


def _images_share_storage(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.shares_memory(left, right))
    left_storage = getattr(left, "untyped_storage", None)
    right_storage = getattr(right, "untyped_storage", None)
    if callable(left_storage) and callable(right_storage):
        try:
            return int(left_storage().data_ptr()) == int(right_storage().data_ptr())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    left_pointer = getattr(left, "data_ptr", None)
    right_pointer = getattr(right, "data_ptr", None)
    if callable(left_pointer) and callable(right_pointer):
        return int(left_pointer()) == int(right_pointer())
    return left is right


def _expected_replay(original: np.ndarray, transforms: Sequence[Mapping[str, Any]]) -> np.ndarray:
    expected = original.copy()
    for transform in transforms:
        if transform.get("type") != "rectangle_occlusion":
            continue
        rectangle = transform["rectangle"]
        _, _, height, width = expected.shape
        x_start = max(0, min(width, math.floor(float(rectangle["x"]) * width)))
        x_end = max(
            x_start + 1,
            min(width, math.ceil((float(rectangle["x"]) + float(rectangle["width"])) * width)),
        )
        y_start = max(0, min(height, math.floor(float(rectangle["y"]) * height)))
        y_end = max(
            y_start + 1,
            min(height, math.ceil((float(rectangle["y"]) + float(rectangle["height"])) * height)),
        )
        fill = transform["fill"]
        for channel, value in enumerate(fill):
            expected[:, channel, y_start:y_end, x_start:x_end] = float(value) / 127.5 - 1.0
    return expected


def _transform_digest(manifest: Any) -> str:
    payload = [
        {
            "frame_index": frame.frame_index,
            "source_index": frame.source_index,
            "source_rgb_sha256": frame.metadata.get("rgb_sha256"),
            "transforms": [dict(transform) for transform in frame.transforms],
        }
        for frame in manifest.frames
    ]
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _pixel_digest(manifest: Any, arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for frame, array in zip(manifest.frames, arrays, strict=True):
        contiguous = np.ascontiguousarray(array)
        header = {
            "frame_index": frame.frame_index,
            "source_index": frame.source_index,
            "source_rgb_sha256": frame.metadata.get("rgb_sha256"),
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "payload_size_bytes": contiguous.nbytes,
        }
        header_bytes = _json_bytes(header)
        digest.update(len(header_bytes).to_bytes(8, "big"))
        digest.update(header_bytes)
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _validate_run_pixels(
    run: _PreparedRun,
    *,
    input_root: Path,
    input_tree_snapshot: Mapping[str, _TreeEntrySnapshot],
    raw_source_snapshots: Mapping[Path, _FileSnapshot],
    torch_module: Any,
    runner_module: Any,
    load_input_manifest: Any,
    apply_deferred_transforms: Any,
) -> dict[str, Any]:
    manifest = load_input_manifest(run.input_path)
    if len(manifest.frames) != FRAME_COUNT or manifest.source_manifest_path != run.source_path:
        raise FormalPilotValidationError(f"{run.spec.run_id} strict loader layout mismatch")
    expected_positions = run.spec.consumed_pool_positions
    if tuple(frame.source_index for frame in manifest.frames) != expected_positions:
        raise FormalPilotValidationError(f"{run.spec.run_id} strict final-source order mismatch")
    source_frames = run.source_payload["frames"]
    expected_paths = tuple(
        _resolve_declared_path(
            source_frames[source_index]["path"],
            base=run.run_root,
            name=f"{run.spec.run_id} final source {position}",
        )
        for position, source_index in enumerate(expected_positions)
    )
    if tuple(frame.path for frame in manifest.frames) != expected_paths:
        raise FormalPilotValidationError(f"{run.spec.run_id} final paths differ from source pool")
    tree_immediately_before = _snapshot_input_tree(input_root)
    _assert_input_tree_equal(
        input_tree_snapshot,
        tree_immediately_before,
        name=f"{run.spec.run_id} input tree immediately before official loader",
    )
    loader_sources_before = _snapshot_loader_sources(
        expected_paths,
        expected=raw_source_snapshots,
        name=f"{run.spec.run_id} immediately-before-loader",
    )
    # The tracked runner deliberately enables the official loader's verbose
    # progress output.  Keep those diagnostics on stderr so the production CLI
    # can reserve stdout for its single machine-readable process attestation.
    with contextlib.redirect_stdout(sys.stderr):
        loaded = runner_module._prepare_views(
            list(expected_paths), IMAGE_SIZE, torch_module
        )
    loader_sources_after = _snapshot_loader_sources(
        expected_paths,
        expected=raw_source_snapshots,
        name=f"{run.spec.run_id} immediately-after-loader",
    )
    _assert_snapshot_maps_equal(
        loader_sources_before,
        loader_sources_after,
        name=f"{run.spec.run_id} loader-source snapshot",
    )
    tree_immediately_after = _snapshot_input_tree(input_root)
    _assert_input_tree_equal(
        input_tree_snapshot,
        tree_immediately_after,
        name=f"{run.spec.run_id} input tree immediately after official loader",
    )
    if _cuda_initialized(torch_module):
        raise FormalPilotValidationError(
            f"{run.spec.run_id} official loader initialized CUDA during CPU replay"
        )
    if not isinstance(loaded, list) or len(loaded) != FRAME_COUNT:
        raise FormalPilotValidationError(
            f"{run.spec.run_id} official loader must return exactly 30 views"
        )
    originals: list[np.ndarray] = []
    for position, view in enumerate(loaded):
        if not isinstance(view, Mapping) or "img" not in view:
            raise FormalPilotValidationError(
                f"{run.spec.run_id} loader view {position} lacks img"
            )
        originals.append(_image_array(view["img"], name=f"{run.spec.run_id} loader frame {position}"))
    replayed = apply_deferred_transforms(loaded, manifest)
    if not isinstance(replayed, list) or len(replayed) != FRAME_COUNT:
        raise FormalPilotValidationError(f"{run.spec.run_id} replay length mismatch")
    replay_arrays: list[np.ndarray] = []
    dynamic_count = 0
    for position, (frame, original_view, output_view, before_array) in enumerate(
        zip(manifest.frames, loaded, replayed, originals, strict=True)
    ):
        if not isinstance(output_view, Mapping) or "img" not in output_view:
            raise FormalPilotValidationError(f"{run.spec.run_id} replay frame {position} lacks img")
        original_after = _image_array(
            original_view["img"], name=f"{run.spec.run_id} loader frame {position} after replay"
        )
        if not np.array_equal(original_after, before_array):
            raise FormalPilotValidationError(
                f"{run.spec.run_id} deferred replay mutated loader tensor {position}"
            )
        if output_view["img"] is original_view["img"] or _images_share_storage(
            output_view["img"], original_view["img"]
        ):
            raise FormalPilotValidationError(
                f"{run.spec.run_id} replay output {position} is not an independent clone"
            )
        output_array = _image_array(
            output_view["img"], name=f"{run.spec.run_id} replay frame {position}"
        )
        transforms = list(frame.transforms)
        expected_types: list[str]
        if run.spec.corruption_type == "dynamic_occlusion" and 15 <= position <= 19:
            expected_types = ["rectangle_occlusion"]
            dynamic_count += 1
        elif run.spec.corruption_type == "wrong_order_segment" and 15 <= position <= 18:
            expected_types = ["temporal_reorder"]
        elif run.spec.corruption_type == "low_overlap_jump" and 15 <= position <= 19:
            expected_types = ["source_frame_substitution"]
        else:
            expected_types = []
        if [transform.get("type") for transform in transforms] != expected_types:
            raise FormalPilotValidationError(
                f"{run.spec.run_id} transform layout mismatch at output {position}"
            )
        if expected_types == ["rectangle_occlusion"]:
            corruption = run.input_payload["corruptions"][0]
            assert isinstance(corruption, Mapping)
            parameters = corruption["parameters"]
            assert isinstance(parameters, Mapping)
            expected_transform = {
                "type": "rectangle_occlusion",
                "coordinate_space": parameters["coordinate_space"],
                "coordinate_reference": parameters["coordinate_reference"],
                "rectangle": parameters["frame_rectangles"][position - EVENT_START],
                "fill": parameters["fill"],
            }
            if transforms != [expected_transform]:
                raise FormalPilotValidationError(
                    f"{run.spec.run_id} strict-loader transform is not bound to "
                    f"the frozen rectangle/fill at output {position}"
                )
        expected_array = _expected_replay(before_array, transforms)
        if not np.array_equal(output_array, expected_array):
            raise FormalPilotValidationError(
                f"{run.spec.run_id} replay pixels differ at output {position}"
            )
        if run.spec.corruption_type in {"wrong_order_segment", "low_overlap_jump"} and not np.array_equal(
            output_array, before_array
        ):
            raise FormalPilotValidationError(
                f"{run.spec.run_id} source-order replay changed final-source pixels at {position}"
            )
        replay_arrays.append(output_array)
    if run.spec.corruption_type == "dynamic_occlusion" and dynamic_count != 5:
        raise FormalPilotValidationError(f"{run.spec.run_id} did not verify five dynamic boxes")
    result = {
        "run_id": run.spec.run_id,
        "frame_count": FRAME_COUNT,
        "manifest_validation_passed": True,
        "pixel_replay_passed": True,
        "transform_digest_sha256": _transform_digest(manifest),
        "pixel_digest_sha256": _pixel_digest(manifest, replay_arrays),
    }
    del replay_arrays, originals, replayed, loaded, manifest
    gc.collect()
    return result


def _resolve_output_path(output_report: Path) -> Path:
    root = _canonical_directory(OUTPUT_ROOT, name="formal output root")
    output = Path(os.path.abspath(output_report))
    if output.name != CPU_VALIDATION_FILENAME:
        raise FormalPilotValidationError(
            f"CPU validation report basename must be {CPU_VALIDATION_FILENAME!r}"
        )
    if output.parent.name != PRE_FORWARD_VALIDATION_DIRNAME:
        raise FormalPilotValidationError(
            "CPU validation report direct parent must be the dedicated "
            f"{PRE_FORWARD_VALIDATION_DIRNAME!r} directory"
        )
    try:
        relative = output.relative_to(root)
    except ValueError as error:
        raise FormalPilotValidationError(
            f"CPU validation output must stay inside {root}"
        ) from error
    if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        raise FormalPilotValidationError("CPU validation output path is unsafe")
    forbidden_components = {"runs", "development", "holdout"}
    forbidden = sorted(
        forbidden_components & {part.lower() for part in relative.parts[:-1]}
    )
    if forbidden:
        raise FormalPilotValidationError(
            "CPU validation report may not be placed in run/split output trees: "
            + ", ".join(forbidden)
        )
    parent = _canonical_directory(output.parent, name="CPU validation output parent")
    try:
        parent.relative_to(root)
    except ValueError as error:
        raise FormalPilotValidationError("CPU validation output parent escapes outputs") from error
    if os.path.lexists(output):
        raise FormalPilotValidationError(f"refusing to overwrite CPU validation report: {output}")
    return output


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as error:
        raise FormalPilotValidationError("atomic no-replace publication is unavailable") from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FormalPilotValidationError(f"refusing to overwrite existing report: {destination}")
    raise FormalPilotValidationError(
        "cannot atomically publish report: " + os.strerror(error_number)
    )


def _publish_report(output: Path, payload: bytes) -> Path:
    temporary: Path | None = None
    published = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="xb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".staging",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        _rename_noreplace(temporary, output)
        published = True
        try:
            _fsync_directory(output.parent)
        except OSError as error:
            try:
                _rename_noreplace(output, temporary)
                published = False
            except (FormalPilotValidationError, OSError) as rollback_error:
                raise FormalPilotValidationError(
                    f"parent fsync failed and report rollback failed: {rollback_error}"
                ) from error
            try:
                _fsync_directory(output.parent)
            except OSError:
                pass
            raise FormalPilotValidationError(
                f"parent fsync failed; report publication rolled back: {error}"
            ) from error
        final_metadata = os.lstat(output)
        if not stat.S_ISREG(final_metadata.st_mode) or stat.S_IMODE(final_metadata.st_mode) != 0o444:
            try:
                _rename_noreplace(output, temporary)
                published = False
                _fsync_directory(output.parent)
            except (FormalPilotValidationError, OSError) as rollback_error:
                raise FormalPilotValidationError(
                    "published report mode is invalid and atomic rollback failed: "
                    f"{rollback_error}"
                )
            raise FormalPilotValidationError(
                "published CPU validation report was not immutable 0444; publication rolled back"
            )
        return output
    finally:
        if not published and temporary is not None and os.path.lexists(temporary):
            try:
                temporary.chmod(0o600)
            except OSError:
                pass
            temporary.unlink(missing_ok=True)


def validate_formal_pilot_inputs(
    input_root: str | os.PathLike[str],
    output_report: str | os.PathLike[str],
    *,
    baseline_root: str | os.PathLike[str] = RECAL3R_ROOT,
) -> Path:
    """Validate, replay, and atomically publish one schema-exact PASS report."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise FormalPilotValidationError(
            "CUDA_VISIBLE_DEVICES must be explicitly set to the empty string"
        )
    output = _resolve_output_path(Path(output_report))
    generator = _validate_generator_provenance(_capture_generator_provenance())
    root = _canonical_directory(
        Path(input_root), name="formal input root", read_only=True
    )
    input_tree_before = _snapshot_input_tree(root)
    _validate_input_tree_layout(input_tree_before)
    root, registry_snapshot, runs, controls_before = _read_input_bundle(root)
    declarations = _declared_sources(runs)
    sources_before = _snapshot_declared_sources(declarations)
    before_digest = _snapshot_digest(sources_before)
    if len(sources_before) != EXPECTED_SOURCE_ARTIFACT_COUNT:
        raise FormalPilotValidationError("raw source artifact count changed unexpectedly")
    tum_associations = _validate_tum_associations(runs, sources_before)

    runtime_before = _capture_production_runtime_provenance(
        Path(baseline_root), require_loader_imported=False
    )
    torch_module, runner_module = _load_production_runtime(Path(baseline_root))
    cuda_before = _cuda_initialized(torch_module)
    if cuda_before:
        raise FormalPilotValidationError("torch CUDA was initialized before CPU replay")
    state_src = REPOSITORY_ROOT / "src"
    if str(state_src) not in sys.path:
        sys.path.insert(0, str(state_src))
    from stateguard3r.input_manifest import (
        apply_deferred_transforms,
        load_input_manifest,
    )

    run_rows = [
        _validate_run_pixels(
            run,
            input_root=root,
            input_tree_snapshot=input_tree_before,
            raw_source_snapshots=sources_before,
            torch_module=torch_module,
            runner_module=runner_module,
            load_input_manifest=load_input_manifest,
            apply_deferred_transforms=apply_deferred_transforms,
        )
        for run in runs
    ]
    runtime_after = _capture_production_runtime_provenance(
        Path(baseline_root), require_loader_imported=True
    )
    if runtime_after != runtime_before:
        raise FormalPilotValidationError(
            "production runtime provenance changed during CPU validation"
        )
    cuda_after = _cuda_initialized(torch_module)
    if cuda_after:
        raise FormalPilotValidationError("torch CUDA was initialized during CPU replay")

    sources_after = _snapshot_declared_sources(declarations)
    after_digest = _snapshot_digest(sources_after)
    _assert_snapshot_maps_equal(
        sources_before, sources_after, name="formal raw source snapshot"
    )
    controls_after = {
        path: _read_snapshot(path, name="formal control artifact")[0]
        for path in controls_before
    }
    _assert_snapshot_maps_equal(
        controls_before, controls_after, name="formal manifest/registry snapshot"
    )
    current_registry = controls_after[root / INPUT_REGISTRY_FILENAME]
    if current_registry != registry_snapshot:
        raise FormalPilotValidationError("formal input registry changed during validation")
    input_tree_after = _snapshot_input_tree(root)
    _assert_input_tree_equal(
        input_tree_before,
        input_tree_after,
        name="formal complete input tree snapshot",
    )
    if _validate_generator_provenance(_capture_generator_provenance()) != generator:
        raise FormalPilotValidationError(
            "generator commit/script provenance changed during CPU validation"
        )
    runtime_final = _capture_production_runtime_provenance(
        Path(baseline_root), require_loader_imported=True
    )
    if runtime_final != runtime_before:
        raise FormalPilotValidationError(
            "production runtime provenance changed before report publication"
        )

    run_ids = {
        split: [spec.run_id for spec in FROZEN_RUN_SPECS if spec.dataset_split == split]
        for split in ("development", "holdout")
    }
    report = {
        "schema_version": CPU_VALIDATION_SCHEMA_VERSION,
        "status": "PASS",
        "input_registry": registry_snapshot.artifact(INPUT_REGISTRY_FILENAME),
        "run_ids": run_ids,
        "environment": {
            "CUDA_VISIBLE_DEVICES": "",
            "torch_cuda_is_initialized_before": cuda_before,
            "torch_cuda_is_initialized_after": cuda_after,
        },
        "production_runtime": runtime_final,
        "source_snapshots_before_sha256": before_digest,
        "source_snapshots_after_sha256": after_digest,
        "source_snapshot_artifact_count": len(sources_before),
        "input_tree_snapshots_before_sha256": _input_tree_digest(input_tree_before),
        "input_tree_snapshots_after_sha256": _input_tree_digest(input_tree_after),
        "input_tree_snapshot_entry_count": len(input_tree_before),
        "tum_associations": tum_associations,
        "runs": run_rows,
        "generator": generator,
        "checks": {name: True for name in sorted(CPU_VALIDATION_CHECKS)},
    }
    payload = _json_bytes(report) + b"\n"
    return _publish_report(output, payload)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_report", type=Path)
    parser.add_argument("--baseline-root", type=Path, default=RECAL3R_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        output = validate_formal_pilot_inputs(
            args.input_root,
            args.output_report,
            baseline_root=args.baseline_root,
        )
    except (FormalPilotValidationError, ImportError, OSError, ValueError) as error:
        parser.error(str(error))
    snapshot, _ = _read_snapshot(output, name="published CPU validation report")
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_report": str(output),
                "sha256": snapshot.sha256,
                "size_bytes": snapshot.size_bytes,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
