#!/usr/bin/env python3
"""Construct the fresh v17 frame-zero CPU-probe capability from raw files."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from stateguard3r.frame_zero_probe_capability_v17 import (
    DATASET_ROOT,
    MODEL_LOADER,
    PROBE_CAPSULE,
    PROBE_CAPSULE_ID,
    PROBE_SCHEMA,
)


RAW_SELECTOR = DATASET_ROOT / ("rgb" + ".txt")
RAW_SELECTOR_SHA256 = "d1bc510ecca08540e03be8df55af8857753b614d8a5c38526bc669ce6c802284"
FIRST_RGB = "rgb/1305031461.059662.png"
FRAME_COUNT = 30
LISTING_ROW_COUNT = 613
PATH_LIST_SHA256 = "de0d7506a0578704410f7b4c25ad760d4260f20533648aee17cd36545fdfe120"


class BuildFrameZeroProbeCapabilityV17Error(RuntimeError):
    """The raw selector/frame-zero builder cannot prove its frozen contract."""


def _read_only_regular(path: Path, *, label: str) -> os.stat_result:
    try:
        value = os.lstat(path)
    except OSError as error:
        raise BuildFrameZeroProbeCapabilityV17Error(f"cannot stat {label}") from error
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode) or stat.S_IMODE(value.st_mode) != 0o444:
        raise BuildFrameZeroProbeCapabilityV17Error(f"{label} must be a non-symlink mode-0444 regular file")
    return value


def _same_snapshot(first: os.stat_result, second: os.stat_result, *, label: str) -> None:
    if (first.st_size, first.st_ino, first.st_mtime_ns, stat.S_IMODE(first.st_mode)) != (
        second.st_size, second.st_ino, second.st_mtime_ns, stat.S_IMODE(second.st_mode),
    ):
        raise BuildFrameZeroProbeCapabilityV17Error(f"{label} changed while it was being read")


def _selector_frame_zero(*, read_bytes: Callable[[Path], bytes]) -> str:
    before = _read_only_regular(RAW_SELECTOR, label="v17 raw selector")
    raw = read_bytes(RAW_SELECTOR)
    _same_snapshot(before, _read_only_regular(RAW_SELECTOR, label="v17 raw selector"), label="v17 raw selector")
    if hashlib.sha256(raw).hexdigest() != RAW_SELECTOR_SHA256:
        raise BuildFrameZeroProbeCapabilityV17Error("v17 raw selector hash differs")
    try:
        rows = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise BuildFrameZeroProbeCapabilityV17Error("v17 raw selector cannot be parsed") from error
    paths: list[str] = []
    for row in rows:
        current = row.strip()
        if not current or current.startswith("#"):
            continue
        fields = current.split()
        candidate = Path(fields[1]) if len(fields) == 2 else Path()
        if len(fields) != 2 or not fields[1].startswith("rgb/") or candidate.is_absolute() or ".." in candidate.parts or str(candidate) != fields[1]:
            raise BuildFrameZeroProbeCapabilityV17Error("v17 raw selector path differs")
        paths.append(fields[1])
    if len(paths) != LISTING_ROW_COUNT or len(set(paths)) != len(paths):
        raise BuildFrameZeroProbeCapabilityV17Error("v17 raw selector cardinality/order differs")
    try:
        start = paths.index(FIRST_RGB)
    except ValueError as error:
        raise BuildFrameZeroProbeCapabilityV17Error("v17 fixed frame-zero RGB is absent") from error
    selected = paths[start : start + FRAME_COUNT]
    if len(selected) != FRAME_COUNT or hashlib.sha256(json.dumps(selected, separators=(",", ":")).encode("utf-8")).hexdigest() != PATH_LIST_SHA256:
        raise BuildFrameZeroProbeCapabilityV17Error("v17 fixed thirty-frame selector differs")
    return selected[0]


def _write_fresh_frozen(path: Path, encoded: bytes) -> None:
    if path != PROBE_CAPSULE or os.path.lexists(path):
        raise BuildFrameZeroProbeCapabilityV17Error("v17 frame-zero capability target must be fresh and fixed")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise BuildFrameZeroProbeCapabilityV17Error("v17 frame-zero capability target became occupied") from error
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.chmod(path, 0o444)
        except OSError:
            pass
        raise


def build_frame_zero_probe_capability_v17(*, read_bytes: Callable[[Path], bytes] | None = None) -> Mapping[str, Any]:
    """Read the selector plus frame zero only, then publish an immutable capsule."""
    reader = read_bytes if read_bytes is not None else Path.read_bytes
    relative = _selector_frame_zero(read_bytes=reader)
    root = DATASET_ROOT.resolve(strict=True)
    image = (root / relative).resolve(strict=True)
    try:
        image.relative_to(root)
    except ValueError as error:
        raise BuildFrameZeroProbeCapabilityV17Error("v17 frame-zero RGB path escapes root") from error
    before = _read_only_regular(image, label="v17 raw frame-zero RGB")
    raw = reader(image)
    _same_snapshot(before, _read_only_regular(image, label="v17 raw frame-zero RGB"), label="v17 raw frame-zero RGB")
    value = {
        "schema": PROBE_SCHEMA,
        "capability_id": PROBE_CAPSULE_ID,
        "dataset_root": str(DATASET_ROOT),
        "rgb_root": str(DATASET_ROOT / "rgb"),
        "loader": MODEL_LOADER,
        "frame": {
            "frame_id": 0,
            "rgb_relative_path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": before.st_size,
            "inode": before.st_ino,
            "mtime_ns": before.st_mtime_ns,
            "mode_octal": "0444",
        },
    }
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _write_fresh_frozen(PROBE_CAPSULE, encoded)
    return {
        "capsule": str(PROBE_CAPSULE),
        "capsule_sha256": hashlib.sha256(encoded).hexdigest(),
        "frame_id": 0,
        "source": "raw_selector_and_frame_zero_only",
    }


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(build_frame_zero_probe_capability_v17(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
