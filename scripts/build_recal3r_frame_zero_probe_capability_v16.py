#!/usr/bin/env python3
"""Construct the isolated v16 frame-zero probe capability."""

from __future__ import annotations

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

from stateguard3r.frame_zero_probe_capability_v16 import (
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


class BuildFrameZeroProbeCapabilityV16Error(RuntimeError):
    pass


def _read_only_regular(path: Path, *, label: str) -> os.stat_result:
    value = os.lstat(path)
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode) or stat.S_IMODE(value.st_mode) != 0o444:
        raise BuildFrameZeroProbeCapabilityV16Error(f"{label} must be a non-symlink mode-0444 regular file")
    return value


def _selector_frame_zero(*, read_bytes: Callable[[Path], bytes]) -> str:
    _read_only_regular(RAW_SELECTOR, label="v16 raw selector")
    raw = read_bytes(RAW_SELECTOR)
    if hashlib.sha256(raw).hexdigest() != RAW_SELECTOR_SHA256:
        raise BuildFrameZeroProbeCapabilityV16Error("v16 raw selector hash differs")
    try:
        rows = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise BuildFrameZeroProbeCapabilityV16Error("v16 raw selector cannot be parsed") from error
    paths: list[str] = []
    for row in rows:
        current = row.strip()
        if not current or current.startswith("#"):
            continue
        fields = current.split()
        if len(fields) != 2 or not fields[1].startswith("rgb/") or ".." in Path(fields[1]).parts:
            raise BuildFrameZeroProbeCapabilityV16Error("v16 raw selector path differs")
        paths.append(fields[1])
    if len(paths) != LISTING_ROW_COUNT or len(set(paths)) != len(paths):
        raise BuildFrameZeroProbeCapabilityV16Error("v16 raw selector cardinality/order differs")
    try:
        start = paths.index(FIRST_RGB)
    except ValueError as error:
        raise BuildFrameZeroProbeCapabilityV16Error("v16 fixed frame-zero RGB is absent") from error
    selected = paths[start : start + FRAME_COUNT]
    if len(selected) != FRAME_COUNT or hashlib.sha256(json.dumps(selected, separators=(",", ":")).encode("utf-8")).hexdigest() != PATH_LIST_SHA256:
        raise BuildFrameZeroProbeCapabilityV16Error("v16 fixed thirty-frame selector differs")
    if selected[0] != FIRST_RGB:
        raise BuildFrameZeroProbeCapabilityV16Error("v16 fixed frame-zero RGB is absent")
    return selected[0]


def _write_fresh_frozen(path: Path, encoded: bytes) -> None:
    if path != PROBE_CAPSULE or os.path.lexists(path):
        raise BuildFrameZeroProbeCapabilityV16Error("v16 probe capability target must be fresh and fixed")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
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


def build_frame_zero_probe_capability_v16(*, read_bytes: Callable[[Path], bytes] | None = None) -> Mapping[str, Any]:
    reader = read_bytes if read_bytes is not None else Path.read_bytes
    relative = _selector_frame_zero(read_bytes=reader)
    root = DATASET_ROOT.resolve(strict=True)
    image = (root / relative).resolve(strict=True)
    try:
        image.relative_to(root)
    except ValueError as error:
        raise BuildFrameZeroProbeCapabilityV16Error("v16 frame-zero RGB path escapes root") from error
    current = _read_only_regular(image, label="v16 raw frame-zero RGB")
    raw = reader(image)
    value = {"schema": PROBE_SCHEMA, "capability_id": PROBE_CAPSULE_ID, "dataset_root": str(DATASET_ROOT), "rgb_root": str(DATASET_ROOT / "rgb"), "loader": MODEL_LOADER, "frame": {"frame_id": 0, "rgb_relative_path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": current.st_size, "inode": current.st_ino, "mtime_ns": current.st_mtime_ns, "mode_octal": "0444"}}
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _write_fresh_frozen(PROBE_CAPSULE, encoded)
    return {"capsule": str(PROBE_CAPSULE), "capsule_sha256": hashlib.sha256(encoded).hexdigest(), "frame_id": 0, "source": "raw_selector_and_frame_zero_only"}


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(build_frame_zero_probe_capability_v16(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
