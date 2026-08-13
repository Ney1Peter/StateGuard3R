"""Parse and verify the v16 probe's isolated one-frame capability.

This module deliberately has no raw selector capability.  A probe can read
only this frozen file and the one RGB it names; selection is established by a
separate non-CUDA builder before the probe begins.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping


PROBE_SCHEMA = "stateguard3r.frame-zero-probe-capability-v16.v1"
PROBE_CAPSULE_ID = "recovery-update-pressure-v16-dynamic-frame0-probe-capsule-0001"
DATASET_ROOT = Path("/data/wangzheng/Project2/baselines/ReCal3R/data/tum/rgbd_dataset_freiburg1_desk")
PROBE_CAPSULE = Path(__file__).resolve().parents[2] / "outputs" / f"{PROBE_CAPSULE_ID}.json"
MODEL_LOADER = {
    "function": "dust3r.utils.image.load_images_for_eval",
    "size": 512,
    "crop": True,
    "square_ok": False,
    "coordinate_reference": "model_input_after_resize_and_center_crop",
    "rectangle_rounding": "floor-start-ceil-end",
}


class FrameZeroProbeCapabilityV16Error(ValueError):
    """The probe's sole RGB capability is malformed or no longer frozen."""


@dataclass(frozen=True)
class FrameZeroProbeCapabilityV16:
    path: Path
    dataset_root: Path
    rgb_path: Path
    sha256: str
    size_bytes: int
    inode: int
    mtime_ns: int


def _read_only_regular(path: Path, *, label: str) -> os.stat_result:
    try:
        value = os.lstat(path)
    except OSError as error:
        raise FrameZeroProbeCapabilityV16Error(f"cannot stat {label}") from error
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode) or stat.S_IMODE(value.st_mode) != 0o444:
        raise FrameZeroProbeCapabilityV16Error(f"{label} must be a non-symlink mode-0444 regular file")
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> Mapping[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise FrameZeroProbeCapabilityV16Error("v16 frame-zero probe capability contains a duplicate JSON key")
        value[key] = item
    return value


def _json_object(path: Path, *, read_bytes: Callable[[Path], bytes] | None = None) -> Mapping[str, Any]:
    _read_only_regular(path, label="v16 frame-zero probe capability")
    try:
        payload = read_bytes(path) if read_bytes is not None else path.read_bytes()
        value = json.loads(payload, object_pairs_hook=_reject_duplicate_pairs, parse_constant=lambda token: (_ for _ in ()).throw(FrameZeroProbeCapabilityV16Error(f"invalid JSON constant {token}")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FrameZeroProbeCapabilityV16Error("v16 frame-zero probe capability is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise FrameZeroProbeCapabilityV16Error("v16 frame-zero probe capability root is not an object")
    return value


def _digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise FrameZeroProbeCapabilityV16Error(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise FrameZeroProbeCapabilityV16Error(f"{label} must be an integer >= {minimum}")
    return value


def _relative_rgb(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("rgb/"):
        raise FrameZeroProbeCapabilityV16Error("v16 frame-zero RGB path is invalid")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts or str(candidate) != value:
        raise FrameZeroProbeCapabilityV16Error("v16 frame-zero RGB path escapes dataset root")
    return value


def load_frame_zero_probe_capability_v16(path: Path, *, read_bytes: Callable[[Path], bytes] | None = None) -> FrameZeroProbeCapabilityV16:
    raw = _json_object(path, read_bytes=read_bytes)
    expected = {"schema", "capability_id", "dataset_root", "rgb_root", "loader", "frame"}
    if set(raw) != expected or raw.get("schema") != PROBE_SCHEMA or raw.get("capability_id") != PROBE_CAPSULE_ID or raw.get("loader") != MODEL_LOADER:
        raise FrameZeroProbeCapabilityV16Error("v16 frame-zero probe capability schema differs")
    if raw.get("dataset_root") != str(DATASET_ROOT) or raw.get("rgb_root") != str(DATASET_ROOT / "rgb"):
        raise FrameZeroProbeCapabilityV16Error("v16 frame-zero probe capability roots differ")
    frame = raw.get("frame")
    required = {"frame_id", "rgb_relative_path", "sha256", "size_bytes", "inode", "mtime_ns", "mode_octal"}
    if not isinstance(frame, Mapping) or set(frame) != required or frame.get("frame_id") != 0 or frame.get("mode_octal") != "0444":
        raise FrameZeroProbeCapabilityV16Error("v16 frame-zero probe frame schema differs")
    root = DATASET_ROOT.resolve(strict=True)
    rgb_path = (root / _relative_rgb(frame.get("rgb_relative_path"))).resolve(strict=False)
    try:
        rgb_path.relative_to(root)
    except ValueError as error:
        raise FrameZeroProbeCapabilityV16Error("v16 frame-zero RGB path escapes root") from error
    return FrameZeroProbeCapabilityV16(path.resolve(strict=True), root, rgb_path, _digest(frame.get("sha256"), label="v16 frame-zero RGB hash"), _integer(frame.get("size_bytes"), label="v16 frame-zero RGB size", minimum=1), _integer(frame.get("inode"), label="v16 frame-zero RGB inode", minimum=1), _integer(frame.get("mtime_ns"), label="v16 frame-zero RGB mtime", minimum=1))


def parse_frame_zero_probe_capability_v16_bytes(payload: bytes, *, capsule_path: Path) -> FrameZeroProbeCapabilityV16:
    """Parse already-ledgered capsule bytes without opening any file."""
    try:
        raw = json.loads(payload, object_pairs_hook=_reject_duplicate_pairs, parse_constant=lambda token: (_ for _ in ()).throw(FrameZeroProbeCapabilityV16Error(f"invalid JSON constant {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FrameZeroProbeCapabilityV16Error("v16 frame-zero probe capability is not valid JSON") from error
    if not isinstance(raw, Mapping):
        raise FrameZeroProbeCapabilityV16Error("v16 frame-zero probe capability root is not an object")
    expected = {"schema", "capability_id", "dataset_root", "rgb_root", "loader", "frame"}
    if set(raw) != expected or raw.get("schema") != PROBE_SCHEMA or raw.get("capability_id") != PROBE_CAPSULE_ID or raw.get("loader") != MODEL_LOADER:
        raise FrameZeroProbeCapabilityV16Error("v16 frame-zero probe capability schema differs")
    if raw.get("dataset_root") != str(DATASET_ROOT) or raw.get("rgb_root") != str(DATASET_ROOT / "rgb"):
        raise FrameZeroProbeCapabilityV16Error("v16 frame-zero probe capability roots differ")
    frame = raw.get("frame")
    required = {"frame_id", "rgb_relative_path", "sha256", "size_bytes", "inode", "mtime_ns", "mode_octal"}
    if not isinstance(frame, Mapping) or set(frame) != required or frame.get("frame_id") != 0 or frame.get("mode_octal") != "0444":
        raise FrameZeroProbeCapabilityV16Error("v16 frame-zero probe frame schema differs")
    root = DATASET_ROOT.resolve(strict=True)
    rgb_path = (root / _relative_rgb(frame.get("rgb_relative_path"))).resolve(strict=False)
    try:
        rgb_path.relative_to(root)
    except ValueError as error:
        raise FrameZeroProbeCapabilityV16Error("v16 frame-zero RGB path escapes root") from error
    return FrameZeroProbeCapabilityV16(capsule_path.resolve(strict=True), root, rgb_path, _digest(frame.get("sha256"), label="v16 frame-zero RGB hash"), _integer(frame.get("size_bytes"), label="v16 frame-zero RGB size", minimum=1), _integer(frame.get("inode"), label="v16 frame-zero RGB inode", minimum=1), _integer(frame.get("mtime_ns"), label="v16 frame-zero RGB mtime", minimum=1))


def verify_frame_zero_probe_capability_v16(value: FrameZeroProbeCapabilityV16, *, read_bytes: Callable[[Path], bytes] | None = None) -> None:
    _read_only_regular(value.path, label="v16 frame-zero probe capability")
    current = _read_only_regular(value.rgb_path, label="v16 frame-zero RGB")
    raw = read_bytes(value.rgb_path) if read_bytes is not None else value.rgb_path.read_bytes()
    if current.st_size != value.size_bytes or current.st_ino != value.inode or current.st_mtime_ns != value.mtime_ns or hashlib.sha256(raw).hexdigest() != value.sha256:
        raise FrameZeroProbeCapabilityV16Error("v16 frame-zero RGB changed")


__all__ = ["DATASET_ROOT", "FrameZeroProbeCapabilityV16", "FrameZeroProbeCapabilityV16Error", "MODEL_LOADER", "PROBE_CAPSULE", "PROBE_CAPSULE_ID", "PROBE_SCHEMA", "load_frame_zero_probe_capability_v16", "parse_frame_zero_probe_capability_v16_bytes", "verify_frame_zero_probe_capability_v16"]
