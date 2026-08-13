"""One-frame opaque capability for v20's CUDA-hidden interface probe."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping


CAPSULE_ID = "native-scalar-hold-v20-frame0-probe-0001"
CAPSULE = Path(__file__).resolve().parents[2] / "outputs" / f"{CAPSULE_ID}.json"
RGB = Path("/data/wangzheng/Project2/baselines/ReCal3R/data/tum/rgbd_dataset_freiburg1_desk/rgb/1305031461.059662.png")
CHECKPOINT = Path("/data/wangzheng/Project2/baselines/ReCal3R/src/cut3r_512_dpt_4_64.pth")
CHECKPOINT_SHA256 = "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103"


class FrameZeroProbeCapabilityV20Error(ValueError):
    """The v20 one-frame CPU probe capability is mutable or malformed."""


def _regular(path: Path, *, label: str) -> os.stat_result:
    try: value = os.lstat(path)
    except OSError as error: raise FrameZeroProbeCapabilityV20Error(f"cannot stat {label}") from error
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode) or stat.S_IMODE(value.st_mode) != 0o444:
        raise FrameZeroProbeCapabilityV20Error(f"{label} must be a mode-0444 regular nonsymlink")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def build_frame_zero_probe_capability_v20() -> Mapping[str, Any]:
    if os.path.lexists(CAPSULE): raise FrameZeroProbeCapabilityV20Error("v20 frame-zero target is occupied")
    rgb, checkpoint = _regular(RGB, label="v20 frame-zero RGB"), _regular(CHECKPOINT, label="v20 checkpoint")
    if _sha(CHECKPOINT) != CHECKPOINT_SHA256: raise FrameZeroProbeCapabilityV20Error("v20 checkpoint differs")
    value = {"schema": "stateguard3r.frame-zero-probe-v20.v1", "capability_id": CAPSULE_ID, "rgb_path": str(RGB), "rgb_sha256": _sha(RGB), "rgb_size_bytes": rgb.st_size, "checkpoint_path": str(CHECKPOINT), "checkpoint_sha256": CHECKPOINT_SHA256, "checkpoint_size_bytes": checkpoint.st_size}
    encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(CAPSULE, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    return {"capsule": str(CAPSULE), "capsule_sha256": hashlib.sha256(encoded).hexdigest()}


def load_frame_zero_probe_capability_v20(path: Path) -> Mapping[str, Any]:
    _regular(path, label="v20 frame-zero capability")
    try: value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error: raise FrameZeroProbeCapabilityV20Error("v20 frame-zero capability JSON differs") from error
    expected = {"schema", "capability_id", "rgb_path", "rgb_sha256", "rgb_size_bytes", "checkpoint_path", "checkpoint_sha256", "checkpoint_size_bytes"}
    if not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "stateguard3r.frame-zero-probe-v20.v1" or value.get("capability_id") != CAPSULE_ID or value.get("rgb_path") != str(RGB) or value.get("checkpoint_path") != str(CHECKPOINT) or value.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise FrameZeroProbeCapabilityV20Error("v20 frame-zero capability schema differs")
    return dict(value)


__all__ = ["CAPSULE", "CAPSULE_ID", "CHECKPOINT", "CHECKPOINT_SHA256", "FrameZeroProbeCapabilityV20Error", "RGB", "build_frame_zero_probe_capability_v20", "load_frame_zero_probe_capability_v20"]
