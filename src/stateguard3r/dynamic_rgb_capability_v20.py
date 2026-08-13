"""Fresh v20 RGB capability built directly from pinned raw data only."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping


SCHEMA = "stateguard3r.dynamic-rgb-capability-v20.v1"
CAPSULE_ID = "native-scalar-hold-v20-dynamic-rgb-0001"
ROOT = Path("/data/wangzheng/Project2/baselines/ReCal3R/data/tum/rgbd_dataset_freiburg1_desk")
LISTING = ROOT / "rgb.txt"
LISTING_SHA256 = "d1bc510ecca08540e03be8df55af8857753b614d8a5c38526bc669ce6c802284"
FIRST = "rgb/1305031461.059662.png"
PATHS_SHA256 = "de0d7506a0578704410f7b4c25ad760d4260f20533648aee17cd36545fdfe120"
FRAME_COUNT = 30
CAPSULE = Path(__file__).resolve().parents[2] / "outputs" / f"{CAPSULE_ID}.json"
LOADER = {"function": "dust3r.utils.image.load_images_for_eval", "size": 512, "crop": True, "square_ok": False}


class DynamicRGBCapabilityV20Error(ValueError):
    """The independent v20 raw RGB capability is unsafe or differs."""


@dataclass(frozen=True, slots=True)
class FrameV20:
    frame_id: int
    relative_path: str
    sha256: str
    size_bytes: int
    inode: int
    mtime_ns: int
    transforms: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class CapabilityV20:
    path: Path
    frames: tuple[FrameV20, ...]

    @property
    def rgb_paths(self) -> tuple[Path, ...]:
        return tuple(ROOT / frame.relative_path for frame in self.frames)


def sha256_v20(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise DynamicRGBCapabilityV20Error(f"cannot stat {label}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o444:
        raise DynamicRGBCapabilityV20Error(f"{label} must be a mode-0444 nonsymlink regular file")
    return metadata


def _paths() -> tuple[str, ...]:
    before = _regular(LISTING, label="v20 raw rgb listing")
    raw = LISTING.read_bytes()
    after = _regular(LISTING, label="v20 raw rgb listing")
    if (before.st_ino, before.st_mtime_ns, before.st_size) != (after.st_ino, after.st_mtime_ns, after.st_size) or hashlib.sha256(raw).hexdigest() != LISTING_SHA256:
        raise DynamicRGBCapabilityV20Error("v20 raw rgb listing differs during read")
    entries: list[str] = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2 or not fields[1].startswith("rgb/") or ".." in Path(fields[1]).parts:
            raise DynamicRGBCapabilityV20Error("v20 raw rgb listing row differs")
        entries.append(fields[1])
    try:
        start = entries.index(FIRST)
    except ValueError as error:
        raise DynamicRGBCapabilityV20Error("v20 fixed first RGB is absent") from error
    chosen = tuple(entries[start:start + FRAME_COUNT])
    if len(chosen) != FRAME_COUNT or hashlib.sha256(json.dumps(list(chosen), separators=(",", ":")).encode("utf-8")).hexdigest() != PATHS_SHA256:
        raise DynamicRGBCapabilityV20Error("v20 ordered RGB selection differs")
    return chosen


def _transform(frame_id: int) -> tuple[Mapping[str, Any], ...]:
    if not 15 <= frame_id <= 19:
        return ()
    index = frame_id - 15
    return ({"type": "rectangle_occlusion", "coordinate_space": "normalized", "rectangle": {"x": 0.25 + 0.04 * index, "y": 0.25 + 0.025 * index, "width": 0.5, "height": 0.5}, "fill": [255, 0, 0]},)


def _encode(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _publish(encoded: bytes) -> None:
    if os.path.lexists(CAPSULE):
        raise DynamicRGBCapabilityV20Error("v20 dynamic capability target is already occupied")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(CAPSULE, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o444)


def build_dynamic_rgb_capability_v20() -> Mapping[str, Any]:
    """Publish a one-use raw selector/RGB-only v20 capability."""

    frames: list[Mapping[str, Any]] = []
    for frame_id, relative in enumerate(_paths()):
        path = (ROOT / relative).resolve(strict=True)
        try:
            path.relative_to(ROOT.resolve(strict=True))
        except ValueError as error:
            raise DynamicRGBCapabilityV20Error("v20 RGB path escapes dataset root") from error
        before = _regular(path, label="v20 raw RGB")
        content = path.read_bytes()
        after = _regular(path, label="v20 raw RGB")
        if (before.st_ino, before.st_mtime_ns, before.st_size) != (after.st_ino, after.st_mtime_ns, after.st_size):
            raise DynamicRGBCapabilityV20Error("v20 raw RGB changed during read")
        frames.append({"frame_id": frame_id, "rgb_relative_path": relative, "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": before.st_size, "inode": before.st_ino, "mtime_ns": before.st_mtime_ns, "mode_octal": "0444", "transforms": list(_transform(frame_id))})
    value = {"schema": SCHEMA, "capability_id": CAPSULE_ID, "dataset_root": str(ROOT), "rgb_listing": {"relative_path": "rgb.txt", "sha256": LISTING_SHA256, "mode_octal": "0444"}, "loader": LOADER, "frames": frames}
    encoded = _encode(value)
    _publish(encoded)
    return {"capsule": str(CAPSULE), "capsule_sha256": hashlib.sha256(encoded).hexdigest(), "frame_count": FRAME_COUNT, "source": "raw_listing_and_rgb_only"}


def load_dynamic_rgb_capability_v20(path: Path) -> CapabilityV20:
    _regular(path, label="v20 dynamic capability")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DynamicRGBCapabilityV20Error("v20 dynamic capability is not valid JSON") from error
    expected = {"schema", "capability_id", "dataset_root", "rgb_listing", "loader", "frames"}
    if not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != SCHEMA or value.get("capability_id") != CAPSULE_ID or value.get("dataset_root") != str(ROOT) or value.get("loader") != LOADER:
        raise DynamicRGBCapabilityV20Error("v20 dynamic capability schema differs")
    listing = value.get("rgb_listing")
    if listing != {"relative_path": "rgb.txt", "sha256": LISTING_SHA256, "mode_octal": "0444"}:
        raise DynamicRGBCapabilityV20Error("v20 listing capability differs")
    rows = value.get("frames")
    paths = _paths()
    if not isinstance(rows, list) or len(rows) != FRAME_COUNT:
        raise DynamicRGBCapabilityV20Error("v20 dynamic capability frame count differs")
    frames: list[FrameV20] = []
    for frame_id, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"frame_id", "rgb_relative_path", "sha256", "size_bytes", "inode", "mtime_ns", "mode_octal", "transforms"} or row.get("frame_id") != frame_id or row.get("rgb_relative_path") != paths[frame_id] or row.get("mode_octal") != "0444" or row.get("transforms") != list(_transform(frame_id)):
            raise DynamicRGBCapabilityV20Error("v20 dynamic frame schema differs")
        digest = row.get("sha256")
        values = (row.get("size_bytes"), row.get("inode"), row.get("mtime_ns"))
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest) or any(type(item) is not int or item < 1 for item in values):
            raise DynamicRGBCapabilityV20Error("v20 dynamic frame metadata differs")
        frames.append(FrameV20(frame_id, paths[frame_id], digest, *values, _transform(frame_id)))
    return CapabilityV20(path.resolve(strict=True), tuple(frames))


def verify_dynamic_rgb_capability_v20(capability: CapabilityV20) -> None:
    _regular(capability.path, label="v20 dynamic capability")
    if len(capability.frames) != FRAME_COUNT or tuple(frame.relative_path for frame in capability.frames) != _paths():
        raise DynamicRGBCapabilityV20Error("v20 capability selection differs")
    for frame, path in zip(capability.frames, capability.rgb_paths, strict=True):
        before = _regular(path, label="v20 raw RGB")
        content = path.read_bytes()
        after = _regular(path, label="v20 raw RGB")
        if (before.st_size, before.st_ino, before.st_mtime_ns, hashlib.sha256(content).hexdigest()) != (frame.size_bytes, frame.inode, frame.mtime_ns, frame.sha256) or (before.st_ino, before.st_mtime_ns, before.st_size) != (after.st_ino, after.st_mtime_ns, after.st_size):
            raise DynamicRGBCapabilityV20Error("v20 raw RGB verification differs")


def _paint(image: Any, transform: Mapping[str, Any]) -> None:
    _, _, height, width = (int(item) for item in image.shape)
    rectangle = transform["rectangle"]
    xs, xe = math.floor(rectangle["x"] * width), math.ceil((rectangle["x"] + rectangle["width"]) * width)
    ys, ye = math.floor(rectangle["y"] * height), math.ceil((rectangle["y"] + rectangle["height"]) * height)
    for channel, value in enumerate(transform["fill"]):
        image[:, channel, ys:ye, xs:xe] = value / 127.5 - 1.0


def prepare_dynamic_rgb_views_v20(capability: CapabilityV20, *, torch: Any) -> list[dict[str, Any]]:
    verify_dynamic_rgb_capability_v20(capability)
    from dust3r.utils.image import load_images_for_eval
    loaded = load_images_for_eval([str(path) for path in capability.rgb_paths], size=512, crop=True, square_ok=False, verbose=True)
    if len(loaded) != FRAME_COUNT:
        raise DynamicRGBCapabilityV20Error("official v20 loader frame count differs")
    output: list[dict[str, Any]] = []
    for frame, row in zip(capability.frames, loaded, strict=True):
        image = row["img"].clone()
        for transform in frame.transforms:
            _paint(image, transform)
        batch, _, height, width = image.shape
        output.append({"img": image, "ray_map": torch.full((batch, 6, height, width), torch.nan), "true_shape": torch.from_numpy(row["true_shape"]), "idx": frame.frame_id, "instance": str(frame.frame_id), "camera_pose": torch.eye(4, dtype=torch.float32).unsqueeze(0), "img_mask": torch.tensor(True).unsqueeze(0), "ray_mask": torch.tensor(False).unsqueeze(0), "update": torch.tensor(True).unsqueeze(0), "reset": torch.tensor(False).unsqueeze(0)})
    return output


__all__ = ["CAPSULE", "CAPSULE_ID", "CapabilityV20", "DynamicRGBCapabilityV20Error", "FRAME_COUNT", "FrameV20", "build_dynamic_rgb_capability_v20", "load_dynamic_rgb_capability_v20", "prepare_dynamic_rgb_views_v20", "sha256_v20", "verify_dynamic_rgb_capability_v20"]
