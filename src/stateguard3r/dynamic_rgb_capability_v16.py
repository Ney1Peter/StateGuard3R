"""Independent v16 runtime RGB capability and its raw-only constructor."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping


DYNAMIC_SCHEMA = "stateguard3r.dynamic-rgb-capability-v16.v1"
DYNAMIC_CAPSULE_ID = "recovery-update-pressure-v16-dynamic-rgb-list-capsule-0001"
FRAME_COUNT = 30
DATASET_ROOT = Path("/data/wangzheng/Project2/baselines/ReCal3R/data/tum/rgbd_dataset_freiburg1_desk")
RGB_LISTING = DATASET_ROOT / "rgb.txt"
RGB_LISTING_SHA256 = "d1bc510ecca08540e03be8df55af8857753b614d8a5c38526bc669ce6c802284"
FIRST_RGB = "rgb/1305031461.059662.png"
PATH_LIST_SHA256 = "de0d7506a0578704410f7b4c25ad760d4260f20533648aee17cd36545fdfe120"
LISTING_ROW_COUNT = 613
DYNAMIC_CAPSULE = Path(__file__).resolve().parents[2] / "outputs" / f"{DYNAMIC_CAPSULE_ID}.json"
MODEL_LOADER = {
    "function": "dust3r.utils.image.load_images_for_eval",
    "size": 512,
    "crop": True,
    "square_ok": False,
    "coordinate_reference": "model_input_after_resize_and_center_crop",
    "rectangle_rounding": "floor-start-ceil-end",
}
FORBIDDEN_CAPABILITY_WORDS = frozenset({
    "groundtruth", "ground_truth", "depth", "source_index", "manifest",
    "archive", "event", "label", "condition", "corruption", "future",
    "timestamp", "metadata",
})


class DynamicRGBCapabilityV16Error(ValueError):
    """The narrow dynamic RGB capability is invalid or cannot be frozen."""


@dataclass(frozen=True)
class DynamicRGBFrameV16:
    frame_id: int
    relative_path: str
    sha256: str
    size_bytes: int
    inode: int
    mtime_ns: int
    transforms: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class DynamicRGBCapabilityV16:
    path: Path
    dataset_root: Path
    listing: Path
    listing_size_bytes: int
    frames: tuple[DynamicRGBFrameV16, ...]

    @property
    def rgb_paths(self) -> tuple[Path, ...]:
        return tuple((self.dataset_root / frame.relative_path).resolve(strict=False) for frame in self.frames)


def sha256_file_v16(path: Path, *, read_bytes: Callable[[Path], bytes] | None = None) -> str:
    payload = read_bytes(path) if read_bytes is not None else path.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def _read_only_regular(path: Path, *, label: str) -> os.stat_result:
    try:
        value = os.lstat(path)
    except OSError as error:
        raise DynamicRGBCapabilityV16Error(f"cannot stat {label}") from error
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode) or stat.S_IMODE(value.st_mode) != 0o444:
        raise DynamicRGBCapabilityV16Error(f"{label} must be a non-symlink mode-0444 regular file")
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> Mapping[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DynamicRGBCapabilityV16Error("v16 dynamic capability contains a duplicate JSON key")
        value[key] = item
    return value


def _json_object(path: Path, *, read_bytes: Callable[[Path], bytes] | None = None) -> Mapping[str, Any]:
    _read_only_regular(path, label="v16 dynamic capability")
    try:
        payload = read_bytes(path) if read_bytes is not None else path.read_bytes()
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(DynamicRGBCapabilityV16Error(f"invalid JSON constant {token}")),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DynamicRGBCapabilityV16Error("v16 dynamic capability is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise DynamicRGBCapabilityV16Error("v16 dynamic capability root is not an object")
    return value


def _forbidden(value: Any, *, location: str = "capability") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DynamicRGBCapabilityV16Error(f"{location} has a non-string key")
            if any(word in key.lower() for word in FORBIDDEN_CAPABILITY_WORDS):
                raise DynamicRGBCapabilityV16Error(f"{location} has a forbidden capability key")
            _forbidden(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _forbidden(item, location=f"{location}[{index}]")
    elif isinstance(value, str) and any(word in value.lower() for word in FORBIDDEN_CAPABILITY_WORDS):
        raise DynamicRGBCapabilityV16Error(f"{location} has a forbidden capability value")


def _digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise DynamicRGBCapabilityV16Error(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DynamicRGBCapabilityV16Error(f"{label} must be an integer >= {minimum}")
    return value


def _relative_rgb(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("rgb/"):
        raise DynamicRGBCapabilityV16Error(f"{label} must be a safe RGB path")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts or str(candidate) != value:
        raise DynamicRGBCapabilityV16Error(f"{label} path escapes the dataset root")
    return value


def _fixed_transform(frame_id: int) -> tuple[Mapping[str, Any], ...]:
    if not 15 <= frame_id <= 19:
        return ()
    k = frame_id - 15
    return ({
        "type": "rectangle_occlusion",
        "coordinate_space": "normalized",
        "coordinate_reference": MODEL_LOADER["coordinate_reference"],
        "rectangle": {"x": 0.25 + 0.04 * k, "y": 0.25 + 0.025 * k, "width": 0.5, "height": 0.5},
        "fill": [255.0, 0.0, 0.0],
    },)


def _validate_transform(value: Any, *, frame_id: int) -> tuple[Mapping[str, Any], ...]:
    expected = _fixed_transform(frame_id)
    if not isinstance(value, list) or value != list(expected):
        raise DynamicRGBCapabilityV16Error("v16 dynamic transform differs from preregistration")
    return expected


def _ordered_raw_paths(*, read_bytes: Callable[[Path], bytes] | None = None) -> tuple[str, ...]:
    _read_only_regular(RGB_LISTING, label="v16 raw RGB selector")
    raw = read_bytes(RGB_LISTING) if read_bytes is not None else RGB_LISTING.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RGB_LISTING_SHA256:
        raise DynamicRGBCapabilityV16Error("v16 raw RGB selector hash differs")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise DynamicRGBCapabilityV16Error("cannot parse v16 raw RGB selector") from error
    paths: list[str] = []
    for line in lines:
        current = line.strip()
        if not current or current.startswith("#"):
            continue
        fields = current.split()
        if len(fields) != 2:
            raise DynamicRGBCapabilityV16Error("v16 raw RGB selector row differs")
        paths.append(_relative_rgb(fields[1], label="v16 raw RGB selector"))
    if len(paths) != LISTING_ROW_COUNT or len(set(paths)) != len(paths):
        raise DynamicRGBCapabilityV16Error("v16 raw RGB selector order differs")
    try:
        start = paths.index(FIRST_RGB)
    except ValueError as error:
        raise DynamicRGBCapabilityV16Error("v16 fixed first RGB is absent") from error
    selected = tuple(paths[start : start + FRAME_COUNT])
    if len(selected) != FRAME_COUNT or hashlib.sha256(json.dumps(list(selected), separators=(",", ":")).encode("utf-8")).hexdigest() != PATH_LIST_SHA256:
        raise DynamicRGBCapabilityV16Error("v16 dynamic RGB sequence differs")
    return selected


def _encode_capability(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_fresh_frozen(path: Path, encoded: bytes) -> None:
    if path != DYNAMIC_CAPSULE or os.path.lexists(path):
        raise DynamicRGBCapabilityV16Error("v16 dynamic capability target must be fresh and fixed")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise DynamicRGBCapabilityV16Error("v16 dynamic capability target became occupied") from error
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


def build_dynamic_rgb_capability_v16(*, read_bytes: Callable[[Path], bytes] | None = None) -> Mapping[str, Any]:
    """Create the first runtime capability from only its raw selector and RGBs."""
    reader = read_bytes if read_bytes is not None else Path.read_bytes
    listing_stat = _read_only_regular(RGB_LISTING, label="v16 raw RGB selector")
    paths = _ordered_raw_paths(read_bytes=reader)
    frames: list[Mapping[str, Any]] = []
    for frame_id, relative_path in enumerate(paths):
        path = (DATASET_ROOT / relative_path).resolve(strict=True)
        try:
            path.relative_to(DATASET_ROOT.resolve(strict=True))
        except ValueError as error:
            raise DynamicRGBCapabilityV16Error("v16 RGB path escapes raw dataset") from error
        current = _read_only_regular(path, label="v16 raw RGB")
        raw = reader(path)
        frames.append({
            "frame_id": frame_id,
            "rgb_relative_path": relative_path,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": current.st_size,
            "inode": current.st_ino,
            "mtime_ns": current.st_mtime_ns,
            "mode_octal": "0444",
            "transforms": list(_fixed_transform(frame_id)),
        })
    value = {
        "schema": DYNAMIC_SCHEMA,
        "capability_id": DYNAMIC_CAPSULE_ID,
        "dataset_root": str(DATASET_ROOT),
        "rgb_root": str(DATASET_ROOT / "rgb"),
        "rgb_listing": {"relative_path": "rgb.txt", "sha256": RGB_LISTING_SHA256, "size_bytes": listing_stat.st_size, "mode_octal": "0444"},
        "loader": MODEL_LOADER,
        "frames": frames,
    }
    encoded = _encode_capability(value)
    _write_fresh_frozen(DYNAMIC_CAPSULE, encoded)
    # The builder must not inspect its own output after O_EXCL creation.
    return {"capsule": str(DYNAMIC_CAPSULE), "capsule_sha256": hashlib.sha256(encoded).hexdigest(), "frame_count": FRAME_COUNT, "source": "raw_selector_and_rgb_only"}


def load_dynamic_rgb_capability_v16(path: Path, *, read_bytes: Callable[[Path], bytes] | None = None) -> DynamicRGBCapabilityV16:
    raw = _json_object(path, read_bytes=read_bytes)
    _forbidden(raw)
    expected = {"schema", "capability_id", "dataset_root", "rgb_root", "rgb_listing", "loader", "frames"}
    if set(raw) != expected or raw.get("schema") != DYNAMIC_SCHEMA or raw.get("capability_id") != DYNAMIC_CAPSULE_ID or raw.get("loader") != MODEL_LOADER:
        raise DynamicRGBCapabilityV16Error("v16 dynamic capability schema differs")
    if raw.get("dataset_root") != str(DATASET_ROOT) or raw.get("rgb_root") != str(DATASET_ROOT / "rgb"):
        raise DynamicRGBCapabilityV16Error("v16 dynamic capability roots differ")
    listing_raw = raw.get("rgb_listing")
    if not isinstance(listing_raw, Mapping) or set(listing_raw) != {"relative_path", "sha256", "size_bytes", "mode_octal"} or listing_raw.get("relative_path") != "rgb.txt" or listing_raw.get("sha256") != RGB_LISTING_SHA256 or listing_raw.get("mode_octal") != "0444":
        raise DynamicRGBCapabilityV16Error("v16 dynamic listing capability differs")
    listing_size = _integer(listing_raw.get("size_bytes"), label="v16 dynamic listing size", minimum=1)
    dataset_root = DATASET_ROOT.resolve(strict=True)
    frames_raw = raw.get("frames")
    if not isinstance(frames_raw, list) or len(frames_raw) != FRAME_COUNT:
        raise DynamicRGBCapabilityV16Error("v16 dynamic frame cardinality differs")
    paths = _ordered_raw_paths(read_bytes=read_bytes)
    frames: list[DynamicRGBFrameV16] = []
    for frame_id, item in enumerate(frames_raw):
        required = {"frame_id", "rgb_relative_path", "sha256", "size_bytes", "inode", "mtime_ns", "mode_octal", "transforms"}
        if not isinstance(item, Mapping) or set(item) != required or item.get("frame_id") != frame_id or item.get("mode_octal") != "0444":
            raise DynamicRGBCapabilityV16Error("v16 dynamic frame schema differs")
        relative_path = _relative_rgb(item.get("rgb_relative_path"), label="v16 dynamic RGB")
        if relative_path != paths[frame_id]:
            raise DynamicRGBCapabilityV16Error("v16 dynamic RGB order differs")
        frames.append(DynamicRGBFrameV16(frame_id, relative_path, _digest(item.get("sha256"), label="v16 dynamic RGB hash"), _integer(item.get("size_bytes"), label="v16 dynamic RGB size", minimum=1), _integer(item.get("inode"), label="v16 dynamic RGB inode", minimum=1), _integer(item.get("mtime_ns"), label="v16 dynamic RGB mtime", minimum=1), _validate_transform(item.get("transforms"), frame_id=frame_id)))
    return DynamicRGBCapabilityV16(path.resolve(strict=True), dataset_root, RGB_LISTING.resolve(strict=True), listing_size, tuple(frames))


def verify_dynamic_rgb_capability_v16(value: DynamicRGBCapabilityV16, *, read_bytes: Callable[[Path], bytes] | None = None) -> None:
    reader = read_bytes if read_bytes is not None else Path.read_bytes
    _read_only_regular(value.path, label="v16 dynamic capability")
    listing = _read_only_regular(value.listing, label="v16 raw RGB selector")
    listing_raw = reader(value.listing)
    if listing.st_size != value.listing_size_bytes or hashlib.sha256(listing_raw).hexdigest() != RGB_LISTING_SHA256:
        raise DynamicRGBCapabilityV16Error("v16 raw RGB selector changed")
    if tuple(frame.relative_path for frame in value.frames) != _ordered_raw_paths(read_bytes=reader):
        raise DynamicRGBCapabilityV16Error("v16 dynamic RGB paths drifted")
    # The pinned ReCal3R environment is Python 3.9, whose built-in ``zip``
    # does not yet accept ``strict``.  Frame cardinality was validated above,
    # so ordinary zip is exact here as well.
    for frame, path in zip(value.frames, value.rgb_paths):
        try:
            path.relative_to(value.dataset_root)
        except ValueError as error:
            raise DynamicRGBCapabilityV16Error("v16 dynamic RGB path escapes") from error
        current = _read_only_regular(path, label="v16 raw RGB")
        raw = reader(path)
        if current.st_size != frame.size_bytes or current.st_ino != frame.inode or current.st_mtime_ns != frame.mtime_ns or hashlib.sha256(raw).hexdigest() != frame.sha256:
            raise DynamicRGBCapabilityV16Error("v16 raw RGB changed")


def _paint(image: Any, transform: Mapping[str, Any]) -> None:
    _, _, height, width = (int(value) for value in image.shape)
    rectangle = transform["rectangle"]
    xs, xe = math.floor(rectangle["x"] * width), math.ceil((rectangle["x"] + rectangle["width"]) * width)
    ys, ye = math.floor(rectangle["y"] * height), math.ceil((rectangle["y"] + rectangle["height"]) * height)
    for channel, value in enumerate(transform["fill"]):
        image[:, channel, ys:ye, xs:xe] = value / 127.5 - 1.0


def prepare_dynamic_rgb_views_v16(value: DynamicRGBCapabilityV16, *, torch: Any) -> list[dict[str, Any]]:
    verify_dynamic_rgb_capability_v16(value)
    from dust3r.utils.image import load_images_for_eval

    loaded = load_images_for_eval([str(path) for path in value.rgb_paths], size=512, crop=True, square_ok=False, verbose=True)
    if len(loaded) != FRAME_COUNT:
        raise DynamicRGBCapabilityV16Error("official loader returned an unexpected RGB count")
    views: list[dict[str, Any]] = []
    for frame, loaded_view in zip(value.frames, loaded, strict=True):
        image = loaded_view["img"].clone()
        if image is loaded_view["img"] or tuple(image.shape[:2]) != (1, 3):
            raise DynamicRGBCapabilityV16Error("official loader image cannot be cloned")
        for transform in frame.transforms:
            _paint(image, transform)
        batch, _, height, width = image.shape
        views.append({"img": image, "ray_map": torch.full((batch, 6, height, width), torch.nan), "true_shape": torch.from_numpy(loaded_view["true_shape"]), "idx": frame.frame_id, "instance": str(frame.frame_id), "camera_pose": torch.eye(4, dtype=torch.float32).unsqueeze(0), "img_mask": torch.tensor(True).unsqueeze(0), "ray_mask": torch.tensor(False).unsqueeze(0), "update": torch.tensor(True).unsqueeze(0), "reset": torch.tensor(False).unsqueeze(0)})
    return views


def dynamic_rgb_capability_provenance_v16(value: DynamicRGBCapabilityV16) -> Mapping[str, Any]:
    return {"capsule_path": str(value.path), "capsule_sha256": sha256_file_v16(value.path), "capsule_id": DYNAMIC_CAPSULE_ID, "frame_count": FRAME_COUNT, "rgb_listing_path": str(value.listing), "rgb_listing_sha256": RGB_LISTING_SHA256, "loader": dict(MODEL_LOADER), "frame_sha256": [frame.sha256 for frame in value.frames]}


__all__ = ["DYNAMIC_CAPSULE", "DYNAMIC_CAPSULE_ID", "DYNAMIC_SCHEMA", "DynamicRGBCapabilityV16", "DynamicRGBCapabilityV16Error", "DynamicRGBFrameV16", "build_dynamic_rgb_capability_v16", "dynamic_rgb_capability_provenance_v16", "load_dynamic_rgb_capability_v16", "prepare_dynamic_rgb_views_v16", "sha256_file_v16", "verify_dynamic_rgb_capability_v16"]
