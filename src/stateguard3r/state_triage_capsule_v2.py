"""Strict, read-only Stage-0 input capsules for StateTriage3R v2.

The capsule records a deterministic input recipe and its event label before a
ReCal3R forward.  It deliberately does not understand model outputs, GT,
future evidence, prior detector artifacts, or any recovery policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "stateguard3r.state-triage-stage0-capsule.v1"
COORDINATE_REFERENCE = "model_input_after_resize_and_center_crop"
CAUSES = frozenset(
    {
        "registration_or_order_fault",
        "bad_observation",
        "transient_local_content",
        "normal_novelty",
    }
)
TRANSFORM_TYPES = frozenset({"temporal_reorder", "rectangle_occlusion", "checkerboard_occlusion", "hashed_tile_occlusion"})


class Stage0CapsuleError(ValueError):
    """Raised when a capsule cannot support an auditable Stage-0 forward."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise Stage0CapsuleError(f"cannot read capsule {path}: {error}") from error

    def reject_constant(value: str) -> None:
        raise Stage0CapsuleError(f"capsule contains non-finite JSON constant {value!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Stage0CapsuleError(f"capsule repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, Stage0CapsuleError) as error:
        raise Stage0CapsuleError(f"capsule is not strict UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise Stage0CapsuleError("capsule root must be an object")
    return value


def _plain_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise Stage0CapsuleError(f"{name} must be an integer")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Stage0CapsuleError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise Stage0CapsuleError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise Stage0CapsuleError(f"{name} must be finite")
    return result


def _regular_read_only(path: Path, *, name: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise Stage0CapsuleError(f"cannot stat {name}: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise Stage0CapsuleError(f"{name} must be a regular non-symlink file")
    if stat.S_IMODE(metadata.st_mode) & 0o222:
        raise Stage0CapsuleError(f"{name} must be read-only")


@dataclass(frozen=True)
class Stage0Event:
    event_id: str
    cause: str
    start_frame: int
    end_frame: int
    coverage_expectation: str

    def __post_init__(self) -> None:
        if not self.event_id or self.event_id != self.event_id.strip():
            raise Stage0CapsuleError("event_id must be non-empty")
        if self.cause not in CAUSES:
            raise Stage0CapsuleError(f"unsupported event cause {self.cause!r}")
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise Stage0CapsuleError("event frame range is invalid")
        expected = "novel" if self.cause == "normal_novelty" else "covered"
        if self.coverage_expectation != expected:
            raise Stage0CapsuleError(f"{self.cause} requires coverage_expectation={expected!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "cause": self.cause,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "coverage_expectation": self.coverage_expectation,
            "label_provenance": "pre_forward_deterministic_recipe",
        }


@dataclass(frozen=True)
class Stage0Frame:
    frame_id: int
    path: Path
    sha256: str
    timestamp: float
    transforms: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise Stage0CapsuleError("frame_id must be non-negative")
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise Stage0CapsuleError("frame SHA-256 must be lowercase hexadecimal")
        if not math.isfinite(self.timestamp):
            raise Stage0CapsuleError("timestamp must be finite")


@dataclass(frozen=True)
class Stage0Capsule:
    path: Path
    capsule_id: str
    source_sequence: str
    recipe_id: str
    event: Stage0Event
    frames: tuple[Stage0Frame, ...]
    source_group: str

    def __post_init__(self) -> None:
        if not self.capsule_id or not self.source_sequence or not self.recipe_id or not self.source_group:
            raise Stage0CapsuleError("capsule identifiers must be non-empty")
        if len(self.frames) < 2:
            raise Stage0CapsuleError("capsule must contain at least two frames")
        if [frame.frame_id for frame in self.frames] != list(range(len(self.frames))):
            raise Stage0CapsuleError("capsule frame IDs must be contiguous")
        if self.event.end_frame >= len(self.frames):
            raise Stage0CapsuleError("event range exceeds capsule frames")

    def frame_paths(self) -> tuple[Path, ...]:
        return tuple(frame.path for frame in self.frames)


def _rectangle_transform(value: Mapping[str, Any], *, frame_id: int) -> dict[str, Any]:
    required = {"type", "coordinate_reference", "rectangle", "fill"}
    if set(value) != required or value.get("type") != "rectangle_occlusion":
        raise Stage0CapsuleError(f"frame {frame_id} has invalid rectangle transform fields")
    if value.get("coordinate_reference") != COORDINATE_REFERENCE:
        raise Stage0CapsuleError(f"frame {frame_id} rectangle coordinate reference differs")
    raw_rectangle = value.get("rectangle")
    if not isinstance(raw_rectangle, Mapping) or set(raw_rectangle) != {"x", "y", "width", "height"}:
        raise Stage0CapsuleError(f"frame {frame_id} rectangle is invalid")
    rectangle = {name: _finite(raw_rectangle[name], f"frame {frame_id} rectangle.{name}") for name in raw_rectangle}
    if not 0.0 <= rectangle["x"] < 1.0 or not 0.0 <= rectangle["y"] < 1.0:
        raise Stage0CapsuleError(f"frame {frame_id} rectangle origin is invalid")
    if not 0.0 < rectangle["width"] <= 1.0 or not 0.0 < rectangle["height"] <= 1.0:
        raise Stage0CapsuleError(f"frame {frame_id} rectangle size is invalid")
    if rectangle["x"] + rectangle["width"] > 1.0 or rectangle["y"] + rectangle["height"] > 1.0:
        raise Stage0CapsuleError(f"frame {frame_id} rectangle exceeds bounds")
    fill = value.get("fill")
    if not isinstance(fill, list) or len(fill) != 3:
        raise Stage0CapsuleError(f"frame {frame_id} rectangle fill is invalid")
    normalized_fill = [_finite(component, f"frame {frame_id} fill") for component in fill]
    if any(component < -1.0 or component > 1.0 for component in normalized_fill):
        raise Stage0CapsuleError(f"frame {frame_id} fill must be normalized to [-1, 1]")
    return {
        "type": "rectangle_occlusion",
        "coordinate_reference": COORDINATE_REFERENCE,
        "rectangle": rectangle,
        "fill": normalized_fill,
    }


def _checkerboard_transform(value: Mapping[str, Any], *, frame_id: int) -> dict[str, Any]:
    required = {"type", "coordinate_reference", "rectangle", "tile_size_pixels", "fills"}
    if set(value) != required or value.get("type") != "checkerboard_occlusion":
        raise Stage0CapsuleError(f"frame {frame_id} has invalid checkerboard transform fields")
    if value.get("coordinate_reference") != COORDINATE_REFERENCE:
        raise Stage0CapsuleError(f"frame {frame_id} checkerboard coordinate reference differs")
    raw_rectangle = value.get("rectangle")
    if not isinstance(raw_rectangle, Mapping) or set(raw_rectangle) != {"x", "y", "width", "height"}:
        raise Stage0CapsuleError(f"frame {frame_id} checkerboard rectangle is invalid")
    rectangle = {name: _finite(raw_rectangle[name], f"frame {frame_id} checkerboard rectangle.{name}") for name in raw_rectangle}
    if not 0.0 <= rectangle["x"] < 1.0 or not 0.0 <= rectangle["y"] < 1.0:
        raise Stage0CapsuleError(f"frame {frame_id} checkerboard rectangle origin is invalid")
    if not 0.0 < rectangle["width"] <= 1.0 or not 0.0 < rectangle["height"] <= 1.0:
        raise Stage0CapsuleError(f"frame {frame_id} checkerboard rectangle size is invalid")
    if rectangle["x"] + rectangle["width"] > 1.0 or rectangle["y"] + rectangle["height"] > 1.0:
        raise Stage0CapsuleError(f"frame {frame_id} checkerboard rectangle exceeds bounds")
    tile_size = _plain_int(value.get("tile_size_pixels"), f"frame {frame_id} checkerboard tile_size_pixels")
    if not 2 <= tile_size <= 128:
        raise Stage0CapsuleError(f"frame {frame_id} checkerboard tile_size_pixels is invalid")
    fills = value.get("fills")
    if not isinstance(fills, list) or len(fills) != 2 or any(not isinstance(fill, list) or len(fill) != 3 for fill in fills):
        raise Stage0CapsuleError(f"frame {frame_id} checkerboard fills are invalid")
    normalized_fills = [[_finite(component, f"frame {frame_id} checkerboard fill") for component in fill] for fill in fills]
    if any(component < -1.0 or component > 1.0 for fill in normalized_fills for component in fill):
        raise Stage0CapsuleError(f"frame {frame_id} checkerboard fills must be normalized to [-1, 1]")
    if normalized_fills[0] == normalized_fills[1]:
        raise Stage0CapsuleError(f"frame {frame_id} checkerboard fills must differ")
    return {
        "type": "checkerboard_occlusion",
        "coordinate_reference": COORDINATE_REFERENCE,
        "rectangle": rectangle,
        "tile_size_pixels": tile_size,
        "fills": normalized_fills,
    }


def _hashed_tile_transform(value: Mapping[str, Any], *, frame_id: int) -> dict[str, Any]:
    required = {"type", "coordinate_reference", "rectangle", "tile_size_pixels", "fills", "seed"}
    if set(value) != required or value.get("type") != "hashed_tile_occlusion":
        raise Stage0CapsuleError(f"frame {frame_id} has invalid hashed-tile transform fields")
    if value.get("coordinate_reference") != COORDINATE_REFERENCE:
        raise Stage0CapsuleError(f"frame {frame_id} hashed-tile coordinate reference differs")
    raw_rectangle = value.get("rectangle")
    if not isinstance(raw_rectangle, Mapping) or set(raw_rectangle) != {"x", "y", "width", "height"}:
        raise Stage0CapsuleError(f"frame {frame_id} hashed-tile rectangle is invalid")
    rectangle = {name: _finite(raw_rectangle[name], f"frame {frame_id} hashed-tile rectangle.{name}") for name in raw_rectangle}
    if not 0.0 <= rectangle["x"] < 1.0 or not 0.0 <= rectangle["y"] < 1.0 or not 0.0 < rectangle["width"] <= 1.0 or not 0.0 < rectangle["height"] <= 1.0 or rectangle["x"] + rectangle["width"] > 1.0 or rectangle["y"] + rectangle["height"] > 1.0:
        raise Stage0CapsuleError(f"frame {frame_id} hashed-tile rectangle is invalid")
    tile_size = _plain_int(value.get("tile_size_pixels"), f"frame {frame_id} hashed-tile tile_size_pixels")
    seed = _plain_int(value.get("seed"), f"frame {frame_id} hashed-tile seed")
    if not 2 <= tile_size <= 128 or not 0 <= seed <= 2**31 - 1:
        raise Stage0CapsuleError(f"frame {frame_id} hashed-tile tile size or seed is invalid")
    fills = value.get("fills")
    if not isinstance(fills, list) or len(fills) != 2 or any(not isinstance(fill, list) or len(fill) != 3 for fill in fills):
        raise Stage0CapsuleError(f"frame {frame_id} hashed-tile fills are invalid")
    normalized_fills = [[_finite(component, f"frame {frame_id} hashed-tile fill") for component in fill] for fill in fills]
    if any(component < -1.0 or component > 1.0 for fill in normalized_fills for component in fill) or normalized_fills[0] == normalized_fills[1]:
        raise Stage0CapsuleError(f"frame {frame_id} hashed-tile fills are invalid")
    return {"type": "hashed_tile_occlusion", "coordinate_reference": COORDINATE_REFERENCE, "rectangle": rectangle, "tile_size_pixels": tile_size, "fills": normalized_fills, "seed": seed}


def _reorder_transform(value: Mapping[str, Any], *, frame_id: int) -> dict[str, Any]:
    required = {"type", "expected_source_index", "replacement_source_index"}
    if set(value) != required or value.get("type") != "temporal_reorder":
        raise Stage0CapsuleError(f"frame {frame_id} has invalid temporal reorder fields")
    expected = _plain_int(value.get("expected_source_index"), f"frame {frame_id} expected_source_index")
    replacement = _plain_int(value.get("replacement_source_index"), f"frame {frame_id} replacement_source_index")
    if expected < 0 or replacement < 0 or expected == replacement:
        raise Stage0CapsuleError(f"frame {frame_id} temporal reorder indices are invalid")
    return {"type": "temporal_reorder", "expected_source_index": expected, "replacement_source_index": replacement}


def _transforms(value: Any, *, frame_id: int) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise Stage0CapsuleError(f"frame {frame_id} transforms must be a list")
    transforms: list[Mapping[str, Any]] = []
    for transform in value:
        if not isinstance(transform, Mapping) or transform.get("type") not in TRANSFORM_TYPES:
            raise Stage0CapsuleError(f"frame {frame_id} has unsupported transform")
        if transform["type"] == "rectangle_occlusion":
            parsed = _rectangle_transform(transform, frame_id=frame_id)
        elif transform["type"] == "checkerboard_occlusion":
            parsed = _checkerboard_transform(transform, frame_id=frame_id)
        elif transform["type"] == "hashed_tile_occlusion":
            parsed = _hashed_tile_transform(transform, frame_id=frame_id)
        else:
            parsed = _reorder_transform(transform, frame_id=frame_id)
        transforms.append(parsed)
    if len({item["type"] for item in transforms}) != len(transforms):
        raise Stage0CapsuleError(f"frame {frame_id} repeats a transform type")
    return tuple(transforms)


def load_stage0_capsule(path: str | os.PathLike[str]) -> Stage0Capsule:
    """Validate one frozen Stage-0 capsule and rehash all raw RGB frames."""

    source = Path(path).resolve(strict=True)
    _regular_read_only(source, name="Stage-0 capsule")
    payload = _strict_json(source)
    required = {
        "schema_version",
        "capsule_id",
        "source_sequence",
        "source_group",
        "recipe_id",
        "event",
        "frames",
        "online_evidence_contract",
    }
    if set(payload) != required or payload.get("schema_version") != SCHEMA_VERSION:
        raise Stage0CapsuleError("capsule schema differs")
    if payload.get("online_evidence_contract") != "current_frame_and_strict_prefix_only":
        raise Stage0CapsuleError("capsule online evidence contract differs")
    event_raw = payload.get("event")
    if not isinstance(event_raw, Mapping) or set(event_raw) != {
        "event_id", "cause", "start_frame", "end_frame", "coverage_expectation", "label_provenance"
    } or event_raw.get("label_provenance") != "pre_forward_deterministic_recipe":
        raise Stage0CapsuleError("capsule event schema differs")
    event = Stage0Event(
        event_id=event_raw["event_id"],
        cause=event_raw["cause"],
        start_frame=_plain_int(event_raw["start_frame"], "event start_frame"),
        end_frame=_plain_int(event_raw["end_frame"], "event end_frame"),
        coverage_expectation=event_raw["coverage_expectation"],
    )
    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list):
        raise Stage0CapsuleError("capsule frames must be a list")
    frames: list[Stage0Frame] = []
    expected_ids: list[int] = []
    for index, raw in enumerate(raw_frames):
        required_frame = {"frame_id", "path", "sha256", "timestamp", "transforms"}
        if not isinstance(raw, Mapping) or set(raw) != required_frame:
            raise Stage0CapsuleError(f"frame {index} schema differs")
        frame_id = _plain_int(raw["frame_id"], f"frame {index} frame_id")
        expected_ids.append(frame_id)
        if not isinstance(raw["path"], str) or not raw["path"]:
            raise Stage0CapsuleError(f"frame {index} path is invalid")
        image = Path(raw["path"]).resolve(strict=True)
        _regular_read_only(image, name=f"frame {index} RGB")
        digest = raw["sha256"]
        if not isinstance(digest, str) or _sha256(image) != digest:
            raise Stage0CapsuleError(f"frame {index} RGB changed from capsule binding")
        frames.append(
            Stage0Frame(
                frame_id=frame_id,
                path=image,
                sha256=digest,
                timestamp=_finite(raw["timestamp"], f"frame {index} timestamp"),
                transforms=_transforms(raw["transforms"], frame_id=frame_id),
            )
        )
    if expected_ids != list(range(len(frames))):
        raise Stage0CapsuleError("capsule frame IDs must be contiguous")
    return Stage0Capsule(
        path=source,
        capsule_id=payload["capsule_id"],
        source_sequence=payload["source_sequence"],
        source_group=payload["source_group"],
        recipe_id=payload["recipe_id"],
        event=event,
        frames=tuple(frames),
    )


def apply_stage0_transforms(views: Sequence[Mapping[str, Any]], capsule: Stage0Capsule) -> list[dict[str, Any]]:
    """Copy model-ready views and apply only the capsule's deferred transforms.

    Temporal reordering is represented by the already-final path order and
    therefore has no pixel mutation.  This function intentionally accepts
    tensor-like values without importing Torch.
    """

    if len(views) != len(capsule.frames):
        raise Stage0CapsuleError("loaded views do not match capsule frame count")
    output: list[dict[str, Any]] = []
    for frame, source in zip(capsule.frames, views, strict=True):
        if not isinstance(source, Mapping) or "img" not in source:
            raise Stage0CapsuleError(f"loaded frame {frame.frame_id} lacks img")
        copied = dict(source)
        image = source["img"]
        clone = getattr(image, "clone", None)
        image_copy = clone() if callable(clone) else np.array(image, copy=True)
        for transform in frame.transforms:
            if transform["type"] == "temporal_reorder":
                continue
            if getattr(image_copy, "ndim", None) != 4 or image_copy.shape[0] != 1 or image_copy.shape[1] != 3:
                raise Stage0CapsuleError("model-ready image must have shape (1, 3, H, W)")
            _, _, height, width = image_copy.shape
            rectangle = transform["rectangle"]
            x0 = max(0, min(width, math.floor(rectangle["x"] * width)))
            x1 = max(x0 + 1, min(width, math.ceil((rectangle["x"] + rectangle["width"]) * width)))
            y0 = max(0, min(height, math.floor(rectangle["y"] * height)))
            y1 = max(y0 + 1, min(height, math.ceil((rectangle["y"] + rectangle["height"]) * height)))
            if transform["type"] == "rectangle_occlusion":
                fill = transform["fill"]
                image_copy[:, 0, y0:y1, x0:x1] = fill[0]
                image_copy[:, 1, y0:y1, x0:x1] = fill[1]
                image_copy[:, 2, y0:y1, x0:x1] = fill[2]
                continue
            tile_size = transform["tile_size_pixels"]
            fills = transform["fills"]
            for tile_y, top in enumerate(range(y0, y1, tile_size)):
                bottom = min(top + tile_size, y1)
                for tile_x, left in enumerate(range(x0, x1, tile_size)):
                    right = min(left + tile_size, x1)
                    if transform["type"] == "checkerboard_occlusion":
                        fill = fills[(tile_x + tile_y) % 2]
                    else:
                        mixed = (int(transform["seed"]) ^ ((tile_x + 1) * 0x9E3779B1) ^ ((tile_y + 1) * 0x85EBCA77)) & 0xFFFFFFFF
                        mixed = ((mixed ^ (mixed >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
                        mixed = ((mixed ^ (mixed >> 15)) * 0x846CA68B) & 0xFFFFFFFF
                        fill = fills[(mixed ^ (mixed >> 16)) & 1]
                    image_copy[:, 0, top:bottom, left:right] = fill[0]
                    image_copy[:, 1, top:bottom, left:right] = fill[1]
                    image_copy[:, 2, top:bottom, left:right] = fill[2]
        copied["img"] = image_copy
        output.append(copied)
    return output


__all__ = [
    "CAUSES",
    "COORDINATE_REFERENCE",
    "SCHEMA_VERSION",
    "Stage0Capsule",
    "Stage0CapsuleError",
    "Stage0Event",
    "Stage0Frame",
    "apply_stage0_transforms",
    "load_stage0_capsule",
]
