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
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "stateguard3r.corruption.v1"
MATERIALIZATION_MODE = "deferred_transforms_no_image_copy"
INDEX_CONVENTION = "zero_based_inclusive"
RECTANGLE_COORDINATE_REFERENCE = "model_input_after_resize_and_center_crop"

CORRUPTION_TO_TRANSFORM = {
    "low_overlap_jump": "source_frame_substitution",
    "dynamic_occlusion": "rectangle_occlusion",
    "wrong_order_segment": "temporal_reorder",
}


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
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InputManifestError(f"{name} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(payload, dict):
        raise InputManifestError(f"{name} must be a JSON object")
    return payload, payload_bytes


def _source_frames(
    payload: Mapping[str, Any], *, base_dir: Path
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
        path = Path(path_text)
        if not path.is_absolute():
            path = base_dir / path
        try:
            path = path.resolve(strict=True)
        except OSError as error:
            raise InputManifestError(
                f"source frame {index} path does not exist: {path}"
            ) from error
        if not path.is_file():
            raise InputManifestError(
                f"source frame {index} path is not a file: {path}"
            )
        result.append((path, metadata))
    return result


def _corruption_types(
    payload: Mapping[str, Any], *, frame_count: int
) -> list[str | None]:
    raw_corruptions = payload.get("corruptions")
    if not isinstance(raw_corruptions, list) or not raw_corruptions:
        raise InputManifestError("corruptions must be a non-empty JSON array")
    expected: list[str | None] = [None] * frame_count
    for index, raw in enumerate(raw_corruptions):
        if not isinstance(raw, Mapping):
            raise InputManifestError(f"corruption {index} must be a JSON object")
        corruption_type = raw.get("type")
        if corruption_type not in CORRUPTION_TO_TRANSFORM:
            raise InputManifestError(
                f"corruption {index} has unsupported type {corruption_type!r}"
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
        for frame_index in range(start, end + 1):
            if expected[frame_index] is not None:
                raise InputManifestError(
                    f"corruption intervals overlap at frame {frame_index}"
                )
            expected[frame_index] = corruption_type
    return expected


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

    source_frames = _source_frames(
        source_payload, base_dir=source_manifest_path.parent
    )
    if len(source_frames) != source_frame_count:
        raise InputManifestError(
            "source_frame_count does not match source manifest frame count"
        )
    expected_corruptions = _corruption_types(payload, frame_count=frame_count)

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
        if expected_transform is None and source_index != frame_index:
            raise InputManifestError(
                f"clean frame {position} must retain its source_index"
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
