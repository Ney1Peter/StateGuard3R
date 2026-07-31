"""Manifest-only corruption protocol for read-only image sequences.

The protocol never copies or edits image files.  It reads a JSON source
manifest, emits a new JSON manifest that references the same paths, and records
deferred transforms for consumers to apply in memory.

Source manifests use either ``frames`` or ``frame_paths``.  Each entry may be a
path string or an object with a non-empty ``path`` field::

    {
      "sequence": "clean_sequence",
      "frames": [
        {"path": "rgb/000000.png", "timestamp": 0.0},
        {"path": "rgb/000001.png", "timestamp": 0.1}
      ]
    }

Corruption intervals are zero-based and inclusive.  Output labels use the
plan's canonical ``start_frame``/``end_frame`` names and also include
``start``/``end`` compatibility aliases.  All randomness comes from the
recorded top-level seed and isolated per-corruption random generators.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "stateguard3r.corruption.v0"
DEFAULT_SEED = 0
RECTANGLE_COORDINATE_REFERENCE = "model_input_after_resize_and_center_crop"
SUPPORTED_CORRUPTIONS = (
    "low_overlap_jump",
    "dynamic_occlusion",
    "wrong_order_segment",
)

EXPECTED_EFFECTS = {
    "low_overlap_jump": "pose drift and local geometry inconsistency",
    "dynamic_occlusion": "static-state contamination and ghost geometry",
    "wrong_order_segment": (
        "temporal inconsistency and recurrent-state update instability"
    ),
}


class CorruptionManifestError(ValueError):
    """Raised when a source manifest or corruption specification is invalid."""


def _require_plain_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise CorruptionManifestError(f"{name} must be an integer")
    return value


def _require_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CorruptionManifestError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CorruptionManifestError(f"{name} must be a finite number")
    return result


def _extract_spec_index(spec: Mapping[str, Any], key: str) -> int:
    alias = f"{key}_frame"
    has_key = key in spec
    has_alias = alias in spec
    if not has_key and not has_alias:
        raise CorruptionManifestError(f"corruption specification is missing '{key}'")
    if has_key and has_alias and spec[key] != spec[alias]:
        raise CorruptionManifestError(f"'{key}' and '{alias}' must agree")
    return _require_plain_int(spec[key] if has_key else spec[alias], key)


def _normalise_source_frames(
    source_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if "frames" in source_manifest and "frame_paths" in source_manifest:
        raise CorruptionManifestError(
            "source manifest must use only one of 'frames' or 'frame_paths'"
        )
    key = "frames" if "frames" in source_manifest else "frame_paths"
    if key not in source_manifest:
        raise CorruptionManifestError(
            "source manifest must contain a 'frames' or 'frame_paths' list"
        )
    raw_frames = source_manifest[key]
    if not isinstance(raw_frames, list) or not raw_frames:
        raise CorruptionManifestError(f"source manifest '{key}' must be a non-empty list")

    frames: list[dict[str, Any]] = []
    for index, entry in enumerate(raw_frames):
        if isinstance(entry, str):
            path = entry
            metadata: dict[str, Any] = {}
        elif isinstance(entry, Mapping):
            path = entry.get("path")
            metadata = copy.deepcopy(
                {name: value for name, value in entry.items() if name != "path"}
            )
        else:
            raise CorruptionManifestError(
                f"frame {index} must be a path string or an object with a path"
            )
        if not isinstance(path, str) or not path.strip():
            raise CorruptionManifestError(f"frame {index} has an invalid path")
        if "\x00" in path:
            raise CorruptionManifestError(f"frame {index} path contains a NUL byte")
        frames.append({"path": path, "metadata": metadata})
    return frames


def _validate_source_paths(
    frames: Sequence[Mapping[str, Any]], base_dir: Path
) -> None:
    for index, frame in enumerate(frames):
        candidate = Path(str(frame["path"]))
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        if not candidate.is_file():
            raise CorruptionManifestError(
                f"frame {index} path does not exist or is not a file: {candidate}"
            )


def _resolve_source_ground_truth(
    source_manifest: Mapping[str, Any], base_dir: Path
) -> tuple[str | None, Path | None]:
    """Return the source GT spelling and its path without modifying either."""

    raw_path = source_manifest.get("ground_truth_path")
    if raw_path is None:
        return None, None
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise CorruptionManifestError(
            "source manifest ground_truth_path must be a non-empty string"
        )
    if "\x00" in raw_path:
        raise CorruptionManifestError(
            "source manifest ground_truth_path contains a NUL byte"
        )
    resolved_path = Path(raw_path)
    if not resolved_path.is_absolute():
        resolved_path = base_dir / resolved_path
    return raw_path, resolved_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_frame(
    source_frame: Mapping[str, Any], output_index: int, source_index: int
) -> dict[str, Any]:
    return {
        "frame_index": output_index,
        "source_index": source_index,
        "path": source_frame["path"],
        "metadata": copy.deepcopy(source_frame["metadata"]),
        "transforms": [],
    }


def _normalise_specs(
    corruption_specs: Sequence[Mapping[str, Any]], frame_count: int
) -> list[dict[str, Any]]:
    if isinstance(corruption_specs, (str, bytes)) or not isinstance(
        corruption_specs, Sequence
    ):
        raise CorruptionManifestError("corruption_specs must be a non-empty sequence")
    if not corruption_specs:
        raise CorruptionManifestError("at least one corruption specification is required")

    normalised: list[dict[str, Any]] = []
    occupied: dict[int, int] = {}
    for spec_index, raw_spec in enumerate(corruption_specs):
        if not isinstance(raw_spec, Mapping):
            raise CorruptionManifestError(
                f"corruption specification {spec_index} must be an object"
            )
        corruption_type = raw_spec.get("type")
        if corruption_type not in SUPPORTED_CORRUPTIONS:
            supported = ", ".join(SUPPORTED_CORRUPTIONS)
            raise CorruptionManifestError(
                f"unsupported corruption type {corruption_type!r}; choose from {supported}"
            )
        start = _extract_spec_index(raw_spec, "start")
        end = _extract_spec_index(raw_spec, "end")
        if start < 0 or end < start or end >= frame_count:
            raise CorruptionManifestError(
                f"invalid inclusive range [{start}, {end}] for {frame_count} frames"
            )
        for frame_index in range(start, end + 1):
            if frame_index in occupied:
                raise CorruptionManifestError(
                    "corruption target ranges must not overlap: "
                    f"specifications {occupied[frame_index]} and {spec_index} both "
                    f"target frame {frame_index}"
                )
            occupied[frame_index] = spec_index

        raw_parameters = raw_spec.get("parameters", {})
        if not isinstance(raw_parameters, Mapping):
            raise CorruptionManifestError("corruption parameters must be an object")
        expected_effect = raw_spec.get(
            "expected_effect", EXPECTED_EFFECTS[corruption_type]
        )
        if not isinstance(expected_effect, str) or not expected_effect.strip():
            raise CorruptionManifestError("expected_effect must be a non-empty string")
        normalised.append(
            {
                "type": corruption_type,
                "start": start,
                "end": end,
                "parameters": copy.deepcopy(dict(raw_parameters)),
                "expected_effect": expected_effect,
            }
        )
    return normalised


def _derived_rng(seed: int, spec_index: int, corruption_type: str) -> random.Random:
    material = f"{seed}:{spec_index}:{corruption_type}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(material).digest(), "big")
    return random.Random(derived_seed)


def _reject_unknown_parameters(
    parameters: Mapping[str, Any], allowed: set[str], corruption_type: str
) -> None:
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise CorruptionManifestError(
            f"unknown {corruption_type} parameter(s): {', '.join(unknown)}"
        )


def _apply_low_overlap_jump(
    frames: list[dict[str, Any]],
    source_frames: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    start = spec["start"]
    end = spec["end"]
    parameters = copy.deepcopy(spec["parameters"])
    _reject_unknown_parameters(parameters, {"source_start"}, "low_overlap_jump")

    length = end - start + 1
    target_indices = set(range(start, end + 1))
    source_start = parameters.get("source_start")
    selection = "explicit"
    if source_start is None:
        candidates = [
            candidate
            for candidate in range(0, len(source_frames) - length + 1)
            if target_indices.isdisjoint(range(candidate, candidate + length))
        ]
        if not candidates:
            raise CorruptionManifestError(
                "low_overlap_jump needs a disjoint source segment of equal length"
            )
        greatest_distance = max(abs(candidate - start) for candidate in candidates)
        farthest = [
            candidate
            for candidate in candidates
            if abs(candidate - start) == greatest_distance
        ]
        source_start = rng.choice(farthest)
        selection = "seeded_farthest_disjoint_index_proxy"
    else:
        source_start = _require_plain_int(source_start, "source_start")

    source_end = source_start + length - 1
    if source_start < 0 or source_end >= len(source_frames):
        raise CorruptionManifestError(
            f"low_overlap_jump source range [{source_start}, {source_end}] is out of bounds"
        )
    if not target_indices.isdisjoint(range(source_start, source_end + 1)):
        raise CorruptionManifestError(
            "low_overlap_jump source segment must not overlap its target segment"
        )

    source_indices = list(range(source_start, source_end + 1))
    for target_index, replacement_index in zip(
        range(start, end + 1), source_indices, strict=True
    ):
        frame = _output_frame(
            source_frames[replacement_index], target_index, replacement_index
        )
        frame["transforms"].append(
            {
                "type": "source_frame_substitution",
                "corruption": "low_overlap_jump",
                "original_source_index": target_index,
                "replacement_source_index": replacement_index,
            }
        )
        frames[target_index] = frame

    return {
        "source_start": source_start,
        "source_end": source_end,
        "source_indices": source_indices,
        "selection": selection,
    }


def _parse_rectangle(
    value: Any, rng: random.Random
) -> tuple[dict[str, float], bool]:
    generated = value is None
    if value is None:
        width = 0.25
        height = 0.25
        x = rng.uniform(0.0, 1.0 - width)
        y = rng.uniform(0.0, 1.0 - height)
    else:
        if not isinstance(value, Mapping):
            raise CorruptionManifestError(
                "dynamic_occlusion rectangle must be an object with x, y, width, height"
            )
        required = {"x", "y", "width", "height"}
        if set(value) != required:
            raise CorruptionManifestError(
                "dynamic_occlusion rectangle must contain exactly x, y, width, height"
            )
        x = _require_finite_number(value["x"], "rectangle.x")
        y = _require_finite_number(value["y"], "rectangle.y")
        width = _require_finite_number(value["width"], "rectangle.width")
        height = _require_finite_number(value["height"], "rectangle.height")

    if width <= 0.0 or height <= 0.0 or width > 1.0 or height > 1.0:
        raise CorruptionManifestError(
            "rectangle width and height must be in the interval (0, 1]"
        )
    if x < 0.0 or y < 0.0 or x + width > 1.0 or y + height > 1.0:
        raise CorruptionManifestError("rectangle must fit inside normalized image bounds")
    return (
        {
            "x": round(x, 8),
            "y": round(y, 8),
            "width": round(width, 8),
            "height": round(height, 8),
        },
        generated,
    )


def _parse_velocity(value: Any, rng: random.Random) -> tuple[dict[str, float], bool]:
    generated = value is None
    if value is None:
        dx = rng.choice((-1.0, 1.0)) * rng.uniform(0.02, 0.06)
        dy = rng.choice((-1.0, 1.0)) * rng.uniform(0.02, 0.06)
    else:
        if not isinstance(value, Mapping) or set(value) != {"dx", "dy"}:
            raise CorruptionManifestError(
                "dynamic_occlusion velocity must contain exactly dx and dy"
            )
        dx = _require_finite_number(value["dx"], "velocity.dx")
        dy = _require_finite_number(value["dy"], "velocity.dy")
    return {"dx": round(dx, 8), "dy": round(dy, 8)}, generated


def _parse_rgb_fill(value: Any) -> list[int | float]:
    if not isinstance(value, list) or len(value) != 3:
        raise CorruptionManifestError(
            "dynamic_occlusion fill must be an RGB list with three channels"
        )
    channels: list[int | float] = []
    for index, raw_channel in enumerate(value):
        channel = _require_finite_number(raw_channel, f"fill[{index}]")
        if channel < 0.0 or channel > 255.0:
            raise CorruptionManifestError(
                "dynamic_occlusion RGB fill channels must be in [0, 255]"
            )
        channels.append(raw_channel if isinstance(raw_channel, int) else channel)
    return channels


def _apply_dynamic_occlusion(
    frames: list[dict[str, Any]],
    spec: Mapping[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    start = spec["start"]
    end = spec["end"]
    parameters = copy.deepcopy(spec["parameters"])
    _reject_unknown_parameters(
        parameters,
        {"rectangle", "velocity", "fill", "coordinate_space"},
        "dynamic_occlusion",
    )
    coordinate_space = parameters.get("coordinate_space", "normalized")
    if coordinate_space != "normalized":
        raise CorruptionManifestError(
            "dynamic_occlusion v0 only supports normalized rectangle coordinates"
        )
    rectangle, rectangle_generated = _parse_rectangle(
        parameters.get("rectangle"), rng
    )
    velocity, velocity_generated = _parse_velocity(parameters.get("velocity"), rng)
    fill = _parse_rgb_fill(parameters.get("fill", [0, 0, 0]))

    frame_rectangles: list[dict[str, float]] = []
    for offset, frame_index in enumerate(range(start, end + 1)):
        x = min(
            max(rectangle["x"] + offset * velocity["dx"], 0.0),
            1.0 - rectangle["width"],
        )
        y = min(
            max(rectangle["y"] + offset * velocity["dy"], 0.0),
            1.0 - rectangle["height"],
        )
        exact_rectangle = {
            "x": round(x, 8),
            "y": round(y, 8),
            "width": rectangle["width"],
            "height": rectangle["height"],
        }
        frame_rectangles.append(exact_rectangle)
        frames[frame_index]["transforms"].append(
            {
                "type": "rectangle_occlusion",
                "coordinate_space": "normalized",
                "coordinate_reference": RECTANGLE_COORDINATE_REFERENCE,
                "rectangle": copy.deepcopy(exact_rectangle),
                "fill": copy.deepcopy(fill),
            }
        )

    return {
        "coordinate_space": "normalized",
        "coordinate_reference": RECTANGLE_COORDINATE_REFERENCE,
        "initial_rectangle": rectangle,
        "velocity": velocity,
        "fill": fill,
        "frame_rectangles": frame_rectangles,
        "rectangle_generated_from_seed": rectangle_generated,
        "velocity_generated_from_seed": velocity_generated,
    }


def _apply_wrong_order_segment(
    frames: list[dict[str, Any]],
    source_frames: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    start = spec["start"]
    end = spec["end"]
    parameters = copy.deepcopy(spec["parameters"])
    _reject_unknown_parameters(parameters, {"mode"}, "wrong_order_segment")
    if end - start + 1 < 2:
        raise CorruptionManifestError(
            "wrong_order_segment must contain at least two frames"
        )

    mode = parameters.get("mode", "shuffle")
    if mode not in {"shuffle", "reverse"}:
        raise CorruptionManifestError(
            "wrong_order_segment mode must be 'shuffle' or 'reverse'"
        )
    original_order = list(range(start, end + 1))
    permutation = original_order.copy()
    if mode == "reverse":
        permutation.reverse()
    else:
        rng.shuffle(permutation)
        if permutation == original_order:
            permutation = permutation[1:] + permutation[:1]

    for target_index, replacement_index in zip(
        original_order, permutation, strict=True
    ):
        frame = _output_frame(
            source_frames[replacement_index], target_index, replacement_index
        )
        frame["transforms"].append(
            {
                "type": "temporal_reorder",
                "corruption": "wrong_order_segment",
                "original_source_index": target_index,
                "replacement_source_index": replacement_index,
            }
        )
        frames[target_index] = frame

    return {
        "mode": mode,
        "permutation": permutation,
        "relative_permutation": [index - start for index in permutation],
    }


def build_corruption_manifest(
    source_manifest: Mapping[str, Any],
    corruption_specs: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    check_paths: bool = False,
    source_base_dir: str | os.PathLike[str] | None = None,
    source_manifest_path: str | None = None,
    source_manifest_sha256: str | None = None,
    source_ground_truth_resolved_path: str | None = None,
    source_ground_truth_sha256: str | None = None,
    sequence: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic corruption manifest without touching source images.

    ``source_manifest`` is treated as immutable.  Relative frame paths are only
    resolved against ``source_base_dir`` when ``check_paths`` is true; their JSON
    spelling is preserved in the output.
    """

    if not isinstance(source_manifest, Mapping):
        raise CorruptionManifestError("source manifest must be a JSON object")
    seed = _require_plain_int(seed, "seed")
    source_frames = _normalise_source_frames(source_manifest)
    base_dir = Path(source_base_dir) if source_base_dir is not None else Path.cwd()
    source_ground_truth_path, ground_truth_path = _resolve_source_ground_truth(
        source_manifest, base_dir
    )
    if (
        ground_truth_path is not None
        and source_ground_truth_resolved_path is None
    ):
        source_ground_truth_resolved_path = str(
            ground_truth_path.resolve(strict=False)
        )
    if check_paths:
        _validate_source_paths(source_frames, base_dir)
        if ground_truth_path is not None and not ground_truth_path.is_file():
            raise CorruptionManifestError(
                "source ground_truth_path does not exist or is not a file: "
                f"{ground_truth_path}"
            )

    specs = _normalise_specs(corruption_specs, len(source_frames))
    frames = [
        _output_frame(frame, frame_index, frame_index)
        for frame_index, frame in enumerate(source_frames)
    ]
    labels: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(specs):
        corruption_type = spec["type"]
        rng = _derived_rng(seed, spec_index, corruption_type)
        if corruption_type == "low_overlap_jump":
            exact_parameters = _apply_low_overlap_jump(
                frames, source_frames, spec, rng
            )
        elif corruption_type == "dynamic_occlusion":
            exact_parameters = _apply_dynamic_occlusion(frames, spec, rng)
        else:
            exact_parameters = _apply_wrong_order_segment(
                frames, source_frames, spec, rng
            )
        labels.append(
            {
                "type": corruption_type,
                "start_frame": spec["start"],
                "end_frame": spec["end"],
                "start": spec["start"],
                "end": spec["end"],
                "parameters": exact_parameters,
                "expected_effect": spec["expected_effect"],
            }
        )

    source_sequence = source_manifest.get("sequence", source_manifest.get("name"))
    if source_sequence is None:
        source_sequence = "source"
    if not isinstance(source_sequence, str) or not source_sequence.strip():
        raise CorruptionManifestError(
            "source manifest sequence/name must be a non-empty string"
        )
    output_sequence = sequence or (
        source_sequence + "__" + "__".join(label["type"] for label in labels)
    )
    if not isinstance(output_sequence, str) or not output_sequence.strip():
        raise CorruptionManifestError("output sequence must be a non-empty string")

    return {
        "schema_version": SCHEMA_VERSION,
        "sequence": output_sequence,
        "source_sequence": source_sequence,
        "source_manifest": source_manifest_path,
        "source_manifest_sha256": source_manifest_sha256,
        "source_ground_truth_path": source_ground_truth_path,
        "source_ground_truth_resolved_path": source_ground_truth_resolved_path,
        "source_ground_truth_sha256": source_ground_truth_sha256,
        "source_is_read_only": True,
        "source_frame_count": len(source_frames),
        "frame_count": len(frames),
        "seed": seed,
        "index_convention": "zero_based_inclusive",
        "materialization": "deferred_transforms_no_image_copy",
        "frames": frames,
        "corruptions": labels,
    }


def _paths_alias(left: Path, right: Path) -> bool:
    """Return whether two paths identify the same target, including hard links."""

    try:
        if left.resolve(strict=False) == right.resolve(strict=False):
            return True
    except (OSError, RuntimeError):
        if os.path.abspath(left) == os.path.abspath(right):
            return True
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def _write_json_atomic(
    path: Path, payload: Mapping[str, Any], *, overwrite: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(payload, temporary, indent=2, ensure_ascii=False, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        if overwrite:
            os.replace(temporary_name, path)
        else:
            try:
                # The hard-link publication is atomic and, unlike rename/replace,
                # cannot clobber a file created after the earlier safety check.
                os.link(temporary_name, path)
            except FileExistsError as error:
                raise CorruptionManifestError(
                    f"output manifest already exists: {path}; use overwrite=True "
                    "or --overwrite only after verifying the target"
                ) from error
            Path(temporary_name).unlink()
            temporary_name = None
    except Exception:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        raise


def generate_corruption_manifest(
    source_manifest_path: str | os.PathLike[str],
    output_manifest_path: str | os.PathLike[str],
    corruption_specs: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    check_paths: bool = False,
    sequence: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Load, transform, and atomically write a corruption manifest."""

    source_path = Path(source_manifest_path)
    output_path = Path(output_manifest_path)
    if _paths_alias(source_path, output_path):
        raise CorruptionManifestError(
            "output manifest must not overwrite the read-only source manifest"
        )
    source_bytes = source_path.read_bytes()
    try:
        source_manifest = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorruptionManifestError(
            f"source manifest is not valid UTF-8 JSON: {source_path}"
        ) from error
    if not isinstance(source_manifest, Mapping):
        raise CorruptionManifestError("source manifest must be a JSON object")
    source_frames = _normalise_source_frames(source_manifest)
    _, ground_truth_path = _resolve_source_ground_truth(
        source_manifest, source_path.parent
    )
    for index, frame in enumerate(source_frames):
        frame_path = Path(frame["path"])
        if not frame_path.is_absolute():
            frame_path = source_path.parent / frame_path
        if _paths_alias(frame_path, output_path):
            raise CorruptionManifestError(
                "output manifest must not overwrite a read-only source frame: "
                f"frame {index} at {frame_path}"
            )
    if ground_truth_path is not None and _paths_alias(ground_truth_path, output_path):
        raise CorruptionManifestError(
            "output manifest must not overwrite the read-only source ground truth: "
            f"{ground_truth_path}"
        )
    if output_path.exists() and not overwrite:
        raise CorruptionManifestError(
            f"output manifest already exists: {output_path}; use overwrite=True "
            "or --overwrite only after verifying the target"
        )
    source_ground_truth_resolved_path = (
        str(ground_truth_path.resolve(strict=False))
        if ground_truth_path is not None
        else None
    )
    source_ground_truth_sha256 = (
        _sha256_file(ground_truth_path)
        if ground_truth_path is not None and ground_truth_path.is_file()
        else None
    )
    result = build_corruption_manifest(
        source_manifest,
        corruption_specs,
        seed=seed,
        check_paths=check_paths,
        source_base_dir=source_path.parent,
        source_manifest_path=str(source_path.resolve(strict=False)),
        source_manifest_sha256=hashlib.sha256(source_bytes).hexdigest(),
        source_ground_truth_resolved_path=source_ground_truth_resolved_path,
        source_ground_truth_sha256=source_ground_truth_sha256,
        sequence=sequence,
    )
    _write_json_atomic(output_path, result, overwrite=overwrite)
    return result


def _load_corruption_specs(
    parser: argparse.ArgumentParser, spec_values: list[str] | None, config: str | None
) -> list[Mapping[str, Any]]:
    if spec_values is not None:
        specs: list[Mapping[str, Any]] = []
        for value in spec_values:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as error:
                parser.error(f"invalid --spec JSON: {error}")
            if not isinstance(parsed, Mapping):
                parser.error("each --spec value must decode to a JSON object")
            specs.append(parsed)
        return specs

    assert config is not None
    try:
        config_payload = json.loads(Path(config).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"cannot read corruption config: {error}")
    if isinstance(config_payload, Mapping):
        config_payload = config_payload.get("corruptions")
    if not isinstance(config_payload, list) or not all(
        isinstance(spec, Mapping) for spec in config_payload
    ):
        parser.error(
            "config must be a JSON list of specifications or an object with "
            "a 'corruptions' list"
        )
    return config_payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic corruption manifest without copying or editing "
            "source frames. Intervals are zero-based and inclusive."
        )
    )
    parser.add_argument("source_manifest", help="read-only JSON source manifest")
    parser.add_argument("output_manifest", help="new corruption manifest path")
    specifications = parser.add_mutually_exclusive_group(required=True)
    specifications.add_argument(
        "--spec",
        action="append",
        help=(
            "inline JSON corruption object; repeat for multiple non-overlapping "
            "corruptions"
        ),
    )
    specifications.add_argument(
        "--config", help="JSON file containing a list or a corruptions list"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="require every referenced source frame to exist",
    )
    parser.add_argument("--sequence", help="override generated output sequence name")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output manifest after explicitly verifying it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``stateguard3r-corrupt``."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    specs = _load_corruption_specs(parser, args.spec, args.config)
    try:
        result = generate_corruption_manifest(
            args.source_manifest,
            args.output_manifest,
            specs,
            seed=args.seed,
            check_paths=args.check_paths,
            sequence=args.sequence,
            overwrite=args.overwrite,
        )
    except (OSError, CorruptionManifestError) as error:
        parser.error(str(error))
    print(
        f"wrote {args.output_manifest} with {result['frame_count']} frames and "
        f"{len(result['corruptions'])} corruption(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
