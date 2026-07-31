#!/usr/bin/env python3
"""Generate tiny deterministic inputs for StateGuard3R CLI smoke tests.

The generated values are deliberately synthetic and must not be reported as a
real model experiment. The source manifest only references the two input files;
it never copies or modifies them, and this script never runs a model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from stateguard3r.health import HealthFrame


FRAME_COUNT = 60
SYNTHETIC_MARKER = "SYNTHETIC_SMOKE_ONLY_NOT_A_REAL_EXPERIMENT"
OUTPUT_FILENAMES = {
    "source_manifest": "source_manifest.json",
    "corruption_specs": "corruption_specs.json",
    "health_jsonl": "health.jsonl",
}

# Intervals are zero-based and inclusive. Five anomalous frames per type leave
# ample stable context for the default trailing detection window.
CORRUPTION_INTERVALS = (
    ("low_overlap_jump", 10, 14),
    ("dynamic_occlusion", 25, 29),
    ("wrong_order_segment", 40, 44),
)

_STABLE_PROFILE = {
    "geometric_residual": 0.01,
    "pose_jump": 0.005,
    "update_magnitude": 0.02,
    "overlap": 0.95,
    "reliability": 0.95,
    "candidate_beta": 0.08,
    "final_beta": 0.04,
}

_ANOMALY_PROFILES = {
    "low_overlap_jump": {
        "geometric_residual": 0.45,
        "pose_jump": 0.80,
        "update_magnitude": 0.85,
        "overlap": 0.10,
        "reliability": 0.20,
        "candidate_beta": 0.90,
        "final_beta": 0.30,
    },
    "dynamic_occlusion": {
        "geometric_residual": 0.70,
        "pose_jump": 0.25,
        "update_magnitude": 0.75,
        "overlap": 0.45,
        "reliability": 0.30,
        "candidate_beta": 0.80,
        "final_beta": 0.25,
    },
    "wrong_order_segment": {
        "geometric_residual": 0.55,
        "pose_jump": 0.95,
        "update_magnitude": 0.90,
        "overlap": 0.25,
        "reliability": 0.15,
        "candidate_beta": 0.95,
        "final_beta": 0.35,
    },
}


class SyntheticSmokeInputError(ValueError):
    """Raised when safe synthetic-smoke generation cannot proceed."""


def _resolve_existing_frame(value: str | os.PathLike[str], name: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise SyntheticSmokeInputError(f"{name} must be an existing file: {path}")
    return path.resolve(strict=True)


def _paths_alias(left: Path, right: Path) -> bool:
    if left == right:
        return True
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _publish_bytes_exclusive(path: Path, payload: bytes) -> None:
    """Atomically publish bytes without replacing an existing output."""

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise SyntheticSmokeInputError(
                f"refusing to replace existing synthetic-smoke output: {path}"
            ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _build_source_manifest(frame_a: Path, frame_b: Path) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    for frame_id in range(FRAME_COUNT):
        # Three-frame blocks make the reversed interval observably different
        # while still using only the two caller-owned files.
        selected = frame_a if (frame_id // 3) % 2 == 0 else frame_b
        frames.append(
            {
                "path": str(selected),
                "timestamp": round(frame_id / 30.0, 9),
                "synthetic": True,
                "not_real_experiment": True,
            }
        )
    return {
        "schema_version": "stateguard3r.synthetic-source.v0",
        "sequence": "synthetic_smoke_not_real_experiment",
        "synthetic": True,
        "not_real_experiment": True,
        "warning": SYNTHETIC_MARKER,
        "frame_count": FRAME_COUNT,
        "frames": frames,
    }


def _build_corruption_config() -> dict[str, Any]:
    return {
        "schema_version": "stateguard3r.synthetic-corruption-config.v0",
        "synthetic": True,
        "not_real_experiment": True,
        "warning": SYNTHETIC_MARKER,
        "recommended_seed": 0,
        "corruptions": [
            {
                "type": "low_overlap_jump",
                "start": 10,
                "end": 14,
                "parameters": {"source_start": 50},
            },
            {
                "type": "dynamic_occlusion",
                "start": 25,
                "end": 29,
                "parameters": {
                    "coordinate_space": "normalized",
                    "rectangle": {
                        "x": 0.25,
                        "y": 0.25,
                        "width": 0.5,
                        "height": 0.5,
                    },
                    "velocity": {"dx": 0.04, "dy": 0.025},
                    "fill": [255, 0, 0],
                },
            },
            {
                "type": "wrong_order_segment",
                "start": 40,
                "end": 44,
                "parameters": {"mode": "reverse"},
            },
        ],
    }


def _corruption_type_for_frame(frame_id: int) -> str | None:
    for corruption_type, start, end in CORRUPTION_INTERVALS:
        if start <= frame_id <= end:
            return corruption_type
    return None


def _build_health_records() -> list[HealthFrame]:
    records: list[HealthFrame] = []
    for frame_id in range(FRAME_COUNT):
        corruption_type = _corruption_type_for_frame(frame_id)
        profile = (
            _STABLE_PROFILE
            if corruption_type is None
            else _ANOMALY_PROFILES[corruption_type]
        )
        label = "stable" if corruption_type is None else corruption_type
        records.append(
            HealthFrame(
                frame_id=frame_id,
                timestamp=round(frame_id / 30.0, 9),
                overlap=profile["overlap"],
                pose_jump=profile["pose_jump"],
                geometric_residual=profile["geometric_residual"],
                update_magnitude=profile["update_magnitude"],
                reliability=profile["reliability"],
                candidate_beta=profile["candidate_beta"],
                final_beta=profile["final_beta"],
                global_state_delta=profile["update_magnitude"],
                local_mem_delta=profile["update_magnitude"] * 0.5,
                decision=f"{SYNTHETIC_MARKER}:{label}",
            )
        )
    return records


def _health_jsonl_bytes(records: Sequence[HealthFrame]) -> bytes:
    lines = [
        json.dumps(
            record.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        for record in records
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def generate_synthetic_smoke_inputs(
    output_dir: str | os.PathLike[str],
    frame_a: str | os.PathLike[str],
    frame_b: str | os.PathLike[str],
) -> dict[str, Path]:
    """Create deterministic manifests and strict health JSONL for CLI smoke tests."""

    resolved_a = _resolve_existing_frame(frame_a, "frame_a")
    resolved_b = _resolve_existing_frame(frame_b, "frame_b")
    if _paths_alias(resolved_a, resolved_b):
        raise SyntheticSmokeInputError("frame_a and frame_b must be distinct files")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        name: destination / filename for name, filename in OUTPUT_FILENAMES.items()
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing:
        formatted = ", ".join(str(path) for path in existing)
        raise SyntheticSmokeInputError(
            "refusing to replace existing synthetic-smoke output(s): " + formatted
        )

    source_manifest = _build_source_manifest(resolved_a, resolved_b)
    corruption_config = _build_corruption_config()
    health_records = _build_health_records()

    _publish_bytes_exclusive(
        outputs["source_manifest"], _json_bytes(source_manifest)
    )
    _publish_bytes_exclusive(
        outputs["corruption_specs"], _json_bytes(corruption_config)
    )
    _publish_bytes_exclusive(
        outputs["health_jsonl"], _health_jsonl_bytes(health_records)
    )
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic synthetic-only inputs for StateGuard3R CLI "
            "smoke tests; this does not run a model or produce experiment results."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-a", type=Path, required=True)
    parser.add_argument("--frame-b", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        outputs = generate_synthetic_smoke_inputs(
            args.output_dir,
            args.frame_a,
            args.frame_b,
        )
    except (OSError, SyntheticSmokeInputError) as error:
        parser.error(str(error))

    print(SYNTHETIC_MARKER)
    print("These files are synthetic CLI smoke inputs, not real experiment results.")
    for name, path in outputs.items():
        print(f"{name}={path.resolve(strict=False)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
