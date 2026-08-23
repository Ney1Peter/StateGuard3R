#!/usr/bin/env python3
"""Build fresh, immutable inputs for the independent Stage 0.5 pilot.

The only source data are already-present TUM RGB images.  This builder never
modifies a Stage 0 capsule or consumes an old event response.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
TUM_ROOT = PROJECT_ROOT / "baselines" / "ReCal3R" / "data" / "tum"
OUTPUT_NAME = "state-triage-v2-stage0p5-inputs-0001"
FRAME_COUNT = 30
EVENT_START = 12
EVENT_END = 17
SCHEMA_VERSION = "stateguard3r.state-triage-stage0-capsule.v1"
COORDINATE_REFERENCE = "model_input_after_resize_and_center_crop"

sys.path.insert(0, str(ROOT / "src"))
from stateguard3r.state_triage_capsule_v2 import load_stage0_capsule  # noqa: E402


class BuildStage0p5CapsulesError(ValueError):
    pass


@dataclass(frozen=True)
class Spec:
    cause: str
    source_sequence: str
    source_start: int
    variant: str
    event_start: int = EVENT_START
    event_end: int = EVENT_END

    @property
    def capsule_id(self) -> str:
        return f"{self.cause}-{self.source_sequence.replace('rgbd_dataset_', '')}-{self.source_start:04d}"


# One fresh 30-frame window for every cause x sequence cell.  The variants are
# recipe diversity, fixed before forward; they are not trial choices.
PILOT_SPECS: tuple[Spec, ...] = (
    Spec("normal_novelty", "rgbd_dataset_freiburg2_desk", 520, "identity", 0, 4),
    Spec("normal_novelty", "rgbd_dataset_freiburg3_walking_static", 330, "identity", 0, 4),
    Spec("normal_novelty", "rgbd_dataset_freiburg3_walking_xyz", 100, "identity", 0, 4),
    Spec("registration_or_order_fault", "rgbd_dataset_freiburg2_desk", 900, "reverse"),
    Spec("registration_or_order_fault", "rgbd_dataset_freiburg3_walking_static", 550, "reverse"),
    Spec("registration_or_order_fault", "rgbd_dataset_freiburg3_walking_xyz", 250, "reverse"),
    Spec("bad_observation", "rgbd_dataset_freiburg2_desk", 1600, "grey"),
    Spec("bad_observation", "rgbd_dataset_freiburg3_walking_static", 220, "black"),
    Spec("bad_observation", "rgbd_dataset_freiburg3_walking_xyz", 430, "white"),
    Spec("transient_local_content", "rgbd_dataset_freiburg2_desk", 2350, "red_left_half"),
    Spec("transient_local_content", "rgbd_dataset_freiburg3_walking_static", 700, "green_right_half"),
    Spec("transient_local_content", "rgbd_dataset_freiburg3_walking_xyz", 600, "blue_top_half"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rgb_records(sequence: str) -> list[tuple[float, Path]]:
    root = (TUM_ROOT / sequence).resolve(strict=True)
    listing = root / "rgb.txt"
    if not listing.is_file() or stat.S_IMODE(listing.stat().st_mode) & 0o222:
        raise BuildStage0p5CapsulesError(f"raw RGB listing must be read-only: {listing}")
    records: list[tuple[float, Path]] = []
    for line_number, raw in enumerate(listing.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise BuildStage0p5CapsulesError(f"{listing}:{line_number} is not timestamp/path")
        try:
            timestamp = float(fields[0])
        except ValueError as error:
            raise BuildStage0p5CapsulesError(f"{listing}:{line_number} timestamp is invalid") from error
        image = (root / fields[1]).resolve(strict=True)
        if not image.is_file() or stat.S_IMODE(image.stat().st_mode) & 0o222:
            raise BuildStage0p5CapsulesError(f"raw RGB must be a read-only file: {image}")
        records.append((timestamp, image))
    return records


def _final_indices(spec: Spec) -> list[int]:
    indices = list(range(spec.source_start, spec.source_start + FRAME_COUNT))
    if spec.cause == "registration_or_order_fault":
        indices[spec.event_start : spec.event_end + 1] = list(reversed(indices[spec.event_start : spec.event_end + 1]))
    return indices


def _rectangle(*, x: float, y: float, width: float, height: float, fill: list[float]) -> dict[str, Any]:
    return {
        "type": "rectangle_occlusion",
        "coordinate_reference": COORDINATE_REFERENCE,
        "rectangle": {"x": x, "y": y, "width": width, "height": height},
        "fill": fill,
    }


def _transforms(spec: Spec, frame_id: int) -> list[dict[str, Any]]:
    if not spec.event_start <= frame_id <= spec.event_end:
        return []
    if spec.cause == "registration_or_order_fault":
        expected = spec.source_start + frame_id
        replacement = spec.source_start + spec.event_end - (frame_id - spec.event_start)
        return [{"type": "temporal_reorder", "expected_source_index": expected, "replacement_source_index": replacement}]
    if spec.cause == "bad_observation":
        fills = {"grey": [0.0, 0.0, 0.0], "black": [-1.0, -1.0, -1.0], "white": [1.0, 1.0, 1.0]}
        if spec.variant not in fills:
            raise BuildStage0p5CapsulesError("bad-observation variant is unsupported")
        return [_rectangle(x=0.0, y=0.0, width=1.0, height=1.0, fill=fills[spec.variant])]
    if spec.cause == "transient_local_content":
        variants = {
            "red_left_half": (0.0, 0.0, 0.5, 1.0, [1.0, -1.0, -1.0]),
            "green_right_half": (0.5, 0.0, 0.5, 1.0, [-1.0, 1.0, -1.0]),
            "blue_top_half": (0.0, 0.0, 1.0, 0.5, [-1.0, -1.0, 1.0]),
        }
        if spec.variant not in variants:
            raise BuildStage0p5CapsulesError("transient-content variant is unsupported")
        x, y, width, height, fill = variants[spec.variant]
        return [_rectangle(x=x, y=y, width=width, height=height, fill=fill)]
    if spec.cause == "normal_novelty":
        return []
    raise BuildStage0p5CapsulesError("cause is unsupported")


def _payload(spec: Spec, records: Sequence[tuple[float, Path]]) -> dict[str, Any]:
    indices = _final_indices(spec)
    if min(indices) < 0 or max(indices) >= len(records):
        raise BuildStage0p5CapsulesError(f"{spec.capsule_id} exceeds raw RGB listing")
    coverage = "novel" if spec.cause == "normal_novelty" else "covered"
    return {
        "schema_version": SCHEMA_VERSION,
        "capsule_id": spec.capsule_id,
        "source_sequence": spec.source_sequence,
        "source_group": f"{spec.source_sequence}:{spec.source_start}:{FRAME_COUNT}",
        "recipe_id": f"stage0p5-{spec.cause}-{spec.variant}-v1",
        "event": {
            "event_id": f"event-{spec.capsule_id}",
            "cause": spec.cause,
            "start_frame": spec.event_start,
            "end_frame": spec.event_end,
            "coverage_expectation": coverage,
            "label_provenance": "pre_forward_deterministic_recipe",
        },
        "online_evidence_contract": "current_frame_and_strict_prefix_only",
        "frames": [
            {
                "frame_id": frame_id,
                "path": str(records[source_index][1]),
                "sha256": _sha256(records[source_index][1]),
                "timestamp": records[source_index][0],
                "transforms": _transforms(spec, frame_id),
            }
            for frame_id, source_index in enumerate(indices)
        ],
    }


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _freeze(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def build(output_dir: Path) -> Path:
    output_dir = output_dir.resolve(strict=False)
    if output_dir.parent != ROOT / "outputs" or output_dir.exists():
        raise BuildStage0p5CapsulesError("output must be a new direct child of StateGuard3R/outputs")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".staging", dir=output_dir.parent))
    published = False
    try:
        artifacts: list[dict[str, Any]] = []
        for spec in PILOT_SPECS:
            payload = _payload(spec, _rgb_records(spec.source_sequence))
            path = staging / "evaluation" / f"{spec.capsule_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_canonical(payload))
            path.chmod(0o444)
            capsule = load_stage0_capsule(path)
            artifacts.append(
                {
                    "capsule_id": capsule.capsule_id,
                    "path": str(path.relative_to(staging)),
                    "sha256": _sha256(path),
                    "event": capsule.event.to_dict(),
                    "source_sequence": capsule.source_sequence,
                    "source_group": capsule.source_group,
                    "recipe_variant": spec.variant,
                }
            )
        protocol = {
            "schema_version": "stateguard3r.state-triage-stage0p5-input-inventory.v1",
            "created_at": datetime.now().astimezone().isoformat(),
            "frame_count_per_capsule": FRAME_COUNT,
            "evaluation_status": "independent_development_only_existing_tum_rgb",
            "online_evidence_contract": "current_frame_and_strict_prefix_only",
            "labels": "pre_forward_deterministic_recipe_only",
            "stage0_relation": "fresh_windows_only_not_a_stage0_rerun_or_revision",
            "persistent_change": "excluded",
            "artifacts": artifacts,
        }
        (staging / "protocol.json").write_bytes(_canonical(protocol))
        _freeze(staging)
        os.replace(staging, output_dir)
        published = True
        return output_dir
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / OUTPUT_NAME)
    args = parser.parse_args(argv)
    try:
        print(build(args.output_dir))
    except (BuildStage0p5CapsulesError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
