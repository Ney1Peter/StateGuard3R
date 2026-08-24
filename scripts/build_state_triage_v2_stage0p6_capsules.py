#!/usr/bin/env python3
"""Build the fresh immutable Stage 0.6 local-before-coverage input matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NAME = "state-triage-v2-stage0p6-inputs-0001"
FRAME_COUNT = 30
EVENT_START, EVENT_END = 12, 17
SCHEMA_VERSION = "stateguard3r.state-triage-stage0-capsule.v1"
COORDINATE_REFERENCE = "model_input_after_resize_and_center_crop"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.build_state_triage_v2_stage0p5_capsules import _rgb_records, _sha256  # noqa: E402
from stateguard3r.state_triage_capsule_v2 import load_stage0_capsule  # noqa: E402


class BuildStage0p6CapsulesError(ValueError):
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


PILOT_SPECS: tuple[Spec, ...] = (
    Spec("normal_novelty", "rgbd_dataset_freiburg2_desk", 1200, "identity", 0, 4),
    Spec("normal_novelty", "rgbd_dataset_freiburg3_walking_static", 30, "identity", 0, 4),
    Spec("normal_novelty", "rgbd_dataset_freiburg3_walking_xyz", 20, "identity", 0, 4),
    Spec("registration_or_order_fault", "rgbd_dataset_freiburg2_desk", 300, "reverse"),
    Spec("registration_or_order_fault", "rgbd_dataset_freiburg3_walking_static", 150, "reverse"),
    Spec("registration_or_order_fault", "rgbd_dataset_freiburg3_walking_xyz", 400, "reverse"),
    Spec("bad_observation", "rgbd_dataset_freiburg2_desk", 2000, "violet_grey"),
    Spec("bad_observation", "rgbd_dataset_freiburg3_walking_static", 250, "dark_grey"),
    Spec("bad_observation", "rgbd_dataset_freiburg3_walking_xyz", 470, "light_grey"),
    Spec("transient_local_content", "rgbd_dataset_freiburg2_desk", 2600, "yellow_bottom_half"),
    Spec("transient_local_content", "rgbd_dataset_freiburg3_walking_static", 580, "cyan_left_half"),
    Spec("transient_local_content", "rgbd_dataset_freiburg3_walking_xyz", 750, "magenta_right_half"),
)


def _final_indices(spec: Spec) -> list[int]:
    indices = list(range(spec.source_start, spec.source_start + FRAME_COUNT))
    if spec.cause == "registration_or_order_fault":
        indices[spec.event_start : spec.event_end + 1] = list(reversed(indices[spec.event_start : spec.event_end + 1]))
    return indices


def _rectangle(*, x: float, y: float, width: float, height: float, fill: list[float]) -> dict[str, Any]:
    return {"type": "rectangle_occlusion", "coordinate_reference": COORDINATE_REFERENCE, "rectangle": {"x": x, "y": y, "width": width, "height": height}, "fill": fill}


def _transforms(spec: Spec, frame_id: int) -> list[dict[str, Any]]:
    if not spec.event_start <= frame_id <= spec.event_end:
        return []
    if spec.cause == "registration_or_order_fault":
        return [{"type": "temporal_reorder", "expected_source_index": spec.source_start + frame_id, "replacement_source_index": spec.source_start + spec.event_end - (frame_id - spec.event_start)}]
    if spec.cause == "bad_observation":
        fills = {"violet_grey": [0.25, -0.25, 0.25], "dark_grey": [-0.50, -0.50, -0.50], "light_grey": [0.75, 0.75, 0.75]}
        if spec.variant not in fills:
            raise BuildStage0p6CapsulesError("bad-observation variant is unsupported")
        return [_rectangle(x=0.0, y=0.0, width=1.0, height=1.0, fill=fills[spec.variant])]
    if spec.cause == "transient_local_content":
        variants = {
            "yellow_bottom_half": (0.0, 0.5, 1.0, 0.5, [1.0, 1.0, -1.0]),
            "cyan_left_half": (0.0, 0.0, 0.5, 1.0, [-1.0, 1.0, 1.0]),
            "magenta_right_half": (0.5, 0.0, 0.5, 1.0, [1.0, -1.0, 1.0]),
        }
        if spec.variant not in variants:
            raise BuildStage0p6CapsulesError("transient-content variant is unsupported")
        x, y, width, height, fill = variants[spec.variant]
        return [_rectangle(x=x, y=y, width=width, height=height, fill=fill)]
    if spec.cause == "normal_novelty":
        return []
    raise BuildStage0p6CapsulesError("cause is unsupported")


def _payload(spec: Spec, records: Sequence[tuple[float, Path]]) -> dict[str, Any]:
    indices = _final_indices(spec)
    if min(indices) < 0 or max(indices) >= len(records):
        raise BuildStage0p6CapsulesError(f"{spec.capsule_id} exceeds raw RGB listing")
    return {
        "schema_version": SCHEMA_VERSION,
        "capsule_id": spec.capsule_id,
        "source_sequence": spec.source_sequence,
        "source_group": f"{spec.source_sequence}:{spec.source_start}:{FRAME_COUNT}",
        "recipe_id": f"stage0p6-{spec.cause}-{spec.variant}-v1",
        "event": {"event_id": f"event-{spec.capsule_id}", "cause": spec.cause, "start_frame": spec.event_start, "end_frame": spec.event_end, "coverage_expectation": "novel" if spec.cause == "normal_novelty" else "covered", "label_provenance": "pre_forward_deterministic_recipe"},
        "online_evidence_contract": "current_frame_and_strict_prefix_only",
        "frames": [{"frame_id": frame_id, "path": str(records[source_index][1]), "sha256": _sha256(records[source_index][1]), "timestamp": records[source_index][0], "transforms": _transforms(spec, frame_id)} for frame_id, source_index in enumerate(indices)],
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
        raise BuildStage0p6CapsulesError("output must be a new direct child of StateGuard3R/outputs")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".staging", dir=output_dir.parent))
    published = False
    try:
        artifacts: list[dict[str, Any]] = []
        for spec in PILOT_SPECS:
            path = staging / "evaluation" / f"{spec.capsule_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_canonical(_payload(spec, _rgb_records(spec.source_sequence))))
            path.chmod(0o444)
            capsule = load_stage0_capsule(path)
            artifacts.append({"capsule_id": capsule.capsule_id, "path": str(path.relative_to(staging)), "sha256": _sha256(path), "event": capsule.event.to_dict(), "source_sequence": capsule.source_sequence, "source_group": capsule.source_group, "recipe_variant": spec.variant})
        (staging / "protocol.json").write_bytes(_canonical({"schema_version": "stateguard3r.state-triage-stage0p6-input-inventory.v1", "created_at": datetime.now().astimezone().isoformat(), "frame_count_per_capsule": FRAME_COUNT, "evaluation_status": "independent_development_only_existing_tum_rgb", "online_evidence_contract": "current_frame_and_strict_prefix_only", "labels": "pre_forward_deterministic_recipe_only", "prior_stage_relation": "fresh_windows_only_not_a_stage0_or_stage0p5_rerun_or_revision", "persistent_change": "excluded", "artifacts": artifacts}))
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
    except (BuildStage0p6CapsulesError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
