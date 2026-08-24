#!/usr/bin/env python3
"""Build 80 fresh immutable windows for the Stage 0.7 robustness benchmark."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NAME = "state-triage-v2-stage0p7-inputs-0001"
FRAME_COUNT, EVENT_START, EVENT_END = 30, 12, 17
SCHEMA_VERSION = "stateguard3r.state-triage-stage0-capsule.v1"
COORDINATE_REFERENCE = "model_input_after_resize_and_center_crop"
CAUSES = ("normal_novelty", "registration_or_order_fault", "bad_observation", "transient_local_content")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.build_state_triage_v2_stage0_capsules import DRY_RUN_SPECS, EVALUATION_SPECS  # noqa: E402
from scripts.build_state_triage_v2_stage0_controls import CONTROL_SPECS  # noqa: E402
from scripts.build_state_triage_v2_stage0p5_capsules import PILOT_SPECS as STAGE0P5_SPECS, _rgb_records, _sha256  # noqa: E402
from scripts.build_state_triage_v2_stage0p6_capsules import PILOT_SPECS as STAGE0P6_SPECS  # noqa: E402
from stateguard3r.state_triage_capsule_v2 import load_stage0_capsule  # noqa: E402


class BuildStage0p7CapsulesError(ValueError):
    pass


@dataclass(frozen=True)
class Spec:
    cause: str
    source_sequence: str
    source_start: int
    ordinal: int
    event_start: int = EVENT_START
    event_end: int = EVENT_END

    @property
    def capsule_id(self) -> str:
        return f"{self.cause}-{self.source_sequence.replace('rgbd_dataset_', '')}-{self.source_start:04d}"

    @property
    def local_tier(self) -> str:
        if self.cause != "transient_local_content":
            return "not_applicable"
        return "stress_75pct" if self.ordinal % 5 in (0, 1, 2) else "moderate_50pct"


SEQUENCE_ORDER = (
    "rgbd_dataset_freiburg1_desk",
    "rgbd_dataset_freiburg2_desk",
    "rgbd_dataset_freiburg3_walking_static",
    "rgbd_dataset_freiburg3_walking_xyz",
)
QUOTAS: dict[str, dict[str, int]] = {
    "rgbd_dataset_freiburg1_desk": {"normal_novelty": 4, "registration_or_order_fault": 4, "bad_observation": 4, "transient_local_content": 3},
    "rgbd_dataset_freiburg2_desk": {"normal_novelty": 13, "registration_or_order_fault": 14, "bad_observation": 14, "transient_local_content": 13},
    "rgbd_dataset_freiburg3_walking_static": {"normal_novelty": 1, "registration_or_order_fault": 0, "bad_observation": 0, "transient_local_content": 1},
    "rgbd_dataset_freiburg3_walking_xyz": {"normal_novelty": 2, "registration_or_order_fault": 2, "bad_observation": 2, "transient_local_content": 3},
}
# These are precomputed 30-frame raw-list slots disjoint from all earlier
# StateTriage capsule windows.  The f2 selection uses 54 evenly spread slots
# from its 69 admissible positions to avoid a cause/time-block confound.
AVAILABLE_STARTS: dict[str, tuple[int, ...]] = {
    "rgbd_dataset_freiburg1_desk": (0, 60, 90, 120, 150, 180, 270, 300, 330, 360, 390, 480, 510, 540, 570),
    "rgbd_dataset_freiburg2_desk": (0, 30, 120, 150, 180, 210, 240, 270, 330, 420, 450, 480, 570, 600, 630, 660, 750, 780, 810, 840, 870, 930, 960, 990, 1020, 1080, 1110, 1140, 1170, 1230, 1260, 1290, 1320, 1350, 1440, 1470, 1500, 1530, 1560, 1650, 1680, 1710, 1740, 1770, 1830, 1860, 1890, 1920, 1950, 2040, 2070, 2100, 2130, 2160, 2250, 2280, 2310, 2400, 2430, 2460, 2550, 2640, 2670, 2700, 2790, 2820, 2850, 2880, 2910),
    "rgbd_dataset_freiburg3_walking_static": (0, 60),
    "rgbd_dataset_freiburg3_walking_xyz": (60, 150, 210, 300, 360, 570, 630, 660, 810),
}


def _selected_slots(sequence: str) -> tuple[int, ...]:
    source = AVAILABLE_STARTS[sequence]
    wanted = sum(QUOTAS[sequence].values())
    if wanted == len(source):
        return source
    if not 0 < wanted < len(source):
        raise BuildStage0p7CapsulesError("slot count differs from quota")
    selected = tuple(source[round(index * (len(source) - 1) / (wanted - 1))] for index in range(wanted))
    if len(set(selected)) != wanted:
        raise BuildStage0p7CapsulesError("even slot selection repeated a source window")
    return selected


def _allocate() -> tuple[Spec, ...]:
    local_ordinals = {cause: 0 for cause in CAUSES}
    result: list[Spec] = []
    for sequence in SEQUENCE_ORDER:
        remaining = dict(QUOTAS[sequence])
        cause_index = 0
        for start in _selected_slots(sequence):
            for _ in CAUSES:
                cause = CAUSES[cause_index % len(CAUSES)]
                cause_index += 1
                if remaining[cause]:
                    remaining[cause] -= 1
                    result.append(Spec(cause, sequence, start, local_ordinals[cause], 0, 4) if cause == "normal_novelty" else Spec(cause, sequence, start, local_ordinals[cause]))
                    local_ordinals[cause] += 1
                    break
            else:
                raise BuildStage0p7CapsulesError("allocation exhausted causes before slots")
        if any(remaining.values()):
            raise BuildStage0p7CapsulesError("allocation did not satisfy a sequence quota")
    if len(result) != 80 or {cause: sum(spec.cause == cause for spec in result) for cause in CAUSES} != {cause: 20 for cause in CAUSES}:
        raise BuildStage0p7CapsulesError("allocation must be balanced at 80 windows")
    return tuple(result)


PILOT_SPECS = _allocate()


def _window(spec: Any) -> set[int]:
    return set(range(spec.source_start, spec.source_start + FRAME_COUNT))


def _assert_fresh(specs: Sequence[Spec]) -> None:
    prior = (*EVALUATION_SPECS, *DRY_RUN_SPECS, *CONTROL_SPECS, *STAGE0P5_SPECS, *STAGE0P6_SPECS)
    for index, spec in enumerate(specs):
        for other in (*prior, *specs[:index], *specs[index + 1 :]):
            if spec.source_sequence == other.source_sequence and not _window(spec).isdisjoint(_window(other)):
                raise BuildStage0p7CapsulesError(f"{spec.capsule_id} reuses source frames")


def _final_indices(spec: Spec) -> list[int]:
    values = list(range(spec.source_start, spec.source_start + FRAME_COUNT))
    if spec.cause == "registration_or_order_fault":
        values[spec.event_start : spec.event_end + 1] = list(reversed(values[spec.event_start : spec.event_end + 1]))
    return values


def _rectangle(x: float, y: float, width: float, height: float, fill: list[float]) -> dict[str, Any]:
    return {"type": "rectangle_occlusion", "coordinate_reference": COORDINATE_REFERENCE, "rectangle": {"x": x, "y": y, "width": width, "height": height}, "fill": fill}


def _transforms(spec: Spec, frame_id: int) -> list[dict[str, Any]]:
    if not spec.event_start <= frame_id <= spec.event_end:
        return []
    if spec.cause == "registration_or_order_fault":
        return [{"type": "temporal_reorder", "expected_source_index": spec.source_start + frame_id, "replacement_source_index": spec.source_start + spec.event_end - (frame_id - spec.event_start)}]
    if spec.cause == "bad_observation":
        fills = ([0.0, 0.0, 0.0], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [0.5, -0.5, 0.5])
        return [_rectangle(0.0, 0.0, 1.0, 1.0, list(fills[spec.ordinal % len(fills)]))]
    if spec.cause == "transient_local_content":
        colours = ([1.0, 1.0, -1.0], [-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, -1.0, -1.0])
        if spec.local_tier == "stress_75pct":
            positions = ((0.0, 0.0, 1.0, 0.75), (0.0, 0.25, 1.0, 0.75), (0.0, 0.0, 0.75, 1.0), (0.25, 0.0, 0.75, 1.0))
        else:
            positions = ((0.0, 0.0, 1.0, 0.5), (0.0, 0.5, 1.0, 0.5), (0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 0.5, 1.0))
        x, y, width, height = positions[spec.ordinal % len(positions)]
        return [_rectangle(x, y, width, height, list(colours[spec.ordinal % len(colours)]))]
    if spec.cause == "normal_novelty":
        return []
    raise BuildStage0p7CapsulesError("unsupported cause")


def _payload(spec: Spec, records: Sequence[tuple[float, Path]]) -> dict[str, Any]:
    indices = _final_indices(spec)
    if min(indices) < 0 or max(indices) >= len(records):
        raise BuildStage0p7CapsulesError(f"{spec.capsule_id} exceeds source listing")
    return {
        "schema_version": SCHEMA_VERSION,
        "capsule_id": spec.capsule_id,
        "source_sequence": spec.source_sequence,
        "source_group": f"{spec.source_sequence}:{spec.source_start}:{FRAME_COUNT}",
        "recipe_id": f"stage0p7-{spec.cause}-{spec.local_tier}-{spec.ordinal:02d}-v1",
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
        raise BuildStage0p7CapsulesError("output must be a new direct child of outputs")
    _assert_fresh(PILOT_SPECS)
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
            artifacts.append({"capsule_id": capsule.capsule_id, "path": str(path.relative_to(staging)), "sha256": _sha256(path), "event": capsule.event.to_dict(), "source_sequence": capsule.source_sequence, "source_group": capsule.source_group, "recipe_tier": spec.local_tier})
        inventory = {"schema_version": "stateguard3r.state-triage-stage0p7-input-inventory.v1", "created_at": datetime.now().astimezone().isoformat(), "frame_count_per_capsule": FRAME_COUNT, "evaluation_status": "large_development_only_existing_tum_rgb_window_benchmark", "online_evidence_contract": "current_frame_and_strict_prefix_only", "labels": "pre_forward_deterministic_recipe_only", "prior_stage_relation": "fresh_windows_only_not_a_prior_stage_rerun_or_revision", "persistent_change": "excluded", "allocation": {"total": 80, "per_cause": 20, "sequence_quotas": QUOTAS}, "artifacts": artifacts}
        (staging / "protocol.json").write_bytes(_canonical(inventory))
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
    except (BuildStage0p7CapsulesError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
