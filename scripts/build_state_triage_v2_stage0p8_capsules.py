#!/usr/bin/env python3
"""Build sealed calibration and reserved-final capsules for Stage 0.8."""

from __future__ import annotations

import argparse
from collections import Counter
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
OUTPUT_NAME = "state-triage-v2-stage0p8-inputs-0001"
DATASET_NAME = "rgbd_dataset_freiburg1_room"
FRAME_COUNT, EVENT_START, EVENT_END = 30, 12, 17
SCHEMA_VERSION = "stateguard3r.state-triage-stage0-capsule.v1"
COORDINATE_REFERENCE = "model_input_after_resize_and_center_crop"
CALIBRATION_COUNT = 12
FINAL_QUOTAS = {
    "normal_novelty": 5,
    "registration_or_order_fault": 5,
    "bad_observation": 5,
    "transient_local_content": 16,
}
TOTAL_COUNT = CALIBRATION_COUNT + sum(FINAL_QUOTAS.values())
ACQUISITION_PATH = ROOT / "outputs" / "state-triage-v2-stage0p8-data-0001" / "acquisition.json"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.build_state_triage_v2_stage0p5_capsules import _rgb_records, _sha256  # noqa: E402
from stateguard3r.state_triage_capsule_v2 import load_stage0_capsule  # noqa: E402


class BuildStage0p8CapsulesError(ValueError):
    pass


@dataclass(frozen=True)
class Spec:
    phase: str
    cause: str
    source_start: int
    ordinal: int
    event_start: int = EVENT_START
    event_end: int = EVENT_END

    @property
    def source_sequence(self) -> str:
        return DATASET_NAME

    @property
    def capsule_id(self) -> str:
        return f"{self.phase}-{self.cause}-{DATASET_NAME.replace('rgbd_dataset_', '')}-{self.source_start:04d}"

    @property
    def local_recipe(self) -> str:
        return "checkerboard_75pct_v1" if self.cause == "transient_local_content" else "not_applicable"


ROLE_CYCLE = (
    ("calibration", "transient_local_content"),
    ("final", "transient_local_content"),
    ("final", "normal_novelty"),
    ("final", "transient_local_content"),
    ("final", "registration_or_order_fault"),
    ("calibration", "transient_local_content"),
    ("final", "transient_local_content"),
    ("final", "bad_observation"),
)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _read_only_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise BuildStage0p8CapsulesError(f"acquisition record must be a read-only regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BuildStage0p8CapsulesError(f"cannot load acquisition record: {error}") from error
    if not isinstance(value, dict):
        raise BuildStage0p8CapsulesError("acquisition record is not an object")
    return value


def _acquisition() -> dict[str, Any]:
    value = _read_only_json(ACQUISITION_PATH)
    raw = value.get("raw_tree")
    if value.get("schema_version") != "stateguard3r.state-triage-stage0p8-scene-acquisition.v1" or value.get("status") != "PASS" or value.get("dataset") != DATASET_NAME or not isinstance(raw, Mapping):
        raise BuildStage0p8CapsulesError("Stage 0.8 source acquisition contract differs")
    source_root = (ROOT.parent / "baselines" / "ReCal3R" / "data" / "tum" / DATASET_NAME).resolve(strict=True)
    if raw.get("path") != str(source_root) or stat.S_IMODE(source_root.stat().st_mode) & 0o222:
        raise BuildStage0p8CapsulesError("Stage 0.8 source tree is not the acquired read-only scene")
    return value


def _selected_starts(record_count: int) -> tuple[int, ...]:
    candidates = tuple(range(0, record_count - FRAME_COUNT + 1, FRAME_COUNT))
    if len(candidates) < TOTAL_COUNT:
        raise BuildStage0p8CapsulesError(f"new scene has only {len(candidates)} disjoint 30-frame slots; need {TOTAL_COUNT}")
    selected = tuple(candidates[round(index * (len(candidates) - 1) / (TOTAL_COUNT - 1))] for index in range(TOTAL_COUNT))
    if len(set(selected)) != TOTAL_COUNT:
        raise BuildStage0p8CapsulesError("evenly spread Stage 0.8 selection repeated a source window")
    return selected


def _roles() -> tuple[tuple[str, str], ...]:
    remaining = {("calibration", "transient_local_content"): CALIBRATION_COUNT, **{("final", cause): count for cause, count in FINAL_QUOTAS.items()}}
    result: list[tuple[str, str]] = []
    cursor = 0
    while any(remaining.values()):
        role = ROLE_CYCLE[cursor % len(ROLE_CYCLE)]
        cursor += 1
        if remaining.get(role, 0) <= 0:
            continue
        remaining[role] -= 1
        result.append(role)
    if len(result) != TOTAL_COUNT or Counter(result)[("calibration", "transient_local_content")] != CALIBRATION_COUNT:
        raise BuildStage0p8CapsulesError("Stage 0.8 role allocation differs")
    return tuple(result)


def _allocate(record_count: int) -> tuple[Spec, ...]:
    ordinals: dict[tuple[str, str], int] = {}
    specs: list[Spec] = []
    for start, role in zip(_selected_starts(record_count), _roles(), strict=True):
        ordinal = ordinals.get(role, 0)
        ordinals[role] = ordinal + 1
        phase, cause = role
        specs.append(Spec(phase=phase, cause=cause, source_start=start, ordinal=ordinal, event_start=0 if cause == "normal_novelty" else EVENT_START, event_end=4 if cause == "normal_novelty" else EVENT_END))
    if Counter((spec.phase, spec.cause) for spec in specs) != Counter({("calibration", "transient_local_content"): CALIBRATION_COUNT, **{("final", cause): count for cause, count in FINAL_QUOTAS.items()}}):
        raise BuildStage0p8CapsulesError("Stage 0.8 allocation quota differs")
    return tuple(specs)


def _window(spec: Spec) -> set[int]:
    return set(range(spec.source_start, spec.source_start + FRAME_COUNT))


def _assert_fresh(specs: Sequence[Spec]) -> None:
    if any(spec.source_sequence != DATASET_NAME for spec in specs):
        raise BuildStage0p8CapsulesError("Stage 0.8 source sequence differs")
    for index, spec in enumerate(specs):
        if any(not _window(spec).isdisjoint(_window(other)) for other in specs[index + 1 :]):
            raise BuildStage0p8CapsulesError(f"{spec.capsule_id} reuses a Stage 0.8 source frame")


def _final_indices(spec: Spec) -> list[int]:
    values = list(range(spec.source_start, spec.source_start + FRAME_COUNT))
    if spec.cause == "registration_or_order_fault":
        values[spec.event_start : spec.event_end + 1] = list(reversed(values[spec.event_start : spec.event_end + 1]))
    return values


def _rectangle(x: float, y: float, width: float, height: float, fill: list[float]) -> dict[str, Any]:
    return {"type": "rectangle_occlusion", "coordinate_reference": COORDINATE_REFERENCE, "rectangle": {"x": x, "y": y, "width": width, "height": height}, "fill": fill}


def _checkerboard(spec: Spec) -> dict[str, Any]:
    positions = ((0.0, 0.0, 1.0, 0.75), (0.0, 0.25, 1.0, 0.75), (0.0, 0.0, 0.75, 1.0), (0.25, 0.0, 0.75, 1.0))
    x, y, width, height = positions[spec.ordinal % len(positions)]
    return {
        "type": "checkerboard_occlusion",
        "coordinate_reference": COORDINATE_REFERENCE,
        "rectangle": {"x": x, "y": y, "width": width, "height": height},
        "tile_size_pixels": 16,
        "fills": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
    }


def _transforms(spec: Spec, frame_id: int) -> list[dict[str, Any]]:
    if not spec.event_start <= frame_id <= spec.event_end:
        return []
    if spec.cause == "normal_novelty":
        return []
    if spec.cause == "registration_or_order_fault":
        return [{"type": "temporal_reorder", "expected_source_index": spec.source_start + frame_id, "replacement_source_index": spec.source_start + spec.event_end - (frame_id - spec.event_start)}]
    if spec.cause == "bad_observation":
        fills = ([0.0, 0.0, 0.0], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [0.5, -0.5, 0.5])
        return [_rectangle(0.0, 0.0, 1.0, 1.0, list(fills[spec.ordinal % len(fills)]))]
    if spec.cause == "transient_local_content":
        return [_checkerboard(spec)]
    raise BuildStage0p8CapsulesError("unsupported Stage 0.8 cause")


def _payload(spec: Spec, records: Sequence[tuple[float, Path]]) -> dict[str, Any]:
    indices = _final_indices(spec)
    if min(indices) < 0 or max(indices) >= len(records):
        raise BuildStage0p8CapsulesError(f"{spec.capsule_id} exceeds raw RGB listing")
    return {
        "schema_version": SCHEMA_VERSION,
        "capsule_id": spec.capsule_id,
        "source_sequence": spec.source_sequence,
        "source_group": f"{spec.source_sequence}:{spec.source_start}:{FRAME_COUNT}",
        "recipe_id": f"stage0p8-{spec.phase}-{spec.cause}-{spec.local_recipe}-{spec.ordinal:02d}-v1",
        "event": {"event_id": f"event-{spec.capsule_id}", "cause": spec.cause, "start_frame": spec.event_start, "end_frame": spec.event_end, "coverage_expectation": "novel" if spec.cause == "normal_novelty" else "covered", "label_provenance": "pre_forward_deterministic_recipe"},
        "online_evidence_contract": "current_frame_and_strict_prefix_only",
        "frames": [{"frame_id": frame_id, "path": str(records[source_index][1]), "sha256": _sha256(records[source_index][1]), "timestamp": records[source_index][0], "transforms": _transforms(spec, frame_id)} for frame_id, source_index in enumerate(indices)],
    }


def _freeze(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def build(output_dir: Path) -> Path:
    output_dir = output_dir.resolve(strict=False)
    if output_dir.parent != ROOT / "outputs" or output_dir.exists():
        raise BuildStage0p8CapsulesError("output must be a new direct child of StateGuard3R/outputs")
    acquisition = _acquisition()
    records = _rgb_records(DATASET_NAME)
    specs = _allocate(len(records))
    _assert_fresh(specs)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".staging", dir=output_dir.parent))
    published = False
    try:
        artifacts: dict[str, list[dict[str, Any]]] = {"calibration": [], "final": []}
        for spec in specs:
            path = staging / spec.phase / f"{spec.capsule_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_canonical(_payload(spec, records)))
            path.chmod(0o444)
            capsule = load_stage0_capsule(path)
            artifacts[spec.phase].append({"capsule_id": capsule.capsule_id, "path": str(path.relative_to(staging)), "sha256": _sha256(path), "event": capsule.event.to_dict(), "source_sequence": capsule.source_sequence, "source_group": capsule.source_group, "recipe": spec.local_recipe})
        inventory = {
            "schema_version": "stateguard3r.state-triage-stage0p8-input-inventory.v1",
            "created_at": datetime.now().astimezone().isoformat(),
            "frame_count_per_capsule": FRAME_COUNT,
            "source_acquisition": {"path": str(ACQUISITION_PATH.relative_to(ROOT)), "sha256": _sha256(ACQUISITION_PATH)},
            "online_evidence_contract": "current_frame_and_strict_prefix_only",
            "labels": "pre_forward_deterministic_recipe_only",
            "prior_stage_relation": "new_scene_and_windows_only_not_a_stage0_to_stage0p7_rerun_or_revision",
            "calibration": {"count": CALIBRATION_COUNT, "acceptance": "at_least_10_first_event_rows_sharpness_gt_0.05_clipped_fraction_gte_0.20_coverage_ratio_lt_0.25", "artifacts": artifacts["calibration"]},
            "reserved_final": {"counts": FINAL_QUOTAS, "total": sum(FINAL_QUOTAS.values()), "excluded_from_calibration_scoring": True, "artifacts": artifacts["final"]},
            "persistent_change": "excluded",
        }
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        acquisition = _acquisition()
        records = _rgb_records(DATASET_NAME)
        specs = _allocate(len(records))
        _assert_fresh(specs)
        if args.dry_run:
            print(json.dumps({"acquisition_sha256": _sha256(ACQUISITION_PATH), "source_frames": len(records), "total": len(specs), "counts": {"calibration": sum(spec.phase == "calibration" for spec in specs), "final": Counter(spec.cause for spec in specs if spec.phase == "final")}, "source_starts": [spec.source_start for spec in specs]}, ensure_ascii=False, sort_keys=True, allow_nan=False))
        else:
            print(build(args.output_dir))
    except (BuildStage0p8CapsulesError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
