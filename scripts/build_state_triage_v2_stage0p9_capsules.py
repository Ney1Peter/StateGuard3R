#!/usr/bin/env python3
"""Build the sealed, fresh Stage 0.9 hashed-tile calibration inputs."""

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
TUM_ROOT = ROOT.parent / "baselines" / "ReCal3R" / "data" / "tum"
OUTPUT_NAME = "state-triage-v2-stage0p9-inputs-0001"
SOURCE_SEQUENCE = "rgbd_dataset_freiburg2_desk"
FRAME_COUNT, EVENT_START, EVENT_END = 30, 12, 17
SOURCE_STARTS = (110, 270, 730, 870, 1230, 1430, 1770, 1950, 2380, 2630, 2760, 2850)
HISTORICAL_INPUT_NAMES = (
    "state-triage-v2-stage0-inputs-0001",
    "state-triage-v2-stage0p5-inputs-0001",
    "state-triage-v2-stage0p6-inputs-0001",
    "state-triage-v2-stage0p7-inputs-0001",
    "state-triage-v2-stage0p8-inputs-0001",
)
CAPACITY_EXPECTED = {
    "rgbd_dataset_freiburg1_desk": 0,
    "rgbd_dataset_freiburg1_room": 2,
    "rgbd_dataset_freiburg2_desk": 17,
    "rgbd_dataset_freiburg3_walking_static": 3,
    "rgbd_dataset_freiburg3_walking_xyz": 1,
}
COORDINATE_REFERENCE = "model_input_after_resize_and_center_crop"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.build_state_triage_v2_stage0p5_capsules import _rgb_records, _sha256  # noqa: E402
from stateguard3r.state_triage_capsule_v2 import load_stage0_capsule  # noqa: E402


class BuildStage0p9CapsulesError(ValueError):
    pass


@dataclass(frozen=True)
class Spec:
    source_start: int
    ordinal: int

    @property
    def capsule_id(self) -> str:
        return f"calibration-transient_local_content-freiburg2_desk-{self.source_start:04d}"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _read_only(path: Path, *, directory: bool = False) -> None:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or (not stat.S_ISDIR(metadata.st_mode) if directory else not stat.S_ISREG(metadata.st_mode)) or stat.S_IMODE(metadata.st_mode) & 0o222:
        raise BuildStage0p9CapsulesError(f"{path} must be a read-only {'directory' if directory else 'file'}")


def _strict_json(path: Path) -> dict[str, Any]:
    def duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BuildStage0p9CapsulesError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda name: (_ for _ in ()).throw(BuildStage0p9CapsulesError(f"non-finite JSON constant {name}")), object_pairs_hook=duplicate)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, BuildStage0p9CapsulesError) as error:
        raise BuildStage0p9CapsulesError(f"cannot load strict JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise BuildStage0p9CapsulesError(f"{path} must contain an object")
    return value


def _artifact_lists(inventory: Mapping[str, Any]) -> list[tuple[str, list[Any]]]:
    result: list[tuple[str, list[Any]]] = []
    if isinstance(inventory.get("artifacts"), list):
        result.append(("artifacts", inventory["artifacts"]))
    for name in ("calibration", "reserved_final"):
        section = inventory.get(name)
        if isinstance(section, Mapping) and isinstance(section.get("artifacts"), list):
            result.append((name, section["artifacts"]))
    if not result:
        raise BuildStage0p9CapsulesError("historical inventory has no artifacts")
    return result


def _historical_occupancy() -> tuple[dict[str, set[Path]], list[dict[str, str]]]:
    occupied: dict[str, set[Path]] = {}
    bindings: list[dict[str, str]] = []
    for name in HISTORICAL_INPUT_NAMES:
        directory = ROOT / "outputs" / name
        protocol_path = directory / "protocol.json"
        _read_only(directory, directory=True)
        _read_only(protocol_path)
        inventory = _strict_json(protocol_path)
        bindings.append({"path": str(protocol_path.relative_to(ROOT)), "sha256": _sha256(protocol_path)})
        for section, artifacts in _artifact_lists(inventory):
            for item in artifacts:
                if not isinstance(item, Mapping) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
                    raise BuildStage0p9CapsulesError(f"{name}:{section} artifact schema differs")
                capsule_path = (directory / item["path"]).resolve(strict=True)
                if capsule_path.parent.parent != directory and capsule_path.parent != directory / "evaluation":
                    raise BuildStage0p9CapsulesError("historical capsule lies outside its inventory")
                _read_only(capsule_path)
                if _sha256(capsule_path) != item["sha256"]:
                    raise BuildStage0p9CapsulesError("historical capsule hash differs")
                capsule = load_stage0_capsule(capsule_path)
                occupied.setdefault(capsule.source_sequence, set()).update(frame.path for frame in capsule.frames)
    return occupied, bindings


def _capacity(occupied: Mapping[str, set[Path]]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for listing in sorted(TUM_ROOT.glob("rgbd_dataset_*/rgb.txt")):
        sequence = listing.parent.name
        records = _rgb_records(sequence)
        blocked = [path in occupied.get(sequence, set()) for _, path in records]
        starts: list[int] = []
        cursor = 0
        while cursor + FRAME_COUNT <= len(records):
            if not any(blocked[cursor : cursor + FRAME_COUNT]):
                starts.append(cursor)
                cursor += FRAME_COUNT
            else:
                cursor += 1
        result[sequence] = starts
    if {name: len(starts) for name, starts in result.items()} != CAPACITY_EXPECTED:
        raise BuildStage0p9CapsulesError("historical fresh-window capacity differs from the Stage 0.9 audit")
    return result


def _specs(capacity: Mapping[str, Sequence[int]]) -> tuple[Spec, ...]:
    candidates = capacity.get(SOURCE_SEQUENCE)
    if candidates is None or any(start not in candidates for start in SOURCE_STARTS) or len(SOURCE_STARTS) != 12 or len(set(SOURCE_STARTS)) != 12:
        raise BuildStage0p9CapsulesError("predeclared Stage 0.9 source starts are not fresh")
    return tuple(Spec(source_start=start, ordinal=index) for index, start in enumerate(SOURCE_STARTS))


def _window(spec: Spec) -> set[int]:
    return set(range(spec.source_start, spec.source_start + FRAME_COUNT))


def _assert_specs(specs: Sequence[Spec]) -> None:
    if len(specs) != 12 or any(not _window(first).isdisjoint(_window(second)) for index, first in enumerate(specs) for second in specs[index + 1 :]):
        raise BuildStage0p9CapsulesError("Stage 0.9 calibration windows overlap")


def _transform(spec: Spec) -> dict[str, Any]:
    positions = ((0.0, 0.0, 1.0, 0.75), (0.0, 0.25, 1.0, 0.75), (0.0, 0.0, 0.75, 1.0), (0.25, 0.0, 0.75, 1.0))
    x, y, width, height = positions[spec.ordinal % len(positions)]
    return {"type": "hashed_tile_occlusion", "coordinate_reference": COORDINATE_REFERENCE, "rectangle": {"x": x, "y": y, "width": width, "height": height}, "tile_size_pixels": 16, "fills": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], "seed": 1909}


def _payload(spec: Spec, records: Sequence[tuple[float, Path]]) -> dict[str, Any]:
    indices = list(range(spec.source_start, spec.source_start + FRAME_COUNT))
    if max(indices) >= len(records):
        raise BuildStage0p9CapsulesError(f"{spec.capsule_id} exceeds RGB listing")
    return {"schema_version": "stateguard3r.state-triage-stage0-capsule.v1", "capsule_id": spec.capsule_id, "source_sequence": SOURCE_SEQUENCE, "source_group": f"{SOURCE_SEQUENCE}:{spec.source_start}:{FRAME_COUNT}", "recipe_id": f"stage0p9-hashed_tile_75pct_seed1909-{spec.ordinal:02d}-v1", "event": {"event_id": f"event-{spec.capsule_id}", "cause": "transient_local_content", "start_frame": EVENT_START, "end_frame": EVENT_END, "coverage_expectation": "covered", "label_provenance": "pre_forward_deterministic_recipe"}, "online_evidence_contract": "current_frame_and_strict_prefix_only", "frames": [{"frame_id": frame_id, "path": str(records[index][1]), "sha256": _sha256(records[index][1]), "timestamp": records[index][0], "transforms": [_transform(spec)] if EVENT_START <= frame_id <= EVENT_END else []} for frame_id, index in enumerate(indices)]}


def _freeze(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def build(output_dir: Path) -> Path:
    output_dir = output_dir.resolve(strict=False)
    if output_dir.parent != ROOT / "outputs" or output_dir.exists():
        raise BuildStage0p9CapsulesError("output must be a new direct child of outputs")
    occupied, bindings = _historical_occupancy()
    capacity = _capacity(occupied)
    specs = _specs(capacity)
    _assert_specs(specs)
    records = _rgb_records(SOURCE_SEQUENCE)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".staging", dir=output_dir.parent))
    published = False
    try:
        artifacts: list[dict[str, Any]] = []
        for spec in specs:
            path = staging / "calibration" / f"{spec.capsule_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_canonical(_payload(spec, records)))
            path.chmod(0o444)
            capsule = load_stage0_capsule(path)
            artifacts.append({"capsule_id": capsule.capsule_id, "path": str(path.relative_to(staging)), "sha256": _sha256(path), "event": capsule.event.to_dict(), "source_sequence": capsule.source_sequence, "source_group": capsule.source_group, "recipe": "hashed_tile_75pct_seed1909_v1"})
        protocol = {"schema_version": "stateguard3r.state-triage-stage0p9-input-inventory.v1", "created_at": datetime.now().astimezone().isoformat(), "frame_count_per_capsule": FRAME_COUNT, "historical_inputs": bindings, "fresh_capacity_30_frame_windows": {name: starts for name, starts in capacity.items()}, "source_selection": {"sequence": SOURCE_SEQUENCE, "starts": list(SOURCE_STARTS), "rule": "fixed_evenly_spread_subset_of_replayed_fresh_capacity"}, "online_evidence_contract": "current_frame_and_strict_prefix_only", "labels": "pre_forward_deterministic_recipe_only", "calibration": {"count": 12, "acceptance": "at_least_10_first_event_rows_sharpness_gt_0.05_clipped_fraction_gte_0.20_coverage_ratio_lt_0.25", "artifacts": artifacts}, "final_scope": "prohibited_without_new_authorized_source_data", "persistent_change": "excluded"}
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        occupied, bindings = _historical_occupancy()
        capacity = _capacity(occupied)
        specs = _specs(capacity)
        _assert_specs(specs)
        if args.dry_run:
            print(json.dumps({"historical_inventory_count": len(bindings), "capacity": {name: len(starts) for name, starts in capacity.items()}, "selection": list(SOURCE_STARTS)}, ensure_ascii=False, sort_keys=True, allow_nan=False))
        else:
            print(build(args.output_dir))
    except (BuildStage0p9CapsulesError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
