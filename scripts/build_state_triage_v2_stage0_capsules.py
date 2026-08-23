#!/usr/bin/env python3
"""Build the pre-forward, development-only StateTriage3R Stage-0 capsules."""

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
PROJECT_ROOT = ROOT.parents[0]
TUM_ROOT = PROJECT_ROOT / "baselines" / "ReCal3R" / "data" / "tum"
OUTPUT_NAME = "state-triage-v2-stage0-inputs-0001"
FRAME_COUNT = 30
EVENT_START = 12
EVENT_END = 17
SCHEMA_VERSION = "stateguard3r.state-triage-stage0-capsule.v1"
COORDINATE_REFERENCE = "model_input_after_resize_and_center_crop"

sys.path.insert(0, str(ROOT / "src"))
from stateguard3r.state_triage_capsule_v2 import load_stage0_capsule


class BuildStage0CapsulesError(ValueError):
    pass


@dataclass(frozen=True)
class Spec:
    cause: str
    source_sequence: str
    source_start: int
    event_start: int
    event_end: int

    @property
    def capsule_id(self) -> str:
        return f"{self.cause}-{self.source_sequence.replace('rgbd_dataset_', '')}-{self.source_start:04d}"


# Evaluation capsules use only already-disclosed data.  Source windows are
# deliberately disjoint across causes so the event-level development report
# does not repeat the same RGB span under different labels.
EVALUATION_SPECS: tuple[Spec, ...] = (
    Spec("normal_novelty", "rgbd_dataset_freiburg2_desk", 80, 0, 4),
    Spec("normal_novelty", "rgbd_dataset_freiburg2_desk", 380, 0, 4),
    Spec("normal_novelty", "rgbd_dataset_freiburg3_walking_static", 90, 0, 4),
    Spec("normal_novelty", "rgbd_dataset_freiburg3_walking_static", 380, 0, 4),
    Spec("normal_novelty", "rgbd_dataset_freiburg3_walking_xyz", 180, 0, 4),
    Spec("registration_or_order_fault", "rgbd_dataset_freiburg2_desk", 700, EVENT_START, EVENT_END),
    Spec("registration_or_order_fault", "rgbd_dataset_freiburg2_desk", 1050, EVENT_START, EVENT_END),
    Spec("registration_or_order_fault", "rgbd_dataset_freiburg3_walking_static", 180, EVENT_START, EVENT_END),
    Spec("registration_or_order_fault", "rgbd_dataset_freiburg3_walking_static", 500, EVENT_START, EVENT_END),
    Spec("registration_or_order_fault", "rgbd_dataset_freiburg3_walking_xyz", 330, EVENT_START, EVENT_END),
    Spec("bad_observation", "rgbd_dataset_freiburg2_desk", 1400, EVENT_START, EVENT_END),
    Spec("bad_observation", "rgbd_dataset_freiburg2_desk", 1800, EVENT_START, EVENT_END),
    Spec("bad_observation", "rgbd_dataset_freiburg3_walking_static", 280, EVENT_START, EVENT_END),
    Spec("bad_observation", "rgbd_dataset_freiburg3_walking_static", 620, EVENT_START, EVENT_END),
    Spec("bad_observation", "rgbd_dataset_freiburg3_walking_xyz", 520, EVENT_START, EVENT_END),
    Spec("transient_local_content", "rgbd_dataset_freiburg2_desk", 2200, EVENT_START, EVENT_END),
    Spec("transient_local_content", "rgbd_dataset_freiburg2_desk", 2500, EVENT_START, EVENT_END),
    Spec("transient_local_content", "rgbd_dataset_freiburg3_walking_static", 120, EVENT_START, EVENT_END),
    Spec("transient_local_content", "rgbd_dataset_freiburg3_walking_static", 440, EVENT_START, EVENT_END),
    Spec("transient_local_content", "rgbd_dataset_freiburg3_walking_xyz", 700, EVENT_START, EVENT_END),
)

# These clean, unlabelled fr1 capsules are only for Gate-B schema/threshold
# construction.  They never enter the event metrics.
DRY_RUN_SPECS: tuple[Spec, ...] = (
    Spec("normal_novelty", "rgbd_dataset_freiburg1_desk", 30, 0, 0),
    Spec("normal_novelty", "rgbd_dataset_freiburg1_desk", 230, 0, 0),
    Spec("normal_novelty", "rgbd_dataset_freiburg1_desk", 430, 0, 0),
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
        raise BuildStage0CapsulesError(f"raw RGB listing must be read-only: {listing}")
    rows: list[tuple[float, Path]] = []
    for line_number, raw in enumerate(listing.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise BuildStage0CapsulesError(f"{listing}:{line_number} is not timestamp/path")
        try:
            timestamp = float(fields[0])
        except ValueError as error:
            raise BuildStage0CapsulesError(f"{listing}:{line_number} timestamp is invalid") from error
        image = (root / fields[1]).resolve(strict=True)
        if not image.is_file() or stat.S_IMODE(image.stat().st_mode) & 0o222:
            raise BuildStage0CapsulesError(f"raw RGB must be a read-only file: {image}")
        rows.append((timestamp, image))
    return rows


def _final_indices(spec: Spec) -> list[int]:
    indices = list(range(spec.source_start, spec.source_start + FRAME_COUNT))
    if spec.cause == "registration_or_order_fault":
        indices[EVENT_START : EVENT_END + 1] = list(reversed(indices[EVENT_START : EVENT_END + 1]))
    return indices


def _transforms(spec: Spec, frame_id: int) -> list[dict[str, Any]]:
    if frame_id < spec.event_start or frame_id > spec.event_end:
        return []
    if spec.cause == "registration_or_order_fault":
        expected = spec.source_start + frame_id
        replacement = spec.source_start + EVENT_END - (frame_id - EVENT_START)
        return [{"type": "temporal_reorder", "expected_source_index": expected, "replacement_source_index": replacement}]
    if spec.cause == "bad_observation":
        return [{"type": "rectangle_occlusion", "coordinate_reference": COORDINATE_REFERENCE, "rectangle": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}, "fill": [0.0, 0.0, 0.0]}]
    if spec.cause == "transient_local_content":
        return [{"type": "rectangle_occlusion", "coordinate_reference": COORDINATE_REFERENCE, "rectangle": {"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5}, "fill": [1.0, -1.0, -1.0]}]
    return []


def _payload(spec: Spec, records: Sequence[tuple[float, Path]], *, dry_run: bool) -> dict[str, Any]:
    indices = _final_indices(spec)
    if min(indices) < 0 or max(indices) >= len(records):
        raise BuildStage0CapsulesError(f"{spec.capsule_id} exceeds raw RGB listing")
    cause = spec.cause if not dry_run else "normal_novelty"
    coverage = "novel" if cause == "normal_novelty" else "covered"
    return {
        "schema_version": SCHEMA_VERSION,
        "capsule_id": spec.capsule_id,
        "source_sequence": spec.source_sequence,
        "source_group": f"{spec.source_sequence}:{spec.source_start}:{FRAME_COUNT}",
        "recipe_id": f"stage0-{spec.cause}-v1",
        "event": {
            "event_id": f"event-{spec.capsule_id}",
            "cause": cause,
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
                "transforms": _transforms(spec, frame_id) if not dry_run else [],
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
        raise BuildStage0CapsulesError("output must be a new direct child of StateGuard3R/outputs")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".staging", dir=output_dir.parent))
    published = False
    try:
        artifacts: list[dict[str, Any]] = []
        for group_name, specs, dry_run in (("evaluation", EVALUATION_SPECS, False), ("schema-dry-run", DRY_RUN_SPECS, True)):
            for spec in specs:
                payload = _payload(spec, _rgb_records(spec.source_sequence), dry_run=dry_run)
                path = staging / group_name / f"{spec.capsule_id}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(_canonical(payload))
                path.chmod(0o444)
                capsule = load_stage0_capsule(path)
                artifacts.append({"group": group_name, "capsule_id": capsule.capsule_id, "path": str(path.relative_to(staging)), "sha256": _sha256(path), "event": capsule.event.to_dict(), "source_group": capsule.source_group})
        protocol = {
            "schema_version": "stateguard3r.state-triage-stage0-input-inventory.v1",
            "created_at": datetime.now().astimezone().isoformat(),
            "frame_count_per_capsule": FRAME_COUNT,
            "evaluation_status": "development_only_all_sequences_previously_disclosed",
            "dry_run_scope": "fr1_desk_only_not_in_event_metrics",
            "online_evidence_contract": "current_frame_and_strict_prefix_only",
            "labels": "pre_forward_deterministic_recipe_only",
            "persistent_change": "excluded_no_real_revisit_label_in_local_data",
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
    except (BuildStage0CapsulesError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
