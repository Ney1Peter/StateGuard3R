#!/usr/bin/env python3
"""Freeze clean, non-evaluation controls for every Stage-0 source sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NAME = "state-triage-v2-stage0-controls-0001"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.build_state_triage_v2_stage0_capsules import (  # noqa: E402
    FRAME_COUNT,
    SCHEMA_VERSION,
    Spec,
    _canonical,
    _freeze,
    _payload,
    _rgb_records,
)
from stateguard3r.state_triage_capsule_v2 import load_stage0_capsule  # noqa: E402


class BuildStage0ControlsError(ValueError):
    pass


# Each 30-frame window is disjoint from the frozen evaluation capsule windows.
CONTROL_SPECS: tuple[Spec, ...] = (
    Spec("normal_novelty", "rgbd_dataset_freiburg2_desk", 2750, 0, 0),
    Spec("normal_novelty", "rgbd_dataset_freiburg3_walking_static", 660, 0, 0),
    Spec("normal_novelty", "rgbd_dataset_freiburg3_walking_xyz", 780, 0, 0),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(output_dir: Path) -> Path:
    output_dir = output_dir.resolve(strict=False)
    if output_dir.parent != ROOT / "outputs" or output_dir.exists():
        raise BuildStage0ControlsError("output must be a new direct child of StateGuard3R/outputs")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".staging", dir=output_dir.parent))
    published = False
    try:
        artifacts: list[dict[str, Any]] = []
        for spec in CONTROL_SPECS:
            payload = _payload(spec, _rgb_records(spec.source_sequence), dry_run=True)
            path = staging / f"{spec.capsule_id}.json"
            path.write_bytes(_canonical(payload))
            path.chmod(0o444)
            capsule = load_stage0_capsule(path)
            artifacts.append(
                {
                    "capsule_id": capsule.capsule_id,
                    "path": path.name,
                    "sha256": _sha256(path),
                    "source_sequence": capsule.source_sequence,
                    "source_group": capsule.source_group,
                    "control_only": True,
                    "event_metrics_excluded": True,
                }
            )
        protocol = {
            "schema_version": "stateguard3r.state-triage-stage0-control-inventory.v1",
            "capsule_schema_version": SCHEMA_VERSION,
            "frame_count_per_capsule": FRAME_COUNT,
            "purpose": "Gate-C vanilla-observer equivalence only; never an event-metric input",
            "online_evidence_contract": "current_frame_and_strict_prefix_only",
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
    except (BuildStage0ControlsError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
