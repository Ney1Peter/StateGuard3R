#!/usr/bin/env python3
"""Freeze the Stage 0.9 hashed-tile calibration protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
INPUT_NAME = "state-triage-v2-stage0p9-inputs-0001"
OUTPUT_PATH = ROOT / "docs" / "protocols" / "state-triage-v2-stage0p9-calibration-gate-b.json"
sys.path.insert(0, str(ROOT / "src"))
from stateguard3r.state_triage_capsule_v2 import load_stage0_capsule  # noqa: E402
from stateguard3r.state_triage_stage0p8 import DECISION_CONTRACT, INTERACTION_CONTRACT  # noqa: E402
from stateguard3r.state_triage_v2 import EvidenceConfig  # noqa: E402


class FreezeStage0p9CalibrationProtocolError(ValueError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    def duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FreezeStage0p9CalibrationProtocolError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda name: (_ for _ in ()).throw(FreezeStage0p9CalibrationProtocolError(f"non-finite JSON constant {name}")), object_pairs_hook=duplicate)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, FreezeStage0p9CalibrationProtocolError) as error:
        raise FreezeStage0p9CalibrationProtocolError(f"cannot load strict JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise FreezeStage0p9CalibrationProtocolError("JSON root must be an object")
    return value


def _read_only(path: Path, *, directory: bool = False) -> None:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or (not stat.S_ISDIR(metadata.st_mode) if directory else not stat.S_ISREG(metadata.st_mode)) or stat.S_IMODE(metadata.st_mode) & 0o222:
        raise FreezeStage0p9CalibrationProtocolError(f"{path} must be a read-only {'directory' if directory else 'file'}")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _inventory(input_dir: Path) -> list[dict[str, str]]:
    _read_only(input_dir, directory=True)
    inventory = _json(input_dir / "protocol.json")
    required = {"schema_version", "created_at", "frame_count_per_capsule", "historical_inputs", "fresh_capacity_30_frame_windows", "source_selection", "online_evidence_contract", "labels", "calibration", "final_scope", "persistent_change"}
    if set(inventory) != required or inventory.get("schema_version") != "stateguard3r.state-triage-stage0p9-input-inventory.v1" or inventory.get("online_evidence_contract") != "current_frame_and_strict_prefix_only" or inventory.get("final_scope") != "prohibited_without_new_authorized_source_data":
        raise FreezeStage0p9CalibrationProtocolError("Stage 0.9 input inventory schema differs")
    calibration = inventory.get("calibration")
    if not isinstance(calibration, Mapping) or set(calibration) != {"count", "acceptance", "artifacts"} or calibration["count"] != 12 or calibration["acceptance"] != "at_least_10_first_event_rows_sharpness_gt_0.05_clipped_fraction_gte_0.20_coverage_ratio_lt_0.25":
        raise FreezeStage0p9CalibrationProtocolError("calibration contract differs")
    artifacts = calibration["artifacts"]
    expected = {"capsule_id", "path", "sha256", "event", "source_sequence", "source_group", "recipe"}
    if not isinstance(artifacts, list) or len(artifacts) != 12 or any(not isinstance(item, Mapping) or set(item) != expected for item in artifacts):
        raise FreezeStage0p9CalibrationProtocolError("calibration artifact schema differs")
    result: list[dict[str, str]] = []
    groups: set[str] = set()
    for item in artifacts:
        path = (input_dir / str(item["path"])).resolve(strict=True)
        if path.parent != input_dir / "calibration":
            raise FreezeStage0p9CalibrationProtocolError("capsule lies outside calibration directory")
        _read_only(path)
        if _sha(path) != item["sha256"]:
            raise FreezeStage0p9CalibrationProtocolError("capsule hash differs")
        capsule = load_stage0_capsule(path)
        if capsule.capsule_id != item["capsule_id"] or capsule.event.cause != "transient_local_content" or item["recipe"] != "hashed_tile_75pct_seed1909_v1" or capsule.source_group in groups:
            raise FreezeStage0p9CalibrationProtocolError("capsule differs from sealed hashed-tile inventory")
        groups.add(capsule.source_group)
        result.append({"capsule_path": str(path.relative_to(ROOT)), "capsule_sha256": str(item["sha256"]), "capsule_id": capsule.capsule_id, "output": f"state-triage-v2-stage0p9-calibration-{capsule.capsule_id}-observer-0001"})
    if len({item["capsule_id"] for item in result}) != 12 or len({item["output"] for item in result}) != 12:
        raise FreezeStage0p9CalibrationProtocolError("duplicate capsule/output name")
    return result


def freeze(*, input_dir: Path, output_path: Path) -> Path:
    input_dir, output_path = input_dir.resolve(strict=True), output_path.resolve(strict=False)
    if input_dir.parent != ROOT / "outputs" or output_path.parent != ROOT / "docs" / "protocols" or output_path.exists():
        raise FreezeStage0p9CalibrationProtocolError("input/output location is invalid or exists")
    calibration_inventory = _inventory(input_dir)
    runner, observer = ROOT / "scripts" / "run_state_triage_v2_stage0.py", ROOT / "src" / "stateguard3r" / "state_triage_v2.py"
    payload = {"schema_version": "stateguard3r.state-triage-stage0p9-calibration-gate-b.v1", "stage": "GATE_B_STAGE0P9_CALIBRATION_FROZEN_PRE_EVALUATION", "input_inventory": {"path": str(input_dir.relative_to(ROOT)), "sha256": _sha(input_dir / "protocol.json")}, "frozen_sources": {"runner_path": str(runner.relative_to(ROOT)), "runner_sha256": _sha(runner), "observer_path": str(observer.relative_to(ROOT)), "observer_sha256": _sha(observer)}, "frozen_evidence_config": EvidenceConfig().to_dict(), "decision": DECISION_CONTRACT, "interaction": INTERACTION_CONTRACT, "calibration_inventory": calibration_inventory, "gates": {"valid_event_count": 12, "activated_event_min": 10, "sharpness_min_exclusive": 0.05, "clipped_fraction_min": 0.20, "coverage_max_exclusive": 0.25}, "scope": "sealed hashed-tile recipe-feasibility calibration only; permanently excluded from final scoring"}
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".staging", dir=output_path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(payload))
        os.chmod(temporary, 0o444)
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs" / INPUT_NAME)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    try:
        print(freeze(input_dir=args.input_dir, output_path=args.output_path))
    except (FreezeStage0p9CalibrationProtocolError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
