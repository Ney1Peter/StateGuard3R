#!/usr/bin/env python3
"""Evaluate the sealed Stage 0.9 hashed-tile calibration exactly once."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SCHEMA_VERSION = "stateguard3r.state-triage-stage0p9-calibration-gate-b.v1"
RESULT_SCHEMA_VERSION = "stateguard3r.state-triage-stage0p9-calibration-evaluation.v1"
sys.path.insert(0, str(ROOT / "src"))
from stateguard3r.state_triage_capsule_v2 import load_stage0_capsule  # noqa: E402
from stateguard3r.state_triage_stage0p8 import DECISION_CONTRACT, INTERACTION_CONTRACT, METHODS, decide_event  # noqa: E402
from stateguard3r.state_triage_v2 import EvidenceConfig  # noqa: E402


class Stage0p9CalibrationEvaluationError(ValueError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise Stage0p9CalibrationEvaluationError(f"non-finite JSON constant {value!r}")
    def duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Stage0p9CalibrationEvaluationError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant, object_pairs_hook=duplicate)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, Stage0p9CalibrationEvaluationError) as error:
        raise Stage0p9CalibrationEvaluationError(f"cannot load strict JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Stage0p9CalibrationEvaluationError(f"{path} must contain an object")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise Stage0p9CalibrationEvaluationError(f"{name} must be finite")
    return float(value)


def _read_only(path: Path, *, directory: bool = False) -> None:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or (not stat.S_ISDIR(metadata.st_mode) if directory else not stat.S_ISREG(metadata.st_mode)) or stat.S_IMODE(metadata.st_mode) & 0o222:
        raise Stage0p9CalibrationEvaluationError(f"{path} must be a read-only {'directory' if directory else 'file'}")


def _direct(relative: str, parent: Path) -> Path:
    path = (ROOT / relative).resolve(strict=True)
    if path.parent != parent:
        raise Stage0p9CalibrationEvaluationError(f"{relative} lies outside approved direct parent")
    return path


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _protocol(path: Path) -> dict[str, Any]:
    _read_only(path)
    value = _json(path)
    required = {"schema_version", "stage", "input_inventory", "frozen_sources", "frozen_evidence_config", "decision", "interaction", "calibration_inventory", "gates", "scope"}
    if set(value) != required or value.get("schema_version") != PROTOCOL_SCHEMA_VERSION or value.get("stage") != "GATE_B_STAGE0P9_CALIBRATION_FROZEN_PRE_EVALUATION":
        raise Stage0p9CalibrationEvaluationError("protocol schema differs")
    artifact = value["input_inventory"]
    if not isinstance(artifact, Mapping) or set(artifact) != {"path", "sha256"}:
        raise Stage0p9CalibrationEvaluationError("input artifact schema differs")
    inputs = _direct(str(artifact["path"]), ROOT / "outputs")
    _read_only(inputs, directory=True)
    if _sha(inputs / "protocol.json") != artifact["sha256"]:
        raise Stage0p9CalibrationEvaluationError("input inventory hash differs")
    sources = value["frozen_sources"]
    if not isinstance(sources, Mapping) or set(sources) != {"runner_path", "runner_sha256", "observer_path", "observer_sha256"}:
        raise Stage0p9CalibrationEvaluationError("source schema differs")
    runner = _direct(str(sources["runner_path"]), ROOT / "scripts")
    observer = _direct(str(sources["observer_path"]), ROOT / "src" / "stateguard3r")
    if _sha(runner) != sources["runner_sha256"] or _sha(observer) != sources["observer_sha256"]:
        raise Stage0p9CalibrationEvaluationError("runner or observer changed after protocol freeze")
    if value["decision"] != DECISION_CONTRACT or value["interaction"] != INTERACTION_CONTRACT or value["frozen_evidence_config"] != EvidenceConfig().to_dict():
        raise Stage0p9CalibrationEvaluationError("decision, interaction, or evidence contract differs")
    items = value["calibration_inventory"]
    expected = {"capsule_path", "capsule_sha256", "capsule_id", "output"}
    if not isinstance(items, list) or len(items) != 12 or any(not isinstance(item, Mapping) or set(item) != expected for item in items) or len({item["capsule_id"] for item in items}) != 12 or len({item["output"] for item in items}) != 12:
        raise Stage0p9CalibrationEvaluationError("calibration inventory schema differs")
    gates = value["gates"]
    expected_gates = {"valid_event_count", "activated_event_min", "sharpness_min_exclusive", "clipped_fraction_min", "coverage_max_exclusive"}
    if not isinstance(gates, Mapping) or set(gates) != expected_gates or gates["valid_event_count"] != 12 or gates["activated_event_min"] != 10:
        raise Stage0p9CalibrationEvaluationError("gate schema differs")
    for name in expected_gates - {"valid_event_count", "activated_event_min"}:
        _finite(gates[name], f"gate {name}")
    return value


def _event(item: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    capsule_path = _direct(str(item["capsule_path"]), ROOT / "outputs" / "state-triage-v2-stage0p9-inputs-0001" / "calibration")
    _read_only(capsule_path)
    if _sha(capsule_path) != item["capsule_sha256"]:
        raise Stage0p9CalibrationEvaluationError("capsule hash differs")
    capsule = load_stage0_capsule(capsule_path)
    if capsule.capsule_id != item["capsule_id"] or capsule.event.cause != "transient_local_content":
        raise Stage0p9CalibrationEvaluationError("capsule identity differs")
    output = _direct(f"outputs/{item['output']}", ROOT / "outputs")
    _read_only(output, directory=True)
    run, evidence = _json(output / "run.json"), _json(output / "evidence.json")
    required_run = {"status", "observer", "input_capsule", "frame_count", "runner", "observer_wall_seconds", "runtime_seconds", "peak_memory_allocated_mib"}
    if run.get("status") != "succeeded" or run.get("observer") != "on" or not required_run.issubset(run) or not isinstance(run["input_capsule"], Mapping) or run["input_capsule"].get("sha256") != item["capsule_sha256"] or run["frame_count"] != len(capsule.frames):
        raise Stage0p9CalibrationEvaluationError("run provenance differs")
    for field in ("observer_wall_seconds", "runtime_seconds", "peak_memory_allocated_mib"):
        if _finite(run[field], field) < 0.0:
            raise Stage0p9CalibrationEvaluationError(f"{field} cannot be negative")
    if evidence.get("config") != protocol["frozen_evidence_config"] or evidence.get("causality") != "each row accepts current frame plus strict observer prefix only":
        raise Stage0p9CalibrationEvaluationError("evidence causality/config differs")
    rows = evidence.get("rows")
    if not isinstance(rows, list) or len(rows) != len(capsule.frames) or [row.get("frame_id") for row in rows if isinstance(row, Mapping)] != list(range(len(rows))):
        raise Stage0p9CalibrationEvaluationError("evidence rows differ")
    first = rows[capsule.event.start_frame]
    if not isinstance(first, Mapping):
        raise Stage0p9CalibrationEvaluationError("first event row is malformed")
    features = {name: _finite(first.get(name), f"first event {name}") for name in ("sharpness", "clipped_fraction", "coverage_ratio")}
    activated = features["sharpness"] > float(protocol["gates"]["sharpness_min_exclusive"]) and features["clipped_fraction"] >= float(protocol["gates"]["clipped_fraction_min"]) and features["coverage_ratio"] < float(protocol["gates"]["coverage_max_exclusive"])
    return {"event_id": capsule.event.event_id, "capsule_id": capsule.capsule_id, "source_sequence": capsule.source_sequence, "truth": capsule.event.cause, "event_window": {"start_frame": capsule.event.start_frame, "end_frame": capsule.event.end_frame}, "first_event_evidence": features, "activated": activated, "run": {"output": output.name, "runner_commit": run["runner"].get("commit"), "observer_wall_seconds": run["observer_wall_seconds"], "runtime_seconds": run["runtime_seconds"], "peak_memory_allocated_mib": run["peak_memory_allocated_mib"]}, "decisions": {method: decide_event(rows, method=method, start_frame=capsule.event.start_frame, end_frame=capsule.event.end_frame) for method in METHODS}}


def _gates(events: Sequence[Mapping[str, Any]], limits: Mapping[str, Any]) -> dict[str, bool]:
    return {"all_12_valid_events": len(events) == int(limits["valid_event_count"]), "activation": sum(bool(event["activated"]) for event in events) >= int(limits["activated_event_min"]), "input_and_output_provenance": True}


def _freeze(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def evaluate(*, protocol_path: Path, output_dir: Path) -> Path:
    protocol_path, output_dir = protocol_path.resolve(strict=True), output_dir.resolve(strict=False)
    if protocol_path.parent != ROOT / "docs" / "protocols" or output_dir.parent != ROOT / "outputs" or output_dir.exists():
        raise Stage0p9CalibrationEvaluationError("protocol/output location is invalid")
    protocol = _protocol(protocol_path)
    events = [_event(item, protocol) for item in protocol["calibration_inventory"]]
    gates = _gates(events, protocol["gates"])
    result = {"schema_version": RESULT_SCHEMA_VERSION, "status": "P0.9_HASHED_TILE_CALIBRATION_ACCEPTED" if all(gates.values()) else "P0.9_HASHED_TILE_CALIBRATION_REJECTED", "evaluated_at": datetime.now().astimezone().isoformat(), "protocol": {"path": str(protocol_path.relative_to(ROOT)), "sha256": _sha(protocol_path)}, "decision_contract": DECISION_CONTRACT, "interaction_contract": INTERACTION_CONTRACT, "events": events, "activation_count": sum(bool(event["activated"]) for event in events), "gates": gates, "scope": "hashed-tile recipe-feasibility calibration only; no final benchmark score or method claim"}
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".staging", dir=output_dir.parent))
    published = False
    try:
        (staging / "result.json").write_bytes(_canonical(result))
        _freeze(staging)
        os.replace(staging, output_dir)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs" / "protocols" / "state-triage-v2-stage0p9-calibration-gate-b.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "state-triage-v2-stage0p9-calibration-evaluation-0001")
    args = parser.parse_args(argv)
    try:
        print(evaluate(protocol_path=args.protocol, output_dir=args.output_dir))
    except (Stage0p9CalibrationEvaluationError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
