#!/usr/bin/env python3
"""Evaluate exactly one frozen Stage 0.6 pilot after all outputs exist."""

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
PROTOCOL_SCHEMA_VERSION = "stateguard3r.state-triage-stage0p6-gate-b.v1"
RESULT_SCHEMA_VERSION = "stateguard3r.state-triage-stage0p6-evaluation.v1"
sys.path.insert(0, str(ROOT / "src"))
from stateguard3r.state_triage_capsule_v2 import load_stage0_capsule  # noqa: E402
from stateguard3r.state_triage_stage0p6 import CAUSES, DECISION_CONTRACT, METHODS, cause_metrics, decide_event  # noqa: E402


class Stage0p6EvaluationError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    def bad_constant(value: str) -> None:
        raise Stage0p6EvaluationError(f"non-finite JSON constant {value!r}")

    def duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise Stage0p6EvaluationError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=bad_constant, object_pairs_hook=duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, Stage0p6EvaluationError) as error:
        raise Stage0p6EvaluationError(f"cannot load strict JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Stage0p6EvaluationError(f"{path} must contain an object")
    return value


def _read_only(path: Path, *, directory: bool = False) -> None:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or (not stat.S_ISDIR(metadata.st_mode) if directory else not stat.S_ISREG(metadata.st_mode)) or stat.S_IMODE(metadata.st_mode) & 0o222:
        raise Stage0p6EvaluationError(f"{path} must be a read-only {'directory' if directory else 'file'}")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _direct(relative: str, parent: Path) -> Path:
    path = (ROOT / relative).resolve(strict=True)
    if path.parent != parent:
        raise Stage0p6EvaluationError(f"{relative} is outside approved parent {parent}")
    return path


def _protocol(path: Path) -> dict[str, Any]:
    _read_only(path)
    value = _json(path)
    required = {"schema_version", "stage", "input_inventory", "reused_stage0_control_evidence", "frozen_sources", "frozen_evidence_config", "decision", "evaluation_inventory", "gates"}
    if set(value) != required or value.get("schema_version") != PROTOCOL_SCHEMA_VERSION or value.get("stage") != "GATE_B_STAGE0P6_FROZEN_PRE_EVALUATION":
        raise Stage0p6EvaluationError("protocol schema differs")
    input_item = value["input_inventory"]
    if not isinstance(input_item, Mapping) or set(input_item) != {"path", "sha256"}:
        raise Stage0p6EvaluationError("input inventory schema differs")
    input_dir = _direct(str(input_item["path"]), ROOT / "outputs")
    _read_only(input_dir, directory=True)
    if _sha256(input_dir / "protocol.json") != input_item["sha256"]:
        raise Stage0p6EvaluationError("input manifest hash differs")
    reuse = value["reused_stage0_control_evidence"]
    if not isinstance(reuse, Mapping) or set(reuse) != {"audit_path", "audit_sha256", "scope"}:
        raise Stage0p6EvaluationError("reuse schema differs")
    audit = _direct(str(reuse["audit_path"]), ROOT / "docs" / "audits")
    if _sha256(audit) != reuse["audit_sha256"]:
        raise Stage0p6EvaluationError("prior-stage audit differs")
    sources = value["frozen_sources"]
    if not isinstance(sources, Mapping) or set(sources) != {"runner_path", "runner_sha256", "observer_path", "observer_sha256"}:
        raise Stage0p6EvaluationError("frozen sources schema differs")
    runner, observer = _direct(str(sources["runner_path"]), ROOT / "scripts"), _direct(str(sources["observer_path"]), ROOT / "src" / "stateguard3r")
    if _sha256(runner) != sources["runner_sha256"] or _sha256(observer) != sources["observer_sha256"]:
        raise Stage0p6EvaluationError("runner/observer changed; required paired controls are absent")
    if value["decision"] != DECISION_CONTRACT or not isinstance(value["frozen_evidence_config"], Mapping):
        raise Stage0p6EvaluationError("decision or evidence contract differs")
    events = value["evaluation_inventory"]
    if not isinstance(events, list) or len(events) != 12:
        raise Stage0p6EvaluationError("evaluation inventory must be 12 events")
    required_event = {"capsule_path", "capsule_sha256", "capsule_id", "output"}
    if any(not isinstance(item, Mapping) or set(item) != required_event for item in events) or len({item["capsule_id"] for item in events}) != 12 or len({item["output"] for item in events}) != 12:
        raise Stage0p6EvaluationError("event inventory schema/uniqueness differs")
    gates = value["gates"]
    required_gates = {"typed_macro_f1_min", "typed_minus_scalar_macro_f1_min", "each_cause_recall_min", "normal_novelty_false_reject_max", "unsafe_confusion_count_max", "valid_event_count"}
    if not isinstance(gates, Mapping) or set(gates) != required_gates or gates["valid_event_count"] != 12 or gates["unsafe_confusion_count_max"] != 1:
        raise Stage0p6EvaluationError("gate schema differs")
    for name in required_gates - {"valid_event_count", "unsafe_confusion_count_max"}:
        if isinstance(gates[name], bool) or not isinstance(gates[name], (int, float)) or not math.isfinite(float(gates[name])):
            raise Stage0p6EvaluationError(f"gate {name} must be finite")
    return value


def _event(item: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    capsule_path = _direct(str(item["capsule_path"]), ROOT / "outputs" / "state-triage-v2-stage0p6-inputs-0001" / "evaluation")
    _read_only(capsule_path)
    if _sha256(capsule_path) != item["capsule_sha256"]:
        raise Stage0p6EvaluationError("capsule hash differs")
    capsule = load_stage0_capsule(capsule_path)
    if capsule.capsule_id != item["capsule_id"]:
        raise Stage0p6EvaluationError("capsule ID differs")
    output = _direct(f"outputs/{item['output']}", ROOT / "outputs")
    _read_only(output, directory=True)
    run, evidence = _json(output / "run.json"), _json(output / "evidence.json")
    required = {"status", "observer", "input_capsule", "frame_count", "runner", "observer_wall_seconds", "runtime_seconds", "peak_memory_allocated_mib"}
    if run.get("status") != "succeeded" or run.get("observer") != "on" or not required.issubset(run) or not isinstance(run["input_capsule"], Mapping) or run["input_capsule"].get("sha256") != item["capsule_sha256"] or run["frame_count"] != len(capsule.frames):
        raise Stage0p6EvaluationError("run provenance differs")
    if evidence.get("config") != protocol["frozen_evidence_config"] or evidence.get("causality") != "each row accepts current frame plus strict observer prefix only":
        raise Stage0p6EvaluationError("evidence contract differs")
    rows = evidence.get("rows")
    if not isinstance(rows, list) or len(rows) != len(capsule.frames) or [row.get("frame_id") for row in rows if isinstance(row, Mapping)] != list(range(len(rows))):
        raise Stage0p6EvaluationError("evidence rows differ")
    return {"event_id": capsule.event.event_id, "capsule_id": capsule.capsule_id, "source_sequence": capsule.source_sequence, "truth": capsule.event.cause, "event_window": {"start_frame": capsule.event.start_frame, "end_frame": capsule.event.end_frame}, "run": {"output": output.name, "runner_commit": run["runner"].get("commit"), "observer_wall_seconds": run["observer_wall_seconds"], "runtime_seconds": run["runtime_seconds"], "peak_memory_allocated_mib": run["peak_memory_allocated_mib"]}, "decisions": {method: decide_event(rows, method=method, start_frame=capsule.event.start_frame, end_frame=capsule.event.end_frame) for method in METHODS}}


def _method_events(events: Sequence[Mapping[str, Any]], method: str) -> list[dict[str, Any]]:
    return [{"event_id": event["event_id"], "source_sequence": event["source_sequence"], "truth": event["truth"], "prediction": event["decisions"][method]["label"], "detection_delay_frames": event["decisions"][method]["detection_delay_frames"]} for event in events]


def _gates(metrics: Mapping[str, Any], limits: Mapping[str, Any]) -> dict[str, bool]:
    typed, scalar = metrics["T0.6"], metrics["S0.6"]
    return {"all_12_valid_events": sum(int(item["support"]) for item in typed["per_cause"].values()) == limits["valid_event_count"], "typed_macro_f1": typed["macro_f1"] >= float(limits["typed_macro_f1_min"]), "typed_beats_scalar": typed["macro_f1"] - scalar["macro_f1"] >= float(limits["typed_minus_scalar_macro_f1_min"]), "each_cause_recall": all(typed["per_cause"][cause]["recall"] >= float(limits["each_cause_recall_min"]) for cause in CAUSES), "normal_novelty_false_reject": typed["normal_novelty_false_reject"] <= float(limits["normal_novelty_false_reject_max"]), "unsafe_confusions": all(count <= int(limits["unsafe_confusion_count_max"]) for count in typed["unsafe_confusion_counts"].values()), "input_and_output_provenance": True}


def _freeze(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def evaluate(*, protocol_path: Path, output_dir: Path) -> Path:
    protocol_path, output_dir = protocol_path.resolve(strict=True), output_dir.resolve(strict=False)
    if protocol_path.parent != ROOT / "docs" / "protocols" or output_dir.parent != ROOT / "outputs" or output_dir.exists():
        raise Stage0p6EvaluationError("protocol/output location is invalid")
    protocol = _protocol(protocol_path)
    events = [_event(item, protocol) for item in protocol["evaluation_inventory"]]
    metrics = {method: cause_metrics(_method_events(events, method)) for method in METHODS}
    gates = _gates(metrics, protocol["gates"])
    result = {"schema_version": RESULT_SCHEMA_VERSION, "status": "STATE_TRIAGE_V2_STAGE0P6_PILOT_GO" if all(gates.values()) else "STATE_TRIAGE_V2_STAGE0P6_PILOT_NO_GO", "evaluated_at": datetime.now().astimezone().isoformat(), "protocol": {"path": str(protocol_path.relative_to(ROOT)), "sha256": _sha256(protocol_path)}, "decision_contract": DECISION_CONTRACT, "events": events, "metrics": metrics, "gates": gates, "interpretation_limit": "development screening pilot only; no action, recovery, persistent-state, generalization, or metric-improvement claim"}
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
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs" / "protocols" / "state-triage-v2-stage0p6-gate-b.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "state-triage-v2-stage0p6-evaluation-0001")
    args = parser.parse_args(argv)
    try:
        print(evaluate(protocol_path=args.protocol, output_dir=args.output_dir))
    except (Stage0p6EvaluationError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
