#!/usr/bin/env python3
"""Evaluate exactly one immutable, pre-registered StateTriage3R Stage 0.5 pilot."""

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
PROTOCOL_SCHEMA_VERSION = "stateguard3r.state-triage-stage0p5-gate-b.v1"
RESULT_SCHEMA_VERSION = "stateguard3r.state-triage-stage0p5-evaluation.v1"

sys.path.insert(0, str(ROOT / "src"))
from stateguard3r.state_triage_capsule_v2 import load_stage0_capsule  # noqa: E402
from stateguard3r.state_triage_stage0p5 import CAUSES, DECISION_CONTRACT, METHODS, cause_metrics, decide_event  # noqa: E402


class Stage0p5EvaluationError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise Stage0p5EvaluationError(f"non-finite JSON constant {value!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Stage0p5EvaluationError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, Stage0p5EvaluationError) as error:
        raise Stage0p5EvaluationError(f"cannot load strict JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Stage0p5EvaluationError(f"{path} must contain a JSON object")
    return value


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _read_only(path: Path, *, directory: bool = False) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise Stage0p5EvaluationError(f"cannot stat {path}") from error
    expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if not expected or stat.S_ISLNK(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o222:
        raise Stage0p5EvaluationError(f"{path} is not a read-only {'directory' if directory else 'file'}")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise Stage0p5EvaluationError(f"{name} must be finite")
    return float(value)


def _under_root(relative: str, parent: Path) -> Path:
    target = (ROOT / relative).resolve(strict=True)
    if target.parent != parent:
        raise Stage0p5EvaluationError(f"{relative} is outside its approved location")
    return target


def _protocol(path: Path) -> dict[str, Any]:
    _read_only(path)
    value = _strict_json(path)
    required = {
        "schema_version",
        "stage",
        "input_inventory",
        "reused_stage0_control_evidence",
        "frozen_sources",
        "frozen_evidence_config",
        "decision",
        "evaluation_inventory",
        "gates",
    }
    if set(value) != required or value.get("schema_version") != PROTOCOL_SCHEMA_VERSION or value.get("stage") != "GATE_B_STAGE0P5_FROZEN_PRE_EVALUATION":
        raise Stage0p5EvaluationError("Stage 0.5 protocol schema differs")
    inventory = value["input_inventory"]
    if not isinstance(inventory, Mapping) or set(inventory) != {"path", "sha256"}:
        raise Stage0p5EvaluationError("input inventory schema differs")
    input_dir = _under_root(str(inventory["path"]), ROOT / "outputs")
    _read_only(input_dir, directory=True)
    if _sha256(input_dir / "protocol.json") != inventory["sha256"]:
        raise Stage0p5EvaluationError("input inventory digest differs")
    reuse = value["reused_stage0_control_evidence"]
    if not isinstance(reuse, Mapping) or set(reuse) != {"audit_path", "audit_sha256", "scope"}:
        raise Stage0p5EvaluationError("reused control evidence schema differs")
    audit = _under_root(str(reuse["audit_path"]), ROOT / "docs" / "audits")
    if _sha256(audit) != reuse["audit_sha256"]:
        raise Stage0p5EvaluationError("frozen Stage 0 audit digest differs")
    sources = value["frozen_sources"]
    if not isinstance(sources, Mapping) or set(sources) != {"runner_path", "runner_sha256", "observer_path", "observer_sha256"}:
        raise Stage0p5EvaluationError("frozen source schema differs")
    runner = _under_root(str(sources["runner_path"]), ROOT / "scripts")
    observer = _under_root(str(sources["observer_path"]), ROOT / "src" / "stateguard3r")
    if _sha256(runner) != sources["runner_sha256"] or _sha256(observer) != sources["observer_sha256"]:
        raise Stage0p5EvaluationError("runner/observer sources changed; paired controls must be rerun before pilot evaluation")
    if not isinstance(value["frozen_evidence_config"], Mapping):
        raise Stage0p5EvaluationError("frozen evidence config differs")
    if value["decision"] != DECISION_CONTRACT:
        raise Stage0p5EvaluationError("fixed Stage 0.5 decision contract differs")
    events = value["evaluation_inventory"]
    if not isinstance(events, list) or len(events) != 12:
        raise Stage0p5EvaluationError("evaluation inventory must contain exactly 12 events")
    seen_capsules, seen_outputs = set(), set()
    for item in events:
        if not isinstance(item, Mapping) or set(item) != {"capsule_path", "capsule_sha256", "capsule_id", "output"}:
            raise Stage0p5EvaluationError("evaluation item schema differs")
        if item["capsule_id"] in seen_capsules or item["output"] in seen_outputs:
            raise Stage0p5EvaluationError("evaluation inventory duplicates an event")
        seen_capsules.add(item["capsule_id"])
        seen_outputs.add(item["output"])
    gates = value["gates"]
    required_gates = {"typed_macro_f1_min", "typed_minus_scalar_macro_f1_min", "each_cause_recall_min", "normal_novelty_false_reject_max", "unsafe_confusion_count_max", "valid_event_count"}
    if not isinstance(gates, Mapping) or set(gates) != required_gates or gates["valid_event_count"] != 12 or gates["unsafe_confusion_count_max"] != 1:
        raise Stage0p5EvaluationError("pilot gate schema differs")
    for name in required_gates - {"valid_event_count", "unsafe_confusion_count_max"}:
        _finite(gates[name], f"gate {name}")
    return value


def _output(name: str) -> Path:
    target = (ROOT / "outputs" / name).resolve(strict=True)
    if target.parent != ROOT / "outputs":
        raise Stage0p5EvaluationError("run output must be a direct child of outputs")
    _read_only(target, directory=True)
    return target


def _event(item: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    capsule_path = _under_root(str(item["capsule_path"]), ROOT / "outputs" / "state-triage-v2-stage0p5-inputs-0001" / "evaluation")
    _read_only(capsule_path)
    if _sha256(capsule_path) != item["capsule_sha256"]:
        raise Stage0p5EvaluationError("capsule digest differs")
    capsule = load_stage0_capsule(capsule_path)
    if capsule.capsule_id != item["capsule_id"]:
        raise Stage0p5EvaluationError("capsule ID differs")
    output = _output(str(item["output"]))
    run = _strict_json(output / "run.json")
    required_run = {"status", "observer", "input_capsule", "frame_count", "runner", "observer_wall_seconds", "runtime_seconds", "peak_memory_allocated_mib"}
    if run.get("status") != "succeeded" or not required_run.issubset(run) or run.get("observer") != "on":
        raise Stage0p5EvaluationError("event run metadata is incomplete")
    if not isinstance(run["input_capsule"], Mapping) or run["input_capsule"].get("sha256") != item["capsule_sha256"] or run["frame_count"] != len(capsule.frames):
        raise Stage0p5EvaluationError("event run input provenance differs")
    for name in ("observer_wall_seconds", "runtime_seconds", "peak_memory_allocated_mib"):
        if _finite(run[name], f"run {name}") < 0.0:
            raise Stage0p5EvaluationError(f"run {name} is negative")
    evidence = _strict_json(output / "evidence.json")
    if evidence.get("config") != protocol["frozen_evidence_config"] or evidence.get("causality") != "each row accepts current frame plus strict observer prefix only":
        raise Stage0p5EvaluationError("event evidence contract differs")
    rows = evidence.get("rows")
    if not isinstance(rows, list) or len(rows) != len(capsule.frames) or [row.get("frame_id") for row in rows if isinstance(row, Mapping)] != list(range(len(rows))):
        raise Stage0p5EvaluationError("event evidence rows differ")
    decisions = {method: decide_event(rows, method=method, start_frame=capsule.event.start_frame, end_frame=capsule.event.end_frame) for method in METHODS}
    return {
        "event_id": capsule.event.event_id,
        "capsule_id": capsule.capsule_id,
        "source_sequence": capsule.source_sequence,
        "truth": capsule.event.cause,
        "event_window": {"start_frame": capsule.event.start_frame, "end_frame": capsule.event.end_frame},
        "run": {"output": output.name, "runner_commit": run["runner"].get("commit"), "observer_wall_seconds": run["observer_wall_seconds"], "runtime_seconds": run["runtime_seconds"], "peak_memory_allocated_mib": run["peak_memory_allocated_mib"]},
        "decisions": decisions,
    }


def _method_events(events: Sequence[Mapping[str, Any]], method: str) -> list[dict[str, Any]]:
    return [
        {
            "event_id": event["event_id"],
            "source_sequence": event["source_sequence"],
            "truth": event["truth"],
            "prediction": event["decisions"][method]["label"],
            "detection_delay_frames": event["decisions"][method]["detection_delay_frames"],
        }
        for event in events
    ]


def _gates(metrics: Mapping[str, Any], limits: Mapping[str, Any]) -> dict[str, bool]:
    typed, scalar = metrics["T0.5"], metrics["S0.5"]
    return {
        "all_12_valid_events": sum(int(item["support"]) for item in typed["per_cause"].values()) == limits["valid_event_count"],
        "typed_macro_f1": typed["macro_f1"] >= float(limits["typed_macro_f1_min"]),
        "typed_beats_scalar": typed["macro_f1"] - scalar["macro_f1"] >= float(limits["typed_minus_scalar_macro_f1_min"]),
        "each_cause_recall": all(typed["per_cause"][cause]["recall"] >= float(limits["each_cause_recall_min"]) for cause in CAUSES),
        "normal_novelty_false_reject": typed["normal_novelty_false_reject"] <= float(limits["normal_novelty_false_reject_max"]),
        "unsafe_confusions": all(count <= int(limits["unsafe_confusion_count_max"]) for count in typed["unsafe_confusion_counts"].values()),
        "input_and_output_provenance": True,
    }


def _freeze(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def evaluate(*, protocol_path: Path, output_dir: Path) -> Path:
    protocol_path = protocol_path.resolve(strict=True)
    output_dir = output_dir.resolve(strict=False)
    if protocol_path.parent != ROOT / "docs" / "protocols" or output_dir.parent != ROOT / "outputs" or output_dir.exists():
        raise Stage0p5EvaluationError("protocol/output path is not an approved new Stage 0.5 location")
    protocol = _protocol(protocol_path)
    events = [_event(item, protocol) for item in protocol["evaluation_inventory"]]
    metrics = {method: cause_metrics(_method_events(events, method)) for method in METHODS}
    gates = _gates(metrics, protocol["gates"])
    status = "STATE_TRIAGE_V2_STAGE0P5_PILOT_GO" if all(gates.values()) else "STATE_TRIAGE_V2_STAGE0P5_PILOT_NO_GO"
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "evaluated_at": datetime.now().astimezone().isoformat(),
        "protocol": {"path": str(protocol_path.relative_to(ROOT)), "sha256": _sha256(protocol_path)},
        "decision_contract": DECISION_CONTRACT,
        "events": events,
        "metrics": metrics,
        "gates": gates,
        "interpretation_limit": "development screening pilot only; no action, recovery, persistent-state, generalization, or metric-improvement claim",
    }
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
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs" / "protocols" / "state-triage-v2-stage0p5-gate-b.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "state-triage-v2-stage0p5-evaluation-0001")
    args = parser.parse_args(argv)
    try:
        print(evaluate(protocol_path=args.protocol, output_dir=args.output_dir))
    except (Stage0p5EvaluationError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
