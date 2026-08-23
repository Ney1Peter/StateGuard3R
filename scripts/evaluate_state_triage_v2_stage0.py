#!/usr/bin/env python3
"""Evaluate one fully frozen StateTriage3R v2 Stage-0 development matrix."""

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
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stateguard3r.state_triage_capsule_v2 import load_stage0_capsule
from stateguard3r.state_triage_stage0 import (
    CAUSES,
    METHODS,
    DecisionThresholds,
    Stage0DecisionError,
    cause_metrics,
    decide_event,
    sequence_bootstrap_delta,
)


SCHEMA_VERSION = "stateguard3r.state-triage-stage0-evaluation.v1"
PROTOCOL_SCHEMA_VERSION = "stateguard3r.state-triage-stage0-gate-b.v1"
NATIVE_ARTIFACTS = ("checkpoint-load-audit.json", "predictions-summary.json", "trajectory.json", "health.jsonl")


class Stage0EvaluationError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise Stage0EvaluationError(f"non-finite JSON constant {value!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Stage0EvaluationError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, Stage0EvaluationError) as error:
        raise Stage0EvaluationError(f"cannot load strict JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise Stage0EvaluationError(f"{path} must contain a JSON object")
    return payload


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _read_only(path: Path, *, directory: bool = False) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise Stage0EvaluationError(f"cannot stat {path}") from error
    expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if not expected or stat.S_ISLNK(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o222:
        raise Stage0EvaluationError(f"{path} is not a read-only {'directory' if directory else 'regular file'}")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise Stage0EvaluationError(f"{label} must be finite")
    return float(value)


def _git_head() -> str:
    try:
        return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise Stage0EvaluationError("cannot record evaluator commit") from error


def _protocol(path: Path) -> dict[str, Any]:
    value = _strict_json(path)
    required = {
        "schema_version",
        "stage",
        "input_inventory",
        "control_inventory",
        "frozen_evidence_config",
        "schema_calibration_outputs",
        "control_pairs",
        "decision",
        "evaluation_inventory",
        "gates",
    }
    if set(value) != required or value.get("schema_version") != PROTOCOL_SCHEMA_VERSION or value.get("stage") != "GATE_B_FROZEN_PRE_EVALUATION":
        raise Stage0EvaluationError("Gate-B protocol schema differs")
    for name in ("input_inventory", "control_inventory"):
        artifact = value[name]
        if not isinstance(artifact, Mapping) or set(artifact) != {"path", "sha256"}:
            raise Stage0EvaluationError(f"protocol {name} artifact differs")
        target = (ROOT / str(artifact["path"])).resolve(strict=True)
        _read_only(target)
        if _sha256(target) != artifact["sha256"]:
            raise Stage0EvaluationError(f"protocol {name} digest differs")
    if not isinstance(value["frozen_evidence_config"], Mapping):
        raise Stage0EvaluationError("frozen evidence config differs")
    if not isinstance(value["schema_calibration_outputs"], list) or len(value["schema_calibration_outputs"]) != 3:
        raise Stage0EvaluationError("schema calibration output inventory differs")
    if not isinstance(value["control_pairs"], list) or len(value["control_pairs"]) != 4:
        raise Stage0EvaluationError("control pair inventory differs")
    if not isinstance(value["evaluation_inventory"], list) or len(value["evaluation_inventory"]) != 20:
        raise Stage0EvaluationError("evaluation inventory must contain exactly 20 events")
    if not isinstance(value["decision"], Mapping) or set(value["decision"]) != {"primary_scalar_baseline", "thresholds", "future_k", "bootstrap_samples", "bootstrap_seed"}:
        raise Stage0EvaluationError("decision protocol differs")
    if value["decision"]["primary_scalar_baseline"] not in ("S0", "S1"):
        raise Stage0EvaluationError("primary scalar baseline must be S0 or S1")
    DecisionThresholds.from_mapping(value["decision"]["thresholds"])
    if type(value["decision"]["future_k"]) is not int or value["decision"]["future_k"] < 0:
        raise Stage0EvaluationError("future K differs")
    if type(value["decision"]["bootstrap_samples"]) is not int or value["decision"]["bootstrap_samples"] < 1 or type(value["decision"]["bootstrap_seed"]) is not int:
        raise Stage0EvaluationError("bootstrap protocol differs")
    required_gates = {"macro_f1_margin", "bootstrap_lower_95_min", "normal_novelty_false_reject_max", "unsafe_confusion_increase_max", "observer_overhead_ratio_max", "gpu_peak_delta_mib_max"}
    if not isinstance(value["gates"], Mapping) or set(value["gates"]) != required_gates:
        raise Stage0EvaluationError("Gate-E thresholds differ")
    for name in required_gates:
        _finite(value["gates"][name], f"gate {name}")
    return value


def _output(path_name: str) -> Path:
    path = (ROOT / "outputs" / path_name).resolve(strict=True)
    if path.parent != ROOT / "outputs":
        raise Stage0EvaluationError("output must be a direct child of outputs")
    _read_only(path, directory=True)
    return path


def _load_run(path: Path) -> dict[str, Any]:
    run = _strict_json(path / "run.json")
    required = {"status", "observer", "input_capsule", "frame_count", "observer_wall_seconds", "runtime_seconds", "peak_memory_allocated_mib", "runner"}
    if run.get("status") != "succeeded" or not required.issubset(run):
        raise Stage0EvaluationError(f"run metadata is incomplete: {path}")
    for name in ("frame_count",):
        if type(run[name]) is not int or run[name] < 1:
            raise Stage0EvaluationError(f"run {name} is invalid")
    for name in ("observer_wall_seconds", "runtime_seconds", "peak_memory_allocated_mib"):
        if _finite(run[name], f"run {name}") < 0.0:
            raise Stage0EvaluationError(f"run {name} is negative")
    if not isinstance(run["input_capsule"], Mapping) or not isinstance(run["runner"], Mapping):
        raise Stage0EvaluationError("run provenance differs")
    return run


def _pair(pair: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"source_sequence", "capsule_sha256", "vanilla_output", "observer_output", "native_artifact_sha256"}
    if set(pair) != expected or not isinstance(pair["native_artifact_sha256"], Mapping) or set(pair["native_artifact_sha256"]) != set(NATIVE_ARTIFACTS):
        raise Stage0EvaluationError("control pair schema differs")
    vanilla, observer = _output(str(pair["vanilla_output"])), _output(str(pair["observer_output"]))
    left, right = _load_run(vanilla), _load_run(observer)
    if left["observer"] != "off" or right["observer"] != "on":
        raise Stage0EvaluationError("control observer mode differs")
    for run in (left, right):
        if run["input_capsule"].get("sha256") != pair["capsule_sha256"]:
            raise Stage0EvaluationError("control input capsule differs")
    hashes: dict[str, str] = {}
    for name in NATIVE_ARTIFACTS:
        left_hash, right_hash = _sha256(vanilla / name), _sha256(observer / name)
        if left_hash != right_hash or left_hash != pair["native_artifact_sha256"][name]:
            raise Stage0EvaluationError(f"native control parity differs for {name}")
        hashes[name] = left_hash
    return {
        "source_sequence": pair["source_sequence"],
        "native_artifact_sha256": hashes,
        "vanilla_runtime_per_frame": left["runtime_seconds"] / left["frame_count"],
        "observer_wall_per_frame": right["observer_wall_seconds"] / right["frame_count"],
        "peak_memory_delta_mib": right["peak_memory_allocated_mib"] - left["peak_memory_allocated_mib"],
    }


def _schema_resource(item: Mapping[str, Any], expected_config: Mapping[str, Any]) -> dict[str, float]:
    if set(item) != {"output", "evidence_sha256"}:
        raise Stage0EvaluationError("schema resource item differs")
    output = _output(str(item["output"]))
    run = _load_run(output)
    evidence_path = output / "evidence.json"
    if _sha256(evidence_path) != item["evidence_sha256"]:
        raise Stage0EvaluationError("schema evidence digest differs")
    evidence = _strict_json(evidence_path)
    if evidence.get("config") != expected_config or not isinstance(evidence.get("rows"), list) or len(evidence["rows"]) != run["frame_count"]:
        raise Stage0EvaluationError("schema evidence differs")
    return {"observer_wall_per_frame": run["observer_wall_seconds"] / run["frame_count"]}


def _event(item: Mapping[str, Any], protocol: Mapping[str, Any], thresholds: DecisionThresholds) -> dict[str, Any]:
    required = {"capsule_path", "capsule_sha256", "output", "capsule_id"}
    if set(item) != required:
        raise Stage0EvaluationError("event inventory item differs")
    capsule_path = (ROOT / str(item["capsule_path"])).resolve(strict=True)
    _read_only(capsule_path)
    if _sha256(capsule_path) != item["capsule_sha256"]:
        raise Stage0EvaluationError("event capsule digest differs")
    capsule = load_stage0_capsule(capsule_path)
    if capsule.capsule_id != item["capsule_id"]:
        raise Stage0EvaluationError("event capsule ID differs")
    output = _output(str(item["output"]))
    run = _load_run(output)
    if run["observer"] != "on" or run["input_capsule"].get("sha256") != item["capsule_sha256"] or run["frame_count"] != len(capsule.frames):
        raise Stage0EvaluationError("event run provenance differs")
    evidence = _strict_json(output / "evidence.json")
    if evidence.get("config") != protocol["frozen_evidence_config"] or evidence.get("causality") != "each row accepts current frame plus strict observer prefix only":
        raise Stage0EvaluationError("event evidence configuration differs")
    rows = evidence.get("rows")
    if not isinstance(rows, list) or len(rows) != len(capsule.frames) or [row.get("frame_id") for row in rows if isinstance(row, Mapping)] != list(range(len(rows))):
        raise Stage0EvaluationError("event evidence rows differ")
    decisions = {
        method: decide_event(
            method,
            rows,
            start_frame=capsule.event.start_frame,
            end_frame=capsule.event.end_frame,
            thresholds=thresholds,
            future_k=protocol["decision"]["future_k"],
        )
        for method in METHODS
    }
    return {
        "event_id": capsule.event.event_id,
        "capsule_id": capsule.capsule_id,
        "source_sequence": capsule.source_sequence,
        "source_group": capsule.source_group,
        "truth": capsule.event.cause,
        "coverage_expectation": capsule.event.coverage_expectation,
        "run": {"output": output.name, "runner_commit": run["runner"].get("commit"), "observer_wall_per_frame": run["observer_wall_seconds"] / run["frame_count"], "peak_memory_mib": run["peak_memory_allocated_mib"]},
        "decisions": decisions,
    }


def _method_events(events: Sequence[Mapping[str, Any]], method: str) -> list[dict[str, Any]]:
    return [
        {
            "event_id": event["event_id"],
            "source_sequence": event["source_sequence"],
            "truth": event["truth"],
            "prediction": event["decisions"][method]["label"],
            "posterior": event["decisions"][method]["posterior"],
            "detection_delay_frames": event["decisions"][method]["detection_delay_frames"],
        }
        for event in events
    ]


def _gates(metrics: Mapping[str, Any], bootstrap: Mapping[str, Any], resource: Mapping[str, Any], primary: str, limits: Mapping[str, Any]) -> dict[str, bool]:
    typed, scalar = metrics["T0"], metrics[primary]
    unsafe = ("registration_as_transient", "bad_observation_as_novelty", "transient_as_registration")
    return {
        "control_parity": bool(resource["control_parity"]),
        "macro_f1_margin": typed["macro_f1"] - scalar["macro_f1"] >= float(limits["macro_f1_margin"]),
        "bootstrap_lower_positive": bootstrap["lower_95"] > float(limits["bootstrap_lower_95_min"]),
        "normal_recall_non_decrease": typed["per_cause"]["normal_novelty"]["recall"] >= scalar["per_cause"]["normal_novelty"]["recall"],
        "normal_false_reject": typed["unsafe_confusions"]["normal_novelty_false_reject"] <= float(limits["normal_novelty_false_reject_max"]),
        "unsafe_confusions": all(typed["unsafe_confusions"][name] <= scalar["unsafe_confusions"][name] + float(limits["unsafe_confusion_increase_max"]) for name in unsafe),
        "observer_overhead": resource["median_observer_to_vanilla_ratio"] <= float(limits["observer_overhead_ratio_max"]),
        "gpu_peak_delta": resource["max_control_gpu_peak_delta_mib"] <= float(limits["gpu_peak_delta_mib_max"]),
    }


def _freeze(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def evaluate(protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    protocol_path = protocol_path.resolve(strict=True)
    output_dir = output_dir.resolve(strict=False)
    if output_dir.parent != ROOT / "outputs" or output_dir.exists():
        raise Stage0EvaluationError("output must be a new direct child of StateGuard3R/outputs")
    protocol = _protocol(protocol_path)
    thresholds = DecisionThresholds.from_mapping(protocol["decision"]["thresholds"])
    controls = [_pair(item) for item in protocol["control_pairs"]]
    schema_resources = [_schema_resource(item, protocol["frozen_evidence_config"]) for item in protocol["schema_calibration_outputs"]]
    events = [_event(item, protocol, thresholds) for item in protocol["evaluation_inventory"]]
    if len({event["event_id"] for event in events}) != len(events) or {event["truth"] for event in events} != set(CAUSES):
        raise Stage0EvaluationError("event inventory lacks unique coverage of all causes")
    by_method = {method: _method_events(events, method) for method in METHODS}
    metrics = {method: cause_metrics(rows) for method, rows in by_method.items()}
    primary = protocol["decision"]["primary_scalar_baseline"]
    bootstrap = sequence_bootstrap_delta(by_method["T0"], by_method[primary], samples=protocol["decision"]["bootstrap_samples"], seed=protocol["decision"]["bootstrap_seed"])
    vanilla = np.asarray([item["vanilla_runtime_per_frame"] for item in controls], dtype=np.float64)
    observers = np.asarray([item["observer_wall_per_frame"] for item in schema_resources], dtype=np.float64)
    resource = {
        "control_parity": True,
        "control_pairs": controls,
        "schema_observer_wall_per_frame": [float(value) for value in observers],
        "median_schema_observer_wall_per_frame": float(np.median(observers)),
        "median_control_vanilla_runtime_per_frame": float(np.median(vanilla)),
        "median_observer_to_vanilla_ratio": float(np.median(observers) / np.median(vanilla)),
        "max_control_gpu_peak_delta_mib": float(max(item["peak_memory_delta_mib"] for item in controls)),
        "event_observer_wall_per_frame": [event["run"]["observer_wall_per_frame"] for event in events],
        "event_peak_memory_mib": [event["run"]["peak_memory_mib"] for event in events],
    }
    gates = _gates(metrics, bootstrap, resource, primary, protocol["gates"])
    decision = "STATE_TRIAGE_V2_STAGE0_GO" if all(gates.values()) else "STATE_TRIAGE_V2_STAGE0_NO_GO"
    result = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "attempt_seal": "single evaluation of the frozen 20-event development matrix; thresholds and inventory were not changed after outputs existed",
        "evaluated_at": datetime.now().astimezone().isoformat(),
        "evaluator_commit": _git_head(),
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "methods": metrics,
        "primary_scalar_baseline": primary,
        "typed_minus_primary_scalar_macro_f1": metrics["T0"]["macro_f1"] - metrics[primary]["macro_f1"],
        "sequence_bootstrap_t0_minus_primary": bootstrap,
        "resources": resource,
        "gates": gates,
        "events": events,
        "limitations": ["development-only: all local TUM sequences were previously disclosed", "registration/order, bad-observation, and transient labels are deterministic proxies", "T0+K is an offline upper bound and is not an online claim", "persistent_change is excluded because there is no audited revisit-change label"],
    }
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".staging", dir=output_dir.parent))
    published = False
    try:
        (staging / "result.json").write_bytes(_canonical(result))
        seal = {"schema_version": "stateguard3r.state-triage-stage0-attempt-seal.v1", "result_sha256": _sha256(staging / "result.json"), "decision": decision, "protocol_sha256": _sha256(protocol_path)}
        (staging / "attempt-seal.json").write_bytes(_canonical(seal))
        _freeze(staging)
        os.replace(staging, output_dir)
        published = True
    finally:
        if not published and staging.exists():
            for path in staging.rglob("*"):
                path.chmod(0o700)
            shutil.rmtree(staging)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(evaluate(args.protocol, args.output_dir), ensure_ascii=False, indent=2, allow_nan=False))
    except (Stage0EvaluationError, Stage0DecisionError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
