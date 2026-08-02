#!/usr/bin/env python3
"""Byte-bound CPU-only diagnosis of the disclosed formal-v1 detector result.

The v1 holdout is disclosed development evidence.  This tool never writes to
its inputs, never imports a CUDA runtime, and refuses to overwrite an output
directory.  It replays the frozen v1 scoring configuration against the six
immutable ledgers, verifies the stored decision byte-for-byte, and records
position-level errors plus rank-only signal ablations for Detector v2 design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import run_formal_detection_pilot as formal_v1
from stateguard3r import detection, detection_suite


EXPECTED_EVALUATION_MANIFEST_SHA256 = (
    "ce17d98ac000d9b1e5df8852553708d03990ba61f42fba743e9626d850792c15"
)
ANALYSIS_SCHEMA_VERSION = "stateguard3r.detector-v2-failure-analysis.v1"
CANONICAL_SIGNALS = tuple(spec.name for spec in detection.SIGNAL_SPECS)


class FailureAnalysisError(ValueError):
    """Raised when the frozen v1 evidence cannot be safely replayed."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode(
        "utf-8"
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular(path: Path, *, name: str, mode: int | None = None) -> bytes:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise FailureAnalysisError(f"cannot read {name}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise FailureAnalysisError(f"{name} must be a regular non-symlink file")
    if mode is not None and stat.S_IMODE(metadata.st_mode) != mode:
        raise FailureAnalysisError(f"{name} must be frozen at mode {mode:04o}")
    try:
        return path.read_bytes()
    except OSError as error:
        raise FailureAnalysisError(f"cannot read {name}: {error}") from error


def _json_object(payload: bytes, *, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise FailureAnalysisError(f"{name} contains non-finite JSON constant {value!r}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FailureAnalysisError(f"{name} has duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, FailureAnalysisError) as error:
        raise FailureAnalysisError(f"{name} is not strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise FailureAnalysisError(f"{name} must contain a JSON object")
    return value


def _artifact_record(
    path: Path,
    *,
    name: str,
    expected: Mapping[str, Any],
    mode: int | None = 0o444,
) -> dict[str, Any]:
    payload = _read_regular(path, name=name, mode=mode)
    actual = {"path": str(path), "size_bytes": len(payload), "sha256": _sha256(payload)}
    if actual["size_bytes"] != expected.get("size_bytes") or actual["sha256"] != expected.get("sha256"):
        raise FailureAnalysisError(
            f"{name} hash/size differs from formal-v1 evaluation manifest"
        )
    return actual


def _require_frozen_directory(path: Path, *, name: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise FailureAnalysisError(f"cannot stat {name}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise FailureAnalysisError(f"{name} must be a real directory")
    if stat.S_IMODE(metadata.st_mode) != 0o555:
        raise FailureAnalysisError(f"{name} must be frozen at mode 0555")


def _relative_frozen_path(root: Path, relative: str, *, name: str) -> Path:
    candidate = root / relative
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise FailureAnalysisError(f"{name} escapes its expected root") from error
    return resolved


def _verify_evaluation_bindings(
    *,
    input_root: Path,
    runs_root: Path,
    calibration_root: Path,
    evaluation_root: Path,
    evaluation_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Verify every artifact named by the immutable evaluation manifest."""

    _require_frozen_directory(input_root, name="formal-v1 input root")
    _require_frozen_directory(calibration_root, name="formal-v1 calibration root")
    _require_frozen_directory(evaluation_root, name="formal-v1 evaluation root")
    for run_directory in sorted(runs_root.iterdir()):
        if run_directory.is_dir():
            _require_frozen_directory(run_directory, name=f"formal-v1 run {run_directory.name}")

    inputs = evaluation_manifest.get("inputs")
    run_artifacts = evaluation_manifest.get("run_artifacts")
    generated = evaluation_manifest.get("artifacts")
    if not isinstance(inputs, Mapping) or not isinstance(run_artifacts, Mapping) or not isinstance(generated, Mapping):
        raise FailureAnalysisError("evaluation manifest lacks artifact bindings")

    locations: dict[str, Path] = {
        "input_registry": input_root,
        "split_registry": input_root,
        "holdout_commitment": input_root,
        "commitment_manifest": input_root.parent / "formal-v1-commit-0001",
        "calibration_manifest": calibration_root,
        "development_metrics": calibration_root,
        "formal_config": calibration_root,
        "search": calibration_root,
    }
    verified: list[dict[str, Any]] = []
    for key, expected in sorted(inputs.items()):
        if not isinstance(expected, Mapping) or not isinstance(expected.get("path"), str):
            raise FailureAnalysisError(f"evaluation input binding {key} is malformed")
        raw_path = Path(expected["path"])
        path = raw_path if raw_path.is_absolute() else _relative_frozen_path(
            locations[key], raw_path, name=f"evaluation input {key}"
        )
        verified.append(
            _artifact_record(
                path,
                name=f"evaluation input {key}",
                expected=expected,
                # The protocol is an immutable tracked source file, not a
                # generated formal artifact with an 0444 filesystem mode.
                mode=None if key == "protocol" else 0o444,
            )
        )

    for run_id, entries in sorted(run_artifacts.items()):
        if not isinstance(entries, Mapping):
            raise FailureAnalysisError(f"evaluation run artifacts for {run_id} are malformed")
        for artifact_name, expected in sorted(entries.items()):
            if not isinstance(expected, Mapping) or not isinstance(expected.get("path"), str):
                raise FailureAnalysisError(f"evaluation run binding {run_id}/{artifact_name} is malformed")
            path = _relative_frozen_path(runs_root, expected["path"], name=f"run binding {run_id}/{artifact_name}")
            verified.append(_artifact_record(path, name=f"run binding {run_id}/{artifact_name}", expected=expected))

    for artifact_name, expected in sorted(generated.items()):
        if artifact_name == "timelines":
            if not isinstance(expected, Mapping):
                raise FailureAnalysisError("evaluation timeline bindings are malformed")
            for timeline_name, timeline_expected in sorted(expected.items()):
                if not isinstance(timeline_expected, Mapping) or not isinstance(timeline_expected.get("path"), str):
                    raise FailureAnalysisError(f"timeline binding {timeline_name} is malformed")
                path = _relative_frozen_path(evaluation_root, timeline_expected["path"], name=f"timeline {timeline_name}")
                verified.append(_artifact_record(path, name=f"timeline {timeline_name}", expected=timeline_expected))
            continue
        if not isinstance(expected, Mapping) or not isinstance(expected.get("path"), str):
            raise FailureAnalysisError(f"evaluation artifact {artifact_name} is malformed")
        path = _relative_frozen_path(evaluation_root, expected["path"], name=f"evaluation artifact {artifact_name}")
        verified.append(_artifact_record(path, name=f"evaluation artifact {artifact_name}", expected=expected))
    return verified


def _make_runs(input_root: Path, runs_root: Path) -> tuple[dict[str, list[detection_suite.DetectionSuiteRun]], dict[str, tuple[dict[str, Any], ...]]]:
    registry = _json_object(
        _read_regular(input_root / "formal-pilot-manifest.json", name="formal-pilot-manifest.json", mode=0o444),
        name="formal-pilot-manifest.json",
    )
    raw_runs = registry.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != 6:
        raise FailureAnalysisError("formal-v1 input registry must contain six runs")

    split_runs: dict[str, list[detection_suite.DetectionSuiteRun]] = {"development": [], "holdout": []}
    records_by_run: dict[str, tuple[dict[str, Any], ...]] = {}
    for row in raw_runs:
        if not isinstance(row, Mapping):
            raise FailureAnalysisError("formal-v1 input registry contains a non-object run")
        run_id = row.get("run_id")
        split = row.get("dataset_split")
        corruption_type = row.get("corruption_type")
        raw_hashes = row.get("raw_frame_sha256s")
        if not isinstance(run_id, str) or split not in split_runs or not isinstance(corruption_type, str):
            raise FailureAnalysisError("formal-v1 run identity is malformed")
        if not isinstance(raw_hashes, list) or len(raw_hashes) != 30 or not all(isinstance(value, str) for value in raw_hashes):
            raise FailureAnalysisError(f"{run_id} has malformed raw RGB hashes")
        run_input_root = _relative_frozen_path(input_root, f"{split}/{run_id}", name=f"input root {run_id}")
        run_output_root = _relative_frozen_path(runs_root, run_id, name=f"output root {run_id}")
        source = _read_regular(run_input_root / "source-manifest.json", name=f"{run_id} source manifest", mode=0o444)
        manifest = _read_regular(run_input_root / "input-manifest.json", name=f"{run_id} input manifest", mode=0o444)
        health = _read_regular(run_output_root / "health.jsonl", name=f"{run_id} health ledger", mode=0o444)
        run_json = _read_regular(run_output_root / "run.json", name=f"{run_id} run metadata", mode=0o444)
        suite_run = detection_suite.DetectionSuiteRun(
            run_id=run_id,
            dataset_split=split,
            corruption_type=corruption_type,
            raw_frame_sha256s=tuple(raw_hashes),
            health_jsonl=health,
            corruption_json=manifest,
            input_manifest_json=manifest,
            source_manifest_json=source,
            run_json=run_json,
        )
        split_runs[split].append(suite_run)
        parsed = tuple(_json_object(line.encode("utf-8"), name=f"{run_id} health line") for line in health.decode("utf-8").splitlines() if line.strip())
        if len(parsed) != detection_suite.FRAME_COUNT:
            raise FailureAnalysisError(f"{run_id} health ledger does not have 30 records")
        records_by_run[run_id] = parsed
    return split_runs, records_by_run


def _result_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in ("schema_version", "evaluation_policy", "detection_config", "run_count", "runs", "methods")}


def _positions_for_method(run: Mapping[str, Any], method: Mapping[str, Any]) -> dict[str, Any]:
    scores = np.asarray(method["scores"], dtype=np.float64)
    threshold = float(method["metrics"]["threshold"])
    labels = np.asarray(run["primary_labels"], dtype=np.int64)
    evaluation = np.asarray(run["evaluation_mask"], dtype=bool)
    washout = np.asarray(run["washout_mask"], dtype=bool)
    if scores.shape != labels.shape or scores.shape != evaluation.shape or not np.all(np.isfinite(scores)):
        raise FailureAnalysisError(f"{run['run_id']} has invalid replay score shape/values")
    predicted = scores >= threshold
    event_start, event_end = int(run["event_start"]), int(run["event_end"])
    position_list = lambda mask: [int(index) for index in np.flatnonzero(mask)]
    return {
        "threshold": threshold,
        "scores": [float(value) for value in scores],
        "true_positive_positions": position_list(evaluation & (labels == 1) & predicted),
        "false_negative_positions": position_list(evaluation & (labels == 1) & ~predicted),
        "primary_false_positive_positions": position_list(evaluation & (labels == 0) & predicted),
        "startup_false_positive_positions": position_list((np.arange(scores.size) < event_start) & predicted),
        "event_alarm_positions": position_list((np.arange(scores.size) >= event_start) & (np.arange(scores.size) <= event_end) & predicted),
        "recovery_boundary_position": event_end + 1,
        "recovery_boundary_alarm": bool(predicted[event_end + 1]),
        "washout_alarm_positions": position_list(washout & predicted),
    }


def _rank_summary(
    runs: Sequence[detection_suite.DetectionSuiteRun],
    result: Mapping[str, Any],
    records_by_run: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recreate v1 components, then measure rank-only single/leave-one analyses."""

    config = result["detection_config"]
    components_by_run: dict[str, dict[str, np.ndarray | None]] = {}
    provenance: dict[str, Any] = {}
    run_rows = {row["run_id"]: row for row in result["runs"]}
    combined_rows = result["methods"]["combined"]["per_run"]
    for suite_run in runs:
        run_id = suite_run.run_id
        records = records_by_run[run_id]
        components: dict[str, np.ndarray | None] = {}
        details: dict[str, Any] = {}
        for spec in detection.SIGNAL_SPECS:
            resolved = detection._resolve_signal(records, spec)
            if resolved is None:
                components[spec.name] = None
                details[spec.name] = {"status": "missing", "reason": "no finite v1 ledger value"}
                continue
            component, transformation = detection._component_for_signal(
                spec.name,
                resolved,
                window=int(config["window"]),
                epsilon=float(config["epsilon"]),
                max_z=config["max_z"],
            )
            components[spec.name] = component
            details[spec.name] = {
                "status": "available",
                "field": resolved["field"],
                "direction": resolved["direction"],
                "finite_frames": int(resolved["finite_frames"]),
                "transformation": transformation,
            }
        reconstructed = np.sum(
            np.vstack(
                [np.where(np.isfinite(value), value, 0.0) for value in components.values() if value is not None]
            ),
            axis=0,
        )
        frozen = np.asarray(combined_rows[run_id]["scores"], dtype=np.float64)
        if not np.array_equal(reconstructed, frozen):
            raise FailureAnalysisError(f"{run_id} v1 combined component reconstruction differs from frozen scores")
        components_by_run[run_id] = components
        provenance[run_id] = details

    def summarize(signal: str, *, leave_one_out: bool) -> dict[str, Any]:
        per_run: dict[str, Any] = {}
        available = 0
        for suite_run in runs:
            run_id = suite_run.run_id
            selected = components_by_run[run_id][signal]
            if not leave_one_out and selected is None:
                per_run[run_id] = {"status": "unavailable", "auroc": None}
                continue
            vectors = [
                value
                for name, value in components_by_run[run_id].items()
                if value is not None and (leave_one_out or name == signal) and (not leave_one_out or name != signal)
            ]
            if not vectors:
                per_run[run_id] = {"status": "unavailable", "auroc": None}
                continue
            score = np.sum(np.vstack([np.where(np.isfinite(value), value, 0.0) for value in vectors]), axis=0)
            run = run_rows[run_id]
            labels = np.asarray(run["primary_labels"], dtype=np.int64)
            mask = np.asarray(run["evaluation_mask"], dtype=bool)
            rank = detection.auroc(labels[mask], score[mask])
            per_run[run_id] = {"status": "available", "auroc": rank}
            available += 1
        values = [row["auroc"] for row in per_run.values() if row["auroc"] is not None]
        return {
            "status": "available" if available == len(runs) else "partially_unavailable",
            "macro_auroc": float(np.mean(values)) if len(values) == len(runs) else None,
            "per_run": per_run,
        }

    return provenance, {
        "single_signal_rank": {signal: summarize(signal, leave_one_out=False) for signal in CANONICAL_SIGNALS},
        "leave_one_signal_out_rank": {signal: summarize(signal, leave_one_out=True) for signal in CANONICAL_SIGNALS},
    }


def _failure_observations(holdout: Mapping[str, Any], positions: Mapping[str, Any]) -> dict[str, Any]:
    combined = positions["holdout"]["combined"]
    reliability = holdout["methods"]["reliability_only"]
    overlap_missing = {
        run_id: "overlap" in row["missing_signals"]
        for run_id, row in holdout["methods"]["combined"]["per_run"].items()
    }
    combined_fp = sum(len(row["primary_false_positive_positions"]) for row in combined.values())
    startup_fp = sum(len(row["startup_false_positive_positions"]) for row in combined.values())
    combined_fn = sum(len(row["false_negative_positions"]) for row in combined.values())
    return {
        "combined_holdout_primary_false_positives": combined_fp,
        "combined_holdout_clean_prefix_false_positives": startup_fp,
        "combined_holdout_false_negatives": combined_fn,
        "all_holdout_combined_runs_missing_overlap": all(overlap_missing.values()),
        "overlap_missing_by_run": overlap_missing,
        "reliability_only_pooled_false_positive_rate": reliability["pooled"]["false_positive_rate"],
        "reliability_only_max_false_positive_streak": reliability["diagnostics"]["max_per_run_consecutive_false_positives"],
        "combined_holdout_macro_auroc": holdout["methods"]["combined"]["macro"]["auroc"],
    }


def _render_report(payload: Mapping[str, Any]) -> str:
    observations = payload["failure_observations"]
    checks = payload["decision_replay"]["checks"]
    failed = [row["id"] for row in checks if not row["passed"]]
    return "\n".join(
        [
            "# Detector v2: formal-v1 failure analysis",
            "",
            "## Scope",
            "",
            "This is a CPU-only replay of disclosed formal-v1 evidence. It is diagnostic development evidence, not a new formal evaluation or a claim about a blind holdout.",
            "",
            "## Byte-bound replay",
            "",
            f"- Frozen evaluation manifest: `{payload['input_evidence']['evaluation_manifest_sha256']}`",
            f"- Verified manifest-bound artifacts: `{payload['input_evidence']['verified_artifact_count']}`",
            f"- Replayed decision equals frozen `go-no-go.json`: `{payload['decision_replay']['matches_frozen_bytes']}`",
            f"- Replayed decision: `{payload['decision_replay']['decision']}`",
            f"- Failed conjunctive gate: `{', '.join(failed)}`",
            "",
            "## What the replay establishes",
            "",
            f"- Combined holdout macro-AUROC was `{observations['combined_holdout_macro_auroc']:.10f}`, below the strict 0.75 gate.",
            f"- Combined had `{observations['combined_holdout_primary_false_positives']}` primary false positives; `{observations['combined_holdout_clean_prefix_false_positives']}` occurred in the clean startup prefix.",
            f"- Combined had `{observations['combined_holdout_false_negatives']}` primary false negatives.",
            f"- Reliability-only saturated at pooled FPR `{observations['reliability_only_pooled_false_positive_rate']:.6f}` with maximum false-positive streak `{observations['reliability_only_max_false_positive_streak']}`.",
            f"- Every holdout combined ledger lacked overlap: `{observations['all_holdout_combined_runs_missing_overlap']}`.",
            "",
            "The accompanying `analysis.json` stores all method/run 30-position scores, classified error positions, source-field provenance, and single/leave-one-signal-out rank diagnostics. The overlap rows intentionally remain unavailable rather than being filled with a proxy.",
            "",
        ]
    )


def analyze(
    input_root: Path,
    runs_root: Path,
    evaluation_root: Path,
    output_dir: Path,
    *,
    calibration_root: Path | None = None,
) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise FailureAnalysisError("CPU-only analysis requires CUDA_VISIBLE_DEVICES=''" )
    if "torch" in sys.modules:
        raise FailureAnalysisError("CPU-only analysis must start before torch is imported")
    output = output_dir.resolve(strict=False)
    if os.path.lexists(output):
        raise FailureAnalysisError(f"refusing to overwrite output directory: {output}")
    calibration = calibration_root or evaluation_root.parent / "formal-v1-calibration-0001"
    evaluation_manifest_path = evaluation_root / "evaluation-manifest.json"
    manifest_bytes = _read_regular(evaluation_manifest_path, name="evaluation manifest", mode=0o444)
    if _sha256(manifest_bytes) != EXPECTED_EVALUATION_MANIFEST_SHA256:
        raise FailureAnalysisError("evaluation manifest SHA-256 is not the frozen formal-v1 value")
    manifest = _json_object(manifest_bytes, name="evaluation manifest")
    verified = _verify_evaluation_bindings(
        input_root=input_root,
        runs_root=runs_root,
        calibration_root=calibration,
        evaluation_root=evaluation_root,
        evaluation_manifest=manifest,
    )
    split_runs, records_by_run = _make_runs(input_root, runs_root)
    dev_frozen = _json_object(_read_regular(calibration / "dev-metrics.json", name="frozen development metrics", mode=0o444), name="frozen development metrics")
    holdout_frozen = _json_object(_read_regular(evaluation_root / "holdout-metrics.json", name="frozen holdout metrics", mode=0o444), name="frozen holdout metrics")
    go_no_go_frozen_bytes = _read_regular(evaluation_root / "go-no-go.json", name="frozen go/no-go", mode=0o444)
    go_no_go_frozen = _json_object(go_no_go_frozen_bytes, name="frozen go/no-go")
    frozen_config = holdout_frozen.get("detection_config")
    if not isinstance(frozen_config, Mapping) or frozen_config != dev_frozen.get("detection_config"):
        raise FailureAnalysisError("development/holdout frozen detection configs differ")
    replayed = {
        "development": detection_suite.evaluate_detection_suite(
            split_runs["development"], frozen_config, expected_split="development"
        )
    }
    formal_config = _json_object(
        _read_regular(calibration / "formal-config.json", name="frozen formal config", mode=0o444),
        name="frozen formal config",
    )
    search = _json_object(
        _read_regular(calibration / "search.json", name="frozen development search", mode=0o444),
        name="frozen development search",
    )
    replayed["holdout"] = detection_suite.evaluate_formal_holdout(
        split_runs["holdout"],
        formal_config,
        split_registry=_read_regular(input_root / "split-registry.json", name="frozen split registry", mode=0o444),
        holdout_commitment=_read_regular(input_root / "holdout-commitment.json", name="frozen holdout commitment", mode=0o444),
        search_result=search,
        development_runs=split_runs["development"],
        runtime_detection_config=frozen_config,
    )
    for split, frozen in (("development", dev_frozen), ("holdout", holdout_frozen)):
        if _canonical_json_bytes(_result_projection(replayed[split])) != _canonical_json_bytes(_result_projection(frozen)):
            raise FailureAnalysisError(f"{split} score replay does not match frozen v1 metrics")
    decision_replay = formal_v1.evaluate_go_no_go(dev_frozen, holdout_frozen, provenance_passed=True)
    if formal_v1._json_bytes(decision_replay) != go_no_go_frozen_bytes:
        raise FailureAnalysisError("replayed v1 decision does not equal frozen go-no-go bytes")

    positions: dict[str, dict[str, Any]] = {}
    signal_provenance: dict[str, Any] = {}
    rank_ablations: dict[str, Any] = {}
    for split, runs in split_runs.items():
        per_run = {row["run_id"]: row for row in replayed[split]["runs"]}
        positions[split] = {
            method: {
                run_id: _positions_for_method(per_run[run_id], row)
                for run_id, row in replayed[split]["methods"][method]["per_run"].items()
            }
            for method in detection.METHOD_NAMES
        }
        provenance, ablations = _rank_summary(runs, replayed[split], records_by_run)
        signal_provenance[split] = provenance
        rank_ablations[split] = ablations

    git_commit = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    script_bytes = Path(__file__).read_bytes()
    payload: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "PASS",
        "cpu_only": {"CUDA_VISIBLE_DEVICES": "", "torch_imported_at_start": False},
        "implementation": {
            "repository_commit": git_commit,
            "script_path": "scripts/analyze_detection_v1_failure.py",
            "script_sha256": _sha256(script_bytes),
        },
        "input_evidence": {
            "evaluation_manifest_sha256": _sha256(manifest_bytes),
            "evaluation_manifest_expected_sha256": EXPECTED_EVALUATION_MANIFEST_SHA256,
            "verified_artifact_count": len(verified),
            "verified_artifacts": verified,
        },
        "decision_replay": {
            "matches_frozen_bytes": True,
            "frozen_go_no_go_sha256": _sha256(go_no_go_frozen_bytes),
            "decision": decision_replay["decision"],
            "checks": decision_replay["checks"],
        },
        "frozen_detection_config": dict(frozen_config),
        "replayed_metrics": replayed,
        "position_level_errors": positions,
        "v1_signal_provenance": signal_provenance,
        "rank_ablations": rank_ablations,
        "failure_observations": _failure_observations(holdout_frozen, positions),
        "limitations": [
            "formal-v1 holdout is disclosed and may only be used for Detector v2 development diagnostics",
            "overlap remains missing in v1; this analysis does not synthesize or impute it",
            "single/leave-one-signal-out values are rank-only diagnostics and do not replace the frozen formal-v1 thresholds",
        ],
    }
    report = _render_report(payload).encode("utf-8")
    formal_v1._publish_immutable_tree(
        output,
        {"analysis.json": _canonical_json_bytes(payload), "report.md": report},
    )
    return {
        "status": "PASS",
        "output_dir": str(output),
        "analysis_sha256": _sha256(_canonical_json_bytes(payload)),
        "decision": decision_replay["decision"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("evaluation_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=None,
        help="frozen formal-v1 calibration directory (defaults beside evaluation_root)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = analyze(
            args.input_root,
            args.runs_root,
            args.evaluation_root,
            args.output_dir,
            calibration_root=args.calibration_root,
        )
    except (FailureAnalysisError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
