#!/usr/bin/env python3
"""Derive immutable Detector v2 development ledgers and candidate search evidence.

This CPU-only program uses only frozen v1 ledgers, labels for offline
development metrics, and frozen online-overlap ledgers. It never calls model
inference or writes to a formal-v1 directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.validate_visual_overlap_v2 import _canonical_json_bytes, _publish
from stateguard3r import detection
from stateguard3r.detection_suite import derive_run_seed
from stateguard3r.detection_v2 import (
    V2ScoringConfig,
    candidate_grid,
    compute_v2_scores,
    derive_development_scale_floors,
    select_constrained_threshold,
    validate_online_overlap,
)
from stateguard3r.health import HealthFrame


SCHEMA_VERSION = "stateguard3r.detector-v2-development.v1"
FRAME_COUNT = 30
CLEAN_PREFIX = 15
METHODS = ("random", "update_magnitude_only", "reliability_only", "combined")
EXPECTED_RUN_IDS = frozenset(
    (
        "development-dynamic",
        "development-low",
        "development-wrong",
        "holdout-dynamic",
        "holdout-low",
        "holdout-wrong",
    )
)


class DetectorV2DevelopmentError(ValueError):
    """Raised when frozen evidence cannot safely form disclosed v2 development."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _artifact(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(payload),
        "size_bytes": len(payload),
    }


def _read_frozen(path: Path, *, label: str) -> bytes:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise DetectorV2DevelopmentError(f"cannot stat {label}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DetectorV2DevelopmentError(f"{label} must be a regular non-symlink file")
    if stat.S_IMODE(metadata.st_mode) != 0o444:
        raise DetectorV2DevelopmentError(f"{label} must be frozen at mode 0444")
    return path.read_bytes()


def _json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise DetectorV2DevelopmentError(f"{label} contains non-finite JSON {value!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DetectorV2DevelopmentError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, DetectorV2DevelopmentError) as error:
        raise DetectorV2DevelopmentError(f"{label} is not strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise DetectorV2DevelopmentError(f"{label} must be a JSON object")
    return value


def _load_regions(
    calibration_metrics: Path, evaluation_metrics: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    regions: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, Any] = {}
    for expected_split, path in (
        ("development", calibration_metrics),
        ("holdout", evaluation_metrics),
    ):
        payload = _read_frozen(path, label=f"{expected_split} metrics")
        value = _json_object(payload, label=f"{expected_split} metrics")
        rows = value.get("runs")
        if not isinstance(rows, list):
            raise DetectorV2DevelopmentError(f"{expected_split} metrics lacks runs")
        artifacts[expected_split] = _artifact(path, payload)
        for row in rows:
            if not isinstance(row, Mapping):
                raise DetectorV2DevelopmentError("metric run must be an object")
            run_id = row.get("run_id")
            labels = row.get("primary_labels")
            mask = row.get("evaluation_mask")
            if (
                not isinstance(run_id, str)
                or row.get("dataset_split") != expected_split
                or not isinstance(row.get("corruption_type"), str)
                or not isinstance(labels, list)
                or not isinstance(mask, list)
                or len(labels) != FRAME_COUNT
                or len(mask) != FRAME_COUNT
                or row.get("event_start") != CLEAN_PREFIX
                or not isinstance(row.get("event_end"), int)
                or not isinstance(row.get("artifact_provenance"), Mapping)
                or run_id in regions
            ):
                raise DetectorV2DevelopmentError(f"malformed metric region for {run_id!r}")
            if not np.all(np.isin(np.asarray(labels), (0, 1, False, True))):
                raise DetectorV2DevelopmentError(f"{run_id} labels must be binary")
            regions[run_id] = {
                "dataset_split": expected_split,
                "corruption_type": row["corruption_type"],
                "event_start": CLEAN_PREFIX,
                "event_end": row["event_end"],
                "primary_labels": [int(value) for value in labels],
                "evaluation_mask": [bool(value) for value in mask],
                "v1_artifacts": dict(row["artifact_provenance"]),
            }
    if set(regions) != EXPECTED_RUN_IDS:
        raise DetectorV2DevelopmentError("metrics must cover exactly the disclosed six v1 runs")
    return regions, artifacts


def merge_online_overlap(
    health_records: Sequence[Mapping[str, Any]],
    online_ledger: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
) -> list[dict[str, Any]]:
    """Return independent v2 health records with only frame_id and overlap joined."""

    if len(health_records) != FRAME_COUNT or len(online_ledger) != FRAME_COUNT:
        raise DetectorV2DevelopmentError(f"{run_id} needs thirty health and online records")
    merged: list[dict[str, Any]] = []
    for frame_id, (health, online) in enumerate(zip(health_records, online_ledger, strict=True)):
        if set(online) != {"frame_id", "overlap"} or online["frame_id"] != frame_id:
            raise DetectorV2DevelopmentError(f"{run_id} online ledger is not frame_id/overlap only")
        if health.get("frame_id") != frame_id or health.get("overlap") is not None:
            raise DetectorV2DevelopmentError(f"{run_id} frozen health ledger is not an expected v1 ledger")
        copied = dict(health)
        copied["overlap"] = online["overlap"]
        try:
            merged.append(HealthFrame.from_dict(copied).to_dict())
        except ValueError as error:
            raise DetectorV2DevelopmentError(f"{run_id} derived frame {frame_id} is invalid: {error}") from error
    try:
        validate_online_overlap(merged)
    except ValueError as error:
        raise DetectorV2DevelopmentError(f"{run_id} online overlap is not finite and causal: {error}") from error
    return merged


def _load_ledgers(
    runs_root: Path,
    replay_path: Path,
    regions: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any], dict[str, Any]]:
    replay_bytes = _read_frozen(replay_path, label="visual overlap replay")
    replay = _json_object(replay_bytes, label="visual overlap replay")
    candidates = replay.get("candidates")
    if replay.get("status") != "PASS" or not isinstance(candidates, list) or len(candidates) != 4:
        raise DetectorV2DevelopmentError("visual overlap replay must be a passing four-candidate output")
    output: dict[str, dict[str, list[dict[str, Any]]]] = {}
    provenance: dict[str, Any] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise DetectorV2DevelopmentError("visual candidate must be an object")
        candidate_id = candidate.get("candidate_id")
        candidate_runs = candidate.get("runs")
        config = candidate.get("config")
        if (
            not isinstance(candidate_id, str)
            or not isinstance(candidate_runs, Mapping)
            or set(candidate_runs) != EXPECTED_RUN_IDS
            or not isinstance(config, Mapping)
            or candidate_id in output
        ):
            raise DetectorV2DevelopmentError("visual candidate identity or run coverage is invalid")
        output[candidate_id] = {}
        provenance[candidate_id] = {"config": dict(config), "runs": {}}
        for run_id in sorted(EXPECTED_RUN_IDS):
            source = candidate_runs[run_id]
            if not isinstance(source, Mapping) or not isinstance(source.get("online_ledger"), list):
                raise DetectorV2DevelopmentError(f"{candidate_id}/{run_id} lacks online ledger")
            health_path = runs_root / run_id / "health.jsonl"
            health_bytes = _read_frozen(health_path, label=f"{run_id} health")
            expected = regions[run_id]["v1_artifacts"].get("health_jsonl")
            if (
                not isinstance(expected, Mapping)
                or expected.get("sha256") != _sha256(health_bytes)
                or expected.get("size_bytes") != len(health_bytes)
            ):
                raise DetectorV2DevelopmentError(f"{run_id} health does not match its v1 metric binding")
            records = [
                _json_object(line.encode("utf-8"), label=f"{run_id} health")
                for line in health_bytes.decode("utf-8").splitlines()
                if line.strip()
            ]
            output[candidate_id][run_id] = merge_online_overlap(
                records, source["online_ledger"], run_id=run_id
            )
            provenance[candidate_id]["runs"][run_id] = {
                "health_jsonl": _artifact(health_path, health_bytes),
                "input_manifest": dict(source.get("input_manifest", {})),
                "online_ledger_sha256": _sha256(
                    _canonical_json_bytes(source["online_ledger"])
                ),
            }
    return output, provenance, {
        "path": str(replay_path),
        "sha256": _sha256(replay_bytes),
        "size_bytes": len(replay_bytes),
    }


def _mean(values: Sequence[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def _longest(values: np.ndarray) -> int:
    result = 0
    running = 0
    for value in values:
        if bool(value):
            running += 1
            result = max(result, running)
        else:
            running = 0
    return result


def _evaluate_method(
    run_ids: Sequence[str],
    regions: Mapping[str, Mapping[str, Any]],
    scores: Mapping[str, np.ndarray],
    threshold: float,
) -> dict[str, Any]:
    per_run: dict[str, Any] = {}
    pooled_labels: list[np.ndarray] = []
    pooled_scores: list[np.ndarray] = []
    event_rows: list[dict[str, Any]] = []
    delays: list[int] = []
    startup_count = 0
    startup_positions: dict[str, int] = {}
    maximum_streak = 0
    for run_id in sorted(run_ids):
        region = regions[run_id]
        labels = np.asarray(region["primary_labels"], dtype=np.int64)
        mask = np.asarray(region["evaluation_mask"], dtype=bool)
        risk = np.asarray(scores[run_id], dtype=np.float64)
        if risk.shape != (FRAME_COUNT,) or not np.all(np.isfinite(risk)):
            raise DetectorV2DevelopmentError(f"{run_id} score timeline must be finite with length thirty")
        metrics = detection.threshold_metrics(labels[mask], risk[mask], threshold)
        metrics["auroc"] = detection.auroc(labels[mask], risk[mask])
        false_positive = mask & (labels == 0) & (risk >= threshold)
        streak = _longest(false_positive)
        metrics["longest_consecutive_false_positives"] = streak
        maximum_streak = max(maximum_streak, streak)
        delay = detection.detection_delay(labels, risk, threshold)
        event = dict(delay["intervals"][0])
        event.update({"run_id": run_id, "corruption_type": region["corruption_type"]})
        event_rows.append(event)
        if event["delay_frames"] is not None:
            delays.append(int(event["delay_frames"]))
        startup = np.flatnonzero(risk[:CLEAN_PREFIX] >= threshold)
        startup_count += int(len(startup))
        for position in startup:
            key = str(int(position))
            startup_positions[key] = startup_positions.get(key, 0) + 1
        per_run[run_id] = {
            "corruption_type": region["corruption_type"],
            "metrics": metrics,
            "detection_delay": delay,
            "startup_false_positive_positions": [int(value) for value in startup],
            "max_false_positive_streak": streak,
        }
        pooled_labels.append(labels[mask])
        pooled_scores.append(risk[mask])
    labels_all = np.concatenate(pooled_labels)
    scores_all = np.concatenate(pooled_scores)
    pooled = detection.threshold_metrics(labels_all, scores_all, threshold)
    pooled["auroc"] = detection.auroc(labels_all, scores_all)
    detected_events = sum(1 for row in event_rows if row["delay_frames"] is not None)
    return {
        "threshold": float(threshold),
        "per_run": per_run,
        "pooled": pooled,
        "macro": {
            "auroc": _mean([row["metrics"]["auroc"] for row in per_run.values()]),
            "f1": _mean([row["metrics"]["f1"] for row in per_run.values()]),
            "false_positive_rate": _mean(
                [row["metrics"]["false_positive_rate"] for row in per_run.values()]
            ),
        },
        "events": {
            "total_events": len(event_rows),
            "detected_events": detected_events,
            "missed_events": len(event_rows) - detected_events,
            "detection_rate": float(detected_events / len(event_rows)),
            "mean_delay_frames": float(np.mean(delays)) if delays else None,
            "intervals": event_rows,
        },
        "diagnostics": {
            "max_per_run_false_positive_streak": maximum_streak,
            "clean_prefix_false_positives": startup_count,
            "clean_prefix_alarm_count_by_position": startup_positions,
        },
    }


def _score_context(
    *,
    score_ids: Sequence[str],
    floor_ids: Sequence[str],
    threshold_ids: Sequence[str],
    regions: Mapping[str, Mapping[str, Any]],
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    config: V2ScoringConfig,
) -> dict[str, Any]:
    floors = derive_development_scale_floors(
        {run_id: records[run_id] for run_id in sorted(floor_ids)},
        clean_prefix_frames=CLEAN_PREFIX,
    )
    payloads: dict[str, Any] = {}
    timelines: dict[str, dict[str, list[float]]] = {method: {} for method in METHODS}
    signals: dict[str, Any] = {}
    computed_ids = sorted(set(score_ids) | set(threshold_ids))
    for run_id in computed_ids:
        payload = compute_v2_scores(
            records[run_id],
            config=config,
            scale_floors=floors,
            seed=derive_run_seed(
                config.master_seed, str(regions[run_id]["dataset_split"]), run_id
            ),
            require_online_overlap=True,
        )
        payloads[run_id] = payload
        if run_id in score_ids:
            for method in METHODS:
                method_payload = payload["methods"][method]
                timelines[method][run_id] = [
                    float(value) for value in method_payload["scores"].tolist()
                ]
            signals[run_id] = {
                method: {
                    "used_signals": list(payload["methods"][method]["used_signals"]),
                    "signal_sources": [
                        dict(value) for value in payload["methods"][method]["signal_sources"]
                    ],
                }
                for method in METHODS
            }
    selections: dict[str, Any] = {}
    evaluations: dict[str, Any] = {}
    for method in METHODS:
        selection = select_constrained_threshold(
            {run_id: regions[run_id]["primary_labels"] for run_id in threshold_ids},
            {
                run_id: np.asarray(
                    payloads[run_id]["methods"][method]["scores"], dtype=np.float64
                )
                for run_id in threshold_ids
            },
            {run_id: regions[run_id]["evaluation_mask"] for run_id in threshold_ids},
        )
        selections[method] = selection
        evaluations[method] = _evaluate_method(
            score_ids,
            regions,
            {
                run_id: np.asarray(
                    payloads[run_id]["methods"][method]["scores"], dtype=np.float64
                )
                for run_id in score_ids
            },
            float(selection["selected"]["threshold"]),
        )
    return {
        "config": config.to_dict(),
        "scale_floors": floors,
        "threshold_selections": selections,
        "methods": evaluations,
        "score_timelines": timelines,
        "signal_sources": signals,
    }


def cross_validation_folds(regions: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    all_ids = set(regions)
    folds: list[dict[str, Any]] = []
    for corruption_type in sorted(
        {str(region["corruption_type"]) for region in regions.values()}
    ):
        test = sorted(
            run_id
            for run_id, region in regions.items()
            if region["corruption_type"] == corruption_type
        )
        folds.append(
            {
                "fold_id": f"leave-one-corruption-type-out:{corruption_type}",
                "kind": "corruption_type",
                "held_out": corruption_type,
                "test_run_ids": test,
                "train_run_ids": sorted(all_ids - set(test)),
            }
        )
    for original_window in ("development", "holdout"):
        test = sorted(
            run_id
            for run_id, region in regions.items()
            if region["dataset_split"] == original_window
        )
        folds.append(
            {
                "fold_id": f"leave-one-window-out:{original_window}",
                "kind": "original_v1_window",
                "held_out": original_window,
                "test_run_ids": test,
                "train_run_ids": sorted(all_ids - set(test)),
            }
        )
    if len(folds) != 5 or any(
        not fold["train_run_ids"] or not fold["test_run_ids"] for fold in folds
    ):
        raise DetectorV2DevelopmentError("must construct three type and two window folds")
    return folds


def _cross_validate(
    *,
    regions: Mapping[str, Mapping[str, Any]],
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    config: V2ScoringConfig,
    folds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        context = _score_context(
            score_ids=fold["test_run_ids"],
            floor_ids=fold["train_run_ids"],
            threshold_ids=fold["train_run_ids"],
            regions=regions,
            records=records,
            config=config,
        )
        rows.append(
            {
                **dict(fold),
                "train_scale_floors": context["scale_floors"],
                "combined": context["methods"]["combined"],
                "combined_threshold_selection": context["threshold_selections"]["combined"],
            }
        )
    combined = [row["combined"] for row in rows]
    return {
        "folds": rows,
        "combined_summary": {
            "macro_auroc": _mean([row["macro"]["auroc"] for row in combined]),
            "pooled_false_positive_rate": _mean(
                [row["pooled"]["false_positive_rate"] for row in combined]
            ),
            "detected_events": _mean(
                [float(row["events"]["detected_events"]) for row in combined]
            ),
            "mean_delay_frames": _mean(
                [row["events"]["mean_delay_frames"] for row in combined]
            ),
        },
    }


def _quality(row: Mapping[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    cross = row["cross_validation"]["combined_summary"]
    visual = row["visual_config"]
    return (
        float(cross["macro_auroc"]),
        -float(cross["pooled_false_positive_rate"])
        if cross["pooled_false_positive_rate"] is not None
        else -math.inf,
        float(cross["detected_events"]),
        -float(cross["mean_delay_frames"])
        if cross["mean_delay_frames"] is not None
        else -math.inf,
        float(visual["ratio_threshold"]),
        -float(visual["ransac_pixel_limit"]),
        -float(row["score_config_order"]),
    )


def _readiness(selected: Mapping[str, Any], regions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    full = selected["full_development"]
    combined = full["methods"]["combined"]
    random = full["methods"]["random"]
    reliability = full["methods"]["reliability_only"]
    detected_types = {
        row["corruption_type"]
        for row in combined["events"]["intervals"]
        if row["delay_frames"] is not None
    }
    repeated_positions = sorted(
        int(position)
        for position, count in combined["diagnostics"][
            "clean_prefix_alarm_count_by_position"
        ].items()
        if count >= 3
    )
    combined_uses_all = all(
        set(full["signal_sources"][run_id]["combined"]["used_signals"])
        == {
            "geometric_residual",
            "pose_jump",
            "update_magnitude",
            "overlap",
            "reliability",
        }
        for run_id in regions
    )
    checks = {
        "full_development_combined_macro_auroc_at_least_0_80": combined["macro"][
            "auroc"
        ]
        >= 0.80,
        "cross_validated_held_out_macro_auroc_at_least_0_75": selected[
            "cross_validation"
        ]["combined_summary"]["macro_auroc"]
        >= 0.75,
        "combined_minus_seeded_random_macro_auroc_at_least_0_15": combined["macro"][
            "auroc"
        ]
        - random["macro"]["auroc"]
        >= 0.15,
        "pooled_primary_fpr_at_most_0_15": combined["pooled"][
            "false_positive_rate"
        ]
        <= 0.15,
        "max_per_run_fp_streak_at_most_2": combined["diagnostics"][
            "max_per_run_false_positive_streak"
        ]
        <= 2,
        "all_three_corruptions_detected_with_mean_delay_at_most_1": len(detected_types)
        == 3
        and combined["events"]["mean_delay_frames"] is not None
        and combined["events"]["mean_delay_frames"] <= 1.0,
        "clean_prefix_fp_at_most_6_without_triplicate_position": combined["diagnostics"][
            "clean_prefix_false_positives"
        ]
        <= 6
        and not repeated_positions,
        "reliability_only_pooled_fpr_below_0_50": reliability["pooled"][
            "false_positive_rate"
        ]
        < 0.50,
        "finite_online_overlap_used_by_combined": combined_uses_all,
        "instrumentation_equivalence_gpu_cleanup": False,
    }
    cpu_pass = all(
        value for key, value in checks.items() if key != "instrumentation_equivalence_gpu_cleanup"
    )
    return {
        "status": "PREINSTRUMENTATION_PASS" if cpu_pass else "FAIL",
        "not_ready_reason": (
            "GPU instrumentation equivalence has not run"
            if cpu_pass
            else "one or more CPU readiness checks failed"
        ),
        "checks": checks,
        "selected_metrics": {
            "combined_macro_auroc": combined["macro"]["auroc"],
            "random_macro_auroc": random["macro"]["auroc"],
            "combined_minus_random_macro_auroc": combined["macro"]["auroc"]
            - random["macro"]["auroc"],
            "combined_pooled_fpr": combined["pooled"]["false_positive_rate"],
            "combined_max_fp_streak": combined["diagnostics"][
                "max_per_run_false_positive_streak"
            ],
            "combined_detected_events": combined["events"]["detected_events"],
            "combined_mean_delay_frames": combined["events"]["mean_delay_frames"],
            "clean_prefix_false_positives": combined["diagnostics"][
                "clean_prefix_false_positives"
            ],
            "repeated_clean_prefix_positions": repeated_positions,
            "reliability_only_pooled_fpr": reliability["pooled"]["false_positive_rate"],
            "cross_validated_held_out_macro_auroc": selected["cross_validation"][
                "combined_summary"
            ]["macro_auroc"],
        },
    }


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise DetectorV2DevelopmentError("derived output cannot contain non-finite float")
    return value


def _report(search: Mapping[str, Any]) -> bytes:
    selected = search["selected"]
    full = selected["full_development"]["methods"]["combined"]
    cross = selected["cross_validation"]["combined_summary"]
    config = selected["score_config"]
    lines = [
        "# Detector v2 disclosed-development derivation",
        "",
        f"Status: **{search['readiness']['status']}**",
        "",
        "All six v1 runs are disclosed development data, not blind formal-v2 evidence.",
        "",
        "| Visual candidate | window | min-history | max-z | full macro-AUROC | held-out CV macro-AUROC | pooled FPR | events | delay |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| {visual} | {window} | {history} | {max_z:g} | {full_auc:.9f} | {cv_auc:.9f} | {fpr:.9f} | {events} | {delay:.9f} |".format(
            visual=selected["visual_candidate_id"],
            window=config["window"],
            history=config["min_history"],
            max_z=config["max_z"],
            full_auc=full["macro"]["auroc"],
            cv_auc=cross["macro_auroc"],
            fpr=full["pooled"]["false_positive_rate"],
            events=full["events"]["detected_events"],
            delay=full["events"]["mean_delay_frames"],
        ),
        "",
        "The complete 32-candidate search, score timelines, threshold candidates, and five held-out folds are in search.json. Derived ledgers contain only copied health records plus online overlap, with no GT or label field.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def derive_detector_v2_development(
    *,
    runs_root: Path,
    visual_replay_dir: Path,
    calibration_metrics: Path,
    evaluation_metrics: Path,
    output_dir: Path,
) -> Path:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise DetectorV2DevelopmentError("derivation requires CUDA_VISIBLE_DEVICES=''")
    replay_path = visual_replay_dir.resolve(strict=True) / "visual-overlap-replay.json"
    regions, metric_artifacts = _load_regions(
        calibration_metrics.resolve(strict=True), evaluation_metrics.resolve(strict=True)
    )
    ledgers, ledger_provenance, replay_artifact = _load_ledgers(
        runs_root.resolve(strict=True), replay_path, regions
    )
    folds = cross_validation_folds(regions)
    results: list[dict[str, Any]] = []
    for visual_id, records in ledgers.items():
        visual_config = ledger_provenance[visual_id]["config"]
        for order, config in enumerate(candidate_grid()):
            try:
                full = _score_context(
                    score_ids=sorted(EXPECTED_RUN_IDS),
                    floor_ids=sorted(EXPECTED_RUN_IDS),
                    threshold_ids=sorted(EXPECTED_RUN_IDS),
                    regions=regions,
                    records=records,
                    config=config,
                )
                cross = _cross_validate(
                    regions=regions, records=records, config=config, folds=folds
                )
                results.append(
                    {
                        "status": "PASS",
                        "visual_candidate_id": visual_id,
                        "visual_config": visual_config,
                        "score_config_order": order,
                        "score_config": config.to_dict(),
                        "full_development": full,
                        "cross_validation": cross,
                    }
                )
            except (ValueError, DetectorV2DevelopmentError) as error:
                results.append(
                    {
                        "status": "FAIL",
                        "visual_candidate_id": visual_id,
                        "visual_config": visual_config,
                        "score_config_order": order,
                        "score_config": config.to_dict(),
                        "failure_reason": str(error),
                    }
                )
    passing = [row for row in results if row["status"] == "PASS"]
    if not passing:
        raise DetectorV2DevelopmentError("all visual/scoring candidates failed")
    selected = dict(max(passing, key=_quality))
    readiness = _readiness(selected, regions)
    ledgers_output = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "disclosed_v1_development_only",
        "selected_visual_candidate_id": selected["visual_candidate_id"],
        "online_ledger_contract": {
            "allowed_fields": ["frame_id", "overlap"],
            "forbidden_online_inputs": [
                "depth",
                "groundtruth",
                "gt_pose",
                "corruption_label",
                "event_interval",
                "future_frame",
                "health_ledger",
                "model_response",
            ],
        },
        "runs": {
            run_id: {
                "source": ledger_provenance[selected["visual_candidate_id"]]["runs"][
                    run_id
                ],
                "records": ledgers[selected["visual_candidate_id"]][run_id],
            }
            for run_id in sorted(EXPECTED_RUN_IDS)
        },
    }
    script_path = Path(__file__).resolve()
    search = {
        "schema_version": SCHEMA_VERSION,
        "status": readiness["status"],
        "purpose": "disclosed_v1_development_only_not_blind_formal_v2",
        "inputs": {
            "visual_replay": replay_artifact,
            "formal_v1_metrics": metric_artifacts,
            "formal_v1_runs_root": str(runs_root.resolve(strict=True)),
        },
        "generator": {
            "script": _artifact(script_path, script_path.read_bytes()),
            "detection_v2_module": _artifact(
                ROOT / "src" / "stateguard3r" / "detection_v2.py",
                (ROOT / "src" / "stateguard3r" / "detection_v2.py").read_bytes(),
            ),
        },
        "candidate_count": len(results),
        "candidate_results": results,
        "selected": selected,
        "fold_definition": {
            "folds": folds,
            "scale_floors": "derived from training-fold clean prefixes only",
            "thresholds": "selected from training folds only with FPR<=0.15 and streak<=2",
        },
        "readiness": readiness,
    }
    _publish(
        output_dir.resolve(strict=False),
        {
            "derived-ledgers.json": _canonical_json_bytes(_plain(ledgers_output)),
            "search.json": _canonical_json_bytes(_plain(search)),
            "report.md": _report(_plain(search)),
        },
    )
    return output_dir.resolve(strict=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("visual_replay_dir", type=Path)
    parser.add_argument("calibration_metrics", type=Path)
    parser.add_argument("evaluation_metrics", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(
        derive_detector_v2_development(
            runs_root=args.runs_root,
            visual_replay_dir=args.visual_replay_dir,
            calibration_metrics=args.calibration_metrics,
            evaluation_metrics=args.evaluation_metrics,
            output_dir=args.output_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
