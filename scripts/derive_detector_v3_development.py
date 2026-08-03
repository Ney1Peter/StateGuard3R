#!/usr/bin/env python3
"""Freeze disclosed v3 timestamp-order development evidence without CUDA.

All nine inputs were already disclosed by formal-v1/formal-v2 evaluation.  The
online half builds timestamp sidecars from only final RGB paths and the raw TUM
``rgb.txt`` listings.  The offline half then joins those sidecars with frozen
Detector-v2 continuous timelines and labels to quantify the predeclared v3 OR
rule.  It never calls model inference or edits an existing artifact.
"""

from __future__ import annotations

import argparse
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stateguard3r.detection import auroc, detection_delay, threshold_metrics
from stateguard3r.detection_v3 import hybridize_v2_continuous_scores
from stateguard3r.timestamp_order_v3 import (
    capture_timestamp_records,
    timestamp_order_sidecar,
    validate_timestamp_order_sidecar,
)


SCHEMA_VERSION = "stateguard3r.detector-v3-development.v1"
THRESHOLD = 2.0686323694270046
V1_ROOT = ROOT / "outputs" / "formal-v1-inputs-0001"
V2_ROOT = ROOT / "outputs" / "formal-v2-inputs-0001"
V1_SEARCH = ROOT / "outputs" / "detector-v2-development-search-0001" / "search.json"
V1_DEVELOPMENT_METRICS = ROOT / "outputs" / "formal-v1-calibration-0001" / "dev-metrics.json"
V1_HOLDOUT_METRICS = ROOT / "outputs" / "formal-v1-evaluation-0001" / "holdout-metrics.json"
V2_HOLDOUT_METRICS = ROOT / "outputs" / "formal-v2-evaluation-0001" / "holdout-metrics.json"
V1_DATASET = ROOT.parent / "baselines" / "ReCal3R" / "data" / "tum" / "rgbd_dataset_freiburg1_desk"
V2_DATASET = ROOT.parent / "baselines" / "ReCal3R" / "data" / "tum" / "rgbd_dataset_freiburg3_walking_static"
V1_RUNS = (
    ("development-dynamic", "development", "dynamic_occlusion"),
    ("development-wrong", "development", "wrong_order_segment"),
    ("development-low", "development", "low_overlap_jump"),
    ("holdout-dynamic", "holdout", "dynamic_occlusion"),
    ("holdout-wrong", "holdout", "wrong_order_segment"),
    ("holdout-low", "holdout", "low_overlap_jump"),
)
V2_RUNS = (
    ("holdout-dynamic", "dynamic_occlusion"),
    ("holdout-wrong", "wrong_order_segment"),
    ("holdout-low", "low_overlap_jump"),
)


class DetectorV3DevelopmentError(ValueError):
    """Raised when frozen disclosed evidence cannot produce v3 development."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_frozen(path: Path, *, label: str) -> bytes:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise DetectorV3DevelopmentError(f"cannot stat {label}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DetectorV3DevelopmentError(f"{label} must be a regular non-symlink file")
    if stat.S_IMODE(metadata.st_mode) != 0o444:
        raise DetectorV3DevelopmentError(f"{label} must be frozen at mode 0444")
    return path.read_bytes()


def _strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise DetectorV3DevelopmentError(f"{label} has non-finite JSON {value!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise DetectorV3DevelopmentError(f"{label} repeats key {key!r}")
            output[key] = value
        return output

    try:
        value = json.loads(payload, parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, DetectorV3DevelopmentError) as error:
        raise DetectorV3DevelopmentError(f"{label} is not strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise DetectorV3DevelopmentError(f"{label} must be a JSON object")
    return value


def _artifact(path: Path, payload: bytes) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(payload), "size_bytes": len(payload)}


def _input_paths(manifest: Mapping[str, Any], *, manifest_path: Path) -> list[Path]:
    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) != 30:
        raise DetectorV3DevelopmentError(f"{manifest_path} must contain thirty frames")
    result: list[Path] = []
    for frame_id, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or frame.get("frame_index") != frame_id:
            raise DetectorV3DevelopmentError(f"{manifest_path} frame IDs are not contiguous")
        path_text = frame.get("path")
        if not isinstance(path_text, str) or not path_text:
            raise DetectorV3DevelopmentError(f"{manifest_path} frame {frame_id} lacks RGB path")
        path = Path(path_text)
        result.append((manifest_path.parent / path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True))
    return result


def _label_context(metrics: Mapping[str, Any], *, run_id: str) -> tuple[np.ndarray, np.ndarray]:
    rows = metrics.get("runs")
    if not isinstance(rows, list):
        raise DetectorV3DevelopmentError("metrics lacks runs")
    found = [row for row in rows if isinstance(row, Mapping) and row.get("run_id") == run_id]
    if len(found) != 1:
        raise DetectorV3DevelopmentError(f"metrics lacks one row for {run_id}")
    row = found[0]
    labels = np.asarray(row.get("primary_labels"), dtype=np.int64)
    mask = np.asarray(row.get("evaluation_mask", row.get("primary_mask")), dtype=bool)
    if labels.shape != (30,) or mask.shape != (30,) or not np.all(np.isin(labels, (0, 1))):
        raise DetectorV3DevelopmentError(f"metrics labels are invalid for {run_id}")
    return labels, mask


def _v2_scores(metrics: Mapping[str, Any], *, run_id: str) -> np.ndarray:
    try:
        result = np.asarray(metrics["methods"]["combined"]["per_run"][run_id]["scores"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise DetectorV3DevelopmentError(f"metrics lacks combined scores for {run_id}") from error
    if result.shape != (30,) or not np.all(np.isfinite(result)):
        raise DetectorV3DevelopmentError(f"combined scores are invalid for {run_id}")
    return result


def _longest_false_positive(labels: np.ndarray, scores: np.ndarray, mask: np.ndarray) -> int:
    active = mask & (labels == 0) & (scores >= THRESHOLD)
    longest = 0
    current = 0
    for value in active:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _evaluate(rows: Sequence[Mapping[str, Any]], *, score_key: str) -> dict[str, Any]:
    per_run: dict[str, Any] = {}
    aucs: list[float] = []
    pooled_labels: list[np.ndarray] = []
    pooled_scores: list[np.ndarray] = []
    delays: list[int] = []
    detected_types: set[str] = set()
    total_clean_prefix_fp = 0
    maximum_streak = 0
    for row in rows:
        labels = np.asarray(row["labels"], dtype=np.int64)
        mask = np.asarray(row["mask"], dtype=bool)
        scores = np.asarray(row[score_key], dtype=np.float64)
        metrics = threshold_metrics(labels[mask], scores[mask], THRESHOLD)
        run_auc = auroc(labels[mask], scores[mask])
        delay = detection_delay(labels, scores, THRESHOLD)
        interval = delay["intervals"][0]
        if run_auc is None:
            raise DetectorV3DevelopmentError(f"{row['development_id']} has undefined AUROC")
        aucs.append(run_auc)
        if interval["delay_frames"] is not None:
            delays.append(int(interval["delay_frames"]))
            detected_types.add(str(row["corruption_type"]))
        startup = (labels[:15] == 0) & (scores[:15] >= THRESHOLD)
        total_clean_prefix_fp += int(startup.sum())
        streak = _longest_false_positive(labels, scores, mask)
        maximum_streak = max(maximum_streak, streak)
        per_run[str(row["development_id"])] = {
            "source_run_id": row["source_run_id"],
            "source_generation": row["source_generation"],
            "corruption_type": row["corruption_type"],
            "auroc": run_auc,
            "primary_metrics": metrics,
            "delay": delay,
            "clean_prefix_false_positives": int(startup.sum()),
            "max_primary_false_positive_streak": streak,
        }
        pooled_labels.append(labels[mask])
        pooled_scores.append(scores[mask])
    labels_all = np.concatenate(pooled_labels)
    scores_all = np.concatenate(pooled_scores)
    return {
        "macro_auroc": float(np.mean(aucs)),
        "pooled_primary_metrics": threshold_metrics(labels_all, scores_all, THRESHOLD),
        "detected_corruption_types": sorted(detected_types),
        "detected_events": len(delays),
        "event_count": len(rows),
        "mean_delay_frames": float(np.mean(delays)) if delays else None,
        "clean_prefix_false_positives": total_clean_prefix_fp,
        "max_primary_false_positive_streak": maximum_streak,
        "per_run": per_run,
    }


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _freeze(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _publish(output_dir: Path, files: Mapping[str, bytes]) -> Path:
    if output_dir.parent != ROOT / "outputs" or output_dir.exists():
        raise DetectorV3DevelopmentError("output_dir must be a new direct child of outputs")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".staging", dir=output_dir.parent))
    published = False
    try:
        for name, content in files.items():
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        _freeze(staging)
        os.replace(staging, output_dir)
        published = True
        return output_dir
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def derive(output_dir: Path) -> Path:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise DetectorV3DevelopmentError("derivation requires CUDA_VISIBLE_DEVICES='' ")
    search_bytes = _read_frozen(V1_SEARCH, label="v2 disclosed development search")
    v1_dev_bytes = _read_frozen(V1_DEVELOPMENT_METRICS, label="v1 development metrics")
    v1_holdout_bytes = _read_frozen(V1_HOLDOUT_METRICS, label="v1 holdout metrics")
    v2_metrics_bytes = _read_frozen(V2_HOLDOUT_METRICS, label="v2 holdout metrics")
    search = _strict_json(search_bytes, label="v2 disclosed development search")
    v1_dev_metrics = _strict_json(v1_dev_bytes, label="v1 development metrics")
    v1_holdout_metrics = _strict_json(v1_holdout_bytes, label="v1 holdout metrics")
    v2_metrics = _strict_json(v2_metrics_bytes, label="v2 holdout metrics")
    try:
        selected = search["selected"]
        source_threshold = float(selected["full_development"]["methods"]["combined"]["threshold"])
        v1_timelines = selected["full_development"]["score_timelines"]["combined"]
    except (KeyError, TypeError, ValueError) as error:
        raise DetectorV3DevelopmentError("v2 disclosed development search lacks selected continuous timeline") from error
    if source_threshold != THRESHOLD or not isinstance(v1_timelines, Mapping):
        raise DetectorV3DevelopmentError("v2 selected threshold does not match frozen formal-v2 threshold")

    rows: list[dict[str, Any]] = []
    online_sidecars: dict[str, Any] = {}
    input_artifacts: dict[str, Any] = {}
    for run_id, split, corruption_type in V1_RUNS:
        manifest_path = V1_ROOT / split / run_id / "input-manifest.json"
        manifest_bytes = _read_frozen(manifest_path, label=f"v1 input {run_id}")
        manifest = _strict_json(manifest_bytes, label=f"v1 input {run_id}")
        captures, provenance = capture_timestamp_records(
            _input_paths(manifest, manifest_path=manifest_path), rgb_txt=V1_DATASET / "rgb.txt", dataset_root=V1_DATASET
        )
        sidecar = timestamp_order_sidecar(captures, provenance=provenance)
        validate_timestamp_order_sidecar(sidecar)
        continuous = np.asarray(v1_timelines.get(run_id), dtype=np.float64)
        if continuous.shape != (30,) or not np.all(np.isfinite(continuous)):
            raise DetectorV3DevelopmentError(f"v1 v2-derived continuous timeline is invalid for {run_id}")
        labels, mask = _label_context(v1_dev_metrics if split == "development" else v1_holdout_metrics, run_id=run_id)
        hybrid = hybridize_v2_continuous_scores(continuous, continuous_threshold=THRESHOLD, timestamp_sidecar=sidecar)
        development_id = f"v1-{run_id}"
        online_sidecars[development_id] = sidecar
        input_artifacts[development_id] = _artifact(manifest_path, manifest_bytes)
        rows.append({
            "development_id": development_id, "source_run_id": run_id, "source_generation": "v1_disclosed_replayed_by_v2",
            "corruption_type": corruption_type, "labels": labels.tolist(), "mask": mask.tolist(),
            "continuous_scores": continuous.tolist(), "hybrid_scores": np.asarray(hybrid["hybrid_scores"], dtype=np.float64).tolist(),
            "timestamp_alarm_positions": [index for index, value in enumerate(validate_timestamp_order_sidecar(sidecar)) if value is True],
        })
    for run_id, corruption_type in V2_RUNS:
        manifest_path = V2_ROOT / run_id / "input-manifest.json"
        manifest_bytes = _read_frozen(manifest_path, label=f"v2 input {run_id}")
        manifest = _strict_json(manifest_bytes, label=f"v2 input {run_id}")
        captures, provenance = capture_timestamp_records(
            _input_paths(manifest, manifest_path=manifest_path), rgb_txt=V2_DATASET / "rgb.txt", dataset_root=V2_DATASET
        )
        sidecar = timestamp_order_sidecar(captures, provenance=provenance)
        validate_timestamp_order_sidecar(sidecar)
        continuous = _v2_scores(v2_metrics, run_id=run_id)
        labels, mask = _label_context(v2_metrics, run_id=run_id)
        hybrid = hybridize_v2_continuous_scores(continuous, continuous_threshold=THRESHOLD, timestamp_sidecar=sidecar)
        development_id = f"v2-{run_id}"
        online_sidecars[development_id] = sidecar
        input_artifacts[development_id] = _artifact(manifest_path, manifest_bytes)
        rows.append({
            "development_id": development_id, "source_run_id": run_id, "source_generation": "v2_disclosed_formal_holdout",
            "corruption_type": corruption_type, "labels": labels.tolist(), "mask": mask.tolist(),
            "continuous_scores": continuous.tolist(), "hybrid_scores": np.asarray(hybrid["hybrid_scores"], dtype=np.float64).tolist(),
            "timestamp_alarm_positions": [index for index, value in enumerate(validate_timestamp_order_sidecar(sidecar)) if value is True],
        })
    continuous_result = _evaluate(rows, score_key="continuous_scores")
    hybrid_result = _evaluate(rows, score_key="hybrid_scores")
    leave_one_type_out = {
        corruption_type: _evaluate(
            [row for row in rows if row["corruption_type"] == corruption_type], score_key="hybrid_scores"
        )
        for corruption_type in ("dynamic_occlusion", "low_overlap_jump", "wrong_order_segment")
    }
    wrong_rows = [row for row in rows if row["corruption_type"] == "wrong_order_segment"]
    wrong_delays = [
        hybrid_result["per_run"][row["development_id"]]["delay"]["intervals"][0]["delay_frames"] for row in wrong_rows
    ]
    readiness = {
        "full_development_hybrid_macro_auroc_at_least_0_85": hybrid_result["macro_auroc"] >= 0.85,
        "leave_one_corruption_type_out_hybrid_macro_auroc_at_least_0_75": all(
            result["macro_auroc"] >= 0.75 for result in leave_one_type_out.values()
        ),
        "hybrid_pooled_primary_fpr_at_most_0_15": hybrid_result["pooled_primary_metrics"]["false_positive_rate"] <= 0.15,
        "hybrid_max_primary_fp_streak_at_most_2": hybrid_result["max_primary_false_positive_streak"] <= 2,
        "all_three_types_detected": set(hybrid_result["detected_corruption_types"]) == {
            "dynamic_occlusion", "low_overlap_jump", "wrong_order_segment"
        },
        "mean_delay_at_most_one": hybrid_result["mean_delay_frames"] is not None and hybrid_result["mean_delay_frames"] <= 1.0,
        "all_wrong_order_runs_detected_at_delay_at_most_one": all(value is not None and value <= 1 for value in wrong_delays),
        "timestamp_clean_prefix_false_positives_zero": all(
            not any(position < 15 for position in row["timestamp_alarm_positions"]) for row in rows
        ),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if all(readiness.values()) else "FAIL",
        "purpose": "disclosed_v1_v2_development_only_not_blind_formal_v3",
        "frozen_continuous_threshold": THRESHOLD,
        "inputs": {
            "v2_development_search": _artifact(V1_SEARCH, search_bytes),
            "v1_development_metrics": _artifact(V1_DEVELOPMENT_METRICS, v1_dev_bytes),
            "v1_holdout_metrics": _artifact(V1_HOLDOUT_METRICS, v1_holdout_bytes),
            "v2_holdout_metrics": _artifact(V2_HOLDOUT_METRICS, v2_metrics_bytes),
            "input_manifests": input_artifacts,
        },
        "online_timestamp_sidecars": online_sidecars,
        "runs": rows,
        "continuous": continuous_result,
        "hybrid": hybrid_result,
        "leave_one_corruption_type_out": leave_one_type_out,
        "wrong_order_hybrid_delays": wrong_delays,
        "readiness": readiness,
    }
    script_path = Path(__file__).resolve()
    result["generator"] = _artifact(script_path, script_path.read_bytes())
    markdown = "\n".join([
        "# Detector v3 disclosed development derivation",
        "",
        f"Status: **{result['status']}** (all v1/v2 inputs are disclosed development data; this is not formal-v3 evidence).",
        "",
        f"- Runs: `{len(rows)}`; frozen continuous threshold: `{THRESHOLD:.12g}`",
        f"- Continuous macro-AUROC/FPR: `{continuous_result['macro_auroc']:.9f}` / `{continuous_result['pooled_primary_metrics']['false_positive_rate']:.9f}`",
        f"- Hybrid macro-AUROC/FPR: `{hybrid_result['macro_auroc']:.9f}` / `{hybrid_result['pooled_primary_metrics']['false_positive_rate']:.9f}`",
        "- Leave-one-corruption-type-out hybrid macro-AUROC: `{}".format(
            {key: round(value["macro_auroc"], 9) for key, value in leave_one_type_out.items()}
        ) + "`",
        f"- Hybrid detected types: `{hybrid_result['detected_corruption_types']}`; wrong-order delays: `{wrong_delays}`",
        f"- Timestamp clean-prefix false positives: `{readiness['timestamp_clean_prefix_false_positives_zero']}`",
        "",
        "Each online timestamp sidecar was constructed before labels/continuous metrics were joined. It reads final RGB paths plus raw rgb.txt only; timestamps rewritten to a monotonic sequence are explicitly outside this channel's claim.",
        "",
    ])
    _publish(output_dir.resolve(strict=False), {"development.json": _canonical_json(result), "report.md": markdown.encode("utf-8")})
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        result = derive(args.output_dir)
    except (DetectorV3DevelopmentError, OSError, ValueError) as error:
        parser.error(str(error))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
