#!/usr/bin/env python3
"""Create immutable CPU-only v3 evidence for a disclosed timestamp reorder.

The online sidecar is derived only from final RGB paths and raw ``rgb.txt``.
Labels and continuous Detector-v2 metrics are loaded later, in the offline
evaluation half of this script, solely to quantify the already disclosed
failure and the fixed v3 hybrid response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from stateguard3r.detection import detection_delay, threshold_metrics
from stateguard3r.detection_v3 import hybridize_v2_continuous_scores
from stateguard3r.timestamp_order_v3 import (
    capture_timestamp_records,
    timestamp_order_sidecar,
    validate_timestamp_order_sidecar,
)


SCHEMA_VERSION = "stateguard3r.detector-v3-order-analysis.v1"


class DetectorV3AnalysisError(ValueError):
    """Raised when disclosed evidence cannot form a strict v3 analysis."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_frozen(path: Path, *, label: str) -> bytes:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise DetectorV3AnalysisError(f"cannot stat {label}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DetectorV3AnalysisError(f"{label} must be a regular non-symlink file")
    if stat.S_IMODE(metadata.st_mode) != 0o444:
        raise DetectorV3AnalysisError(f"{label} must be frozen at mode 0444")
    return path.read_bytes()


def _json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise DetectorV3AnalysisError(f"{label} contains non-finite JSON {value!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DetectorV3AnalysisError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, DetectorV3AnalysisError) as error:
        raise DetectorV3AnalysisError(f"{label} is not strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise DetectorV3AnalysisError(f"{label} must be a JSON object")
    return value


def _artifact(path: Path, payload: bytes) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(payload), "size_bytes": len(payload)}


def _input_paths(input_manifest: Mapping[str, Any]) -> list[Path]:
    """Select only final RGB paths; deliberately ignore all frame metadata."""

    frames = input_manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise DetectorV3AnalysisError("input manifest lacks a non-empty frames list")
    paths: list[Path] = []
    for frame_id, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or frame.get("frame_index") != frame_id:
            raise DetectorV3AnalysisError("input manifest frame IDs are not contiguous")
        path = frame.get("path")
        if not isinstance(path, str) or not path:
            raise DetectorV3AnalysisError(f"input frame {frame_id} lacks RGB path")
        paths.append(Path(path))
    return paths


def _continuous_context(metrics: Mapping[str, Any], *, run_id: str) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    try:
        method = metrics["methods"]["combined"]
        run = method["per_run"][run_id]
        score = np.asarray(run["scores"], dtype=np.float64)
        threshold = float(run["threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise DetectorV3AnalysisError(f"formal metrics lacks combined context for {run_id!r}") from error
    if score.ndim != 1 or score.size == 0 or not np.all(np.isfinite(score)) or not np.isfinite(threshold):
        raise DetectorV3AnalysisError("combined score/threshold is non-finite")
    rows = metrics.get("runs")
    if not isinstance(rows, list):
        raise DetectorV3AnalysisError("formal metrics lacks run labels")
    matching = [row for row in rows if isinstance(row, Mapping) and row.get("run_id") == run_id]
    if len(matching) != 1:
        raise DetectorV3AnalysisError(f"formal metrics lacks exactly one labels row for {run_id!r}")
    labels = np.asarray(matching[0].get("primary_labels"), dtype=np.int64)
    mask = np.asarray(matching[0].get("primary_mask"), dtype=bool)
    if labels.shape != score.shape or mask.shape != score.shape or not np.all(np.isin(labels, (0, 1))):
        raise DetectorV3AnalysisError("formal labels/mask are incompatible with combined score")
    return score, threshold, labels, mask


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _freeze_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _publish(output_dir: Path, files: Mapping[str, bytes]) -> Path:
    if output_dir.parent != ROOT / "outputs" or output_dir.exists():
        raise DetectorV3AnalysisError("output_dir must be a new direct child of outputs")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".staging", dir=output_dir.parent))
    published = False
    try:
        for name, payload in files.items():
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        _freeze_tree(staging)
        os.replace(staging, output_dir)
        published = True
        return output_dir
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def analyze(
    *,
    input_manifest_path: Path,
    rgb_txt_path: Path,
    dataset_root: Path,
    formal_metrics_path: Path,
    run_id: str,
    output_dir: Path,
) -> Path:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise DetectorV3AnalysisError("analysis requires CUDA_VISIBLE_DEVICES='' ")
    input_bytes = _read_frozen(input_manifest_path.resolve(strict=True), label="input manifest")
    metrics_bytes = _read_frozen(formal_metrics_path.resolve(strict=True), label="formal metrics")
    input_manifest = _json_object(input_bytes, label="input manifest")
    metrics = _json_object(metrics_bytes, label="formal metrics")
    captures, provenance = capture_timestamp_records(
        _input_paths(input_manifest), rgb_txt=rgb_txt_path.resolve(strict=True), dataset_root=dataset_root.resolve(strict=True)
    )
    sidecar = timestamp_order_sidecar(captures, provenance=provenance)
    predicates = validate_timestamp_order_sidecar(sidecar)
    continuous, threshold, labels, mask = _continuous_context(metrics, run_id=run_id)
    hybrid = hybridize_v2_continuous_scores(
        continuous, continuous_threshold=threshold, timestamp_sidecar=sidecar
    )
    hybrid_scores = np.asarray(hybrid["hybrid_scores"], dtype=np.float64)
    continuous_metrics = threshold_metrics(labels[mask], continuous[mask], threshold)
    hybrid_metrics = threshold_metrics(labels[mask], hybrid_scores[mask], threshold)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "scope": "disclosed_v2_failure_analysis_only_not_blind_formal_v3",
        "run_id": run_id,
        "inputs": {
            "input_manifest": _artifact(input_manifest_path.resolve(strict=True), input_bytes),
            "formal_metrics": _artifact(formal_metrics_path.resolve(strict=True), metrics_bytes),
        },
        "timestamp_sidecar": sidecar,
        "continuous": {
            "threshold": threshold,
            "scores": continuous.tolist(),
            "primary_metrics": continuous_metrics,
            "delay": detection_delay(labels, continuous, threshold),
        },
        "hybrid": {
            "scores": hybrid_scores.tolist(),
            "alarms": [bool(value) for value in np.asarray(hybrid["alarms"], dtype=bool)],
            "attribution": hybrid["attribution"],
            "primary_metrics": hybrid_metrics,
            "delay": detection_delay(labels, hybrid_scores, threshold),
        },
        "timestamp_violation_positions": [index for index, value in enumerate(predicates) if value is True],
    }
    markdown = "\n".join(
        [
            "# Detector v3 disclosed timestamp-order analysis",
            "",
            "Status: **PASS** (disclosed development/failure analysis only; not blind formal-v3 evidence).",
            "",
            f"- Run: `{run_id}`",
            f"- Frozen Detector-v2 continuous threshold: `{threshold:.12g}`",
            f"- Timestamp violation positions: `{report['timestamp_violation_positions']}`",
            f"- Continuous TP/FN/FP: `{continuous_metrics['true_positive']}/{continuous_metrics['false_negative']}/{continuous_metrics['false_positive']}`",
            f"- Hybrid TP/FN/FP: `{hybrid_metrics['true_positive']}/{hybrid_metrics['false_negative']}/{hybrid_metrics['false_positive']}`",
            f"- Continuous delay: `{report['continuous']['delay']['mean_delay_frames']}`; hybrid delay: `{report['hybrid']['delay']['mean_delay_frames']}`",
            "",
            "The sidecar was derived only from final RGB paths and raw rgb.txt capture timestamps. Labels were used only after sidecar construction for this disclosed metric report.",
            "",
        ]
    )
    script_path = Path(__file__).resolve()
    report["generator"] = _artifact(script_path, script_path.read_bytes())
    _publish(output_dir.resolve(strict=False), {"analysis.json": _canonical_bytes(report), "report.md": markdown.encode("utf-8")})
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_manifest", type=Path)
    parser.add_argument("rgb_txt", type=Path)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("formal_metrics", type=Path)
    parser.add_argument("run_id")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        result = analyze(
            input_manifest_path=args.input_manifest,
            rgb_txt_path=args.rgb_txt,
            dataset_root=args.dataset_root,
            formal_metrics_path=args.formal_metrics,
            run_id=args.run_id,
            output_dir=args.output_dir,
        )
    except (DetectorV3AnalysisError, OSError, ValueError) as error:
        parser.error(str(error))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
