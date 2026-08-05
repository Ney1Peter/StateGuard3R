#!/usr/bin/env python3
"""Construct frozen Detector-v3 shadow alarms without exposing GT to the detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs"
SCHEMA_VERSION = "stateguard3r.recovery-quality-alarm.v1"
FORBIDDEN_ONLINE_TOKENS = ("groundtruth", "ground_truth", "depth", "label", "event", "source_index", "future")


class RecoveryQualityAlarmError(ValueError):
    """Raised when a quality alarm would not be bound to causal, non-GT inputs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryQualityAlarmError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, *, label: str, mode: int | None = None) -> Path:
    try:
        details = os.lstat(path)
    except OSError as error:
        raise RecoveryQualityAlarmError(f"cannot stat {label}: {error}") from error
    _require(not stat.S_ISLNK(details.st_mode) and stat.S_ISREG(details.st_mode), f"{label} must be a regular non-symlink file")
    if mode is not None:
        _require(stat.S_IMODE(details.st_mode) == mode, f"{label} must have mode {mode:04o}")
    return path.resolve(strict=True)


def _directory(path: Path, *, label: str, mode: int = 0o555) -> Path:
    try:
        details = os.lstat(path)
    except OSError as error:
        raise RecoveryQualityAlarmError(f"cannot stat {label}: {error}") from error
    _require(not stat.S_ISLNK(details.st_mode) and stat.S_ISDIR(details.st_mode), f"{label} must be a directory")
    _require(stat.S_IMODE(details.st_mode) == mode, f"{label} must have mode {mode:04o}")
    return path.resolve(strict=True)


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise RecoveryQualityAlarmError(f"{label} contains non-finite JSON {value!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RecoveryQualityAlarmError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_bytes(), parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryQualityAlarmError(f"cannot parse {label}: {error}") from error
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    regular = _regular(path, label=str(path), mode=0o444)
    return {"path": str(regular), "sha256": _sha256(regular), "size_bytes": regular.stat().st_size, "mode_octal": "0444"}


def _health_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RecoveryQualityAlarmError(f"health line {line_number} is invalid JSON") from error
        _require(isinstance(row, dict) and row.get("frame_id") == len(rows), "health frame IDs must be contiguous")
        for key in row:
            normalized = key.lower()
            _require(not any(token in normalized for token in FORBIDDEN_ONLINE_TOKENS), f"health contains forbidden online key {key!r}")
        rows.append(row)
    _require(rows, "health ledger is empty")
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _freeze(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        _require(not path.is_symlink(), f"refusing to freeze symlink {path}")
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def build(
    input_manifest: Path,
    baseline_dir: Path,
    formal_config: Path,
    rgb_listing: Path,
    dataset_root: Path,
    output_dir: Path,
) -> Path:
    """Publish a v3 hybrid alarm artifact from one frozen baseline health ledger."""

    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "alarm build requires CUDA_VISIBLE_DEVICES='' ")
    manifest_path = _regular(input_manifest, label="input manifest", mode=0o444)
    baseline = _directory(baseline_dir, label="baseline output")
    health_path = _regular(baseline / "health.jsonl", label="baseline health", mode=0o444)
    run_path = _regular(baseline / "run.json", label="baseline run metadata", mode=0o444)
    config_path = _regular(formal_config, label="frozen v3 formal config", mode=0o444)
    listing = _regular(rgb_listing, label="raw rgb.txt", mode=0o444)
    raw_root = _directory(dataset_root, label="raw dataset root")
    _require(listing.parent == raw_root, "rgb listing must be the canonical dataset-root/rgb.txt")
    output = output_dir.resolve(strict=False)
    _require(output.parent == OUTPUT_ROOT and not output.exists(), "output must be a new direct child of StateGuard3R/outputs")
    manifest = _strict_json(manifest_path, label="input manifest")
    _require(manifest.get("source_is_read_only") is True, "input manifest source is not read-only")
    health = _health_records(health_path)
    _require(len(health) == manifest.get("frame_count"), "health and input frame counts differ")
    run = _strict_json(run_path, label="baseline run metadata")
    _require(run.get("health_profile") == "v2", "quality shadow baseline must use health-profile v2")
    _require(isinstance(run.get("input_manifest"), Mapping), "baseline run lacks input binding")
    _require(run["input_manifest"].get("path") == str(manifest_path) and run["input_manifest"].get("sha256") == _sha256(manifest_path), "baseline run is not bound to this input manifest")
    config = _strict_json(config_path, label="frozen v3 formal config")
    continuous = config.get("continuous_scoring")
    thresholds = config.get("thresholds")
    floors = config.get("scale_floors")
    _require(isinstance(continuous, Mapping) and isinstance(thresholds, Mapping) and isinstance(floors, Mapping), "formal config lacks frozen v3 fields")
    try:
        scoring = {name: continuous[name] for name in ("window", "min_history", "max_z", "master_seed")}
        threshold = float(thresholds["combined"])
    except (KeyError, TypeError, ValueError) as error:
        raise RecoveryQualityAlarmError("formal config continuous threshold is invalid") from error
    _require(math.isfinite(threshold), "frozen continuous threshold is not finite")
    sys.path.insert(0, str(ROOT / "src"))
    from stateguard3r.detection_v2 import V2ScoringConfig, compute_v2_scores
    from stateguard3r.detection_v3 import hybridize_v2_continuous_scores
    from stateguard3r.input_manifest import load_input_manifest
    from stateguard3r.timestamp_order_v3 import capture_timestamp_records, timestamp_order_sidecar

    loaded = load_input_manifest(manifest_path)
    captures, provenance = capture_timestamp_records(list(loaded.frame_paths), rgb_txt=listing, dataset_root=raw_root)
    sidecar = timestamp_order_sidecar(captures, provenance=provenance)
    v2 = compute_v2_scores(health, config=V2ScoringConfig(**scoring), scale_floors=floors, require_online_overlap=True)
    hybrid = hybridize_v2_continuous_scores(v2["methods"]["combined"]["scores"], continuous_threshold=threshold, timestamp_sidecar=sidecar)
    hybrid_alarms = [bool(value) for value in hybrid["alarms"]]
    policy_alarms = [value and (index == 0 or not hybrid_alarms[index - 1]) for index, value in enumerate(hybrid_alarms)]
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".staging", dir=output.parent))
    try:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "input_manifest": _artifact(manifest_path),
            "baseline": {"run": _artifact(run_path), "health": _artifact(health_path)},
            "formal_v3_config": _artifact(config_path),
            "online_input_policy": {
                "allowed": ["baseline_health_v2_fields", "model_ready_rgb_visual_overlap", "raw_rgb_capture_timestamp"],
                "forbidden": list(FORBIDDEN_ONLINE_TOKENS),
                "future_frames": False,
            },
            "continuous": {
                "config": scoring,
                "threshold": threshold,
                "used_signals": v2["methods"]["combined"]["used_signals"],
                "scores": [float(value) for value in v2["methods"]["combined"]["scores"]],
            },
            "timestamp_order": sidecar,
            "attribution": hybrid["attribution"],
            "hybrid_alarm_positions": [index for index, value in enumerate(hybrid_alarms) if value],
            "policy_alarm_positions": [index for index, value in enumerate(policy_alarms) if value],
            "policy_filter": "causal_hybrid_alarm_rising_edge",
            "causality": "policy frame t may read only frozen detector alarm t-1",
        }
        _write_json(staging / "alarm.json", payload)
        _freeze(staging)
        os.replace(staging, output)
    except Exception:
        for path in sorted(staging.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        staging.rmdir()
        raise
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--formal-config", required=True, type=Path)
    parser.add_argument("--rgb-listing", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    output = build(args.input_manifest, args.baseline_dir, args.formal_config, args.rgb_listing, args.dataset_root, args.output_dir)
    print(json.dumps({"output_dir": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
