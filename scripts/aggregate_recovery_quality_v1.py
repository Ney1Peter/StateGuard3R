#!/usr/bin/env python3
"""Irreversibly aggregate the eight predeclared recovery-quality v1 evaluations."""

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
SCENES = ("rgbd_dataset_freiburg3_walking_static", "rgbd_dataset_freiburg3_walking_xyz")
CONDITIONS = ("clean", "dynamic", "wrong-order", "low-overlap")
EVENT_CONDITIONS = CONDITIONS[1:]
SCHEMA_VERSION = "stateguard3r.recovery-quality-v1-aggregation.v1"


class RecoveryQualityAggregationError(ValueError):
    """Raised before an incomplete or substituted formal result can be published."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryQualityAggregationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, *, label: str) -> Path:
    details = os.lstat(path)
    _require(not stat.S_ISLNK(details.st_mode) and stat.S_ISREG(details.st_mode), f"{label} must be a regular non-symlink file")
    _require(stat.S_IMODE(details.st_mode) == 0o444, f"{label} must be mode 0444")
    return path.resolve(strict=True)


def _directory(path: Path, *, label: str) -> Path:
    details = os.lstat(path)
    _require(not stat.S_ISLNK(details.st_mode) and stat.S_ISDIR(details.st_mode), f"{label} must be a directory")
    _require(stat.S_IMODE(details.st_mode) == 0o555, f"{label} must be mode 0555")
    return path.resolve(strict=True)


def _json(path: Path, *, label: str) -> dict[str, Any]:
    def no_constant(value: str) -> None:
        raise RecoveryQualityAggregationError(f"{label} contains non-finite JSON {value!r}")

    def no_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise RecoveryQualityAggregationError(f"{label} repeats key {key!r}")
            output[key] = value
        return output

    payload = json.loads(path.read_bytes(), parse_constant=no_constant, object_pairs_hook=no_duplicate)
    _require(isinstance(payload, dict), f"{label} must be an object")
    return payload


def _finite(value: object, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RecoveryQualityAggregationError(f"{label} must be numeric") from error
    _require(math.isfinite(number), f"{label} must be finite")
    return number


def _read_evaluation(path: Path, *, scene: str, condition: str) -> dict[str, Any]:
    root = _directory(path, label=f"{scene}/{condition} evaluation")
    evaluation_path = _regular(root / "evaluation.json", label=f"{scene}/{condition} evaluation JSON")
    payload = _json(evaluation_path, label=f"{scene}/{condition} evaluation JSON")
    _require(payload.get("schema_version") == "stateguard3r.recovery-quality-evaluation.v1", f"{scene}/{condition} schema differs")
    equivalence = payload.get("baseline_always_commit_byte_equivalence")
    _require(isinstance(equivalence, Mapping) and all(value is True for value in equivalence.values()), f"{scene}/{condition} baseline/always equivalence failed")
    effects = payload.get("effects")
    metrics = payload.get("metrics")
    runs = payload.get("runs")
    _require(isinstance(effects, Mapping) and isinstance(metrics, Mapping) and isinstance(runs, Mapping), f"{scene}/{condition} evaluation layout differs")
    ate_effect = _finite(effects.get("ATE_RMSE_effect"), label=f"{scene}/{condition} ATE effect")
    rpe_effect = _finite(effects.get("RPE_translation_RMSE_effect"), label=f"{scene}/{condition} RPE effect")
    baseline_run = _regular(Path(str(runs.get("baseline", {}).get("path"))), label=f"{scene}/{condition} baseline run")
    policy_run = _regular(Path(str(runs.get("detector_policy", {}).get("path"))), label=f"{scene}/{condition} policy run")
    _require(runs["baseline"].get("sha256") == _sha256(baseline_run) and runs["detector_policy"].get("sha256") == _sha256(policy_run), f"{scene}/{condition} run artifact hash differs")
    baseline_metadata = _json(baseline_run, label=f"{scene}/{condition} baseline run")
    policy_metadata = _json(policy_run, label=f"{scene}/{condition} policy run")
    baseline_runtime = _finite(baseline_metadata.get("runtime_seconds"), label=f"{scene}/{condition} baseline runtime")
    policy_runtime = _finite(policy_metadata.get("runtime_seconds"), label=f"{scene}/{condition} policy runtime")
    state = policy_metadata.get("state_policy")
    _require(isinstance(state, Mapping) and state.get("name") == "detector-v3-quality-prior-alarm", f"{scene}/{condition} policy identity differs")
    timeline_path = _regular(Path(str(state.get("timeline_path"))), label=f"{scene}/{condition} state timeline")
    timeline = _json(timeline_path, label=f"{scene}/{condition} state timeline")
    transactions = timeline.get("transactions")
    _require(isinstance(transactions, list), f"{scene}/{condition} transactions are missing")
    actions = [str(row.get("action")) for row in transactions if isinstance(row, Mapping)]
    dropped = timeline.get("dropped_transaction_frame_ids")
    _require(dropped == [], f"{scene}/{condition} silently dropped a transaction")
    return {
        "scene": scene,
        "condition": condition,
        "evaluation": {"path": str(evaluation_path), "sha256": _sha256(evaluation_path)},
        "effects": {"ATE_RMSE_effect": ate_effect, "RPE_translation_RMSE_effect": rpe_effect},
        "metrics": metrics,
        "runtime_ratio": policy_runtime / baseline_runtime if baseline_runtime > 0 else float("inf"),
        "alarm_positions": state.get("alarm_positions"),
        "transaction_actions": actions,
        "restore_failure": False,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def aggregate(evaluation_dirs: Mapping[tuple[str, str], Path], output_dir: Path) -> Path:
    """Publish a single attempt seal and `GO`/`NO_GO` decision for exactly 8 inputs."""

    expected = {(scene, condition) for scene in SCENES for condition in CONDITIONS}
    _require(set(evaluation_dirs) == expected, "formal aggregation requires exactly two scenes times four conditions")
    output = output_dir.resolve(strict=False)
    _require(output.parent == OUTPUT_ROOT and not output.exists(), "aggregation output must be a new direct child of outputs")
    rows = [_read_evaluation(evaluation_dirs[(scene, condition)], scene=scene, condition=condition) for scene in SCENES for condition in CONDITIONS]
    event_rows = [row for row in rows if row["condition"] in EVENT_CONDITIONS]
    clean_rows = [row for row in rows if row["condition"] == "clean"]
    _require(len(event_rows) == 6 and len(clean_rows) == 2, "formal event/clean inventory differs")
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    from stateguard3r.recovery_quality import scene_stratified_bootstrap_lower_bound

    def summary(metric: str) -> tuple[float, float, int]:
        values = np.asarray([row["effects"][metric] for row in event_rows], dtype=np.float64)
        grouped = {scene: [row["effects"][metric] for row in event_rows if row["scene"] == scene] for scene in SCENES}
        return float(np.median(values)), scene_stratified_bootstrap_lower_bound(grouped, seed=20260806, samples=10_000), int(np.sum(values > 0.0))

    ate_median, ate_lower, ate_positive = summary("ATE_RMSE_effect")
    rpe_median, rpe_lower, _ = summary("RPE_translation_RMSE_effect")
    clean_ok = all(row["effects"][metric] >= -0.05 for row in clean_rows for metric in ("ATE_RMSE_effect", "RPE_translation_RMSE_effect"))
    runtime_median = float(np.median(np.asarray([row["runtime_ratio"] for row in rows], dtype=np.float64)))
    wrong_order_actions = [row["transaction_actions"] for row in event_rows if row["condition"] == "wrong-order"]
    wrong_order_action_ok = all(any("hold" in action for action in actions) and any("replay" in action for action in actions) for actions in wrong_order_actions)
    gates = {
        "all_evaluations_and_equivalence_passed": True,
        "no_restore_failure": all(not row["restore_failure"] for row in rows),
        "median_ATE_effect_gte_10_percent": ate_median >= 0.10,
        "median_translation_RPE_effect_gte_10_percent": rpe_median >= 0.10,
        "at_least_four_positive_ATE_trials": ate_positive >= 4,
        "ATE_bootstrap_lower_bound_positive": ate_lower > 0.0,
        "translation_RPE_bootstrap_lower_bound_positive": rpe_lower > 0.0,
        "clean_degradation_within_5_percent": clean_ok,
        "median_runtime_increase_within_20_percent": runtime_median <= 1.20,
        "wrong_order_hold_and_replay_each_scene": wrong_order_action_ok,
    }
    decision = "RECOVERY_QUALITY_GO" if all(gates.values()) else "RECOVERY_QUALITY_NO_GO"
    output.mkdir()
    result = {"schema_version": SCHEMA_VERSION, "decision": decision, "gates": gates, "summary": {"event_trial_count": 6, "ATE_median_effect": ate_median, "ATE_bootstrap_lower_95": ate_lower, "ATE_positive_trial_count": ate_positive, "translation_RPE_median_effect": rpe_median, "translation_RPE_bootstrap_lower_95": rpe_lower, "policy_to_baseline_runtime_median_ratio": runtime_median}, "evaluations": rows, "attempt_seal": "this directory is the sole aggregation attempt for its frozen run inventory"}
    _write_json(output / "result.json", result)
    _write_json(output / "attempt-seal.json", {"schema_version": SCHEMA_VERSION, "result_sha256": _sha256(output / "result.json"), "sealed": True})
    for path in output.iterdir():
        path.chmod(0o444)
    output.chmod(0o555)
    return output


def _parse_evaluation(value: str) -> tuple[tuple[str, str], Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("evaluation must be scene=condition=directory")
    return (parts[0], parts[1]), Path(parts[2])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="append", required=True, type=_parse_evaluation)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    values: dict[tuple[str, str], Path] = {}
    for key, path in args.evaluation:
        if key in values:
            parser.error(f"duplicate evaluation {key}")
        values[key] = path
    output = aggregate(values, args.output_dir)
    print(json.dumps({"output_dir": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
