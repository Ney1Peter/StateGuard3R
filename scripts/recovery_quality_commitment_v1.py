"""Validate one frozen recovery-quality v1 commitment at every formal boundary."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping


SCHEMA_VERSION = "stateguard3r.recovery-quality-v1-commitment.v1"
SCENES = ("rgbd_dataset_freiburg3_walking_static", "rgbd_dataset_freiburg3_walking_xyz")
CONDITIONS = ("clean", "dynamic", "wrong-order", "low-overlap")
FORWARDS = ("baseline", "always-commit", "detector-policy")


class RecoveryQualityCommitmentError(ValueError):
    """Raised when a formal artifact is not exactly authorized pre-forward."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryQualityCommitmentError(message)


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
        raise RecoveryQualityCommitmentError(f"cannot stat {label}: {error}") from error
    _require(not stat.S_ISLNK(details.st_mode) and stat.S_ISREG(details.st_mode), f"{label} must be a regular non-symlink file")
    if mode is not None:
        _require(stat.S_IMODE(details.st_mode) == mode, f"{label} must be mode {mode:04o}")
    return path.resolve(strict=True)


def _directory(path: Path, *, label: str, mode: int | None = None) -> Path:
    try:
        details = os.lstat(path)
    except OSError as error:
        raise RecoveryQualityCommitmentError(f"cannot stat {label}: {error}") from error
    _require(not stat.S_ISLNK(details.st_mode) and stat.S_ISDIR(details.st_mode), f"{label} must be a directory")
    if mode is not None:
        _require(stat.S_IMODE(details.st_mode) == mode, f"{label} must be mode {mode:04o}")
    return path.resolve(strict=True)


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise RecoveryQualityCommitmentError(f"{label} contains non-finite JSON {value!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, f"{label} repeats key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(path.read_bytes(), parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryQualityCommitmentError(f"cannot parse {label}: {error}") from error
    _require(isinstance(payload, dict), f"{label} must be an object")
    return payload


def artifact(path: Path, *, mode: int | None = 0o444) -> dict[str, Any]:
    """Return immutable provenance for an existing regular artifact."""

    regular = _regular(path, label=str(path), mode=mode)
    return {
        "path": str(regular),
        "sha256": _sha256(regular),
        "size_bytes": regular.stat().st_size,
        "mode_octal": f"{stat.S_IMODE(regular.stat().st_mode):04o}",
    }


def _validate_artifact(value: object, *, label: str, mode: int | None = None) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} artifact is absent")
    raw_path = value.get("path")
    _require(isinstance(raw_path, str), f"{label} artifact path is invalid")
    path = _regular(Path(raw_path), label=label, mode=mode)
    _require(value.get("path") == str(path), f"{label} artifact path differs")
    _require(value.get("sha256") == _sha256(path), f"{label} artifact hash differs")
    _require(value.get("size_bytes") == path.stat().st_size, f"{label} artifact size differs")
    if "mode_octal" in value:
        _require(value.get("mode_octal") == f"{stat.S_IMODE(path.stat().st_mode):04o}", f"{label} artifact mode differs")
    return artifact(path, mode=mode)


def _commitment(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    commitment_path = _regular(path, label="recovery-quality commitment", mode=0o444)
    _directory(commitment_path.parent, label="recovery-quality commitment directory", mode=0o555)
    payload = _strict_json(commitment_path, label="recovery-quality commitment")
    _require(payload.get("schema_version") == SCHEMA_VERSION, "commitment schema differs")
    _require(payload.get("status") == "COMMITTED_PRE_FORMAL_FORWARD", "commitment is not pre-forward frozen")
    _require(payload.get("no_reconfiguration_after_commitment") is True, "commitment does not prohibit reconfiguration")
    return payload, artifact(commitment_path)


def _expected_runs(payload: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows = payload.get("run_inventory")
    _require(isinstance(rows, list), "commitment run inventory is absent")
    expected = {(scene, condition) for scene in SCENES for condition in CONDITIONS}
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), "commitment run inventory row is invalid")
        scene, condition = row.get("scene"), row.get("condition")
        key = (scene, condition)
        _require(key in expected and key not in result, "commitment run inventory differs")
        _require(row.get("run_id") == f"{scene}-{condition}", "commitment run ID differs")
        _require(row.get("forwards") == list(FORWARDS), "commitment forward inventory differs")
        result[key] = row
    _require(set(result) == expected, "commitment run inventory is incomplete")
    return result


def _input_inventory(payload: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    scenes = payload.get("scenes")
    _require(isinstance(scenes, Mapping) and set(scenes) == set(SCENES), "commitment scene inventory differs")
    inventory: dict[tuple[str, str], dict[str, Any]] = {}
    for scene in SCENES:
        scene_payload = scenes.get(scene)
        _require(isinstance(scene_payload, Mapping), f"commitment scene {scene} is invalid")
        conditions = scene_payload.get("conditions")
        _require(isinstance(conditions, Mapping) and set(conditions) == set(CONDITIONS), f"commitment conditions differ for {scene}")
        for condition in CONDITIONS:
            row = conditions[condition]
            _require(isinstance(row, Mapping), f"commitment input row differs for {scene}/{condition}")
            inventory[(scene, condition)] = _validate_artifact(row.get("input_manifest"), label=f"commitment input {scene}/{condition}", mode=0o444)
    return inventory


def _validate_static_provenance(
    payload: Mapping[str, Any],
    *,
    state_guard_commit: str,
    recal3r_commit: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    runner_contract: Mapping[str, Any],
) -> None:
    state_guard = payload.get("StateGuard3R")
    recal3r = payload.get("ReCal3R")
    _require(isinstance(state_guard, Mapping) and state_guard.get("clean") is True and state_guard.get("commit") == state_guard_commit, "StateGuard3R commit is not authorized by commitment")
    _require(isinstance(recal3r, Mapping) and recal3r.get("clean") is True and recal3r.get("commit") == recal3r_commit, "ReCal3R commit is not authorized by commitment")
    _validate_artifact(payload.get("protocol"), label="commitment protocol")
    _validate_artifact(payload.get("frozen_detector_v3_config"), label="commitment detector config", mode=0o444)
    committed_checkpoint = _validate_artifact(payload.get("checkpoint"), label="commitment checkpoint")
    actual_checkpoint = _regular(checkpoint, label="requested checkpoint")
    _require(committed_checkpoint["path"] == str(actual_checkpoint), "requested checkpoint path is not committed")
    _require(committed_checkpoint["sha256"] == checkpoint_sha256 == _sha256(actual_checkpoint), "requested checkpoint hash is not committed")
    _require(payload.get("checkpoint_expected_sha256") == checkpoint_sha256, "commitment expected checkpoint hash differs")
    _require(payload.get("runner_contract") == dict(runner_contract), "formal runner contract differs from commitment")
    policy = payload.get("policy")
    _require(isinstance(policy, Mapping) and policy.get("name") == "detector-v3-quality-prior-alarm" and policy.get("max_hold") == 3, "commitment policy differs")


def authorize(
    commitment_path: Path,
    *,
    input_manifest: Path,
    forward: str,
    state_guard_commit: str,
    recal3r_commit: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    runner_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Authorize exactly one formal forward against the full frozen inventory."""

    payload, commitment_artifact = _commitment(commitment_path)
    _validate_static_provenance(
        payload,
        state_guard_commit=state_guard_commit,
        recal3r_commit=recal3r_commit,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        runner_contract=runner_contract,
    )
    expected_runs = _expected_runs(payload)
    inputs = _input_inventory(payload)
    manifest = artifact(input_manifest, mode=0o444)
    matches = [key for key, committed in inputs.items() if committed == manifest]
    _require(len(matches) == 1, "input manifest is not admitted by the commitment")
    scene, condition = matches[0]
    _require(forward in FORWARDS and forward in expected_runs[(scene, condition)]["forwards"], "forward is not admitted by the commitment")
    return {
        "commitment": commitment_artifact,
        "run_id": f"{scene}-{condition}",
        "scene": scene,
        "condition": condition,
        "forward": forward,
        "input_manifest": manifest,
    }


def same_binding(left: object, right: object) -> bool:
    """Compare serialized formal bindings without accepting missing fields."""

    return isinstance(left, Mapping) and isinstance(right, Mapping) and dict(left) == dict(right)
