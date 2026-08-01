#!/usr/bin/env python3
"""Reproducible CPU orchestration for the formal StateGuard3R pilot.

The three stages deliberately have separate commands.  ``commit`` freezes the
pre-forward split, ``calibrate`` may read development outputs only, and
``evaluate`` refuses to read any holdout output until all three holdout runs
are complete.  Every published directory is immutable and uses atomic
no-replace publication.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from stateguard3r import detection_suite
from stateguard3r.timeline import render_timeline_svg


INPUT_REGISTRY_FILENAME = "formal-pilot-manifest.json"
SOURCE_MANIFEST_FILENAME = "source-manifest.json"
INPUT_MANIFEST_FILENAME = "input-manifest.json"
SPLIT_REGISTRY_FILENAME = "split-registry.json"
HOLDOUT_COMMITMENT_FILENAME = "holdout-commitment.json"
COMMITMENT_MANIFEST_FILENAME = "commitment-manifest.json"
SEARCH_FILENAME = "search.json"
DEVELOPMENT_METRICS_FILENAME = "dev-metrics.json"
FORMAL_CONFIG_FILENAME = "formal-config.json"
CALIBRATION_MANIFEST_FILENAME = "calibration-manifest.json"
HOLDOUT_METRICS_FILENAME = "holdout-metrics.json"
GO_NO_GO_FILENAME = "go-no-go.json"
RUNTIME_SUMMARY_FILENAME = "runtime-summary.json"
EVALUATION_MANIFEST_FILENAME = "evaluation-manifest.json"
TIMELINE_DIRECTORY = "timelines"
HOLDOUT_UNLOCK_FILENAME = "holdout-unlock.json"
RUN_OUTPUT_ARTIFACT_FILENAMES = {
    "checkpoint_load_audit": "checkpoint-load-audit.json",
    "health_jsonl": "health.jsonl",
    "predictions_summary": "predictions-summary.json",
    "trajectory": "trajectory.json",
    "run_json": "run.json",
}

INPUT_SCHEMA_VERSION = "stateguard3r.formal-pilot-inputs.v1"
COMMITMENT_SCHEMA_VERSION = "stateguard3r.formal-detection-commitment.v1"
CALIBRATION_SCHEMA_VERSION = "stateguard3r.formal-detection-calibration.v1"
GATE_SCHEMA_VERSION = "stateguard3r.formal-detection-go-no-go.v1"
RUNTIME_SCHEMA_VERSION = "stateguard3r.formal-detection-runtime-summary.v1"
EVALUATION_SCHEMA_VERSION = "stateguard3r.formal-detection-evaluation.v1"
HOLDOUT_UNLOCK_SCHEMA_VERSION = "stateguard3r.formal-holdout-unlock.v1"
CPU_VALIDATION_SCHEMA_VERSION = "stateguard3r.formal-input-cpu-validation.v1"
CPU_VALIDATION_GENERATOR_RELATIVE_PATH = Path(
    "scripts/validate_formal_pilot_inputs.py"
)
CPU_VALIDATION_DIRECTORY_NAME = "pre-forward-validation"
CPU_VALIDATION_REPORT_FILENAME = "cpu-validation.json"
EXPECTED_SOURCE_SNAPSHOT_ARTIFACT_COUNT = 385
EXPECTED_INPUT_TREE_SNAPSHOT_ENTRY_COUNT = 24
CPU_VALIDATION_CHECKS = frozenset(
    {
        "strict_manifest_validation",
        "cpu_pixel_replay",
        "raw_source_snapshot_unchanged",
        "input_tree_snapshot_unchanged",
        "official_tum_associations",
        "production_runtime_provenance",
        "cuda_hidden",
        "six_run_coverage",
    }
)

SPLITS = ("development", "holdout")
CORRUPTION_ORDER = (
    "dynamic_occlusion",
    "wrong_order_segment",
    "low_overlap_jump",
)
FROZEN_RUN_LAYOUT = {
    "development": (
        ("development-dynamic", "dynamic_occlusion"),
        ("development-wrong", "wrong_order_segment"),
        ("development-low", "low_overlap_jump"),
    ),
    "holdout": (
        ("holdout-dynamic", "dynamic_occlusion"),
        ("holdout-wrong", "wrong_order_segment"),
        ("holdout-low", "low_overlap_jump"),
    ),
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_RECAL3R_COMMIT = "466c7cdf3acd2f589f1d82e5f6391966f19db9ff"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPOSITORY_ROOT / "outputs"
RUNNER_RELATIVE_PATH = Path("scripts/run_recal3r_smoke.py")
PROTOCOL_RELATIVE_PATH = Path("docs/protocols/detection-formal-v1.md")
RECAL3R_ROOT = REPOSITORY_ROOT.parent / "baselines" / "ReCal3R"
RECAL3R_PYTHON = RECAL3R_ROOT / ".venv" / "bin" / "python"
OFFICIAL_LOADER_RELATIVE_PATH = Path("src/dust3r/utils/image.py")
EXPECTED_OFFICIAL_LOADER_SHA256 = (
    "a2738085cdf0f713289a3a42dbd49a1f39325eb3ad590af65970fe4609209d96"
)


class FormalPilotOrchestrationError(ValueError):
    """Raised when a formal stage cannot prove its frozen inputs."""


@dataclass(frozen=True)
class _Snapshot:
    path: Path
    payload: bytes

    def artifact(self, path_text: str) -> dict[str, Any]:
        return {
            "path": path_text,
            "sha256": hashlib.sha256(self.payload).hexdigest(),
            "size_bytes": len(self.payload),
        }


@dataclass(frozen=True)
class _InputRun:
    run_id: str
    dataset_split: str
    corruption_type: str
    raw_frame_sha256s: tuple[str, ...]
    source_manifest: _Snapshot
    input_manifest: _Snapshot

    def suite_input(self) -> detection_suite.DetectionSuiteInput:
        return detection_suite.DetectionSuiteInput(
            run_id=self.run_id,
            dataset_split=self.dataset_split,
            corruption_type=self.corruption_type,
            raw_frame_sha256s=self.raw_frame_sha256s,
            corruption_json=self.input_manifest.payload,
            input_manifest_json=self.input_manifest.payload,
            source_manifest_json=self.source_manifest.payload,
        )

    def suite_run(
        self, health_jsonl: bytes, run_json: bytes
    ) -> detection_suite.DetectionSuiteRun:
        return detection_suite.DetectionSuiteRun(
            run_id=self.run_id,
            dataset_split=self.dataset_split,
            corruption_type=self.corruption_type,
            raw_frame_sha256s=self.raw_frame_sha256s,
            health_jsonl=health_jsonl,
            corruption_json=self.input_manifest.payload,
            input_manifest_json=self.input_manifest.payload,
            source_manifest_json=self.source_manifest.payload,
            run_json=run_json,
        )


@dataclass(frozen=True)
class _InputBundle:
    root: Path
    registry: _Snapshot
    registry_value: Mapping[str, Any]
    runs: tuple[_InputRun, ...]
    builder_split_registry: _Snapshot
    builder_holdout_commitment: _Snapshot


@dataclass(frozen=True)
class _CommitBundle:
    root: Path
    manifest: _Snapshot
    manifest_value: Mapping[str, Any]
    split_registry: _Snapshot
    holdout_commitment: _Snapshot
    protocol: _Snapshot
    validation_report: _Snapshot
    runner_provenance: Mapping[str, Any]
    execution_unlocked_at: datetime


@dataclass(frozen=True)
class _CalibrationBundle:
    root: Path
    manifest: _Snapshot
    manifest_value: Mapping[str, Any]
    search: _Snapshot
    development_metrics: _Snapshot
    formal_config: _Snapshot
    development_outputs: Mapping[str, Mapping[str, _Snapshot]]
    deterministic_replay_verified: bool
    holdout_unlock: _Snapshot
    holdout_unlocked_at: datetime


def _strict_json_loads(payload: bytes, *, name: str) -> Any:
    def reject_constant(value: str) -> None:
        raise FormalPilotOrchestrationError(
            f"{name} contains non-finite JSON constant {value!r}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FormalPilotOrchestrationError(
                    f"{name} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            payload,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalPilotOrchestrationError(
            f"{name} is not strict UTF-8 JSON: {error}"
        ) from error


def _strict_json_object(payload: bytes, *, name: str) -> Mapping[str, Any]:
    value = _strict_json_loads(payload, name=name)
    if not isinstance(value, Mapping):
        raise FormalPilotOrchestrationError(f"{name} must contain a JSON object")
    return value


def _strict_health_records(payload: bytes, *, name: str) -> list[Mapping[str, Any]]:
    try:
        lines = payload.splitlines()
    except AttributeError as error:
        raise FormalPilotOrchestrationError(f"{name} snapshot must be bytes") from error
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        value = _strict_json_loads(line, name=f"{name}:{line_number}")
        if not isinstance(value, Mapping):
            raise FormalPilotOrchestrationError(
                f"{name}:{line_number} must be a JSON object"
            )
        records.append(value)
    if not records:
        raise FormalPilotOrchestrationError(f"{name} must not be empty")
    return records


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FormalPilotOrchestrationError(
            f"cannot encode strict JSON artifact: {error}"
        ) from error


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_text() -> str:
    return _utc_now().isoformat(timespec="microseconds")


def _aware_iso_datetime(value: Any, *, name: str, require_utc: bool = False) -> datetime:
    if (
        not isinstance(value, str)
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
            r"(?:\.\d{1,6})?[+-]\d{2}:\d{2}",
            value,
        )
        is None
    ):
        raise FormalPilotOrchestrationError(
            f"{name} must be a canonical timezone-aware ISO-8601 string"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise FormalPilotOrchestrationError(
            f"{name} is not strict ISO-8601: {error}"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise FormalPilotOrchestrationError(
            f"{name} must be canonical and timezone-aware"
        )
    if require_utc and parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise FormalPilotOrchestrationError(f"{name} must use an explicit UTC offset")
    return parsed


def _expect_exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise FormalPilotOrchestrationError(
            f"{name} has invalid keys; " + "; ".join(details)
        )


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FormalPilotOrchestrationError(f"{name} must be a non-empty string")
    return value


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FormalPilotOrchestrationError(
            f"{name} must be a lowercase 64-character SHA-256"
        )
    return value


def _plain_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise FormalPilotOrchestrationError(
            f"{name} must be an integer of at least {minimum}"
        )
    return value


def _validated_artifact(
    value: Any, *, name: str, expected_path: str | None = None
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalPilotOrchestrationError(f"{name} must be a JSON object")
    _expect_exact_keys(value, {"path", "sha256", "size_bytes"}, name=name)
    path_text = _nonempty_string(value["path"], name=f"{name}.path")
    if expected_path is not None and path_text != expected_path:
        raise FormalPilotOrchestrationError(
            f"{name}.path must equal {expected_path!r}"
        )
    return {
        "path": path_text,
        "sha256": _sha256(value["sha256"], name=f"{name}.sha256"),
        "size_bytes": _plain_integer(
            value["size_bytes"], name=f"{name}.size_bytes"
        ),
    }


def _assert_artifact(
    snapshot: _Snapshot, declared: Mapping[str, Any], *, name: str
) -> None:
    actual = snapshot.artifact(str(declared["path"]))
    if actual != dict(declared):
        raise FormalPilotOrchestrationError(
            f"{name} exact SHA-256/size does not match its manifest"
        )


def _preflight_regular_file(path: Path, *, name: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise FormalPilotOrchestrationError(
            f"required {name} is not complete: {path}: {error}"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise FormalPilotOrchestrationError(
            f"required {name} must be a regular non-symlink file: {path}"
        )


def _read_snapshot(path: Path, *, name: str) -> _Snapshot:
    _preflight_regular_file(path, name=name)
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            payload = stream.read()
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise FormalPilotOrchestrationError(f"cannot read {name} {path}: {error}") from error
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_mode,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_mode,
    )
    if identity_before != identity_after or len(payload) != after.st_size:
        raise FormalPilotOrchestrationError(f"{name} changed while being snapshotted")
    return _Snapshot(path=path, payload=payload)


def _git_output(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FormalPilotOrchestrationError(
            f"cannot inspect Git repository {repository}: {error}"
        ) from error
    return completed.stdout.strip()


def _tracked_file_sha256_at_head(repository: Path, relative_path: Path) -> str:
    """Return the byte SHA-256 of one uniquely tracked file at ``HEAD``."""

    path_text = relative_path.as_posix()
    tracked = _git_output(
        repository,
        "ls-files",
        "--error-unmatch",
        "--",
        path_text,
    )
    if tracked != path_text:
        raise FormalPilotOrchestrationError(
            f"required provenance file is not uniquely tracked: {path_text}"
        )
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "show", f"HEAD:{path_text}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FormalPilotOrchestrationError(
            f"cannot read tracked HEAD bytes for {path_text}: {error}"
        ) from error
    return hashlib.sha256(completed.stdout).hexdigest()


def _clean_repository_commit(repository: Path, *, name: str) -> str:
    root = Path(
        _git_output(repository, "rev-parse", "--show-toplevel")
    ).resolve(strict=True)
    if root != repository.resolve(strict=True):
        raise FormalPilotOrchestrationError(f"{name} repository root mismatch")
    commit = _git_output(repository, "rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise FormalPilotOrchestrationError(f"{name} HEAD is not a full Git commit")
    dirty = _git_output(repository, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise FormalPilotOrchestrationError(
            f"{name} repository must be clean; visible changes:\n{dirty}"
        )
    return commit


def _capture_runner_provenance() -> dict[str, Any]:
    state_guard_commit = _clean_repository_commit(
        REPOSITORY_ROOT, name="StateGuard3R"
    )
    tracked_runner = _git_output(
        REPOSITORY_ROOT,
        "ls-files",
        "--error-unmatch",
        "--",
        RUNNER_RELATIVE_PATH.as_posix(),
    )
    if tracked_runner != RUNNER_RELATIVE_PATH.as_posix():
        raise FormalPilotOrchestrationError(
            "ReCal3R runner must be a uniquely tracked StateGuard3R file"
        )
    runner = _read_snapshot(
        REPOSITORY_ROOT / RUNNER_RELATIVE_PATH, name="tracked ReCal3R runner"
    )
    recal3r_commit = _clean_repository_commit(RECAL3R_ROOT, name="ReCal3R")
    if recal3r_commit != EXPECTED_RECAL3R_COMMIT:
        raise FormalPilotOrchestrationError(
            f"ReCal3R HEAD must equal {EXPECTED_RECAL3R_COMMIT}"
        )
    return {
        "state_guard_repository": {
            "path": str(REPOSITORY_ROOT),
            "commit": state_guard_commit,
            "tracked_worktree_clean": True,
        },
        "runner_script": runner.artifact(RUNNER_RELATIVE_PATH.as_posix()),
        "recal3r_repository": {
            "path": str(RECAL3R_ROOT.resolve(strict=True)),
            "commit": recal3r_commit,
            "tracked_worktree_clean": True,
        },
    }


def _validated_runner_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalPilotOrchestrationError("runner_provenance must be an object")
    _expect_exact_keys(
        value,
        {"state_guard_repository", "runner_script", "recal3r_repository"},
        name="runner_provenance",
    )
    result: dict[str, Any] = {}
    for key, expected_path in (
        ("state_guard_repository", str(REPOSITORY_ROOT)),
        ("recal3r_repository", str(RECAL3R_ROOT.resolve(strict=True))),
    ):
        repository = value[key]
        if not isinstance(repository, Mapping):
            raise FormalPilotOrchestrationError(f"runner_provenance.{key} must be an object")
        _expect_exact_keys(
            repository,
            {"path", "commit", "tracked_worktree_clean"},
            name=f"runner_provenance.{key}",
        )
        commit = repository.get("commit")
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise FormalPilotOrchestrationError(
                f"runner_provenance.{key}.commit must be a full Git SHA"
            )
        if repository.get("path") != expected_path or repository.get(
            "tracked_worktree_clean"
        ) is not True:
            raise FormalPilotOrchestrationError(
                f"runner_provenance.{key} path/clean flag mismatch"
            )
        result[key] = dict(repository)
    if result["recal3r_repository"]["commit"] != EXPECTED_RECAL3R_COMMIT:
        raise FormalPilotOrchestrationError("bound ReCal3R commit mismatch")
    result["runner_script"] = _validated_artifact(
        value["runner_script"],
        name="runner_provenance.runner_script",
        expected_path=RUNNER_RELATIVE_PATH.as_posix(),
    )
    return result


def _verify_current_runner_provenance(value: Any) -> dict[str, Any]:
    declared = _validated_runner_provenance(value)
    current = _capture_runner_provenance()
    if current != declared:
        raise FormalPilotOrchestrationError(
            "current repositories/runner differ from the pre-forward commitment"
        )
    return declared


def _snapshot_frozen_protocol(path: str | os.PathLike[str]) -> _Snapshot:
    expected = REPOSITORY_ROOT / PROTOCOL_RELATIVE_PATH
    try:
        requested = Path(path).resolve(strict=True)
        expected_resolved = expected.resolve(strict=True)
    except OSError as error:
        raise FormalPilotOrchestrationError(
            f"cannot resolve frozen formal protocol: {error}"
        ) from error
    if requested != expected_resolved:
        raise FormalPilotOrchestrationError(
            f"--protocol must be the fixed tracked {PROTOCOL_RELATIVE_PATH.as_posix()}"
        )
    snapshot = _read_snapshot(expected, name="fixed formal protocol")
    tracked_sha256 = _tracked_file_sha256_at_head(
        REPOSITORY_ROOT, PROTOCOL_RELATIVE_PATH
    )
    if hashlib.sha256(snapshot.payload).hexdigest() != tracked_sha256:
        raise FormalPilotOrchestrationError(
            "fixed formal protocol bytes differ from the clean tracked HEAD blob"
        )
    return snapshot


def _validate_run_json_runner(
    snapshot: _Snapshot, runner_provenance: Mapping[str, Any], *, run_id: str
) -> None:
    value = _strict_json_object(snapshot.payload, name=f"{run_id} run.json")
    if value.get("baseline_commit") != EXPECTED_RECAL3R_COMMIT:
        raise FormalPilotOrchestrationError(
            f"{run_id} baseline_commit differs from frozen ReCal3R"
        )
    runner = value.get("runner")
    if not isinstance(runner, Mapping):
        raise FormalPilotOrchestrationError(f"{run_id} run.json lacks runner provenance")
    if runner.get("commit") != runner_provenance["state_guard_repository"]["commit"]:
        raise FormalPilotOrchestrationError(
            f"{run_id} runner.commit differs from pre-forward commitment"
        )
    if runner.get("script_sha256") != runner_provenance["runner_script"]["sha256"]:
        raise FormalPilotOrchestrationError(
            f"{run_id} runner script SHA differs from pre-forward commitment"
        )
    started_at = _aware_iso_datetime(
        value.get("started_at"), name=f"{run_id} run.json started_at"
    )
    finished_at = _aware_iso_datetime(
        value.get("finished_at"), name=f"{run_id} run.json finished_at"
    )
    if finished_at <= started_at:
        raise FormalPilotOrchestrationError(
            f"{run_id} run interval must have finished_at strictly after started_at"
        )


def _validated_split_run_timeline(
    runs: Sequence[_InputRun],
    outputs: Mapping[str, Mapping[str, _Snapshot]],
    *,
    split: str,
    earliest_start: datetime,
    latest_finish: datetime,
) -> tuple[tuple[datetime, datetime], ...]:
    intervals: list[tuple[datetime, datetime]] = []
    previous_finish: datetime | None = None
    for run in runs:
        value = _strict_json_object(
            outputs[run.run_id]["run_json"].payload,
            name=f"{run.run_id} run.json",
        )
        started_at = _aware_iso_datetime(
            value.get("started_at"), name=f"{run.run_id} run.json started_at"
        )
        finished_at = _aware_iso_datetime(
            value.get("finished_at"), name=f"{run.run_id} run.json finished_at"
        )
        if finished_at <= started_at:
            raise FormalPilotOrchestrationError(
                f"{run.run_id} run interval is not strictly positive"
            )
        if started_at < earliest_start:
            raise FormalPilotOrchestrationError(
                f"{run.run_id} {split} run started before its immutable unlock"
            )
        if finished_at > latest_finish:
            raise FormalPilotOrchestrationError(
                f"{run.run_id} {split} run finished after the consuming stage began"
            )
        if previous_finish is not None and started_at < previous_finish:
            raise FormalPilotOrchestrationError(
                f"{split} runs violate frozen dynamic->wrong->low serial order"
            )
        intervals.append((started_at, finished_at))
        previous_finish = finished_at
    if len(intervals) != 3:
        raise FormalPilotOrchestrationError(
            f"{split} timeline must contain exactly three runs"
        )
    return tuple(intervals)


def _resolve_root(path: str | os.PathLike[str], *, name: str) -> Path:
    unresolved = Path(path)
    try:
        unresolved_metadata = os.lstat(unresolved)
    except OSError as error:
        raise FormalPilotOrchestrationError(f"cannot resolve {name}: {error}") from error
    if stat.S_ISLNK(unresolved_metadata.st_mode):
        raise FormalPilotOrchestrationError(f"{name} may not be a symlink")
    try:
        root = unresolved.resolve(strict=True)
    except OSError as error:
        raise FormalPilotOrchestrationError(f"cannot resolve {name}: {error}") from error
    if not root.is_dir():
        raise FormalPilotOrchestrationError(f"{name} must be a directory: {root}")
    return root


def _validated_child_directory(
    root: Path, components: Sequence[str], *, name: str
) -> Path:
    current = root
    for component in components:
        if (
            not isinstance(component, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", component) is None
        ):
            raise FormalPilotOrchestrationError(
                f"{name} contains an unsafe path component {component!r}"
            )
        current = current / component
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise FormalPilotOrchestrationError(
                f"required {name} directory is missing: {current}: {error}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise FormalPilotOrchestrationError(
                f"{name} must be a real non-symlink directory: {current}"
            )
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise FormalPilotOrchestrationError(
            f"{name} escapes its expected root {root}"
        ) from error
    if resolved != current:
        raise FormalPilotOrchestrationError(
            f"{name} resolves through an unexpected alias or symlink"
        )
    return resolved


def _read_input_registry(
    input_root: str | os.PathLike[str],
) -> tuple[Path, _Snapshot, Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    root = _resolve_root(input_root, name="INPUT_ROOT")
    snapshot = _read_snapshot(
        root / INPUT_REGISTRY_FILENAME, name="formal pilot input registry"
    )
    value = _strict_json_object(snapshot.payload, name=INPUT_REGISTRY_FILENAME)
    if value.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise FormalPilotOrchestrationError("input registry schema_version mismatch")
    if value.get("seed") != 0 or value.get("frame_count") != 30:
        raise FormalPilotOrchestrationError(
            "input registry must retain formal seed=0 and frame_count=30"
        )
    raw_runs = value.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != 6:
        raise FormalPilotOrchestrationError("input registry must contain exactly six runs")
    records: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(raw_runs):
        if not isinstance(record, Mapping):
            raise FormalPilotOrchestrationError(
                f"input registry runs[{index}] must be an object"
            )
        run_id = _nonempty_string(record.get("run_id"), name=f"runs[{index}].run_id")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) is None:
            raise FormalPilotOrchestrationError(
                f"runs[{index}].run_id is not a safe path component"
            )
        if run_id in seen_ids:
            raise FormalPilotOrchestrationError(f"duplicate input run ID {run_id!r}")
        seen_ids.add(run_id)
        split = record.get("dataset_split")
        corruption_type = record.get("corruption_type")
        if split not in SPLITS or corruption_type not in CORRUPTION_ORDER:
            raise FormalPilotOrchestrationError(
                f"input registry run {run_id!r} has invalid split/type"
            )
        records.append(record)
    execution_order = value.get("fixed_execution_order")
    if not isinstance(execution_order, Mapping) or set(execution_order) != set(SPLITS):
        raise FormalPilotOrchestrationError(
            "input registry lacks the exact development/holdout execution order"
        )
    for split in SPLITS:
        split_records = [record for record in records if record["dataset_split"] == split]
        if len(split_records) != 3:
            raise FormalPilotOrchestrationError(f"input registry needs three {split} runs")
        by_position = sorted(
            split_records,
            key=lambda record: _plain_integer(
                record.get("execution_position_within_split"),
                name=f"{record['run_id']}.execution_position_within_split",
            ),
        )
        if [record["execution_position_within_split"] for record in by_position] != [0, 1, 2]:
            raise FormalPilotOrchestrationError(f"{split} execution positions must be 0,1,2")
        if tuple(record["corruption_type"] for record in by_position) != CORRUPTION_ORDER:
            raise FormalPilotOrchestrationError(f"{split} corruption order is not frozen v1")
        ordered_ids = [record["run_id"] for record in by_position]
        actual_layout = tuple(
            (str(record["run_id"]), str(record["corruption_type"]))
            for record in by_position
        )
        if actual_layout != FROZEN_RUN_LAYOUT[split]:
            raise FormalPilotOrchestrationError(
                f"input registry {split} run IDs/types are not the frozen v1 layout"
            )
        if execution_order[split] != ordered_ids:
            raise FormalPilotOrchestrationError(
                f"input registry {split} fixed_execution_order mismatch"
            )
    records.sort(
        key=lambda record: (
            SPLITS.index(str(record["dataset_split"])),
            int(record["execution_position_within_split"]),
        )
    )
    return root, snapshot, value, tuple(records)


def _load_input_bundle(
    input_root: str | os.PathLike[str],
    *,
    registry_snapshot: _Snapshot | None = None,
    registry_value: Mapping[str, Any] | None = None,
    registry_records: Sequence[Mapping[str, Any]] | None = None,
) -> _InputBundle:
    if registry_snapshot is None or registry_value is None or registry_records is None:
        root, registry_snapshot, registry_value, records = _read_input_registry(input_root)
    else:
        root = _resolve_root(input_root, name="INPUT_ROOT")
        records = tuple(registry_records)

    runs: list[_InputRun] = []
    for record in records:
        run_id = str(record["run_id"])
        split = str(record["dataset_split"])
        run_root = _validated_child_directory(
            root, (split, run_id), name=f"{run_id} input run"
        )
        expected_source = f"{split}/{run_id}/{SOURCE_MANIFEST_FILENAME}"
        expected_input = f"{split}/{run_id}/{INPUT_MANIFEST_FILENAME}"
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise FormalPilotOrchestrationError(f"{run_id}.artifacts must be an object")
        _expect_exact_keys(
            artifacts,
            {"source_manifest", "input_manifest", "corruption_json"},
            name=f"{run_id}.artifacts",
        )
        source_declared = _validated_artifact(
            artifacts["source_manifest"],
            name=f"{run_id}.artifacts.source_manifest",
            expected_path=expected_source,
        )
        input_declared = _validated_artifact(
            artifacts["input_manifest"],
            name=f"{run_id}.artifacts.input_manifest",
            expected_path=expected_input,
        )
        corruption_declared = _validated_artifact(
            artifacts["corruption_json"],
            name=f"{run_id}.artifacts.corruption_json",
            expected_path=expected_input,
        )
        if corruption_declared != input_declared:
            raise FormalPilotOrchestrationError(
                f"{run_id} corruption_json must be the exact input-manifest bytes"
            )
        source_snapshot = _read_snapshot(
            run_root / SOURCE_MANIFEST_FILENAME, name=f"{run_id} source manifest"
        )
        input_snapshot = _read_snapshot(
            run_root / INPUT_MANIFEST_FILENAME, name=f"{run_id} input manifest"
        )
        _assert_artifact(source_snapshot, source_declared, name=f"{run_id} source manifest")
        _assert_artifact(input_snapshot, input_declared, name=f"{run_id} input manifest")
        _strict_json_object(source_snapshot.payload, name=f"{run_id} source manifest")
        _strict_json_object(input_snapshot.payload, name=f"{run_id} input manifest")
        raw_shas = record.get("raw_frame_sha256s")
        if not isinstance(raw_shas, list) or len(raw_shas) != 30:
            raise FormalPilotOrchestrationError(f"{run_id} needs 30 raw RGB hashes")
        validated_raw = tuple(
            _sha256(value, name=f"{run_id}.raw_frame_sha256s[{index}]")
            for index, value in enumerate(raw_shas)
        )
        if len(set(validated_raw)) != 30:
            raise FormalPilotOrchestrationError(f"{run_id} raw RGB hashes must be unique")
        run = _InputRun(
            run_id=run_id,
            dataset_split=split,
            corruption_type=str(record["corruption_type"]),
            raw_frame_sha256s=validated_raw,
            source_manifest=source_snapshot,
            input_manifest=input_snapshot,
        )
        computed = detection_suite.make_run_commitment(run.suite_input())
        if record.get("suite_run_commitment") != computed:
            raise FormalPilotOrchestrationError(
                f"{run_id} suite commitment differs from the input registry"
            )
        runs.append(run)

    formal_artifacts = registry_value.get("formal_commitment_artifacts")
    if not isinstance(formal_artifacts, Mapping):
        raise FormalPilotOrchestrationError(
            "input registry formal_commitment_artifacts must be an object"
        )
    _expect_exact_keys(
        formal_artifacts,
        {"split_registry", "holdout_commitment"},
        name="formal_commitment_artifacts",
    )
    split_declared = _validated_artifact(
        formal_artifacts["split_registry"],
        name="formal_commitment_artifacts.split_registry",
        expected_path=SPLIT_REGISTRY_FILENAME,
    )
    holdout_declared = _validated_artifact(
        formal_artifacts["holdout_commitment"],
        name="formal_commitment_artifacts.holdout_commitment",
        expected_path=HOLDOUT_COMMITMENT_FILENAME,
    )
    split_snapshot = _read_snapshot(
        root / SPLIT_REGISTRY_FILENAME, name="builder split registry"
    )
    holdout_snapshot = _read_snapshot(
        root / HOLDOUT_COMMITMENT_FILENAME, name="builder holdout commitment"
    )
    _assert_artifact(split_snapshot, split_declared, name="builder split registry")
    _assert_artifact(
        holdout_snapshot, holdout_declared, name="builder holdout commitment"
    )
    _strict_json_object(split_snapshot.payload, name="builder split registry")
    _strict_json_object(holdout_snapshot.payload, name="builder holdout commitment")
    return _InputBundle(
        root=root,
        registry=registry_snapshot,
        registry_value=registry_value,
        runs=tuple(runs),
        builder_split_registry=split_snapshot,
        builder_holdout_commitment=holdout_snapshot,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for candidate in [path, *path.rglob("*")]:
        if candidate.is_dir() and not candidate.is_symlink():
            candidate.chmod(0o700)
    shutil.rmtree(path)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as error:
        raise FormalPilotOrchestrationError(
            "atomic no-replace directory publication is unavailable"
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FormalPilotOrchestrationError(
            f"refusing to overwrite existing output directory: {destination}"
        )
    raise FormalPilotOrchestrationError(
        "cannot atomically publish output directory with no-replace semantics: "
        + os.strerror(error_number)
    )


def _publish_immutable_tree(
    output_dir: str | os.PathLike[str], files: Mapping[str, bytes]
) -> Path:
    output_root = _resolve_root(OUTPUT_ROOT, name="StateGuard3R OUTPUT_ROOT")
    output = Path(os.path.abspath(output_dir))
    parent_lexical = Path(os.path.abspath(output.parent))
    try:
        parent = output.parent.resolve(strict=True)
        parent.relative_to(output_root)
    except (OSError, ValueError) as error:
        raise FormalPilotOrchestrationError(
            f"formal stage OUTPUT_DIR parent must stay inside {output_root}"
        ) from error
    if parent != parent_lexical:
        raise FormalPilotOrchestrationError(
            "formal stage OUTPUT_DIR parent may not resolve through a symlink"
        )
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", output.name) is None:
        raise FormalPilotOrchestrationError(
            "formal stage OUTPUT_DIR must end in a safe path component"
        )
    if os.path.lexists(output):
        raise FormalPilotOrchestrationError(
            f"refusing to overwrite existing output directory: {output}"
        )
    if not files:
        raise FormalPilotOrchestrationError("cannot publish an empty formal stage")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".staging", dir=output.parent
        )
    )
    published = False
    try:
        seen: set[Path] = set()
        for relative_text, payload in sorted(files.items()):
            relative = Path(relative_text)
            if (
                relative.is_absolute()
                or not relative.parts
                or any(part in ("", ".", "..") for part in relative.parts)
            ):
                raise FormalPilotOrchestrationError(
                    f"invalid staged output path {relative_text!r}"
                )
            destination = staging / relative
            if destination in seen:
                raise FormalPilotOrchestrationError(
                    f"duplicate staged output path {relative_text!r}"
                )
            seen.add(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                with destination.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError as error:
                raise FormalPilotOrchestrationError(
                    f"cannot stage output {relative_text!r}: {error}"
                ) from error
            destination.chmod(0o444)
        directories = [candidate for candidate in staging.rglob("*") if candidate.is_dir()]
        for directory in sorted(
            directories, key=lambda candidate: len(candidate.parts), reverse=True
        ):
            directory.chmod(0o555)
            _fsync_directory(directory)
        staging.chmod(0o555)
        _fsync_directory(staging)
        _rename_directory_noreplace(staging, output)
        try:
            _fsync_directory(output.parent)
        except OSError as error:
            try:
                _rename_directory_noreplace(output, staging)
            except (FormalPilotOrchestrationError, OSError) as rollback_error:
                raise FormalPilotOrchestrationError(
                    "published output but parent fsync failed and atomic rollback "
                    f"also failed: {rollback_error}"
                ) from error
            try:
                _fsync_directory(output.parent)
            except OSError:
                pass
            raise FormalPilotOrchestrationError(
                "cannot fsync published output parent; publication was atomically "
                f"rolled back: {error}"
            ) from error
        published = True
    finally:
        if not published and staging.exists():
            _remove_tree(staging)
    return output


def _assert_output_absent(output_dir: str | os.PathLike[str]) -> None:
    output = Path(output_dir).resolve(strict=False)
    if os.path.lexists(output):
        raise FormalPilotOrchestrationError(
            f"refusing to overwrite existing output directory: {output}"
        )


def _rollback_newly_published_tree(path: Path, *, name: str) -> None:
    rollback = path.parent / f".{path.name}.rollback.{os.urandom(12).hex()}"
    try:
        _rename_directory_noreplace(path, rollback)
        _fsync_directory(path.parent)
        _remove_tree(rollback)
        _fsync_directory(path.parent)
    except (FormalPilotOrchestrationError, OSError) as error:
        raise FormalPilotOrchestrationError(
            f"{name} failed and its newly published tree could not be rolled back: {error}"
        ) from error


def _holdout_unlock_directory(calibration_dir: str | os.PathLike[str]) -> Path:
    calibration = Path(os.path.abspath(calibration_dir))
    return calibration.parent / f"{calibration.name}-holdout-unlock"


def _publish_holdout_unlock(
    calibration_root: Path, calibration_manifest: _Snapshot
) -> tuple[_Snapshot, datetime]:
    unlocked_at_text = _utc_now_text()
    payload = {
        "schema_version": HOLDOUT_UNLOCK_SCHEMA_VERSION,
        "stage": "holdout_unlock",
        "unlocked_at_utc": unlocked_at_text,
        "calibration_directory": str(calibration_root),
        "calibration_manifest": calibration_manifest.artifact(
            str(calibration_manifest.path)
        ),
    }
    payload_bytes = _json_bytes(payload)
    unlock_root = _publish_immutable_tree(
        _holdout_unlock_directory(calibration_root),
        {HOLDOUT_UNLOCK_FILENAME: payload_bytes},
    )
    snapshot = _read_snapshot(
        unlock_root / HOLDOUT_UNLOCK_FILENAME,
        name="immutable holdout unlock",
    )
    return snapshot, _aware_iso_datetime(
        unlocked_at_text, name="holdout unlock unlocked_at_utc", require_utc=True
    )


def _load_holdout_unlock(
    calibration_root: Path, calibration_manifest: _Snapshot
) -> tuple[_Snapshot, datetime]:
    unlock_root = _resolve_root(
        _holdout_unlock_directory(calibration_root), name="HOLDOUT_UNLOCK_DIR"
    )
    if stat.S_IMODE(os.lstat(unlock_root).st_mode) != 0o555:
        raise FormalPilotOrchestrationError(
            "holdout unlock directory must remain immutable 0555"
        )
    unlock_path = unlock_root / HOLDOUT_UNLOCK_FILENAME
    unlock_metadata = os.lstat(unlock_path)
    if (
        stat.S_ISLNK(unlock_metadata.st_mode)
        or not stat.S_ISREG(unlock_metadata.st_mode)
        or stat.S_IMODE(unlock_metadata.st_mode) != 0o444
    ):
        raise FormalPilotOrchestrationError(
            "holdout unlock artifact must remain a regular immutable 0444 file"
        )
    snapshot = _read_snapshot(
        unlock_path,
        name="immutable holdout unlock",
    )
    value = _strict_json_object(snapshot.payload, name=HOLDOUT_UNLOCK_FILENAME)
    _expect_exact_keys(
        value,
        {
            "schema_version",
            "stage",
            "unlocked_at_utc",
            "calibration_directory",
            "calibration_manifest",
        },
        name="holdout unlock",
    )
    if (
        value["schema_version"] != HOLDOUT_UNLOCK_SCHEMA_VERSION
        or value["stage"] != "holdout_unlock"
        or value["calibration_directory"] != str(calibration_root)
    ):
        raise FormalPilotOrchestrationError("holdout unlock schema/stage/path mismatch")
    declared = _validated_artifact(
        value["calibration_manifest"],
        name="holdout unlock calibration_manifest",
        expected_path=str(calibration_manifest.path),
    )
    _assert_artifact(
        calibration_manifest,
        declared,
        name="holdout unlock calibration manifest",
    )
    unlocked_at = _aware_iso_datetime(
        value["unlocked_at_utc"],
        name="holdout unlock unlocked_at_utc",
        require_utc=True,
    )
    return snapshot, unlocked_at


def _runs_by_split(bundle: _InputBundle, split: str) -> tuple[_InputRun, ...]:
    return tuple(run for run in bundle.runs if run.dataset_split == split)


def _run_ids(bundle: _InputBundle) -> dict[str, list[str]]:
    return {
        split: [run.run_id for run in _runs_by_split(bundle, split)]
        for split in SPLITS
    }


def _validate_cpu_validation_report_location(path: Path) -> Path:
    output_root = _resolve_root(OUTPUT_ROOT, name="StateGuard3R OUTPUT_ROOT")
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(output_root)
    except (OSError, ValueError) as error:
        raise FormalPilotOrchestrationError(
            "CPU validation report must stay inside StateGuard3R outputs"
        ) from error
    if (
        resolved.name != CPU_VALIDATION_REPORT_FILENAME
        or resolved.parent.name != CPU_VALIDATION_DIRECTORY_NAME
        or len(relative.parts) < 2
        or {part.lower() for part in relative.parts}
        & {"runs", "development", "holdout"}
    ):
        raise FormalPilotOrchestrationError(
            "CPU validation report must be named "
            f"{CPU_VALIDATION_DIRECTORY_NAME}/{CPU_VALIDATION_REPORT_FILENAME}"
        )
    return resolved


def _create_cpu_validation_staging(
    output_dir: str | os.PathLike[str],
) -> tuple[Path, Path]:
    output_root = _resolve_root(OUTPUT_ROOT, name="StateGuard3R OUTPUT_ROOT")
    output = Path(os.path.abspath(output_dir))
    parent_lexical = Path(os.path.abspath(output.parent))
    try:
        parent = output.parent.resolve(strict=True)
        parent.relative_to(output_root)
    except (OSError, ValueError) as error:
        raise FormalPilotOrchestrationError(
            "commit OUTPUT_DIR parent must be a real directory inside "
            f"{output_root}"
        ) from error
    if parent != parent_lexical:
        raise FormalPilotOrchestrationError(
            "commit OUTPUT_DIR parent may not resolve through a symlink"
        )
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", output.name) is None:
        raise FormalPilotOrchestrationError(
            "commit OUTPUT_DIR must end in a safe path component"
        )
    try:
        staging = Path(
            tempfile.mkdtemp(
                dir=parent,
                prefix=f".{output.name}.cpu-validation.",
            )
        )
        staging.chmod(0o700)
        validation_root = staging / CPU_VALIDATION_DIRECTORY_NAME
        validation_root.mkdir(mode=0o700)
        _fsync_directory(staging)
        _fsync_directory(parent)
    except OSError as error:
        raise FormalPilotOrchestrationError(
            f"cannot create private CPU validation staging directory: {error}"
        ) from error
    return staging, validation_root / CPU_VALIDATION_REPORT_FILENAME


def _invoke_fixed_cpu_validator(input_root: Path, report_path: Path) -> _Snapshot:
    """Run the one tracked validator and bind its process attestation immediately."""

    if os.path.lexists(report_path):
        raise FormalPilotOrchestrationError(
            "refusing to invoke the CPU validator over a pre-existing report"
        )
    validator_path = REPOSITORY_ROOT / CPU_VALIDATION_GENERATOR_RELATIVE_PATH
    _preflight_regular_file(validator_path, name="fixed CPU validation generator")
    try:
        interpreter_realpath = RECAL3R_PYTHON.resolve(strict=True)
    except OSError as error:
        raise FormalPilotOrchestrationError(
            f"cannot resolve the fixed ReCal3R interpreter: {error}"
        ) from error
    if not interpreter_realpath.is_file() or not os.access(RECAL3R_PYTHON, os.X_OK):
        raise FormalPilotOrchestrationError(
            f"fixed ReCal3R interpreter is not executable: {RECAL3R_PYTHON}"
        )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    command = [
        str(RECAL3R_PYTHON),
        str(validator_path),
        str(input_root),
        str(report_path),
        "--baseline-root",
        str(RECAL3R_ROOT),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise FormalPilotOrchestrationError(
            f"cannot execute the fixed CPU validator: {error}"
        ) from error
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")[-4000:]
        raise FormalPilotOrchestrationError(
            "fixed CPU validator failed before commitment"
            + (f": {stderr.strip()}" if stderr.strip() else "")
        )
    attestation = _strict_json_object(
        completed.stdout, name="fixed CPU validator process attestation"
    )
    _expect_exact_keys(
        attestation,
        {"status", "output_report", "sha256", "size_bytes"},
        name="fixed CPU validator process attestation",
    )
    if attestation["status"] != "PASS" or attestation["output_report"] != str(
        report_path
    ):
        raise FormalPilotOrchestrationError(
            "fixed CPU validator process attestation status/path mismatch"
        )
    snapshot = _read_snapshot(report_path, name="fresh CPU validation report")
    _validate_cpu_validation_report_location(snapshot.path)
    expected_attestation = snapshot.artifact(str(report_path))
    if {
        "path": attestation["output_report"],
        "sha256": attestation["sha256"],
        "size_bytes": attestation["size_bytes"],
    } != expected_attestation:
        raise FormalPilotOrchestrationError(
            "fresh CPU validation report differs from validator process attestation"
        )
    return snapshot


def _input_tree_snapshot(input_root: Path) -> tuple[str, int]:
    """Recompute the validator's canonical 24-entry input-tree identity."""

    root = _resolve_root(input_root, name="formal input tree root")
    records: dict[str, dict[str, Any]] = {}
    inode_owners: dict[tuple[int, int], str] = {}

    def visit(path: Path, relative: str) -> None:
        try:
            before = os.lstat(path)
        except OSError as error:
            raise FormalPilotOrchestrationError(
                f"cannot inspect formal input tree entry {path}: {error}"
            ) from error
        if stat.S_ISLNK(before.st_mode):
            raise FormalPilotOrchestrationError(
                f"formal input tree may not contain symlinks: {path}"
            )
        identity = (before.st_dev, before.st_ino)
        previous = inode_owners.get(identity)
        if previous is not None:
            raise FormalPilotOrchestrationError(
                "formal input tree paths alias one inode: "
                f"{previous!r} and {relative!r}"
            )
        inode_owners[identity] = relative
        if stat.S_ISREG(before.st_mode):
            snapshot = _read_snapshot(path, name=f"formal input tree file {relative}")
            try:
                after = os.lstat(path)
            except OSError as error:
                raise FormalPilotOrchestrationError(
                    f"formal input tree file disappeared: {path}: {error}"
                ) from error
            if (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise FormalPilotOrchestrationError(
                    f"formal input tree file changed while snapshotted: {path}"
                )
            records[relative] = {
                "relative_path": relative,
                "kind": "file",
                "device": before.st_dev,
                "inode": before.st_ino,
                "mode_octal": f"{stat.S_IMODE(before.st_mode):04o}",
                "link_count": before.st_nlink,
                "size_bytes": before.st_size,
                "mtime_ns": before.st_mtime_ns,
                "sha256": hashlib.sha256(snapshot.payload).hexdigest(),
            }
            return
        if not stat.S_ISDIR(before.st_mode):
            raise FormalPilotOrchestrationError(
                f"formal input tree contains a special entry: {path}"
            )
        try:
            children = sorted(os.scandir(path), key=lambda entry: entry.name)
        except OSError as error:
            raise FormalPilotOrchestrationError(
                f"cannot enumerate formal input tree directory {path}: {error}"
            ) from error
        for child in children:
            child_relative = child.name if relative == "." else f"{relative}/{child.name}"
            visit(path / child.name, child_relative)
        try:
            after = os.lstat(path)
        except OSError as error:
            raise FormalPilotOrchestrationError(
                f"formal input tree directory disappeared: {path}: {error}"
            ) from error
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise FormalPilotOrchestrationError(
                f"formal input tree directory changed while snapshotted: {path}"
            )
        records[relative] = {
            "relative_path": relative,
            "kind": "directory",
            "device": before.st_dev,
            "inode": before.st_ino,
            "mode_octal": f"{stat.S_IMODE(before.st_mode):04o}",
            "link_count": before.st_nlink,
            "size_bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns,
            "sha256": None,
        }

    visit(root, ".")
    expected_layout = {
        ".",
        INPUT_REGISTRY_FILENAME,
        SPLIT_REGISTRY_FILENAME,
        HOLDOUT_COMMITMENT_FILENAME,
        "development",
        "holdout",
    }
    for split in SPLITS:
        for run_id, _ in FROZEN_RUN_LAYOUT[split]:
            run_relative = f"{split}/{run_id}"
            expected_layout.update(
                {
                    run_relative,
                    f"{run_relative}/{SOURCE_MANIFEST_FILENAME}",
                    f"{run_relative}/{INPUT_MANIFEST_FILENAME}",
                }
            )
    if set(records) != expected_layout:
        raise FormalPilotOrchestrationError(
            "formal input tree differs from the exact frozen 24-entry layout"
        )
    payload = [records[relative] for relative in sorted(records)]
    return hashlib.sha256(_json_bytes(payload)).hexdigest(), len(records)


def _validate_cpu_validation_report(
    snapshot: _Snapshot,
    input_bundle: _InputBundle,
    runner_provenance: Mapping[str, Any],
) -> Mapping[str, Any]:
    _validate_cpu_validation_report_location(snapshot.path)
    value = _strict_json_object(snapshot.payload, name="CPU validation report")
    _expect_exact_keys(
        value,
        {
            "schema_version",
            "status",
            "input_registry",
            "run_ids",
            "environment",
            "source_snapshots_before_sha256",
            "source_snapshots_after_sha256",
            "source_snapshot_artifact_count",
            "input_tree_snapshots_before_sha256",
            "input_tree_snapshots_after_sha256",
            "input_tree_snapshot_entry_count",
            "runs",
            "generator",
            "production_runtime",
            "tum_associations",
            "checks",
        },
        name="CPU validation report",
    )
    if value["schema_version"] != CPU_VALIDATION_SCHEMA_VERSION:
        raise FormalPilotOrchestrationError("CPU validation report schema mismatch")
    if value["status"] != "PASS":
        raise FormalPilotOrchestrationError("CPU validation report status must be PASS")
    registry = _validated_artifact(
        value["input_registry"],
        name="CPU validation report input_registry",
        expected_path=INPUT_REGISTRY_FILENAME,
    )
    _assert_artifact(
        input_bundle.registry, registry, name="CPU validation report input registry"
    )
    if value["run_ids"] != _run_ids(input_bundle):
        raise FormalPilotOrchestrationError("CPU validation report run IDs mismatch")
    environment = value["environment"]
    if not isinstance(environment, Mapping):
        raise FormalPilotOrchestrationError(
            "CPU validation report environment must be an object"
        )
    _expect_exact_keys(
        environment,
        {
            "CUDA_VISIBLE_DEVICES",
            "torch_cuda_is_initialized_before",
            "torch_cuda_is_initialized_after",
        },
        name="CPU validation report environment",
    )
    if environment["CUDA_VISIBLE_DEVICES"] != "" or any(
        environment[field] is not False
        for field in (
            "torch_cuda_is_initialized_before",
            "torch_cuda_is_initialized_after",
        )
    ):
        raise FormalPilotOrchestrationError(
            "CPU validation must hide CUDA and leave torch CUDA uninitialized"
        )
    before = _sha256(
        value["source_snapshots_before_sha256"],
        name="CPU validation source_snapshots_before_sha256",
    )
    after = _sha256(
        value["source_snapshots_after_sha256"],
        name="CPU validation source_snapshots_after_sha256",
    )
    if before != after:
        raise FormalPilotOrchestrationError(
            "CPU validation source snapshots changed during replay"
        )
    if value["source_snapshot_artifact_count"] != (
        EXPECTED_SOURCE_SNAPSHOT_ARTIFACT_COUNT
    ):
        raise FormalPilotOrchestrationError(
            "CPU validation report must prove exactly "
            f"{EXPECTED_SOURCE_SNAPSHOT_ARTIFACT_COUNT} raw source artifacts"
        )
    input_tree_before = _sha256(
        value["input_tree_snapshots_before_sha256"],
        name="CPU validation input_tree_snapshots_before_sha256",
    )
    input_tree_after = _sha256(
        value["input_tree_snapshots_after_sha256"],
        name="CPU validation input_tree_snapshots_after_sha256",
    )
    if input_tree_before != input_tree_after:
        raise FormalPilotOrchestrationError(
            "CPU validation input tree changed during replay"
        )
    if value["input_tree_snapshot_entry_count"] != (
        EXPECTED_INPUT_TREE_SNAPSHOT_ENTRY_COUNT
    ):
        raise FormalPilotOrchestrationError(
            "CPU validation report must prove the exact 24-entry formal input tree"
        )
    current_input_tree_digest, current_input_tree_count = _input_tree_snapshot(
        input_bundle.root
    )
    if (
        current_input_tree_count != EXPECTED_INPUT_TREE_SNAPSHOT_ENTRY_COUNT
        or current_input_tree_digest != input_tree_after
    ):
        raise FormalPilotOrchestrationError(
            "current formal input tree differs from the CPU-validated tree identity"
        )
    run_rows = value["runs"]
    if not isinstance(run_rows, list) or len(run_rows) != 6:
        raise FormalPilotOrchestrationError(
            "CPU validation report must contain exactly six run summaries"
        )
    expected_runs = list(input_bundle.runs)
    for index, (row, expected_run) in enumerate(zip(run_rows, expected_runs, strict=True)):
        if not isinstance(row, Mapping):
            raise FormalPilotOrchestrationError(
                f"CPU validation runs[{index}] must be an object"
            )
        _expect_exact_keys(
            row,
            {
                "run_id",
                "frame_count",
                "manifest_validation_passed",
                "pixel_replay_passed",
                "transform_digest_sha256",
                "pixel_digest_sha256",
            },
            name=f"CPU validation runs[{index}]",
        )
        if row["run_id"] != expected_run.run_id or row["frame_count"] != 30:
            raise FormalPilotOrchestrationError(
                f"CPU validation runs[{index}] layout/frame count mismatch"
            )
        if (
            row["manifest_validation_passed"] is not True
            or row["pixel_replay_passed"] is not True
        ):
            raise FormalPilotOrchestrationError(
                f"CPU validation run {expected_run.run_id} did not pass replay"
            )
        _sha256(
            row["transform_digest_sha256"],
            name=f"CPU validation runs[{index}].transform_digest_sha256",
        )
        _sha256(
            row["pixel_digest_sha256"],
            name=f"CPU validation runs[{index}].pixel_digest_sha256",
        )
    generator = value["generator"]
    if not isinstance(generator, Mapping):
        raise FormalPilotOrchestrationError(
            "CPU validation report generator must be an object"
        )
    _expect_exact_keys(
        generator,
        {
            "repository_commit",
            "tracked_worktree_clean",
            "script",
            "git_blob_sha256",
        },
        name="CPU validation report generator",
    )
    expected_commit = runner_provenance["state_guard_repository"]["commit"]
    if (
        generator["repository_commit"] != expected_commit
        or generator["tracked_worktree_clean"] is not True
    ):
        raise FormalPilotOrchestrationError(
            "CPU validation generator commit/clean state mismatch"
        )
    script = _validated_artifact(
        generator["script"],
        name="CPU validation report generator.script",
        expected_path=CPU_VALIDATION_GENERATOR_RELATIVE_PATH.as_posix(),
    )
    generator_snapshot = _read_snapshot(
        REPOSITORY_ROOT / CPU_VALIDATION_GENERATOR_RELATIVE_PATH,
        name="fixed CPU validation generator",
    )
    _assert_artifact(
        generator_snapshot, script, name="CPU validation generator script"
    )
    git_blob_sha256 = _sha256(
        generator["git_blob_sha256"],
        name="CPU validation report generator.git_blob_sha256",
    )
    current_head_sha256 = _tracked_file_sha256_at_head(
        REPOSITORY_ROOT, CPU_VALIDATION_GENERATOR_RELATIVE_PATH
    )
    if not (
        git_blob_sha256
        == current_head_sha256
        == generator_snapshot.artifact(
            CPU_VALIDATION_GENERATOR_RELATIVE_PATH.as_posix()
        )["sha256"]
    ):
        raise FormalPilotOrchestrationError(
            "CPU validation generator bytes do not equal the tracked HEAD blob"
        )

    production_runtime = value["production_runtime"]
    if not isinstance(production_runtime, Mapping):
        raise FormalPilotOrchestrationError(
            "CPU validation production_runtime must be an object"
        )
    _expect_exact_keys(
        production_runtime,
        {"recal3r_repository", "python", "official_loader"},
        name="CPU validation production_runtime",
    )
    runtime_repository = production_runtime["recal3r_repository"]
    if not isinstance(runtime_repository, Mapping):
        raise FormalPilotOrchestrationError(
            "CPU validation ReCal3R runtime provenance must be an object"
        )
    _expect_exact_keys(
        runtime_repository,
        {"path", "commit", "tracked_worktree_clean"},
        name="CPU validation production_runtime.recal3r_repository",
    )
    expected_recal3r_root = str(RECAL3R_ROOT.resolve(strict=True))
    if dict(runtime_repository) != {
        "path": expected_recal3r_root,
        "commit": EXPECTED_RECAL3R_COMMIT,
        "tracked_worktree_clean": True,
    } or dict(runtime_repository) != dict(
        runner_provenance["recal3r_repository"]
    ):
        raise FormalPilotOrchestrationError(
            "CPU validation runtime did not use the frozen clean ReCal3R revision"
        )
    runtime_python = production_runtime["python"]
    if not isinstance(runtime_python, Mapping):
        raise FormalPilotOrchestrationError(
            "CPU validation Python runtime provenance must be an object"
        )
    _expect_exact_keys(
        runtime_python,
        {"executable", "executable_realpath", "prefix"},
        name="CPU validation production_runtime.python",
    )
    try:
        expected_python_realpath = str(RECAL3R_PYTHON.resolve(strict=True))
        expected_python_prefix = str((RECAL3R_ROOT / ".venv").resolve(strict=True))
    except OSError as error:
        raise FormalPilotOrchestrationError(
            f"cannot resolve frozen ReCal3R Python runtime: {error}"
        ) from error
    if dict(runtime_python) != {
        "executable": str(RECAL3R_PYTHON),
        "executable_realpath": expected_python_realpath,
        "prefix": expected_python_prefix,
    }:
        raise FormalPilotOrchestrationError(
            "CPU validation report Python executable/prefix mismatch"
        )
    official_loader = _validated_artifact(
        production_runtime["official_loader"],
        name="CPU validation production_runtime.official_loader",
        expected_path=OFFICIAL_LOADER_RELATIVE_PATH.as_posix(),
    )
    loader_snapshot = _read_snapshot(
        RECAL3R_ROOT / OFFICIAL_LOADER_RELATIVE_PATH,
        name="frozen official ReCal3R image loader",
    )
    _assert_artifact(
        loader_snapshot, official_loader, name="official ReCal3R image loader"
    )
    if official_loader["sha256"] != EXPECTED_OFFICIAL_LOADER_SHA256:
        raise FormalPilotOrchestrationError(
            "official ReCal3R image loader SHA differs from the frozen revision"
        )
    tum_associations = value["tum_associations"]
    if not isinstance(tum_associations, Mapping):
        raise FormalPilotOrchestrationError(
            "CPU validation TUM association evidence must be an object"
        )
    _expect_exact_keys(
        tum_associations,
        {
            "method",
            "tie_policy",
            "max_absolute_delta_seconds",
            "allocated_rgb_count",
            "globally_unique_depth_count",
            "globally_unique_groundtruth_count",
            "association_digest_sha256",
        },
        name="CPU validation tum_associations",
    )
    expected_associations = {
        "method": "unique_nearest_absolute_timestamp",
        "tie_policy": "reject",
        "max_absolute_delta_seconds": 0.02,
        "allocated_rgb_count": 190,
        "globally_unique_depth_count": 190,
        "globally_unique_groundtruth_count": 190,
    }
    if {
        key: tum_associations[key] for key in expected_associations
    } != expected_associations:
        raise FormalPilotOrchestrationError(
            "CPU validation TUM association policy/counts are not frozen v1"
        )
    _sha256(
        tum_associations["association_digest_sha256"],
        name="CPU validation tum_associations.association_digest_sha256",
    )
    checks = value["checks"]
    if not isinstance(checks, Mapping) or set(checks) != CPU_VALIDATION_CHECKS:
        raise FormalPilotOrchestrationError(
            "CPU validation report must contain the exact frozen check set"
        )
    if any(checks[name] is not True for name in CPU_VALIDATION_CHECKS):
        raise FormalPilotOrchestrationError("every CPU validation check must PASS")
    return value


def commit_inputs(
    input_root: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    protocol: str | os.PathLike[str],
) -> dict[str, Any]:
    """Run CPU validation, then atomically freeze all pre-forward evidence."""

    _assert_output_absent(output_dir)
    inputs = _load_input_bundle(input_root)
    protocol_snapshot = _snapshot_frozen_protocol(protocol)
    protocol_path = protocol_snapshot.path.resolve(strict=True)
    runner_provenance = _capture_runner_provenance()
    validation_staging, validation_path = _create_cpu_validation_staging(output_dir)
    try:
        staged_validation = _invoke_fixed_cpu_validator(inputs.root, validation_path)
        runner_after_validation = _capture_runner_provenance()
        if runner_after_validation != runner_provenance:
            raise FormalPilotOrchestrationError(
                "repository/runner provenance changed while CPU validation ran"
            )
        _validate_cpu_validation_report(
            staged_validation, inputs, runner_provenance
        )
        development = [
            detection_suite.make_run_commitment(run.suite_input())
            for run in _runs_by_split(inputs, "development")
        ]
        holdout = [
            detection_suite.make_run_commitment(run.suite_input())
            for run in _runs_by_split(inputs, "holdout")
        ]
        split_registry = detection_suite.make_split_registry(development, holdout)
        holdout_commitment = detection_suite.make_holdout_commitment(holdout)
        if split_registry != inputs.builder_split_registry.payload:
            raise FormalPilotOrchestrationError(
                "recomputed split registry differs from input-builder artifact"
            )
        if holdout_commitment != inputs.builder_holdout_commitment.payload:
            raise FormalPilotOrchestrationError(
                "recomputed holdout commitment differs from input-builder artifact"
            )
        split_snapshot = _Snapshot(Path(SPLIT_REGISTRY_FILENAME), split_registry)
        holdout_snapshot = _Snapshot(
            Path(HOLDOUT_COMMITMENT_FILENAME), holdout_commitment
        )
        output = Path(os.path.abspath(output_dir))
        validation_relative = (
            Path(CPU_VALIDATION_DIRECTORY_NAME) / CPU_VALIDATION_REPORT_FILENAME
        )
        final_validation_path = output / validation_relative
        final_validation = _Snapshot(
            path=final_validation_path,
            payload=staged_validation.payload,
        )
        execution_not_before_utc = _utc_now_text()
        manifest = {
            "schema_version": COMMITMENT_SCHEMA_VERSION,
            "stage": "commit",
            "execution_not_before_utc": execution_not_before_utc,
            "input_registry": inputs.registry.artifact(INPUT_REGISTRY_FILENAME),
            "protocol": protocol_snapshot.artifact(str(protocol_path)),
            "validation_report": final_validation.artifact(
                str(final_validation_path)
            ),
            "runner_provenance": runner_provenance,
            "artifacts": {
                "split_registry": split_snapshot.artifact(SPLIT_REGISTRY_FILENAME),
                "holdout_commitment": holdout_snapshot.artifact(
                    HOLDOUT_COMMITMENT_FILENAME
                ),
            },
            "run_ids": _run_ids(inputs),
        }
        manifest_bytes = _json_bytes(manifest)
        published_output = _publish_immutable_tree(
            output,
            {
                SPLIT_REGISTRY_FILENAME: split_registry,
                HOLDOUT_COMMITMENT_FILENAME: holdout_commitment,
                validation_relative.as_posix(): staged_validation.payload,
                COMMITMENT_MANIFEST_FILENAME: manifest_bytes,
            },
        )
    finally:
        _remove_tree(validation_staging)
    return {
        "stage": "commit",
        "output_dir": str(published_output),
        "commitment_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }


def _load_commit_bundle(
    commit_dir: str | os.PathLike[str], input_bundle: _InputBundle
) -> _CommitBundle:
    root = _resolve_root(commit_dir, name="COMMIT_DIR")
    manifest_snapshot = _read_snapshot(
        root / COMMITMENT_MANIFEST_FILENAME, name="commitment manifest"
    )
    value = _strict_json_object(
        manifest_snapshot.payload, name=COMMITMENT_MANIFEST_FILENAME
    )
    _expect_exact_keys(
        value,
        {
            "schema_version",
            "stage",
            "execution_not_before_utc",
            "input_registry",
            "protocol",
            "validation_report",
            "runner_provenance",
            "artifacts",
            "run_ids",
        },
        name="commitment manifest",
    )
    if value["schema_version"] != COMMITMENT_SCHEMA_VERSION or value["stage"] != "commit":
        raise FormalPilotOrchestrationError("commitment manifest schema/stage mismatch")
    execution_unlocked_at = _aware_iso_datetime(
        value["execution_not_before_utc"],
        name="commitment manifest execution_not_before_utc",
        require_utc=True,
    )
    registry_declared = _validated_artifact(
        value["input_registry"],
        name="commitment manifest input_registry",
        expected_path=INPUT_REGISTRY_FILENAME,
    )
    _assert_artifact(
        input_bundle.registry,
        registry_declared,
        name="commitment manifest input registry",
    )
    if value["run_ids"] != _run_ids(input_bundle):
        raise FormalPilotOrchestrationError("commitment manifest run IDs mismatch")
    protocol_declared = _validated_artifact(
        value["protocol"], name="commitment manifest protocol"
    )
    protocol_snapshot = _read_snapshot(
        Path(protocol_declared["path"]), name="bound formal protocol"
    )
    _assert_artifact(
        protocol_snapshot, protocol_declared, name="bound formal protocol"
    )
    expected_validation_path = (
        root
        / CPU_VALIDATION_DIRECTORY_NAME
        / CPU_VALIDATION_REPORT_FILENAME
    )
    validation_declared = _validated_artifact(
        value["validation_report"],
        name="commitment manifest validation_report",
        expected_path=str(expected_validation_path),
    )
    validation_snapshot = _read_snapshot(
        expected_validation_path, name="bound CPU validation report"
    )
    _assert_artifact(
        validation_snapshot,
        validation_declared,
        name="bound CPU validation report",
    )
    runner_provenance = _verify_current_runner_provenance(value["runner_provenance"])
    _validate_cpu_validation_report(
        validation_snapshot, input_bundle, runner_provenance
    )
    artifacts = value["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise FormalPilotOrchestrationError("commitment manifest artifacts must be an object")
    _expect_exact_keys(
        artifacts,
        {"split_registry", "holdout_commitment"},
        name="commitment manifest artifacts",
    )
    split_declared = _validated_artifact(
        artifacts["split_registry"],
        name="commitment manifest split_registry",
        expected_path=SPLIT_REGISTRY_FILENAME,
    )
    holdout_declared = _validated_artifact(
        artifacts["holdout_commitment"],
        name="commitment manifest holdout_commitment",
        expected_path=HOLDOUT_COMMITMENT_FILENAME,
    )
    split_snapshot = _read_snapshot(root / SPLIT_REGISTRY_FILENAME, name="split registry")
    holdout_snapshot = _read_snapshot(
        root / HOLDOUT_COMMITMENT_FILENAME, name="holdout commitment"
    )
    _assert_artifact(split_snapshot, split_declared, name="split registry")
    _assert_artifact(holdout_snapshot, holdout_declared, name="holdout commitment")
    _strict_json_object(split_snapshot.payload, name="split registry")
    _strict_json_object(holdout_snapshot.payload, name="holdout commitment")
    if split_snapshot.payload != input_bundle.builder_split_registry.payload:
        raise FormalPilotOrchestrationError(
            "committed split registry differs from bound input-builder artifact"
        )
    if holdout_snapshot.payload != input_bundle.builder_holdout_commitment.payload:
        raise FormalPilotOrchestrationError(
            "committed holdout commitment differs from bound input-builder artifact"
        )
    return _CommitBundle(
        root=root,
        manifest=manifest_snapshot,
        manifest_value=value,
        split_registry=split_snapshot,
        holdout_commitment=holdout_snapshot,
        protocol=protocol_snapshot,
        validation_report=validation_snapshot,
        runner_provenance=runner_provenance,
        execution_unlocked_at=execution_unlocked_at,
    )


def _snapshot_run_outputs(
    runs_root: Path, runs: Sequence[_InputRun]
) -> dict[str, dict[str, _Snapshot]]:
    snapshots: dict[str, dict[str, _Snapshot]] = {}
    for run in runs:
        run_root = _validated_child_directory(
            runs_root, (run.run_id,), name=f"{run.run_id} output run"
        )
        directory_metadata = os.lstat(run_root)
        if stat.S_IMODE(directory_metadata.st_mode) != 0o555:
            raise FormalPilotOrchestrationError(
                f"{run.run_id} output directory must be frozen at mode 0555"
            )
        try:
            children = {entry.name: entry for entry in os.scandir(run_root)}
        except OSError as error:
            raise FormalPilotOrchestrationError(
                f"cannot enumerate {run.run_id} frozen outputs: {error}"
            ) from error
        expected_filenames = set(RUN_OUTPUT_ARTIFACT_FILENAMES.values())
        if set(children) != expected_filenames:
            raise FormalPilotOrchestrationError(
                f"{run.run_id} output directory must contain exactly the five "
                "frozen formal artifacts"
            )
        run_snapshots: dict[str, _Snapshot] = {}
        for artifact_name, filename in RUN_OUTPUT_ARTIFACT_FILENAMES.items():
            path = run_root / filename
            metadata = os.lstat(path)
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o444
            ):
                raise FormalPilotOrchestrationError(
                    f"{run.run_id} {filename} must be a regular non-symlink "
                    "file frozen at mode 0444"
                )
            run_snapshots[artifact_name] = _read_snapshot(
                path, name=f"{run.run_id} {filename}"
            )
        snapshots[run.run_id] = run_snapshots
    return snapshots


def _run_output_artifacts(
    outputs: Mapping[str, Mapping[str, _Snapshot]],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        run_id: {
            artifact_name: snapshots[artifact_name].artifact(
                f"{run_id}/{filename}"
            )
            for artifact_name, filename in RUN_OUTPUT_ARTIFACT_FILENAMES.items()
        }
        for run_id, snapshots in outputs.items()
    }


def calibrate_development(
    input_root: str | os.PathLike[str],
    commit_dir: str | os.PathLike[str],
    runs_root: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Search and freeze development only; no holdout output path is probed."""

    _assert_output_absent(output_dir)
    _assert_output_absent(_holdout_unlock_directory(output_dir))
    inputs = _load_input_bundle(input_root)
    committed = _load_commit_bundle(commit_dir, inputs)
    runs_directory = _resolve_root(runs_root, name="RUNS_ROOT")
    development_inputs = _runs_by_split(inputs, "development")
    # This helper receives development inputs only.  It therefore cannot even
    # construct, list, or stat a holdout output path.
    development_outputs = _snapshot_run_outputs(runs_directory, development_inputs)
    for run in development_inputs:
        _validate_run_json_runner(
            development_outputs[run.run_id]["run_json"],
            committed.runner_provenance,
            run_id=run.run_id,
        )
    _validated_split_run_timeline(
        development_inputs,
        development_outputs,
        split="development",
        earliest_start=committed.execution_unlocked_at,
        latest_finish=_utc_now(),
    )
    development_runs = [
        run.suite_run(
            development_outputs[run.run_id]["health_jsonl"].payload,
            development_outputs[run.run_id]["run_json"].payload,
        )
        for run in development_inputs
    ]
    search = detection_suite.search_development_suite(
        development_runs,
        detection_suite.fixed_candidate_order(),
        master_seed=0,
    )
    development_metrics = detection_suite.evaluate_detection_suite(
        development_runs,
        search["frozen_detection_config"],
        expected_split="development",
    )
    formal_config = detection_suite.build_formal_v1_config(
        development_runs,
        search,
        development_run_id="FORMAL-DEV-SUITE-0001",
        split_registry=committed.split_registry.payload,
        holdout_commitment=committed.holdout_commitment.payload,
    )
    search_bytes = _json_bytes(search)
    development_metrics_bytes = _json_bytes(development_metrics)
    formal_config_bytes = _json_bytes(formal_config)
    output_snapshots = {
        "search": _Snapshot(Path(SEARCH_FILENAME), search_bytes),
        "development_metrics": _Snapshot(
            Path(DEVELOPMENT_METRICS_FILENAME), development_metrics_bytes
        ),
        "formal_config": _Snapshot(Path(FORMAL_CONFIG_FILENAME), formal_config_bytes),
    }
    manifest = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "stage": "calibrate",
        "inputs": {
            "input_registry": inputs.registry.artifact(INPUT_REGISTRY_FILENAME),
            "commitment_manifest": committed.manifest.artifact(
                COMMITMENT_MANIFEST_FILENAME
            ),
            "split_registry": committed.split_registry.artifact(
                SPLIT_REGISTRY_FILENAME
            ),
            "holdout_commitment": committed.holdout_commitment.artifact(
                HOLDOUT_COMMITMENT_FILENAME
            ),
            "protocol": committed.protocol.artifact(
                str(committed.protocol.path.resolve(strict=True))
            ),
            "validation_report": committed.validation_report.artifact(
                str(committed.validation_report.path.resolve(strict=True))
            ),
        },
        "development_run_artifacts": _run_output_artifacts(development_outputs),
        "artifacts": {
            "search": output_snapshots["search"].artifact(SEARCH_FILENAME),
            "development_metrics": output_snapshots[
                "development_metrics"
            ].artifact(DEVELOPMENT_METRICS_FILENAME),
            "formal_config": output_snapshots["formal_config"].artifact(
                FORMAL_CONFIG_FILENAME
            ),
        },
        "run_ids": _run_ids(inputs),
    }
    manifest_bytes = _json_bytes(manifest)
    output = _publish_immutable_tree(
        output_dir,
        {
            SEARCH_FILENAME: search_bytes,
            DEVELOPMENT_METRICS_FILENAME: development_metrics_bytes,
            FORMAL_CONFIG_FILENAME: formal_config_bytes,
            CALIBRATION_MANIFEST_FILENAME: manifest_bytes,
        },
    )
    published_manifest = _read_snapshot(
        output / CALIBRATION_MANIFEST_FILENAME,
        name="published calibration manifest",
    )
    try:
        holdout_unlock, _ = _publish_holdout_unlock(output, published_manifest)
    except (FormalPilotOrchestrationError, OSError):
        _rollback_newly_published_tree(output, name="holdout unlock publication")
        raise
    return {
        "stage": "calibrate",
        "output_dir": str(output),
        "holdout_unlock": str(holdout_unlock.path),
        "calibration_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }


def _load_calibration_bundle(
    calibration_dir: str | os.PathLike[str],
    input_bundle: _InputBundle,
    commit_bundle: _CommitBundle,
    runs_root: Path,
) -> _CalibrationBundle:
    root = _resolve_root(calibration_dir, name="CALIB_DIR")
    manifest_snapshot = _read_snapshot(
        root / CALIBRATION_MANIFEST_FILENAME, name="calibration manifest"
    )
    value = _strict_json_object(
        manifest_snapshot.payload, name=CALIBRATION_MANIFEST_FILENAME
    )
    _expect_exact_keys(
        value,
        {
            "schema_version",
            "stage",
            "inputs",
            "development_run_artifacts",
            "artifacts",
            "run_ids",
        },
        name="calibration manifest",
    )
    if (
        value["schema_version"] != CALIBRATION_SCHEMA_VERSION
        or value["stage"] != "calibrate"
    ):
        raise FormalPilotOrchestrationError("calibration manifest schema/stage mismatch")
    holdout_unlock, holdout_unlocked_at = _load_holdout_unlock(
        root, manifest_snapshot
    )
    if value["run_ids"] != _run_ids(input_bundle):
        raise FormalPilotOrchestrationError("calibration manifest run IDs mismatch")
    raw_inputs = value["inputs"]
    if not isinstance(raw_inputs, Mapping):
        raise FormalPilotOrchestrationError("calibration manifest inputs must be an object")
    expected_input_paths = {
        "input_registry": INPUT_REGISTRY_FILENAME,
        "commitment_manifest": COMMITMENT_MANIFEST_FILENAME,
        "split_registry": SPLIT_REGISTRY_FILENAME,
        "holdout_commitment": HOLDOUT_COMMITMENT_FILENAME,
        "protocol": str(commit_bundle.protocol.path.resolve(strict=True)),
        "validation_report": str(
            commit_bundle.validation_report.path.resolve(strict=True)
        ),
    }
    _expect_exact_keys(raw_inputs, set(expected_input_paths), name="calibration inputs")
    actual_input_snapshots = {
        "input_registry": input_bundle.registry,
        "commitment_manifest": commit_bundle.manifest,
        "split_registry": commit_bundle.split_registry,
        "holdout_commitment": commit_bundle.holdout_commitment,
        "protocol": commit_bundle.protocol,
        "validation_report": commit_bundle.validation_report,
    }
    for artifact_name, expected_path in expected_input_paths.items():
        declared = _validated_artifact(
            raw_inputs[artifact_name],
            name=f"calibration inputs.{artifact_name}",
            expected_path=expected_path,
        )
        _assert_artifact(
            actual_input_snapshots[artifact_name],
            declared,
            name=f"calibration input {artifact_name}",
        )

    raw_run_artifacts = value["development_run_artifacts"]
    development_inputs = _runs_by_split(input_bundle, "development")
    expected_run_ids = {run.run_id for run in development_inputs}
    if not isinstance(raw_run_artifacts, Mapping) or set(raw_run_artifacts) != expected_run_ids:
        raise FormalPilotOrchestrationError(
            "calibration manifest must bind exactly three development run outputs"
        )
    development_outputs = _snapshot_run_outputs(runs_root, development_inputs)
    for run in development_inputs:
        declarations = raw_run_artifacts[run.run_id]
        if not isinstance(declarations, Mapping):
            raise FormalPilotOrchestrationError(
                f"calibration development artifacts for {run.run_id} must be an object"
            )
        _expect_exact_keys(
            declarations,
            set(RUN_OUTPUT_ARTIFACT_FILENAMES),
            name=f"calibration development artifacts.{run.run_id}",
        )
        for artifact_name, filename in RUN_OUTPUT_ARTIFACT_FILENAMES.items():
            declared = _validated_artifact(
                declarations[artifact_name],
                name=f"calibration development artifacts.{run.run_id}.{artifact_name}",
                expected_path=f"{run.run_id}/{filename}",
            )
            _assert_artifact(
                development_outputs[run.run_id][artifact_name],
                declared,
                name=f"frozen development {run.run_id} {artifact_name}",
            )

    raw_artifacts = value["artifacts"]
    if not isinstance(raw_artifacts, Mapping):
        raise FormalPilotOrchestrationError(
            "calibration manifest artifacts must be an object"
        )
    filenames = {
        "search": SEARCH_FILENAME,
        "development_metrics": DEVELOPMENT_METRICS_FILENAME,
        "formal_config": FORMAL_CONFIG_FILENAME,
    }
    _expect_exact_keys(raw_artifacts, set(filenames), name="calibration artifacts")
    snapshots: dict[str, _Snapshot] = {}
    for artifact_name, filename in filenames.items():
        declared = _validated_artifact(
            raw_artifacts[artifact_name],
            name=f"calibration artifacts.{artifact_name}",
            expected_path=filename,
        )
        snapshot = _read_snapshot(root / filename, name=f"calibration {artifact_name}")
        _assert_artifact(snapshot, declared, name=f"calibration {artifact_name}")
        _strict_json_object(snapshot.payload, name=f"calibration {artifact_name}")
        snapshots[artifact_name] = snapshot
    for run in development_inputs:
        _validate_run_json_runner(
            development_outputs[run.run_id]["run_json"],
            commit_bundle.runner_provenance,
            run_id=run.run_id,
        )
    _validated_split_run_timeline(
        development_inputs,
        development_outputs,
        split="development",
        earliest_start=commit_bundle.execution_unlocked_at,
        latest_finish=holdout_unlocked_at,
    )
    development_runs = [
        run.suite_run(
            development_outputs[run.run_id]["health_jsonl"].payload,
            development_outputs[run.run_id]["run_json"].payload,
        )
        for run in development_inputs
    ]
    recomputed_search = detection_suite.search_development_suite(
        development_runs,
        detection_suite.fixed_candidate_order(),
        master_seed=0,
    )
    recomputed_metrics = detection_suite.evaluate_detection_suite(
        development_runs,
        recomputed_search["frozen_detection_config"],
        expected_split="development",
    )
    recomputed_config = detection_suite.build_formal_v1_config(
        development_runs,
        recomputed_search,
        development_run_id="FORMAL-DEV-SUITE-0001",
        split_registry=commit_bundle.split_registry.payload,
        holdout_commitment=commit_bundle.holdout_commitment.payload,
    )
    for artifact_name, recomputed in (
        ("search", recomputed_search),
        ("development_metrics", recomputed_metrics),
        ("formal_config", recomputed_config),
    ):
        if _json_bytes(recomputed) != snapshots[artifact_name].payload:
            raise FormalPilotOrchestrationError(
                f"calibration {artifact_name} differs from deterministic replay"
            )
    return _CalibrationBundle(
        root=root,
        manifest=manifest_snapshot,
        manifest_value=value,
        search=snapshots["search"],
        development_metrics=snapshots["development_metrics"],
        formal_config=snapshots["formal_config"],
        development_outputs=development_outputs,
        deterministic_replay_verified=True,
        holdout_unlock=holdout_unlock,
        holdout_unlocked_at=holdout_unlocked_at,
    )


def _finite_metric(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormalPilotOrchestrationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise FormalPilotOrchestrationError(f"{name} must be a finite number")
    return result


def _method(metrics: Mapping[str, Any], method_name: str) -> Mapping[str, Any]:
    methods = metrics.get("methods")
    if not isinstance(methods, Mapping) or not isinstance(
        methods.get(method_name), Mapping
    ):
        raise FormalPilotOrchestrationError(
            f"metrics lacks method {method_name!r}"
        )
    return methods[method_name]


def _macro_auroc(metrics: Mapping[str, Any], method_name: str) -> float:
    method = _method(metrics, method_name)
    macro = method.get("macro")
    if not isinstance(macro, Mapping):
        raise FormalPilotOrchestrationError(
            f"metrics method {method_name!r} lacks macro metrics"
        )
    return _finite_metric(macro.get("auroc"), name=f"{method_name}.macro.auroc")


def _detected_corruption_types(metrics: Mapping[str, Any]) -> set[str]:
    combined = _method(metrics, "combined")
    events = combined.get("events")
    if not isinstance(events, Mapping) or not isinstance(events.get("intervals"), list):
        raise FormalPilotOrchestrationError("combined events.intervals is missing")
    detected: set[str] = set()
    for index, interval in enumerate(events["intervals"]):
        if not isinstance(interval, Mapping):
            raise FormalPilotOrchestrationError(
                f"combined events.intervals[{index}] must be an object"
            )
        corruption_type = interval.get("corruption_type")
        if corruption_type not in CORRUPTION_ORDER:
            raise FormalPilotOrchestrationError(
                f"combined event {index} has invalid corruption type"
            )
        if interval.get("delay_frames") is not None:
            detected.add(str(corruption_type))
    return detected


def _combined_signals_present(metrics: Mapping[str, Any]) -> bool:
    try:
        combined = _method(metrics, "combined")
    except FormalPilotOrchestrationError:
        return False
    per_run = combined.get("per_run")
    if not isinstance(per_run, Mapping) or len(per_run) != 3:
        return False
    for row in per_run.values():
        if not isinstance(row, Mapping):
            return False
        used = row.get("used_signals")
        sources = row.get("signal_sources")
        if (
            not isinstance(used, list)
            or not {"update_magnitude", "reliability"}.issubset(used)
            or not isinstance(sources, list)
            or not sources
        ):
            return False
    return True


def evaluate_go_no_go(
    development_metrics: Mapping[str, Any],
    holdout_metrics: Mapping[str, Any],
    *,
    provenance_passed: bool,
) -> dict[str, Any]:
    """Evaluate the seven conjunctive pre-registered conditions exactly once."""

    combined = _macro_auroc(holdout_metrics, "combined")
    random = _macro_auroc(holdout_metrics, "random")
    update = _macro_auroc(holdout_metrics, "update_magnitude_only")
    reliability = _macro_auroc(holdout_metrics, "reliability_only")
    pooled = _method(holdout_metrics, "combined").get("pooled")
    diagnostics = _method(holdout_metrics, "combined").get("diagnostics")
    if not isinstance(pooled, Mapping) or not isinstance(diagnostics, Mapping):
        raise FormalPilotOrchestrationError(
            "holdout combined pooled/diagnostic metrics are missing"
        )
    false_positive_rate = _finite_metric(
        pooled.get("false_positive_rate"),
        name="combined.pooled.false_positive_rate",
    )
    longest_streak = _plain_integer(
        diagnostics.get("max_per_run_consecutive_false_positives"),
        name="combined.max_per_run_consecutive_false_positives",
    )
    dev_detected = _detected_corruption_types(development_metrics)
    holdout_detected = _detected_corruption_types(holdout_metrics)
    shared_detected = dev_detected & holdout_detected
    signals_passed = _combined_signals_present(
        development_metrics
    ) and _combined_signals_present(holdout_metrics)
    combined_decimal = Decimal(str(combined))
    random_margin = combined_decimal - Decimal(str(random))
    best_single = max(Decimal(str(update)), Decimal(str(reliability)))
    checks = [
        {
            "id": "combined_macro_auroc_strictly_above_0_75",
            "passed": combined > 0.75,
            "actual": combined,
            "operator": ">",
            "threshold": 0.75,
        },
        {
            "id": "combined_minus_random_at_least_0_10",
            "passed": random_margin >= Decimal("0.10"),
            "actual": float(random_margin),
            "operator": ">=",
            "threshold": 0.10,
        },
        {
            "id": "combined_within_0_02_of_best_single_signal",
            "passed": combined_decimal >= best_single - Decimal("0.02"),
            "actual": combined,
            "best_single_signal": float(best_single),
            "operator": ">= best_single_signal -",
            "tolerance": 0.02,
        },
        {
            "id": "same_at_least_two_detected_corruption_types",
            "passed": len(shared_detected) >= 2,
            "development_detected": sorted(dev_detected),
            "holdout_detected": sorted(holdout_detected),
            "shared_detected": sorted(shared_detected),
            "minimum_shared_types": 2,
        },
        {
            "id": "holdout_pooled_false_positive_rate_at_most_0_20",
            "passed": false_positive_rate <= 0.20,
            "actual": false_positive_rate,
            "operator": "<=",
            "threshold": 0.20,
        },
        {
            "id": "holdout_max_false_positive_streak_at_most_3",
            "passed": longest_streak <= 3,
            "actual": longest_streak,
            "operator": "<=",
            "threshold": 3,
        },
        {
            "id": "provenance_runtime_reproducibility_and_real_signals",
            "passed": provenance_passed is True and signals_passed,
            "provenance_passed": provenance_passed is True,
            "combined_real_signals_passed": signals_passed,
        },
    ]
    passed = all(check["passed"] is True for check in checks)
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "decision": "GO" if passed else "NO-GO",
        "passed": passed,
        "checks": checks,
        "scope": "within-sequence pilot only",
        "next_step": (
            "minimal quarantine experiment may be designed"
            if passed
            else "stop before quarantine or rollback"
        ),
    }


def _runtime_summary(
    input_bundle: _InputBundle,
    outputs: Mapping[str, Mapping[str, _Snapshot]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in input_bundle.runs:
        value = _strict_json_object(
            outputs[run.run_id]["run_json"].payload, name=f"{run.run_id} run.json"
        )
        runtime = _finite_metric(
            value.get("runtime_seconds"), name=f"{run.run_id}.runtime_seconds"
        )
        peak = _finite_metric(
            value.get("peak_memory_allocated_mib"),
            name=f"{run.run_id}.peak_memory_allocated_mib",
        )
        if runtime <= 0 or peak <= 0:
            raise FormalPilotOrchestrationError(
                f"{run.run_id} runtime and peak memory must be positive"
            )
        rows.append(
            {
                "run_id": run.run_id,
                "dataset_split": run.dataset_split,
                "corruption_type": run.corruption_type,
                "runtime_seconds": runtime,
                "peak_memory_allocated_mib": peak,
            }
        )
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "run_count": 6,
        "total_runtime_seconds": math.fsum(row["runtime_seconds"] for row in rows),
        "maximum_peak_memory_allocated_mib": max(
            row["peak_memory_allocated_mib"] for row in rows
        ),
        "runs": rows,
    }


def _timeline_view(metrics: Mapping[str, Any], run: _InputRun) -> dict[str, Any]:
    raw_runs = metrics.get("runs")
    if not isinstance(raw_runs, list):
        raise FormalPilotOrchestrationError("metrics runs array is missing")
    run_regions = next(
        (
            row
            for row in raw_runs
            if isinstance(row, Mapping) and row.get("run_id") == run.run_id
        ),
        None,
    )
    combined = _method(metrics, "combined")
    per_run = combined.get("per_run")
    if run_regions is None or not isinstance(per_run, Mapping) or not isinstance(
        per_run.get(run.run_id), Mapping
    ):
        raise FormalPilotOrchestrationError(
            f"metrics lacks timeline data for {run.run_id}"
        )
    labels = run_regions.get("primary_labels")
    scores = per_run[run.run_id].get("scores")
    return {
        "schema_version": "stateguard3r.formal-timeline-compat.v1",
        "frame_count": 30,
        "frame_ids": list(range(30)),
        "labels": labels,
        "methods": {"combined": {"scores": scores}},
        "by_corruption_type": {run.corruption_type: {"labels": labels}},
    }


def _timeline_payloads(
    input_bundle: _InputBundle,
    development_metrics: Mapping[str, Any],
    holdout_metrics: Mapping[str, Any],
    outputs: Mapping[str, Mapping[str, _Snapshot]],
) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for run in input_bundle.runs:
        metrics = (
            development_metrics
            if run.dataset_split == "development"
            else holdout_metrics
        )
        records = _strict_health_records(
            outputs[run.run_id]["health_jsonl"].payload,
            name=f"{run.run_id} health.jsonl",
        )
        svg = render_timeline_svg(
            records,
            _timeline_view(metrics, run),
            title=f"Formal pilot: {run.run_id}",
        )
        payloads[f"{TIMELINE_DIRECTORY}/{run.run_id}.svg"] = svg.encode("utf-8")
    return payloads


def _preflight_holdout_outputs(runs_root: Path, run_ids: Sequence[str]) -> None:
    problems: list[str] = []
    for run_id in run_ids:
        try:
            run_directory = _validated_child_directory(
                runs_root, (run_id,), name=f"{run_id} holdout output run"
            )
        except FormalPilotOrchestrationError as error:
            problems.append(str(error))
            continue
        try:
            directory_metadata = os.lstat(run_directory)
            if stat.S_IMODE(directory_metadata.st_mode) != 0o555:
                problems.append(f"{run_id} output directory is not frozen 0555")
            filenames = {entry.name for entry in os.scandir(run_directory)}
            if filenames != set(RUN_OUTPUT_ARTIFACT_FILENAMES.values()):
                problems.append(f"{run_id} does not contain exactly five artifacts")
        except OSError as error:
            problems.append(f"cannot inspect {run_id} frozen output tree: {error}")
            continue
        for filename in RUN_OUTPUT_ARTIFACT_FILENAMES.values():
            try:
                _preflight_regular_file(
                    run_directory / filename,
                    name=f"{run_id} {filename}",
                )
                if stat.S_IMODE(os.lstat(run_directory / filename).st_mode) != 0o444:
                    problems.append(f"{run_id} {filename} is not frozen 0444")
            except (FormalPilotOrchestrationError, OSError) as error:
                problems.append(str(error))
    if problems:
        raise FormalPilotOrchestrationError(
            "all three holdout runs must be complete before any holdout output is read; "
            + " | ".join(problems)
        )


def evaluate_holdout(
    input_root: str | os.PathLike[str],
    commit_dir: str | os.PathLike[str],
    calibration_dir: str | os.PathLike[str],
    runs_root: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Evaluate the already-frozen holdout once and publish the formal gate."""

    _assert_output_absent(output_dir)
    (
        _,
        registry_snapshot,
        registry_value,
        registry_records,
    ) = _read_input_registry(input_root)
    runs_directory = _resolve_root(runs_root, name="RUNS_ROOT")
    holdout_ids = [
        str(record["run_id"])
        for record in registry_records
        if record["dataset_split"] == "holdout"
    ]
    # This is intentionally before source/input, commitment, calibration, or
    # run-output content reads.  Incomplete holdout execution leaks zero bytes.
    _preflight_holdout_outputs(runs_directory, holdout_ids)

    inputs = _load_input_bundle(
        input_root,
        registry_snapshot=registry_snapshot,
        registry_value=registry_value,
        registry_records=registry_records,
    )
    committed = _load_commit_bundle(commit_dir, inputs)
    calibrated = _load_calibration_bundle(
        calibration_dir, inputs, committed, runs_directory
    )
    holdout_inputs = _runs_by_split(inputs, "holdout")
    # Snapshot all six holdout files before parsing or evaluating any one run.
    holdout_outputs = _snapshot_run_outputs(runs_directory, holdout_inputs)
    all_outputs: dict[str, Mapping[str, _Snapshot]] = {
        **calibrated.development_outputs,
        **holdout_outputs,
    }
    for run in inputs.runs:
        _validate_run_json_runner(
            all_outputs[run.run_id]["run_json"],
            committed.runner_provenance,
            run_id=run.run_id,
        )
    holdout_timeline = _validated_split_run_timeline(
        holdout_inputs,
        holdout_outputs,
        split="holdout",
        earliest_start=calibrated.holdout_unlocked_at,
        latest_finish=_utc_now(),
    )
    holdout_runs = [
        run.suite_run(
            holdout_outputs[run.run_id]["health_jsonl"].payload,
            holdout_outputs[run.run_id]["run_json"].payload,
        )
        for run in holdout_inputs
    ]
    development_metrics = _strict_json_object(
        calibrated.development_metrics.payload, name=DEVELOPMENT_METRICS_FILENAME
    )
    formal_config = _strict_json_object(
        calibrated.formal_config.payload, name=FORMAL_CONFIG_FILENAME
    )
    search_result = _strict_json_object(
        calibrated.search.payload, name=SEARCH_FILENAME
    )
    development_runs = [
        run.suite_run(
            calibrated.development_outputs[run.run_id]["health_jsonl"].payload,
            calibrated.development_outputs[run.run_id]["run_json"].payload,
        )
        for run in _runs_by_split(inputs, "development")
    ]
    holdout_metrics = detection_suite.evaluate_formal_holdout(
        holdout_runs,
        formal_config,
        split_registry=committed.split_registry.payload,
        holdout_commitment=committed.holdout_commitment.payload,
        search_result=search_result,
        development_runs=development_runs,
        runtime_detection_config=formal_config.get("frozen_detection_config"),
    )
    runtime_summary = _runtime_summary(inputs, all_outputs)
    provenance_evidence = {
        "input_cpu_validation_report_verified": bool(
            _validate_cpu_validation_report(
                committed.validation_report, inputs, committed.runner_provenance
            )
        ),
        "commitment_chain_verified": committed.manifest_value.get("stage") == "commit",
        "calibration_chain_verified": calibrated.manifest_value.get("stage")
        == "calibrate",
        "deterministic_development_replay_verified": (
            calibrated.deterministic_replay_verified
        ),
        "current_repositories_and_runner_verified": bool(
            committed.runner_provenance
        ),
        "six_run_snapshots_verified": len(all_outputs) == 6,
        "formal_holdout_run_bindings_verified": holdout_metrics.get("formal") is True,
        "runtime_checks_verified": runtime_summary.get("run_count") == 6,
        "frozen_serial_execution_timeline_verified": len(holdout_timeline) == 3,
    }
    provenance_passed = all(value is True for value in provenance_evidence.values())
    gate = evaluate_go_no_go(
        development_metrics,
        holdout_metrics,
        provenance_passed=provenance_passed,
    )
    timeline_payloads = _timeline_payloads(
        inputs,
        development_metrics,
        holdout_metrics,
        all_outputs,
    )
    if len(timeline_payloads) != 6:
        raise FormalPilotOrchestrationError("formal evaluation must render six timelines")

    holdout_metrics_bytes = _json_bytes(holdout_metrics)
    gate_bytes = _json_bytes(gate)
    runtime_bytes = _json_bytes(runtime_summary)
    output_snapshots = {
        "holdout_metrics": _Snapshot(Path(HOLDOUT_METRICS_FILENAME), holdout_metrics_bytes),
        "go_no_go": _Snapshot(Path(GO_NO_GO_FILENAME), gate_bytes),
        "runtime_summary": _Snapshot(Path(RUNTIME_SUMMARY_FILENAME), runtime_bytes),
    }
    run_artifacts = _run_output_artifacts(all_outputs)
    evaluation_manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "stage": "evaluate",
        "inputs": {
            "input_registry": inputs.registry.artifact(INPUT_REGISTRY_FILENAME),
            "commitment_manifest": committed.manifest.artifact(
                COMMITMENT_MANIFEST_FILENAME
            ),
            "calibration_manifest": calibrated.manifest.artifact(
                CALIBRATION_MANIFEST_FILENAME
            ),
            "holdout_unlock": calibrated.holdout_unlock.artifact(
                str(calibrated.holdout_unlock.path.resolve(strict=True))
            ),
            "protocol": committed.protocol.artifact(
                str(committed.protocol.path.resolve(strict=True))
            ),
            "validation_report": committed.validation_report.artifact(
                str(committed.validation_report.path.resolve(strict=True))
            ),
            "split_registry": committed.split_registry.artifact(
                SPLIT_REGISTRY_FILENAME
            ),
            "holdout_commitment": committed.holdout_commitment.artifact(
                HOLDOUT_COMMITMENT_FILENAME
            ),
            "search": calibrated.search.artifact(SEARCH_FILENAME),
            "development_metrics": calibrated.development_metrics.artifact(
                DEVELOPMENT_METRICS_FILENAME
            ),
            "formal_config": calibrated.formal_config.artifact(
                FORMAL_CONFIG_FILENAME
            ),
        },
        "run_artifacts": run_artifacts,
        "artifacts": {
            "holdout_metrics": output_snapshots["holdout_metrics"].artifact(
                HOLDOUT_METRICS_FILENAME
            ),
            "go_no_go": output_snapshots["go_no_go"].artifact(GO_NO_GO_FILENAME),
            "runtime_summary": output_snapshots["runtime_summary"].artifact(
                RUNTIME_SUMMARY_FILENAME
            ),
            "timelines": {
                path: _Snapshot(Path(path), payload).artifact(path)
                for path, payload in sorted(timeline_payloads.items())
            },
        },
        "run_ids": _run_ids(inputs),
        "decision": gate["decision"],
        "provenance_gate_evidence": provenance_evidence,
    }
    evaluation_manifest_bytes = _json_bytes(evaluation_manifest)
    files = {
        HOLDOUT_METRICS_FILENAME: holdout_metrics_bytes,
        GO_NO_GO_FILENAME: gate_bytes,
        RUNTIME_SUMMARY_FILENAME: runtime_bytes,
        EVALUATION_MANIFEST_FILENAME: evaluation_manifest_bytes,
        **timeline_payloads,
    }
    output = _publish_immutable_tree(output_dir, files)
    return {
        "stage": "evaluate",
        "output_dir": str(output),
        "decision": gate["decision"],
        "evaluation_manifest_sha256": hashlib.sha256(
            evaluation_manifest_bytes
        ).hexdigest(),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    commit_parser = subparsers.add_parser("commit", help="freeze pre-forward inputs")
    commit_parser.add_argument("input_root", type=Path)
    commit_parser.add_argument("output_dir", type=Path)
    commit_parser.add_argument("--protocol", required=True, type=Path)

    calibrate_parser = subparsers.add_parser(
        "calibrate", help="search development outputs and freeze calibration"
    )
    calibrate_parser.add_argument("input_root", type=Path)
    calibrate_parser.add_argument("commit_dir", type=Path)
    calibrate_parser.add_argument("runs_root", type=Path)
    calibrate_parser.add_argument("output_dir", type=Path)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate the complete frozen holdout once"
    )
    evaluate_parser.add_argument("input_root", type=Path)
    evaluate_parser.add_argument("commit_dir", type=Path)
    evaluate_parser.add_argument("calibration_dir", type=Path)
    evaluate_parser.add_argument("runs_root", type=Path)
    evaluate_parser.add_argument("output_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "commit":
            result = commit_inputs(
                args.input_root,
                args.output_dir,
                protocol=args.protocol,
            )
        elif args.command == "calibrate":
            result = calibrate_development(
                args.input_root,
                args.commit_dir,
                args.runs_root,
                args.output_dir,
            )
        else:
            result = evaluate_holdout(
                args.input_root,
                args.commit_dir,
                args.calibration_dir,
                args.runs_root,
                args.output_dir,
            )
    except (FormalPilotOrchestrationError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
