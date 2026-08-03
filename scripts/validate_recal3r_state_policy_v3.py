#!/usr/bin/env python3
"""Validate external ReCal3R state-policy feasibility evidence on CPU only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


CONTROL_ARTIFACTS = (
    "checkpoint-load-audit.json",
    "health.jsonl",
    "predictions-summary.json",
    "trajectory.json",
)
POLICY_ARTIFACTS = (*CONTROL_ARTIFACTS, "run.json", "state-timeline.json")


class StatePolicyValidationError(RuntimeError):
    """Raised when the bounded state-policy feasibility contract is violated."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v2_control", type=Path)
    parser.add_argument("always_commit", type=Path)
    parser.add_argument("forced_hold", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--always-postflight-log", required=True, type=Path)
    parser.add_argument("--forced-postflight-log", required=True, type=Path)
    return parser


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StatePolicyValidationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StatePolicyValidationError(f"cannot read JSON {path}: {error}") from error


def _regular(path: Path, *, label: str) -> None:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular non-symlink file")


def _artifacts(directory: Path, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        path = directory / name
        _regular(path, label=f"{directory.name}/{name}")
        result[name] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    return result


def _assert_immutable(directory: Path, names: Sequence[str]) -> None:
    _require(directory.is_dir() and not directory.is_symlink(), f"output directory is invalid: {directory}")
    _require((directory.stat().st_mode & 0o777) == 0o555, f"output directory is not mode 0555: {directory}")
    for name in names:
        path = directory / name
        _regular(path, label=f"immutable artifact {name}")
        _require((path.stat().st_mode & 0o777) == 0o444, f"artifact is not mode 0444: {path}")


def _assert_postflight(path: Path, *, pid: Any, label: str) -> dict[str, Any]:
    _regular(path, label=f"{label} postflight log")
    _require(isinstance(pid, int) and not isinstance(pid, bool), f"{label} PID is invalid")
    text = path.read_text(encoding="utf-8")
    _require("NVIDIA-SMI" in text or "NVSMI LOG" in text, f"{label} postflight lacks nvidia-smi")
    _require(str(pid) not in text, f"{label} PID remains in postflight snapshot")
    return {"path": str(path.resolve()), "sha256": _sha256(path), "run_pid_absent": True}


def _assert_shared_run_identity(v2: Mapping[str, Any], always: Mapping[str, Any]) -> None:
    for key in (
        "baseline_root",
        "baseline_commit",
        "baseline_tracked_worktree_clean",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "input_manifest",
        "images",
        "input_frames",
        "frame_count",
        "size",
        "seed",
        "model_update_type",
        "loaded_model_interface",
        "checkpoint_state_dict",
        "beta_base",
        "recal3r_runtime_config",
        "recal3r_runtime_config_source",
        "model_eval",
        "calibrated_update_calls",
        "expected_calibrated_update_calls",
        "trace_frame_steps",
        "health_profile",
    ):
        _require(v2.get(key) == always.get(key), f"v2/always run identity differs: {key}")


def _assert_forced_timeline(timeline: Mapping[str, Any], always_timeline: Mapping[str, Any]) -> dict[str, Any]:
    _require(timeline.get("policy") == "forced-prior-alarm", "forced policy name is invalid")
    _require(timeline.get("max_hold") == 3, "forced max_hold must be three")
    _require(timeline.get("alarm_positions") == [1], "forced run must use only synthetic alarm frame one")
    _require(timeline.get("dropped_transaction_frame_ids") == [], "forced run silently dropped held state")
    records = timeline.get("transactions")
    control_records = always_timeline.get("transactions")
    _require(isinstance(records, list) and isinstance(control_records, list), "state transactions are invalid")
    _require(len(records) == len(control_records) == 30, "state transaction count must be 30")
    for index, record in enumerate(records):
        _require(isinstance(record, Mapping) and record.get("frame_id") == index, f"forced transaction frame mismatch at {index}")
    changed = [record for record in records if record.get("action") != "commit"]
    _require(len(changed) == 2, "forced run must contain exactly hold and release/replay")
    hold, replay = changed
    _require(
        (hold.get("frame_id"), hold.get("action"), hold.get("reason"), hold.get("prior_alarm"))
        == (2, "hold", "prior_detector_alarm", True),
        "forced hold is not the causal frame-two transaction",
    )
    _require(
        hold.get("pre_state_digest_sha256") != hold.get("proposed_state_digest_sha256")
        and hold.get("committed_state_digest_sha256") == hold.get("pre_state_digest_sha256"),
        "held transaction did not restore its pre-state closure",
    )
    _require(
        (replay.get("frame_id"), replay.get("action"), replay.get("prior_alarm"), replay.get("replayed_transaction_frame_id"))
        == (3, "release_replay_then_commit", False, 2),
        "forced replay is not the next-frame causal release",
    )
    _require(
        replay.get("pre_state_digest_sha256") == hold.get("proposed_state_digest_sha256")
        and replay.get("committed_state_digest_sha256") == replay.get("proposed_state_digest_sha256"),
        "replayed state was not committed before the next proposal",
    )
    _require(
        all(record.get("action") == "commit" for record in control_records),
        "always-commit control contains a state intervention",
    )
    return {
        "hold_frame": 2,
        "replay_frame": 3,
        "held_proposal_digest": hold["proposed_state_digest_sha256"],
        "restored_pre_digest": hold["committed_state_digest_sha256"],
        "replayed_pre_digest": replay["pre_state_digest_sha256"],
        "post_replay_commit_digest": replay["committed_state_digest_sha256"],
    }


def _assert_detector_timeline(
    timeline: Mapping[str, Any],
    always_timeline: Mapping[str, Any],
    *,
    alarm_positions: list[int],
) -> dict[str, Any]:
    """Verify a real frozen-v3 alarm caused causal hold/replay transitions."""
    _require(timeline.get("policy") == "detector-v3-prior-alarm", "detector policy name is invalid")
    _require(timeline.get("max_hold") == 3, "detector max_hold must be three")
    _require(timeline.get("alarm_positions") == alarm_positions, "detector alarm positions differ")
    _require(timeline.get("dropped_transaction_frame_ids") == [], "detector policy silently dropped held state")
    records = timeline.get("transactions")
    control_records = always_timeline.get("transactions")
    _require(isinstance(records, list) and isinstance(control_records, list), "detector state transactions are invalid")
    _require(len(records) == len(control_records) == 30, "detector state transaction count must be 30")
    holds = [record for record in records if isinstance(record, Mapping) and record.get("action") == "hold"]
    replays = [
        record
        for record in records
        if isinstance(record, Mapping) and str(record.get("action", "")).startswith("release_replay_then_")
    ]
    _require(holds, "frozen detector alarm caused no held state transaction")
    _require(replays, "frozen detector alarm caused no replayed state transaction")
    held_proposals: dict[int, str] = {}
    for hold in holds:
        frame_id = hold.get("frame_id")
        _require(isinstance(frame_id, int) and frame_id > 0, "detector hold frame is invalid")
        _require(frame_id - 1 in alarm_positions and hold.get("prior_alarm") is True, "detector hold is not caused by the prior frozen alarm")
        _require(
            hold.get("pre_state_digest_sha256") != hold.get("proposed_state_digest_sha256")
            and hold.get("committed_state_digest_sha256") == hold.get("pre_state_digest_sha256"),
            "detector-held transaction did not restore its pre-state closure",
        )
        proposal = hold.get("proposed_state_digest_sha256")
        _require(isinstance(proposal, str), "detector-held proposal digest is invalid")
        held_proposals[frame_id] = proposal
    for replay in replays:
        replayed = replay.get("replayed_transaction_frame_id")
        _require(
            isinstance(replayed, int)
            and replayed in held_proposals
            and replay.get("prior_alarm") is False
            and replay.get("pre_state_digest_sha256") == held_proposals[replayed],
            "detector replay did not restore a held proposal before the next candidate",
        )
    _require(
        all(isinstance(record, Mapping) and record.get("action") == "commit" for record in control_records),
        "always-commit control contains a state intervention",
    )
    return {
        "hold_frames": [record["frame_id"] for record in holds],
        "replay_frames": [record["frame_id"] for record in replays],
        "held_proposal_digests": {str(frame): digest for frame, digest in held_proposals.items()},
    }


def _assert_detector_alarm_source(forced_run: Mapping[str, Any]) -> tuple[list[int], dict[str, Any]]:
    policy = forced_run.get("state_policy")
    _require(isinstance(policy, Mapping), "detector state policy metadata is invalid")
    source = policy.get("alarm_source")
    _require(isinstance(source, Mapping) and source.get("kind") == "frozen_formal_v3_hybrid_alarm", "detector alarm source is missing")
    input_artifact = source.get("input_manifest")
    forced_input = forced_run.get("input_manifest")
    _require(isinstance(input_artifact, Mapping) and isinstance(forced_input, Mapping), "detector alarm source input binding is missing")
    _require(
        all(input_artifact.get(key) == forced_input.get(key) for key in ("path", "sha256"))
        and input_artifact.get("size_bytes") == Path(str(forced_input.get("path", ""))).stat().st_size,
        "detector alarm source input binding differs",
    )
    run_artifact = source.get("run")
    timeline_artifact = source.get("timeline")
    _require(isinstance(run_artifact, Mapping) and isinstance(timeline_artifact, Mapping), "detector alarm source artifacts are missing")
    evidence: dict[str, Any] = {}
    values: dict[str, Mapping[str, Any]] = {}
    for label, artifact in (("run", run_artifact), ("timeline", timeline_artifact)):
        path = Path(str(artifact.get("path", "")))
        _regular(path, label=f"detector source {label}")
        _require((path.stat().st_mode & 0o777) == 0o444, f"detector source {label} is not frozen")
        _require(artifact.get("sha256") == _sha256(path) and artifact.get("size_bytes") == path.stat().st_size, f"detector source {label} binding differs")
        value = _read_json(path)
        _require(isinstance(value, Mapping), f"detector source {label} JSON is invalid")
        values[label] = value
        evidence[label] = {"path": str(path.resolve()), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
    _require(values["run"].get("health_profile") == "v3", "detector source run is not v3")
    run_input = values["run"].get("input_manifest")
    _require(
        isinstance(run_input, Mapping)
        and all(run_input.get(key) == input_artifact.get(key) for key in ("path", "sha256")),
        "detector source run input differs",
    )
    _require(values["timeline"].get("run_id") == "blind-wrong-order", "detector source timeline differs")
    rows = values["timeline"].get("attribution")
    _require(isinstance(rows, list) and len(rows) == forced_run.get("frame_count"), "detector source attribution count differs")
    hybrid_positions: list[int] = []
    for frame_id, row in enumerate(rows):
        _require(isinstance(row, Mapping) and row.get("frame_id") == frame_id and isinstance(row.get("hybrid_alarm"), bool), "detector source attribution row differs")
        if row["hybrid_alarm"]:
            hybrid_positions.append(frame_id)
    policy_positions = [
        frame_id
        for frame_id, row in enumerate(rows)
        if row["hybrid_alarm"] and (frame_id == 0 or not rows[frame_id - 1]["hybrid_alarm"])
    ]
    _require(
        hybrid_positions
        and source.get("hybrid_alarm_positions") == hybrid_positions
        and source.get("policy_alarm_positions") == policy_positions
        and source.get("policy_filter") == "causal_hybrid_alarm_rising_edge",
        "detector source hybrid alarm positions differ",
    )
    evidence["hybrid_alarm_positions"] = hybrid_positions
    evidence["policy_alarm_positions"] = policy_positions
    return policy_positions, evidence


def validate_state_policy(
    *,
    v2_control: Path,
    always_commit: Path,
    forced_hold: Path,
    always_postflight_log: Path,
    forced_postflight_log: Path,
) -> dict[str, Any]:
    """Validate completed external-policy runs without modifying them."""

    v2_control = v2_control.resolve()
    always_commit = always_commit.resolve()
    forced_hold = forced_hold.resolve()
    for directory in (v2_control, always_commit, forced_hold):
        _require(directory.is_dir(), f"run directory does not exist: {directory}")
    _assert_immutable(v2_control, CONTROL_ARTIFACTS)
    _assert_immutable(always_commit, POLICY_ARTIFACTS)
    _assert_immutable(forced_hold, POLICY_ARTIFACTS)
    v2_artifacts = _artifacts(v2_control, CONTROL_ARTIFACTS)
    always_artifacts = _artifacts(always_commit, POLICY_ARTIFACTS)
    forced_artifacts = _artifacts(forced_hold, POLICY_ARTIFACTS)
    for name in CONTROL_ARTIFACTS:
        _require(v2_artifacts[name] == always_artifacts[name], f"always-commit is not byte-equivalent to v2: {name}")
        _require(always_artifacts[name] == forced_artifacts[name], f"forced policy changed model/health artifact: {name}")
    v2_run = _read_json(v2_control / "run.json")
    always_run = _read_json(always_commit / "run.json")
    forced_run = _read_json(forced_hold / "run.json")
    _require(all(isinstance(item, Mapping) for item in (v2_run, always_run, forced_run)), "run metadata is invalid")
    _assert_shared_run_identity(v2_run, always_run)
    _require(always_run.get("state_policy", {}).get("name") == "always-commit", "always control policy is invalid")
    _assert_shared_run_identity(always_run, forced_run)
    policy_name = forced_run.get("state_policy", {}).get("name")
    _require(policy_name in {"forced-prior-alarm", "detector-v3-prior-alarm"}, "forced policy metadata is invalid")
    always_timeline = _read_json(always_commit / "state-timeline.json")
    forced_timeline = _read_json(forced_hold / "state-timeline.json")
    _require(isinstance(always_timeline, Mapping) and isinstance(forced_timeline, Mapping), "state timeline is invalid")
    detector_source: dict[str, Any] | None = None
    if policy_name == "forced-prior-alarm":
        timeline = _assert_forced_timeline(forced_timeline, always_timeline)
    else:
        positions, detector_source = _assert_detector_alarm_source(forced_run)
        timeline = _assert_detector_timeline(forced_timeline, always_timeline, alarm_positions=positions)
    return {
        "schema_version": "stateguard3r.recal3r-state-policy-v3-feasibility.v1",
        "status": "PASS",
        "scope": "transactional_state_policy_validated_no_recovery_quality_claim",
        "v2_control_dir": str(v2_control),
        "always_commit_dir": str(always_commit),
        "forced_hold_dir": str(forced_hold),
        "checks": {
            "always_commit_byte_equivalent_to_v2": True,
            "forced_policy_model_and_health_artifacts_unchanged": True,
            "source_bound_transaction_timeline_present": True,
            "held_state_restored_to_pre_state": True,
            "subsequent_clear_score_replayed_held_state": True,
            "bounded_hold_no_silent_drop": True,
            "always_postflight_pid_absent": True,
            "forced_postflight_pid_absent": True,
            "detector_alarm_source_bound": policy_name != "detector-v3-prior-alarm" or detector_source is not None,
        },
        "timeline": timeline,
        "detector_alarm_source": detector_source,
        "artifacts": {"v2": v2_artifacts, "always": always_artifacts, "forced": forced_artifacts},
        "postflight": {
            "always": _assert_postflight(always_postflight_log, pid=always_run.get("pid"), label="always"),
            "forced": _assert_postflight(forced_postflight_log, pid=forced_run.get("pid"), label="forced"),
        },
    }


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write((json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = args.output_dir.resolve(strict=False)
    _require(not output_dir.exists(), f"refusing to overwrite existing output directory: {output_dir}")
    report = validate_state_policy(
        v2_control=args.v2_control,
        always_commit=args.always_commit,
        forced_hold=args.forced_hold,
        always_postflight_log=args.always_postflight_log,
        forced_postflight_log=args.forced_postflight_log,
    )
    script_path = Path(__file__).resolve()
    report["validator"] = {"path": str(script_path), "sha256": _sha256(script_path)}
    _write_atomic(output_dir / "state-policy-feasibility.json", report)
    for path in output_dir.iterdir():
        if path.is_file():
            path.chmod(0o444)
    output_dir.chmod(0o555)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
