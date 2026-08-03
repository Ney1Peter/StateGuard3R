from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_recal3r_state_policy_v3 import (
    CONTROL_ARTIFACTS,
    POLICY_ARTIFACTS,
    StatePolicyValidationError,
    validate_state_policy,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _run_metadata(*, policy: str | None, pid: int) -> dict[str, object]:
    metadata: dict[str, object] = {
        "baseline_root": "/baseline",
        "baseline_commit": "commit",
        "baseline_tracked_worktree_clean": True,
        "checkpoint": "/checkpoint",
        "checkpoint_sha256": "checkpoint-sha",
        "checkpoint_size_bytes": 1,
        "input_manifest": {"sha256": "manifest"},
        "images": [{"sha256": "image"}],
        "input_frames": [{"sha256": "image"}],
        "frame_count": 30,
        "size": 512,
        "seed": 0,
        "model_update_type": "recal3r",
        "loaded_model_interface": {"head": "dpt"},
        "checkpoint_state_dict": {"strict": False},
        "beta_base": 0.1,
        "recal3r_runtime_config": {"decay": 0.95},
        "recal3r_runtime_config_source": {"sha256": "source"},
        "model_eval": True,
        "calibrated_update_calls": 29,
        "expected_calibrated_update_calls": 29,
        "trace_frame_steps": list(range(1, 30)),
        "health_profile": "v2",
        "pid": pid,
    }
    if policy is not None:
        metadata["state_policy"] = {"name": policy}
    return metadata


def _record(frame_id: int, action: str = "commit") -> dict[str, object]:
    digest = f"{frame_id + 1:064x}"
    return {
        "frame_id": frame_id,
        "action": action,
        "reason": "prior_score_clear",
        "prior_alarm": False,
        "replayed_transaction_frame_id": None,
        "proposal_digest_sha256": digest,
        "pre_state_digest_sha256": digest,
        "proposed_state_digest_sha256": digest,
        "committed_state_digest_sha256": digest,
    }


def _timeline(*, forced: bool) -> dict[str, object]:
    records = [_record(index) for index in range(30)]
    if forced:
        pre = "a" * 64
        proposed = "b" * 64
        records[2] = {
            **_record(2, "hold"),
            "reason": "prior_detector_alarm",
            "prior_alarm": True,
            "pre_state_digest_sha256": pre,
            "proposed_state_digest_sha256": proposed,
            "committed_state_digest_sha256": pre,
        }
        records[3] = {
            **_record(3, "release_replay_then_commit"),
            "reason": "released_held_frame_2;prior_score_clear",
            "replayed_transaction_frame_id": 2,
            "pre_state_digest_sha256": proposed,
            "proposed_state_digest_sha256": "c" * 64,
            "committed_state_digest_sha256": "c" * 64,
        }
    return {
        "policy": "forced-prior-alarm" if forced else "always-commit",
        "max_hold": 3,
        "alarm_positions": [1] if forced else [],
        "dropped_transaction_frame_ids": [],
        "transactions": records,
    }


def _freeze(directory: Path) -> None:
    for path in directory.iterdir():
        path.chmod(0o444)
    directory.chmod(0o555)


def _write_run(directory: Path, *, policy: str | None, pid: int, forced: bool = False) -> None:
    directory.mkdir()
    for name in CONTROL_ARTIFACTS:
        if name == "health.jsonl":
            (directory / name).write_text('{"frame_id": 0}\n', encoding="utf-8")
        elif name == "trajectory.json":
            _write_json(directory / name, {"frames": []})
        elif name == "predictions-summary.json":
            _write_json(directory / name, [])
        else:
            _write_json(directory / name, {"strict": False})
    _write_json(directory / "run.json", _run_metadata(policy=policy, pid=pid))
    if policy is not None:
        _write_json(directory / "state-timeline.json", _timeline(forced=forced))
    _freeze(directory)


def _postflight(path: Path) -> None:
    path.write_text("NVSMI LOG\nNo runner remains\n", encoding="utf-8")


def test_validates_external_state_policy_hold_and_replay(tmp_path: Path) -> None:
    v2, always, forced = tmp_path / "v2", tmp_path / "always", tmp_path / "forced"
    _write_run(v2, policy=None, pid=1)
    _write_run(always, policy="always-commit", pid=2)
    _write_run(forced, policy="forced-prior-alarm", pid=3, forced=True)
    always_log, forced_log = tmp_path / "always.log", tmp_path / "forced.log"
    _postflight(always_log)
    _postflight(forced_log)

    report = validate_state_policy(
        v2_control=v2,
        always_commit=always,
        forced_hold=forced,
        always_postflight_log=always_log,
        forced_postflight_log=forced_log,
    )

    assert report["status"] == "PASS"
    assert all(report["checks"].values())
    assert report["timeline"]["hold_frame"] == 2


def test_rejects_hold_that_does_not_restore_pre_state(tmp_path: Path) -> None:
    v2, always, forced = tmp_path / "v2", tmp_path / "always", tmp_path / "forced"
    _write_run(v2, policy=None, pid=1)
    _write_run(always, policy="always-commit", pid=2)
    _write_run(forced, policy="forced-prior-alarm", pid=3, forced=True)
    forced.chmod(0o755)
    timeline_path = forced / "state-timeline.json"
    timeline_path.chmod(0o644)
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    timeline["transactions"][2]["committed_state_digest_sha256"] = "d" * 64
    _write_json(timeline_path, timeline)
    _freeze(forced)
    always_log, forced_log = tmp_path / "always.log", tmp_path / "forced.log"
    _postflight(always_log)
    _postflight(forced_log)

    with pytest.raises(StatePolicyValidationError, match="did not restore"):
        validate_state_policy(
            v2_control=v2,
            always_commit=always,
            forced_hold=forced,
            always_postflight_log=always_log,
            forced_postflight_log=forced_log,
        )
