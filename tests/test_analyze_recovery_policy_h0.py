from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_recovery_policy_h0 import RecoveryPolicyH0Error, analyze_pair


def _freeze(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _pair(tmp_path: Path, *, altered_output: bool = False, invalid_replay: bool = False) -> tuple[Path, Path]:
    baseline = tmp_path / "baseline"
    policy = tmp_path / "policy"
    baseline.mkdir()
    policy.mkdir()
    for name in ("trajectory.json", "health.jsonl", "predictions-summary.json"):
        (baseline / name).write_text('{"same":true}\n', encoding="utf-8")
        (policy / name).write_text('{"different":true}\n' if altered_output and name == "trajectory.json" else '{"same":true}\n', encoding="utf-8")
    timeline = {
        "transactions": [
            {"frame_id": 0, "action": "commit", "replayed_transaction_frame_id": None, "pre_state_digest_sha256": "s0", "proposed_state_digest_sha256": "s1", "committed_state_digest_sha256": "s1"},
            {"frame_id": 1, "action": "hold", "replayed_transaction_frame_id": None, "pre_state_digest_sha256": "s1", "proposed_state_digest_sha256": "s2", "committed_state_digest_sha256": "s1"},
            {"frame_id": 2, "action": "release_replay_then_commit", "replayed_transaction_frame_id": 1, "pre_state_digest_sha256": "wrong" if invalid_replay else "s2", "proposed_state_digest_sha256": "s3", "committed_state_digest_sha256": "s3"},
        ],
        "dropped_transaction_frame_ids": [],
    }
    (policy / "state-timeline.json").write_text(json.dumps(timeline), encoding="utf-8")
    _freeze(baseline)
    _freeze(policy)
    return baseline, policy


def test_h0_pair_requires_byte_identical_outputs_and_reinstalled_held_state(tmp_path: Path) -> None:
    baseline, policy = _pair(tmp_path)

    result = analyze_pair(label="fixture", baseline_dir=baseline, policy_dir=policy)

    assert result["h0_supported"] is True
    assert result["hold_frame_ids"] == [1]
    assert result["replay_reinstalls"][0]["held_frame_id"] == 1


def test_h0_pair_rejects_a_policy_output_difference(tmp_path: Path) -> None:
    baseline, policy = _pair(tmp_path, altered_output=True)

    with pytest.raises(RecoveryPolicyH0Error, match="not byte-identical"):
        analyze_pair(label="fixture", baseline_dir=baseline, policy_dir=policy)


def test_h0_pair_rejects_release_without_the_held_post_state(tmp_path: Path) -> None:
    baseline, policy = _pair(tmp_path, invalid_replay=True)

    with pytest.raises(RecoveryPolicyH0Error, match="did not reinstall"):
        analyze_pair(label="fixture", baseline_dir=baseline, policy_dir=policy)
