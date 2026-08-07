from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts import validate_recovery_early_spatial_pooled_pose_v9 as validator


RUN_ID = "recovery-early-spatial-pooled-pose-v9-dynamic-always-commit-0001"


def _write(path: Path, value: object, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    else:
        path.write_text(str(value), encoding="utf-8")
    os.chmod(path, mode)


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> validator.Paths:
    root = tmp_path / "StateGuard3R"
    for name in ("outputs", "logs", "tmp"):
        (root / name).mkdir(parents=True)
    monkeypatch.setattr(validator, "ROOT", root)
    monkeypatch.setattr(validator, "BASELINE", root / "outputs" / "recovery-policy-development-v1-dynamic-baseline-0001")
    baseline = validator.BASELINE
    baseline.mkdir()
    protected = {
        "checkpoint-load-audit.json": "{}\n",
        "health.jsonl": "{\"frame_id\": 0}\n",
        "predictions-summary.json": "[]\n",
        "trajectory.json": "{}\n",
    }
    for name, value in protected.items():
        _write(baseline / name, value)
    _write(baseline / "run.json", {"runtime_seconds": 10.0})
    os.chmod(baseline, 0o555)

    paths = validator._paths(RUN_ID)
    paths.output.mkdir()
    command_sha, pid = "a" * 64, 731
    snapshots = [
        {"selected": {"uuid": validator.GPU_UUID, "memory_free_mib": validator.MIN_FREE_MIB}},
        {"selected": {"uuid": validator.GPU_UUID, "memory_free_mib": validator.MIN_FREE_MIB}},
    ]
    _write(paths.preflight, {"run_id": RUN_ID, "command_sha256": command_sha, "gpu_snapshots": snapshots})
    result = {"run_id": RUN_ID, "command_sha256": command_sha, "child_pid": pid, "exit_code": 0}
    _write(paths.result, result)
    _write(paths.postflight, {"run_id": RUN_ID, "result": result, "child_pid_absent": True})
    _write(paths.driver, "#!/usr/bin/env bash\n")
    markers = "\n".join((
        f"V9_DRIVER_START run_id={RUN_ID}",
        f"V9_DRIVER_COMMAND_SHA256 {command_sha}",
        f"V9_DRIVER_DISPATCHED run_id={RUN_ID}",
        f"V9_FORWARD_DISPATCHED run_id={RUN_ID} command_sha256={command_sha} child_pid={pid}",
        f"V9_DRIVER_EXIT run_id={RUN_ID} child_pid={pid} exit_code=0",
        "",
    ))
    _write(paths.main, markers)
    _write(paths.transcript, "\n".join(markers.splitlines()[2:]) + "\n")
    for name, value in protected.items():
        _write(paths.output / name, value)
    transactions = [
        {
            "frame_id": index, "action": "commit", "current_alarm": False,
            "export_action": "export_real_camera_pose",
            "pending_transaction_count": 0, "restore_witness": None,
        }
        for index in range(30)
    ]
    _write(paths.output / "run.json", {"status": "succeeded", "pid": pid, "output_dir": str(paths.output), "runtime_seconds": 11.0, "state_policy": {"name": "always-commit"}})
    _write(paths.output / "state-timeline.json", {"policy": "always-commit", "pending_transaction_count": 0, "transactions": transactions})
    os.chmod(paths.output, 0o555)
    return paths


def test_accepts_complete_ordered_control_and_computes_protected_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    report = validator.validate_control(RUN_ID)
    assert report["status"] == "PASS"
    assert report["runtime_ratio"] == pytest.approx(1.1)
    assert all(report["protected_file_byte_equivalence"].values())
    assert set(report["terminal_artifacts"]) == {"preflight", "main", "postflight", "transcript", "driver", "result"}
    assert not paths.report.exists()


def test_rejects_protected_byte_drift_and_writes_one_frozen_failure_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    os.chmod(paths.output, 0o755)
    os.chmod(paths.output / "health.jsonl", 0o644)
    (paths.output / "health.jsonl").write_text("different\n", encoding="utf-8")
    os.chmod(paths.output / "health.jsonl", 0o444)
    os.chmod(paths.output, 0o555)
    with pytest.raises(validator.V9ValidationError, match="byte-identical"):
        validator.validate_control(RUN_ID)
    with pytest.raises(validator.V9ValidationError, match="byte-identical"):
        validator.main(["--run-id", RUN_ID])
    assert (paths.report.stat().st_mode & 0o777) == 0o444
    payload = json.loads(paths.report.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"


def test_rejects_missing_transcript_marker_and_candidate_timeline_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    os.chmod(paths.transcript, 0o644)
    paths.transcript.write_text("V9_DRIVER_DISPATCHED\n", encoding="utf-8")
    os.chmod(paths.transcript, 0o444)
    with pytest.raises(validator.V9ValidationError, match="transcript"):
        validator.validate_control(RUN_ID)

    paths = _fixture(tmp_path / "again", monkeypatch)
    os.chmod(paths.output, 0o755)
    timeline_path = paths.output / "state-timeline.json"
    os.chmod(timeline_path, 0o644)
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    timeline["transactions"][1]["early_spatial_pooled_pose_token"] = {"bad": True}
    timeline_path.write_text(json.dumps(timeline) + "\n", encoding="utf-8")
    os.chmod(timeline_path, 0o444)
    os.chmod(paths.output, 0o555)
    with pytest.raises(validator.V9ValidationError, match="candidate"):
        validator.validate_control(RUN_ID)
