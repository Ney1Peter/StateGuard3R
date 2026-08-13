from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import threading
from types import SimpleNamespace

import pytest

from scripts import dispatch_recal3r_selective_state_memory_repair_v13 as dispatch


RUN_ID = "recovery-selective-state-memory-repair-v13-test-dispatch-0001"


def _root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "StateGuard3R"
    for name in ("outputs", "logs", "tmp"):
        (root / name).mkdir(parents=True)
    monkeypatch.setattr(dispatch, "ROOT", root)
    return root


def _argv(root: Path) -> list[str]:
    return [
        "python", str(root / dispatch.RUNNER_RELATIVE), "--output-dir", str(root / "outputs" / RUN_ID),
        "--state-policy", "always-commit", "--device", "cpu",
    ]


def _source() -> dict[str, object]:
    return {"stateguard_commit": "commit", "recal3r_commit": "baseline", "component_sha256": {}, "external_sha256": {}}


def _snapshot() -> dict[str, object]:
    return {
        "timestamp": "2026-08-13T22:00:00+08:00",
        "selected": {"uuid": dispatch.GPU_UUID, "memory_free_mib": dispatch.MIN_FREE_MIB},
        "rows": [], "project_gpu2_processes": [],
    }


def _result(argv: list[str]) -> dict[str, object]:
    return {
        "schema_version": "stateguard3r.v13-driver-result.v1", "run_id": RUN_ID,
        "command_sha256": dispatch._command_hash(argv), "child_pid": 731,
        "child_start_ticks": "123456", "exit_code": 0, "reaped_at": "2026-08-13T22:00:00+08:00",
    }


def _markers(argv: list[str]) -> str:
    result = _result(argv)
    return "\n".join((
        f"V13_DRIVER_START run_id={RUN_ID}",
        f"V13_DRIVER_COMMAND={shlex.join(argv)}",
        f"V13_DRIVER_COMMAND_SHA256={result['command_sha256']}",
        f"V13_DRIVER_DISPATCHED run_id={RUN_ID} kind=wrapper_only",
        f"V13_DRIVER_CHILD_PID run_id={RUN_ID} child_pid=731 child_start_ticks=123456",
        f"V13_DRIVER_PAYLOAD_RELEASE_ARMED run_id={RUN_ID} child_pid=731 child_start_ticks=123456",
        f"V13_DRIVER_PAYLOAD_RELEASED run_id={RUN_ID} child_pid=731 child_start_ticks=123456",
        f"V13_DRIVER_EXIT run_id={RUN_ID} child_pid=731 child_start_ticks=123456 exit_code=0 reaped_at=2026-08-13T22:00:00+08:00",
        f"V13_DRIVER_RESULT_WRITTEN run_id={RUN_ID} child_pid=731 child_start_ticks=123456 exit_code=0", "",
    ))


def test_v13_dispatcher_owns_only_the_preregistered_selective_state_route() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "dispatch_recal3r_selective_state_memory_repair_v13.py").read_text(encoding="utf-8")
    assert dispatch.CONTROL_RUN_ID == "recovery-selective-state-memory-repair-v13-dynamic-always-commit-0001"
    assert dispatch.CANDIDATE_RUN_ID == "recovery-selective-state-memory-repair-v13-dynamic-candidate-0001"
    assert dispatch.CANDIDATE_POLICY == "detector-v3-incremental-selective-state-memory-repair"
    assert dispatch.RUNNER_RELATIVE == Path("scripts/run_recal3r_selective_state_memory_repair_v13.py")
    assert Path("scripts/wait_and_dispatch_recal3r_selective_state_memory_repair_v13_control.sh") in dispatch.PINNED_COMPONENTS
    for forbidden in ("patch_embed_global", "encoder_global", "early_spatial", "current_pointmap", "pose_query", "anchor_export"):
        assert forbidden not in source


def test_v13_driver_releases_generic_payload_only_after_complete_identity_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path, monkeypatch)
    paths = dispatch.artifact_paths(RUN_ID)
    dispatch._acquire_lease(paths)
    dispatch._write(paths.main, "")
    payload, sentinel = tmp_path / "payload.sh", tmp_path / "executed"
    payload.write_text(f"#!/usr/bin/env bash\nprintf payload > {shlex.quote(str(sentinel))}\n", encoding="utf-8")
    payload.chmod(0o755)
    argv, token = [str(payload)], "b" * 64
    dispatch._write(paths.driver, dispatch._driver(RUN_ID, argv, paths, dispatch._command_hash(argv), shlex.join(argv), token), 0o555)
    completed = subprocess.run(["bash", str(paths.driver)], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert sentinel.read_text(encoding="utf-8") == "payload"
    assert dispatch._parse_result(paths.result, run_id=RUN_ID, command_hash=dispatch._command_hash(argv))["exit_code"] == 0


def test_v13_driver_rejects_authorized_token_prefix_with_extra_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _root(tmp_path, monkeypatch)
    paths = dispatch.artifact_paths(RUN_ID)
    dispatch._acquire_lease(paths); dispatch._write(paths.main, "")
    payload, sentinel = tmp_path / "payload.sh", tmp_path / "executed"
    payload.write_text(f"#!/usr/bin/env bash\nprintf bad > {shlex.quote(str(sentinel))}\n", encoding="utf-8")
    payload.chmod(0o755)
    argv, token = [str(payload)], "c" * 64
    dispatch._write(paths.driver, dispatch._driver(RUN_ID, argv, paths, dispatch._command_hash(argv), shlex.join(argv), token), 0o555)
    (paths.lease / "child-go").write_text(token + "\nunrecorded-suffix\n", encoding="utf-8")
    completed = subprocess.run(["bash", str(paths.driver)], text=True, capture_output=True)
    assert completed.returncode == 70
    assert not sentinel.exists() and not paths.result.exists()
    assert dispatch._preexec_failure(paths.main, run_id=RUN_ID) is not None


def test_v13_snapshot_and_candidate_predecessor_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _root(tmp_path, monkeypatch)
    monkeypatch.setattr(dispatch, "_project_gpu_processes", lambda: [])
    monkeypatch.setattr(dispatch.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout=f"{dispatch.GPU_UUID}, 1\n"))
    with pytest.raises(dispatch.V13DispatchError, match="free-memory"):
        dispatch._snapshot()
    assert dispatch._snapshot(require_minimum=False)["selected"]["memory_free_mib"] == 1
    with pytest.raises(dispatch.V13DispatchError, match="blocked"):
        dispatch._gate_b_predecessor(dispatch.CANDIDATE_RUN_ID)


def test_v13_lease_has_exactly_one_concurrent_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _root(tmp_path, monkeypatch)
    results: list[str] = []; lock = threading.Lock()

    def claim() -> None:
        try:
            dispatch._acquire_lease(dispatch.artifact_paths(RUN_ID)); value = "owner"
        except dispatch.V13DispatchError:
            value = "refused"
        with lock:
            results.append(value)

    first, second = threading.Thread(target=claim), threading.Thread(target=claim)
    first.start(); second.start(); first.join(); second.join()
    assert sorted(results) == ["owner", "refused"]


def test_v13_timeout_terminates_only_matching_pid_start_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    states = iter(("123456", None)); signals: list[tuple[int, int]] = []
    monkeypatch.setattr(dispatch, "_process_start_ticks", lambda _pid: next(states))
    monkeypatch.setattr(dispatch.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    dispatch._terminate_owned_child((731, "123456"), grace_seconds=0.01)
    assert signals == [(731, dispatch.signal.SIGTERM)]


def test_v13_candidate_validator_requires_complete_selective_repair_semantics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path, monkeypatch)
    paths = dispatch.artifact_paths(RUN_ID)
    monkeypatch.setattr(dispatch, "V1_DYNAMIC_CONTROL", root / "baseline")
    baseline = root / "baseline"
    baseline.mkdir()
    (baseline / "run.json").write_text(json.dumps({"runtime_seconds": 10.0}), encoding="utf-8")
    (baseline / "trajectory.json").write_text(json.dumps({"frames": [{"camera_to_reference": [index]} for index in range(30)]}), encoding="utf-8")
    paths.output.mkdir()
    (paths.output / "run.json").write_text(json.dumps({"status": "succeeded", "runtime_seconds": 11.0, "state_policy": {"name": dispatch.CANDIDATE_POLICY}}), encoding="utf-8")
    repaired = {
        "operator": "selective_state_memory_repair_v13", "repair_denominator": 8,
        "fallback_used": False, "no_fallback": True, "raw_camera_pose_numeric_input": False, "cuda_resident": True,
        "state_feat": {"shape": [1, 768, 768], "selected_row_count": 96, "selected_rows": list(range(96)), "tie_free_boundary": True, "selected_nonzero_delta_count": 96, "selected_rows_equal_pre": True, "unselected_rows_equal_proposed": True, "pre_gpu_digest": "gpu-fingerprint-v1:a", "proposed_gpu_digest": "gpu-fingerprint-v1:b", "committed_gpu_digest": "gpu-fingerprint-v1:c"},
        "mem": {"shape": [1, 256, 1536], "selected_row_count": 32, "selected_rows": list(range(32)), "tie_free_boundary": True, "selected_nonzero_delta_count": 32, "selected_rows_equal_pre": True, "unselected_rows_equal_proposed": True, "pre_gpu_digest": "gpu-fingerprint-v1:d", "proposed_gpu_digest": "gpu-fingerprint-v1:e", "committed_gpu_digest": "gpu-fingerprint-v1:f"},
    }
    timeline = []
    for frame in range(30):
        item: dict[str, object] = {"frame_id": frame, "action": "commit", "current_alarm": False, "pending_transaction_count": 0, "repair": None}
        if frame == 3:
            item.update({"action": "selective_state_memory_repair", "current_alarm": True, "repair": repaired, "online_detector": {"frame_id": 3, "hybrid_alarm": True}, "export_action": "export_real_camera_pose", "raw_pose_unchanged": True, "raw_candidate_pose_gpu_digest": "pose", "exported_camera_pose_gpu_digest": "pose", "selective_state_memory_repair": repaired})
        timeline.append(item)
    (paths.output / "state-timeline.json").write_text(json.dumps(timeline), encoding="utf-8")
    frames = [{"camera_to_reference": [index]} for index in range(30)]; frames[4] = {"camera_to_reference": [99]}
    (paths.output / "trajectory.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")
    result = dispatch._candidate_check(paths)
    assert result["status"] == "PASS" and result["repair_count"] == 1
    timeline[3]["repair"] = None
    (paths.output / "state-timeline.json").write_text(json.dumps(timeline), encoding="utf-8")
    with pytest.raises(dispatch.V13DispatchError, match="repair evidence"):
        dispatch._candidate_check(paths)


def test_v13_control_validator_requires_raw_identity_and_30_direct_commits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path, monkeypatch)
    paths = dispatch.artifact_paths(RUN_ID)
    baseline = root / "baseline"; baseline.mkdir()
    monkeypatch.setattr(dispatch, "V1_DYNAMIC_CONTROL", baseline)
    protected = {
        "checkpoint-load-audit.json": "{}\n", "health.jsonl": "health\n",
        "predictions-summary.json": "[]\n", "trajectory.json": "{}\n",
    }
    for name, value in protected.items():
        (baseline / name).write_text(value, encoding="utf-8")
    (baseline / "run.json").write_text(json.dumps({"runtime_seconds": 10.0}), encoding="utf-8")
    paths.output.mkdir()
    for name, value in protected.items():
        (paths.output / name).write_text(value, encoding="utf-8")
    (paths.output / "run.json").write_text(json.dumps({"status": "succeeded", "runtime_seconds": 11.0, "state_policy": {"name": "always-commit"}}), encoding="utf-8")
    timeline = [
        {"frame_id": frame, "action": "commit", "reason": "always_commit_control", "current_alarm": False,
         "pending_transaction_count": 0, "export_action": "export_real_camera_pose", "repair": None}
        for frame in range(30)
    ]
    (paths.output / "state-timeline.json").write_text(json.dumps(timeline), encoding="utf-8")
    assert dispatch._control_check(paths)["status"] == "PASS"
    timeline[-1]["repair"] = {"unexpected": True}
    (paths.output / "state-timeline.json").write_text(json.dumps(timeline), encoding="utf-8")
    with pytest.raises(dispatch.V13DispatchError, match="non-direct"):
        dispatch._control_check(paths)


def test_v13_production_contract_rejects_non_preregistered_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path, monkeypatch)
    paths = dispatch.artifact_paths(dispatch.CONTROL_RUN_ID)
    rgb = dispatch.RECAL3R_ROOT / "data" / "tum" / "rgbd_dataset_freiburg1_desk" / "rgb.txt"
    argv = [
        str(dispatch.RECAL3R_ROOT / ".venv" / "bin" / "python"), str(root / dispatch.RUNNER_RELATIVE),
        "--baseline-root", str(dispatch.RECAL3R_ROOT), "--checkpoint", str(dispatch.CHECKPOINT),
        "--checkpoint-sha256", dispatch.CHECKPOINT_SHA256,
        "--input-manifest", str(root / "outputs" / "formal-v1-inputs-0001" / "development" / "development-dynamic" / "input-manifest.json"),
        "--output-dir", str(paths.output), "--device", "cuda", "--size", "512", "--seed", "0",
        "--beta-base", "0.1", "--health-profile", "v3", "--rgb-timestamp-listing", str(rgb),
        "--timestamp-dataset-root", str(rgb.parent), "--state-policy", "always-commit",
        "--detector-config", str(root / "outputs" / "formal-v3-calibration-0001" / "formal-config.json"), "--watchdog", "8",
    ]
    monkeypatch.setattr(dispatch, "DETECTOR_CONFIG", root / "outputs" / "formal-v3-calibration-0001" / "formal-config.json")
    dispatch._production_contract(argv, paths)
    altered = list(argv); altered[altered.index("--state-policy") + 1] = dispatch.CANDIDATE_POLICY
    with pytest.raises(dispatch.V13DispatchError, match="differs"):
        dispatch._production_contract(altered, paths)
