from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from scripts import dispatch_recal3r_patch_embed_global_pooled_pose_v12 as dispatch


RUN_ID = "recovery-patch-embed-global-pooled-pose-v12-test-dispatch-0001"


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
        "timestamp": "2026-08-07T22:00:00+08:00",
        "selected": {"uuid": dispatch.GPU_UUID, "memory_free_mib": dispatch.MIN_FREE_MIB},
        "rows": [], "project_gpu2_processes": [],
    }


def _result(argv: list[str]) -> dict[str, object]:
    return {
        "schema_version": "stateguard3r.v12-driver-result.v1", "run_id": RUN_ID,
        "command_sha256": dispatch._command_hash(argv), "child_pid": 731,
        "child_start_ticks": "123456", "exit_code": 0,
        "reaped_at": "2026-08-07T22:00:00+08:00",
    }


def _markers(argv: list[str]) -> str:
    result = _result(argv)
    return "\n".join((
        f"V12_DRIVER_START run_id={RUN_ID}",
        f"V12_DRIVER_COMMAND={shlex.join(argv)}",
        f"V12_DRIVER_COMMAND_SHA256={result['command_sha256']}",
        f"V12_DRIVER_DISPATCHED run_id={RUN_ID} kind=wrapper_only",
        "V12_DRIVER_CHILD_PID run_id=" + RUN_ID + " child_pid=731 child_start_ticks=123456",
        "V12_DRIVER_PAYLOAD_RELEASE_ARMED run_id=" + RUN_ID + " child_pid=731 child_start_ticks=123456",
        "V12_DRIVER_PAYLOAD_RELEASED run_id=" + RUN_ID + " child_pid=731 child_start_ticks=123456",
        "V12_DRIVER_EXIT run_id=" + RUN_ID + " child_pid=731 child_start_ticks=123456 exit_code=0 reaped_at=2026-08-07T22:00:00+08:00",
        "V12_DRIVER_RESULT_WRITTEN run_id=" + RUN_ID + " child_pid=731 child_start_ticks=123456 exit_code=0",
        "",
    ))


def _preexec_markers(argv: list[str]) -> str:
    return "\n".join((
        f"V12_DRIVER_START run_id={RUN_ID}", f"V12_DRIVER_COMMAND={shlex.join(argv)}",
        f"V12_DRIVER_COMMAND_SHA256={dispatch._command_hash(argv)}",
        f"V12_DRIVER_DISPATCHED run_id={RUN_ID} kind=wrapper_only",
        f"V12_DRIVER_PREEXEC_IDENTITY_FAILURE run_id={RUN_ID} child_pid=731 reason=start_time_unavailable exit_code=72", "",
    ))


def test_atomic_lease_paths_and_symlinked_outputs_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path, monkeypatch)
    paths = dispatch.artifact_paths(RUN_ID)
    dispatch._fresh(paths)
    dispatch._acquire_lease(paths)
    with pytest.raises(dispatch.V12DispatchError, match="lease"):
        dispatch._acquire_lease(paths)
    paths.main.write_text("prior\n", encoding="utf-8")
    with pytest.raises(dispatch.V12DispatchError, match="already exists"):
        dispatch._fresh(paths)
    paths.output.mkdir()
    (paths.output / "aliased").symlink_to(root / "logs" / "outside")
    with pytest.raises(dispatch.V12DispatchError, match="symlink"):
        dispatch._freeze_output(paths.output)


def test_driver_releases_generic_payload_only_after_complete_identity_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path, monkeypatch)
    paths = dispatch.artifact_paths(RUN_ID)
    dispatch._acquire_lease(paths)
    dispatch._write(paths.main, "")
    payload, sentinel = tmp_path / "payload.sh", tmp_path / "executed"
    payload.write_text(f"#!/usr/bin/env bash\nprintf payload > {shlex.quote(str(sentinel))}\n", encoding="utf-8")
    payload.chmod(0o755)
    token, argv = "b" * 64, [str(payload)]
    dispatch._write(paths.driver, dispatch._driver(RUN_ID, argv, paths, dispatch._command_hash(argv), shlex.join(argv), token), 0o555)
    completed = subprocess.run(["bash", str(paths.driver)], text=True, capture_output=True, timeout=5)
    assert completed.returncode == 0, completed.stderr
    assert sentinel.read_text(encoding="utf-8") == "payload"
    assert dispatch._parse_result(paths.result, run_id=RUN_ID, command_hash=dispatch._command_hash(argv))["exit_code"] == 0
    assert (paths.lease / "child-go").read_text(encoding="utf-8") == token + "\n"


def test_driver_rejects_authorized_token_prefix_with_extra_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _root(tmp_path, monkeypatch)
    paths = dispatch.artifact_paths(RUN_ID)
    dispatch._acquire_lease(paths)
    dispatch._write(paths.main, "")
    payload, sentinel = tmp_path / "payload.sh", tmp_path / "executed"
    payload.write_text(f"#!/usr/bin/env bash\nprintf bad > {shlex.quote(str(sentinel))}\n", encoding="utf-8")
    payload.chmod(0o755)
    token, argv = "c" * 64, [str(payload)]
    dispatch._write(paths.driver, dispatch._driver(RUN_ID, argv, paths, dispatch._command_hash(argv), shlex.join(argv), token), 0o555)
    (paths.lease / "child-go").write_text(token + "\nunrecorded-suffix\n", encoding="utf-8")
    completed = subprocess.run(["bash", str(paths.driver)], text=True, capture_output=True, timeout=5)
    # The wrapper may already have exited with its safe pre-execution status
    # before the parent rechecks start ticks.  Either way the exact-byte gate
    # prevented payload execution and no result may be created.
    assert completed.returncode == 70
    assert not sentinel.exists() and not paths.result.exists()
    assert dispatch._preexec_failure(paths.main, run_id=RUN_ID) is not None


def test_dispatch_orders_pipe_ready_before_driver_freezes_then_cpu_validates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path, monkeypatch)
    paths, argv, events = dispatch.artifact_paths(RUN_ID), _argv(root), []
    monkeypatch.setattr(dispatch, "_component_provenance", _source)
    monkeypatch.setattr(dispatch, "_snapshot", lambda **_kwargs: _snapshot())
    monkeypatch.setattr(dispatch, "_pane", lambda: "%42")
    monkeypatch.setattr(dispatch, "_pane_alive", lambda _pane: False)
    monkeypatch.setattr(dispatch, "_pid_pair_absent", lambda _result: {"expected_start_ticks": "123456", "pair_absent": True, "pid_reused": False})

    def tmux(args: list[str], **_kwargs: object) -> SimpleNamespace:
        if args[0] == "pipe-pane":
            events.append("pipe")
        elif args[0] == "send-keys":
            command = args[3]
            if "V12_PIPE_READY" in command:
                events.append("ready")
                paths.transcript.write_text(f"V12_PIPE_READY run_id={RUN_ID}\n", encoding="utf-8")
            else:
                assert events == ["pipe", "ready"]
                events.append("driver")
                stream = _markers(argv)
                paths.main.write_text(stream, encoding="utf-8")
                paths.transcript.write_text(paths.transcript.read_text(encoding="utf-8") + stream, encoding="utf-8")
                paths.result.write_text(json.dumps(_result(argv)) + "\n", encoding="utf-8")
                paths.output.mkdir()
                (paths.output / "evidence.json").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(dispatch, "_tmux", tmux)
    monkeypatch.setattr(dispatch, "_run_independent_validator", lambda value: {"status": "PASS", "run_id": value})
    result = dispatch.dispatch(RUN_ID, argv, timeout_seconds=1, allow_noncuda_child_for_test=True)
    assert result["status"] == "PASS" and events == ["pipe", "ready", "driver"]
    for path in (paths.preflight, paths.main, paths.postflight, paths.transcript, paths.driver, paths.result, paths.output / "evidence.json"):
        assert stat_mode(path) == 0o444
    assert stat_mode(paths.output) == 0o555


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_validator_accepts_exact_frozen_no_result_preexec_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path, monkeypatch)
    paths, argv = dispatch.artifact_paths(RUN_ID), _argv(root)
    source, snapshot = _source(), _snapshot()
    monkeypatch.setattr(dispatch, "_component_provenance", lambda: source)
    dispatch._acquire_lease(paths)
    quoted, command_hash, token = shlex.join(argv), dispatch._command_hash(argv), "d" * 64
    dispatch._write_json(paths.preflight, {"schema_version": "stateguard3r.v12-preflight.v1", "run_id": RUN_ID, "command": argv, "quoted_command": quoted, "command_sha256": command_hash, "test_child": True, "interpreter": None, "gpu_preflights": [snapshot, snapshot], "source_provenance": source})
    evidence = _preexec_markers(argv)
    dispatch._write(paths.main, evidence)
    dispatch._write(paths.transcript, f"V12_PIPE_READY run_id={RUN_ID}\n" + evidence)
    dispatch._write(paths.driver, dispatch._driver(RUN_ID, argv, paths, command_hash, quoted, token), 0o555)
    failure = dispatch._preexec_failure(paths.main, run_id=RUN_ID)
    assert failure is not None
    dispatch._write_json(paths.postflight, {"schema_version": "stateguard3r.v12-postflight.v1", "run_id": RUN_ID, "result": None, "source_provenance": source, "pipe_close_and_drain": {"drained": True}, "error": "PREEXEC_IDENTITY_NO_GO", "preexec_no_go": {"terminal_kind": "PREEXEC_IDENTITY_NO_GO", "payload_not_executed": True, "failure": failure, "driver_result_exists": False, "output_kind": "missing", "go_gate_authorized": False, "go_token_sha256": hashlib.sha256((token + "\n").encode("ascii")).hexdigest()}})
    for path in (paths.preflight, paths.main, paths.transcript, paths.driver, paths.postflight):
        dispatch._freeze(path)
    report = dispatch.validate_sealed_run(RUN_ID)
    assert report["status"] == "PREEXEC_IDENTITY_NO_GO_EVIDENCE_COMPLETE"
    assert stat_mode(paths.validator) == 0o444
    with pytest.raises(dispatch.V12DispatchError, match="fresh validator"):
        dispatch.validate_sealed_run(RUN_ID)


def test_snapshot_requires_memory_and_project_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dispatch, "_project_gpu_processes", lambda: [])
    monkeypatch.setattr(dispatch.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout=f"{dispatch.GPU_UUID}, 1\n"))
    with pytest.raises(dispatch.V12DispatchError, match="free-memory"):
        dispatch._snapshot()
    assert dispatch._snapshot(require_minimum=False)["selected"]["memory_free_mib"] == 1
    monkeypatch.setattr(dispatch, "_project_gpu_processes", lambda: [{"pid": 1}])
    with pytest.raises(dispatch.V12DispatchError, match="project process"):
        dispatch._snapshot(require_minimum=False)


def test_lease_has_exactly_one_concurrent_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _root(tmp_path, monkeypatch)
    results: list[str] = []
    lock = threading.Lock()

    def claim() -> None:
        try:
            dispatch._acquire_lease(dispatch.artifact_paths(RUN_ID))
            value = "owner"
        except dispatch.V12DispatchError:
            value = "refused"
        with lock:
            results.append(value)

    first, second = threading.Thread(target=claim), threading.Thread(target=claim)
    first.start(); second.start(); first.join(); second.join()
    assert sorted(results) == ["owner", "refused"]


def test_candidate_is_blocked_without_frozen_passing_control(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _root(tmp_path, monkeypatch)
    with pytest.raises(dispatch.V12DispatchError, match="blocked"):
        dispatch._gate_b_predecessor(dispatch.CANDIDATE_RUN_ID)


def test_timeout_terminates_only_matching_pid_start_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    states = iter(("123456", None))
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(dispatch, "_process_start_ticks", lambda _pid: next(states))
    monkeypatch.setattr(dispatch.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    dispatch._terminate_owned_child((731, "123456"), grace_seconds=0.01)
    assert signals == [(731, dispatch.signal.SIGTERM)]


def test_cli_has_no_repair_or_postflight_surface() -> None:
    parser = dispatch._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--repair"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--postflight"])


def test_v12_dispatcher_owns_only_the_preregistered_patch_embed_route() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "dispatch_recal3r_patch_embed_global_pooled_pose_v12.py").read_text(encoding="utf-8")
    assert dispatch.CONTROL_RUN_ID == "recovery-patch-embed-global-pooled-pose-v12-dynamic-always-commit-0001"
    assert dispatch.CANDIDATE_RUN_ID == "recovery-patch-embed-global-pooled-pose-v12-dynamic-candidate-0001"
    assert dispatch.CANDIDATE_POLICY == "detector-v3-incremental-patch-embed-global-pooled-pose-export"
    assert dispatch.RUNNER_RELATIVE == Path("scripts/run_recal3r_patch_embed_global_pooled_pose_export_v12.py")
    assert Path("scripts/wait_and_dispatch_recal3r_patch_embed_global_pooled_pose_v12_control.sh") in dispatch.PINNED_COMPONENTS
    assert "dispatch_recal3r_encoder_global_pooled_pose_v11" not in source
    assert "dispatch_recal3r_early_spatial_pooled_pose_v9" not in source
    assert "importlib" not in source


def test_v12_production_contract_rejects_non_preregistered_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path, monkeypatch)
    paths = dispatch.artifact_paths(dispatch.CONTROL_RUN_ID)
    rgb = dispatch.RECAL3R_ROOT / "data" / "tum" / "rgbd_dataset_freiburg1_desk" / "rgb.txt"
    argv = [
        str(dispatch.RECAL3R_ROOT / ".venv" / "bin" / "python"), str(root / dispatch.RUNNER_RELATIVE),
        "--baseline-root", str(dispatch.RECAL3R_ROOT),
        "--checkpoint", str(dispatch.CHECKPOINT),
        "--checkpoint-sha256", dispatch.CHECKPOINT_SHA256,
        "--input-manifest", str(root / "outputs" / "formal-v1-inputs-0001" / "development" / "development-dynamic" / "input-manifest.json"),
        "--output-dir", str(paths.output),
        "--device", "cuda",
        "--size", "512",
        "--seed", "0",
        "--beta-base", "0.1",
        "--health-profile", "v3",
        "--rgb-timestamp-listing", str(rgb),
        "--timestamp-dataset-root", str(rgb.parent),
        "--state-policy", "always-commit",
        "--detector-config", str(root / "outputs" / "formal-v3-calibration-0001" / "formal-config.json"),
        "--watchdog", "8",
    ]
    # Rebase the fixed project paths into this isolated non-CUDA test root.
    monkeypatch.setattr(dispatch, "DETECTOR_CONFIG", root / "outputs" / "formal-v3-calibration-0001" / "formal-config.json")
    monkeypatch.setattr(dispatch, "V1_DYNAMIC_CONTROL", root / "outputs" / "recovery-policy-development-v1-dynamic-always-commit-0001")
    dispatch._production_contract(argv, paths)
    altered = list(argv)
    altered[altered.index("--state-policy") + 1] = dispatch.CANDIDATE_POLICY
    with pytest.raises(dispatch.V12DispatchError, match="differs"):
        dispatch._production_contract(altered, paths)
