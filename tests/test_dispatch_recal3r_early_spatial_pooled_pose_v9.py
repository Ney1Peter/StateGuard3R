from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import threading
from types import SimpleNamespace

import pytest

from scripts import dispatch_recal3r_early_spatial_pooled_pose_v9 as dispatch


RUN_ID = "recovery-early-spatial-pooled-pose-v9-dynamic-always-commit-0001"


def _root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "StateGuard3R"
    for name in ("outputs", "logs", "tmp"):
        (root / name).mkdir(parents=True)
    monkeypatch.setattr(dispatch, "ROOT", root)
    return root


def _argv(root: Path) -> list[str]:
    return [
        "python", str(root / "scripts" / "run_recal3r_early_spatial_pooled_pose_export_v9.py"),
        "--output-dir", str(root / "outputs" / RUN_ID),
        "--state-policy", "always-commit", "--device", "cpu",
    ]


def _result(run_id: str, argv: list[str]) -> dict[str, object]:
    return {
        "schema_version": "stateguard3r.v9-driver-result.v1",
        "run_id": run_id,
        "command_sha256": dispatch._command_hash(argv),
        "child_pid": 731, "child_start_ticks": "123456",
        "exit_code": 0, "reaped_at": "2026-08-07T22:00:00+08:00",
    }


def _markers(run_id: str, argv: list[str]) -> str:
    result = _result(run_id, argv)
    return "\n".join((
        f"V9_DRIVER_START run_id={run_id}",
        f"V9_DRIVER_COMMAND={shlex.join(argv)}",
        f"V9_DRIVER_COMMAND_SHA256={result['command_sha256']}",
        f"V9_DRIVER_DISPATCHED run_id={run_id}",
        f"V9_DRIVER_CHILD_PID run_id={run_id} child_pid=731 child_start_ticks=123456",
        f"V9_DRIVER_EXIT run_id={run_id} child_pid=731 child_start_ticks=123456 exit_code=0 reaped_at=2026-08-07T22:00:00+08:00",
        f"V9_DRIVER_RESULT_WRITTEN run_id={run_id} child_pid=731 child_start_ticks=123456 exit_code=0",
        "",
    ))


def test_canonical_paths_refuse_preexisting_artifact_and_atomic_lease(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path, monkeypatch)
    paths = dispatch.artifact_paths(RUN_ID)
    dispatch._fresh(paths)
    dispatch._acquire_lease(paths)
    with pytest.raises(dispatch.V9DispatchError, match="lease"):
        dispatch._acquire_lease(paths)
    assert paths.lease.is_dir()
    (paths.main).write_text("prior\n", encoding="utf-8")
    with pytest.raises(dispatch.V9DispatchError, match="already exists"):
        dispatch._fresh(paths)
    assert root == dispatch.ROOT


def test_driver_is_direct_child_writer_without_process_substitution_tees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path, monkeypatch)
    paths, argv = dispatch.artifact_paths(RUN_ID), _argv(root)
    source = dispatch._driver(RUN_ID, argv, paths, dispatch._command_hash(argv), shlex.join(argv))
    assert '>> "$main_log" 2>&1 &' in source
    assert "> >(" not in source
    assert "V9_DRIVER_DISPATCHED" in source
    assert "V9_DRIVER_CHILD_PID" in source
    assert "V9_DRIVER_RESULT_WRITTEN" in source


def test_dispatch_orders_pipe_before_driver_and_seals_original_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path, monkeypatch)
    paths, argv, events = dispatch.artifact_paths(RUN_ID), _argv(root), []
    # This simulation exercises the generic dispatcher protocol.  The separate
    # Gate-B control validator has its own exact-output fixtures below.
    monkeypatch.setattr(dispatch, "CONTROL_RUN_ID", "recovery-early-spatial-pooled-pose-v9-synthetic-0001")
    source = {"stateguard_commit": "commit", "recal3r_commit": "baseline", "component_sha256": {"x": "y"}}
    snapshot = {"timestamp": "2026-08-07T22:00:00+08:00", "selected": {"uuid": dispatch.GPU_UUID, "memory_free_mib": dispatch.MIN_FREE_MIB}, "rows": [], "project_gpu2_processes": []}
    monkeypatch.setattr(dispatch, "_component_provenance", lambda: source)
    monkeypatch.setattr(dispatch, "_snapshot", lambda **_kwargs: snapshot)
    monkeypatch.setattr(dispatch, "_pane", lambda: "%42")
    monkeypatch.setattr(dispatch, "_pane_alive", lambda _pane: False)
    monkeypatch.setattr(dispatch, "_pid_pair_absent", lambda _result: {"child_pid": 731, "expected_start_ticks": "123456", "observed_start_ticks": None, "pair_absent": True, "pid_reused": False})
    monkeypatch.setattr(dispatch, "_run_independent_validator", lambda _run_id: {"status": "PASS", "cpu_only": True})

    def fake_tmux(args: list[str], **_kwargs: object) -> SimpleNamespace:
        if args[0] == "pipe-pane" and "-o" in args:
            events.append("pipe")
        elif args[0] == "send-keys":
            command = args[3]
            if "V9_PIPE_READY" in command:
                assert events == ["pipe"]
                events.append("ready")
                paths.transcript.write_text(f"V9_PIPE_READY run_id={RUN_ID}\n", encoding="utf-8")
            else:
                assert events == ["pipe", "ready"]
                events.append("driver")
                marker_text = _markers(RUN_ID, argv)
                paths.main.write_text(marker_text, encoding="utf-8")
                paths.transcript.write_text(paths.transcript.read_text(encoding="utf-8") + marker_text, encoding="utf-8")
                paths.result.write_text(json.dumps(_result(RUN_ID, argv)) + "\n", encoding="utf-8")
                paths.output.mkdir()
                (paths.output / "run.json").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(dispatch, "_tmux", fake_tmux)
    result = dispatch.dispatch(RUN_ID, argv, timeout_seconds=1, allow_noncuda_child_for_test=True)
    assert result["status"] == "PASS"
    assert events == ["pipe", "ready", "driver"]
    for path in (paths.preflight, paths.main, paths.postflight, paths.transcript, paths.driver, paths.result, paths.output / "run.json"):
        assert (path.stat().st_mode & 0o777) == 0o444
        assert b"\0" not in path.read_bytes()
    assert (paths.output.stat().st_mode & 0o777) == 0o555
    postflight = json.loads(paths.postflight.read_text(encoding="utf-8"))
    assert postflight["pipe_close_and_drain"]["drained"] is True
    assert postflight["pid_start_time_check"]["pair_absent"] is True
    assert postflight["output_inventory"][-1]["path"] == "run.json"
    validator = dispatch.validate_sealed_run(RUN_ID)
    assert validator["status"] == "PASS" and validator["cpu_only"] is True
    assert (paths.validator.stat().st_mode & 0o777) == 0o444
    with pytest.raises(dispatch.V9DispatchError, match="already exists"):
        dispatch.validate_sealed_run(RUN_ID)


def test_snapshot_requires_memory_only_in_preflight_not_postflight(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    def fake_run(_command: list[str], **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout=f"{dispatch.GPU_UUID}, 1\n")

    monkeypatch.setattr(dispatch.subprocess, "run", fake_run)
    with pytest.raises(dispatch.V9DispatchError, match="free-memory"):
        dispatch._snapshot()
    assert dispatch._snapshot(require_minimum=False)["selected"]["memory_free_mib"] == 1


def test_atomic_lease_allows_exactly_one_concurrent_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path, monkeypatch)
    outcomes: list[str] = []
    lock = threading.Lock()

    def claim() -> None:
        try:
            dispatch._acquire_lease(dispatch.artifact_paths(RUN_ID))
            value = "owner"
        except dispatch.V9DispatchError:
            value = "refused"
        with lock:
            outcomes.append(value)

    first, second = threading.Thread(target=claim), threading.Thread(target=claim)
    first.start(); second.start(); first.join(); second.join()
    assert sorted(outcomes) == ["owner", "refused"]
    assert (root / "tmp" / f"{RUN_ID}.owner-lease").is_dir()


def test_fresh_refuses_broken_symlink_and_child_rejects_wrapper_injection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path, monkeypatch)
    paths = dispatch.artifact_paths(RUN_ID)
    paths.main.symlink_to(paths.main.parent / "missing-prior-artifact")
    with pytest.raises(dispatch.V9DispatchError, match="already exists"):
        dispatch._fresh(paths)
    paths.main.unlink()
    injected = _argv(root)
    injected[1] = "-c"
    with pytest.raises(dispatch.V9DispatchError, match="exact pinned"):
        dispatch._child(injected, paths, test=True)


def test_nul_and_output_symlink_are_rejected_without_repair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path, monkeypatch)
    paths = dispatch.artifact_paths(RUN_ID)
    paths.main.write_bytes(b"marker\0corruption")
    with pytest.raises(dispatch.V9DispatchError, match="contains NUL"):
        dispatch._read_regular_nul_free(paths.main, label="test main")
    with pytest.raises(dispatch.V9DispatchError, match="NUL-containing"):
        dispatch._write(root / "logs" / "nul-write.log", "never\0write")
    paths.output.mkdir()
    (paths.output / "aliased-output").symlink_to(paths.main)
    with pytest.raises(dispatch.V9DispatchError, match="symlink"):
        dispatch._freeze_output(paths.output)
    assert (paths.output / "aliased-output").is_symlink()


def test_cli_exposes_no_manual_postflight_or_repair_operation() -> None:
    parser = dispatch._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--postflight"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--repair"])


def test_wait_for_result_retries_partial_json_until_the_exclusive_writer_finishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path, monkeypatch)
    paths, argv = dispatch.artifact_paths(RUN_ID), _argv(root)
    paths.result.write_text("{", encoding="utf-8")

    def finish() -> None:
        paths.result.write_text(json.dumps(_result(RUN_ID, argv)) + "\n", encoding="utf-8")

    timer = threading.Timer(0.08, finish)
    timer.start()
    try:
        observed = dispatch._wait_for_result(
            paths.result, run_id=RUN_ID, command_hash=dispatch._command_hash(argv),
            quoted=shlex.join(argv), deadline=dispatch.time.monotonic() + 1,
        )
    finally:
        timer.join()
    assert observed["child_start_ticks"] == "123456"


def test_timeout_terminates_only_matching_pid_start_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    states = iter(("123456", None))
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(dispatch, "_process_start_ticks", lambda _pid: next(states))
    monkeypatch.setattr(dispatch.os, "kill", lambda pid, signal: signals.append((pid, signal)))
    dispatch._terminate_owned_child((731, "123456"), grace_seconds=0.01)
    assert signals == [(731, dispatch.signal.SIGTERM)]


def test_marker_order_and_live_pane_are_rejected_before_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path, monkeypatch)
    paths, argv = dispatch.artifact_paths(RUN_ID), _argv(root)
    result = _result(RUN_ID, argv)
    unordered = (
        f"V9_DRIVER_RESULT_WRITTEN run_id={RUN_ID} child_pid=731 child_start_ticks=123456 exit_code=0\n"
        + _markers(RUN_ID, argv)
    )
    with pytest.raises(dispatch.V9DispatchError, match="ill ordered"):
        dispatch._ordered_stream(unordered.encode(), run_id=RUN_ID, quoted=shlex.join(argv), command_hash=dispatch._command_hash(argv), result=result, transcript=False)
    paths.transcript.write_text("ready\n", encoding="utf-8")
    paths.main.write_text("main\n", encoding="utf-8")
    monkeypatch.setattr(dispatch, "_pane_alive", lambda _pane: True)
    with pytest.raises(dispatch.V9DispatchError, match="still live"):
        dispatch._close_pipe_and_drain("%42", paths.transcript, paths.main)


def test_missing_pipe_ready_never_sends_driver_or_starts_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path, monkeypatch)
    paths, argv, events = dispatch.artifact_paths(RUN_ID), _argv(root), []
    source = {"stateguard_commit": "commit", "recal3r_commit": "baseline", "component_sha256": {}, "external_sha256": {}}
    snapshot = {"timestamp": "2026-08-07T22:00:00+08:00", "selected": {"uuid": dispatch.GPU_UUID, "memory_free_mib": dispatch.MIN_FREE_MIB}, "rows": [], "project_gpu2_processes": []}
    monkeypatch.setattr(dispatch, "_component_provenance", lambda: source)
    monkeypatch.setattr(dispatch, "_snapshot", lambda **_kwargs: snapshot)
    monkeypatch.setattr(dispatch, "_pane", lambda: "%42")
    monkeypatch.setattr(dispatch, "_pane_alive", lambda _pane: False)

    def fake_tmux(args: list[str], **_kwargs: object) -> SimpleNamespace:
        if args[0] == "pipe-pane":
            events.append("pipe")
        elif args[0] == "send-keys":
            events.append("ready-attempt" if "V9_PIPE_READY" in args[3] else "driver")
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(dispatch, "_tmux", fake_tmux)
    with pytest.raises(dispatch.V9DispatchError, match="V9_PIPE_READY"):
        dispatch.dispatch(RUN_ID, argv, timeout_seconds=0.1, allow_noncuda_child_for_test=True)
    assert events == ["pipe", "ready-attempt"]
    for path in (paths.preflight, paths.main, paths.transcript, paths.driver, paths.postflight):
        assert path.exists() and (path.stat().st_mode & 0o777) == 0o444
    assert not paths.result.exists()


def test_validator_output_inventory_is_read_only_and_rejects_mutable_or_symlinked_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path, monkeypatch)
    output = root / "outputs" / RUN_ID
    (output / "nested").mkdir(parents=True)
    artifact = output / "nested" / "evidence.json"
    artifact.write_text("{}\n", encoding="utf-8")
    dispatch._freeze_output(output)
    artifact.chmod(0o644)
    with pytest.raises(dispatch.V9DispatchError, match="mode"):
        dispatch._output_inventory(output, require_frozen=True)
    assert (artifact.stat().st_mode & 0o777) == 0o644  # validator did not repair it
    artifact.chmod(0o444)
    (output / "nested").chmod(0o755)
    with pytest.raises(dispatch.V9DispatchError, match="directory mode"):
        dispatch._output_inventory(output, require_frozen=True)


def test_postflight_snapshot_requires_no_project_process_but_not_12_gib() -> None:
    low_free = {
        "timestamp": "2026-08-07T22:00:00+08:00",
        "selected": {"uuid": dispatch.GPU_UUID, "memory_free_mib": 1},
        "project_gpu2_processes": [],
    }
    dispatch._validate_snapshot(low_free, require_minimum=False)
    low_free["project_gpu2_processes"] = [{"pid": 1}]
    with pytest.raises(dispatch.V9DispatchError, match="project-process"):
        dispatch._validate_snapshot(low_free, require_minimum=False)


def test_production_child_contract_is_exact_and_rejects_single_argument_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path, monkeypatch)
    recal3r = tmp_path / "ReCal3R"
    checkpoint = recal3r / "src" / "cut3r_512_dpt_4_64.pth"
    rgb = recal3r / "data" / "tum" / "rgbd_dataset_freiburg1_desk" / "rgb.txt"
    config = root / "outputs" / "formal-v3-calibration-0001" / "formal-config.json"
    manifest = root / "outputs" / "formal-v1-inputs-0001" / "development" / "development-dynamic" / "input-manifest.json"
    for path in (checkpoint, rgb, config, manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pinned\n", encoding="utf-8")
    monkeypatch.setattr(dispatch, "RECAL3R_ROOT", recal3r)
    monkeypatch.setattr(dispatch, "CHECKPOINT", checkpoint)
    monkeypatch.setattr(dispatch, "DETECTOR_CONFIG", config)
    monkeypatch.setattr(dispatch, "DEVELOPMENT_MANIFESTS", (manifest,))
    paths = dispatch.artifact_paths(RUN_ID)
    argv = [
        str(recal3r / ".venv" / "bin" / "python"), str(root / dispatch.RUNNER_RELATIVE),
        "--baseline-root", str(recal3r), "--checkpoint", str(checkpoint),
        "--checkpoint-sha256", dispatch.CHECKPOINT_SHA256, "--input-manifest", str(manifest),
        "--output-dir", str(paths.output), "--device", "cuda", "--size", "512", "--seed", "0",
        "--beta-base", "0.1", "--health-profile", "v3", "--rgb-timestamp-listing", str(rgb),
        "--timestamp-dataset-root", str(rgb.parent), "--state-policy", "always-commit",
        "--detector-config", str(config), "--watchdog", "8",
    ]
    assert dispatch._child(argv, paths, test=False) == argv
    drifted = list(argv)
    drifted[drifted.index("--watchdog") + 1] = "7"
    with pytest.raises(dispatch.V9DispatchError, match="scalar"):
        dispatch._child(drifted, paths, test=False)
    candidate_paths = dispatch.artifact_paths(dispatch.CANDIDATE_RUN_ID)
    candidate = list(argv)
    candidate[candidate.index("--output-dir") + 1] = str(candidate_paths.output)
    candidate[candidate.index("--state-policy") + 1] = "detector-v3-incremental-early-spatial-pooled-pose-export"
    with pytest.raises(dispatch.V9DispatchError, match="blocked"):
        dispatch._child(candidate, candidate_paths, test=False)
    with pytest.raises(dispatch.V9DispatchError, match="wrong/low"):
        dispatch._require_gate_b_predecessor("recovery-early-spatial-pooled-pose-v9-wrong-always-commit-0001")


def test_dynamic_control_validator_rejects_output_difference_bad_timeline_and_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path, monkeypatch)
    baseline, output = root / "outputs" / "v1", root / "outputs" / RUN_ID
    baseline.mkdir(); output.mkdir()
    protected = ("checkpoint-load-audit.json", "health.jsonl", "predictions-summary.json", "trajectory.json")
    for filename in protected:
        (baseline / filename).write_text(f"{filename}\n", encoding="utf-8")
        (output / filename).write_text(f"{filename}\n", encoding="utf-8")
    (baseline / "run.json").write_text(json.dumps({"runtime_seconds": 2.0}), encoding="utf-8")
    (output / "run.json").write_text(json.dumps({"status": "succeeded", "state_policy": {"name": "always-commit"}, "runtime_seconds": 2.4}), encoding="utf-8")
    transactions = [{"frame_id": index, "action": "commit", "reason": "always_commit_control", "current_alarm": False, "consecutive_rollbacks": 0, "export_action": "export_real_camera_pose", "pending_transaction_count": 0, "restore_witness": None, "anchor_frame_ids": None} for index in range(30)]
    (output / "state-timeline.json").write_text(json.dumps({"policy": "always-commit", "pending_transaction_count": 0, "transactions": transactions}), encoding="utf-8")
    monkeypatch.setattr(dispatch, "V1_DYNAMIC_CONTROL", baseline)
    paths = dispatch.artifact_paths(RUN_ID)
    report = dispatch._validate_dynamic_always_control(paths)
    assert report["runtime_ratio"] == pytest.approx(1.2)
    (output / "health.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(dispatch.V9DispatchError, match="protected outputs differ"):
        dispatch._validate_dynamic_always_control(paths)
    (output / "health.jsonl").write_text("health.jsonl\n", encoding="utf-8")
    bad_pending = [dict(transaction) for transaction in transactions]
    bad_pending[0]["pending_transaction_count"] = 1
    (output / "state-timeline.json").write_text(json.dumps({"policy": "always-commit", "pending_transaction_count": 1, "transactions": bad_pending}), encoding="utf-8")
    with pytest.raises(dispatch.V9DispatchError, match="timeline"):
        dispatch._validate_dynamic_always_control(paths)
    (output / "state-timeline.json").write_text(json.dumps({"policy": "always-commit", "pending_transaction_count": 0, "transactions": transactions}), encoding="utf-8")
    (output / "run.json").write_text(json.dumps({"status": "succeeded", "state_policy": {"name": "always-commit"}, "runtime_seconds": 2.40001}), encoding="utf-8")
    with pytest.raises(dispatch.V9DispatchError, match="runtime ratio"):
        dispatch._validate_dynamic_always_control(paths)


def test_large_external_hash_uses_the_expected_digest(tmp_path: Path) -> None:
    artifact = tmp_path / "checkpoint-like.bin"
    artifact.write_bytes((b"0123456789abcdef" * 1024 * 64) + b"tail")
    assert dispatch._sha256_file(artifact) == hashlib.sha256(artifact.read_bytes()).hexdigest()
