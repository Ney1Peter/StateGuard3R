from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
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
    source = {"stateguard_commit": "commit", "recal3r_commit": "baseline", "component_sha256": {"x": "y"}}
    snapshot = {"timestamp": "2026-08-07T22:00:00+08:00", "selected": {"uuid": dispatch.GPU_UUID, "memory_free_mib": dispatch.MIN_FREE_MIB}, "rows": []}
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
