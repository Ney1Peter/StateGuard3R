from __future__ import annotations

import ast
import json
from pathlib import Path
import shlex
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from scripts import dispatch_recal3r_bounded_update_pressure_v16 as dispatch


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = dispatch.CONTROL


def _sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "StateGuard3R"
    for name in ("outputs", "logs", "tmp"):
        (root / name).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(dispatch, "ROOT", root)
    return root


def _fake_snapshot() -> dict[str, object]:
    return {
        "at": "2026-08-14T00:00:00+08:00",
        "command": ["nvidia-smi"], "rows": [],
        "selected": {"uuid": dispatch.GPU_UUID, "memory_free_mib": dispatch.MIN_FREE_MIB},
        "project_gpu2_processes": [],
    }


def test_v16_dispatcher_is_own_protocol_and_pins_split_capabilities() -> None:
    script = ROOT / "scripts" / "dispatch_recal3r_bounded_update_pressure_v16.py"
    source = script.read_text(encoding="utf-8")
    modules: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    assert not any("v15" in name or "v14" in name for name in modules)
    assert dispatch.CONTROL == "recovery-update-pressure-v16-dynamic-always-commit-0001"
    assert dispatch.CANDIDATE == "recovery-update-pressure-v16-dynamic-candidate-0001"
    expected = {
        "src/stateguard3r/dynamic_rgb_capability_v16.py",
        "src/stateguard3r/frame_zero_probe_capability_v16.py",
        "scripts/build_recal3r_dynamic_rgb_capability_v16.py",
        "scripts/build_recal3r_frame_zero_probe_capability_v16.py",
        "scripts/probe_recal3r_bounded_update_pressure_interface_v16.py",
        "src/stateguard3r/bounded_update_pressure_v16.py",
        "src/stateguard3r/recal3r_bounded_update_pressure_runner_v16.py",
        "scripts/run_recal3r_bounded_update_pressure_v16.py",
        "scripts/dispatch_recal3r_bounded_update_pressure_v16.py",
        "scripts/wait_and_dispatch_recal3r_bounded_update_pressure_v16_control.sh",
    }
    component_names = {str(value.relative_to(ROOT)) for value in dispatch.COMPONENTS}
    assert expected <= component_names
    for marker in ("V16_PIPE_READY", "secrets.token_hex(32)", "child_start_ticks", "_pid_absent", "_freeze_output", "_fresh_validator", "CUDA_VISIBLE_DEVICES"):
        assert marker in source


def test_v16_driver_runs_fake_child_only_after_complete_identity_and_token_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _sandbox(tmp_path, monkeypatch)
    item = dispatch.paths(RUN_ID)
    dispatch._acquire(item)
    dispatch._write_new(item.main, "")
    marker = tmp_path / "fake-child-ran"
    payload = tmp_path / "fake-child.sh"
    payload.write_text(f"#!/usr/bin/env bash\nprintf ran > {shlex.quote(str(marker))}\n", encoding="utf-8")
    payload.chmod(0o755)
    argv, token = [str(payload)], "a" * 64
    command_hash = dispatch._command_hash(argv)
    dispatch._write_new(item.driver, dispatch._driver(RUN_ID, item, argv, command_hash, token), 0o555)
    complete = subprocess.run(["bash", str(item.driver)], text=True, capture_output=True)
    assert complete.returncode == 0, complete.stderr
    assert marker.read_text(encoding="utf-8") == "ran"
    assert dispatch._result(item.result, run_id=RUN_ID, command_hash=command_hash)["exit_code"] == 0
    source = item.driver.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=2" in source and "cmp -s" in source


def test_v16_driver_records_nonzero_fake_child_without_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _sandbox(tmp_path, monkeypatch)
    item = dispatch.paths(RUN_ID)
    dispatch._acquire(item)
    dispatch._write_new(item.main, "")
    argv, token = ["bash", "-c", "exit 23"], "b" * 64
    command_hash = dispatch._command_hash(argv)
    dispatch._write_new(item.driver, dispatch._driver(RUN_ID, item, argv, command_hash, token), 0o555)
    complete = subprocess.run(["bash", str(item.driver)], text=True, capture_output=True)
    assert complete.returncode == 0, complete.stderr
    assert dispatch._result(item.result, run_id=RUN_ID, command_hash=command_hash)["exit_code"] == 23
    assert item.main.read_text(encoding="utf-8").count("V16_DRIVER_PAYLOAD_RELEASED") == 1


def test_v16_driver_rejects_bad_preexisting_go_token_before_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _sandbox(tmp_path, monkeypatch)
    item = dispatch.paths(RUN_ID)
    dispatch._acquire(item)
    dispatch._write_new(item.main, "")
    marker = tmp_path / "must-not-run"
    payload = tmp_path / "fake-child.sh"
    payload.write_text(f"#!/usr/bin/env bash\nprintf bad > {shlex.quote(str(marker))}\n", encoding="utf-8")
    payload.chmod(0o755)
    argv, token = [str(payload)], "c" * 64
    dispatch._write_new(item.driver, dispatch._driver(RUN_ID, item, argv, dispatch._command_hash(argv), token), 0o555)
    (item.lease / "child-go").write_text(token + "\nextra\n", encoding="utf-8")
    complete = subprocess.run(["bash", str(item.driver)], text=True, capture_output=True)
    assert complete.returncode == 70
    assert not marker.exists() and not item.result.exists()
    assert dispatch._preexec(item, RUN_ID) is not None


def test_v16_one_use_lease_has_exactly_one_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _sandbox(tmp_path, monkeypatch)
    observed: list[str] = []
    lock = threading.Lock()

    def claim() -> None:
        try:
            dispatch._acquire(dispatch.paths(RUN_ID))
            value = "owner"
        except dispatch.DispatchV16Error:
            value = "refused"
        with lock:
            observed.append(value)

    left, right = threading.Thread(target=claim), threading.Thread(target=claim)
    left.start(); right.start(); left.join(); right.join()
    assert sorted(observed) == ["owner", "refused"]


def test_v16_timeout_signals_only_matching_pid_start_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    states = iter(("123456", None))
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(dispatch, "_process_start_ticks", lambda _pid: next(states))
    monkeypatch.setattr(dispatch.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    dispatch._terminate_owned((731, "123456"), grace_seconds=0.01)
    assert signals == [(731, dispatch.signal.SIGTERM)]
    monkeypatch.setattr(dispatch, "_process_start_ticks", lambda _pid: "reused")
    with pytest.raises(dispatch.DispatchV16Error, match="cannot prove"):
        dispatch._terminate_owned((731, "123456"), grace_seconds=0.01)


def test_v16_timeout_aborts_if_pid_is_reused_after_safe_term(monkeypatch: pytest.MonkeyPatch) -> None:
    states = iter(("123456", "reused"))
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(dispatch, "_process_start_ticks", lambda _pid: next(states))
    monkeypatch.setattr(dispatch.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(dispatch.DispatchV16Error, match="was reused"):
        dispatch._terminate_owned((731, "123456"), grace_seconds=0.01)
    assert signals == [(731, dispatch.signal.SIGTERM)]


def test_v16_missing_preexec_pid_and_pipe_ready_failure_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _sandbox(tmp_path, monkeypatch)
    item = dispatch.paths(RUN_ID)
    dispatch._write_new(item.main, "")
    dispatch._write_new(item.transcript, "")
    with pytest.raises(dispatch.DispatchV16Error, match="did not publish"):
        dispatch._wait_result(item, RUN_ID, "a" * 64, time.monotonic())
    with pytest.raises(dispatch.DispatchV16Error, match="timed out waiting"):
        dispatch._wait_text(item.transcript, f"V16_PIPE_READY run_id={RUN_ID}", time.monotonic())


def test_v16_early_dispatch_failure_seals_exclusive_journal_and_terminal_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _sandbox(tmp_path, monkeypatch)
    monkeypatch.setattr(dispatch, "_source", lambda: (_ for _ in ()).throw(dispatch.DispatchV16Error("synthetic early source failure")))
    with pytest.raises(dispatch.DispatchV16Error, match="synthetic early source failure"):
        dispatch.dispatch(RUN_ID, timeout_seconds=1)
    journal, terminal = root / "logs" / f"{RUN_ID}-dispatch-journal.jsonl", root / "logs" / f"{RUN_ID}-validator.json"
    assert journal.stat().st_mode & 0o777 == 0o444
    assert terminal.stat().st_mode & 0o777 == 0o444
    rows = [json.loads(row) for row in journal.read_text(encoding="utf-8").splitlines()]
    assert [row["stage"] for row in rows] == ["LEASE_AND_JOURNAL_CREATED", "SOURCE_AND_GATE_B_CHECK", "EXCEPTION", "TERMINAL"]
    assert json.loads(terminal.read_text(encoding="utf-8"))["status"] == "FAIL"


def test_v16_dispatcher_exception_after_source_is_terminal_without_taking_tmux_or_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _sandbox(tmp_path, monkeypatch)
    source = {"stateguard_commit": "test", "recal3r_commit": dispatch.RECAL3R_COMMIT, "component_sha256": {}, "external_sha256": {}}
    monkeypatch.setattr(dispatch, "_source", lambda: source)
    monkeypatch.setattr(dispatch, "_production_contract", lambda *_args: None)
    monkeypatch.setattr(dispatch, "_snapshot", lambda **_kwargs: (_ for _ in ()).throw(dispatch.DispatchV16Error("synthetic snapshot failure")))
    monkeypatch.setattr(dispatch, "_pane", lambda: (_ for _ in ()).throw(AssertionError("tmux must not start")))
    with pytest.raises(dispatch.DispatchV16Error, match="synthetic snapshot failure"):
        dispatch.dispatch(RUN_ID, timeout_seconds=1)
    journal = root / "logs" / f"{RUN_ID}-dispatch-journal.jsonl"
    rows = [json.loads(row) for row in journal.read_text(encoding="utf-8").splitlines()]
    assert [row["stage"] for row in rows] == [
        "LEASE_AND_JOURNAL_CREATED", "SOURCE_AND_GATE_B_CHECK", "GPU_PREFLIGHT_FIRST", "EXCEPTION", "TERMINAL",
    ]
    assert (root / "logs" / f"{RUN_ID}-validator.json").stat().st_mode & 0o777 == 0o444


def test_v16_pipe_setup_failure_is_terminal_before_the_driver_can_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _sandbox(tmp_path, monkeypatch)
    source = {"stateguard_commit": "test", "recal3r_commit": dispatch.RECAL3R_COMMIT, "component_sha256": {}, "external_sha256": {}}
    monkeypatch.setattr(dispatch, "_source", lambda: source)
    monkeypatch.setattr(dispatch, "_production_contract", lambda *_args: None)
    monkeypatch.setattr(dispatch, "_snapshot", lambda **_kwargs: _fake_snapshot())
    monkeypatch.setattr(dispatch, "_pane", lambda: "%fake")

    def failed_pipe(arguments: object, **_kwargs: object) -> SimpleNamespace:
        if list(arguments)[0] == "pipe-pane":
            raise subprocess.CalledProcessError(1, ["tmux", "pipe-pane"])
        raise AssertionError("only pipe-pane is allowed in this fake")

    monkeypatch.setattr(dispatch, "_tmux", failed_pipe)
    with pytest.raises(dispatch.DispatchV16Error):
        dispatch.dispatch(RUN_ID, timeout_seconds=1)
    journal = root / "logs" / f"{RUN_ID}-dispatch-journal.jsonl"
    rows = [json.loads(row) for row in journal.read_text(encoding="utf-8").splitlines()]
    # A failed pipe setup is terminal before the driver can start.  The
    # dispatcher can still retain a postflight snapshot; it must not invent a
    # secondary postflight-write failure merely to satisfy this branch.
    assert [row["stage"] for row in rows] == [
        "LEASE_AND_JOURNAL_CREATED", "SOURCE_AND_GATE_B_CHECK", "GPU_PREFLIGHT_FIRST", "GPU_PREFLIGHT_SECOND", "TMUX_PANE_CREATED", "EXCEPTION", "TERMINAL",
    ]
    assert (root / "logs" / f"{RUN_ID}-postflight.json").is_file()
    assert "V16_DRIVER_START" in (root / "logs" / f"{RUN_ID}-driver.sh").read_text(encoding="utf-8")
    assert "V16_DRIVER_PAYLOAD_RELEASED" not in (root / "logs" / f"{RUN_ID}-main.log").read_text(encoding="utf-8")
    assert (root / "logs" / f"{RUN_ID}-validator.json").stat().st_mode & 0o777 == 0o444


def test_v16_source_status_allows_only_exact_old_terminal_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    allowed = "\n".join(f"?? {name}" for name in sorted(dispatch.PREEXISTING_TERMINAL_UNTRACKED))
    monkeypatch.setattr(dispatch, "_git", lambda _repo, *_args: allowed)
    dispatch._v16_relevant_clean()
    monkeypatch.setattr(dispatch, "_git", lambda _repo, *_args: "?? scripts/new-v16.py")
    with pytest.raises(dispatch.DispatchV16Error, match="relevant dirty"):
        dispatch._v16_relevant_clean()
    monkeypatch.setattr(dispatch, "_git", lambda _repo, *_args: " M scripts/dispatch_recal3r_bounded_update_pressure_v16.py")
    with pytest.raises(dispatch.DispatchV16Error, match="relevant dirty"):
        dispatch._v16_relevant_clean()


def test_v16_candidate_cannot_bypass_the_frozen_control_predecessor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _sandbox(tmp_path, monkeypatch)
    with pytest.raises(dispatch.DispatchV16Error, match="requires frozen PASS dynamic control"):
        dispatch._control_predecessor(dispatch.CANDIDATE)


def test_v16_pipe_drain_inventory_and_cuda_hidden_validator_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _sandbox(tmp_path, monkeypatch)
    item = dispatch.paths(RUN_ID)
    dispatch._write_new(item.main, "main\n")
    dispatch._write_new(item.transcript, "V16_PIPE_READY run_id=" + RUN_ID + "\n")
    monkeypatch.setattr(dispatch, "_pane_alive", lambda _pane: False)
    assert dispatch._drain("%fake", item)["drained"] is True
    monkeypatch.setattr(dispatch, "_pane_alive", lambda _pane: True)
    with pytest.raises(dispatch.DispatchV16Error, match="pane is live"):
        dispatch._drain("%fake", item)
    monkeypatch.setattr(dispatch, "_pane_alive", lambda _pane: False)
    output = item.output
    (output / "nested").mkdir(parents=True)
    (output / "nested" / "out.json").write_text("{}\n", encoding="utf-8")
    inventory = dispatch._freeze_output(output)
    assert inventory[-1]["path"] == "nested/out.json"
    assert (output / "nested" / "out.json").stat().st_mode & 0o777 == 0o444
    captured: dict[str, object] = {}

    def fake_run(_argv: object, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout='{"status":"PASS"}', stderr="")

    monkeypatch.setattr(dispatch.subprocess, "run", fake_run)
    assert dispatch._fresh_validator(RUN_ID) == {"status": "PASS"}
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == ""


def test_v16_control_validator_uses_hash_constants_not_historical_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _sandbox(tmp_path, monkeypatch)
    item = dispatch.paths(RUN_ID)
    item.output.mkdir()
    values = {name: f"{name}\n".encode("utf-8") for name in dispatch.PROTECTED}
    for name, value in values.items():
        (item.output / name).write_bytes(value)
    expected = {name: dispatch._sha(item.output / name) for name in dispatch.PROTECTED}
    monkeypatch.setattr(dispatch, "PREREGISTERED_PROTECTED_SHA256", expected)
    (item.output / "run.json").write_text(json.dumps({"status": "succeeded", "runtime_seconds": 6.0, "state_policy": {"name": "always-commit"}}), encoding="utf-8")
    timeline = {"transactions": [{"frame_id": index, "action": "commit", "reason": "always_commit_control", "current_alarm": False, "pending_transaction_count": 0, "pressure": None} for index in range(30)]}
    (item.output / "state-timeline.json").write_text(json.dumps(timeline), encoding="utf-8")
    assert dispatch._control_check(item)["status"] == "PASS"
    assert "recovery-policy-development" not in (ROOT / "scripts" / "dispatch_recal3r_bounded_update_pressure_v16.py").read_text(encoding="utf-8")
