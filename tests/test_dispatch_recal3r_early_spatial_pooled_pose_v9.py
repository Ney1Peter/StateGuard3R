from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import dispatch_recal3r_early_spatial_pooled_pose_v9 as dispatch


RUN_ID = "recovery-early-spatial-pooled-pose-v9-dynamic-always-commit-0001"


def _prepare_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "StateGuard3R"
    for name in ("outputs", "logs", "tmp"):
        (root / name).mkdir(parents=True, exist_ok=True)
    baseline = tmp_path / "baselines" / "ReCal3R"
    baseline.mkdir(parents=True)
    monkeypatch.setattr(dispatch, "ROOT", root)
    return root


def _argv(root: Path) -> list[str]:
    return [
        "python", "scripts/run_recal3r_early_spatial_pooled_pose_export_v9.py",
        "--output-dir", str(root / "outputs" / RUN_ID),
        "--state-policy", "always-commit", "--device", "cpu",
    ]


def test_fresh_id_refuses_every_terminal_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _prepare_root(tmp_path, monkeypatch)
    paths = dispatch.artifact_paths(RUN_ID)
    for path in (paths.output, paths.preflight, paths.main, paths.postflight, paths.transcript, paths.driver, paths.result):
        if path == paths.output:
            path.mkdir()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("prior\n", encoding="utf-8")
        with pytest.raises(dispatch.V9DispatchError, match="already exists"):
            dispatch._fresh(paths)
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink()
    assert root == dispatch.ROOT


def test_dispatch_orders_pipe_before_driver_and_freezes_original_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _prepare_root(tmp_path, monkeypatch)
    events: list[str] = []
    monkeypatch.setattr(dispatch, "_git_clean", lambda _path: "clean-commit")
    monkeypatch.setattr(dispatch, "_snapshot", lambda **_kwargs: {"selected": {"uuid": dispatch.GPU_UUID, "memory_free_mib": dispatch.MIN_FREE_MIB}, "rows": []})
    monkeypatch.setattr(dispatch, "_pane", lambda: "%42")
    monkeypatch.setattr(dispatch, "_pid_absent", lambda _pid: True)
    monkeypatch.setattr(dispatch, "_pinned_component_provenance", lambda: [{"path": "component", "sha256": "hash", "git_blob": "blob"}])

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        if command[:2] == ["tmux", "pipe-pane"]:
            events.append("pipe")
        elif command[:2] == ["tmux", "send-keys"]:
            paths = dispatch.artifact_paths(RUN_ID)
            assert events == ["pipe"]
            assert (paths.preflight.stat().st_mode & 0o777) == 0o444
            events.append("send")
            paths.main.write_text(
                "V9_DRIVER_START\nV9_DRIVER_DISPATCHED\nV9_FORWARD_DISPATCHED child_pid=731\nV9_DRIVER_EXIT\n",
                encoding="utf-8",
            )
            paths.transcript.write_text("V9_DRIVER_DISPATCHED\nV9_FORWARD_DISPATCHED child_pid=731\n", encoding="utf-8")
            paths.output.mkdir()
            (paths.output / "run.json").write_text("{}\n", encoding="utf-8")
            digest = hashlib_for(_argv(root))
            paths.result.write_text(json.dumps({"run_id": RUN_ID, "command_sha256": digest, "child_pid": 731, "exit_code": 0}) + "\n", encoding="utf-8")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(dispatch.subprocess, "run", fake_run)
    result = dispatch.dispatch(RUN_ID, _argv(root), timeout_seconds=1, allow_noncuda_child_for_test=True)
    assert result["status"] == "PASS"
    assert events[:2] == ["pipe", "send"]
    paths = dispatch.artifact_paths(RUN_ID)
    for path in (paths.preflight, paths.main, paths.postflight, paths.transcript, paths.driver, paths.result, paths.output / "run.json"):
        assert (path.stat().st_mode & 0o777) == 0o444
        assert b"\0" not in path.read_bytes()
    assert (paths.output.stat().st_mode & 0o777) == 0o555
    postflight = json.loads(paths.postflight.read_text(encoding="utf-8"))
    assert postflight["child_pid_absent"] is True


def hashlib_for(argv: list[str]) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(argv, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def test_dispatch_rejects_nul_terminal_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_root(tmp_path, monkeypatch)
    path = dispatch.artifact_paths(RUN_ID).main
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"bad\0")
    with pytest.raises(dispatch.V9DispatchError, match="NUL"):
        dispatch._freeze(path)
