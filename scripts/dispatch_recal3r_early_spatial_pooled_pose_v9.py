#!/usr/bin/env python3
"""Single-owner tmux dispatcher for v9's one-use GPU forwards.

No command can add a postflight, repair a transcript, or reuse an ID.  This is
intentional: the v8 failure showed that a numerically successful run without
an original ordered evidence chain is not a scientific result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SESSION, GPU_INDEX = "stateguard", "2"
GPU_UUID, MIN_FREE_MIB = "GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250", 12288
V9_COMPONENTS = (
    ROOT / "scripts" / "dispatch_recal3r_early_spatial_pooled_pose_v9.py",
    ROOT / "scripts" / "run_recal3r_early_spatial_pooled_pose_export_v9.py",
    ROOT / "src" / "stateguard3r" / "recal3r_early_spatial_pooled_pose_runner_v9.py",
    ROOT / "src" / "stateguard3r" / "early_spatial_pooled_pose_export_v9.py",
    ROOT / "src" / "stateguard3r" / "early_spatial_pooled_pose_v9.py",
)


class V9DispatchError(RuntimeError):
    """A v9 forward has no valid single-owner dispatch path."""


@dataclass(frozen=True)
class ArtifactPaths:
    output: Path
    preflight: Path
    main: Path
    postflight: Path
    transcript: Path
    driver: Path
    result: Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--command-json", type=Path, required=True, help="regular JSON array of the pinned child argv")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--allow-noncuda-child-for-test", action="store_true")
    return parser


def _run_id(run_id: str) -> None:
    valid = set("abcdefghijklmnopqrstuvwxyz0123456789-")
    if not run_id.startswith("recovery-early-spatial-pooled-pose-v9-") or not run_id or any(char not in valid for char in run_id):
        raise V9DispatchError("run id is not a canonical v9 one-use id")


def artifact_paths(run_id: str) -> ArtifactPaths:
    _run_id(run_id)
    logs, tmp = ROOT / "logs", ROOT / "tmp"
    return ArtifactPaths(
        ROOT / "outputs" / run_id, logs / f"{run_id}-preflight.json",
        logs / f"{run_id}-main.log", logs / f"{run_id}-postflight.json",
        logs / f"{run_id}-tmux-transcript.log", tmp / f"{run_id}-driver.sh",
        logs / f"{run_id}-result.json",
    )


def _write(path: Path, data: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)


def _write_json(path: Path, payload: Any) -> None:
    _write(path, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _freeze(path: Path) -> None:
    if not path.is_file() or path.is_symlink() or b"\0" in path.read_bytes():
        raise V9DispatchError(f"artifact is missing, nonregular, or NUL-containing: {path}")
    os.chmod(path, 0o444)


def _freeze_output(path: Path) -> None:
    if not path.is_dir() or path.is_symlink():
        raise V9DispatchError("child did not produce its direct output directory")
    for child in sorted(path.rglob("*")):
        if child.is_symlink() or (child.is_file() and b"\0" in child.read_bytes()):
            raise V9DispatchError("output has symlink or NUL-containing artifact")
        if child.is_file():
            os.chmod(child, 0o444)
    os.chmod(path, 0o555)


def _fresh(paths: ArtifactPaths) -> None:
    if paths.output.parent != ROOT / "outputs" or any(path.exists() for path in asdict(paths).values()):
        raise V9DispatchError("a one-use output or terminal artifact path already exists")
    if not (ROOT / "logs").is_dir() or not (ROOT / "tmp").is_dir():
        raise V9DispatchError("logs/tmp parent is unavailable")


def _git_clean(path: Path) -> str:
    try:
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], check=True, text=True, capture_output=True).stdout
        if dirty.strip():
            raise V9DispatchError(f"tracked worktree is not clean: {path}")
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise V9DispatchError(f"cannot inspect worktree: {path}") from error


def _snapshot(*, require_minimum: bool = True) -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 2:
            rows.append({"uuid": values[0], "memory_free_mib": int(values[1])})
    selected = [row for row in rows if row["uuid"] == GPU_UUID]
    if len(selected) != 1 or (require_minimum and selected[0]["memory_free_mib"] < MIN_FREE_MIB):
        raise V9DispatchError("GPU 2 UUID/free-memory precondition failed")
    return {"timestamp": datetime.now().astimezone().isoformat(), "command": command, "rows": rows, "selected": selected[0]}


def _child(argv: Sequence[str], *, test: bool) -> list[str]:
    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise V9DispatchError("child argv is not a nonempty string array")
    if "run_recal3r_early_spatial_pooled_pose_export_v9.py" not in " ".join(argv) or "--output-dir" not in argv or "--state-policy" not in argv:
        raise V9DispatchError("child is not the pinned v9 runner")
    if not test and ("--device" not in argv or argv[argv.index("--device") + 1] != "cuda"):
        raise V9DispatchError("production dispatch requires a CUDA child")
    output = Path(argv[argv.index("--output-dir") + 1]).resolve(strict=False)
    if output.parent != ROOT / "outputs":
        raise V9DispatchError("child output is not a direct outputs child")
    return list(argv)


def _driver(run_id: str, argv: Sequence[str], paths: ArtifactPaths, command_hash: str) -> str:
    command = " ".join(shlex.quote(item) for item in argv)
    return "\n".join((
        "#!/usr/bin/env bash", "set -u",
        f"main={shlex.quote(str(paths.main))}", f"result={shlex.quote(str(paths.result))}",
        f"echo 'V9_DRIVER_START run_id={run_id}' | tee -a \"$main\"",
        f"echo 'V9_DRIVER_COMMAND_SHA256 {command_hash}' | tee -a \"$main\"",
        f"echo 'V9_DRIVER_DISPATCHED run_id={run_id}' | tee -a \"$main\"",
        f"CUDA_VISIBLE_DEVICES={GPU_INDEX} {command} >>\"$main\" 2>&1 &",
        "child=$!",
        f"echo \"V9_FORWARD_DISPATCHED run_id={run_id} command_sha256={command_hash} child_pid=$child\" | tee -a \"$main\"",
        "wait \"$child\"; code=$?",
        f"echo \"V9_DRIVER_EXIT run_id={run_id} child_pid=$child exit_code=$code\" | tee -a \"$main\"",
        f"printf '%s\\n' \"{{\\\"run_id\\\":\\\"{run_id}\\\",\\\"command_sha256\\\":\\\"{command_hash}\\\",\\\"child_pid\\\":$child,\\\"exit_code\\\":$code}}\" > \"$result\"",
        "exit 0", "",
    ))


def _pane() -> str:
    try:
        subprocess.run(["tmux", "has-session", "-t", SESSION], check=True, capture_output=True)
        return subprocess.run(["tmux", "new-window", "-d", "-P", "-F", "#{pane_id}", "-t", SESSION], check=True, text=True, capture_output=True).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise V9DispatchError("existing stateguard tmux session is unavailable") from error


def _pid_absent(pid: int) -> bool:
    return not subprocess.run(["ps", "-p", str(pid), "-o", "pid="], text=True, capture_output=True).stdout.strip()


def _pinned_component_provenance() -> list[dict[str, str]]:
    """Bind the owner and child code to their exact committed blobs."""
    result: list[dict[str, str]] = []
    for path in V9_COMPONENTS:
        try:
            relative = path.resolve(strict=True).relative_to(ROOT)
            subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", str(relative)], check=True, capture_output=True)
            committed = subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"HEAD:{relative}"], check=True, text=True, capture_output=True).stdout.strip()
            current = subprocess.run(["git", "-C", str(ROOT), "hash-object", "--", str(relative)], check=True, text=True, capture_output=True).stdout.strip()
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            raise V9DispatchError(f"v9 component is not a tracked committed file: {path}") from error
        if not committed or committed != current:
            raise V9DispatchError(f"v9 component differs from its committed blob: {path}")
        result.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "git_blob": committed})
    return result


def dispatch(run_id: str, argv: Sequence[str], *, timeout_seconds: int, allow_noncuda_child_for_test: bool = False) -> dict[str, Any]:
    """Do the full ordered forward/terminal-evidence protocol exactly once."""
    paths = artifact_paths(run_id)
    _fresh(paths)
    child = _child(argv, test=allow_noncuda_child_for_test)
    if Path(child[child.index("--output-dir") + 1]).resolve(strict=False) != paths.output:
        raise V9DispatchError("run id and child output directory differ")
    state_commit, baseline_commit = _git_clean(ROOT), _git_clean(ROOT.parent / "baselines" / "ReCal3R")
    components = _pinned_component_provenance()
    snapshots = [_snapshot(), _snapshot()]
    command_hash = hashlib.sha256(json.dumps(child, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    _write_json(paths.preflight, {"schema_version": "stateguard3r.v9-preflight.v1", "run_id": run_id, "state_commit": state_commit, "baseline_commit": baseline_commit, "components": components, "command": child, "command_sha256": command_hash, "gpu_snapshots": snapshots})
    _freeze(paths.preflight)
    _write(paths.driver, _driver(run_id, child, paths, command_hash), 0o555)
    _write(paths.transcript, "")
    pane = _pane()
    pipe_open = False
    try:
        subprocess.run(["tmux", "pipe-pane", "-o", "-t", pane, f"cat >> {shlex.quote(str(paths.transcript))}"], check=True)
        pipe_open = True
        subprocess.run(["tmux", "send-keys", "-t", pane, f"bash {shlex.quote(str(paths.driver))}; exit", "C-m"], check=True)
        end = time.monotonic() + timeout_seconds
        while not paths.result.exists() and time.monotonic() < end:
            time.sleep(0.25)
        if not paths.result.exists():
            raise V9DispatchError("child did not publish result before timeout")
        subprocess.run(["tmux", "pipe-pane", "-t", pane], check=True)
        pipe_open = False
        result = json.loads(paths.result.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or result.get("run_id") != run_id or result.get("command_sha256") != command_hash or type(result.get("child_pid")) is not int or type(result.get("exit_code")) is not int:
            raise V9DispatchError("child result is malformed")
        postflight = {"schema_version": "stateguard3r.v9-postflight.v1", "run_id": run_id, "result": result, "gpu_snapshot": _snapshot(require_minimum=False), "child_pid_absent": _pid_absent(result["child_pid"]), "state_commit": _git_clean(ROOT), "baseline_commit": _git_clean(ROOT.parent / "baselines" / "ReCal3R")}
        if not postflight["child_pid_absent"]:
            raise V9DispatchError("recorded CUDA child is still live at postflight")
        _write_json(paths.postflight, postflight)
        if result["exit_code"] != 0:
            raise V9DispatchError(f"v9 child exited {result['exit_code']}")
        _freeze_output(paths.output)
        for path in (paths.main, paths.postflight, paths.transcript, paths.driver, paths.result):
            _freeze(path)
        return {"status": "PASS", "run_id": run_id, "pane": pane, "command_sha256": command_hash, "paths": {name: str(path) for name, path in asdict(paths).items()}}
    finally:
        if pipe_open:
            subprocess.run(["tmux", "pipe-pane", "-t", pane], check=False, capture_output=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.command_json.is_file() or args.command_json.is_symlink() or args.command_json.parent != ROOT / "tmp":
        raise V9DispatchError("command JSON must be a regular direct tmp child")
    payload = json.loads(args.command_json.read_text(encoding="utf-8"))
    print(json.dumps(dispatch(args.run_id, payload, timeout_seconds=args.timeout_seconds, allow_noncuda_child_for_test=args.allow_noncuda_child_for_test), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
