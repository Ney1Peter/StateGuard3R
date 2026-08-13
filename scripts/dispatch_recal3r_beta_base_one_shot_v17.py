#!/usr/bin/env python3
"""One-use terminal-evidence dispatcher for v17's native beta-base experiment.

This program deliberately owns its protocol rather than delegating release,
validation, or child lifecycle management to an earlier experiment.  It does
not itself import a model.  CUDA is reachable only by a token-gated child in a
new pane of the existing ``stateguard`` tmux session.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shlex
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SESSION = "stateguard"
GPU_INDEX = "2"
GPU_UUID = "GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250"
MIN_FREE_MIB = 12288
RECAL3R_COMMIT = "466c7cdf3acd2f589f1d82e5f6391966f19db9ff"
CHECKPOINT_RELATIVE = Path("src/cut3r_512_dpt_4_64.pth")
CHECKPOINT_SHA256 = "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103"
CAPSULE_RELATIVE = Path("outputs/recovery-beta-floor-v17-dynamic-rgb-list-capsule-0001.json")
DETECTOR_CONFIG_RELATIVE = Path("outputs/formal-v3-calibration-0001/formal-config.json")
CONTROL = "recovery-beta-floor-v17-dynamic-always-commit-0001"
CANDIDATE = "recovery-beta-floor-v17-dynamic-candidate-0001"
POLICIES = {
    CONTROL: "always-commit",
    CANDIDATE: "detector-v3-one-shot-beta-base-floor",
}
PROTECTED = (
    "checkpoint-load-audit.json",
    "health.jsonl",
    "predictions-summary.json",
    "trajectory.json",
)
PREREGISTERED_PROTECTED_SHA256 = {
    "checkpoint-load-audit.json": "3e12875a701e0afc534d8f3a90b6a5fb54a58cfca59e2ce055109e6db4a4153d",
    "health.jsonl": "9de0c0b690c00e603fd16c81ff6258ce3c275388e48d66a0c1a64672848131fe",
    "predictions-summary.json": "cb11415bfe3765e7ad87fd9655873f319cdaaa553ef948f3e3f935037a8504ed",
    "trajectory.json": "fdba3312b792e4c67f56c8a0f8feba908846cd61190fa5a8665f0b0be81d013b",
}
REFERENCE_RUNTIME_SECONDS = 5.660642675124109
RUNTIME_RATIO_LIMIT = 1.20
RUNTIME_LIMIT_SECONDS = 6.79277121014893

# This is an explicit v17 inventory.  A release refuses any source file not
# committed at HEAD; paths are never discovered with a glob.
COMPONENT_RELATIVES = (
    Path("src/stateguard3r/beta_base_floor_v17.py"),
    Path("src/stateguard3r/recal3r_beta_base_floor_runner_v17.py"),
    Path("src/stateguard3r/online_detector_v17.py"),
    Path("src/stateguard3r/dynamic_rgb_capability_v17.py"),
    Path("src/stateguard3r/frame_zero_probe_capability_v17.py"),
    Path("src/stateguard3r/online_visual_overlap_v17.py"),
    Path("src/stateguard3r/timestamp_order_v17.py"),
    Path("src/stateguard3r/v17_source_import_audit.py"),
    Path("scripts/build_recal3r_dynamic_rgb_capability_v17.py"),
    Path("scripts/build_recal3r_frame_zero_probe_capability_v17.py"),
    Path("scripts/probe_recal3r_beta_base_floor_interface_v17.py"),
    Path("scripts/run_recal3r_beta_base_one_shot_v17.py"),
    Path("scripts/dispatch_recal3r_beta_base_one_shot_v17.py"),
    Path("scripts/wait_and_dispatch_recal3r_beta_base_one_shot_v17_control.sh"),
)
RUNNER_RELATIVE = Path("scripts/run_recal3r_beta_base_one_shot_v17.py")

# These terminal v14/v15 files pre-date v17 and must remain untouched.  The
# list is intentionally exact: it is not a version-prefix bypass.
PREEXISTING_TERMINAL_UNTRACKED = frozenset({
    "scripts/build_recal3r_dynamic_input_capsule_v14.py",
    "scripts/build_recal3r_rgb_listing_capsule_v15.py",
    "scripts/dispatch_recal3r_bounded_update_pressure_v15.py",
    "scripts/dispatch_recal3r_update_pressure_bounded_state_memory_v14.py",
    "scripts/probe_recal3r_bounded_update_pressure_interface_v15.py",
    "scripts/probe_recal3r_update_pressure_interface_v14.py",
    "scripts/run_recal3r_bounded_update_pressure_v15.py",
    "scripts/run_recal3r_update_pressure_bounded_state_memory_v14.py",
    "scripts/wait_and_dispatch_recal3r_bounded_update_pressure_v15_control.sh",
    "scripts/wait_and_dispatch_recal3r_update_pressure_bounded_state_memory_v14_control.sh",
    "src/stateguard3r/bounded_update_pressure_v15.py",
    "src/stateguard3r/recal3r_bounded_update_pressure_runner_v15.py",
    "src/stateguard3r/recal3r_update_pressure_bounded_state_memory_runner_v14.py",
    "src/stateguard3r/rgb_listing_capsule_v15.py",
    "src/stateguard3r/runtime_input_capsule_v14.py",
    "src/stateguard3r/update_pressure_bounded_state_memory_v14.py",
    "tests/test_bounded_update_pressure_v15.py",
    "tests/test_build_recal3r_dynamic_input_capsule_v14.py",
    "tests/test_dispatch_recal3r_update_pressure_bounded_state_memory_v14.py",
    "tests/test_probe_recal3r_bounded_update_pressure_interface_v15.py",
    "tests/test_probe_recal3r_update_pressure_interface_v14.py",
    "tests/test_recal3r_bounded_update_pressure_runner_v15.py",
    "tests/test_recal3r_update_pressure_bounded_state_memory_runner_v14.py",
    "tests/test_rgb_listing_capsule_v15.py",
    "tests/test_run_recal3r_bounded_update_pressure_v15.py",
    "tests/test_run_recal3r_update_pressure_bounded_state_memory_v14.py",
    "tests/test_runtime_input_capsule_v14.py",
    "tests/test_update_pressure_bounded_state_memory_v14.py",
})


class DispatchV17Error(RuntimeError):
    """Raised when a one-use v17 release lacks complete evidence."""


class _PreexecV17NoGo(DispatchV17Error):
    def __init__(self, evidence: Mapping[str, Any]) -> None:
        super().__init__("v17 child failed before payload release")
        self.evidence = evidence


@dataclass(frozen=True)
class ArtifactPaths:
    output: Path
    lease: Path
    journal: Path
    preflight: Path
    main: Path
    transcript: Path
    driver: Path
    result: Path
    postflight: Path
    validator: Path


def _recal3r_root() -> Path:
    return ROOT.parent / "baselines" / "ReCal3R"


def _runtime_python() -> Path:
    return _recal3r_root() / ".venv" / "bin" / "python"


def paths(run_id: str) -> ArtifactPaths:
    if run_id not in POLICIES:
        raise DispatchV17Error("unknown fixed v17 run id")
    return ArtifactPaths(
        output=ROOT / "outputs" / run_id,
        lease=ROOT / "tmp" / f"{run_id}.owner-lease",
        journal=ROOT / "logs" / f"{run_id}-dispatch-journal.jsonl",
        preflight=ROOT / "logs" / f"{run_id}-preflight.json",
        main=ROOT / "logs" / f"{run_id}-main.log",
        transcript=ROOT / "logs" / f"{run_id}-tmux-transcript.log",
        driver=ROOT / "logs" / f"{run_id}-driver.sh",
        result=ROOT / "logs" / f"{run_id}-result.json",
        postflight=ROOT / "logs" / f"{run_id}-postflight.json",
        validator=ROOT / "logs" / f"{run_id}-validator.json",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("--run-id", choices=tuple(POLICIES))
    command.add_argument("--validate-run-id", choices=tuple(POLICIES))
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    return parser


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _kind(path: Path) -> str:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return "missing"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISREG(mode):
        return "regular"
    if stat.S_ISDIR(mode):
        return "directory"
    return "special"


def _read(path: Path, *, label: str) -> bytes:
    if _kind(path) != "regular":
        raise DispatchV17Error(f"{label} is missing, nonregular, or a symlink")
    data = path.read_bytes()
    if b"\0" in data:
        raise DispatchV17Error(f"{label} contains NUL")
    return data


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, text: str, mode: int = 0o644) -> None:
    if "\0" in text or _lexists(path):
        raise DispatchV17Error("refusing overwrite or NUL in one-use evidence")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, mode)


def _write_json(path: Path, value: Any, mode: int = 0o644) -> None:
    _write_new(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", mode)


def _freeze(path: Path) -> None:
    _read(path, label="terminal artifact")
    os.chmod(path, 0o444)
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise DispatchV17Error("could not freeze terminal artifact")


def _json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(_read(path, label=label))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DispatchV17Error(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise DispatchV17Error(f"{label} is not an object")
    return value


def _canonical(item: ArtifactPaths) -> None:
    if item != paths(item.output.name):
        raise DispatchV17Error("artifact paths are not canonical")
    for parent in (ROOT / "outputs", ROOT / "logs", ROOT / "tmp"):
        if _kind(parent) != "directory":
            raise DispatchV17Error("an artifact parent is missing or unsafe")


def _journal(item: ArtifactPaths, stage: str, **values: Any) -> None:
    row = {"at": datetime.now().astimezone().isoformat(), "stage": stage, **values}
    with item.journal.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _acquire(item: ArtifactPaths) -> None:
    _canonical(item)
    if any(_lexists(value) for value in asdict(item).values()):
        raise DispatchV17Error("one-use artifact or owner lease already exists")
    try:
        item.lease.mkdir(mode=0o700)
    except FileExistsError as error:
        raise DispatchV17Error("same-id owner lease already exists") from error
    if _kind(item.lease) != "directory" or list(item.lease.iterdir()):
        raise DispatchV17Error("owner lease is unsafe")
    _write_new(item.journal, "")


def _git(repository: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repository), *args], check=True, text=True, capture_output=True).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise DispatchV17Error("cannot inspect source provenance") from error


def _git_bytes(repository: Path, *args: str) -> bytes:
    try:
        return subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as error:
        raise DispatchV17Error("cannot inspect source provenance bytes") from error


def _relevant_clean() -> None:
    status = _git(ROOT, "status", "--porcelain=v1", "--untracked-files=all")
    unexpected = [line for line in status.splitlines() if len(line) < 4 or line[:2] != "??" or line[3:] not in PREEXISTING_TERMINAL_UNTRACKED]
    if unexpected:
        raise DispatchV17Error(f"StateGuard3R has release-relevant dirty entries: {unexpected}")


def _regular_frozen(path: Path, *, label: str) -> None:
    if _kind(path) != "regular" or stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise DispatchV17Error(f"{label} must be a frozen regular 0444 file")


def _source() -> Mapping[str, Any]:
    _relevant_clean()
    recal3r = _recal3r_root()
    if _git(recal3r, "status", "--porcelain"):
        raise DispatchV17Error("ReCal3R worktree is dirty")
    if _git(recal3r, "rev-parse", "HEAD") != RECAL3R_COMMIT:
        raise DispatchV17Error("ReCal3R commit differs from pin")
    components: dict[str, str] = {}
    for relative in COMPONENT_RELATIVES:
        path = ROOT / relative
        if _kind(path) != "regular":
            raise DispatchV17Error(f"v17 component is unsafe: {relative}")
        _git(ROOT, "ls-files", "--error-unmatch", "--", str(relative))
        if _git_bytes(ROOT, "show", f"HEAD:{relative}") != path.read_bytes():
            raise DispatchV17Error(f"component differs from committed HEAD: {relative}")
        components[str(relative)] = _sha(path)
    capsule, detector, checkpoint = ROOT / CAPSULE_RELATIVE, ROOT / DETECTOR_CONFIG_RELATIVE, recal3r / CHECKPOINT_RELATIVE
    _regular_frozen(capsule, label="dynamic capsule")
    _regular_frozen(detector, label="frozen detector config")
    if _kind(checkpoint) != "regular" or _sha(checkpoint) != CHECKPOINT_SHA256:
        raise DispatchV17Error("checkpoint is missing or differs from pin")
    model = recal3r / "src" / "dust3r" / "model.py"
    if _kind(model) != "regular":
        raise DispatchV17Error("pinned ReCal3R model source is absent")
    return {
        "stateguard_commit": _git(ROOT, "rev-parse", "HEAD"),
        "recal3r_commit": RECAL3R_COMMIT,
        "component_sha256": components,
        "external_sha256": {str(path): _sha(path) for path in (capsule, detector, checkpoint, model)},
    }


def _project_processes() -> list[Mapping[str, Any]]:
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True)
    observed: list[Mapping[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 4 or fields[0] != GPU_UUID or not fields[1].isdigit():
            continue
        pid = int(fields[1])
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        except OSError:
            command = ""
        if str(ROOT) in command or str(_recal3r_root()) in command:
            observed.append({"pid": pid, "process_name": fields[2], "used_memory_mib": fields[3], "command": command})
    return observed


def _snapshot(*, require_minimum: bool = True) -> Mapping[str, Any]:
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True)
    rows = []
    selected: Mapping[str, Any] | None = None
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 3 or not fields[0].isdigit() or not fields[2].isdigit():
            raise DispatchV17Error("GPU snapshot contains malformed row")
        row = {"index": fields[0], "uuid": fields[1], "memory_free_mib": int(fields[2])}
        rows.append(row)
        if fields[0] == GPU_INDEX:
            selected = row
    if selected is None or selected["uuid"] != GPU_UUID:
        raise DispatchV17Error("configured GPU identity differs")
    processes = _project_processes()
    if require_minimum and (selected["memory_free_mib"] < MIN_FREE_MIB or processes):
        raise DispatchV17Error("GPU preflight lacks memory or has a Project2 process")
    return {"at": datetime.now().astimezone().isoformat(), "command": ["nvidia-smi", "--query-gpu=index,uuid,memory.free"], "rows": rows, "selected": selected, "project_gpu2_processes": processes}


def _command(run_id: str, item: ArtifactPaths) -> list[str]:
    return [
        str(_runtime_python()), str(ROOT / RUNNER_RELATIVE),
        "--capsule", str(ROOT / CAPSULE_RELATIVE),
        "--output-dir", str(item.output),
        "--device", "cuda", "--size", "512", "--seed", "0",
        "--beta-base", "0.1", "--state-policy", POLICIES[run_id],
        "--detector-config", str(ROOT / DETECTOR_CONFIG_RELATIVE),
    ]


def _command_hash(command: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(command), separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _control_predecessor(run_id: str) -> None:
    if run_id == CONTROL:
        return
    control = paths(CONTROL)
    if _kind(control.validator) != "regular" or stat.S_IMODE(control.validator.stat().st_mode) != 0o444:
        raise DispatchV17Error("candidate requires frozen PASS dynamic control")
    report = _json(control.validator, label="v17 control validator")
    if report.get("status") != "PASS" or report.get("dynamic_always_control", {}).get("status") != "PASS":
        raise DispatchV17Error("candidate is blocked by dynamic control")


def _production_contract(run_id: str, item: ArtifactPaths, command: Sequence[str]) -> None:
    if list(command) != _command(run_id, item):
        raise DispatchV17Error("child argv differs from fixed production contract")
    _control_predecessor(run_id)


def _driver(run_id: str, item: ArtifactPaths, command: Sequence[str], command_hash: str, token: str) -> str:
    if len(token) != 64 or any(char not in "0123456789abcdef" for char in token):
        raise DispatchV17Error("go token is not 256-bit lowercase hex")
    quoted = tuple(shlex.quote(value) for value in (run_id, str(item.main), str(item.result), str(item.lease), command_hash, token))
    run_q, main_q, result_q, lease_q, hash_q, token_q = quoted
    command_q = shlex.join(command)
    return f'''#!/usr/bin/env bash
set +e
run_id={run_q}
main_log={main_q}
result_path={result_q}
lease_path={lease_q}
command_sha256={hash_q}
go_token={token_q}
start_path="$lease_path/child-start"
go_path="$lease_path/child-go"
record() {{ printf '%s\\n' "$1" | tee -a "$main_log"; }}
record "V17_DRIVER_START run_id=$run_id"
record "V17_DRIVER_COMMAND_SHA256=$command_sha256"
record "V17_DRIVER_DISPATCHED run_id=$run_id kind=token-gated-wrapper"
(
  gate_pid="$BASHPID"
  gate_ticks="$(awk '{{print $22}}' "/proc/$gate_pid/stat" 2>/dev/null)"
  if [[ ! "$gate_ticks" =~ ^[0-9]+$ ]]; then
    (set -o noclobber; printf 'FAIL %s unavailable\\n' "$gate_pid" > "$start_path") 2>/dev/null
    exit 72
  fi
  if ! (set -o noclobber; printf 'READY %s %s\\n' "$gate_pid" "$gate_ticks" > "$start_path") 2>/dev/null; then exit 72; fi
  while true; do
    if [[ -e "$go_path" || -L "$go_path" ]]; then
      if [[ -L "$go_path" || ! -f "$go_path" ]] || ! printf '%s\\n' "$go_token" | cmp -s - "$go_path"; then exit 73; fi
      exec env CUDA_VISIBLE_DEVICES={GPU_INDEX} {command_q}
    fi
    sleep 0.01
  done
) >> "$main_log" 2>&1 &
child_pid=$!
deadline=$((SECONDS + 30))
while [[ ! -f "$start_path" && $SECONDS -lt $deadline ]]; do
  if ! kill -0 "$child_pid" 2>/dev/null; then break; fi
  sleep 0.01
done
if [[ ! -f "$start_path" ]]; then
  wait "$child_pid"; child_exit=$?
  record "V17_DRIVER_PREEXEC_IDENTITY_FAILURE run_id=$run_id child_pid=$child_pid reason=start_gate_missing exit_code=$child_exit"
  exit 70
fi
read -r kind reported_pid child_start_ticks < "$start_path"
if [[ "$kind" != READY || "$reported_pid" != "$child_pid" || ! "$child_start_ticks" =~ ^[0-9]+$ ]]; then
  wait "$child_pid"; child_exit=$?
  record "V17_DRIVER_PREEXEC_IDENTITY_FAILURE run_id=$run_id child_pid=$child_pid reason=start_time_unavailable exit_code=$child_exit"
  exit 70
fi
observed_ticks="$(awk '{{print $22}}' "/proc/$child_pid/stat" 2>/dev/null)"
if [[ "$observed_ticks" != "$child_start_ticks" ]]; then
  kill "$child_pid" 2>/dev/null; wait "$child_pid"; child_exit=$?
  record "V17_DRIVER_PREEXEC_IDENTITY_FAILURE run_id=$run_id child_pid=$child_pid reason=parent_start_time_mismatch exit_code=$child_exit"
  exit 70
fi
record "V17_DRIVER_CHILD_PID run_id=$run_id child_pid=$child_pid child_start_ticks=$child_start_ticks"
record "V17_DRIVER_PAYLOAD_RELEASE_ARMED run_id=$run_id child_pid=$child_pid child_start_ticks=$child_start_ticks"
if ! (set -o noclobber; printf '%s\\n' "$go_token" > "$go_path") 2>/dev/null; then
  kill "$child_pid" 2>/dev/null; wait "$child_pid"; child_exit=$?
  record "V17_DRIVER_PREEXEC_IDENTITY_FAILURE run_id=$run_id child_pid=$child_pid reason=go_gate_create_failed exit_code=$child_exit"
  exit 70
fi
record "V17_DRIVER_PAYLOAD_RELEASED run_id=$run_id child_pid=$child_pid child_start_ticks=$child_start_ticks"
wait "$child_pid"; child_exit=$?
reaped_at="$(date --iso-8601=seconds)"
record "V17_DRIVER_EXIT run_id=$run_id child_pid=$child_pid child_start_ticks=$child_start_ticks exit_code=$child_exit reaped_at=$reaped_at"
if ! (set -o noclobber; : > "$result_path") 2>/dev/null; then record "V17_DRIVER_RESULT_CREATE_FAILED run_id=$run_id"; exit 70; fi
printf '{{"schema_version":"stateguard3r.v17-driver-result.v1","run_id":"%s","command_sha256":"%s","child_pid":%s,"child_start_ticks":"%s","exit_code":%s,"reaped_at":"%s"}}\\n' "$run_id" "$command_sha256" "$child_pid" "$child_start_ticks" "$child_exit" "$reaped_at" >> "$result_path"
record "V17_DRIVER_RESULT_WRITTEN run_id=$run_id child_pid=$child_pid child_start_ticks=$child_start_ticks exit_code=$child_exit"
exit 0
'''


def _tmux(arguments: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["tmux", *arguments], check=check, text=True, capture_output=True)


def _pane() -> str:
    try:
        _tmux(["has-session", "-t", SESSION])
        pane = _tmux(["new-window", "-d", "-P", "-F", "#{pane_id}", "-t", SESSION, "-c", str(ROOT)]).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise DispatchV17Error("existing stateguard tmux session is unavailable") from error
    if not pane.startswith("%"):
        raise DispatchV17Error("tmux did not return a v17 pane")
    return pane


def _pane_alive(pane: str) -> bool:
    return _tmux(["list-panes", "-t", pane, "-F", "#{pane_id}"], check=False).returncode == 0


def _wait_text(path: Path, marker: str, deadline: float) -> None:
    while time.monotonic() < deadline:
        if _lexists(path) and marker in _read(path, label="tmux transcript").decode("utf-8", errors="strict"):
            return
        time.sleep(0.05)
    raise DispatchV17Error(f"timed out waiting for {marker}")


def _wait_pane_exit(pane: str, deadline: float) -> None:
    while time.monotonic() < deadline:
        if not _pane_alive(pane):
            return
        time.sleep(0.05)
    raise DispatchV17Error("driver pane did not exit")


def _drain(pane: str, item: ArtifactPaths) -> Mapping[str, Any]:
    if _pane_alive(pane):
        raise DispatchV17Error("pane is live; streams cannot freeze")
    before = (_read(item.transcript, label="tmux transcript"), _read(item.main, label="main log"))
    time.sleep(0.15)
    after = (_read(item.transcript, label="tmux transcript"), _read(item.main, label="main log"))
    if before != after:
        raise DispatchV17Error("pipe did not drain before sealing")
    return {"drained": True, "pane_alive_at_close": False, "transcript_sha256": hashlib.sha256(after[0]).hexdigest(), "main_sha256": hashlib.sha256(after[1]).hexdigest()}


def _process_start_ticks(pid: int) -> str | None:
    try:
        tail = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return tail[19] if len(tail) > 19 and tail[19].isdigit() else None
    except (FileNotFoundError, OSError, IndexError):
        return None


def _pid_absent(result: Mapping[str, Any]) -> Mapping[str, Any]:
    observed = _process_start_ticks(result["child_pid"])
    return {"child_pid": result["child_pid"], "expected_start_ticks": result["child_start_ticks"], "observed_start_ticks": observed, "pair_absent": observed is None, "pid_reused": observed is not None and observed != result["child_start_ticks"]}


def _partial_identity(item: ArtifactPaths, run_id: str) -> tuple[int, str] | None:
    try:
        rows = _read(item.main, label="main log").decode("utf-8", errors="strict").splitlines()
    except (UnicodeDecodeError, DispatchV17Error):
        return None
    prefix = f"V17_DRIVER_CHILD_PID run_id={run_id} "
    matches = [row for row in rows if row.startswith(prefix)]
    if len(matches) != 1:
        return None
    fields = dict(part.split("=", 1) for part in matches[0].split() if "=" in part)
    return (int(fields["child_pid"]), fields["child_start_ticks"]) if fields.get("child_pid", "").isdigit() and fields.get("child_start_ticks", "").isdigit() else None


def _terminate_owned(identity: tuple[int, str], *, grace_seconds: float = 5.0) -> None:
    pid, expected = identity
    if _process_start_ticks(pid) != expected:
        raise DispatchV17Error("timeout cannot prove child PID/start identity")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        observed = _process_start_ticks(pid)
        if observed is None:
            return
        if observed != expected:
            raise DispatchV17Error("child PID was reused during timeout")
        time.sleep(0.05)
    if _process_start_ticks(pid) == expected:
        os.kill(pid, signal.SIGKILL)


def _result(path: Path, *, run_id: str, command_hash: str) -> Mapping[str, Any]:
    value = _json(path, label="driver result")
    required = {"schema_version", "run_id", "command_sha256", "child_pid", "child_start_ticks", "exit_code", "reaped_at"}
    if set(value) != required or value.get("schema_version") != "stateguard3r.v17-driver-result.v1" or value.get("run_id") != run_id or value.get("command_sha256") != command_hash or type(value.get("child_pid")) is not int or value["child_pid"] <= 0 or not isinstance(value.get("child_start_ticks"), str) or not value["child_start_ticks"].isdigit() or type(value.get("exit_code")) is not int or not isinstance(value.get("reaped_at"), str):
        raise DispatchV17Error("driver result is malformed or unbound")
    return value


def _preexec(item: ArtifactPaths, run_id: str) -> Mapping[str, Any] | None:
    try:
        lines = _read(item.main, label="main log").decode("utf-8", errors="strict").splitlines()
    except (UnicodeDecodeError, DispatchV17Error):
        return None
    prefix = f"V17_DRIVER_PREEXEC_IDENTITY_FAILURE run_id={run_id} "
    matches = [line for line in lines if line.startswith(prefix)]
    if not matches:
        return None
    if len(matches) != 1 or any("V17_DRIVER_PAYLOAD_RELEASED" in line for line in lines):
        raise DispatchV17Error("pre-execution evidence is ambiguous")
    fields = dict(part.split("=", 1) for part in matches[0].split() if "=" in part)
    valid_reasons = {"start_gate_missing", "start_time_unavailable", "parent_start_time_mismatch", "go_gate_create_failed"}
    if not fields.get("child_pid", "").isdigit() or not fields.get("exit_code", "").isdigit() or fields.get("reason") not in valid_reasons:
        raise DispatchV17Error("pre-execution marker is malformed")
    return {"child_pid": int(fields["child_pid"]), "reason": fields["reason"], "exit_code": int(fields["exit_code"])}


def _wait_result(item: ArtifactPaths, run_id: str, command_hash: str, deadline: float) -> Mapping[str, Any]:
    while time.monotonic() < deadline:
        failure = _preexec(item, run_id)
        if failure is not None:
            raise _PreexecV17NoGo(failure)
        if _lexists(item.result):
            return _result(item.result, run_id=run_id, command_hash=command_hash)
        time.sleep(0.05)
    raise DispatchV17Error("driver did not publish an exclusive result before timeout")


def _ordered(path: Path, *, run_id: str, command_hash: str, result: Mapping[str, Any] | None, transcript: bool) -> None:
    text = _read(path, label="terminal stream").decode("utf-8", errors="strict")
    markers = [f"V17_DRIVER_START run_id={run_id}", f"V17_DRIVER_COMMAND_SHA256={command_hash}", f"V17_DRIVER_DISPATCHED run_id={run_id} kind=token-gated-wrapper"]
    if transcript:
        markers.insert(0, f"V17_PIPE_READY run_id={run_id}")
    if result is None:
        failure = _preexec(paths(run_id), run_id)
        if failure is None:
            raise DispatchV17Error("preexec stream lacks failure marker")
        markers.append(f"V17_DRIVER_PREEXEC_IDENTITY_FAILURE run_id={run_id} child_pid={failure['child_pid']} reason={failure['reason']} exit_code={failure['exit_code']}")
    else:
        pid, ticks, code = result["child_pid"], result["child_start_ticks"], result["exit_code"]
        markers.extend((f"V17_DRIVER_CHILD_PID run_id={run_id} child_pid={pid} child_start_ticks={ticks}", f"V17_DRIVER_PAYLOAD_RELEASE_ARMED run_id={run_id} child_pid={pid} child_start_ticks={ticks}", f"V17_DRIVER_PAYLOAD_RELEASED run_id={run_id} child_pid={pid} child_start_ticks={ticks}", f"V17_DRIVER_EXIT run_id={run_id} child_pid={pid} child_start_ticks={ticks} exit_code={code}", f"V17_DRIVER_RESULT_WRITTEN run_id={run_id} child_pid={pid} child_start_ticks={ticks} exit_code={code}"))
    positions = [text.find(marker) for marker in markers]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise DispatchV17Error("terminal stream markers are missing or ill ordered")


def _output_inventory(path: Path, *, frozen: bool) -> list[Mapping[str, Any]]:
    target_mode = 0o555
    if _kind(path) != "directory" or (frozen and stat.S_IMODE(path.stat().st_mode) != target_mode):
        raise DispatchV17Error("output directory is absent or not frozen")
    inventory: list[Mapping[str, Any]] = []
    for base, directories, files in os.walk(path, topdown=False, followlinks=False):
        current = Path(base)
        for name in sorted(files):
            child = current / name
            _read(child, label="output artifact")
            if frozen and stat.S_IMODE(child.stat().st_mode) != 0o444:
                raise DispatchV17Error("output file is mutable")
            inventory.append({"path": str(child.relative_to(path)), "kind": "file", "mode": "0444", "sha256": _sha(child)})
        for name in sorted(directories):
            child = current / name
            if _kind(child) != "directory" or (frozen and stat.S_IMODE(child.stat().st_mode) != target_mode):
                raise DispatchV17Error("output contains unsafe directory")
            inventory.append({"path": str(child.relative_to(path)), "kind": "directory", "mode": "0555"})
    inventory.append({"path": ".", "kind": "directory", "mode": "0555"})
    return sorted(inventory, key=lambda row: (str(row["path"]), str(row["kind"])))


def _freeze_output(path: Path) -> list[Mapping[str, Any]]:
    _output_inventory(path, frozen=False)
    for base, directories, files in os.walk(path, topdown=False, followlinks=False):
        current = Path(base)
        for name in files:
            os.chmod(current / name, 0o444)
        for name in directories:
            os.chmod(current / name, 0o555)
    os.chmod(path, 0o555)
    return _output_inventory(path, frozen=True)


def _runtime_ratio(run: Mapping[str, Any]) -> tuple[float, float]:
    runtime = run.get("runtime_seconds")
    if type(runtime) not in (int, float) or not math.isfinite(float(runtime)):
        raise DispatchV17Error("run lacks finite runtime_seconds")
    ratio = float(runtime) / REFERENCE_RUNTIME_SECONDS
    if ratio > RUNTIME_RATIO_LIMIT:
        raise DispatchV17Error("run exceeds preregistered runtime limit")
    return float(runtime), ratio


def _control_check(item: ArtifactPaths) -> Mapping[str, Any]:
    identical, hashes = {}, {}
    for name in PROTECTED:
        actual, expected = _sha(item.output / name), PREREGISTERED_PROTECTED_SHA256[name]
        identical[name] = actual == expected
        hashes[name] = {"preregistered_sha256": expected, "v17_sha256": actual}
    if not all(identical.values()):
        raise DispatchV17Error("control differs from preregistered protected hashes")
    run = _json(item.output / "run.json", label="control run")
    timeline = _json(item.output / "state-timeline.json", label="control timeline")
    rows = timeline.get("frames")
    if run.get("status") != "succeeded" or run.get("state_policy", {}).get("name") != POLICIES[CONTROL] or not isinstance(rows, list) or len(rows) != 30:
        raise DispatchV17Error("control metadata is incomplete")
    runtime, ratio = _runtime_ratio(run)
    for frame_id, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("frame_id") != frame_id or row.get("action") != "native_commit" or row.get("current_alarm") is not False or row.get("arm_pending") is not False or row.get("beta_base_override") is not None or row.get("detector_constructed") is not False or row.get("operator_constructed") is not False:
            raise DispatchV17Error("control is not thirty direct native commits")
    return {"status": "PASS", "protected_model_outputs_byte_identical": identical, "protected_model_output_sha256": hashes, "direct_commits": 30, "detector_constructed": False, "operator_constructed": False, "runtime_seconds": runtime, "reference_runtime_seconds": REFERENCE_RUNTIME_SECONDS, "runtime_ratio": ratio, "runtime_ratio_limit": RUNTIME_RATIO_LIMIT}


def _candidate_check(item: ArtifactPaths) -> Mapping[str, Any]:
    run = _json(item.output / "run.json", label="candidate run")
    timeline = _json(item.output / "state-timeline.json", label="candidate timeline")
    rows = timeline.get("frames")
    if run.get("status") != "succeeded" or run.get("state_policy", {}).get("name") != POLICIES[CANDIDATE] or not isinstance(rows, list) or len(rows) != 30:
        raise DispatchV17Error("candidate metadata is incomplete")
    runtime, ratio = _runtime_ratio(run)
    alarms = [row for row in rows if isinstance(row, Mapping) and row.get("current_alarm") is True]
    consumed = [row for row in rows if isinstance(row, Mapping) and row.get("action") == "one_shot_beta_base_floor"]
    if not alarms or not consumed:
        raise DispatchV17Error("candidate had no real alarm or arm consumption")
    for row in alarms:
        if row.get("action") not in {"native_commit", "armed_one_future_native_update"} or row.get("raw_current_state_gpu_fingerprint_unchanged") is not True or row.get("raw_current_mem_gpu_fingerprint_unchanged") is not True or row.get("raw_current_pose_gpu_fingerprint_unchanged") is not True:
            raise DispatchV17Error("alarm-frame preservation witness is incomplete")
    for row in consumed:
        if row.get("beta_base_before") != 0.1 or row.get("beta_base_during_mask") != 0.0 or row.get("beta_base_restored") != 0.1 or row.get("strict_mask_reduction") is not True or row.get("arm_pending") is not False:
            raise DispatchV17Error("one-shot floor witness is incomplete")
    control = paths(CONTROL).output
    changed = _read(item.output / "trajectory.json", label="candidate trajectory") != _read(control / "trajectory.json", label="control trajectory") or _read(item.output / "predictions-summary.json", label="candidate predictions") != _read(control / "predictions-summary.json", label="control predictions")
    if not changed:
        raise DispatchV17Error("candidate caused no later raw output change")
    return {"status": "PASS", "alarm_count": len(alarms), "arm_consumptions": len(consumed), "later_raw_output_changed": True, "runtime_seconds": runtime, "reference_runtime_seconds": REFERENCE_RUNTIME_SECONDS, "runtime_ratio": ratio, "runtime_ratio_limit": RUNTIME_RATIO_LIMIT}


def _validate_driver(item: ArtifactPaths, run_id: str, command: Sequence[str], command_hash: str) -> str:
    source = _read(item.driver, label="driver").decode("utf-8", errors="strict")
    tokens = [line.split("=", 1)[1] for line in source.splitlines() if line.startswith("go_token=")]
    if len(tokens) != 1 or len(tokens[0]) != 64 or any(char not in "0123456789abcdef" for char in tokens[0]) or source != _driver(run_id, item, command, command_hash, tokens[0]):
        raise DispatchV17Error("frozen driver differs from token-gated template")
    return hashlib.sha256((tokens[0] + "\n").encode("ascii")).hexdigest()


def _validator(item: ArtifactPaths, run_id: str) -> Mapping[str, Any]:
    report: dict[str, Any] = {"schema_version": "stateguard3r.v17-validator.v1", "run_id": run_id, "cpu_only": True, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "started_at": datetime.now().astimezone().isoformat()}
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise DispatchV17Error("validator must run with CUDA hidden")
        primary = (item.journal, item.preflight, item.main, item.transcript, item.driver, item.postflight)
        for path in primary:
            if _kind(path) != "regular" or stat.S_IMODE(path.stat().st_mode) != 0o444:
                raise DispatchV17Error("sealed primary artifact is absent or mutable")
            _read(path, label="sealed primary artifact")
        preflight, postflight = _json(item.preflight, label="preflight"), _json(item.postflight, label="postflight")
        command = preflight.get("command")
        if preflight.get("schema_version") != "stateguard3r.v17-preflight.v1" or preflight.get("run_id") != run_id or not isinstance(command, list) or any(type(part) is not str for part in command):
            raise DispatchV17Error("preflight schema is invalid")
        command_hash = _command_hash(command)
        if preflight.get("command_sha256") != command_hash or command != _command(run_id, item):
            raise DispatchV17Error("preflight command does not recompute")
        _production_contract(run_id, item, command)
        snapshots = preflight.get("gpu_preflights")
        if not isinstance(snapshots, list) or len(snapshots) != 2:
            raise DispatchV17Error("preflight lacks two GPU snapshots")
        for snapshot in snapshots:
            selected = snapshot.get("selected") if isinstance(snapshot, Mapping) else None
            if not isinstance(selected, Mapping) or selected.get("uuid") != GPU_UUID or selected.get("memory_free_mib", -1) < MIN_FREE_MIB or snapshot.get("project_gpu2_processes") != []:
                raise DispatchV17Error("preflight GPU snapshot is invalid")
        source = _source()
        if preflight.get("source_provenance") != source or postflight.get("source_provenance") != source:
            raise DispatchV17Error("source provenance no longer recomputes")
        token_sha = _validate_driver(item, run_id, command, command_hash)
        if _kind(item.result) != "regular" or stat.S_IMODE(item.result.stat().st_mode) != 0o444:
            raise DispatchV17Error("result is absent or mutable")
        result = _result(item.result, run_id=run_id, command_hash=command_hash)
        _ordered(item.main, run_id=run_id, command_hash=command_hash, result=result, transcript=False)
        _ordered(item.transcript, run_id=run_id, command_hash=command_hash, result=result, transcript=True)
        if postflight.get("result") != result or postflight.get("go_token_sha256") != token_sha or postflight.get("pipe_close_and_drain", {}).get("drained") is not True or "gpu_postflight_error" in postflight:
            raise DispatchV17Error("postflight evidence is incomplete")
        identity = postflight.get("pid_start_time_check")
        if not isinstance(identity, Mapping) or identity.get("expected_start_ticks") != result["child_start_ticks"] or identity.get("pair_absent") is not True or identity.get("pid_reused") is not False:
            raise DispatchV17Error("postflight lacks dead PID/start proof")
        if result["exit_code"] != 0:
            report["status"] = "CHILD_NONZERO_EVIDENCE_COMPLETE"
        else:
            inventory = _output_inventory(item.output, frozen=True)
            if postflight.get("output_inventory") != inventory:
                raise DispatchV17Error("frozen output inventory does not recompute")
            report["dynamic_always_control" if run_id == CONTROL else "dynamic_candidate"] = _control_check(item) if run_id == CONTROL else _candidate_check(item)
            report["status"] = "PASS"
        report["artifact_sha256"] = {path.name: _sha(path) for path in (*primary, item.result)}
    except Exception as error:
        report.update({"status": "FAIL", "error": str(error), "finished_at": datetime.now().astimezone().isoformat()})
        _write_json(item.validator, report)
        _freeze(item.validator)
        raise DispatchV17Error(f"fresh CUDA-hidden validator failed: {error}") from error
    report["finished_at"] = datetime.now().astimezone().isoformat()
    _write_json(item.validator, report)
    _freeze(item.validator)
    return report


def validate_sealed_run(run_id: str) -> Mapping[str, Any]:
    item = paths(run_id)
    _canonical(item)
    if _kind(item.lease) != "directory" or _lexists(item.validator):
        raise DispatchV17Error("validator requires owner lease and a fresh report")
    return _validator(item, run_id)


def _fresh_validator(run_id: str) -> Mapping[str, Any]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--validate-run-id", run_id], text=True, capture_output=True, env=environment)
    if result.returncode != 0:
        raise DispatchV17Error(f"CUDA-hidden validator subprocess failed: {result.stderr.strip() or result.stdout.strip()}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise DispatchV17Error("CUDA-hidden validator emitted invalid JSON") from error
    if not isinstance(value, Mapping):
        raise DispatchV17Error("CUDA-hidden validator did not emit an object")
    return value


def _postflight(item: ArtifactPaths, *, run_id: str, result: Mapping[str, Any] | None, source: Mapping[str, Any], pipe: Mapping[str, Any], error: str | None, token_sha: str | None) -> Mapping[str, Any]:
    value: dict[str, Any] = {"schema_version": "stateguard3r.v17-postflight.v1", "run_id": run_id, "result": result, "source_provenance": source, "pipe_close_and_drain": pipe, "error": error, "go_token_sha256": token_sha}
    try:
        value["gpu_postflight"] = _snapshot(require_minimum=False)
    except Exception as caught:
        value["gpu_postflight_error"] = str(caught)
    return value


def _freeze_primary(item: ArtifactPaths) -> None:
    for path in (item.journal, item.preflight, item.main, item.transcript, item.driver, item.result, item.postflight):
        if _lexists(path):
            _freeze(path)


def dispatch(run_id: str, *, timeout_seconds: float) -> Mapping[str, Any]:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise DispatchV17Error("timeout must be positive and finite")
    item = paths(run_id)
    _acquire(item)
    _journal(item, "LEASE_AND_JOURNAL_CREATED", run_id=run_id)
    source: Mapping[str, Any] | None = None
    pane: str | None = None
    pipe: Mapping[str, Any] | None = None
    result: Mapping[str, Any] | None = None
    caught: Exception | None = None
    command = _command(run_id, item)
    command_hash = _command_hash(command)
    token = secrets.token_hex(32)
    token_sha = hashlib.sha256((token + "\n").encode("ascii")).hexdigest()
    try:
        _journal(item, "SOURCE_AND_GATE_B_CHECK")
        source = _source()
        _production_contract(run_id, item, command)
        _journal(item, "GPU_PREFLIGHT_FIRST")
        first = _snapshot()
        time.sleep(1.0)
        _journal(item, "GPU_PREFLIGHT_SECOND")
        second = _snapshot()
        _write_json(item.preflight, {"schema_version": "stateguard3r.v17-preflight.v1", "run_id": run_id, "command": command, "command_sha256": command_hash, "gpu_preflights": [first, second], "source_provenance": source})
        _freeze(item.preflight)
        _write_new(item.main, "")
        _write_new(item.transcript, "")
        _write_new(item.driver, _driver(run_id, item, command, command_hash, token), 0o555)
        pane, deadline = _pane(), time.monotonic() + timeout_seconds
        _journal(item, "TMUX_PANE_CREATED", pane=pane)
        _tmux(["pipe-pane", "-o", "-t", pane, f"cat >> {shlex.quote(str(item.transcript))}"])
        ready = f"V17_PIPE_READY run_id={run_id}"
        _tmux(["send-keys", "-t", pane, f"printf '%s\\n' {shlex.quote(ready)}", "C-m"])
        _wait_text(item.transcript, ready, deadline)
        _journal(item, "PIPE_READY", pane=pane)
        _tmux(["send-keys", "-t", pane, f"bash {shlex.quote(str(item.driver))}; exit", "C-m"])
        _journal(item, "DRIVER_RELEASED", pane=pane)
        try:
            result = _wait_result(item, run_id, command_hash, deadline)
        except _PreexecV17NoGo as event:
            _wait_pane_exit(pane, time.monotonic() + 10.0)
            pipe = _drain(pane, item)
            raise DispatchV17Error(f"pre-execution identity NO-GO: {event.evidence}") from event
        except DispatchV17Error as timeout:
            identity = _partial_identity(item, run_id)
            if identity is None:
                raise DispatchV17Error("timeout lacks a proven PID/start identity; refusing to seal possible live CUDA") from timeout
            _terminate_owned(identity)
            result = _wait_result(item, run_id, command_hash, time.monotonic() + 10.0)
        _wait_pane_exit(pane, deadline)
        pipe = _drain(pane, item)
        _ordered(item.main, run_id=run_id, command_hash=command_hash, result=result, transcript=False)
        _ordered(item.transcript, run_id=run_id, command_hash=command_hash, result=result, transcript=True)
        identity = _pid_absent(result)
        if not identity["pair_absent"] or identity["pid_reused"]:
            raise DispatchV17Error("PID/start pair is live or reused after driver exit")
        post_source = _source()
        if post_source != source:
            raise DispatchV17Error("source drifted during child execution")
        postflight = _postflight(item, run_id=run_id, result=result, source=post_source, pipe=pipe, error=None, token_sha=token_sha)
        if "gpu_postflight_error" in postflight:
            raise DispatchV17Error("postflight GPU snapshot failed")
        postflight["pid_start_time_check"] = identity
        postflight["output_inventory"] = _freeze_output(item.output) if result["exit_code"] == 0 else None
        _write_json(item.postflight, postflight)
        _journal(item, "POSTFLIGHT_COMPLETE", child_exit_code=result["exit_code"])
    except Exception as error:
        caught = error
        _journal(item, "EXCEPTION", error_class=type(error).__name__, error=str(error))
        if pane is not None and pipe is None and _lexists(item.transcript) and _lexists(item.main):
            try:
                pipe = _drain(pane, item)
            except Exception as drain_error:
                pipe = {"drained": False, "close_error": str(drain_error)}
        if source is not None and _lexists(item.preflight) and not _lexists(item.postflight):
            try:
                _write_json(item.postflight, _postflight(item, run_id=run_id, result=result, source=source, pipe=pipe or {"drained": False}, error=str(error), token_sha=token_sha))
            except Exception as postflight_error:
                _journal(item, "POSTFLIGHT_WRITE_FAILURE", error=str(postflight_error))
    finally:
        _journal(item, "TERMINAL", status="PASS" if caught is None else "FAIL")
        try:
            _freeze(item.journal)
            if pane is None or (pipe is not None and pipe.get("drained") is True and pipe.get("pane_alive_at_close") is False):
                _freeze_primary(item)
        except Exception as freeze_error:
            caught = caught or freeze_error
    validator: Mapping[str, Any] | None = None
    if caught is None:
        try:
            validator = _fresh_validator(run_id)
            if validator.get("status") != "PASS":
                raise DispatchV17Error("independent validator did not pass")
        except Exception as error:
            caught = error
    if caught is not None:
        if not _lexists(item.validator):
            _write_json(item.validator, {"schema_version": "stateguard3r.v17-dispatch-terminal.v1", "run_id": run_id, "status": "FAIL", "error_class": type(caught).__name__, "error": str(caught), "finished_at": datetime.now().astimezone().isoformat()})
            _freeze(item.validator)
        raise DispatchV17Error(f"terminal evidence sealed as failure: {caught}") from caught
    assert result is not None
    if result["exit_code"] != 0:
        raise DispatchV17Error(f"child exited nonzero with immutable evidence: {result['exit_code']}")
    return {"status": "PASS", "run_id": run_id, "pane": pane, "command_sha256": command_hash, "validator": validator, "paths": {name: str(path) for name, path in asdict(item).items()}}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        value = validate_sealed_run(args.validate_run_id) if args.validate_run_id is not None else dispatch(args.run_id, timeout_seconds=args.timeout_seconds)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return 0
    except DispatchV17Error as error:
        print(f"V17_DISPATCH_ERROR={error}", file=sys.stderr)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
