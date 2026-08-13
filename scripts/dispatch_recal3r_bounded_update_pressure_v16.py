#!/usr/bin/env python3
"""One-use terminal-evidence dispatcher for v16 split RGB capabilities.

The implementation is owned by v16 and never imports an earlier recovery or
input route.  CUDA is reachable only in the token-gated child of a fresh pane
inside the existing ``stateguard`` tmux session.
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
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from stateguard3r.v16_source_import_audit import audit_v16_production_script

SESSION, GPU_INDEX = "stateguard", "2"
GPU_UUID, MIN_FREE_MIB = "GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250", 12288
RECAL3R_ROOT = ROOT.parent / "baselines" / "ReCal3R"
RECAL3R_COMMIT = "466c7cdf3acd2f589f1d82e5f6391966f19db9ff"
PYTHON = RECAL3R_ROOT / ".venv" / "bin" / "python"
RUNNER = ROOT / "scripts" / "run_recal3r_bounded_update_pressure_v16.py"
CAPSULE = ROOT / "outputs" / "recovery-update-pressure-v16-dynamic-rgb-list-capsule-0001.json"
CHECKPOINT = RECAL3R_ROOT / "src" / "cut3r_512_dpt_4_64.pth"
CHECKPOINT_SHA256 = "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103"
DETECTOR_CONFIG = ROOT / "outputs" / "formal-v3-calibration-0001" / "formal-config.json"
CONTROL = "recovery-update-pressure-v16-dynamic-always-commit-0001"
CANDIDATE = "recovery-update-pressure-v16-dynamic-candidate-0001"
POLICIES = {CONTROL: "always-commit", CANDIDATE: "detector-v3-bounded-update-pressure"}
PROTECTED = ("checkpoint-load-audit.json", "health.jsonl", "predictions-summary.json", "trajectory.json")
PREREGISTERED_PROTECTED_SHA256 = {
    "checkpoint-load-audit.json": "3e12875a701e0afc534d8f3a90b6a5fb54a58cfca59e2ce055109e6db4a4153d",
    "health.jsonl": "9de0c0b690c00e603fd16c81ff6258ce3c275388e48d66a0c1a64672848131fe",
    "predictions-summary.json": "cb11415bfe3765e7ad87fd9655873f319cdaaa553ef948f3e3f935037a8504ed",
    "trajectory.json": "fdba3312b792e4c67f56c8a0f8feba908846cd61190fa5a8665f0b0be81d013b",
}
PREREGISTERED_CONTROL_RUNTIME_SECONDS = 5.660642675124109
RUNTIME_RATIO_LIMIT = 1.20
# This list is intentionally exhaustive rather than a glob.  All entries are
# v16-owned and must be committed byte-for-byte before a release is eligible.
COMPONENTS = (
    ROOT / "src" / "stateguard3r" / "dynamic_rgb_capability_v16.py",
    ROOT / "src" / "stateguard3r" / "frame_zero_probe_capability_v16.py",
    ROOT / "scripts" / "build_recal3r_dynamic_rgb_capability_v16.py",
    ROOT / "scripts" / "build_recal3r_frame_zero_probe_capability_v16.py",
    ROOT / "scripts" / "probe_recal3r_bounded_update_pressure_interface_v16.py",
    ROOT / "src" / "stateguard3r" / "bounded_update_pressure_v16.py",
    ROOT / "src" / "stateguard3r" / "recal3r_bounded_update_pressure_runner_v16.py",
    ROOT / "src" / "stateguard3r" / "online_detector_v16.py",
    ROOT / "src" / "stateguard3r" / "online_visual_overlap_v16.py",
    ROOT / "src" / "stateguard3r" / "timestamp_order_v16.py",
    ROOT / "src" / "stateguard3r" / "health.py",
    ROOT / "src" / "stateguard3r" / "v16_source_import_audit.py",
    RUNNER,
    Path(__file__).resolve(),
    ROOT / "scripts" / "wait_and_dispatch_recal3r_bounded_update_pressure_v16_control.sh",
)
EXTERNALS = (CAPSULE, CHECKPOINT, DETECTOR_CONFIG, RECAL3R_ROOT / "src" / "dust3r" / "model.py")
RUNTIME_IMPORT_SOURCES = (
    ROOT / "src" / "stateguard3r" / "bounded_update_pressure_v16.py",
    ROOT / "src" / "stateguard3r" / "recal3r_bounded_update_pressure_runner_v16.py",
    ROOT / "src" / "stateguard3r" / "online_detector_v16.py",
    ROOT / "src" / "stateguard3r" / "online_visual_overlap_v16.py",
    ROOT / "src" / "stateguard3r" / "timestamp_order_v16.py",
    ROOT / "src" / "stateguard3r" / "dynamic_rgb_capability_v16.py",
    ROOT / "src" / "stateguard3r" / "health.py",
)

# v14/v15 terminal untracked content is preserved for audit.  This permission
# is an exact historical inventory, not a version-name pattern or a bypass for
# any v16 worktree change.  It must stay in sync with the Gate-A audit.
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
    "src/stateguard3r/online_detector_v15.py",
    "src/stateguard3r/online_visual_overlap_v15.py",
    "src/stateguard3r/recal3r_bounded_update_pressure_runner_v15.py",
    "src/stateguard3r/recal3r_update_pressure_bounded_state_memory_runner_v14.py",
    "src/stateguard3r/rgb_listing_capsule_v15.py",
    "src/stateguard3r/runtime_input_capsule_v14.py",
    "src/stateguard3r/timestamp_order_v15.py",
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


class DispatchV16Error(RuntimeError):
    """A v16 release has no complete, valid one-use evidence route."""


class _PreexecV16NoGo(DispatchV16Error):
    def __init__(self, failure: Mapping[str, Any]) -> None:
        super().__init__("v16 driver reached a pre-execution identity NO-GO")
        self.failure = failure


@dataclass(frozen=True)
class Paths:
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


def paths(run_id: str) -> Paths:
    if run_id not in POLICIES:
        raise DispatchV16Error("unknown fixed v16 run ID")
    return Paths(
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
    command.add_argument("--run-id")
    command.add_argument("--validate-run-id")
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


def _canonical(item: Paths) -> None:
    if item != paths(item.output.name):
        raise DispatchV16Error("v16 artifact paths are not canonical")
    for directory in (ROOT / "outputs", ROOT / "logs", ROOT / "tmp"):
        if _kind(directory) != "directory":
            raise DispatchV16Error("v16 artifact parent is missing or unsafe")


def _read(path: Path, *, label: str) -> bytes:
    if _kind(path) != "regular":
        raise DispatchV16Error(f"{label} is missing, nonregular, or a symlink")
    data = path.read_bytes()
    if b"\0" in data:
        raise DispatchV16Error(f"{label} contains NUL")
    return data


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, data: str, mode: int = 0o644) -> None:
    if "\0" in data or _lexists(path):
        raise DispatchV16Error("refusing overwrite or NUL in one-use v16 evidence")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, mode)


def _write_json(path: Path, value: Any, mode: int = 0o644) -> None:
    _write_new(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", mode)


def _freeze(path: Path) -> None:
    _read(path, label="terminal artifact")
    os.chmod(path, 0o444)
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise DispatchV16Error("could not freeze v16 terminal artifact")


def _journal(item: Paths, stage: str, **values: Any) -> None:
    row = json.dumps({"at": datetime.now().astimezone().isoformat(), "stage": stage, **values}, sort_keys=True, allow_nan=False) + "\n"
    with item.journal.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(row)
        stream.flush()
        os.fsync(stream.fileno())


def _acquire(item: Paths) -> None:
    _canonical(item)
    if any(_lexists(value) for value in asdict(item).values()):
        raise DispatchV16Error("a v16 one-use artifact or lease already exists")
    try:
        item.lease.mkdir(mode=0o700)
    except FileExistsError as error:
        raise DispatchV16Error("v16 same-ID owner lease already exists") from error
    if _kind(item.lease) != "directory" or list(item.lease.iterdir()):
        raise DispatchV16Error("v16 owner lease is unsafe")
    _write_new(item.journal, "")


def _regular_frozen(path: Path, *, label: str) -> None:
    if _kind(path) != "regular" or stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise DispatchV16Error(f"{label} must be a frozen regular 0444 file")


def _git(repository: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repository), *args], check=True, text=True, capture_output=True).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise DispatchV16Error("cannot inspect v16 source provenance") from error


def _git_bytes(repository: Path, *args: str) -> bytes:
    try:
        return subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as error:
        raise DispatchV16Error("cannot inspect v16 source provenance bytes") from error


def _v16_relevant_clean() -> None:
    status = _git(ROOT, "status", "--porcelain=v1", "--untracked-files=all")
    unexpected: list[str] = []
    for line in status.splitlines():
        if len(line) < 4 or line[:2] != "??" or line[3:] not in PREEXISTING_TERMINAL_UNTRACKED:
            unexpected.append(line)
    if unexpected:
        raise DispatchV16Error(f"StateGuard3R has v16-relevant dirty entries: {unexpected}")


def _source() -> Mapping[str, Any]:
    _v16_relevant_clean()
    if _git(RECAL3R_ROOT, "status", "--porcelain"):
        raise DispatchV16Error("ReCal3R worktree is dirty")
    if _git(RECAL3R_ROOT, "rev-parse", "HEAD") != RECAL3R_COMMIT:
        raise DispatchV16Error("ReCal3R commit differs from v16 pin")
    components: dict[str, str] = {}
    for path in COMPONENTS:
        if _kind(path) != "regular":
            raise DispatchV16Error("v16 component is unsafe")
        relative = path.relative_to(ROOT)
        _git(ROOT, "ls-files", "--error-unmatch", "--", str(relative))
        if _git_bytes(ROOT, "show", f"HEAD:{relative}") != path.read_bytes():
            raise DispatchV16Error(f"v16 component differs from committed HEAD: {relative}")
        components[str(relative)] = _sha(path)
    for path in (CAPSULE, DETECTOR_CONFIG):
        _regular_frozen(path, label="v16 frozen input")
    if _kind(CHECKPOINT) != "regular" or _sha(CHECKPOINT) != CHECKPOINT_SHA256:
        raise DispatchV16Error("v16 checkpoint is missing or differs")
    return {
        "stateguard_commit": _git(ROOT, "rev-parse", "HEAD"),
        "recal3r_commit": RECAL3R_COMMIT,
        "component_sha256": components,
        "external_sha256": {str(path): _sha(path) for path in EXTERNALS},
        "source_import_audit": audit_v16_production_script(RUNNER, RUNTIME_IMPORT_SOURCES),
    }


def _project_processes() -> list[Mapping[str, Any]]:
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True)
    found: list[Mapping[str, Any]] = []
    for line in result.stdout.splitlines():
        row = [part.strip() for part in line.split(",")]
        if len(row) != 4 or row[0] != GPU_UUID or not row[1].isdigit():
            continue
        try:
            command = Path(f"/proc/{row[1]}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        except OSError:
            command = "<unavailable>"
        if str(ROOT) in command or str(RECAL3R_ROOT) in command:
            found.append({"gpu_uuid": row[0], "pid": int(row[1]), "process_name": row[2], "used_memory_mib": row[3], "command": command})
    return found


def _snapshot(*, require_minimum: bool = True) -> Mapping[str, Any]:
    command = ["nvidia-smi", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[1].isdigit():
            rows.append({"uuid": parts[0], "memory_free_mib": int(parts[1])})
    selected = [row for row in rows if row["uuid"] == GPU_UUID]
    project = _project_processes()
    if len(selected) != 1 or (require_minimum and selected[0]["memory_free_mib"] < MIN_FREE_MIB):
        raise DispatchV16Error("GPU-2 UUID/free-memory condition fails")
    if project:
        raise DispatchV16Error("a Project2 process already uses GPU-2")
    return {"at": datetime.now().astimezone().isoformat(), "command": command, "rows": rows, "selected": selected[0], "project_gpu2_processes": project}


def _validate_snapshot(value: Mapping[str, Any], *, require_minimum: bool) -> None:
    selected = value.get("selected")
    if not isinstance(selected, Mapping) or selected.get("uuid") != GPU_UUID or type(selected.get("memory_free_mib")) is not int or (require_minimum and selected["memory_free_mib"] < MIN_FREE_MIB) or value.get("project_gpu2_processes") != [] or not isinstance(value.get("at"), str):
        raise DispatchV16Error("stored v16 GPU snapshot is invalid")


def _command(run_id: str, item: Paths) -> list[str]:
    return [str(PYTHON), str(RUNNER), "--capsule", str(CAPSULE), "--output-dir", str(item.output), "--state-policy", POLICIES[run_id], "--device", "cuda", "--seed", "0", "--beta-base", "0.1", "--detector-config", str(DETECTOR_CONFIG)]


def _command_hash(argv: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(argv), ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(_read(path, label=label))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DispatchV16Error(f"{label} is not UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise DispatchV16Error(f"{label} is not an object")
    return value


def _control_predecessor(run_id: str) -> None:
    if run_id == CONTROL:
        return
    item = paths(CONTROL)
    if _kind(item.validator) != "regular" or stat.S_IMODE(item.validator.stat().st_mode) != 0o444:
        raise DispatchV16Error("v16 candidate requires frozen PASS dynamic control")
    report = _json(item.validator, label="v16 control validator")
    if report.get("status") != "PASS" or report.get("dynamic_always_control", {}).get("status") != "PASS":
        raise DispatchV16Error("v16 candidate is blocked by dynamic control")


def _production_contract(run_id: str, item: Paths, argv: Sequence[str]) -> None:
    if list(argv) != _command(run_id, item):
        raise DispatchV16Error("v16 child argv differs from fixed production contract")
    _control_predecessor(run_id)


def _driver(run_id: str, item: Paths, argv: Sequence[str], command_hash: str, token: str) -> str:
    if len(token) != 64 or any(ch not in "0123456789abcdef" for ch in token):
        raise DispatchV16Error("v16 go token is not 256-bit lowercase hex")
    command = shlex.join(argv)
    run_q, main_q, result_q, lease_q, hash_q, token_q = tuple(shlex.quote(value) for value in (run_id, str(item.main), str(item.result), str(item.lease), command_hash, token))
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
record "V16_DRIVER_START run_id=$run_id"
record "V16_DRIVER_COMMAND_SHA256=$command_sha256"
record "V16_DRIVER_DISPATCHED run_id=$run_id kind=token-gated-wrapper"
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
      exec env CUDA_VISIBLE_DEVICES={GPU_INDEX} {command}
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
  record "V16_DRIVER_PREEXEC_IDENTITY_FAILURE run_id=$run_id child_pid=$child_pid reason=start_gate_missing exit_code=$child_exit"
  exit 70
fi
read -r kind reported_pid child_start_ticks < "$start_path"
if [[ "$kind" != READY || "$reported_pid" != "$child_pid" || ! "$child_start_ticks" =~ ^[0-9]+$ ]]; then
  wait "$child_pid"; child_exit=$?
  record "V16_DRIVER_PREEXEC_IDENTITY_FAILURE run_id=$run_id child_pid=$child_pid reason=start_time_unavailable exit_code=$child_exit"
  exit 70
fi
observed_ticks="$(awk '{{print $22}}' "/proc/$child_pid/stat" 2>/dev/null)"
if [[ "$observed_ticks" != "$child_start_ticks" ]]; then
  kill "$child_pid" 2>/dev/null; wait "$child_pid"; child_exit=$?
  record "V16_DRIVER_PREEXEC_IDENTITY_FAILURE run_id=$run_id child_pid=$child_pid reason=parent_start_time_mismatch exit_code=$child_exit"
  exit 70
fi
record "V16_DRIVER_CHILD_PID run_id=$run_id child_pid=$child_pid child_start_ticks=$child_start_ticks"
record "V16_DRIVER_PAYLOAD_RELEASE_ARMED run_id=$run_id child_pid=$child_pid child_start_ticks=$child_start_ticks"
if ! (set -o noclobber; printf '%s\\n' "$go_token" > "$go_path") 2>/dev/null; then
  kill "$child_pid" 2>/dev/null; wait "$child_pid"; child_exit=$?
  record "V16_DRIVER_PREEXEC_IDENTITY_FAILURE run_id=$run_id child_pid=$child_pid reason=go_gate_create_failed exit_code=$child_exit"
  exit 70
fi
record "V16_DRIVER_PAYLOAD_RELEASED run_id=$run_id child_pid=$child_pid child_start_ticks=$child_start_ticks"
wait "$child_pid"; child_exit=$?
reaped_at="$(date --iso-8601=seconds)"
record "V16_DRIVER_EXIT run_id=$run_id child_pid=$child_pid child_start_ticks=$child_start_ticks exit_code=$child_exit reaped_at=$reaped_at"
if ! (set -o noclobber; : > "$result_path") 2>/dev/null; then record "V16_DRIVER_RESULT_CREATE_FAILED run_id=$run_id"; exit 70; fi
printf '{{"schema_version":"stateguard3r.v16-driver-result.v1","run_id":"%s","command_sha256":"%s","child_pid":%s,"child_start_ticks":"%s","exit_code":%s,"reaped_at":"%s"}}\\n' "$run_id" "$command_sha256" "$child_pid" "$child_start_ticks" "$child_exit" "$reaped_at" >> "$result_path"
record "V16_DRIVER_RESULT_WRITTEN run_id=$run_id child_pid=$child_pid child_start_ticks=$child_start_ticks exit_code=$child_exit"
exit 0
'''


def _tmux(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["tmux", *args], check=check, text=True, capture_output=True)


def _pane() -> str:
    try:
        _tmux(["has-session", "-t", SESSION])
        pane = _tmux(["new-window", "-d", "-P", "-F", "#{pane_id}", "-t", SESSION, "-c", str(ROOT)]).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise DispatchV16Error("existing stateguard tmux session is unavailable") from error
    if not pane.startswith("%"):
        raise DispatchV16Error("tmux did not return a v16 pane")
    return pane


def _pane_alive(pane: str) -> bool:
    return _tmux(["list-panes", "-t", pane, "-F", "#{pane_id}"], check=False).returncode == 0


def _wait_text(path: Path, marker: str, deadline: float) -> None:
    while time.monotonic() < deadline:
        if _lexists(path) and marker in _read(path, label="v16 tmux transcript").decode("utf-8", errors="strict"):
            return
        time.sleep(0.05)
    raise DispatchV16Error(f"timed out waiting for {marker}")


def _wait_pane_exit(pane: str, deadline: float) -> None:
    while time.monotonic() < deadline:
        if not _pane_alive(pane):
            return
        time.sleep(0.05)
    raise DispatchV16Error("v16 driver pane did not exit")


def _drain(pane: str, item: Paths) -> Mapping[str, Any]:
    if _pane_alive(pane):
        raise DispatchV16Error("v16 pane is live; terminal streams cannot freeze")
    first = (_read(item.transcript, label="v16 transcript"), _read(item.main, label="v16 main log"))
    time.sleep(0.15)
    second = (_read(item.transcript, label="v16 transcript"), _read(item.main, label="v16 main log"))
    if first != second:
        raise DispatchV16Error("v16 pipe did not drain before sealing")
    return {"drained": True, "pane_alive_at_close": False, "transcript_sha256": hashlib.sha256(second[0]).hexdigest(), "main_sha256": hashlib.sha256(second[1]).hexdigest()}


def _process_start_ticks(pid: int) -> str | None:
    try:
        tail = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return tail[19] if len(tail) > 19 and tail[19].isdigit() else None
    except (FileNotFoundError, OSError, IndexError):
        return None


def _pid_absent(result: Mapping[str, Any]) -> Mapping[str, Any]:
    observed = _process_start_ticks(result["child_pid"])
    return {"child_pid": result["child_pid"], "expected_start_ticks": result["child_start_ticks"], "observed_start_ticks": observed, "pair_absent": observed is None, "pid_reused": observed is not None and observed != result["child_start_ticks"]}


def _partial_identity(item: Paths, run_id: str) -> tuple[int, str] | None:
    try:
        lines = _read(item.main, label="v16 main log").decode("utf-8", errors="strict").splitlines()
    except (UnicodeDecodeError, DispatchV16Error):
        return None
    marker = f"V16_DRIVER_CHILD_PID run_id={run_id} "
    matches = [line for line in lines if line.startswith(marker)]
    if len(matches) != 1:
        return None
    fields = dict(piece.split("=", 1) for piece in matches[0].split() if "=" in piece)
    if fields.get("child_pid", "").isdigit() and fields.get("child_start_ticks", "").isdigit():
        return int(fields["child_pid"]), fields["child_start_ticks"]
    return None


def _terminate_owned(identity: tuple[int, str], *, grace_seconds: float = 5.0) -> None:
    pid, ticks = identity
    if _process_start_ticks(pid) != ticks:
        raise DispatchV16Error("v16 timeout cannot prove child PID/start identity")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        observed = _process_start_ticks(pid)
        if observed is None:
            return
        if observed != ticks:
            raise DispatchV16Error("v16 child PID was reused during timeout")
        time.sleep(0.05)
    if _process_start_ticks(pid) == ticks:
        os.kill(pid, signal.SIGKILL)


def _result(path: Path, *, run_id: str, command_hash: str) -> Mapping[str, Any]:
    value = _json(path, label="v16 driver result")
    expected = {"schema_version", "run_id", "command_sha256", "child_pid", "child_start_ticks", "exit_code", "reaped_at"}
    if set(value) != expected or value.get("schema_version") != "stateguard3r.v16-driver-result.v1" or value.get("run_id") != run_id or value.get("command_sha256") != command_hash or type(value.get("child_pid")) is not int or value["child_pid"] <= 0 or not isinstance(value.get("child_start_ticks"), str) or not value["child_start_ticks"].isdigit() or type(value.get("exit_code")) is not int or not isinstance(value.get("reaped_at"), str):
        raise DispatchV16Error("v16 driver result is malformed or unbound")
    return value


def _preexec(item: Paths, run_id: str) -> Mapping[str, Any] | None:
    try:
        lines = _read(item.main, label="v16 main log").decode("utf-8", errors="strict").splitlines()
    except (UnicodeDecodeError, DispatchV16Error):
        return None
    marker = f"V16_DRIVER_PREEXEC_IDENTITY_FAILURE run_id={run_id} "
    found = [line for line in lines if line.startswith(marker)]
    if not found:
        return None
    if len(found) != 1 or any("V16_DRIVER_PAYLOAD_RELEASED" in line for line in lines):
        raise DispatchV16Error("v16 pre-execution evidence is ambiguous")
    fields = dict(piece.split("=", 1) for piece in found[0].split() if "=" in piece)
    if not fields.get("child_pid", "").isdigit() or fields.get("reason") not in {"start_gate_missing", "start_time_unavailable", "parent_start_time_mismatch", "go_gate_create_failed"} or not fields.get("exit_code", "").isdigit():
        raise DispatchV16Error("v16 pre-execution marker is malformed")
    return {"child_pid": int(fields["child_pid"]), "reason": fields["reason"], "exit_code": int(fields["exit_code"])}


def _wait_result(item: Paths, run_id: str, command_hash: str, deadline: float) -> Mapping[str, Any]:
    while time.monotonic() < deadline:
        failure = _preexec(item, run_id)
        if failure is not None:
            raise _PreexecV16NoGo(failure)
        if _lexists(item.result):
            return _result(item.result, run_id=run_id, command_hash=command_hash)
        time.sleep(0.05)
    raise DispatchV16Error("v16 driver did not publish an exclusive result before timeout")


def _ordered(path: Path, *, run_id: str, command_hash: str, result: Mapping[str, Any] | None, transcript: bool) -> None:
    text = _read(path, label="v16 terminal stream").decode("utf-8", errors="strict")
    markers = [f"V16_DRIVER_START run_id={run_id}", f"V16_DRIVER_COMMAND_SHA256={command_hash}", f"V16_DRIVER_DISPATCHED run_id={run_id} kind=token-gated-wrapper"]
    if transcript:
        markers.insert(0, f"V16_PIPE_READY run_id={run_id}")
    if result is None:
        failure = _preexec(paths(run_id), run_id)
        if failure is None:
            raise DispatchV16Error("v16 preexec stream lacks failure marker")
        markers.append(f"V16_DRIVER_PREEXEC_IDENTITY_FAILURE run_id={run_id} child_pid={failure['child_pid']} reason={failure['reason']} exit_code={failure['exit_code']}")
    else:
        pid, ticks, code = result["child_pid"], result["child_start_ticks"], result["exit_code"]
        markers.extend((f"V16_DRIVER_CHILD_PID run_id={run_id} child_pid={pid} child_start_ticks={ticks}", f"V16_DRIVER_PAYLOAD_RELEASE_ARMED run_id={run_id} child_pid={pid} child_start_ticks={ticks}", f"V16_DRIVER_PAYLOAD_RELEASED run_id={run_id} child_pid={pid} child_start_ticks={ticks}", f"V16_DRIVER_EXIT run_id={run_id} child_pid={pid} child_start_ticks={ticks} exit_code={code}", f"V16_DRIVER_RESULT_WRITTEN run_id={run_id} child_pid={pid} child_start_ticks={ticks} exit_code={code}"))
    positions = [text.find(marker) for marker in markers]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise DispatchV16Error("v16 terminal stream markers are missing or ill ordered")


def _output_inventory(path: Path, *, frozen: bool) -> list[Mapping[str, Any]]:
    if _kind(path) != "directory" or (frozen and stat.S_IMODE(path.stat().st_mode) != 0o555):
        raise DispatchV16Error("v16 output directory is missing or not frozen")
    entries: list[Mapping[str, Any]] = []
    for base, directories, files in os.walk(path, topdown=False, followlinks=False):
        current = Path(base)
        for name in sorted(files):
            child = current / name
            _read(child, label="v16 output artifact")
            if frozen and stat.S_IMODE(child.stat().st_mode) != 0o444:
                raise DispatchV16Error("v16 output file is mutable")
            entries.append({"path": str(child.relative_to(path)), "kind": "file", "mode": "0444", "sha256": _sha(child)})
        for name in sorted(directories):
            child = current / name
            if _kind(child) != "directory" or (frozen and stat.S_IMODE(child.stat().st_mode) != 0o555):
                raise DispatchV16Error("v16 output has unsafe directory")
            entries.append({"path": str(child.relative_to(path)), "kind": "directory", "mode": "0555"})
    entries.append({"path": ".", "kind": "directory", "mode": "0555"})
    return sorted(entries, key=lambda value: (str(value["path"]), str(value["kind"])))


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


def _control_check(item: Paths) -> Mapping[str, Any]:
    identical: dict[str, bool] = {}
    hashes: dict[str, Mapping[str, str]] = {}
    for name in PROTECTED:
        actual = _sha(item.output / name)
        expected = PREREGISTERED_PROTECTED_SHA256[name]
        identical[name] = actual == expected
        hashes[name] = {"preregistered_sha256": expected, "v16_sha256": actual}
    if not all(identical.values()):
        raise DispatchV16Error("v16 control differs from preregistered protected output hashes")
    run, timeline = _json(item.output / "run.json", label="v16 control run"), _json(item.output / "state-timeline.json", label="v16 control timeline")
    runtime, rows = run.get("runtime_seconds"), timeline.get("transactions")
    if run.get("status") != "succeeded" or run.get("state_policy", {}).get("name") != "always-commit" or type(runtime) not in (int, float) or not math.isfinite(float(runtime)) or not isinstance(rows, list) or len(rows) != 30:
        raise DispatchV16Error("v16 control metadata is incomplete")
    ratio = float(runtime) / PREREGISTERED_CONTROL_RUNTIME_SECONDS
    if ratio > RUNTIME_RATIO_LIMIT:
        raise DispatchV16Error("v16 control exceeds the preregistered runtime limit")
    for frame_id, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("frame_id") != frame_id or row.get("action") != "commit" or row.get("reason") != "always_commit_control" or row.get("current_alarm") is not False or row.get("pending_transaction_count") != 0 or row.get("pressure") is not None:
            raise DispatchV16Error("v16 control is not thirty direct native commits")
    return {"status": "PASS", "protected_model_outputs_byte_identical": identical, "protected_model_output_sha256": hashes, "direct_commits": 30, "pressure_operator_constructed": False, "runtime_seconds": float(runtime), "preregistered_control_runtime_seconds": PREREGISTERED_CONTROL_RUNTIME_SECONDS, "runtime_ratio": ratio, "runtime_ratio_limit": RUNTIME_RATIO_LIMIT}


def _candidate_check(item: Paths) -> Mapping[str, Any]:
    run, timeline = _json(item.output / "run.json", label="v16 candidate run"), _json(item.output / "state-timeline.json", label="v16 candidate timeline")
    rows, runtime = timeline.get("transactions"), run.get("runtime_seconds")
    if run.get("status") != "succeeded" or run.get("state_policy", {}).get("name") != POLICIES[CANDIDATE] or not isinstance(rows, list) or len(rows) != 30 or type(runtime) not in (int, float) or not math.isfinite(float(runtime)):
        raise DispatchV16Error("v16 candidate metadata is incomplete")
    ratio = float(runtime) / PREREGISTERED_CONTROL_RUNTIME_SECONDS
    if ratio > RUNTIME_RATIO_LIMIT:
        raise DispatchV16Error("v16 candidate exceeds the preregistered runtime limit")
    alarms = [row for row in rows if isinstance(row, Mapping) and row.get("current_alarm") is True]
    if not alarms:
        raise DispatchV16Error("v16 candidate had no real alarm and is a no-op")
    for frame_id, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("frame_id") != frame_id or row.get("pending_transaction_count") != 0:
            raise DispatchV16Error("v16 candidate transaction sequence differs")
        if row.get("current_alarm") is not True and row.get("action") != "commit":
            raise DispatchV16Error("v16 clear candidate frame is not a direct commit")
    for row in alarms:
        pressure = row.get("pressure")
        if row.get("action") != "bounded_update_pressure_injection" or row.get("reason") != "current_online_detector_alarm" or row.get("detector_history_action") != "commit_raw_current_health" or not isinstance(pressure, Mapping) or pressure.get("operator") != "bounded_native_update_pressure_v16" or pressure.get("increment") != 0.25 or pressure.get("cap") != 1.0 or not all(pressure.get(name) is True for name in ("raw_current_state_gpu_fingerprint_unchanged", "raw_current_mem_gpu_fingerprint_unchanged", "raw_current_pose_gpu_fingerprint_unchanged")):
            raise DispatchV16Error("v16 candidate alarm witness is incomplete")
    control = paths(CONTROL).output
    changed = _read(item.output / "trajectory.json", label="v16 candidate trajectory") != _read(control / "trajectory.json", label="v16 control trajectory") or _read(item.output / "predictions-summary.json", label="v16 candidate predictions") != _read(control / "predictions-summary.json", label="v16 control predictions")
    if not changed:
        raise DispatchV16Error("v16 candidate caused no later raw trajectory/prediction change")
    return {"status": "PASS", "alarm_count": len(alarms), "later_raw_output_changed": True, "runtime_seconds": float(runtime), "preregistered_control_runtime_seconds": PREREGISTERED_CONTROL_RUNTIME_SECONDS, "runtime_ratio": ratio, "runtime_ratio_limit": RUNTIME_RATIO_LIMIT}


def _validate_driver(item: Paths, run_id: str, command: Sequence[str], command_hash: str) -> str:
    source = _read(item.driver, label="v16 driver").decode("utf-8", errors="strict")
    tokens = [line.split("=", 1)[1] for line in source.splitlines() if line.startswith("go_token=")]
    if len(tokens) != 1 or len(tokens[0]) != 64 or any(ch not in "0123456789abcdef" for ch in tokens[0]) or source != _driver(run_id, item, command, command_hash, tokens[0]):
        raise DispatchV16Error("v16 frozen driver differs from its token-gated template")
    return hashlib.sha256((tokens[0] + "\n").encode("ascii")).hexdigest()


def _validator(item: Paths, run_id: str) -> Mapping[str, Any]:
    report: dict[str, Any] = {"schema_version": "stateguard3r.v16-validator.v1", "run_id": run_id, "cpu_only": True, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "started_at": datetime.now().astimezone().isoformat()}
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise DispatchV16Error("v16 validator must run with CUDA hidden")
        primary = (item.journal, item.preflight, item.main, item.transcript, item.driver, item.postflight)
        for path in primary:
            if _kind(path) != "regular" or stat.S_IMODE(path.stat().st_mode) != 0o444:
                raise DispatchV16Error("v16 sealed primary artifact is absent or mutable")
            _read(path, label="v16 sealed primary artifact")
        preflight, postflight = _json(item.preflight, label="v16 preflight"), _json(item.postflight, label="v16 postflight")
        command = preflight.get("command")
        if preflight.get("schema_version") != "stateguard3r.v16-preflight.v1" or preflight.get("run_id") != run_id or not isinstance(command, list) or any(type(part) is not str for part in command):
            raise DispatchV16Error("v16 preflight schema is invalid")
        command_hash = _command_hash(command)
        if preflight.get("command_sha256") != command_hash or command != _command(run_id, item):
            raise DispatchV16Error("v16 preflight command does not recompute")
        _production_contract(run_id, item, command)
        snapshots = preflight.get("gpu_preflights")
        if not isinstance(snapshots, list) or len(snapshots) != 2:
            raise DispatchV16Error("v16 preflight lacks two snapshots")
        for snapshot in snapshots:
            _validate_snapshot(snapshot, require_minimum=True)
        source = _source()
        if preflight.get("source_provenance") != source or postflight.get("source_provenance") != source:
            raise DispatchV16Error("v16 source provenance no longer recomputes")
        token_sha = _validate_driver(item, run_id, command, command_hash)
        if _kind(item.result) != "regular" or stat.S_IMODE(item.result.stat().st_mode) != 0o444:
            raise DispatchV16Error("v16 result is absent or mutable")
        result = _result(item.result, run_id=run_id, command_hash=command_hash)
        _ordered(item.main, run_id=run_id, command_hash=command_hash, result=result, transcript=False)
        _ordered(item.transcript, run_id=run_id, command_hash=command_hash, result=result, transcript=True)
        if postflight.get("result") != result or postflight.get("go_token_sha256") != token_sha or not postflight.get("pipe_close_and_drain", {}).get("drained") or "gpu_postflight_error" in postflight:
            raise DispatchV16Error("v16 postflight evidence is incomplete")
        _validate_snapshot(postflight.get("gpu_postflight", {}), require_minimum=False)
        identity = postflight.get("pid_start_time_check")
        if not isinstance(identity, Mapping) or identity.get("expected_start_ticks") != result["child_start_ticks"] or identity.get("pair_absent") is not True or identity.get("pid_reused") is not False:
            raise DispatchV16Error("v16 postflight lacks dead PID/start identity proof")
        if result["exit_code"] != 0:
            report["status"] = "CHILD_NONZERO_EVIDENCE_COMPLETE"
        else:
            inventory = _output_inventory(item.output, frozen=True)
            if postflight.get("output_inventory") != inventory:
                raise DispatchV16Error("v16 frozen output inventory does not recompute")
            if run_id == CONTROL:
                report["dynamic_always_control"] = _control_check(item)
            else:
                report["dynamic_candidate"] = _candidate_check(item)
            report["status"] = "PASS"
        report["artifact_sha256"] = {path.name: _sha(path) for path in (*primary, item.result)}
    except Exception as error:
        report.update({"status": "FAIL", "error": str(error), "finished_at": datetime.now().astimezone().isoformat()})
        _write_json(item.validator, report)
        _freeze(item.validator)
        raise DispatchV16Error(f"fresh CUDA-hidden v16 validator failed: {error}") from error
    report["finished_at"] = datetime.now().astimezone().isoformat()
    _write_json(item.validator, report)
    _freeze(item.validator)
    return report


def validate_sealed_run(run_id: str) -> Mapping[str, Any]:
    item = paths(run_id)
    _canonical(item)
    if _kind(item.lease) != "directory" or _lexists(item.validator):
        raise DispatchV16Error("v16 validator requires its owner lease and a fresh report")
    return _validator(item, run_id)


def _fresh_validator(run_id: str) -> Mapping[str, Any]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--validate-run-id", run_id], text=True, capture_output=True, env=environment)
    if result.returncode != 0:
        raise DispatchV16Error(f"v16 CUDA-hidden validator subprocess failed: {result.stderr.strip() or result.stdout.strip()}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise DispatchV16Error("v16 CUDA-hidden validator emitted invalid JSON") from error
    if not isinstance(value, Mapping):
        raise DispatchV16Error("v16 CUDA-hidden validator did not emit an object")
    return value


def _postflight(item: Paths, *, run_id: str, result: Mapping[str, Any] | None, source: Mapping[str, Any], pipe: Mapping[str, Any], error: str | None, token_sha: str | None) -> Mapping[str, Any]:
    value: dict[str, Any] = {"schema_version": "stateguard3r.v16-postflight.v1", "run_id": run_id, "result": result, "source_provenance": source, "pipe_close_and_drain": pipe, "error": error, "go_token_sha256": token_sha}
    try:
        value["gpu_postflight"] = _snapshot(require_minimum=False)
    except Exception as caught:
        value["gpu_postflight_error"] = str(caught)
    return value


def _freeze_primary(item: Paths) -> None:
    for path in (item.journal, item.preflight, item.main, item.transcript, item.driver, item.result, item.postflight):
        if _lexists(path):
            _freeze(path)


def dispatch(run_id: str, *, timeout_seconds: float) -> Mapping[str, Any]:
    if timeout_seconds <= 0:
        raise DispatchV16Error("v16 timeout must be positive")
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
        _write_json(item.preflight, {"schema_version": "stateguard3r.v16-preflight.v1", "run_id": run_id, "command": command, "command_sha256": command_hash, "gpu_preflights": [first, second], "source_provenance": source})
        _freeze(item.preflight)
        _write_new(item.main, "")
        _write_new(item.transcript, "")
        _write_new(item.driver, _driver(run_id, item, command, command_hash, token), 0o555)
        pane, deadline = _pane(), time.monotonic() + timeout_seconds
        _journal(item, "TMUX_PANE_CREATED", pane=pane)
        _tmux(["pipe-pane", "-o", "-t", pane, f"cat >> {shlex.quote(str(item.transcript))}"])
        ready = f"V16_PIPE_READY run_id={run_id}"
        _tmux(["send-keys", "-t", pane, f"printf '%s\\n' {shlex.quote(ready)}", "C-m"])
        _wait_text(item.transcript, ready, deadline)
        _journal(item, "PIPE_READY", pane=pane)
        _tmux(["send-keys", "-t", pane, f"bash {shlex.quote(str(item.driver))}; exit", "C-m"])
        _journal(item, "DRIVER_RELEASED", pane=pane)
        try:
            result = _wait_result(item, run_id, command_hash, deadline)
        except _PreexecV16NoGo as event:
            _wait_pane_exit(pane, time.monotonic() + 10.0)
            pipe = _drain(pane, item)
            raise DispatchV16Error(f"v16 pre-execution identity NO-GO: {event.failure}") from event
        except DispatchV16Error as timeout:
            identity = _partial_identity(item, run_id)
            if identity is None:
                raise DispatchV16Error("v16 timeout without a proven PID/start identity; refusing to seal possible live CUDA") from timeout
            _terminate_owned(identity)
            result = _wait_result(item, run_id, command_hash, time.monotonic() + 10.0)
        _wait_pane_exit(pane, deadline)
        pipe = _drain(pane, item)
        _ordered(item.main, run_id=run_id, command_hash=command_hash, result=result, transcript=False)
        _ordered(item.transcript, run_id=run_id, command_hash=command_hash, result=result, transcript=True)
        identity = _pid_absent(result)
        if not identity["pair_absent"] or identity["pid_reused"]:
            raise DispatchV16Error("v16 PID/start pair is live or reused after driver exit")
        post_source = _source()
        if post_source != source:
            raise DispatchV16Error("v16 source drifted during child execution")
        postflight = _postflight(item, run_id=run_id, result=result, source=post_source, pipe=pipe, error=None, token_sha=token_sha)
        if "gpu_postflight_error" in postflight:
            raise DispatchV16Error("v16 postflight GPU snapshot failed")
        _validate_snapshot(postflight["gpu_postflight"], require_minimum=False)
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
                raise DispatchV16Error("v16 independent validator did not pass")
        except Exception as error:
            caught = error
    if caught is not None:
        if not _lexists(item.validator):
            _write_json(item.validator, {"schema_version": "stateguard3r.v16-dispatch-terminal.v1", "run_id": run_id, "status": "FAIL", "error_class": type(caught).__name__, "error": str(caught), "finished_at": datetime.now().astimezone().isoformat()})
            _freeze(item.validator)
        raise DispatchV16Error(f"v16 terminal evidence sealed as failure: {caught}") from caught
    assert result is not None
    if result["exit_code"] != 0:
        raise DispatchV16Error(f"v16 child exited nonzero with immutable evidence: {result['exit_code']}")
    return {"status": "PASS", "run_id": run_id, "pane": pane, "command_sha256": command_hash, "validator": validator, "paths": {name: str(path) for name, path in asdict(item).items()}}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.validate_run_id is not None:
            print(json.dumps(validate_sealed_run(args.validate_run_id), ensure_ascii=False, sort_keys=True))
        else:
            print(json.dumps(dispatch(args.run_id, timeout_seconds=args.timeout_seconds), ensure_ascii=False, sort_keys=True))
        return 0
    except DispatchV16Error as error:
        print(f"V16_DISPATCH_ERROR={error}", file=sys.stderr)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
