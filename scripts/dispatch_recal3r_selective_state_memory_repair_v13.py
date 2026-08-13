#!/usr/bin/env python3
"""One-use terminal-evidence dispatcher for the distinct v13 GPU route.

This is intentionally a v13-owned implementation: it does not import or call
an earlier recovery dispatcher.  A successful forward is meaningful only when its original ordered
preflight, tmux transcript, child identity, postflight, and independent CPU
validator all exist.  It has no rerun, repair, or post-hoc sealing command.
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
SESSION, GPU_INDEX = "stateguard", "2"
GPU_UUID, MIN_FREE_MIB = "GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250", 12288

# The v13 runner is a pinned input.  Its exact name is part of provenance, so
# a renamed or substituted child fails closed rather than loosening argv checks.
RUNNER_RELATIVE = Path("scripts/run_recal3r_selective_state_memory_repair_v13.py")
PINNED_COMPONENTS = (
    RUNNER_RELATIVE,
    Path("src/stateguard3r/recal3r_selective_state_memory_repair_runner_v13.py"),
    Path("src/stateguard3r/selective_state_memory_repair_v13.py"),
    Path("scripts/dispatch_recal3r_selective_state_memory_repair_v13.py"),
    Path("scripts/wait_and_dispatch_recal3r_selective_state_memory_repair_v13_control.sh"),
)
RECAL3R_ROOT = ROOT.parent / "baselines" / "ReCal3R"
RECAL3R_COMMIT = "466c7cdf3acd2f589f1d82e5f6391966f19db9ff"
CHECKPOINT = RECAL3R_ROOT / "src" / "cut3r_512_dpt_4_64.pth"
CHECKPOINT_SHA256 = "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103"
DETECTOR_CONFIG = ROOT / "outputs" / "formal-v3-calibration-0001" / "formal-config.json"
DEVELOPMENT_MANIFESTS = tuple(
    ROOT / "outputs" / "formal-v1-inputs-0001" / "development" / f"development-{condition}" / "input-manifest.json"
    for condition in ("dynamic", "wrong", "low")
)
CONTROL_RUN_ID = "recovery-selective-state-memory-repair-v13-dynamic-always-commit-0001"
CANDIDATE_RUN_ID = "recovery-selective-state-memory-repair-v13-dynamic-candidate-0001"
CANDIDATE_POLICY = "detector-v3-incremental-selective-state-memory-repair"
RUN_SPECS = {
    CONTROL_RUN_ID: ("dynamic", "always-commit"),
    CANDIDATE_RUN_ID: ("dynamic", CANDIDATE_POLICY),
}
PINNED_EXTERNALS = (
    CHECKPOINT,
    RECAL3R_ROOT / "src" / "dust3r" / "model.py",
    RECAL3R_ROOT / "src" / "dust3r" / "utils" / "image.py",
    DETECTOR_CONFIG,
    *DEVELOPMENT_MANIFESTS,
)
V1_DYNAMIC_CONTROL = ROOT / "outputs" / "recovery-policy-development-v1-dynamic-always-commit-0001"
PROTECTED_MODEL_OUTPUTS = (
    "checkpoint-load-audit.json", "health.jsonl", "predictions-summary.json", "trajectory.json",
)
RUNTIME_RATIO_LIMIT = 1.20


class V13DispatchError(RuntimeError):
    """No valid single-owner v13 dispatch/evidence path exists."""


class V13PreexecNoGo(V13DispatchError):
    """The token-gated wrapper died before it could release a payload."""

    def __init__(self, failure: Mapping[str, Any]) -> None:
        super().__init__("driver reached a pre-execution identity NO-GO")
        self.failure = failure


@dataclass(frozen=True)
class ArtifactPaths:
    output: Path
    preflight: Path
    main: Path
    postflight: Path
    transcript: Path
    driver: Path
    result: Path
    validator: Path
    lease: Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--command-json", type=Path)
    commands.add_argument("--validate-run-id")
    parser.add_argument("--run-id")
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    return parser


def _run_id(run_id: str) -> None:
    valid = set("abcdefghijklmnopqrstuvwxyz0123456789-")
    if type(run_id) is not str or not run_id.startswith("recovery-selective-state-memory-repair-v13-") or not run_id or any(ch not in valid for ch in run_id):
        raise V13DispatchError("run id is not a canonical v13 one-use id")


def artifact_paths(run_id: str) -> ArtifactPaths:
    _run_id(run_id)
    return ArtifactPaths(
        output=ROOT / "outputs" / run_id,
        preflight=ROOT / "logs" / f"{run_id}-preflight.json",
        main=ROOT / "logs" / f"{run_id}-main.log",
        postflight=ROOT / "logs" / f"{run_id}-postflight.json",
        transcript=ROOT / "logs" / f"{run_id}-tmux-transcript.log",
        driver=ROOT / "logs" / f"{run_id}-driver.sh",
        result=ROOT / "logs" / f"{run_id}-result.json",
        validator=ROOT / "logs" / f"{run_id}-validator-0001.log",
        lease=ROOT / "tmp" / f"{run_id}.owner-lease",
    )


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _lstat_kind(path: Path) -> str:
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


def _direct_directory(path: Path, label: str) -> None:
    if _lstat_kind(path) != "directory":
        raise V13DispatchError(f"{label} parent must be an existing non-symlink directory")


def _canonical_paths(paths: ArtifactPaths) -> None:
    expected = {
        "output": ROOT / "outputs" / paths.output.name,
        "preflight": ROOT / "logs" / paths.preflight.name,
        "main": ROOT / "logs" / paths.main.name,
        "postflight": ROOT / "logs" / paths.postflight.name,
        "transcript": ROOT / "logs" / paths.transcript.name,
        "driver": ROOT / "logs" / paths.driver.name,
        "result": ROOT / "logs" / paths.result.name,
        "validator": ROOT / "logs" / paths.validator.name,
        "lease": ROOT / "tmp" / paths.lease.name,
    }
    for name, wanted in expected.items():
        actual = getattr(paths, name)
        if actual != wanted or actual.parent != wanted.parent:
            raise V13DispatchError(f"{name} path is not a direct canonical child")
    for directory, label in ((ROOT / "outputs", "outputs"), (ROOT / "logs", "logs"), (ROOT / "tmp", "tmp")):
        _direct_directory(directory, label)


def _read_regular_nul_free(path: Path, *, label: str) -> bytes:
    if _lstat_kind(path) != "regular":
        raise V13DispatchError(f"{label} is missing, a symlink, or nonregular: {path}")
    data = path.read_bytes()
    if b"\0" in data:
        raise V13DispatchError(f"{label} contains NUL: {path}")
    return data


def _write(path: Path, data: str, mode: int = 0o644) -> None:
    if "\0" in data or _lexists(path):
        raise V13DispatchError(f"refusing to overwrite or NUL-write one-use artifact: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)


def _write_json(path: Path, value: Any, mode: int = 0o644) -> None:
    _write(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", mode)


def _freeze(path: Path) -> None:
    _read_regular_nul_free(path, label="terminal artifact")
    os.chmod(path, 0o444)
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise V13DispatchError(f"could not freeze terminal artifact: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_inventory(path: Path, *, frozen: bool) -> list[dict[str, Any]]:
    if _lstat_kind(path) != "directory":
        raise V13DispatchError("child did not produce a direct regular output directory")
    if frozen and stat.S_IMODE(path.stat().st_mode) != 0o555:
        raise V13DispatchError("output root mode is not frozen 0555")
    items: list[dict[str, Any]] = []
    for current_text, directories, files in os.walk(path, topdown=False, followlinks=False):
        current = Path(current_text)
        for name in sorted(files):
            child = current / name
            if _lstat_kind(child) != "regular":
                raise V13DispatchError(f"output contains symlink or special artifact: {child}")
            _read_regular_nul_free(child, label="output artifact")
            if frozen and stat.S_IMODE(child.stat().st_mode) != 0o444:
                raise V13DispatchError(f"output artifact mode is not frozen 0444: {child}")
            items.append({"path": str(child.relative_to(path)), "kind": "file", "mode": "0444", "sha256": _sha256_file(child)})
        for name in sorted(directories):
            child = current / name
            if _lstat_kind(child) != "directory":
                raise V13DispatchError(f"output contains symlink or special directory: {child}")
            if frozen and stat.S_IMODE(child.stat().st_mode) != 0o555:
                raise V13DispatchError(f"output directory mode is not frozen 0555: {child}")
            items.append({"path": str(child.relative_to(path)), "kind": "directory", "mode": "0555"})
    items.append({"path": ".", "kind": "directory", "mode": "0555"})
    return sorted(items, key=lambda item: (item["path"], item["kind"]))


def _freeze_output(path: Path) -> list[dict[str, Any]]:
    _output_inventory(path, frozen=False)
    for current_text, directories, files in os.walk(path, topdown=False, followlinks=False):
        current = Path(current_text)
        for name in files:
            os.chmod(current / name, 0o444)
        for name in directories:
            os.chmod(current / name, 0o555)
    os.chmod(path, 0o555)
    return _output_inventory(path, frozen=True)


def _fresh(paths: ArtifactPaths) -> None:
    _canonical_paths(paths)
    existing = [name for name, value in asdict(paths).items() if _lexists(value)]
    if existing:
        raise V13DispatchError(f"a one-use output or terminal artifact path already exists: {existing}")


def _acquire_lease(paths: ArtifactPaths) -> None:
    _canonical_paths(paths)
    try:
        paths.lease.mkdir(mode=0o700)
    except FileExistsError as error:
        raise V13DispatchError("same-ID owner lease already exists") from error
    if _lstat_kind(paths.lease) != "directory" or list(paths.lease.iterdir()):
        raise V13DispatchError("same-ID owner lease is unsafe")


def _component_provenance() -> Mapping[str, Any]:
    """Bind every v13 component and fixed external to clean HEAD bytes."""
    try:
        if subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], check=True, text=True, capture_output=True).stdout.strip():
            raise V13DispatchError("StateGuard3R tracked worktree is not clean")
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
        components: dict[str, str] = {}
        for relative in PINNED_COMPONENTS:
            path = ROOT / relative
            subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", str(relative)], check=True, text=True, capture_output=True)
            head = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{relative}"], check=True, capture_output=True).stdout
            current = path.read_bytes()
            if hashlib.sha256(head).digest() != hashlib.sha256(current).digest():
                raise V13DispatchError(f"pinned component differs from HEAD: {relative}")
            components[str(relative)] = hashlib.sha256(current).hexdigest()
        if subprocess.run(["git", "-C", str(RECAL3R_ROOT), "status", "--porcelain"], check=True, text=True, capture_output=True).stdout.strip():
            raise V13DispatchError("ReCal3R tracked worktree is not clean")
        recal_commit = subprocess.run(["git", "-C", str(RECAL3R_ROOT), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
        if recal_commit != RECAL3R_COMMIT:
            raise V13DispatchError("ReCal3R commit differs from preregistered revision")
    except subprocess.CalledProcessError as error:
        raise V13DispatchError("cannot inspect pinned source provenance") from error
    external: dict[str, str] = {}
    for path in PINNED_EXTERNALS:
        if _lstat_kind(path) != "regular":
            raise V13DispatchError(f"pinned external input is missing or unsafe: {path}")
        external[str(path)] = _sha256_file(path)
    if external.get(str(CHECKPOINT)) != CHECKPOINT_SHA256:
        raise V13DispatchError("pinned checkpoint SHA-256 differs")
    return {"stateguard_commit": commit, "recal3r_commit": recal_commit, "component_sha256": components, "external_sha256": external}


def _project_gpu_processes() -> list[Mapping[str, Any]]:
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True)
    project: list[Mapping[str, Any]] = []
    for line in result.stdout.splitlines():
        row = [item.strip() for item in line.split(",")]
        if len(row) != 4 or row[0] != GPU_UUID or not row[1].isdigit():
            continue
        pid = int(row[1])
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        except OSError:
            command = "<unavailable>"
        if str(ROOT) in command or str(RECAL3R_ROOT) in command:
            project.append({"gpu_uuid": row[0], "pid": pid, "process_name": row[2], "used_memory_mib": row[3], "command": command})
    return project


def _snapshot(*, require_minimum: bool = True) -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    rows = []
    for line in result.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) == 2 and values[1].isdigit():
            rows.append({"uuid": values[0], "memory_free_mib": int(values[1])})
    selected = [row for row in rows if row["uuid"] == GPU_UUID]
    project = _project_gpu_processes()
    if len(selected) != 1 or (require_minimum and selected[0]["memory_free_mib"] < MIN_FREE_MIB):
        raise V13DispatchError("GPU 2 UUID/free-memory precondition failed")
    if project:
        raise V13DispatchError("a StateGuard3R/ReCal3R project process is already on GPU 2")
    return {"timestamp": datetime.now().astimezone().isoformat(), "command": command, "rows": rows, "selected": selected[0], "project_gpu2_processes": project}


def _validate_snapshot(snapshot: Mapping[str, Any], *, require_minimum: bool = True) -> None:
    selected = snapshot.get("selected")
    if not isinstance(selected, Mapping) or selected.get("uuid") != GPU_UUID or type(selected.get("memory_free_mib")) is not int or (require_minimum and selected["memory_free_mib"] < MIN_FREE_MIB):
        raise V13DispatchError("stored GPU preflight snapshot is invalid")
    if not isinstance(snapshot.get("timestamp"), str) or not snapshot["timestamp"] or snapshot.get("project_gpu2_processes") != []:
        raise V13DispatchError("stored GPU snapshot lacks timestamp/project-process absence")


def _flag(argv: Sequence[str], name: str) -> str:
    matches = [index for index, item in enumerate(argv) if item == name]
    if len(matches) != 1 or matches[0] + 1 >= len(argv):
        raise V13DispatchError(f"child must contain exactly one {name}")
    return argv[matches[0] + 1]


def _path(value: str) -> Path:
    raw = Path(value)
    return (ROOT / raw).resolve(strict=False) if not raw.is_absolute() else raw.resolve(strict=False)


def _gate_b_predecessor(run_id: str) -> None:
    if run_id == CONTROL_RUN_ID:
        return
    if run_id != CANDIDATE_RUN_ID:
        raise V13DispatchError("v13 wrong/low runs remain forbidden until Gate C authorization")
    control = artifact_paths(CONTROL_RUN_ID)
    if _lstat_kind(control.validator) != "regular" or stat.S_IMODE(control.validator.stat().st_mode) != 0o444:
        raise V13DispatchError("dynamic candidate is blocked until frozen dynamic control validator PASS exists")
    report = _json(control.validator, label="frozen dynamic control validator")
    if report.get("status") != "PASS" or report.get("dynamic_always_control", {}).get("status") != "PASS":
        raise V13DispatchError("dynamic candidate is blocked because dynamic control did not pass")


def _production_contract(argv: Sequence[str], paths: ArtifactPaths) -> None:
    flags = (
        "--baseline-root", "--checkpoint", "--checkpoint-sha256", "--input-manifest", "--output-dir", "--device", "--size", "--seed", "--beta-base", "--health-profile", "--rgb-timestamp-listing", "--timestamp-dataset-root", "--state-policy", "--detector-config", "--watchdog",
    )
    if len(argv) != 2 + 2 * len(flags) or tuple(argv[2::2]) != flags:
        raise V13DispatchError("production child argv is not canonical v13 flag order")
    values = {argv[index]: argv[index + 1] for index in range(2, len(argv), 2)}
    spec = RUN_SPECS.get(paths.output.name)
    if spec is None:
        raise V13DispatchError("production v13 run ID is not preregistered")
    condition, policy = spec
    rgb = RECAL3R_ROOT / "data" / "tum" / "rgbd_dataset_freiburg1_desk" / "rgb.txt"
    expected_manifest = ROOT / "outputs" / "formal-v1-inputs-0001" / "development" / f"development-{condition}" / "input-manifest.json"
    checks = (
        _path(values["--baseline-root"]) == RECAL3R_ROOT.resolve(strict=False),
        _path(values["--checkpoint"]) == CHECKPOINT.resolve(strict=False),
        values["--checkpoint-sha256"] == CHECKPOINT_SHA256,
        _path(values["--input-manifest"]) == expected_manifest.resolve(strict=False),
        values["--output-dir"] == str(paths.output), values["--device"] == "cuda", values["--size"] == "512",
        values["--seed"] == "0", values["--beta-base"] == "0.1", values["--health-profile"] == "v3",
        _path(values["--rgb-timestamp-listing"]) == rgb.resolve(strict=False),
        _path(values["--timestamp-dataset-root"]) == rgb.parent.resolve(strict=False),
        values["--state-policy"] == policy, _path(values["--detector-config"]) == DETECTOR_CONFIG.resolve(strict=False), values["--watchdog"] == "8",
    )
    if not all(checks):
        raise V13DispatchError("production child differs from pinned v13 contract")


def _child(argv: Sequence[str], paths: ArtifactPaths, *, test: bool) -> list[str]:
    if not argv or any(type(item) is not str or not item or "\0" in item or "\n" in item or "\r" in item for item in argv):
        raise V13DispatchError("child argv is not a clean nonempty string array")
    if len(argv) < 2 or argv[1] not in {str(RUNNER_RELATIVE), str(ROOT / RUNNER_RELATIVE)}:
        raise V13DispatchError("child is not the exact pinned v13 runner")
    if _flag(argv, "--output-dir") != str(paths.output) or _flag(argv, "--state-policy") not in {"always-commit", CANDIDATE_POLICY}:
        raise V13DispatchError("child output or state policy is not pinned")
    device = _flag(argv, "--device")
    if (not test and (argv[0] != str(RECAL3R_ROOT / ".venv" / "bin" / "python") or device != "cuda")) or (test and device not in {"cpu", "cuda"}):
        raise V13DispatchError("child interpreter/device does not satisfy v13 contract")
    child = list(argv)
    if not test:
        _production_contract(child, paths)
        _gate_b_predecessor(paths.output.name)
    return child


def _command_hash(argv: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(argv), ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _interpreter_provenance(argv: Sequence[str], *, test: bool) -> Mapping[str, Any] | None:
    if test:
        return None
    executable = Path(argv[0])
    if _lstat_kind(executable) not in {"regular", "symlink"} or not executable.exists():
        raise V13DispatchError("pinned ReCal3R interpreter is unavailable")
    resolved = executable.resolve(strict=True)
    return {"path": str(executable), "resolved_path": str(resolved), "sha256": _sha256_file(resolved)}


def _driver(run_id: str, argv: Sequence[str], paths: ArtifactPaths, command_hash: str, quoted: str, token: str) -> str:
    if len(token) != 64 or any(ch not in "0123456789abcdef" for ch in token):
        raise V13DispatchError("driver go token is not 256-bit lowercase hex")
    values = (run_id, str(paths.main), str(paths.result), str(paths.lease), quoted, command_hash, token)
    run_q, main_q, result_q, lease_q, quoted_q, hash_q, token_q = (shlex.quote(item) for item in values)
    command = shlex.join(argv)
    return f'''#!/usr/bin/env bash
set +e
run_id={run_q}
main_log={main_q}
result_path={result_q}
lease_path={lease_q}
start_path="$lease_path/child-start"
start_tmp="$lease_path/child-start.tmp"
go_path="$lease_path/child-go"
command_quoted={quoted_q}
command_sha256={hash_q}
go_token={token_q}
record() {{ printf '%s\\n' "$1" | tee -a "$main_log"; }}
record "V13_DRIVER_START run_id=$run_id"
record "V13_DRIVER_COMMAND=$command_quoted"
record "V13_DRIVER_COMMAND_SHA256=$command_sha256"
record "V13_DRIVER_DISPATCHED run_id=$run_id kind=wrapper_only"
(
    gate_pid="$BASHPID"
    gate_start_ticks="$(awk '{{print $22}}' "/proc/$gate_pid/stat" 2>/dev/null)"
    if [[ ! "$gate_start_ticks" =~ ^[0-9]+$ ]]; then
        printf 'FAIL %s unavailable\\n' "$gate_pid" > "$start_tmp"; mv "$start_tmp" "$start_path"; exit 72
    fi
    printf 'READY %s %s\\n' "$gate_pid" "$gate_start_ticks" > "$start_tmp"; mv "$start_tmp" "$start_path"
    while true; do
        if [[ -e "$go_path" || -L "$go_path" ]]; then
            if [[ -L "$go_path" || ! -f "$go_path" ]]; then exit 73; fi
            if ! printf '%s\\n' "$go_token" | cmp -s - "$go_path"; then exit 73; fi
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
    kill "$child_pid" 2>/dev/null; wait "$child_pid"; child_exit=$?
    record "V13_DRIVER_PREEXEC_IDENTITY_FAILURE run_id=$run_id child_pid=$child_pid reason=start_gate_missing exit_code=$child_exit"; exit 70
fi
read -r kind reported_pid child_start_ticks < "$start_path"
if [[ "$kind" != READY || "$reported_pid" != "$child_pid" || ! "$child_start_ticks" =~ ^[0-9]+$ ]]; then
    wait "$child_pid"; child_exit=$?
    record "V13_DRIVER_PREEXEC_IDENTITY_FAILURE run_id=$run_id child_pid=$child_pid reason=start_time_unavailable exit_code=$child_exit"; exit 70
fi
driver_ticks="$(awk '{{print $22}}' "/proc/$child_pid/stat" 2>/dev/null)"
if [[ "$driver_ticks" != "$child_start_ticks" ]]; then
    kill "$child_pid" 2>/dev/null; wait "$child_pid"; child_exit=$?
    record "V13_DRIVER_PREEXEC_IDENTITY_FAILURE run_id=$run_id child_pid=$child_pid reason=parent_start_time_mismatch exit_code=$child_exit"; exit 70
fi
record "V13_DRIVER_CHILD_PID run_id=$run_id child_pid=$child_pid child_start_ticks=$child_start_ticks"
record "V13_DRIVER_PAYLOAD_RELEASE_ARMED run_id=$run_id child_pid=$child_pid child_start_ticks=$child_start_ticks"
if ! (set -o noclobber; printf '%s\\n' "$go_token" > "$go_path") 2>/dev/null; then
    kill "$child_pid" 2>/dev/null; wait "$child_pid"; child_exit=$?
    record "V13_DRIVER_PREEXEC_IDENTITY_FAILURE run_id=$run_id child_pid=$child_pid reason=go_gate_create_failed exit_code=$child_exit"; exit 70
fi
record "V13_DRIVER_PAYLOAD_RELEASED run_id=$run_id child_pid=$child_pid child_start_ticks=$child_start_ticks"
wait "$child_pid"; child_exit=$?
reaped_at="$(date --iso-8601=seconds)"
record "V13_DRIVER_EXIT run_id=$run_id child_pid=$child_pid child_start_ticks=$child_start_ticks exit_code=$child_exit reaped_at=$reaped_at"
if ! (set -o noclobber; : > "$result_path") 2>/dev/null; then record "V13_DRIVER_RESULT_CREATE_FAILED run_id=$run_id"; exit 70; fi
printf '{{"schema_version":"stateguard3r.v13-driver-result.v1","run_id":"%s","command_sha256":"%s","child_pid":%s,"child_start_ticks":"%s","exit_code":%s,"reaped_at":"%s"}}\\n' "$run_id" "$command_sha256" "$child_pid" "$child_start_ticks" "$child_exit" "$reaped_at" >> "$result_path"
record "V13_DRIVER_RESULT_WRITTEN run_id=$run_id child_pid=$child_pid child_start_ticks=$child_start_ticks exit_code=$child_exit"
exit 0
'''


def _tmux(args: Sequence[str], *, check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["tmux", *args], check=check, text=True, capture_output=capture_output)


def _pane() -> str:
    try:
        _tmux(["has-session", "-t", SESSION])
        pane = _tmux(["new-window", "-d", "-P", "-F", "#{pane_id}", "-t", SESSION, "-c", str(ROOT)]).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise V13DispatchError("existing stateguard tmux session is unavailable") from error
    if not pane.startswith("%"):
        raise V13DispatchError("tmux did not return a fresh pane id")
    return pane


def _wait_for_text(path: Path, marker: str, deadline: float) -> None:
    while time.monotonic() < deadline:
        if _lexists(path) and marker in _read_regular_nul_free(path, label="tmux transcript").decode("utf-8", errors="strict"):
            return
        time.sleep(0.05)
    raise V13DispatchError(f"timed out waiting for ordered marker: {marker}")


def _pane_alive(pane: str) -> bool:
    return _tmux(["list-panes", "-t", pane, "-F", "#{pane_id}"], check=False).returncode == 0


def _wait_for_driver_exit(pane: str, deadline: float) -> None:
    while time.monotonic() < deadline:
        if not _pane_alive(pane):
            return
        time.sleep(0.05)
    raise V13DispatchError("driver pane did not exit after publishing its result")


def _close_pipe_and_drain(pane: str, transcript: Path, main: Path) -> Mapping[str, Any]:
    if _pane_alive(pane):
        raise V13DispatchError("driver pane is still live; terminal writers cannot be sealed")
    first_t, first_m = _read_regular_nul_free(transcript, label="tmux transcript"), _read_regular_nul_free(main, label="main log")
    time.sleep(0.15)
    second_t, second_m = _read_regular_nul_free(transcript, label="tmux transcript"), _read_regular_nul_free(main, label="main log")
    if first_t != second_t or first_m != second_m:
        raise V13DispatchError("terminal stream changed after driver pane closed")
    return {"pane_alive_at_close": False, "transcript_sha256": hashlib.sha256(second_t).hexdigest(), "main_sha256": hashlib.sha256(second_m).hexdigest(), "drained": True}


def _process_start_ticks(pid: int) -> str | None:
    try:
        tail = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return tail[19] if len(tail) > 19 and tail[19].isdigit() else None
    except (FileNotFoundError, OSError, IndexError):
        return None


def _pid_pair_absent(result: Mapping[str, Any]) -> Mapping[str, Any]:
    observed = _process_start_ticks(result["child_pid"])
    return {"child_pid": result["child_pid"], "expected_start_ticks": result["child_start_ticks"], "observed_start_ticks": observed, "pair_absent": observed is None, "pid_reused": observed is not None and observed != result["child_start_ticks"]}


def _partial_child_identity(path: Path, *, run_id: str) -> tuple[int, str] | None:
    """Read only the ordered identity marker needed for a timeout cleanup."""
    try:
        lines = _read_regular_nul_free(path, label="main log").decode("utf-8", errors="strict").splitlines()
    except (UnicodeDecodeError, V13DispatchError):
        return None
    prefix = f"V13_DRIVER_CHILD_PID run_id={run_id} child_pid="
    matches = [line for line in lines if line.startswith(prefix)]
    if len(matches) != 1:
        return None
    fields = dict(piece.split("=", 1) for piece in matches[0].split() if "=" in piece)
    if fields.get("run_id") != run_id or not fields.get("child_pid", "").isdigit() or not fields.get("child_start_ticks", "").isdigit():
        return None
    return int(fields["child_pid"]), fields["child_start_ticks"]


def _terminate_owned_child(identity: tuple[int, str], *, grace_seconds: float = 5.0) -> None:
    """Never signal a PID unless it still has the recorded start time."""
    pid, expected_ticks = identity
    if _process_start_ticks(pid) != expected_ticks:
        raise V13DispatchError("timeout cannot prove the recorded child PID/start-time identity")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        observed = _process_start_ticks(pid)
        if observed is None:
            return
        if observed != expected_ticks:
            raise V13DispatchError("child PID was reused while waiting for timeout termination")
        time.sleep(0.05)
    if _process_start_ticks(pid) == expected_ticks:
        os.kill(pid, signal.SIGKILL)


def _parse_result(path: Path, *, run_id: str, command_hash: str) -> Mapping[str, Any]:
    try:
        result = json.loads(_read_regular_nul_free(path, label="driver result"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V13DispatchError("driver result is not valid UTF-8 JSON") from error
    expected = {"schema_version", "run_id", "command_sha256", "child_pid", "child_start_ticks", "exit_code", "reaped_at"}
    if not isinstance(result, dict) or set(result) != expected or result.get("schema_version") != "stateguard3r.v13-driver-result.v1" or result.get("run_id") != run_id or result.get("command_sha256") != command_hash or type(result.get("child_pid")) is not int or result["child_pid"] <= 0 or not isinstance(result.get("child_start_ticks"), str) or not result["child_start_ticks"].isdigit() or type(result.get("exit_code")) is not int or not isinstance(result.get("reaped_at"), str) or not result["reaped_at"]:
        raise V13DispatchError("driver result is malformed or unbound")
    return result


def _preexec_failure(path: Path, *, run_id: str) -> Mapping[str, Any] | None:
    try:
        lines = _read_regular_nul_free(path, label="main log").decode("utf-8", errors="strict").splitlines()
    except (UnicodeDecodeError, V13DispatchError):
        return None
    marker = f"V13_DRIVER_PREEXEC_IDENTITY_FAILURE run_id={run_id} "
    found = [line for line in lines if line.startswith(marker)]
    if not found:
        return None
    if len(found) != 1 or any("V13_DRIVER_PAYLOAD_RELEASED" in line for line in lines):
        raise V13DispatchError("pre-exec identity evidence is ambiguous or released payload")
    fields = dict(piece.split("=", 1) for piece in found[0].split() if "=" in piece)
    if fields.get("run_id") != run_id or not fields.get("child_pid", "").isdigit() or fields.get("reason") not in {"start_gate_missing", "start_time_unavailable", "parent_start_time_mismatch", "go_gate_create_failed"} or not fields.get("exit_code", "").isdigit():
        raise V13DispatchError("pre-exec identity marker is malformed")
    return {"run_id": run_id, "child_pid": int(fields["child_pid"]), "reason": fields["reason"], "exit_code": int(fields["exit_code"])}


def _wait_for_result(path: Path, *, run_id: str, command_hash: str, deadline: float, main: Path) -> Mapping[str, Any]:
    last: Exception | None = None
    while time.monotonic() < deadline:
        failure = _preexec_failure(main, run_id=run_id)
        if failure is not None:
            raise V13PreexecNoGo(failure)
        if _lexists(path):
            try:
                return _parse_result(path, run_id=run_id, command_hash=command_hash)
            except V13DispatchError as error:
                last = error
        time.sleep(0.05)
    raise V13DispatchError(f"driver did not publish a complete exclusive result before timeout{': ' + str(last) if last else ''}")


def _ordered_stream(path: Path, *, run_id: str, quoted: str, command_hash: str, result: Mapping[str, Any] | None, transcript: bool) -> None:
    text = _read_regular_nul_free(path, label="terminal stream").decode("utf-8", errors="strict")
    markers = [
        f"V13_DRIVER_START run_id={run_id}", f"V13_DRIVER_COMMAND={quoted}", f"V13_DRIVER_COMMAND_SHA256={command_hash}",
        f"V13_DRIVER_DISPATCHED run_id={run_id} kind=wrapper_only",
    ]
    if transcript:
        markers.insert(0, f"V13_PIPE_READY run_id={run_id}")
    if result is None:
        failure = _preexec_failure(path, run_id=run_id)
        if failure is None:
            raise V13DispatchError("pre-exec terminal stream has no identity failure")
        markers.append(f"V13_DRIVER_PREEXEC_IDENTITY_FAILURE run_id={run_id} child_pid={failure['child_pid']} reason={failure['reason']} exit_code={failure['exit_code']}")
    else:
        pid, ticks, exit_code = result["child_pid"], result["child_start_ticks"], result["exit_code"]
        markers.extend((
            f"V13_DRIVER_CHILD_PID run_id={run_id} child_pid={pid} child_start_ticks={ticks}",
            f"V13_DRIVER_PAYLOAD_RELEASE_ARMED run_id={run_id} child_pid={pid} child_start_ticks={ticks}",
            f"V13_DRIVER_PAYLOAD_RELEASED run_id={run_id} child_pid={pid} child_start_ticks={ticks}",
            f"V13_DRIVER_EXIT run_id={run_id} child_pid={pid} child_start_ticks={ticks} exit_code={exit_code}",
            f"V13_DRIVER_RESULT_WRITTEN run_id={run_id} child_pid={pid} child_start_ticks={ticks} exit_code={exit_code}",
        ))
    positions = [text.find(marker) for marker in markers]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise V13DispatchError("terminal stream markers are missing or ill ordered")


def _json_value(path: Path, *, label: str) -> Any:
    try:
        value = json.loads(_read_regular_nul_free(path, label=label))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V13DispatchError(f"{label} is not valid UTF-8 JSON") from error
    return value


def _json(path: Path, *, label: str) -> Mapping[str, Any]:
    value = _json_value(path, label=label)
    if not isinstance(value, Mapping):
        raise V13DispatchError(f"{label} must be a JSON object")
    return value


def _postflight(paths: ArtifactPaths, *, run_id: str, result: Mapping[str, Any] | None, source: Mapping[str, Any], pipe: Mapping[str, Any], error: str | None, token_sha256: str | None = None, preexec: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    payload: dict[str, Any] = {"schema_version": "stateguard3r.v13-postflight.v1", "run_id": run_id, "result": result, "source_provenance": source, "pipe_close_and_drain": pipe, "error": error}
    try:
        payload["gpu_postflight"] = _snapshot(require_minimum=False)
    except Exception as snapshot_error:
        payload["gpu_postflight_error"] = str(snapshot_error)
    if preexec is not None:
        payload["preexec_no_go"] = {"terminal_kind": "PREEXEC_IDENTITY_NO_GO", "payload_not_executed": True, "failure": preexec, "driver_result_exists": _lexists(paths.result), "output_kind": _lstat_kind(paths.output), "go_gate_authorized": False, "go_token_sha256": token_sha256}
    return payload


def _freeze_primary(paths: ArtifactPaths) -> None:
    for path in (paths.preflight, paths.main, paths.postflight, paths.transcript, paths.driver, paths.result):
        if _lexists(path):
            _freeze(path)


def _control_check(paths: ArtifactPaths) -> Mapping[str, Any]:
    equality: dict[str, bool] = {}
    hashes: dict[str, Mapping[str, str]] = {}
    for name in PROTECTED_MODEL_OUTPUTS:
        old, new = _read_regular_nul_free(V1_DYNAMIC_CONTROL / name, label=f"v1 {name}"), _read_regular_nul_free(paths.output / name, label=f"v13 control {name}")
        equality[name] = old == new
        hashes[name] = {"v1_dynamic_sha256": hashlib.sha256(old).hexdigest(), "v13_control_sha256": hashlib.sha256(new).hexdigest()}
    if not all(equality.values()):
        raise V13DispatchError("v13 dynamic always-control protected outputs differ from frozen baseline")
    run = _json(paths.output / "run.json", label="v13 control run.json")
    timeline = _json_value(paths.output / "state-timeline.json", label="v13 control timeline")
    runtime, baseline = run.get("runtime_seconds"), _json(V1_DYNAMIC_CONTROL / "run.json", label="frozen baseline run.json").get("runtime_seconds")
    if run.get("status") != "succeeded" or run.get("state_policy", {}).get("name") != "always-commit" or type(runtime) not in (int, float) or type(baseline) not in (int, float) or not math.isfinite(float(runtime)) or not math.isfinite(float(baseline)) or baseline <= 0:
        raise V13DispatchError("v13 control run metadata/runtime is invalid")
    ratio = float(runtime) / float(baseline)
    if ratio > RUNTIME_RATIO_LIMIT or not isinstance(timeline, list) or len(timeline) != 30:
        raise V13DispatchError("v13 control runtime or timeline does not pass")
    for frame_id, item in enumerate(timeline):
        if not isinstance(item, Mapping) or item.get("frame_id") != frame_id or item.get("action") != "commit" or item.get("reason") != "always_commit_control" or item.get("current_alarm") is not False or item.get("pending_transaction_count") != 0 or item.get("repair") is not None or item.get("export_action") != "export_real_camera_pose":
            raise V13DispatchError("v13 control has a non-direct transaction")
    return {"status": "PASS", "protected_model_outputs_byte_identical": equality, "protected_model_output_sha256": hashes, "transaction_count": 30, "pending_transaction_count": 0, "runtime_seconds": float(runtime), "baseline_runtime_seconds": float(baseline), "runtime_ratio": ratio, "runtime_ratio_limit": RUNTIME_RATIO_LIMIT}


def _candidate_check(paths: ArtifactPaths) -> Mapping[str, Any]:
    """Prove v13's fixed causal repair semantics after a successful child."""
    run = _json(paths.output / "run.json", label="v13 candidate run.json")
    timeline = _json_value(paths.output / "state-timeline.json", label="v13 candidate timeline")
    trajectory = _json(paths.output / "trajectory.json", label="v13 candidate trajectory")
    control_trajectory = _json(V1_DYNAMIC_CONTROL / "trajectory.json", label="frozen baseline dynamic trajectory")
    runtime = run.get("runtime_seconds")
    baseline = _json(V1_DYNAMIC_CONTROL / "run.json", label="frozen baseline dynamic run.json").get("runtime_seconds")
    if (
        run.get("status") != "succeeded"
        or run.get("state_policy", {}).get("name") != CANDIDATE_POLICY
        or type(runtime) not in (int, float)
        or type(baseline) not in (int, float)
        or not math.isfinite(float(runtime))
        or not math.isfinite(float(baseline))
        or float(baseline) <= 0
        or not isinstance(timeline, list)
        or len(timeline) != 30
    ):
        raise V13DispatchError("v13 candidate run metadata, runtime, or timeline is invalid")
    ratio = float(runtime) / float(baseline)
    if ratio > RUNTIME_RATIO_LIMIT:
        raise V13DispatchError("v13 candidate runtime exceeds the frozen limit")

    repairs: list[Mapping[str, Any]] = []
    for frame_id, item in enumerate(timeline):
        if not isinstance(item, Mapping) or item.get("frame_id") != frame_id or item.get("pending_transaction_count") != 0:
            raise V13DispatchError("v13 candidate timeline has an invalid transaction")
        if item.get("action") == "commit":
            if item.get("current_alarm") is not False or item.get("repair") is not None:
                raise V13DispatchError("v13 clear transaction is not a direct commit")
            continue
        if item.get("action") != "selective_state_memory_repair" or item.get("current_alarm") is not True:
            raise V13DispatchError("v13 candidate has an unsupported state action")
        detector, repair = item.get("online_detector"), item.get("repair")
        if not isinstance(detector, Mapping) or detector.get("frame_id") != frame_id or detector.get("hybrid_alarm") is not True:
            raise V13DispatchError("v13 repair lacks its preceding raw detector alarm")
        if not isinstance(repair, Mapping) or repair.get("operator") != "selective_state_memory_repair_v13":
            raise V13DispatchError("v13 repair evidence is missing its fixed operator")
        if repair.get("repair_denominator") != 8 or repair.get("fallback_used") is not False or repair.get("no_fallback") is not True or repair.get("raw_camera_pose_numeric_input") is not False or repair.get("cuda_resident") is not True:
            raise V13DispatchError("v13 repair evidence violates the fixed no-fallback CUDA contract")
        for name, shape, count in (("state_feat", [1, 768, 768], 96), ("mem", [1, 256, 1536], 32)):
            section = repair.get(name)
            if not isinstance(section, Mapping):
                raise V13DispatchError(f"v13 repair lacks {name} evidence")
            selected = section.get("selected_rows")
            digests = (section.get("pre_gpu_digest"), section.get("proposed_gpu_digest"), section.get("committed_gpu_digest"))
            if (
                section.get("shape") != shape
                or section.get("selected_row_count") != count
                or not isinstance(selected, list)
                or len(selected) != count
                or len(set(selected)) != count
                or any(type(value) is not int or value < 0 or value >= shape[1] for value in selected)
                or section.get("tie_free_boundary") is not True
                or type(section.get("selected_nonzero_delta_count")) is not int
                or section["selected_nonzero_delta_count"] <= 0
                or section.get("selected_rows_equal_pre") is not True
                or section.get("unselected_rows_equal_proposed") is not True
                or not all(isinstance(value, str) and value.startswith("gpu-fingerprint-v1:") for value in digests)
            ):
                raise V13DispatchError(f"v13 repair {name} evidence is incomplete")
        if (
            item.get("export_action") != "export_real_camera_pose"
            or item.get("raw_pose_unchanged") is not True
            or item.get("raw_candidate_pose_gpu_digest") != item.get("exported_camera_pose_gpu_digest")
            or item.get("selective_state_memory_repair") != repair
        ):
            raise V13DispatchError("v13 repair changed or failed to prove its raw current pose export")
        repairs.append(item)
    if not repairs:
        raise V13DispatchError("v13 candidate is a no-op: no detector alarm received a repair")

    candidate_frames, control_frames = trajectory.get("frames"), control_trajectory.get("frames")
    if not isinstance(candidate_frames, list) or not isinstance(control_frames, list) or len(candidate_frames) != 30 or len(control_frames) != 30:
        raise V13DispatchError("v13 trajectory evidence is incomplete")
    earliest = int(repairs[0]["frame_id"])
    later_changed = any(
        candidate_frames[index].get("camera_to_reference") != control_frames[index].get("camera_to_reference")
        for index in range(earliest + 1, len(candidate_frames))
        if isinstance(candidate_frames[index], Mapping) and isinstance(control_frames[index], Mapping)
    )
    if not later_changed:
        raise V13DispatchError("v13 candidate has no future-frame trajectory intervention evidence")
    return {
        "status": "PASS", "repair_count": len(repairs), "first_repair_frame_id": earliest,
        "future_frame_trajectory_changed": True, "runtime_seconds": float(runtime),
        "baseline_runtime_seconds": float(baseline), "runtime_ratio": ratio,
        "runtime_ratio_limit": RUNTIME_RATIO_LIMIT,
    }


def _validate_driver(paths: ArtifactPaths, *, run_id: str, argv: Sequence[str], command_hash: str, quoted: str) -> str:
    text = _read_regular_nul_free(paths.driver, label="frozen driver").decode("utf-8", errors="strict")
    tokens = [line.split("=", 1)[1] for line in text.splitlines() if line.startswith("go_token=")]
    if len(tokens) != 1 or len(tokens[0]) != 64 or any(ch not in "0123456789abcdef" for ch in tokens[0]) or text != _driver(run_id, argv, paths, command_hash, quoted, tokens[0]):
        raise V13DispatchError("frozen driver differs from pinned token-gated template")
    return hashlib.sha256((tokens[0] + "\n").encode("ascii")).hexdigest()


def _validator_report(paths: ArtifactPaths, *, run_id: str) -> Mapping[str, Any]:
    report: dict[str, Any] = {"schema_version": "stateguard3r.v13-validator.v1", "run_id": run_id, "started_at": datetime.now().astimezone().isoformat(), "cpu_only": True}
    try:
        primary = (paths.preflight, paths.main, paths.postflight, paths.transcript, paths.driver)
        for path in primary:
            if _lstat_kind(path) != "regular" or stat.S_IMODE(path.stat().st_mode) != 0o444:
                raise V13DispatchError("sealed primary artifact mode/path invalid")
            _read_regular_nul_free(path, label="sealed primary artifact")
        preflight, postflight = _json(paths.preflight, label="preflight"), _json(paths.postflight, label="postflight")
        if preflight.get("schema_version") != "stateguard3r.v13-preflight.v1" or preflight.get("run_id") != run_id or not isinstance(preflight.get("command"), list):
            raise V13DispatchError("sealed preflight schema is invalid")
        argv, command_hash, quoted = preflight["command"], preflight.get("command_sha256"), preflight.get("quoted_command")
        if quoted != shlex.join(argv) or command_hash != _command_hash(argv) or _child(argv, paths, test=preflight.get("test_child") is True) != argv:
            raise V13DispatchError("sealed child command does not recompute")
        snapshots = preflight.get("gpu_preflights")
        if not isinstance(snapshots, list) or len(snapshots) != 2:
            raise V13DispatchError("sealed preflight does not contain exactly two snapshots")
        for snapshot in snapshots:
            _validate_snapshot(snapshot)
        source = _component_provenance()
        if preflight.get("source_provenance") != source or postflight.get("source_provenance") != source:
            raise V13DispatchError("sealed source provenance does not recompute")
        token_sha = _validate_driver(paths, run_id=run_id, argv=argv, command_hash=command_hash, quoted=quoted)
        if not _lexists(paths.result):
            failure = _preexec_failure(paths.main, run_id=run_id)
            if failure is None or _preexec_failure(paths.transcript, run_id=run_id) != failure or postflight.get("error") != "PREEXEC_IDENTITY_NO_GO" or postflight.get("preexec_no_go", {}).get("payload_not_executed") is not True or postflight["preexec_no_go"].get("go_token_sha256") != token_sha or postflight["preexec_no_go"].get("driver_result_exists") is not False or postflight["preexec_no_go"].get("output_kind") != "missing":
                raise V13DispatchError("sealed pre-exec no-go boundary does not recompute")
            _ordered_stream(paths.main, run_id=run_id, quoted=quoted, command_hash=command_hash, result=None, transcript=False)
            _ordered_stream(paths.transcript, run_id=run_id, quoted=quoted, command_hash=command_hash, result=None, transcript=True)
            report.update({"status": "PREEXEC_IDENTITY_NO_GO_EVIDENCE_COMPLETE", "payload_not_executed": True})
        else:
            if _lstat_kind(paths.result) != "regular" or stat.S_IMODE(paths.result.stat().st_mode) != 0o444:
                raise V13DispatchError("sealed result mode/path invalid")
            result = _parse_result(paths.result, run_id=run_id, command_hash=command_hash)
            _ordered_stream(paths.main, run_id=run_id, quoted=quoted, command_hash=command_hash, result=result, transcript=False)
            _ordered_stream(paths.transcript, run_id=run_id, quoted=quoted, command_hash=command_hash, result=result, transcript=True)
            if postflight.get("result") != result or not postflight.get("pipe_close_and_drain", {}).get("drained") or "gpu_postflight_error" in postflight:
                raise V13DispatchError("sealed postflight is incomplete")
            _validate_snapshot(postflight.get("gpu_postflight", {}), require_minimum=False)
            pid = postflight.get("pid_start_time_check")
            if not isinstance(pid, Mapping) or pid.get("expected_start_ticks") != result["child_start_ticks"] or pid.get("pair_absent") is not True or pid.get("pid_reused") is not False:
                raise V13DispatchError("sealed postflight lacks PID/start-time absence proof")
            if result["exit_code"] == 0:
                inventory = _output_inventory(paths.output, frozen=True)
                if postflight.get("output_inventory") != inventory:
                    raise V13DispatchError("sealed output inventory does not recompute")
                if run_id == CONTROL_RUN_ID:
                    report["dynamic_always_control"] = _control_check(paths)
                if run_id == CANDIDATE_RUN_ID:
                    report["dynamic_candidate"] = _candidate_check(paths)
                report["status"] = "PASS"
            else:
                report["status"] = "CHILD_NONZERO_EVIDENCE_COMPLETE"
        report["artifact_sha256"] = {path.name: _sha256_file(path) for path in (*primary, *((paths.result,) if _lexists(paths.result) else ()))}
    except Exception as error:
        report.update({"status": "FAIL", "error": str(error)})
        _write_json(paths.validator, {**report, "finished_at": datetime.now().astimezone().isoformat()})
        _freeze(paths.validator)
        raise V13DispatchError(f"independent v13 validator failed: {error}") from error
    report["finished_at"] = datetime.now().astimezone().isoformat()
    _write_json(paths.validator, report)
    _freeze(paths.validator)
    return report


def validate_sealed_run(run_id: str) -> Mapping[str, Any]:
    paths = artifact_paths(run_id)
    _canonical_paths(paths)
    if _lstat_kind(paths.lease) != "directory" or _lexists(paths.validator):
        raise V13DispatchError("validator requires owner lease and a fresh validator artifact")
    return _validator_report(paths, run_id=run_id)


def _run_independent_validator(run_id: str) -> Mapping[str, Any]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--validate-run-id", run_id], text=True, capture_output=True, env=environment)
    if result.returncode != 0:
        raise V13DispatchError(f"independent validator subprocess failed: {result.stderr.strip() or result.stdout.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise V13DispatchError("independent validator did not emit JSON") from error


def dispatch(run_id: str, argv: Sequence[str], *, timeout_seconds: float, allow_noncuda_child_for_test: bool = False) -> Mapping[str, Any]:
    """Dispatch once, seal original artifacts, then validate from fresh CPU Python."""
    if timeout_seconds <= 0:
        raise V13DispatchError("dispatch timeout must be positive")
    paths = artifact_paths(run_id)
    _canonical_paths(paths)
    _acquire_lease(paths)
    existing = [name for name, value in asdict(paths).items() if name != "lease" and _lexists(value)]
    if existing:
        raise V13DispatchError(f"same-ID lease acquired but owner artifacts already exist: {existing}")
    child = _child(argv, paths, test=allow_noncuda_child_for_test)
    command_hash, quoted = _command_hash(child), shlex.join(child)
    token = secrets.token_hex(32)
    token_sha = hashlib.sha256((token + "\n").encode("ascii")).hexdigest()
    source: Mapping[str, Any] | None = None
    pane: str | None = None
    pipe: Mapping[str, Any] | None = None
    result: Mapping[str, Any] | None = None
    preexec: Mapping[str, Any] | None = None
    error: Exception | None = None
    try:
        source = _component_provenance()
        snapshots = [_snapshot(), _snapshot()]
        for snapshot in snapshots:
            _validate_snapshot(snapshot)
        _write_json(paths.preflight, {"schema_version": "stateguard3r.v13-preflight.v1", "run_id": run_id, "command": child, "quoted_command": quoted, "command_sha256": command_hash, "test_child": allow_noncuda_child_for_test, "interpreter": _interpreter_provenance(child, test=allow_noncuda_child_for_test), "gpu_preflights": snapshots, "source_provenance": source})
        _freeze(paths.preflight)
        _write(paths.main, "")
        _write(paths.transcript, "")
        _write(paths.driver, _driver(run_id, child, paths, command_hash, quoted, token), 0o555)
        pane, deadline = _pane(), time.monotonic() + timeout_seconds
        _tmux(["pipe-pane", "-o", "-t", pane, f"cat >> {shlex.quote(str(paths.transcript))}"])
        ready = f"V13_PIPE_READY run_id={run_id}"
        _tmux(["send-keys", "-t", pane, f"printf '%s\\n' {shlex.quote(ready)}", "C-m"])
        _wait_for_text(paths.transcript, ready, deadline)
        _tmux(["send-keys", "-t", pane, f"bash {shlex.quote(str(paths.driver))}; exit", "C-m"])
        try:
            result = _wait_for_result(paths.result, run_id=run_id, command_hash=command_hash, deadline=deadline, main=paths.main)
        except V13PreexecNoGo as event:
            preexec = event.failure
            _wait_for_driver_exit(pane, time.monotonic() + 10.0)
            pipe = _close_pipe_and_drain(pane, paths.transcript, paths.main)
            _ordered_stream(paths.main, run_id=run_id, quoted=quoted, command_hash=command_hash, result=None, transcript=False)
            _ordered_stream(paths.transcript, run_id=run_id, quoted=quoted, command_hash=command_hash, result=None, transcript=True)
            if _lexists(paths.result) or _lstat_kind(paths.output) != "missing":
                raise V13DispatchError("pre-exec identity failure unexpectedly has result or output")
            post_source = _component_provenance()
            if post_source != source:
                raise V13DispatchError("source/worktree drifted during pre-exec identity gate")
            postflight = _postflight(paths, run_id=run_id, result=None, source=post_source, pipe=pipe, error="PREEXEC_IDENTITY_NO_GO", token_sha256=token_sha, preexec=preexec)
            if "gpu_postflight_error" in postflight:
                raise V13DispatchError("pre-exec no-go lacks GPU postflight snapshot")
            _validate_snapshot(postflight["gpu_postflight"], require_minimum=False)
            _write_json(paths.postflight, postflight)
            raise V13DispatchError("driver terminated before payload release because PID/start-time identity could not be established") from event
        except V13DispatchError as timeout_error:
            # A missing result is never allowed to turn into a post-hoc freeze
            # while an unverified wrapper might still own CUDA.  Only the exact
            # PID/start-time pair written before payload release is terminable.
            identity = _partial_child_identity(paths.main, run_id=run_id)
            if identity is None:
                raise V13DispatchError("driver timed out without a PID/start-time marker; refusing to seal a possibly live child") from timeout_error
            _terminate_owned_child(identity)
            deadline = time.monotonic() + 10.0
            result = _wait_for_result(paths.result, run_id=run_id, command_hash=command_hash, deadline=deadline, main=paths.main)
        _wait_for_driver_exit(pane, deadline)
        pipe = _close_pipe_and_drain(pane, paths.transcript, paths.main)
        _ordered_stream(paths.main, run_id=run_id, quoted=quoted, command_hash=command_hash, result=result, transcript=False)
        _ordered_stream(paths.transcript, run_id=run_id, quoted=quoted, command_hash=command_hash, result=result, transcript=True)
        pid = _pid_pair_absent(result)
        if not pid["pair_absent"] or pid["pid_reused"]:
            raise V13DispatchError("recorded child PID/start-time pair is live or reused at postflight")
        post_source = _component_provenance()
        if post_source != source:
            raise V13DispatchError("source/worktree drifted during v13 child")
        postflight = dict(_postflight(paths, run_id=run_id, result=result, source=post_source, pipe=pipe, error=None))
        if "gpu_postflight_error" in postflight:
            raise V13DispatchError("postflight GPU snapshot could not be collected")
        _validate_snapshot(postflight["gpu_postflight"], require_minimum=False)
        postflight["pid_start_time_check"] = pid
        postflight["output_inventory"] = _freeze_output(paths.output) if result["exit_code"] == 0 else None
        _write_json(paths.postflight, postflight)
    except Exception as caught:
        error = caught
        if pane is not None and pipe is None and _lexists(paths.transcript):
            try:
                pipe = _close_pipe_and_drain(pane, paths.transcript, paths.main)
            except Exception as pipe_error:
                pipe = {"drained": False, "close_error": str(pipe_error)}
        if _lexists(paths.preflight) and not _lexists(paths.postflight) and source is not None:
            try:
                _write_json(paths.postflight, _postflight(paths, run_id=run_id, result=result, source=source, pipe=pipe or {"drained": False}, error=str(caught), token_sha256=token_sha if preexec else None, preexec=preexec))
            except Exception:
                pass
    finally:
        if pane is None or (pipe is not None and pipe.get("drained") is True and pipe.get("pane_alive_at_close") is False):
            try:
                _freeze_primary(paths)
            except Exception as freeze_error:
                error = error or freeze_error
        else:
            error = error or V13DispatchError("terminal writers were not closed/drained; refusing to freeze live evidence")
    try:
        primary = (paths.preflight, paths.main, paths.postflight, paths.transcript, paths.driver)
        normal = (*primary, paths.result)
        if error is None or (preexec is not None and all(_lexists(path) for path in primary)) or (preexec is None and all(_lexists(path) for path in normal)):
            validator = _run_independent_validator(run_id)
        else:
            validator = None
    except Exception as validator_error:
        error = error or validator_error
        validator = None
    if error is not None:
        raise V13DispatchError(f"v13 terminal evidence sealed as failure: {error}") from error
    assert result is not None
    if result["exit_code"] != 0:
        raise V13DispatchError(f"v13 child exited nonzero with immutable evidence: {result['exit_code']}")
    return {"status": "PASS", "run_id": run_id, "pane": pane, "command_sha256": command_hash, "validator": validator, "paths": {name: str(path) for name, path in asdict(paths).items()}}


def _read_command_json(path: Path) -> list[str]:
    if path.parent != ROOT / "tmp" or _lstat_kind(path) != "regular":
        raise V13DispatchError("command JSON must be a regular direct tmp child")
    try:
        value = json.loads(_read_regular_nul_free(path, label="command JSON"))
    except json.JSONDecodeError as error:
        raise V13DispatchError("command JSON is malformed") from error
    if not isinstance(value, list):
        raise V13DispatchError("command JSON must be an argv array")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.validate_run_id is not None:
            print(json.dumps(validate_sealed_run(args.validate_run_id), ensure_ascii=False, sort_keys=True))
        else:
            if args.run_id is None:
                raise V13DispatchError("--run-id is required with --command-json")
            print(json.dumps(dispatch(args.run_id, _read_command_json(args.command_json), timeout_seconds=args.timeout_seconds), ensure_ascii=False, sort_keys=True))
        return 0
    except V13DispatchError as error:
        print(f"V13_DISPATCH_ERROR={error}", file=sys.stderr)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
