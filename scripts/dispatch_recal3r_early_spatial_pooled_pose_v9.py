#!/usr/bin/env python3
"""One-use, single-owner terminal-evidence dispatcher for v9 GPU forwards.

This launcher is deliberately stricter than a convenient job wrapper.  A v9
forward is scientifically usable only if its original, ordered terminal
evidence exists.  In particular, this module never offers a repair,
postflight-only, or rerun command.
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
RUNNER_RELATIVE = Path("scripts/run_recal3r_early_spatial_pooled_pose_export_v9.py")
PINNED_COMPONENTS = (
    RUNNER_RELATIVE,
    Path("src/stateguard3r/recal3r_early_spatial_pooled_pose_runner_v9.py"),
    Path("src/stateguard3r/early_spatial_pooled_pose_export_v9.py"),
    Path("src/stateguard3r/early_spatial_pooled_pose_v9.py"),
    Path("scripts/dispatch_recal3r_early_spatial_pooled_pose_v9.py"),
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
CONTROL_RUN_ID = "recovery-early-spatial-pooled-pose-v9-dynamic-always-commit-0001"
CANDIDATE_RUN_ID = "recovery-early-spatial-pooled-pose-v9-dynamic-candidate-0001"
RUN_SPECS = {
    CONTROL_RUN_ID: ("dynamic", "always-commit"),
    CANDIDATE_RUN_ID: ("dynamic", "detector-v3-incremental-early-spatial-pooled-pose-export"),
    "recovery-early-spatial-pooled-pose-v9-wrong-always-commit-0001": ("wrong", "always-commit"),
    "recovery-early-spatial-pooled-pose-v9-wrong-candidate-0001": ("wrong", "detector-v3-incremental-early-spatial-pooled-pose-export"),
    "recovery-early-spatial-pooled-pose-v9-low-always-commit-0001": ("low", "always-commit"),
    "recovery-early-spatial-pooled-pose-v9-low-candidate-0001": ("low", "detector-v3-incremental-early-spatial-pooled-pose-export"),
}
PINNED_EXTERNALS = (
    CHECKPOINT,
    RECAL3R_ROOT / "src" / "dust3r" / "model.py",
    RECAL3R_ROOT / "src" / "dust3r" / "heads" / "dpt_head.py",
    RECAL3R_ROOT / "src" / "dust3r" / "heads" / "postprocess.py",
    DETECTOR_CONFIG,
    *DEVELOPMENT_MANIFESTS,
)
V1_DYNAMIC_CONTROL = ROOT / "outputs" / "recovery-policy-development-v1-dynamic-always-commit-0001"
PROTECTED_MODEL_OUTPUTS = (
    "checkpoint-load-audit.json",
    "health.jsonl",
    "predictions-summary.json",
    "trajectory.json",
)
RUNTIME_RATIO_LIMIT = 1.20


class V9DispatchError(RuntimeError):
    """A v9 forward has no valid single-owner dispatch path."""


class V9PreexecNoGo(V9DispatchError):
    """The driver reaped an unreleased wrapper and emitted no child result."""

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
    commands.add_argument("--command-json", type=Path, help="regular direct tmp JSON array of the pinned child argv")
    commands.add_argument("--validate-run-id", help="run the separate CPU-only, one-use validator")
    parser.add_argument("--run-id", help="one-use v9 run ID (required with --command-json)")
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    # This is intentionally an API/testing escape hatch only.  The public
    # launcher cannot select it, so a real v9 execution always has --device cuda.
    return parser


def _run_id(run_id: str) -> None:
    valid = set("abcdefghijklmnopqrstuvwxyz0123456789-")
    if (
        type(run_id) is not str
        or not run_id.startswith("recovery-early-spatial-pooled-pose-v9-")
        or not run_id
        or any(character not in valid for character in run_id)
    ):
        raise V9DispatchError("run id is not a canonical v9 one-use id")


def artifact_paths(run_id: str) -> ArtifactPaths:
    _run_id(run_id)
    logs, tmp = ROOT / "logs", ROOT / "tmp"
    return ArtifactPaths(
        output=ROOT / "outputs" / run_id,
        preflight=logs / f"{run_id}-preflight.json",
        main=logs / f"{run_id}-main.log",
        postflight=logs / f"{run_id}-postflight.json",
        transcript=logs / f"{run_id}-tmux-transcript.log",
        driver=logs / f"{run_id}-driver.sh",
        result=logs / f"{run_id}-result.json",
        validator=logs / f"{run_id}-validator-0001.log",
        lease=tmp / f"{run_id}.owner-lease",
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


def _require_direct_directory(path: Path, *, label: str) -> None:
    if _lstat_kind(path) != "directory":
        raise V9DispatchError(f"{label} parent must be an existing non-symlink directory")


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
    for name, expected_path in expected.items():
        actual = getattr(paths, name)
        if actual != expected_path or actual.parent != expected_path.parent:
            raise V9DispatchError(f"{name} path is not a direct canonical child")
    for parent, label in ((ROOT / "outputs", "outputs"), (ROOT / "logs", "logs"), (ROOT / "tmp", "tmp")):
        _require_direct_directory(parent, label=label)


def _write(path: Path, data: str, mode: int = 0o644) -> None:
    if "\0" in data:
        raise V9DispatchError(f"refusing NUL-containing write: {path}")
    if _lexists(path):
        raise V9DispatchError(f"refusing to overwrite one-use artifact: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # The incomplete artifact remains visible and prevents an ID reuse.
        raise
    os.chmod(path, mode)


def _write_json(path: Path, payload: Any, mode: int = 0o644) -> None:
    _write(path, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", mode)


def _sha256_file(path: Path) -> str:
    """Hash large pinned artifacts without loading a multi-GB checkpoint at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_regular_nul_free(path: Path, *, label: str) -> bytes:
    if _lstat_kind(path) != "regular":
        raise V9DispatchError(f"{label} is missing, a symlink, or nonregular: {path}")
    data = path.read_bytes()
    if b"\0" in data:
        raise V9DispatchError(f"{label} contains NUL: {path}")
    return data


def _freeze(path: Path) -> None:
    _read_regular_nul_free(path, label="terminal artifact")
    os.chmod(path, 0o444)
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise V9DispatchError(f"could not freeze terminal artifact: {path}")


def _output_inventory(path: Path, *, require_frozen: bool) -> list[dict[str, Any]]:
    """Read an output tree without ever changing it (validator-safe)."""
    if _lstat_kind(path) != "directory":
        raise V9DispatchError("child did not produce a direct regular output directory")
    if require_frozen and stat.S_IMODE(path.stat().st_mode) != 0o555:
        raise V9DispatchError("output root mode is not frozen 0555")
    inventory: list[dict[str, Any]] = []
    for current_text, directories, filenames in os.walk(path, topdown=False, followlinks=False):
        current = Path(current_text)
        for name in sorted(filenames):
            child = current / name
            if _lstat_kind(child) != "regular":
                raise V9DispatchError(f"output contains symlink or special artifact: {child}")
            _read_regular_nul_free(child, label="output artifact")
            if require_frozen and stat.S_IMODE(child.stat().st_mode) != 0o444:
                raise V9DispatchError(f"output artifact mode is not frozen 0444: {child}")
            inventory.append({"path": str(child.relative_to(path)), "kind": "file", "mode": "0444", "sha256": hashlib.sha256(child.read_bytes()).hexdigest()})
        for name in sorted(directories):
            child = current / name
            if _lstat_kind(child) != "directory":
                raise V9DispatchError(f"output contains symlink or special directory: {child}")
            if require_frozen and stat.S_IMODE(child.stat().st_mode) != 0o555:
                raise V9DispatchError(f"output directory mode is not frozen 0555: {child}")
            inventory.append({"path": str(child.relative_to(path)), "kind": "directory", "mode": "0555"})
    inventory.append({"path": ".", "kind": "directory", "mode": "0555"})
    return sorted(inventory, key=lambda item: (item["path"], item["kind"]))


def _freeze_output(path: Path) -> list[dict[str, Any]]:
    """Reject links/special files and recursively make a child output immutable."""
    _output_inventory(path, require_frozen=False)
    for current_text, directories, filenames in os.walk(path, topdown=False, followlinks=False):
        current = Path(current_text)
        for name in filenames:
            os.chmod(current / name, 0o444)
        for name in directories:
            os.chmod(current / name, 0o555)
    os.chmod(path, 0o555)
    return _output_inventory(path, require_frozen=True)


def _fresh(paths: ArtifactPaths) -> None:
    _canonical_paths(paths)
    # Validator is also an owner artifact for a previous run, but is never
    # created before the owner has sealed its primary evidence.
    existing = [name for name, path in asdict(paths).items() if _lexists(path)]
    if existing:
        raise V9DispatchError(f"a one-use output or terminal artifact path already exists: {existing}")


def _acquire_lease(paths: ArtifactPaths) -> None:
    """Claim the same-ID owner slot atomically before GPU/pane inspection."""
    _canonical_paths(paths)
    try:
        paths.lease.mkdir(mode=0o700)
    except FileExistsError as error:
        raise V9DispatchError("same-ID owner lease already exists") from error
    if _lstat_kind(paths.lease) != "directory":
        raise V9DispatchError("same-ID owner lease is not a directory")
    _require_empty_lease(paths)


def _lease_entry(paths: ArtifactPaths, name: str) -> Path:
    if name not in {"child-start", "child-start.tmp", "child-go"}:
        raise V9DispatchError("unrecognized owner-lease entry")
    return paths.lease / name


def _require_empty_lease(paths: ArtifactPaths) -> None:
    """A fresh owner lease is a private protocol channel, never a cache."""
    if _lstat_kind(paths.lease) != "directory":
        raise V9DispatchError("same-ID owner lease is not a direct directory")
    entries = list(paths.lease.iterdir())
    if entries:
        raise V9DispatchError("same-ID owner lease is not empty")


def _component_provenance() -> Mapping[str, Any]:
    """Bind this launcher and every v9 scientific component to HEAD bytes."""
    try:
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            check=True, text=True, capture_output=True,
        ).stdout
        if dirty.strip():
            raise V9DispatchError("StateGuard3R tracked worktree is not clean")
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, text=True, capture_output=True,
        ).stdout.strip()
        components: dict[str, str] = {}
        for relative in PINNED_COMPONENTS:
            absolute = ROOT / relative
            subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", str(relative)], check=True, text=True, capture_output=True)
            head_bytes = subprocess.run(
                ["git", "-C", str(ROOT), "show", f"HEAD:{relative}"],
                check=True, capture_output=True,
            ).stdout
            digest = hashlib.sha256(absolute.read_bytes()).hexdigest()
            if hashlib.sha256(head_bytes).hexdigest() != digest:
                raise V9DispatchError(f"pinned component differs from HEAD: {relative}")
            components[str(relative)] = digest
        baseline = ROOT.parent / "baselines" / "ReCal3R"
        baseline_dirty = subprocess.run(
            ["git", "-C", str(baseline), "status", "--porcelain"],
            check=True, text=True, capture_output=True,
        ).stdout
        if baseline_dirty.strip():
            raise V9DispatchError("ReCal3R tracked worktree is not clean")
        baseline_commit = subprocess.run(
            ["git", "-C", str(baseline), "rev-parse", "HEAD"], check=True, text=True, capture_output=True,
        ).stdout.strip()
        if baseline_commit != RECAL3R_COMMIT:
            raise V9DispatchError("ReCal3R commit differs from the preregistered v9 revision")
    except subprocess.CalledProcessError as error:
        raise V9DispatchError("cannot inspect pinned source provenance") from error
    external: dict[str, str] = {}
    for path in PINNED_EXTERNALS:
        if _lstat_kind(path) != "regular":
            raise V9DispatchError(f"pinned external input is missing or unsafe: {path}")
        external[str(path)] = _sha256_file(path)
    if external[str(CHECKPOINT)] != CHECKPOINT_SHA256:
        raise V9DispatchError("pinned checkpoint SHA-256 differs")
    return {"stateguard_commit": commit, "recal3r_commit": baseline_commit, "component_sha256": components, "external_sha256": external}


def _project_gpu2_processes() -> list[Mapping[str, Any]]:
    command = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    project: list[Mapping[str, Any]] = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 4 or values[0] != GPU_UUID or not values[1].isdigit():
            continue
        pid = int(values[1])
        try:
            command_line = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        except OSError:
            command_line = "<unavailable>"
        if str(ROOT) in command_line or str(RECAL3R_ROOT) in command_line:
            project.append({"gpu_uuid": values[0], "pid": pid, "process_name": values[2], "used_memory_mib": values[3], "command": command_line})
    return project


def _snapshot(*, require_minimum: bool = True, require_project_absent: bool = True) -> dict[str, Any]:
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
    project = _project_gpu2_processes()
    if require_project_absent and project:
        raise V9DispatchError("a StateGuard3R/ReCal3R project process is already on GPU 2")
    return {"timestamp": datetime.now().astimezone().isoformat(), "command": command, "rows": rows, "selected": selected[0], "project_gpu2_processes": project}


def _validate_snapshot(snapshot: Mapping[str, Any], *, require_minimum: bool = True) -> None:
    selected = snapshot.get("selected")
    if not isinstance(selected, Mapping) or selected.get("uuid") != GPU_UUID or type(selected.get("memory_free_mib")) is not int or (require_minimum and selected["memory_free_mib"] < MIN_FREE_MIB):
        raise V9DispatchError("stored GPU preflight snapshot is invalid")
    if not isinstance(snapshot.get("timestamp"), str) or not snapshot["timestamp"]:
        raise V9DispatchError("stored GPU preflight lacks timestamp")
    if snapshot.get("project_gpu2_processes") != []:
        raise V9DispatchError("stored GPU snapshot does not prove project-process absence")


def _flag_value(argv: Sequence[str], flag: str) -> str:
    indices = [index for index, value in enumerate(argv) if value == flag]
    if len(indices) != 1 or indices[0] + 1 >= len(argv):
        raise V9DispatchError(f"child must contain exactly one {flag}")
    return argv[indices[0] + 1]


def _path_argument(value: str) -> Path:
    raw = Path(value)
    return (ROOT / raw).resolve(strict=False) if not raw.is_absolute() else raw.resolve(strict=False)


def _production_child_contract(argv: Sequence[str], paths: ArtifactPaths) -> None:
    """Pin every formal v9 command argument before any GPU inspection."""
    expected_flags = (
        "--baseline-root", "--checkpoint", "--checkpoint-sha256", "--input-manifest", "--output-dir",
        "--device", "--size", "--seed", "--beta-base", "--health-profile", "--rgb-timestamp-listing",
        "--timestamp-dataset-root", "--state-policy", "--detector-config", "--watchdog",
    )
    if len(argv) != 2 + 2 * len(expected_flags) or tuple(argv[2::2]) != expected_flags:
        raise V9DispatchError("production child argv is not the exact canonical v9 flag order")
    values = {argv[index]: argv[index + 1] for index in range(2, len(argv), 2)}
    if _path_argument(values["--baseline-root"]) != RECAL3R_ROOT.resolve(strict=False):
        raise V9DispatchError("production child baseline root differs")
    if _path_argument(values["--checkpoint"]) != CHECKPOINT.resolve(strict=False) or values["--checkpoint-sha256"] != CHECKPOINT_SHA256:
        raise V9DispatchError("production child checkpoint binding differs")
    manifest = _path_argument(values["--input-manifest"])
    if manifest not in {path.resolve(strict=False) for path in DEVELOPMENT_MANIFESTS}:
        raise V9DispatchError("production child manifest is not a pinned development manifest")
    if values["--output-dir"] != str(paths.output) or values["--device"] != "cuda" or values["--size"] != "512" or values["--seed"] != "0" or values["--beta-base"] != "0.1" or values["--health-profile"] != "v3" or values["--watchdog"] != "8":
        raise V9DispatchError("production child scalar v9 contract differs")
    rgb = RECAL3R_ROOT / "data" / "tum" / "rgbd_dataset_freiburg1_desk" / "rgb.txt"
    timestamp_root = rgb.parent
    if _path_argument(values["--rgb-timestamp-listing"]) != rgb.resolve(strict=False) or _path_argument(values["--timestamp-dataset-root"]) != timestamp_root.resolve(strict=False):
        raise V9DispatchError("production child timestamp inputs differ")
    if _path_argument(values["--detector-config"]) != DETECTOR_CONFIG.resolve(strict=False):
        raise V9DispatchError("production child Detector-v3 config differs")
    spec = RUN_SPECS.get(paths.output.name)
    if spec is None:
        raise V9DispatchError("production v9 run ID is not pre-registered")
    condition, expected_policy = spec
    expected_manifest = ROOT / "outputs" / "formal-v1-inputs-0001" / "development" / f"development-{condition}" / "input-manifest.json"
    if manifest != expected_manifest.resolve(strict=False) or values["--state-policy"] != expected_policy:
        raise V9DispatchError("production v9 run ID, manifest, and policy do not match the pre-registration")


def _require_gate_b_predecessor(run_id: str) -> None:
    """Make Gate-B/C ordering executable rather than an advisory document."""
    if run_id == CONTROL_RUN_ID:
        return
    if run_id == CANDIDATE_RUN_ID:
        paths = artifact_paths(CONTROL_RUN_ID)
        if _lstat_kind(paths.validator) != "regular" or stat.S_IMODE(paths.validator.stat().st_mode) != 0o444:
            raise V9DispatchError("dynamic candidate is blocked until the frozen dynamic control validator PASS exists")
        report = _regular_json(paths.validator, label="frozen dynamic control validator")
        if report.get("status") != "PASS" or report.get("dynamic_always_control", {}).get("status") != "PASS":
            raise V9DispatchError("dynamic candidate is blocked because dynamic control did not pass")
        return
    raise V9DispatchError("wrong/low v9 runs are blocked until a frozen dynamic candidate Gate-B PASS exists")


def _child(argv: Sequence[str], paths: ArtifactPaths, *, test: bool) -> list[str]:
    if not argv or any(type(item) is not str or not item or "\0" in item or "\n" in item or "\r" in item for item in argv):
        raise V9DispatchError("child argv is not a clean nonempty string array")
    if sum(item == "--output-dir" for item in argv) != 1 or sum(item == "--state-policy" for item in argv) != 1 or sum(item == "--device" for item in argv) != 1:
        raise V9DispatchError("child has duplicate or missing pinned flags")
    if len(argv) < 2 or argv[1] not in {str(RUNNER_RELATIVE), str(ROOT / RUNNER_RELATIVE)}:
        raise V9DispatchError("child is not the exact pinned v9 runner")
    if not test and argv[0] != str(RECAL3R_ROOT / ".venv" / "bin" / "python"):
        raise V9DispatchError("production child interpreter is not the pinned ReCal3R virtualenv Python")
    output = _flag_value(argv, "--output-dir")
    if output != str(paths.output):
        raise V9DispatchError("child output must be the exact absolute direct run output")
    policy = _flag_value(argv, "--state-policy")
    if policy not in {"always-commit", "detector-v3-incremental-early-spatial-pooled-pose-export"}:
        raise V9DispatchError("child state policy is not pinned")
    device = _flag_value(argv, "--device")
    if not test and device != "cuda":
        raise V9DispatchError("production dispatch requires a CUDA child")
    if test and device not in {"cpu", "cuda"}:
        raise V9DispatchError("test child device must be cpu or cuda")
    child = list(argv)
    if not test:
        _production_child_contract(child, paths)
        _require_gate_b_predecessor(paths.output.name)
    return child


def _command_hash(argv: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(argv), ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _interpreter_provenance(argv: Sequence[str], *, test: bool) -> Mapping[str, Any] | None:
    if test:
        return None
    interpreter = Path(argv[0])
    if _lstat_kind(interpreter) not in {"regular", "symlink"} or not interpreter.exists():
        raise V9DispatchError("pinned ReCal3R interpreter is unavailable")
    resolved = interpreter.resolve(strict=True)
    return {"path": str(interpreter), "resolved_path": str(resolved), "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()}


def _driver(run_id: str, argv: Sequence[str], paths: ArtifactPaths, command_hash: str, quoted: str, go_token: str) -> str:
    """Generate the sole writer of ordered main/result child evidence."""
    if len(go_token) != 64 or any(character not in "0123456789abcdef" for character in go_token):
        raise V9DispatchError("driver go token is not a 256-bit lowercase-hex value")
    run_q, main_q, result_q, lease_q, quoted_q, hash_q, token_q = (shlex.quote(value) for value in (run_id, str(paths.main), str(paths.result), str(paths.lease), quoted, command_hash, go_token))
    command = shlex.join(argv)
    return f"""#!/usr/bin/env bash
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
record "V9_DRIVER_START run_id=$run_id"
record "V9_DRIVER_COMMAND=$command_quoted"
record "V9_DRIVER_COMMAND_SHA256=$command_sha256"
record "V9_DRIVER_DISPATCHED run_id=$run_id kind=wrapper_only"
# The payload cannot exec until its wrapper has supplied a PID/start-time pair
# and this driver has written the unpredictable one-use go token.  Thus a
# missing /proc start time is a pre-exec failure, never an unidentifiable GPU
# forward.  A mere pre-existing path cannot release the payload: its contents
# must equal the token embedded in this frozen driver.
(
    gate_pid="$BASHPID"
    gate_start_ticks="$(awk '{{print $22}}' "/proc/$gate_pid/stat" 2>/dev/null)"
    if [[ ! "$gate_start_ticks" =~ ^[0-9]+$ ]]; then
        printf 'FAIL %s unavailable\\n' "$gate_pid" > "$start_tmp"
        mv "$start_tmp" "$start_path"
        exit 72
    fi
    printf 'READY %s %s\\n' "$gate_pid" "$gate_start_ticks" > "$start_tmp"
    mv "$start_tmp" "$start_path"
    while true; do
        if [[ -e "$go_path" || -L "$go_path" ]]; then
            if [[ -L "$go_path" || ! -f "$go_path" ]]; then exit 73; fi
            # The postflight evidence hashes the complete gate file, so the
            # wrapper must accept precisely those same bytes, not merely an
            # authorized first line followed by unrecorded suffix data.
            if ! printf '%s\\n' "$go_token" | cmp -s - "$go_path"; then exit 73; fi
            exec env CUDA_VISIBLE_DEVICES={GPU_INDEX} {command}
        fi
        sleep 0.01
    done
) >> "$main_log" 2>&1 &
child_pid=$!
start_deadline=$((SECONDS + 30))
while [[ ! -f "$start_path" && $SECONDS -lt $start_deadline ]]; do
    if ! kill -0 "$child_pid" 2>/dev/null; then break; fi
    sleep 0.01
done
if [[ ! -f "$start_path" ]]; then
    kill "$child_pid" 2>/dev/null; wait "$child_pid"; child_exit=$?
    record "V9_DRIVER_PREEXEC_IDENTITY_FAILURE run_id=$run_id child_pid=$child_pid reason=start_gate_missing exit_code=$child_exit"
    exit 70
fi
read -r start_kind reported_pid child_start_ticks < "$start_path"
if [[ "$start_kind" != READY || "$reported_pid" != "$child_pid" || ! "$child_start_ticks" =~ ^[0-9]+$ ]]; then
    wait "$child_pid"; child_exit=$?
    record "V9_DRIVER_PREEXEC_IDENTITY_FAILURE run_id=$run_id child_pid=$child_pid reason=start_time_unavailable exit_code=$child_exit"
    exit 70
fi
driver_start_ticks="$(awk '{{print $22}}' "/proc/$child_pid/stat" 2>/dev/null)"
if [[ "$driver_start_ticks" != "$child_start_ticks" ]]; then
    kill "$child_pid" 2>/dev/null; wait "$child_pid"; child_exit=$?
    record "V9_DRIVER_PREEXEC_IDENTITY_FAILURE run_id=$run_id child_pid=$child_pid reason=parent_start_time_mismatch exit_code=$child_exit"
    exit 70
fi
if ! record "V9_DRIVER_CHILD_PID run_id=$run_id child_pid=$child_pid child_start_ticks=$child_start_ticks"; then
    kill "$child_pid" 2>/dev/null; wait "$child_pid"
    exit 70
fi
record "V9_DRIVER_PAYLOAD_RELEASE_ARMED run_id=$run_id child_pid=$child_pid child_start_ticks=$child_start_ticks"
if ! (set -o noclobber; printf '%s\\n' "$go_token" > "$go_path") 2>/dev/null; then
    kill "$child_pid" 2>/dev/null; wait "$child_pid"; child_exit=$?
    record "V9_DRIVER_PREEXEC_IDENTITY_FAILURE run_id=$run_id child_pid=$child_pid reason=go_gate_create_failed exit_code=$child_exit"
    exit 70
fi
record "V9_DRIVER_PAYLOAD_RELEASED run_id=$run_id child_pid=$child_pid child_start_ticks=$child_start_ticks"
wait "$child_pid"; child_exit=$?
reaped_at="$(date --iso-8601=seconds)"
record "V9_DRIVER_EXIT run_id=$run_id child_pid=$child_pid child_start_ticks=$child_start_ticks exit_code=$child_exit reaped_at=$reaped_at"
if ! (set -o noclobber; : > "$result_path") 2>/dev/null; then
    record "V9_DRIVER_RESULT_CREATE_FAILED run_id=$run_id"
    exit 70
fi
# Exact command text is frozen in preflight and in the ordered stream.  Keep
# result JSON scalar-only so shell-legal quotes cannot corrupt JSON output.
printf '{{"schema_version":"stateguard3r.v9-driver-result.v1","run_id":"%s","command_sha256":"%s","child_pid":%s,"child_start_ticks":"%s","exit_code":%s,"reaped_at":"%s"}}\\n' "$run_id" "$command_sha256" "$child_pid" "$child_start_ticks" "$child_exit" "$reaped_at" >> "$result_path"
record "V9_DRIVER_RESULT_WRITTEN run_id=$run_id child_pid=$child_pid child_start_ticks=$child_start_ticks exit_code=$child_exit"
exit 0
"""


def _tmux(args: Sequence[str], *, check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["tmux", *args], check=check, text=True, capture_output=capture_output)


def _pane() -> str:
    try:
        _tmux(["has-session", "-t", SESSION])
        result = _tmux(["new-window", "-d", "-P", "-F", "#{pane_id}", "-t", SESSION, "-c", str(ROOT)])
    except subprocess.CalledProcessError as error:
        raise V9DispatchError("existing stateguard tmux session is unavailable") from error
    pane = result.stdout.strip()
    if not pane.startswith("%"):
        raise V9DispatchError("tmux did not return a fresh pane id")
    return pane


def _wait_for_text(path: Path, marker: str, deadline: float) -> None:
    while time.monotonic() < deadline:
        if _lexists(path) and marker in _read_regular_nul_free(path, label="tmux transcript").decode("utf-8", errors="strict"):
            return
        time.sleep(0.05)
    raise V9DispatchError(f"timed out waiting for ordered marker: {marker}")


def _pane_alive(pane: str) -> bool:
    return _tmux(["list-panes", "-t", pane, "-F", "#{pane_id}"], check=False).returncode == 0


def _wait_for_driver_exit(pane: str, deadline: float) -> None:
    while time.monotonic() < deadline:
        if not _pane_alive(pane):
            return
        time.sleep(0.05)
    raise V9DispatchError("driver pane did not exit after publishing its result")


def _stable_bytes(path: Path, *, label: str) -> bytes:
    first = _read_regular_nul_free(path, label=label)
    time.sleep(0.15)
    second = _read_regular_nul_free(path, label=label)
    if first != second:
        raise V9DispatchError(f"{label} changed after its writer was closed")
    return second


def _close_pipe_and_drain(pane: str, transcript: Path, main: Path) -> Mapping[str, Any]:
    """Detach a live pipe, or prove it closed with an exited pane, then drain."""
    alive = _pane_alive(pane)
    if alive:
        raise V9DispatchError("driver pane is still live; terminal writers cannot be sealed")
    transcript_bytes = _stable_bytes(transcript, label="tmux transcript")
    main_bytes = _stable_bytes(main, label="main log")
    return {"pane_alive_at_close": alive, "transcript_sha256": hashlib.sha256(transcript_bytes).hexdigest(), "main_sha256": hashlib.sha256(main_bytes).hexdigest(), "drained": True}


def _process_start_ticks(pid: int) -> str | None:
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        # Field 22 follows the final ')' of the process name, which may itself
        # contain spaces or parentheses.
        tail = stat_text.rsplit(")", 1)[1].split()
        return tail[19] if len(tail) > 19 and tail[19].isdigit() else None
    except (FileNotFoundError, OSError, IndexError):
        return None


def _pid_pair_absent(result: Mapping[str, Any]) -> Mapping[str, Any]:
    pid, expected = result["child_pid"], result["child_start_ticks"]
    observed = _process_start_ticks(pid)
    return {"child_pid": pid, "expected_start_ticks": expected, "observed_start_ticks": observed, "pair_absent": observed is None, "pid_reused": observed is not None and observed != expected}


def _parse_result(path: Path, *, run_id: str, command_hash: str, quoted: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(_read_regular_nul_free(path, label="driver result").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V9DispatchError("driver result is not valid UTF-8 JSON") from error
    expected = {"schema_version", "run_id", "command_sha256", "child_pid", "child_start_ticks", "exit_code", "reaped_at"}
    if not isinstance(payload, dict) or set(payload) != expected or payload.get("schema_version") != "stateguard3r.v9-driver-result.v1" or payload.get("run_id") != run_id or payload.get("command_sha256") != command_hash or type(payload.get("child_pid")) is not int or payload["child_pid"] <= 0 or type(payload.get("child_start_ticks")) is not str or not payload["child_start_ticks"].isdigit() or type(payload.get("exit_code")) is not int or not isinstance(payload.get("reaped_at"), str) or not payload["reaped_at"]:
        raise V9DispatchError("driver result is malformed or does not bind the pinned command")
    return payload


def _wait_for_result(path: Path, *, run_id: str, command_hash: str, quoted: str, deadline: float, preexec_main: Path | None = None) -> Mapping[str, Any]:
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if preexec_main is not None:
            failure = _preexec_identity_failure(preexec_main, run_id=run_id)
            if failure is not None:
                raise V9PreexecNoGo(failure)
        if _lexists(path):
            try:
                return _parse_result(path, run_id=run_id, command_hash=command_hash, quoted=quoted)
            except V9DispatchError as error:
                last_error = error
        time.sleep(0.05)
    suffix = f": {last_error}" if last_error is not None else ""
    raise V9DispatchError(f"driver did not publish a complete exclusive result before timeout{suffix}")


def _partial_child_identity(main: Path, *, run_id: str) -> tuple[int, str] | None:
    try:
        text = _read_regular_nul_free(main, label="main log").decode("utf-8")
    except (UnicodeDecodeError, V9DispatchError):
        return None
    prefix = f"V9_DRIVER_CHILD_PID run_id={run_id} child_pid="
    for line in text.splitlines():
        if line.startswith(prefix) and " child_start_ticks=" in line:
            pid_text, start = line[len(prefix):].split(" child_start_ticks=", 1)
            if pid_text.isdigit() and start.isdigit():
                return int(pid_text), start
    return None


def _preexec_identity_failure(main: Path, *, run_id: str) -> Mapping[str, Any] | None:
    """Parse the driver's only legitimate no-result terminal marker.

    This marker is emitted only after ``wait`` has reaped the wrapper.  It is
    intentionally not a driver result: the payload never received a valid go
    token, so a result object would falsely claim that a model invocation had
    occurred.
    """
    try:
        text = _read_regular_nul_free(main, label="main log").decode("utf-8", errors="strict")
    except (UnicodeDecodeError, V9DispatchError):
        return None
    prefix = f"V9_DRIVER_PREEXEC_IDENTITY_FAILURE run_id={run_id} "
    lines = [line for line in text.splitlines() if line.startswith(prefix)]
    if not lines:
        return None
    if len(lines) != 1:
        raise V9DispatchError("pre-exec identity failure marker is ambiguous")
    parts = lines[0].split()
    if len(parts) != 5:
        raise V9DispatchError("pre-exec identity failure marker is malformed")
    fields: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise V9DispatchError("pre-exec identity failure marker is malformed")
        key, value = part.split("=", 1)
        if key in fields:
            raise V9DispatchError("pre-exec identity failure marker is malformed")
        fields[key] = value
    if (
        fields.get("run_id") != run_id
        or not fields.get("child_pid", "").isdigit()
        or int(fields["child_pid"]) <= 0
        or fields.get("reason") not in {
            "start_gate_missing", "start_time_unavailable", "parent_start_time_mismatch", "go_gate_create_failed",
        }
        or not fields.get("exit_code", "").isdigit()
    ):
        raise V9DispatchError("pre-exec identity failure marker is malformed")
    forbidden = [
        f"V9_DRIVER_PAYLOAD_RELEASED run_id={run_id} ",
        f"V9_DRIVER_EXIT run_id={run_id} ",
        f"V9_DRIVER_RESULT_WRITTEN run_id={run_id} ",
    ]
    # A failed exclusive write can occur after identity logging and arming but
    # before the random token exists.  It remains a no-go only if postflight
    # proves that the on-disk gate hash is *not* the authorized token.
    if fields["reason"] != "go_gate_create_failed":
        forbidden.extend((
            f"V9_DRIVER_CHILD_PID run_id={run_id} ",
            f"V9_DRIVER_PAYLOAD_RELEASE_ARMED run_id={run_id} ",
        ))
    if any(marker in text for marker in forbidden):
        raise V9DispatchError("pre-exec identity failure follows a possible payload release")
    return {
        "run_id": run_id,
        "child_pid": int(fields["child_pid"]),
        "reason": fields["reason"],
        "exit_code": int(fields["exit_code"]),
    }


def _terminate_owned_child(identity: tuple[int, str], *, grace_seconds: float = 5.0) -> None:
    pid, expected_start = identity
    current = _process_start_ticks(pid)
    if current is None:
        return
    if current != expected_start:
        raise V9DispatchError("timeout child PID has been reused; refusing to signal another process")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if _process_start_ticks(pid) is None:
            return
        time.sleep(0.05)
    if _process_start_ticks(pid) != expected_start:
        raise V9DispatchError("timeout child PID changed during TERM grace period")
    os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if _process_start_ticks(pid) is None:
            return
        time.sleep(0.05)
    raise V9DispatchError("owned timeout child remained live after TERM/KILL")


def _ordered_stream(data: bytes, *, run_id: str, quoted: str, command_hash: str, result: Mapping[str, Any], transcript: bool) -> None:
    text = data.decode("utf-8", errors="strict")
    markers = [
        f"V9_DRIVER_START run_id={run_id}",
        f"V9_DRIVER_COMMAND={quoted}",
        f"V9_DRIVER_COMMAND_SHA256={command_hash}",
        f"V9_DRIVER_DISPATCHED run_id={run_id} kind=wrapper_only",
        f"V9_DRIVER_CHILD_PID run_id={run_id} child_pid={result['child_pid']} child_start_ticks={result['child_start_ticks']}",
        f"V9_DRIVER_EXIT run_id={run_id} child_pid={result['child_pid']} child_start_ticks={result['child_start_ticks']} exit_code={result['exit_code']} reaped_at={result['reaped_at']}",
        f"V9_DRIVER_RESULT_WRITTEN run_id={run_id} child_pid={result['child_pid']} child_start_ticks={result['child_start_ticks']} exit_code={result['exit_code']}",
    ]
    offsets = [text.find(marker) for marker in markers]
    if any(offset < 0 for offset in offsets) or offsets != sorted(offsets):
        raise V9DispatchError("driver terminal marker sequence is missing or ill ordered")
    if transcript:
        ready = f"V9_PIPE_READY run_id={run_id}"
        ready_at = text.find(ready)
        if ready_at < 0 or ready_at > offsets[0]:
            raise V9DispatchError("tmux pipe-ready marker is missing or late")


def _ordered_preexec_stream(data: bytes, *, run_id: str, quoted: str, command_hash: str, failure: Mapping[str, Any], transcript: bool, ready_start_ticks: str | None = None) -> None:
    """Require the complete, distinctly no-payload terminal stream."""
    text = data.decode("utf-8", errors="strict")
    markers = [
        f"V9_DRIVER_START run_id={run_id}",
        f"V9_DRIVER_COMMAND={quoted}",
        f"V9_DRIVER_COMMAND_SHA256={command_hash}",
        f"V9_DRIVER_DISPATCHED run_id={run_id} kind=wrapper_only",
    ]
    if failure["reason"] == "go_gate_create_failed":
        if ready_start_ticks is None or not ready_start_ticks.isdigit():
            raise V9DispatchError("go-gate pre-exec terminal lacks its READY start-time binding")
        markers.extend((
            f"V9_DRIVER_CHILD_PID run_id={run_id} child_pid={failure['child_pid']} child_start_ticks={ready_start_ticks}",
            f"V9_DRIVER_PAYLOAD_RELEASE_ARMED run_id={run_id} child_pid={failure['child_pid']} child_start_ticks={ready_start_ticks}",
        ))
    markers.append(f"V9_DRIVER_PREEXEC_IDENTITY_FAILURE run_id={run_id} child_pid={failure['child_pid']} reason={failure['reason']} exit_code={failure['exit_code']}")
    offsets = [text.find(marker) for marker in markers]
    if any(offset < 0 for offset in offsets) or offsets != sorted(offsets) or any(text.count(marker) != 1 for marker in markers):
        raise V9DispatchError("pre-exec terminal marker sequence is missing, duplicated, or ill ordered")
    forbidden = [
        f"V9_DRIVER_PAYLOAD_RELEASED run_id={run_id} ",
        f"V9_DRIVER_EXIT run_id={run_id} ",
        f"V9_DRIVER_RESULT_WRITTEN run_id={run_id} ",
    ]
    if failure["reason"] != "go_gate_create_failed":
        forbidden.extend((
            f"V9_DRIVER_CHILD_PID run_id={run_id} ",
            f"V9_DRIVER_PAYLOAD_RELEASE_ARMED run_id={run_id} ",
        ))
    if any(marker in text for marker in forbidden):
        raise V9DispatchError("pre-exec terminal stream contains a payload-capable marker")
    if transcript:
        ready = f"V9_PIPE_READY run_id={run_id}"
        ready_at = text.find(ready)
        if text.count(ready) != 1 or ready_at < 0 or ready_at > offsets[0]:
            raise V9DispatchError("pre-exec tmux pipe-ready marker is missing or late")


def _artifact_digest(path: Path) -> str:
    return hashlib.sha256(_read_regular_nul_free(path, label="terminal artifact")).hexdigest()


def _lease_entry_evidence(paths: ArtifactPaths, name: str) -> Mapping[str, Any]:
    path = _lease_entry(paths, name)
    kind = _lstat_kind(path)
    evidence: dict[str, Any] = {"path": name, "kind": kind}
    if kind == "regular":
        data = path.read_bytes()
        evidence.update({"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "nul_free": b"\0" not in data})
        if b"\0" not in data and len(data) <= 4096:
            try:
                evidence["utf8_text"] = data.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                evidence["utf8_text"] = None
    return evidence


def _preexec_ready_start_ticks(paths: ArtifactPaths, failure: Mapping[str, Any]) -> str | None:
    """Return the exact READY handoff only for the post-identity go failure."""
    if failure["reason"] != "go_gate_create_failed":
        return None
    text = _read_regular_nul_free(_lease_entry(paths, "child-start"), label="pre-exec start gate").decode("utf-8", errors="strict")
    fields = text.split()
    if len(fields) != 3 or fields[0] != "READY" or fields[1] != str(failure["child_pid"]) or not fields[2].isdigit() or text != f"READY {failure['child_pid']} {fields[2]}\n":
        raise V9DispatchError("go-gate pre-exec terminal lacks its exact READY start gate")
    return fields[2]


def _preexec_no_go_evidence(paths: ArtifactPaths, failure: Mapping[str, Any], *, go_token_sha256: str) -> Mapping[str, Any]:
    """Record the immutable boundary that makes a no-result failure safe."""
    go = _lease_entry_evidence(paths, "child-go")
    go_is_authorized = go.get("kind") == "regular" and go.get("sha256") == go_token_sha256
    return {
        "terminal_kind": "PREEXEC_IDENTITY_NO_GO",
        "payload_not_executed": True,
        "wrapper_reaped": True,
        "failure": dict(failure),
        "driver_result_exists": _lexists(paths.result),
        "output_kind": _lstat_kind(paths.output),
        "go_gate": go,
        "go_token_sha256": go_token_sha256,
        "go_gate_authorized": go_is_authorized,
        "start_gate": _lease_entry_evidence(paths, "child-start"),
    }


def _postflight_payload(paths: ArtifactPaths, *, run_id: str, result: Mapping[str, Any] | None, source: Mapping[str, Any] | None, pipe: Mapping[str, Any] | None, error: str | None, preexec_failure: Mapping[str, Any] | None = None, go_token_sha256: str | None = None) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "stateguard3r.v9-postflight.v3",
        "run_id": run_id,
        "finished_at": datetime.now().astimezone().isoformat(),
        "source_provenance": source,
        "pipe_close_and_drain": pipe,
        "error": error,
        "result": result,
        "preexec_no_go": None,
        "output_exists": _lstat_kind(paths.output) != "missing",
    }
    if result is not None:
        payload["pid_start_time_check"] = _pid_pair_absent(result)
    if preexec_failure is not None:
        if go_token_sha256 is None:
            raise V9DispatchError("pre-exec postflight lacks its go-token digest")
        payload["preexec_no_go"] = _preexec_no_go_evidence(paths, preexec_failure, go_token_sha256=go_token_sha256)
    try:
        # A pre-exec NO-GO ran no model forward, so it records GPU state but
        # never demands the 12-GiB admission threshold of a real launch.
        payload["gpu_postflight"] = _snapshot(require_minimum=False)
    except Exception as snapshot_error:  # Preserve a complete failure record.
        payload["gpu_postflight_error"] = str(snapshot_error)
    return payload


def _freeze_primary(paths: ArtifactPaths) -> None:
    for path in (paths.preflight, paths.main, paths.postflight, paths.transcript, paths.driver, paths.result):
        if _lexists(path):
            _freeze(path)


def _regular_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(_read_regular_nul_free(path, label=label).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V9DispatchError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(payload, Mapping):
        raise V9DispatchError(f"{label} must be a JSON object")
    return payload


def _v1_dynamic_runtime() -> float:
    baseline = _regular_json(V1_DYNAMIC_CONTROL / "run.json", label="frozen v1 dynamic run.json")
    runtime = baseline.get("runtime_seconds")
    if type(runtime) not in (int, float) or not math.isfinite(float(runtime)) or not runtime > 0:
        raise V9DispatchError("frozen v1 dynamic runtime is invalid")
    return float(runtime)


def _validate_dynamic_always_control(paths: ArtifactPaths) -> Mapping[str, Any]:
    """Verify the pre-registered Gate-B short circuit without using CUDA."""
    equal: dict[str, bool] = {}
    digests: dict[str, Mapping[str, str]] = {}
    for filename in PROTECTED_MODEL_OUTPUTS:
        baseline = V1_DYNAMIC_CONTROL / filename
        candidate = paths.output / filename
        baseline_bytes = _read_regular_nul_free(baseline, label=f"frozen v1 dynamic {filename}")
        candidate_bytes = _read_regular_nul_free(candidate, label=f"v9 control {filename}")
        equal[filename] = candidate_bytes == baseline_bytes
        digests[filename] = {
            "v1_dynamic_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
            "v9_control_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        }
    if not all(equal.values()):
        mismatched = [name for name, matches in equal.items() if not matches]
        raise V9DispatchError(f"v9 dynamic always-control protected outputs differ from v1: {mismatched}")

    run = _regular_json(paths.output / "run.json", label="v9 control run.json")
    if run.get("status") != "succeeded" or run.get("state_policy", {}).get("name") != "always-commit":
        raise V9DispatchError("v9 dynamic always-control run metadata is not an always-commit success")
    runtime = run.get("runtime_seconds")
    if type(runtime) not in (int, float) or not math.isfinite(float(runtime)) or not runtime > 0:
        raise V9DispatchError("v9 dynamic always-control runtime is invalid")
    baseline_runtime = _v1_dynamic_runtime()
    ratio = float(runtime) / baseline_runtime
    if ratio > RUNTIME_RATIO_LIMIT:
        raise V9DispatchError(f"v9 dynamic always-control runtime ratio exceeds {RUNTIME_RATIO_LIMIT:.2f}: {ratio:.12g}")

    timeline = _regular_json(paths.output / "state-timeline.json", label="v9 control state timeline")
    transactions = timeline.get("transactions")
    if timeline.get("policy") != "always-commit" or timeline.get("pending_transaction_count") != 0 or not isinstance(transactions, list) or len(transactions) != 30:
        raise V9DispatchError("v9 dynamic always-control timeline is not 30 final always-commit transactions")
    for frame_id, transaction in enumerate(transactions):
        expected_keys = {"frame_id", "action", "reason", "current_alarm", "consecutive_rollbacks", "pending_transaction_count", "restore_witness", "export_action", "anchor_frame_ids"}
        if (
            not isinstance(transaction, Mapping)
            or set(transaction) != expected_keys
            or transaction.get("frame_id") != frame_id
            or transaction.get("action") != "commit"
            or transaction.get("reason") != "always_commit_control"
            or transaction.get("current_alarm") is not False
            or transaction.get("consecutive_rollbacks") != 0
            or transaction.get("export_action") != "export_real_camera_pose"
            or transaction.get("pending_transaction_count") != 0
            or transaction.get("restore_witness") is not None
            or transaction.get("anchor_frame_ids") is not None
        ):
            raise V9DispatchError("v9 dynamic always-control transaction is not final/direct raw control")
    return {
        "status": "PASS",
        "protected_model_outputs_byte_identical": equal,
        "protected_model_output_sha256": digests,
        "transaction_count": len(transactions),
        "pending_transaction_count": timeline["pending_transaction_count"],
        "runtime_seconds": float(runtime),
        "v1_dynamic_runtime_seconds": baseline_runtime,
        "runtime_ratio": ratio,
        "runtime_ratio_limit": RUNTIME_RATIO_LIMIT,
    }


def _validate_frozen_driver(paths: ArtifactPaths, *, run_id: str, argv: Sequence[str], command_hash: str, quoted: str) -> str:
    """Bind the generated token-gated driver byte-for-byte to this dispatcher."""
    text = _read_regular_nul_free(paths.driver, label="frozen driver").decode("utf-8", errors="strict")
    lines = [line for line in text.splitlines() if line.startswith("go_token=")]
    if len(lines) != 1:
        raise V9DispatchError("frozen driver does not contain exactly one go token")
    token = lines[0].split("=", 1)[1]
    if len(token) != 64 or any(character not in "0123456789abcdef" for character in token):
        raise V9DispatchError("frozen driver go token is malformed")
    if text != _driver(run_id, argv, paths, command_hash, quoted, token):
        raise V9DispatchError("frozen driver differs from the pinned token-gated template")
    return hashlib.sha256((token + "\n").encode("ascii")).hexdigest()


def _validate_preexec_no_go(paths: ArtifactPaths, *, run_id: str, command_hash: str, quoted: str, postflight: Mapping[str, Any], driver_token_sha256: str) -> Mapping[str, Any]:
    """Validate the deliberately no-result pre-execution terminal state."""
    main_failure = _preexec_identity_failure(paths.main, run_id=run_id)
    transcript_failure = _preexec_identity_failure(paths.transcript, run_id=run_id)
    if main_failure is None or transcript_failure != main_failure:
        raise V9DispatchError("sealed pre-exec failure is absent or differs between terminal streams")
    ready_start_ticks = _preexec_ready_start_ticks(paths, main_failure)
    _ordered_preexec_stream(_read_regular_nul_free(paths.main, label="main"), run_id=run_id, quoted=quoted, command_hash=command_hash, failure=main_failure, transcript=False, ready_start_ticks=ready_start_ticks)
    _ordered_preexec_stream(_read_regular_nul_free(paths.transcript, label="transcript"), run_id=run_id, quoted=quoted, command_hash=command_hash, failure=main_failure, transcript=True, ready_start_ticks=ready_start_ticks)
    if _lexists(paths.result) or _lstat_kind(paths.output) != "missing":
        raise V9DispatchError("sealed pre-exec no-go unexpectedly contains a result or output")
    if postflight.get("schema_version") != "stateguard3r.v9-postflight.v3" or postflight.get("result") is not None or postflight.get("error") != "PREEXEC_IDENTITY_NO_GO":
        raise V9DispatchError("sealed pre-exec postflight schema is invalid")
    no_go = postflight.get("preexec_no_go")
    if not isinstance(no_go, Mapping):
        raise V9DispatchError("sealed pre-exec postflight lacks no-go evidence")
    if (
        no_go.get("terminal_kind") != "PREEXEC_IDENTITY_NO_GO"
        or no_go.get("payload_not_executed") is not True
        or no_go.get("wrapper_reaped") is not True
        or no_go.get("failure") != main_failure
        or no_go.get("driver_result_exists") is not False
        or no_go.get("output_kind") != "missing"
        or no_go.get("go_gate_authorized") is not False
        or no_go.get("go_token_sha256") != driver_token_sha256
    ):
        raise V9DispatchError("sealed pre-exec no-go boundary does not recompute")
    go = no_go.get("go_gate")
    if not isinstance(go, Mapping) or go.get("path") != "child-go":
        raise V9DispatchError("sealed pre-exec go-gate evidence is malformed")
    if go.get("kind") == "regular" and go.get("sha256") == no_go["go_token_sha256"]:
        raise V9DispatchError("sealed pre-exec gate contains the authorized token")
    start_gate = no_go.get("start_gate")
    if not isinstance(start_gate, Mapping) or start_gate.get("path") != "child-start":
        raise V9DispatchError("sealed pre-exec start-gate evidence is malformed")
    reason = main_failure["reason"]
    if reason == "start_gate_missing" and start_gate.get("kind") != "missing":
        raise V9DispatchError("sealed missing-start pre-exec terminal has a start gate")
    if reason == "start_time_unavailable" and start_gate.get("utf8_text") != f"FAIL {main_failure['child_pid']} unavailable\n":
        raise V9DispatchError("sealed unavailable-start pre-exec terminal lacks its exact FAIL gate")
    if reason in {"parent_start_time_mismatch", "go_gate_create_failed"}:
        text = start_gate.get("utf8_text")
        prefix = f"READY {main_failure['child_pid']} "
        if not isinstance(text, str) or not text.startswith(prefix) or not text.endswith("\n") or not text[len(prefix):-1].isdigit():
            raise V9DispatchError("sealed READY-start pre-exec terminal lacks its exact READY gate")
    if not postflight.get("pipe_close_and_drain", {}).get("drained"):
        raise V9DispatchError("sealed pre-exec postflight lacks pipe drain")
    if "gpu_postflight_error" in postflight:
        raise V9DispatchError("sealed pre-exec postflight lacks a GPU snapshot")
    _validate_snapshot(postflight.get("gpu_postflight", {}), require_minimum=False)
    return {"status": "PREEXEC_IDENTITY_NO_GO_EVIDENCE_COMPLETE", "failure": main_failure, "payload_not_executed": True}


def _validator_report(paths: ArtifactPaths, *, run_id: str) -> Mapping[str, Any]:
    """Independently recompute the sealed evidence without any CUDA API call."""
    report: dict[str, Any] = {"schema_version": "stateguard3r.v9-validator.v1", "run_id": run_id, "started_at": datetime.now().astimezone().isoformat(), "cpu_only": True}
    error: str | None = None
    try:
        primary = (paths.preflight, paths.main, paths.postflight, paths.transcript, paths.driver)
        for path in primary:
            if _lstat_kind(path) != "regular" or stat.S_IMODE(path.stat().st_mode) != 0o444:
                raise V9DispatchError(f"sealed primary artifact mode/path invalid: {path}")
            _read_regular_nul_free(path, label="sealed primary artifact")
        preflight = json.loads(_read_regular_nul_free(paths.preflight, label="preflight"))
        postflight = json.loads(_read_regular_nul_free(paths.postflight, label="postflight"))
        if preflight.get("schema_version") != "stateguard3r.v9-preflight.v2" or preflight.get("run_id") != run_id or not isinstance(preflight.get("command"), list):
            raise V9DispatchError("sealed preflight schema is invalid")
        snapshots = preflight.get("gpu_preflights")
        if not isinstance(snapshots, list) or len(snapshots) != 2:
            raise V9DispatchError("sealed preflight does not contain exactly two snapshots")
        for snapshot in snapshots:
            _validate_snapshot(snapshot)
        argv = preflight["command"]
        quoted, command_hash = preflight.get("quoted_command"), preflight.get("command_sha256")
        if not isinstance(quoted, str) or quoted != shlex.join(argv) or command_hash != _command_hash(argv):
            raise V9DispatchError("sealed preflight command/hash does not recompute")
        test_child = preflight.get("test_child")
        if type(test_child) is not bool or _child(argv, paths, test=test_child) != argv:
            raise V9DispatchError("sealed preflight child no longer satisfies its pinned contract")
        recorded_interpreter = preflight.get("interpreter")
        if recorded_interpreter != _interpreter_provenance(argv, test=test_child):
            raise V9DispatchError("sealed production interpreter provenance does not recompute")
        driver_token_sha256 = _validate_frozen_driver(paths, run_id=run_id, argv=argv, command_hash=command_hash, quoted=quoted)
        current_source = _component_provenance()
        if preflight.get("source_provenance") != current_source or postflight.get("source_provenance") != current_source:
            raise V9DispatchError("sealed source provenance does not recompute")
        if not _lexists(paths.result):
            report.update(_validate_preexec_no_go(paths, run_id=run_id, command_hash=command_hash, quoted=quoted, postflight=postflight, driver_token_sha256=driver_token_sha256))
            report.update({"command_sha256": command_hash, "artifact_sha256": {path.name: _artifact_digest(path) for path in primary}})
            raise StopIteration
        if _lstat_kind(paths.result) != "regular" or stat.S_IMODE(paths.result.stat().st_mode) != 0o444:
            raise V9DispatchError("sealed driver result mode/path invalid")
        _read_regular_nul_free(paths.result, label="sealed driver result")
        if postflight.get("schema_version") != "stateguard3r.v9-postflight.v3":
            raise V9DispatchError("sealed postflight schema is invalid")
        result = _parse_result(paths.result, run_id=run_id, command_hash=command_hash, quoted=quoted)
        _ordered_stream(_read_regular_nul_free(paths.main, label="main"), run_id=run_id, quoted=quoted, command_hash=command_hash, result=result, transcript=False)
        _ordered_stream(_read_regular_nul_free(paths.transcript, label="transcript"), run_id=run_id, quoted=quoted, command_hash=command_hash, result=result, transcript=True)
        if postflight.get("result") != result or not postflight.get("pipe_close_and_drain", {}).get("drained"):
            raise V9DispatchError("sealed postflight does not bind result/pipe drain")
        if "gpu_postflight_error" in postflight:
            raise V9DispatchError("sealed postflight lacks a GPU snapshot")
        _validate_snapshot(postflight.get("gpu_postflight", {}), require_minimum=False)
        pid_check = postflight.get("pid_start_time_check")
        if not isinstance(pid_check, Mapping) or pid_check.get("expected_start_ticks") != result["child_start_ticks"] or pid_check.get("pair_absent") is not True or pid_check.get("pid_reused") is not False:
            raise V9DispatchError("sealed postflight lacks PID/start-time absence proof")
        if result["exit_code"] == 0:
            if _lstat_kind(paths.output) != "directory":
                raise V9DispatchError("successful child lacks direct output")
            expected_inventory = _output_inventory(paths.output, require_frozen=True)
            if postflight.get("output_inventory") != expected_inventory:
                raise V9DispatchError("sealed output inventory does not recompute")
            if run_id == CONTROL_RUN_ID:
                report["dynamic_always_control"] = _validate_dynamic_always_control(paths)
            terminal_status = "PASS"
        else:
            terminal_status = "CHILD_NONZERO_EVIDENCE_COMPLETE"
        report.update({"status": terminal_status, "command_sha256": command_hash, "artifact_sha256": {path.name: _artifact_digest(path) for path in (*primary, paths.result)}})
    except StopIteration:
        pass
    except Exception as validation_error:
        error = str(validation_error)
        report.update({"status": "FAIL", "error": error})
    report["finished_at"] = datetime.now().astimezone().isoformat()
    _write_json(paths.validator, report)
    _freeze(paths.validator)
    if error is not None:
        raise V9DispatchError(f"independent v9 validator failed: {error}")
    return report


def _run_independent_validator(run_id: str) -> Mapping[str, Any]:
    """Start a fresh CPU-only interpreter; it alone creates validator-0001."""
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--validate-run-id", run_id], text=True, capture_output=True, env=environment)
    if result.returncode != 0:
        raise V9DispatchError(f"independent validator subprocess failed: {result.stderr.strip() or result.stdout.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise V9DispatchError("independent validator did not emit JSON") from error


def validate_sealed_run(run_id: str) -> Mapping[str, Any]:
    """Public CPU-only validator entry point; it is necessarily one-use."""
    paths = artifact_paths(run_id)
    _canonical_paths(paths)
    if _lstat_kind(paths.lease) != "directory":
        raise V9DispatchError("validator refuses a run without its atomic owner lease")
    if _lexists(paths.validator):
        raise V9DispatchError("validator-0001 already exists; sealed run cannot be revalidated")
    return _validator_report(paths, run_id=run_id)


def dispatch(run_id: str, argv: Sequence[str], *, timeout_seconds: float, allow_noncuda_child_for_test: bool = False) -> Mapping[str, Any]:
    """Dispatch one v9 child once, then seal and independently validate it."""
    if timeout_seconds <= 0:
        raise V9DispatchError("dispatch timeout must be positive")
    paths = artifact_paths(run_id)
    _canonical_paths(paths)
    _acquire_lease(paths)
    _require_empty_lease(paths)
    _fresh_after_lease = [name for name, path in asdict(paths).items() if name != "lease" and _lexists(path)]
    if _fresh_after_lease:
        raise V9DispatchError(f"same-ID lease acquired but owner artifacts already exist: {_fresh_after_lease}")
    child = _child(argv, paths, test=allow_noncuda_child_for_test)
    command_hash, quoted = _command_hash(child), shlex.join(child)
    source: Mapping[str, Any] | None = None
    pane: str | None = None
    result: Mapping[str, Any] | None = None
    pipe: Mapping[str, Any] | None = None
    preexec_failure: Mapping[str, Any] | None = None
    go_token = secrets.token_hex(32)
    go_token_sha256 = hashlib.sha256((go_token + "\n").encode("ascii")).hexdigest()
    error: Exception | None = None
    try:
        source = _component_provenance()
        snapshots = [_snapshot(), _snapshot()]
        if len(snapshots) != 2:
            raise V9DispatchError("dispatcher did not obtain exactly two preflights")
        for snapshot in snapshots:
            _validate_snapshot(snapshot)
        _write_json(paths.preflight, {"schema_version": "stateguard3r.v9-preflight.v2", "run_id": run_id, "command": child, "quoted_command": quoted, "command_sha256": command_hash, "test_child": allow_noncuda_child_for_test, "interpreter": _interpreter_provenance(child, test=allow_noncuda_child_for_test), "gpu_preflights": snapshots, "source_provenance": source})
        _freeze(paths.preflight)
        _write(paths.main, "")
        _write(paths.transcript, "")
        _write(paths.driver, _driver(run_id, child, paths, command_hash, quoted, go_token), 0o555)
        pane = _pane()
        deadline = time.monotonic() + timeout_seconds
        _tmux(["pipe-pane", "-o", "-t", pane, f"cat >> {shlex.quote(str(paths.transcript))}"])
        ready = f"V9_PIPE_READY run_id={run_id}"
        _tmux(["send-keys", "-t", pane, f"printf '%s\\n' {shlex.quote(ready)}", "C-m"])
        _wait_for_text(paths.transcript, ready, deadline)
        _tmux(["send-keys", "-t", pane, f"bash {shlex.quote(str(paths.driver))}; exit", "C-m"])
        try:
            result = _wait_for_result(paths.result, run_id=run_id, command_hash=command_hash, quoted=quoted, deadline=deadline, preexec_main=paths.main)
        except V9PreexecNoGo as preexec_event:
            preexec_failure = preexec_event.failure
            # The exact marker is written only after bash has waited for its
            # wrapper.  Close the pane and drain both original streams
            # immediately; there is no result because no payload was released,
            # not because a result writer merely timed out.
            _wait_for_driver_exit(pane, time.monotonic() + 10.0)
            pipe = _close_pipe_and_drain(pane, paths.transcript, paths.main)
            transcript_failure = _preexec_identity_failure(paths.transcript, run_id=run_id)
            if transcript_failure != preexec_failure:
                raise V9DispatchError("pre-exec identity failure is not identically recorded in both terminal streams") from preexec_event
            ready_start_ticks = _preexec_ready_start_ticks(paths, preexec_failure)
            _ordered_preexec_stream(_read_regular_nul_free(paths.main, label="main"), run_id=run_id, quoted=quoted, command_hash=command_hash, failure=preexec_failure, transcript=False, ready_start_ticks=ready_start_ticks)
            _ordered_preexec_stream(_read_regular_nul_free(paths.transcript, label="transcript"), run_id=run_id, quoted=quoted, command_hash=command_hash, failure=preexec_failure, transcript=True, ready_start_ticks=ready_start_ticks)
            if _lexists(paths.result) or _lstat_kind(paths.output) != "missing":
                raise V9DispatchError("pre-exec identity failure unexpectedly has a result or output") from preexec_event
            post_source = _component_provenance()
            if post_source != source:
                raise V9DispatchError("source/worktree drifted during pre-exec identity gate") from preexec_event
            postflight = dict(_postflight_payload(
                paths, run_id=run_id, result=None, source=post_source, pipe=pipe,
                error="PREEXEC_IDENTITY_NO_GO", preexec_failure=preexec_failure,
                go_token_sha256=go_token_sha256,
            ))
            no_go = postflight["preexec_no_go"]
            if (
                not isinstance(no_go, Mapping)
                or no_go.get("go_gate_authorized") is not False
                or no_go.get("driver_result_exists") is not False
                or no_go.get("output_kind") != "missing"
            ):
                raise V9DispatchError("pre-exec identity failure cannot prove that the payload go gate stayed closed") from preexec_event
            if "gpu_postflight_error" in postflight:
                raise V9DispatchError("pre-exec identity failure lacks a GPU postflight snapshot") from preexec_event
            _validate_snapshot(postflight.get("gpu_postflight", {}), require_minimum=False)
            _write_json(paths.postflight, postflight)
            raise V9DispatchError("driver terminated before payload release because its PID/start-time identity could not be established") from preexec_event
        except V9DispatchError as timeout_error:
            identity = _partial_child_identity(paths.main, run_id=run_id)
            if identity is None:
                raise V9DispatchError("driver timed out without a PID/start-time marker; refusing to seal a possibly live child") from timeout_error
            _terminate_owned_child(identity)
            result = _wait_for_result(paths.result, run_id=run_id, command_hash=command_hash, quoted=quoted, deadline=time.monotonic() + 10.0)
            _wait_for_driver_exit(pane, time.monotonic() + 10.0)
            pipe = _close_pipe_and_drain(pane, paths.transcript, paths.main)
            _ordered_stream(_read_regular_nul_free(paths.main, label="main"), run_id=run_id, quoted=quoted, command_hash=command_hash, result=result, transcript=False)
            _ordered_stream(_read_regular_nul_free(paths.transcript, label="transcript"), run_id=run_id, quoted=quoted, command_hash=command_hash, result=result, transcript=True)
            raise V9DispatchError("driver timed out; verified owned child was terminated and reaped") from timeout_error
        _wait_for_driver_exit(pane, deadline)
        pipe = _close_pipe_and_drain(pane, paths.transcript, paths.main)
        _ordered_stream(_read_regular_nul_free(paths.main, label="main"), run_id=run_id, quoted=quoted, command_hash=command_hash, result=result, transcript=False)
        _ordered_stream(_read_regular_nul_free(paths.transcript, label="transcript"), run_id=run_id, quoted=quoted, command_hash=command_hash, result=result, transcript=True)
        pid_check = _pid_pair_absent(result)
        if not pid_check["pair_absent"] or pid_check["pid_reused"]:
            raise V9DispatchError("recorded child PID/start-time pair is live or reused at postflight")
        post_source = _component_provenance()
        if post_source != source:
            raise V9DispatchError("source/worktree drifted during v9 child")
        postflight = dict(_postflight_payload(paths, run_id=run_id, result=result, source=post_source, pipe=pipe, error=None))
        postflight["pid_start_time_check"] = pid_check
        postflight["output_inventory"] = _freeze_output(paths.output) if result["exit_code"] == 0 else None
        if "gpu_postflight_error" in postflight:
            raise V9DispatchError("postflight GPU snapshot could not be collected")
        _validate_snapshot(postflight.get("gpu_postflight", {}), require_minimum=False)
        _write_json(paths.postflight, postflight)
    except Exception as caught:
        error = caught
        # If pipe setup occurred, never freeze a potentially live transcript.
        if pane is not None and pipe is None and _lexists(paths.transcript):
            try:
                pipe = _close_pipe_and_drain(pane, paths.transcript, paths.main)
            except Exception as pipe_error:
                pipe = {"drained": False, "close_error": str(pipe_error)}
        if _lexists(paths.preflight) and not _lexists(paths.postflight):
            try:
                postflight = dict(_postflight_payload(
                    paths, run_id=run_id, result=result, source=source, pipe=pipe, error=str(caught),
                    preexec_failure=preexec_failure, go_token_sha256=go_token_sha256 if preexec_failure is not None else None,
                ))
                if result is not None and result.get("exit_code") == 0 and _lstat_kind(paths.output) == "directory":
                    try:
                        postflight["output_inventory"] = _freeze_output(paths.output)
                    except Exception as output_error:
                        postflight["output_freeze_error"] = str(output_error)
                _write_json(paths.postflight, postflight)
            except Exception:
                pass
    finally:
        # Driver/result/output failure is still terminal evidence: freeze every
        # successfully created primary artifact only after every terminal
        # writer has been observed closed and drained.
        if pane is None or (pipe is not None and pipe.get("drained") is True and pipe.get("pane_alive_at_close") is False):
            try:
                _freeze_primary(paths)
            except Exception as freeze_error:
                error = error or freeze_error
        else:
            error = error or V9DispatchError("terminal writers were not closed/drained; refusing to freeze live evidence")
    if error is None:
        try:
            validator = _run_independent_validator(run_id)
        except Exception as validator_error:
            error = validator_error
    else:
        # A nonzero child or protocol failure is still independently audited
        # when all primary artifacts are complete enough to do so.
        normal_primary = (paths.preflight, paths.main, paths.postflight, paths.transcript, paths.driver, paths.result)
        preexec_primary = (paths.preflight, paths.main, paths.postflight, paths.transcript, paths.driver)
        if all(_lexists(path) for path in normal_primary) or (preexec_failure is not None and all(_lexists(path) for path in preexec_primary)):
            try:
                _run_independent_validator(run_id)
            except Exception:
                pass
    if error is not None:
        raise V9DispatchError(f"v9 terminal evidence sealed as failure: {error}") from error
    assert result is not None
    if result["exit_code"] != 0:
        raise V9DispatchError(f"v9 child exited nonzero with immutable evidence: {result['exit_code']}")
    return {"status": "PASS", "run_id": run_id, "pane": pane, "command_sha256": command_hash, "validator": validator, "paths": {name: str(path) for name, path in asdict(paths).items()}}


def _read_command_json(path: Path) -> list[str]:
    expected_parent = ROOT / "tmp"
    if path.parent != expected_parent or _lstat_kind(path) != "regular":
        raise V9DispatchError("command JSON must be a regular direct tmp child")
    try:
        payload = json.loads(_read_regular_nul_free(path, label="command JSON"))
    except json.JSONDecodeError as error:
        raise V9DispatchError("command JSON is malformed") from error
    if not isinstance(payload, list):
        raise V9DispatchError("command JSON must be an argv array")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.validate_run_id is not None:
            print(json.dumps(validate_sealed_run(args.validate_run_id), ensure_ascii=False, sort_keys=True))
        else:
            if args.run_id is None:
                raise V9DispatchError("--run-id is required with --command-json")
            command = _read_command_json(args.command_json)
            print(json.dumps(dispatch(args.run_id, command, timeout_seconds=args.timeout_seconds), ensure_ascii=False, sort_keys=True))
        return 0
    except V9DispatchError as error:
        print(f"V9_DISPATCH_ERROR={error}", file=sys.stderr)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
