#!/usr/bin/env python3
"""One-use CPU-only v17 frame-zero interface probe with a three-file ledger."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from stateguard3r.frame_zero_probe_capability_v17 import (
    MODEL_LOADER,
    PROBE_CAPSULE_ID,
    parse_frame_zero_probe_capability_v17_bytes,
)


RUN_ID = "recovery-beta-floor-v17-dynamic-frame0-cpu-interface-probe-0001"
CAPSULE = ROOT / "outputs" / f"{PROBE_CAPSULE_ID}.json"
EVIDENCE = ROOT / "logs" / f"{RUN_ID}.json"
RECAL3R_ROOT = ROOT.parent / "baselines" / "ReCal3R"
RECAL3R_COMMIT = "466c7cdf3acd2f589f1d82e5f6391966f19db9ff"
CHECKPOINT = RECAL3R_ROOT / "src" / "cut3r_512_dpt_4_64.pth"
CHECKPOINT_SHA256 = "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103"
STATE_FEATURE_SHAPE = (1, 768, 768)
MEMORY_SHAPE = (1, 256, 1536)
UPDATE_MASK_SHAPE = (1, 768, 1)


class InterfaceProbeV17Error(RuntimeError):
    """The frame-zero CPU interface boundary is not established."""


def _absolute(path: Path | str) -> Path:
    """Normalize a supplied path without following a possible final symlink."""
    return Path(os.path.abspath(os.fspath(path)))


def _read_only_regular(path: Path, *, label: str) -> os.stat_result:
    try:
        value = os.lstat(path)
    except OSError as error:
        raise InterfaceProbeV17Error(f"cannot stat {label}") from error
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode) or stat.S_IMODE(value.st_mode) != 0o444:
        raise InterfaceProbeV17Error(f"{label} must be a non-symlink mode-0444 regular file")
    return value


def _same_snapshot(first: os.stat_result, second: os.stat_result, *, label: str) -> None:
    if (first.st_size, first.st_ino, first.st_mtime_ns, stat.S_IMODE(first.st_mode)) != (
        second.st_size, second.st_ino, second.st_mtime_ns, stat.S_IMODE(second.st_mode),
    ):
        raise InterfaceProbeV17Error(f"{label} changed while it was being read")


class DataAccessLedgerV17:
    """Fail closed to exactly the capsule, its RGB, and the checkpoint."""

    def __init__(self, *, capsule: Path, frame_zero_rgb: Path | None, checkpoint: Path) -> None:
        self._expected = {
            "probe_capsule": _absolute(capsule),
            "checkpoint": _absolute(checkpoint),
        }
        if frame_zero_rgb is not None:
            self._expected["frame_zero_rgb"] = _absolute(frame_zero_rgb)
        self._entries: list[dict[str, str]] = []

    @property
    def entries(self) -> list[Mapping[str, str]]:
        return [dict(entry) for entry in self._entries]

    def _claim(self, path: Path | str, label: str) -> tuple[Path, dict[str, str]]:
        candidate = _absolute(path)
        if label not in self._expected or candidate != self._expected[label]:
            raise InterfaceProbeV17Error("v17 probe attempted a forbidden data resource")
        entry = next((current for current in self._entries if current["resource"] == label), None)
        if entry is None:
            entry = {"resource": label, "path": str(self._expected[label]), "access_count": "0"}
            self._entries.append(entry)
        entry["access_count"] = str(int(entry["access_count"]) + 1)
        return self._expected[label], entry

    def bind_frame_zero_rgb(self, path: Path) -> None:
        if "frame_zero_rgb" in self._expected:
            raise InterfaceProbeV17Error("v17 probe frame-zero resource is already bound")
        self._expected["frame_zero_rgb"] = _absolute(path)

    def read(self, path: Path, label: str) -> bytes:
        resource, entry = self._claim(path, label)
        before = _read_only_regular(resource, label=f"v17 probe {label}")
        try:
            payload = resource.read_bytes()
        except OSError as error:
            raise InterfaceProbeV17Error(f"cannot read v17 probe {label}") from error
        after = _read_only_regular(resource, label=f"v17 probe {label}")
        _same_snapshot(before, after, label=f"v17 probe {label}")
        entry.update({
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": str(before.st_size),
            "inode": str(before.st_ino),
            "mtime_ns": str(before.st_mtime_ns),
            "mode_octal": "0444",
        })
        return payload

    def permit_external_open(self, path: str | os.PathLike[str], label: str) -> None:
        self._claim(path, label)

    def sha256(self, label: str) -> str:
        for entry in self._entries:
            if entry["resource"] == label and "sha256" in entry:
                return entry["sha256"]
        raise InterfaceProbeV17Error("v17 probe resource has no recorded digest")

    def assert_exact_resources(self) -> None:
        names = [entry["resource"] for entry in self._entries]
        if names != ["probe_capsule", "frame_zero_rgb", "checkpoint"]:
            raise InterfaceProbeV17Error("v17 probe data-access ledger does not have exactly three entries")
        if any("sha256" not in entry for entry in self._entries):
            raise InterfaceProbeV17Error("v17 probe data-access ledger has an unhashed resource")


def _verify_frame_zero_snapshot(capability: Any, raw: bytes) -> None:
    current = _read_only_regular(capability.rgb_path, label="v17 frame-zero RGB")
    if (current.st_size != capability.size_bytes or current.st_ino != capability.inode or current.st_mtime_ns != capability.mtime_ns or hashlib.sha256(raw).hexdigest() != capability.sha256):
        raise InterfaceProbeV17Error("v17 frame-zero RGB differs from the isolated capability")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, default=CAPSULE)
    parser.add_argument("--evidence-json", type=Path, default=EVIDENCE)
    parser.add_argument("--run-id", default=RUN_ID)
    return parser


def _require_fixed_contract(args: argparse.Namespace) -> None:
    if (
        args.run_id != RUN_ID
        or _absolute(args.capsule) != _absolute(CAPSULE)
        or _absolute(args.evidence_json) != _absolute(EVIDENCE)
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise InterfaceProbeV17Error("v17 probe fixed path/run/CUDA contract differs")


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repository), *args], check=True, text=True, capture_output=True).stdout.strip()


def _verify_baseline_pin() -> None:
    if _git(RECAL3R_ROOT, "rev-parse", "HEAD") != RECAL3R_COMMIT:
        raise InterfaceProbeV17Error("v17 probe baseline commit differs from the pin")
    if _git(RECAL3R_ROOT, "status", "--porcelain", "--untracked-files=no"):
        raise InterfaceProbeV17Error("v17 probe requires a clean tracked baseline worktree")


def _blocked(name: str) -> Any:
    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(f"v17 CPU interface probe called prohibited path: {name}")

    return fail


def _block_module_forward(module: Any, name: str) -> None:
    if module is not None and callable(getattr(module, "forward", None)):
        module.forward = _blocked(name)


def _block_model_execution(model: Any) -> None:
    method_names = (
        "_forward_impl", "_encode_image", "_encode_ray_map", "_init_state",
        "_get_img_level_feat", "_recurrent_rollout", "_downstream_head",
        "_compute_recal3r_update_mask", "_advance_recal3r_sequence_age",
        "_maybe_record_u_calibration_step", "_reset_update" + "_pressure_if_needed",
        "_reset_recal3r_reference_state_if_needed", "forward", "forward_recurrent",
        "forward_recurrent_lighter", "inference_step", "_det" + "ector",
        "beta_base_floor_" + "operator",
    )
    for name in method_names:
        setattr(model, name, _blocked(name))
    group_names = (
        "patch" + "_embed", "patch" + "_embed_ray_map", "enc" + "_blocks",
        "enc" + "_blocks_ray_map", "dec" + "_blocks", "dec" + "_blocks_state",
        "downstream_head",
    )
    for group_name in group_names:
        group = getattr(model, group_name, None)
        if isinstance(group, (tuple, list)) or hasattr(group, "__iter__"):
            for index, block in enumerate(group):
                _block_module_forward(block, f"{group_name}[{index}]")
        else:
            _block_module_forward(group, group_name)
    _block_module_forward(getattr(model, "rope", None), "RoPE")
    retriever = getattr(model, "pose_retriever", None)
    if retriever is not None:
        retriever.inquire = _blocked("pose_retriever.inquire")
        retriever.update_mem = _blocked("pose_retriever.update_mem")
        _block_module_forward(retriever, "pose_retriever.forward")
        for group_name in ("read_blocks", "write_blocks"):
            for index, block in enumerate(getattr(retriever, group_name, ())):
                _block_module_forward(block, f"pose_retriever.{group_name}[{index}]")


def _tensor_metadata(value: Any) -> Mapping[str, Any]:
    shape = tuple(getattr(value, "shape", ()))
    return {
        "shape": [int(dimension) for dimension in shape],
        "dtype": str(getattr(value, "dtype", "")),
        "device": str(getattr(getattr(value, "device", None), "type", "")),
    }


def _interfaces(model: Any) -> Mapping[str, Mapping[str, Any]]:
    register = getattr(getattr(model, "register_tokens", None), "weight", None)
    memory = getattr(getattr(model, "pose_retriever", None), "mem", None)
    projection = getattr(model, "dec" + "oder_embed_state", None)
    if tuple(getattr(register, "shape", ())) != (768, 1024):
        raise InterfaceProbeV17Error("v17 register-token interface differs")
    if tuple(getattr(memory, "shape", ())) != MEMORY_SHAPE:
        raise InterfaceProbeV17Error("v17 pose-memory interface differs")
    if getattr(projection, "in_features", None) != 1024 or getattr(projection, "out_features", None) != 768:
        raise InterfaceProbeV17Error("v17 state-projection interface differs")
    if getattr(getattr(register, "device", None), "type", None) != "cpu" or getattr(getattr(memory, "device", None), "type", None) != "cpu":
        raise InterfaceProbeV17Error("v17 interface tensors are not CPU-resident")
    return {
        "state_feature": {"shape": list(STATE_FEATURE_SHAPE), "dtype": str(register.dtype), "device": "cpu"},
        "pose_memory": _tensor_metadata(memory),
        "native_update_mask": {"shape": list(UPDATE_MASK_SHAPE), "dtype": str(register.dtype), "device": "cpu"},
        "register_tokens": _tensor_metadata(register),
        "state_projection": {"in_features": 1024, "out_features": 768},
    }


def _write_immutable_evidence(path: Path, payload: Mapping[str, Any]) -> None:
    target = _absolute(path)
    logs_root = _absolute(ROOT / "logs")
    if target.parent != logs_root:
        raise InterfaceProbeV17Error("v17 probe evidence must be a direct logs child")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError as error:
        raise InterfaceProbeV17Error("v17 probe evidence path is already occupied") from error
    try:
        encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.chmod(target, 0o444)
        except OSError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _require_fixed_contract(args)
    _verify_baseline_pin()
    ledger = DataAccessLedgerV17(capsule=CAPSULE, frame_zero_rgb=None, checkpoint=CHECKPOINT)
    capability = parse_frame_zero_probe_capability_v17_bytes(
        ledger.read(CAPSULE, "probe_capsule"), capsule_path=CAPSULE,
    )
    if capability.path != CAPSULE.resolve(strict=True):
        raise InterfaceProbeV17Error("v17 frame-zero capability differs")
    ledger.bind_frame_zero_rgb(capability.rgb_path)
    frame_path = capability.rgb_path
    _verify_frame_zero_snapshot(capability, ledger.read(frame_path, "frame_zero_rgb"))
    if hashlib.sha256(ledger.read(CHECKPOINT, "checkpoint")).hexdigest() != CHECKPOINT_SHA256:
        raise InterfaceProbeV17Error("v17 checkpoint hash differs")
    # CUDA is hidden by the fixed contract before this first torch import.
    sys.path[:0] = [str(RECAL3R_ROOT), str(RECAL3R_ROOT / "src")]
    import torch

    if torch.cuda.is_initialized():
        raise InterfaceProbeV17Error("v17 CPU probe found initialized CUDA")
    import add_ckpt_path

    add_ckpt_path.add_path_to_dust3r(str(CHECKPOINT))
    import dust3r.model as dust3r_model
    from dust3r.utils.image import load_images_for_eval
    import PIL.Image

    original_image_open = PIL.Image.open

    def audited_image_open(fp: Any, *open_args: Any, **open_kwargs: Any) -> Any:
        ledger.permit_external_open(fp, "frame_zero_rgb")
        return original_image_open(fp, *open_args, **open_kwargs)

    PIL.Image.open = audited_image_open
    try:
        loaded = load_images_for_eval(
            [str(frame_path)], size=MODEL_LOADER["size"], crop=MODEL_LOADER["crop"],
            square_ok=MODEL_LOADER["square_ok"], verbose=True,
        )
    finally:
        PIL.Image.open = original_image_open
    if len(loaded) != 1:
        raise InterfaceProbeV17Error("v17 loader returned a non-singleton frame-zero batch")
    image = loaded[0].get("img")
    original_shape = loaded[0].get("true_shape")
    if (tuple(getattr(image, "shape", ()))[:2] != (1, 3) or tuple(getattr(original_shape, "shape", ())) != (1, 2) or getattr(getattr(image, "device", None), "type", None) != "cpu"):
        raise InterfaceProbeV17Error("v17 frame-zero loader interface differs")
    original_torch_load = torch.load

    def audited_torch_load(file_name: Any, *load_args: Any, **load_kwargs: Any) -> Any:
        ledger.permit_external_open(file_name, "checkpoint")
        return original_torch_load(file_name, *load_args, **load_kwargs)

    torch.load = audited_torch_load
    try:
        model = dust3r_model.ARCroco3DStereo.from_pretrained(str(CHECKPOINT)).to("cpu")
    finally:
        torch.load = original_torch_load
    model.eval()
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise InterfaceProbeV17Error("v17 checkpoint model is not entirely CPU-resident")
    _block_model_execution(model)
    interfaces = _interfaces(model)
    if torch.cuda.is_initialized():
        raise InterfaceProbeV17Error("v17 CPU probe unexpectedly initialized CUDA")
    ledger.assert_exact_resources()
    payload = {
        "schema_version": "stateguard3r.v17-cpu-interface-probe.v1",
        "run_id": RUN_ID,
        "status": "passed",
        "started_at": datetime.now().astimezone().isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_visible_devices": "",
        "cuda_initialized": False,
        "baseline_root": str(RECAL3R_ROOT),
        "baseline_commit": RECAL3R_COMMIT,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "probe_capsule": str(capability.path),
        "probe_capsule_sha256": ledger.sha256("probe_capsule"),
        "probe_capsule_id": PROBE_CAPSULE_ID,
        "frame_id": 0,
        "frame_zero_rgb_path": str(frame_path),
        "frame_zero_rgb_sha256": capability.sha256,
        "loaded_image": _tensor_metadata(image),
        "loaded_true_shape": _tensor_metadata(original_shape),
        "interfaces": interfaces,
        "data_access_ledger": ledger.entries,
        "model_forward_executed": False,
        "image_values_serialized": False,
        "tensor_values_serialized": False,
        "blocked_execution_routes": True,
    }
    _write_immutable_evidence(EVIDENCE, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
