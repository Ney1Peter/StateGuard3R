#!/usr/bin/env python3
"""CPU-only Gate-A metadata probe for v13 persistent ReCal3R state.

This is deliberately *not* a partial inference.  It loads one disclosed
dynamic frame only to bind the input boundary and loads the existing checkpoint
only to bind its state/memory interface.  It must not enter the image encoder,
RoPE, recurrent rollout, detector, state repair, output head, or any GPU path.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import scripts.run_recal3r_smoke as smoke


RUN_ID = "recovery-selective-state-memory-repair-v13-dynamic-frame0-cpu-interface-probe-0001"
DEVELOPMENT_DYNAMIC_MANIFEST = (
    ROOT / "outputs" / "formal-v1-inputs-0001" / "development" /
    "development-dynamic" / "input-manifest.json"
)
PINNED_RECAL3R_COMMIT = "466c7cdf3acd2f589f1d82e5f6391966f19db9ff"
CHECKPOINT_SHA256 = "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103"
STATE_REGISTER_SHAPE = (768, 1024)
STATE_FEATURE_SHAPE = (1, 768, 768)
MEMORY_SHAPE = (1, 256, 1536)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--evidence-json", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    return parser


def _require_cuda_disabled_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("v13 CPU interface probe requires CUDA_VISIBLE_DEVICES='' before torch import")


def _require_fixed_run_id(value: Any) -> str:
    if type(value) is not str or value != RUN_ID:
        raise RuntimeError("v13 CPU interface probe run id is fixed and one-use")
    return value


def _require_fixed_evidence_path(value: Path) -> Path:
    expected = ROOT / "logs" / f"{RUN_ID}.json"
    if value.resolve(strict=False) != expected.resolve(strict=False):
        raise RuntimeError("v13 CPU interface probe evidence path is fixed and one-use")
    return expected


def _exact_dynamic_frame_zero(manifest: Path) -> tuple[Path, Mapping[str, Any]]:
    resolved = manifest.resolve(strict=True)
    if resolved != DEVELOPMENT_DYNAMIC_MANIFEST.resolve(strict=True):
        raise RuntimeError("v13 CPU interface probe accepts only the disclosed dynamic manifest")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    frames = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(frames, list) or not frames or not isinstance(frames[0], dict):
        raise RuntimeError("v13 dynamic manifest frame zero is malformed")
    frame = frames[0]
    if frame.get("frame_index") != 0 or not isinstance(frame.get("path"), str):
        raise RuntimeError("v13 dynamic manifest frame zero is malformed")
    return (resolved.parent / frame["path"]).resolve(strict=True), frame


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.resolve(strict=False)
    logs_root = (ROOT / "logs").resolve(strict=True)
    if target.parent != logs_root or target.exists() or os.path.lexists(target):
        raise RuntimeError("v13 CPU interface evidence must be a new direct logs child")
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=logs_root)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, 0o444)
        os.link(temporary_name, target)
    except FileExistsError as error:
        raise RuntimeError("v13 CPU interface evidence target already exists") from error
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _prohibited(name: str) -> Any:
    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(f"v13 CPU interface probe called prohibited path: {name}")

    return blocked


def _guard_all_model_execution(model: Any) -> None:
    """Bind this probe to construction/interface inspection, never a forward."""
    for name in (
        "_encode_image", "_encode_ray_map", "_init_state", "_encode_state",
        "_recurrent_rollout", "_downstream_head", "forward", "forward_recurrent",
        "forward_recurrent_lighter",
    ):
        setattr(model, name, _prohibited(name))
    for group_name in ("enc_blocks", "enc_blocks_ray_map", "dec_blocks", "dec_blocks_state"):
        for block in getattr(model, group_name, ()):
            if callable(getattr(block, "forward", None)):
                block.forward = _prohibited(group_name)
    rope = getattr(model, "rope", None)
    if rope is not None and callable(getattr(rope, "forward", None)):
        rope.forward = _prohibited("RoPE")
    retriever = getattr(model, "pose_retriever", None)
    if retriever is not None:
        retriever.inquire = _prohibited("pose_retriever.inquire")
        retriever.update_mem = _prohibited("pose_retriever.update_mem")
        if callable(getattr(retriever, "forward", None)):
            retriever.forward = _prohibited("pose_retriever.forward")


def _interface_metadata(model: Any, *, torch: Any) -> Mapping[str, Any]:
    config = getattr(model, "config", None)
    register = getattr(getattr(model, "register_tokens", None), "weight", None)
    memory = getattr(getattr(model, "pose_retriever", None), "mem", None)
    decoder_state = getattr(model, "decoder_embed_state", None)
    if tuple(getattr(register, "shape", ())) != STATE_REGISTER_SHAPE:
        raise RuntimeError("v13 checkpoint register-token interface differs")
    if tuple(getattr(memory, "shape", ())) != MEMORY_SHAPE:
        raise RuntimeError("v13 checkpoint pose-memory interface differs")
    # The pinned checkpoint loader intentionally reconstructs only the
    # checkpoint's explicit constructor arguments.  `state_size`,
    # `local_mem_size`, `state_pe`, and `pose_head` are therefore absent from
    # its runtime config when they equal upstream defaults.  Bind the actual
    # instantiated tensor/module interfaces instead of treating omitted
    # default-valued config fields as a different model.
    if getattr(memory, "device", None).type != "cpu" or getattr(register, "device", None).type != "cpu":
        raise RuntimeError("v13 CPU interface tensors are not CPU resident")
    if not bool(torch.isfinite(register).all()) or not bool(torch.isfinite(memory).all()):
        raise RuntimeError("v13 checkpoint persistent interface tensors are nonfinite")
    if getattr(decoder_state, "in_features", None) != STATE_REGISTER_SHAPE[-1] or getattr(decoder_state, "out_features", None) != STATE_FEATURE_SHAPE[-1]:
        raise RuntimeError("v13 checkpoint state projection interface differs")
    return {
        "checkpoint_loader_omits_default_state_fields": True,
        "state_size_bound_by_register_tokens": STATE_REGISTER_SHAPE[0],
        "local_mem_size_bound_by_pose_retriever_memory": MEMORY_SHAPE[1],
        "state_pe_bound_by_pinned_source": "2d",
        "pose_head_bound_by_pose_retriever_interface": True,
        "register_tokens_shape": list(register.shape),
        "register_tokens_dtype": str(register.dtype),
        "register_tokens_device": str(register.device),
        "decoder_embed_state_in_features": int(decoder_state.in_features),
        "decoder_embed_state_out_features": int(decoder_state.out_features),
        "derived_state_feature_shape": list(STATE_FEATURE_SHAPE),
        "pose_retriever_memory_shape": list(memory.shape),
        "pose_retriever_memory_dtype": str(memory.dtype),
        "pose_retriever_memory_device": str(memory.device),
        "persistent_tensors_finite": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _require_fixed_run_id(args.run_id)
    evidence = _require_fixed_evidence_path(args.evidence_json)
    _require_cuda_disabled_environment()
    baseline_root = args.baseline_root.resolve(strict=True)
    baseline_src = (baseline_root / "src").resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    if args.checkpoint_sha256 != CHECKPOINT_SHA256 or _sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("v13 CPU interface probe checkpoint SHA-256 differs")
    if smoke._git_output(baseline_root, "rev-parse", "HEAD") != PINNED_RECAL3R_COMMIT:
        raise RuntimeError("v13 CPU interface probe ReCal3R commit differs from the pin")
    if smoke._git_output(baseline_root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("v13 CPU interface probe requires a clean tracked ReCal3R worktree")
    image_path, manifest_frame = _exact_dynamic_frame_zero(args.input_manifest)
    if str(baseline_root) not in sys.path:
        sys.path.insert(0, str(baseline_root))
    if str(baseline_src) not in sys.path:
        sys.path.insert(0, str(baseline_src))

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("v13 CPU interface probe refuses an initialized CUDA runtime")
    import add_ckpt_path

    add_ckpt_path.add_path_to_dust3r(str(baseline_src / checkpoint.name))
    import dust3r.model as dust3r_model

    started_at = datetime.now().astimezone().isoformat()
    model, checkpoint_audit = smoke._load_model_with_state_dict_audit(
        dust3r_model.ARCroco3DStereo, checkpoint, torch,
    )
    if checkpoint_audit["strict"] is not False or checkpoint_audit["missing_keys"] or checkpoint_audit["unexpected_keys"]:
        raise RuntimeError("v13 CPU interface checkpoint compatibility audit failed")
    model = model.to("cpu")
    model.eval()
    _guard_all_model_execution(model)
    interface = _interface_metadata(model, torch=torch)
    if torch.cuda.is_initialized():
        raise RuntimeError("v13 CPU interface probe unexpectedly initialized CUDA")
    payload = {
        "schema_version": "stateguard3r.v13-cpu-state-memory-interface-probe.v1",
        "run_id": args.run_id,
        "status": "passed",
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_initialized": False,
        "baseline_root": str(baseline_root),
        "baseline_commit": PINNED_RECAL3R_COMMIT,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_load_audit": checkpoint_audit,
        "checkpoint_compatible_without_missing_or_unexpected_keys": True,
        "input_manifest": str(args.input_manifest.resolve(strict=True)),
        "input_manifest_sha256": _sha256(args.input_manifest.resolve(strict=True)),
        "frame_id": 0,
        "frame_path": str(image_path),
        "frame_sha256": _sha256(image_path),
        "manifest_frame_rgb_sha256": manifest_frame.get("metadata", {}).get("rgb_sha256"),
        "interface": interface,
        "prohibited_paths": [
            "_encode_image", "_encode_ray_map", "encoder_blocks", "RoPE",
            "_init_state", "_recurrent_rollout", "pose_retriever.inquire",
            "pose_retriever.update_mem", "_downstream_head", "detector",
            "selective_state_memory_repair", "ground_truth", "future_frame",
        ],
        "model_forward_executed": False,
        "state_memory_values_serialized": False,
        "image_values_serialized": False,
    }
    _write_immutable_json(evidence, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
