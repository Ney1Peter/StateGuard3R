#!/usr/bin/env python3
"""CPU-only one-frame Gate-A probe for the v12 patch-embedding boundary.

The only model computation permitted here is the current-image path
``load_images_for_eval -> patch_embed -> mean/projection -> official pose head
-> official postprocess -> official camera decoder``.  In particular, this is
not a shortened recurrent inference: the native full image encoder, RoPE,
pose-retriever memory, recurrent rollout, detector, ground truth, and every
other frame are deliberately unavailable.

It writes a new immutable direct child of ``logs/`` only after that complete
token-only path succeeds.  The evidence contains geometry and health facts,
never patch, projected-token, pose, or camera values.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import scripts.run_recal3r_smoke as smoke

from stateguard3r.patch_embed_global_pooled_pose_v12 import (
    PATCH_FEATURE_WIDTH,
    PATCH_GLOBAL_SHAPE,
    PINNED_RECAL3R_SOURCE_SHA256,
    PROJECTED_TOKEN_SHAPE,
    capture_patch_embed_global_pose_token,
    decode_patch_embed_global_pose_token,
)


RUN_ID = "recovery-patch-embed-global-pooled-pose-v12-dynamic-frame0-cpu-probe-0001"
DEVELOPMENT_DYNAMIC_MANIFEST = (
    ROOT / "outputs" / "formal-v1-inputs-0001" / "development" /
    "development-dynamic" / "input-manifest.json"
)
PINNED_RECAL3R_COMMIT = "466c7cdf3acd2f589f1d82e5f6391966f19db9ff"


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
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _require_cuda_disabled_environment() -> None:
    """Require an explicit CUDA-disabled process before importing torch."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("v12 CPU probe requires CUDA_VISIBLE_DEVICES='' before torch import")


def _exact_dynamic_frame_zero(manifest: Path) -> tuple[Path, Mapping[str, Any]]:
    resolved = manifest.resolve(strict=True)
    if resolved != DEVELOPMENT_DYNAMIC_MANIFEST.resolve(strict=True):
        raise RuntimeError("v12 CPU probe accepts only the disclosed dynamic manifest")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    frames = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("dynamic manifest has no frame list")
    frame = frames[0]
    if (
        not isinstance(frame, dict)
        or frame.get("frame_index") != 0
        or not isinstance(frame.get("path"), str)
    ):
        raise RuntimeError("dynamic manifest frame zero is malformed")
    image_path = (resolved.parent / frame["path"]).resolve(strict=True)
    return image_path, frame


def _pinned_pose_head_interface(model: Any, dpt_head_module: Any) -> tuple[Any, Any]:
    expected = getattr(dpt_head_module, "DPTPts3dPose", None)
    downstream = getattr(model, "downstream_head", None)
    if expected is None or type(downstream) is not expected:
        raise RuntimeError("v12 model downstream head is not pinned DPTPts3dPose")
    pose_head = getattr(downstream, "pose_head", None)
    pose_mode = getattr(downstream, "pose_mode", None)
    if not callable(pose_head) or not isinstance(pose_mode, tuple) or len(pose_mode) != 3:
        raise RuntimeError("v12 pinned DPT pose-head interface is unavailable")
    return pose_head, pose_mode


def _pinned_patch_embed_interface(model: Any, patch_embed_module: Any) -> Any:
    expected = getattr(patch_embed_module, "PatchEmbedDust3R", None)
    patch_embed = getattr(model, "patch_embed", None)
    if expected is None or type(patch_embed) is not expected:
        raise RuntimeError(
            "v12 model patch embedder is not the pinned PatchEmbedDust3R; "
            "ManyAR_PatchEmbed is forbidden"
        )
    return patch_embed


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create one frozen, non-overwritable direct logs child."""
    target = path.resolve(strict=False)
    logs_root = (ROOT / "logs").resolve(strict=True)
    if logs_root not in target.parents or target.exists() or target.parent != logs_root:
        raise RuntimeError("v12 CPU probe evidence must be a new direct logs child")
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, 0o444)
        os.link(temporary_name, target)
    except FileExistsError as error:
        raise RuntimeError("v12 CPU probe evidence target already exists") from error
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _prohibited(name: str) -> Any:
    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(f"v12 CPU probe called prohibited path: {name}")

    return blocked


def _block_module_forward(module: Any, name: str) -> None:
    if module is not None and callable(getattr(module, "forward", None)):
        module.forward = _prohibited(name)


def _guard_prohibited_model_paths(model: Any) -> None:
    """Make every unavailable inference branch fail at its first invocation."""
    model._encode_image = _prohibited("_encode_image")
    model._recurrent_rollout = _prohibited("_recurrent_rollout")

    # PatchEmbedDust3R has no encoder/RoPE dependency.  Guard the full image
    # and ray encoder blocks separately so a future call cannot silently enter
    # their attention/RoPE path during this deliberately pre-RoPE probe.
    for group_name in ("enc_blocks", "enc_blocks_ray_map"):
        for index, block in enumerate(getattr(model, group_name, ())):
            _block_module_forward(block, f"{group_name}[{index}]")
    _block_module_forward(getattr(model, "rope", None), "RoPE")

    retriever = getattr(model, "pose_retriever", None)
    if retriever is not None:
        retriever.inquire = _prohibited("pose_retriever.inquire")
        retriever.update_mem = _prohibited("pose_retriever.update_mem")
        _block_module_forward(retriever, "pose_retriever.forward")
        for group_name in ("read_blocks", "write_blocks"):
            for index, block in enumerate(getattr(retriever, group_name, ())):
                _block_module_forward(block, f"pose_retriever.{group_name}[{index}]")


def _require_cpu_tensor(value: Any, *, name: str) -> None:
    device = getattr(value, "device", None)
    if getattr(device, "type", None) != "cpu":
        raise RuntimeError(f"v12 CPU probe {name} is not a CPU tensor")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _require_cuda_disabled_environment()
    baseline_root = args.baseline_root.resolve(strict=True)
    baseline_src = (baseline_root / "src").resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    if _sha256(checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("v12 CPU probe checkpoint SHA-256 differs")
    if smoke._git_output(baseline_root, "rev-parse", "HEAD") != PINNED_RECAL3R_COMMIT:
        raise RuntimeError("v12 CPU probe ReCal3R commit differs from the pin")
    if smoke._git_output(baseline_root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("v12 CPU probe requires a clean tracked ReCal3R worktree")
    patch_embed_path = baseline_src / "dust3r" / "patch_embed.py"
    if _sha256(patch_embed_path) != PINNED_RECAL3R_SOURCE_SHA256["patch_embed_sha256"]:
        raise RuntimeError("v12 CPU probe pinned upstream patch embedder bytes differ")
    image_path, manifest_frame = _exact_dynamic_frame_zero(args.input_manifest)
    if str(baseline_root) not in sys.path:
        sys.path.insert(0, str(baseline_root))
    if str(baseline_src) not in sys.path:
        sys.path.insert(0, str(baseline_src))

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("v12 CPU probe refuses an already initialized CUDA runtime")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    import add_ckpt_path

    add_ckpt_path.add_path_to_dust3r(str(baseline_src / checkpoint.name))
    import dust3r.heads.dpt_head as dpt_head
    import dust3r.heads.postprocess as postprocess
    import dust3r.model as dust3r_model
    import dust3r.patch_embed as patch_embed_module
    from dust3r.utils.camera import pose_encoding_to_camera
    from dust3r.utils.image import load_images_for_eval

    # Supplying a list with exactly one canonical path is the only image input:
    # it prevents a future frame, GT listing, detector input, or history input.
    loaded = load_images_for_eval([str(image_path)], size=512, crop=True, verbose=True)
    if len(loaded) != 1:
        raise RuntimeError("v12 CPU probe loader did not return exactly frame zero")
    image = loaded[0]
    image_tensor = image.get("img")
    true_shape = image.get("true_shape")
    if (
        tuple(getattr(image_tensor, "shape", ()))[:2] != (1, 3)
        or tuple(getattr(true_shape, "shape", ())) != (1, 2)
    ):
        raise RuntimeError("v12 CPU probe loader returned malformed image metadata")
    _require_cpu_tensor(image_tensor, name="loaded image")
    selected_shapes = torch.from_numpy(true_shape)
    _require_cpu_tensor(selected_shapes, name="loaded true_shape")

    model, checkpoint_audit = smoke._load_model_with_state_dict_audit(
        dust3r_model.ARCroco3DStereo, checkpoint, torch,
    )
    if (
        checkpoint_audit["strict"] is not False
        or checkpoint_audit["missing_keys"]
        or checkpoint_audit["unexpected_keys"]
    ):
        raise RuntimeError("v12 CPU probe checkpoint compatibility audit failed")
    model = model.to("cpu")
    model.eval()
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("v12 CPU probe checkpoint model is not entirely CPU-resident")
    patch_embed = _pinned_patch_embed_interface(model, patch_embed_module)
    pose_head, pose_mode = _pinned_pose_head_interface(model, dpt_head)
    _guard_prohibited_model_paths(model)

    started_at = datetime.now().astimezone().isoformat()
    with torch.no_grad():
        patch_tokens, patch_positions = model.patch_embed(
            image_tensor, true_shape=selected_shapes,
        )
        _require_cpu_tensor(patch_tokens, name="patch feature")
        _require_cpu_tensor(patch_positions, name="patch positions")
        capture = capture_patch_embed_global_pose_token(
            patch_tokens, projection=model.decoder_embed, torch=torch,
        )
        decoded = decode_patch_embed_global_pose_token(
            capture.token,
            torch=torch,
            pose_head=pose_head,
            postprocess_pose=postprocess.postprocess_pose,
            pose_mode=pose_mode,
            pose_encoding_to_camera=pose_encoding_to_camera,
        )
    if (
        tuple(patch_tokens.shape)[0] != 1
        or tuple(patch_tokens.shape)[2] != PATCH_FEATURE_WIDTH
        or tuple(capture.patch_global_shape) != PATCH_GLOBAL_SHAPE
        or tuple(capture.token.shape) != PROJECTED_TOKEN_SHAPE
        or tuple(decoded.pose.shape) != (1, 7)
        or tuple(decoded.camera.shape) != (1, 4, 4)
    ):
        raise RuntimeError("v12 CPU probe patch-global shape contract failed")
    if torch.cuda.is_initialized():
        raise RuntimeError("v12 CPU probe unexpectedly initialized CUDA")

    # All data-dependent computation and every fail-closed guard above have
    # completed before constructing or writing the immutable evidence payload.
    payload = {
        "schema_version": "stateguard3r.v12-cpu-patch-embed-probe.v1",
        "run_id": RUN_ID,
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
        "checkpoint_sha256": args.checkpoint_sha256,
        "checkpoint_load_audit": checkpoint_audit,
        "checkpoint_compatible_without_missing_or_unexpected_keys": True,
        "patch_embed_source_sha256": _sha256(patch_embed_path),
        "patch_embed_class": type(patch_embed).__name__,
        "input_manifest": str(args.input_manifest.resolve(strict=True)),
        "input_manifest_sha256": _sha256(args.input_manifest.resolve(strict=True)),
        "frame_id": 0,
        "frame_path": str(image_path),
        "frame_sha256": _sha256(image_path),
        "manifest_frame_rgb_sha256": manifest_frame.get("metadata", {}).get("rgb_sha256"),
        "loaded_image_shape": list(image_tensor.shape),
        "true_shape": [int(value) for value in true_shape.reshape(-1)],
        "patch_feature_shape": list(patch_tokens.shape),
        "patch_spatial_token_count": int(patch_tokens.shape[1]),
        "patch_position_shape": list(patch_positions.shape),
        "patch_feature_dtype": str(patch_tokens.dtype),
        "patch_feature_device": str(patch_tokens.device),
        "patch_position_dtype": str(patch_positions.dtype),
        "patch_position_device": str(patch_positions.device),
        "patch_global_shape": list(capture.patch_global_shape),
        "projected_token_shape": list(capture.token.shape),
        "token_dtype": str(capture.token.dtype),
        "token_device": str(capture.token.device),
        "pose_shape": list(decoded.pose.shape),
        "camera_shape": list(decoded.camera.shape),
        "rotation_determinant": decoded.rotation_determinant,
        "orthonormality_max_abs_error": decoded.orthonormality_max_abs_error,
        "homogeneous_max_abs_error": decoded.homogeneous_max_abs_error,
        "call_chain": [
            "load_images_for_eval",
            "model.patch_embed",
            "torch.mean",
            "model.decoder_embed",
            "pose_head",
            "postprocess_pose",
            "pose_encoding_to_camera",
        ],
        "prohibited_paths": [
            "_encode_image",
            "encoder_blocks",
            "RoPE",
            "pose_retriever.inquire",
            "pose_retriever.update_mem",
            "pose_retriever.memory",
            "_recurrent_rollout",
            "detector",
            "ground_truth",
            "future_frame",
        ],
        "latent_values_serialized": False,
        "pose_or_camera_values_serialized": False,
    }
    _write_immutable_json(args.evidence_json, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
