#!/usr/bin/env python3
"""CPU-only, one-frame Gate-A probe for the v11 encoder-global boundary.

This probe deliberately performs only the current-frame path
``load_images_for_eval -> _encode_image -> _get_img_level_feat ->
decoder_embed -> pose_head/postprocess/camera``.  It neither constructs nor
uses recurrent state, pose-retriever memory, detector inputs, GT, or another
frame.  It writes immutable shape/health evidence, never latent values.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_recal3r_smoke as smoke

from stateguard3r.encoder_global_pooled_pose_v11 import (
    ENCODER_GLOBAL_FEATURE_SHAPE,
    PROJECTED_TOKEN_SHAPE,
    capture_encoder_global_pose_token,
    decode_encoder_global_pose_token,
)


RUN_ID = "recovery-encoder-global-pooled-pose-v11-dynamic-frame0-cpu-probe-0001"
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


def _exact_dynamic_frame_zero(manifest: Path) -> tuple[Path, Mapping[str, Any]]:
    resolved = manifest.resolve(strict=True)
    if resolved != DEVELOPMENT_DYNAMIC_MANIFEST.resolve(strict=True):
        raise RuntimeError("v11 CPU probe accepts only the disclosed dynamic manifest")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    frames = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("dynamic manifest has no frame list")
    frame = frames[0]
    if not isinstance(frame, dict) or frame.get("frame_index") != 0 or not isinstance(frame.get("path"), str):
        raise RuntimeError("dynamic manifest frame zero is malformed")
    image_path = (resolved.parent / frame["path"]).resolve(strict=True)
    return image_path, frame


def _pinned_pose_head_interface(model: Any, dpt_head_module: Any) -> tuple[Any, Any]:
    expected, downstream = getattr(dpt_head_module, "DPTPts3dPose", None), getattr(model, "downstream_head", None)
    if expected is None or type(downstream) is not expected:
        raise RuntimeError("v11 model downstream head is not pinned DPTPts3dPose")
    pose_head, pose_mode = getattr(downstream, "pose_head", None), getattr(downstream, "pose_mode", None)
    if not callable(pose_head) or not isinstance(pose_mode, tuple) or len(pose_mode) != 3:
        raise RuntimeError("v11 pinned DPT pose-head interface is unavailable")
    return pose_head, pose_mode


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.resolve(strict=False)
    logs_root = (ROOT / "logs").resolve(strict=True)
    if logs_root not in target.parents or target.exists() or target.parent != logs_root:
        raise RuntimeError("v11 CPU probe evidence must be a new direct logs child")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, 0o444)
        os.link(temporary_name, target)
    except FileExistsError as error:
        raise RuntimeError("v11 CPU probe evidence target already exists") from error
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    baseline_root = args.baseline_root.resolve(strict=True)
    baseline_src = (baseline_root / "src").resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    if _sha256(checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("v11 CPU probe checkpoint SHA-256 differs")
    if smoke._git_output(baseline_root, "rev-parse", "HEAD") != PINNED_RECAL3R_COMMIT:
        raise RuntimeError("v11 CPU probe ReCal3R commit differs from the pin")
    if smoke._git_output(baseline_root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("v11 CPU probe requires a clean tracked ReCal3R worktree")
    image_path, manifest_frame = _exact_dynamic_frame_zero(args.input_manifest)
    if str(baseline_root) not in sys.path:
        sys.path.insert(0, str(baseline_root))
    if str(baseline_src) not in sys.path:
        sys.path.insert(0, str(baseline_src))

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("v11 CPU probe refuses an already initialized CUDA runtime")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    import add_ckpt_path

    add_ckpt_path.add_path_to_dust3r(str(baseline_src / checkpoint.name))
    import dust3r.heads.dpt_head as dpt_head
    import dust3r.heads.postprocess as postprocess
    import dust3r.model as dust3r_model
    from dust3r.utils.camera import pose_encoding_to_camera
    from dust3r.utils.image import load_images_for_eval

    loaded = load_images_for_eval([str(image_path)], size=512, crop=True, verbose=True)
    if len(loaded) != 1:
        raise RuntimeError("v11 CPU probe loader did not return exactly frame zero")
    image = loaded[0]
    image_tensor = image.get("img")
    true_shape = image.get("true_shape")
    if tuple(getattr(image_tensor, "shape", ()))[:2] != (1, 3) or tuple(getattr(true_shape, "shape", ())) != (1, 2):
        raise RuntimeError("v11 CPU probe loader returned malformed image metadata")
    model, checkpoint_audit = smoke._load_model_with_state_dict_audit(dust3r_model.ARCroco3DStereo, checkpoint, torch)
    if checkpoint_audit["strict"] is not False or checkpoint_audit["missing_keys"] or checkpoint_audit["unexpected_keys"]:
        raise RuntimeError("v11 CPU probe checkpoint audit failed")
    model = model.to("cpu")
    model.eval()
    pose_head, pose_mode = _pinned_pose_head_interface(model, dpt_head)

    # These hard failures make the executed call graph itself reject any
    # accidental recurrence, retrieval, or memory update during this probe.
    def prohibited(name: str) -> Any:
        def blocked(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError(f"v11 CPU probe called prohibited path: {name}")
        return blocked

    model._recurrent_rollout = prohibited("_recurrent_rollout")
    model.pose_retriever.inquire = prohibited("pose_retriever.inquire")
    model.pose_retriever.update_mem = prohibited("pose_retriever.update_mem")
    started_at = datetime.now().astimezone().isoformat()
    with torch.no_grad():
        encoder_outputs, _positions, _ = model._encode_image(image_tensor, torch.from_numpy(true_shape))
        encoder_feature = encoder_outputs[-1]
        global_feature = model._get_img_level_feat(encoder_feature)
        capture = capture_encoder_global_pose_token(
            global_feature,
            encoder_feature_shape=tuple(encoder_feature.shape),
            projection=model.decoder_embed,
            torch=torch,
        )
        decoded = decode_encoder_global_pose_token(
            capture.token,
            torch=torch,
            pose_head=pose_head,
            postprocess_pose=postprocess.postprocess_pose,
            pose_mode=pose_mode,
            pose_encoding_to_camera=pose_encoding_to_camera,
        )
    if tuple(encoder_feature.shape)[0] != 1 or tuple(encoder_feature.shape)[2] != 1024 or tuple(global_feature.shape) != ENCODER_GLOBAL_FEATURE_SHAPE or tuple(capture.token.shape) != PROJECTED_TOKEN_SHAPE:
        raise RuntimeError("v11 CPU probe encoder-global shape contract failed")
    if torch.cuda.is_initialized():
        raise RuntimeError("v11 CPU probe unexpectedly initialized CUDA")
    payload = {
        "schema_version": "stateguard3r.v11-cpu-encoder-probe.v1",
        "run_id": RUN_ID,
        "status": "passed",
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_initialized": False,
        "baseline_root": str(baseline_root),
        "baseline_commit": PINNED_RECAL3R_COMMIT,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "checkpoint_load_audit": checkpoint_audit,
        "input_manifest": str(args.input_manifest.resolve(strict=True)),
        "input_manifest_sha256": _sha256(args.input_manifest.resolve(strict=True)),
        "frame_id": 0,
        "frame_path": str(image_path),
        "frame_sha256": _sha256(image_path),
        "manifest_frame_rgb_sha256": manifest_frame.get("metadata", {}).get("rgb_sha256"),
        "loaded_image_shape": list(image_tensor.shape),
        "true_shape": [int(value) for value in true_shape.reshape(-1)],
        "encoder_feature_shape": list(encoder_feature.shape),
        "encoder_spatial_token_count": int(encoder_feature.shape[1]),
        "encoder_global_shape": list(global_feature.shape),
        "projected_token_shape": list(capture.token.shape),
        "encoder_dtype": str(encoder_feature.dtype),
        "encoder_device": str(encoder_feature.device),
        "token_dtype": str(capture.token.dtype),
        "token_device": str(capture.token.device),
        "pose_shape": list(decoded.pose.shape),
        "camera_shape": list(decoded.camera.shape),
        "rotation_determinant": decoded.rotation_determinant,
        "orthonormality_max_abs_error": decoded.orthonormality_max_abs_error,
        "homogeneous_max_abs_error": decoded.homogeneous_max_abs_error,
        "call_chain": ["load_images_for_eval", "_encode_image", "_get_img_level_feat", "decoder_embed", "pose_head", "postprocess_pose", "pose_encoding_to_camera"],
        "prohibited_paths": ["_recurrent_rollout", "pose_retriever.inquire", "pose_retriever.update_mem", "detector", "ground_truth", "future_frame"],
        "latent_values_serialized": False,
    }
    _write_immutable_json(args.evidence_json, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
