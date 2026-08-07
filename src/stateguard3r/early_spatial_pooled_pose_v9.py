"""Pinned, current-frame spatial-pooled pose decoding for recovery v9.

This module deliberately accepts only the already projected ``(1, 1, 768)``
latent.  It does not know a prediction, a raw camera pose, RGB, point maps, or
detector state; this makes the alarm export boundary auditable and fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping


POOLED_TOKEN_SHAPE = (1, 1, 768)
PREPROJECTION_SPATIAL_SHAPE = (1, 576, 1024)
ROTATION_ATOL = 1e-8
PINNED_RECAL3R_SOURCE_SHA256 = {
    "model_sha256": "32785a6f29fded66aa142207b29a33d7522f0c39aa068fe2175a956ec8dbc3c1",
    "dpt_head_sha256": "c4f4b08c844b5fea47b67cedbcf4888719e8664f4b3b7e7ab6218e33cf63a66e",
    "postprocess_sha256": "ee93ae7fa16897ed9163a21efe225f46193054b19bd7d316402545ed766133d7",
}


class EarlySpatialPooledPoseError(RuntimeError):
    """The spatial pooled token cannot safely provide an alarm export."""


@dataclass(frozen=True)
class EarlySpatialPooledPoseResult:
    pose: Any
    camera: Any
    token_shape: tuple[int, int, int]
    rotation_determinant: float
    orthonormality_max_abs_error: float
    homogeneous_max_abs_error: float


def decode_early_spatial_pooled_pose_token(
    token: Any, *, torch: Any, pose_head: Callable[[Any], Any],
    postprocess_pose: Callable[[Any, Any], Any], pose_mode: Any,
    pose_encoding_to_camera: Callable[[Any], Any],
) -> EarlySpatialPooledPoseResult:
    """Decode one projected spatial mean through the pinned official head only."""
    if not bool(torch.is_tensor(token)) or tuple(getattr(token, "shape", ())) != POOLED_TOKEN_SHAPE:
        raise EarlySpatialPooledPoseError("early spatial pooled pose must have shape (1,1,768)")
    if not bool(torch.is_floating_point(token)) or not bool(torch.isfinite(token).all()):
        raise EarlySpatialPooledPoseError("early spatial pooled pose must be finite floating point")
    parameters = getattr(pose_head, "parameters", None)
    if callable(parameters):
        try:
            head_device = next(parameters()).device
        except StopIteration:
            head_device = None
        if head_device is not None and token.device != head_device:
            raise EarlySpatialPooledPoseError("early spatial pooled pose token is not on the pinned head device")
    try:
        raw = pose_head(token[:, 0])
        if tuple(getattr(raw, "shape", ())) != (1, 7):
            raise EarlySpatialPooledPoseError("pinned pose head has malformed output shape")
        if not bool(torch.is_floating_point(raw)) or not bool(torch.isfinite(raw).all()):
            raise EarlySpatialPooledPoseError("pinned pose head output is nonfinite")
        # Preserve native model/head precision.  Only the seven head values are
        # promoted on the same device for the official SO(3) postprocess check.
        pose = postprocess_pose(raw.to(dtype=torch.float64), pose_mode)
        camera = pose_encoding_to_camera(pose)
    except EarlySpatialPooledPoseError:
        raise
    except Exception as error:
        raise EarlySpatialPooledPoseError(f"pinned early spatial pooled pose decode failed: {error}") from error
    if tuple(getattr(pose, "shape", ())) != (1, 7) or tuple(getattr(camera, "shape", ())) != (1, 4, 4):
        raise EarlySpatialPooledPoseError("pinned early spatial pooled pose decode has malformed output shape")
    if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(pose).all()) or not bool(torch.isfinite(camera).all()):
        raise EarlySpatialPooledPoseError("pinned early spatial pooled pose decode is nonfinite")
    rotation = camera[0, :3, :3]
    determinant = float(torch.linalg.det(rotation).item())
    orthonormality = float(torch.max(torch.abs(rotation.transpose(0, 1) @ rotation - torch.eye(3, dtype=rotation.dtype, device=rotation.device))).item())
    homogeneous = float(torch.max(torch.abs(camera[0, 3] - torch.tensor((0.0, 0.0, 0.0, 1.0), dtype=camera.dtype, device=camera.device))).item())
    if not math.isfinite(determinant) or abs(determinant - 1.0) > ROTATION_ATOL or not math.isfinite(orthonormality) or orthonormality > ROTATION_ATOL or not math.isfinite(homogeneous) or homogeneous > ROTATION_ATOL:
        raise EarlySpatialPooledPoseError("decoded early spatial pooled pose is not proper homogeneous SO(3)")
    return EarlySpatialPooledPoseResult(pose, camera, POOLED_TOKEN_SHAPE, determinant, orthonormality, homogeneous)


def audit_pinned_early_spatial_pooled_pose_sources(
    model_path: Path, dpt_head_path: Path, postprocess_path: Path, runner_path: Path,
) -> Mapping[str, Any]:
    """Bind v9 to the frozen pre-projection spatial sequence and projection."""
    model, dpt, postprocess, runner = (path.read_text(encoding="utf-8") for path in (model_path, dpt_head_path, postprocess_path, runner_path))
    decoder_start = model.find("    def _decoder(")
    decoder_end = model.find("\n    def ", decoder_start + 1) if decoder_start >= 0 else -1
    decoder = model[decoder_start: decoder_end if decoder_end >= 0 else None]
    lighter_start = model.find("    def forward_recurrent_lighter(")
    lighter_end = model.find("\n    def ", lighter_start + 1) if lighter_start >= 0 else -1
    lighter = model[lighter_start: lighter_end if lighter_end >= 0 else None]
    required_decoder = (
        "final_output = [(f_state, f_img)]  # before projection",
        "f_img = self.decoder_embed(f_img)",
        "f_img = torch.cat([f_pose, f_img], dim=1)",
        "del final_output[1]  # duplicate with final_output[0]",
        "return zip(*final_output), zip(*attention_maps)",
    )
    required_lighter = (
        "new_state_feat, dec",
        "self._recurrent_rollout(",
        "out_pose_feat_i = dec[-1][:, 0:1]",
        "self.pose_retriever.update_mem(",
        "dec[0].float()",
    )
    required_dpt = (
        "class DPTPts3dPose",
        "self.pose_head = PoseDecoder(hidden_size=in_dim)",
        "pose_token = x[-1][:, 0].clone()",
        "pose = self.pose_head(pose_token)",
        "pose = postprocess_pose(pose, self.pose_mode)",
        'final_output["camera_pose"] = pose',
    )
    if any(fragment not in decoder for fragment in required_decoder) or any(fragment not in lighter for fragment in required_lighter) or any(fragment not in dpt for fragment in required_dpt) or "def postprocess_pose(out, mode, inverse=False):" not in postprocess:
        raise EarlySpatialPooledPoseError("pinned ReCal3R spatial-pooling topology changed")
    try:
        projection_at = decoder.index(required_decoder[0])
        pose_concat_at = decoder.index(required_decoder[1], projection_at)
        rollout_at = lighter.index(required_lighter[1])
        update_at = lighter.index(required_lighter[3], rollout_at)
        runner_body = runner[runner.index("\ndef run_early_spatial_pooled_pose_recurrent_lighter(") + 1:]
        runner_rollout_at = runner_body.index("model._recurrent_rollout(")
        runner_capture_at = runner_body.index("early_spatial_pooled_pose_token = model.decoder_embed(dec[0].mean(dim=1, keepdim=True)).detach().clone()")
        runner_update_at = runner_body.index("model.pose_retriever.update_mem(", runner_rollout_at)
    except ValueError as error:
        raise EarlySpatialPooledPoseError("pinned spatial pooling/source update order changed") from error
    if not (projection_at < pose_concat_at and rollout_at < update_at and runner_rollout_at < runner_capture_at < runner_update_at):
        raise EarlySpatialPooledPoseError("pinned spatial pooling/source update order changed")
    source_hashes = {
        "model_sha256": hashlib.sha256(model.encode("utf-8")).hexdigest(),
        "dpt_head_sha256": hashlib.sha256(dpt.encode("utf-8")).hexdigest(),
        "postprocess_sha256": hashlib.sha256(postprocess.encode("utf-8")).hexdigest(),
    }
    if source_hashes != PINNED_RECAL3R_SOURCE_SHA256:
        raise EarlySpatialPooledPoseError("pinned ReCal3R spatial-pooling source SHA-256 differs")
    return {
        "model_path": str(model_path.resolve(strict=True)),
        "model_sha256": source_hashes["model_sha256"],
        "dpt_head_path": str(dpt_head_path.resolve(strict=True)),
        "dpt_head_sha256": source_hashes["dpt_head_sha256"],
        "postprocess_path": str(postprocess_path.resolve(strict=True)),
        "postprocess_sha256": source_hashes["postprocess_sha256"],
        "runner_path": str(runner_path.resolve(strict=True)),
        "runner_sha256": hashlib.sha256(runner.encode("utf-8")).hexdigest(),
        "decoder_projection_interface": "torch.nn.Linear(1024,768): model.decoder_embed",
        "decoder_projection_sha256": source_hashes["model_sha256"],
        "early_spatial_source": "decoder_layer_0_preprojection_spatial_mean_after_frozen_decoder_embed_before_update_mem",
        "early_decoder_layer_index": 0,
        "spatial_token_count": 576,
        "preprojection_shape": list(PREPROJECTION_SPATIAL_SHAPE),
        "pooled_shape": list(POOLED_TOKEN_SHAPE),
        "aggregation": "mean",
        "capture_relation": "after_rollout_before_update_mem",
        "raw_camera_pose_numeric_input_to_v9_decoder": False,
    }


__all__ = [
    "EarlySpatialPooledPoseError", "EarlySpatialPooledPoseResult", "PINNED_RECAL3R_SOURCE_SHA256",
    "POOLED_TOKEN_SHAPE", "PREPROJECTION_SPATIAL_SHAPE", "ROTATION_ATOL",
    "audit_pinned_early_spatial_pooled_pose_sources", "decode_early_spatial_pooled_pose_token",
]
