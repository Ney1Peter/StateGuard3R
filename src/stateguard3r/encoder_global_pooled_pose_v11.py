"""Pinned encoder-global pose decoding for the distinct v11 recovery route.

The decoder accepts only an already captured current-frame encoder-global
projection.  It deliberately has no prediction, raw camera pose, detector,
RGB, point-map, history, ground-truth, or CPU-transfer input.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ENCODER_GLOBAL_FEATURE_SHAPE = (1, 1, 1024)
PROJECTED_TOKEN_SHAPE = (1, 1, 768)
ROTATION_ATOL = 1e-8
PINNED_RECAL3R_SOURCE_SHA256 = {
    "model_sha256": "32785a6f29fded66aa142207b29a33d7522f0c39aa068fe2175a956ec8dbc3c1",
    "dpt_head_sha256": "c4f4b08c844b5fea47b67cedbcf4888719e8664f4b3b7e7ab6218e33cf63a66e",
    "postprocess_sha256": "ee93ae7fa16897ed9163a21efe225f46193054b19bd7d316402545ed766133d7",
    "patch_embed_sha256": "c0dae36e876d2124f0e31403a3a306db7e6d5bd69d8e9c6f16bbfb81c82eef3d",
    "image_loader_sha256": "a2738085cdf0f713289a3a42dbd49a1f39325eb3ad590af65970fe4609209d96",
}


class EncoderGlobalPooledPoseError(RuntimeError):
    """The current encoder-global latent cannot safely provide an alarm pose."""


@dataclass(frozen=True)
class EncoderGlobalPoseCapture:
    """One isolated v11 projected latent plus non-sensitive shape provenance."""

    token: Any
    encoder_feature_shape: tuple[int, int, int]
    encoder_global_shape: tuple[int, int, int]
    encoder_feature_dtype: str
    encoder_feature_device: str


@dataclass(frozen=True)
class EncoderGlobalPooledPoseResult:
    pose: Any
    camera: Any
    token_shape: tuple[int, int, int]
    rotation_determinant: float
    orthonormality_max_abs_error: float
    homogeneous_max_abs_error: float


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in getattr(value, "shape", ()))


def capture_encoder_global_pose_token(
    encoder_global_feature: Any, *, encoder_feature_shape: Sequence[int], projection: Any, torch: Any,
) -> EncoderGlobalPoseCapture:
    """Project one native global feature and detach it before retrieval/rollout.

    ``encoder_feature_shape`` contains shape metadata only.  Its spatial length
    is intentionally recorded rather than constrained, so a valid aspect ratio
    cannot become a concealed retry of v9's fixed decoder-grid guard.
    """
    source_shape = tuple(int(item) for item in encoder_feature_shape)
    if len(source_shape) != 3 or source_shape[0] != 1 or source_shape[1] <= 0 or source_shape[2] != 1024:
        raise EncoderGlobalPooledPoseError("current encoder feature shape is unavailable")
    if not bool(torch.is_tensor(encoder_global_feature)) or _shape(encoder_global_feature) != ENCODER_GLOBAL_FEATURE_SHAPE:
        raise EncoderGlobalPooledPoseError("encoder global feature must have shape (1,1,1024)")
    if not bool(torch.is_floating_point(encoder_global_feature)) or not bool(torch.isfinite(encoder_global_feature).all()):
        raise EncoderGlobalPooledPoseError("encoder global feature must be finite floating point")
    if type(projection) is not torch.nn.Linear or projection.in_features != 1024 or projection.out_features != 768:
        raise EncoderGlobalPooledPoseError("pinned frozen decoder projection changed")
    if projection.weight.device != encoder_global_feature.device or projection.weight.dtype != encoder_global_feature.dtype:
        raise EncoderGlobalPooledPoseError("pinned frozen decoder projection device or dtype changed")
    try:
        token = projection(encoder_global_feature).detach().clone()
    except Exception as error:
        raise EncoderGlobalPooledPoseError(f"encoder global projection failed: {error}") from error
    if not bool(torch.is_tensor(token)) or _shape(token) != PROJECTED_TOKEN_SHAPE:
        raise EncoderGlobalPooledPoseError("encoder global projected token has malformed shape")
    if not bool(torch.is_floating_point(token)) or not bool(torch.isfinite(token).all()):
        raise EncoderGlobalPooledPoseError("encoder global projected token is nonfinite")
    return EncoderGlobalPoseCapture(
        token=token,
        encoder_feature_shape=source_shape,
        encoder_global_shape=ENCODER_GLOBAL_FEATURE_SHAPE,
        encoder_feature_dtype=str(encoder_global_feature.dtype),
        encoder_feature_device=str(encoder_global_feature.device),
    )


def decode_encoder_global_pose_token(
    token: Any, *, torch: Any, pose_head: Callable[[Any], Any],
    postprocess_pose: Callable[[Any, Any], Any], pose_mode: Any,
    pose_encoding_to_camera: Callable[[Any], Any],
) -> EncoderGlobalPooledPoseResult:
    """Decode an already-projected v11 token with the pinned official head."""
    if not bool(torch.is_tensor(token)) or _shape(token) != PROJECTED_TOKEN_SHAPE:
        raise EncoderGlobalPooledPoseError("encoder global pose token must have shape (1,1,768)")
    if not bool(torch.is_floating_point(token)) or not bool(torch.isfinite(token).all()):
        raise EncoderGlobalPooledPoseError("encoder global pose token must be finite floating point")
    parameters = getattr(pose_head, "parameters", None)
    if callable(parameters):
        try:
            head_device = next(parameters()).device
        except StopIteration:
            head_device = None
        if head_device is not None and token.device != head_device:
            raise EncoderGlobalPooledPoseError("encoder global pose token is not on the pinned head device")
    try:
        raw = pose_head(token[:, 0])
        if _shape(raw) != (1, 7):
            raise EncoderGlobalPooledPoseError("pinned pose head has malformed output shape")
        if not bool(torch.is_floating_point(raw)) or not bool(torch.isfinite(raw).all()):
            raise EncoderGlobalPooledPoseError("pinned pose head output is nonfinite")
        pose = postprocess_pose(raw.to(dtype=torch.float64), pose_mode)
        camera = pose_encoding_to_camera(pose)
    except EncoderGlobalPooledPoseError:
        raise
    except Exception as error:
        raise EncoderGlobalPooledPoseError(f"pinned encoder global pose decode failed: {error}") from error
    if _shape(pose) != (1, 7) or _shape(camera) != (1, 4, 4):
        raise EncoderGlobalPooledPoseError("pinned encoder global pose decode has malformed output shape")
    if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(pose).all()) or not bool(torch.isfinite(camera).all()):
        raise EncoderGlobalPooledPoseError("pinned encoder global pose decode is nonfinite")
    rotation = camera[0, :3, :3]
    determinant = float(torch.linalg.det(rotation).item())
    orthonormality = float(torch.max(torch.abs(rotation.transpose(0, 1) @ rotation - torch.eye(3, dtype=rotation.dtype, device=rotation.device))).item())
    homogeneous = float(torch.max(torch.abs(camera[0, 3] - torch.tensor((0.0, 0.0, 0.0, 1.0), dtype=camera.dtype, device=camera.device))).item())
    if not math.isfinite(determinant) or abs(determinant - 1.0) > ROTATION_ATOL or not math.isfinite(orthonormality) or orthonormality > ROTATION_ATOL or not math.isfinite(homogeneous) or homogeneous > ROTATION_ATOL:
        raise EncoderGlobalPooledPoseError("decoded encoder global pose is not proper homogeneous SO(3)")
    return EncoderGlobalPooledPoseResult(pose, camera, PROJECTED_TOKEN_SHAPE, determinant, orthonormality, homogeneous)


def audit_pinned_encoder_global_pose_sources(
    model_path: Path, dpt_head_path: Path, postprocess_path: Path, patch_embed_path: Path,
    image_loader_path: Path, runner_path: Path,
) -> Mapping[str, Any]:
    """Bind v11 to native encoder-global capture before retrieval and rollout."""
    model, dpt, postprocess, patch_embed, image_loader, runner = (
        path.read_text(encoding="utf-8")
        for path in (model_path, dpt_head_path, postprocess_path, patch_embed_path, image_loader_path, runner_path)
    )
    global_start = model.find("    def _get_img_level_feat(")
    global_end = model.find("\n    def ", global_start + 1) if global_start >= 0 else -1
    global_method = model[global_start:global_end if global_end >= 0 else None]
    lighter_start = model.find("    def forward_recurrent_lighter(")
    lighter_end = model.find("\n    def ", lighter_start + 1) if lighter_start >= 0 else -1
    lighter = model[lighter_start:lighter_end if lighter_end >= 0 else None]
    required_global = ("def _get_img_level_feat(self, feat):", "torch.mean(feat, dim=1, keepdim=True)")
    required_lighter = ("global_img_feat_i = self._get_img_level_feat(feat_i)", "self.pose_retriever.inquire(global_img_feat_i, mem)", "self._recurrent_rollout(", "self.pose_retriever.update_mem(")
    required_dpt = ("class DPTPts3dPose", "self.pose_head = PoseDecoder(hidden_size=in_dim)", "pose_token = x[-1][:, 0].clone()", "pose = self.pose_head(pose_token)", "pose = postprocess_pose(pose, self.pose_mode)")
    if any(fragment not in global_method for fragment in required_global) or any(fragment not in lighter for fragment in required_lighter) or any(fragment not in dpt for fragment in required_dpt) or "def postprocess_pose(out, mode, inverse=False):" not in postprocess or "img = img.resize((512, 384))" not in image_loader or "n_tokens = H * W" not in patch_embed:
        raise EncoderGlobalPooledPoseError("pinned ReCal3R encoder-global topology changed")
    try:
        native_global_at = lighter.index(required_lighter[0])
        native_inquire_at = lighter.index(required_lighter[1], native_global_at)
        native_rollout_at = lighter.index(required_lighter[2], native_inquire_at)
        native_update_at = lighter.index(required_lighter[3], native_rollout_at)
        body = runner[runner.index("\ndef run_encoder_global_pooled_pose_recurrent_lighter(") + 1:]
        runner_global_at = body.index("global_img_feat_i = model._get_img_level_feat(feat_i)")
        runner_capture_at = body.index("capture_encoder_global_pose_token(", runner_global_at)
        runner_inquire_at = body.index("model.pose_retriever.inquire(global_img_feat_i, mem)", runner_capture_at)
        runner_rollout_at = body.index("model._recurrent_rollout(", runner_inquire_at)
        runner_update_at = body.index("model.pose_retriever.update_mem(", runner_rollout_at)
    except ValueError as error:
        raise EncoderGlobalPooledPoseError("pinned encoder-global capture/source order changed") from error
    if not (native_global_at < native_inquire_at < native_rollout_at < native_update_at and runner_global_at < runner_capture_at < runner_inquire_at < runner_rollout_at < runner_update_at):
        raise EncoderGlobalPooledPoseError("pinned encoder-global capture/source order changed")
    capture_region = body[runner_global_at:runner_inquire_at]
    forbidden_capture = ("dec[", "pose_token", "pose_retriever.inquire", "camera_pose", "prediction", ".cpu(", ".numpy(", ".tolist(", "ground_truth", "future", "anchor", "history", "fallback")
    if any(marker in capture_region for marker in forbidden_capture):
        raise EncoderGlobalPooledPoseError("v11 capture boundary contains a forbidden input")
    source_hashes = {
        "model_sha256": hashlib.sha256(model.encode("utf-8")).hexdigest(),
        "dpt_head_sha256": hashlib.sha256(dpt.encode("utf-8")).hexdigest(),
        "postprocess_sha256": hashlib.sha256(postprocess.encode("utf-8")).hexdigest(),
        "patch_embed_sha256": hashlib.sha256(patch_embed.encode("utf-8")).hexdigest(),
        "image_loader_sha256": hashlib.sha256(image_loader.encode("utf-8")).hexdigest(),
    }
    if source_hashes != PINNED_RECAL3R_SOURCE_SHA256:
        raise EncoderGlobalPooledPoseError("pinned ReCal3R encoder-global source SHA-256 differs")
    return {
        "model_path": str(model_path.resolve(strict=True)),
        "dpt_head_path": str(dpt_head_path.resolve(strict=True)),
        "postprocess_path": str(postprocess_path.resolve(strict=True)),
        "patch_embed_path": str(patch_embed_path.resolve(strict=True)),
        "image_loader_path": str(image_loader_path.resolve(strict=True)),
        **source_hashes,
        "runner_path": str(runner_path.resolve(strict=True)),
        "runner_sha256": hashlib.sha256(runner.encode("utf-8")).hexdigest(),
        "encoder_global_source": "native_encoder_global_mean_after_feature_selection_before_pose_retriever",
        "encoder_global_shape": list(ENCODER_GLOBAL_FEATURE_SHAPE),
        "projected_shape": list(PROJECTED_TOKEN_SHAPE),
        "projection_interface": "torch.nn.Linear(1024,768): model.decoder_embed",
        "actual_spatial_token_count_is_runtime_evidence": True,
        "raw_camera_pose_numeric_input_to_v11_decoder": False,
    }


__all__ = [
    "ENCODER_GLOBAL_FEATURE_SHAPE", "PROJECTED_TOKEN_SHAPE", "ROTATION_ATOL",
    "EncoderGlobalPoseCapture", "EncoderGlobalPooledPoseError", "EncoderGlobalPooledPoseResult",
    "PINNED_RECAL3R_SOURCE_SHA256", "audit_pinned_encoder_global_pose_sources",
    "capture_encoder_global_pose_token", "decode_encoder_global_pose_token",
]
