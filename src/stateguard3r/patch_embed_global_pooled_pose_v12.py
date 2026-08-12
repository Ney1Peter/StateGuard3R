"""Pinned pre-RoPE patch-global pose decoding for the distinct v12 route.

The decoder receives only a detached projection of a native current-frame
patch embedding.  It deliberately accepts no prediction, raw camera pose,
detector, RGB, point-map, history, ground truth, future frame or CPU transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping


PATCH_FEATURE_WIDTH = 1024
PATCH_GLOBAL_SHAPE = (1, 1, PATCH_FEATURE_WIDTH)
PROJECTED_TOKEN_SHAPE = (1, 1, 768)
ROTATION_ATOL = 1e-8
PINNED_RECAL3R_SOURCE_SHA256 = {
    "model_sha256": "32785a6f29fded66aa142207b29a33d7522f0c39aa068fe2175a956ec8dbc3c1",
    "dpt_head_sha256": "c4f4b08c844b5fea47b67cedbcf4888719e8664f4b3b7e7ab6218e33cf63a66e",
    "postprocess_sha256": "ee93ae7fa16897ed9163a21efe225f46193054b19bd7d316402545ed766133d7",
    "patch_embed_sha256": "c0dae36e876d2124f0e31403a3a306db7e6d5bd69d8e9c6f16bbfb81c82eef3d",
    "image_loader_sha256": "a2738085cdf0f713289a3a42dbd49a1f39325eb3ad590af65970fe4609209d96",
}


class PatchEmbedGlobalPooledPoseError(RuntimeError):
    """The current pre-RoPE patch latent cannot safely provide an alarm pose."""


@dataclass(frozen=True)
class PatchEmbedGlobalPoseCapture:
    """One isolated v12 token and non-sensitive patch shape provenance."""

    token: Any
    patch_feature_shape: tuple[int, int, int]
    patch_global_shape: tuple[int, int, int]
    patch_feature_dtype: str
    patch_feature_device: str


@dataclass(frozen=True)
class PatchEmbedGlobalPooledPoseResult:
    pose: Any
    camera: Any
    token_shape: tuple[int, int, int]
    rotation_determinant: float
    orthonormality_max_abs_error: float
    homogeneous_max_abs_error: float


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in getattr(value, "shape", ()))


def capture_patch_embed_global_pose_token(
    patch_feature: Any, *, projection: Any, torch: Any,
) -> PatchEmbedGlobalPoseCapture:
    """Pool one native pre-RoPE patch sequence then project it once.

    The runtime patch length is retained as evidence, never compared to a
    geometry-specific eligibility constant.
    """
    shape = _shape(patch_feature)
    if len(shape) != 3 or shape[0] != 1 or shape[1] <= 0 or shape[2] != PATCH_FEATURE_WIDTH:
        raise PatchEmbedGlobalPooledPoseError("current patch feature shape is unavailable")
    if not bool(torch.is_tensor(patch_feature)) or not bool(torch.is_floating_point(patch_feature)):
        raise PatchEmbedGlobalPooledPoseError("current patch feature must be floating point")
    if not bool(torch.isfinite(patch_feature).all()):
        raise PatchEmbedGlobalPooledPoseError("current patch feature must be finite")
    if type(projection) is not torch.nn.Linear or projection.in_features != PATCH_FEATURE_WIDTH or projection.out_features != 768:
        raise PatchEmbedGlobalPooledPoseError("pinned frozen decoder projection changed")
    if projection.weight.device != patch_feature.device or projection.weight.dtype != patch_feature.dtype:
        raise PatchEmbedGlobalPooledPoseError("pinned frozen decoder projection device or dtype changed")
    try:
        patch_global = torch.mean(patch_feature, dim=1, keepdim=True)
        token = projection(patch_global).detach().clone()
    except Exception as error:
        raise PatchEmbedGlobalPooledPoseError(f"patch-global projection failed: {error}") from error
    if _shape(patch_global) != PATCH_GLOBAL_SHAPE or not bool(torch.isfinite(patch_global).all()):
        raise PatchEmbedGlobalPooledPoseError("patch global mean is malformed or nonfinite")
    if _shape(token) != PROJECTED_TOKEN_SHAPE or not bool(torch.is_floating_point(token)) or not bool(torch.isfinite(token).all()):
        raise PatchEmbedGlobalPooledPoseError("patch global projected token is malformed or nonfinite")
    return PatchEmbedGlobalPoseCapture(
        token=token,
        patch_feature_shape=(1, shape[1], PATCH_FEATURE_WIDTH),
        patch_global_shape=PATCH_GLOBAL_SHAPE,
        patch_feature_dtype=str(patch_feature.dtype),
        patch_feature_device=str(patch_feature.device),
    )


def decode_patch_embed_global_pose_token(
    token: Any, *, torch: Any, pose_head: Callable[[Any], Any],
    postprocess_pose: Callable[[Any, Any], Any], pose_mode: Any,
    pose_encoding_to_camera: Callable[[Any], Any],
) -> PatchEmbedGlobalPooledPoseResult:
    """Decode an already-projected v12 token with the pinned official head."""
    if not bool(torch.is_tensor(token)) or _shape(token) != PROJECTED_TOKEN_SHAPE:
        raise PatchEmbedGlobalPooledPoseError("patch global pose token must have shape (1,1,768)")
    if not bool(torch.is_floating_point(token)) or not bool(torch.isfinite(token).all()):
        raise PatchEmbedGlobalPooledPoseError("patch global pose token must be finite floating point")
    parameters = getattr(pose_head, "parameters", None)
    if callable(parameters):
        try:
            head_device = next(parameters()).device
        except StopIteration:
            head_device = None
        if head_device is not None and token.device != head_device:
            raise PatchEmbedGlobalPooledPoseError("patch global pose token is not on the pinned head device")
    try:
        raw = pose_head(token[:, 0])
        if _shape(raw) != (1, 7) or not bool(torch.is_floating_point(raw)) or not bool(torch.isfinite(raw).all()):
            raise PatchEmbedGlobalPooledPoseError("pinned pose head has malformed output")
        pose = postprocess_pose(raw.to(dtype=torch.float64), pose_mode)
        camera = pose_encoding_to_camera(pose)
    except PatchEmbedGlobalPooledPoseError:
        raise
    except Exception as error:
        raise PatchEmbedGlobalPooledPoseError(f"pinned patch-global pose decode failed: {error}") from error
    if _shape(pose) != (1, 7) or _shape(camera) != (1, 4, 4):
        raise PatchEmbedGlobalPooledPoseError("pinned patch-global decode has malformed output shape")
    if not bool(torch.isfinite(pose).all()) or not bool(torch.isfinite(camera).all()):
        raise PatchEmbedGlobalPooledPoseError("pinned patch-global decode is nonfinite")
    rotation = camera[0, :3, :3]
    determinant = float(torch.linalg.det(rotation).item())
    orthonormality = float(torch.max(torch.abs(rotation.transpose(0, 1) @ rotation - torch.eye(3, dtype=rotation.dtype, device=rotation.device))).item())
    homogeneous = float(torch.max(torch.abs(camera[0, 3] - torch.tensor((0.0, 0.0, 0.0, 1.0), dtype=camera.dtype, device=camera.device))).item())
    if not math.isfinite(determinant) or abs(determinant - 1.0) > ROTATION_ATOL or not math.isfinite(orthonormality) or orthonormality > ROTATION_ATOL or not math.isfinite(homogeneous) or homogeneous > ROTATION_ATOL:
        raise PatchEmbedGlobalPooledPoseError("decoded patch-global pose is not proper homogeneous SO(3)")
    return PatchEmbedGlobalPooledPoseResult(pose, camera, PROJECTED_TOKEN_SHAPE, determinant, orthonormality, homogeneous)


def audit_pinned_patch_embed_global_pose_sources(
    model_path: Path, dpt_head_path: Path, postprocess_path: Path, patch_embed_path: Path,
    image_loader_path: Path, runner_path: Path,
) -> Mapping[str, Any]:
    """Bind v12 to native patch embedding before encoder/RoPE and recurrence."""
    model, dpt, postprocess, patch_embed, image_loader, runner = (
        path.read_text(encoding="utf-8")
        for path in (model_path, dpt_head_path, postprocess_path, patch_embed_path, image_loader_path, runner_path)
    )
    encode_start = model.find("    def _encode_image(self, image, true_shape):")
    encode_end = model.find("\n    def ", encode_start + 1) if encode_start >= 0 else -1
    encode_method = model[encode_start:encode_end if encode_end >= 0 else None]
    lighter_start = model.find("    def forward_recurrent_lighter(")
    lighter_end = model.find("\n    def ", lighter_start + 1) if lighter_start >= 0 else -1
    lighter = model[lighter_start:lighter_end if lighter_end >= 0 else None]
    required_encode = ("x, pos = self.patch_embed(image, true_shape=true_shape)", "for blk in self.enc_blocks:", "x = blk(x, pos)")
    required_lighter = ("self._encode_image(selected_imgs, selected_shapes)", "self.pose_retriever.inquire(global_img_feat_i, mem)", "self._recurrent_rollout(", "self.pose_retriever.update_mem(")
    required_patch = (
        "def get_patch_embed(patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3):",
        'assert patch_embed_cls in ["PatchEmbedDust3R", "ManyAR_PatchEmbed"]',
        "class PatchEmbedDust3R(PatchEmbed):",
        "def forward(self, x, **kw):",
        "x = self.proj(x)",
        "x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC",
        "x = self.norm(x)",
        "return x, pos",
    )
    required_dpt = ("class DPTPts3dPose", "self.pose_head = PoseDecoder(hidden_size=in_dim)", "pose_token = x[-1][:, 0].clone()", "pose = self.pose_head(pose_token)", "pose = postprocess_pose(pose, self.pose_mode)")
    required_loader = (
        "def load_images_for_eval(",
        "# resize long side to 512",
        "img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))",
    )
    required_checkpoint_loader = 'args = ckpt["args"].model.replace(\n        "ManyAR_PatchEmbed", "PatchEmbedDust3R"\n    )'
    if any(fragment not in encode_method for fragment in required_encode) or any(fragment not in lighter for fragment in required_lighter) or any(fragment not in patch_embed for fragment in required_patch) or any(fragment not in dpt for fragment in required_dpt) or any(fragment not in image_loader for fragment in required_loader) or required_checkpoint_loader not in model or "def postprocess_pose(out, mode, inverse=False):" not in postprocess:
        raise PatchEmbedGlobalPooledPoseError("pinned ReCal3R patch-global topology changed")
    try:
        native_patch_at = encode_method.index(required_encode[0])
        native_block_at = encode_method.index(required_encode[1], native_patch_at)
        native_global_at = lighter.index("global_img_feat_i = self._get_img_level_feat(feat_i)")
        native_inquire_at = lighter.index(required_lighter[1], native_global_at)
        native_rollout_at = lighter.index(required_lighter[2], native_inquire_at)
        native_update_at = lighter.index(required_lighter[3], native_rollout_at)
        body = runner[runner.index("\ndef run_patch_embed_global_pooled_pose_recurrent_lighter(") + 1:]
        runner_patch_at = body.index("patch_tokens_i, _patch_pos_i = model.patch_embed(selected_imgs, true_shape=selected_shapes)")
        runner_capture_at = body.index("capture_patch_embed_global_pose_token(", runner_patch_at)
        runner_encode_at = body.index("model._encode_image(selected_imgs, selected_shapes)", runner_capture_at)
        runner_inquire_at = body.index("model.pose_retriever.inquire(global_img_feat_i, mem)", runner_encode_at)
        runner_rollout_at = body.index("model._recurrent_rollout(", runner_inquire_at)
        runner_update_at = body.index("model.pose_retriever.update_mem(", runner_rollout_at)
    except ValueError as error:
        raise PatchEmbedGlobalPooledPoseError("pinned patch-global capture/source order changed") from error
    if not (native_patch_at < native_block_at and native_global_at < native_inquire_at < native_rollout_at < native_update_at and runner_patch_at < runner_capture_at < runner_encode_at < runner_inquire_at < runner_rollout_at < runner_update_at):
        raise PatchEmbedGlobalPooledPoseError("pinned patch-global capture/source order changed")
    capture_region = body[runner_patch_at:runner_encode_at]
    forbidden_capture = ("dec[", "model.pose_token", "pose_retriever.inquire", "camera_pose", "prediction", ".cpu(", ".numpy(", ".tolist(", "ground_truth", "future", "anchor", "history", "fallback")
    if any(marker in capture_region for marker in forbidden_capture):
        raise PatchEmbedGlobalPooledPoseError("v12 patch capture boundary contains a forbidden input")
    source_hashes = {
        "model_sha256": hashlib.sha256(model.encode("utf-8")).hexdigest(),
        "dpt_head_sha256": hashlib.sha256(dpt.encode("utf-8")).hexdigest(),
        "postprocess_sha256": hashlib.sha256(postprocess.encode("utf-8")).hexdigest(),
        "patch_embed_sha256": hashlib.sha256(patch_embed.encode("utf-8")).hexdigest(),
        "image_loader_sha256": hashlib.sha256(image_loader.encode("utf-8")).hexdigest(),
    }
    if source_hashes != PINNED_RECAL3R_SOURCE_SHA256:
        raise PatchEmbedGlobalPooledPoseError("pinned ReCal3R patch-global source SHA-256 differs")
    return {
        "model_path": str(model_path.resolve(strict=True)),
        "dpt_head_path": str(dpt_head_path.resolve(strict=True)),
        "postprocess_path": str(postprocess_path.resolve(strict=True)),
        "patch_embed_path": str(patch_embed_path.resolve(strict=True)),
        "image_loader_path": str(image_loader_path.resolve(strict=True)),
        "runner_path": str(runner_path.resolve(strict=True)),
        "runner_sha256": hashlib.sha256(runner.encode("utf-8")).hexdigest(),
        **source_hashes,
        "patch_feature_source": "native_model_patch_embed_before_encoder_blocks_and_rope",
        "patch_global_shape": list(PATCH_GLOBAL_SHAPE),
        "projected_shape": list(PROJECTED_TOKEN_SHAPE),
        "projection_interface": "torch.nn.Linear(1024,768): model.decoder_embed",
        "actual_patch_token_count_is_runtime_evidence": True,
        "raw_camera_pose_numeric_input_to_v12_decoder": False,
    }


__all__ = [
    "PATCH_FEATURE_WIDTH", "PATCH_GLOBAL_SHAPE", "PROJECTED_TOKEN_SHAPE", "ROTATION_ATOL",
    "PatchEmbedGlobalPoseCapture", "PatchEmbedGlobalPooledPoseError", "PatchEmbedGlobalPooledPoseResult",
    "PINNED_RECAL3R_SOURCE_SHA256", "audit_pinned_patch_embed_global_pose_sources",
    "capture_patch_embed_global_pose_token", "decode_patch_embed_global_pose_token",
]
