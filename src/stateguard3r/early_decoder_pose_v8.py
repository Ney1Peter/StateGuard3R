"""Pinned, current-frame early-decoder pose decoding for recovery v8."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping


QUERY_SHAPE = (1, 1, 768)
ROTATION_ATOL = 1e-8
PINNED_RECAL3R_SOURCE_SHA256 = {
    "model_sha256": "32785a6f29fded66aa142207b29a33d7522f0c39aa068fe2175a956ec8dbc3c1",
    "dpt_head_sha256": "c4f4b08c844b5fea47b67cedbcf4888719e8664f4b3b7e7ab6218e33cf63a66e",
    "postprocess_sha256": "ee93ae7fa16897ed9163a21efe225f46193054b19bd7d316402545ed766133d7",
}


class EarlyDecoderPoseError(RuntimeError):
    """The early decoder token cannot safely provide a pose export."""


@dataclass(frozen=True)
class EarlyDecoderPoseResult:
    """A direct frozen-head decode of one captured early decoder token."""

    pose: Any
    camera: Any
    token_shape: tuple[int, int, int]
    rotation_determinant: float
    orthonormality_max_abs_error: float
    homogeneous_max_abs_error: float


def decode_early_decoder_pose_token(
    token: Any, *, torch: Any, pose_head: Callable[[Any], Any], postprocess_pose: Callable[[Any, Any], Any], pose_mode: Any,
    pose_encoding_to_camera: Callable[[Any], Any],
) -> EarlyDecoderPoseResult:
    """Decode only a captured layer-0 decoder token through pinned head APIs."""
    if not bool(torch.is_tensor(token)) or tuple(getattr(token, "shape", ())) != QUERY_SHAPE:
        raise EarlyDecoderPoseError("early decoder pose must have shape (1,1,768)")
    if not bool(torch.is_floating_point(token)) or not bool(torch.isfinite(token).all()):
        raise EarlyDecoderPoseError("early decoder pose must be finite floating point")
    parameters = getattr(pose_head, "parameters", None)
    if callable(parameters):
        try:
            head_device = next(parameters()).device
        except StopIteration:
            head_device = None
        if head_device is not None and token.device != head_device:
            raise EarlyDecoderPoseError("early decoder pose token is not on the pinned head device")
    try:
        raw = pose_head(token[:, 0])
        if tuple(getattr(raw, "shape", ())) != (1, 7):
            raise EarlyDecoderPoseError("pinned pose head has malformed output shape")
        if not bool(torch.is_floating_point(raw)) or not bool(torch.isfinite(raw).all()):
            raise EarlyDecoderPoseError("pinned pose head output is nonfinite")
        # The frozen head itself remains in its model dtype.  Promote its seven
        # numeric outputs on-device before the frozen postprocess/camera path:
        # binary32 quaternion arithmetic cannot meet the preregistered 1e-8
        # proper-SO(3) check even for valid rotations.  This is not a second
        # pose estimator and does not alter the captured token or head weights.
        pose = postprocess_pose(raw.to(dtype=torch.float64), pose_mode)
        camera = pose_encoding_to_camera(pose)
    except EarlyDecoderPoseError:
        raise
    except Exception as error:
        raise EarlyDecoderPoseError(f"pinned early decoder pose decode failed: {error}") from error
    if tuple(getattr(pose, "shape", ())) != (1, 7) or tuple(getattr(camera, "shape", ())) != (1, 4, 4):
        raise EarlyDecoderPoseError("pinned early decoder pose decode has malformed output shape")
    if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(pose).all()) or not bool(torch.isfinite(camera).all()):
        raise EarlyDecoderPoseError("pinned early decoder pose decode is nonfinite")
    rotation = camera[0, :3, :3]
    determinant = float(torch.linalg.det(rotation).item())
    orthonormality = float(torch.max(torch.abs(rotation.transpose(0, 1) @ rotation - torch.eye(3, dtype=rotation.dtype, device=rotation.device))).item())
    homogeneous = float(torch.max(torch.abs(camera[0, 3] - torch.tensor((0.0, 0.0, 0.0, 1.0), dtype=camera.dtype, device=camera.device))).item())
    if not math.isfinite(determinant) or abs(determinant - 1.0) > ROTATION_ATOL or not math.isfinite(orthonormality) or orthonormality > ROTATION_ATOL or not math.isfinite(homogeneous) or homogeneous > ROTATION_ATOL:
        raise EarlyDecoderPoseError("decoded early decoder pose is not proper homogeneous SO(3)")
    return EarlyDecoderPoseResult(pose, camera, QUERY_SHAPE, determinant, orthonormality, homogeneous)


def audit_pinned_early_decoder_pose_sources(model_path: Path, dpt_head_path: Path, postprocess_path: Path, runner_path: Path) -> Mapping[str, Any]:
    """Bind v8 to the official early decoder token and pose-head topology."""
    model, dpt, postprocess, runner = (path.read_text(encoding="utf-8") for path in (model_path, dpt_head_path, postprocess_path, runner_path))
    lighter_start = model.find("    def forward_recurrent_lighter(")
    lighter_end = model.find("\n    def ", lighter_start + 1) if lighter_start >= 0 else -1
    lighter = model[lighter_start: lighter_end if lighter_end >= 0 else None]
    required_model = (
        "global_img_feat_i = self._get_img_level_feat(feat_i)",
        "new_state_feat, dec",
        "self._recurrent_rollout(",
        "out_pose_feat_i = dec[-1][:, 0:1]",
        "self.pose_retriever.update_mem(",
        "mem, global_img_feat_i, out_pose_feat_i",
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
    if any(fragment not in lighter for fragment in required_model) or any(fragment not in dpt for fragment in required_dpt) or "def postprocess_pose(out, mode, inverse=False):" not in postprocess:
        raise EarlyDecoderPoseError("pinned ReCal3R early-decoder pose topology changed")
    try:
        rollout_at = lighter.index(required_model[2])
        final_token_at = lighter.index(required_model[3], rollout_at)
        update_at = lighter.index(required_model[4], final_token_at)
        runner_body = runner[runner.index("\ndef run_early_decoder_pose_recurrent_lighter(") + 1:]
        runner_rollout_at = runner_body.index("model._recurrent_rollout(")
        runner_capture_at = runner_body.index("early_decoder_pose_token = dec[0][:, 0:1].detach().clone() if observer is not None")
        runner_update_at = runner_body.index("model.pose_retriever.update_mem(", runner_rollout_at)
    except ValueError as error:
        raise EarlyDecoderPoseError("pinned early decoder token/source update order changed") from error
    if not (rollout_at < final_token_at < update_at and runner_rollout_at < runner_capture_at < runner_update_at):
        raise EarlyDecoderPoseError("pinned early decoder token/source update order changed")
    source_hashes = {
        "model_sha256": hashlib.sha256(model.encode("utf-8")).hexdigest(),
        "dpt_head_sha256": hashlib.sha256(dpt.encode("utf-8")).hexdigest(),
        "postprocess_sha256": hashlib.sha256(postprocess.encode("utf-8")).hexdigest(),
    }
    if source_hashes != PINNED_RECAL3R_SOURCE_SHA256:
        raise EarlyDecoderPoseError("pinned ReCal3R early-decoder source SHA-256 differs")
    return {
        "model_path": str(model_path.resolve(strict=True)),
        "model_sha256": source_hashes["model_sha256"],
        "dpt_head_path": str(dpt_head_path.resolve(strict=True)),
        "dpt_head_sha256": source_hashes["dpt_head_sha256"],
        "postprocess_path": str(postprocess_path.resolve(strict=True)),
        "postprocess_sha256": source_hashes["postprocess_sha256"],
        "runner_path": str(runner_path.resolve(strict=True)),
        "runner_sha256": hashlib.sha256(runner.encode("utf-8")).hexdigest(),
        "early_decoder_token_source": "decoder_layer_0_after_rollout_before_update_mem",
        "early_decoder_layer_index": 0,
        "capture_relation": "after_rollout_before_update_mem",
        "raw_camera_pose_numeric_input_to_v8_decoder": False,
        "current_image_feature_in_token": True,
        "causal_interpretation": "early decoder latent pose; not current-image-independent and not a raw-pose correction",
    }


__all__ = ["EarlyDecoderPoseError", "EarlyDecoderPoseResult", "PINNED_RECAL3R_SOURCE_SHA256", "QUERY_SHAPE", "ROTATION_ATOL", "audit_pinned_early_decoder_pose_sources", "decode_early_decoder_pose_token"]
