"""Pinned, current-frame pre-rollout pose-query decoding for recovery v6."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping


QUERY_SHAPE = (1, 1, 768)
ROTATION_ATOL = 1e-8


class PreRolloutPoseQueryError(RuntimeError):
    """The pre-rollout query cannot safely provide a pose export."""


@dataclass(frozen=True)
class PreRolloutPoseQueryResult:
    """A direct frozen-head decode of one captured query token."""

    pose: Any
    camera: Any
    query_shape: tuple[int, int, int]
    rotation_determinant: float
    orthonormality_max_abs_error: float
    homogeneous_max_abs_error: float


def decode_pre_rollout_pose_query(
    query: Any, *, torch: Any, pose_head: Callable[[Any], Any], postprocess_pose: Callable[[Any, Any], Any], pose_mode: Any,
    pose_encoding_to_camera: Callable[[Any], Any],
) -> PreRolloutPoseQueryResult:
    """Decode only a captured pre-rollout query through pinned head APIs."""
    if not bool(torch.is_tensor(query)) or tuple(getattr(query, "shape", ())) != QUERY_SHAPE:
        raise PreRolloutPoseQueryError("pre-rollout pose query must have shape (1,1,768)")
    if not bool(torch.is_floating_point(query)) or not bool(torch.isfinite(query).all()):
        raise PreRolloutPoseQueryError("pre-rollout pose query must be finite floating point")
    try:
        raw = pose_head(query[:, 0])
        if tuple(getattr(raw, "shape", ())) != (1, 7):
            raise PreRolloutPoseQueryError("pinned pose head has malformed output shape")
        if not bool(torch.is_floating_point(raw)) or not bool(torch.isfinite(raw).all()):
            raise PreRolloutPoseQueryError("pinned pose head output is nonfinite")
        # The frozen head itself remains in its model dtype.  Promote its seven
        # numeric outputs on-device before the frozen postprocess/camera path:
        # binary32 quaternion arithmetic cannot meet the preregistered 1e-8
        # proper-SO(3) check even for valid rotations.  This is not a second
        # pose estimator and does not alter the captured query or head weights.
        pose = postprocess_pose(raw.to(dtype=torch.float64), pose_mode)
        camera = pose_encoding_to_camera(pose)
    except Exception as error:
        raise PreRolloutPoseQueryError(f"pinned pre-rollout pose decode failed: {error}") from error
    if tuple(getattr(pose, "shape", ())) != (1, 7) or tuple(getattr(camera, "shape", ())) != (1, 4, 4):
        raise PreRolloutPoseQueryError("pinned pre-rollout pose decode has malformed output shape")
    if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(pose).all()) or not bool(torch.isfinite(camera).all()):
        raise PreRolloutPoseQueryError("pinned pre-rollout pose decode is nonfinite")
    rotation = camera[0, :3, :3]
    determinant = float(torch.linalg.det(rotation).item())
    orthonormality = float(torch.max(torch.abs(rotation.transpose(0, 1) @ rotation - torch.eye(3, dtype=rotation.dtype, device=rotation.device))).item())
    homogeneous = float(torch.max(torch.abs(camera[0, 3] - torch.tensor((0.0, 0.0, 0.0, 1.0), dtype=camera.dtype, device=camera.device))).item())
    if not math.isfinite(determinant) or abs(determinant - 1.0) > ROTATION_ATOL or not math.isfinite(orthonormality) or orthonormality > ROTATION_ATOL or not math.isfinite(homogeneous) or homogeneous > ROTATION_ATOL:
        raise PreRolloutPoseQueryError("decoded pre-rollout pose is not proper homogeneous SO(3)")
    return PreRolloutPoseQueryResult(pose, camera, QUERY_SHAPE, determinant, orthonormality, homogeneous)


def audit_pinned_pose_query_sources(model_path: Path, dpt_head_path: Path, postprocess_path: Path) -> Mapping[str, Any]:
    """Bind v6 to the official pre-rollout query and pose-head topology."""
    model, dpt, postprocess = (path.read_text(encoding="utf-8") for path in (model_path, dpt_head_path, postprocess_path))
    lighter_start = model.find("    def forward_recurrent_lighter(")
    lighter_end = model.find("\n    def ", lighter_start + 1) if lighter_start >= 0 else -1
    lighter = model[lighter_start: lighter_end if lighter_end >= 0 else None]
    required_model = (
        "global_img_feat_i = self._get_img_level_feat(feat_i)",
        "pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)",
        "new_state_feat, dec",
        "self._recurrent_rollout(",
        "self.pose_retriever.update_mem(",
        "mem, global_img_feat_i, out_pose_feat_i",
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
        raise PreRolloutPoseQueryError("pinned ReCal3R pre-rollout pose-query topology changed")
    query_at = lighter.index(required_model[1])
    rollout_at = lighter.index(required_model[3], query_at)
    update_at = lighter.index(required_model[4], rollout_at)
    if not (query_at < rollout_at < update_at):
        raise PreRolloutPoseQueryError("pinned pre-rollout query/source update order changed")
    return {
        "model_path": str(model_path.resolve(strict=True)),
        "model_sha256": hashlib.sha256(model.encode("utf-8")).hexdigest(),
        "dpt_head_path": str(dpt_head_path.resolve(strict=True)),
        "dpt_head_sha256": hashlib.sha256(dpt.encode("utf-8")).hexdigest(),
        "postprocess_path": str(postprocess_path.resolve(strict=True)),
        "postprocess_sha256": hashlib.sha256(postprocess.encode("utf-8")).hexdigest(),
        "pre_rollout_query_source": "pose_retriever.inquire(current_global_img_feature, committed_pre_state_memory)",
        "raw_camera_pose_numeric_input_to_v6_decoder": False,
        "current_image_feature_in_query": True,
        "causal_interpretation": "pre-rollout latent pose query; not current-image-independent and not a raw-pose correction",
    }


__all__ = ["PreRolloutPoseQueryError", "PreRolloutPoseQueryResult", "QUERY_SHAPE", "ROTATION_ATOL", "audit_pinned_pose_query_sources", "decode_pre_rollout_pose_query"]
