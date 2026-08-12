"""Fail-closed alarm export boundary for v12 pre-RoPE patch-global tokens."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .patch_embed_global_pooled_pose_v12 import (
    PatchEmbedGlobalPoseCapture,
    PatchEmbedGlobalPooledPoseError,
    PatchEmbedGlobalPooledPoseResult,
    decode_patch_embed_global_pose_token,
)


TOKEN_SOURCE = "native_patch_embed_mean_before_encoder_blocks_and_rope"


class PatchEmbedGlobalPooledPoseExportError(RuntimeError):
    """A quarantined frame cannot safely receive a v12 alarm export."""


@dataclass(frozen=True)
class PatchEmbedGlobalPooledPoseAction:
    frame_id: int
    action: str
    evidence: Mapping[str, Any] | None


def tensor_gpu_digest(value: Any, *, torch: Any) -> str:
    """Record a deterministic GPU-side fingerprint without serializing values."""
    if not bool(torch.is_tensor(value)) or not bool(torch.isfinite(value).all()):
        raise PatchEmbedGlobalPooledPoseExportError("patch-global evidence tensor is unavailable")
    flat = value.detach().contiguous().reshape(-1).to(dtype=torch.float64)
    positions = torch.arange(1, flat.numel() + 1, dtype=torch.float64, device=flat.device)
    moments = (flat.sum(), flat.abs().sum(), (flat * positions).sum(), flat.square().sum(), flat.min(), flat.max())
    payload = "|".join((str(tuple(value.shape)), str(value.dtype), *(float(item.item()).hex() for item in moments)))
    return "gpu-fingerprint-v1:" + hashlib.sha256(payload.encode("ascii")).hexdigest()


class PatchEmbedGlobalPooledPoseExport:
    """Clear frames stay raw; alarms decode only the captured v12 token."""

    def __init__(self, *, torch: Any, pose_head: Callable[[Any], Any], postprocess_pose: Callable[[Any, Any], Any], pose_mode: Any, pose_encoding_to_camera: Callable[[Any], Any]) -> None:
        self._torch = torch
        self._pose_head = pose_head
        self._postprocess = postprocess_pose
        self._mode = pose_mode
        self._camera_decode = pose_encoding_to_camera
        self._next_frame = 0

    def commit_real(self, frame_id: int, prediction: Mapping[str, Any]) -> tuple[Mapping[str, Any], PatchEmbedGlobalPooledPoseAction]:
        self._require(frame_id)
        self._next_frame += 1
        return prediction, PatchEmbedGlobalPooledPoseAction(frame_id, "export_real_camera_pose", None)

    def export_alarm(self, frame_id: int, prediction: Mapping[str, Any], capture: PatchEmbedGlobalPoseCapture, *, token_source: str) -> tuple[Mapping[str, Any], PatchEmbedGlobalPooledPoseAction]:
        self._require(frame_id)
        if token_source != TOKEN_SOURCE:
            raise PatchEmbedGlobalPooledPoseExportError("patch-global token source is not pinned")
        if not isinstance(capture, PatchEmbedGlobalPoseCapture):
            raise PatchEmbedGlobalPooledPoseExportError("patch-global capture is unavailable")
        try:
            decoded = decode_patch_embed_global_pose_token(
                capture.token,
                torch=self._torch,
                pose_head=self._pose_head,
                postprocess_pose=self._postprocess,
                pose_mode=self._mode,
                pose_encoding_to_camera=self._camera_decode,
            )
        except PatchEmbedGlobalPooledPoseError as error:
            raise PatchEmbedGlobalPooledPoseExportError(f"patch-global alarm export unavailable: {error}") from error
        exported = dict(prediction)
        exported["camera_pose"] = decoded.pose
        self._next_frame += 1
        return exported, PatchEmbedGlobalPooledPoseAction(frame_id, "export_patch_embed_global_pose", self._evidence(decoded, capture))

    def _evidence(self, decoded: PatchEmbedGlobalPooledPoseResult, capture: PatchEmbedGlobalPoseCapture) -> Mapping[str, Any]:
        return {
            "token_source": TOKEN_SOURCE,
            "patch_feature_shape": list(capture.patch_feature_shape),
            "patch_spatial_token_count": capture.patch_feature_shape[1],
            "patch_global_shape": list(capture.patch_global_shape),
            "patch_feature_dtype": capture.patch_feature_dtype,
            "patch_feature_device": capture.patch_feature_device,
            "token_shape": list(decoded.token_shape),
            "token_gpu_digest": tensor_gpu_digest(capture.token, torch=self._torch),
            "rotation_determinant": decoded.rotation_determinant,
            "orthonormality_max_abs_error": decoded.orthonormality_max_abs_error,
            "homogeneous_max_abs_error": decoded.homogeneous_max_abs_error,
            "proper_rotation": True,
            "fallback_used": False,
            "no_fallback": True,
            "exported_pose_gpu_digest": tensor_gpu_digest(decoded.pose, torch=self._torch),
        }

    def _require(self, frame_id: int) -> None:
        if type(frame_id) is not int or frame_id != self._next_frame:
            raise PatchEmbedGlobalPooledPoseExportError("v12 frames must be contiguous and causal")


__all__ = [
    "PatchEmbedGlobalPooledPoseAction", "PatchEmbedGlobalPooledPoseExport",
    "PatchEmbedGlobalPooledPoseExportError", "TOKEN_SOURCE", "tensor_gpu_digest",
]
