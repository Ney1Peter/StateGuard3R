"""Fail-closed alarm export boundary for v11 encoder-global tokens."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .encoder_global_pooled_pose_v11 import (
    EncoderGlobalPoseCapture,
    EncoderGlobalPooledPoseError,
    EncoderGlobalPooledPoseResult,
    decode_encoder_global_pose_token,
)


TOKEN_SOURCE = "native_encoder_global_mean_after_feature_selection_before_pose_retriever"


class EncoderGlobalPooledPoseExportError(RuntimeError):
    """A quarantined frame cannot safely receive a v11 alarm export."""


@dataclass(frozen=True)
class EncoderGlobalPooledPoseAction:
    frame_id: int
    action: str
    evidence: Mapping[str, Any] | None


def tensor_gpu_digest(value: Any, *, torch: Any) -> str:
    """Record a deterministic GPU-side fingerprint without serializing a token."""
    if not bool(torch.is_tensor(value)) or not bool(torch.isfinite(value).all()):
        raise EncoderGlobalPooledPoseExportError("encoder global evidence tensor is unavailable")
    flat = value.detach().contiguous().reshape(-1).to(dtype=torch.float64)
    positions = torch.arange(1, flat.numel() + 1, dtype=torch.float64, device=flat.device)
    moments = (flat.sum(), flat.abs().sum(), (flat * positions).sum(), flat.square().sum(), flat.min(), flat.max())
    payload = "|".join((str(tuple(value.shape)), str(value.dtype), *(float(item.item()).hex() for item in moments)))
    return "gpu-fingerprint-v1:" + hashlib.sha256(payload.encode("ascii")).hexdigest()


class EncoderGlobalPooledPoseExport:
    """Clear frames stay raw; quarantined frames use only a captured v11 token."""

    def __init__(self, *, torch: Any, pose_head: Callable[[Any], Any], postprocess_pose: Callable[[Any, Any], Any], pose_mode: Any, pose_encoding_to_camera: Callable[[Any], Any]) -> None:
        self._torch = torch
        self._pose_head = pose_head
        self._postprocess = postprocess_pose
        self._mode = pose_mode
        self._camera_decode = pose_encoding_to_camera
        self._next_frame = 0

    def commit_real(self, frame_id: int, prediction: Mapping[str, Any]) -> tuple[Mapping[str, Any], EncoderGlobalPooledPoseAction]:
        self._require(frame_id)
        return prediction, EncoderGlobalPooledPoseAction(frame_id, "export_real_camera_pose", None)

    def export_alarm(self, frame_id: int, prediction: Mapping[str, Any], capture: EncoderGlobalPoseCapture, *, token_source: str) -> tuple[Mapping[str, Any], EncoderGlobalPooledPoseAction]:
        self._require(frame_id)
        if token_source != TOKEN_SOURCE:
            raise EncoderGlobalPooledPoseExportError("encoder global token source is not pinned")
        if not isinstance(capture, EncoderGlobalPoseCapture):
            raise EncoderGlobalPooledPoseExportError("encoder global capture is unavailable")
        try:
            decoded = decode_encoder_global_pose_token(capture.token, torch=self._torch, pose_head=self._pose_head, postprocess_pose=self._postprocess, pose_mode=self._mode, pose_encoding_to_camera=self._camera_decode)
        except EncoderGlobalPooledPoseError as error:
            raise EncoderGlobalPooledPoseExportError(f"encoder global alarm export unavailable: {error}") from error
        exported = dict(prediction)
        exported["camera_pose"] = decoded.pose
        return exported, EncoderGlobalPooledPoseAction(frame_id, "export_encoder_global_pose", self._evidence(decoded, capture))

    def _evidence(self, decoded: EncoderGlobalPooledPoseResult, capture: EncoderGlobalPoseCapture) -> Mapping[str, Any]:
        return {
            "token_source": TOKEN_SOURCE,
            "encoder_feature_shape": list(capture.encoder_feature_shape),
            "encoder_spatial_token_count": capture.encoder_feature_shape[1],
            "encoder_global_shape": list(capture.encoder_global_shape),
            "encoder_feature_dtype": capture.encoder_feature_dtype,
            "encoder_feature_device": capture.encoder_feature_device,
            "token_shape": list(decoded.token_shape),
            "token_gpu_digest": tensor_gpu_digest(capture.token, torch=self._torch),
            "rotation_determinant": decoded.rotation_determinant,
            "orthonormality_max_abs_error": decoded.orthonormality_max_abs_error,
            "homogeneous_max_abs_error": decoded.homogeneous_max_abs_error,
            "proper_rotation": True,
            "exported_pose_gpu_digest": tensor_gpu_digest(decoded.pose, torch=self._torch),
        }

    def _require(self, frame_id: int) -> None:
        if type(frame_id) is not int or frame_id != self._next_frame:
            raise EncoderGlobalPooledPoseExportError("v11 frames must be contiguous and causal")
        self._next_frame += 1


__all__ = [
    "EncoderGlobalPooledPoseAction", "EncoderGlobalPooledPoseExport",
    "EncoderGlobalPooledPoseExportError", "TOKEN_SOURCE", "tensor_gpu_digest",
]
