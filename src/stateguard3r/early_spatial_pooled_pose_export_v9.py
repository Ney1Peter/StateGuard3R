"""External fail-closed spatial-pooled pose export policy for v9."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Mapping

from .early_spatial_pooled_pose_v9 import (
    EarlySpatialPooledPoseError,
    EarlySpatialPooledPoseResult,
    decode_early_spatial_pooled_pose_token,
)


TOKEN_SOURCE = "decoder_layer_0_preprojection_spatial_mean_after_frozen_decoder_embed_before_update_mem"


class EarlySpatialPooledPoseExportError(RuntimeError):
    """An alarm cannot safely export the current spatial pooled token pose."""


def _clone(value: Any) -> Any:
    detached, clone = getattr(value, "detach", None), getattr(value, "clone", None)
    if callable(detached) and callable(clone):
        return detached().clone()
    if callable(clone):
        return clone()
    raise EarlySpatialPooledPoseExportError("exported pose tensor must expose clone()")


def tensor_gpu_digest(value: Any, *, torch: Any) -> str:
    """Make a deterministic digest while keeping tensor values on their device."""
    tensor = value.detach() if callable(getattr(value, "detach", None)) else value
    if not bool(torch.is_tensor(tensor)) or not bool(torch.isfinite(tensor).all()):
        raise EarlySpatialPooledPoseExportError("audit tensor is nonfinite")
    flat = tensor.reshape(-1).to(dtype=torch.float64)
    positions = torch.arange(1, flat.numel() + 1, dtype=torch.float64, device=flat.device)
    moments = (flat.sum(), flat.abs().sum(), (flat * positions).sum(), flat.square().sum(), flat.min(), flat.max())
    payload = "|".join((str(tuple(tensor.shape)), str(tensor.dtype), *(float(item.item()).hex() for item in moments)))
    return "gpu-fingerprint-v1:" + hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class EarlySpatialPooledPoseAction:
    frame_id: int
    action: str
    evidence: Mapping[str, Any] | None


class EarlySpatialPooledPoseExport:
    """Clear frames remain raw; alarm frames use only one captured pooled latent."""

    def __init__(self, *, torch: Any, pose_head: Callable[[Any], Any], postprocess_pose: Callable[[Any, Any], Any], pose_mode: Any, pose_encoding_to_camera: Callable[[Any], Any]) -> None:
        self._torch, self._pose_head, self._postprocess, self._mode, self._camera_decode, self._next_frame = torch, pose_head, postprocess_pose, pose_mode, pose_encoding_to_camera, 0

    def commit_real(self, frame_id: int, prediction: Mapping[str, Any]) -> tuple[Mapping[str, Any], EarlySpatialPooledPoseAction]:
        self._require(frame_id)
        if tuple(getattr(prediction.get("camera_pose"), "shape", ())) != (1, 7):
            raise EarlySpatialPooledPoseExportError("clear prediction lacks camera pose")
        self._next_frame += 1
        return prediction, EarlySpatialPooledPoseAction(frame_id, "export_real_camera_pose", None)

    def export_alarm(self, frame_id: int, prediction: Mapping[str, Any], early_spatial_pooled_pose_token: Any, *, token_source: str) -> tuple[Mapping[str, Any], EarlySpatialPooledPoseAction]:
        self._require(frame_id)
        if token_source != TOKEN_SOURCE:
            raise EarlySpatialPooledPoseExportError("v9 spatial pooled token source is not pinned")
        try:
            decoded = decode_early_spatial_pooled_pose_token(early_spatial_pooled_pose_token, torch=self._torch, pose_head=self._pose_head, postprocess_pose=self._postprocess, pose_mode=self._mode, pose_encoding_to_camera=self._camera_decode)
        except EarlySpatialPooledPoseError as error:
            raise EarlySpatialPooledPoseExportError(f"early spatial pooled pose unavailable: {error}") from error
        exported = dict(prediction)
        exported["camera_pose"] = _clone(decoded.pose)
        self._next_frame += 1
        return exported, EarlySpatialPooledPoseAction(frame_id, "export_early_spatial_pooled_pose", self._evidence(decoded, early_spatial_pooled_pose_token))

    def _evidence(self, decoded: EarlySpatialPooledPoseResult, token: Any) -> Mapping[str, Any]:
        return {
            "token_source": TOKEN_SOURCE,
            "early_decoder_layer_index": 0,
            "capture_relation": "after_rollout_before_update_mem",
            "preprojection_spatial_token_count": 576,
            "preprojection_shape": [1, 576, 1024],
            "aggregation": "mean",
            "frozen_decoder_projection": "model.decoder_embed",
            "token_shape": list(decoded.token_shape),
            "token_dtype": str(token.dtype), "token_device": str(token.device),
            "token_digest": tensor_gpu_digest(token, torch=self._torch),
            "postprocess_mode": repr(self._mode),
            "rotation_determinant": decoded.rotation_determinant,
            "orthonormality_max_abs_error": decoded.orthonormality_max_abs_error,
            "homogeneous_max_abs_error": decoded.homogeneous_max_abs_error,
            "proper_rotation": True,
            "fallback_used": False,
            "no_fallback": True,
            "exported_pose_digest": tensor_gpu_digest(decoded.pose, torch=self._torch),
        }

    def _require(self, frame_id: int) -> None:
        if type(frame_id) is not int or frame_id != self._next_frame:
            raise EarlySpatialPooledPoseExportError("v9 frames must be contiguous and causal")


__all__ = ["EarlySpatialPooledPoseAction", "EarlySpatialPooledPoseExport", "EarlySpatialPooledPoseExportError", "TOKEN_SOURCE", "tensor_gpu_digest"]
