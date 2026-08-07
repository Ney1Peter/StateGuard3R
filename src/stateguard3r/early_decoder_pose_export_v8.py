"""External no-fallback early-decoder pose export policy for v8."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Mapping

from .early_decoder_pose_v8 import EarlyDecoderPoseError, EarlyDecoderPoseResult, decode_early_decoder_pose_token


class EarlyDecoderPoseExportError(RuntimeError):
    """An alarm cannot safely export a early decoder token pose."""


def _clone(value: Any) -> Any:
    detached, clone = getattr(value, "detach", None), getattr(value, "clone", None)
    if callable(detached) and callable(clone):
        return detached().clone()
    if callable(clone):
        return clone()
    raise EarlyDecoderPoseExportError("exported pose tensor must expose clone()")


def tensor_gpu_digest(value: Any, *, torch: Any) -> str:
    """Return a deterministic GPU-resident evidence digest without a tensor CPU copy."""
    tensor = value.detach() if callable(getattr(value, "detach", None)) else value
    if not bool(torch.is_tensor(tensor)) or not bool(torch.isfinite(tensor).all()):
        raise EarlyDecoderPoseExportError("audit tensor is nonfinite")
    flat = tensor.reshape(-1).to(dtype=torch.float64)
    positions = torch.arange(1, flat.numel() + 1, dtype=torch.float64, device=flat.device)
    moments = (
        flat.sum(),
        flat.abs().sum(),
        (flat * positions).sum(),
        flat.square().sum(),
        flat.min(),
        flat.max(),
    )
    payload = "|".join((str(tuple(tensor.shape)), str(tensor.dtype), *(float(item.item()).hex() for item in moments)))
    return "gpu-fingerprint-v1:" + hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class EarlyDecoderPoseAction:
    frame_id: int
    action: str
    evidence: Mapping[str, Any] | None


class EarlyDecoderPoseExport:
    """Export raw clear poses and direct captured-token alarm poses only."""

    def __init__(self, *, torch: Any, pose_head: Callable[[Any], Any], postprocess_pose: Callable[[Any, Any], Any], pose_mode: Any, pose_encoding_to_camera: Callable[[Any], Any]) -> None:
        self._torch, self._pose_head, self._postprocess, self._mode, self._camera_decode, self._next_frame = torch, pose_head, postprocess_pose, pose_mode, pose_encoding_to_camera, 0

    def commit_real(self, frame_id: int, prediction: Mapping[str, Any]) -> tuple[Mapping[str, Any], EarlyDecoderPoseAction]:
        self._require(frame_id)
        if tuple(getattr(prediction.get("camera_pose"), "shape", ())) != (1, 7):
            raise EarlyDecoderPoseExportError("clear prediction lacks camera pose")
        self._next_frame += 1
        return prediction, EarlyDecoderPoseAction(frame_id, "export_real_camera_pose", None)

    def export_alarm(self, frame_id: int, prediction: Mapping[str, Any], early_decoder_pose_token: Any, *, token_source: str) -> tuple[Mapping[str, Any], EarlyDecoderPoseAction]:
        self._require(frame_id)
        if token_source != "decoder_layer_0_after_rollout_before_update_mem":
            raise EarlyDecoderPoseExportError("v8 early decoder token source is not pinned")
        try:
            decoded = decode_early_decoder_pose_token(early_decoder_pose_token, torch=self._torch, pose_head=self._pose_head, postprocess_pose=self._postprocess, pose_mode=self._mode, pose_encoding_to_camera=self._camera_decode)
        except EarlyDecoderPoseError as error:
            raise EarlyDecoderPoseExportError(f"early decoder pose unavailable: {error}") from error
        exported = dict(prediction)
        exported["camera_pose"] = _clone(decoded.pose)
        self._next_frame += 1
        return exported, EarlyDecoderPoseAction(frame_id, "export_early_decoder_pose_token", self._evidence(decoded, early_decoder_pose_token, token_source))

    def _evidence(self, decoded: EarlyDecoderPoseResult, token: Any, token_source: str) -> Mapping[str, Any]:
        return {"token_source": token_source, "early_decoder_layer_index": 0, "capture_relation": "after_rollout_before_update_mem", "token_shape": list(decoded.token_shape), "token_dtype": str(token.dtype), "token_device": str(token.device), "token_digest": tensor_gpu_digest(token, torch=self._torch), "postprocess_mode": repr(self._mode), "rotation_determinant": decoded.rotation_determinant, "orthonormality_max_abs_error": decoded.orthonormality_max_abs_error, "homogeneous_max_abs_error": decoded.homogeneous_max_abs_error, "proper_rotation": True, "exported_pose_digest": tensor_gpu_digest(decoded.pose, torch=self._torch)}

    def _require(self, frame_id: int) -> None:
        if type(frame_id) is not int or frame_id != self._next_frame:
            raise EarlyDecoderPoseExportError("v8 frames must be contiguous and causal")


__all__ = ["EarlyDecoderPoseAction", "EarlyDecoderPoseExport", "EarlyDecoderPoseExportError", "tensor_gpu_digest"]
