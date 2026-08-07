"""External no-fallback pre-rollout pose-query export policy for v6."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Mapping

import numpy as np

from .prerollout_pose_query_v6 import PreRolloutPoseQueryError, PreRolloutPoseQueryResult, decode_pre_rollout_pose_query


class PreRolloutPoseQueryExportError(RuntimeError):
    """An alarm cannot safely export a pre-rollout query pose."""


def _clone(value: Any) -> Any:
    detached, clone = getattr(value, "detach", None), getattr(value, "clone", None)
    if callable(detached) and callable(clone):
        return detached().clone()
    if callable(clone):
        return clone()
    raise PreRolloutPoseQueryExportError("exported pose tensor must expose clone()")


def tensor_sha256(value: Any) -> str:
    tensor = value.detach() if callable(getattr(value, "detach", None)) else value
    tensor = tensor.cpu() if callable(getattr(tensor, "cpu", None)) else tensor
    array = np.asarray(tensor)
    if not np.isfinite(array).all():
        raise PreRolloutPoseQueryExportError("audit tensor is nonfinite")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class PreRolloutPoseQueryAction:
    frame_id: int
    action: str
    evidence: Mapping[str, Any] | None


class PreRolloutPoseQueryExport:
    """Export raw clear poses and direct captured-query alarm poses only."""

    def __init__(self, *, torch: Any, pose_head: Callable[[Any], Any], postprocess_pose: Callable[[Any, Any], Any], pose_mode: Any, pose_encoding_to_camera: Callable[[Any], Any]) -> None:
        self._torch, self._pose_head, self._postprocess, self._mode, self._camera_decode, self._next_frame = torch, pose_head, postprocess_pose, pose_mode, pose_encoding_to_camera, 0

    def commit_real(self, frame_id: int, prediction: Mapping[str, Any]) -> tuple[Mapping[str, Any], PreRolloutPoseQueryAction]:
        self._require(frame_id)
        if tuple(getattr(prediction.get("camera_pose"), "shape", ())) != (1, 7):
            raise PreRolloutPoseQueryExportError("clear prediction lacks camera pose")
        self._next_frame += 1
        return prediction, PreRolloutPoseQueryAction(frame_id, "export_real_camera_pose", None)

    def export_alarm(self, frame_id: int, prediction: Mapping[str, Any], pre_rollout_pose_query: Any, *, query_source: str) -> tuple[Mapping[str, Any], PreRolloutPoseQueryAction]:
        self._require(frame_id)
        if query_source not in ("initial_pose_token", "pose_retriever_inquire_pre_rollout"):
            raise PreRolloutPoseQueryExportError("v6 query source is not pinned")
        try:
            decoded = decode_pre_rollout_pose_query(pre_rollout_pose_query, torch=self._torch, pose_head=self._pose_head, postprocess_pose=self._postprocess, pose_mode=self._mode, pose_encoding_to_camera=self._camera_decode)
        except PreRolloutPoseQueryError as error:
            raise PreRolloutPoseQueryExportError(f"pre-rollout pose query unavailable: {error}") from error
        exported = dict(prediction)
        exported["camera_pose"] = _clone(decoded.pose)
        self._next_frame += 1
        return exported, PreRolloutPoseQueryAction(frame_id, "export_pre_rollout_pose_query", self._evidence(decoded, pre_rollout_pose_query, query_source))

    def _evidence(self, decoded: PreRolloutPoseQueryResult, query: Any, query_source: str) -> Mapping[str, Any]:
        return {"query_source": query_source, "query_shape": list(decoded.query_shape), "query_dtype": str(query.dtype), "query_device": str(query.device), "query_sha256": tensor_sha256(query), "postprocess_mode": repr(self._mode), "camera_to_reference": decoded.camera.detach().cpu().tolist(), "rotation_determinant": decoded.rotation_determinant, "orthonormality_max_abs_error": decoded.orthonormality_max_abs_error, "homogeneous_max_abs_error": decoded.homogeneous_max_abs_error, "proper_rotation": True, "exported_pose_sha256": tensor_sha256(decoded.pose)}

    def _require(self, frame_id: int) -> None:
        if type(frame_id) is not int or frame_id != self._next_frame:
            raise PreRolloutPoseQueryExportError("v6 frames must be contiguous and causal")


__all__ = ["PreRolloutPoseQueryAction", "PreRolloutPoseQueryExport", "PreRolloutPoseQueryExportError", "tensor_sha256"]
