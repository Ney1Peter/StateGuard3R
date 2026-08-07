"""External GPU current-pointmap consensus export policy for recovery v5."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Mapping

import numpy as np

from .current_pointmap_consensus_v5 import (
    CurrentPointmapConsensusError,
    CurrentPointmapConsensusResult,
    current_self_cross_consensus,
    matrix_sha256,
)


class CurrentPointmapConsensusExportError(RuntimeError):
    """A v5 alarm cannot safely export a current pointmap consensus pose."""


def _clone(value: Any) -> Any:
    detached, clone = getattr(value, "detach", None), getattr(value, "clone", None)
    if callable(detached) and callable(clone):
        return detached().clone()
    if callable(clone):
        return clone()
    raise CurrentPointmapConsensusExportError("exported tensor must expose clone()")


def pose_sha256(value: Any) -> str:
    tensor = value.detach() if callable(getattr(value, "detach", None)) else value
    tensor = tensor.cpu() if callable(getattr(tensor, "cpu", None)) else tensor
    array = np.asarray(tensor)
    if not np.isfinite(array).all():
        raise CurrentPointmapConsensusExportError("pose audit tensor is nonfinite")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


class PinnedConsensusPoseEncoder:
    """Encode a GPU transform; raw pose is used only as dtype/device template."""

    def __init__(self, *, torch: Any, camera_to_pose_encoding: Callable[[Any], Any], pose_encoding_to_camera: Callable[[Any], Any]) -> None:
        self._torch, self._encode, self._decode = torch, camera_to_pose_encoding, pose_encoding_to_camera

    def __call__(self, matrix: Any, pose_template: Any) -> tuple[Any, Mapping[str, float]]:
        if not bool(self._torch.is_tensor(matrix)) or tuple(getattr(matrix, "shape", ())) != (4, 4):
            raise CurrentPointmapConsensusExportError("consensus matrix must have shape (4,4)")
        if not bool(self._torch.is_floating_point(matrix)) or not bool(self._torch.isfinite(matrix).all()):
            raise CurrentPointmapConsensusExportError("consensus matrix must be finite floating point")
        rotation = matrix[:3, :3]
        identity = self._torch.eye(3, dtype=matrix.dtype, device=matrix.device)
        determinant = float(self._torch.linalg.det(rotation).item())
        orthonormality = float(self._torch.max(self._torch.abs(rotation.transpose(0, 1) @ rotation - identity)).item())
        homogeneous = float(self._torch.max(self._torch.abs(matrix[3] - self._torch.tensor((0.0, 0.0, 0.0, 1.0), dtype=matrix.dtype, device=matrix.device))).item())
        if not math.isfinite(determinant) or determinant <= 0.0 or abs(determinant - 1.0) > 1e-8 or not math.isfinite(orthonormality) or orthonormality > 1e-8 or not math.isfinite(homogeneous) or homogeneous > 1e-8:
            raise CurrentPointmapConsensusExportError("consensus matrix is not a proper homogeneous SO(3) transform")
        if not bool(self._torch.is_tensor(pose_template)) or tuple(getattr(pose_template, "shape", ())) != (1, 7):
            raise CurrentPointmapConsensusExportError("raw pose template must have shape (1,7)")
        try:
            camera = matrix.to(device=pose_template.device, dtype=pose_template.dtype).unsqueeze(0)
            pose = self._encode(camera)
            decoded = self._decode(pose)
        except Exception as error:
            raise CurrentPointmapConsensusExportError(f"pinned camera encoding failed: {error}") from error
        error = float(self._torch.max(self._torch.abs(decoded - camera)).item())
        if tuple(getattr(pose, "shape", ())) != (1, 7) or tuple(getattr(decoded, "shape", ())) != (1, 4, 4) or not bool(self._torch.isfinite(pose).all()) or not bool(self._torch.isfinite(decoded).all()) or error > 1e-5:
            raise CurrentPointmapConsensusExportError("pinned camera round-trip failed")
        return pose, {"encoder_roundtrip_max_abs_error": error, "encoder_roundtrip_atol": 1e-5}


@dataclass(frozen=True)
class ConsensusExportAction:
    frame_id: int
    action: str
    evidence: Mapping[str, Any] | None


class CurrentPointmapConsensusExport:
    """No-history export policy: clear raw pose, alarm current-only consensus."""

    def __init__(self, *, torch: Any, registered_pose_encoder: Callable[[Any, Any], tuple[Any, Mapping[str, float]]]) -> None:
        self._torch, self._encode, self._next_frame = torch, registered_pose_encoder, 0

    def commit_real(self, frame_id: int, prediction: Mapping[str, Any]) -> tuple[Mapping[str, Any], ConsensusExportAction]:
        self._require(frame_id)
        if tuple(getattr(prediction.get("camera_pose"), "shape", ())) != (1, 7):
            raise CurrentPointmapConsensusExportError("clear prediction lacks camera pose")
        self._next_frame += 1
        return prediction, ConsensusExportAction(frame_id, "export_real_camera_pose", None)

    def export_alarm(self, frame_id: int, prediction: Mapping[str, Any]) -> tuple[Mapping[str, Any], ConsensusExportAction]:
        self._require(frame_id)
        pose_template = prediction.get("camera_pose")
        if tuple(getattr(pose_template, "shape", ())) != (1, 7):
            raise CurrentPointmapConsensusExportError("alarm prediction lacks camera-pose template")
        try:
            consensus = current_self_cross_consensus(prediction.get("pts3d_in_self_view"), prediction.get("pts3d_in_other_view"), prediction.get("conf_self"), prediction.get("conf"), torch=self._torch)
            pose, encoding = self._encode(consensus.camera_to_reference, pose_template)
        except (CurrentPointmapConsensusError, CurrentPointmapConsensusExportError) as error:
            raise CurrentPointmapConsensusExportError(f"current pointmap consensus unavailable: {error}") from error
        exported = dict(prediction)
        exported["camera_pose"] = _clone(pose)
        self._next_frame += 1
        return exported, ConsensusExportAction(frame_id, "export_current_pointmap_consensus_se3", self._evidence(consensus, encoding))

    @staticmethod
    def _evidence(consensus: CurrentPointmapConsensusResult, encoding: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "lattice": {"side": 16, "pairs": 256, "row_indices": [int(item) for item in consensus.lattice_rows.detach().cpu().tolist()], "col_indices": [int(item) for item in consensus.lattice_cols.detach().cpu().tolist()]},
            "finite_pairs": consensus.finite_pairs,
            "finite_positive_pairs": consensus.finite_positive_pairs,
            "irls_positive_weights": list(consensus.irls_positive_weights),
            "final_positive_weights": consensus.final_positive_weights,
            "source_rank": consensus.source_rank,
            "target_rank": consensus.target_rank,
            "mad_scales": list(consensus.mad_scales),
            "normalized_residual": consensus.normalized_residual,
            "camera_to_reference": consensus.camera_to_reference.detach().cpu().tolist(),
            "transform_sha256": matrix_sha256(consensus.camera_to_reference),
            "proper_rotation": True,
            "rotation_determinant": consensus.rotation_determinant,
            "orthonormality_max_abs_error": consensus.orthonormality_max_abs_error,
            **dict(encoding),
        }

    def _require(self, frame_id: int) -> None:
        if type(frame_id) is not int or frame_id != self._next_frame:
            raise CurrentPointmapConsensusExportError("v5 frames must be contiguous and causal")


__all__ = ["ConsensusExportAction", "CurrentPointmapConsensusExport", "CurrentPointmapConsensusExportError", "PinnedConsensusPoseEncoder", "pose_sha256"]
