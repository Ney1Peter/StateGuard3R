"""External current-frame consensus pose export for recovery v5."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Mapping

import numpy as np

from .current_pointmap_consensus_v5 import current_self_cross_consensus


class CurrentPointmapConsensusExportError(RuntimeError):
    """A v5 alarm cannot safely export a current pointmap consensus pose."""


def _clone(value: Any) -> Any:
    value = value.detach() if callable(getattr(value, "detach", None)) else value
    if callable(getattr(value, "clone", None)):
        return value.clone()
    raise CurrentPointmapConsensusExportError("exported tensor must expose clone")


def pose_sha256(value: Any) -> str:
    value = value.detach() if callable(getattr(value, "detach", None)) else value
    value = value.cpu() if callable(getattr(value, "cpu", None)) else value
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise CurrentPointmapConsensusExportError("pose audit tensor is nonfinite")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


class PinnedConsensusPoseEncoder:
    """Validate and round-trip a consensus matrix through pinned camera APIs."""

    def __init__(self, *, torch: Any, camera_to_pose_encoding: Callable[[Any], Any], pose_encoding_to_camera: Callable[[Any], Any]) -> None:
        self._torch, self._encode, self._decode = torch, camera_to_pose_encoding, pose_encoding_to_camera

    def __call__(self, matrix: Any, pose_template: Any) -> tuple[Any, Mapping[str, float]]:
        matrix = np.asarray(matrix, dtype=np.float64)
        rotation = matrix[:3, :3] if matrix.shape == (4, 4) else None
        if rotation is None or not np.isfinite(matrix).all() or not np.allclose(matrix[3], (0, 0, 0, 1), rtol=0.0, atol=1e-10):
            raise CurrentPointmapConsensusExportError("consensus matrix is malformed")
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-8) or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-8):
            raise CurrentPointmapConsensusExportError("consensus matrix is not proper SO(3)")
        if tuple(getattr(pose_template, "shape", ())) != (1, 7):
            raise CurrentPointmapConsensusExportError("raw pose template must have shape (1, 7)")
        try:
            camera = self._torch.as_tensor(matrix, device=pose_template.device, dtype=pose_template.dtype).unsqueeze(0)
            pose = self._encode(camera)
            decoded = self._decode(pose)
        except Exception as error:
            raise CurrentPointmapConsensusExportError(f"pinned camera encoding failed: {error}") from error
        error = float(self._torch.max(self._torch.abs(decoded - camera)).item())
        if tuple(getattr(pose, "shape", ())) != (1, 7) or not bool(self._torch.isfinite(pose).all()) or error > 1e-5:
            raise CurrentPointmapConsensusExportError("pinned camera round-trip failed")
        return pose, {"encoder_roundtrip_max_abs_error": error, "encoder_roundtrip_atol": 1e-5}


@dataclass(frozen=True)
class ConsensusExportAction:
    frame_id: int
    action: str
    evidence: Mapping[str, Any] | None


class CurrentPointmapConsensusExport:
    """No-history, export-only policy: clear raw pose or current consensus pose."""

    def __init__(self, *, registered_pose_encoder: Callable[[Any, Any], tuple[Any, Mapping[str, float]]]) -> None:
        self._encode, self._next_frame = registered_pose_encoder, 0

    def commit_real(self, frame_id: int, prediction: Mapping[str, Any]) -> tuple[Mapping[str, Any], ConsensusExportAction]:
        self._require(frame_id)
        self._next_frame += 1
        return prediction, ConsensusExportAction(frame_id, "export_real_camera_pose", None)

    def export_alarm(self, frame_id: int, prediction: Mapping[str, Any]) -> tuple[Mapping[str, Any], ConsensusExportAction]:
        self._require(frame_id)
        required = ("pts3d_in_self_view", "pts3d_in_other_view", "conf_self", "conf", "camera_pose")
        if any(prediction.get(key) is None for key in required):
            raise CurrentPointmapConsensusExportError("alarm prediction lacks required current pointmap field")
        try:
            consensus = current_self_cross_consensus(prediction["pts3d_in_self_view"], prediction["pts3d_in_other_view"], prediction["conf_self"], prediction["conf"])
            pose, encoding = self._encode(consensus.camera_to_reference, prediction["camera_pose"])
        except Exception as error:
            raise CurrentPointmapConsensusExportError(f"current pointmap consensus unavailable: {error}") from error
        exported = dict(prediction)
        exported["camera_pose"] = _clone(pose)
        self._next_frame += 1
        return exported, ConsensusExportAction(frame_id, "export_current_pointmap_consensus_se3", {"lattice_side": 16, "lattice_pairs": 256, "finite_positive_pairs": consensus.finite_positive_pairs, "final_positive_weights": consensus.final_positive_weights, "source_rank": consensus.source_rank, "target_rank": consensus.target_rank, "mad_scales": list(consensus.mad_scales), "normalized_residual": consensus.normalized_residual, "camera_to_reference": consensus.camera_to_reference.tolist(), "transform_sha256": consensus.transform_sha256, "proper_rotation": True, "rotation_determinant": consensus.rotation_determinant, "orthonormality_max_abs_error": consensus.orthonormality_max_abs_error, **dict(encoding)})

    def _require(self, frame_id: int) -> None:
        if type(frame_id) is not int or frame_id != self._next_frame:
            raise CurrentPointmapConsensusExportError("v5 frames must be contiguous and causal")


__all__ = ["ConsensusExportAction", "CurrentPointmapConsensusExport", "CurrentPointmapConsensusExportError", "PinnedConsensusPoseEncoder", "pose_sha256"]
