"""External real-safe anchor policy for v4 geometric pose export."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Mapping

import numpy as np

from .orb_pointmap_registration_v4 import register_anchor_orb_3d3d


class GeometricExportError(RuntimeError):
    """A v4 alarm cannot safely produce a registered pose export."""


def _clone(value: Any) -> Any:
    detach, clone = getattr(value, "detach", None), getattr(value, "clone", None)
    if callable(detach) and callable(clone):
        return detach().clone()
    if callable(clone):
        return clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    raise GeometricExportError("external anchor value must expose clone()")


def _value_sha256(value: Any) -> str:
    detached = value.detach() if callable(getattr(value, "detach", None)) else value
    cpu = detached.cpu() if callable(getattr(detached, "cpu", None)) else detached
    array = np.asarray(cpu)
    if not np.isfinite(array).all():
        raise GeometricExportError("pose audit value must be finite")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _matrix_sha256(value: Any) -> str:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise GeometricExportError("registration audit matrix must be finite 4x4")
    return hashlib.sha256(matrix.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class GeometricExportAction:
    frame_id: int
    action: str
    anchor_frame_id: int | None
    registration_evidence: Mapping[str, Any] | None
    anchor_real_pose_sha256: str | None


class PinnedCameraPoseEncoder:
    """Convert a validated NumPy registration matrix through pinned ReCal3R APIs.

    The candidate pose is accepted only as a tensor device/dtype template.  It
    is never read for a numerical value and therefore cannot initialize or
    otherwise influence geometric registration.
    """

    def __init__(self, *, torch: Any, camera_to_pose_encoding: Callable[[Any], Any], pose_encoding_to_camera: Callable[[Any], Any]) -> None:
        self._torch = torch
        self._encode = camera_to_pose_encoding
        self._decode = pose_encoding_to_camera

    def __call__(self, camera_to_reference: Any, pose_template: Any) -> tuple[Any, Mapping[str, Any]]:
        matrix = np.asarray(camera_to_reference, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise GeometricExportError("registration camera matrix must be finite 4x4")
        rotation = matrix[:3, :3]
        determinant = float(np.linalg.det(rotation))
        if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=1e-10):
            raise GeometricExportError("registration camera matrix has invalid homogeneous row")
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-8) or not np.isclose(determinant, 1.0, rtol=0.0, atol=1e-8):
            raise GeometricExportError("registration camera matrix is not proper SO(3)")
        if tuple(getattr(pose_template, "shape", ())) != (1, 7):
            raise GeometricExportError("candidate pose template must have shape (1, 7)")
        try:
            camera = self._torch.as_tensor(matrix, device=pose_template.device, dtype=pose_template.dtype).unsqueeze(0)
            pose = self._encode(camera)
            decoded = self._decode(pose)
        except Exception as error:
            raise GeometricExportError(f"pinned camera pose encoding failed: {error}") from error
        if tuple(getattr(pose, "shape", ())) != (1, 7) or not bool(self._torch.isfinite(pose).all()):
            raise GeometricExportError("pinned camera encoder returned a nonfinite or malformed pose")
        if tuple(getattr(decoded, "shape", ())) != (1, 4, 4) or not bool(self._torch.isfinite(decoded).all()):
            raise GeometricExportError("pinned camera decoder returned a nonfinite or malformed matrix")
        error = float(self._torch.max(self._torch.abs(decoded - camera)).item())
        if not bool(self._torch.allclose(decoded, camera, rtol=0.0, atol=1e-5)):
            raise GeometricExportError("pinned camera pose encoding did not round-trip the registration")
        return pose, {"encoder_roundtrip_max_abs_error": error, "encoder_roundtrip_atol": 1e-5}


class RealSafeGeometricAnchor:
    """Keep one clear-frame RGB/reference-pointmap anchor outside the model."""

    def __init__(self, *, registered_pose_encoder: Callable[[Any, Any], tuple[Any, Mapping[str, Any]]]) -> None:
        self._encode_registered_pose = registered_pose_encoder
        # The pose is retained solely as real-safe-anchor provenance.  The
        # registration routine deliberately receives only RGB and pointmaps.
        self._anchor: tuple[int, Any, Any, Any, str] | None = None
        self._next_frame = 0

    @property
    def anchor_frame_id(self) -> int | None:
        return None if self._anchor is None else self._anchor[0]

    def commit_real(self, frame_id: int, *, rgb: Any, prediction: Mapping[str, Any]) -> tuple[Mapping[str, Any], GeometricExportAction]:
        self._require(frame_id)
        pointmap = prediction.get("pts3d_in_other_view")
        if pointmap is None:
            raise GeometricExportError("real-safe prediction lacks reference pointmap")
        raw_pose = prediction.get("camera_pose")
        if raw_pose is None:
            raise GeometricExportError("real-safe prediction lacks camera pose")
        pose_digest = _value_sha256(raw_pose)
        self._anchor = (frame_id, _clone(rgb), _clone(pointmap), _clone(raw_pose), pose_digest)
        self._next_frame += 1
        return prediction, GeometricExportAction(frame_id, "export_real_camera_pose", None, None, pose_digest)

    def register_alarm(self, frame_id: int, *, rgb: Any, prediction: Mapping[str, Any]) -> tuple[Mapping[str, Any], GeometricExportAction]:
        self._require(frame_id)
        if self._anchor is None:
            raise GeometricExportError("alarm has no prior real-safe geometric anchor")
        anchor_frame, anchor_rgb, anchor_points, _anchor_real_pose, anchor_pose_digest = self._anchor
        current_points = prediction.get("pts3d_in_self_view")
        if current_points is None:
            raise GeometricExportError("alarm prediction lacks current self pointmap")
        pose_template = prediction.get("camera_pose")
        if pose_template is None:
            raise GeometricExportError("alarm prediction lacks candidate pose template")
        try:
            evidence = register_anchor_orb_3d3d(anchor_rgb, rgb, anchor_points, current_points)
            pose, encoder_evidence = self._encode_registered_pose(evidence.registration.camera_to_reference, pose_template)
        except Exception as error:
            raise GeometricExportError(f"registration unavailable: {error}") from error
        if tuple(getattr(pose, "shape", ())) != (1, 7):
            raise GeometricExportError("camera encoder must return shape (1, 7)")
        exported = dict(prediction)
        exported["camera_pose"] = _clone(pose)
        self._next_frame += 1
        rotation = evidence.registration.camera_to_reference[:3, :3]
        return exported, GeometricExportAction(frame_id, "export_anchor_orb_3d3d_registration", anchor_frame, {"anchor_keypoints": evidence.anchor_keypoints, "current_keypoints": evidence.current_keypoints, "ratio_matches": evidence.ratio_matches, "mutual_matches": evidence.mutual_matches, "finite_3d_pairs": evidence.finite_3d_pairs, "inliers": int(evidence.registration.inlier_mask.sum()), "scale": evidence.registration.scale, "normalized_residual": evidence.registration.normalized_residual, "camera_to_reference": evidence.registration.camera_to_reference.tolist(), "registration_matrix_sha256": _matrix_sha256(evidence.registration.camera_to_reference), "proper_rotation": bool(np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-8) and np.isclose(float(np.linalg.det(rotation)), 1.0, rtol=0.0, atol=1e-8)), "rotation_determinant": float(np.linalg.det(rotation)), "orthonormality_max_abs_error": float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))), **dict(encoder_evidence)}, anchor_pose_digest)

    def _require(self, frame_id: int) -> None:
        if type(frame_id) is not int or frame_id != self._next_frame:
            raise GeometricExportError("external anchor frames must be contiguous and causal")


__all__ = ["GeometricExportAction", "GeometricExportError", "PinnedCameraPoseEncoder", "RealSafeGeometricAnchor"]
