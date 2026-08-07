"""External real-safe anchor policy for v4 geometric pose export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .orb_pointmap_registration_v4 import register_anchor_orb_3d3d


class GeometricExportError(RuntimeError):
    """A v4 alarm cannot safely produce a registered pose export."""


def _clone(value: Any) -> Any:
    detach, clone = getattr(value, "detach", None), getattr(value, "clone", None)
    if callable(detach) and callable(clone):
        return detach().clone()
    if callable(clone):
        return clone()
    raise GeometricExportError("external anchor value must expose clone()")


@dataclass(frozen=True)
class GeometricExportAction:
    frame_id: int
    action: str
    anchor_frame_id: int | None
    registration_evidence: Mapping[str, Any] | None


class RealSafeGeometricAnchor:
    """Keep one clear-frame RGB/reference-pointmap anchor outside the model."""

    def __init__(self, *, camera_to_pose_encoding: Callable[[Any], Any]) -> None:
        self._encode = camera_to_pose_encoding
        self._anchor: tuple[int, Any, Any] | None = None
        self._next_frame = 0

    @property
    def anchor_frame_id(self) -> int | None:
        return None if self._anchor is None else self._anchor[0]

    def commit_real(self, frame_id: int, *, rgb: Any, prediction: Mapping[str, Any]) -> tuple[Mapping[str, Any], GeometricExportAction]:
        self._require(frame_id)
        pointmap = prediction.get("pts3d_in_other_view")
        if pointmap is None:
            raise GeometricExportError("real-safe prediction lacks reference pointmap")
        self._anchor = (frame_id, _clone(rgb), _clone(pointmap))
        self._next_frame += 1
        return prediction, GeometricExportAction(frame_id, "export_real_camera_pose", None, None)

    def register_alarm(self, frame_id: int, *, rgb: Any, prediction: Mapping[str, Any]) -> tuple[Mapping[str, Any], GeometricExportAction]:
        self._require(frame_id)
        if self._anchor is None:
            raise GeometricExportError("alarm has no prior real-safe geometric anchor")
        anchor_frame, anchor_rgb, anchor_points = self._anchor
        current_points = prediction.get("pts3d_in_self_view")
        if current_points is None:
            raise GeometricExportError("alarm prediction lacks current self pointmap")
        try:
            evidence = register_anchor_orb_3d3d(anchor_rgb, rgb, anchor_points, current_points)
            pose = self._encode(evidence.registration.camera_to_reference)
        except Exception as error:
            raise GeometricExportError(f"registration unavailable: {error}") from error
        if tuple(getattr(pose, "shape", ())) != (1, 7):
            raise GeometricExportError("camera encoder must return shape (1, 7)")
        exported = dict(prediction)
        exported["camera_pose"] = _clone(pose)
        self._next_frame += 1
        return exported, GeometricExportAction(frame_id, "export_anchor_orb_3d3d_registration", anchor_frame, {"anchor_keypoints": evidence.anchor_keypoints, "current_keypoints": evidence.current_keypoints, "ratio_matches": evidence.ratio_matches, "mutual_matches": evidence.mutual_matches, "finite_3d_pairs": evidence.finite_3d_pairs, "inliers": int(evidence.registration.inlier_mask.sum()), "scale": evidence.registration.scale, "normalized_residual": evidence.registration.normalized_residual})

    def _require(self, frame_id: int) -> None:
        if type(frame_id) is not int or frame_id != self._next_frame:
            raise GeometricExportError("external anchor frames must be contiguous and causal")


__all__ = ["GeometricExportAction", "GeometricExportError", "RealSafeGeometricAnchor"]
