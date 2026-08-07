"""Causal SE(3) export fallback for recovery development v3.

This module intentionally knows nothing about ReCal3R state or the detector.
It only keeps externally exported, previously safe camera poses.  An alarmed
candidate can therefore never become an anchor, nor can this history feed back
into model state or detector health.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


class SafeAnchorExportError(RuntimeError):
    """Raised when a causal safe-anchor export cannot be formed."""


def _clone(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    clone = getattr(value, "clone", None)
    if callable(detach) and callable(clone):
        return detach().clone()
    if callable(clone):
        return clone()
    raise SafeAnchorExportError("camera pose must expose clone()")


def _camera_pose(prediction: Mapping[str, Any]) -> Any:
    pose = prediction.get("camera_pose")
    shape = getattr(pose, "shape", None)
    if pose is None or tuple(shape) != (1, 7):
        raise SafeAnchorExportError("prediction camera_pose must have fixed shape (1, 7)")
    return pose


@dataclass(frozen=True)
class ExportAction:
    """One explicit external-output action, for the state timeline audit."""

    frame_id: int
    action: str
    anchor_frame_ids: tuple[int, int] | None


class SafeAnchorSE3Export:
    """Store clear exports and replace only alarmed ``camera_pose`` values.

    For two prior exported camera matrices ``T(t-2), T(t-1)`` the fallback is
    exactly ``(T(t-1) @ inverse(T(t-2))) @ T(t-1)``.  A fallback itself becomes
    the latest *external* export, which is necessary for a causal consecutive
    alarm episode but remains disconnected from the recurrent model.
    """

    def __init__(
        self,
        *,
        pose_encoding_to_camera: Callable[[Any], Any],
        camera_to_pose_encoding: Callable[[Any], Any],
        inverse: Callable[[Any], Any],
    ) -> None:
        self._decode = pose_encoding_to_camera
        self._encode = camera_to_pose_encoding
        self._inverse = inverse
        self._exports: list[tuple[int, Any]] = []
        self._next_frame_id = 0

    @property
    def exported_frame_ids(self) -> tuple[int, ...]:
        return tuple(frame_id for frame_id, _ in self._exports)

    def commit_real(self, frame_id: int, prediction: Mapping[str, Any]) -> tuple[Mapping[str, Any], ExportAction]:
        self._require_frame_id(frame_id)
        pose = _clone(_camera_pose(prediction))
        self._exports.append((frame_id, pose))
        self._next_frame_id += 1
        return prediction, ExportAction(frame_id=frame_id, action="export_real_camera_pose", anchor_frame_ids=None)

    def fallback(self, frame_id: int, prediction: Mapping[str, Any]) -> tuple[Mapping[str, Any], ExportAction]:
        self._require_frame_id(frame_id)
        _camera_pose(prediction)  # validates shape but is deliberately never read as an anchor
        if len(self._exports) < 2:
            raise SafeAnchorExportError("alarm has fewer than two prior safe exported poses")
        older_frame, older = self._exports[-2]
        newer_frame, newer = self._exports[-1]
        older_camera = self._decode(_clone(older))
        newer_camera = self._decode(_clone(newer))
        if tuple(getattr(older_camera, "shape", ())) != (1, 4, 4) or tuple(getattr(newer_camera, "shape", ())) != (1, 4, 4):
            raise SafeAnchorExportError("safe anchor decoder must return shape (1, 4, 4)")
        camera = (newer_camera @ self._inverse(older_camera)) @ newer_camera
        pose = self._encode(camera)
        if tuple(getattr(pose, "shape", ())) != (1, 7):
            raise SafeAnchorExportError("safe anchor encoder must return shape (1, 7)")
        exported = dict(prediction)
        exported["camera_pose"] = _clone(pose)
        self._exports.append((frame_id, _clone(pose)))
        self._next_frame_id += 1
        return exported, ExportAction(
            frame_id=frame_id,
            action="export_safe_anchor_se3_constant_velocity",
            anchor_frame_ids=(older_frame, newer_frame),
        )

    def _require_frame_id(self, frame_id: int) -> None:
        if type(frame_id) is not int or frame_id != self._next_frame_id:
            raise SafeAnchorExportError("safe-anchor export frame IDs must be a contiguous causal sequence")


__all__ = ["ExportAction", "SafeAnchorExportError", "SafeAnchorSE3Export"]
