from __future__ import annotations

import pytest

from stateguard3r.geometric_registration_export_v4 import GeometricExportError, RealSafeGeometricAnchor


class _Value:
    def __init__(self, value: object, shape: tuple[int, ...] = (1, 7)) -> None:
        self.value, self.shape = value, shape
    def clone(self) -> "_Value": return _Value(self.value, self.shape)


def test_real_anchor_is_replaced_only_by_clear_frame_and_alarm_needs_anchor() -> None:
    policy = RealSafeGeometricAnchor(camera_to_pose_encoding=lambda matrix: _Value(matrix))
    with pytest.raises(GeometricExportError, match="no prior"):
        policy.register_alarm(0, rgb=_Value("rgb"), prediction={"pts3d_in_self_view": _Value("self")})
    prediction = {"camera_pose": _Value("pose"), "pts3d_in_other_view": _Value("reference")}
    exported, action = policy.commit_real(0, rgb=_Value("safe-rgb"), prediction=prediction)
    assert exported is prediction and action.action == "export_real_camera_pose" and policy.anchor_frame_id == 0
