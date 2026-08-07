from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

import stateguard3r.geometric_registration_export_v4 as policy_module
from stateguard3r.geometric_registration_export_v4 import GeometricExportError, PinnedCameraPoseEncoder, RealSafeGeometricAnchor
from stateguard3r.geometric_registration_v4 import RegistrationResult
from stateguard3r.orb_pointmap_registration_v4 import ORBPointmapRegistrationEvidence


class _Value:
    def __init__(self, value: object, shape: tuple[int, ...] = (1, 7)) -> None:
        self.value, self.shape = value, shape
    def clone(self) -> "_Value": return _Value(self.value, self.shape)


def _encoder(matrix: object, template: object) -> tuple[_Value, dict[str, float]]:
    assert np.asarray(matrix).shape == (4, 4)
    assert np.asarray(template).shape == (1, 7)
    return _Value("registered"), {"encoder_roundtrip_max_abs_error": 0.0, "encoder_roundtrip_atol": 1e-5}


def test_real_anchor_is_replaced_only_by_clear_frame_and_alarm_needs_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = RealSafeGeometricAnchor(registered_pose_encoder=_encoder)
    with pytest.raises(GeometricExportError, match="no prior"):
        policy.register_alarm(0, rgb=_Value("rgb"), prediction={"camera_pose": np.zeros((1, 7)), "pts3d_in_self_view": _Value("self")})
    prediction = {"camera_pose": np.zeros((1, 7)), "pts3d_in_other_view": _Value("reference")}
    exported, action = policy.commit_real(0, rgb=_Value("safe-rgb"), prediction=prediction)
    assert exported is prediction and action.action == "export_real_camera_pose" and action.anchor_real_pose_sha256 is not None and policy.anchor_frame_id == 0
    registration = RegistrationResult(np.eye(4), np.ones(24, dtype=bool), 1.0, 0.0)
    evidence = ORBPointmapRegistrationEvidence(30, 30, 28, 26, 24, registration)
    monkeypatch.setattr(policy_module, "register_anchor_orb_3d3d", lambda *args: evidence)
    alarm_prediction = {"camera_pose": np.ones((1, 7)), "pts3d_in_self_view": _Value("self"), "untouched": _Value("value")}
    registered, registered_action = policy.register_alarm(1, rgb=_Value("alarm-rgb"), prediction=alarm_prediction)
    assert registered is not alarm_prediction
    assert registered["camera_pose"].value == "registered"
    assert registered["untouched"] is alarm_prediction["untouched"]
    assert registered_action.anchor_frame_id == 0
    assert registered_action.anchor_real_pose_sha256 == action.anchor_real_pose_sha256
    assert registered_action.registration_evidence is not None
    assert registered_action.registration_evidence["proper_rotation"] is True
    assert registered_action.registration_evidence["camera_to_reference"] == np.eye(4).tolist()


def test_pinned_recal3r_camera_encoder_tensorizes_and_round_trips() -> None:
    torch = pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1]
    baseline_src = root.parent / "baselines" / "ReCal3R" / "src"
    sys.path.insert(0, str(baseline_src))
    try:
        import dust3r.model
        from dust3r.utils.camera import camera_to_pose_encoding, pose_encoding_to_camera
    finally:
        sys.path.remove(str(baseline_src))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = (0.2, -0.3, 0.4)
    encoder = PinnedCameraPoseEncoder(torch=torch, camera_to_pose_encoding=camera_to_pose_encoding, pose_encoding_to_camera=pose_encoding_to_camera)
    pose, evidence = encoder(matrix, torch.zeros((1, 7), dtype=torch.float32))
    assert tuple(pose.shape) == (1, 7)
    assert evidence["encoder_roundtrip_max_abs_error"] <= evidence["encoder_roundtrip_atol"]
