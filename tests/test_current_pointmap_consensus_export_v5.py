from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import stateguard3r.current_pointmap_consensus_export_v5 as export_module
from stateguard3r.current_pointmap_consensus_export_v5 import CurrentPointmapConsensusExport, CurrentPointmapConsensusExportError, PinnedConsensusPoseEncoder
from stateguard3r.current_pointmap_consensus_v5 import CurrentPointmapConsensusError


torch = pytest.importorskip("torch")


class _Value:
    def __init__(self, value: object, shape: tuple[int, ...] = (1, 7)) -> None:
        self.value, self.shape = value, shape

    def clone(self) -> "_Value":
        return _Value(self.value, self.shape)

    def detach(self) -> "_Value":
        return self


def test_export_is_current_only_and_replaces_only_camera_pose(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = CurrentPointmapConsensusExport(torch=torch, registered_pose_encoder=lambda matrix, template: (_Value("registered"), {"encoder_roundtrip_max_abs_error": 0.0, "encoder_roundtrip_atol": 1e-5}))
    clear = {"camera_pose": torch.zeros((1, 7)), "other": _Value("untouched")}
    committed, action = policy.commit_real(0, clear)
    assert committed is clear and action.action == "export_real_camera_pose"

    class _Consensus:
        camera_to_reference = torch.eye(4, dtype=torch.float64)
        lattice_rows = torch.arange(256) // 16
        lattice_cols = torch.arange(256) % 16
        finite_pairs = finite_positive_pairs = final_positive_weights = 256
        irls_positive_weights = (256, 256)
        source_rank = target_rank = 3
        mad_scales = (0.1, 0.1)
        normalized_residual = 0.01
        rotation_determinant = 1.0
        orthonormality_max_abs_error = 0.0

    monkeypatch.setattr(export_module, "current_self_cross_consensus", lambda *args, **kwargs: _Consensus())
    raw = {"camera_pose": torch.ones((1, 7)), "pts3d_in_self_view": torch.ones((1, 32, 32, 3)), "pts3d_in_other_view": torch.ones((1, 32, 32, 3)), "conf_self": torch.ones((1, 32, 32)), "conf": torch.ones((1, 32, 32)), "other": _Value("untouched")}
    exported, alarm = policy.export_alarm(1, raw)
    assert exported is not raw and exported["camera_pose"].value == "registered" and exported["other"] is raw["other"]
    assert alarm.action == "export_current_pointmap_consensus_se3"
    assert alarm.evidence is not None and alarm.evidence["lattice"]["pairs"] == 256
    assert alarm.evidence["transform_sha256"] == export_module.matrix_sha256(_Consensus.camera_to_reference)
    with pytest.raises(CurrentPointmapConsensusExportError, match="contiguous"):
        policy.commit_real(3, clear)


def test_consensus_failure_never_returns_or_mutates_raw_pose(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = CurrentPointmapConsensusExport(
        torch=torch,
        registered_pose_encoder=lambda matrix, template: pytest.fail("encoder must not run after unavailable consensus"),
    )
    raw_pose = torch.ones((1, 7))
    prediction = {
        "camera_pose": raw_pose,
        "pts3d_in_self_view": torch.ones((1, 32, 32, 3)),
        "pts3d_in_other_view": torch.ones((1, 32, 32, 3)),
        "conf_self": torch.ones((1, 32, 32)),
        "conf": torch.ones((1, 32, 32)),
    }
    monkeypatch.setattr(export_module, "current_self_cross_consensus", lambda *args, **kwargs: (_ for _ in ()).throw(CurrentPointmapConsensusError("rank deficient")))
    with pytest.raises(CurrentPointmapConsensusExportError, match="unavailable"):
        policy.export_alarm(0, prediction)
    assert prediction["camera_pose"] is raw_pose
    torch.testing.assert_close(raw_pose, torch.ones((1, 7)))
    policy.commit_real(0, prediction)


def test_pinned_encoder_rejects_malformed_or_improper_transform() -> None:
    encoder = PinnedConsensusPoseEncoder(
        torch=torch,
        camera_to_pose_encoding=lambda camera: torch.zeros((1, 7), dtype=camera.dtype, device=camera.device),
        pose_encoding_to_camera=lambda pose: torch.eye(4, dtype=pose.dtype, device=pose.device).unsqueeze(0),
    )
    template = torch.zeros((1, 7), dtype=torch.float64)
    with pytest.raises(CurrentPointmapConsensusExportError, match="shape"):
        encoder(torch.eye(3, dtype=torch.float64), template)
    reflection = torch.eye(4, dtype=torch.float64)
    reflection[0, 0] = -1.0
    with pytest.raises(CurrentPointmapConsensusExportError, match="proper homogeneous"):
        encoder(reflection, template)
    malformed = torch.eye(4, dtype=torch.float64)
    malformed[3, 3] = 0.0
    with pytest.raises(CurrentPointmapConsensusExportError, match="proper homogeneous"):
        encoder(malformed, template)


def test_pinned_encoder_round_trips_real_recal3r_camera_api() -> None:
    baseline_src = Path(__file__).resolve().parents[2] / "baselines" / "ReCal3R" / "src"
    if str(baseline_src) not in sys.path:
        sys.path.insert(0, str(baseline_src))
    import dust3r.model  # noqa: F401  # initializes heads before camera's circular imports
    from dust3r.utils.camera import camera_to_pose_encoding, pose_encoding_to_camera

    matrix = torch.eye(4, dtype=torch.float64)
    matrix[:3, 3] = torch.tensor((0.2, -0.3, 0.4), dtype=torch.float64)
    encoder = PinnedConsensusPoseEncoder(
        torch=torch,
        camera_to_pose_encoding=camera_to_pose_encoding,
        pose_encoding_to_camera=pose_encoding_to_camera,
    )
    pose, evidence = encoder(matrix, torch.zeros((1, 7), dtype=torch.float32))
    assert tuple(pose.shape) == (1, 7)
    assert evidence["encoder_roundtrip_max_abs_error"] <= evidence["encoder_roundtrip_atol"]
