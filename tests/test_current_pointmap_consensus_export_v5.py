from __future__ import annotations

import numpy as np
import pytest

import stateguard3r.current_pointmap_consensus_export_v5 as export_module
from stateguard3r.current_pointmap_consensus_export_v5 import CurrentPointmapConsensusExport, CurrentPointmapConsensusExportError


class _Value:
    def __init__(self, value: object, shape: tuple[int, ...] = (1, 7)) -> None:
        self.value, self.shape = value, shape

    def clone(self) -> "_Value":
        return _Value(self.value, self.shape)


def test_export_is_current_only_and_replaces_only_camera_pose(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = CurrentPointmapConsensusExport(registered_pose_encoder=lambda matrix, template: (_Value("registered"), {"encoder_roundtrip_max_abs_error": 0.0, "encoder_roundtrip_atol": 1e-5}))
    clear = {"camera_pose": np.zeros((1, 7)), "other": _Value("untouched")}
    committed, action = policy.commit_real(0, clear)
    assert committed is clear and action.action == "export_real_camera_pose"
    class _Consensus:
        camera_to_reference = np.eye(4)
        finite_positive_pairs = final_positive_weights = 256
        source_rank = target_rank = 3
        mad_scales = (0.1, 0.1)
        normalized_residual = 0.01
        transform_sha256 = "digest"
        rotation_determinant = 1.0
        orthonormality_max_abs_error = 0.0
    monkeypatch.setattr(export_module, "current_self_cross_consensus", lambda *args: _Consensus())
    raw = {"camera_pose": np.ones((1, 7)), "pts3d_in_self_view": _Value("self"), "pts3d_in_other_view": _Value("cross"), "conf_self": _Value("self-conf"), "conf": _Value("cross-conf"), "other": _Value("untouched")}
    exported, alarm = policy.export_alarm(1, raw)
    assert exported is not raw and exported["camera_pose"].value == "registered" and exported["other"] is raw["other"]
    assert alarm.action == "export_current_pointmap_consensus_se3"
    assert alarm.evidence is not None and alarm.evidence["lattice_pairs"] == 256
    with pytest.raises(CurrentPointmapConsensusExportError, match="contiguous"):
        policy.commit_real(3, clear)
