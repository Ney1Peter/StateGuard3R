from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from stateguard3r.encoder_global_pooled_pose_export_v11 import (
    EncoderGlobalPooledPoseExport,
    EncoderGlobalPooledPoseExportError,
    TOKEN_SOURCE,
)
from stateguard3r.encoder_global_pooled_pose_v11 import EncoderGlobalPoseCapture


def _torch() -> object:
    return pytest.importorskip("torch")


def _camera(torch: object, pose: object) -> object:
    camera = torch.eye(4, dtype=pose.dtype, device=pose.device).unsqueeze(0)
    camera[:, :3, 3] = pose[:, :3]
    return camera


def _head(torch: object, token: object) -> object:
    pose = torch.zeros((1, 7), dtype=token.dtype, device=token.device)
    pose[:, :3] = token[:, :3]
    pose[:, 3] = 1.0
    return pose


def _capture(torch: object, token: object, *, spatial_count: int = 768) -> EncoderGlobalPoseCapture:
    return EncoderGlobalPoseCapture(
        token=token,
        encoder_feature_shape=(1, spatial_count, 1024),
        encoder_global_shape=(1, 1, 1024),
        encoder_feature_dtype=str(token.dtype),
        encoder_feature_device=str(token.device),
    )


def _exporter(torch: object, *, head: object | None = None) -> EncoderGlobalPooledPoseExport:
    return EncoderGlobalPooledPoseExport(
        torch=torch,
        pose_head=head or (lambda token: _head(torch, token)),
        postprocess_pose=lambda pose, _mode: pose,
        pose_mode=("exp", -float("inf"), float("inf")),
        pose_encoding_to_camera=lambda pose: _camera(torch, pose),
    )


def test_v11_alarm_export_never_subscripts_raw_prediction_before_copying_output() -> None:
    module_path = inspect.getsourcefile(EncoderGlobalPooledPoseExport)
    assert module_path is not None
    module = ast.parse(Path(module_path).read_text(encoding="utf-8"))
    alarm = next(
        node for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "export_alarm"
    )
    prediction_loads = [
        node for node in ast.walk(alarm)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "prediction"
        and isinstance(node.ctx, ast.Load)
    ]
    assert prediction_loads == []
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in {"cpu", "numpy", "tolist"}
        for node in ast.walk(module)
    )


def test_v11_alarm_export_uses_captured_token_not_raw_camera_and_keeps_raw_intact() -> None:
    torch = _torch()
    exporter = _exporter(torch)
    raw_pose = torch.full((1, 7), 19.0, dtype=torch.float64)
    sentinel = object()
    prediction = {"camera_pose": raw_pose, "sentinel": sentinel}
    token = torch.zeros((1, 1, 768), dtype=torch.float64)
    token[:, :, :3] = torch.tensor((0.25, -0.5, 1.0), dtype=torch.float64)

    exported, action = exporter.export_alarm(0, prediction, _capture(torch, token), token_source=TOKEN_SOURCE)

    assert prediction["camera_pose"] is raw_pose
    torch.testing.assert_close(raw_pose, torch.full((1, 7), 19.0, dtype=torch.float64))
    assert exported is not prediction
    assert exported["sentinel"] is sentinel
    assert exported["camera_pose"] is not raw_pose
    torch.testing.assert_close(
        exported["camera_pose"][0, :3],
        torch.tensor((0.25, -0.5, 1.0), dtype=torch.float64),
    )
    assert action.action == "export_encoder_global_pose"


def test_v11_alarm_decode_failure_fails_closed_and_preserves_raw_prediction() -> None:
    torch = _torch()
    exporter = _exporter(torch, head=lambda token: torch.zeros((1, 8), dtype=token.dtype))
    raw_pose = torch.full((1, 7), 7.0)
    prediction = {"camera_pose": raw_pose}

    with pytest.raises(EncoderGlobalPooledPoseExportError, match="unavailable"):
        exporter.export_alarm(
            0,
            prediction,
            _capture(torch, torch.zeros((1, 1, 768), dtype=torch.float32)),
            token_source=TOKEN_SOURCE,
        )
    assert prediction["camera_pose"] is raw_pose
    torch.testing.assert_close(raw_pose, torch.full((1, 7), 7.0))


def test_v11_token_source_and_causal_frame_order_fail_closed() -> None:
    torch = _torch()
    prediction = {"camera_pose": torch.zeros((1, 7), dtype=torch.float32)}
    capture = _capture(torch, torch.zeros((1, 1, 768), dtype=torch.float32))
    with pytest.raises(EncoderGlobalPooledPoseExportError, match="source"):
        _exporter(torch).export_alarm(0, prediction, capture, token_source="untrusted-source")

    exporter = _exporter(torch)
    exporter.commit_real(0, prediction)
    with pytest.raises(EncoderGlobalPooledPoseExportError, match="contiguous and causal"):
        exporter.export_alarm(2, prediction, capture, token_source=TOKEN_SOURCE)


def test_v11_token_evidence_records_provenance_shape_grid_and_so3_without_token_values() -> None:
    torch = _torch()
    token = torch.zeros((1, 1, 768), dtype=torch.float64)
    token[:, :, :3] = torch.tensor((1.5, -2.0, 0.125), dtype=torch.float64)
    exported, action = _exporter(torch).export_alarm(
        0,
        {"camera_pose": torch.full((1, 7), 99.0, dtype=torch.float64)},
        _capture(torch, token, spatial_count=768),
        token_source=TOKEN_SOURCE,
    )
    evidence = action.evidence
    assert evidence is not None
    assert evidence["token_source"] == TOKEN_SOURCE
    assert evidence["encoder_feature_shape"] == [1, 768, 1024]
    assert evidence["encoder_spatial_token_count"] == 768
    assert evidence["encoder_global_shape"] == [1, 1, 1024]
    assert evidence["token_shape"] == [1, 1, 768]
    assert evidence["proper_rotation"] is True
    assert evidence["rotation_determinant"] == pytest.approx(1.0, abs=1e-8)
    assert evidence["orthonormality_max_abs_error"] == pytest.approx(0.0, abs=1e-8)
    assert "token_values" not in evidence
    assert all(value is not token for value in evidence.values())
    assert "1.5" not in json.dumps(evidence, sort_keys=True)
    torch.testing.assert_close(exported["camera_pose"][0, :3], torch.tensor((1.5, -2.0, 0.125), dtype=torch.float64))
