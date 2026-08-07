from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from stateguard3r.early_spatial_pooled_pose_export_v9 import (
    EarlySpatialPooledPoseExport,
    EarlySpatialPooledPoseExportError,
    TOKEN_SOURCE,
)


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


def test_v9_alarm_export_never_loads_raw_camera_pose_value() -> None:
    module_path = inspect.getsourcefile(EarlySpatialPooledPoseExport)
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


def test_v9_alarm_export_uses_only_pooled_token_and_keeps_raw_input_unmutated() -> None:
    torch = _torch()
    exporter = EarlySpatialPooledPoseExport(
        torch=torch,
        pose_head=lambda token: _head(torch, token),
        postprocess_pose=lambda pose, _mode: pose,
        pose_mode=("exp", -float("inf"), float("inf")),
        pose_encoding_to_camera=lambda pose: _camera(torch, pose),
    )
    raw_pose = torch.full((1, 7), 9.0)
    prediction = {"camera_pose": raw_pose, "sentinel": object()}
    token = torch.zeros((1, 1, 768), dtype=torch.float32)
    token[:, :, :3] = torch.tensor((0.25, -0.5, 1.0))
    exported, action = exporter.export_alarm(0, prediction, token, token_source=TOKEN_SOURCE)
    assert prediction["camera_pose"] is raw_pose
    torch.testing.assert_close(raw_pose, torch.full((1, 7), 9.0))
    assert exported is not prediction
    assert exported["sentinel"] is prediction["sentinel"]
    assert exported["camera_pose"] is not raw_pose
    assert action.action == "export_early_spatial_pooled_pose"
    assert action.evidence is not None
    assert action.evidence["token_source"] == TOKEN_SOURCE
    assert action.evidence["preprojection_spatial_token_count"] == 576


def test_v9_alarm_decode_failure_fails_closed_without_raw_fallback_or_mutation() -> None:
    torch = _torch()
    exporter = EarlySpatialPooledPoseExport(
        torch=torch,
        pose_head=lambda token: torch.zeros((1, 8), dtype=token.dtype),
        postprocess_pose=lambda pose, _mode: pose,
        pose_mode=("exp", -float("inf"), float("inf")),
        pose_encoding_to_camera=lambda pose: _camera(torch, pose),
    )
    raw_pose = torch.full((1, 7), 7.0)
    prediction = {"camera_pose": raw_pose}
    with pytest.raises(EarlySpatialPooledPoseExportError, match="unavailable"):
        exporter.export_alarm(
            0,
            prediction,
            torch.zeros((1, 1, 768), dtype=torch.float32),
            token_source=TOKEN_SOURCE,
        )
    assert prediction["camera_pose"] is raw_pose
    torch.testing.assert_close(raw_pose, torch.full((1, 7), 7.0))


def test_v9_clear_export_is_raw_and_alarm_token_source_is_pinned() -> None:
    torch = _torch()
    exporter = EarlySpatialPooledPoseExport(
        torch=torch,
        pose_head=lambda token: _head(torch, token),
        postprocess_pose=lambda pose, _mode: pose,
        pose_mode=("exp", -float("inf"), float("inf")),
        pose_encoding_to_camera=lambda pose: _camera(torch, pose),
    )
    prediction = {"camera_pose": torch.zeros((1, 7))}
    exported, action = exporter.commit_real(0, prediction)
    assert exported is prediction
    assert action.action == "export_real_camera_pose"
    with pytest.raises(EarlySpatialPooledPoseExportError, match="source"):
        exporter.export_alarm(
            1,
            prediction,
            torch.zeros((1, 1, 768), dtype=torch.float32),
            token_source="other",
        )
