from __future__ import annotations

import ast
import inspect
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from stateguard3r.patch_embed_global_pooled_pose_export_v12 import (
    TOKEN_SOURCE,
    PatchEmbedGlobalPooledPoseExport,
    PatchEmbedGlobalPooledPoseExportError,
    tensor_gpu_digest,
)
from stateguard3r.patch_embed_global_pooled_pose_v12 import PatchEmbedGlobalPoseCapture


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


def _capture(torch: object, token: object, *, patch_count: int = 5) -> PatchEmbedGlobalPoseCapture:
    return PatchEmbedGlobalPoseCapture(
        token=token,
        patch_feature_shape=(1, patch_count, 1024),
        patch_global_shape=(1, 1, 1024),
        patch_feature_dtype=str(token.dtype),
        patch_feature_device=str(token.device),
    )


def _exporter(torch: object, *, head: object | None = None) -> PatchEmbedGlobalPooledPoseExport:
    return PatchEmbedGlobalPooledPoseExport(
        torch=torch,
        pose_head=head or (lambda token: _head(torch, token)),
        postprocess_pose=lambda pose, _mode: pose,
        pose_mode=("exp", -float("inf"), float("inf")),
        pose_encoding_to_camera=lambda pose: _camera(torch, pose),
    )


class _CameraPoseReadForbiddenMapping(Mapping[str, Any]):
    """A mapping that proves copying does not load the raw camera pose value."""

    def __init__(self, camera_pose: object, **other: object) -> None:
        self._camera_pose = camera_pose
        self._other = dict(other)

    def __getitem__(self, key: str) -> object:
        if key == "camera_pose":
            raise AssertionError("v12 export must not read raw prediction['camera_pose']")
        return self._other[key]

    def __iter__(self) -> Iterator[str]:
        yield "camera_pose"
        yield from self._other

    def __len__(self) -> int:
        return len(self._other) + 1


def test_v12_alarm_export_never_reads_raw_camera_pose_before_copying_output() -> None:
    module_path = inspect.getsourcefile(PatchEmbedGlobalPooledPoseExport)
    assert module_path is not None
    module = ast.parse(Path(module_path).read_text(encoding="utf-8"))
    alarm = next(
        node for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "export_alarm"
    )
    raw_camera_pose_loads = [
        node for node in ast.walk(alarm)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "prediction"
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "camera_pose"
        and isinstance(node.ctx, ast.Load)
    ]
    assert raw_camera_pose_loads == []
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "prediction"
        for node in ast.walk(alarm)
    )
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in {"cpu", "numpy", "tolist"}
        for node in ast.walk(module)
    )


def test_v12_alarm_export_uses_capture_only_and_preserves_non_pose_mapping_entries() -> None:
    torch = _torch()
    raw_pose = torch.full((1, 7), 19.0, dtype=torch.float64)
    sentinel = object()
    prediction = _CameraPoseReadForbiddenMapping(raw_pose, sentinel=sentinel, health="raw-health")
    token = torch.zeros((1, 1, 768), dtype=torch.float64)
    token[:, :, :3] = torch.tensor((0.25, -0.5, 1.0), dtype=torch.float64)

    exported, action = _exporter(torch).export_alarm(
        0,
        prediction,
        _capture(torch, token, patch_count=5),
        token_source=TOKEN_SOURCE,
    )

    torch.testing.assert_close(raw_pose, torch.full((1, 7), 19.0, dtype=torch.float64))
    assert exported["sentinel"] is sentinel
    assert exported["health"] == "raw-health"
    assert exported["camera_pose"] is not raw_pose
    torch.testing.assert_close(
        exported["camera_pose"][0, :3],
        torch.tensor((0.25, -0.5, 1.0), dtype=torch.float64),
    )
    assert action.action == "export_patch_embed_global_pose"


def test_v12_alarm_decode_failure_fails_closed_without_raw_or_fallback() -> None:
    torch = _torch()
    raw_pose = torch.full((1, 7), 7.0)
    prediction = _CameraPoseReadForbiddenMapping(raw_pose, sentinel=object())
    with pytest.raises(PatchEmbedGlobalPooledPoseExportError, match="unavailable"):
        _exporter(torch, head=lambda token: torch.zeros((1, 8), dtype=token.dtype)).export_alarm(
            0,
            prediction,
            _capture(torch, torch.zeros((1, 1, 768), dtype=torch.float32)),
            token_source=TOKEN_SOURCE,
        )
    torch.testing.assert_close(raw_pose, torch.full((1, 7), 7.0))


def test_v12_clear_path_is_original_raw_mapping_and_alarm_source_and_order_fail_closed() -> None:
    torch = _torch()
    prediction = {"camera_pose": torch.zeros((1, 7), dtype=torch.float32)}
    capture = _capture(torch, torch.zeros((1, 1, 768), dtype=torch.float32))
    exporter = _exporter(torch)
    exported, action = exporter.commit_real(0, prediction)
    assert exported is prediction
    assert action.action == "export_real_camera_pose"
    assert action.evidence is None
    with pytest.raises(PatchEmbedGlobalPooledPoseExportError, match="source"):
        _exporter(torch).export_alarm(0, prediction, capture, token_source="untrusted-source")
    with pytest.raises(PatchEmbedGlobalPooledPoseExportError, match="contiguous and causal"):
        exporter.export_alarm(2, prediction, capture, token_source=TOKEN_SOURCE)


def test_v12_evidence_has_non_sensitive_patch_provenance_and_no_token_values() -> None:
    torch = _torch()
    raw_pose = torch.full((1, 7), 99.0, dtype=torch.float64)
    token = torch.zeros((1, 1, 768), dtype=torch.float64)
    token[:, :, :3] = torch.tensor((1.5, -2.0, 0.125), dtype=torch.float64)
    exported, action = _exporter(torch).export_alarm(
        0,
        {"camera_pose": raw_pose},
        _capture(torch, token, patch_count=768),
        token_source=TOKEN_SOURCE,
    )
    evidence = action.evidence
    assert evidence is not None
    assert evidence["token_source"] == TOKEN_SOURCE
    assert evidence["patch_feature_shape"] == [1, 768, 1024]
    assert evidence["patch_spatial_token_count"] == 768
    assert evidence["patch_global_shape"] == [1, 1, 1024]
    assert evidence["patch_feature_dtype"] == str(torch.float64)
    assert evidence["patch_feature_device"] == "cpu"
    assert evidence["token_shape"] == [1, 1, 768]
    assert evidence["proper_rotation"] is True
    assert evidence["fallback_used"] is False
    assert evidence["no_fallback"] is True
    assert evidence["rotation_determinant"] == pytest.approx(1.0, abs=1e-8)
    assert evidence["orthonormality_max_abs_error"] == pytest.approx(0.0, abs=1e-8)
    assert evidence["homogeneous_max_abs_error"] == pytest.approx(0.0, abs=1e-8)
    assert evidence["token_gpu_digest"].startswith("gpu-fingerprint-v1:")
    assert evidence["exported_pose_gpu_digest"].startswith("gpu-fingerprint-v1:")
    assert "token_values" not in evidence
    assert all(value is not token and value is not raw_pose for value in evidence.values())
    serialized = json.dumps(evidence, sort_keys=True)
    for secret in ("1.5", "-2.0", "0.125", "99.0"):
        assert secret not in serialized
    torch.testing.assert_close(
        exported["camera_pose"][0, :3],
        torch.tensor((1.5, -2.0, 0.125), dtype=torch.float64),
    )


def test_v12_tensor_digest_is_deterministic_and_rejects_nonfinite_values() -> None:
    torch = _torch()
    token = torch.arange(768, dtype=torch.float64).reshape(1, 1, 768)
    assert tensor_gpu_digest(token, torch=torch) == tensor_gpu_digest(token.clone(), torch=torch)
    changed = token.clone()
    changed[:, :, 7] += 1.0
    assert tensor_gpu_digest(changed, torch=torch) != tensor_gpu_digest(token, torch=torch)
    with pytest.raises(PatchEmbedGlobalPooledPoseExportError, match="unavailable"):
        tensor_gpu_digest(torch.full((1, 1, 768), float("nan")), torch=torch)
