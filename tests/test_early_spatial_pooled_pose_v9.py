from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sys
import textwrap

import pytest

from stateguard3r.early_spatial_pooled_pose_export_v9 import (
    EarlySpatialPooledPoseExport,
    EarlySpatialPooledPoseExportError,
    TOKEN_SOURCE,
)
from stateguard3r.early_spatial_pooled_pose_v9 import (
    EarlySpatialPooledPoseError,
    POOLED_TOKEN_SHAPE,
    PREPROJECTION_SPATIAL_SHAPE,
    audit_pinned_early_spatial_pooled_pose_sources,
    decode_early_spatial_pooled_pose_token,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECAL3R_SRC = PROJECT_ROOT.parent / "baselines" / "ReCal3R" / "src"
RUNNER = PROJECT_ROOT / "src" / "stateguard3r" / "recal3r_early_spatial_pooled_pose_runner_v9.py"


def _torch() -> object:
    return pytest.importorskip("torch")


def _camera_from_pose(torch: object, pose: object) -> object:
    camera = torch.eye(4, dtype=pose.dtype, device=pose.device).unsqueeze(0)
    camera[:, :3, 3] = pose[:, :3]
    return camera


def _valid_head(torch: object, value: object) -> object:
    pose = torch.zeros((1, 7), dtype=value.dtype, device=value.device)
    pose[:, :3] = value[:, :3]
    pose[:, 3] = 1.0
    return pose


def test_v9_pool_is_all_576_preprojection_tokens_then_frozen_projection() -> None:
    torch = _torch()
    projection = torch.nn.Linear(1024, 768)
    spatial = torch.arange(576 * 1024, dtype=torch.float32).reshape(PREPROJECTION_SPATIAL_SHAPE) / 1024.0
    token = projection(spatial.mean(dim=1, keepdim=True))
    expected = projection(spatial).mean(dim=1, keepdim=True)
    assert tuple(token.shape) == POOLED_TOKEN_SHAPE
    # Both orders are affine-equivalent; float32 reduction order has bounded
    # rounding noise, so this checks the same frozen transform numerically.
    torch.testing.assert_close(token, expected, rtol=1e-5, atol=1e-4)


def test_v9_decoder_is_token_only_and_never_uses_cpu_transfer() -> None:
    source = inspect.getsource(decode_early_spatial_pooled_pose_token).lower()
    for forbidden in (
        "camera_pose",
        "prediction",
        "rgb",
        "pointmap",
        "confidence",
        "anchor",
        "history",
        "detector",
        "ground_truth",
        "future",
        ".cpu(",
        ".numpy(",
        ".tolist(",
    ):
        assert forbidden not in source
    node = ast.parse(inspect.getsource(decode_early_spatial_pooled_pose_token))
    assert not any(
        isinstance(item, ast.Attribute) and item.attr in {"cpu", "numpy", "tolist"}
        for item in ast.walk(node)
    )


def test_v9_decoder_keeps_head_dtype_and_validates_proper_camera() -> None:
    torch = _torch()
    token = torch.zeros(POOLED_TOKEN_SHAPE, dtype=torch.float32)
    token[:, :, :3] = torch.tensor((0.25, -0.5, 1.0), dtype=torch.float32)
    seen: dict[str, object] = {}

    def head(value: object) -> object:
        seen["head_dtype"] = value.dtype
        return _valid_head(torch, value)

    def postprocess(pose: object, _mode: object) -> object:
        seen["postprocess_dtype"] = pose.dtype
        return pose

    result = decode_early_spatial_pooled_pose_token(
        token,
        torch=torch,
        pose_head=head,
        postprocess_pose=postprocess,
        pose_mode=("exp", -float("inf"), float("inf")),
        pose_encoding_to_camera=lambda pose: _camera_from_pose(torch, pose),
    )
    assert seen == {"head_dtype": torch.float32, "postprocess_dtype": torch.float64}
    assert tuple(result.pose.shape) == (1, 7)
    assert result.rotation_determinant == pytest.approx(1.0, abs=1e-8)


def test_v9_decoder_rejects_unprojected_1024_wide_input_and_bad_camera() -> None:
    torch = _torch()
    raw = torch.zeros((1, 1, 1024), dtype=torch.float32)
    with pytest.raises(EarlySpatialPooledPoseError, match="shape"):
        decode_early_spatial_pooled_pose_token(
            raw,
            torch=torch,
            pose_head=lambda value: _valid_head(torch, value),
            postprocess_pose=lambda pose, _mode: pose,
            pose_mode=("exp", -float("inf"), float("inf")),
            pose_encoding_to_camera=lambda pose: _camera_from_pose(torch, pose),
        )

    def reflection(pose: object) -> object:
        camera = _camera_from_pose(torch, pose)
        camera[:, 0, 0] = -1.0
        return camera

    with pytest.raises(EarlySpatialPooledPoseError, match="proper homogeneous"):
        decode_early_spatial_pooled_pose_token(
            torch.zeros(POOLED_TOKEN_SHAPE, dtype=torch.float64),
            torch=torch,
            pose_head=lambda value: _valid_head(torch, value),
            postprocess_pose=lambda pose, _mode: pose,
            pose_mode=("exp", -float("inf"), float("inf")),
            pose_encoding_to_camera=reflection,
        )


def test_v9_export_alarm_uses_no_numeric_raw_prediction_pose_and_records_spatial_evidence() -> None:
    torch = _torch()
    policy = EarlySpatialPooledPoseExport(
        torch=torch, pose_head=lambda value: _valid_head(torch, value),
        postprocess_pose=lambda pose, _mode: pose,
        pose_mode=("exp", -float("inf"), float("inf")),
        pose_encoding_to_camera=lambda pose: _camera_from_pose(torch, pose),
    )
    raw_pose = torch.full((1, 7), 19.0, dtype=torch.float64)
    raw = {"camera_pose": raw_pose, "untouched": object()}
    token = torch.zeros(POOLED_TOKEN_SHAPE, dtype=torch.float64)
    token[:, :, :3] = torch.tensor((1.0, 2.0, 3.0), dtype=torch.float64)
    exported, action = policy.export_alarm(0, raw, token, token_source=TOKEN_SOURCE)
    assert exported is not raw and exported["untouched"] is raw["untouched"]
    torch.testing.assert_close(raw_pose, torch.full((1, 7), 19.0, dtype=torch.float64))
    torch.testing.assert_close(exported["camera_pose"][0, :3], torch.tensor((1.0, 2.0, 3.0), dtype=torch.float64))
    assert action.evidence is not None
    assert action.evidence["preprojection_spatial_token_count"] == 576
    assert action.evidence["aggregation"] == "mean"
    assert action.evidence["frozen_decoder_projection"] == "model.decoder_embed"
    source = inspect.getsource(EarlySpatialPooledPoseExport.export_alarm)
    tree = ast.parse(textwrap.dedent(source))
    assert not any(
        isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
        and node.value.id == "prediction" and isinstance(node.ctx, ast.Load)
        for node in ast.walk(tree)
    )
    with pytest.raises(EarlySpatialPooledPoseExportError, match="unavailable"):
        EarlySpatialPooledPoseExport(
            torch=torch, pose_head=lambda value: _valid_head(torch, value),
            postprocess_pose=lambda pose, _mode: pose,
            pose_mode=("exp", -float("inf"), float("inf")),
            pose_encoding_to_camera=lambda pose: _camera_from_pose(torch, pose),
        ).export_alarm(0, raw, torch.full(POOLED_TOKEN_SHAPE, float("nan")), token_source=TOKEN_SOURCE)


def test_v9_source_audit_binds_576_preprojection_spatial_mean_and_rejects_runner_mutation(tmp_path: Path) -> None:
    dust3r = RECAL3R_SRC / "dust3r"
    audit = audit_pinned_early_spatial_pooled_pose_sources(
        dust3r / "model.py", dust3r / "heads" / "dpt_head.py", dust3r / "heads" / "postprocess.py", RUNNER,
    )
    assert audit["spatial_token_count"] == 576
    assert audit["preprojection_shape"] == [1, 576, 1024]
    assert audit["pooled_shape"] == [1, 1, 768]
    assert audit["aggregation"] == "mean"
    original = RUNNER.read_text(encoding="utf-8")
    entry = original.index("\ndef run_early_spatial_pooled_pose_recurrent_lighter(")
    mutated = tmp_path / "special-token.py"
    mutated.write_text(
        original[:entry] + original[entry:].replace("dec[0].mean(dim=1, keepdim=True)", "dec[-1][:, 0:1]", 1),
        encoding="utf-8",
    )
    with pytest.raises(EarlySpatialPooledPoseError, match="order"):
        audit_pinned_early_spatial_pooled_pose_sources(
            dust3r / "model.py", dust3r / "heads" / "dpt_head.py", dust3r / "heads" / "postprocess.py", mutated,
        )


def test_checkpoint_loaded_official_v9_pose_head_obeys_float64_camera_contract() -> None:
    torch = _torch()
    baseline_root = PROJECT_ROOT.parent / "baselines" / "ReCal3R"
    checkpoint = baseline_root / "src" / "cut3r_512_dpt_4_64.pth"
    if str(baseline_root) not in sys.path:
        sys.path.insert(0, str(baseline_root))
    if str(RECAL3R_SRC) not in sys.path:
        sys.path.insert(0, str(RECAL3R_SRC))
    import add_ckpt_path
    add_ckpt_path.add_path_to_dust3r(str(checkpoint))
    import dust3r.heads.dpt_head as dpt_head
    import dust3r.heads.postprocess as postprocess
    import dust3r.model as dust3r_model
    from dust3r.utils.camera import pose_encoding_to_camera
    from scripts import run_recal3r_early_spatial_pooled_pose_export_v9 as script

    model, checkpoint_audit = script.smoke._load_model_with_state_dict_audit(dust3r_model.ARCroco3DStereo, checkpoint, torch)
    assert checkpoint_audit["strict"] is False and not checkpoint_audit["missing_keys"] and not checkpoint_audit["unexpected_keys"]
    pose_head, pose_mode = script._pinned_pose_head_interface(model, dpt_head)
    result = decode_early_spatial_pooled_pose_token(
        torch.zeros(POOLED_TOKEN_SHAPE, dtype=torch.float32), torch=torch,
        pose_head=pose_head, postprocess_pose=postprocess.postprocess_pose,
        pose_mode=pose_mode, pose_encoding_to_camera=pose_encoding_to_camera,
    )
    assert result.pose.dtype == torch.float64
    assert result.rotation_determinant == pytest.approx(1.0, abs=1e-8)
