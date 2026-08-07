from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from stateguard3r.encoder_global_pooled_pose_v11 import (
    ENCODER_GLOBAL_FEATURE_SHAPE,
    PROJECTED_TOKEN_SHAPE,
    EncoderGlobalPooledPoseError,
    audit_pinned_encoder_global_pose_sources,
    capture_encoder_global_pose_token,
    decode_encoder_global_pose_token,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECAL3R_SRC = PROJECT_ROOT.parent / "baselines" / "ReCal3R" / "src"
RUNNER = PROJECT_ROOT / "src" / "stateguard3r" / "recal3r_encoder_global_pooled_pose_runner_v11.py"


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


def _decode(torch: object, token: object, *, camera_decode: object | None = None) -> object:
    return decode_encoder_global_pose_token(
        token,
        torch=torch,
        pose_head=lambda value: _head(torch, value),
        postprocess_pose=lambda pose, _mode: pose,
        pose_mode=("exp", -float("inf"), float("inf")),
        pose_encoding_to_camera=camera_decode or (lambda pose: _camera(torch, pose)),
    )


def test_v11_capture_projects_only_a_finite_encoder_global_feature() -> None:
    torch = _torch()
    projection = torch.nn.Linear(1024, 768)
    global_feature = torch.randn(ENCODER_GLOBAL_FEATURE_SHAPE, dtype=torch.float32)

    capture = capture_encoder_global_pose_token(
        global_feature,
        encoder_feature_shape=(1, 768, 1024),
        projection=projection,
        torch=torch,
    )

    assert tuple(capture.token.shape) == PROJECTED_TOKEN_SHAPE
    assert capture.encoder_feature_shape == (1, 768, 1024)
    assert capture.encoder_global_shape == ENCODER_GLOBAL_FEATURE_SHAPE
    assert capture.encoder_feature_dtype == str(torch.float32)
    assert capture.encoder_feature_device == "cpu"
    assert capture.token.requires_grad is False
    torch.testing.assert_close(capture.token, projection(global_feature))


def test_v11_capture_rejects_malformed_shape_dtype_finiteness_projection_and_device() -> None:
    torch = _torch()
    projection = torch.nn.Linear(1024, 768)
    good = torch.zeros(ENCODER_GLOBAL_FEATURE_SHAPE, dtype=torch.float32)

    for malformed in (
        torch.zeros((1, 2, 1024), dtype=torch.float32),
        torch.zeros(ENCODER_GLOBAL_FEATURE_SHAPE, dtype=torch.int64),
        torch.full(ENCODER_GLOBAL_FEATURE_SHAPE, float("nan"), dtype=torch.float32),
    ):
        with pytest.raises(EncoderGlobalPooledPoseError):
            capture_encoder_global_pose_token(
                malformed,
                encoder_feature_shape=(1, 768, 1024),
                projection=projection,
                torch=torch,
            )

    for malformed_shape in ((1, 0, 1024), (1, 768, 768), (2, 768, 1024)):
        with pytest.raises(EncoderGlobalPooledPoseError, match="feature shape"):
            capture_encoder_global_pose_token(
                good,
                encoder_feature_shape=malformed_shape,
                projection=projection,
                torch=torch,
            )

    for malformed_projection in (object(), torch.nn.Linear(1024, 767), torch.nn.Sequential(projection)):
        with pytest.raises(EncoderGlobalPooledPoseError, match="projection"):
            capture_encoder_global_pose_token(
                good,
                encoder_feature_shape=(1, 768, 1024),
                projection=malformed_projection,
                torch=torch,
            )

    meta_projection = torch.nn.Linear(1024, 768, device="meta")
    with pytest.raises(EncoderGlobalPooledPoseError, match="device or dtype"):
        capture_encoder_global_pose_token(
            good,
            encoder_feature_shape=(1, 768, 1024),
            projection=meta_projection,
            torch=torch,
        )
    with pytest.raises(EncoderGlobalPooledPoseError, match="device or dtype"):
        capture_encoder_global_pose_token(
            good,
            encoder_feature_shape=(1, 768, 1024),
            projection=torch.nn.Linear(1024, 768, dtype=torch.float64),
            torch=torch,
        )


def test_v11_decoder_is_token_only_and_promotes_only_pose_postprocess_input() -> None:
    torch = _torch()
    token = torch.zeros(PROJECTED_TOKEN_SHAPE, dtype=torch.float32)
    seen: dict[str, object] = {}

    def head(value: object) -> object:
        seen["head_dtype"] = value.dtype
        return _head(torch, value)

    def postprocess(pose: object, _mode: object) -> object:
        seen["postprocess_dtype"] = pose.dtype
        return pose

    result = decode_encoder_global_pose_token(
        token,
        torch=torch,
        pose_head=head,
        postprocess_pose=postprocess,
        pose_mode=("exp", -float("inf"), float("inf")),
        pose_encoding_to_camera=lambda pose: _camera(torch, pose),
    )
    assert seen == {"head_dtype": torch.float32, "postprocess_dtype": torch.float64}
    assert tuple(result.pose.shape) == (1, 7)
    assert result.rotation_determinant == pytest.approx(1.0, abs=1e-8)

    source = inspect.getsource(decode_encoder_global_pose_token).lower()
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
    tree = ast.parse(inspect.getsource(decode_encoder_global_pose_token))
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in {"cpu", "numpy", "tolist"}
        for node in ast.walk(tree)
    )


def test_v11_decoder_rejects_malformed_nonfinite_token_and_invalid_camera() -> None:
    torch = _torch()
    for malformed in (
        torch.zeros((1, 1, 1024), dtype=torch.float32),
        torch.full(PROJECTED_TOKEN_SHAPE, float("nan"), dtype=torch.float32),
    ):
        with pytest.raises(EncoderGlobalPooledPoseError):
            _decode(torch, malformed)

    def reflection(pose: object) -> object:
        camera = _camera(torch, pose)
        camera[:, 0, 0] = -1.0
        return camera

    def non_so3(pose: object) -> object:
        camera = _camera(torch, pose)
        camera[:, 0, 1] = 0.25
        return camera

    def bad_homogeneous(pose: object) -> object:
        camera = _camera(torch, pose)
        camera[:, 3, 3] = 0.0
        return camera

    for bad_camera in (reflection, non_so3, bad_homogeneous):
        with pytest.raises(EncoderGlobalPooledPoseError, match="proper homogeneous"):
            _decode(torch, torch.zeros(PROJECTED_TOKEN_SHAPE, dtype=torch.float64), camera_decode=bad_camera)


def test_v11_source_audit_pins_native_pre_retrieval_boundary_and_rejects_mutations(tmp_path: Path) -> None:
    dust3r = RECAL3R_SRC / "dust3r"
    audit = audit_pinned_encoder_global_pose_sources(
        dust3r / "model.py",
        dust3r / "heads" / "dpt_head.py",
        dust3r / "heads" / "postprocess.py",
        dust3r / "patch_embed.py",
        dust3r / "utils" / "image.py",
        RUNNER,
    )
    assert audit["encoder_global_source"] == "native_encoder_global_mean_after_feature_selection_before_pose_retriever"
    assert audit["encoder_global_shape"] == [1, 1, 1024]
    assert audit["projected_shape"] == [1, 1, 768]
    assert audit["actual_spatial_token_count_is_runtime_evidence"] is True
    assert audit["raw_camera_pose_numeric_input_to_v11_decoder"] is False

    loader_mutation = tmp_path / "image-mutated.py"
    loader_mutation.write_text(
        (dust3r / "utils" / "image.py").read_text(encoding="utf-8").replace(
            "# resize long side to 512", "# resize long side to 513", 1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(EncoderGlobalPooledPoseError, match="topology"):
        audit_pinned_encoder_global_pose_sources(
            dust3r / "model.py",
            dust3r / "heads" / "dpt_head.py",
            dust3r / "heads" / "postprocess.py",
            dust3r / "patch_embed.py",
            loader_mutation,
            RUNNER,
        )

    original_runner = RUNNER.read_text(encoding="utf-8")
    entry = original_runner.index("\ndef run_encoder_global_pooled_pose_recurrent_lighter(")
    runner_mutation = tmp_path / "runner-mutated.py"
    runner_mutation.write_text(
        original_runner[:entry] + original_runner[entry:].replace(
            "model.pose_retriever.inquire(global_img_feat_i, mem)",
            "model.pose_retriever.inquire(global_img_feat_i, altered_mem)",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(EncoderGlobalPooledPoseError, match="order"):
        audit_pinned_encoder_global_pose_sources(
            dust3r / "model.py",
            dust3r / "heads" / "dpt_head.py",
            dust3r / "heads" / "postprocess.py",
            dust3r / "patch_embed.py",
            dust3r / "utils" / "image.py",
            runner_mutation,
        )
