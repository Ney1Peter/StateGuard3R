from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from stateguard3r.patch_embed_global_pooled_pose_v12 import (
    PATCH_FEATURE_WIDTH,
    PATCH_GLOBAL_SHAPE,
    PROJECTED_TOKEN_SHAPE,
    PatchEmbedGlobalPooledPoseError,
    audit_pinned_patch_embed_global_pose_sources,
    capture_patch_embed_global_pose_token,
    decode_patch_embed_global_pose_token,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECAL3R_SRC = PROJECT_ROOT.parent / "baselines" / "ReCal3R" / "src"
RUNNER = PROJECT_ROOT / "src" / "stateguard3r" / "recal3r_patch_embed_global_pooled_pose_runner_v12.py"


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
    return decode_patch_embed_global_pose_token(
        token,
        torch=torch,
        pose_head=lambda value: _head(torch, value),
        postprocess_pose=lambda pose, _mode: pose,
        pose_mode=("exp", -float("inf"), float("inf")),
        pose_encoding_to_camera=camera_decode or (lambda pose: _camera(torch, pose)),
    )


@pytest.mark.parametrize("patch_count", (5, 768))
@pytest.mark.parametrize("dtype_name", ("float32", "float64"))
def test_v12_capture_mean_projects_each_positive_runtime_patch_count(
    patch_count: int, dtype_name: str,
) -> None:
    torch = _torch()
    dtype = getattr(torch, dtype_name)
    projection = torch.nn.Linear(PATCH_FEATURE_WIDTH, 768, dtype=dtype)
    patch_feature = torch.randn((1, patch_count, PATCH_FEATURE_WIDTH), dtype=dtype, requires_grad=True)

    capture = capture_patch_embed_global_pose_token(
        patch_feature,
        projection=projection,
        torch=torch,
    )

    assert tuple(capture.token.shape) == PROJECTED_TOKEN_SHAPE
    assert capture.patch_feature_shape == (1, patch_count, PATCH_FEATURE_WIDTH)
    assert capture.patch_global_shape == PATCH_GLOBAL_SHAPE
    assert capture.patch_feature_dtype == str(dtype)
    assert capture.patch_feature_device == "cpu"
    assert capture.token.requires_grad is False
    torch.testing.assert_close(capture.token, projection(patch_feature.mean(dim=1, keepdim=True)))


def test_v12_capture_rejects_invalid_patch_feature_and_frozen_projection() -> None:
    torch = _torch()
    projection = torch.nn.Linear(PATCH_FEATURE_WIDTH, 768)
    good = torch.zeros((1, 5, PATCH_FEATURE_WIDTH), dtype=torch.float32)

    for malformed in (
        torch.zeros((1, 0, PATCH_FEATURE_WIDTH), dtype=torch.float32),
        torch.zeros((2, 5, PATCH_FEATURE_WIDTH), dtype=torch.float32),
        torch.zeros((1, 5, 768), dtype=torch.float32),
        torch.zeros((1, 5, PATCH_FEATURE_WIDTH), dtype=torch.int64),
        torch.full((1, 5, PATCH_FEATURE_WIDTH), float("nan"), dtype=torch.float32),
        torch.full((1, 5, PATCH_FEATURE_WIDTH), float("inf"), dtype=torch.float32),
    ):
        with pytest.raises(PatchEmbedGlobalPooledPoseError, match="patch feature"):
            capture_patch_embed_global_pose_token(malformed, projection=projection, torch=torch)

    for malformed_projection in (
        object(),
        torch.nn.Linear(PATCH_FEATURE_WIDTH, 767),
        torch.nn.Linear(768, 768),
        torch.nn.Sequential(projection),
    ):
        with pytest.raises(PatchEmbedGlobalPooledPoseError, match="projection"):
            capture_patch_embed_global_pose_token(good, projection=malformed_projection, torch=torch)

    with pytest.raises(PatchEmbedGlobalPooledPoseError, match="device or dtype"):
        capture_patch_embed_global_pose_token(
            good,
            projection=torch.nn.Linear(PATCH_FEATURE_WIDTH, 768, dtype=torch.float64),
            torch=torch,
        )
    with pytest.raises(PatchEmbedGlobalPooledPoseError, match="device or dtype"):
        capture_patch_embed_global_pose_token(
            good,
            projection=torch.nn.Linear(PATCH_FEATURE_WIDTH, 768, device="meta"),
            torch=torch,
        )

    nonfinite_projection = torch.nn.Linear(PATCH_FEATURE_WIDTH, 768)
    with torch.no_grad():
        nonfinite_projection.weight.fill_(float("nan"))
    with pytest.raises(PatchEmbedGlobalPooledPoseError, match="projected token"):
        capture_patch_embed_global_pose_token(good, projection=nonfinite_projection, torch=torch)


def test_v12_capture_uses_one_patch_axis_mean_without_fixed_grid_gate() -> None:
    torch = _torch()
    source = inspect.getsource(capture_patch_embed_global_pose_token)
    tree = ast.parse(source)
    mean_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "mean"
    ]
    assert len(mean_calls) == 1
    mean = mean_calls[0]
    keywords = {keyword.arg: keyword.value for keyword in mean.keywords}
    assert isinstance(keywords.get("dim"), ast.Constant) and keywords["dim"].value == 1
    assert isinstance(keywords.get("keepdim"), ast.Constant) and keywords["keepdim"].value is True
    assert "576" not in source

    numeric_shape_comparisons = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(
            isinstance(child, ast.Subscript)
            and isinstance(child.value, ast.Name)
            and child.value.id == "shape"
            and isinstance(child.slice, ast.Constant)
            and child.slice.value == 1
            for child in ast.walk(node.left)
        )
        and any(isinstance(comparator, ast.Constant) and isinstance(comparator.value, int) for comparator in node.comparators)
    ]
    assert len(numeric_shape_comparisons) == 1
    assert isinstance(numeric_shape_comparisons[0].comparators[0], ast.Constant)
    assert numeric_shape_comparisons[0].comparators[0].value == 0


def test_v12_decoder_is_token_only_uses_no_cpu_or_raw_inputs_and_promotes_only_pose() -> None:
    torch = _torch()
    token = torch.zeros(PROJECTED_TOKEN_SHAPE, dtype=torch.float32)
    seen: dict[str, object] = {}

    def head(value: object) -> object:
        seen["head_shape"] = tuple(value.shape)
        seen["head_dtype"] = value.dtype
        return _head(torch, value)

    def postprocess(pose: object, _mode: object) -> object:
        seen["postprocess_dtype"] = pose.dtype
        return pose

    result = decode_patch_embed_global_pose_token(
        token,
        torch=torch,
        pose_head=head,
        postprocess_pose=postprocess,
        pose_mode=("exp", -float("inf"), float("inf")),
        pose_encoding_to_camera=lambda pose: _camera(torch, pose),
    )
    assert seen == {"head_shape": (1, 768), "head_dtype": torch.float32, "postprocess_dtype": torch.float64}
    assert tuple(result.pose.shape) == (1, 7)
    assert tuple(result.camera.shape) == (1, 4, 4)
    assert result.rotation_determinant == pytest.approx(1.0, abs=1e-8)

    signature = inspect.signature(decode_patch_embed_global_pose_token)
    assert tuple(signature.parameters) == (
        "token", "torch", "pose_head", "postprocess_pose", "pose_mode", "pose_encoding_to_camera",
    )
    source = inspect.getsource(decode_patch_embed_global_pose_token).lower()
    for forbidden in (
        "camera_pose", "prediction", "rgb", "image", "pointmap", "confidence", "anchor",
        "history", "detector", "ground_truth", "future", ".cpu(", ".numpy(", ".tolist(",
    ):
        assert forbidden not in source
    tree = ast.parse(inspect.getsource(decode_patch_embed_global_pose_token))
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in {"cpu", "numpy", "tolist"}
        for node in ast.walk(tree)
    )


def test_v12_decoder_rejects_bad_token_head_and_non_so3_camera() -> None:
    torch = _torch()
    for malformed in (
        torch.zeros((1, 1, PATCH_FEATURE_WIDTH), dtype=torch.float32),
        torch.full(PROJECTED_TOKEN_SHAPE, float("nan"), dtype=torch.float32),
        torch.full(PROJECTED_TOKEN_SHAPE, float("inf"), dtype=torch.float32),
    ):
        with pytest.raises(PatchEmbedGlobalPooledPoseError):
            _decode(torch, malformed)

    with pytest.raises(PatchEmbedGlobalPooledPoseError, match="malformed output"):
        decode_patch_embed_global_pose_token(
            torch.zeros(PROJECTED_TOKEN_SHAPE),
            torch=torch,
            pose_head=lambda value: torch.zeros((1, 8), dtype=value.dtype),
            postprocess_pose=lambda pose, _mode: pose,
            pose_mode=("exp", -float("inf"), float("inf")),
            pose_encoding_to_camera=lambda pose: _camera(torch, pose),
        )

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

    for camera_decode in (reflection, non_so3, bad_homogeneous):
        with pytest.raises(PatchEmbedGlobalPooledPoseError, match="proper homogeneous"):
            _decode(torch, torch.zeros(PROJECTED_TOKEN_SHAPE, dtype=torch.float64), camera_decode=camera_decode)


def test_v12_source_audit_pins_patch_embed_checkpoint_rewrite_and_rejects_mutants(tmp_path: Path) -> None:
    dust3r = RECAL3R_SRC / "dust3r"
    audit = audit_pinned_patch_embed_global_pose_sources(
        dust3r / "model.py",
        dust3r / "heads" / "dpt_head.py",
        dust3r / "heads" / "postprocess.py",
        dust3r / "patch_embed.py",
        dust3r / "utils" / "image.py",
        RUNNER,
    )
    assert audit["patch_feature_source"] == "native_model_patch_embed_before_encoder_blocks_and_rope"
    assert audit["patch_global_shape"] == [1, 1, PATCH_FEATURE_WIDTH]
    assert audit["projected_shape"] == [1, 1, 768]
    assert audit["projection_interface"] == "torch.nn.Linear(1024,768): model.decoder_embed"
    assert audit["actual_patch_token_count_is_runtime_evidence"] is True
    assert audit["raw_camera_pose_numeric_input_to_v12_decoder"] is False

    mutated_model = tmp_path / "model-mutated.py"
    mutated_model.write_text(
        (dust3r / "model.py").read_text(encoding="utf-8").replace(
            '"ManyAR_PatchEmbed", "PatchEmbedDust3R"',
            '"ManyAR_PatchEmbed", "ManyAR_PatchEmbed"',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(PatchEmbedGlobalPooledPoseError, match="topology"):
        audit_pinned_patch_embed_global_pose_sources(
            mutated_model,
            dust3r / "heads" / "dpt_head.py",
            dust3r / "heads" / "postprocess.py",
            dust3r / "patch_embed.py",
            dust3r / "utils" / "image.py",
            RUNNER,
        )

    original_runner = RUNNER.read_text(encoding="utf-8")
    entry = original_runner.index("\ndef run_patch_embed_global_pooled_pose_recurrent_lighter(")
    mutated_runner = tmp_path / "runner-mutated.py"
    mutated_runner.write_text(
        original_runner[:entry] + original_runner[entry:].replace(
            "patch_tokens_i, _patch_pos_i = model.patch_embed(selected_imgs, true_shape=selected_shapes)",
            "patch_tokens_i, _patch_pos_i = model._encode_image(selected_imgs, selected_shapes)",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(PatchEmbedGlobalPooledPoseError, match="order"):
        audit_pinned_patch_embed_global_pose_sources(
            dust3r / "model.py",
            dust3r / "heads" / "dpt_head.py",
            dust3r / "heads" / "postprocess.py",
            dust3r / "patch_embed.py",
            dust3r / "utils" / "image.py",
            mutated_runner,
        )
