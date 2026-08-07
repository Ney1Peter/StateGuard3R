from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sys

import pytest

from stateguard3r.prerollout_pose_query_export_v6 import (
    PreRolloutPoseQueryExport,
    PreRolloutPoseQueryExportError,
)
from stateguard3r.prerollout_pose_query_v6 import (
    PreRolloutPoseQueryError,
    ROTATION_ATOL,
    audit_pinned_pose_query_sources,
    decode_pre_rollout_pose_query,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECAL3R_SRC = PROJECT_ROOT.parent / "baselines" / "ReCal3R" / "src"


def _function_node(callable_object: object) -> ast.FunctionDef:
    module = ast.parse(Path(inspect.getsourcefile(callable_object) or "").read_text(encoding="utf-8"))
    function_name = getattr(callable_object, "__name__")
    return next(node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == function_name)


def test_v6_decoder_signature_and_ast_are_token_only() -> None:
    signature = inspect.signature(decode_pre_rollout_pose_query)
    assert tuple(signature.parameters) == (
        "query",
        "torch",
        "pose_head",
        "postprocess_pose",
        "pose_mode",
        "pose_encoding_to_camera",
    )
    node = _function_node(decode_pre_rollout_pose_query)
    source = ast.unparse(node).lower()
    for forbidden in (
        "camera_pose",
        "prediction",
        "rgb",
        "pointmap",
        "pts3d",
        "conf",
        "anchor",
        "history",
        "detector",
        "ground_truth",
        "future",
    ):
        assert forbidden not in source
    assert "raw.to(dtype=torch.float64)" in source
    assert not any(
        isinstance(item, ast.Attribute) and item.attr in {"cpu", "numpy", "tolist"}
        for item in ast.walk(node)
    )


def test_alarm_export_never_loads_raw_camera_pose() -> None:
    module_path = inspect.getsourcefile(PreRolloutPoseQueryExport)
    assert module_path is not None
    node = next(
        item
        for item in ast.walk(ast.parse(Path(module_path).read_text(encoding="utf-8")))
        if isinstance(item, ast.FunctionDef) and item.name == "export_alarm"
    )
    prediction_loads = [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Subscript)
        and isinstance(item.value, ast.Name)
        and item.value.id == "prediction"
        and isinstance(item.ctx, ast.Load)
    ]
    assert prediction_loads == []


def test_v6_runner_captures_query_before_rollout_without_prior_recovery_delegation() -> None:
    runner_path = PROJECT_ROOT / "src" / "stateguard3r" / "recal3r_prerollout_pose_query_runner_v6.py"
    source = runner_path.read_text(encoding="utf-8")
    capture = source.index("pre_rollout_pose_query = pose_feat_i.detach().clone()")
    rollout = source.index("model._recurrent_rollout(", capture)
    memory_update = source.index("model.pose_retriever.update_mem(", rollout)
    assert capture < rollout < memory_update
    for forbidden in (
        "recal3r_safe_anchor_runner_v3",
        "recal3r_geometric_registration_runner_v4",
        "recal3r_current_pointmap_runner_v5",
        "run_current_pointmap_recurrent_lighter",
        "raw_prediction = to_cpu(res)",
    ):
        assert forbidden not in source


def test_source_audit_binds_the_pinned_pre_rollout_topology(tmp_path: Path) -> None:
    dust3r = RECAL3R_SRC / "dust3r"
    audit = audit_pinned_pose_query_sources(
        dust3r / "model.py", dust3r / "heads" / "dpt_head.py", dust3r / "heads" / "postprocess.py"
    )
    assert audit["raw_camera_pose_numeric_input_to_v6_decoder"] is False
    assert audit["current_image_feature_in_query"] is True
    assert len(audit["model_sha256"]) == len(audit["dpt_head_sha256"]) == len(audit["postprocess_sha256"]) == 64

    changed_model = tmp_path / "model.py"
    changed_model.write_text(
        (dust3r / "model.py").read_text(encoding="utf-8").replace(
            "pose_retriever.inquire",
            "pose_retriever.invalid",
        ),
        encoding="utf-8",
    )
    with pytest.raises(PreRolloutPoseQueryError, match="topology"):
        audit_pinned_pose_query_sources(changed_model, dust3r / "heads" / "dpt_head.py", dust3r / "heads" / "postprocess.py")


def _torch() -> object:
    return pytest.importorskip("torch")


def _camera_from_pose(torch: object, pose: object) -> object:
    camera = torch.eye(4, dtype=pose.dtype, device=pose.device).unsqueeze(0)
    camera[:, :3, 3] = pose[:, :3]
    return camera


def _valid_pose_head(torch: object, token: object) -> object:
    pose = torch.zeros((1, 7), dtype=token.dtype, device=token.device)
    pose[:, :3] = token[:, :3]
    pose[:, 3] = 1.0
    return pose


def test_decoder_accepts_only_a_finite_pinned_query_and_validates_camera() -> None:
    torch = _torch()
    query = torch.zeros((1, 1, 768), dtype=torch.float64)
    query[:, :, :3] = torch.tensor((0.25, -0.5, 1.0), dtype=torch.float64)
    result = decode_pre_rollout_pose_query(
        query,
        torch=torch,
        pose_head=lambda token: _valid_pose_head(torch, token),
        postprocess_pose=lambda pose, _mode: pose,
        pose_mode=("exp", -float("inf"), float("inf")),
        pose_encoding_to_camera=lambda pose: _camera_from_pose(torch, pose),
    )
    assert tuple(result.pose.shape) == (1, 7)
    assert tuple(result.camera.shape) == (1, 4, 4)
    assert result.rotation_determinant == pytest.approx(1.0, abs=ROTATION_ATOL)
    assert result.orthonormality_max_abs_error <= ROTATION_ATOL
    torch.testing.assert_close(result.camera[0, :3, 3], torch.tensor((0.25, -0.5, 1.0), dtype=torch.float64))


@pytest.mark.parametrize(
    ("query", "pose_head", "camera", "message"),
    (
        ("wrong_shape", None, None, "shape"),
        ("nonfinite", None, None, "finite"),
        ("valid", "nonfinite", None, "nonfinite"),
        ("valid", None, "reflection", "proper homogeneous"),
    ),
)
def test_decoder_fails_closed_for_malformed_query_head_or_camera(query: str, pose_head: str | None, camera: str | None, message: str) -> None:
    torch = _torch()
    value = torch.zeros((1, 1, 768), dtype=torch.float64)
    if query == "wrong_shape":
        value = torch.zeros((1, 768), dtype=torch.float64)
    elif query == "nonfinite":
        value[0, 0, 0] = float("nan")

    def head(token: object) -> object:
        if pose_head == "nonfinite":
            return torch.full((1, 7), float("nan"), dtype=token.dtype, device=token.device)
        return _valid_pose_head(torch, token)

    def decode(pose: object) -> object:
        result = _camera_from_pose(torch, pose)
        if camera == "reflection":
            result[:, 0, 0] = -1.0
        return result

    with pytest.raises(PreRolloutPoseQueryError, match=message):
        decode_pre_rollout_pose_query(
            value,
            torch=torch,
            pose_head=head,
            postprocess_pose=lambda pose, _mode: pose,
            pose_mode=("exp", -float("inf"), float("inf")),
            pose_encoding_to_camera=decode,
        )


def test_export_policy_replaces_only_alarm_camera_pose_and_has_no_raw_fallback() -> None:
    torch = _torch()
    policy = PreRolloutPoseQueryExport(
        torch=torch,
        pose_head=lambda token: _valid_pose_head(torch, token),
        postprocess_pose=lambda pose, _mode: pose,
        pose_mode=("exp", -float("inf"), float("inf")),
        pose_encoding_to_camera=lambda pose: _camera_from_pose(torch, pose),
    )
    clear = {"camera_pose": torch.zeros((1, 7), dtype=torch.float64), "untouched": object()}
    committed, clear_action = policy.commit_real(0, clear)
    assert committed is clear
    assert clear_action.action == "export_real_camera_pose"

    raw_pose = torch.full((1, 7), 17.0, dtype=torch.float64)
    raw = {"camera_pose": raw_pose, "untouched": clear["untouched"]}
    query = torch.zeros((1, 1, 768), dtype=torch.float64)
    query[:, :, :3] = torch.tensor((1.0, 2.0, 3.0), dtype=torch.float64)
    exported, action = policy.export_alarm(1, raw, query, query_source="pose_retriever_inquire_pre_rollout")
    assert exported is not raw
    assert exported["untouched"] is raw["untouched"]
    assert exported["camera_pose"] is not raw_pose
    torch.testing.assert_close(raw_pose, torch.full((1, 7), 17.0, dtype=torch.float64))
    torch.testing.assert_close(exported["camera_pose"][0, :3], torch.tensor((1.0, 2.0, 3.0), dtype=torch.float64))
    assert action.action == "export_pre_rollout_pose_query"
    assert action.evidence is not None and action.evidence["query_source"] == "pose_retriever_inquire_pre_rollout"

    failed = PreRolloutPoseQueryExport(
        torch=torch,
        pose_head=lambda token: _valid_pose_head(torch, token),
        postprocess_pose=lambda pose, _mode: pose,
        pose_mode=("exp", -float("inf"), float("inf")),
        pose_encoding_to_camera=lambda pose: _camera_from_pose(torch, pose),
    )
    with pytest.raises(PreRolloutPoseQueryExportError, match="unavailable"):
        failed.export_alarm(0, raw, torch.full((1, 1, 768), float("nan"), dtype=torch.float64), query_source="pose_retriever_inquire_pre_rollout")
    committed_after_failure, _ = failed.commit_real(0, raw)
    assert committed_after_failure is raw


def test_real_recal3r_pose_head_and_postprocess_obey_the_pinned_camera_contract() -> None:
    torch = _torch()
    if str(RECAL3R_SRC) not in sys.path:
        sys.path.insert(0, str(RECAL3R_SRC))
    import dust3r.model  # noqa: F401  # resolves the upstream camera/head import cycle
    from dust3r.heads.postprocess import postprocess_pose
    from dust3r.utils.camera import PoseDecoder, pose_encoding_to_camera

    torch.manual_seed(0)
    result = decode_pre_rollout_pose_query(
        torch.zeros((1, 1, 768), dtype=torch.float32),
        torch=torch,
        pose_head=PoseDecoder(hidden_size=768),
        postprocess_pose=postprocess_pose,
        pose_mode=("exp", -float("inf"), float("inf")),
        pose_encoding_to_camera=pose_encoding_to_camera,
    )
    assert result.rotation_determinant == pytest.approx(1.0, abs=ROTATION_ATOL)
    assert result.orthonormality_max_abs_error <= ROTATION_ATOL
