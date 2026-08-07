from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sys

import pytest

from stateguard3r.early_decoder_pose_export_v8 import (
    EarlyDecoderPoseExport,
    EarlyDecoderPoseExportError,
)
from stateguard3r.early_decoder_pose_v8 import (
    EarlyDecoderPoseError,
    PINNED_RECAL3R_SOURCE_SHA256,
    ROTATION_ATOL,
    audit_pinned_early_decoder_pose_sources,
    decode_early_decoder_pose_token,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECAL3R_SRC = PROJECT_ROOT.parent / "baselines" / "ReCal3R" / "src"
RUNNER = PROJECT_ROOT / "src" / "stateguard3r" / "recal3r_early_decoder_pose_runner_v8.py"
TOKEN_SOURCE = "decoder_layer_0_after_rollout_before_update_mem"


def _function_node(callable_object: object) -> ast.FunctionDef:
    module = ast.parse(Path(inspect.getsourcefile(callable_object) or "").read_text(encoding="utf-8"))
    name = getattr(callable_object, "__name__")
    return next(node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == name)


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


def test_v8_decoder_signature_and_ast_are_token_only() -> None:
    signature = inspect.signature(decode_early_decoder_pose_token)
    assert tuple(signature.parameters) == (
        "token",
        "torch",
        "pose_head",
        "postprocess_pose",
        "pose_mode",
        "pose_encoding_to_camera",
    )
    node = _function_node(decode_early_decoder_pose_token)
    source = inspect.getsource(decode_early_decoder_pose_token).lower()
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


def test_alarm_export_never_numerically_loads_raw_camera_pose() -> None:
    module_path = inspect.getsourcefile(EarlyDecoderPoseExport)
    assert module_path is not None
    module = ast.parse(Path(module_path).read_text(encoding="utf-8"))
    node = next(item for item in ast.walk(module) if isinstance(item, ast.FunctionDef) and item.name == "export_alarm")
    prediction_loads = [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Subscript)
        and isinstance(item.value, ast.Name)
        and item.value.id == "prediction"
        and isinstance(item.ctx, ast.Load)
    ]
    assert prediction_loads == []
    assert not any(isinstance(item, ast.Attribute) and item.attr == "cpu" for item in ast.walk(node))
    assert not any(
        isinstance(item, ast.Attribute) and item.attr in {"cpu", "numpy", "tolist"}
        for item in ast.walk(module)
    )


def test_source_audit_binds_v8_early_layer_and_runner_capture(tmp_path: Path) -> None:
    dust3r = RECAL3R_SRC / "dust3r"
    audit = audit_pinned_early_decoder_pose_sources(
        dust3r / "model.py", dust3r / "heads" / "dpt_head.py", dust3r / "heads" / "postprocess.py", RUNNER
    )
    assert audit["early_decoder_token_source"] == TOKEN_SOURCE
    assert audit["early_decoder_layer_index"] == 0
    assert audit["capture_relation"] == "after_rollout_before_update_mem"
    assert audit["raw_camera_pose_numeric_input_to_v8_decoder"] is False
    assert len(audit["model_sha256"]) == len(audit["dpt_head_sha256"]) == len(audit["postprocess_sha256"]) == len(audit["runner_sha256"]) == 64
    assert {key: audit[key] for key in PINNED_RECAL3R_SOURCE_SHA256} == PINNED_RECAL3R_SOURCE_SHA256

    changed_model = tmp_path / "model.py"
    changed_model.write_text((dust3r / "model.py").read_text(encoding="utf-8").replace("dec[0].float()", "dec[-1].float()"), encoding="utf-8")
    with pytest.raises(EarlyDecoderPoseError, match="topology"):
        audit_pinned_early_decoder_pose_sources(changed_model, dust3r / "heads" / "dpt_head.py", dust3r / "heads" / "postprocess.py", RUNNER)

    changed_runner = tmp_path / "runner.py"
    runner_source = RUNNER.read_text(encoding="utf-8")
    runner_entry = runner_source.index("\ndef run_early_decoder_pose_recurrent_lighter(")
    changed_runner.write_text(runner_source[:runner_entry] + runner_source[runner_entry:].replace("dec[0][:, 0:1].detach().clone()", "dec[-1][:, 0:1].detach().clone()", 1), encoding="utf-8")
    with pytest.raises(EarlyDecoderPoseError, match="order"):
        audit_pinned_early_decoder_pose_sources(dust3r / "model.py", dust3r / "heads" / "dpt_head.py", dust3r / "heads" / "postprocess.py", changed_runner)

    comment_only = tmp_path / "comment_only.py"
    comment_only.write_text((dust3r / "model.py").read_text(encoding="utf-8") + "\n# v8 audit mutation\n", encoding="utf-8")
    with pytest.raises(EarlyDecoderPoseError, match="SHA-256"):
        audit_pinned_early_decoder_pose_sources(comment_only, dust3r / "heads" / "dpt_head.py", dust3r / "heads" / "postprocess.py", RUNNER)


def test_decoder_accepts_only_a_finite_layer_zero_token_and_validates_camera() -> None:
    torch = _torch()
    token = torch.zeros((1, 1, 768), dtype=torch.float64)
    token[:, :, :3] = torch.tensor((0.25, -0.5, 1.0), dtype=torch.float64)
    result = decode_early_decoder_pose_token(
        token,
        torch=torch,
        pose_head=lambda input_token: _valid_pose_head(torch, input_token),
        postprocess_pose=lambda pose, _mode: pose,
        pose_mode=("exp", -float("inf"), float("inf")),
        pose_encoding_to_camera=lambda pose: _camera_from_pose(torch, pose),
    )
    assert tuple(result.pose.shape) == (1, 7)
    assert tuple(result.camera.shape) == (1, 4, 4)
    assert result.rotation_determinant == pytest.approx(1.0, abs=ROTATION_ATOL)
    assert result.orthonormality_max_abs_error <= ROTATION_ATOL
    torch.testing.assert_close(result.camera[0, :3, 3], torch.tensor((0.25, -0.5, 1.0), dtype=torch.float64))


def test_decoder_keeps_head_dtype_then_promotes_only_the_seven_output() -> None:
    torch = _torch()
    seen: dict[str, object] = {}
    token = torch.zeros((1, 1, 768), dtype=torch.float32)

    def head(input_token: object) -> object:
        seen["head_dtype"] = input_token.dtype
        return _valid_pose_head(torch, input_token)

    def postprocess(raw: object, _mode: object) -> object:
        seen["postprocess_dtype"] = raw.dtype
        seen["postprocess_device"] = raw.device
        return raw

    result = decode_early_decoder_pose_token(token, torch=torch, pose_head=head, postprocess_pose=postprocess, pose_mode=("exp", -float("inf"), float("inf")), pose_encoding_to_camera=lambda pose: _camera_from_pose(torch, pose))
    assert seen == {"head_dtype": torch.float32, "postprocess_dtype": torch.float64, "postprocess_device": token.device}
    assert result.pose.dtype == torch.float64


def test_decoder_accepts_a_proper_rotation_camera() -> None:
    torch = _torch()
    token = torch.zeros((1, 1, 768), dtype=torch.float64)

    def rotated_camera(pose: object) -> object:
        camera = _camera_from_pose(torch, pose)
        camera[0, :3, :3] = torch.tensor(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=pose.dtype)
        return camera

    result = decode_early_decoder_pose_token(token, torch=torch, pose_head=lambda input_token: _valid_pose_head(torch, input_token), postprocess_pose=lambda pose, _mode: pose, pose_mode=("exp", -float("inf"), float("inf")), pose_encoding_to_camera=rotated_camera)
    assert result.rotation_determinant == pytest.approx(1.0, abs=ROTATION_ATOL)
    assert result.orthonormality_max_abs_error <= ROTATION_ATOL


@pytest.mark.parametrize(
    ("token_kind", "head_kind", "camera_kind", "message"),
    (
        ("wrong_shape", "valid", "valid", "shape"),
        ("integer", "valid", "valid", "finite floating"),
        ("nonfinite", "valid", "valid", "finite"),
        ("infinite", "valid", "valid", "finite"),
        ("valid", "malformed", "valid", "malformed"),
        ("valid", "nonfinite", "valid", "nonfinite"),
        ("valid", "valid", "reflection", "proper homogeneous"),
        ("valid", "valid", "nonfinite", "nonfinite"),
    ),
)
def test_decoder_fails_closed_for_malformed_token_head_or_camera(token_kind: str, head_kind: str, camera_kind: str, message: str) -> None:
    torch = _torch()
    token = torch.zeros((1, 1, 768), dtype=torch.float64)
    if token_kind == "wrong_shape":
        token = torch.zeros((1, 768), dtype=torch.float64)
    elif token_kind == "integer":
        token = torch.zeros((1, 1, 768), dtype=torch.int64)
    elif token_kind == "nonfinite":
        token[0, 0, 0] = float("nan")
    elif token_kind == "infinite":
        token[0, 0, 0] = float("inf")

    def head(input_token: object) -> object:
        if head_kind == "malformed":
            return torch.zeros((1, 8), dtype=input_token.dtype, device=input_token.device)
        if head_kind == "nonfinite":
            return torch.full((1, 7), float("nan"), dtype=input_token.dtype, device=input_token.device)
        return _valid_pose_head(torch, input_token)

    def decode(pose: object) -> object:
        camera = _camera_from_pose(torch, pose)
        if camera_kind == "reflection":
            camera[:, 0, 0] = -1.0
        if camera_kind == "nonfinite":
            camera[:, 0, 0] = float("nan")
        return camera

    with pytest.raises(EarlyDecoderPoseError, match=message):
        decode_early_decoder_pose_token(
            token,
            torch=torch,
            pose_head=head,
            postprocess_pose=lambda pose, _mode: pose,
            pose_mode=("exp", -float("inf"), float("inf")),
            pose_encoding_to_camera=decode,
        )


@pytest.mark.parametrize("camera_kind", ("nonorthogonal", "bad_homogeneous"))
def test_decoder_fails_closed_for_invalid_camera_geometry(camera_kind: str) -> None:
    torch = _torch()

    def decode(pose: object) -> object:
        camera = _camera_from_pose(torch, pose)
        if camera_kind == "nonorthogonal":
            camera[0, 0, 1] = 0.25
        else:
            camera[0, 3, 3] = 0.5
        return camera

    with pytest.raises(EarlyDecoderPoseError, match="proper homogeneous"):
        decode_early_decoder_pose_token(torch.zeros((1, 1, 768), dtype=torch.float64), torch=torch, pose_head=lambda input_token: _valid_pose_head(torch, input_token), postprocess_pose=lambda pose, _mode: pose, pose_mode=("exp", -float("inf"), float("inf")), pose_encoding_to_camera=decode)


def test_decoder_fails_closed_when_token_and_pinned_head_devices_differ() -> None:
    torch = _torch()
    head = torch.nn.Linear(768, 7).to("meta")
    with pytest.raises(EarlyDecoderPoseError, match="head device"):
        decode_early_decoder_pose_token(torch.zeros((1, 1, 768), dtype=torch.float32), torch=torch, pose_head=head, postprocess_pose=lambda pose, _mode: pose, pose_mode=("exp", -float("inf"), float("inf")), pose_encoding_to_camera=lambda pose: _camera_from_pose(torch, pose))


def test_export_policy_replaces_only_alarm_camera_pose_and_records_v8_evidence() -> None:
    torch = _torch()
    policy = EarlyDecoderPoseExport(
        torch=torch,
        pose_head=lambda input_token: _valid_pose_head(torch, input_token),
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
    token = torch.zeros((1, 1, 768), dtype=torch.float64)
    token[:, :, :3] = torch.tensor((1.0, 2.0, 3.0), dtype=torch.float64)
    exported, action = policy.export_alarm(1, raw, token, token_source=TOKEN_SOURCE)
    assert exported is not raw
    assert exported["untouched"] is raw["untouched"]
    assert exported["camera_pose"] is not raw_pose
    torch.testing.assert_close(raw_pose, torch.full((1, 7), 17.0, dtype=torch.float64))
    torch.testing.assert_close(exported["camera_pose"][0, :3], torch.tensor((1.0, 2.0, 3.0), dtype=torch.float64))
    assert action.action == "export_early_decoder_pose_token"
    assert action.evidence is not None
    assert action.evidence["early_decoder_layer_index"] == 0
    assert action.evidence["capture_relation"] == "after_rollout_before_update_mem"
    assert action.evidence["token_source"] == TOKEN_SOURCE
    assert action.evidence["proper_rotation"] is True
    assert action.evidence["token_digest"].startswith("gpu-fingerprint-v1:")
    assert action.evidence["exported_pose_digest"].startswith("gpu-fingerprint-v1:")

    unavailable = EarlyDecoderPoseExport(
        torch=torch,
        pose_head=lambda input_token: _valid_pose_head(torch, input_token),
        postprocess_pose=lambda pose, _mode: pose,
        pose_mode=("exp", -float("inf"), float("inf")),
        pose_encoding_to_camera=lambda pose: _camera_from_pose(torch, pose),
    )
    with pytest.raises(EarlyDecoderPoseExportError, match="unavailable"):
        unavailable.export_alarm(0, raw, torch.full((1, 1, 768), float("nan"), dtype=torch.float64), token_source=TOKEN_SOURCE)
    committed_after_failure, _ = unavailable.commit_real(0, raw)
    assert committed_after_failure is raw


def test_real_recal3r_pose_head_and_postprocess_obey_the_pinned_camera_contract() -> None:
    torch = _torch()
    if str(RECAL3R_SRC) not in sys.path:
        sys.path.insert(0, str(RECAL3R_SRC))
    import dust3r.model  # noqa: F401  # resolves upstream camera/head import cycle
    from dust3r.heads.postprocess import postprocess_pose
    from dust3r.utils.camera import PoseDecoder, pose_encoding_to_camera

    torch.manual_seed(0)
    result = decode_early_decoder_pose_token(
        torch.zeros((1, 1, 768), dtype=torch.float32),
        torch=torch,
        pose_head=PoseDecoder(hidden_size=768),
        postprocess_pose=postprocess_pose,
        pose_mode=("exp", -float("inf"), float("inf")),
        pose_encoding_to_camera=pose_encoding_to_camera,
    )
    assert result.rotation_determinant == pytest.approx(1.0, abs=ROTATION_ATOL)
    assert result.orthonormality_max_abs_error <= ROTATION_ATOL


def test_checkpoint_loaded_recal3r_dpt_pose_head_obeys_the_pinned_camera_contract() -> None:
    torch = _torch()
    baseline_root = PROJECT_ROOT.parent / "baselines" / "ReCal3R"
    checkpoint = baseline_root / "src" / "cut3r_512_dpt_4_64.pth"
    assert checkpoint.is_file()
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
    from scripts import run_recal3r_early_decoder_pose_export_v8 as script

    model, checkpoint_audit = script.smoke._load_model_with_state_dict_audit(dust3r_model.ARCroco3DStereo, checkpoint, torch)
    assert checkpoint_audit["strict"] is False
    pose_head, pose_mode = script._pinned_pose_head_interface(model, dpt_head)
    result = decode_early_decoder_pose_token(torch.zeros((1, 1, 768), dtype=torch.float32), torch=torch, pose_head=pose_head, postprocess_pose=postprocess.postprocess_pose, pose_mode=pose_mode, pose_encoding_to_camera=pose_encoding_to_camera)
    assert result.rotation_determinant == pytest.approx(1.0, abs=ROTATION_ATOL)
    assert result.orthonormality_max_abs_error <= ROTATION_ATOL
