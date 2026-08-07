from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from scripts import run_recal3r_early_spatial_pooled_pose_export_v9 as script
from stateguard3r.early_spatial_pooled_pose_v9 import (
    POOLED_TOKEN_SHAPE,
    PREPROJECTION_SPATIAL_SHAPE,
    ROTATION_ATOL,
    decode_early_spatial_pooled_pose_token,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECAL3R_ROOT = PROJECT_ROOT.parent / "baselines" / "ReCal3R"
RECAL3R_SRC = RECAL3R_ROOT / "src"


def _torch() -> object:
    return pytest.importorskip("torch")


def _args(
    manifest: Path,
    *,
    watchdog: int = 8,
    commitment: object | None = None,
    detector_config: Path | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        health_profile="v3",
        watchdog=watchdog,
        recovery_quality_commitment=commitment,
        input_manifest=manifest,
        detector_config=script.FORMAL_V3_CONFIG if detector_config is None else detector_config,
    )


def test_v9_accepts_only_disclosed_read_only_manifests() -> None:
    parser = script._parser()
    assert {path.resolve(strict=True) for path in script.DEVELOPMENT_MANIFESTS} == {
        script.DEVELOPMENT_INPUT_ROOT / "development-dynamic" / "input-manifest.json",
        script.DEVELOPMENT_INPUT_ROOT / "development-wrong" / "input-manifest.json",
        script.DEVELOPMENT_INPUT_ROOT / "development-low" / "input-manifest.json",
    }
    for manifest in script.DEVELOPMENT_MANIFESTS:
        script._validate_args(_args(manifest), parser)
    with pytest.raises(SystemExit):
        script._validate_args(_args(next(iter(script.DEVELOPMENT_MANIFESTS)), watchdog=7), parser)
    with pytest.raises(SystemExit):
        script._validate_args(_args(next(iter(script.DEVELOPMENT_MANIFESTS)), commitment=object()), parser)


def test_v9_requires_direct_new_output_and_pinned_policy_components(tmp_path: Path) -> None:
    allowed = script.ROOT / "outputs" / "test-v9-direct-output"
    script._validate_new_output_dir(allowed)
    with pytest.raises(RuntimeError, match="new direct"):
        script._validate_new_output_dir(script.ROOT / "outputs" / "nested" / "child")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(RuntimeError, match="new direct"):
        script._validate_new_output_dir(existing)

    action = next(item for item in script._parser()._actions if item.dest == "state_policy")
    assert action.choices == (
        "always-commit",
        "detector-v3-incremental-early-spatial-pooled-pose-export",
    )
    source = Path(script.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "recal3r_early_decoder_pose_runner_v8",
        "EarlyDecoderPoseExport",
        "PreRolloutPoseQueryExport",
        "CurrentPointmapConsensusExport",
        "register_anchor_orb_3d3d",
    ):
        assert forbidden not in source


def test_v9_pinned_dpt_head_interface_fails_closed() -> None:
    class _Pinned:
        pose_mode = ("exp", -float("inf"), float("inf"))

        def pose_head(self, value: object) -> object:
            return value

    module = SimpleNamespace(DPTPts3dPose=_Pinned)
    pose_head, pose_mode = script._pinned_pose_head_interface(
        SimpleNamespace(downstream_head=_Pinned()), module
    )
    assert callable(pose_head)
    assert pose_mode == _Pinned.pose_mode
    with pytest.raises(RuntimeError, match="not pinned"):
        script._pinned_pose_head_interface(SimpleNamespace(downstream_head=object()), module)


def test_v9_provenance_rejects_tracked_component_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    component = Path(script.__file__).resolve()
    monkeypatch.setattr(script, "V9_COMPONENTS", (component,))

    def dirty(_root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return "commit"
        if args == ("status", "--porcelain", "--untracked-files=no"):
            return " M tracked.py"
        raise AssertionError(args)

    monkeypatch.setattr(script.smoke, "_git_output", dirty)
    with pytest.raises(RuntimeError, match="tracked changes"):
        script._self_provenance()


def test_v9_checkpoint_loaded_dpt_accepts_only_projected_spatial_mean_on_cpu() -> None:
    torch = _torch()
    assert not torch.cuda.is_initialized()
    checkpoint = RECAL3R_SRC / "cut3r_512_dpt_4_64.pth"
    assert checkpoint.is_file()
    for path in (RECAL3R_ROOT, RECAL3R_SRC):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import add_ckpt_path

    add_ckpt_path.add_path_to_dust3r(str(checkpoint))
    import dust3r.heads.dpt_head as dpt_head
    import dust3r.heads.postprocess as postprocess
    import dust3r.model as dust3r_model
    from dust3r.utils.camera import pose_encoding_to_camera

    model, checkpoint_audit = script.smoke._load_model_with_state_dict_audit(
        dust3r_model.ARCroco3DStereo, checkpoint, torch
    )
    assert checkpoint_audit["strict"] is False
    assert type(model.decoder_embed) is torch.nn.Linear
    assert (model.decoder_embed.in_features, model.decoder_embed.out_features) == (1024, 768)
    pose_head, pose_mode = script._pinned_pose_head_interface(model, dpt_head)
    spatial = torch.zeros(PREPROJECTION_SPATIAL_SHAPE, dtype=model.decoder_embed.weight.dtype)
    pooled = model.decoder_embed(spatial.mean(dim=1, keepdim=True))
    assert tuple(pooled.shape) == POOLED_TOKEN_SHAPE
    with pytest.raises(RuntimeError):
        pose_head(spatial[:, 0])
    result = decode_early_spatial_pooled_pose_token(
        pooled,
        torch=torch,
        pose_head=pose_head,
        postprocess_pose=postprocess.postprocess_pose,
        pose_mode=pose_mode,
        pose_encoding_to_camera=pose_encoding_to_camera,
    )
    assert result.rotation_determinant == pytest.approx(1.0, abs=ROTATION_ATOL)
    assert result.orthonormality_max_abs_error <= ROTATION_ATOL
    assert not torch.cuda.is_initialized()
