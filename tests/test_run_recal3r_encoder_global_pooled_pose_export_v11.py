from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace

import pytest

from scripts import run_recal3r_encoder_global_pooled_pose_export_v11 as script


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _args(manifest: Path, *, watchdog: int = 8, detector_config: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        health_profile="v3",
        watchdog=watchdog,
        recovery_quality_commitment=None,
        input_manifest=manifest,
        detector_config=script.FORMAL_V3_CONFIG if detector_config is None else detector_config,
    )


def test_v11_cli_accepts_only_disclosed_read_only_manifests_and_fixed_policy_inputs() -> None:
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
        script._validate_args(_args(next(iter(script.DEVELOPMENT_MANIFESTS)), detector_config=PROJECT_ROOT / "other.json"), parser)


def test_v11_cli_requires_new_direct_output_and_only_its_two_policies(tmp_path: Path) -> None:
    direct = script.ROOT / "outputs" / "test-v11-direct-output"
    script._validate_new_output_dir(direct)
    with pytest.raises(RuntimeError, match="new direct"):
        script._validate_new_output_dir(script.ROOT / "outputs" / "nested" / "child")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(RuntimeError, match="new direct"):
        script._validate_new_output_dir(existing)
    action = next(item for item in script._parser()._actions if item.dest == "state_policy")
    assert action.choices == ("always-commit", "detector-v3-incremental-encoder-global-pooled-pose-export")


def test_v11_cli_components_are_separate_and_head_interface_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    source = Path(script.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "recal3r_early_spatial_pooled_pose_runner_v9",
        "EarlySpatialPooledPoseExport",
        "PreRolloutPoseQueryExport",
        "CurrentPointmapConsensusExport",
        "register_anchor_orb_3d3d",
    ):
        assert forbidden not in source

    class _Pinned:
        pose_mode = ("exp", -float("inf"), float("inf"))

        def pose_head(self, value: object) -> object:
            return value

    pose_head, pose_mode = script._pinned_pose_head_interface(
        SimpleNamespace(downstream_head=_Pinned()), SimpleNamespace(DPTPts3dPose=_Pinned)
    )
    assert callable(pose_head) and pose_mode == _Pinned.pose_mode
    with pytest.raises(RuntimeError, match="not pinned"):
        script._pinned_pose_head_interface(SimpleNamespace(downstream_head=object()), SimpleNamespace(DPTPts3dPose=_Pinned))

    component = Path(script.__file__).resolve()
    monkeypatch.setattr(script, "V11_COMPONENTS", (component,))

    def dirty(_root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return "commit"
        if args == ("status", "--porcelain", "--untracked-files=no"):
            return " M tracked.py"
        raise AssertionError(args)

    monkeypatch.setattr(script.smoke, "_git_output", dirty)
    with pytest.raises(RuntimeError, match="tracked changes"):
        script._self_provenance()


def test_v11_observer_keeps_decoder_boundary_token_only_and_records_loaded_shape() -> None:
    alarm_source = inspect.getsource(script._V11Observer.finalize)
    tree = ast.parse(textwrap.dedent(alarm_source))
    prediction_subscripts = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
        and node.value.id == "prediction" and isinstance(node.ctx, ast.Load)
    ]
    # Raw prediction can provide only detector accounting/digests and the
    # opaque mapping passed to the export wrapper; the pose decoder is invoked
    # solely inside the wrapper with its capture token.
    assert prediction_subscripts
    assert "export_alarm(observation.frame_id, prediction, observation.capture" in alarm_source
    assert "loaded_image_shape" in alarm_source
    assert "decode_encoder_global_pose_token" not in alarm_source


def test_v11_metadata_causality_text_excludes_gt_and_future_export_inputs() -> None:
    source = Path(script.__file__).read_text(encoding="utf-8")
    assert "alarm export consumes only current encoder-global token after rollback" in source
    assert "no raw camera_pose numeric input, fallback, future input or export feedback" in source
    assert json.loads(json.dumps({"components": [str(path) for path in script.V11_COMPONENTS]}))["components"]
