from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_recal3r_selective_state_memory_repair_v13 as script


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _args(manifest: Path, *, watchdog: int = 8, detector_config: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        health_profile="v3", watchdog=watchdog, recovery_quality_commitment=None,
        input_manifest=manifest,
        detector_config=script.FORMAL_V3_CONFIG if detector_config is None else detector_config,
    )


def test_v13_cli_accepts_only_disclosed_read_only_manifests_and_fixed_inputs() -> None:
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
    action = next(item for item in parser._actions if item.dest == "state_policy")
    assert action.choices == ("always-commit", "detector-v3-incremental-selective-state-memory-repair")


def test_v13_cli_requires_new_direct_output_and_tracks_only_v13_components(tmp_path: Path) -> None:
    script._validate_new_output_dir(script.ROOT / "outputs" / "test-v13-direct-output")
    with pytest.raises(RuntimeError, match="new direct"):
        script._validate_new_output_dir(script.ROOT / "outputs" / "nested" / "child")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(RuntimeError, match="new direct"):
        script._validate_new_output_dir(existing)
    source = Path(script.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "PatchEmbedGlobal", "EncoderGlobal", "EarlySpatial", "CurrentPointmap",
        "PreRolloutPose", "anchor_export_v3", "camera_to_pose_encoding",
    ):
        assert forbidden not in source
    assert json.loads(json.dumps({"components": [str(path) for path in script.V13_COMPONENTS]}))["components"]


def test_v13_observer_finalization_preserves_raw_pose_and_never_reads_cpu_state_mirror() -> None:
    observe_source = inspect.getsource(script._V13Observer.observe)
    finalize_source = inspect.getsource(script._V13Observer.finalize)
    module = ast.parse(Path(script.__file__).read_text(encoding="utf-8"))
    assert "get_u_calibration_last_state" not in observe_source + finalize_source
    assert "_u_calibration_last_state" not in observe_source + finalize_source
    assert "prediction[\"camera_pose\"]" in observe_source
    assert "raw current camera pose" in finalize_source
    finalize = next(node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == "finalize")
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in {"cpu", "numpy", "tolist"}
        for node in ast.walk(finalize)
    )


def test_v13_production_main_requires_cuda_before_model_execution() -> None:
    source = inspect.getsource(script.main)
    assert 'if args.device != "cuda" or not torch.cuda.is_available()' in source
    assert 'raise RuntimeError("v13 production runner requires CUDA")' in source
