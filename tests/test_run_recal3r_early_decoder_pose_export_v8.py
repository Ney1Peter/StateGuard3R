from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_recal3r_early_decoder_pose_export_v8 as script


def _args(manifest: Path, *, watchdog: int = 8, commitment: object | None = None, detector_config: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        health_profile="v3",
        watchdog=watchdog,
        recovery_quality_commitment=commitment,
        input_manifest=manifest,
        detector_config=script.FORMAL_V3_CONFIG if detector_config is None else detector_config,
    )


def test_v8_accepts_exactly_the_three_disclosed_read_only_manifests(tmp_path: Path) -> None:
    parser = script._parser()
    assert {path.resolve(strict=True) for path in script.DEVELOPMENT_MANIFESTS} == {
        script.DEVELOPMENT_INPUT_ROOT / "development-dynamic" / "input-manifest.json",
        script.DEVELOPMENT_INPUT_ROOT / "development-wrong" / "input-manifest.json",
        script.DEVELOPMENT_INPUT_ROOT / "development-low" / "input-manifest.json",
    }
    for manifest in script.DEVELOPMENT_MANIFESTS:
        script._validate_args(_args(manifest), parser)

    undisclosed = tmp_path / "input-manifest.json"
    undisclosed.write_text(json.dumps({"schema_version": "stateguard3r.corruption.v1", "source_is_read_only": True}), encoding="utf-8")
    with pytest.raises(SystemExit):
        script._validate_args(_args(undisclosed), parser)
    with pytest.raises(SystemExit):
        script._validate_args(_args(next(iter(script.DEVELOPMENT_MANIFESTS)), watchdog=7), parser)
    with pytest.raises(SystemExit):
        script._validate_args(_args(next(iter(script.DEVELOPMENT_MANIFESTS)), commitment=object()), parser)
    with pytest.raises(SystemExit):
        script._validate_args(_args(next(iter(script.DEVELOPMENT_MANIFESTS)), detector_config=tmp_path / "other-config.json"), parser)


def test_v8_output_must_be_a_new_direct_outputs_child(tmp_path: Path) -> None:
    allowed = script.ROOT / "outputs" / "test-v8-output-validation-unique"
    script._validate_new_output_dir(allowed)
    with pytest.raises(RuntimeError, match="new direct"):
        script._validate_new_output_dir(script.ROOT / "outputs" / "nested" / "child")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(RuntimeError, match="new direct"):
        script._validate_new_output_dir(existing)


def test_v8_provenance_rejects_tracked_worktree_changes_and_component_blob_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    component = Path(script.__file__).resolve()
    monkeypatch.setattr(script, "V8_COMPONENTS", (component,))

    def dirty_git(_root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return "commit"
        if args == ("status", "--porcelain", "--untracked-files=no"):
            return " M tracked.py"
        raise AssertionError(args)

    monkeypatch.setattr(script.smoke, "_git_output", dirty_git)
    with pytest.raises(RuntimeError, match="tracked changes"):
        script._self_provenance()

    def drift_git(_root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return "commit"
        if args == ("status", "--porcelain", "--untracked-files=no"):
            return ""
        if args[:2] == ("ls-files", "--error-unmatch"):
            return str(component)
        if args[0] == "rev-parse" and args[1].startswith("HEAD:"):
            return "committed-blob"
        if args[:2] == ("hash-object", "--"):
            return "working-blob"
        raise AssertionError(args)

    monkeypatch.setattr(script.smoke, "_git_output", drift_git)
    with pytest.raises(RuntimeError, match="differs from its HEAD blob"):
        script._self_provenance()


def test_v8_runtime_components_exclude_previous_recovery_exports() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in script.V8_COMPONENTS)
    for forbidden in (
        "recal3r_safe_anchor_runner_v3",
        "recal3r_geometric_registration_runner_v4",
        "recal3r_current_pointmap_runner_v5",
        "recal3r_prerollout_pose_query_runner_v6",
        "register_anchor_orb_3d3d",
    ):
        assert forbidden not in source
