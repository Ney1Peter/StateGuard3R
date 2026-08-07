from __future__ import annotations

from pathlib import Path

from scripts import run_recal3r_current_pointmap_consensus_export_v5 as runner


def test_v5_runner_exposes_only_preregistered_policy() -> None:
    action = next(item for item in runner._parser()._actions if item.dest == "state_policy")
    assert action.choices == ("always-commit", "detector-v3-incremental-current-self-cross-consensus-export")
    assert runner._parser().get_default("watchdog") == 8


def test_v5_runner_is_source_bound_and_does_not_delegate_to_prior_recovery_versions() -> None:
    source = Path(runner.__file__).read_text()
    for forbidden in (
        "run_recal3r_safe_anchor_export_v3",
        "run_recal3r_geometric_registration_export_v4",
        "SafeAnchorSE3Export",
        "RealSafeGeometricAnchor",
        "register_anchor_orb_3d3d",
        "orb_pointmap_registration_v4",
    ):
        assert forbidden not in source
    assert "run_current_pointmap_recurrent_lighter" in source
    component_paths = {str(item.resolve()) for item in runner.V5_COMPONENTS}
    assert str(Path(runner.__file__).resolve()) in component_paths
    assert any(path.endswith("recal3r_current_pointmap_runner_v5.py") for path in component_paths)
