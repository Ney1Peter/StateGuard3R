from __future__ import annotations

from pathlib import Path

from scripts import run_recal3r_geometric_registration_export_v4 as runner


def test_v4_runner_exposes_only_preregistered_policy() -> None:
    action = next(item for item in runner._parser()._actions if item.dest == "state_policy")
    assert action.choices == ("always-commit", "detector-v3-incremental-anchor-orb-3d3d-registration-export")
    assert runner._parser().get_default("watchdog") == 8


def test_v4_runner_is_source_bound_and_does_not_delegate_to_v3() -> None:
    source = Path(runner.__file__).read_text()
    assert "run_recal3r_safe_anchor_export_v3" not in source
    assert "SafeAnchorSE3Export" not in source
    assert "v3.main" not in source
    components = {item["path"] for item in runner._component_provenance()}
    assert str(Path(runner.__file__).resolve()) in components
    assert any(path.endswith("recal3r_geometric_registration_runner_v4.py") for path in components)
