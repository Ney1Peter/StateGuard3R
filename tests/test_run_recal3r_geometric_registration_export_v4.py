from __future__ import annotations

from scripts import run_recal3r_geometric_registration_export_v4 as runner


def test_v4_runner_exposes_only_preregistered_policy() -> None:
    action = next(item for item in runner._parser()._actions if item.dest == "state_policy")
    assert action.choices == ("always-commit", "detector-v3-incremental-anchor-orb-3d3d-registration-export")
    assert runner._parser().get_default("watchdog") == 8
