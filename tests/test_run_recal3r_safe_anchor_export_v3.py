from __future__ import annotations

from scripts import run_recal3r_safe_anchor_export_v3 as runner


def test_v3_runner_parser_exposes_only_preregistered_policy_and_fixed_watchdog() -> None:
    parser = runner._parser()
    action = next(item for item in parser._actions if item.dest == "state_policy")
    assert action.choices == ("always-commit", "detector-v3-incremental-safe-anchor-se3-export")
    assert parser.get_default("health_profile") == "v3"
    assert parser.get_default("watchdog") == 8
