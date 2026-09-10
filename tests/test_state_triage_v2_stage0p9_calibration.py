from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.evaluate_state_triage_v2_stage0p9_calibration import _gates


ROOT = Path(__file__).parents[1]


def test_calibration_gate_requires_ten_activated_valid_events() -> None:
    limits = {"valid_event_count": 12, "activated_event_min": 10}

    assert all(_gates([{"activated": index < 10} for index in range(12)], limits).values())
    assert not _gates([{"activated": index < 9} for index in range(12)], limits)["activation"]
    assert not _gates([{"activated": True} for _ in range(11)], limits)["all_12_valid_events"]


def test_dispatcher_is_parseable_and_requires_frozen_protocol() -> None:
    dispatcher = ROOT / "scripts" / "run_state_triage_v2_stage0p9_calibration.sh"
    parsed = subprocess.run(["bash", "-n", str(dispatcher)], text=True, capture_output=True)
    absent = subprocess.run(["bash", str(dispatcher), "--dry-run"], cwd=ROOT, text=True, capture_output=True)

    assert parsed.returncode == 0, parsed.stderr
    assert absent.returncode != 0
    assert "missing immutable Stage 0.9 calibration protocol" in absent.stderr
