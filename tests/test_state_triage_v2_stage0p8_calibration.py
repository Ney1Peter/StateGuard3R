from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.evaluate_state_triage_v2_stage0p8_calibration import _gates
from scripts.freeze_state_triage_v2_stage0p8_calibration_protocol import _inventory


ROOT = Path(__file__).parents[1]


def test_sealed_calibration_inventory_is_local_only_and_separate_from_final() -> None:
    items = _inventory(ROOT / "outputs" / "state-triage-v2-stage0p8-inputs-0001")

    assert len(items) == 12
    assert len({item["output"] for item in items}) == 12
    assert all("/calibration/" in item["capsule_path"] for item in items)
    assert all("stage0p8-calibration-" in item["output"] for item in items)


def test_calibration_gate_requires_ten_activated_valid_responses() -> None:
    events = [{"activated": index < 10} for index in range(12)]
    gates = _gates(events, {"valid_event_count": 12, "activated_event_min": 10})

    assert all(gates.values())
    assert not _gates(events[:11], {"valid_event_count": 12, "activated_event_min": 10})["all_12_valid_events"]
    assert not _gates([{"activated": index < 9} for index in range(12)], {"valid_event_count": 12, "activated_event_min": 10})["activation"]


def test_calibration_dispatcher_is_parseable_and_has_a_dry_run_path() -> None:
    dispatcher = ROOT / "scripts" / "run_state_triage_v2_stage0p8_calibration.sh"
    parsed = subprocess.run(["bash", "-n", str(dispatcher)], text=True, capture_output=True)

    assert parsed.returncode == 0, parsed.stderr
    assert "--dry-run" in dispatcher.read_text(encoding="utf-8")
