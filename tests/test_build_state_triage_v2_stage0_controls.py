from __future__ import annotations

from scripts.build_state_triage_v2_stage0_capsules import EVALUATION_SPECS, FRAME_COUNT
from scripts.build_state_triage_v2_stage0_controls import CONTROL_SPECS


def test_controls_cover_each_evaluation_source_without_reusing_event_windows() -> None:
    evaluation_sources = {spec.source_sequence for spec in EVALUATION_SPECS}
    assert {spec.source_sequence for spec in CONTROL_SPECS} == evaluation_sources
    assert len({spec.capsule_id for spec in CONTROL_SPECS}) == len(CONTROL_SPECS)
    for control in CONTROL_SPECS:
        control_window = set(range(control.source_start, control.source_start + FRAME_COUNT))
        for evaluation in EVALUATION_SPECS:
            if evaluation.source_sequence == control.source_sequence:
                event_window = set(range(evaluation.source_start, evaluation.source_start + FRAME_COUNT))
                assert control_window.isdisjoint(event_window)
