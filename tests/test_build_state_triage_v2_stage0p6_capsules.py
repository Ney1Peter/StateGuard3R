from __future__ import annotations

from collections import Counter

from scripts.build_state_triage_v2_stage0_capsules import EVALUATION_SPECS, FRAME_COUNT
from scripts.build_state_triage_v2_stage0_controls import CONTROL_SPECS
from scripts.build_state_triage_v2_stage0p5_capsules import PILOT_SPECS as P5_SPECS
from scripts.build_state_triage_v2_stage0p6_capsules import EVENT_END, EVENT_START, PILOT_SPECS, _final_indices, _transforms


def _window(spec) -> set[int]:
    return set(range(spec.source_start, spec.source_start + FRAME_COUNT))


def test_p6_has_twelve_balanced_fresh_capsules() -> None:
    assert len(PILOT_SPECS) == 12
    assert Counter(spec.cause for spec in PILOT_SPECS) == Counter({"normal_novelty": 3, "registration_or_order_fault": 3, "bad_observation": 3, "transient_local_content": 3})


def test_p6_windows_are_disjoint_from_all_prior_and_current_windows() -> None:
    for pilot in PILOT_SPECS:
        for other in (*EVALUATION_SPECS, *CONTROL_SPECS, *P5_SPECS, *PILOT_SPECS):
            if other is not pilot and other.source_sequence == pilot.source_sequence:
                assert _window(pilot).isdisjoint(_window(other))


def test_p6_recipes_are_schema_compatible_and_new_local_variants_cover_half_an_image() -> None:
    for spec in PILOT_SPECS:
        assert _transforms(spec, spec.event_start - 1) == []
        transforms = _transforms(spec, spec.event_start)
        if spec.cause == "normal_novelty":
            assert transforms == [] and (spec.event_start, spec.event_end) == (0, 4)
        else:
            assert len(transforms) == 1
    for spec in (item for item in PILOT_SPECS if item.cause == "transient_local_content"):
        rectangle = _transforms(spec, EVENT_START)[0]["rectangle"]
        assert rectangle["width"] * rectangle["height"] == 0.5
    for spec in (item for item in PILOT_SPECS if item.cause == "registration_or_order_fault"):
        assert _final_indices(spec)[EVENT_START : EVENT_END + 1] == list(reversed(range(spec.source_start + EVENT_START, spec.source_start + EVENT_END + 1)))
