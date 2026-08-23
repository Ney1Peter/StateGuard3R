from __future__ import annotations

from collections import Counter

from scripts.build_state_triage_v2_stage0_capsules import EVALUATION_SPECS, FRAME_COUNT
from scripts.build_state_triage_v2_stage0_controls import CONTROL_SPECS
from scripts.build_state_triage_v2_stage0p5_capsules import EVENT_END, EVENT_START, PILOT_SPECS, _final_indices, _transforms


def _window(spec) -> set[int]:
    return set(range(spec.source_start, spec.source_start + FRAME_COUNT))


def test_pilot_is_exactly_one_fresh_window_per_cause_and_sequence() -> None:
    assert len(PILOT_SPECS) == 12
    counts = Counter(spec.cause for spec in PILOT_SPECS)
    assert counts == Counter({"normal_novelty": 3, "registration_or_order_fault": 3, "bad_observation": 3, "transient_local_content": 3})
    for cause in counts:
        assert len({spec.source_sequence for spec in PILOT_SPECS if spec.cause == cause}) == 3


def test_all_pilot_source_windows_are_disjoint_from_stage0_eval_and_controls() -> None:
    frozen = (*EVALUATION_SPECS, *CONTROL_SPECS)
    for pilot in PILOT_SPECS:
        for old in frozen:
            if pilot.source_sequence == old.source_sequence:
                assert _window(pilot).isdisjoint(_window(old))
        for other in PILOT_SPECS:
            if other is not pilot and other.source_sequence == pilot.source_sequence:
                assert _window(pilot).isdisjoint(_window(other))


def test_transforms_are_schema_compatible_and_variants_are_fixed() -> None:
    for spec in PILOT_SPECS:
        assert _transforms(spec, spec.event_start - 1) == []
        transforms = _transforms(spec, spec.event_start)
        if spec.cause == "normal_novelty":
            assert transforms == []
            assert (spec.event_start, spec.event_end) == (0, 4)
        else:
            assert len(transforms) == 1
            assert transforms[0]["type"] in {"rectangle_occlusion", "temporal_reorder"}
    bad_fills = {tuple(_transforms(spec, EVENT_START)[0]["fill"]) for spec in PILOT_SPECS if spec.cause == "bad_observation"}
    assert bad_fills == {(0.0, 0.0, 0.0), (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)}
    for spec in (item for item in PILOT_SPECS if item.cause == "transient_local_content"):
        transform = _transforms(spec, EVENT_START)[0]
        assert transform["rectangle"]["width"] * transform["rectangle"]["height"] == 0.5


def test_registration_recipe_is_a_local_temporal_reversal() -> None:
    for spec in (item for item in PILOT_SPECS if item.cause == "registration_or_order_fault"):
        indices = _final_indices(spec)
        assert indices[EVENT_START : EVENT_END + 1] == list(reversed(range(spec.source_start + EVENT_START, spec.source_start + EVENT_END + 1)))
