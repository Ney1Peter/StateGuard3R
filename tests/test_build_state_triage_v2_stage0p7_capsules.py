from __future__ import annotations

from collections import Counter

from scripts.build_state_triage_v2_stage0_capsules import DRY_RUN_SPECS, EVALUATION_SPECS, FRAME_COUNT
from scripts.build_state_triage_v2_stage0_controls import CONTROL_SPECS
from scripts.build_state_triage_v2_stage0p5_capsules import PILOT_SPECS as P5_SPECS
from scripts.build_state_triage_v2_stage0p6_capsules import PILOT_SPECS as P6_SPECS
from scripts.build_state_triage_v2_stage0p7_capsules import CAUSES, PILOT_SPECS, QUOTAS, _final_indices, _selected_slots, _transforms


def _window(spec) -> set[int]:
    return set(range(spec.source_start, spec.source_start + FRAME_COUNT))


def test_window_benchmark_is_large_balanced_and_quota_matched() -> None:
    assert len(PILOT_SPECS) == 80
    assert Counter(spec.cause for spec in PILOT_SPECS) == Counter({cause: 20 for cause in CAUSES})
    for sequence, counts in QUOTAS.items():
        assert Counter(spec.cause for spec in PILOT_SPECS if spec.source_sequence == sequence) == Counter(counts)
        assert len(_selected_slots(sequence)) == sum(counts.values())


def test_every_p7_window_is_disjoint_from_all_prior_and_current_windows() -> None:
    previous = (*EVALUATION_SPECS, *DRY_RUN_SPECS, *CONTROL_SPECS, *P5_SPECS, *P6_SPECS)
    for index, spec in enumerate(PILOT_SPECS):
        for other in (*previous, *PILOT_SPECS[:index], *PILOT_SPECS[index + 1 :]):
            if spec.source_sequence == other.source_sequence:
                assert _window(spec).isdisjoint(_window(other))


def test_local_recipes_have_fixed_moderate_and_stress_tiers() -> None:
    local = [spec for spec in PILOT_SPECS if spec.cause == "transient_local_content"]
    assert Counter(spec.local_tier for spec in local) == Counter({"stress_75pct": 12, "moderate_50pct": 8})
    for spec in local:
        transform = _transforms(spec, spec.event_start)[0]
        rectangle = transform["rectangle"]
        expected = 0.75 if spec.local_tier == "stress_75pct" else 0.5
        assert rectangle["width"] * rectangle["height"] == expected
    for spec in (item for item in PILOT_SPECS if item.cause == "registration_or_order_fault"):
        assert _final_indices(spec)[12:18] == list(reversed(range(spec.source_start + 12, spec.source_start + 18)))
