from __future__ import annotations

from collections import Counter

from scripts.build_state_triage_v2_stage0p8_capsules import CALIBRATION_COUNT, FINAL_QUOTAS, FRAME_COUNT, TOTAL_COUNT, _allocate, _final_indices, _transforms


def _window(spec) -> set[int]:
    return set(range(spec.source_start, spec.source_start + FRAME_COUNT))


def test_allocation_reserves_calibration_and_final_windows_before_any_forward() -> None:
    specs = _allocate(TOTAL_COUNT * FRAME_COUNT)

    assert len(specs) == TOTAL_COUNT
    assert Counter((spec.phase, spec.cause) for spec in specs) == Counter({("calibration", "transient_local_content"): CALIBRATION_COUNT, **{("final", cause): count for cause, count in FINAL_QUOTAS.items()}})
    for index, spec in enumerate(specs):
        assert all(_window(spec).isdisjoint(_window(other)) for other in specs[index + 1 :])


def test_textured_local_recipe_is_fixed_partial_and_registration_is_reversed() -> None:
    specs = _allocate(TOTAL_COUNT * FRAME_COUNT)
    local = [spec for spec in specs if spec.cause == "transient_local_content"]
    assert len(local) == CALIBRATION_COUNT + FINAL_QUOTAS["transient_local_content"]
    for spec in local:
        transform = _transforms(spec, spec.event_start)[0]
        assert transform["type"] == "checkerboard_occlusion"
        assert transform["rectangle"]["width"] * transform["rectangle"]["height"] == 0.75
        assert transform["tile_size_pixels"] == 16
        assert transform["fills"] == [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
    for spec in (item for item in specs if item.cause == "registration_or_order_fault"):
        assert _final_indices(spec)[12:18] == list(reversed(range(spec.source_start + 12, spec.source_start + 18)))


def test_normal_events_are_predeclared_at_the_prefix() -> None:
    specs = _allocate(TOTAL_COUNT * FRAME_COUNT)
    normal = [spec for spec in specs if spec.cause == "normal_novelty"]
    assert len(normal) == FINAL_QUOTAS["normal_novelty"]
    assert all((spec.event_start, spec.event_end) == (0, 4) for spec in normal)
