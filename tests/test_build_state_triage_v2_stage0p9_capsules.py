from __future__ import annotations

from scripts.build_state_triage_v2_stage0p9_capsules import FRAME_COUNT, SOURCE_STARTS, Spec, _assert_specs, _specs, _transform, _window


def test_predeclared_selection_has_twelve_disjoint_30_frame_windows() -> None:
    specs = tuple(Spec(source_start=start, ordinal=index) for index, start in enumerate(SOURCE_STARTS))

    _assert_specs(specs)
    assert len(specs) == 12
    assert all(len(_window(spec)) == FRAME_COUNT for spec in specs)
    for index, spec in enumerate(specs):
        assert all(_window(spec).isdisjoint(_window(other)) for other in specs[index + 1 :])


def test_hashed_tile_recipe_is_fixed_partial_and_nonperiodic() -> None:
    transform = _transform(Spec(source_start=SOURCE_STARTS[0], ordinal=0))

    assert transform["type"] == "hashed_tile_occlusion"
    assert transform["rectangle"]["width"] * transform["rectangle"]["height"] == 0.75
    assert transform["tile_size_pixels"] == 16
    assert transform["seed"] == 1909


def test_selection_rejects_any_nonfresh_start() -> None:
    capacity = {"rgbd_dataset_freiburg2_desk": list(SOURCE_STARTS[:-1])}

    try:
        _specs(capacity)
    except ValueError as error:
        assert "not fresh" in str(error)
    else:
        raise AssertionError("a missing predeclared start must be rejected")
