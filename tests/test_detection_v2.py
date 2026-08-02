from __future__ import annotations

import numpy as np
import pytest

from stateguard3r.detection_v2 import (
    DetectorV2Error,
    V2ScoringConfig,
    candidate_grid,
    compute_v2_scores,
    derive_development_scale_floors,
    directional_causal_component,
    select_constrained_threshold,
    validate_online_overlap,
)


def _records(length: int = 6) -> list[dict[str, object]]:
    return [
        {
            "frame_id": index,
            "geometric_residual": 1.0 if index < length - 1 else 4.0,
            "pose_jump": 1.0,
            "global_state_delta": 1.0 if index < length - 1 else 5.0,
            "reliability": 0.9 if index < length - 1 else 0.2,
            "overlap": None if index == 0 else 0.9,
        }
        for index in range(length)
    ]


def _floors() -> dict[str, float]:
    return {
        "geometric_residual": 1.0,
        "pose_jump": 1.0,
        "update_magnitude": 1.0,
        "overlap": 1.0,
        "reliability": 1.0,
    }


def test_directional_component_is_causal_one_sided_and_warmup_neutral() -> None:
    values = [1.0, 1.0, 1.0, 3.0, 0.0]
    high, high_warmup = directional_causal_component(
        values,
        direction="high",
        window=3,
        min_history=3,
        scale_floor=1.0,
        max_z=5.0,
    )
    low, low_warmup = directional_causal_component(
        values,
        direction="low",
        window=3,
        min_history=3,
        scale_floor=1.0,
        max_z=5.0,
    )

    np.testing.assert_allclose(high[:3], 0.0)
    np.testing.assert_allclose(low[:3], 0.0)
    assert high_warmup.tolist() == low_warmup.tolist() == [True, True, True, False, False]
    assert high[3] == pytest.approx(2.0)
    assert low[3] == 0.0
    assert high[4] == 0.0
    assert low[4] > 0.0

    first, _ = directional_causal_component(
        [1.0, 1.0, 1.0, 3.0, 5.0],
        direction="high",
        window=3,
        min_history=3,
        scale_floor=1.0,
        max_z=5.0,
    )
    second, _ = directional_causal_component(
        [1.0, 1.0, 1.0, 3.0, -1000.0],
        direction="high",
        window=3,
        min_history=3,
        scale_floor=1.0,
        max_z=5.0,
    )
    np.testing.assert_allclose(first[:4], second[:4])


def test_scale_floors_use_only_positive_clean_prefix_differences() -> None:
    development = {
        "a": [
            {"geometric_residual": value, "global_state_delta": value, "reliability": 0.9}
            for value in (1.0, 1.1, 1.3, 1.6)
        ],
        "b": [
            {"geometric_residual": value, "global_state_delta": value, "reliability": 0.9}
            for value in (2.0, 2.2, 2.6, 3.2)
        ],
    }

    floors = derive_development_scale_floors(development, clean_prefix_frames=4)

    assert floors["geometric_residual"] == pytest.approx(0.15)
    assert floors["update_magnitude"] == pytest.approx(0.15)
    assert floors["reliability"] == pytest.approx(1e-6)
    assert floors["overlap"] == pytest.approx(1e-6)


def test_v2_combined_uses_finite_component_mean_and_requires_real_overlap() -> None:
    records = _records()
    validate_online_overlap(records)
    result = compute_v2_scores(
        records,
        config=V2ScoringConfig(window=3, min_history=3, max_z=5.0),
        scale_floors=_floors(),
        seed=7,
        require_online_overlap=True,
    )
    combined = result["methods"]["combined"]

    assert combined["used_signals"] == [
        "geometric_residual",
        "pose_jump",
        "update_magnitude",
        "overlap",
        "reliability",
    ]
    assert combined["finite_component_count"].tolist()[0] == 4
    assert combined["finite_component_count"].tolist()[1:] == [5, 5, 5, 5, 5]
    assert np.isfinite(combined["scores"]).all()
    assert combined["scores"][-1] > 0.0
    assert result["methods"]["random"]["scores"].tolist() == compute_v2_scores(
        records,
        config=V2ScoringConfig(window=3, min_history=3, max_z=5.0),
        scale_floors=_floors(),
        seed=7,
        require_online_overlap=True,
    )["methods"]["random"]["scores"].tolist()

    broken = [dict(row) for row in records]
    broken[3]["overlap"] = None
    with pytest.raises(DetectorV2Error, match="requires finite overlap"):
        validate_online_overlap(broken)


def test_threshold_selection_obeys_hard_fpr_and_streak_constraint() -> None:
    selection = select_constrained_threshold(
        {"run": [0, 0, 1, 1]},
        {"run": [0.9, 0.1, 0.8, 0.7]},
        {"run": [True, True, True, True]},
        max_pooled_fpr=0.0,
        max_per_run_fp_streak=0,
    )

    selected = selection["selected"]
    assert selected["pooled"]["false_positive_rate"] == 0.0
    assert selected["max_per_run_false_positive_streak"] == 0
    assert selected["threshold"] > 0.9
    assert any(not row["feasible"] for row in selection["candidates"])


def test_v2_rejects_missing_floor_nonfinite_score_and_invalid_grid() -> None:
    with pytest.raises(DetectorV2Error, match="exactly canonical"):
        compute_v2_scores(_records(), config=V2ScoringConfig(), scale_floors={})
    with pytest.raises(DetectorV2Error, match="scores/mask"):
        select_constrained_threshold(
            {"run": [0, 1]}, {"run": [0.0, np.nan]}, {"run": [True, True]}
        )
    assert len(candidate_grid()) == 8
    assert len({(item.window, item.min_history, item.max_z) for item in candidate_grid()}) == 8
