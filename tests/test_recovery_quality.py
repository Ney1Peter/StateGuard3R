from __future__ import annotations

import math

import numpy as np
import pytest

from stateguard3r.recovery_quality import (
    RecoveryQualityError,
    evaluate_tail,
    positive_effect,
    proper_umeyama_sim3,
    quaternion_xyzw_to_rotation,
    scene_stratified_bootstrap_lower_bound,
    trajectory_arrays,
)


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def _trajectory(positions: np.ndarray, rotations: np.ndarray) -> dict[str, object]:
    return {
        "frames": [
            {
                "frame_id": index,
                "camera_to_reference": np.block(
                    [[rotations[index], positions[index, :, None]], [np.zeros((1, 3)), np.ones((1, 1))]]
                ).tolist(),
            }
            for index in range(len(positions))
        ]
    }


def test_proper_umeyama_recovers_non_reflective_similarity() -> None:
    predicted = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.5, 1.0, 0.5], [2.0, 1.5, 1.0]])
    rotation = _rotation_z(0.35)
    target = 2.5 * (predicted @ rotation.T) + np.asarray([1.0, -2.0, 3.0])
    transform = proper_umeyama_sim3(predicted, target)
    assert transform.scale == pytest.approx(2.5)
    assert transform.rotation == pytest.approx(rotation)
    assert transform.translation == pytest.approx([1.0, -2.0, 3.0])
    assert transform.apply_positions(predicted) == pytest.approx(target)


def test_tail_metrics_fit_only_prefix_not_tail() -> None:
    target_positions = np.asarray([[float(index), 0.0, 0.0] for index in range(8)])
    predicted_positions = target_positions.copy()
    predicted_positions[5:, 1] = 2.0
    rotations = np.repeat(np.eye(3)[None, ...], 8, axis=0)
    metrics = evaluate_tail(
        predicted_positions,
        rotations,
        target_positions,
        rotations,
        prefix_frame_ids=(0, 1, 2, 3, 4),
        tail_frame_ids=(5, 6, 7),
    )
    assert metrics.ate_rmse_m == pytest.approx(2.0)
    assert metrics.rpe_translation_rmse_m == pytest.approx(0.0)
    assert metrics.rpe_rotation_rmse_deg == pytest.approx(0.0)


def test_tail_metrics_rejects_degenerate_prefix_and_nonconsecutive_tail() -> None:
    positions = np.zeros((5, 3), dtype=np.float64)
    rotations = np.repeat(np.eye(3)[None, ...], 5, axis=0)
    with pytest.raises(RecoveryQualityError, match="zero variance"):
        evaluate_tail(
            positions,
            rotations,
            positions,
            rotations,
            prefix_frame_ids=(0, 1, 2),
            tail_frame_ids=(3, 4),
        )
    positions[:, 0] = np.arange(5)
    with pytest.raises(RecoveryQualityError, match="consecutive"):
        evaluate_tail(
            positions,
            rotations,
            positions,
            rotations,
            prefix_frame_ids=(0, 1, 2),
            tail_frame_ids=(4,),
        )


def test_trajectory_arrays_and_quaternion_validation() -> None:
    positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    rotations = np.repeat(np.eye(3)[None, ...], 2, axis=0)
    actual_positions, actual_rotations = trajectory_arrays(_trajectory(positions, rotations))
    assert actual_positions == pytest.approx(positions)
    assert actual_rotations == pytest.approx(rotations)
    assert quaternion_xyzw_to_rotation([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]) == pytest.approx(
        _rotation_z(math.pi / 2)
    )
    with pytest.raises(RecoveryQualityError, match="zero"):
        quaternion_xyzw_to_rotation([0.0, 0.0, 0.0, 0.0])


def test_effect_and_scene_stratified_bootstrap_are_deterministic() -> None:
    assert positive_effect(10.0, 8.0, name="ATE") == pytest.approx(0.2)
    with pytest.raises(RecoveryQualityError, match="positive"):
        positive_effect(0.0, 0.0, name="ATE")
    effects = {"scene-a": [0.1, 0.3, 0.2], "scene-b": [0.2, 0.4, 0.3]}
    first = scene_stratified_bootstrap_lower_bound(effects, seed=20260806, samples=200)
    second = scene_stratified_bootstrap_lower_bound(effects, seed=20260806, samples=200)
    assert first == second
    assert 0.1 <= first <= 0.4
