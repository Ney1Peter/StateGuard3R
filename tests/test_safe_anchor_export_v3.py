from __future__ import annotations

import numpy as np
import pytest

from stateguard3r.safe_anchor_export_v3 import SafeAnchorExportError, SafeAnchorSE3Export


class _Pose:
    shape = (1, 7)

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float64).reshape(1, 4, 4).copy()

    def clone(self) -> "_Pose":
        return _Pose(self.matrix)


def _matrix(*, translation: tuple[float, float, float] = (0.0, 0.0, 0.0), rotation: np.ndarray | None = None) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = translation
    if rotation is not None:
        result[:3, :3] = rotation
    return result


def _prediction(matrix: np.ndarray) -> dict[str, object]:
    return {"camera_pose": _Pose(matrix), "pts3d_in_self_view": object(), "pts3d_in_other_view": object()}


def _policy() -> SafeAnchorSE3Export:
    return SafeAnchorSE3Export(
        pose_encoding_to_camera=lambda pose: pose.matrix.copy(),
        camera_to_pose_encoding=lambda matrix: _Pose(matrix),
        inverse=np.linalg.inv,
    )


def test_safe_anchor_uses_preregistered_left_world_increment_and_preserves_candidate() -> None:
    policy = _policy()
    first = _prediction(_matrix())
    second = _prediction(_matrix(translation=(1.0, 0.0, 0.0)))
    alarmed = _prediction(_matrix(translation=(99.0, 5.0, 0.0)))
    policy.commit_real(0, first)
    policy.commit_real(1, second)
    exported, evidence = policy.fallback(2, alarmed)
    assert evidence.action == "export_safe_anchor_se3_constant_velocity"
    assert evidence.anchor_frame_ids == (0, 1)
    np.testing.assert_allclose(exported["camera_pose"].matrix[0], _matrix(translation=(2.0, 0.0, 0.0)))  # type: ignore[index,union-attr]
    np.testing.assert_allclose(alarmed["camera_pose"].matrix[0], _matrix(translation=(99.0, 5.0, 0.0)))  # type: ignore[index,union-attr]


def test_consecutive_alarm_advances_from_previous_export_and_clear_export_resets_history() -> None:
    policy = _policy()
    policy.commit_real(0, _prediction(_matrix()))
    policy.commit_real(1, _prediction(_matrix(translation=(1.0, 0.0, 0.0))))
    first_fallback, _ = policy.fallback(2, _prediction(_matrix(translation=(100.0, 0.0, 0.0))))
    second_fallback, _ = policy.fallback(3, _prediction(_matrix(translation=(101.0, 0.0, 0.0))))
    np.testing.assert_allclose(first_fallback["camera_pose"].matrix[0], _matrix(translation=(2.0, 0.0, 0.0)))  # type: ignore[index,union-attr]
    np.testing.assert_allclose(second_fallback["camera_pose"].matrix[0], _matrix(translation=(3.0, 0.0, 0.0)))  # type: ignore[index,union-attr]
    clear, action = policy.commit_real(4, _prediction(_matrix(translation=(4.0, 0.0, 0.0))))
    assert action.action == "export_real_camera_pose"
    assert clear["camera_pose"].matrix[0, 0, 3] == 4.0  # type: ignore[index,union-attr]
    assert policy.exported_frame_ids == (0, 1, 2, 3, 4)


def test_rotation_formula_is_left_multiplication_not_body_increment() -> None:
    policy = _policy()
    ninety_z = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    first = _matrix(translation=(1.0, 0.0, 0.0))
    second = _matrix(translation=(1.0, 1.0, 0.0), rotation=ninety_z)
    policy.commit_real(0, _prediction(first))
    policy.commit_real(1, _prediction(second))
    exported, _ = policy.fallback(2, _prediction(_matrix(translation=(40.0, 0.0, 0.0))) )
    expected = (second @ np.linalg.inv(first)) @ second
    np.testing.assert_allclose(exported["camera_pose"].matrix[0], expected)  # type: ignore[index,union-attr]


def test_fails_closed_without_two_safe_anchors_or_contiguous_frame_ids() -> None:
    policy = _policy()
    policy.commit_real(0, _prediction(_matrix()))
    with pytest.raises(SafeAnchorExportError, match="fewer than two"):
        policy.fallback(1, _prediction(_matrix()))
    with pytest.raises(SafeAnchorExportError, match="contiguous"):
        policy.commit_real(2, _prediction(_matrix()))
