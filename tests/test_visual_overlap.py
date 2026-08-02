from __future__ import annotations

import numpy as np
import pytest

from stateguard3r.visual_overlap import (
    VisualOverlapConfig,
    VisualOverlapError,
    grid_endpoint_coverage,
    gt_depth_reprojection_overlap,
    normalized_tensor_to_uint8_rgb,
    online_visual_correspondence_coverage,
    online_visual_correspondence_series,
)


class _Keypoint:
    def __init__(self, x: float, y: float) -> None:
        self.pt = (x, y)


class _Match:
    def __init__(self, query: int, train: int, distance: float) -> None:
        self.queryIdx = query
        self.trainIdx = train
        self.distance = distance


class _Matcher:
    def knnMatch(self, query: np.ndarray, train: np.ndarray, *, k: int) -> list[list[_Match]]:
        assert k == 2
        return [
            [_Match(index, index, 10.0), _Match(index, (index + 1) % len(query), 20.0)]
            for index in range(len(query))
        ]


class _ORB:
    def __init__(self, points: list[_Keypoint], *, descriptors: bool) -> None:
        self._points = points
        self._descriptors = descriptors
        self._calls = 0

    def detectAndCompute(self, image: np.ndarray, mask: object) -> tuple[list[_Keypoint], np.ndarray | None]:
        del image, mask
        self._calls += 1
        if not self._descriptors:
            return self._points, None
        descriptor_value = self._calls
        return self._points, np.full((len(self._points), 32), descriptor_value, dtype=np.uint8)


class _OCL:
    def __init__(self) -> None:
        self.enabled = True

    def setUseOpenCL(self, enabled: bool) -> None:
        self.enabled = enabled

    def useOpenCL(self) -> bool:
        return self.enabled


class _FakeCV2:
    NORM_HAMMING = 6
    FM_RANSAC = 8

    def __init__(self, *, descriptors: bool = True, geometric_ok: bool = True) -> None:
        self.ocl = _OCL()
        self.descriptors = descriptors
        self.geometric_ok = geometric_ok
        self.threads = 99
        self.rng_seed = None
        self.points = [
            _Keypoint(5 + 20 * column, 5 + 20 * row)
            for row in range(3)
            for column in range(4)
        ]

    def setNumThreads(self, count: int) -> None:
        self.threads = count

    def setRNGSeed(self, seed: int) -> None:
        self.rng_seed = seed

    def ORB_create(self, *, nfeatures: int) -> _ORB:
        assert nfeatures == 2000
        return _ORB(self.points, descriptors=self.descriptors)

    def BFMatcher(self, norm: int, *, crossCheck: bool) -> _Matcher:
        assert norm == self.NORM_HAMMING
        assert crossCheck is False
        return _Matcher()

    def findFundamentalMat(
        self, previous: np.ndarray, current: np.ndarray, method: int, limit: float
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        assert previous.shape == current.shape == (12, 2)
        assert method == self.FM_RANSAC
        assert limit == 1.0
        if not self.geometric_ok:
            return None, None
        return np.eye(3), np.ones((12, 1), dtype=np.uint8)


def _normalized_image(value: float = 0.0) -> np.ndarray:
    return np.full((3, 80, 100), value, dtype=np.float32)


def test_normalized_conversion_copies_and_preserves_rgb_channel_order() -> None:
    source = np.array(
        [
            [[-1.0, 1.0]],
            [[0.0, -1.0]],
            [[1.0, 0.0]],
        ],
        dtype=np.float32,
    )

    actual = normalized_tensor_to_uint8_rgb(source)

    assert actual.shape == (1, 2, 3)
    assert actual.dtype == np.uint8
    assert actual.tolist() == [[[0, 128, 255], [255, 0, 128]]]
    actual[0, 0, 0] = 99
    assert source[0, 0, 0] == -1.0


@pytest.mark.parametrize("points,expected", [([], 0.0), ([(0.0, 0.0), (99.0, 79.0)], 2 / 4)])
def test_grid_endpoint_coverage_handles_blank_and_corner_geometry(
    points: list[tuple[float, float]], expected: float
) -> None:
    actual = grid_endpoint_coverage(
        points,
        image_width=100,
        image_height=80,
        grid_columns=2,
        grid_rows=2,
    )
    assert actual == pytest.approx(expected)


def test_orb_ransac_coverage_is_finite_deterministic_and_causal() -> None:
    fake = _FakeCV2()
    config = VisualOverlapConfig()

    first = online_visual_correspondence_coverage(
        _normalized_image(-0.2),
        _normalized_image(0.2),
        config=config,
        cv2_module=fake,
        reference_frame_id=4,
        frame_id=5,
    )
    second = online_visual_correspondence_coverage(
        _normalized_image(-0.2),
        _normalized_image(0.2),
        config=config,
        cv2_module=fake,
        reference_frame_id=4,
        frame_id=5,
    )

    assert first == second
    assert first.status == "ok"
    assert first.score == pytest.approx(0.25)
    assert first.inlier_ratio == 1.0
    assert fake.threads == 1
    assert fake.rng_seed == 0
    assert fake.ocl.enabled is False

    series_a = online_visual_correspondence_series(
        [_normalized_image(-0.5), _normalized_image(0.0), _normalized_image(0.5)],
        config=config,
        cv2_module=_FakeCV2(),
    )
    series_b = online_visual_correspondence_series(
        [_normalized_image(-0.5), _normalized_image(0.0), _normalized_image(-1.0)],
        config=config,
        cv2_module=_FakeCV2(),
    )
    assert series_a[0] is None
    assert series_a[1] == series_b[1]


@pytest.mark.parametrize(
    "fake,status",
    [(_FakeCV2(descriptors=False), "descriptor_insufficient"), (_FakeCV2(geometric_ok=False), "geometric_verification_failed")],
)
def test_low_texture_and_geometry_failure_are_explicit_finite_zeros(
    fake: _FakeCV2, status: str
) -> None:
    result = online_visual_correspondence_coverage(
        _normalized_image(), _normalized_image(), cv2_module=fake
    )
    assert result.status == status
    assert result.score == 0.0
    assert result.inliers == 0


def test_offline_gt_depth_reprojection_is_bidirectional_and_isolated() -> None:
    depth = np.full((8, 8), 5000, dtype=np.uint16)
    intrinsics = {"fx": 2.0, "fy": 2.0, "cx": 3.5, "cy": 3.5}
    identity = np.eye(4)

    same = gt_depth_reprojection_overlap(depth, depth, identity, identity, intrinsics, stride=1)
    far_pose = np.eye(4)
    far_pose[0, 3] = 100.0
    disjoint = gt_depth_reprojection_overlap(depth, depth, identity, far_pose, intrinsics, stride=1)

    assert same.score == same.reference_to_target == same.target_to_reference == 1.0
    assert disjoint.score == 0.0
    assert disjoint.reference_consistent_samples == disjoint.target_consistent_samples == 0


def test_visual_overlap_rejects_nonfinite_or_non_normalized_model_input() -> None:
    with pytest.raises(VisualOverlapError, match="normalized"):
        normalized_tensor_to_uint8_rgb(np.full((3, 2, 2), 2.0))
    with pytest.raises(VisualOverlapError, match="finite"):
        normalized_tensor_to_uint8_rgb(np.full((3, 2, 2), np.nan))
