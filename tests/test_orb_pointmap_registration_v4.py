from __future__ import annotations

import numpy as np
import pytest

from stateguard3r.orb_pointmap_registration_v4 import ORBPointmapRegistrationError, _nearest_pairs


def test_nearest_pair_sampling_uses_anchor_and_current_in_declared_direction() -> None:
    anchor = np.zeros((4, 5, 3)); current = np.zeros((4, 5, 3))
    anchor[2, 1] = (1.0, 2.0, 3.0); current[1, 3] = (4.0, 5.0, 6.0)
    left, right = _nearest_pairs(anchor, current, [(1.2, 1.6)], [(2.6, 1.2)])
    np.testing.assert_allclose(left, [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(right, [[4.0, 5.0, 6.0]])


def test_nearest_pair_sampling_rejects_shape_mismatch_and_filters_invalid_coordinates() -> None:
    with pytest.raises(ORBPointmapRegistrationError, match="shapes differ"):
        _nearest_pairs(np.zeros((2, 2, 3)), np.zeros((3, 2, 3)), [(0, 0)], [(0, 0)])
    left, right = _nearest_pairs(np.zeros((2, 2, 3)), np.zeros((2, 2, 3)), [(-1, 0)], [(0, 0)])
    assert left.shape == right.shape == (0, 3)
