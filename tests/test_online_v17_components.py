from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stateguard3r.online_detector_v17 import (
    FrozenDetectorV17Config,
    IncrementalOnlinePrefixDetectorV17,
    OnlineDetectorV17Error,
)
from stateguard3r.online_visual_overlap_v17 import (
    OnlineVisualOverlapConfigV17,
    OnlineVisualOverlapV17Error,
    normalized_tensor_to_uint8_rgb_v17,
)
from stateguard3r.timestamp_order_v17 import (
    TimestampOrderV17Error,
    capture_timestamp_records_v17,
    timestamp_order_sidecar_v17,
)


ROOT = Path(__file__).resolve().parents[1]


def _row(frame_id: int, *, order: bool = False) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "overlap": None if frame_id == 0 else 0.5,
        "pose_jump": float(frame_id),
        "geometric_residual": float(frame_id),
        "update_magnitude": float(frame_id),
        "reliability": 0.9,
        "uncertainty_u": 0.1,
        "timestamp_order_alarm": order,
    }


def test_v17_detector_is_current_scalar_only_and_commit_gated() -> None:
    config = FrozenDetectorV17Config.from_mapping(
        json.loads((ROOT / "outputs" / "formal-v3-calibration-0001" / "formal-config.json").read_text())
    )
    detector = IncrementalOnlinePrefixDetectorV17(config)
    decision = detector.observe(_row(0))
    assert decision.hybrid_alarm is False
    with pytest.raises(OnlineDetectorV17Error, match="not committed"):
        detector.observe(_row(1))
    detector.commit(_row(0))
    ordered = detector.observe(_row(1, order=True))
    assert ordered.timestamp_order_alarm is True and ordered.hybrid_alarm is True
    detector.commit(_row(1, order=True))
    with pytest.raises(OnlineDetectorV17Error, match="schema"):
        detector.observe({"frame_id": 2})


def test_v17_rgb_projection_is_owned_and_rejects_invalid_values() -> None:
    source = np.zeros((1, 3, 2, 3), dtype=np.float32)
    copied = normalized_tensor_to_uint8_rgb_v17(source)
    source[...] = 1.0
    assert copied.shape == (2, 3, 3) and int(copied[0, 0, 0]) == 128
    with pytest.raises(OnlineVisualOverlapV17Error, match="normalized"):
        normalized_tensor_to_uint8_rgb_v17(np.full((3, 2, 2), 2.0))
    assert OnlineVisualOverlapConfigV17().minimum_matches == 12


def test_v17_timestamp_binds_only_raw_listing_and_selected_paths(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    rgb = root / "rgb"
    rgb.mkdir(parents=True)
    first, second = rgb / "a.png", rgb / "b.png"
    first.write_bytes(b"a"); second.write_bytes(b"b")
    listing = root / "rgb.txt"
    listing.write_text("1.0 rgb/a.png\n2.0 rgb/b.png\n", encoding="utf-8")
    for item in (first, second, listing):
        item.chmod(0o444)
    captures, provenance = capture_timestamp_records_v17([first, second], rgb_txt=listing, dataset_root=root)
    sidecar = timestamp_order_sidecar_v17(captures, provenance=provenance)
    assert sidecar["records"][0]["timestamp_order_violation"] is None
    assert sidecar["records"][1]["timestamp_order_violation"] is False
    with pytest.raises(TimestampOrderV17Error, match="unique"):
        capture_timestamp_records_v17([first, first], rgb_txt=listing, dataset_root=root)
