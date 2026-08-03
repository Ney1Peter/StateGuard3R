from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from stateguard3r.detection_v3 import DetectorV3Error, hybridize_v2_continuous_scores
from stateguard3r.timestamp_order_v3 import (
    TimestampOrderError,
    capture_timestamp_records,
    parse_tum_rgb_capture_index,
    timestamp_order_sidecar,
    validate_timestamp_order_sidecar,
    write_timestamp_order_sidecar_atomic,
)


def _raw_rgb_fixture(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    root = tmp_path / "dataset"
    rgb = root / "rgb"
    rgb.mkdir(parents=True)
    paths = [rgb / f"{index}.png" for index in range(4)]
    for index, path in enumerate(paths):
        path.write_bytes(f"rgb-{index}".encode("ascii"))
    listing = root / "rgb.txt"
    listing.write_text(
        "# TUM RGB listing\n"
        "10.000 rgb/0.png\n"
        "10.010 rgb/1.png\n"
        "10.020 rgb/2.png\n"
        "10.030 rgb/3.png\n",
        encoding="utf-8",
    )
    return root, listing, paths


def _sidecar(tmp_path: Path, order: list[int]) -> dict[str, object]:
    root, listing, paths = _raw_rgb_fixture(tmp_path)
    captures, provenance = capture_timestamp_records(
        [paths[index] for index in order], rgb_txt=listing, dataset_root=root
    )
    return timestamp_order_sidecar(captures, provenance=provenance)


def test_raw_rgb_listing_binds_paths_and_strictly_increasing_order(tmp_path: Path) -> None:
    root, listing, paths = _raw_rgb_fixture(tmp_path)
    index = parse_tum_rgb_capture_index(listing, dataset_root=root)
    assert list(index) == paths
    captures, provenance = capture_timestamp_records(paths[:3], rgb_txt=listing, dataset_root=root)
    sidecar = timestamp_order_sidecar(captures, provenance=provenance)
    assert validate_timestamp_order_sidecar(sidecar) == [None, False, False]
    assert sidecar["records"][1]["rgb_capture_timestamp_text"] == "10.010"


def test_reverse_and_duplicate_capture_timestamps_are_hard_violations(tmp_path: Path) -> None:
    reversed_sidecar = _sidecar(tmp_path / "reverse", [0, 2, 1, 3])
    duplicate_sidecar = _sidecar(tmp_path / "duplicate", [0, 1, 1, 3])
    assert validate_timestamp_order_sidecar(reversed_sidecar) == [None, False, True, False]
    assert validate_timestamp_order_sidecar(duplicate_sidecar) == [None, False, True, False]


def test_timestamp_order_is_future_invariant(tmp_path: Path) -> None:
    prefix = _sidecar(tmp_path / "prefix", [0, 1, 2])
    extended = _sidecar(tmp_path / "extended", [0, 1, 2, 1])
    assert validate_timestamp_order_sidecar(prefix) == validate_timestamp_order_sidecar(extended)[:3]


def test_missing_or_decreasing_raw_listing_is_rejected(tmp_path: Path) -> None:
    root, listing, paths = _raw_rgb_fixture(tmp_path)
    unlisted = root / "rgb" / "unlisted.png"
    unlisted.write_bytes(b"unlisted")
    with pytest.raises(TimestampOrderError, match="absent"):
        capture_timestamp_records([paths[0], unlisted], rgb_txt=listing, dataset_root=root)
    listing.write_text("10.010 rgb/1.png\n10.000 rgb/0.png\n", encoding="utf-8")
    with pytest.raises(TimestampOrderError, match="decrease"):
        parse_tum_rgb_capture_index(listing, dataset_root=root)


def test_sidecar_rejects_injected_gt_or_predicate_mutation(tmp_path: Path) -> None:
    sidecar = _sidecar(tmp_path, [0, 2, 1])
    injected = copy.deepcopy(sidecar)
    injected["records"][1]["groundtruth"] = {"timestamp": "pretend"}
    with pytest.raises(TimestampOrderError, match="schema"):
        validate_timestamp_order_sidecar(injected)
    mutated = copy.deepcopy(sidecar)
    mutated["records"][2]["timestamp_order_violation"] = False
    with pytest.raises(TimestampOrderError, match="replay"):
        validate_timestamp_order_sidecar(mutated)


def test_atomic_sidecar_publication(tmp_path: Path) -> None:
    sidecar = _sidecar(tmp_path / "source", [0, 1, 2])
    destination = tmp_path / "out" / "timestamp-order.json"
    written = write_timestamp_order_sidecar_atomic(destination, sidecar)
    assert written == destination
    assert destination.read_text(encoding="utf-8").endswith("\n")


def test_hybrid_uses_hard_timestamp_channel_without_retuning_threshold(tmp_path: Path) -> None:
    sidecar = _sidecar(tmp_path, [0, 2, 1, 3])
    result = hybridize_v2_continuous_scores(
        np.asarray([0.2, 0.3, 0.4, 2.1]), continuous_threshold=2.0, timestamp_sidecar=sidecar
    )
    assert result["alarms"].tolist() == [False, False, True, True]
    assert result["hybrid_scores"].tolist() == [0.2, 0.3, 3.0, 2.1]
    assert result["attribution"][2] == {
        "frame_id": 2,
        "continuous_alarm": False,
        "timestamp_order_alarm": True,
        "hybrid_alarm": True,
    }


def test_hybrid_rejects_nonfinite_scores_and_unavailable_order(tmp_path: Path) -> None:
    sidecar = _sidecar(tmp_path, [0, 1, 2])
    with pytest.raises(DetectorV3Error, match="finite"):
        hybridize_v2_continuous_scores([0.0, float("nan"), 0.0], continuous_threshold=1.0, timestamp_sidecar=sidecar)
    unavailable = copy.deepcopy(sidecar)
    unavailable["records"][1]["timestamp_order_violation"] = None
    with pytest.raises(DetectorV3Error, match="invalid timestamp sidecar"):
        hybridize_v2_continuous_scores([0.0, 0.0, 0.0], continuous_threshold=1.0, timestamp_sidecar=unavailable)
