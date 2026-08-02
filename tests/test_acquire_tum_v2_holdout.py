from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from scripts.acquire_tum_v2_holdout import (
    AcquisitionError,
    DATASET_NAME,
    _safe_member,
    _tree_manifest,
    _validate_archive,
)


def _png(*, bit_depth: int, color_type: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + (640).to_bytes(4, "big")
        + (480).to_bytes(4, "big")
        + bytes((bit_depth, color_type, 0, 0, 0))
        + b"\x00\x00\x00\x00"
    )


def _raw_tree(tmp_path: Path) -> Path:
    root = tmp_path / DATASET_NAME
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    rgb_rows: list[str] = []
    depth_rows: list[str] = []
    gt_rows: list[str] = []
    for index in range(30):
        timestamp = f"1.{index:09d}"
        rgb_name = f"rgb/{index}.png"
        depth_name = f"depth/{index}.png"
        (root / rgb_name).write_bytes(_png(bit_depth=8, color_type=2))
        (root / depth_name).write_bytes(_png(bit_depth=16, color_type=0))
        rgb_rows.append(f"{timestamp} {rgb_name}")
        depth_rows.append(f"{timestamp} {depth_name}")
        gt_rows.append(f"{timestamp} 0 0 0 0 0 0 1")
    (root / "rgb.txt").write_text("\n".join(rgb_rows) + "\n", encoding="utf-8")
    (root / "depth.txt").write_text("\n".join(depth_rows) + "\n", encoding="utf-8")
    (root / "groundtruth.txt").write_text("\n".join(gt_rows) + "\n", encoding="utf-8")
    return root


def test_tree_manifest_validates_tum_stream_layout(tmp_path: Path) -> None:
    manifest = _tree_manifest(_raw_tree(tmp_path))

    assert manifest["regular_file_count"] == 63
    assert manifest["rgb"]["rows"] == 30
    assert manifest["depth"]["rows"] == 30
    assert manifest["groundtruth"]["rows"] == 30
    assert manifest["rgb_frames_with_depth_within_20ms"] == 30


def test_safe_member_rejects_path_escape() -> None:
    member = tarfile.TarInfo(f"{DATASET_NAME}/../escape.txt")

    with pytest.raises(AcquisitionError, match="escapes"):
        _safe_member(member)


def test_archive_validation_requires_expected_tum_members(tmp_path: Path) -> None:
    archive = tmp_path / "scene.tgz"
    with tarfile.open(archive, "w:gz") as stream:
        for name in (
            DATASET_NAME,
            f"{DATASET_NAME}/rgb",
            f"{DATASET_NAME}/depth",
            f"{DATASET_NAME}/rgb.txt",
            f"{DATASET_NAME}/depth.txt",
            f"{DATASET_NAME}/groundtruth.txt",
        ):
            member = tarfile.TarInfo(name)
            member.type = tarfile.DIRTYPE if name.endswith((DATASET_NAME, "/rgb", "/depth")) else tarfile.REGTYPE
            member.size = 0
            stream.addfile(member)

    report = _validate_archive(archive)

    assert report["required_entries_present"] == ["depth", "depth.txt", "groundtruth.txt", "rgb", "rgb.txt"]
