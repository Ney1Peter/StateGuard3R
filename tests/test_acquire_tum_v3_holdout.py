from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

import scripts.acquire_tum_v3_holdout as acquire
from scripts.acquire_tum_v3_holdout import (
    ACQUISITION_OUTPUT_NAME,
    AcquisitionError,
    CANONICAL_URL,
    DATASET_NAME,
    EXPECTED_ARCHIVE_BYTES,
    MAX_ARCHIVE_AND_RAW_TREE_BYTES,
    _safe_member,
    _source_response_search,
    _tree_manifest,
    _validate_archive,
    _validate_groundtruth,
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


def test_v3_candidate_and_budget_are_fixed() -> None:
    assert DATASET_NAME == "rgbd_dataset_freiburg2_desk"
    assert CANONICAL_URL.endswith("/freiburg2/rgbd_dataset_freiburg2_desk.tgz")
    assert EXPECTED_ARCHIVE_BYTES == 1_893_351_095
    assert MAX_ARCHIVE_AND_RAW_TREE_BYTES == 5 * 1024 * 1024 * 1024
    assert ACQUISITION_OUTPUT_NAME == "formal-v3-data-0001"


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


def test_archive_validation_requires_expected_tum_members_and_reports_budget(
    tmp_path: Path,
) -> None:
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
            member.type = (
                tarfile.DIRTYPE
                if name.endswith((DATASET_NAME, "/rgb", "/depth"))
                else tarfile.REGTYPE
            )
            member.size = 0
            stream.addfile(member)

    report = _validate_archive(archive)

    assert report["required_entries_present"] == [
        "depth",
        "depth.txt",
        "groundtruth.txt",
        "rgb",
        "rgb.txt",
    ]
    assert report["archive_plus_regular_file_bytes"] <= report[
        "archive_plus_raw_tree_budget_bytes"
    ]


def test_groundtruth_allows_duplicate_timestamp_but_reports_it(tmp_path: Path) -> None:
    root = _raw_tree(tmp_path)
    lines = (root / "groundtruth.txt").read_text(encoding="utf-8").splitlines()
    lines[1] = lines[0]
    (root / "groundtruth.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = _validate_groundtruth(root)

    assert report["rows"] == 30
    assert report["duplicate_timestamp_count"] == 1


def test_response_search_excludes_own_preflight_but_finds_prior_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stateguard_root = tmp_path / "StateGuard3R"
    recalc_root = tmp_path / "ReCal3R"
    preflight = stateguard_root / "outputs" / acquire.PREFLIGHT_OUTPUT_NAME
    preflight.mkdir(parents=True)
    (preflight / f"{DATASET_NAME}-metadata.json").write_text("metadata", encoding="utf-8")
    prior = recalc_root / "outputs" / f"run-{DATASET_NAME}-response"
    prior.parent.mkdir(parents=True)
    prior.write_text("already run", encoding="utf-8")
    generic_log = stateguard_root / "logs" / "runner.log"
    generic_log.parent.mkdir(parents=True)
    generic_log.write_text(f"input scene: {DATASET_NAME}", encoding="utf-8")

    monkeypatch.setattr(acquire, "STATEGUARD_ROOT", stateguard_root)
    monkeypatch.setattr(acquire, "RECAL3R_ROOT", recalc_root)
    monkeypatch.setattr(acquire, "OUTPUT_ROOT", stateguard_root / "outputs")

    assert _source_response_search() == [str(prior), str(generic_log)]
