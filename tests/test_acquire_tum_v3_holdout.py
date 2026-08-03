from __future__ import annotations

import json
import stat
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


def _acquisition_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    tum = tmp_path / "tum"
    outputs = tmp_path / "outputs"
    tum.mkdir()
    outputs.mkdir()
    monkeypatch.setattr(acquire, "TUM_ROOT", tum)
    monkeypatch.setattr(acquire, "OUTPUT_ROOT", outputs)
    monkeypatch.setattr(acquire, "EXPECTED_ARCHIVE_BYTES", len(b"fixture"))
    monkeypatch.setattr(
        acquire,
        "_validate_archive",
        lambda _archive: {"fixture_archive_validation": True},
    )
    archive = tum / f"{DATASET_NAME}.tgz"
    archive.write_bytes(b"fixture")
    preflight = outputs / acquire.PREFLIGHT_OUTPUT_NAME
    preflight.mkdir()
    preflight_record = {
        "schema_version": "stateguard3r.formal-v3-tum-preflight.v1",
        "status": "PASS",
        "dataset": DATASET_NAME,
        "expected_archive_bytes": len(b"fixture"),
        "archive_target": str(archive),
        "raw_target": str(tum / DATASET_NAME),
        "model_response_search_hits": [],
        "official_source": {
            "canonical_url": CANONICAL_URL,
            "license": acquire.LICENSE,
            "license_reference": acquire.LICENSE_REFERENCE,
        },
    }
    (preflight / "preflight.json").write_text(
        json.dumps(preflight_record, sort_keys=True) + "\n", encoding="utf-8"
    )
    return tum, outputs, preflight


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


def test_acquire_recovers_the_single_verified_staging_tree_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tum, outputs, preflight = _acquisition_fixture(tmp_path, monkeypatch)
    staging = tum / f".{DATASET_NAME}.formal-v3-staging-fixture"
    staging.mkdir()
    _raw_tree(staging)

    report = acquire.acquire(preflight)

    raw = tum / DATASET_NAME
    acquisition = outputs / ACQUISITION_OUTPUT_NAME / "acquisition.json"
    assert report["staging_recovery"] is True
    assert raw.is_dir()
    assert not staging.exists()
    assert not list(tum.glob(f".{DATASET_NAME}.formal-v3-staging-*"))
    assert stat.S_IMODE(raw.stat().st_mode) == 0o555
    assert stat.S_IMODE((raw / "rgb.txt").stat().st_mode) == 0o444
    assert stat.S_IMODE((tum / f"{DATASET_NAME}.tgz").stat().st_mode) == 0o444
    assert json.loads(acquisition.read_text(encoding="utf-8"))["staging_recovery"] is True
    assert stat.S_IMODE(acquisition.stat().st_mode) == 0o444


def test_acquire_rejects_ambiguous_multiple_staging_trees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tum, _outputs, preflight = _acquisition_fixture(tmp_path, monkeypatch)
    for suffix in ("one", "two"):
        (tum / f".{DATASET_NAME}.formal-v3-staging-{suffix}").mkdir()

    with pytest.raises(AcquisitionError, match="multiple unfinished formal-v3 staging"):
        acquire.acquire(preflight)
