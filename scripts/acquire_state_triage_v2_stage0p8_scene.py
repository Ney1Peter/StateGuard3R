#!/usr/bin/env python3
"""Acquire the sole new official TUM scene permitted for StateTriage Stage 0.8.

The archive safety and raw-layout validator is reused under an explicit,
temporary scene binding.  This script never imports a model or launches an
online forward.  It publishes only a read-only archive/raw tree and two
read-only provenance records.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import acquire_tum_v3_holdout as tum


DATASET_NAME = "rgbd_dataset_freiburg1_room"
CANONICAL_URL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz"
EXPECTED_ARCHIVE_BYTES = 782_381_450
MINIMUM_DATA_BUDGET_BYTES = 5 * 1024 * 1024 * 1024
SCHEMA_PREFIX = "stateguard3r.state-triage-stage0p8-scene"
PREFLIGHT_OUTPUT_NAME = "state-triage-v2-stage0p8-data-preflight-0001"
ACQUISITION_OUTPUT_NAME = "state-triage-v2-stage0p8-data-0001"


@contextmanager
def _scene_binding() -> Iterator[None]:
    """Bind the proven tar/index checker to exactly the Stage 0.8 scene."""

    original = {
        "DATASET_NAME": tum.DATASET_NAME,
        "CANONICAL_URL": tum.CANONICAL_URL,
        "EXPECTED_ARCHIVE_BYTES": tum.EXPECTED_ARCHIVE_BYTES,
        "EXPECTED_TOP_LEVEL": tum.EXPECTED_TOP_LEVEL,
        "MAX_ARCHIVE_AND_RAW_TREE_BYTES": tum.MAX_ARCHIVE_AND_RAW_TREE_BYTES,
        "MINIMUM_FREE_BEFORE_DOWNLOAD_BYTES": tum.MINIMUM_FREE_BEFORE_DOWNLOAD_BYTES,
    }
    tum.DATASET_NAME = DATASET_NAME
    tum.CANONICAL_URL = CANONICAL_URL
    tum.EXPECTED_ARCHIVE_BYTES = EXPECTED_ARCHIVE_BYTES
    tum.EXPECTED_TOP_LEVEL = DATASET_NAME
    tum.MAX_ARCHIVE_AND_RAW_TREE_BYTES = MINIMUM_DATA_BUDGET_BYTES
    tum.MINIMUM_FREE_BEFORE_DOWNLOAD_BYTES = MINIMUM_DATA_BUDGET_BYTES + 64 * 1024 * 1024
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(tum, name, value)


def _freeze_directory(output_dir: Path) -> None:
    for path in output_dir.iterdir():
        if not path.is_file() or path.is_symlink():
            raise tum.AcquisitionError(f"unexpected Stage 0.8 acquisition artifact: {path}")
        path.chmod(0o444)
    output_dir.chmod(0o555)


def _output_path(output_dir: Path, *, expected_name: str) -> Path:
    resolved = output_dir.resolve(strict=False)
    expected = (tum.OUTPUT_ROOT / expected_name).resolve(strict=False)
    tum._require(resolved == expected, f"Stage 0.8 output must be exactly {expected}")
    tum._require(not resolved.exists(), f"refusing to overwrite Stage 0.8 output: {resolved}")
    return resolved


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    acquire = commands.add_parser("acquire")
    acquire.add_argument("preflight_dir", type=Path)
    return parser


def preflight() -> dict[str, Any]:
    with _scene_binding():
        destination = _output_path(
            tum.OUTPUT_ROOT / PREFLIGHT_OUTPUT_NAME,
            expected_name=PREFLIGHT_OUTPUT_NAME,
        )
        report = tum._preflight_report()
        report["schema_version"] = f"{SCHEMA_PREFIX}-preflight.v1"
        report["stage"] = "state_triage_v2_stage0p8"
        report["policy"] = {
            **report["policy"],
            "single_new_archive_for_stage0p8": True,
            "archive_plus_raw_budget_bytes": MINIMUM_DATA_BUDGET_BYTES,
            "no_calibration_or_final_forward_before_read_only_publication": True,
        }
        script = Path(__file__).resolve()
        report["script"] = {"path": str(script), "sha256": tum._sha256(script)}
        tum._write_json_atomic(destination / "preflight.json", report)
        _freeze_directory(destination)
        return report


def _read_preflight(preflight_dir: Path) -> dict[str, Any]:
    path = preflight_dir.resolve(strict=True) / "preflight.json"
    tum._require(path.is_file() and not path.is_symlink(), f"Stage 0.8 preflight record is missing: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise tum.AcquisitionError(f"cannot read Stage 0.8 preflight: {error}") from error
    tum._require(isinstance(report, dict), "Stage 0.8 preflight is not an object")
    tum._require(report.get("schema_version") == f"{SCHEMA_PREFIX}-preflight.v1", "Stage 0.8 preflight schema differs")
    tum._require(report.get("status") == "PASS", "Stage 0.8 preflight did not pass")
    tum._require(report.get("dataset") == DATASET_NAME, "Stage 0.8 preflight dataset differs")
    tum._require(report.get("expected_archive_bytes") == EXPECTED_ARCHIVE_BYTES, "Stage 0.8 preflight archive size differs")
    tum._require(report.get("archive_target") == str(tum._archive_path()), "Stage 0.8 preflight archive target differs")
    tum._require(report.get("raw_target") == str(tum._raw_path()), "Stage 0.8 preflight raw target differs")
    tum._require(report.get("model_response_search_hits") == [], "Stage 0.8 candidate already has response evidence")
    official = report.get("official_source")
    tum._require(isinstance(official, Mapping) and official.get("canonical_url") == CANONICAL_URL, "Stage 0.8 official URL differs")
    return report


def acquire(preflight_dir: Path) -> dict[str, Any]:
    with _scene_binding():
        destination = _output_path(
            tum.OUTPUT_ROOT / ACQUISITION_OUTPUT_NAME,
            expected_name=ACQUISITION_OUTPUT_NAME,
        )
        preflight_record = _read_preflight(preflight_dir)
        archive = tum._archive_path()
        raw = tum._raw_path()
        part = tum._part_path()
        tum._require(not raw.exists(), f"Stage 0.8 raw target already exists: {raw}")
        tum._require(shutil.disk_usage(tum.TUM_ROOT).free >= MINIMUM_DATA_BUDGET_BYTES, "insufficient disk space before Stage 0.8 download")
        tum.TUM_ROOT.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            tum._require(archive.is_file() and not archive.is_symlink(), "existing Stage 0.8 archive is not a regular file")
            tum._require(archive.stat().st_size == EXPECTED_ARCHIVE_BYTES, "existing Stage 0.8 archive size differs from preflight")
            tum._require(not (archive.stat().st_mode & 0o222), "existing Stage 0.8 archive must be read-only")
            tum._require(not part.exists(), "published Stage 0.8 archive and partial archive coexist")
            download: Mapping[str, Any] = {"canonical_url": CANONICAL_URL, "reused_verified_archive": True, "final_bytes": archive.stat().st_size}
        else:
            tum._assert_target_absent(permit_part=True)
            download = tum._download_resume()
            os.replace(part, archive)
            archive.chmod(0o444)
        archive_validation = tum._validate_archive(archive)
        archive_sha256 = tum._sha256(archive)
        staging_parent, staged_root = tum._extract_to_staging(archive)
        try:
            raw_manifest = tum._tree_manifest(staged_root)
            tum._freeze_raw_tree(staged_root, freeze_root=False)
            os.replace(staged_root, raw)
            raw.chmod(0o555)
        finally:
            if staging_parent.exists():
                shutil.rmtree(staging_parent)
        tum._require(not tum._part_path().exists(), "Stage 0.8 partial archive remains after publication")
        script = Path(__file__).resolve()
        report = {
            "schema_version": f"{SCHEMA_PREFIX}-acquisition.v1",
            "status": "PASS",
            "created_at": datetime.now().astimezone().isoformat(),
            "dataset": DATASET_NAME,
            "preflight": {
                "path": str(preflight_dir.resolve() / "preflight.json"),
                "sha256": tum._sha256(preflight_dir.resolve() / "preflight.json"),
                "record": preflight_record,
            },
            "download": dict(download),
            "archive": {
                "path": str(archive),
                "size_bytes": archive.stat().st_size,
                "sha256": archive_sha256,
                "mode_octal": format(stat.S_IMODE(archive.stat().st_mode), "04o"),
                "validation": archive_validation,
            },
            "raw_tree": {
                "path": str(raw),
                "root_mode_octal": format(stat.S_IMODE(raw.stat().st_mode), "04o"),
                "manifest": raw_manifest,
            },
            "postconditions": {
                "part_absent": True,
                "staging_absent": not any(tum.TUM_ROOT.glob(f".{DATASET_NAME}.formal-v3-staging-*")),
                "no_model_or_online_overlap_execution": True,
                "no_calibration_or_final_response_before_data_publication": True,
            },
            "script": {"path": str(script), "sha256": tum._sha256(script)},
        }
        tum._write_json_atomic(destination / "acquisition.json", report)
        _freeze_directory(destination)
        return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = preflight() if args.command == "preflight" else acquire(args.preflight_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
