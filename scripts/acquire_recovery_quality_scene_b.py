#!/usr/bin/env python3
"""Acquire only the predeclared second TUM scene for recovery-quality v1.

The safety-critical archive validation implementation is reused under a
temporary, explicit scene binding.  This wrapper adds a recovery-quality
schema and never launches a model or reads an existing model response.
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

from scripts import acquire_tum_v2_holdout as tum


DATASET_NAME = "rgbd_dataset_freiburg3_walking_xyz"
CANONICAL_URL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz"
EXPECTED_ARCHIVE_BYTES = 527_550_055
MINIMUM_DATA_BUDGET_BYTES = 5 * 1024 * 1024 * 1024
SCHEMA_PREFIX = "stateguard3r.recovery-quality-v1-scene-b"


@contextmanager
def _scene_binding() -> Iterator[None]:
    """Scope reuse of the existing tar/index validator to exactly scene B."""

    original = {
        "DATASET_NAME": tum.DATASET_NAME,
        "CANONICAL_URL": tum.CANONICAL_URL,
        "EXPECTED_ARCHIVE_BYTES": tum.EXPECTED_ARCHIVE_BYTES,
        "EXPECTED_TOP_LEVEL": tum.EXPECTED_TOP_LEVEL,
        "MINIMUM_DATA_BUDGET_BYTES": tum.MINIMUM_DATA_BUDGET_BYTES,
    }
    tum.DATASET_NAME = DATASET_NAME
    tum.CANONICAL_URL = CANONICAL_URL
    tum.EXPECTED_ARCHIVE_BYTES = EXPECTED_ARCHIVE_BYTES
    tum.EXPECTED_TOP_LEVEL = DATASET_NAME
    tum.MINIMUM_DATA_BUDGET_BYTES = MINIMUM_DATA_BUDGET_BYTES
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(tum, name, value)


def _freeze_directory(output_dir: Path) -> None:
    for path in output_dir.iterdir():
        if path.is_file() and not path.is_symlink():
            path.chmod(0o444)
        else:
            raise tum.AcquisitionError(f"unexpected acquisition output entry: {path}")
    output_dir.chmod(0o555)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight").add_argument("output_dir", type=Path)
    acquire = commands.add_parser("acquire")
    acquire.add_argument("preflight_dir", type=Path)
    acquire.add_argument("output_dir", type=Path)
    return parser


def preflight(output_dir: Path) -> dict[str, Any]:
    with _scene_binding():
        destination = tum._assert_output_path(output_dir)
        report = tum._preflight_report()
        report["schema_version"] = f"{SCHEMA_PREFIX}-preflight.v1"
        report["stage"] = "recovery_quality_formal_scene_b"
        report["policy"] = {
            **report["policy"],
            "single_new_archive_for_recovery_quality_v1": True,
            "archive_plus_raw_budget_bytes": MINIMUM_DATA_BUDGET_BYTES,
        }
        script = Path(__file__).resolve()
        report["script"] = {"path": str(script), "sha256": tum._sha256(script)}
        tum._write_json_atomic(destination / "preflight.json", report)
        _freeze_directory(destination)
        return report


def acquire(preflight_dir: Path, output_dir: Path) -> dict[str, Any]:
    with _scene_binding():
        destination = tum._assert_output_path(output_dir)
        preflight_record = tum._read_preflight(preflight_dir)
        tum._require(not tum._raw_path().exists(), f"formal scene B raw target already exists: {tum._raw_path()}")
        tum._require(shutil.disk_usage(tum.TUM_ROOT).free >= MINIMUM_DATA_BUDGET_BYTES, "insufficient disk space before scene B download")
        tum.TUM_ROOT.mkdir(parents=True, exist_ok=True)
        archive = tum._archive_path()
        if archive.exists():
            tum._require(archive.is_file() and archive.stat().st_size == EXPECTED_ARCHIVE_BYTES, "existing scene B archive differs from predeclared candidate")
            tum._require(not tum._part_path().exists(), "scene B archive and partial coexist")
            download: Mapping[str, Any] = {"canonical_url": CANONICAL_URL, "reused_verified_archive": True, "final_bytes": archive.stat().st_size}
        else:
            download = tum._download_resume()
            os.replace(tum._part_path(), archive)
            archive.chmod(0o444)
        archive_validation = tum._validate_archive(archive)
        archive_sha256 = tum._sha256(archive)
        staging_parent, staged_root = tum._extract_to_staging(archive)
        try:
            raw_manifest = tum._tree_manifest(staged_root)
            tum._freeze_tree(staged_root)
            staged_root.chmod(0o755)
            os.replace(staged_root, tum._raw_path())
            tum._raw_path().chmod(0o555)
        finally:
            if staging_parent.exists():
                shutil.rmtree(staging_parent)
        tum._require(not tum._part_path().exists(), "scene B partial remains after publication")
        script = Path(__file__).resolve()
        report = {
            "schema_version": f"{SCHEMA_PREFIX}-acquisition.v1",
            "status": "PASS",
            "created_at": datetime.now().astimezone().isoformat(),
            "dataset": DATASET_NAME,
            "preflight": {"path": str(preflight_dir.resolve() / "preflight.json"), "sha256": tum._sha256(preflight_dir.resolve() / "preflight.json"), "record": preflight_record},
            "download": dict(download),
            "archive": {"path": str(archive), "size_bytes": archive.stat().st_size, "sha256": archive_sha256, "mode_octal": format(stat.S_IMODE(archive.stat().st_mode), "04o"), "validation": archive_validation},
            "raw_tree": {"path": str(tum._raw_path()), "root_mode_octal": format(stat.S_IMODE(tum._raw_path().stat().st_mode), "04o"), "manifest": raw_manifest},
            "postconditions": {"part_absent": True, "staging_absent": not any(tum.TUM_ROOT.glob(f".{DATASET_NAME}.formal-v2-staging-*")), "no_model_or_online_overlap_execution": True, "no_response_inspection_before_quality_commitment": True},
            "script": {"path": str(script), "sha256": tum._sha256(script)},
        }
        tum._write_json_atomic(destination / "acquisition.json", report)
        _freeze_directory(destination)
        return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = preflight(args.output_dir) if args.command == "preflight" else acquire(args.preflight_dir, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
