#!/usr/bin/env python3
"""Acquire exactly one predeclared, uninspected TUM RGB-D formal-v2 scene.

The only supported scene is ``fr3_walking_static``.  ``preflight`` is
network-metadata-only; ``acquire`` binds to that immutable preflight, resumes
only the same ``.part`` archive, validates archive/tree safety and publishes a
read-only raw tree.  Neither mode runs a model or computes online overlap.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path("/data/wangzheng/Project2")
RECAL3R_ROOT = PROJECT_ROOT / "baselines" / "ReCal3R"
TUM_ROOT = RECAL3R_ROOT / "data" / "tum"
DATASET_NAME = "rgbd_dataset_freiburg3_walking_static"
CANONICAL_URL = (
    "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/"
    f"{DATASET_NAME}.tgz"
)
EXPECTED_ARCHIVE_BYTES = 474_074_105
EXPECTED_TOP_LEVEL = DATASET_NAME
REQUIRED_PATHS = frozenset({"rgb", "depth", "rgb.txt", "depth.txt", "groundtruth.txt"})
MINIMUM_DATA_BUDGET_BYTES = 1_500 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 1024 * 1024


class AcquisitionError(RuntimeError):
    """Raised when a fresh formal-v2 holdout acquisition is unsafe or invalid."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("output_dir", type=Path)
    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("preflight_dir", type=Path)
    acquire.add_argument("output_dir", type=Path)
    return parser


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AcquisitionError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _freeze_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AcquisitionError(f"refusing to freeze symlink in raw tree: {path}")
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
        else:
            raise AcquisitionError(f"unexpected raw-tree entry: {path}")
    root.chmod(0o555)


def _inside_project(path: Path) -> bool:
    resolved = path.resolve(strict=False)
    return resolved == PROJECT_ROOT or PROJECT_ROOT in resolved.parents


def _assert_output_path(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    _require(_inside_project(resolved), f"output must stay in Project2: {resolved}")
    _require(not resolved.exists(), f"refusing to overwrite existing output: {resolved}")
    return resolved


def _archive_path() -> Path:
    return TUM_ROOT / f"{DATASET_NAME}.tgz"


def _raw_path() -> Path:
    return TUM_ROOT / DATASET_NAME


def _part_path() -> Path:
    return TUM_ROOT / f"{DATASET_NAME}.tgz.part"


def _assert_target_absent(*, permit_part: bool) -> None:
    for path in (_archive_path(), _raw_path()):
        _require(not path.exists(), f"holdout target already exists: {path}")
    part = _part_path()
    if not permit_part:
        _require(not part.exists(), f"unfinished holdout archive already exists: {part}")
    for path in TUM_ROOT.glob(f".{DATASET_NAME}.formal-v2-staging-*"):
        raise AcquisitionError(f"unfinished holdout staging directory exists: {path}")


def _head() -> dict[str, Any]:
    request = Request(CANONICAL_URL, method="HEAD")
    try:
        with urlopen(request, timeout=60) as response:
            headers = response.headers
            content_length = headers.get("Content-Length")
            _require(content_length is not None, "official HEAD has no Content-Length")
            try:
                bytes_on_server = int(content_length)
            except ValueError as error:
                raise AcquisitionError(f"invalid official Content-Length: {content_length!r}") from error
            _require(
                bytes_on_server == EXPECTED_ARCHIVE_BYTES,
                "official archive Content-Length differs from the predeclared single candidate",
            )
            return {
                "canonical_url": CANONICAL_URL,
                "final_url": response.geturl(),
                "status": int(getattr(response, "status", response.getcode())),
                "content_length": bytes_on_server,
                "accept_ranges": headers.get("Accept-Ranges"),
                "last_modified": headers.get("Last-Modified"),
                "etag": headers.get("ETag"),
                "content_type": headers.get("Content-Type"),
            }
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise AcquisitionError(f"official TUM HEAD failed: {error}") from error


def _source_search() -> list[str]:
    roots = (PROJECT_ROOT / "StateGuard3R" / "outputs", PROJECT_ROOT / "StateGuard3R" / "logs", RECAL3R_ROOT / "outputs")
    hits: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for candidate in root.rglob("*"):
            if DATASET_NAME in candidate.name:
                hits.append(str(candidate))
    return sorted(hits)


def _preflight_report() -> dict[str, Any]:
    _assert_target_absent(permit_part=False)
    _require(TUM_ROOT.is_dir(), f"TUM target parent does not exist: {TUM_ROOT}")
    available = shutil.disk_usage(TUM_ROOT).free
    _require(
        available >= MINIMUM_DATA_BUDGET_BYTES,
        f"only {available} bytes free; need at least {MINIMUM_DATA_BUDGET_BYTES}",
    )
    response_hits = _source_search()
    _require(
        not response_hits,
        "candidate appears under a model-output/log root and is not uninspected: " + ", ".join(response_hits),
    )
    return {
        "schema_version": "stateguard3r.formal-v2-tum-preflight.v1",
        "status": "PASS",
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset": DATASET_NAME,
        "official_source": {
            **_head(),
            "license": "CC BY 4.0",
            "license_reference": "https://creativecommons.org/licenses/by/4.0/",
        },
        "archive_target": str(_archive_path()),
        "raw_target": str(_raw_path()),
        "part_target": str(_part_path()),
        "expected_archive_bytes": EXPECTED_ARCHIVE_BYTES,
        "minimum_data_budget_bytes": MINIMUM_DATA_BUDGET_BYTES,
        "available_bytes_before_download": available,
        "model_response_search_roots": [str(item) for item in (PROJECT_ROOT / "StateGuard3R" / "outputs", PROJECT_ROOT / "StateGuard3R" / "logs", RECAL3R_ROOT / "outputs")],
        "model_response_search_hits": response_hits,
        "policy": {
            "single_archive_only": True,
            "download_to_part_then_atomic_rename": True,
            "no_model_or_online_overlap_execution": True,
            "no_formal_response_inspection_before_commitment": True,
        },
    }


def _read_preflight(path: Path) -> dict[str, Any]:
    record_path = path.resolve() / "preflight.json"
    _require(record_path.is_file(), f"preflight record is missing: {record_path}")
    try:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AcquisitionError(f"cannot read preflight record: {error}") from error
    _require(isinstance(payload, dict), "preflight record is not an object")
    _require(payload.get("status") == "PASS", "preflight did not pass")
    _require(payload.get("dataset") == DATASET_NAME, "preflight selected a different dataset")
    _require(payload.get("expected_archive_bytes") == EXPECTED_ARCHIVE_BYTES, "preflight size binding differs")
    _require(payload.get("archive_target") == str(_archive_path()), "preflight archive target differs")
    _require(payload.get("raw_target") == str(_raw_path()), "preflight raw target differs")
    _require(payload.get("model_response_search_hits") == [], "preflight disclosed candidate response evidence")
    return payload


def _download_resume() -> dict[str, Any]:
    part = _part_path()
    previous_size = part.stat().st_size if part.is_file() else 0
    _require(previous_size <= EXPECTED_ARCHIVE_BYTES, "partial archive is larger than expected")
    headers = {"Range": f"bytes={previous_size}-"} if previous_size else {}
    request = Request(CANONICAL_URL, headers=headers)
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=60) as response:
            status = int(getattr(response, "status", response.getcode()))
            if previous_size:
                _require(status == 206, f"resume requested but server returned HTTP {status}")
            else:
                _require(status == 200, f"fresh download returned HTTP {status}")
            mode = "ab" if previous_size else "wb"
            received = 0
            with part.open(mode) as stream:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    stream.write(chunk)
                    received += len(chunk)
                stream.flush()
                os.fsync(stream.fileno())
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise AcquisitionError(f"archive download failed; retained {part} for resume: {error}") from error
    final_size = part.stat().st_size
    _require(
        final_size == EXPECTED_ARCHIVE_BYTES,
        f"archive size {final_size} differs from expected {EXPECTED_ARCHIVE_BYTES}",
    )
    return {
        "canonical_url": CANONICAL_URL,
        "part_path": str(part),
        "resumed_from_bytes": previous_size,
        "received_bytes": received,
        "final_bytes": final_size,
        "runtime_seconds": time.perf_counter() - started,
    }


def _safe_member(member: tarfile.TarInfo) -> PurePosixPath:
    relative = PurePosixPath(member.name)
    _require(not relative.is_absolute(), f"archive member is absolute: {member.name}")
    _require(".." not in relative.parts and "." not in relative.parts, f"archive member escapes root: {member.name}")
    _require(relative.parts and relative.parts[0] == EXPECTED_TOP_LEVEL, f"unexpected archive top-level: {member.name}")
    _require(member.isdir() or member.isfile(), f"archive has unsupported member type: {member.name}")
    _require(not member.issym() and not member.islnk(), f"archive has link member: {member.name}")
    return relative


def _validate_archive(archive: Path) -> dict[str, Any]:
    _require(archive.is_file(), f"archive does not exist: {archive}")
    try:
        with gzip.open(archive, "rb") as stream:
            while stream.read(8 * 1024 * 1024):
                pass
    except (OSError, EOFError) as error:
        raise AcquisitionError(f"gzip integrity check failed: {error}") from error
    try:
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
    except (tarfile.TarError, OSError) as error:
        raise AcquisitionError(f"cannot read TUM tar archive: {error}") from error
    _require(members, "TUM archive has no members")
    names: set[PurePosixPath] = set()
    total_regular_bytes = 0
    directories = 0
    regular_files = 0
    required: set[str] = set()
    for member in members:
        relative = _safe_member(member)
        _require(relative not in names, f"duplicate archive member: {member.name}")
        names.add(relative)
        tail = relative.parts[1:]
        if tail:
            required.add("/".join(tail))
        if member.isdir():
            directories += 1
        else:
            regular_files += 1
            total_regular_bytes += member.size
    _require(REQUIRED_PATHS <= required, f"archive lacks required entries: {sorted(REQUIRED_PATHS - required)}")
    return {
        "member_count": len(members),
        "directory_count": directories,
        "regular_file_count": regular_files,
        "regular_file_bytes": total_regular_bytes,
        "required_entries_present": sorted(REQUIRED_PATHS),
    }


def _extract_to_staging(archive: Path) -> tuple[Path, Path]:
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{DATASET_NAME}.formal-v2-staging-", dir=TUM_ROOT)
    )
    try:
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            for member in members:
                _safe_member(member)
            tar.extractall(path=staging_parent, members=members, numeric_owner=False)
        root = staging_parent / DATASET_NAME
        _require(root.is_dir() and not root.is_symlink(), "staged archive root is invalid")
        return staging_parent, root
    except BaseException:
        raise


def _decimal_timestamp(text: str, *, label: str) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise AcquisitionError(f"invalid {label} timestamp: {text!r}") from error
    _require(value.is_finite(), f"non-finite {label} timestamp: {text!r}")
    return value


def _data_lines(path: Path) -> Iterable[tuple[int, list[str]]]:
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        yield line_number, line.split()


def _png_header(path: Path) -> tuple[int, int, int, int]:
    data = path.read_bytes()[:33]
    _require(data[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {path}")
    _require(data[12:16] == b"IHDR", f"PNG lacks IHDR: {path}")
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    bit_depth = data[24]
    color_type = data[25]
    return width, height, bit_depth, color_type


def _validate_stream_index(root: Path, *, kind: str) -> tuple[list[Decimal], dict[str, Any]]:
    index = root / f"{kind}.txt"
    directory = root / kind
    _require(index.is_file() and directory.is_dir(), f"missing {kind} stream")
    timestamps: list[Decimal] = []
    referenced: set[str] = set()
    expected_bit_depth, expected_color = (8, 2) if kind == "rgb" else (16, 0)
    for line_number, fields in _data_lines(index):
        _require(len(fields) == 2, f"{kind}.txt line {line_number} must have two fields")
        timestamp = _decimal_timestamp(fields[0], label=kind)
        _require(not timestamps or timestamp > timestamps[-1], f"{kind}.txt timestamps are not increasing")
        relative = PurePosixPath(fields[1])
        _require(relative.parts and relative.parts[0] == kind and ".." not in relative.parts, f"unsafe {kind} path at line {line_number}")
        image = root.joinpath(*relative.parts)
        _require(image.is_file() and not image.is_symlink(), f"missing {kind} image at line {line_number}")
        width, height, bit_depth, color_type = _png_header(image)
        _require((width, height) == (640, 480), f"unexpected {kind} image geometry: {image}")
        _require(bit_depth == expected_bit_depth and color_type == expected_color, f"unexpected {kind} PNG format: {image}")
        timestamps.append(timestamp)
        referenced.add(relative.as_posix())
    on_disk = {item.relative_to(root).as_posix() for item in directory.glob("*.png")}
    _require(referenced == on_disk, f"{kind}.txt does not exactly cover on-disk PNGs")
    _require(len(timestamps) >= 30, f"{kind} has fewer than 30 frames")
    return timestamps, {"rows": len(timestamps), "first_timestamp": str(timestamps[0]), "last_timestamp": str(timestamps[-1])}


def _validate_groundtruth(root: Path) -> dict[str, Any]:
    path = root / "groundtruth.txt"
    _require(path.is_file(), "groundtruth.txt is missing")
    timestamps: list[Decimal] = []
    for line_number, fields in _data_lines(path):
        _require(len(fields) == 8, f"groundtruth line {line_number} must have eight fields")
        timestamp = _decimal_timestamp(fields[0], label="groundtruth")
        _require(
            not timestamps or timestamp >= timestamps[-1],
            "groundtruth timestamps are decreasing",
        )
        try:
            values = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise AcquisitionError(f"invalid groundtruth value at line {line_number}") from error
        _require(all(math.isfinite(value) for value in values), f"non-finite groundtruth value at line {line_number}")
        quaternion_norm = math.sqrt(sum(value * value for value in values[3:]))
        _require(abs(quaternion_norm - 1.0) <= 0.02, f"groundtruth quaternion is not normalized at line {line_number}")
        timestamps.append(timestamp)
    _require(len(timestamps) >= 30, "groundtruth has fewer than 30 poses")
    return {
        "rows": len(timestamps),
        "first_timestamp": str(timestamps[0]),
        "last_timestamp": str(timestamps[-1]),
        "duplicate_timestamp_count": sum(
            left == right for left, right in zip(timestamps, timestamps[1:])
        ),
    }


def _nearest_distance(reference: Decimal, targets: Sequence[Decimal]) -> Decimal:
    return min(abs(reference - target) for target in targets)


def _tree_manifest(root: Path) -> dict[str, Any]:
    rgb_times, rgb = _validate_stream_index(root, kind="rgb")
    depth_times, depth = _validate_stream_index(root, kind="depth")
    groundtruth = _validate_groundtruth(root)
    associated_prefix = sum(
        _nearest_distance(timestamp, depth_times) <= Decimal("0.02")
        for timestamp in rgb_times
    )
    _require(associated_prefix >= 30, "fewer than 30 RGB frames have a depth match within 20 ms")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        _require(not path.is_symlink(), f"raw tree has symlink: {relative}")
        if path.is_file():
            entries.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
        elif not path.is_dir():
            raise AcquisitionError(f"raw tree has unsupported entry: {relative}")
    return {
        "dataset": DATASET_NAME,
        "root_name": root.name,
        "regular_file_count": len(entries),
        "regular_file_bytes": sum(entry["size_bytes"] for entry in entries),
        "rgb": rgb,
        "depth": depth,
        "groundtruth": groundtruth,
        "rgb_frames_with_depth_within_20ms": associated_prefix,
        "entries": entries,
    }


def _publish_acquisition(output_dir: Path, report: Mapping[str, Any]) -> None:
    _write_json_atomic(output_dir / "acquisition.json", report)
    for path in output_dir.iterdir():
        if path.is_file():
            path.chmod(0o444)
    output_dir.chmod(0o555)


def preflight(output_dir: Path) -> dict[str, Any]:
    output_dir = _assert_output_path(output_dir)
    report = _preflight_report()
    script = Path(__file__).resolve()
    report["script"] = {"path": str(script), "sha256": _sha256(script)}
    _write_json_atomic(output_dir / "preflight.json", report)
    for path in output_dir.iterdir():
        if path.is_file():
            path.chmod(0o444)
    output_dir.chmod(0o555)
    return report


def acquire(preflight_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = _assert_output_path(output_dir)
    preflight_record = _read_preflight(preflight_dir)
    _require(not _raw_path().exists(), f"holdout raw target already exists: {_raw_path()}")
    for path in TUM_ROOT.glob(f".{DATASET_NAME}.formal-v2-staging-*"):
        raise AcquisitionError(f"unfinished holdout staging directory exists: {path}")
    _require(shutil.disk_usage(TUM_ROOT).free >= MINIMUM_DATA_BUDGET_BYTES, "insufficient disk space before download")
    TUM_ROOT.mkdir(parents=True, exist_ok=True)
    archive = _archive_path()
    if archive.exists():
        _require(archive.is_file(), f"existing archive is not a regular file: {archive}")
        _require(not _part_path().exists(), "archive and .part both exist")
        _require(
            archive.stat().st_size == EXPECTED_ARCHIVE_BYTES,
            "existing archive size differs from the predeclared candidate",
        )
        download = {
            "canonical_url": CANONICAL_URL,
            "reused_verified_archive": True,
            "final_bytes": archive.stat().st_size,
        }
    else:
        download = _download_resume()
        part = _part_path()
        os.replace(part, archive)
        archive.chmod(0o444)
    archive_validation = _validate_archive(archive)
    archive_sha256 = _sha256(archive)
    staging_parent, staged_root = _extract_to_staging(archive)
    try:
        raw_manifest = _tree_manifest(staged_root)
        _freeze_tree(staged_root)
        staged_root.chmod(0o755)
        os.replace(staged_root, _raw_path())
        _raw_path().chmod(0o555)
    finally:
        if staging_parent.exists():
            shutil.rmtree(staging_parent)
    _require(not _part_path().exists(), "archive part remains after successful publication")
    script = Path(__file__).resolve()
    report = {
        "schema_version": "stateguard3r.formal-v2-tum-acquisition.v1",
        "status": "PASS",
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset": DATASET_NAME,
        "preflight": {
            "path": str((preflight_dir.resolve() / "preflight.json")),
            "sha256": _sha256(preflight_dir.resolve() / "preflight.json"),
            "record": preflight_record,
        },
        "download": download,
        "archive": {
            "path": str(archive),
            "size_bytes": archive.stat().st_size,
            "sha256": archive_sha256,
            "mode_octal": format(stat.S_IMODE(archive.stat().st_mode), "04o"),
            "validation": archive_validation,
        },
        "raw_tree": {
            "path": str(_raw_path()),
            "root_mode_octal": format(stat.S_IMODE(_raw_path().stat().st_mode), "04o"),
            "manifest": raw_manifest,
        },
        "postconditions": {
            "part_absent": True,
            "staging_absent": not any(TUM_ROOT.glob(f".{DATASET_NAME}.formal-v2-staging-*")),
            "no_model_or_online_overlap_execution": True,
        },
        "script": {"path": str(script), "sha256": _sha256(script)},
    }
    _publish_acquisition(output_dir, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        report = preflight(args.output_dir)
    else:
        report = acquire(args.preflight_dir, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
