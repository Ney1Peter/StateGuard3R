"""Narrow v16 capture-order binding from raw RGB paths and a raw listing only."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence


PARSER_VERSION = "tum-rgb-txt-decimal-strict-v16.v1"


class TimestampOrderV16Error(ValueError):
    """The v16 raw RGB timestamp capability was violated."""


def _regular_0444(path: Path, *, label: str) -> Path:
    try:
        value = os.lstat(path)
    except OSError as error:
        raise TimestampOrderV16Error(f"cannot stat {label}") from error
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode) or stat.S_IMODE(value.st_mode) != 0o444:
        raise TimestampOrderV16Error(f"{label} must be a non-symlink mode-0444 regular file")
    return path.resolve(strict=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decimal(value: str, *, label: str) -> Decimal:
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise TimestampOrderV16Error(f"{label} is not a finite decimal") from error
    if not result.is_finite():
        raise TimestampOrderV16Error(f"{label} is not a finite decimal")
    return result


def capture_timestamp_records_v16(frame_paths: Sequence[str | os.PathLike[str]], *, rgb_txt: str | os.PathLike[str], dataset_root: str | os.PathLike[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind just the selected ordered RGB files to timestamp rows in ``rgb.txt``."""
    if not frame_paths:
        raise TimestampOrderV16Error("selected RGB paths must not be empty")
    root = Path(dataset_root).resolve(strict=True)
    if not root.is_dir():
        raise TimestampOrderV16Error("dataset root is unavailable")
    listing = _regular_0444(Path(rgb_txt), label="v16 raw rgb listing")
    selected = tuple(_regular_0444(Path(value), label=f"v16 selected RGB {index}") for index, value in enumerate(frame_paths))
    if len(set(selected)) != len(selected):
        raise TimestampOrderV16Error("v16 selected RGB paths must be unique")
    selected_relative: dict[str, Path] = {}
    for path in selected:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise TimestampOrderV16Error("selected RGB path escapes dataset root") from error
        if not relative.startswith("rgb/"):
            raise TimestampOrderV16Error("selected path is not an RGB listing path")
        selected_relative[relative] = path
    found: dict[Path, tuple[Decimal, str, int]] = {}
    try:
        lines = listing.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise TimestampOrderV16Error("cannot read v16 raw rgb listing") from error
    previous: Decimal | None = None
    for physical_line, raw in enumerate(lines, start=1):
        row = raw.strip()
        if not row or row.startswith("#"):
            continue
        fields = row.split()
        if len(fields) != 2:
            raise TimestampOrderV16Error("v16 raw listing row differs")
        timestamp = _decimal(fields[0], label=f"v16 raw listing line {physical_line}")
        if previous is not None and timestamp < previous:
            raise TimestampOrderV16Error("v16 raw listing timestamps decrease")
        previous = timestamp
        selected_path = selected_relative.get(fields[1])
        if selected_path is not None:
            if selected_path in found:
                raise TimestampOrderV16Error("v16 raw listing repeats a selected RGB path")
            found[selected_path] = (timestamp, fields[0], physical_line)
    if set(found) != set(selected):
        raise TimestampOrderV16Error("v16 selected RGB path is absent from raw listing")
    records = [
        {
            "frame_id": frame_id,
            "rgb_capture_timestamp": str(found[path][0]),
            "rgb_capture_timestamp_text": found[path][1],
            "rgb_txt_physical_line": found[path][2],
            "rgb_path_sha256": _sha256(path),
        }
        for frame_id, path in enumerate(selected)
    ]
    provenance = {
        "parser_version": PARSER_VERSION,
        "rgb_txt_path": str(listing),
        "rgb_txt_sha256": _sha256(listing),
        "dataset_root": str(root),
        "input_contract": "ordered_rgb_paths_only_no_index_or_annotation_metadata",
    }
    return records, provenance


def timestamp_order_sidecar_v16(captures: Sequence[Mapping[str, Any]], *, provenance: Mapping[str, Any]) -> dict[str, Any]:
    if not captures or not isinstance(provenance, Mapping):
        raise TimestampOrderV16Error("v16 capture records and provenance are required")
    expected = {"frame_id", "rgb_capture_timestamp", "rgb_capture_timestamp_text", "rgb_txt_physical_line", "rgb_path_sha256"}
    records: list[dict[str, Any]] = []
    previous: Decimal | None = None
    for frame_id, raw in enumerate(captures):
        if not isinstance(raw, Mapping) or set(raw) != expected or raw.get("frame_id") != frame_id:
            raise TimestampOrderV16Error("v16 capture record schema differs")
        timestamp = _decimal(str(raw.get("rgb_capture_timestamp")), label=f"v16 capture {frame_id}")
        if _decimal(str(raw.get("rgb_capture_timestamp_text")), label=f"v16 capture text {frame_id}") != timestamp:
            raise TimestampOrderV16Error("v16 capture timestamp text differs")
        line, digest = raw.get("rgb_txt_physical_line"), raw.get("rgb_path_sha256")
        if type(line) is not int or line < 1 or not isinstance(digest, str) or len(digest) != 64:
            raise TimestampOrderV16Error("v16 capture provenance differs")
        records.append({**dict(raw), "timestamp_order_violation": None if previous is None else bool(timestamp <= previous), "predicate": "null_first_frame_else_current_capture_timestamp_lte_previous"})
        previous = timestamp
    return {
        "schema_version": "stateguard3r.timestamp-order-v16.v1",
        "purpose": "online_capture_timestamp_order_invariant",
        "provenance": dict(provenance),
        "records": records,
    }


__all__ = ["PARSER_VERSION", "TimestampOrderV16Error", "capture_timestamp_records_v16", "timestamp_order_sidecar_v16"]
