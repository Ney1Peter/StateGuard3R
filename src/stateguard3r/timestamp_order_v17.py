"""v17 raw RGB capture-order binding with no auxiliary input channel."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence


PARSER_VERSION = "tum-rgb-txt-decimal-strict-v17.v1"


class TimestampOrderV17Error(ValueError):
    """A raw capture-order input differs from the v17 contract."""


def _regular_0444(path: Path, *, label: str) -> Path:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise TimestampOrderV17Error(f"cannot stat {label}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o444:
        raise TimestampOrderV17Error(f"{label} must be a non-symlink mode-0444 regular file")
    return path.resolve(strict=True)


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _decimal(text: str, *, label: str) -> Decimal:
    try:
        result = Decimal(text)
    except InvalidOperation as error:
        raise TimestampOrderV17Error(f"{label} is not a finite decimal") from error
    if not result.is_finite():
        raise TimestampOrderV17Error(f"{label} is not a finite decimal")
    return result


def capture_timestamp_records_v17(frame_paths: Sequence[str | os.PathLike[str]], *, rgb_txt: str | os.PathLike[str], dataset_root: str | os.PathLike[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind selected ordered RGB paths to their rows in raw ``rgb.txt`` only."""

    if not frame_paths:
        raise TimestampOrderV17Error("selected RGB paths cannot be empty")
    root = Path(dataset_root).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise TimestampOrderV17Error("dataset root is unavailable")
    listing = _regular_0444(Path(rgb_txt), label="raw RGB listing")
    selected = tuple(_regular_0444(Path(value), label=f"selected RGB {index}") for index, value in enumerate(frame_paths))
    if len(set(selected)) != len(selected):
        raise TimestampOrderV17Error("selected RGB paths must be unique")
    relative: dict[str, Path] = {}
    for value in selected:
        try:
            text = value.relative_to(root).as_posix()
        except ValueError as error:
            raise TimestampOrderV17Error("selected RGB path escapes dataset root") from error
        if not text.startswith("rgb/"):
            raise TimestampOrderV17Error("selected path is not an RGB listing path")
        relative[text] = value
    found: dict[Path, tuple[Decimal, str, int]] = {}
    prior: Decimal | None = None
    try:
        rows = listing.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise TimestampOrderV17Error("cannot read raw RGB listing") from error
    for number, raw in enumerate(rows, start=1):
        row = raw.strip()
        if not row or row.startswith("#"):
            continue
        fields = row.split()
        if len(fields) != 2:
            raise TimestampOrderV17Error("raw RGB listing row differs")
        value = _decimal(fields[0], label=f"raw RGB listing line {number}")
        if prior is not None and value < prior:
            raise TimestampOrderV17Error("raw RGB listing timestamps decrease")
        prior = value
        selected_path = relative.get(fields[1])
        if selected_path is not None:
            if selected_path in found:
                raise TimestampOrderV17Error("raw RGB listing repeats a selected path")
            found[selected_path] = (value, fields[0], number)
    if set(found) != set(selected):
        raise TimestampOrderV17Error("a selected RGB path is absent from raw listing")
    records = [{"frame_id": index, "rgb_capture_timestamp": str(found[path][0]), "rgb_capture_timestamp_text": found[path][1], "rgb_txt_physical_line": found[path][2], "rgb_path_sha256": _digest(path)} for index, path in enumerate(selected)]
    return records, {"parser_version": PARSER_VERSION, "rgb_txt_path": str(listing), "rgb_txt_sha256": _digest(listing), "dataset_root": str(root), "input_contract": "ordered_rgb_paths_only"}


def timestamp_order_sidecar_v17(captures: Sequence[Mapping[str, Any]], *, provenance: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"frame_id", "rgb_capture_timestamp", "rgb_capture_timestamp_text", "rgb_txt_physical_line", "rgb_path_sha256"}
    if not captures or not isinstance(provenance, Mapping):
        raise TimestampOrderV17Error("capture records and provenance are required")
    previous: Decimal | None = None
    rows: list[dict[str, Any]] = []
    for frame_id, raw in enumerate(captures):
        if not isinstance(raw, Mapping) or set(raw) != expected or raw.get("frame_id") != frame_id:
            raise TimestampOrderV17Error("capture record schema differs")
        value = _decimal(str(raw.get("rgb_capture_timestamp")), label=f"capture {frame_id}")
        if _decimal(str(raw.get("rgb_capture_timestamp_text")), label=f"capture text {frame_id}") != value:
            raise TimestampOrderV17Error("capture timestamp text differs")
        line, digest = raw.get("rgb_txt_physical_line"), raw.get("rgb_path_sha256")
        if type(line) is not int or line < 1 or not isinstance(digest, str) or len(digest) != 64:
            raise TimestampOrderV17Error("capture record provenance differs")
        rows.append({**dict(raw), "timestamp_order_violation": None if previous is None else bool(value <= previous), "predicate": "null_first_frame_else_current_capture_timestamp_lte_previous"})
        previous = value
    return {"schema_version": "stateguard3r.timestamp-order-v17.v1", "purpose": "online_capture_timestamp_order_invariant", "provenance": dict(provenance), "records": rows}


__all__ = ["PARSER_VERSION", "TimestampOrderV17Error", "capture_timestamp_records_v17", "timestamp_order_sidecar_v17"]
