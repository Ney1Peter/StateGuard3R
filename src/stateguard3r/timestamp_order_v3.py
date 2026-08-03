"""Causal RGB capture-timestamp order evidence for Detector v3.

This module deliberately accepts only a raw ``rgb.txt`` listing plus final
RGB paths.  It has no API parameter for source indices, ground truth, labels,
corruption metadata, model responses, or future frames.  The resulting
sidecar is therefore a narrow capability: it detects reordering only when the
original capture timestamps are preserved with the packets.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "stateguard3r.timestamp-order-v3.v1"
PARSER_VERSION = "tum-rgb-txt-decimal-strict.v1"


class TimestampOrderError(ValueError):
    """Raised when capture-timestamp provenance or order is invalid."""


@dataclass(frozen=True)
class CaptureTimestamp:
    """One raw RGB listing entry, tied to a concrete immutable RGB file."""

    path: Path
    timestamp: Decimal
    timestamp_text: str
    physical_line: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, *, label: str) -> Path:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise TimestampOrderError(f"cannot stat {label}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise TimestampOrderError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve(strict=True)


def _finite_decimal(text: str, *, label: str) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise TimestampOrderError(f"{label} is not a decimal timestamp") from error
    if not value.is_finite():
        raise TimestampOrderError(f"{label} must be finite")
    return value


def parse_tum_rgb_capture_index(
    rgb_txt: str | os.PathLike[str],
    *,
    dataset_root: str | os.PathLike[str],
) -> dict[Path, CaptureTimestamp]:
    """Parse raw TUM ``rgb.txt`` entries without deriving timestamps elsewhere.

    Paths in the listing must be relative, resolve beneath ``dataset_root``,
    name regular non-symlink files, and occur only once.  TUM listings are
    expected to be non-decreasing in their own capture order; this check makes
    a malformed raw listing fail rather than becoming a detector input.
    """

    listing = _regular_file(Path(rgb_txt), label="rgb.txt")
    root = Path(dataset_root).resolve(strict=True)
    if not root.is_dir():
        raise TimestampOrderError(f"dataset_root is not a directory: {root}")
    try:
        lines = listing.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise TimestampOrderError(f"cannot read rgb.txt: {error}") from error

    entries: dict[Path, CaptureTimestamp] = {}
    previous: Decimal | None = None
    for physical_line, raw_line in enumerate(lines, start=1):
        text = raw_line.strip()
        if not text or text.startswith("#"):
            continue
        fields = text.split()
        if len(fields) != 2:
            raise TimestampOrderError(f"rgb.txt line {physical_line} must contain timestamp and path")
        timestamp = _finite_decimal(fields[0], label=f"rgb.txt line {physical_line}")
        relative = Path(fields[1])
        if relative.is_absolute() or ".." in relative.parts:
            raise TimestampOrderError(f"rgb.txt line {physical_line} has unsafe relative path")
        candidate = root / relative
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise TimestampOrderError(f"rgb.txt line {physical_line} path is missing: {candidate}") from error
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise TimestampOrderError(f"rgb.txt line {physical_line} escapes dataset_root") from error
        _regular_file(resolved, label=f"rgb.txt line {physical_line} RGB")
        if resolved in entries:
            raise TimestampOrderError(f"rgb.txt repeats RGB path at line {physical_line}")
        if previous is not None and timestamp < previous:
            raise TimestampOrderError(f"rgb.txt capture timestamps decrease at line {physical_line}")
        entries[resolved] = CaptureTimestamp(
            path=resolved,
            timestamp=timestamp,
            timestamp_text=fields[0],
            physical_line=physical_line,
        )
        previous = timestamp
    if not entries:
        raise TimestampOrderError("rgb.txt contains no RGB entries")
    return entries


def capture_timestamp_records(
    frame_paths: Sequence[str | os.PathLike[str]],
    *,
    rgb_txt: str | os.PathLike[str],
    dataset_root: str | os.PathLike[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind ordered input RGB files to their raw capture timestamps.

    The input is intentionally just an ordered path sequence.  No caller can
    pass source indices or other metadata to this function, which prevents
    deriving an apparent order signal from corruption construction details.
    """

    if not frame_paths:
        raise TimestampOrderError("frame_paths must not be empty")
    listing = _regular_file(Path(rgb_txt), label="rgb.txt")
    index = parse_tum_rgb_capture_index(listing, dataset_root=dataset_root)
    records: list[dict[str, Any]] = []
    for frame_id, value in enumerate(frame_paths):
        path = _regular_file(Path(value), label=f"frame {frame_id}")
        capture = index.get(path)
        if capture is None:
            raise TimestampOrderError(f"frame {frame_id} is absent from raw rgb.txt")
        records.append(
            {
                "frame_id": frame_id,
                "rgb_capture_timestamp": str(capture.timestamp),
                "rgb_capture_timestamp_text": capture.timestamp_text,
                "rgb_txt_physical_line": capture.physical_line,
                "rgb_path_sha256": _sha256(path),
            }
        )
    provenance = {
        "parser_version": PARSER_VERSION,
        "rgb_txt_path": str(listing),
        "rgb_txt_sha256": _sha256(listing),
        "dataset_root": str(Path(dataset_root).resolve(strict=True)),
        "input_contract": "ordered_rgb_paths_only_no_source_index_gt_label_or_event_metadata",
    }
    return records, provenance


def timestamp_order_sidecar(
    captures: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a finite, causal timestamp-order sidecar from bound captures."""

    if not captures:
        raise TimestampOrderError("captures must not be empty")
    expected_capture_keys = {
        "frame_id",
        "rgb_capture_timestamp",
        "rgb_capture_timestamp_text",
        "rgb_txt_physical_line",
        "rgb_path_sha256",
    }
    rows: list[dict[str, Any]] = []
    previous: Decimal | None = None
    for frame_id, raw in enumerate(captures):
        if not isinstance(raw, Mapping) or set(raw) != expected_capture_keys:
            raise TimestampOrderError("capture record has an invalid schema")
        if raw.get("frame_id") != frame_id:
            raise TimestampOrderError("capture records must have contiguous frame_id")
        text = raw.get("rgb_capture_timestamp_text")
        encoded = raw.get("rgb_capture_timestamp")
        if not isinstance(text, str) or not isinstance(encoded, str):
            raise TimestampOrderError("capture timestamp text and encoding must be strings")
        timestamp = _finite_decimal(encoded, label=f"capture frame {frame_id}")
        if _finite_decimal(text, label=f"capture frame {frame_id} text") != timestamp:
            raise TimestampOrderError("capture timestamp text does not match its decimal encoding")
        line = raw.get("rgb_txt_physical_line")
        digest = raw.get("rgb_path_sha256")
        if type(line) is not int or line < 1 or not isinstance(digest, str) or len(digest) != 64:
            raise TimestampOrderError("capture provenance is malformed")
        violation = None if previous is None else bool(timestamp <= previous)
        rows.append(
            {
                **dict(raw),
                "timestamp_order_violation": violation,
                "predicate": "null_first_frame_else_current_capture_timestamp_lte_previous",
            }
        )
        previous = timestamp
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "parser_version",
        "rgb_txt_path",
        "rgb_txt_sha256",
        "dataset_root",
        "input_contract",
    }:
        raise TimestampOrderError("timestamp provenance schema is invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "online_capture_timestamp_order_invariant",
        "provenance": dict(provenance),
        "records": rows,
    }


def validate_timestamp_order_sidecar(
    sidecar: Mapping[str, Any],
    *,
    require_available: bool = True,
) -> list[bool | None]:
    """Validate a sidecar and return its causal predicates in frame order."""

    if not isinstance(sidecar, Mapping) or set(sidecar) != {
        "schema_version",
        "purpose",
        "provenance",
        "records",
    }:
        raise TimestampOrderError("timestamp sidecar schema is invalid")
    if sidecar.get("schema_version") != SCHEMA_VERSION or sidecar.get("purpose") != "online_capture_timestamp_order_invariant":
        raise TimestampOrderError("timestamp sidecar version or purpose is invalid")
    records = sidecar.get("records")
    if not isinstance(records, list):
        raise TimestampOrderError("timestamp sidecar records must be a list")
    capture_keys = {
        "frame_id",
        "rgb_capture_timestamp",
        "rgb_capture_timestamp_text",
        "rgb_txt_physical_line",
        "rgb_path_sha256",
    }
    sidecar_keys = capture_keys | {"timestamp_order_violation", "predicate"}
    captures: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != sidecar_keys:
            raise TimestampOrderError("timestamp sidecar record has an invalid schema")
        captures.append({key: record[key] for key in capture_keys})
    recomputed = timestamp_order_sidecar(captures, provenance=sidecar.get("provenance", {}))
    if json.dumps(recomputed, sort_keys=True, separators=(",", ":")) != json.dumps(dict(sidecar), sort_keys=True, separators=(",", ":")):
        raise TimestampOrderError("timestamp sidecar predicate or provenance does not replay")
    predicates = [record["timestamp_order_violation"] for record in records]
    if require_available and any(value is None for value in predicates[1:]):
        raise TimestampOrderError("timestamp order is unavailable after frame zero")
    return predicates


def write_timestamp_order_sidecar_atomic(
    path: str | os.PathLike[str], sidecar: Mapping[str, Any]
) -> Path:
    """Validate then atomically publish a canonical timestamp sidecar."""

    validate_timestamp_order_sidecar(sidecar)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(sidecar, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass
    return destination


__all__ = [
    "CaptureTimestamp",
    "PARSER_VERSION",
    "SCHEMA_VERSION",
    "TimestampOrderError",
    "capture_timestamp_records",
    "parse_tum_rgb_capture_index",
    "timestamp_order_sidecar",
    "validate_timestamp_order_sidecar",
    "write_timestamp_order_sidecar_atomic",
]
