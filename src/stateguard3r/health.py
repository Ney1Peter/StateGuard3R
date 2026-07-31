"""Validated per-frame health records and JSONL persistence.

The module deliberately depends only on the Python standard library and NumPy.
It does not import ReCal3R or PyTorch; the trace adapter uses small amounts of
duck typing so detached CPU torch tensors can still be consumed when present.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
import json
import math
import numbers
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


RELATION_ABS_TOL = 1e-6


class HealthValidationError(ValueError):
    """Raised when a health record violates the v0 schema."""


class HealthJSONLError(ValueError):
    """Raised when a JSONL stream cannot be decoded as health records."""


def _validated_frame_id(value: Any, *, name: str = "frame_id") -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise HealthValidationError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise HealthValidationError(f"{name} must be a non-negative integer")
    return result


def _validated_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise HealthValidationError(f"{name} must be a real number or null")
    result = float(value)
    if not math.isfinite(result):
        raise HealthValidationError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise HealthValidationError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise HealthValidationError(f"{name} must be <= {maximum}")
    return result


def _validated_decision(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise HealthValidationError("decision must be a string or null")
    if not value or value != value.strip():
        raise HealthValidationError(
            "decision must be non-empty and have no leading/trailing whitespace"
        )
    if len(value) > 128 or any(ord(char) < 32 for char in value):
        raise HealthValidationError(
            "decision must contain at most 128 printable characters"
        )
    return value


def _is_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=RELATION_ABS_TOL)


@dataclass(frozen=True, slots=True)
class HealthFrame:
    """Health Ledger v0 record for one stream frame.

    Every unavailable observation remains ``None`` and is serialized as JSON
    ``null``. ``uncertainty_u`` follows the current ReCal3R trace convention:
    it is the fallback/uncertainty weight, so reliability is exactly ``1-u``.
    When only one of the pair is supplied, the other is derived; contradictory
    values are rejected.
    """

    frame_id: int
    timestamp: float | None = None
    overlap: float | None = None
    pose_jump: float | None = None
    geometric_residual: float | None = None
    update_magnitude: float | None = None
    uncertainty_u: float | None = None
    reliability: float | None = None
    candidate_beta: float | None = None
    final_beta: float | None = None
    global_state_delta: float | None = None
    local_mem_delta: float | None = None
    risk_score: float | None = None
    decision: str | None = None

    # Optional ReCal3R frame-summary details. They preserve measured trace
    # statistics without pretending that missing internal signals were logged.
    uncertainty_u_min: float | None = None
    uncertainty_u_max: float | None = None
    reliability_min: float | None = None
    reliability_max: float | None = None
    attention_entropy_mean: float | None = None
    attention_entropy_min: float | None = None
    attention_entropy_max: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _validated_frame_id(self.frame_id))

        # The proposed health-risk formula subtracts reliability, so its raw
        # score may legitimately be negative before threshold calibration.
        finite_fields = ("timestamp", "risk_score")
        non_negative_fields = (
            "pose_jump",
            "geometric_residual",
            "update_magnitude",
            "global_state_delta",
            "local_mem_delta",
        )
        unit_interval_fields = (
            "overlap",
            "uncertainty_u",
            "reliability",
            "candidate_beta",
            "final_beta",
            "uncertainty_u_min",
            "uncertainty_u_max",
            "reliability_min",
            "reliability_max",
            "attention_entropy_mean",
            "attention_entropy_min",
            "attention_entropy_max",
        )

        for name in finite_fields:
            object.__setattr__(
                self, name, _validated_float(getattr(self, name), name=name)
            )
        for name in non_negative_fields:
            object.__setattr__(
                self,
                name,
                _validated_float(getattr(self, name), name=name, minimum=0.0),
            )
        for name in unit_interval_fields:
            object.__setattr__(
                self,
                name,
                _validated_float(
                    getattr(self, name), name=name, minimum=0.0, maximum=1.0
                ),
            )

        object.__setattr__(self, "decision", _validated_decision(self.decision))
        self._complete_and_validate_uncertainty_relation()
        self._validate_summary_order()

    def _complete_and_validate_uncertainty_relation(self) -> None:
        uncertainty = self.uncertainty_u
        reliability = self.reliability
        if uncertainty is None and reliability is not None:
            uncertainty = 1.0 - reliability
            object.__setattr__(self, "uncertainty_u", uncertainty)
        elif reliability is None and uncertainty is not None:
            reliability = 1.0 - uncertainty
            object.__setattr__(self, "reliability", reliability)
        elif uncertainty is not None and reliability is not None:
            expected = 1.0 - uncertainty
            if not _is_close(reliability, expected):
                raise HealthValidationError(
                    "reliability must equal 1 - uncertainty_u "
                    f"(got reliability={reliability}, expected={expected})"
                )
            object.__setattr__(self, "reliability", expected)

        u_bounds = (self.uncertainty_u_min, self.uncertainty_u_max)
        r_bounds = (self.reliability_min, self.reliability_max)
        if (u_bounds[0] is None) != (u_bounds[1] is None):
            raise HealthValidationError(
                "uncertainty_u_min and uncertainty_u_max must be supplied together"
            )
        if (r_bounds[0] is None) != (r_bounds[1] is None):
            raise HealthValidationError(
                "reliability_min and reliability_max must be supplied together"
            )

        if u_bounds[0] is not None and self.uncertainty_u is None:
            raise HealthValidationError(
                "uncertainty summary bounds require uncertainty_u"
            )
        if r_bounds[0] is not None and self.reliability is None:
            raise HealthValidationError("reliability summary bounds require reliability")

        if u_bounds[0] is not None and r_bounds[0] is None:
            object.__setattr__(self, "reliability_min", 1.0 - u_bounds[1])
            object.__setattr__(self, "reliability_max", 1.0 - u_bounds[0])
        elif r_bounds[0] is not None and u_bounds[0] is None:
            object.__setattr__(self, "uncertainty_u_min", 1.0 - r_bounds[1])
            object.__setattr__(self, "uncertainty_u_max", 1.0 - r_bounds[0])
        elif u_bounds[0] is not None and r_bounds[0] is not None:
            expected_r_min = 1.0 - u_bounds[1]
            expected_r_max = 1.0 - u_bounds[0]
            if not _is_close(r_bounds[0], expected_r_min) or not _is_close(
                r_bounds[1], expected_r_max
            ):
                raise HealthValidationError(
                    "reliability bounds must be the complement of uncertainty bounds"
                )
            object.__setattr__(self, "reliability_min", expected_r_min)
            object.__setattr__(self, "reliability_max", expected_r_max)

    def _validate_summary_order(self) -> None:
        summaries = (
            (
                "uncertainty_u",
                self.uncertainty_u_min,
                self.uncertainty_u,
                self.uncertainty_u_max,
            ),
            (
                "reliability",
                self.reliability_min,
                self.reliability,
                self.reliability_max,
            ),
            (
                "attention_entropy",
                self.attention_entropy_min,
                self.attention_entropy_mean,
                self.attention_entropy_max,
            ),
        )
        for name, minimum, mean, maximum in summaries:
            if minimum is None and maximum is None:
                continue
            if mean is None or minimum is None or maximum is None:
                raise HealthValidationError(
                    f"{name} min/mean/max must be supplied together"
                )
            if minimum > mean + RELATION_ABS_TOL or mean > maximum + RELATION_ABS_TOL:
                raise HealthValidationError(
                    f"{name} summary must satisfy min <= mean <= max"
                )

    @property
    def overlap_score(self) -> float | None:
        """Read-only compatibility alias for documentation using overlap_score."""

        return self.overlap

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary including explicit null fields."""

        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HealthFrame":
        """Construct a record from a strict schema mapping."""

        if not isinstance(value, Mapping):
            raise HealthValidationError("health record must be a JSON object")
        allowed = {field.name for field in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise HealthValidationError(f"unknown health field(s): {names}")
        if "frame_id" not in value:
            raise HealthValidationError("missing required health field: frame_id")
        return cls(**dict(value))


# A semantic alias for callers that prefer "record" terminology.
HealthRecord = HealthFrame


def _coerce_record(value: HealthFrame | Mapping[str, Any]) -> HealthFrame:
    if isinstance(value, HealthFrame):
        return value
    if isinstance(value, Mapping):
        return HealthFrame.from_dict(value)
    raise HealthValidationError("JSONL entries must be HealthFrame or mapping values")


def _encode_record(value: HealthFrame | Mapping[str, Any]) -> str:
    record = _coerce_record(value)
    return json.dumps(
        record.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            # Directory fsync is unsupported on some filesystems. The file has
            # already been flushed and atomically replaced, so treat this as a
            # best-effort durability step rather than reporting a false failure.
            pass
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def write_health_jsonl_atomic(
    path: str | os.PathLike[str],
    records: Iterable[HealthFrame | Mapping[str, Any]],
) -> Path:
    """Atomically replace ``path`` with validated records.

    Records are validated and written one at a time to a temporary file in the
    destination directory. If validation, serialization, or I/O fails, the old
    destination remains untouched.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            for record in records:
                stream.write(_encode_record(record))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        _fsync_directory(destination.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def append_health_jsonl(
    path: str | os.PathLike[str],
    records: Iterable[HealthFrame | Mapping[str, Any]],
    *,
    fsync: bool = False,
) -> Path:
    """Append validated records incrementally to a JSONL stream.

    This is the streaming counterpart to :func:`write_health_jsonl_atomic`.
    Appending is intentionally not advertised as whole-file atomic.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(_encode_record(record))
            stream.write("\n")
        stream.flush()
        if fsync:
            os.fsync(stream.fileno())
    return destination


def iter_health_jsonl(path: str | os.PathLike[str]) -> Iterator[HealthFrame]:
    """Yield validated records without loading the complete ledger into memory."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise HealthJSONLError(
                    f"{source}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            try:
                yield HealthFrame.from_dict(payload)
            except (HealthValidationError, TypeError) as error:
                raise HealthJSONLError(
                    f"{source}:{line_number}: invalid health record: {error}"
                ) from error


def read_health_jsonl(path: str | os.PathLike[str]) -> list[HealthFrame]:
    """Read a complete health ledger into a list."""

    return list(iter_health_jsonl(path))


def _detach_to_numpy(value: Any, *, name: str) -> np.ndarray:
    current = value
    for method_name in ("detach", "cpu"):
        method = getattr(current, method_name, None)
        if callable(method):
            current = method()
    try:
        result = np.asarray(current)
    except Exception as error:  # pragma: no cover - backend-specific message
        raise HealthValidationError(f"{name} cannot be converted to NumPy") from error
    return result


def _trace_entries(value: Any, *, name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    array = _detach_to_numpy(value, name=name)
    if array.ndim == 0:
        return [array]
    return [array[index] for index in range(array.shape[0])]


def _trace_scalar(value: Any, *, name: str) -> float:
    array = _detach_to_numpy(value, name=name)
    if array.size != 1:
        raise HealthValidationError(f"{name} must contain exactly one value")
    scalar = array.reshape(-1)[0]
    return _validated_float(scalar, name=name)  # type: ignore[return-value]


def _trace_frame_id(value: Any, *, name: str) -> int:
    scalar = _trace_scalar(value, name=name)
    if not scalar.is_integer():
        raise HealthValidationError(f"{name} must be a non-negative integer")
    return _validated_frame_id(int(scalar), name=name)


def _optional_summary_series(
    trace: Mapping[str, Any], key: str, expected_length: int
) -> list[float | None]:
    if key not in trace or trace[key] is None:
        return [None] * expected_length
    entries = _trace_entries(trace[key], name=key)
    if len(entries) != expected_length:
        raise HealthValidationError(
            f"trace {key} has {len(entries)} entries; expected {expected_length}"
        )
    result: list[float | None] = []
    for index, entry in enumerate(entries):
        if entry is None:
            result.append(None)
        else:
            result.append(_trace_scalar(entry, name=f"{key}[{index}]"))
    return result


def _mean_delta_by_frame(trace: Mapping[str, Any]) -> dict[int, float]:
    if trace.get("delta_norm") is None:
        return {}
    deltas = _trace_entries(trace["delta_norm"], name="delta_norm")
    if trace.get("frame_idx") is None:
        raise HealthValidationError("trace delta_norm requires frame_idx")
    frame_indices = _trace_entries(trace["frame_idx"], name="frame_idx")
    if len(deltas) != len(frame_indices):
        raise HealthValidationError(
            "trace delta_norm and frame_idx must have the same number of entries"
        )

    grouped: dict[int, list[np.ndarray]] = {}
    for entry_index, (delta_value, frame_value) in enumerate(
        zip(deltas, frame_indices, strict=True)
    ):
        delta_array = _detach_to_numpy(
            delta_value, name=f"delta_norm[{entry_index}]"
        ).astype(np.float64, copy=False)
        frame_array = _detach_to_numpy(
            frame_value, name=f"frame_idx[{entry_index}]"
        )
        delta_flat = delta_array.reshape(-1)
        frame_flat = frame_array.reshape(-1)
        if frame_flat.size == 1 and delta_flat.size != 1:
            frame_flat = np.repeat(frame_flat, delta_flat.size)
        if delta_flat.size != frame_flat.size:
            raise HealthValidationError(
                f"delta_norm[{entry_index}] and frame_idx[{entry_index}] size mismatch"
            )
        if not np.isfinite(delta_flat).all() or np.any(delta_flat < 0.0):
            raise HealthValidationError(
                f"delta_norm[{entry_index}] must contain finite non-negative values"
            )

        for frame_raw in np.unique(frame_flat):
            frame_id = _trace_frame_id(frame_raw, name="trace frame_idx")
            selected = delta_flat[frame_flat == frame_raw]
            grouped.setdefault(frame_id, []).append(selected)

    return {
        frame_id: float(np.concatenate(parts).mean())
        for frame_id, parts in grouped.items()
        if parts
    }


def _resolve_timestamps(
    timestamps: Mapping[int, Any] | Sequence[Any] | None,
    frame_ids: Sequence[int],
) -> list[float | None]:
    if timestamps is None:
        return [None] * len(frame_ids)
    if isinstance(timestamps, Mapping):
        result: list[float | None] = []
        for frame_id in frame_ids:
            value = timestamps.get(frame_id)
            result.append(
                _validated_float(value, name=f"timestamp[{frame_id}]")
                if value is not None
                else None
            )
        return result
    values = list(timestamps)
    if len(values) != len(frame_ids):
        raise HealthValidationError(
            f"timestamps has {len(values)} entries; expected {len(frame_ids)}"
        )
    return [
        _validated_float(value, name=f"timestamps[{index}]")
        if value is not None
        else None
        for index, value in enumerate(values)
    ]


def adapt_recal3r_trace(
    trace: Mapping[str, Any],
    *,
    frame_ids: Sequence[int] | None = None,
    all_frame_ids: Sequence[int] | None = None,
    timestamps: Mapping[int, Any] | Sequence[Any] | None = None,
    batch_size: int | None = None,
) -> list[HealthFrame]:
    """Convert ReCal3R ``get_u_calibration_trace`` frame summaries.

    The adapter never synthesizes unavailable ReCal3R internals. In particular,
    candidate/final beta, overlap, pose jump, geometric residual, update
    magnitude, and local-memory delta remain null unless another logger provides
    them. ReCal3R's trace ``u`` is interpreted as uncertainty/fallback weight;
    ledger reliability is derived as ``1-u``.

    ``frame_ids`` identifies the summary rows when ``frame_step`` is unavailable
    (or verifies it when present). ReCal3R normally has no summary for its first
    input frame or reset frames. Callers that need a record for every input can
    pass ``all_frame_ids``; frames without a summary are then emitted with null
    summary fields. This optional expansion leaves the default behavior intact.

    ReCal3R collapses batch and token dimensions in parts of this trace, so
    records from more than one sequence cannot be mapped back unambiguously.
    This adapter therefore supports batch size one only. Callers should pass the
    actual inference ``batch_size`` (or include integer ``batch_size`` metadata
    in the trace); any value other than one is rejected.
    """

    if not isinstance(trace, Mapping):
        raise HealthValidationError("ReCal3R trace must be a mapping")

    resolved_batch_size = batch_size
    trace_batch_size = trace.get("batch_size")
    if trace_batch_size is not None:
        if (
            isinstance(trace_batch_size, (bool, np.bool_))
            or not isinstance(trace_batch_size, numbers.Integral)
            or int(trace_batch_size) < 1
        ):
            raise HealthValidationError("trace batch_size must be a positive integer")
        trace_batch_size = int(trace_batch_size)
        if resolved_batch_size is not None and resolved_batch_size != trace_batch_size:
            raise HealthValidationError(
                "batch_size argument disagrees with trace batch_size metadata"
            )
        resolved_batch_size = trace_batch_size

    if resolved_batch_size is None:
        resolved_batch_size = 1
    if (
        isinstance(resolved_batch_size, (bool, np.bool_))
        or not isinstance(resolved_batch_size, numbers.Integral)
        or int(resolved_batch_size) < 1
    ):
        raise HealthValidationError("batch_size must be a positive integer")
    if int(resolved_batch_size) != 1:
        raise HealthValidationError(
            "ReCal3R trace adapter only supports batch_size=1; batch>1 trace "
            "rows cannot be mapped to ledger frames unambiguously"
        )

    if trace.get("frame_u_mean") is None:
        raise HealthValidationError("ReCal3R trace is missing frame_u_mean")

    u_mean_entries = _trace_entries(trace["frame_u_mean"], name="frame_u_mean")
    record_count = len(u_mean_entries)
    u_mean = [
        _trace_scalar(entry, name=f"frame_u_mean[{index}]")
        for index, entry in enumerate(u_mean_entries)
    ]

    trace_steps: list[int] | None = None
    if trace.get("frame_step") is not None:
        step_entries = _trace_entries(trace["frame_step"], name="frame_step")
        if len(step_entries) != record_count:
            raise HealthValidationError(
                f"trace frame_step has {len(step_entries)} entries; "
                f"expected {record_count}"
            )
        trace_steps = [
            _trace_frame_id(entry, name=f"frame_step[{index}]")
            for index, entry in enumerate(step_entries)
        ]

    if frame_ids is None:
        if trace_steps is None:
            if record_count:
                raise HealthValidationError(
                    "ReCal3R trace needs frame_step or explicit frame_ids"
                )
            resolved_frame_ids = []
        else:
            resolved_frame_ids = trace_steps
    else:
        resolved_frame_ids = [
            _validated_frame_id(value, name=f"frame_ids[{index}]")
            for index, value in enumerate(frame_ids)
        ]
        if len(resolved_frame_ids) != record_count:
            raise HealthValidationError(
                f"frame_ids has {len(resolved_frame_ids)} entries; "
                f"expected {record_count}"
            )
        if trace_steps is not None and resolved_frame_ids != trace_steps:
            raise HealthValidationError(
                "explicit frame_ids disagree with ReCal3R frame_step"
            )

    if len(set(resolved_frame_ids)) != len(resolved_frame_ids):
        raise HealthValidationError(
            "ReCal3R frame summaries contain duplicate frame IDs"
        )

    if all_frame_ids is None:
        output_frame_ids = resolved_frame_ids
    else:
        output_frame_ids = [
            _validated_frame_id(value, name=f"all_frame_ids[{index}]")
            for index, value in enumerate(all_frame_ids)
        ]
        if len(set(output_frame_ids)) != len(output_frame_ids):
            raise HealthValidationError("all_frame_ids contains duplicate frame IDs")
        output_frame_id_set = set(output_frame_ids)
        missing_summary_ids = [
            frame_id
            for frame_id in resolved_frame_ids
            if frame_id not in output_frame_id_set
        ]
        if missing_summary_ids:
            raise HealthValidationError(
                "all_frame_ids is missing ReCal3R summary frame ID(s): "
                + ", ".join(str(frame_id) for frame_id in missing_summary_ids)
            )

    u_min = _optional_summary_series(trace, "frame_u_min", record_count)
    u_max = _optional_summary_series(trace, "frame_u_max", record_count)
    h_mean = _optional_summary_series(trace, "frame_h_mean", record_count)
    h_min = _optional_summary_series(trace, "frame_h_min", record_count)
    h_max = _optional_summary_series(trace, "frame_h_max", record_count)
    resolved_timestamps = _resolve_timestamps(timestamps, output_frame_ids)
    delta_by_frame = _mean_delta_by_frame(trace)

    output_frame_id_set = set(output_frame_ids)
    unmatched_delta_ids = sorted(set(delta_by_frame) - output_frame_id_set)
    if unmatched_delta_ids:
        raise HealthValidationError(
            "trace delta frame ID(s) have no output ledger frame: "
            + ", ".join(str(frame_id) for frame_id in unmatched_delta_ids)
        )

    summary_index_by_frame = {
        frame_id: index for index, frame_id in enumerate(resolved_frame_ids)
    }
    records: list[HealthFrame] = []
    for output_index, frame_id in enumerate(output_frame_ids):
        summary_index = summary_index_by_frame.get(frame_id)
        records.append(
            HealthFrame(
                frame_id=frame_id,
                timestamp=resolved_timestamps[output_index],
                uncertainty_u=(
                    u_mean[summary_index] if summary_index is not None else None
                ),
                # Reliability is derived in HealthFrame.__post_init__ as 1-u.
                uncertainty_u_min=(
                    u_min[summary_index] if summary_index is not None else None
                ),
                uncertainty_u_max=(
                    u_max[summary_index] if summary_index is not None else None
                ),
                attention_entropy_mean=(
                    h_mean[summary_index] if summary_index is not None else None
                ),
                attention_entropy_min=(
                    h_min[summary_index] if summary_index is not None else None
                ),
                attention_entropy_max=(
                    h_max[summary_index] if summary_index is not None else None
                ),
                global_state_delta=delta_by_frame.get(frame_id),
            )
        )
    return records


# Concise aliases for callers that already operate in a health-ledger namespace.
write_jsonl_atomic = write_health_jsonl_atomic
write_health_jsonl = write_health_jsonl_atomic
append_jsonl = append_health_jsonl
iter_jsonl = iter_health_jsonl
read_jsonl = read_health_jsonl
recal3r_trace_to_health_frames = adapt_recal3r_trace


__all__ = [
    "HealthFrame",
    "HealthRecord",
    "HealthValidationError",
    "HealthJSONLError",
    "adapt_recal3r_trace",
    "append_health_jsonl",
    "append_jsonl",
    "iter_health_jsonl",
    "iter_jsonl",
    "read_health_jsonl",
    "read_jsonl",
    "recal3r_trace_to_health_frames",
    "write_health_jsonl_atomic",
    "write_health_jsonl",
    "write_jsonl_atomic",
]
