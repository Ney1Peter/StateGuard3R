"""Bounded Detector-v3 implementation with a reference-compatible API.

The original online v2 adapter rescored every health prefix on every frame.
This version retains only the frozen rolling window plus persistent alias
availability bits.  That is enough to reproduce the current Detector-v2 score
exactly while keeping quarantined health restricted to its overlap placeholder.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from . import detection
from .detection_v2 import directional_causal_component
from .detection_v3 import TIMESTAMP_SCORE_MARGIN
from .online_detector_v2 import (
    ALLOWED_HEALTH_FIELDS,
    FORBIDDEN_ONLINE_TOKENS,
    FrozenDetectorV3Config,
    OnlineDetectorDecision,
    OnlineDetectorError,
    _online_row,
)


def _numeric(value: Any) -> float:
    if value is None or isinstance(value, (dict, list, tuple)):
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _input_digest(rows: Sequence[Mapping[str, Any]], captures: Sequence[Mapping[str, Any]]) -> str:
    payload = {"health": list(rows), "capture_prefix": list(captures)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Pending:
    row: Mapping[str, Any]
    decision: OnlineDetectorDecision


class IncrementalOnlinePrefixDetector:
    """Current-only bounded scorer for the frozen combined Detector-v3 path."""

    def __init__(self, config: FrozenDetectorV3Config, *, captures: Sequence[Mapping[str, Any]], capture_provenance: Mapping[str, Any]) -> None:
        if not captures:
            raise OnlineDetectorError("online detector needs capture timestamp records")
        self._config = config
        self._captures = tuple(dict(value) for value in captures)
        self._capture_provenance = dict(capture_provenance)
        self._history: deque[dict[str, Any]] = deque(maxlen=config.scoring.window)
        self._history_count = 0
        # Availability is updated only with finalized detector history.  In
        # particular, an alarmed candidate may influence its own decision but
        # cannot make a future score select one of its health fields.
        self._seen_aliases = {spec.name: {alias.field: False for alias in spec.aliases} for spec in detection.SIGNAL_SPECS}
        self._pending: _Pending | None = None

    @property
    def history_frame_ids(self) -> tuple[int, ...]:
        return tuple(row["frame_id"] for row in self._history)

    @property
    def retained_row_count(self) -> int:
        return len(self._history)

    def observe(self, candidate: Any) -> OnlineDetectorDecision:
        if self._pending is not None:
            raise OnlineDetectorError("previous online detector decision was not finalized")
        row = _online_row(candidate)
        frame_id = int(row["frame_id"])
        if frame_id != self._history_count or frame_id >= len(self._captures):
            raise OnlineDetectorError("online detector candidate does not extend the causal prefix")
        scores: list[float] = []
        for spec in detection.SIGNAL_SPECS:
            seen = self._seen_aliases[spec.name]
            selected = next(
                (
                    alias
                    for alias in spec.aliases
                    if seen[alias.field] or math.isfinite(_numeric(row.get(alias.field)))
                ),
                None,
            )
            if selected is None:
                continue
            values = np.asarray([_numeric(item.get(selected.field)) for item in self._history] + [_numeric(row.get(selected.field))], dtype=np.float64)
            component, _ = directional_causal_component(
                values,
                direction=selected.direction,
                window=self._config.scoring.window,
                min_history=self._config.scoring.min_history,
                scale_floor=float(self._config.scale_floors[spec.name]),
                max_z=self._config.scoring.max_z,
            )
            current = float(component[-1])
            if math.isfinite(current):
                scores.append(current)
        continuous = float(sum(scores) / len(scores)) if scores else 0.0
        timestamp_alarm = self._timestamp_alarm(frame_id)
        hybrid = max(continuous, self._config.continuous_threshold + TIMESTAMP_SCORE_MARGIN) if timestamp_alarm else continuous
        decision = OnlineDetectorDecision(
            frame_id=frame_id,
            continuous_score=continuous,
            hybrid_score=hybrid,
            continuous_alarm=continuous >= self._config.continuous_threshold,
            timestamp_order_alarm=timestamp_alarm,
            hybrid_alarm=continuous >= self._config.continuous_threshold or timestamp_alarm,
            finite_component_count=len(scores),
            policy_input_sha256=_input_digest([*self._history, row], self._captures[: frame_id + 1]),
        )
        self._pending = _Pending(row=row, decision=decision)
        return decision

    def commit(self, candidate: Any) -> None:
        self._finalize(candidate, quarantined=False)

    def quarantine(self, candidate: Any) -> None:
        self._finalize(candidate, quarantined=True)

    def _timestamp_alarm(self, frame_id: int) -> bool:
        if frame_id == 0:
            return False
        current = self._captures[frame_id]
        previous = self._captures[frame_id - 1]
        try:
            from decimal import Decimal

            return Decimal(str(current["rgb_capture_timestamp"])) <= Decimal(str(previous["rgb_capture_timestamp"]))
        except Exception as error:
            raise OnlineDetectorError("capture timestamp records are invalid") from error

    def _finalize(self, candidate: Any, *, quarantined: bool) -> None:
        row = _online_row(candidate)
        if self._pending is None or row != self._pending.row:
            raise OnlineDetectorError("online detector finalization differs from pending candidate")
        committed = {"frame_id": row["frame_id"], "overlap": row.get("overlap")} if quarantined else row
        for spec in detection.SIGNAL_SPECS:
            seen = self._seen_aliases[spec.name]
            for alias in spec.aliases:
                if math.isfinite(_numeric(committed.get(alias.field))):
                    seen[alias.field] = True
        self._history.append(committed)
        self._history_count += 1
        self._pending = None


__all__ = ["IncrementalOnlinePrefixDetector"]
