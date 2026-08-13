"""Self-contained causal Detector-v3 arithmetic for the v16 runtime graph."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


TIMESTAMP_SCORE_MARGIN = 1.0
CANONICAL_SIGNALS = ("geometric_residual", "pose_jump", "update_magnitude", "overlap", "reliability")
ALLOWED_HEALTH_FIELDS = frozenset({"frame_id", "overlap", "pose_jump", "geometric_residual", "update_magnitude", "uncertainty_u", "reliability", "global_state_delta", "local_mem_delta"})
SIGNAL_ALIASES = {
    "geometric_residual": (("geometric_residual", "high"),),
    "pose_jump": (("pose_jump", "high"),),
    "update_magnitude": (("update_magnitude", "high"), ("global_state_delta", "high")),
    "overlap": (("overlap", "low"),),
    "reliability": (("reliability", "low"), ("uncertainty_u", "high")),
}


class OnlineDetectorV16Error(ValueError):
    """The frozen v16 detector's allowed online capability was violated."""


@dataclass(frozen=True)
class V16ScoringConfig:
    window: int
    min_history: int
    max_z: float
    master_seed: int

    def __post_init__(self) -> None:
        if any(isinstance(getattr(self, key), bool) or not isinstance(getattr(self, key), int) or getattr(self, key) < 1 for key in ("window", "min_history")):
            raise OnlineDetectorV16Error("window and min_history must be positive integers")
        if self.min_history > self.window:
            raise OnlineDetectorV16Error("min_history may not exceed window")
        if not math.isfinite(self.max_z) or self.max_z <= 0.0:
            raise OnlineDetectorV16Error("max_z must be finite and positive")
        if isinstance(self.master_seed, bool) or not isinstance(self.master_seed, int) or self.master_seed < 0:
            raise OnlineDetectorV16Error("master_seed must be a non-negative integer")


@dataclass(frozen=True)
class FrozenDetectorV16Config:
    scoring: V16ScoringConfig
    scale_floors: Mapping[str, float]
    continuous_threshold: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FrozenDetectorV16Config":
        if not isinstance(payload, Mapping):
            raise OnlineDetectorV16Error("frozen detector config must be an object")
        continuous, floors, thresholds = payload.get("continuous_scoring"), payload.get("scale_floors"), payload.get("thresholds")
        if not isinstance(continuous, Mapping) or not isinstance(floors, Mapping) or not isinstance(thresholds, Mapping):
            raise OnlineDetectorV16Error("frozen detector config lacks required sections")
        if set(continuous) != {"window", "min_history", "max_z", "master_seed"} or set(floors) != set(CANONICAL_SIGNALS) or set(thresholds) != {"combined", "random", "reliability_only", "update_magnitude_only"}:
            raise OnlineDetectorV16Error("frozen detector config schema differs")
        if payload.get("hybrid_rule") != "continuous_score_gte_threshold_or_timestamp_order_violation":
            raise OnlineDetectorV16Error("frozen detector hybrid rule differs")
        try:
            validated_floors = {name: float(floors[name]) for name in CANONICAL_SIGNALS}
            threshold = float(thresholds["combined"])
        except (TypeError, ValueError) as error:
            raise OnlineDetectorV16Error("frozen detector scalar is invalid") from error
        if not math.isfinite(threshold) or any(not math.isfinite(value) or value < 1e-6 for value in validated_floors.values()):
            raise OnlineDetectorV16Error("frozen detector scalar is outside its fixed domain")
        return cls(V16ScoringConfig(**dict(continuous)), validated_floors, threshold)


@dataclass(frozen=True)
class OnlineDetectorDecisionV16:
    frame_id: int
    continuous_score: float
    hybrid_score: float
    continuous_alarm: bool
    timestamp_order_alarm: bool
    hybrid_alarm: bool
    finite_component_count: int
    policy_input_sha256: str


@dataclass(frozen=True)
class _Pending:
    row: Mapping[str, Any]
    decision: OnlineDetectorDecisionV16


def _numeric(value: Any) -> float:
    if value is None or isinstance(value, (bool, dict, list, tuple)):
        return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _row(candidate: Any) -> dict[str, Any]:
    is_record = callable(getattr(candidate, "to_dict", None))
    raw = candidate.to_dict() if is_record else dict(candidate) if isinstance(candidate, Mapping) else None
    if raw is None or (not is_record and set(raw) - ALLOWED_HEALTH_FIELDS):
        raise OnlineDetectorV16Error("online detector health has a non-v16 field")
    if type(raw.get("frame_id")) is not int or raw["frame_id"] < 0:
        raise OnlineDetectorV16Error("online detector health frame_id must be a non-negative integer")
    return {name: raw.get(name) for name in sorted(ALLOWED_HEALTH_FIELDS) if name in raw}


def _component(values: np.ndarray, *, direction: str, config: V16ScoringConfig, floor: float) -> float:
    current = float(values[-1])
    if not math.isfinite(current):
        return float("nan")
    history = values[max(0, values.size - 1 - config.window) : -1]
    history = history[np.isfinite(history)]
    if history.size < config.min_history:
        return 0.0
    center = float(np.median(history))
    mad = float(np.median(np.abs(history - center)))
    raw = (current - center) / max(1.4826 * mad, floor) if direction == "high" else (center - current) / max(1.4826 * mad, floor)
    return min(max(raw, 0.0), config.max_z)


def _digest(rows: Sequence[Mapping[str, Any]], captures: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(json.dumps({"health": list(rows), "capture_prefix": list(captures)}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


class IncrementalOnlinePrefixDetectorV16:
    """One-frame frozen scoring that stores only a bounded committed prefix."""

    def __init__(self, config: FrozenDetectorV16Config, *, captures: Sequence[Mapping[str, Any]], capture_provenance: Mapping[str, Any]) -> None:
        if not captures or not isinstance(capture_provenance, Mapping):
            raise OnlineDetectorV16Error("v16 detector needs RGB capture records and provenance")
        self._config = config
        self._captures = tuple(dict(value) for value in captures)
        self._capture_provenance = dict(capture_provenance)
        self._history: deque[dict[str, Any]] = deque(maxlen=config.scoring.window)
        self._history_count = 0
        self._seen = {signal: {field: False for field, _ in aliases} for signal, aliases in SIGNAL_ALIASES.items()}
        self._pending: _Pending | None = None

    @property
    def history_frame_ids(self) -> tuple[int, ...]:
        return tuple(int(row["frame_id"]) for row in self._history)

    @property
    def retained_row_count(self) -> int:
        return len(self._history)

    def observe(self, candidate: Any) -> OnlineDetectorDecisionV16:
        if self._pending is not None:
            raise OnlineDetectorV16Error("previous v16 detector decision was not finalized")
        row = _row(candidate)
        frame_id = int(row["frame_id"])
        if frame_id != self._history_count or frame_id >= len(self._captures):
            raise OnlineDetectorV16Error("v16 detector candidate does not extend the causal prefix")
        scores: list[float] = []
        for signal, aliases in SIGNAL_ALIASES.items():
            selected = next(((field, direction) for field, direction in aliases if self._seen[signal][field] or math.isfinite(_numeric(row.get(field)))), None)
            if selected is None:
                continue
            field, direction = selected
            values = np.asarray([_numeric(item.get(field)) for item in self._history] + [_numeric(row.get(field))], dtype=np.float64)
            score = _component(values, direction=direction, config=self._config.scoring, floor=self._config.scale_floors[signal])
            if math.isfinite(score):
                scores.append(score)
        continuous = float(sum(scores) / len(scores)) if scores else 0.0
        timestamp_alarm = self._timestamp_alarm(frame_id)
        decision = OnlineDetectorDecisionV16(frame_id, continuous, max(continuous, self._config.continuous_threshold + TIMESTAMP_SCORE_MARGIN) if timestamp_alarm else continuous, continuous >= self._config.continuous_threshold, timestamp_alarm, continuous >= self._config.continuous_threshold or timestamp_alarm, len(scores), _digest([*self._history, row], self._captures[: frame_id + 1]))
        self._pending = _Pending(row, decision)
        return decision

    def commit(self, candidate: Any) -> None:
        row = _row(candidate)
        if self._pending is None or row != self._pending.row:
            raise OnlineDetectorV16Error("v16 detector finalization differs from pending candidate")
        for signal, aliases in SIGNAL_ALIASES.items():
            for field, _ in aliases:
                if math.isfinite(_numeric(row.get(field))):
                    self._seen[signal][field] = True
        self._history.append(row)
        self._history_count += 1
        self._pending = None

    def _timestamp_alarm(self, frame_id: int) -> bool:
        if frame_id == 0:
            return False
        try:
            current = Decimal(str(self._captures[frame_id]["rgb_capture_timestamp"]))
            previous = Decimal(str(self._captures[frame_id - 1]["rgb_capture_timestamp"]))
        except (InvalidOperation, KeyError, TypeError, ValueError) as error:
            raise OnlineDetectorV16Error("v16 RGB capture timestamps are invalid") from error
        if not current.is_finite() or not previous.is_finite():
            raise OnlineDetectorV16Error("v16 RGB capture timestamps must be finite")
        return current <= previous


__all__ = ["ALLOWED_HEALTH_FIELDS", "CANONICAL_SIGNALS", "FrozenDetectorV16Config", "IncrementalOnlinePrefixDetectorV16", "OnlineDetectorDecisionV16", "OnlineDetectorV16Error", "TIMESTAMP_SCORE_MARGIN", "V16ScoringConfig"]
