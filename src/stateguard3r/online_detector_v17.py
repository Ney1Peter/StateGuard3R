"""Fresh bounded scalar Detector-v3 runtime arithmetic for v17.

The detector accepts only a current scalar health row and a Boolean order
predicate.  It cannot receive image arrays, model objects, tensors, output
files, labels, or any offline artifact.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


SIGNALS = ("geometric_residual", "pose_jump", "update_magnitude", "overlap", "reliability")
ALIASES = {
    "geometric_residual": (("geometric_residual", "high"),),
    "pose_jump": (("pose_jump", "high"),),
    "update_magnitude": (("update_magnitude", "high"),),
    "overlap": (("overlap", "low"),),
    "reliability": (("reliability", "low"), ("uncertainty_u", "high")),
}
ROW_FIELDS = frozenset({"frame_id", "overlap", "pose_jump", "geometric_residual", "update_magnitude", "reliability", "uncertainty_u", "timestamp_order_alarm"})


class OnlineDetectorV17Error(ValueError):
    """The v17 scalar current-health capability is malformed."""


@dataclass(frozen=True)
class DetectorScoringConfigV17:
    window: int
    min_history: int
    max_z: float
    master_seed: int

    def __post_init__(self) -> None:
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 1 for name in ("window", "min_history")):
            raise OnlineDetectorV17Error("window and min_history must be positive integers")
        if self.min_history > self.window or not math.isfinite(self.max_z) or self.max_z <= 0.0 or type(self.master_seed) is not int or self.master_seed < 0:
            raise OnlineDetectorV17Error("frozen detector scoring configuration differs")


@dataclass(frozen=True)
class FrozenDetectorV17Config:
    scoring: DetectorScoringConfigV17
    scale_floors: Mapping[str, float]
    continuous_threshold: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FrozenDetectorV17Config":
        if not isinstance(value, Mapping):
            raise OnlineDetectorV17Error("frozen detector config must be an object")
        scoring, floors, thresholds = value.get("continuous_scoring"), value.get("scale_floors"), value.get("thresholds")
        if not isinstance(scoring, Mapping) or not isinstance(floors, Mapping) or not isinstance(thresholds, Mapping):
            raise OnlineDetectorV17Error("frozen detector config lacks a required section")
        if set(scoring) != {"window", "min_history", "max_z", "master_seed"} or set(floors) != set(SIGNALS) or set(thresholds) != {"combined", "random", "reliability_only", "update_magnitude_only"} or value.get("hybrid_rule") != "continuous_score_gte_threshold_or_timestamp_order_violation":
            raise OnlineDetectorV17Error("frozen detector config schema differs")
        try:
            parsed_floors = {name: float(floors[name]) for name in SIGNALS}
            threshold = float(thresholds["combined"])
        except (TypeError, ValueError) as error:
            raise OnlineDetectorV17Error("frozen detector scalar is invalid") from error
        if not math.isfinite(threshold) or any(not math.isfinite(item) or item < 1e-6 for item in parsed_floors.values()):
            raise OnlineDetectorV17Error("frozen detector scalar is outside its fixed domain")
        return cls(DetectorScoringConfigV17(**dict(scoring)), parsed_floors, threshold)


@dataclass(frozen=True)
class OnlineDetectorDecisionV17:
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
    decision: OnlineDetectorDecisionV17


def _number(value: Any) -> float:
    if value is None or isinstance(value, (bool, Mapping, list, tuple)):
        return float("nan")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def _row(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != ROW_FIELDS or type(value.get("frame_id")) is not int or value["frame_id"] < 0 or type(value.get("timestamp_order_alarm")) is not bool:
        raise OnlineDetectorV17Error("raw scalar health row schema differs")
    for name in ROW_FIELDS - {"frame_id", "timestamp_order_alarm"}:
        scalar = value[name]
        if scalar is not None and not math.isfinite(_number(scalar)):
            raise OnlineDetectorV17Error("raw scalar health value is nonfinite")
    return {name: value[name] for name in sorted(ROW_FIELDS)}


def _component(values: np.ndarray, *, direction: str, config: DetectorScoringConfigV17, floor: float) -> float:
    current = float(values[-1])
    if not math.isfinite(current):
        return float("nan")
    history = values[max(0, len(values) - 1 - config.window): -1]
    history = history[np.isfinite(history)]
    if len(history) < config.min_history:
        return 0.0
    center = float(np.median(history))
    mad = float(np.median(np.abs(history - center)))
    signed = (current - center) if direction == "high" else (center - current)
    return min(max(signed / max(1.4826 * mad, floor), 0.0), config.max_z)


def _digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(list(rows), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


class IncrementalOnlinePrefixDetectorV17:
    """Commit-only rolling detector over its own bounded scalar prefix."""

    def __init__(self, config: FrozenDetectorV17Config) -> None:
        self._config = config
        self._history: deque[dict[str, Any]] = deque(maxlen=config.scoring.window)
        self._count = 0
        self._seen = {signal: {name: False for name, _ in aliases} for signal, aliases in ALIASES.items()}
        self._pending: _Pending | None = None

    @property
    def retained_row_count(self) -> int:
        return len(self._history)

    def observe(self, raw_current_health: Mapping[str, Any]) -> OnlineDetectorDecisionV17:
        if self._pending is not None:
            raise OnlineDetectorV17Error("prior current health was not committed")
        row = _row(raw_current_health)
        if row["frame_id"] != self._count:
            raise OnlineDetectorV17Error("current health does not extend the causal prefix")
        components: list[float] = []
        for signal, aliases in ALIASES.items():
            chosen = next(
                ((name, direction) for name, direction in aliases if self._seen[signal][name] or math.isfinite(_number(row[name]))),
                None,
            )
            if chosen is None:
                continue
            name, direction = chosen
            score = _component(np.asarray([_number(prior[name]) for prior in self._history] + [_number(row[name])], dtype=np.float64), direction=direction, config=self._config.scoring, floor=self._config.scale_floors[signal])
            if math.isfinite(score):
                components.append(score)
        continuous = float(sum(components) / len(components)) if components else 0.0
        order_alarm = bool(row["timestamp_order_alarm"])
        decision = OnlineDetectorDecisionV17(row["frame_id"], continuous, max(continuous, self._config.continuous_threshold + 1.0) if order_alarm else continuous, continuous >= self._config.continuous_threshold, order_alarm, continuous >= self._config.continuous_threshold or order_alarm, len(components), _digest([*self._history, row]))
        self._pending = _Pending(row, decision)
        return decision

    def commit(self, raw_current_health: Mapping[str, Any]) -> None:
        row = _row(raw_current_health)
        if self._pending is None or row != self._pending.row:
            raise OnlineDetectorV17Error("current health commit differs from the pending row")
        for signal, aliases in ALIASES.items():
            for name, _direction in aliases:
                if math.isfinite(_number(row[name])):
                    self._seen[signal][name] = True
        self._history.append(row)
        self._count += 1
        self._pending = None


__all__ = ["DetectorScoringConfigV17", "FrozenDetectorV17Config", "IncrementalOnlinePrefixDetectorV17", "OnlineDetectorDecisionV17", "OnlineDetectorV17Error", "ROW_FIELDS", "SIGNALS"]
