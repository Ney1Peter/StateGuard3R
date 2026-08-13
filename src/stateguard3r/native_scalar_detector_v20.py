"""Closed scalar-only online detector for v20 post-commit native health."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping


FIELDS = frozenset({
    "frame_id", "uncertainty_u", "reliability", "global_state_delta",
    "pose_jump", "geometric_residual",
})
SIGNALS = ("uncertainty_u", "global_state_delta", "pose_jump", "geometric_residual")


class NativeScalarDetectorV20Error(ValueError):
    """A v20 detector input is not a closed scalar current-health row."""


@dataclass(frozen=True, slots=True)
class NativeScalarDetectorV20Config:
    history: int = 5
    minimum_history: int = 3
    scale_floor: float = 1e-6
    max_score: float = 20.0
    threshold: float = 2.0

    def __post_init__(self) -> None:
        if type(self.history) is not int or type(self.minimum_history) is not int or self.history < 1 or self.minimum_history < 1 or self.minimum_history > self.history:
            raise NativeScalarDetectorV20Error("frozen v20 history configuration differs")
        for name in ("scale_floor", "max_score", "threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise NativeScalarDetectorV20Error(f"frozen v20 {name} differs")


@dataclass(frozen=True, slots=True)
class NativeScalarDecisionV20:
    frame_id: int
    score: float
    alarm: bool
    finite_component_count: int
    row_sha256: str


def _number(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NativeScalarDetectorV20Error(f"{label} must be a scalar or null")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise NativeScalarDetectorV20Error(f"{label} must be finite and nonnegative")
    return number


def canonical_native_scalar_row_v20(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a detector row with no channel capable of holding a tensor."""

    if not isinstance(value, Mapping) or set(value) != FIELDS:
        raise NativeScalarDetectorV20Error("v20 health row schema differs")
    if type(value.get("frame_id")) is not int or value["frame_id"] < 0:
        raise NativeScalarDetectorV20Error("v20 frame id differs")
    row = {"frame_id": value["frame_id"]}
    for name in SIGNALS:
        row[name] = _number(value[name], label=name)
    reliability = _number(value["reliability"], label="reliability")
    uncertainty = row["uncertainty_u"]
    if uncertainty is None and reliability is not None:
        uncertainty = 1.0 - reliability
        row["uncertainty_u"] = uncertainty
    if uncertainty is not None:
        if uncertainty > 1.0:
            raise NativeScalarDetectorV20Error("uncertainty_u must be <= 1")
        expected = 1.0 - uncertainty
        if reliability is not None and not math.isclose(reliability, expected, rel_tol=0.0, abs_tol=1e-6):
            raise NativeScalarDetectorV20Error("reliability must equal 1 - uncertainty_u")
        row["reliability"] = expected
    else:
        row["reliability"] = None
    return row


def _digest(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


class IncrementalNativeScalarDetectorV20:
    """A current-row-only robust detector with bounded committed history."""

    def __init__(self, config: NativeScalarDetectorV20Config = NativeScalarDetectorV20Config()) -> None:
        self._config = config
        self._history: deque[dict[str, Any]] = deque(maxlen=config.history)
        self._pending: tuple[dict[str, Any], NativeScalarDecisionV20] | None = None
        self._next_frame = 0

    def observe(self, raw: Mapping[str, Any]) -> NativeScalarDecisionV20:
        if self._pending is not None:
            raise NativeScalarDetectorV20Error("current v20 health was not committed")
        row = canonical_native_scalar_row_v20(raw)
        if row["frame_id"] != self._next_frame:
            raise NativeScalarDetectorV20Error("v20 health does not extend the current prefix")
        components: list[float] = []
        for name in SIGNALS:
            current = row[name]
            historic = [previous[name] for previous in self._history if previous[name] is not None]
            if current is None or len(historic) < self._config.minimum_history:
                continue
            center = float(sorted(historic)[len(historic) // 2])
            deviations = sorted(abs(value - center) for value in historic)
            mad = float(deviations[len(deviations) // 2])
            components.append(min(max((current - center) / max(1.4826 * mad, self._config.scale_floor), 0.0), self._config.max_score))
        score = float(sum(components) / len(components)) if components else 0.0
        decision = NativeScalarDecisionV20(row["frame_id"], score, score >= self._config.threshold, len(components), _digest(row))
        self._pending = (row, decision)
        return decision

    def commit(self, raw: Mapping[str, Any]) -> None:
        row = canonical_native_scalar_row_v20(raw)
        if self._pending is None or row != self._pending[0]:
            raise NativeScalarDetectorV20Error("v20 current health commit differs")
        self._history.append(row)
        self._next_frame += 1
        self._pending = None


__all__ = [
    "FIELDS", "SIGNALS", "IncrementalNativeScalarDetectorV20",
    "NativeScalarDecisionV20", "NativeScalarDetectorV20Config",
    "NativeScalarDetectorV20Error", "canonical_native_scalar_row_v20",
]
