"""Prefix-only Detector-v3 adapter for online recovery development.

The formal Detector-v3 configuration is immutable, but its original batch
evaluation API receives a whole health ledger.  This adapter exposes the same
calculation one candidate frame at a time.  Its retained history contains only
committed model health; a quarantined candidate is represented solely by its
safe RGB-overlap placeholder.  The full observed candidate is deliberately
kept outside this object by the runner's audit ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .detection_v2 import V2ScoringConfig, compute_v2_scores
from .detection_v3 import hybridize_v2_continuous_scores
from .health import HealthFrame
from .timestamp_order_v3 import timestamp_order_sidecar


SCHEMA_VERSION = "stateguard3r.online-detector-v2.v1"
FORBIDDEN_ONLINE_TOKENS = (
    "groundtruth",
    "ground_truth",
    "depth",
    "label",
    "event",
    "source_index",
    "future",
)
# These are the only health fields passed to Detector-v2.  In particular,
# ``timestamp`` is not accepted: Detector-v3's order channel is bound directly
# to raw RGB capture records below rather than to manifest association metadata.
ALLOWED_HEALTH_FIELDS = frozenset(
    {
        "frame_id",
        "overlap",
        "pose_jump",
        "geometric_residual",
        "update_magnitude",
        "uncertainty_u",
        "reliability",
        "global_state_delta",
        "local_mem_delta",
    }
)


class OnlineDetectorError(ValueError):
    """Raised when online detector inputs or frozen configuration are unsafe."""


@dataclass(frozen=True)
class FrozenDetectorV3Config:
    """The subset of the frozen formal-v3 configuration needed online."""

    scoring: V2ScoringConfig
    scale_floors: Mapping[str, float]
    continuous_threshold: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FrozenDetectorV3Config":
        if not isinstance(payload, Mapping):
            raise OnlineDetectorError("frozen detector config must be an object")
        continuous = payload.get("continuous_scoring")
        floors = payload.get("scale_floors")
        thresholds = payload.get("thresholds")
        if not isinstance(continuous, Mapping) or not isinstance(floors, Mapping) or not isinstance(thresholds, Mapping):
            raise OnlineDetectorError("frozen detector config lacks continuous_scoring, scale_floors, or thresholds")
        expected_scoring = {"window", "min_history", "max_z", "master_seed"}
        if set(continuous) != expected_scoring:
            raise OnlineDetectorError("frozen detector continuous scoring keys differ")
        if set(thresholds) != {"combined", "random", "reliability_only", "update_magnitude_only"}:
            raise OnlineDetectorError("frozen detector threshold keys differ")
        if payload.get("hybrid_rule") != "continuous_score_gte_threshold_or_timestamp_order_violation":
            raise OnlineDetectorError("frozen detector hybrid rule differs")
        try:
            threshold = float(thresholds["combined"])
        except (TypeError, ValueError) as error:
            raise OnlineDetectorError("frozen combined threshold is invalid") from error
        if not math.isfinite(threshold):
            raise OnlineDetectorError("frozen combined threshold must be finite")
        return cls(
            scoring=V2ScoringConfig(**dict(continuous)),
            scale_floors={str(name): float(value) for name, value in floors.items()},
            continuous_threshold=threshold,
        )


@dataclass(frozen=True)
class OnlineDetectorDecision:
    """One current-frame decision plus scalar evidence for the audit timeline."""

    frame_id: int
    continuous_score: float
    hybrid_score: float
    continuous_alarm: bool
    timestamp_order_alarm: bool
    hybrid_alarm: bool
    finite_component_count: int
    policy_input_sha256: str


def _online_row(record: HealthFrame | Mapping[str, Any]) -> dict[str, Any]:
    """Project a candidate health record into the narrow detector capability."""

    is_validated_health = isinstance(record, HealthFrame)
    raw = record.to_dict() if is_validated_health else dict(record)
    if not is_validated_health and set(raw) - ALLOWED_HEALTH_FIELDS:
        unknown = sorted(str(value) for value in set(raw) - ALLOWED_HEALTH_FIELDS)
        raise OnlineDetectorError("online detector health contains forbidden field(s): " + ", ".join(unknown))
    for key in raw:
        lowered = key.lower()
        if any(token in lowered for token in FORBIDDEN_ONLINE_TOKENS):
            raise OnlineDetectorError(f"online detector health field {key!r} is forbidden")
    if type(raw.get("frame_id")) is not int or int(raw["frame_id"]) < 0:
        raise OnlineDetectorError("online detector health frame_id must be a non-negative integer")
    result = {key: raw.get(key) for key in sorted(ALLOWED_HEALTH_FIELDS) if key in raw}
    if "frame_id" not in result:
        raise OnlineDetectorError("online detector health lacks frame_id")
    return result


def _input_digest(rows: Sequence[Mapping[str, Any]], captures: Sequence[Mapping[str, Any]]) -> str:
    payload = {"health": list(rows), "capture_prefix": list(captures)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class OnlinePrefixDetector:
    """Current-frame detector whose score can observe only the causal prefix.

    ``observe`` has no mutating effect.  The runner must subsequently call
    exactly one of ``commit`` or ``quarantine`` for that decision.  This makes
    the policy's history explicit and prevents observed rollback health from
    accidentally becoming a later detector feature.
    """

    def __init__(
        self,
        config: FrozenDetectorV3Config,
        *,
        captures: Sequence[Mapping[str, Any]],
        capture_provenance: Mapping[str, Any],
    ) -> None:
        if not captures:
            raise OnlineDetectorError("online detector needs capture timestamp records")
        self._config = config
        self._captures = tuple(dict(value) for value in captures)
        self._capture_provenance = dict(capture_provenance)
        self._history: list[dict[str, Any]] = []
        self._pending: OnlineDetectorDecision | None = None

    @property
    def history_frame_ids(self) -> tuple[int, ...]:
        return tuple(int(row["frame_id"]) for row in self._history)

    def observe(self, candidate: HealthFrame | Mapping[str, Any]) -> OnlineDetectorDecision:
        if self._pending is not None:
            raise OnlineDetectorError("previous online detector decision was not finalized")
        row = _online_row(candidate)
        frame_id = int(row["frame_id"])
        if frame_id != len(self._history) or frame_id >= len(self._captures):
            raise OnlineDetectorError("online detector candidate does not extend the causal prefix")
        rows = [*self._history, row]
        sidecar = timestamp_order_sidecar(self._captures[: frame_id + 1], provenance=self._capture_provenance)
        scores = compute_v2_scores(
            rows,
            config=self._config.scoring,
            scale_floors=self._config.scale_floors,
            require_online_overlap=True,
        )
        hybrid = hybridize_v2_continuous_scores(
            scores["methods"]["combined"]["scores"],
            continuous_threshold=self._config.continuous_threshold,
            timestamp_sidecar=sidecar,
        )
        attribution = hybrid["attribution"][-1]
        decision = OnlineDetectorDecision(
            frame_id=frame_id,
            continuous_score=float(scores["methods"]["combined"]["scores"][-1]),
            hybrid_score=float(hybrid["hybrid_scores"][-1]),
            continuous_alarm=bool(attribution["continuous_alarm"]),
            timestamp_order_alarm=bool(attribution["timestamp_order_alarm"]),
            hybrid_alarm=bool(attribution["hybrid_alarm"]),
            finite_component_count=int(scores["methods"]["combined"]["finite_component_count"][-1]),
            policy_input_sha256=_input_digest(rows, self._captures[: frame_id + 1]),
        )
        self._pending = decision
        return decision

    def commit(self, candidate: HealthFrame | Mapping[str, Any]) -> None:
        self._finalize(candidate, quarantined=False)

    def quarantine(self, candidate: HealthFrame | Mapping[str, Any]) -> None:
        self._finalize(candidate, quarantined=True)

    def _finalize(self, candidate: HealthFrame | Mapping[str, Any], *, quarantined: bool) -> None:
        row = _online_row(candidate)
        if self._pending is None or self._pending.frame_id != row["frame_id"]:
            raise OnlineDetectorError("online detector finalization differs from pending candidate")
        if quarantined:
            # Overlap is raw current/previous RGB evidence, not model response.
            # Every other candidate health field is discarded from policy history.
            row = {"frame_id": row["frame_id"], "overlap": row.get("overlap")}
        self._history.append(row)
        self._pending = None


__all__ = [
    "ALLOWED_HEALTH_FIELDS",
    "FORBIDDEN_ONLINE_TOKENS",
    "FrozenDetectorV3Config",
    "OnlineDetectorDecision",
    "OnlineDetectorError",
    "OnlinePrefixDetector",
    "SCHEMA_VERSION",
]
