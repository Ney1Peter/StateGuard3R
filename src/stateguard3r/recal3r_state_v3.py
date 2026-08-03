"""State-transaction primitives for the experimental ReCal3R v3 wrapper.

The pinned ``forward_recurrent_lighter`` routine returns an empty state list,
so its local recurrent tensors cannot be recovered through its public return
value.  This module deliberately does not patch that source.  Instead an
external wrapper captures the full local closure plus every known model-side
mutable ReCal3R attribute before a candidate update, and either commits the
candidate or restores that snapshot.

It has no dependency on torch at import time.  The isolated ReCal3R execution
environment supplies torch only when it needs an RNG snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Protocol, Sequence


PINNED_RECAL3R_COMMIT = "466c7cdf3acd2f589f1d82e5f6391966f19db9ff"
RECAL3R_ROOT = Path("/data/wangzheng/Project2/baselines/ReCal3R")
RECAL3R_LIGHTER_SOURCE = RECAL3R_ROOT / "src/dust3r/model.py"
RECAL3R_LIGHTER_SOURCE_SHA256 = "32785a6f29fded66aa142207b29a33d7522f0c39aa068fe2175a956ec8dbc3c1"
RECAL3R_LIGHTER_LINE_RANGE = (1661, 1833)
RECAL3R_LIGHTER_LINE_SHA256 = "03d3c534f5f59f6eb1fa1852ff1cbeb8e874e97b027cd863e48fb6347b100023"

# These are the model attributes mutated or cleared in the pinned ReCal3R
# recurrent path and its called calibration helpers.  They must be restored as
# a unit; holding just ``state_feat`` or setting a view's ``update=False`` is
# not a transaction.
MODEL_MUTABLE_ATTRIBUTES = (
    "update_pressure",
    "recal3r_state0",
    "_recal3r_sequence_age",
    "_u_calibration_trace",
    "_u_calibration_pending_u",
    "_u_calibration_pending_h",
    "_u_calibration_last_state",
    "_u_calibration_final_state",
    "_u_calibration_oracle_window",
    "_u_calibration_queue",
)
MODEL_CONFIG_ATTRIBUTES = ("model_update_type",)
STATE_FIELDS = ("state_feat", "state_pos", "init_state_feat", "mem", "init_mem")


class StateTransactionError(RuntimeError):
    """Raised when an external state transaction is malformed or unsafe."""


class _TorchLike(Protocol):
    def get_rng_state(self) -> Any: ...


def _clone_value(value: Any) -> Any:
    """Deep-copy mutable state while giving tensors an alias-free clone."""

    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    detach = getattr(value, "detach", None)
    clone = getattr(value, "clone", None)
    if callable(detach) and callable(clone):
        return detach().clone()
    if callable(clone):
        return clone()
    if isinstance(value, Mapping):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    return copy.deepcopy(value)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _value_digest(value: Any) -> dict[str, Any]:
    """Return a content digest without serialising raw state into a report."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return {"kind": type(value).__name__, "sha256": _sha256_bytes(repr(value).encode())}
    detach = getattr(value, "detach", None)
    if callable(detach):
        tensor = detach()
        cpu = getattr(tensor, "cpu", None)
        if callable(cpu):
            tensor = cpu()
        contiguous = getattr(tensor, "contiguous", None)
        if callable(contiguous):
            tensor = contiguous()
        shape = tuple(int(item) for item in getattr(tensor, "shape", ()))
        raw: bytes
        try:
            as_uint8 = tensor.view(getattr(type(tensor), "uint8", None))
            raw = as_uint8.numpy().tobytes()
        except (AttributeError, RuntimeError, TypeError):
            try:
                raw = tensor.numpy().tobytes()
            except (AttributeError, RuntimeError, TypeError):
                raw = repr(tensor).encode()
        return {
            "kind": "tensor",
            "dtype": str(getattr(tensor, "dtype", "unknown")),
            "shape": shape,
            "sha256": _sha256_bytes(raw),
        }
    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "items": {str(key): _value_digest(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))},
        }
    if isinstance(value, (tuple, list)):
        return {"kind": type(value).__name__, "items": [_value_digest(item) for item in value]}
    return {"kind": type(value).__name__, "sha256": _sha256_bytes(repr(value).encode())}


@dataclass(frozen=True)
class AttributeSnapshot:
    """The prior value of one possibly-absent mutable object attribute."""

    exists: bool
    value: Any = None


@dataclass(frozen=True)
class StateClosure:
    """All five local tensors needed to continue the pinned recurrent path."""

    state_feat: Any
    state_pos: Any
    init_state_feat: Any
    mem: Any
    init_mem: Any

    @classmethod
    def capture(
        cls,
        state_feat: Any,
        state_pos: Any,
        init_state_feat: Any,
        mem: Any,
        init_mem: Any,
    ) -> "StateClosure":
        return cls(*(_clone_value(value) for value in (state_feat, state_pos, init_state_feat, mem, init_mem)))

    def restored(self) -> tuple[Any, Any, Any, Any, Any]:
        return tuple(_clone_value(getattr(self, field)) for field in STATE_FIELDS)

    def digest(self) -> dict[str, Any]:
        return {field: _value_digest(getattr(self, field)) for field in STATE_FIELDS}


@dataclass(frozen=True)
class ModelStateSnapshot:
    """Known model-side recurrent/calibration state and optional RNG state."""

    attributes: Mapping[str, AttributeSnapshot]
    config_attributes: Mapping[str, AttributeSnapshot]
    cpu_rng_state: Any | None
    cuda_rng_states: Any | None

    def restore(self, model: Any, *, torch: _TorchLike | None = None) -> None:
        for name, snapshot in self.attributes.items():
            _restore_attribute(model, name, snapshot)
        config = getattr(model, "config", None)
        if config is None:
            raise StateTransactionError("model has no config while restoring transaction")
        for name, snapshot in self.config_attributes.items():
            _restore_attribute(config, name, snapshot)
        if torch is not None and self.cpu_rng_state is not None:
            torch.set_rng_state(_clone_value(self.cpu_rng_state))
            cuda = getattr(torch, "cuda", None)
            if self.cuda_rng_states is not None and cuda is not None:
                cuda.set_rng_state_all([_clone_value(value) for value in self.cuda_rng_states])

    def digest(self) -> dict[str, Any]:
        return {
            "attributes": {
                name: {"exists": snapshot.exists, "value": _value_digest(snapshot.value) if snapshot.exists else None}
                for name, snapshot in self.attributes.items()
            },
            "config_attributes": {
                name: {"exists": snapshot.exists, "value": _value_digest(snapshot.value) if snapshot.exists else None}
                for name, snapshot in self.config_attributes.items()
            },
            "rng": {
                "cpu": _value_digest(self.cpu_rng_state) if self.cpu_rng_state is not None else None,
                "cuda": _value_digest(self.cuda_rng_states) if self.cuda_rng_states is not None else None,
            },
        }


def _restore_attribute(target: Any, name: str, snapshot: AttributeSnapshot) -> None:
    if snapshot.exists:
        setattr(target, name, _clone_value(snapshot.value))
    elif hasattr(target, name):
        delattr(target, name)


def capture_model_state(model: Any, *, torch: _TorchLike | None = None) -> ModelStateSnapshot:
    """Capture all model attributes that the recurrent transaction may mutate."""

    config = getattr(model, "config", None)
    if config is None:
        raise StateTransactionError("model has no config")
    attributes = {
        name: AttributeSnapshot(hasattr(model, name), _clone_value(getattr(model, name)) if hasattr(model, name) else None)
        for name in MODEL_MUTABLE_ATTRIBUTES
    }
    config_attributes = {
        name: AttributeSnapshot(hasattr(config, name), _clone_value(getattr(config, name)) if hasattr(config, name) else None)
        for name in MODEL_CONFIG_ATTRIBUTES
    }
    cpu_rng_state = None
    cuda_rng_states = None
    if torch is not None:
        cpu_rng_state = _clone_value(torch.get_rng_state())
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and bool(cuda.is_available()):
            cuda_rng_states = tuple(_clone_value(value) for value in cuda.get_rng_state_all())
    return ModelStateSnapshot(attributes, config_attributes, cpu_rng_state, cuda_rng_states)


@dataclass(frozen=True)
class StateTransaction:
    """One candidate update, including pre/post state and source-bound evidence."""

    frame_id: int
    pre_state: StateClosure
    post_state: StateClosure
    pre_model: ModelStateSnapshot
    post_model: ModelStateSnapshot
    pre_reset_mask: Any
    post_reset_mask: Any

    def digest(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "pre_state": self.pre_state.digest(),
            "post_state": self.post_state.digest(),
            "pre_model": self.pre_model.digest(),
            "post_model": self.post_model.digest(),
            "pre_reset_mask": _value_digest(self.pre_reset_mask),
            "post_reset_mask": _value_digest(self.post_reset_mask),
        }


@dataclass(frozen=True)
class PolicyDecision:
    """Causal state-policy decision for one post-frame candidate."""

    frame_id: int
    action: str
    prior_alarm: bool
    reason: str


class CausalHoldReplayController:
    """Bounded hold controller that reads only the immediately prior alarm.

    At frame ``t`` the controller may see only ``alarm[t-1]``.  It holds the
    newly proposed transaction when that prior alarm is true.  On a subsequent
    non-alarm it restores the newest held post-state before processing the
    current frame; this is an explicit replayed commit.  At most ``max_hold``
    candidate updates can remain pending.
    """

    def __init__(self, alarms: Sequence[bool], *, max_hold: int = 3) -> None:
        if max_hold < 1:
            raise StateTransactionError("max_hold must be at least one")
        self._alarms = tuple(bool(value) for value in alarms)
        self.max_hold = max_hold
        self._pending: list[StateTransaction] = []

    def prior_alarm(self, frame_id: int) -> bool:
        if frame_id < 0 or frame_id >= len(self._alarms):
            raise StateTransactionError(f"frame {frame_id} is outside alarm sequence")
        return frame_id > 0 and self._alarms[frame_id - 1]

    def take_replay(self, frame_id: int) -> StateTransaction | None:
        """Return a prior held transaction only after a subsequent safe score."""

        if not self._pending or self.prior_alarm(frame_id):
            return None
        transaction = self._pending[-1]
        self._pending.clear()
        return transaction

    def decide_candidate(self, transaction: StateTransaction) -> PolicyDecision:
        frame_id = transaction.frame_id
        prior_alarm = self.prior_alarm(frame_id)
        if frame_id == 0:
            return PolicyDecision(frame_id, "commit", False, "initial_state_must_commit")
        if not prior_alarm:
            return PolicyDecision(frame_id, "commit", False, "prior_score_clear")
        if len(self._pending) < self.max_hold:
            self._pending.append(transaction)
            return PolicyDecision(frame_id, "hold", True, "prior_detector_alarm")
        self._pending.clear()
        return PolicyDecision(frame_id, "commit", True, "hold_bound_reached")

    def drain(self) -> list[StateTransaction]:
        """Discard held transactions at end of stream instead of hidden commits."""

        result = list(self._pending)
        self._pending.clear()
        return result


def _git_output(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def verify_pinned_lighter_source(
    *,
    repository: Path = RECAL3R_ROOT,
    source_path: Path = RECAL3R_LIGHTER_SOURCE,
) -> dict[str, Any]:
    """Require the exact pinned source and cloned method span before execution."""

    try:
        commit = _git_output(repository, "rev-parse", "HEAD")
        tracked_changes = _git_output(repository, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError) as error:
        raise StateTransactionError(f"cannot inspect ReCal3R provenance: {error}") from error
    if commit != PINNED_RECAL3R_COMMIT or tracked_changes:
        raise StateTransactionError("ReCal3R must be pinned and have a clean tracked worktree")
    try:
        source_bytes = source_path.read_bytes()
    except OSError as error:
        raise StateTransactionError(f"cannot read pinned lighter source: {error}") from error
    source_sha = _sha256_bytes(source_bytes)
    if source_sha != RECAL3R_LIGHTER_SOURCE_SHA256:
        raise StateTransactionError("pinned ReCal3R model source hash differs")
    start, end = RECAL3R_LIGHTER_LINE_RANGE
    lines = source_bytes.splitlines(keepends=True)
    method_sha = _sha256_bytes(b"".join(lines[start - 1 : end]))
    if method_sha != RECAL3R_LIGHTER_LINE_SHA256:
        raise StateTransactionError("pinned ReCal3R lighter method span differs")
    return {
        "baseline_root": str(repository.resolve(strict=True)),
        "baseline_commit": commit,
        "baseline_tracked_worktree_clean": True,
        "source_path": str(source_path.resolve(strict=True)),
        "source_sha256": source_sha,
        "line_range": list(RECAL3R_LIGHTER_LINE_RANGE),
        "line_sha256": method_sha,
    }


def canonical_digest(payload: Mapping[str, Any]) -> str:
    """Hash a state-policy report with deterministic JSON encoding."""

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _sha256_bytes(encoded)


__all__ = [
    "CausalHoldReplayController",
    "MODEL_CONFIG_ATTRIBUTES",
    "MODEL_MUTABLE_ATTRIBUTES",
    "ModelStateSnapshot",
    "PINNED_RECAL3R_COMMIT",
    "PolicyDecision",
    "RECAL3R_LIGHTER_LINE_RANGE",
    "RECAL3R_LIGHTER_LINE_SHA256",
    "RECAL3R_LIGHTER_SOURCE",
    "RECAL3R_LIGHTER_SOURCE_SHA256",
    "StateClosure",
    "StateTransaction",
    "StateTransactionError",
    "canonical_digest",
    "capture_model_state",
    "verify_pinned_lighter_source",
]
