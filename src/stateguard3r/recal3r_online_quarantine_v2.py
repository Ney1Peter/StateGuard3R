"""Fast structural rollback primitives for current-frame online quarantine.

The old v1 wrapper deep-copied both pre- and post-frame recurrent closures on
every frame.  That was correct but its normal-path runtime was already about
1.65x baseline.  The pinned lighter source instead rebinds local recurrent
tensors and its model-side tensors; the only in-place growth is in registered
calibration trace lists.  This module records references plus those list
lengths before a candidate, and restores them only when its *current* online
score asks for quarantine.  It has no dependency on ReCal3R at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Protocol

from .recal3r_state_v3 import MODEL_CONFIG_ATTRIBUTES, MODEL_MUTABLE_ATTRIBUTES, STATE_FIELDS, _value_digest


class OnlineQuarantineStateError(RuntimeError):
    """Raised when a structural snapshot cannot safely restore a candidate."""


class _TorchLike(Protocol):
    def get_rng_state(self) -> Any: ...


@dataclass(frozen=True)
class _AttributeReference:
    exists: bool
    value: Any = None


def _clone_rng(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    clone = getattr(value, "clone", None)
    if callable(detach) and callable(clone):
        return detach().clone()
    if callable(clone):
        return clone()
    return value


def _restore_attribute(target: Any, name: str, snapshot: _AttributeReference) -> None:
    if snapshot.exists:
        setattr(target, name, snapshot.value)
    elif hasattr(target, name):
        delattr(target, name)


def _trace_lengths(value: Any, *, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise OnlineQuarantineStateError(f"{label} must be a mapping")
    lengths: dict[str, int] = {}
    for name, entries in value.items():
        if not isinstance(name, str) or not isinstance(entries, list):
            raise OnlineQuarantineStateError(f"{label} must contain string/list entries only")
        lengths[name] = len(entries)
    return lengths


def _truncate_trace(value: Any, lengths: Mapping[str, int], *, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(lengths):
        raise OnlineQuarantineStateError(f"{label} schema changed during candidate frame")
    for name, expected in lengths.items():
        entries = value[name]
        if not isinstance(entries, list) or len(entries) < expected:
            raise OnlineQuarantineStateError(f"{label}.{name} cannot be structurally restored")
        del entries[expected:]


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class StructuralPreState:
    """Reference-only pre-frame closure plus structural model/RNG restoration."""

    local_state: Mapping[str, Any]
    reset_mask: Any
    model_attributes: Mapping[str, _AttributeReference]
    config_attributes: Mapping[str, _AttributeReference]
    trace_lengths: Mapping[str, int] | None
    queue_length: int | None
    cpu_rng_state: Any | None
    cuda_rng_states: tuple[Any, ...] | None

    @classmethod
    def capture(
        cls,
        state_feat: Any,
        state_pos: Any,
        init_state_feat: Any,
        mem: Any,
        init_mem: Any,
        reset_mask: Any,
        model: Any,
        *,
        torch: _TorchLike | None,
    ) -> "StructuralPreState":
        config = getattr(model, "config", None)
        if config is None:
            raise OnlineQuarantineStateError("model has no config")
        attributes = {
            name: _AttributeReference(hasattr(model, name), getattr(model, name) if hasattr(model, name) else None)
            for name in MODEL_MUTABLE_ATTRIBUTES
        }
        config_attributes = {
            name: _AttributeReference(hasattr(config, name), getattr(config, name) if hasattr(config, name) else None)
            for name in MODEL_CONFIG_ATTRIBUTES
        }
        trace_snapshot = attributes.get("_u_calibration_trace")
        trace_lengths = _trace_lengths(trace_snapshot.value, label="_u_calibration_trace") if trace_snapshot and trace_snapshot.exists else None
        queue_snapshot = attributes.get("_u_calibration_queue")
        queue_length: int | None = None
        if queue_snapshot and queue_snapshot.exists:
            if not isinstance(queue_snapshot.value, list):
                raise OnlineQuarantineStateError("_u_calibration_queue must be a list")
            queue_length = len(queue_snapshot.value)
        cpu_rng_state = None
        cuda_rng_states = None
        if torch is not None:
            cpu_rng_state = _clone_rng(torch.get_rng_state())
            cuda = getattr(torch, "cuda", None)
            if cuda is not None and bool(cuda.is_available()):
                cuda_rng_states = tuple(_clone_rng(item) for item in cuda.get_rng_state_all())
        return cls(
            local_state=dict(zip(STATE_FIELDS, (state_feat, state_pos, init_state_feat, mem, init_mem), strict=True)),
            reset_mask=reset_mask,
            model_attributes=attributes,
            config_attributes=config_attributes,
            trace_lengths=trace_lengths,
            queue_length=queue_length,
            cpu_rng_state=cpu_rng_state,
            cuda_rng_states=cuda_rng_states,
        )

    def restore(self, model: Any, *, torch: _TorchLike | None) -> tuple[Any, Any, Any, Any, Any, Any]:
        config = getattr(model, "config", None)
        if config is None:
            raise OnlineQuarantineStateError("model has no config while restoring")
        # Reinstall reference-valued attributes first.  The trace and queue
        # objects are intentionally the exact pre-frame containers, then their
        # append-only tails are removed below.
        for name, snapshot in self.model_attributes.items():
            _restore_attribute(model, name, snapshot)
        for name, snapshot in self.config_attributes.items():
            _restore_attribute(config, name, snapshot)
        if self.trace_lengths is not None:
            _truncate_trace(getattr(model, "_u_calibration_trace", None), self.trace_lengths, label="_u_calibration_trace")
        if self.queue_length is not None:
            queue = getattr(model, "_u_calibration_queue", None)
            if not isinstance(queue, list) or len(queue) < self.queue_length:
                raise OnlineQuarantineStateError("_u_calibration_queue cannot be structurally restored")
            del queue[self.queue_length:]
        if torch is not None and self.cpu_rng_state is not None:
            torch.set_rng_state(_clone_rng(self.cpu_rng_state))
            cuda = getattr(torch, "cuda", None)
            if self.cuda_rng_states is not None and cuda is not None:
                cuda.set_rng_state_all([_clone_rng(item) for item in self.cuda_rng_states])
        return (*tuple(self.local_state[field] for field in STATE_FIELDS), self.reset_mask)

    def digest(self) -> str:
        """Compute a full content digest only for an actual rollback audit."""

        return _digest(
            {
                "local_state": {name: _value_digest(value) for name, value in self.local_state.items()},
                "reset_mask": _value_digest(self.reset_mask),
                "model_attributes": {
                    name: {"exists": snapshot.exists, "value": _value_digest(snapshot.value) if snapshot.exists else None}
                    for name, snapshot in self.model_attributes.items()
                },
                "config_attributes": {
                    name: {"exists": snapshot.exists, "value": _value_digest(snapshot.value) if snapshot.exists else None}
                    for name, snapshot in self.config_attributes.items()
                },
                "trace_lengths": dict(self.trace_lengths) if self.trace_lengths is not None else None,
                "queue_length": self.queue_length,
            }
        )


def current_state_digest(
    state_feat: Any,
    state_pos: Any,
    init_state_feat: Any,
    mem: Any,
    init_mem: Any,
    reset_mask: Any,
    model: Any,
) -> str:
    """Content digest of the current closure, used only after a rollback action."""

    config = getattr(model, "config", None)
    if config is None:
        raise OnlineQuarantineStateError("model has no config while digesting candidate")
    attributes = {
        name: {"exists": hasattr(model, name), "value": _value_digest(getattr(model, name)) if hasattr(model, name) else None}
        for name in MODEL_MUTABLE_ATTRIBUTES
    }
    config_attributes = {
        name: {"exists": hasattr(config, name), "value": _value_digest(getattr(config, name)) if hasattr(config, name) else None}
        for name in MODEL_CONFIG_ATTRIBUTES
    }
    return _digest(
        {
            "local_state": {
                name: _value_digest(value)
                for name, value in zip(STATE_FIELDS, (state_feat, state_pos, init_state_feat, mem, init_mem), strict=True)
            },
            "reset_mask": _value_digest(reset_mask),
            "model_attributes": attributes,
            "config_attributes": config_attributes,
        }
    )


class QuarantineWatchdog:
    """Fixed liveness boundary; it never falls back to a hidden commit."""

    def __init__(self, limit: int = 8) -> None:
        if type(limit) is not int or limit != 8:
            raise OnlineQuarantineStateError("online quarantine watchdog is fixed at eight consecutive rollbacks")
        self.limit = limit
        self._consecutive = 0

    @property
    def consecutive(self) -> int:
        return self._consecutive

    def record(self, *, quarantine: bool) -> int:
        if quarantine:
            self._consecutive += 1
            if self._consecutive > self.limit:
                raise OnlineQuarantineStateError("online quarantine liveness watchdog exceeded eight consecutive rollbacks")
        else:
            self._consecutive = 0
        return self._consecutive


__all__ = [
    "current_state_digest",
    "OnlineQuarantineStateError",
    "QuarantineWatchdog",
    "StructuralPreState",
]
