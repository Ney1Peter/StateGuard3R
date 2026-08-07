"""Cheap alias/in-place witness for v3 reference-closure restoration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .recal3r_online_quarantine_v2 import OnlineQuarantineStateError, StructuralPreState
from .recal3r_state_v3 import MODEL_CONFIG_ATTRIBUTES, MODEL_MUTABLE_ATTRIBUTES, STATE_FIELDS


class StructuralWitnessError(RuntimeError):
    """Raised when the fast reference-only snapshot cannot be trusted."""


@dataclass(frozen=True)
class _Identity:
    object_id: int
    storage_ptr: int | None
    version: int | None


def _identity(value: Any) -> _Identity:
    storage_ptr: int | None = None
    version: int | None = None
    data_ptr = getattr(value, "data_ptr", None)
    if callable(data_ptr):
        storage_ptr = int(data_ptr())
    raw_version = getattr(value, "_version", None)
    if isinstance(raw_version, int):
        version = raw_version
    return _Identity(id(value), storage_ptr, version)


@dataclass(frozen=True)
class StructuralIdentityWitness:
    """Identity/storage/version facts captured alongside a ``StructuralPreState``."""

    local: Mapping[str, _Identity]
    model: Mapping[str, _Identity | None]
    config: Mapping[str, _Identity | None]
    trace_lengths: Mapping[str, int] | None
    queue_length: int | None

    @classmethod
    def capture(cls, snapshot: StructuralPreState, model: Any) -> "StructuralIdentityWitness":
        config = getattr(model, "config", None)
        if config is None:
            raise StructuralWitnessError("model has no config")
        return cls(
            local={name: _identity(value) for name, value in snapshot.local_state.items()},
            model={name: _identity(getattr(model, name)) if hasattr(model, name) else None for name in MODEL_MUTABLE_ATTRIBUTES},
            config={name: _identity(getattr(config, name)) if hasattr(config, name) else None for name in MODEL_CONFIG_ATTRIBUTES},
            trace_lengths=dict(snapshot.trace_lengths) if snapshot.trace_lengths is not None else None,
            queue_length=snapshot.queue_length,
        )

    def reject_in_place_mutation(self, snapshot: StructuralPreState) -> None:
        for name, before in self.local.items():
            after = _identity(snapshot.local_state[name])
            if before.storage_ptr is not None and after.storage_ptr != before.storage_ptr:
                raise StructuralWitnessError(f"pre-state local {name} storage changed")
            if before.version is not None and after.version != before.version:
                raise StructuralWitnessError(f"pre-state local {name} was mutated in place")
        for name, before in self.model.items():
            reference = snapshot.model_attributes[name]
            if before is None or not reference.exists:
                continue
            after = _identity(reference.value)
            if before.storage_ptr is not None and after.storage_ptr != before.storage_ptr:
                raise StructuralWitnessError(f"pre-state model attribute {name} storage changed")
            if before.version is not None and after.version != before.version:
                raise StructuralWitnessError(f"pre-state model attribute {name} was mutated in place")

    def verify_restored(self, restored: tuple[Any, Any, Any, Any, Any, Any], model: Any) -> None:
        config = getattr(model, "config", None)
        if config is None:
            raise StructuralWitnessError("model has no config after restore")
        for name, value in zip(STATE_FIELDS, restored[:-1], strict=True):
            if _identity(value) != self.local[name]:
                raise StructuralWitnessError(f"restored local {name} does not rebind pre-state reference")
        for name, before in self.model.items():
            exists = hasattr(model, name)
            if (before is None) != (not exists):
                raise StructuralWitnessError(f"restored model attribute presence differs for {name}")
            if before is not None and _identity(getattr(model, name)) != before:
                raise StructuralWitnessError(f"restored model attribute identity differs for {name}")
        for name, before in self.config.items():
            exists = hasattr(config, name)
            if (before is None) != (not exists):
                raise StructuralWitnessError(f"restored config attribute presence differs for {name}")
            if before is not None and _identity(getattr(config, name)) != before:
                raise StructuralWitnessError(f"restored config attribute identity differs for {name}")
        if self.trace_lengths is not None:
            trace = getattr(model, "_u_calibration_trace", None)
            if not isinstance(trace, Mapping) or {name: len(value) for name, value in trace.items()} != dict(self.trace_lengths):
                raise StructuralWitnessError("restored trace lengths differ")
        if self.queue_length is not None:
            queue = getattr(model, "_u_calibration_queue", None)
            if not isinstance(queue, list) or len(queue) != self.queue_length:
                raise StructuralWitnessError("restored queue length differs")

    def timeline_evidence(self) -> dict[str, object]:
        return {
            "contract": "reference_identity_storage_version_and_trace_length",
            "local_fields": list(STATE_FIELDS),
            "model_mutable_attributes": list(MODEL_MUTABLE_ATTRIBUTES),
            "config_attributes": list(MODEL_CONFIG_ATTRIBUTES),
            "trace_lengths": dict(self.trace_lengths) if self.trace_lengths is not None else None,
            "queue_length": self.queue_length,
        }


__all__ = ["StructuralIdentityWitness", "StructuralWitnessError"]
