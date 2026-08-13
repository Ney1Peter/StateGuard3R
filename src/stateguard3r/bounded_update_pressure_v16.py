"""CUDA-only fixed bounded native update-pressure write for v16."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping


INCREMENT = 0.25
CAP = 1.0
PRESSURE_SHAPE = (1, 768, 1)


class BoundedUpdatePressureV16Error(RuntimeError):
    """The narrow v16 pressure-write contract was violated."""


def gpu_fingerprint_v16(value: Any, *, torch: Any, label: str) -> str:
    """Return a scalar-only CUDA witness without serializing tensor values."""
    if not bool(torch.is_tensor(value)) or getattr(value.device, "type", None) != "cuda" or not bool(torch.is_floating_point(value)) or not bool(torch.isfinite(value).all()):
        raise BoundedUpdatePressureV16Error(f"{label} must be a finite floating CUDA tensor")
    flat = value.detach().contiguous().reshape(-1).to(dtype=torch.float64)
    positions = torch.arange(1, flat.numel() + 1, dtype=torch.float64, device=flat.device)
    moments = (flat.sum(), flat.abs().sum(), (flat * positions).sum(), flat.square().sum(), flat.min(), flat.max())
    encoded = "|".join((str(tuple(value.shape)), str(value.dtype), *(float(item.item()).hex() for item in moments)))
    return "v16-gpu-fingerprint:" + hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _validate_pressure(value: Any, *, torch: Any, expected_shape: tuple[int, int, int]) -> None:
    if not bool(torch.is_tensor(value)) or getattr(value.device, "type", None) != "cuda" or not bool(torch.is_floating_point(value)) or tuple(int(item) for item in value.shape) != expected_shape:
        raise BoundedUpdatePressureV16Error("native update pressure has the wrong CUDA tensor contract")
    if not bool(torch.isfinite(value).all()) or bool((value < 0.0).any()) or bool((value > CAP).any()):
        raise BoundedUpdatePressureV16Error("native update pressure is nonfinite or outside [0, 1]")


def inject_bounded_pressure_v16(native_pressure: Any | None, *, alarm: bool, device: Any, dtype: Any, expected_shape: tuple[int, int, int] = PRESSURE_SHAPE, torch: Any) -> tuple[Any | None, Mapping[str, Any] | None]:
    """Apply the preregistered add/cap after a plain current-frame alarm."""
    if type(alarm) is not bool:
        raise BoundedUpdatePressureV16Error("alarm must be a plain boolean")
    if not alarm:
        return native_pressure, None
    if tuple(expected_shape) != PRESSURE_SHAPE or getattr(device, "type", None) != "cuda":
        raise BoundedUpdatePressureV16Error("v16 pressure shape/device pin differs")
    if native_pressure is None:
        committed = torch.full(PRESSURE_SHAPE, INCREMENT, device=device, dtype=dtype)
        _validate_pressure(committed, torch=torch, expected_shape=PRESSURE_SHAPE)
        return committed, {
            "operator": "bounded_native_update_pressure_v16",
            "action": "bootstrap_after_native_reset",
            "increment": INCREMENT,
            "cap": CAP,
            "shape": list(PRESSURE_SHAPE),
            "dtype": str(committed.dtype),
            "device": str(committed.device),
            "native_pressure_absent_after_reset": True,
            "pre_gpu_fingerprint": None,
            "committed_gpu_fingerprint": gpu_fingerprint_v16(committed, torch=torch, label="committed pressure"),
        }
    _validate_pressure(native_pressure, torch=torch, expected_shape=PRESSURE_SHAPE)
    committed = torch.clamp(native_pressure + INCREMENT, min=0.0, max=CAP)
    _validate_pressure(committed, torch=torch, expected_shape=PRESSURE_SHAPE)
    return committed, {
        "operator": "bounded_native_update_pressure_v16",
        "action": "add_and_cap_native_pressure",
        "increment": INCREMENT,
        "cap": CAP,
        "shape": list(PRESSURE_SHAPE),
        "dtype": str(committed.dtype),
        "device": str(committed.device),
        "native_pressure_absent_after_reset": False,
        "pre_gpu_fingerprint": gpu_fingerprint_v16(native_pressure, torch=torch, label="native pressure"),
        "committed_gpu_fingerprint": gpu_fingerprint_v16(committed, torch=torch, label="committed pressure"),
    }


__all__ = ["CAP", "INCREMENT", "PRESSURE_SHAPE", "BoundedUpdatePressureV16Error", "gpu_fingerprint_v16", "inject_bounded_pressure_v16"]
