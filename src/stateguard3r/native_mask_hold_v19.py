"""Scalar one-shot policy and narrow official native-mask wrapper for v19."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import ast
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping


MODEL = Path("/data/wangzheng/Project2/baselines/ReCal3R/src/dust3r/model.py")
MODEL_SHA256 = "32785a6f29fded66aa142207b29a33d7522f0c39aa068fe2175a956ec8dbc3c1"


class NativeMaskHoldV19Error(RuntimeError):
    """A v19 scalar hold or official mask wrapper is out of contract."""


@dataclass(frozen=True, slots=True)
class NativeMaskHoldSelectionV19:
    armed_before: bool
    armed_after: bool
    consumed: bool
    reset_cancelled: bool
    return_zero_mask: bool


def _bool(value: bool, *, label: str) -> bool:
    if type(value) is not bool:
        raise NativeMaskHoldV19Error(f"{label} must be a plain boolean")
    return value


def arm_one_native_state_mask_v19(armed: bool, *, alarm: bool, has_next_view: bool) -> bool:
    armed = _bool(armed, label="armed")
    alarm = _bool(alarm, label="alarm")
    has_next_view = _bool(has_next_view, label="has_next_view")
    if armed:
        raise NativeMaskHoldV19Error("prior v19 arm was neither consumed nor reset-cancelled")
    return bool(alarm and has_next_view)


def select_native_mask_hold_v19(armed: bool, *, reset: bool) -> NativeMaskHoldSelectionV19:
    armed = _bool(armed, label="armed")
    reset = _bool(reset, label="reset")
    if armed and reset:
        return NativeMaskHoldSelectionV19(True, False, False, True, False)
    if armed:
        return NativeMaskHoldSelectionV19(True, False, True, False, True)
    return NativeMaskHoldSelectionV19(False, False, False, False, False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_native_mask_wrapper_v19() -> Mapping[str, Any]:
    """Bind the one native method v19 may wrap and its official use sites."""

    if _sha256(MODEL) != MODEL_SHA256:
        raise NativeMaskHoldV19Error("pinned ReCal3R source differs")
    tree = ast.parse(MODEL.read_text(encoding="utf-8"), filename=str(MODEL))
    functions = {
        item.name: item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name in {"forward_recurrent_lighter", "_compute_recal3r_update_mask"}
    }
    if set(functions) != {"forward_recurrent_lighter", "_compute_recal3r_update_mask"}:
        raise NativeMaskHoldV19Error("pinned native mask interface is absent")
    lighter = functions["forward_recurrent_lighter"]
    calls = [
        node for node in ast.walk(lighter)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
        and node.func.attr == "_compute_recal3r_update_mask"
    ]
    if len(calls) != 1 or [argument.id if isinstance(argument, ast.Name) else None for argument in calls[0].args] != ["update_mask", "raw_cross_attn_state", "dec"]:
        raise NativeMaskHoldV19Error("official native mask call differs")
    if {keyword.arg: keyword.value.id if isinstance(keyword.value, ast.Name) else None for keyword in calls[0].keywords} != {"prev_state_feat": "state_feat"}:
        raise NativeMaskHoldV19Error("official native mask previous-state argument differs")
    source = MODEL.read_text(encoding="utf-8")
    start = source.index("def forward_recurrent_lighter")
    positions = (
        source.index("update_mask1 = self._compute_recal3r_update_mask(", start),
        source.index("update_mask2 = update_mask", start),
        source.index("state_feat = new_state_feat * update_mask1", start),
        source.index("mem = new_mem * update_mask2", start),
        source.index("self._maybe_record_u_calibration_step(i, prev_state_feat, state_feat)", start),
    )
    if positions != tuple(sorted(positions)):
        raise NativeMaskHoldV19Error("official mask/state/memory order differs")
    return {
        "schema_version": "stateguard3r.native-mask-wrapper-audit-v19.v1",
        "model_path": str(MODEL),
        "model_sha256": MODEL_SHA256,
        "official_inference": "dust3r.inference.inference_recurrent_lighter",
        "native_mask_method": "model._compute_recal3r_update_mask",
        "native_mask_call_args": ["update_mask", "raw_cross_attn_state", "dec"],
        "native_mask_call_keywords": {"prev_state_feat": "state_feat"},
        "state_mask_only": True,
        "memory_mask_remains_native": True,
        "native_mask_before_state_memory_commit": True,
    }


def _zero_like(mask: Any) -> Any:
    try:
        return mask.clone().zero_()
    except AttributeError as error:
        raise NativeMaskHoldV19Error("native mask is not a cloneable tensor") from error


@contextmanager
def temporary_native_mask_hold_v19(
    model: Any,
    *,
    selection: NativeMaskHoldSelectionV19,
    on_native_mask: Callable[[Any], None] | None = None,
) -> Iterator[None]:
    """Wrap exactly one instance method and only replace its returned mask.

    The native method always executes first.  A consumed arm then replaces only
    that return tensor by a same-shape/dtype/device zero tensor.  The official
    loop retains ownership of all following state/memory computations.
    """

    if not isinstance(selection, NativeMaskHoldSelectionV19):
        raise NativeMaskHoldV19Error("v19 selection type differs")
    original = model._compute_recal3r_update_mask
    calls = 0

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        native = original(*args, **kwargs)
        if on_native_mask is not None:
            on_native_mask(native)
        return _zero_like(native) if selection.return_zero_mask else native

    model._compute_recal3r_update_mask = wrapped
    try:
        yield
        if calls != 1:
            raise NativeMaskHoldV19Error("v19 wrapper must observe exactly one native mask call")
    finally:
        model._compute_recal3r_update_mask = original
        if model._compute_recal3r_update_mask is not original:
            raise NativeMaskHoldV19Error("v19 native mask wrapper was not restored")


def require_cuda_mask_identity_v19(native: Any, held: Any, *, torch: Any) -> Mapping[str, Any]:
    """Check that a held final mask is a CUDA zeros-like native mask."""

    for label, value in (("native", native), ("held", held)):
        if not bool(torch.is_tensor(value)) or not bool(torch.is_floating_point(value)):
            raise NativeMaskHoldV19Error(f"{label} mask is not floating")
        if getattr(value.device, "type", None) != "cuda" or not bool(torch.isfinite(value).all().item()):
            raise NativeMaskHoldV19Error(f"{label} mask lacks finite CUDA residency")
    if tuple(native.shape) != tuple(held.shape) or native.dtype != held.dtype or native.device != held.device:
        raise NativeMaskHoldV19Error("held mask interface differs from native")
    if not bool((held == 0).all().item()):
        raise NativeMaskHoldV19Error("held final mask is not exactly zero")
    return {
        "native_mask_shape": list(native.shape),
        "native_mask_dtype": str(native.dtype),
        "native_mask_device": str(native.device),
        "held_final_mask_exact_zero": True,
    }


__all__ = [
    "MODEL", "MODEL_SHA256", "NativeMaskHoldSelectionV19", "NativeMaskHoldV19Error",
    "arm_one_native_state_mask_v19", "audit_native_mask_wrapper_v19",
    "require_cuda_mask_identity_v19", "select_native_mask_hold_v19",
    "temporary_native_mask_hold_v19",
]
