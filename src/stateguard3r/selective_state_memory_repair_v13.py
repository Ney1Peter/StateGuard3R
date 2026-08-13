"""GPU-only bounded repair of ReCal3R's persistent state and pose memory.

The v13 repair operator is deliberately small: after a raw current prediction
has already been produced and assessed by the frozen detector, it restores the
largest one-eighth row updates in ``state_feat`` and ``mem`` from their own
pre-frame tensors.  It has no camera-pose, image, pointmap, decoder, detector,
history, GT, or future-frame input and never serializes tensor values.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping


STATE_SHAPE = (1, 768, 768)
MEMORY_SHAPE = (1, 256, 1536)
REPAIR_DENOMINATOR = 8


class SelectiveStateMemoryRepairError(RuntimeError):
    """A selective persistent-state repair cannot be made safely."""


@dataclass(frozen=True)
class StateMemoryPreState:
    """Reference-only pre-update state, with an in-place mutation witness."""

    state_feat: Any
    mem: Any
    state_identity: tuple[int, int | None, int | None]
    mem_identity: tuple[int, int | None, int | None]


@dataclass(frozen=True)
class SelectiveStateMemoryRepairResult:
    """The repaired tensors and non-sensitive causal evidence."""

    state_feat: Any
    mem: Any
    evidence: Mapping[str, Any]


def _identity(value: Any) -> tuple[int, int | None, int | None]:
    pointer = getattr(value, "data_ptr", None)
    return (
        id(value),
        int(pointer()) if callable(pointer) else None,
        getattr(value, "_version", None) if isinstance(getattr(value, "_version", None), int) else None,
    )


def _require_tensor(value: Any, *, label: str, torch: Any) -> tuple[int, int, int]:
    if not bool(torch.is_tensor(value)) or not bool(torch.is_floating_point(value)):
        raise SelectiveStateMemoryRepairError(f"{label} must be a floating tensor")
    shape = tuple(int(item) for item in getattr(value, "shape", ()))
    if len(shape) != 3 or shape[0] != 1 or shape[1] <= 0 or shape[2] <= 0:
        raise SelectiveStateMemoryRepairError(f"{label} must have shape (1,rows,width)")
    if shape[1] % REPAIR_DENOMINATOR:
        raise SelectiveStateMemoryRepairError(f"{label} row count is not divisible by the fixed repair denominator")
    if not bool(torch.isfinite(value).all()):
        raise SelectiveStateMemoryRepairError(f"{label} must be finite")
    return shape


def _require_pair(pre: Any, proposed: Any, *, label: str, torch: Any) -> tuple[int, int, int]:
    shape = _require_tensor(pre, label=f"pre {label}", torch=torch)
    if _require_tensor(proposed, label=f"proposed {label}", torch=torch) != shape:
        raise SelectiveStateMemoryRepairError(f"{label} pre/proposed shapes differ")
    if pre.dtype != proposed.dtype or pre.device != proposed.device:
        raise SelectiveStateMemoryRepairError(f"{label} pre/proposed dtype or device differs")
    return shape


def _tensor_fingerprint(value: Any, *, torch: Any) -> str:
    """Hash device-side moments; no state or memory value enters the evidence.

    CPU unit tests use this internal primitive only to verify the mathematics.
    The v13 production runner separately requires CUDA tensors before it can
    invoke a repair, so only CUDA evidence can reach a real timeline.
    """
    if not bool(torch.is_tensor(value)) or not bool(torch.isfinite(value).all()):
        raise SelectiveStateMemoryRepairError("repair evidence tensor is unavailable")
    flat = value.detach().contiguous().reshape(-1).to(dtype=torch.float64)
    positions = torch.arange(1, flat.numel() + 1, dtype=torch.float64, device=flat.device)
    moments = (
        flat.sum(), flat.abs().sum(), (flat * positions).sum(), flat.square().sum(),
        flat.min(), flat.max(),
    )
    payload = "|".join((str(tuple(value.shape)), str(value.dtype), *(float(item.item()).hex() for item in moments)))
    return "gpu-fingerprint-v1:" + hashlib.sha256(payload.encode("ascii")).hexdigest()


def capture_state_memory_prestate(state_feat: Any, mem: Any, *, torch: Any) -> StateMemoryPreState:
    """Capture only references before a native recurrent update begins."""
    _require_tensor(state_feat, label="pre state_feat", torch=torch)
    _require_tensor(mem, label="pre mem", torch=torch)
    return StateMemoryPreState(state_feat, mem, _identity(state_feat), _identity(mem))


def require_unmutated_prestate(snapshot: StateMemoryPreState) -> None:
    """Fail closed if the native forward changed a pre-frame tensor in place."""
    if not isinstance(snapshot, StateMemoryPreState):
        raise SelectiveStateMemoryRepairError("state/memory pre-state snapshot is unavailable")
    if _identity(snapshot.state_feat) != snapshot.state_identity:
        raise SelectiveStateMemoryRepairError("pre state_feat was mutated in place")
    if _identity(snapshot.mem) != snapshot.mem_identity:
        raise SelectiveStateMemoryRepairError("pre mem was mutated in place")


def _repair_one(pre: Any, proposed: Any, *, label: str, torch: Any) -> tuple[Any, Mapping[str, Any]]:
    shape = _require_pair(pre, proposed, label=label, torch=torch)
    rows = shape[1]
    selected_count = rows // REPAIR_DENOMINATOR
    if selected_count <= 0:
        raise SelectiveStateMemoryRepairError(f"{label} has no selectable rows")
    delta = torch.sqrt(torch.mean((proposed - pre).square(), dim=2))
    if tuple(delta.shape) != (1, rows) or not bool(torch.isfinite(delta).all()):
        raise SelectiveStateMemoryRepairError(f"{label} row delta is malformed or nonfinite")
    ordered = torch.sort(delta, dim=1, descending=True).values
    # The operator is intentionally undefined when a tie crosses the fixed
    # top-k boundary: otherwise library tie ordering would silently select a
    # different repair set across devices or versions.
    if selected_count < rows and bool((ordered[:, selected_count - 1] == ordered[:, selected_count]).any().item()):
        raise SelectiveStateMemoryRepairError(f"{label} has a tie across the fixed repair boundary")
    indices = torch.topk(delta, k=selected_count, dim=1, largest=True, sorted=True).indices
    selected_delta = torch.gather(delta, 1, indices)
    if not bool((selected_delta > 0).any().item()):
        raise SelectiveStateMemoryRepairError(f"{label} selected rows make no repair")
    mask = torch.zeros((1, rows), dtype=torch.bool, device=pre.device)
    mask.scatter_(1, indices, True)
    repaired = torch.where(mask.unsqueeze(-1), pre, proposed)
    if not bool(torch.isfinite(repaired).all()):
        raise SelectiveStateMemoryRepairError(f"{label} repaired tensor is nonfinite")
    if not bool(torch.equal(repaired[mask], pre[mask])) or not bool(torch.equal(repaired[~mask], proposed[~mask])):
        raise SelectiveStateMemoryRepairError(f"{label} repair did not preserve its fixed selected/unselected contract")
    # Evidence is materialized only after GPU selection has completed.  It
    # contains indices (not tensor values) and scalar summaries; no CPU value
    # can influence the committed tensor or a later selection.
    selected_rows = [int(item) for item in indices.detach().cpu().reshape(-1).tolist()]
    return repaired, {
        "shape": list(shape),
        "dtype": str(pre.dtype),
        "device": str(pre.device),
        "row_count": rows,
        "selected_row_count": selected_count,
        "selected_rows": selected_rows,
        "selection": "largest_rowwise_rms_delta_fixed_one_eighth",
        "tie_free_boundary": True,
        "selected_nonzero_delta_count": int((selected_delta > 0).sum().item()),
        "row_delta_summary": {
            "min": float(delta.min().item()),
            "max": float(delta.max().item()),
            "mean": float(delta.mean().item()),
        },
        "pre_tensor_digest": _tensor_fingerprint(pre, torch=torch),
        "proposed_tensor_digest": _tensor_fingerprint(proposed, torch=torch),
        "committed_tensor_digest": _tensor_fingerprint(repaired, torch=torch),
        "selected_rows_equal_pre": True,
        "unselected_rows_equal_proposed": True,
    }


def selective_state_memory_repair(
    pre_state_feat: Any,
    proposed_state_feat: Any,
    pre_mem: Any,
    proposed_mem: Any,
    *,
    torch: Any,
) -> SelectiveStateMemoryRepairResult:
    """Restore only the fixed highest-delta state and memory rows on device."""
    state_feat, state_evidence = _repair_one(pre_state_feat, proposed_state_feat, label="state_feat", torch=torch)
    mem, mem_evidence = _repair_one(pre_mem, proposed_mem, label="mem", torch=torch)
    if tuple(state_feat.shape) != STATE_SHAPE or tuple(mem.shape) != MEMORY_SHAPE:
        raise SelectiveStateMemoryRepairError("pinned ReCal3R state/memory shapes changed")
    evidence = {
        "operator": "selective_state_memory_repair_v13",
        "repair_denominator": REPAIR_DENOMINATOR,
        "state_feat": state_evidence,
        "mem": mem_evidence,
        "raw_camera_pose_numeric_input": False,
        "fallback_used": False,
        "no_fallback": True,
    }
    return SelectiveStateMemoryRepairResult(state_feat, mem, evidence)


__all__ = [
    "MEMORY_SHAPE", "REPAIR_DENOMINATOR", "STATE_SHAPE",
    "SelectiveStateMemoryRepairError", "SelectiveStateMemoryRepairResult",
    "StateMemoryPreState", "capture_state_memory_prestate",
    "require_unmutated_prestate", "selective_state_memory_repair",
]
