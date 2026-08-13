"""Static Gate-A audit for v18's official-loop-only intervention boundary."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any, Mapping


MODEL = Path("/data/wangzheng/Project2/baselines/ReCal3R/src/dust3r/model.py")
MODEL_SHA256 = "32785a6f29fded66aa142207b29a33d7522f0c39aa068fe2175a956ec8dbc3c1"


class OfficialLoopHookAuditV18Error(ValueError):
    """The pinned official loop cannot satisfy the pre-registered v18 hook."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _method(tree: ast.Module, name: str) -> ast.FunctionDef:
    for item in ast.walk(tree):
        if isinstance(item, ast.FunctionDef) and item.name == name:
            return item
    raise OfficialLoopHookAuditV18Error(f"pinned official method is absent: {name}")


def _method_call_arguments(function: ast.FunctionDef, method: str) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "self" or node.func.attr != method:
            continue
        if not all(isinstance(argument, ast.Name) for argument in node.args):
            raise OfficialLoopHookAuditV18Error(f"official {method} call arguments differ")
        rows.append(tuple(argument.id for argument in node.args))
    return tuple(rows)


def audit_official_loop_hook_v18() -> Mapping[str, Any]:
    """Prove whether v18 can obtain every promised witness without copying a loop.

    The preregistered v18 hook allows only instance-local wrappers around the
    official downstream head and post-commit calibration call.  The latter has
    only ``(frame_idx, state_prev, state_post)`` arguments.  Since native
    ``mem`` is a lexical local and no permitted callback receives it after its
    commit, the required held-frame *memory GPU summary* is unavailable unless
    v18 copies the loop, uses stack-frame reflection, or wraps another method.
    Every option is prohibited by the plan.
    """

    if _sha256(MODEL) != MODEL_SHA256:
        raise OfficialLoopHookAuditV18Error("pinned ReCal3R source hash differs")
    source = MODEL.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(MODEL))
    lighter = _method(tree, "forward_recurrent_lighter")
    downstream = _method_call_arguments(lighter, "_downstream_head")
    calibration = _method_call_arguments(lighter, "_maybe_record_u_calibration_step")
    if downstream != (("head_input", "shape"),):
        raise OfficialLoopHookAuditV18Error("official downstream hook call differs")
    if calibration != (("i", "prev_state_feat", "state_feat"),):
        raise OfficialLoopHookAuditV18Error("official calibration hook call differs")
    source_order = (
        source.index("res = self._downstream_head(", source.index("def forward_recurrent_lighter")),
        source.index("state_feat = new_state_feat * update_mask1", source.index("def forward_recurrent_lighter")),
        source.index("mem = new_mem * update_mask2", source.index("def forward_recurrent_lighter")),
        source.index("self._maybe_record_u_calibration_step(i, prev_state_feat, state_feat)", source.index("def forward_recurrent_lighter")),
    )
    if source_order != tuple(sorted(source_order)):
        raise OfficialLoopHookAuditV18Error("official native causal order differs")
    return {
        "schema_version": "stateguard3r.official-loop-hook-audit-v18.v1",
        "model_path": str(MODEL),
        "model_sha256": MODEL_SHA256,
        "official_lighter_loop": "forward_recurrent_lighter",
        "permitted_downstream_callback_args": ["head_input", "shape"],
        "permitted_post_commit_callback_args": ["frame_idx", "state_prev", "state_post"],
        "state_gpu_witness_available": True,
        "memory_gpu_witness_available": False,
        "forbidden_routes_needed_for_memory_witness": [
            "copied_recurrent_loop",
            "stack_frame_reflection",
            "additional_native_method_wrapper",
        ],
    }


def require_v18_official_hook_contract() -> Mapping[str, Any]:
    """Fail closed because Gate-A's registered memory witness is impossible."""

    report = audit_official_loop_hook_v18()
    if report["memory_gpu_witness_available"] is not True:
        raise OfficialLoopHookAuditV18Error(
            "official-loop-only hook cannot witness held native memory identity"
        )
    return report


__all__ = [
    "MODEL",
    "MODEL_SHA256",
    "OfficialLoopHookAuditV18Error",
    "audit_official_loop_hook_v18",
    "require_v18_official_hook_contract",
]
