"""Gate-A audit of whether v19's approved official-loop hooks are causal."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any, Mapping


MODEL = Path("/data/wangzheng/Project2/baselines/ReCal3R/src/dust3r/model.py")
MODEL_SHA256 = "32785a6f29fded66aa142207b29a33d7522f0c39aa068fe2175a956ec8dbc3c1"
APPROVED_V19_INSTANCE_WRAPPERS = frozenset({
    "_downstream_head", "_compute_recal3r_update_mask", "_maybe_record_u_calibration_step",
})


class V19CausalHookAuditError(ValueError):
    """The registered v19 hooks cannot satisfy its causal order."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    candidates = [item for item in ast.walk(tree) if isinstance(item, ast.FunctionDef) and item.name == name]
    if len(candidates) != 1:
        raise V19CausalHookAuditError(f"pinned official function differs: {name}")
    return candidates[0]


def _self_calls(function: ast.FunctionDef) -> dict[str, int]:
    calls: dict[str, int] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
            calls[node.func.attr] = calls.get(node.func.attr, 0) + 1
    return calls


def audit_v19_causal_hook_order() -> Mapping[str, Any]:
    """Prove that registered v19 wrappers occur after the model-ready stage.

    `forward_recurrent_lighter` directly invokes `_encode_image` before its
    rollout.  The three wrappers admitted by the pre-registered v19 plan are
    all after that point.  Thus none can calculate the required current RGB
    overlap *before* current native rollout without an unapproved encode/device
    wrapper or a copied loop.
    """

    if _sha256(MODEL) != MODEL_SHA256:
        raise V19CausalHookAuditError("pinned ReCal3R source differs")
    source = MODEL.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(MODEL))
    lighter = _function(tree, "forward_recurrent_lighter")
    calls = _self_calls(lighter)
    if calls.get("_encode_image") != 1 or calls.get("_recurrent_rollout") != 1:
        raise V19CausalHookAuditError("official encode/rollout call cardinality differs")
    if any(calls.get(name) != 1 for name in APPROVED_V19_INSTANCE_WRAPPERS):
        raise V19CausalHookAuditError("registered v19 wrapper call cardinality differs")
    start = source.index("def forward_recurrent_lighter")
    positions = {
        "encode_image": source.index("img_out, img_pos, _ = self._encode_image", start),
        "recurrent_rollout": source.index("new_state_feat, dec, self_attn_state", start),
        "downstream_head": source.index("res = self._downstream_head(", start),
        "native_mask": source.index("update_mask1 = self._compute_recal3r_update_mask(", start),
        "post_commit_calibration": source.index("self._maybe_record_u_calibration_step(i, prev_state_feat, state_feat)", start),
    }
    if list(positions.values()) != sorted(positions.values()):
        raise V19CausalHookAuditError("official frame order differs")
    return {
        "schema_version": "stateguard3r.v19-causal-hook-audit.v1",
        "model_path": str(MODEL),
        "model_sha256": MODEL_SHA256,
        "approved_v19_instance_wrappers": sorted(APPROVED_V19_INSTANCE_WRAPPERS),
        "official_order": list(positions),
        "pre_rollout_rgb_hook_available": False,
        "approved_wrappers_all_after_rollout": True,
        "forbidden_routes_needed_for_registered_overlap_order": [
            "additional_encode_or_device_wrapper",
            "copied_recurrent_loop",
            "offline_future_frame_precomputation",
        ],
    }


def require_v19_registered_causal_order() -> Mapping[str, Any]:
    report = audit_v19_causal_hook_order()
    if report["pre_rollout_rgb_hook_available"] is not True:
        raise V19CausalHookAuditError(
            "v19 approved official-loop hooks cannot compute current RGB overlap before rollout"
        )
    return report


__all__ = [
    "APPROVED_V19_INSTANCE_WRAPPERS", "MODEL", "MODEL_SHA256",
    "V19CausalHookAuditError", "audit_v19_causal_hook_order",
    "require_v19_registered_causal_order",
]
