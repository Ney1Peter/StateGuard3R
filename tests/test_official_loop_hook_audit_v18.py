from __future__ import annotations

from pathlib import Path

import pytest

from stateguard3r.official_loop_hook_audit_v18 import (
    MODEL,
    OfficialLoopHookAuditV18Error,
    audit_official_loop_hook_v18,
    require_v18_official_hook_contract,
)


def test_v18_audit_binds_the_official_callback_arguments_and_missing_memory() -> None:
    report = audit_official_loop_hook_v18()
    assert report["state_gpu_witness_available"] is True
    assert report["memory_gpu_witness_available"] is False
    assert report["permitted_post_commit_callback_args"] == ["frame_idx", "state_prev", "state_post"]
    assert report["forbidden_routes_needed_for_memory_witness"] == [
        "copied_recurrent_loop", "stack_frame_reflection", "additional_native_method_wrapper",
    ]


def test_v18_gate_a_fails_closed_when_the_registered_memory_witness_is_impossible() -> None:
    with pytest.raises(OfficialLoopHookAuditV18Error, match="memory identity"):
        require_v18_official_hook_contract()


def test_v18_audit_is_pinned_to_the_official_source_only() -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "stateguard3r" / "official_loop_hook_audit_v18.py"
    text = source.read_text(encoding="utf-8")
    assert "_v17" not in text
    assert "outputs/" not in text
    assert str(MODEL) in text
