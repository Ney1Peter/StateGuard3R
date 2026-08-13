from __future__ import annotations

import pytest

from stateguard3r.v19_causal_hook_audit import (
    APPROVED_V19_INSTANCE_WRAPPERS,
    V19CausalHookAuditError,
    audit_v19_causal_hook_order,
    require_v19_registered_causal_order,
)


def test_v19_audit_binds_all_allowed_wrappers_after_the_official_rollout() -> None:
    report = audit_v19_causal_hook_order()
    assert report["approved_v19_instance_wrappers"] == sorted(APPROVED_V19_INSTANCE_WRAPPERS)
    assert report["official_order"] == [
        "encode_image", "recurrent_rollout", "downstream_head", "native_mask", "post_commit_calibration",
    ]
    assert report["approved_wrappers_all_after_rollout"] is True
    assert report["pre_rollout_rgb_hook_available"] is False


def test_v19_gate_a_fails_closed_instead_of_moving_overlap_after_rollout() -> None:
    with pytest.raises(V19CausalHookAuditError, match="before rollout"):
        require_v19_registered_causal_order()
