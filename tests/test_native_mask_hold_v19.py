from __future__ import annotations

from types import SimpleNamespace

import pytest

from stateguard3r.native_mask_hold_v19 import (
    NativeMaskHoldV19Error,
    arm_one_native_state_mask_v19,
    audit_native_mask_wrapper_v19,
    select_native_mask_hold_v19,
    temporary_native_mask_hold_v19,
)


def test_v19_scalar_arm_contract() -> None:
    assert arm_one_native_state_mask_v19(False, alarm=False, has_next_view=True) is False
    assert arm_one_native_state_mask_v19(False, alarm=True, has_next_view=False) is False
    armed = arm_one_native_state_mask_v19(False, alarm=True, has_next_view=True)
    held = select_native_mask_hold_v19(armed, reset=False)
    assert held.consumed is True and held.return_zero_mask is True and held.armed_after is False
    reset = select_native_mask_hold_v19(True, reset=True)
    assert reset.reset_cancelled is True and reset.return_zero_mask is False
    with pytest.raises(NativeMaskHoldV19Error, match="prior"):
        arm_one_native_state_mask_v19(True, alarm=True, has_next_view=True)


@pytest.mark.parametrize("value", (0, None, "false", object()))
def test_v19_policy_rejects_nonbooleans(value: object) -> None:
    with pytest.raises(NativeMaskHoldV19Error):
        arm_one_native_state_mask_v19(value, alarm=False, has_next_view=True)  # type: ignore[arg-type]
    with pytest.raises(NativeMaskHoldV19Error):
        select_native_mask_hold_v19(False, reset=value)  # type: ignore[arg-type]


class _Mask:
    def __init__(self, value: int) -> None:
        self.value = value

    def clone(self) -> "_Mask":
        return _Mask(self.value)

    def zero_(self) -> "_Mask":
        self.value = 0
        return self


def test_v19_wrapper_calls_native_first_holds_only_return_and_restores() -> None:
    calls: list[str] = []
    native = _Mask(7)
    model = SimpleNamespace(_compute_recal3r_update_mask=lambda *_args, **_kwargs: calls.append("native") or native)
    original = model._compute_recal3r_update_mask
    selection = select_native_mask_hold_v19(True, reset=False)
    seen: list[_Mask] = []
    with temporary_native_mask_hold_v19(model, selection=selection, on_native_mask=seen.append):
        held = model._compute_recal3r_update_mask("a", prev_state_feat="b")
        assert calls == ["native"]
        assert seen == [native]
        assert held is not native and held.value == 0 and native.value == 7
    assert model._compute_recal3r_update_mask is original


def test_v19_wrapper_clear_is_native_identity_and_exception_restores() -> None:
    native = _Mask(5)
    model = SimpleNamespace(_compute_recal3r_update_mask=lambda *_args, **_kwargs: native)
    original = model._compute_recal3r_update_mask
    with temporary_native_mask_hold_v19(model, selection=select_native_mask_hold_v19(False, reset=False)):
        assert model._compute_recal3r_update_mask() is native
    with pytest.raises(RuntimeError):
        with temporary_native_mask_hold_v19(model, selection=select_native_mask_hold_v19(True, reset=False)):
            model._compute_recal3r_update_mask()
            raise RuntimeError("intentional")
    assert model._compute_recal3r_update_mask is original


def test_v19_wrapper_requires_exactly_one_native_call() -> None:
    model = SimpleNamespace(_compute_recal3r_update_mask=lambda: _Mask(1))
    with pytest.raises(NativeMaskHoldV19Error, match="exactly one"):
        with temporary_native_mask_hold_v19(model, selection=select_native_mask_hold_v19(False, reset=False)):
            pass


def test_v19_source_audit_binds_official_native_mask_and_memory_continuation() -> None:
    report = audit_native_mask_wrapper_v19()
    assert report["official_inference"] == "dust3r.inference.inference_recurrent_lighter"
    assert report["native_mask_before_state_memory_commit"] is True
    assert report["state_mask_only"] is True
    assert report["memory_mask_remains_native"] is True
