from __future__ import annotations

import pytest

from stateguard3r.native_update_hold_v18 import (
    NativeUpdateHoldV18Error,
    arm_one_native_update_v18,
    select_native_update_hold_v18,
)


def test_v18_clear_and_last_alarm_are_identity() -> None:
    assert arm_one_native_update_v18(False, alarm=False, has_next_view=True) is False
    assert arm_one_native_update_v18(False, alarm=True, has_next_view=False) is False
    selected = select_native_update_hold_v18(False, reset=False)
    assert selected.consumed is False
    assert selected.inject_update_false is False


def test_v18_one_arm_consumes_exactly_one_future_nonreset_view() -> None:
    armed = arm_one_native_update_v18(False, alarm=True, has_next_view=True)
    selected = select_native_update_hold_v18(armed, reset=False)
    assert selected.armed_before is True
    assert selected.armed_after is False
    assert selected.consumed is True
    assert selected.reset_cancelled is False
    assert selected.inject_update_false is True


def test_v18_reset_cancels_an_arm_without_injecting() -> None:
    selected = select_native_update_hold_v18(True, reset=True)
    assert selected.armed_before is True
    assert selected.armed_after is False
    assert selected.consumed is False
    assert selected.reset_cancelled is True
    assert selected.inject_update_false is False


@pytest.mark.parametrize("value", (0, 1, None, "false", object()))
def test_v18_policy_rejects_nonboolean_input(value: object) -> None:
    with pytest.raises(NativeUpdateHoldV18Error):
        arm_one_native_update_v18(value, alarm=False, has_next_view=True)  # type: ignore[arg-type]
    with pytest.raises(NativeUpdateHoldV18Error):
        select_native_update_hold_v18(False, reset=value)  # type: ignore[arg-type]


def test_v18_policy_rejects_a_second_outstanding_arm() -> None:
    with pytest.raises(NativeUpdateHoldV18Error, match="prior"):
        arm_one_native_update_v18(True, alarm=True, has_next_view=True)
