"""Pure one-shot arming for v18's native update-eligibility hold.

The policy knows only booleans.  It cannot inspect a model, tensor, RGB value,
prediction, timestamp, health scalar or output artifact.
"""

from __future__ import annotations

from dataclasses import dataclass


class NativeUpdateHoldV18Error(ValueError):
    """A caller violated v18's scalar-only hold policy."""


@dataclass(frozen=True, slots=True)
class NativeUpdateHoldSelectionV18:
    """Decision for the immediate next official native view."""

    armed_before: bool
    armed_after: bool
    consumed: bool
    reset_cancelled: bool
    inject_update_false: bool


def _boolean(value: bool, *, label: str) -> bool:
    if type(value) is not bool:
        raise NativeUpdateHoldV18Error(f"{label} must be a plain boolean")
    return value


def arm_one_native_update_v18(armed: bool, *, alarm: bool, has_next_view: bool) -> bool:
    """Commit a current alarm into one future eligible native view only."""

    armed = _boolean(armed, label="armed")
    alarm = _boolean(alarm, label="alarm")
    has_next_view = _boolean(has_next_view, label="has_next_view")
    if armed:
        raise NativeUpdateHoldV18Error("prior v18 arm was neither consumed nor reset-cancelled")
    return bool(alarm and has_next_view)


def select_native_update_hold_v18(armed: bool, *, reset: bool) -> NativeUpdateHoldSelectionV18:
    """Consume or reset-cancel an arm before the official native frame starts."""

    armed = _boolean(armed, label="armed")
    reset = _boolean(reset, label="reset")
    if armed and reset:
        return NativeUpdateHoldSelectionV18(
            armed_before=True,
            armed_after=False,
            consumed=False,
            reset_cancelled=True,
            inject_update_false=False,
        )
    if armed:
        return NativeUpdateHoldSelectionV18(
            armed_before=True,
            armed_after=False,
            consumed=True,
            reset_cancelled=False,
            inject_update_false=True,
        )
    return NativeUpdateHoldSelectionV18(
        armed_before=False,
        armed_after=False,
        consumed=False,
        reset_cancelled=False,
        inject_update_false=False,
    )


__all__ = [
    "NativeUpdateHoldSelectionV18",
    "NativeUpdateHoldV18Error",
    "arm_one_native_update_v18",
    "select_native_update_hold_v18",
]
