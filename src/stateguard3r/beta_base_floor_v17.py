"""Pure one-shot arming logic for the v17 native beta-base floor.

This module deliberately contains no model or tensor dependency.  It only
decides whether the next native ReCal3R mask calculation should use the fixed
floor, whether an existing arm is consumed, or whether a reset cancels it.
"""

from __future__ import annotations

from dataclasses import dataclass


NATIVE_BETA_BASE = 0.1
FLOORED_BETA_BASE = 0.0


class BetaBaseFloorV17Error(ValueError):
    """Raised when the narrow v17 scalar arm contract is violated."""


@dataclass(frozen=True, slots=True)
class BetaBaseFloorSelectionV17:
    """The scalar-only decision made before one native mask calculation."""

    armed_before: bool
    armed_after: bool
    beta_base_for_mask: float
    consumed: bool
    reset_cancelled: bool


def _plain_bool(value: bool, *, name: str) -> bool:
    if type(value) is not bool:
        raise BetaBaseFloorV17Error(f"{name} must be a plain boolean")
    return value


def arm_one_future_update_v17(armed: bool, *, alarm: bool) -> bool:
    """Record one alarm only after its raw health has been committed.

    A second arm cannot overwrite an outstanding one.  The runner consumes or
    cancels every old arm before it permits the observer to call this function
    again, so rejection here is a fail-closed causal-order check.
    """

    armed = _plain_bool(armed, name="armed")
    alarm = _plain_bool(alarm, name="alarm")
    if armed:
        raise BetaBaseFloorV17Error("prior v17 arm was neither consumed nor reset-cancelled")
    return alarm


def select_one_shot_beta_base_v17(
    armed: bool,
    *,
    reset: bool,
) -> BetaBaseFloorSelectionV17:
    """Choose the scalar beta base for the next native mask computation.

    This function deliberately cannot inspect eligibility tensors or model
    values.  The native recurrent runner supplies the already-resolved reset
    boolean, uses the returned scalar around exactly one native mask call, and
    restores that scalar before proceeding.
    """

    armed = _plain_bool(armed, name="armed")
    reset = _plain_bool(reset, name="reset")
    if armed and reset:
        return BetaBaseFloorSelectionV17(
            armed_before=True,
            armed_after=False,
            beta_base_for_mask=NATIVE_BETA_BASE,
            consumed=False,
            reset_cancelled=True,
        )
    if armed:
        return BetaBaseFloorSelectionV17(
            armed_before=True,
            armed_after=False,
            beta_base_for_mask=FLOORED_BETA_BASE,
            consumed=True,
            reset_cancelled=False,
        )
    return BetaBaseFloorSelectionV17(
        armed_before=False,
        armed_after=False,
        beta_base_for_mask=NATIVE_BETA_BASE,
        consumed=False,
        reset_cancelled=False,
    )


__all__ = [
    "FLOORED_BETA_BASE",
    "NATIVE_BETA_BASE",
    "BetaBaseFloorSelectionV17",
    "BetaBaseFloorV17Error",
    "arm_one_future_update_v17",
    "select_one_shot_beta_base_v17",
]
