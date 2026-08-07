from __future__ import annotations

import pytest

from stateguard3r.recal3r_safe_anchor_runner_v3 import SafeAnchorRunnerError, _reset_requested


class _Flag:
    def __init__(self, value: bool) -> None:
        self.value = value

    def detach(self) -> "_Flag":
        return self

    def any(self) -> "_Flag":
        return self

    def item(self) -> bool:
        return self.value


def test_reset_predicate_accepts_bool_and_tensor_like_values() -> None:
    assert _reset_requested(None) is False
    assert _reset_requested(False) is False
    assert _reset_requested(_Flag(True)) is True


def test_runner_error_is_distinct_recovery_failure() -> None:
    assert issubclass(SafeAnchorRunnerError, RuntimeError)
