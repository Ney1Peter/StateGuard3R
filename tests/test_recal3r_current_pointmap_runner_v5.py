from __future__ import annotations

from pathlib import Path

from stateguard3r import recal3r_current_pointmap_runner_v5 as runner


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
    assert runner._reset_requested(None) is False
    assert runner._reset_requested(False) is False
    assert runner._reset_requested(_Flag(True)) is True


def test_runner_error_is_distinct_and_runner_does_not_delegate_to_v3_or_v4() -> None:
    assert issubclass(runner.CurrentPointmapRunnerError, RuntimeError)
    source = Path(runner.__file__).read_text()
    assert "recal3r_safe_anchor_runner_v3" not in source
    assert "recal3r_geometric_registration_runner_v4" not in source
    assert "raw_prediction = to_cpu(res)" not in source
    assert "observer.finalize(observation, quarantined=quarantine, prediction=res)" in source
    assert "predictions.append(to_cpu(exported))" in source
