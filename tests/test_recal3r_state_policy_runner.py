from __future__ import annotations

import argparse

import pytest

from scripts.run_recal3r_state_policy_v3 import _parser, _validate_policy_args
from stateguard3r.recal3r_transactional_v3 import (
    TransactionalReCal3RError,
    run_transactional_recurrent_lighter,
)


def _policy_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "health_profile": "v2",
        "max_hold": 3,
        "state_policy": "always-commit",
        "alarm_frame": [],
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_state_policy_parser_defaults_to_v2_and_accepts_control_modes() -> None:
    parser = _parser()

    assert parser.get_default("health_profile") == "v2"
    _validate_policy_args(_policy_args(), parser)
    _validate_policy_args(
        _policy_args(state_policy="forced-prior-alarm", alarm_frame=[1, 4]), parser
    )


@pytest.mark.parametrize(
    "args",
    [
        _policy_args(health_profile="v1"),
        _policy_args(max_hold=4),
        _policy_args(alarm_frame=[1]),
        _policy_args(state_policy="forced-prior-alarm", alarm_frame=[]),
        _policy_args(state_policy="forced-prior-alarm", alarm_frame=[-1]),
    ],
)
def test_state_policy_parser_rejects_unsafe_or_ambiguous_controls(
    args: argparse.Namespace,
) -> None:
    with pytest.raises(SystemExit):
        _validate_policy_args(args, _parser())


def test_transactional_runner_rejects_misaligned_alarm_sequence_before_execution() -> None:
    with pytest.raises(TransactionalReCal3RError, match="alarm count"):
        run_transactional_recurrent_lighter(
            [object()],
            object(),
            "cpu",
            torch=object(),
            to_gpu=lambda value, device: value,
            to_cpu=lambda value: value,
            canonicalize_model_update_type=lambda value: str(value),
            alarms=[],
            verify_source=False,
        )
