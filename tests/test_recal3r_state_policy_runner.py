from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_recal3r_state_policy_v3 import (
    _detector_v3_alarms,
    _parser,
    _run_without_grad,
    _validate_policy_args,
)
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
        "detector_run_json": None,
        "detector_alarm_timeline": None,
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
    _validate_policy_args(
        _policy_args(
            state_policy="detector-v3-prior-alarm",
            detector_run_json=Path("run.json"),
            detector_alarm_timeline=Path("timeline.json"),
        ),
        parser,
    )


def test_state_policy_script_is_directly_importable_outside_repository(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_recal3r_state_policy_v3.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--state-policy" in completed.stdout


def test_state_policy_forward_runs_inside_no_grad_context() -> None:
    events: list[str] = []

    class _Context:
        def __enter__(self) -> None:
            events.append("enter")

        def __exit__(self, *args: object) -> None:
            events.append("exit")

    class _Torch:
        def no_grad(self) -> _Context:
            return _Context()

    result = _run_without_grad(_Torch(), lambda: events.append("forward") or 7)

    assert result == 7
    assert events == ["enter", "forward", "exit"]


@pytest.mark.parametrize(
    "args",
    [
        _policy_args(health_profile="v1"),
        _policy_args(max_hold=4),
        _policy_args(alarm_frame=[1]),
        _policy_args(state_policy="forced-prior-alarm", alarm_frame=[]),
        _policy_args(state_policy="forced-prior-alarm", alarm_frame=[-1]),
        _policy_args(state_policy="detector-v3-prior-alarm"),
        _policy_args(
            state_policy="detector-v3-prior-alarm",
            alarm_frame=[1],
            detector_run_json=Path("run.json"),
            detector_alarm_timeline=Path("timeline.json"),
        ),
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


def test_detector_policy_reads_only_frozen_hybrid_alarm_and_bound_input(tmp_path: Path) -> None:
    input_manifest = tmp_path / "input-manifest.json"
    input_manifest.write_text("{}\n", encoding="utf-8")
    run = tmp_path / "run.json"
    run.write_text(
        json.dumps(
            {
                "health_profile": "v3",
                "input_manifest": {
                    "path": str(input_manifest),
                    "sha256": hashlib.sha256(input_manifest.read_bytes()).hexdigest(),
                    "size_bytes": input_manifest.stat().st_size,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    timeline = tmp_path / "timeline.json"
    timeline.write_text(
        '{"run_id":"blind-wrong-order","labels":[99,99,99],"attribution":['
        '{"frame_id":0,"continuous_alarm":false,"timestamp_order_alarm":false,"hybrid_alarm":false},'
        '{"frame_id":1,"continuous_alarm":true,"timestamp_order_alarm":false,"hybrid_alarm":true},'
        '{"frame_id":2,"continuous_alarm":false,"timestamp_order_alarm":true,"hybrid_alarm":true}'
        ']}\n',
        encoding="utf-8",
    )
    for path in (input_manifest, run, timeline):
        path.chmod(0o444)

    alarms, source = _detector_v3_alarms(
        _policy_args(
            state_policy="detector-v3-prior-alarm",
            input_manifest=input_manifest,
            detector_run_json=run,
            detector_alarm_timeline=timeline,
        )
    )

    assert alarms == [False, True, True]
    assert source["alarm_positions"] == [1, 2]
    assert source["causality"].endswith("not consumed")
