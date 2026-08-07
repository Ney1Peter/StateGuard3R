from __future__ import annotations

from typing import Any

from scripts import run_recal3r_online_quarantine_v2 as runner


class _NoGradContext:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __enter__(self) -> "_NoGradContext":
        self.events.append("enter")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.events.append("exit")
        return False


class _FakeTorch:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def no_grad(self) -> _NoGradContext:
        self.events.append("create")
        return _NoGradContext(self.events)


def test_online_runner_executes_operation_inside_no_grad() -> None:
    events: list[str] = []

    def operation() -> str:
        events.append("operation")
        return "result"

    assert runner._run_without_grad(_FakeTorch(events), operation) == "result"
    assert events == ["create", "enter", "operation", "exit"]


def test_online_runner_parser_defaults_to_v3_and_fixed_watchdog() -> None:
    parser = runner._parser()
    assert parser.get_default("health_profile") == "v3"
    assert parser.get_default("watchdog") == 8
