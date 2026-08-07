#!/usr/bin/env python3
"""Run one command in a freshly piped tmux pane and preserve its transcript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from typing import Sequence


class TmuxCapturedCommandError(RuntimeError):
    """The v7 terminal-evidence protocol could not complete safely."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--window-name", required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--main-log", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _require_new(paths: Sequence[Path]) -> None:
    duplicated = [str(path) for path in paths if path.exists()]
    if duplicated:
        raise TmuxCapturedCommandError(f"v7 refuses reused terminal-evidence path: {duplicated}")


def _quoted_command(command: Sequence[str]) -> str:
    if not command:
        raise TmuxCapturedCommandError("v7 requires a child command after --")
    return shlex.join(command)


def _driver_text(run_id: str, command: Sequence[str], main_log: Path, result: Path) -> str:
    """Build the separate pane driver; output is simultaneously pane and main log."""
    command_line = _quoted_command(command)
    run_id_q, command_q = shlex.quote(run_id), shlex.quote(command_line)
    main_q, result_q = shlex.quote(str(main_log)), shlex.quote(str(result))
    return f"""#!/usr/bin/env bash
set -u
run_id={run_id_q}
main_log={main_q}
result_path={result_q}
printf 'V7_DRIVER_START run_id=%s\\n' \"$run_id\" | tee -a \"$main_log\"
printf 'V7_DRIVER_COMMAND=%s\\n' {command_q} | tee -a \"$main_log\"
{command_line} > >(tee -a \"$main_log\") 2> >(tee -a \"$main_log\" >&2) &
child_pid=$!
printf 'V7_DRIVER_CHILD_PID=%s\\n' \"$child_pid\" | tee -a \"$main_log\"
wait \"$child_pid\"
child_status=$?
sleep 0.1
printf 'RUN_ID=%s\\nEXIT_CODE=%s\\n' \"$run_id\" \"$child_status\" > \"$result_path\"
printf 'V7_DRIVER_EXIT run_id=%s exit_code=%s\\n' \"$run_id\" \"$child_status\" | tee -a \"$main_log\"
exit \"$child_status\"
"""


def _parse_result(path: Path, *, run_id: str) -> int:
    try:
        rows = dict(
            line.split("=", 1)
            for line in path.read_text(encoding="utf-8").splitlines()
            if "=" in line
        )
        if rows != {"RUN_ID": run_id, "EXIT_CODE": rows.get("EXIT_CODE", "")}:
            raise ValueError("run ID mismatch or unexpected result fields")
        return int(rows["EXIT_CODE"])
    except (OSError, ValueError, KeyError) as error:
        raise TmuxCapturedCommandError(f"invalid v7 driver result: {path}") from error


def _verify_transcript(path: Path, *, run_id: str, command: Sequence[str], exit_code: int) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise TmuxCapturedCommandError("v7 tmux transcript is missing") from error
    expected = (
        f"V7_DRIVER_START run_id={run_id}",
        f"V7_DRIVER_COMMAND={_quoted_command(command)}",
        "V7_DRIVER_CHILD_PID=",
        f"V7_DRIVER_EXIT run_id={run_id} exit_code={exit_code}",
    )
    if not text or any(marker not in text for marker in expected):
        raise TmuxCapturedCommandError("v7 tmux transcript is incomplete or mismatched")


def _tmux(*args: str, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("tmux", *args),
        check=True,
        text=True,
        capture_output=capture_output,
    )


def _freeze(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.exists():
            path.chmod(0o444)


def run_captured_command(args: argparse.Namespace) -> int:
    command = tuple(args.command[1:] if args.command[:1] == ["--"] else args.command)
    if args.timeout_seconds <= 0:
        raise TmuxCapturedCommandError("v7 timeout must be positive")
    cwd = args.cwd.resolve(strict=True)
    artifacts = (args.transcript, args.main_log, args.driver, args.result)
    _require_new(artifacts)
    for artifact in artifacts:
        artifact.parent.mkdir(parents=True, exist_ok=True)
    args.driver.write_text(_driver_text(args.run_id, command, args.main_log, args.result), encoding="utf-8")
    args.driver.chmod(0o700)

    pane: str | None = None
    try:
        pane = _tmux(
            "new-window", "-d", "-P", "-F", "#{pane_id}", "-t", args.session,
            "-n", args.window_name, "-c", str(cwd), capture_output=True,
        ).stdout.strip()
        if not pane.startswith("%"):
            raise TmuxCapturedCommandError("tmux did not return a new pane ID")
        _tmux("pipe-pane", "-o", "-t", pane, f"cat >> {shlex.quote(str(args.transcript))}")
        _tmux("send-keys", "-t", pane, f"bash {shlex.quote(str(args.driver))}", "C-m")
        deadline = time.monotonic() + args.timeout_seconds
        while not args.result.exists():
            if time.monotonic() >= deadline:
                raise TmuxCapturedCommandError("v7 driver result timed out")
            time.sleep(0.05)
        exit_code = _parse_result(args.result, run_id=args.run_id)
        _verify_transcript(args.transcript, run_id=args.run_id, command=command, exit_code=exit_code)
        return exit_code
    finally:
        if pane is not None:
            try:
                _tmux("kill-pane", "-t", pane)
            except subprocess.CalledProcessError:
                pass
        _freeze(artifacts)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return run_captured_command(args)
    except (TmuxCapturedCommandError, OSError, subprocess.CalledProcessError) as error:
        print(f"TMUX_CAPTURE_V7_ERROR={error}")
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
