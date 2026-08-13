from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

from scripts import dispatch_recal3r_bounded_update_pressure_v16 as dispatch


ROOT = Path(__file__).resolve().parents[1]
WAITER = ROOT / "scripts" / "wait_and_dispatch_recal3r_bounded_update_pressure_v16_control.sh"


def test_v16_waiter_is_shell_valid_and_owns_exclusive_terminal_log() -> None:
    completed = subprocess.run(["bash", "-n", str(WAITER)], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    source = WAITER.read_text(encoding="utf-8")
    assert 'RUN_ID="recovery-update-pressure-v16-dynamic-always-commit-0001"' in source
    assert 'GPU_UUID="GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250"' in source
    assert "MIN_FREE_MIB=12288" in source and "CONFIRM_SECONDS=30" in source
    assert "os.O_EXCL" in source and "freeze_terminal_log" in source
    assert '"${PYTHON_BIN}" "${DISPATCHER}" --run-id "${RUN_ID}"' in source
    for forbidden in ("dynamic-candidate", "development-wrong", "development-low", "torch"):
        assert forbidden not in source


def test_v16_waiter_log_create_is_o_excl_and_freezes(tmp_path: Path) -> None:
    log = tmp_path / "wait.log"
    command = f'''source {str(WAITER)!r}
PYTHON_BIN={sys.executable!r}
WAIT_LOG={str(log)!r}
create_terminal_log || exit $?
record one
freeze_terminal_log
'''
    completed = subprocess.run(["bash", "-c", command], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert log.read_text(encoding="utf-8") == "one\n"
    assert log.stat().st_mode & 0o777 == 0o444
    repeated = subprocess.run(["bash", "-c", command], text=True, capture_output=True)
    assert repeated.returncode != 0


def test_v16_waiter_and_dispatcher_share_the_same_exact_old_terminal_allowlist() -> None:
    source = WAITER.read_text(encoding="utf-8")
    literal = source.split("allowed = ", 1)[1].split("\nstatus = ", 1)[0]
    assert ast.literal_eval(literal) == dispatch.PREEXISTING_TERMINAL_UNTRACKED
