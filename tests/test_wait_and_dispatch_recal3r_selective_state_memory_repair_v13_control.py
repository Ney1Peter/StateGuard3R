from __future__ import annotations

import json
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAITER = PROJECT_ROOT / "scripts" / "wait_and_dispatch_recal3r_selective_state_memory_repair_v13_control.sh"


def test_v13_control_waiter_is_shell_valid_and_has_only_the_fixed_gate_b_route() -> None:
    completed = subprocess.run(["bash", "-n", str(WAITER)], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    source = WAITER.read_text(encoding="utf-8")
    assert 'RUN_ID="recovery-selective-state-memory-repair-v13-dynamic-always-commit-0001"' in source
    assert 'GPU_UUID="GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250"' in source
    assert "MIN_FREE_MIB=12288" in source
    assert "CONFIRM_SECONDS=30" in source
    assert source.count('gpu_free_mib)') == 2
    assert 'git -C "${ROOT}" status --porcelain' in source
    assert 'git -C "${RECAL3R_ROOT}" status --porcelain' in source
    assert "return 0" in source
    assert '"${PYTHON_BIN}" "${DISPATCHER}" --run-id "${RUN_ID}"' in source
    assert '"--state-policy", "always-commit"' in source
    assert '"--device", "cuda"' in source
    for forbidden in ("torch", "dynamic-candidate", "development-wrong", "development-low", "patch_embed"):
        assert forbidden not in source


def test_v13_waiter_executes_clean_worktree_success_branch_and_builds_canonical_command(tmp_path: Path) -> None:
    source = f'''source {str(WAITER)!r}
record() {{ :; }}
git() {{ return 0; }}
require_clean_tracked_worktrees
ROOT={str(tmp_path / "StateGuard3R")!r}
RECAL3R_ROOT={str(tmp_path / "ReCal3R")!r}
PYTHON_BIN=$(command -v python3)
RUN_ID="recovery-selective-state-memory-repair-v13-dynamic-always-commit-0001"
COMMAND_JSON={str(tmp_path / "command.json")!r}
mkdir -p "$ROOT/outputs/formal-v1-inputs-0001/development/development-dynamic" "$ROOT/outputs/formal-v3-calibration-0001" "$RECAL3R_ROOT/src" "$RECAL3R_ROOT/data/tum/rgbd_dataset_freiburg1_desk"
touch "$ROOT/outputs/formal-v1-inputs-0001/development/development-dynamic/input-manifest.json" "$ROOT/outputs/formal-v3-calibration-0001/formal-config.json" "$RECAL3R_ROOT/src/cut3r_512_dpt_4_64.pth" "$RECAL3R_ROOT/data/tum/rgbd_dataset_freiburg1_desk/rgb.txt"
write_one_use_command_json
'''
    completed = subprocess.run(["bash", "-c", source], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    argv = json.loads((tmp_path / "command.json").read_text(encoding="utf-8"))
    assert argv[1].endswith("scripts/run_recal3r_selective_state_memory_repair_v13.py")
    assert argv[argv.index("--state-policy") + 1] == "always-commit"
    assert argv[argv.index("--output-dir") + 1].endswith("recovery-selective-state-memory-repair-v13-dynamic-always-commit-0001")
