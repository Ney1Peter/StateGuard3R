from __future__ import annotations

from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAITER = PROJECT_ROOT / "scripts" / "wait_and_dispatch_recal3r_patch_embed_global_pooled_pose_v12_control.sh"


def test_v12_control_waiter_is_shell_valid_and_has_only_the_fixed_gate_b_route() -> None:
    completed = subprocess.run(["bash", "-n", str(WAITER)], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    source = WAITER.read_text(encoding="utf-8")
    assert 'RUN_ID="recovery-patch-embed-global-pooled-pose-v12-dynamic-always-commit-0001"' in source
    assert 'GPU_UUID="GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250"' in source
    assert "MIN_FREE_MIB=12288" in source
    assert "CONFIRM_SECONDS=30" in source
    assert source.count('gpu_free_mib)') == 2
    assert 'git -C "${ROOT}" status --porcelain' in source
    assert 'git -C "${RECAL3R_ROOT}" status --porcelain' in source
    assert '"${PYTHON_BIN}" "${DISPATCHER}" --run-id "${RUN_ID}"' in source
    assert '"--state-policy", "always-commit"' in source
    assert '"--device", "cuda"' in source
    for forbidden in (
        "torch", "model.patch_embed", "run_recal3r_encoder_global_pooled_pose",
        "dynamic-candidate", "development-wrong", "development-low",
    ):
        assert forbidden not in source
