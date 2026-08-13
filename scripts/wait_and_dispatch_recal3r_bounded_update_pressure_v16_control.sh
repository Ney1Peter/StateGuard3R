#!/usr/bin/env bash
# Wait without reserving GPU, then release only v16's preregistered control.
# This program never imports a model.  CUDA can exist only in the independent
# dispatcher's token-gated child.  Its terminal log is one-use/O_EXCL.
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
RECAL3R_ROOT="${ROOT}/../baselines/ReCal3R"
PYTHON_BIN="${RECAL3R_ROOT}/.venv/bin/python"
DISPATCHER="${ROOT}/scripts/dispatch_recal3r_bounded_update_pressure_v16.py"
RUN_ID="recovery-update-pressure-v16-dynamic-always-commit-0001"
GPU_UUID="GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250"
MIN_FREE_MIB=12288
POLL_SECONDS=60
CONFIRM_SECONDS=30
WAIT_LOG="${ROOT}/logs/${RUN_ID}-wait.log"

create_terminal_log() {
    WAIT_LOG="${WAIT_LOG}" "${PYTHON_BIN}" - <<'PY'
import os
import stat

path = os.environ["WAIT_LOG"]
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
descriptor = os.open(path, flags, 0o644)
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
if stat.S_IMODE(os.lstat(path).st_mode) != 0o644:
    raise RuntimeError("wait log mode differs")
PY
}

record() {
    printf '%s\n' "$1" >> "${WAIT_LOG}"
}

freeze_terminal_log() {
    chmod 0444 "${WAIT_LOG}"
    [[ "$(stat -c '%a' "${WAIT_LOG}")" == "444" ]]
}

gpu_free_mib() {
    nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits |
        awk -F ',' -v uuid="${GPU_UUID}" '
            {
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
                if ($1 == uuid) { print $2; found += 1 }
            }
            END { if (found != 1) exit 1 }
        '
}

require_clean_worktrees() {
    # Only the exact older terminal inventory may remain untracked.  No new
    # file (including uncommitted v16 source) is tolerated before dispatch.
    if ! ROOT="${ROOT}" "${PYTHON_BIN}" - <<'PY'
import os
import subprocess

allowed = {
    "scripts/build_recal3r_dynamic_input_capsule_v14.py",
    "scripts/build_recal3r_rgb_listing_capsule_v15.py",
    "scripts/dispatch_recal3r_bounded_update_pressure_v15.py",
    "scripts/dispatch_recal3r_update_pressure_bounded_state_memory_v14.py",
    "scripts/probe_recal3r_bounded_update_pressure_interface_v15.py",
    "scripts/probe_recal3r_update_pressure_interface_v14.py",
    "scripts/run_recal3r_bounded_update_pressure_v15.py",
    "scripts/run_recal3r_update_pressure_bounded_state_memory_v14.py",
    "scripts/wait_and_dispatch_recal3r_bounded_update_pressure_v15_control.sh",
    "scripts/wait_and_dispatch_recal3r_update_pressure_bounded_state_memory_v14_control.sh",
    "src/stateguard3r/bounded_update_pressure_v15.py",
    "src/stateguard3r/online_detector_v15.py",
    "src/stateguard3r/online_visual_overlap_v15.py",
    "src/stateguard3r/recal3r_bounded_update_pressure_runner_v15.py",
    "src/stateguard3r/recal3r_update_pressure_bounded_state_memory_runner_v14.py",
    "src/stateguard3r/rgb_listing_capsule_v15.py",
    "src/stateguard3r/runtime_input_capsule_v14.py",
    "src/stateguard3r/timestamp_order_v15.py",
    "src/stateguard3r/update_pressure_bounded_state_memory_v14.py",
    "tests/test_bounded_update_pressure_v15.py",
    "tests/test_build_recal3r_dynamic_input_capsule_v14.py",
    "tests/test_dispatch_recal3r_update_pressure_bounded_state_memory_v14.py",
    "tests/test_probe_recal3r_bounded_update_pressure_interface_v15.py",
    "tests/test_probe_recal3r_update_pressure_interface_v14.py",
    "tests/test_recal3r_bounded_update_pressure_runner_v15.py",
    "tests/test_recal3r_update_pressure_bounded_state_memory_runner_v14.py",
    "tests/test_rgb_listing_capsule_v15.py",
    "tests/test_run_recal3r_bounded_update_pressure_v15.py",
    "tests/test_run_recal3r_update_pressure_bounded_state_memory_v14.py",
    "tests/test_runtime_input_capsule_v14.py",
    "tests/test_update_pressure_bounded_state_memory_v14.py",
}
status = subprocess.run(
    ["git", "-C", os.environ["ROOT"], "status", "--porcelain=v1", "--untracked-files=all"],
    check=True, text=True, capture_output=True,
).stdout
for row in status.splitlines():
    if len(row) < 4 or row[:2] != "??" or row[3:] not in allowed:
        raise SystemExit(1)
PY
    then
        record "V16_WAIT_ABORT run_id=${RUN_ID} reason=dirty_v16_relevant_tree at=$(date --iso-8601=seconds)"
        return 1
    fi
    if [[ -n "$(git -C "${RECAL3R_ROOT}" status --porcelain)" ]]; then
        record "V16_WAIT_ABORT run_id=${RUN_ID} reason=dirty_recal3r at=$(date --iso-8601=seconds)"
        return 1
    fi
    return 0
}

main() {
    if ! create_terminal_log; then
        return 70
    fi
    record "V16_WAIT_START run_id=${RUN_ID} min_free_mib=${MIN_FREE_MIB} at=$(date --iso-8601=seconds)"
    while true; do
        if first="$(gpu_free_mib)" && [[ "${first}" =~ ^[0-9]+$ ]] && (( first >= MIN_FREE_MIB )); then
            sleep "${CONFIRM_SECONDS}"
            if second="$(gpu_free_mib)" && [[ "${second}" =~ ^[0-9]+$ ]] && (( second >= MIN_FREE_MIB )); then
                record "V16_WAIT_EXTERNAL_GATE_PASS run_id=${RUN_ID} first_free_mib=${first} second_free_mib=${second} at=$(date --iso-8601=seconds)"
                if ! require_clean_worktrees; then
                    code=70
                else
                    record "V16_WAIT_DISPATCH run_id=${RUN_ID} at=$(date --iso-8601=seconds)"
                    "${PYTHON_BIN}" "${DISPATCHER}" --run-id "${RUN_ID}" --timeout-seconds 7200 >>"${WAIT_LOG}" 2>&1
                    code=$?
                    record "V16_WAIT_DISPATCH_EXIT=${code}"
                fi
                record "V16_WAIT_TERMINAL run_id=${RUN_ID} exit_code=${code} at=$(date --iso-8601=seconds)"
                freeze_terminal_log
                return "${code}"
            fi
            record "V16_WAIT_CONFIRM_REJECTED run_id=${RUN_ID} first_free_mib=${first} second_free_mib=${second:-unavailable} at=$(date --iso-8601=seconds)"
        else
            record "V16_WAIT_POLL run_id=${RUN_ID} free_mib=${first:-unavailable} at=$(date --iso-8601=seconds)"
        fi
        unset first second
        sleep "${POLL_SECONDS}"
    done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
