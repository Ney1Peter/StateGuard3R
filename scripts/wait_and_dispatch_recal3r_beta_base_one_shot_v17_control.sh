#!/usr/bin/env bash
# Wait without reserving GPU, then release only v17's preregistered control.
# The waiter never imports a model and cannot reach CUDA.  Its own log is
# O_EXCL and immutable so a second invocation cannot appear to be the first.
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
RECAL3R_ROOT="${ROOT}/../baselines/ReCal3R"
PYTHON_BIN="${RECAL3R_ROOT}/.venv/bin/python"
DISPATCHER="${ROOT}/scripts/dispatch_recal3r_beta_base_one_shot_v17.py"
RUN_ID="recovery-beta-floor-v17-dynamic-always-commit-0001"
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
                if ($1 == uuid && $2 ~ /^[0-9]+$/) { print $2; found=1; exit }
            }
            END { if (!found) exit 1 }
        '
}

project_gpu_process_count() {
    local result
    result="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)"
    [[ -z "${result}" ]] && { printf '0\n'; return; }
    while IFS=, read -r uuid pid; do
        uuid="${uuid//[[:space:]]/}"
        pid="${pid//[[:space:]]/}"
        if [[ "${uuid}" == "${GPU_UUID}" && "${pid}" =~ ^[0-9]+$ && -r "/proc/${pid}/cmdline" ]]; then
            if tr '\0' ' ' < "/proc/${pid}/cmdline" | grep -F -q "${ROOT}"; then
                printf '1\n'
                return
            fi
        fi
    done <<< "${result}"
    printf '0\n'
}

ready_once() {
    local free
    free="$(gpu_free_mib)" || return 1
    [[ "${free}" =~ ^[0-9]+$ && "${free}" -ge "${MIN_FREE_MIB}" ]] || return 1
    [[ "$(project_gpu_process_count)" == "0" ]]
}

main() {
    create_terminal_log
    trap 'freeze_terminal_log' EXIT
    record "V17_WAITER_START run_id=${RUN_ID} gpu_uuid=${GPU_UUID} minimum_free_mib=${MIN_FREE_MIB}"
    while ! ready_once; do
        record "V17_WAITER_NOT_READY"
        sleep "${POLL_SECONDS}"
    done
    record "V17_WAITER_FIRST_READY"
    sleep "${CONFIRM_SECONDS}"
    if ! ready_once; then
        record "V17_WAITER_CONFIRM_FAILED"
        exit 70
    fi
    record "V17_WAITER_DISPATCH"
    "${PYTHON_BIN}" "${DISPATCHER}" --run-id "${RUN_ID}"
    record "V17_WAITER_TERMINAL"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
