#!/usr/bin/env bash
# Wait without reserving GPU, then release only the preregistered v13 control
# through its terminal-evidence dispatcher.  This wrapper never imports a
# model; all CUDA work remains inside the single token-gated dispatcher child.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
RECAL3R_ROOT="${ROOT}/../baselines/ReCal3R"
PYTHON_BIN="${RECAL3R_ROOT}/.venv/bin/python"
DISPATCHER="${ROOT}/scripts/dispatch_recal3r_selective_state_memory_repair_v13.py"
RUN_ID="recovery-selective-state-memory-repair-v13-dynamic-always-commit-0001"
GPU_UUID="GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250"
MIN_FREE_MIB=12288
POLL_SECONDS=60
CONFIRM_SECONDS=30
WAIT_LOG="${ROOT}/logs/${RUN_ID}-wait.log"
COMMAND_JSON="${ROOT}/tmp/${RUN_ID}-command.json"

record() {
    printf '%s\n' "$1" >> "${WAIT_LOG}"
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

require_clean_tracked_worktrees() {
    if [[ -n "$(git -C "${ROOT}" status --porcelain)" ]]; then
        record "V13_WAIT_ABORT run_id=${RUN_ID} reason=dirty_stateguard at=$(date --iso-8601=seconds)"
        return 1
    fi
    if [[ -n "$(git -C "${RECAL3R_ROOT}" status --porcelain)" ]]; then
        record "V13_WAIT_ABORT run_id=${RUN_ID} reason=dirty_recal3r at=$(date --iso-8601=seconds)"
        return 1
    fi
    # An explicit successful return is essential: a false conditional's
    # status otherwise becomes the function status under `set -e`.
    return 0
}

write_one_use_command_json() {
    if [[ -e "${COMMAND_JSON}" || -L "${COMMAND_JSON}" ]]; then
        record "V13_WAIT_ABORT run_id=${RUN_ID} reason=command_json_already_exists at=$(date --iso-8601=seconds)"
        return 1
    fi
    ROOT="${ROOT}" RECAL3R_ROOT="${RECAL3R_ROOT}" RUN_ID="${RUN_ID}" COMMAND_JSON="${COMMAND_JSON}" \
        "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
recal3r = Path(os.environ["RECAL3R_ROOT"])
run_id = os.environ["RUN_ID"]
target = Path(os.environ["COMMAND_JSON"])
argv = [
    str(recal3r / ".venv" / "bin" / "python"),
    str(root / "scripts" / "run_recal3r_selective_state_memory_repair_v13.py"),
    "--baseline-root", str(recal3r),
    "--checkpoint", str(recal3r / "src" / "cut3r_512_dpt_4_64.pth"),
    "--checkpoint-sha256", "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103",
    "--input-manifest", str(root / "outputs" / "formal-v1-inputs-0001" / "development" / "development-dynamic" / "input-manifest.json"),
    "--output-dir", str(root / "outputs" / run_id),
    "--device", "cuda",
    "--size", "512",
    "--seed", "0",
    "--beta-base", "0.1",
    "--health-profile", "v3",
    "--rgb-timestamp-listing", str(recal3r / "data" / "tum" / "rgbd_dataset_freiburg1_desk" / "rgb.txt"),
    "--timestamp-dataset-root", str(recal3r / "data" / "tum" / "rgbd_dataset_freiburg1_desk"),
    "--state-policy", "always-commit",
    "--detector-config", str(root / "outputs" / "formal-v3-calibration-0001" / "formal-config.json"),
    "--watchdog", "8",
]
with target.open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(argv, ensure_ascii=False, separators=(",", ":")) + "\n")
PY
}

main() {
record "V13_WAIT_START run_id=${RUN_ID} min_free_mib=${MIN_FREE_MIB} at=$(date --iso-8601=seconds)"
while true; do
    if first="$(gpu_free_mib)" && [[ "${first}" =~ ^[0-9]+$ ]] && (( first >= MIN_FREE_MIB )); then
        sleep "${CONFIRM_SECONDS}"
        if second="$(gpu_free_mib)" && [[ "${second}" =~ ^[0-9]+$ ]] && (( second >= MIN_FREE_MIB )); then
            record "V13_WAIT_EXTERNAL_GATE_PASS run_id=${RUN_ID} first_free_mib=${first} second_free_mib=${second} at=$(date --iso-8601=seconds)"
            require_clean_tracked_worktrees
            write_one_use_command_json
            record "V13_WAIT_DISPATCH run_id=${RUN_ID} at=$(date --iso-8601=seconds)"
            exec "${PYTHON_BIN}" "${DISPATCHER}" --run-id "${RUN_ID}" --command-json "${COMMAND_JSON}" --timeout-seconds 7200
        fi
        record "V13_WAIT_CONFIRM_REJECTED run_id=${RUN_ID} first_free_mib=${first} second_free_mib=${second:-unavailable} at=$(date --iso-8601=seconds)"
    else
        record "V13_WAIT_POLL run_id=${RUN_ID} free_mib=${first:-unavailable} at=$(date --iso-8601=seconds)"
    fi
    unset first second
    sleep "${POLL_SECONDS}"
done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
