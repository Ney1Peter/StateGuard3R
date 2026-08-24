#!/usr/bin/env bash
# Serial Stage 0.6 dispatcher. It refuses to overwrite or retry a response.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu_index="1"
run_id="0001"
dry_run="false"
while (($#)); do
    case "$1" in
        --gpu-index) gpu_index="$2"; shift 2 ;;
        --run-id) run_id="$2"; shift 2 ;;
        --dry-run) dry_run="true"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$gpu_index" =~ ^[0-9]+$ && "$run_id" =~ ^[0-9]{4}$ ]] || { echo "invalid GPU/run ID" >&2; exit 2; }

cd "$root"
input_root="outputs/state-triage-v2-stage0p6-inputs-0001/evaluation"
prefix="outputs/state-triage-v2-stage0p6-eval-"
suffix="-observer-0001"
log="logs/state-triage-v2-stage0p6-pilot-${run_id}.log"
mkdir -p logs tmp
items=(
    normal_novelty-freiburg2_desk-1200
    normal_novelty-freiburg3_walking_static-0030
    normal_novelty-freiburg3_walking_xyz-0020
    registration_or_order_fault-freiburg2_desk-0300
    registration_or_order_fault-freiburg3_walking_static-0150
    registration_or_order_fault-freiburg3_walking_xyz-0400
    bad_observation-freiburg2_desk-2000
    bad_observation-freiburg3_walking_static-0250
    bad_observation-freiburg3_walking_xyz-0470
    transient_local_content-freiburg2_desk-2600
    transient_local_content-freiburg3_walking_static-0580
    transient_local_content-freiburg3_walking_xyz-0750
)
for item in "${items[@]}"; do
    capsule="${input_root}/${item}.json"; output="${prefix}${item}${suffix}"
    [[ -f "$capsule" && ! -w "$capsule" ]] || { echo "invalid immutable capsule: $capsule" >&2; exit 1; }
    [[ ! -e "$output" ]] || { echo "refusing to overwrite/retry: $output" >&2; exit 1; }
done
if [[ "$dry_run" == "true" ]]; then printf '%s\n' "${items[@]}"; exit 0; fi

exec > >(tee -a "$log") 2>&1
echo "[$(date -Is)] Stage0.6 pilot ${run_id} start on physical GPU ${gpu_index}"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits -i "$gpu_index"
for item in "${items[@]}"; do
    capsule="${input_root}/${item}.json"; output="${prefix}${item}${suffix}"
    echo "[$(date -Is)] START ${item}"
    CUDA_VISIBLE_DEVICES="$gpu_index" TMPDIR="$root/tmp" PYTHONUNBUFFERED=1 \
        /data/wangzheng/Project2/baselines/ReCal3R/.venv/bin/python scripts/run_state_triage_v2_stage0.py \
        --baseline-root /data/wangzheng/Project2/baselines/ReCal3R \
        --checkpoint /data/wangzheng/Project2/baselines/ReCal3R/src/cut3r_512_dpt_4_64.pth \
        --input-capsule "$capsule" --output-dir "$output" --device cuda --size 512 --seed 0 --beta-base 0.1 --observer on
    [[ -f "$output/run.json" && ! -w "$output/run.json" ]] || { echo "output was not immutably published: $output" >&2; exit 1; }
    echo "[$(date -Is)] DONE ${item}"
done
touch "logs/state-triage-v2-stage0p6-pilot-${run_id}.done"
echo "[$(date -Is)] ALL_DONE"
