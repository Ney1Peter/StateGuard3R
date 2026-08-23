#!/usr/bin/env bash
# Serial, pre-registered Stage 0.5 forward dispatcher.  This script contains
# no decision logic and never overwrites/retries an existing response.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu_index="3"
run_id="0002"
dry_run="false"

while (($#)); do
    case "$1" in
        --gpu-index) gpu_index="$2"; shift 2 ;;
        --run-id) run_id="$2"; shift 2 ;;
        --dry-run) dry_run="true"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ ! "$gpu_index" =~ ^[0-9]+$ ]] || [[ ! "$run_id" =~ ^[0-9]{4}$ ]]; then
    echo "gpu index must be numeric and run ID must be four digits" >&2
    exit 2
fi

cd "$root"
input_root="outputs/state-triage-v2-stage0p5-inputs-0001/evaluation"
output_prefix="outputs/state-triage-v2-stage0p5-eval-"
output_suffix="-observer-0001"
log="logs/state-triage-v2-stage0p5-pilot-${run_id}.log"
mkdir -p logs tmp

items=(
    normal_novelty-freiburg2_desk-0520
    normal_novelty-freiburg3_walking_static-0330
    normal_novelty-freiburg3_walking_xyz-0100
    registration_or_order_fault-freiburg2_desk-0900
    registration_or_order_fault-freiburg3_walking_static-0550
    registration_or_order_fault-freiburg3_walking_xyz-0250
    bad_observation-freiburg2_desk-1600
    bad_observation-freiburg3_walking_static-0220
    bad_observation-freiburg3_walking_xyz-0430
    transient_local_content-freiburg2_desk-2350
    transient_local_content-freiburg3_walking_static-0700
    transient_local_content-freiburg3_walking_xyz-0600
)

for item in "${items[@]}"; do
    capsule="${input_root}/${item}.json"
    output="${output_prefix}${item}${output_suffix}"
    [[ -f "$capsule" && ! -w "$capsule" ]] || { echo "invalid immutable capsule: $capsule" >&2; exit 1; }
    [[ ! -e "$output" ]] || { echo "refusing to overwrite or retry: $output" >&2; exit 1; }
done

if [[ "$dry_run" == "true" ]]; then
    printf '%s\n' "${items[@]}"
    exit 0
fi

exec > >(tee -a "$log") 2>&1
echo "[$(date -Is)] Stage0.5 pilot ${run_id} start on physical GPU ${gpu_index}"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits -i "$gpu_index"

for item in "${items[@]}"; do
    capsule="${input_root}/${item}.json"
    output="${output_prefix}${item}${output_suffix}"
    echo "[$(date -Is)] START ${item}"
    CUDA_VISIBLE_DEVICES="$gpu_index" \
    TMPDIR="$root/tmp" \
    PYTHONUNBUFFERED=1 \
    /data/wangzheng/Project2/baselines/ReCal3R/.venv/bin/python \
    scripts/run_state_triage_v2_stage0.py \
        --baseline-root /data/wangzheng/Project2/baselines/ReCal3R \
        --checkpoint /data/wangzheng/Project2/baselines/ReCal3R/src/cut3r_512_dpt_4_64.pth \
        --input-capsule "$capsule" \
        --output-dir "$output" \
        --device cuda --size 512 --seed 0 --beta-base 0.1 --observer on
    [[ -f "$output/run.json" && ! -w "$output/run.json" ]] || { echo "run did not publish immutable output: $output" >&2; exit 1; }
    echo "[$(date -Is)] DONE ${item}"
done

touch "logs/state-triage-v2-stage0p5-pilot-${run_id}.done"
echo "[$(date -Is)] ALL_DONE"
