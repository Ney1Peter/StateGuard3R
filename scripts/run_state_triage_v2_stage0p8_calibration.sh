#!/usr/bin/env bash
# Serial dispatcher for the sealed 12-capsule Stage 0.8 calibration only.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu_index=""
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
[[ "$run_id" =~ ^[0-9]{4}$ ]] || { echo "invalid run ID" >&2; exit 2; }
[[ "$dry_run" == "true" || "$gpu_index" =~ ^[0-9]+$ ]] || { echo "--gpu-index is required unless --dry-run" >&2; exit 2; }

cd "$root"
protocol="docs/protocols/state-triage-v2-stage0p8-calibration-gate-b.json"
[[ -f "$protocol" && ! -w "$protocol" ]] || { echo "missing immutable calibration protocol" >&2; exit 1; }
mkdir -p logs tmp
mapfile -t records < <(.venv/bin/python -c '
import json
from pathlib import Path
p = Path("docs/protocols/state-triage-v2-stage0p8-calibration-gate-b.json")
value = json.loads(p.read_text(encoding="utf-8"))
items = value["calibration_inventory"]
assert len(items) == 12 and len({item["output"] for item in items}) == 12
for item in items:
    print(item["capsule_path"] + "\t" + item["output"])
')
[[ ${#records[@]} -eq 12 ]] || { echo "frozen protocol did not yield 12 records" >&2; exit 1; }
for record in "${records[@]}"; do
    IFS=$'\t' read -r capsule output <<< "$record"
    [[ -f "$capsule" && ! -w "$capsule" ]] || { echo "invalid immutable capsule: $capsule" >&2; exit 1; }
    [[ ! -e "outputs/$output" ]] || { echo "refusing to overwrite/retry: outputs/$output" >&2; exit 1; }
done
if [[ "$dry_run" == "true" ]]; then
    printf '%s\n' "${records[@]}"
    exit 0
fi

log="logs/state-triage-v2-stage0p8-calibration-${run_id}.log"
exec > >(tee -a "$log") 2>&1
echo "[$(date -Is)] Stage0.8 calibration ${run_id} start on physical GPU ${gpu_index}"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits -i "$gpu_index"
index=0
for record in "${records[@]}"; do
    index=$((index + 1))
    IFS=$'\t' read -r capsule output <<< "$record"
    echo "[$(date -Is)] START ${index}/12 ${output}"
    CUDA_VISIBLE_DEVICES="$gpu_index" TMPDIR="$root/tmp" PYTHONUNBUFFERED=1 \
        /data/wangzheng/Project2/baselines/ReCal3R/.venv/bin/python scripts/run_state_triage_v2_stage0.py \
        --baseline-root /data/wangzheng/Project2/baselines/ReCal3R \
        --checkpoint /data/wangzheng/Project2/baselines/ReCal3R/src/cut3r_512_dpt_4_64.pth \
        --input-capsule "$capsule" --output-dir "outputs/$output" --device cuda --size 512 --seed 0 --beta-base 0.1 --observer on
    [[ -f "outputs/$output/run.json" && ! -w "outputs/$output/run.json" ]] || { echo "output was not immutably published: $output" >&2; exit 1; }
    echo "[$(date -Is)] DONE ${index}/12 ${output}"
done
touch "logs/state-triage-v2-stage0p8-calibration-${run_id}.done"
echo "[$(date -Is)] ALL_DONE 12/12"
