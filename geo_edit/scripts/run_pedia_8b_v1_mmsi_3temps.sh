#!/usr/bin/env bash
# Run PEDIA_8B_v1 on visworld_mmsi (Real-world Spatial Reasoning) at 3 temperatures
set -uo pipefail

cd /storage/openpsi/users/lichangye.lcy/antoinegg1/AReaL

: "${JUDGE_API_KEY:?}"
: "${JUDGE_API_BASE:?}"

MODEL_PATH="/storage/openpsi/data/lcy_image_edit/pedia_model/PEDIA_8B_v1"
MODEL_NAME="PEDIA_8B_v1"
API_BASE="http://127.0.0.1:8000"
DS_PATH="/storage/openpsi/data/lcy_image_edit/VisWorld-Eval/mmsi/mmsi.parquet"
OUT_ROOT="/storage/openpsi/data/lcy_image_edit/eval_results/visworld_mmsi"
EVAL_ROOT="/storage/openpsi/data/lcy_image_edit/eval_output/visworld_mmsi"
LOG_DIR="/tmp/log/pedia_8b_v1_tool_7more_3temps"
mkdir -p "$LOG_DIR"

TEMPS=(0.0 0.6 0.9)
SUFS=(t00 t06 t09)
EVAL_PIDS=()

for i in "${!TEMPS[@]}"; do
    temp="${TEMPS[$i]}"
    suf="${SUFS[$i]}"
    out_dir="${OUT_ROOT}/${MODEL_NAME}_tool_${suf}"
    inf_log="$LOG_DIR/inference_visworld_mmsi_${suf}.log"
    eval_log="$LOG_DIR/eval_visworld_mmsi_${suf}.log"

    echo "[$(date '+%F %T')] [INF] visworld_mmsi ${suf} (temp=$temp) -> $out_dir"
    python -m geo_edit.scripts.async_generate_with_tool_call_api \
        --api_base "$API_BASE" \
        --dataset_path "$DS_PATH" \
        --dataset_name visworld_eval \
        --output_dir "$out_dir" \
        --model_name_or_path "$MODEL_PATH" \
        --model_type vLLM \
        --sample_rate 1.0 \
        --use_tools auto \
        --max_concurrent_requests 64 \
        --max_tool_calls 10 \
        --enable_tools map general \
        --no_image_compression \
        --temperature "$temp" \
        > "$inf_log" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] [INF] visworld_mmsi ${suf} rc=$rc"
    if [ $rc -eq 0 ]; then
        (python -m geo_edit.evaluation.eval_unified \
            --dataset_name visworld_eval \
            --result_path "$out_dir" \
            --output_path "${EVAL_ROOT}/${MODEL_NAME}_tool_${suf}" \
            --use_judge --judge_model gpt-4.1-mini-2025-04-14 \
            --judge_api_key "$JUDGE_API_KEY" --judge_api_base "$JUDGE_API_BASE" \
            > "$eval_log" 2>&1) &
        EVAL_PIDS+=($!)
        echo "[$(date '+%F %T')] [EVAL] launched visworld_mmsi ${suf} bg pid=${EVAL_PIDS[-1]}"
    fi
done

echo "[$(date '+%F %T')] Waiting for ${#EVAL_PIDS[@]} eval jobs ..."
for pid in "${EVAL_PIDS[@]}"; do wait "$pid"; echo "  eval pid=$pid rc=$?"; done

echo "[$(date '+%F %T')] visworld_mmsi pipeline complete."
for suf in "${SUFS[@]}"; do
    s="${EVAL_ROOT}/${MODEL_NAME}_tool_${suf}/summary.txt"
    [ -f "$s" ] && { printf "visworld_mmsi %s " "$suf"; grep -E "^Accuracy:" "$s" | head -1; }
done
