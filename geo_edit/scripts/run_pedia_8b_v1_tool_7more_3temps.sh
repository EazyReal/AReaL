#!/usr/bin/env bash
# ==============================================================================
# Test PEDIA_8B_v1 on 7 additional benchmarks at 3 temperatures (0.0, 0.6, 0.9)
# Tool mode with --enable_tools map general (two categories)
#
# Datasets:
#   mapeval_visual         (400 rows)
#   babyvision             (388 rows)
#   visworld_ballgame      (1024 rows)   -- Ball Tracing
#   visworld_paperfolding  (480 rows)    -- Paper Folding
#   visworld_cube          (480 rows)    -- Cube 3-View Reasoning
#   3dsrbench              (5157 rows)   -- Real-world Spatial Reasoning
#   vstar_bench            (191 rows)    -- VStar
#
# Pipeline: per-dataset inference -> per-dataset bg eval (judge API parallel)
#
# Required env: JUDGE_API_KEY, JUDGE_API_BASE  (vLLM assumed already running)
# ==============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${JUDGE_API_KEY:?JUDGE_API_KEY must be set}"
: "${JUDGE_API_BASE:?JUDGE_API_BASE must be set}"

MODEL_PATH="/storage/openpsi/data/lcy_image_edit/pedia_model/PEDIA_8B_v1"
MODEL_NAME="PEDIA_8B_v1"
PORT="${PORT:-8000}"
API_BASE="http://127.0.0.1:${PORT}"
MAX_CONCURRENT="${MAX_CONCURRENT:-64}"
SAMPLE_RATE="${SAMPLE_RATE:-1.0}"
MAX_TOOL_CALLS="${MAX_TOOL_CALLS:-10}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-mini-2025-04-14}"

OUTPUT_ROOT="/storage/openpsi/data/lcy_image_edit/eval_results"
EVAL_OUTPUT_ROOT="/storage/openpsi/data/lcy_image_edit/eval_output"

DATASETS=(mapeval_visual babyvision visworld_ballgame visworld_paperfolding visworld_cube 3dsrbench vstar_bench)

declare -A DATASET_PATHS=(
    [mapeval_visual]="/storage/openpsi/data/lcy_image_edit/mapeval_visual.parquet"
    [babyvision]="/storage/openpsi/data/lcy_image_edit/BabyVision/data/train-00000-of-00001.parquet"
    [visworld_ballgame]="/storage/openpsi/data/lcy_image_edit/VisWorld-Eval/ballgame/ballgame.parquet"
    [visworld_paperfolding]="/storage/openpsi/data/lcy_image_edit/VisWorld-Eval/paperfolding/paperfolding.parquet"
    [visworld_cube]="/storage/openpsi/data/lcy_image_edit/VisWorld-Eval/cube/cube.parquet"
    [3dsrbench]="/storage/openpsi/data/lcy_image_edit/3DSRBench/3dsrbench.parquet"
    [vstar_bench]="/storage/openpsi/data/Vstar_Bench/vstar_bench.parquet"
)

declare -A DATASET_EVAL_NAMES=(
    [mapeval_visual]="mapeval_visual"
    [babyvision]="babyvision"
    [visworld_ballgame]="visworld_eval"
    [visworld_paperfolding]="visworld_eval"
    [visworld_cube]="visworld_eval"
    [3dsrbench]="3dsrbench"
    [vstar_bench]="vstar_bench"
)

TEMP_VALUES=(0.0 0.6 0.9)
TEMP_SUFFIXES=(t00 t06 t09)

LOG_DIR="/tmp/log/pedia_8b_v1_tool_7more_3temps"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/main.log"; }

wait_vllm_ready() {
    local max_wait=120 waited=0
    until curl -sf "${API_BASE}/v1/models" > /dev/null 2>&1; do
        sleep 3; waited=$((waited + 3))
        if [ $waited -ge $max_wait ]; then
            log "ERROR: vLLM not reachable at ${API_BASE}"; return 1
        fi
    done
    log "vLLM ready"
}

run_inference_one() {
    local ds_key="$1" temp="$2" suffix="$3"
    local ds_path="${DATASET_PATHS[$ds_key]}"
    local ds_name="${DATASET_EVAL_NAMES[$ds_key]}"
    local output_dir="${OUTPUT_ROOT}/${ds_key}/${MODEL_NAME}_tool_${suffix}"
    local inf_log="$LOG_DIR/inference_${ds_key}_${suffix}.log"

    if [ ! -f "$ds_path" ]; then
        log "  SKIP $ds_key ($suffix): dataset file not found: $ds_path"
        return 1
    fi

    log "  [INF] $ds_key ${suffix} (temp=$temp) -> $output_dir"

    python -m geo_edit.scripts.async_generate_with_tool_call_api \
        --api_base "$API_BASE" \
        --dataset_path "$ds_path" \
        --dataset_name "$ds_name" \
        --output_dir "$output_dir" \
        --model_name_or_path "$MODEL_PATH" \
        --model_type vLLM \
        --sample_rate "$SAMPLE_RATE" \
        --use_tools auto \
        --max_concurrent_requests "$MAX_CONCURRENT" \
        --max_tool_calls "$MAX_TOOL_CALLS" \
        --enable_tools map general \
        --no_image_compression \
        --temperature "$temp" \
        > "$inf_log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        log "  [INF] FAILED $ds_key ${suffix} (rc=$rc); see $inf_log"
        return $rc
    fi
    log "  [INF] DONE $ds_key ${suffix}"
    return 0
}

launch_eval_one() {
    local ds_key="$1" suffix="$2"
    local ds_name="${DATASET_EVAL_NAMES[$ds_key]}"
    local result_path="${OUTPUT_ROOT}/${ds_key}/${MODEL_NAME}_tool_${suffix}"
    local eval_output="${EVAL_OUTPUT_ROOT}/${ds_key}/${MODEL_NAME}_tool_${suffix}"
    local eval_log="$LOG_DIR/eval_${ds_key}_${suffix}.log"

    [ ! -d "$result_path" ] && { log "  [EVAL] SKIP $ds_key ${suffix}: no results"; return; }

    log "  [EVAL] launching $ds_key ${suffix} -> $eval_output (bg)"
    (
        python -m geo_edit.evaluation.eval_unified \
            --dataset_name "$ds_name" \
            --result_path "$result_path" \
            --output_path "$eval_output" \
            --use_judge --judge_model "$JUDGE_MODEL" \
            --judge_api_key "$JUDGE_API_KEY" \
            --judge_api_base "$JUDGE_API_BASE" \
            > "$eval_log" 2>&1
        echo $? > "$eval_log.rc"
    ) &
    EVAL_PIDS+=($!)
    EVAL_TAGS+=("${ds_key}_${suffix}")
}

print_summary() {
    log "================================================================"
    log "  BENCHMARK SUMMARY -- 7 additional datasets"
    log "  Model: $MODEL_NAME    Temperatures: ${TEMP_VALUES[*]}    Judge: $JUDGE_MODEL"
    log "================================================================"
    printf "%-25s" "Dataset" | tee -a "$LOG_DIR/main.log"
    for suf in "${TEMP_SUFFIXES[@]}"; do
        printf " | %-20s" "$suf" | tee -a "$LOG_DIR/main.log"
    done
    printf "\n" | tee -a "$LOG_DIR/main.log"
    for ds_key in "${DATASETS[@]}"; do
        printf "%-25s" "$ds_key" | tee -a "$LOG_DIR/main.log"
        for suf in "${TEMP_SUFFIXES[@]}"; do
            local s="${EVAL_OUTPUT_ROOT}/${ds_key}/${MODEL_NAME}_tool_${suf}/summary.txt"
            local acc="N/A"
            [ -f "$s" ] && acc=$(grep -E "^Accuracy:" "$s" | head -1 | sed 's/Accuracy: //')
            [ -z "$acc" ] && acc="?"
            printf " | %-20s" "$acc" | tee -a "$LOG_DIR/main.log"
        done
        printf "\n" | tee -a "$LOG_DIR/main.log"
    done
    log "================================================================"
}

EVAL_PIDS=()
EVAL_TAGS=()

log "================================================================"
log "  PEDIA_8B_v1 tool-mode 7-additional-datasets benchmark (3 temperatures)"
log "================================================================"

wait_vllm_ready || { log "FATAL: vLLM not reachable"; exit 1; }

for i in "${!TEMP_VALUES[@]}"; do
    temp="${TEMP_VALUES[$i]}"
    suffix="${TEMP_SUFFIXES[$i]}"
    log "================================================================"
    log "  Temperature pass [$((i + 1))/${#TEMP_VALUES[@]}]: temp=$temp  suffix=$suffix"
    log "================================================================"
    for ds_key in "${DATASETS[@]}"; do
        if run_inference_one "$ds_key" "$temp" "$suffix"; then
            launch_eval_one "$ds_key" "$suffix"
        fi
    done
    log "  Temperature pass for temp=$temp done; eval running in background ..."
done

log "================================================================"
log "  All inference done. Waiting for ${#EVAL_PIDS[@]} background eval jobs ..."
log "================================================================"

for i in "${!EVAL_PIDS[@]}"; do
    pid="${EVAL_PIDS[$i]}"
    tag="${EVAL_TAGS[$i]}"
    wait "$pid"
    log "  [EVAL] $tag finished (pid=$pid rc=$?)"
done

print_summary
log "Pipeline complete."
exit 0
