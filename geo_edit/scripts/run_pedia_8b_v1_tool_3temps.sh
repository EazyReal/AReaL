#!/usr/bin/env bash
# ==============================================================================
# Test PEDIA_8B_v1 on 6 geo/map datasets at 3 temperatures (0.0, 0.6, 0.9)
# Tool mode with --enable_tools map general (two categories)
#
# Pipeline:
#   1. Launch vLLM once (DP=8 TP=1 on this node)
#   2. For each temperature: run inference for all 6 datasets sequentially
#      (datasets share the same vLLM endpoint)
#   3. After each temperature's inference completes, fork eval to background
#      (eval is CPU-only via judge API, runs in parallel with next temp's
#       inference)
#   4. Wait for all background evals and print summary
#
# Required env:
#   JUDGE_API_KEY, JUDGE_API_BASE (already set in env)
# ==============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${JUDGE_API_KEY:?JUDGE_API_KEY must be set}"
: "${JUDGE_API_BASE:?JUDGE_API_BASE must be set}"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH="/storage/openpsi/data/lcy_image_edit/pedia_model/PEDIA_8B_v1"
MODEL_NAME="PEDIA_8B_v1"
PORT="${PORT:-8000}"
API_BASE="http://127.0.0.1:${PORT}"
MAX_CONCURRENT="${MAX_CONCURRENT:-32}"
SAMPLE_RATE="${SAMPLE_RATE:-1.0}"
MAX_TOOL_CALLS="${MAX_TOOL_CALLS:-10}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-30}"
DP_SIZE="${DP_SIZE:-8}"
TP_SIZE="${TP_SIZE:-1}"

JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-mini-2025-04-14}"

OUTPUT_ROOT="/storage/openpsi/data/lcy_image_edit/eval_results"
EVAL_OUTPUT_ROOT="/storage/openpsi/data/lcy_image_edit/eval_output"

# 6 datasets requested by the user
DATASETS=(visual_probe_easy visual_probe_medium visual_probe_hard map_trace reason_map reason_map_plus)

declare -A DATASET_PATHS=(
    [visual_probe_easy]="/storage/openpsi/data/VisualProbe_Easy/val.parquet"
    [visual_probe_medium]="/storage/openpsi/data/VisualProbe_Medium/val.parquet"
    [visual_probe_hard]="/storage/openpsi/data/VisualProbe_Hard/val.parquet"
    [map_trace]="/storage/openpsi/data/lcy_image_edit/maptrace_val_2851.parquet"
    [reason_map]="/storage/openpsi/data/ReasonMap/reasonmap_base_validation_dataset.parquet"
    [reason_map_plus]="/storage/openpsi/data/ReasonMap_plus/reasonmap_plus_test.parquet"
)

declare -A DATASET_EVAL_NAMES=(
    [visual_probe_easy]="visual_probe"
    [visual_probe_medium]="visual_probe"
    [visual_probe_hard]="visual_probe"
    [map_trace]="map_trace"
    [reason_map]="reason_map"
    [reason_map_plus]="reason_map_plus"
)

# map_trace uses NDTW; no LLM judge
declare -A SKIP_JUDGE=(
    [map_trace]=1
)

# Temperature -> output suffix
TEMP_VALUES=(0.0 0.6 0.9)
TEMP_SUFFIXES=(t00 t06 t09)

LOG_DIR="/tmp/log/pedia_8b_v1_tool_3temps"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/main.log"; }

# ── Helpers ───────────────────────────────────────────────────────────────────
kill_vllm() {
    log "Killing existing vLLM/api_server processes ..."
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    sleep 5
}

wait_vllm_ready() {
    local max_wait=600 waited=0
    until curl -sf "${API_BASE}/v1/models" > /dev/null 2>&1; do
        sleep 5; waited=$((waited + 5))
        if [ $waited -ge $max_wait ]; then
            log "ERROR: vLLM not ready after ${max_wait}s; dumping tail of api log:"
            tail -50 /tmp/log/vllm_api.log 2>&1 | tee -a "$LOG_DIR/main.log"
            return 1
        fi
    done
    log "vLLM ready after ${waited}s"
    return 0
}

launch_vllm() {
    log "Launching vLLM: model=$MODEL_NAME port=$PORT DP=$DP_SIZE TP=$TP_SIZE gpu_mem=$GPU_MEM_UTIL max_len=$MAX_MODEL_LEN"
    DP_SIZE="$DP_SIZE" TP_SIZE="$TP_SIZE" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" GPU_MEM_UTIL="$GPU_MEM_UTIL" \
    MAX_IMAGES_PER_PROMPT="$MAX_IMAGES_PER_PROMPT" \
        bash "$SCRIPT_DIR/launch_vllm_generate.sh" "$MODEL_PATH" "$PORT" \
        >> "$LOG_DIR/vllm_launch.log" 2>&1 &
    # Don't wait on launcher script; it itself waits for endpoint with a curl loop
    sleep 20
    wait_vllm_ready
}

# Run one (dataset, temperature) inference
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

# Run eval for one (dataset, temperature) in background
launch_eval_one() {
    local ds_key="$1" suffix="$2"
    local ds_name="${DATASET_EVAL_NAMES[$ds_key]}"
    local result_path="${OUTPUT_ROOT}/${ds_key}/${MODEL_NAME}_tool_${suffix}"
    local eval_output="${EVAL_OUTPUT_ROOT}/${ds_key}/${MODEL_NAME}_tool_${suffix}"
    local eval_log="$LOG_DIR/eval_${ds_key}_${suffix}.log"

    if [ ! -d "$result_path" ]; then
        log "  [EVAL] SKIP $ds_key ${suffix}: no inference results at $result_path"
        return
    fi

    log "  [EVAL] launching $ds_key ${suffix} -> $eval_output (bg)"

    local judge_args=""
    if [ "${SKIP_JUDGE[$ds_key]:-0}" != "1" ]; then
        judge_args="--use_judge --judge_model $JUDGE_MODEL --judge_api_key $JUDGE_API_KEY --judge_api_base $JUDGE_API_BASE"
    fi

    (
        python -m geo_edit.evaluation.eval_unified \
            --dataset_name "$ds_name" \
            --result_path "$result_path" \
            --output_path "$eval_output" \
            $judge_args > "$eval_log" 2>&1
        echo $? > "$eval_log.rc"
    ) &
    EVAL_PIDS+=($!)
    EVAL_TAGS+=("${ds_key}_${suffix}")
}

print_summary() {
    log "================================================================"
    log "  BENCHMARK SUMMARY"
    log "  Model:        $MODEL_NAME"
    log "  Path:         $MODEL_PATH"
    log "  Mode:         tool (enable_tools=map general)"
    log "  Temperatures: ${TEMP_VALUES[*]}"
    log "  Judge:        $JUDGE_MODEL"
    log "  Time:         $(date '+%F %T')"
    log "================================================================"

    # header
    printf "%-25s" "Dataset" | tee -a "$LOG_DIR/main.log"
    for suf in "${TEMP_SUFFIXES[@]}"; do
        printf " | %-20s" "$suf" | tee -a "$LOG_DIR/main.log"
    done
    printf "\n" | tee -a "$LOG_DIR/main.log"
    printf -- "-%.0s" {1..95}; echo | tee -a "$LOG_DIR/main.log"

    for ds_key in "${DATASETS[@]}"; do
        printf "%-25s" "$ds_key" | tee -a "$LOG_DIR/main.log"
        for suf in "${TEMP_SUFFIXES[@]}"; do
            local summ="${EVAL_OUTPUT_ROOT}/${ds_key}/${MODEL_NAME}_tool_${suf}/summary.txt"
            local acc="N/A"
            if [ -f "$summ" ]; then
                # Prefer "Accuracy:" for most datasets; for map_trace use Avg NDTW
                if [ "$ds_key" = "map_trace" ]; then
                    acc=$(grep -E "^Avg NDTW:" "$summ" | head -1 | sed 's/Avg NDTW: //')
                    [ -z "$acc" ] && acc=$(grep -E "^Accuracy:" "$summ" | head -1 | sed 's/Accuracy: //')
                    [ -n "$acc" ] && acc="NDTW=$acc"
                else
                    acc=$(grep -E "^Accuracy:" "$summ" | head -1 | sed 's/Accuracy: //')
                fi
                [ -z "$acc" ] && acc="?"
            fi
            printf " | %-20s" "$acc" | tee -a "$LOG_DIR/main.log"
        done
        printf "\n" | tee -a "$LOG_DIR/main.log"
    done
    log "================================================================"
    log "Inference outputs: ${OUTPUT_ROOT}/<ds>/${MODEL_NAME}_tool_t{00,06,09}"
    log "Eval outputs:      ${EVAL_OUTPUT_ROOT}/<ds>/${MODEL_NAME}_tool_t{00,06,09}"
    log "Log dir:           ${LOG_DIR}"
}

# ── Main ──────────────────────────────────────────────────────────────────────
EVAL_PIDS=()
EVAL_TAGS=()

log "================================================================"
log "  PEDIA_8B_v1 tool-mode benchmark (3 temperatures)"
log "================================================================"

SKIP_VLLM="${SKIP_VLLM:-0}"
if [ "$SKIP_VLLM" = "1" ]; then
    log "SKIP_VLLM=1; assuming vLLM already running at ${API_BASE}"
    wait_vllm_ready || { log "FATAL: vLLM not reachable at ${API_BASE}"; exit 1; }
else
    kill_vllm
    launch_vllm || { log "FATAL: vLLM failed to start"; exit 1; }
fi

for i in "${!TEMP_VALUES[@]}"; do
    temp="${TEMP_VALUES[$i]}"
    suffix="${TEMP_SUFFIXES[$i]}"
    log "================================================================"
    log "  Temperature pass [$((i + 1))/${#TEMP_VALUES[@]}]: temp=$temp  suffix=$suffix"
    log "================================================================"

    for ds_key in "${DATASETS[@]}"; do
        if run_inference_one "$ds_key" "$temp" "$suffix"; then
            launch_eval_one "$ds_key" "$suffix"  # CPU-only judge, runs alongside next inference
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
    rc=$?
    log "  [EVAL] $tag finished (pid=$pid rc=$rc)"
done

print_summary

log "Pipeline complete."
exit 0
