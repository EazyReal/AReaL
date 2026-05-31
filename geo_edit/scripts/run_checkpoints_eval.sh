#!/usr/bin/env bash
# ==============================================================================
# Multi-checkpoint benchmark pipeline (tool mode, temperature=0).
#
# For each checkpoint in CHECKPOINT_STEPS:
#   1. Kill any running vLLM / GPU processes
#   2. Launch vLLM via geo_edit/scripts/launch_vllm_generate.sh
#   3. Loop over 7 datasets sequentially:
#        a. Run async tool-call inference (GPU-bound, blocks)
#        b. As soon as inference finishes, fire eval_unified in BACKGROUND
#           (CPU + judge-API only - does not compete with the next inference)
#        c. Move on to next dataset's inference while previous eval is still running
#      Datasets:
#       (reason_map_plus, reason_map, visual_probe_easy/medium/hard,
#        map_trace, omnispatial)
#   4. After all 7 inferences finish, wait for all background eval PIDs.
#   5. Parse per-dataset accuracy from eval log files.
# After all checkpoints: print combined summary table to stdout + summary file.
#
# Tools enabled (matches training config): --enable_tools map general
#   This expands (order-invariant set union) to 16 unique tools:
#     map     -> text_spotting, map_text_ocr
#     general -> image_crop, image_label, draw_line, draw_path, bounding_box,
#                image_highlight, text_ocr, auto_segment, bbox_segment,
#                text_segment, exemplar_segment, concept_count, presence_check,
#                grounding_dino
#   Requires a Ray cluster with `tool_agent` actors (PaddleOCR + SAM3 +
#   GroundingDINO) reachable from this process.
#
# Output naming pattern (unique per checkpoint + temperature):
#   ${RUN_NAME}_step${N}_tool_temp0
#   e.g. mixed-atgigpo-v2-0528_step357_tool_temp0
#
# Environment (with defaults):
#   RUN_NAME            mixed-atgigpo-v2-0528
#   CKPT_ROOT           /storage/openpsi/data/lcy_image_edit/mixed_rl_v2/checkpoints
#   CHECKPOINT_STEPS    "340 350 357"   (space-separated)
#   DATASETS            "reason_map_plus reason_map visual_probe_easy visual_probe_medium visual_probe_hard map_trace omnispatial"
#   PORT                8000
#   GPU_MEM_UTIL        0.7    (RL checkpoints have 8 shards -> use 0.7 to avoid OOM)
#   DP_SIZE             8
#   TP_SIZE             1
#   MAX_MODEL_LEN       65536
#   MAX_IMAGES_PER_PROMPT  5
#   MAX_CONCURRENT      64
#   MAX_TOOL_CALLS      10     (matches training max_turns=10)
#   TEMPERATURE         0      (greedy)
#   SAMPLE_RATE         1.0
#   NO_IMAGE_COMPRESSION  1    (matches training)
#   JUDGE_MODEL         gpt-4.1-mini-2025-04-14
#   JUDGE_API_KEY / JUDGE_API_BASE  required (from env)
#   OUTPUT_ROOT         /storage/openpsi/data/lcy_image_edit/eval_results
#   EVAL_OUTPUT_ROOT    /storage/openpsi/data/lcy_image_edit/eval_output
#   LOG_DIR             /storage/openpsi/data/lcy_image_edit/eval_logs/<RUN_NAME>_temp0
#
# Usage:
#   JUDGE_API_KEY=... JUDGE_API_BASE=... \
#       bash geo_edit/scripts/run_checkpoints_eval.sh
#
#   # Only 1 step + 1 dataset (smoke test)
#   CHECKPOINT_STEPS="357" DATASETS="reason_map_plus" \
#       bash geo_edit/scripts/run_checkpoints_eval.sh
# ==============================================================================
set -uo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
RUN_NAME="${RUN_NAME:-mixed-atgigpo-v2-0528}"
CKPT_ROOT="${CKPT_ROOT:-/storage/openpsi/data/lcy_image_edit/mixed_rl_v2/checkpoints}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-340 350 357}"
DATASETS="${DATASETS:-reason_map_plus reason_map visual_probe_easy visual_probe_medium visual_probe_hard map_trace omnispatial}"

PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.7}"
DP_SIZE="${DP_SIZE:-8}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-5}"

MAX_CONCURRENT="${MAX_CONCURRENT:-64}"
MAX_TOOL_CALLS="${MAX_TOOL_CALLS:-10}"
TEMPERATURE="${TEMPERATURE:-0}"
SAMPLE_RATE="${SAMPLE_RATE:-1.0}"
NO_IMAGE_COMPRESSION="${NO_IMAGE_COMPRESSION:-1}"

JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-mini-2025-04-14}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"
JUDGE_API_BASE="${JUDGE_API_BASE:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/storage/openpsi/data/lcy_image_edit/eval_results}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-/storage/openpsi/data/lcy_image_edit/eval_output}"
LOG_DIR="${LOG_DIR:-/storage/openpsi/data/lcy_image_edit/eval_logs/${RUN_NAME}_temp${TEMPERATURE}}"
SUMMARY_FILE="${LOG_DIR}/summary.txt"
RESULTS_CSV="${LOG_DIR}/results.csv"

API_BASE="http://127.0.0.1:${PORT}"

mkdir -p "$LOG_DIR"

# ── Dataset registry ──────────────────────────────────────────────────────────
declare -A DATASET_PATHS=(
    ["visual_probe_easy"]="/storage/openpsi/data/VisualProbe_Easy/val.parquet"
    ["visual_probe_medium"]="/storage/openpsi/data/VisualProbe_Medium/val.parquet"
    ["visual_probe_hard"]="/storage/openpsi/data/VisualProbe_Hard/val.parquet"
    ["map_trace"]="/storage/openpsi/data/lcy_image_edit/maptrace_val_2851.parquet"
    ["reason_map"]="/storage/openpsi/data/ReasonMap/reasonmap_base_validation_dataset.parquet"
    ["reason_map_plus"]="/storage/openpsi/data/ReasonMap_plus/reasonmap_plus_test.parquet"
    ["omnispatial"]="/storage/openpsi/data/lcy_image_edit/OmniSpatial/omnispatial.parquet"
    ["3dsrbench"]="/storage/openpsi/data/lcy_image_edit/3DSRBench/3dsrbench.parquet"
    ["babyvision"]="/storage/openpsi/data/lcy_image_edit/BabyVision/data/train-00000-of-00001.parquet"
    ["mapeval_visual"]="/storage/openpsi/data/lcy_image_edit/mapeval_visual.parquet"
    ["vstar_bench"]="/storage/openpsi/data/Vstar_Bench/vstar_bench.parquet"
    ["visworld_ballgame"]="/storage/openpsi/data/lcy_image_edit/VisWorld-Eval/ballgame/ballgame.parquet"
    ["visworld_cube"]="/storage/openpsi/data/lcy_image_edit/VisWorld-Eval/cube/cube.parquet"
    ["visworld_paperfolding"]="/storage/openpsi/data/lcy_image_edit/VisWorld-Eval/paperfolding/paperfolding.parquet"
    ["visworld_mmsi"]="/storage/openpsi/data/lcy_image_edit/VisWorld-Eval/mmsi/mmsi.parquet"
)

declare -A DATASET_EVAL_NAMES=(
    ["visual_probe_easy"]="visual_probe"
    ["visual_probe_medium"]="visual_probe"
    ["visual_probe_hard"]="visual_probe"
    ["map_trace"]="map_trace"
    ["reason_map"]="reason_map"
    ["reason_map_plus"]="reason_map_plus"
    ["omnispatial"]="omnispatial"
    ["3dsrbench"]="3dsrbench"
    ["babyvision"]="babyvision"
    ["mapeval_visual"]="mapeval_visual"
    ["vstar_bench"]="vstar_bench"
    ["visworld_ballgame"]="visworld_eval"
    ["visworld_cube"]="visworld_eval"
    ["visworld_paperfolding"]="visworld_eval"
    ["visworld_mmsi"]="visworld_eval"
)

# map_trace uses NDTW only - no judge needed
declare -A SKIP_JUDGE=(
    ["map_trace"]=1
)

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/run.log"; }

kill_vllm() {
    log "Killing existing vLLM/GPU processes..."
    ps aux | grep -E "vllm|api_server" | grep -v grep \
        | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    sleep 2
    # second pass for stragglers + any leftover Python on GPUs not owned by ray workers
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | sort -u | while read -r pid; do
            # Only kill the PID if its process command line contains "vllm" or "api_server".
            # Avoid nuking Ray actors / tool_agent processes that may share GPU.
            if [ -n "$pid" ] && [ -f "/proc/$pid/cmdline" ]; then
                cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
                if echo "$cmd" | grep -qE "vllm|api_server"; then
                    kill -9 "$pid" 2>/dev/null || true
                fi
            fi
        done
    sleep 3
}

wait_vllm() {
    local max_wait=600 waited=0
    until curl -sf "${API_BASE}/v1/models" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [ $waited -ge $max_wait ]; then
            log "ERROR: vLLM not ready after ${max_wait}s"
            tail -n 50 /tmp/log/vllm_api.log 2>/dev/null | tee -a "$LOG_DIR/run.log"
            return 1
        fi
    done
    log "vLLM ready after ${waited}s"
    return 0
}

# Extract Accuracy from eval_unified stdout. Falls back to Avg NDTW for map_trace.
parse_accuracy() {
    local txt="$1"
    local acc
    acc=$(echo "$txt" | grep -E "^Accuracy:" | tail -1 | awk -F: '{print $2}' | xargs)
    if [ -z "$acc" ]; then
        acc=$(echo "$txt" | grep -E "^Avg NDTW:" | tail -1 | awk -F: '{print $2}' | xargs)
    fi
    echo "${acc:-N/A}"
}

# ── Pre-flight ────────────────────────────────────────────────────────────────
if [ -z "$JUDGE_API_KEY" ] || [ -z "$JUDGE_API_BASE" ]; then
    log "WARNING: JUDGE_API_KEY or JUDGE_API_BASE not set; judge will be skipped."
fi

# Print run plan
log "=========================================================================="
log " Multi-checkpoint benchmark"
log "   Run name      : $RUN_NAME"
log "   Checkpoints   : $CHECKPOINT_STEPS"
log "   Datasets      : $DATASETS"
log "   Temperature   : $TEMPERATURE"
log "   Tool mode     : auto, max_tool_calls=$MAX_TOOL_CALLS"
log "   vLLM port     : $PORT (DP=$DP_SIZE TP=$TP_SIZE mem=$GPU_MEM_UTIL)"
log "   Judge         : $JUDGE_MODEL @ ${JUDGE_API_BASE:-<unset>}"
log "   Logs          : $LOG_DIR"
log "=========================================================================="

# Initialize CSV header
{
    echo -n "step"
    for ds_key in $DATASETS; do
        echo -n ",${ds_key}"
    done
    echo
} > "$RESULTS_CSV"

# Per-checkpoint results storage (associative array of "step,ds" -> acc)
declare -A RESULTS
declare -A FAILED_DS

# ── Main loop over checkpoints ───────────────────────────────────────────────
for step in $CHECKPOINT_STEPS; do
    CKPT_HF_PATH="${CKPT_ROOT}/${RUN_NAME}/global_step_${step}/actor/huggingface"
    MODEL_NAME="${RUN_NAME}_step${step}_tool_temp${TEMPERATURE}"

    log ""
    log "##########################################################################"
    log "# Checkpoint step $step"
    log "#   HF path     : $CKPT_HF_PATH"
    log "#   Model name  : $MODEL_NAME"
    log "##########################################################################"

    if [ ! -d "$CKPT_HF_PATH" ] || [ ! -f "$CKPT_HF_PATH/config.json" ]; then
        log "ERROR: checkpoint not found or missing config.json at $CKPT_HF_PATH"
        log "       skipping step $step"
        continue
    fi

    # Stage 1: kill old vLLM and launch new
    kill_vllm
    log "Launching vLLM for step $step ..."
    GPU_MEM_UTIL="$GPU_MEM_UTIL" \
    DP_SIZE="$DP_SIZE" \
    TP_SIZE="$TP_SIZE" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    MAX_IMAGES_PER_PROMPT="$MAX_IMAGES_PER_PROMPT" \
        bash geo_edit/scripts/launch_vllm_generate.sh "$CKPT_HF_PATH" "$PORT" \
        2>&1 | tee -a "$LOG_DIR/vllm_step${step}.log" >/dev/null &

    if ! wait_vllm; then
        log "ERROR: vLLM failed to start for step $step; skipping checkpoint."
        kill_vllm
        continue
    fi

    # Stage 2+3: inference (foreground, GPU) + eval (background, CPU + judge API)
    # As each dataset's inference finishes, eval fires in background while the
    # next dataset's inference runs. After the loop, we wait for all eval PIDs.
    COMPRESS_FLAG=""
    [ -n "$NO_IMAGE_COMPRESSION" ] && COMPRESS_FLAG="--no_image_compression"

    log "=== Inference + parallel eval (step $step) ==="
    total_ds=$(echo "$DATASETS" | wc -w)
    idx=0
    declare -A EVAL_PIDS=()
    declare -A EVAL_LOGS=()
    declare -A INFERENCE_OK=()

    for ds_key in $DATASETS; do
        idx=$((idx + 1))
        ds_path="${DATASET_PATHS[$ds_key]:-}"
        ds_name="${DATASET_EVAL_NAMES[$ds_key]:-}"
        output_dir="${OUTPUT_ROOT}/${ds_key}/${MODEL_NAME}"
        inf_log="${LOG_DIR}/inf_step${step}_${ds_key}.log"
        eval_log="${LOG_DIR}/eval_step${step}_${ds_key}.log"
        eval_output="${EVAL_OUTPUT_ROOT}/${ds_key}/${MODEL_NAME}"

        if [ -z "$ds_path" ]; then
            log "  [$idx/$total_ds] SKIP unknown dataset: $ds_key"
            continue
        fi
        if [ ! -f "$ds_path" ]; then
            log "  [$idx/$total_ds] SKIP missing parquet: $ds_path"
            continue
        fi

        log "  [$idx/$total_ds] inference: $ds_key  ->  $output_dir"
        if python -m geo_edit.scripts.async_generate_with_tool_call_api \
                --api_base "$API_BASE" \
                --dataset_path "$ds_path" \
                --dataset_name "$ds_name" \
                --output_dir "$output_dir" \
                --model_name_or_path "$CKPT_HF_PATH" \
                --model_type vLLM \
                --sample_rate "$SAMPLE_RATE" \
                --use_tools auto \
                --max_concurrent_requests "$MAX_CONCURRENT" \
                --max_tool_calls "$MAX_TOOL_CALLS" \
                --enable_tools map general \
                --temperature "$TEMPERATURE" \
                $COMPRESS_FLAG \
                > "$inf_log" 2>&1; then
            log "  [$idx/$total_ds] inference done: $ds_key (-> background eval)"
            INFERENCE_OK[$ds_key]=1
        else
            log "  [$idx/$total_ds] INFERENCE FAILED on $ds_key (see $inf_log)"
            FAILED_DS["${step}_${ds_key}"]="inference"
            tail -n 20 "$inf_log" | sed 's/^/    /' | tee -a "$LOG_DIR/run.log"
            continue
        fi

        # Inference succeeded - launch eval in background while we move to the next inference.
        judge_args=""
        if [ "${SKIP_JUDGE[$ds_key]:-0}" != "1" ] && [ -n "$JUDGE_API_KEY" ]; then
            judge_args="--use_judge --judge_model $JUDGE_MODEL --judge_api_key $JUDGE_API_KEY"
            [ -n "$JUDGE_API_BASE" ] && judge_args="$judge_args --judge_api_base $JUDGE_API_BASE"
        fi

        (
            python -m geo_edit.evaluation.eval_unified \
                    --dataset_name "$ds_name" \
                    --result_path "$output_dir" \
                    --output_path "$eval_output" \
                    $judge_args \
                    > "$eval_log" 2>&1
            echo "__EVAL_EXIT__=$?" >> "$eval_log"
        ) &
        EVAL_PIDS[$ds_key]=$!
        EVAL_LOGS[$ds_key]="$eval_log"
    done

    # Wait for all background evals to finish before parsing accuracy
    n_pending=${#EVAL_PIDS[@]}
    log "=== Waiting for $n_pending background eval(s) (step $step) ==="
    for ds_key in $DATASETS; do
        pid="${EVAL_PIDS[$ds_key]:-}"
        [ -z "$pid" ] && continue
        log "  waiting eval: $ds_key (pid=$pid)"
        wait "$pid"
        eval_log="${EVAL_LOGS[$ds_key]}"
        if [ -f "$eval_log" ] && grep -q "^__EVAL_EXIT__=0" "$eval_log"; then
            acc=$(parse_accuracy "$(cat "$eval_log")")
            RESULTS["${step},${ds_key}"]="$acc"
            log "  eval done: $ds_key -> $acc"
        else
            log "  EVAL FAILED on $ds_key (see $eval_log)"
            RESULTS["${step},${ds_key}"]="ERR"
            FAILED_DS["${step}_${ds_key}"]="eval"
        fi
    done

    # Datasets that never got an eval (inference failed): fill cells.
    for ds_key in $DATASETS; do
        if [ -z "${RESULTS["${step},${ds_key}"]:-}" ]; then
            RESULTS["${step},${ds_key}"]="N/A"
        fi
    done

    unset EVAL_PIDS EVAL_LOGS INFERENCE_OK

    # Append row to CSV
    {
        echo -n "$step"
        for ds_key in $DATASETS; do
            echo -n ",${RESULTS["${step},${ds_key}"]:-N/A}"
        done
        echo
    } >> "$RESULTS_CSV"

    log "##########################################################################"
    log "# Checkpoint step $step complete"
    log "##########################################################################"
done

# ── Final summary ─────────────────────────────────────────────────────────────
{
    echo
    echo "=========================================================================="
    echo "  BENCHMARK SUMMARY"
    echo "  Run name    : $RUN_NAME"
    echo "  Steps       : $CHECKPOINT_STEPS"
    echo "  Datasets    : $DATASETS"
    echo "  Temperature : $TEMPERATURE"
    echo "  Mode        : tool (max_tool_calls=$MAX_TOOL_CALLS)"
    echo "  Judge       : $JUDGE_MODEL"
    echo "  Time        : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================================================="
    # column widths
    col_w=26
    printf "%-10s" "step"
    for ds_key in $DATASETS; do
        printf " %-${col_w}s" "$ds_key"
    done
    echo
    printf "%-10s" "----"
    for ds_key in $DATASETS; do
        printf " %-${col_w}s" "$(printf -- '-%.0s' $(seq 1 $col_w))"
    done
    echo
    for step in $CHECKPOINT_STEPS; do
        printf "%-10s" "$step"
        for ds_key in $DATASETS; do
            printf " %-${col_w}s" "${RESULTS["${step},${ds_key}"]:-N/A}"
        done
        echo
    done
    echo
    if [ ${#FAILED_DS[@]} -gt 0 ]; then
        echo "Failed combinations:"
        for k in "${!FAILED_DS[@]}"; do
            echo "  $k -> ${FAILED_DS[$k]}"
        done
    fi
    echo
    echo "Inference results : ${OUTPUT_ROOT}/<dataset>/${RUN_NAME}_step<N>_tool_temp0/"
    echo "Eval outputs      : ${EVAL_OUTPUT_ROOT}/<dataset>/${RUN_NAME}_step<N>_tool_temp0/"
    echo "Logs              : $LOG_DIR"
    echo "CSV               : $RESULTS_CSV"
    echo "=========================================================================="
} | tee "$SUMMARY_FILE" | tee -a "$LOG_DIR/run.log"

# Final cleanup: stop vLLM so GPUs are free for other work.
kill_vllm
log "Pipeline complete."
