#!/usr/bin/env bash
# Run inference on one dataset (default: visual_probe_easy).
#
# 1. Launches vLLM (DP=8 by default — all 8 local GPUs) in the background.
# 2. Waits for the OpenAI-compatible endpoint to come up.
# 3. Runs async_generate_with_tool_call_api against the dataset.
# 4. Stops vLLM on exit (trap).
#
# The dataset id auto-resolves to its parquet path + eval template via
# geo_edit.eval_datasets.DATASET_REGISTRY.
#
# Examples:
#   bash geo_edit/scripts/run_inference.sh                       # visual_probe_easy
#   DATASET=reason_map bash geo_edit/scripts/run_inference.sh    # any registered id
#   DP_SIZE=4 VLLM_PORT=8001 bash geo_edit/scripts/run_inference.sh  # tweak vLLM
#
# Prereq: a tool server reachable via the local Ray cluster (Ray tool_agent
# actors must be registered — either via `bash train_tool_server/scripts/
# launch_tool_server.sh` on the same machine, or by joining a remote Ray
# head with `ray start --address=<head-ip>:6379`).
set -euo pipefail

# ─── inference config ───
DATASET="${DATASET:-visual_probe_easy}"
PEDIA_DATA="${PEDIA_DATA:-./pedia_data}"
PEDIA_MODEL="${PEDIA_MODEL:-./pedia_model}"
MODEL_PATH="${MODEL_PATH:-${PEDIA_MODEL}/PEDIA_8B_v1}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/eval_results}"
OUT_DIR="${OUTPUT_ROOT}/${DATASET}/${MODEL_NAME}"

# ─── vLLM config ───
VLLM_PORT="${VLLM_PORT:-8000}"
API_BASE="${API_BASE:-http://127.0.0.1:${VLLM_PORT}}"
DP_SIZE="${DP_SIZE:-8}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-5}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
VLLM_LOG="${VLLM_LOG:-/tmp/log/vllm_${MODEL_NAME}.log}"

mkdir -p "$OUT_DIR" "$(dirname "$VLLM_LOG")"

# ─── 1. Background-launch vLLM ───
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
echo "[run_inference] launching vLLM dp=$DP_SIZE tp=$TP_SIZE model=$MODEL_PATH log=$VLLM_LOG"
nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    --data-parallel-size "$DP_SIZE" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype auto \
    --allowed-local-media-path "$PEDIA_DATA" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-prefix-caching \
    --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES_PER_PROMPT}}" \
    $EXTRA_VLLM_ARGS \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'echo "[run_inference] stopping vLLM pid=$VLLM_PID"; kill "$VLLM_PID" 2>/dev/null || true' EXIT

# ─── 2. Wait for endpoint ───
echo "[run_inference] vLLM pid=$VLLM_PID — waiting for http://127.0.0.1:${VLLM_PORT}/v1/models"
until curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[run_inference] vLLM process died — last 30 log lines:"
        tail -n 30 "$VLLM_LOG" || true
        exit 1
    fi
    sleep 5
done
echo "[run_inference] vLLM endpoint ready"

# ─── 3. Run inference ───
python -m geo_edit.scripts.async_generate_with_tool_call_api \
    --dataset "$DATASET" \
    --data_root "$PEDIA_DATA" \
    --output_dir "$OUT_DIR" \
    --model_name_or_path "$MODEL_PATH" \
    --model_type vLLM --api_base "$API_BASE" \
    --temperature 0 --sample_rate 1.0 \
    --use_tools auto --enable_tools map general \
    --max_concurrent_requests 16 --max_tool_calls 10 \
    --no_image_compression

echo "[run_inference] done — output at $OUT_DIR"
