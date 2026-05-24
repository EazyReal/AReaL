#!/usr/bin/env bash
# Fix the 3 residual context-overflow failures by raising max_model_len.
#
# Targets:
#   lr1e5: visual_probe_hard      (1 task: visual_probe_hard_40)
#   lr7e6: visual_probe_easy      (1 task: visual_probe_easy_39)
#         visual_probe_medium     (1 task: visual_probe_medium_75)
#
# The 3 failed task subdirs were already deleted; resume logic will retry them.
# vLLM gets MAX_MODEL_LEN=65536 (up from 40960) - model max_position_embeddings=262144 so safe.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${JUDGE_API_KEY:?JUDGE_API_KEY must be set}"
: "${JUDGE_API_BASE:?JUDGE_API_BASE must be set}"

# CRITICAL: bump context window from 40960 → 65536
export MAX_MODEL_LEN=65536
export MAX_IMAGES_PER_PROMPT=30

MODE="tool"
PORT=8000
SAMPLE_RATE=1.0
MAX_CONCURRENT=32

declare -A MODEL_PATH=(
  [lr1e5]="/storage/openpsi/models/lcy_image_edit/sft_workspace/qwen3vl8b-thinking-5ds-v3-0520-ct65536-lr1e5"
  [lr7e6]="/storage/openpsi/models/lcy_image_edit/sft_workspace/qwen3vl8b-thinking-5ds-v3-0520-ct65536-lr7e6"
)
declare -A MODEL_DATASETS=(
  [lr1e5]="visual_probe_hard"
  [lr7e6]="visual_probe_easy visual_probe_medium"
)

LOG_DIR=/tmp/log/three_0520_fix_context
mkdir -p "$LOG_DIR"

run_one() {
  local tag="$1"
  local mpath="${MODEL_PATH[$tag]}"
  local datasets="${MODEL_DATASETS[$tag]}"
  local model_name
  model_name=$(basename "$mpath")
  local log="$LOG_DIR/${model_name}.log"

  echo "================================================================"
  echo "  [$(date '+%F %T')] CONTEXT-FIX model=$tag datasets=$datasets max_model_len=$MAX_MODEL_LEN"
  echo "    log=$log"
  echo "================================================================"

  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -9 -f "EngineCore" 2>/dev/null || true
  sleep 5

  DP_SIZE=8 TP_SIZE=1 MAX_MODEL_LEN="$MAX_MODEL_LEN" GPU_MEM_UTIL=0.85 \
  MAX_IMAGES_PER_PROMPT="$MAX_IMAGES_PER_PROMPT" \
  bash "$SCRIPT_DIR/run_model_benchmark.sh" \
      --model "$mpath" \
      --mode "$MODE" \
      --datasets "$datasets" \
      --no-image-compression \
      --judge-api-key "$JUDGE_API_KEY" \
      --judge-api-base "$JUDGE_API_BASE" \
      --gpu-mem-util "0.85" \
      --port "$PORT" \
      --max-concurrent "$MAX_CONCURRENT" \
      --sample-rate "$SAMPLE_RATE" \
      2>&1 | tee "$log"

  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -9 -f "EngineCore" 2>/dev/null || true
  sleep 5
}

for tag in lr1e5 lr7e6; do
  run_one "$tag"
done

echo "================================================================"
echo "  CONTEXT-FIX DONE. Logs: $LOG_DIR"
echo "================================================================"
