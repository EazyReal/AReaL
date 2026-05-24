#!/usr/bin/env bash
# Re-run ONLY the 103 failed tasks (image cap=5 → 30) for the affected (model, dataset) pairs.
#
# Affected pairs:
#   lr1e5: visual_probe_easy/medium/hard          (27 failures)
#   lr3e6: reason_map_plus + visual_probe_*       (32)
#   lr7e6: reason_map_plus + visual_probe_*       (44)
#
# The failed task subdirs were already deleted, so async_generate's resume
# logic will only process those.
#
# Required env: JUDGE_API_KEY, JUDGE_API_BASE
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${JUDGE_API_KEY:?JUDGE_API_KEY must be set}"
: "${JUDGE_API_BASE:?JUDGE_API_BASE must be set}"

MODE="tool"
PORT=8000
SAMPLE_RATE="${SAMPLE_RATE:-1.0}"
MAX_CONCURRENT="${MAX_CONCURRENT:-32}"
# CRITICAL FIX: bump image cap from default 5 → 30 to accommodate tool-call image accumulation
export MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-30}"

MODEL_ROOT="/storage/openpsi/models/lcy_image_edit/sft_workspace"

# tag -> (path|datasets-to-rerun)
declare -A MODEL_PATH=(
  [lr1e5]="${MODEL_ROOT}/qwen3vl8b-thinking-5ds-v3-0520-ct65536-lr1e5"
  [lr3e6]="${MODEL_ROOT}/qwen3vl8b-thinking-5ds-v3-0520-ct65536-lr3e6"
  [lr7e6]="${MODEL_ROOT}/qwen3vl8b-thinking-5ds-v3-0520-ct65536-lr7e6"
)
declare -A MODEL_DATASETS=(
  [lr1e5]="visual_probe_easy visual_probe_medium visual_probe_hard"
  [lr3e6]="reason_map_plus visual_probe_easy visual_probe_medium visual_probe_hard"
  [lr7e6]="reason_map_plus visual_probe_easy visual_probe_medium visual_probe_hard"
)

TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(lr1e5 lr3e6 lr7e6)

LOG_DIR=/tmp/log/three_0520_rerun
mkdir -p "$LOG_DIR"

run_one() {
  local tag="$1"
  local mpath="${MODEL_PATH[$tag]:-}"
  local datasets="${MODEL_DATASETS[$tag]:-}"
  [ -z "$mpath" ] && { echo "[ERROR] unknown $tag"; return 1; }
  local model_name
  model_name=$(basename "$mpath")
  local log="$LOG_DIR/${model_name}.log"

  echo "================================================================"
  echo "  [$(date '+%F %T')] RERUN model=$tag datasets=$datasets"
  echo "    MAX_IMAGES_PER_PROMPT=$MAX_IMAGES_PER_PROMPT"
  echo "    log=$log"
  echo "================================================================"

  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -9 -f "EngineCore" 2>/dev/null || true
  sleep 5

  DP_SIZE=8 TP_SIZE=1 MAX_MODEL_LEN=40960 GPU_MEM_UTIL=0.85 \
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
  local rc=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] Finished $tag (exit=$rc)"

  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -9 -f "EngineCore" 2>/dev/null || true
  sleep 5
  return $rc
}

OVERALL_RC=0
for tag in "${TARGETS[@]}"; do
  if ! run_one "$tag"; then
    echo "[WARN] tag=$tag failed; continuing"
    OVERALL_RC=1
  fi
done

echo "================================================================"
echo "  RERUN ALL DONE. Logs: $LOG_DIR"
echo "================================================================"
exit $OVERALL_RC
