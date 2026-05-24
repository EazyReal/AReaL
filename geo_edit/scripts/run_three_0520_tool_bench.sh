#!/usr/bin/env bash
# ==============================================================================
# Benchmark three 0520 SFT models in TOOL mode (enable_tools = map general)
# on the 7-dataset eval suite requested by the user:
#   ReasonMap_plus, ReasonMap, VisualProbe_easy/medium/hard, MapTrace, OmniSpatial.
#
# Hardware: 8x NVIDIA L20X (~144GB each) per node.
# Per model: DP=8 TP=1, max_model_len=40960, gpu_mem_util=0.85
# Models run SERIALLY (each kills vLLM after finishing).
#
# Required env:
#   JUDGE_API_KEY, JUDGE_API_BASE
#
# Usage:
#   bash geo_edit/scripts/run_three_0520_tool_bench.sh           # all 3 models
#   bash geo_edit/scripts/run_three_0520_tool_bench.sh lr1e5     # subset
# ==============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${JUDGE_API_KEY:?JUDGE_API_KEY must be set}"
: "${JUDGE_API_BASE:?JUDGE_API_BASE must be set}"

# 7 datasets specified by the user (omnispatial replaces mapqa here).
DATASETS="reason_map_plus reason_map visual_probe_easy visual_probe_medium visual_probe_hard map_trace omnispatial"
MODE="tool"
PORT=8000
SAMPLE_RATE="${SAMPLE_RATE:-1.0}"
MAX_CONCURRENT="${MAX_CONCURRENT:-32}"

# Model registry: tag -> "path|dp|tp|max_model_len|gpu_mem_util|extra_args"
MODEL_ROOT="/storage/openpsi/models/lcy_image_edit/sft_workspace"
declare -A MODEL_CFG=(
  [lr1e5]="${MODEL_ROOT}/qwen3vl8b-thinking-5ds-v3-0520-ct65536-lr1e5|8|1|40960|0.85|"
  [lr3e6]="${MODEL_ROOT}/qwen3vl8b-thinking-5ds-v3-0520-ct65536-lr3e6|8|1|40960|0.85|"
  [lr7e6]="${MODEL_ROOT}/qwen3vl8b-thinking-5ds-v3-0520-ct65536-lr7e6|8|1|40960|0.85|"
)

# Default order: lr1e5 -> lr3e6 -> lr7e6
TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(lr1e5 lr3e6 lr7e6)

LOG_DIR=/tmp/log/three_0520_tool_bench
mkdir -p "$LOG_DIR"

run_one() {
  local tag="$1"
  local cfg="${MODEL_CFG[$tag]:-}"
  if [ -z "$cfg" ]; then
    echo "[ERROR] Unknown model tag: $tag"; return 1
  fi
  IFS='|' read -r MODEL_PATH DP_SIZE TP_SIZE MAX_MODEL_LEN GPU_MEM_UTIL EXTRA_VLLM_ARGS <<< "$cfg"
  local MODEL_NAME
  MODEL_NAME=$(basename "$MODEL_PATH")
  local model_log="$LOG_DIR/${MODEL_NAME}.log"

  echo "================================================================"
  echo "  [$(date '+%F %T')] Running model: $MODEL_NAME  (mode=tool, tools=map general)"
  echo "    path=$MODEL_PATH"
  echo "    DP=$DP_SIZE  TP=$TP_SIZE  max_model_len=$MAX_MODEL_LEN  gpu_mem=$GPU_MEM_UTIL"
  echo "    sample_rate=$SAMPLE_RATE  max_concurrent=$MAX_CONCURRENT"
  echo "    datasets=$DATASETS"
  echo "    log=$model_log"
  echo "================================================================"

  # Hard kill any leftover vllm before each run.
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -9 -f "EngineCore" 2>/dev/null || true
  sleep 5

  DP_SIZE="$DP_SIZE" TP_SIZE="$TP_SIZE" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" GPU_MEM_UTIL="$GPU_MEM_UTIL" \
  EXTRA_VLLM_ARGS="$EXTRA_VLLM_ARGS" \
  bash "$SCRIPT_DIR/run_model_benchmark.sh" \
      --model "$MODEL_PATH" \
      --mode "$MODE" \
      --datasets "$DATASETS" \
      --no-image-compression \
      --judge-api-key "$JUDGE_API_KEY" \
      --judge-api-base "$JUDGE_API_BASE" \
      --gpu-mem-util "$GPU_MEM_UTIL" \
      --port "$PORT" \
      --max-concurrent "$MAX_CONCURRENT" \
      --sample-rate "$SAMPLE_RATE" \
      2>&1 | tee "$model_log"
  local rc=${PIPESTATUS[0]}

  echo "[$(date '+%F %T')] Finished $MODEL_NAME (exit=$rc)"
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -9 -f "EngineCore" 2>/dev/null || true
  sleep 5
  return $rc
}

OVERALL_RC=0
for tag in "${TARGETS[@]}"; do
  if ! run_one "$tag"; then
    echo "[WARN] tag=$tag failed; continuing to next model."
    OVERALL_RC=1
  fi
done

echo "================================================================"
echo "  ALL DONE. Logs:        $LOG_DIR"
echo "  Inference outputs:     /storage/openpsi/data/lcy_image_edit/eval_results/<ds>/<MODEL>_tool"
echo "  Eval outputs:          /storage/openpsi/data/lcy_image_edit/eval_output/<ds>/<MODEL>_tool"
echo "================================================================"
exit $OVERALL_RC
