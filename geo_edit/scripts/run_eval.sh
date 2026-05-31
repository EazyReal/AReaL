#!/usr/bin/env bash
# Score inference outputs for one dataset. Default: visual_probe_easy.
#
# Default scoring: rule-based only (fast sanity check, no external API).
# To enable LLM-judge fallback (paper numbers), export JUDGE_API_KEY.
# JUDGE_API_BASE defaults to OpenAI's official endpoint; override only when
# using a non-OpenAI provider.
#
# Examples:
#   bash geo_edit/scripts/run_eval.sh                                  # rule-based
#   JUDGE_API_KEY=sk-... bash geo_edit/scripts/run_eval.sh             # + LLM judge (OpenAI)
#   JUDGE_API_KEY=... JUDGE_API_BASE=https://... bash run_eval.sh      # custom provider
#   DATASET=reason_map bash geo_edit/scripts/run_eval.sh               # other dataset
set -euo pipefail

DATASET="${DATASET:-visual_probe_easy}"
PEDIA_MODEL="${PEDIA_MODEL:-./pedia_model}"
MODEL_PATH="${MODEL_PATH:-${PEDIA_MODEL}/PEDIA_8B_v1}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/eval_results}"
OUT_DIR="${OUTPUT_ROOT}/${DATASET}/${MODEL_NAME}"
EVAL_OUT_DIR="${OUTPUT_ROOT/eval_results/eval_output}/${DATASET}/${MODEL_NAME}"

JUDGE_API_BASE="${JUDGE_API_BASE:-https://api.openai.com/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-mini-2025-04-14}"

mkdir -p "$EVAL_OUT_DIR"

CMD=(python -m geo_edit.evaluation.eval_unified
     --dataset "$DATASET"
     --result_path "$OUT_DIR"
     --output_path "$EVAL_OUT_DIR")

if [ -n "${JUDGE_API_KEY:-}" ]; then
    CMD+=(--use_judge
          --judge_model "$JUDGE_MODEL"
          --judge_api_key "$JUDGE_API_KEY"
          --judge_api_base "$JUDGE_API_BASE")
    echo "[run_eval] mode: rule-based + LLM judge (model=$JUDGE_MODEL base=$JUDGE_API_BASE)"
else
    echo "[run_eval] mode: rule-based only — export JUDGE_API_KEY to enable LLM-judge fallback"
fi

"${CMD[@]}"
