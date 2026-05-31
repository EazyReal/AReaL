#!/usr/bin/env bash
# Score inference outputs for one dataset. Default: visual_probe_easy.
# The dataset id auto-resolves to its eval template via
# geo_edit.eval_datasets.DATASET_REGISTRY.
#
# Examples:
#   JUDGE_API_KEY=... JUDGE_API_BASE=... bash geo_edit/scripts/run_eval.sh
#   DATASET=reason_map JUDGE_API_KEY=... JUDGE_API_BASE=... bash geo_edit/scripts/run_eval.sh
set -euo pipefail

DATASET="${DATASET:-visual_probe_easy}"
PEDIA_MODEL="${PEDIA_MODEL:-./pedia_model}"
MODEL_PATH="${MODEL_PATH:-${PEDIA_MODEL}/PEDIA_8B_v1}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/eval_results}"
OUT_DIR="${OUTPUT_ROOT}/${DATASET}/${MODEL_NAME}"
EVAL_OUT_DIR="${OUTPUT_ROOT/eval_results/eval_output}/${DATASET}/${MODEL_NAME}"

JUDGE_API_KEY="${JUDGE_API_KEY:?need JUDGE_API_KEY}"
JUDGE_API_BASE="${JUDGE_API_BASE:?need JUDGE_API_BASE}"

mkdir -p "$EVAL_OUT_DIR"

python -m geo_edit.evaluation.eval_unified \
    --dataset "$DATASET" \
    --result_path "$OUT_DIR" \
    --output_path "$EVAL_OUT_DIR" \
    --use_judge \
    --judge_model "${JUDGE_MODEL:-gpt-4.1-mini-2025-04-14}" \
    --judge_api_key "$JUDGE_API_KEY" \
    --judge_api_base "$JUDGE_API_BASE"
