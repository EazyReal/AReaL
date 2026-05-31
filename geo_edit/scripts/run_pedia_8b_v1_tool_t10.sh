#!/usr/bin/env bash
# Run PEDIA_8B_v1 on all 13 datasets at temp=1.0
# In-dist (6): visual_probe_{easy,medium,hard}, map_trace, reason_map, reason_map_plus
# OOD (7):     mapeval_visual, babyvision, visworld_{ballgame,paperfolding,cube,mmsi}, vstar_bench
set -uo pipefail

cd /storage/openpsi/users/lichangye.lcy/antoinegg1/AReaL

: "${JUDGE_API_KEY:?}"
: "${JUDGE_API_BASE:?}"

MODEL_PATH="/storage/openpsi/data/lcy_image_edit/pedia_model/PEDIA_8B_v1"
MODEL_NAME="PEDIA_8B_v1"
API_BASE="http://127.0.0.1:8000"
MAX_CONCURRENT=64
MAX_TOOL_CALLS=10
JUDGE_MODEL="gpt-4.1-mini-2025-04-14"
TEMP=1.0
SUFFIX=t10

OUTPUT_ROOT="/storage/openpsi/data/lcy_image_edit/eval_results"
EVAL_OUTPUT_ROOT="/storage/openpsi/data/lcy_image_edit/eval_output"
LOG_DIR="/tmp/log/pedia_8b_v1_tool_t10"
mkdir -p "$LOG_DIR"

DATASETS=(
    visual_probe_easy visual_probe_medium visual_probe_hard
    map_trace reason_map reason_map_plus
    mapeval_visual babyvision
    visworld_ballgame visworld_paperfolding visworld_cube visworld_mmsi
    vstar_bench
)

declare -A DS_PATH=(
    [visual_probe_easy]="/storage/openpsi/data/VisualProbe_Easy/val.parquet"
    [visual_probe_medium]="/storage/openpsi/data/VisualProbe_Medium/val.parquet"
    [visual_probe_hard]="/storage/openpsi/data/VisualProbe_Hard/val.parquet"
    [map_trace]="/storage/openpsi/data/lcy_image_edit/maptrace_val_2851.parquet"
    [reason_map]="/storage/openpsi/data/ReasonMap/reasonmap_base_validation_dataset.parquet"
    [reason_map_plus]="/storage/openpsi/data/ReasonMap_plus/reasonmap_plus_test.parquet"
    [mapeval_visual]="/storage/openpsi/data/lcy_image_edit/mapeval_visual.parquet"
    [babyvision]="/storage/openpsi/data/lcy_image_edit/BabyVision/data/train-00000-of-00001.parquet"
    [visworld_ballgame]="/storage/openpsi/data/lcy_image_edit/VisWorld-Eval/ballgame/ballgame.parquet"
    [visworld_paperfolding]="/storage/openpsi/data/lcy_image_edit/VisWorld-Eval/paperfolding/paperfolding.parquet"
    [visworld_cube]="/storage/openpsi/data/lcy_image_edit/VisWorld-Eval/cube/cube.parquet"
    [visworld_mmsi]="/storage/openpsi/data/lcy_image_edit/VisWorld-Eval/mmsi/mmsi.parquet"
    [vstar_bench]="/storage/openpsi/data/Vstar_Bench/vstar_bench.parquet"
)

declare -A DS_EVAL=(
    [visual_probe_easy]="visual_probe"
    [visual_probe_medium]="visual_probe"
    [visual_probe_hard]="visual_probe"
    [map_trace]="map_trace"
    [reason_map]="reason_map"
    [reason_map_plus]="reason_map_plus"
    [mapeval_visual]="mapeval_visual"
    [babyvision]="babyvision"
    [visworld_ballgame]="visworld_eval"
    [visworld_paperfolding]="visworld_eval"
    [visworld_cube]="visworld_eval"
    [visworld_mmsi]="visworld_eval"
    [vstar_bench]="vstar_bench"
)

declare -A SKIP_JUDGE=([map_trace]=1)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/main.log"; }

EVAL_PIDS=()
EVAL_TAGS=()

log "================================================================"
log "  PEDIA_8B_v1 temp=$TEMP (suffix=$SUFFIX) — 13 datasets"
log "================================================================"

for ds_key in "${DATASETS[@]}"; do
    ds_path="${DS_PATH[$ds_key]}"
    ds_name="${DS_EVAL[$ds_key]}"
    out_dir="${OUTPUT_ROOT}/${ds_key}/${MODEL_NAME}_tool_${SUFFIX}"
    inf_log="$LOG_DIR/inference_${ds_key}_${SUFFIX}.log"
    eval_log="$LOG_DIR/eval_${ds_key}_${SUFFIX}.log"

    if [ ! -f "$ds_path" ]; then
        log "  SKIP $ds_key: missing $ds_path"; continue
    fi

    log "  [INF] $ds_key (temp=$TEMP) -> $out_dir"
    python -m geo_edit.scripts.async_generate_with_tool_call_api \
        --api_base "$API_BASE" \
        --dataset_path "$ds_path" \
        --dataset_name "$ds_name" \
        --output_dir "$out_dir" \
        --model_name_or_path "$MODEL_PATH" \
        --model_type vLLM \
        --sample_rate 1.0 \
        --use_tools auto \
        --max_concurrent_requests "$MAX_CONCURRENT" \
        --max_tool_calls "$MAX_TOOL_CALLS" \
        --enable_tools map general \
        --no_image_compression \
        --temperature "$TEMP" \
        > "$inf_log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        log "  [INF] FAILED $ds_key (rc=$rc); see $inf_log"
        continue
    fi
    log "  [INF] DONE $ds_key"

    judge_args=""
    if [ "${SKIP_JUDGE[$ds_key]:-0}" != "1" ]; then
        judge_args="--use_judge --judge_model $JUDGE_MODEL --judge_api_key $JUDGE_API_KEY --judge_api_base $JUDGE_API_BASE"
    fi

    log "  [EVAL] launching $ds_key bg"
    (python -m geo_edit.evaluation.eval_unified \
        --dataset_name "$ds_name" \
        --result_path "$out_dir" \
        --output_path "${EVAL_OUTPUT_ROOT}/${ds_key}/${MODEL_NAME}_tool_${SUFFIX}" \
        $judge_args > "$eval_log" 2>&1) &
    EVAL_PIDS+=($!)
    EVAL_TAGS+=("$ds_key")
done

log "  All inference done. Waiting for ${#EVAL_PIDS[@]} background evals ..."
for i in "${!EVAL_PIDS[@]}"; do
    pid="${EVAL_PIDS[$i]}"
    tag="${EVAL_TAGS[$i]}"
    wait "$pid"
    log "  [EVAL] $tag finished pid=$pid rc=$?"
done

log "================================================================"
log "  temp=$TEMP summary"
log "================================================================"
for ds_key in "${DATASETS[@]}"; do
    s="${EVAL_OUTPUT_ROOT}/${ds_key}/${MODEL_NAME}_tool_${SUFFIX}/summary.txt"
    if [ -f "$s" ]; then
        acc=$(grep -E "^Accuracy:|^Avg NDTW:" "$s" | head -1)
        printf "  %-25s %s\n" "$ds_key" "$acc" | tee -a "$LOG_DIR/main.log"
    fi
done

log "Pipeline complete."
