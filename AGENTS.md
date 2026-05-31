<!-- Project brief for AI coding agents. README.md is the human-facing version. -->

# AGENTS.md — PERIA Minimal Repo Guide

## TL;DR

- **Project**: PERIA (paper: *Perceive, Interact, Reason: Building
  Tool-Augmented Visual Agents for Spatial Reasoning*). Derivative of
  [AReaL](https://github.com/inclusionAI/AReaL) +
  [verl-tool](https://github.com/volcengine/verl), stripped to only what we use.
- **Default model**: `./pedia_model/PEDIA_8B_v1` (8B RL ckpt).
- **Naming gotcha**: project is **PERIA**, on-disk prefix stays `pedia_*`
  (model dirs, data dirs, training scripts: `run_pedia_rl_*`). Conda envs use
  `peria-*`. Don't "fix" by renaming — many configs would break.
- **Runtime**: 3 mutually-incompatible conda envs, Python 3.12, CUDA 12.8 via
  `nvidia-*` pip wheels. Singularity sif is legacy; don't suggest it for new
  work.
- **State paths** (HARD RULE): `./pedia_model/`, `./pedia_data/`,
  `./outputs/{mixed_rl,eval_results,eval_output,eval_logs,trajectories}/`.
  NEVER hard-code absolute paths. Override base dirs via `PEDIA_MODEL` /
  `PEDIA_DATA` env vars only.

## Repository layout

```
AReaL/
├── llamafactory/        # SFT training (LLaMA-Factory wrapper)
├── geo_edit/            # Tool catalog + inference + eval entries
├── train_tool_server/   # Ray+HTTP tool backend for RL rollouts
├── verl-tool/           # Vendored verl + RL training entry
├── AGENTS.md            # this file
├── README.md            # paper / project README (human-facing)
└── LICENSE
```

| Workspace | Purpose | Env |
|---|---|---|
| `llamafactory/`      | SFT training (LLaMA-Factory PyPI wrapper)                              | `peria-sft`   |
| `geo_edit/`          | Tool catalog, inference driver, eval driver, eval dataset registry     | `peria-tools` |
| `train_tool_server/` | Ray+HTTP tool backend (4 tool agents + router on :30888)               | `peria-tools` |
| `verl-tool/`         | Vendored verl + RL training entry (renamed from `verl-tool_060`)        | `peria-rl`    |

## Conda envs (3 mutually-incompatible)

| Env | Install command | Versions | Stages |
|---|---|---|---|
| `peria-sft`   | `cd llamafactory && pip install -r requirements.txt`                                                                                                                                                                                  | torch 2.8 + transformers 4.57.1 + DeepSpeed | SFT                                |
| `peria-rl`    | `cd verl-tool && unset ROCR_VISIBLE_DEVICES && TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=48 NVCC_THREADS=4 pip install -r requirements.txt && pip uninstall -y deep_ep deep_gemm`                                                            | torch 2.8 + vllm 0.11 + flash-attn 2.7.4 (built from source, ~15 min) | RL training |
| `peria-tools` | `cd geo_edit && pip install -U -r requirements.txt && cd ../train_tool_server && pip install -r requirements.txt`                                                                                                                       | torch ≥ 2.10 + vllm 0.17                    | Tool server, eval inference, SFT data synthesis |

Activation cheat-sheet:

```bash
conda activate peria-sft     # → llamafactory/train_v{1,2}.sh
conda activate peria-rl      # → verl-tool/examples/train/geo_edit/run_pedia_rl_v1_*node.sh
conda activate peria-tools   # → train_tool_server/scripts/launch_tool_server.sh
                             #   or geo_edit/scripts/run_inference.sh / run_eval.sh
```

## Models in `./pedia_model/`

All weights are mirrored to a single HF model repo
`<your-org>/PERIA-Models`; download a subset with `hf download ... --include`.

| Dir | Purpose | HF `--include` glob |
|---|---|---|
| `PEDIA_8B_v1`         | 8B RL ckpt (default eval target)                  | `PEDIA_8B_v1/*` |
| `pedia_8b_SFT_v1`     | 8B SFT ckpt (RL start point)                       | `pedia_8b_SFT_v1/*` |
| `pedia_4b_v1`         | 4B RL (optional)                                   | `pedia_4b_v1/*` |
| `pedia_2b_v1`         | 2B RL (optional)                                   | `pedia_2b_v1/*` |
| `Qwen3-VL-8B-Thinking`| Base VLM (SFT start + data synthesis)              | `Qwen3-VL-8B-Thinking/*` |
| `PaddleOCR-VL-1.5`    | Tool backend (`geo_paddleocr`, 7 sub-tools, **2 Ray replicas**) | `PaddleOCR-VL-1.5/*` |
| `sam3.1`              | Tool backend (`geo_sam3`, 6 sub-tools); needs `sam3.1_multiplex.pt` | `sam3.1/*` |
| `grounding-dino-base` | Tool backend (`geo_grounding_dino`)                | `grounding-dino-base/*` |

## Data in `./pedia_data/`

All data is mirrored to a single HF dataset repo `<your-org>/PERIA-Data`.

| Path | Contents | HF `--include` glob |
|---|---|---|
| `pedia_sft_v1/`              | SFT data (`train.json` + `images/`)             | `pedia_sft_v1/*` |
| `pedia_rl_v1/`               | RL data (`train.parquet` + `val.parquet` + `images/`) | `pedia_rl_v1/*` |
| `eval/id/<dataset>.parquet`  | 6 in-distribution benchmarks                    | `eval/id/<dataset>.parquet` |
| `eval/ood/<dataset>.parquet` | 7 out-of-distribution benchmarks                | `eval/ood/<dataset>.parquet` |

## Eval dataset registry — single source of truth

[`geo_edit/eval_datasets.py`](./geo_edit/eval_datasets.py) maps `dataset_id ->
(parquet_relpath, eval_template)`. Both `run_inference.sh` and `run_eval.sh`
take a single `--dataset <id>` and auto-resolve everything else.

Registered ids (13):

```
ID  : visual_probe_{easy,medium,hard}  reason_map{,_plus}  map_trace
OOD : visworld_{cube,mmsi,ballgame,paperfolding}  mapeval_visual  babyvision  vstar_bench
```

**To add a new eval dataset**: append one tuple to `DATASET_REGISTRY` in
`geo_edit/eval_datasets.py`. No shell edits needed.

## Entry scripts (canonical)

| Script | Role |
|---|---|
| `geo_edit/scripts/run_inference.sh`               | Auto-launches vLLM (`DP=8`) in background, waits for `/v1/models`, runs inference for `$DATASET`, `trap`-kills vLLM on exit. |
| `geo_edit/scripts/run_eval.sh`                    | Scores inference outputs. **Defaults to rule-based only** (fast sanity check). Export `JUDGE_API_KEY` to enable LLM-judge fallback (needed to reproduce paper numbers); `JUDGE_API_BASE` defaults to `https://api.openai.com/v1`. |
| `geo_edit/scripts/iterative_sampling_generate.py` | SFT trajectory sampling against a **raw** third-party dataset (uses `--dataset_path` + `--dataset_name`, NOT the registry). |
| `geo_edit/scripts/async_generate_with_tool_call_api.py` | Inner inference runner (called by `run_inference.sh`). |
| `geo_edit/scripts/launch_cantainer.sh`            | Legacy: `srun + singularity shell` into base sif. |
| `llamafactory/train_v{1,2}.sh`                    | 8-GPU SFT with DeepSpeed ZeRO-2; configs at `llamafactory/configs/pedia_sft_v{1,2}.yaml`. |
| `train_tool_server/scripts/launch_tool_server.sh` | Boots 4 tool backends (ports 30889-30892) + router on 30888. |
| `verl-tool/examples/train/geo_edit/run_pedia_rl_v1_singlenode.sh` | RL training, 1 × 8 GPU (embedded `ray start --head`). |
| `verl-tool/examples/train/geo_edit/run_pedia_rl_v1_multinode.sh`  | RL training, 4 × 8 GPU (external Ray cluster). |
| `verl-tool/examples/train/geo_edit/ray_start_{head,worker}.sh`     | Ray cluster setup for multinode RL. |

## Workflows

### Eval (2-node minimum: tool node + inference node)

```bash
# ─── Node A: tool server (all 8 GPUs) ───
conda activate peria-tools
unset ROCR_VISIBLE_DEVICES
ray start --head --port=6379 --num-gpus=8 --resources='{"tool_agent": 8}'
bash train_tool_server/scripts/launch_tool_server.sh        # router on :30888

# ─── Node B: run_inference auto-launches vLLM DP=8 → runs inference ───
conda activate peria-tools
unset ROCR_VISIBLE_DEVICES
ray start --address=<node-a-ip>:6379
bash geo_edit/scripts/run_inference.sh                       # DATASET=visual_probe_easy
# DATASET=reason_map bash geo_edit/scripts/run_inference.sh  # other registered id

# ─── Node B (or anywhere CPU): score outputs ───
# Rule-based by default. To reproduce paper numbers, export JUDGE_API_KEY
# (JUDGE_API_BASE defaults to OpenAI's official endpoint).
bash geo_edit/scripts/run_eval.sh
```

Single-node fallback: `DP_SIZE=4 CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_inference.sh`
+ `CUDA_VISIBLE_DEVICES=0,1,2,3 bash launch_tool_server.sh` on the same machine.

### SFT (1 × 8 GPU)

```bash
conda activate peria-sft
cd llamafactory && bash train_v1.sh           # → ./pedia_model/pedia_8b_SFT_v1/
```

### RL (2-node minimum: tool node + training node)

```bash
# ─── Node A: tool server (same as Eval Node A) ───

# ─── Node B: RL training ───
conda activate peria-rl
unset ROCR_VISIBLE_DEVICES
# JUDGE_API_KEY optional (reward manager uses it if set);
# JUDGE_API_BASE defaults to https://api.openai.com/v1.
TOOL_SERVER_URL=http://<node-a-ip>:30888/get_observation \
    bash verl-tool/examples/train/geo_edit/run_pedia_rl_v1_singlenode.sh
# For 4×8 GPU training: use ray_start_{head,worker}.sh + run_pedia_rl_v1_multinode.sh.
```

> RL training uses **HTTP** (`TOOL_SERVER_URL`) to reach the tool server.
> Eval inference uses **Ray actors** (both nodes must be in the same Ray
> cluster, `tool_agent` resource). Different code paths — don't conflate.

### SFT data synthesis (3-stage pipeline)

```bash
conda activate peria-tools
export JUDGE_API_KEY=<your-openai-key>      # REQUIRED — trajectory filter calls the judge
# JUDGE_API_BASE defaults to https://api.openai.com/v1

# 1. Tool server (Node A) + Qwen3-VL-8B-Thinking served via run_inference-style vLLM.

# 2. Iterative sampling — uses RAW source data (not registry).
python -m geo_edit.scripts.iterative_sampling_generate \
    --api_base http://127.0.0.1:8000 \
    --dataset_path ./pedia_data/raw/<source>/train.parquet \
    --dataset_name <eval_template_name> \
    --output_dir ./outputs/trajectories/<source> \
    --model_name_or_path ./pedia_model/Qwen3-VL-8B-Thinking \
    --model_type vLLM --temperature 0.7 \
    --use_tools auto --enable_tools map general --max_tool_calls 10

# 3. Filter + diversify
python -m geo_edit.data_preprocess.augment_traj_data \
    --src_dir ./outputs/trajectories/<source> \
    --dst_dir ./outputs/trajectories/<source>_augmented

# 4. Convert to LLaMA-Factory SFT format
python -m geo_edit.data_preprocess.convert_trajectory_to_sft \
    --src_dir ./outputs/trajectories/<source>_augmented \
    --dst_dir ./pedia_data/pedia_sft_v1
```

## Gotchas (READ before editing)

1. **`unset ROCR_VISIBLE_DEVICES` before any `ray start`**. Ray workers
   inherit it and crash with ROCm errors on NVIDIA boxes.

2. **`flash-attn` in `peria-rl` MUST be built from source** against torch 2.8.
   Build env: `TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=48 NVCC_THREADS=4 pip install -r requirements.txt`.
   ~15 min on a 192-CPU node. Wheel for torch 2.10 silently mis-versions
   and crashes at runtime.

3. **NEVER install `deep_ep` or `deep_gemm` in `peria-rl`** — torch 2.10
   ABI; crashes the torch 2.8 baseline at import time. Sanity check after
   any `peria-rl` reinstall: `pip uninstall -y deep_ep deep_gemm`.

4. **Tool server uses two transports in different code paths**:
   - Eval inference (`async_generate_with_tool_call_api.py`) → **Ray actors**
     (both nodes must be in the same Ray cluster).
   - RL rollouts (verl-tool) → **HTTP** via `TOOL_SERVER_URL` env var.
   Don't mix them up.

5. **`_AREAL_ROOT` post-flatten**: in
   `train_tool_server/train_tool_server/tools/geo_edit_base.py`, uses **3
   `..`** levels (not 4). Don't reintroduce the 4-level assumption when
   refactoring tool paths.

6. **PaddleOCR-VL `num_replicas: 2`** in
   `geo_edit/tool_definitions/agents/paddleocr_tool.py`. Earlier was 6.
   Bumping back will OOM — 2 PaddleOCR replicas + SAM3 + Grounding-DINO +
   vLLM `DP=8` already don't co-locate on one 8-GPU node.

7. **No wandb**: every training script uses `trainer.logger=['console']`.
   Don't add wandb back.

8. **Path discipline**: NEVER hard-code `/storage/...` or any absolute path.
   Override base dirs via `PEDIA_MODEL` / `PEDIA_DATA` env vars only.

9. **Singularity sif is legacy.** `launch_cantainer.sh` is kept for
   back-compat (pass `IMAGE=/path/to/your.sif`; historical base was
   `pytorch280-1210-v1.sif`), but Conda is the default for new work.
