<!-- Project brief for AI coding agents working on this minimal repo. -->

# AGENTS.md — Pedia Minimal Repo Guide

## TL;DR

Minimal 4-environment workspace for the **PEDIA-8B** RL+tool training & evaluation
pipeline (paper: *"Perceive, Interact, Reason: Building Tool-Augmented Visual
Agents for Spatial Reasoning"*). This is a derivative of
[AReaL](https://github.com/inclusionAI/AReaL) +
[verl-tool](https://github.com/volcengine/verl) stripped to only what we use.

- **Runtime**: GPU cluster (L20X 143 GB × 8 per node). Two supported launch
  modes:
  - **Conda envs (default going forward)**: one conda env per workspace, all
    pinned to Python 3.12 + CUDA 12.8. See **Conda-based setup** below.
  - **Singularity sif (legacy)**: drop into a base sif via
    `geo_edit/scripts/launch_cantainer.sh`, then `pip install -r requirements.txt`
    inside. Pass `IMAGE=/path/to/your.sif` to the launcher; the historical base
    image was `pytorch280-1210-v1.sif`. This mode is being phased out.
- **State**: paths are expressed relative to the repo root —
  `./pedia_model/` for weights, `./pedia_data/` for SFT/RL/eval data.
- **Install**: each env is one `pip install -r requirements.txt` (no setup scripts).
- **Cluster ops**: start Ray
  (`verl-tool/examples/train/geo_edit/ray_start_head.sh` + `ray_start_worker.sh`
  for multinode; embedded in the single-node training script).

## Conda-based setup (recommended)

Each workspace is built around an incompatible combination of `torch` / `vllm` /
`transformers`, so a single shared env is not possible. Use **three** conda
envs (geo_edit + train_tool_server share one):

```bash
# Prerequisites: miniconda installed, CUDA 12.8 driver on host.
# CUDA toolchain is pulled in as pip wheels via nvidia-* packages in
# each requirements.txt, so no host-level CUDA toolkit is required.

# ────────────────────────────────────────────────────────────────────
# Env 1 — SFT (llamafactory: torch 2.8 + transformers 4.57.1)
# ────────────────────────────────────────────────────────────────────
conda create -n pedia-sft python=3.12 -y
conda activate pedia-sft
cd llamafactory && pip install -r requirements.txt

# ────────────────────────────────────────────────────────────────────
# Env 2 — RL training (verl-tool: torch 2.8 + vllm 0.11 + flash-attn 2.7.4)
# flash-attn must be built from source against torch 2.8 (~15 min on a
# 192-CPU node with the env vars below).
# ────────────────────────────────────────────────────────────────────
conda create -n pedia-rl python=3.12 -y
conda activate pedia-rl
unset ROCR_VISIBLE_DEVICES                                # required for Ray
cd verl-tool
TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=48 NVCC_THREADS=4 \
    pip install -r requirements.txt
# Sanity check: deep_ep / deep_gemm MUST stay un-installed.
pip uninstall -y deep_ep deep_gemm 2>/dev/null || true

# ────────────────────────────────────────────────────────────────────
# Env 3 — Tool server + inference / eval
#         (geo_edit + train_tool_server: torch≥2.10 + vllm 0.17)
# ────────────────────────────────────────────────────────────────────
conda create -n pedia-tools python=3.12 -y
conda activate pedia-tools
cd geo_edit             && pip install -U -r requirements.txt
cd ../train_tool_server && pip install -r requirements.txt
```

Activating an env later:
```bash
conda activate pedia-sft     # → run llamafactory/train_v1.sh
conda activate pedia-rl      # → run verl-tool/examples/train/geo_edit/run_pedia_rl_v1_*node.sh
conda activate pedia-tools   # → run train_tool_server/scripts/launch_tool_server.sh
                             #    or  geo_edit/scripts/run_inference.sh / run_eval.sh
```

## Repository layout (4 top-level workspaces)

```
AReaL/
├── llamafactory/             # SFT training (LLaMA-Factory PyPI)
├── geo_edit/                 # tool definitions + inference + eval entry
├── train_tool_server/        # HTTP tool server for RL rollouts (forked from verl-tool)
├── verl-tool/                # vendored verl + RL training entry (renamed from verl-tool_060)
├── AGENTS.md                 # this file
├── README.md                 # paper / project README
└── LICENSE
```

### `llamafactory/`
- Single-machine SFT for `qwen3vl8b-thinking` (and `pedia_8b_SFT_v1` outputs).
- Entry: `train_v1.sh` (and `train_v2.sh`).
- Configs: `configs/pedia_sft_v{1,2}.yaml`, `configs/ds_z2_config.json` (DeepSpeed
  ZeRO-2).
- Trainer: `train_no_pil_limit_direct.py` (PIL bomb guard off + drop-overlong
  samples + nvtx workaround).
- Install: `cd llamafactory && pip install -r requirements.txt` (pulls
  `llamafactory[metrics,deepspeed]==0.9.4` + pinned torch 2.8.0).

### `geo_edit/` (tool definitions + inference + eval)
Owns the tool catalogue used by both the HTTP tool server and the inference
client. Highly stripped:
- `tool_definitions/agents/` — only 3 agents kept:
  `paddleocr_tool.py` (7 tools), `sam3.py` (6 tools), `grounding_dino.py` (1).
- `tool_definitions/functions/` — 6 CPU image-edit functions (`crop`, `label`,
  `draw_line`, `draw_path`, `bbox`, `highlight`).
- `tool_definitions/router.py` — categories, model routing.
- `agents/` — `api_agent.py` is the OpenAI-compatible inference loop.
- `data_preprocess/` — only the 3 user-facing scripts kept (`augment_traj_data`,
  `convert_trajectory_to_sft`, `iterative_sampling_generate` helpers
  `diversification`, `trajectory_filter`, `trajectory_utils`).
- `evaluation/` — `eval_unified.py` (single entry for all 13 datasets) +
  `map_trace_verifier`, `reason_map_verifier`, `openai_as_judge`,
  `trajectory_judge`.
- `models/sam3/` — vendored SAM 3.1 model code (other vendored models deleted).
- `scripts/` — only 6 entry points kept (see below).
- Install: `cd geo_edit && pip install -U -r requirements.txt` (also auto
  `pip install -e .` for Ray actors).

Entry scripts in `geo_edit/scripts/`:
- `iterative_sampling_generate.py` — SFT trajectory sampling
- `async_generate_with_tool_call_api.py` — inference + tool-call (used by
  `run_inference.sh` as the inner runner)
- `launch_cantainer.sh` — `srun + singularity shell` to enter the container
- `run_inference.sh` — auto-launches vLLM (DP=8 by default) in the background,
  waits for the endpoint, then runs inference for one dataset. Default
  `DATASET=visual_probe_easy`; override via `DATASET=<id>` for any id registered
  in `geo_edit.eval_datasets.DATASET_REGISTRY`. vLLM is killed on exit (trap).
- `run_eval.sh` — eval-only: scores inference outputs with rule-based +
  LLM-judge fallback. Default `DATASET=visual_probe_easy`. Requires
  `JUDGE_API_KEY` + `JUDGE_API_BASE`.

Both `run_inference.sh` and `run_eval.sh` default to
`MODEL_PATH=./pedia_model/PEDIA_8B_v1` and `TEMPERATURE=0`.
and require `JUDGE_API_KEY` + `JUDGE_API_BASE` env vars.

### `train_tool_server/` (HTTP tool server for RL training)
Fork of `verl-tool/verl_tool/servers/` flattened to one package. Boots a tool
backend per agent + a tool-aware router; `verl-tool`'s rollout loop POSTs
`/get_observation` to the router.

- Python package `train_tool_server/` (flat — no `servers/` sublayer):
  - `server.py` — `python -m train_tool_server.server`
  - `router.py` — `router_factory` for uvicorn
  - `ray_manager.py`, `utils.py`
  - `tools/` — only `geo_edit_function`, `geo_paddleocr`, `geo_sam3`,
    `geo_grounding_dino` plus `base.py` and `finish.py`.
- Entry: `scripts/launch_tool_server.sh` (launches 4 backends on ports
  30889-30892 + router on 30888).
- Install: `cd train_tool_server && pip install -r requirements.txt`
  (auto `pip install -e .` and `pip install -e ../geo_edit/`).
- Verification: `image_crop` / `text_ocr` / `bbox_segment` / `grounding_dino`
  all tested live in this env; GPUs 0-5 carry PaddleOCR-VL (DP=6, ~115 GB each),
  GPU 6 SAM3 (4 GB), GPU 7 Grounding-DINO (1.5 GB).

### `verl-tool/` (RL training entry, renamed from `verl-tool_060`)
Vendored verl + `verl_tool` integration for GiGPO/PPO-style RL with the tool
server above.

- `verl_tool/` — server proxies + trainer + workers
- `verl/` — upstream verl, installed editable
- `examples/train/geo_edit/` — only what's needed for pedia 8B RL:
  - `run_pedia_rl_v1_singlenode.sh` (1 × 8 GPU, embedded `ray start --head`)
  - `run_pedia_rl_v1_multinode.sh` (4 × 8 GPU, expects external Ray cluster)
  - `ray_start_head.sh`, `ray_start_worker.sh`
- Install: see `verl-tool/requirements.txt` — must run
  `unset ROCR_VISIBLE_DEVICES` before Ray and rebuild `flash-attn` from source
  for torch 2.8 (`TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=48 NVCC_THREADS=4 pip
  install -r requirements.txt`).
- **Do not install** `deep_ep` or `deep_gemm` (torch 2.10 ABI; will crash on
  torch 2.8 baseline).

## Models in `pedia_model/`

| Path | Purpose |
|---|---|
| `pedia_model/PEDIA_8B_v1`     | 8B RL-trained model (default for eval) |
| `pedia_model/pedia_8b_SFT_v1` | 8B SFT model (RL start) |
| `pedia_model/pedia_4b_v1`     | 4B RL |
| `pedia_model/pedia_2b_v1`     | 2B RL |
| `pedia_model/PaddleOCR-VL-1.5`    | tool backend (`geo_paddleocr`) |
| `pedia_model/sam3.1`              | tool backend (`geo_sam3`) — uses `sam3.1_multiplex.pt` |
| `pedia_model/grounding-dino-base` | tool backend (`geo_grounding_dino`) |

## Data in `pedia_data/`

| Path | Purpose |
|---|---|
| `pedia_data/pedia_sft_v1.tar`  | SFT training data (extract in place to `pedia_sft_v1/`) |
| `pedia_data/pedia_rl_v1.tar`   | RL training data (extract in place to `pedia_rl_v1/train.parquet` + `val.parquet`) |
| `pedia_data/eval/id_data.tar`  | 6 in-distribution eval parquets |
| `pedia_data/eval/ood_data.tar` | 7 out-of-distribution eval parquets |

## Common workflows

### Run an eval (2-node minimum — tool node + inference node)
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

# ─── Node B (or anywhere CPU): score outputs ───
JUDGE_API_KEY=... JUDGE_API_BASE=... bash geo_edit/scripts/run_eval.sh
```

### Run RL training (single node)
```bash
bash geo_edit/scripts/launch_cantainer.sh
bash train_tool_server/scripts/launch_tool_server.sh  # provides tool agents
TOOL_SERVER_URL=http://127.0.0.1:30888/get_observation \
    bash verl-tool/examples/train/geo_edit/run_pedia_rl_v1_singlenode.sh
```

### Run SFT
```bash
bash geo_edit/scripts/launch_cantainer.sh
cd llamafactory && bash train_v1.sh           # 8 GPU SFT with DeepSpeed ZeRO-2
```

## Conventions

- **Path discipline**: All non-code state lives under repo-relative
  `./pedia_model/`, `./pedia_data/`, and `./outputs/{mixed_rl,eval_results,
  eval_logs}/`. Code never hard-codes absolute system paths; configure base
  dirs via env vars (e.g. `PEDIA_MODEL`, `PEDIA_DATA`) if you need to point
  elsewhere.
- **Default runtime**: 3 conda envs (`pedia-sft` / `pedia-rl` / `pedia-tools`),
  Python 3.12 + CUDA 12.8 via nvidia-* pip wheels. Singularity sif is legacy.
- **No `setup_env_*.sh`**: every env install is captured in its own
  `requirements.txt` including editable installs.
- **No wandb**: training scripts log to console only (logger='console').
- **Tool server bug to remember**: after flattening `servers/` we had to
  reduce `_AREAL_ROOT` in `train_tool_server/train_tool_server/tools/
  geo_edit_base.py` from 4 `..` to 3 `..`. Don't reintroduce the 4-level
  assumption when refactoring.

## Things that intentionally don't exist

Removed from upstream AReaL/verl-tool because they aren't used:
- AReaL `areal/`, `realhf/`, `csrc/`, `evaluation/`, `examples/`, `recipe/`,
  `functioncall/`, `benchmark/`, `assets/`, `notebook/`, `docs/`, `wandb/`,
  `outputs/`
- AReaL meta files: `CONTRIBUTING.md`, `Dockerfile`, `LEGAL.md`, `MANIFEST.in`,
  `pyproject.toml`, `README_AReaL.md`, `ROADMAP.md`
- verl-tool: most `examples/train/*` (kept only `geo_edit/`), `assets/`,
  `benchmarks/`, `patches/`, `scripts/`, `eval_service/`, `outputs/`, `wandb/`,
  internal `LlamaFactory/`
- geo_edit: `baseline/`, `tests/`, `assets/`, `qualitative_example/`, `images/`,
  `docs/`, deprecated agent files (chartmoe, chartr1, gllava, multimath, ovr,
  thinkmorph) + their models, dataset-specific preprocess + eval scripts (we
  use `eval_unified.py` now)
- Tool agents reduced from 8 to 3 (paddleocr + sam3 + grounding_dino). The
  `geo_paddleocr` PaddleOCR-VL backend exposes 7 sub-tools.

If you need to bring something back, check git history of the upstream
repos — nothing here is intentionally salvaged.
