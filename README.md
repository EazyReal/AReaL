<h1 align="center"> PERIA: PERception-Interaction-reason Agent for Spatial Reasoning </h1>

This repository releases the official implementation of [**PERIA: Perceive, Interact, Reason — Building Tool-Augmented Visual Agents for Spatial Reasoning**]().

<!-- TODO: project page banner / teaser image -->

## Abstract

While recent vision-language models (VLMs) demonstrate strong multimodal
understanding, they remain limited in spatial reasoning tasks that require active
evidence acquisition and multi-step visual interaction. This limitation suggests that
relying solely on implicit visual representations from vision encoders is insufficient
for recovering fine-grained spatial evidence. We introduce **PERception-Interaction-
reason Agent (PERIA)**, a tool-augmented visual agent for spatial reasoning tasks
across map reasoning, visual probing, and vision reconstruction. PERIA uses two
lightweight tool families: **vision perception tools** for exposing textual, symbolic,
and spatial evidence, and **vision interaction tools** for manipulating visual context,
tracing paths, and verifying spatial relations. To train PERIA, we develop a
unified recipe that combines supervised tool-use trajectory synthesis, composite
rewards, and **Observation-Relaxed Group-in-Group Policy Optimization (OR-
GIGPO)** for effective multi-tool behavior. Experiments on 13 benchmarks from 8
datasets show that **PERIA-8B improves over the Qwen3-8B backbone by 10.0%
on in-distribution benchmarks and 4.4% on out-of-distribution benchmarks**, while
outperforming previous state-of-the-art baselines of similar size by 7.0%–14.8%.
It also achieves performance comparable to much larger models such as Qwen3-
VL-235B-A22B-Thinking and GPT-5, demonstrating the effectiveness of PERIA
in enhancing spatial reasoning capabilities.

<!-- TODO: project page link -->

### Table of Contents  <!-- omit in toc -->

  - [PERIA: PERception-Interaction-reason Agent for Spatial Reasoning](#peria-perception-interaction-reason-agent-for-spatial-reasoning)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Dataset and Models](#dataset-and-models)
  - [Evaluation](#evaluation)
  - [SFT Training](#sft-training)
  - [RL Training](#rl-training)
  - [SFT Data Synthesis](#sft-data-synthesis)
  - [Acknowledgment](#acknowledgment)

## Introduction

<!-- TODO: fill in motivation + method overview + key results -->

<!-- TODO: images/teaser.png -->
<!-- TODO: images/method.png -->
<!-- TODO: images/results.png -->

## Installation

```bash
git clone <repo-url> PERIA
cd PERIA
```

The repo is split into four workspaces with mutually incompatible
`torch` / `vllm` / `transformers` versions, so we use **three conda envs**:
`peria-sft`, `peria-rl`, and `peria-tools`. Each stage below ships with the
env-setup snippet it needs — install only the ones for the stages you actually
run.

> **Note**: if `pip` cannot reach HuggingFace, use a mirror first:
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

## Dataset and Models

All data and weights live at the repo root, under `./pedia_data/` and
`./pedia_model/`. The full contents are mirrored on HuggingFace as two
repositories:

- `<your-org>/PERIA-Models` &nbsp;— model repo (everything under `./pedia_model/`)
- `<your-org>/PERIA-Data` &nbsp;&nbsp;— dataset repo (everything under `./pedia_data/`)

**Each stage section below downloads only the subset it needs**, using
`hf download ... --include "<glob>/*"` so you never pull files you won't use.
For reference, the full repo layout (once all stages have run) is:

```
./pedia_model/                                            (<your-org>/PERIA-Models)
├── Qwen3-VL-8B-Thinking/    base VLM       (SFT, SFT Data Synthesis)
├── pedia_8b_SFT_v1/         8B SFT ckpt    (RL start)
├── PEDIA_8B_v1/             8B RL ckpt     (default eval target)
├── PaddleOCR-VL-1.5/        tool backend   (Eval, RL, Data Synthesis)
├── sam3.1/                  tool backend   (Eval, RL, Data Synthesis)
└── grounding-dino-base/     tool backend   (Eval, RL, Data Synthesis)

./pedia_data/                                             (<your-org>/PERIA-Data)
├── pedia_sft_v1/            SFT data        (SFT)
├── pedia_rl_v1/             RL data         (RL)
└── eval/
    ├── id/<dataset>.parquet  in-distribution eval (Eval)
    └── ood/<dataset>.parquet out-of-distribution eval (Eval)
```

Install the HF CLI once and (optionally) point at a mirror:

```bash
pip install -U huggingface_hub
# (optional) HF mirror if direct download is slow:
# export HF_ENDPOINT=https://hf-mirror.com
```

## Evaluation

PERIA evaluation = **vLLM serving the PERIA model + tool server providing 18
tools (3 GPU agents + 6 CPU function tools) + the eval driver
`async_generate_with_tool_call_api.py`**.

[`run_inference.sh`](file:///storage/openpsi/users/lichangye.lcy/antoinegg1/AReaL/geo_edit/scripts/run_inference.sh)
auto-launches vLLM with `DP=8` (all 8 local GPUs) in the background, then runs
inference, then stops vLLM on exit. Because vLLM claims all 8 GPUs, the tool
server runs on a **separate node** — same 2-node minimum as RL training.

### Download — PERIA-8B + tool backends + the eval dataset you want

```bash
# PERIA-8B-RL checkpoint + the 3 tool backends (~38 GB total)
hf download <your-org>/PERIA-Models \
    --include "PEDIA_8B_v1/*" "PaddleOCR-VL-1.5/*" "sam3.1/*" "grounding-dino-base/*" \
    --local-dir ./pedia_model

# Eval parquet for the dataset you want to score (default example: visual_probe_easy).
# To score a different benchmark, swap the --include glob:
#   eval/id/visual_probe_{easy,medium,hard}.parquet | eval/id/reason_map{,_plus}.parquet
#   eval/id/map_trace.parquet
#   eval/ood/visworld_{cube,mmsi,ballgame,paperfolding}.parquet
#   eval/ood/{mapeval_visual,babyvision,vstar_bench}.parquet
hf download <your-org>/PERIA-Data --repo-type dataset \
    --include "eval/id/visual_probe_easy.parquet" \
    --local-dir ./pedia_data
```

### Env setup — `peria-tools` (torch ≥ 2.10 + vllm 0.17)

```bash
conda create -n peria-tools python=3.12 -y
conda activate peria-tools
cd geo_edit          && pip install -U -r requirements.txt && cd ..
cd train_tool_server && pip install -r requirements.txt    && cd ..
```

### 1 ─ Tool server on Node A (all 8 GPUs)

```bash
conda activate peria-tools
unset ROCR_VISIBLE_DEVICES
ray start --head --port=6379 --num-gpus=8 --resources='{"tool_agent": 8}'

# Spawns paddleocr (2 replicas) + sam3 + grounding_dino + geo_edit_function (CPU)
# backends and a tool-aware router on port 30888.
bash train_tool_server/scripts/launch_tool_server.sh

# Note Node A's reachable IP for Node B
echo "Node A IP for Ray join: $(hostname -I | awk '{print $1}'):6379"
```

### 2 ─ Run inference on Node B (auto-launches vLLM DP=8)

```bash
conda activate peria-tools
unset ROCR_VISIBLE_DEVICES
ray start --address=<node-a-ip>:6379          # join Node A's Ray cluster

# Defaults to DATASET=visual_probe_easy. The script launches vLLM (DP=8),
# waits for the endpoint, runs inference, then kills vLLM on exit.
# Dataset id auto-resolves to its parquet path + eval template via the
# geo_edit.eval_datasets.DATASET_REGISTRY.
bash geo_edit/scripts/run_inference.sh
# DATASET=reason_map bash geo_edit/scripts/run_inference.sh    # any registered id
# DP_SIZE=4 VLLM_PORT=8001 bash geo_edit/scripts/run_inference.sh  # tweak vLLM
```

### 3 ─ Score outputs (CPU only)

```bash
bash geo_edit/scripts/run_eval.sh
# DATASET=reason_map bash geo_edit/scripts/run_eval.sh
```

By default `run_eval.sh` does **rule-based scoring only** — fast, no external
API, good for quick sanity checks. To reproduce the paper numbers, enable the
LLM-judge fallback by exporting `JUDGE_API_KEY`. `JUDGE_API_BASE` defaults to
OpenAI's official endpoint (`https://api.openai.com/v1`), so for OpenAI you
only need the key:

```bash
export JUDGE_API_KEY=<your-openai-key>
# export JUDGE_API_BASE=<custom-endpoint>     # only for non-OpenAI providers
bash geo_edit/scripts/run_eval.sh
```

Raw inference outputs land in `./outputs/eval_results/<dataset>/<model_name>/`;
per-dataset accuracy + summary go to `./outputs/eval_output/<dataset>/<model_name>/`.

> **Single-node alternative**: set `DP_SIZE=4` on the inference script
> to leave 4 GPUs free, then run `CUDA_VISIBLE_DEVICES=0,1,2,3 bash
> train_tool_server/scripts/launch_tool_server.sh` on the same node before
> `CUDA_VISIBLE_DEVICES=4,5,6,7 DP_SIZE=4 bash geo_edit/scripts/run_inference.sh`.

## SFT Training

### Download — base VLM + SFT data

```bash
# Base VLM (Qwen3-VL-8B-Thinking is the SFT starting point)
hf download <your-org>/PERIA-Models \
    --include "Qwen3-VL-8B-Thinking/*" \
    --local-dir ./pedia_model

# SFT training data (train.json + images/)
hf download <your-org>/PERIA-Data --repo-type dataset \
    --include "pedia_sft_v1/*" \
    --local-dir ./pedia_data
```

### Env setup — `peria-sft` (torch 2.8 + transformers 4.57.1)

```bash
conda create -n peria-sft python=3.12 -y
conda activate peria-sft
cd llamafactory && pip install -r requirements.txt && cd ..
```

SFT uses LLaMA-Factory on a single 8-GPU node.

```bash
conda activate peria-sft
cd llamafactory
bash train_v1.sh
```

`train_v1.sh` is a thin wrapper around
`torchrun --standalone --nnodes=1 --nproc_per_node=8 train_no_pil_limit_direct.py configs/pedia_sft_v1.yaml`,
with patches for over-long multimodal samples + DeepSpeed ZeRO-2.

The checkpoint is written to `./pedia_model/pedia_8b_SFT_v1/`.
Hyper-parameters and dataset paths live in `llamafactory/configs/pedia_sft_v1.yaml`.

## RL Training

RL fine-tunes the SFT checkpoint with **OR-GIGPO** against the tool server.
**RL requires at least two 8-GPU nodes** — one dedicated to the tool server
(GPU-heavy: PaddleOCR-VL replicas + SAM 3.1 + Grounding-DINO) and one running
the training loop. The two cannot share a node without OOM.

### Download — SFT checkpoint + RL data + tool backends

```bash
# SFT checkpoint (RL starting point) + 3 tool backends (skip the tool
# globs if you already downloaded them in Evaluation).
hf download <your-org>/PERIA-Models \
    --include "pedia_8b_SFT_v1/*" "PaddleOCR-VL-1.5/*" "sam3.1/*" "grounding-dino-base/*" \
    --local-dir ./pedia_model

# RL training data (train.parquet + val.parquet + images/)
hf download <your-org>/PERIA-Data --repo-type dataset \
    --include "pedia_rl_v1/*" \
    --local-dir ./pedia_data
```

### Prerequisites

1. SFT checkpoint at `./pedia_model/pedia_8b_SFT_v1/` (download above or
   train via [SFT Training](#sft-training)).
2. RL data at `./pedia_data/pedia_rl_v1/{train,val}.parquet`.
3. Tool models under `./pedia_model/` (PaddleOCR-VL-1.5, sam3.1, grounding-dino-base).

### Minimum setup — 1 tool node + 1 training node

**Node A — tool server** (uses `peria-tools` from Evaluation):

```bash
conda activate peria-tools
unset ROCR_VISIBLE_DEVICES
ray start --head --port=6379 --num-gpus=8 --resources='{"tool_agent": 8}'
bash train_tool_server/scripts/launch_tool_server.sh

# Take note of this node's reachable IP for node B
TOOL_SERVER_IP=$(hostname -I | awk '{print $1}')
echo "TOOL_SERVER_URL=http://${TOOL_SERVER_IP}:30888/get_observation"
```

**Node B — RL training:**

```bash
# Env setup — peria-rl (torch 2.8 + vllm 0.11 + flash-attn 2.7.4)
conda create -n peria-rl python=3.12 -y
conda activate peria-rl
unset ROCR_VISIBLE_DEVICES
cd verl-tool
TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=48 NVCC_THREADS=4 \
    pip install -r requirements.txt
pip uninstall -y deep_ep deep_gemm 2>/dev/null || true
cd ..

# Kick off training, pointing at node A's tool server
TOOL_SERVER_URL=http://<node-a-ip>:30888/get_observation \
    bash verl-tool/examples/train/geo_edit/run_pedia_rl_v1_singlenode.sh
```

Outputs land under `./outputs/mixed_rl/`.

The reward manager can optionally call an LLM judge for trajectory rewards.
This is opt-in via `JUDGE_API_KEY`; `JUDGE_API_BASE` defaults to OpenAI's
official endpoint, so for OpenAI you only need the key:

```bash
export JUDGE_API_KEY=<your-openai-key>
# export JUDGE_API_BASE=<custom-endpoint>     # only for non-OpenAI providers
TOOL_SERVER_URL=http://<node-a-ip>:30888/get_observation \
    bash verl-tool/examples/train/geo_edit/run_pedia_rl_v1_singlenode.sh
```

### Larger multi-node training

For scaling beyond a single training node (e.g. 4 × 8 GPU + tool node), we also
provide `run_pedia_rl_v1_multinode.sh` and the matching
`ray_start_{head,worker}.sh` helpers under
`verl-tool/examples/train/geo_edit/`.

## SFT Data Synthesis

(Uses the `peria-tools` env from the Evaluation section — install that first.)

We provide three composable scripts under `geo_edit/`:

1. **Iterative trajectory sampling** — generate raw tool-use trajectories on
   a base VLM (e.g. Qwen3-VL-8B-Thinking) over a target dataset.
2. **Trajectory augmentation** — diversify + reward-filter the raw trajectories.
3. **SFT format conversion** — turn the filtered trajectories into the
   LLaMA-Factory SFT JSON layout.

### Download — base VLM + tool backends + source dataset

```bash
# Base VLM + 3 tool backends (skip globs you've already pulled in earlier stages)
hf download <your-org>/PERIA-Models \
    --include "Qwen3-VL-8B-Thinking/*" "PaddleOCR-VL-1.5/*" "sam3.1/*" "grounding-dino-base/*" \
    --local-dir ./pedia_model

# Source dataset to sample trajectories over (ReasonMap-Plus example — third-party repo)
hf download <your-org>/ReasonMap-Plus --repo-type dataset \
    --local-dir ./pedia_data/raw/reasonmap_plus
```

### Example — synthesise SFT data from ReasonMap-Plus

SFT data synthesis **requires an LLM-judge API key** — the trajectory
filter and augmentation stages call the judge to score candidate
trajectories, so unlike eval/RL the key is mandatory. `JUDGE_API_BASE`
defaults to OpenAI's official endpoint; override only for non-OpenAI
providers.

```bash
conda activate peria-tools
export JUDGE_API_KEY=<your-openai-key>             # REQUIRED for data synthesis
# export JUDGE_API_BASE=<custom-endpoint>          # only for non-OpenAI providers

# 1. Start tool server on a dedicated node + serve Qwen3-VL-8B-Thinking with
#    run_inference.sh's auto-launched vLLM (set MODEL_PATH accordingly).
#    See Evaluation steps 1-2; swap MODEL_PATH=./pedia_model/Qwen3-VL-8B-Thinking.

# 2. Iterative sampling — generates ./outputs/trajectories/reason_map_plus/
python -m geo_edit.scripts.iterative_sampling_generate \
    --api_base http://127.0.0.1:8000 \
    --dataset_path ./pedia_data/raw/reasonmap_plus/train.parquet \
    --dataset_name reason_map_plus \
    --output_dir ./outputs/trajectories/reason_map_plus \
    --model_name_or_path ./pedia_model/Qwen3-VL-8B-Thinking \
    --model_type vLLM --temperature 0.7 \
    --use_tools auto --enable_tools map general --max_tool_calls 10

# 3. Filter + diversify (reward-driven, drops low-quality trajectories)
python -m geo_edit.data_preprocess.augment_traj_data \
    --src_dir ./outputs/trajectories/reason_map_plus \
    --dst_dir ./outputs/trajectories/reason_map_plus_augmented

# 4. Convert to LLaMA-Factory SFT format → drop into ./pedia_data/pedia_sft_v1/
python -m geo_edit.data_preprocess.convert_trajectory_to_sft \
    --src_dir ./outputs/trajectories/reason_map_plus_augmented \
    --dst_dir ./pedia_data/pedia_sft_v1
```

After step 4 you can re-run [SFT Training](#sft-training) on the freshly
synthesised data.

## Citation

```bibtex
@article{peria2026,
    author  = {<TODO authors>},
    title   = {Perceive, Interact, Reason: Building Tool-Augmented Visual Agents for Spatial Reasoning},
    journal = {arXiv},
    year    = {2026}
}
```

## Acknowledgment

This repository benefits from
[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL),
[verl](https://github.com/volcengine/verl),
[verl-tool](https://github.com/volcengine/verl),
[AReaL](https://github.com/inclusionAI/AReaL),
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory),
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR),
[SAM 3.1](https://github.com/facebookresearch/sam2),
and [Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO).
Thanks to the authors for releasing these excellent codebases.
