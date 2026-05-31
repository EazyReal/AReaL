# llamafactory/

Single-machine, 8-GPU SFT training kit for **Qwen3-VL-8B-Thinking** variants.

Two ready-to-run experiments:

| Run | Dataset | Samples | Config | Launcher | Output (trained model) |
|---|---|---:|---|---|---|
| **v1** | `pedia_sft_v1` | 10,661 | [`configs/pedia_sft_v1.yaml`](configs/pedia_sft_v1.yaml) | [`train_v1.sh`](train_v1.sh) | [`pedia_model/pedia_sft_v1/`](file://./pedia_model/pedia_sft_v1) |
| **v2** | `pedia_sft_v2` | 13,398 | [`configs/pedia_sft_v2.yaml`](configs/pedia_sft_v2.yaml) | [`train_v2.sh`](train_v2.sh) | [`pedia_model/pedia_sft_v2/`](file://./pedia_model/pedia_sft_v2) |

Both runs use DeepSpeed ZeRO-2, read from the consolidated `pedia_data/` layout, and ship with three production patches needed for safe multimodal training (see [Patches](#patches)).

## Layout

```
llamafactory/
├── README.md                                                 (this file)
├── requirements.txt                                          (all Python deps, one-shot pip install)
├── train_no_pil_limit_direct.py                              (training entry + 3 runtime patches)
├── train_v1.sh                                               (single-machine 8-GPU launcher, v1)
├── train_v2.sh                                               (single-machine 8-GPU launcher, v2)
├── configs/
│   ├── ds_z2_config.json                                     (DeepSpeed ZeRO-2 settings)
│   ├── pedia_sft_v1.yaml
│   └── pedia_sft_v2.yaml
└── logs/                                                     (auto-created at run time)
```

## Environment setup

One command. Verified target: Python 3.12, CUDA 12.8, 8x GPU node.

```bash
cd llamafactory
pip install -r requirements.txt
```

That single line installs torch+torchvision+torchaudio, tensorboard, and `llamafactory==0.9.4` from PyPI together with its `[metrics]` and `[deepspeed]` extras (nltk / jieba / rouge-chinese / deepspeed). No `git clone` step needed.

> For a non-cu128 driver, override the PyTorch index URL:
> `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124` (or your CUDA version).

> The training entry script monkey-patches `llamafactory.data.processor.supervised` (see [Patches](#patches)). The patched API surface (`SupervisedDatasetProcessor._encode_data_example` + `preprocess_dataset`) is identical between PyPI 0.9.4 and the cluster's prior 0.9.5.dev0, so the patch behaves the same.

### ⚠️ Version drift vs prior cluster install

PyPI `llamafactory==0.9.4` bakes in stricter dependency caps than the 0.9.5.dev0 HEAD editable install previously running on the cluster. Installing this kit will **force these downgrades**:

| package | cluster (prior) | after `pip install -r requirements.txt` | reason |
|---|---|---|---|
| transformers | 5.0.0.dev0 | <= 4.57.1 (likely 4.57.1) | 0.9.4 pyproject cap |
| peft | 0.18.1 | <= 0.17.1 | 0.9.4 pyproject cap |
| deepspeed | 0.19.0 | <= 0.16.9 | `[deepspeed]` extra cap in 0.9.4 |

Re-run both `train_v{1,2}.sh` after install to re-verify SFT stability before trusting longer runs. If a downgrade breaks training, fall back to the editable install from HEAD (see [Alternative](#alternative-editable-install-from-head)).

### Verified version stack

What pip will resolve out of `pip install -r requirements.txt`. Exact patch versions for LlamaFactory deps depend on pip's resolver - the ceilings are what 0.9.4 declares.

| package | version | source |
|---|---|---|
| torch | 2.8.0 | requirements.txt |
| torchvision | 0.23.0 | requirements.txt |
| torchaudio | 2.8.0 | requirements.txt |
| tensorboard | 2.16.2 | requirements.txt |
| llamafactory | 0.9.4 | requirements.txt (PyPI) |
| nltk, jieba, rouge-chinese | latest | `llamafactory[metrics]` |
| deepspeed | <= 0.16.9 | `llamafactory[deepspeed]` |
| transformers | <= 4.57.1 (!= 4.52.0, 4.57.0) | LlamaFactory 0.9.4 deps |
| peft | <= 0.17.1 | LlamaFactory 0.9.4 deps |
| accelerate | <= 1.11.0 | LlamaFactory 0.9.4 deps |
| trl | <= 0.24.0 | LlamaFactory 0.9.4 deps |
| datasets | <= 4.0.0 | LlamaFactory 0.9.4 deps |
| Pillow | latest | LlamaFactory 0.9.4 deps (via torchvision) |

### Alternative: editable install from HEAD

If you need transformers 5.x / peft 0.18.1 / deepspeed 0.19 (the cluster's pre-existing stack), skip `llamafactory` from `requirements.txt` and install LlamaFactory from source instead:

```bash
# 1) torch + tensorboard (drop the `llamafactory[...]` line from requirements.txt)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 tensorboard==2.16.2 deepspeed==0.19.0

# 2) LlamaFactory HEAD, editable, plus the 3 metrics packages
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
pip install -r requirements/metrics.txt
```

## Dataset prerequisites

Both YAMLs expect the consolidated `pedia_data/` dataset layout (already in place on the cluster):

| Run | `dataset_dir` |
|---|---|
| v1 | `./pedia_data/pedia_sft_v1/` |
| v2 | `./pedia_data/pedia_sft_v2/` |

Each folder contains `train.json` + `dataset_info.json` (whose top-level key matches the dataset name, e.g. `pedia_sft_v1`). All image paths inside `train.json` are absolute paths under `./pedia_data/images/<source>/`.

Base model (referenced by both YAMLs):

- `./pedia_model/Qwen3-VL-8B-Thinking`

## Trained outputs

Each run's `output_dir` points at the canonical model archive under `pedia_model/`. After our cleanup the dirs hold only inference-ready weights + tokenizer + config (~16.3 GB each, DeepSpeed `global_step*` state + intermediate `checkpoint-*` subdirs stripped):

| Model | Origin | Tokenizer |
|---|---|---|
| [`pedia_model/pedia_sft_v1/`](file://./pedia_model/pedia_sft_v1) | exported from former `sft_workspace/qwen3vl8b-thinking-5ds-v2-0419-ct65536/checkpoint-280/` | as-trained, byte-identical to base Qwen3-VL-8B-Thinking |
| [`pedia_model/pedia_sft_v2/`](file://./pedia_model/pedia_sft_v2) | exported from former `sft_workspace/qwen3vl8b-thinking-5ds-v4-0526-ct65536-lr1e5/checkpoint-419/` | re-aligned to base after migration (same `vocab_size=151936`, more compact serialization) |

Re-running `train_v{1,2}.sh` will write new `checkpoint-N/` subdirs into these same dirs (auto-resume picks up the latest if present). Override `output_dir=...` via the `--` CLI mechanism (below) if you want a fresh location.

## Training

```bash
cd llamafactory

bash train_v1.sh        # 10,661 samples
bash train_v2.sh        # 13,398 samples
```

Under the hood both launchers run `torchrun --standalone --nnodes=1 --nproc_per_node=8 train_no_pil_limit_direct.py <config>.yaml`. Logs are tee'd to `logs/<config>.<timestamp>.log`.

### Auto-resume

If `output_dir/checkpoint-*` already exists, the launcher detects the latest checkpoint and appends `resume_from_checkpoint=<path>` automatically. To start fresh, point `output_dir` somewhere empty (see overrides below).

### Environment overrides

```bash
NPROC=4 bash train_v1.sh                    # 4 GPUs instead of 8
MASTER_PORT=29503 bash train_v2.sh          # change torchrun rendezvous port
```

Defaults: `NPROC=8`, `MASTER_PORT=29501` (v1) / `29502` (v2) - distinct ports so the two runs can be launched in parallel.

### LlamaFactory key=value overrides

Any args after `--` are forwarded verbatim to LlamaFactory:

```bash
bash train_v1.sh -- output_dir=/tmp/run1 learning_rate=5e-6 num_train_epochs=2
bash train_v2.sh -- save_steps=100 logging_steps=5
```

## Patches

Three runtime patches applied in `train_no_pil_limit_direct.py` BEFORE `run_exp()`:

1. **PIL bomb guard disabled** - `PIL.Image.MAX_IMAGE_PIXELS = None`. Vision-RL data routinely exceeds PIL's default decompression limit and would otherwise raise.

2. **Drop, don't truncate, over-long samples** - replaces `SupervisedDatasetProcessor._encode_data_example` so any sample whose UNTRUNCATED token length exceeds `cutoff_len` is DROPPED. The vanilla implementation truncates, which for multimodal models slices out `<|image_pad|>` placeholders while vision tokens (computed from `image_grid_thw`) are still generated in full - this corrupts GPU memory and triggers `CUDA illegal memory access` deep inside a later forward pass, far from the real cause.

3. **DeepSpeed nvtx fallback** - the installed `nvtx==0.2.11` lacks `Domain.push_range`, but DeepSpeed 0.19.0 calls it unconditionally on successful import. Nulling out `_ds_cuda_acc.nvtx` forces the fallback path to `torch.cuda.nvtx`.

## Differences from the original multi-node setup

Original scripts under `./outputs/legacy_sft_workspace/` use `deepspeed --hostfile` or multi-node `torchrun` across **4 nodes x 8 GPUs = 32 GPUs total** (effective batch = `nodes x per_node_gpus x per_device_train_batch_size x gradient_accumulation_steps` = `4 x 8 x 1 x 1 = 32`).

The single-machine YAMLs here run on **1 node x 8 GPUs**, so they bump `gradient_accumulation_steps` from `1` to `4` to keep the same **effective batch = 32** (`1 x 8 x 1 x 4 = 32`). Total optimizer steps are therefore identical to the original (v1: 334, v2: 419), and `trainer_state.json` from the original runs confirms the math.

Other deltas vs the original cluster runs:

- read from `pedia_data/pedia_8b_SFT_v{1,2}/` (the renamed-and-consolidated layout) instead of the original `sft_workspace/batch_0414/data/merged_5ds_v{2,4}_*_sft/`;
- write outputs to `pedia_model/pedia_8b_SFT_v{1,2}/` instead of the original `sft_workspace/qwen3vl8b-thinking-5ds-v{2,4}-*-ct65536*/`;
- have `resume_from_checkpoint` removed (the launcher discovers the latest checkpoint automatically);
- reference `configs/ds_z2_config.json` with a relative path (the launcher `cd`s into `llamafactory/` first).

All other hyperparameters - model path, `cutoff_len`, learning rate, `per_device_train_batch_size`, save cadence, warmup, bf16 - are identical to the original cluster runs.
