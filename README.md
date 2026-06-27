# AReaL Loss Aggregation Run Proofs

Static proof artifacts for AReaL loss aggregation runtime runs on commit
`86b48e5c4`.

## 2026-06-27

These plots are generated from captured AReaL logs and use a trailing 5-step
moving average over `ppo_actor/task_reward/avg`.

The run is a controlled arithmetic overfit demo through the normal
PPOTrainer/FSDP/SGLang/weight-update path. It is meant to show that each loss
aggregation mode can carry a learning signal; it is not a GSM8K convergence
benchmark. The linked W&B runs are sanitized public replays of the captured
reward metrics, with only public-safe config fields.

| Mode | Updates | First-5 reward avg | Last-5 reward avg | W&B |
| --- | ---: | ---: | ---: | --- |
| `token_mean` | 40 | 0.378 | 0.919 | [run](https://wandb.ai/augustinevmax-vmax/areal-loss-aggregation-public/runs/lossagg-public-token-mean-train) |
| `seq_mean` | 40 | 0.341 | 0.897 | [run](https://wandb.ai/augustinevmax-vmax/areal-loss-aggregation-public/runs/lossagg-public-seq-mean-train) |
| `prompt_mean` | 40 | 0.419 | 0.763 | [run](https://wandb.ai/augustinevmax-vmax/areal-loss-aggregation-public/runs/lossagg-public-prompt-mean-train) |
| `constant` | 80 | 0.384 | 1.000 | [run](https://wandb.ai/augustinevmax-vmax/areal-loss-aggregation-public/runs/lossagg-public-constant-train) |

Images:

- [`lossagg_reward_moving_average_all.png`](lossagg-run-proofs/2026-06-27/lossagg_reward_moving_average_all.png)
- [`lossagg_reward_moving_average_token_mean.png`](lossagg-run-proofs/2026-06-27/lossagg_reward_moving_average_token_mean.png)
- [`lossagg_reward_moving_average_seq_mean.png`](lossagg-run-proofs/2026-06-27/lossagg_reward_moving_average_seq_mean.png)
- [`lossagg_reward_moving_average_prompt_mean.png`](lossagg-run-proofs/2026-06-27/lossagg_reward_moving_average_prompt_mean.png)
- [`lossagg_reward_moving_average_constant.png`](lossagg-run-proofs/2026-06-27/lossagg_reward_moving_average_constant.png)

Data slice:

- [`easy8.jsonl`](lossagg-run-proofs/2026-06-27/easy8.jsonl)

Command template from the AReaL repo root:

```bash
DATA=/tmp/areal-lossagg-easy8.jsonl
MODEL=Qwen/Qwen2.5-0.5B-Instruct
MODE=token_mean  # token_mean, seq_mean, prompt_mean, or constant

curl -L -o "$DATA" \
  https://raw.githubusercontent.com/EazyReal/AReaL/refs/heads/codex/lossagg-run-proofs-20260626/lossagg-run-proofs/2026-06-27/easy8.jsonl

EXTRA_ARGS=(total_train_epochs=40 ++total_train_steps=40)
if [ "$MODE" = constant ]; then
  EXTRA_ARGS=(total_train_epochs=80 ++total_train_steps=80 ++actor.loss_aggregation_divisor=16)
fi

uv run python examples/math/boba_grpo.py \
  --config examples/math/boba_grpo.yaml \
  experiment_name=lossagg-boba-demo \
  trial_name="$MODE" \
  cluster.n_gpus_per_node=2 \
  scheduler.type=local \
  rollout.backend=sglang:d1 \
  rollout.max_concurrent_rollouts=64 \
  rollout.dump_to_file=false \
  rollout.enable_rollout_tracing=false \
  actor.backend=fsdp:d1 \
  actor.path="$MODEL" \
  ++actor.attn_impl=sdpa \
  ++actor.use_kernels=false \
  ref.path="$MODEL" \
  tokenizer_path="$MODEL" \
  sglang.model_path="$MODEL" \
  ++sglang.disable_cuda_graph=true \
  train_dataset.path="$DATA" \
  train_dataset.batch_size=8 \
  train_dataset.pin_memory=false \
  gconfig.n_samples=8 \
  gconfig.max_new_tokens=32 \
  gconfig.max_tokens=320 \
  actor.mb_spec.max_tokens_per_mb=8192 \
  actor.ppo_n_minibatches=1 \
  ++actor.loss_aggregation="$MODE" \
  saver.freq_epochs=null \
  saver.freq_steps=null \
  saver.freq_secs=null \
  perf_tracer.enabled=false \
  perf_tracer.session_tracer.enabled=false \
  stats_logger.wandb.mode=disabled \
  "${EXTRA_ARGS[@]}"
```
