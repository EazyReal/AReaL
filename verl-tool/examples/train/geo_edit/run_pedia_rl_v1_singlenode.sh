#!/usr/bin/env bash
set -x

# ============================================================
# Single-node (1×8 GPU) RL training for pedia 8B v1.
# Ray head is started inline (no separate ray_start_*.sh needed).
#
# Environment variables (optional):
#   WORKSPACE        – default ./outputs/mixed_rl
#   MODEL_PATH       – SFT checkpoint (default pedia_8b_SFT_v1)
#   TOOL_SERVER_URL  – default http://127.0.0.1:30888/get_observation
#   JUDGE_API_KEY / JUDGE_API_BASE / JUDGE_MODEL
# ============================================================

WORKSPACE=${WORKSPACE:-./outputs/mixed_rl}
model_name=${MODEL_PATH:-./pedia_model/pedia_8b_SFT_v1}

train_data="./pedia_data/pedia_rl_v1/train.parquet"
val_data="./pedia_data/pedia_rl_v1/val.parquet"
run_name="pedia-rl-v1-1node"
rl_alg=gigpo
gigpo_sim_threshold=0.9

# ---- Cluster topology (single node) ----
n_gpus_per_node=8
n_nodes=1

# ---- Batch sizes (scaled to 1/4 of 4-node version) ----
n=2
batch_size=16
ppo_mini_batch_size=64

# ---- Sequence lengths ----
max_prompt_length=16384
max_response_length=32768
max_action_length=4096
max_obs_length=8192
max_obs_length_image=8192
max_obs_length_text=6144
ppo_max_token_len_per_gpu=$(expr $max_prompt_length + $max_response_length)

# ---- Sampling ----
temperature=1.0
top_p=1.0

# ---- Agent / tool ----
enable_agent=True
action_stop_tokens='</action>'
max_turns=10
mask_observations=True
enable_mtrl=True
additional_eos_token_ids=[151645]
reward_manager=geo_vision_qa

# ---- Training ----
strategy="fsdp2"
lr=1e-6
kl_loss_coef=0.0
kl_coef=0.0
entropy_coeff=0
kl_loss_type=low_var_kl

# ---- Per-GPU micro batches ----
ppo_micro_batch_size_per_gpu=4
log_prob_micro_batch_size_per_gpu=16

# ---- Parallelism ----
tensor_model_parallel_size=1
ulysses_sequence_parallel_size=1
fsdp_size=-1

# ---- Memory ----
gpu_memory_utilization=0.8
do_offload=False
use_dynamic_bsz=True

# ---- Rollout ----
max_num_batched_tokens=$(expr $max_prompt_length + $max_response_length)
rollout_mode='async'

# ---- Schedule ----
total_epochs=3
save_freq=10
test_freq=20

# ============================================================
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=WARN
unset ROCR_VISIBLE_DEVICES
mkdir -p $WORKSPACE/logs/$run_name

action_stop_tokens_file="$WORKSPACE/logs/$run_name/action_stop_tokens.txt"
echo -e -n "$action_stop_tokens" | tee $action_stop_tokens_file

# ---- Start local Ray head (1 node, all 8 GPUs) ----
ray stop --force 2>/dev/null || true
sleep 2
ray start --head --port=6379 --num-gpus=$n_gpus_per_node --resources='{"tool_agent": 8}'
sleep 4

# ---- Tool server URL (default localhost) ----
tool_server_url=${TOOL_SERVER_URL:-http://127.0.0.1:30888/get_observation}
echo "Using tool server at $tool_server_url"

# ---- Verify Ray ----
python3 -c "
import ray
ray.init(address='auto', ignore_reinit_error=True)
gpus = sum(n['Resources'].get('GPU', 0) for n in ray.nodes() if n['Alive'])
print(f'Ray cluster: 1 node, {int(gpus)} GPUs')
assert int(gpus) >= $n_gpus_per_node, f'expected {$n_gpus_per_node} GPUs but got {int(gpus)}'
ray.shutdown()
"

# ---- Training ----
trap 'ray stop --force 2>/dev/null || true' EXIT

PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg  \
    algorithm.gigpo_omega=1.0 \
    algorithm.gigpo_gamma=0.99 \
    +algorithm.gigpo_sim_threshold=$gigpo_sim_threshold \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=64 \
    data.dataloader_num_workers=16 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    data.shuffle=True \
    reward_model.reward_manager=$reward_manager \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    +actor_rollout_ref.agent.max_obs_length_image=$max_obs_length_image \
    +actor_rollout_ref.agent.max_obs_length_text=$max_obs_length_text \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    actor_rollout_ref.agent.tool_call_timeout=600 \
    actor_rollout_ref.agent.max_concurrent_trajectories=64 \
    +actor_rollout_ref.agent.dispatch_mode=work_queue \
    +actor_rollout_ref.agent.logprobs=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.agent.num_workers=$(expr $n_nodes \* $n_gpus_per_node) \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm-processor-cache-gb=8 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    algorithm.use_kl_in_reward=False \
    +algorithm.overturn_masking=False \
    trainer.logger=['console'] \
    trainer.project_name=pedia_rl \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$WORKSPACE/checkpoints/$run_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.rollout_data_dir=$WORKSPACE/logs/$run_name/step_records \
    trainer.nnodes=$n_nodes \
    +trainer.max_actor_ckpt_to_keep=20 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs \
    trainer.resume_mode=auto \
    2>&1 | tee $WORKSPACE/logs/$run_name/train.log

echo "Training finished"
