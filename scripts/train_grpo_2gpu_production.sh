#!/usr/bin/env bash
set -xe

export PYTHONUNBUFFERED=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# Production config for 2 V100 GPUs (balanced throughput, 1-4 hour training)
# Increases batch size, response length, epochs, and uses full dataset vs _small

ARGS=""
ARGS+=" algorithm.adv_estimator=grpo"

# Use FULL Lithuanian dataset (not _small)
ARGS+=" data.train_files=./data/kk/instruct/lithuanian/train.parquet"
ARGS+=" data.val_files=./data/kk/instruct/lithuanian/test.parquet"

# Increase batch size from 2 to 8, val to 4
ARGS+=" data.train_batch_size=8"
ARGS+=" data.val_batch_size=4"
ARGS+=" data.max_prompt_length=512"
ARGS+=" data.max_response_length=256"

# Model config - keep float16 (V100 requirement)
ARGS+=" actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct"
ARGS+=" +actor_rollout_ref.model.override_config.torch_dtype=float16"
ARGS+=" +actor_rollout_ref.model.override_config.low_cpu_mem_usage=true"
ARGS+=" +actor_rollout_ref.model.override_config.trust_remote_code=true"
ARGS+=" actor_rollout_ref.model.use_remove_padding=True"
ARGS+=" actor_rollout_ref.model.enable_gradient_checkpointing=True"

# Actor training - increase mini batch size from 4 to 8
ARGS+=" actor_rollout_ref.actor.optim.lr=3e-07"
ARGS+=" actor_rollout_ref.actor.ppo_mini_batch_size=8"
ARGS+=" actor_rollout_ref.actor.ppo_micro_batch_size=4"
ARGS+=" actor_rollout_ref.actor.use_kl_loss=True"
ARGS+=" actor_rollout_ref.actor.kl_loss_coef=0.001"

# Memory optimization - FSDP offloading
ARGS+=" ++actor_rollout_ref.actor.fsdp_config.param_offload=true"
ARGS+=" ++actor_rollout_ref.actor.fsdp_config.grad_offload=true"
ARGS+=" ++actor_rollout_ref.actor.fsdp_config.optimizer_offload=true"
ARGS+=" ++actor_rollout_ref.ref.fsdp_config.param_offload=true"

# vLLM rollout - 2 GPUs with optimized settings
ARGS+=" actor_rollout_ref.rollout.name=vllm"
ARGS+=" actor_rollout_ref.rollout.gpu_memory_utilization=0.6"
ARGS+=" actor_rollout_ref.rollout.tensor_model_parallel_size=2"
ARGS+=" actor_rollout_ref.rollout.n=1"
ARGS+=" actor_rollout_ref.rollout.max_num_seqs=8"
ARGS+=" actor_rollout_ref.rollout.max_num_batched_tokens=4096"
ARGS+=" actor_rollout_ref.rollout.prompt_length=512"
ARGS+=" actor_rollout_ref.rollout.response_length=128"
ARGS+=" actor_rollout_ref.rollout.dtype=float16"

# Training config - 2-3 epochs, enable checkpointing
ARGS+=" trainer.critic_warmup=0"
ARGS+=" trainer.total_epochs=2"
ARGS+=" trainer.test_freq=1"
ARGS+=" trainer.save_freq=500"
ARGS+=" trainer.output_dir='./outputs/grpo_kk_production'"
ARGS+=" trainer.logger=['console']"
ARGS+=" trainer.project_name='grpo_kk_production'"
ARGS+=" trainer.experiment_name='qwen2_1.5b_v100_2gpu_ep2'"
ARGS+=" trainer.n_gpus_per_node=2"
ARGS+=" trainer.nnodes=1"

python3 -u -m verl.trainer.main_ppo $ARGS
