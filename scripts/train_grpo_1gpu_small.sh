#!/usr/bin/env bash
set -xe

export VLLM_ATTENTION_BACKEND=XFORMERS

ARGS=""
ARGS+=" algorithm.adv_estimator=grpo"
ARGS+=" data.train_files=./data/kk/instruct/lithuanian/train.parquet"
ARGS+=" data.val_files=./data/kk/instruct/lithuanian/test.parquet"
ARGS+=" data.train_batch_size=2"
ARGS+=" data.val_batch_size=2"
ARGS+=" data.max_prompt_length=512"
ARGS+=" data.max_response_length=256"
ARGS+=" actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct"
ARGS+=" +actor_rollout_ref.model.override_config.torch_dtype=bfloat16"
ARGS+=" +actor_rollout_ref.model.override_config.device_map=auto"
ARGS+=" +actor_rollout_ref.model.override_config.low_cpu_mem_usage=true"
ARGS+=" +actor_rollout_ref.model.override_config.trust_remote_code=true"
ARGS+=" actor_rollout_ref.actor.optim.lr=3e-07"
ARGS+=" actor_rollout_ref.model.use_remove_padding=True"
ARGS+=" actor_rollout_ref.actor.ppo_mini_batch_size=4"
ARGS+=" actor_rollout_ref.actor.ppo_micro_batch_size=2"
ARGS+=" actor_rollout_ref.actor.use_kl_loss=True"
ARGS+=" actor_rollout_ref.actor.kl_loss_coef=0.001"
ARGS+=" actor_rollout_ref.model.enable_gradient_checkpointing=True"

# Offload to reduce host memory
ARGS+=" ++actor_rollout_ref.actor.fsdp_config.param_offload=true"
ARGS+=" ++actor_rollout_ref.actor.fsdp_config.grad_offload=true"
ARGS+=" ++actor_rollout_ref.actor.fsdp_config.optimizer_offload=true"

# Use vLLM rollout on 2 GPUs
ARGS+=" actor_rollout_ref.rollout.name=vllm"
ARGS+=" actor_rollout_ref.rollout.gpu_memory_utilization=0.6"
ARGS+=" actor_rollout_ref.rollout.tensor_model_parallel_size=2"
ARGS+=" actor_rollout_ref.rollout.n=1"
ARGS+=" actor_rollout_ref.rollout.max_num_seqs=4"
ARGS+=" actor_rollout_ref.rollout.max_num_batched_tokens=2048"
ARGS+=" actor_rollout_ref.rollout.prompt_length=512"
ARGS+=" actor_rollout_ref.rollout.response_length=64"

# Reference model offloaded
ARGS+=" ++actor_rollout_ref.ref.fsdp_config.param_offload=true"

ARGS+=" trainer.critic_warmup=0"
ARGS+=" trainer.total_epochs=1"
ARGS+=" trainer.test_freq=1"
ARGS+=" trainer.save_freq=-1"
ARGS+=" trainer.logger=['console']"
ARGS+=" trainer.project_name='grpo_kk_small_test'"
ARGS+=" trainer.experiment_name='grpo_qwen2_small'" 
ARGS+=" trainer.n_gpus_per_node=2"
ARGS+=" trainer.nnodes=1"

python3 -u -m verl.trainer.main_ppo $ARGS
