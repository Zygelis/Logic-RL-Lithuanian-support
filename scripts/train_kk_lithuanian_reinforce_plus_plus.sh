#!/bin/bash

# Reinforce++ training script for Lithuanian KK dataset
# Expects preprocessed parquet files at:
#   data/kk/instruct/lithuanian/train.parquet
#   data/kk/instruct/lithuanian/test.parquet

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

# Build args safely to avoid stray token execution
ARGS=""
ARGS+="algorithm.adv_estimator=reinforce_plus_plus "
ARGS+="data.train_files=./data/kk/instruct/lithuanian/train.parquet "
ARGS+="data.val_files=./data/kk/instruct/lithuanian/test.parquet "
ARGS+="data.train_batch_size=1 "
ARGS+="data.val_batch_size=1 "
ARGS+="data.max_prompt_length=512 "
ARGS+="data.max_response_length=512 "
ARGS+="actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct "
ARGS+="actor_rollout_ref.actor.optim.lr=1e-6 "
# Load model with low CPU memory usage and bfloat16 for FlashAttention
# Use Hydra override_config (+) so keys are accepted by the config system
ARGS+="+actor_rollout_ref.model.override_config.torch_dtype=bfloat16 "
ARGS+="+actor_rollout_ref.model.override_config.device_map=auto "
ARGS+="+actor_rollout_ref.model.override_config.low_cpu_mem_usage=true "
ARGS+="+actor_rollout_ref.model.override_config.trust_remote_code=true "
ARGS+="actor_rollout_ref.model.use_remove_padding=true "
ARGS+="actor_rollout_ref.actor.ppo_mini_batch_size=1 "
ARGS+="actor_rollout_ref.actor.ppo_micro_batch_size=1 "
ARGS+="actor_rollout_ref.actor.kl_loss_coef=0.001 "
ARGS+="actor_rollout_ref.model.enable_gradient_checkpointing=true "
ARGS+="actor_rollout_ref.actor.fsdp_config.param_offload=false "
ARGS+="actor_rollout_ref.actor.fsdp_config.grad_offload=false "
ARGS+="actor_rollout_ref.actor.fsdp_config.optimizer_offload=false "
ARGS+="actor_rollout_ref.rollout.log_prob_micro_batch_size=1 "
ARGS+="actor_rollout_ref.rollout.tensor_model_parallel_size=1 "
ARGS+="actor_rollout_ref.rollout.name=vllm "
ARGS+="actor_rollout_ref.rollout.dtype=bfloat16 "
# Increase GPU memory utilization so vllm can allocate cache blocks
ARGS+="actor_rollout_ref.rollout.gpu_memory_utilization=0.9 "
ARGS+="actor_rollout_ref.rollout.n=1 "
ARGS+="actor_rollout_ref.ref.log_prob_micro_batch_size=1 "
ARGS+="actor_rollout_ref.ref.fsdp_config.param_offload=true "
ARGS+="algorithm.kl_ctrl.kl_coef=0.001 "
ARGS+="trainer.critic_warmup=0 "
ARGS+="trainer.logger=[console] "
ARGS+="trainer.project_name='kk_lithuanian_rf++_test' "
ARGS+="trainer.experiment_name='rf++_1.5b_small_test' "
ARGS+="trainer.n_gpus_per_node=1 "
ARGS+="trainer.nnodes=1 "
ARGS+="trainer.save_freq=-1 "
ARGS+="trainer.test_freq=1 "
ARGS+="trainer.total_epochs=1 "

python3 -m verl.trainer.main_ppo $ARGS "$@"
