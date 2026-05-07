#!/usr/bin/env bash
set -xe

export PYTHONUNBUFFERED=1
export VLLM_ATTENTION_BACKEND=XFORMERS
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Production config for 2 V100 GPUs (balanced throughput, 1-4 hour training)
# Increases batch size, response length, epochs, and uses full dataset vs _small
python3 -u -m verl.trainer.main_ppo \
	algorithm.adv_estimator=grpo \
	data.train_files=./data/kk/instruct/lithuanian/train.parquet \
	data.val_files=./data/kk/instruct/lithuanian/test.parquet \
	data.train_batch_size=8 \
	data.val_batch_size=4 \
	data.max_prompt_length=1024 \
	data.max_response_length=1024 \
	actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
	+actor_rollout_ref.model.override_config.torch_dtype=float16 \
	+actor_rollout_ref.model.override_config.low_cpu_mem_usage=true \
	+actor_rollout_ref.model.override_config.trust_remote_code=true \
	actor_rollout_ref.model.use_remove_padding=True \
	actor_rollout_ref.model.enable_gradient_checkpointing=True \
	actor_rollout_ref.actor.optim.lr=3e-07 \
	actor_rollout_ref.actor.ppo_mini_batch_size=8 \
	actor_rollout_ref.actor.ppo_micro_batch_size=4 \
	actor_rollout_ref.actor.use_kl_loss=True \
	actor_rollout_ref.actor.kl_loss_coef=0.001 \
	++actor_rollout_ref.actor.fsdp_config.param_offload=true \
	++actor_rollout_ref.actor.fsdp_config.grad_offload=true \
	++actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
	++actor_rollout_ref.ref.fsdp_config.param_offload=true \
	actor_rollout_ref.rollout.name=vllm \
	actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
	actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
	actor_rollout_ref.rollout.n=1 \
	actor_rollout_ref.rollout.max_num_seqs=8 \
	actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
	actor_rollout_ref.rollout.prompt_length=600 \
	actor_rollout_ref.rollout.response_length=1024 \
	actor_rollout_ref.rollout.dtype=float16 \
	trainer.critic_warmup=0 \
	trainer.total_epochs=1 \
	trainer.test_freq=1 \
	trainer.save_freq=50 \
	trainer.logger=['console'] \
	trainer.project_name='grpo_kk_production' \
	trainer.experiment_name='qwen2_1.5b_v100_2gpu_ep1' \
	trainer.n_gpus_per_node=2 \
	trainer.nnodes=1
