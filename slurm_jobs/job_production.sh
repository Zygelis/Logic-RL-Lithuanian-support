#!/bin/bash
#SBATCH --job-name=grpo_kk_prod_v100
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

# Production GRPO training on 2 V100 GPUs
# Expected wall-clock: 1.5-3 hours depending on data size

cd /scratch/lustre/home/zygi9184/Logic-RL-Lithuanian-support
source venv/bin/activate

# Log GPU info
nvidia-smi

# Run production training
chmod +x scripts/train_grpo_2gpu_production.sh
bash scripts/train_grpo_2gpu_production.sh
