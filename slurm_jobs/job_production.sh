#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH -n1
#SBATCH --job-name=reinforce-2gpu-1.5B
#SBATCH --output=reinforce_2gpu_%j.out

cd /scratch/lustre/home/zygi9184/Logic-RL-Lithuanian-support
source venv/bin/activate

# Log GPU info
nvidia-smi

# Run production training
chmod +x scripts/train_grpo_2gpu_production.sh
bash scripts/train_grpo_2gpu_production.sh
