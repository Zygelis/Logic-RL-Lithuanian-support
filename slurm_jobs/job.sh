#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:2             # Change to more GPUs if needed
#SBATCH --time=2:00:00            # Adjust time limit as you expect
#SBATCH -n1
#SBATCH --job-name=logic-train
#SBATCH --output=slurm-train-%j.out

cd /scratch/lustre/home/zygi9184/Logic-RL-Lithuanian-support
source venv/bin/activate

# Run your training (e.g., your shell script)
chmod +x scripts/train_grpo_4gpu_7Binstruct.sh
bash scripts/train_grpo_4gpu_7Binstruct.sh