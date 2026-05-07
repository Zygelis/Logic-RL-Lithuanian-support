#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH -n1
#SBATCH --job-name=reinforce-2gpu-1.5B
#SBATCH --output=reinforce_2gpu_%j.out

exec > >(tee -a train_reinforce_2gpu.log) 2>&1

cd /scratch/lustre/home/zygi9184/Logic-RL-Lithuanian-support
source venv/bin/activate


# Ensure script is executable and run
chmod +x scripts/train_reinforce_plus_2gpu_1.5Binstruct.sh
bash scripts/train_reinforce_plus_2gpu_1.5Binstruct.sh $@
