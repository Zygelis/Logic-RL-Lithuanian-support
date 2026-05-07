#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:2             # Change to more GPUs if needed
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00            # Adjust time limit as you expect
#SBATCH -n1
#SBATCH --job-name=logic-train
#SBATCH --output=slurm-train-%j.out

exec > >(tee -a train.log) 2>&1

cd /scratch/lustre/home/zygi9184/Logic-RL-Lithuanian-support
source venv/bin/activate

python3 --version
which python3
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

# Run your training (e.g., your shell script)
chmod +x scripts/train_grpo_2gpu_small.sh
bash scripts/train_grpo_2gpu_small.sh