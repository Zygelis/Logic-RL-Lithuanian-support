#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1           # Change to more GPUs if needed
#SBATCH --time=0:10:00            # Adjust time limit as you expect
#SBATCH -n1
#SBATCH --job-name=test-job
#SBATCH --output=slurm-test-%j.out

cd /scratch/lustre/home/zygi9184/Logic-RL-Lithuanian-support
source venv/bin/activate

# Run your training (e.g., your shell script)
python3 test_qwen25_lithuanian_compat.py
