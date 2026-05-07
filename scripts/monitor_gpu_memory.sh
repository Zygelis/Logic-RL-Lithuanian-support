#!/usr/bin/env bash

# Quick memory monitoring script for V100 GRPO training
# Usage: bash scripts/monitor_gpu_memory.sh <interval_seconds>

INTERVAL=${1:-2}

echo "GPU Memory Monitoring (V100 GRPO Training)"
echo "==========================================="
echo "Interval: ${INTERVAL}s | Press Ctrl+C to stop"
echo ""

watch -n $INTERVAL 'nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits | awk -F, "{printf \"GPU %s | Used: %6d MB | Free: %6d MB | GPU: %3d%% | Mem: %3d%% | Temp: %2d°C\n\", \$1, \$3, \$4, \$6, \$7, \$8}"'
