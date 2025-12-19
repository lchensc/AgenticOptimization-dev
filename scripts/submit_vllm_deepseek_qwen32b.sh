#!/bin/bash
#SBATCH --job-name=vllm-deepseek-qwen32b
#SBATCH --partition=gpuidle
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=128G
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/vllm_server_%j.log
#SBATCH --error=logs/vllm_server_%j.err

# ============================================================================
# SLURM Job Script for vLLM Server (DeepSeek-R1-Distill-Qwen-32B) - H100
# ============================================================================
#
# PURPOSE:
#   Run DeepSeek-R1-Distill-Qwen-32B on 2x H100 GPUs.
#   This model has 128K context and superior reasoning (distilled from R1 671B).
#
# USAGE:
#   sbatch scripts/submit_vllm_deepseek_qwen32b.sh
#
# REQUIREMENTS:
#   - 2x H100 GPUs (80GB each = 160GB total)
#   - Model at /scratch/longchen/LLM/DeepSeek-R1-Distill-Qwen-32B
#
# COMPARISON:
#   | Model                        | Context | Params |
#   |------------------------------|---------|--------|
#   | DeepSeek-R1-Distill-Qwen-32B | 128K    | 32B    |
#   | Qwen3-32B                    | 40K     | 32B    |
#
# ============================================================================

# Exit if not running under SLURM
if [ -z "$SLURM_JOB_ID" ]; then
    echo "=============================================="
    echo "ERROR: This script must be submitted via sbatch"
    echo "=============================================="
    echo ""
    echo "Usage:"
    echo "  sbatch scripts/submit_vllm_deepseek_qwen32b.sh"
    echo ""
    exit 1
fi

# Create logs directory if needed
mkdir -p /scratch/longchen/AgenticOptimization-dev/logs

echo "=============================================="
echo "vLLM Server - DeepSeek-R1-Distill-Qwen-32B"
echo "=============================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "Time:      $(date)"
echo "=============================================="
echo ""

# Activate conda environment
source /home/longchen/.bashrc
eval "$(conda shell.bash hook)"
conda activate ml

# Print GPU info
echo "GPU Information:"
nvidia-smi
echo ""

# Configuration
MODEL_PATH="/scratch/longchen/LLM/DeepSeek-R1-Distill-Qwen-32B"
MODEL_NAME="deepseek-r1-qwen32b"
PORT=8000
MAX_MODEL_LEN=32768  # Conservative; can go up to 128K on H100

# Count GPUs allocated by SLURM
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

# Ensure GPU_COUNT is valid for tensor parallelism (power of 2)
case $GPU_COUNT in
    1|2|4|8) ;; # Valid, keep as is
    3) GPU_COUNT=2 ;;
    5|6|7) GPU_COUNT=4 ;;
    *) GPU_COUNT=2 ;;  # Safe default
esac

echo "Configuration:"
echo "  Model:        $MODEL_NAME"
echo "  Context:      $MAX_MODEL_LEN tokens"
echo "  GPUs:         $GPU_COUNT (tensor parallel)"
echo "  Tool Calling: Enabled (hermes parser)"
echo ""

# Print connection instructions
echo "=============================================="
echo "CONNECTION INSTRUCTIONS"
echo "=============================================="
echo ""
echo "Once the server shows 'Uvicorn running', connect using:"
echo ""
echo "Option 1: Direct Access (from cluster)"
echo "  export VLLM_API_BASE=http://$SLURM_NODELIST:$PORT/v1"
echo "  python -m paola.cli --model vllm:$MODEL_NAME"
echo ""
echo "Option 2: Test with curl (from cluster)"
echo "  curl http://$SLURM_NODELIST:$PORT/v1/models"
echo ""
echo "=============================================="
echo ""
echo "Starting vLLM server with DeepSeek-R1-Distill-Qwen-32B..."
echo ""

# Start vLLM server with tool calling enabled
# Using hermes parser since this is a Qwen-based model
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size "$GPU_COUNT" \
    --gpu-memory-utilization 0.90 \
    --max-model-len "$MAX_MODEL_LEN" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

echo ""
echo "vLLM server exited at $(date)"
