#!/bin/bash
#SBATCH --job-name=vllm-deepseek
#SBATCH --partition=gpuidle
#SBATCH --gres=gpu:v100:16
#SBATCH --nodelist=dgx001
#SBATCH --mem=128G
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/vllm_server_%j.log
#SBATCH --error=logs/vllm_server_%j.err

# ============================================================================
# SLURM Job Script for vLLM Server (DeepSeek-R1-Distill-Llama-70B)
# ============================================================================
#
# USAGE:
#   sbatch scripts/submit_vllm_deepseek.sh
#
# REQUIREMENTS:
#   - 16x V100 GPUs on DGX node (faster loading from NVMe)
#   - Model at /scratch/longchen/LLM/DeepSeek-R1-Distill-Llama-70B
#
# After submission:
#   1. Check job status:  squeue -u $USER
#   2. Once running, check logs:  tail -f logs/vllm_server_<jobid>.log
#   3. Connect to the server (see instructions in log file)
#
# ============================================================================

# Exit if not running under SLURM
if [ -z "$SLURM_JOB_ID" ]; then
    echo "=============================================="
    echo "ERROR: This script must be submitted via sbatch"
    echo "=============================================="
    echo ""
    echo "Usage:"
    echo "  sbatch scripts/submit_vllm_deepseek.sh"
    echo ""
    exit 1
fi

# Create logs directory if needed
mkdir -p /scratch/longchen/AgenticOptimization-dev/logs

echo "=============================================="
echo "vLLM Server - DeepSeek-R1-Distill-Llama-70B"
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
MODEL_PATH="/scratch/longchen/LLM/DeepSeek-R1-Distill-Llama-70B"
MODEL_NAME="deepseek-r1"
PORT=8000
MAX_MODEL_LEN=8192

# Count GPUs allocated by SLURM
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

# Ensure GPU_COUNT is valid for tensor parallelism (power of 2)
case $GPU_COUNT in
    1|2|4|8|16) ;; # Valid, keep as is
    3) GPU_COUNT=2 ;;
    5|6|7) GPU_COUNT=4 ;;
    9|10|11|12|13|14|15) GPU_COUNT=8 ;;
    *) GPU_COUNT=8 ;;  # Safe default for large allocations
esac

echo "Configuration:"
echo "  Model:        $MODEL_NAME"
echo "  Context:      $MAX_MODEL_LEN tokens"
echo "  GPUs:         $GPU_COUNT (tensor parallel)"
echo "  Tool Calling: Enabled (llama3_json parser)"
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
echo "Starting vLLM server with DeepSeek-R1..."
echo ""

# Start vLLM server
# Using --enforce-eager to avoid CUDA graph compilation issues
# Tool calling enabled with llama3_json parser (DeepSeek R1 Distill is Llama-based)
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size "$GPU_COUNT" \
    --gpu-memory-utilization 0.90 \
    --max-model-len "$MAX_MODEL_LEN" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --served-model-name "$MODEL_NAME" \
    --enforce-eager \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json

echo ""
echo "vLLM server exited at $(date)"
