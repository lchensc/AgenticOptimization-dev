#!/bin/bash
#SBATCH --job-name=vllm-qwen3
#SBATCH --partition=gpuidle
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=128G
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/vllm_server_%j.log
#SBATCH --error=logs/vllm_server_%j.err

# ============================================================================
# SLURM Job Script for vLLM Server (Qwen3-32B)
# ============================================================================
#
# USAGE:
#   sbatch scripts/submit_vllm_qwen3.sh
#
# This script is flexible - it requests generic GPUs (not specific type)
# and auto-adjusts settings based on available GPU memory.
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
    echo "  sbatch scripts/submit_vllm_qwen3.sh"
    echo ""
    exit 1
fi

# Create logs directory if needed
mkdir -p /scratch/longchen/AgenticOptimization-dev/logs

echo "=============================================="
echo "vLLM Server Job Starting - Qwen3-32B"
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
MODEL_PATH="/scratch/longchen/LLM/Qwen3-32B"
MODEL_NAME="qwen3-32b"
PORT=8000

# Auto-detect GPU memory and count (using SLURM-allocated GPUs only)
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # Count GPUs allocated by SLURM
    GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    # Fallback: count all visible GPUs
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

echo "Detected: $GPU_COUNT x $GPU_NAME with ${GPU_MEM}MiB each"

# Calculate total GPU memory
TOTAL_MEM=$((GPU_MEM * GPU_COUNT))

# Qwen3-32B needs ~60GB for weights. With YaRN factor=2.0, we can use 64K context.
# Adjust max_model_len based on available memory
if [ $TOTAL_MEM -ge 160000 ]; then
    # 160GB+ (e.g., 2x A100/H100): Extended context with YaRN
    MAX_MODEL_LEN=65536
elif [ $TOTAL_MEM -ge 80000 ]; then
    # 80-160GB (e.g., 2x A40 or 1x A100): Native context
    MAX_MODEL_LEN=40960
else
    # <80GB (e.g., 2x V100): Reduced context
    MAX_MODEL_LEN=16384
fi

echo "Total GPU memory: ${TOTAL_MEM}MiB"
echo "Using max_model_len=$MAX_MODEL_LEN"
echo ""

# Ensure GPU_COUNT is valid for tensor parallelism (power of 2)
case $GPU_COUNT in
    1|2|4|8) ;; # Valid, keep as is
    3) GPU_COUNT=2 ;;
    5|6|7) GPU_COUNT=4 ;;
    *) GPU_COUNT=2 ;;  # Safe default
esac

echo "Tensor parallel size: $GPU_COUNT"
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
echo "Option 3: SSH Port Forwarding (from your local machine)"
echo "  ssh -L 8000:localhost:8000 $SLURM_NODELIST"
echo "  Then access: http://localhost:8000/v1"
echo ""
echo "=============================================="
echo ""
echo "Starting vLLM server with Qwen3-32B..."
echo ""

# Start vLLM server with tool calling enabled
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
