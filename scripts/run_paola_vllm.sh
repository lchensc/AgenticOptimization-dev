#!/bin/bash
# ============================================================================
# Run Paola with vLLM (auto-detect server)
# ============================================================================
#
# USAGE:
#   ./scripts/run_paola_vllm.sh
#
# This script automatically:
#   1. Finds a running vLLM server job
#   2. Sets VLLM_API_BASE to the correct node
#   3. Launches Paola with the vLLM model
#
# ============================================================================

set -e

# Find running vLLM job (try new name first, fallback to old name)
VLLM_JOB=$(squeue -u $USER -n vllm-server -h -o "%N" 2>/dev/null | head -1)

if [ -z "$VLLM_JOB" ]; then
    # Fallback to old job name
    VLLM_JOB=$(squeue -u $USER -n vllm-qwen3 -h -o "%N" 2>/dev/null | head -1)
fi

if [ -z "$VLLM_JOB" ]; then
    echo "❌ No running vLLM server found."
    echo ""
    echo "Start one with:"
    echo "  sbatch scripts/submit_vllm_server.sh"
    echo ""
    echo "Then wait for it to show 'Uvicorn running' in the logs:"
    echo "  tail -f logs/vllm_server_*.log"
    exit 1
fi

# Set the API base
export VLLM_API_BASE="http://${VLLM_JOB}:8000/v1"

echo "✓ Found vLLM server on: $VLLM_JOB"
echo "✓ VLLM_API_BASE=$VLLM_API_BASE"
echo ""

# Test if server is responding
if ! curl -s "$VLLM_API_BASE/models" > /dev/null 2>&1; then
    echo "⏳ Server is starting up. Waiting for it to be ready..."
    echo "   (Check logs: tail -f logs/vllm_server_*.log)"
    echo ""

    # Wait up to 5 minutes for server to be ready
    for i in {1..60}; do
        if curl -s "$VLLM_API_BASE/models" > /dev/null 2>&1; then
            echo "✓ Server is ready!"
            break
        fi
        sleep 5
    done

    if ! curl -s "$VLLM_API_BASE/models" > /dev/null 2>&1; then
        echo "❌ Server not responding after 5 minutes. Check logs for errors."
        exit 1
    fi
fi

echo "Launching Paola with vllm:qwen3-32b..."
echo ""

# Change to project directory and run Paola
cd /scratch/longchen/AgenticOptimization-dev
python -m paola.cli --model vllm:qwen3-32b "$@"
