# vLLM Server Setup for Paola

This guide explains how to run a local LLM server using vLLM on the RPTU cluster for use with Paola.

## Quick Start

```bash
# 1. Submit the vLLM server job
cd /scratch/longchen/AgenticOptimization-dev
sbatch scripts/submit_vllm_qwen3.sh

# 2. Check job status and wait for it to start
squeue -u $USER

# 3. Once running, check the log for the node name
tail -f logs/vllm_server_<jobid>.log
# Look for: "Node: gpuXXX" and "Uvicorn running"

# 4. Connect to Paola (replace gpuXXX with actual node)
export VLLM_API_BASE=http://gpuXXX:8000/v1
python -m paola.cli --model vllm:qwen3-32b
```

## Prerequisites

### 1. Download the Model

The Qwen3-32B model should be downloaded to `/scratch/longchen/LLM/Qwen3-32B`:

```bash
# If not already downloaded:
conda activate ml
huggingface-cli download Qwen/Qwen3-32B --local-dir /scratch/longchen/LLM/Qwen3-32B
```

### 2. Conda Environment

Ensure `vllm` and `langchain-openai` are installed in the `ml` environment:

```bash
conda activate ml
pip install vllm langchain-openai
```

## How It Works

### Architecture

```
┌─────────────────┐     HTTP/OpenAI API     ┌─────────────────┐
│   Paola CLI     │ ──────────────────────► │   vLLM Server   │
│ (login node)    │                         │  (GPU node)     │
└─────────────────┘                         └─────────────────┘
        │                                           │
        │ VLLM_API_BASE env var                     │ Qwen3-32B model
        │ http://gpuXXX:8000/v1                     │ /scratch/longchen/LLM/
```

### Key Components

1. **vLLM Server** (`scripts/submit_vllm_qwen3.sh`): SLURM job that serves the model
2. **Paola LLM Client** (`paola/llm/models.py`): Connects via OpenAI-compatible API
3. **Tool Calling**: Enabled via `--enable-auto-tool-choice --tool-call-parser hermes`

## Configuration Details

### SLURM Job Settings

| Setting | Value | Notes |
|---------|-------|-------|
| Partition | `gpuidle` | Preemptible, but usually available |
| GPUs | 2 (any type) | Auto-adjusts to available hardware |
| Memory | 128GB | System RAM |
| Time | 10 days | Maximum job duration |

### vLLM Server Settings

| Setting | Value | Notes |
|---------|-------|-------|
| Model | Qwen3-32B | 32B parameters, ~55GB |
| Context Length | Auto (8K-32K) | Based on GPU memory |
| Tool Calling | Enabled | `hermes` parser |
| Memory Utilization | 90% | GPU VRAM usage |

### GPU Memory Requirements

The script auto-detects GPU type and adjusts context length:

| GPU Type | Memory | Context Length |
|----------|--------|----------------|
| H100/A100 | 80GB each | 32K tokens |
| A40 | 48GB each | 16K tokens |
| V100 | 32GB each | 8K tokens |

## Performance

Current setup on 2x A40 GPUs:

| Metric | Value |
|--------|-------|
| Generation Speed | ~14.5 tokens/s |
| Prompt Processing | ~1000 tokens/s |
| Model Load Time | ~15 minutes |

### Performance Tips

1. **Request better GPUs**: Add `--gres=gpu:h100:2` or `--gres=gpu:a100:2` to the script
2. **Use more GPUs**: Change `--gres=gpu:4` for more parallelism
3. **Smaller model**: Download Qwen3-8B-Instruct for faster inference

## Troubleshooting

### Job Stays Pending

Check available resources:
```bash
sinfo -p gpuidle -o "%P %a %D %t %G %N"
```

Try requesting more common GPU types by editing the script:
```bash
#SBATCH --gres=gpu:4   # Request 4 GPUs of any type
```

### Out of Memory Error

If you see "No available memory for cache blocks":
1. The model is too large for allocated GPUs
2. Reduce `--max-model-len` in the script
3. Request more GPUs

### Tool Calling Not Working

Ensure the server was started with:
```bash
--enable-auto-tool-choice \
--tool-call-parser hermes
```

Check the log file for these settings.

### Connection Refused

1. Verify the job is running: `squeue -u $USER`
2. Check the correct node: `grep "Node:" logs/vllm_server_<jobid>.log`
3. Test with curl: `curl http://gpuXXX:8000/v1/models`

## Files Reference

| File | Purpose |
|------|---------|
| `scripts/submit_vllm_qwen3.sh` | SLURM job script for vLLM server |
| `paola/llm/models.py` | LLM client code (reads `VLLM_API_BASE`) |
| `logs/vllm_server_*.log` | Server output logs |
| `/scratch/longchen/LLM/Qwen3-32B/` | Model weights |

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `VLLM_API_BASE` | vLLM server URL | `http://gpu009:8000/v1` |

## Comparison: vLLM vs Cloud API

| Aspect | vLLM (Local) | Alibaba DashScope |
|--------|--------------|-------------------|
| Cost | Free (cluster time) | Pay per token |
| Speed | ~15 tok/s (A40) | ~50-100 tok/s |
| Privacy | Data stays local | Data sent to cloud |
| Availability | Depends on cluster | Always available |
| Setup | Requires job submission | Just API key |

## Alternative Models

To use a different model, edit `submit_vllm_qwen3.sh`:

```bash
# For smaller/faster model:
MODEL_PATH="/scratch/longchen/LLM/Qwen3-8B-Instruct"
MODEL_NAME="qwen3-8b"

# For DeepSeek:
MODEL_PATH="/scratch/longchen/LLM/DeepSeek-R1-Distill-Llama-70B"
MODEL_NAME="deepseek-r1"
```

Then download the model:
```bash
huggingface-cli download Qwen/Qwen3-8B-Instruct --local-dir /scratch/longchen/LLM/Qwen3-8B-Instruct
```
