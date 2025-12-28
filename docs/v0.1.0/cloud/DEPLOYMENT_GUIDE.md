# Paola Deployment Guide for Cluster

A complete guide for running Paola with local LLMs on the SLURM cluster.

## Quick Start (3 Steps)

### Step 1: Submit the vLLM server job
```bash
cd /scratch/longchen/AgenticOptimization-dev
sbatch scripts/submit_vllm_job.sh
```

### Step 2: Wait for server to start
```bash
# Check job status
squeue -u $USER

# Once running, monitor the log (replace JOBID with your job ID)
tail -f logs/vllm_server_JOBID.log
```

Wait until you see: `Uvicorn running on http://0.0.0.0:8000`

### Step 3: Connect Paola to the server
```bash
# Get the GPU node name from the log or squeue output
export VLLM_API_BASE=http://gpuXXX:8000/v1

# Run Paola
conda activate ml
python -m paola.cli --model vllm:deepseek-r1
```

---

## Detailed Setup

### Prerequisites

1. **Conda environment**: `ml` (with all dependencies installed)
2. **Model files**: `/scratch/longchen/LLM/DeepSeek-R1-Distill-Llama-70B`
3. **Cluster access**: SLURM with GPU partition access

### File Locations

| Component | Path |
|-----------|------|
| Paola source | `/scratch/longchen/AgenticOptimization-dev` |
| Model weights | `/scratch/longchen/LLM/DeepSeek-R1-Distill-Llama-70B` |
| Job logs | `/scratch/longchen/AgenticOptimization-dev/logs/` |
| SLURM script | `/scratch/longchen/AgenticOptimization-dev/scripts/submit_vllm_job.sh` |

---

## Usage Scenarios

### Scenario 1: Using API Models (No GPU Required)

If you just want to use Paola with cloud APIs (Qwen, Claude, GPT):

```bash
conda activate ml
cd /scratch/longchen/AgenticOptimization-dev

# Use Qwen (default, uses DASHSCOPE_API_KEY from .env)
python -m paola.cli

# Or specify a model
python -m paola.cli --model claude-sonnet-4
python -m paola.cli --model gpt-4
```

### Scenario 2: Using Local LLM (Requires GPU)

#### Option A: Submit as batch job (recommended)

```bash
# Submit the job
sbatch scripts/submit_vllm_job.sh

# Check status
squeue -u $USER

# Get the node name
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
```

Once running, the log file will show connection instructions.

#### Option B: Interactive session

```bash
# Request GPU node
srun --partition=gpuidle --gres=gpu:2 --mem=128G --time=8:00:00 --pty bash

# On the GPU node:
conda activate ml
cd /scratch/longchen/AgenticOptimization-dev
./scripts/start_vllm_server.sh
```

---

## Connecting to the vLLM Server

### From Another Terminal on the Cluster

```bash
# Set the server URL (replace gpuXXX with actual node name)
export VLLM_API_BASE=http://gpuXXX:8000/v1

# Run Paola
conda activate ml
cd /scratch/longchen/AgenticOptimization-dev
python -m paola.cli --model vllm:deepseek-r1
```

### Test the Connection

```bash
# Check if server is running
curl http://gpuXXX:8000/v1/models

# Expected output:
# {"object":"list","data":[{"id":"deepseek-r1",...}]}
```

### From Your Local Machine (SSH Tunnel)

```bash
# Create SSH tunnel
ssh -L 8000:gpuXXX:8000 username@cluster.example.com

# Then locally:
export VLLM_API_BASE=http://localhost:8000/v1
python -m paola.cli --model vllm:deepseek-r1
```

---

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_API_BASE` | `http://localhost:8000/v1` | vLLM server URL |
| `DASHSCOPE_API_KEY` | (from .env) | Qwen API key |
| `ANTHROPIC_API_KEY` | (from .env) | Claude API key |
| `OPENAI_API_KEY` | (not set) | OpenAI API key |

### Model Selection

```bash
# Local vLLM model
python -m paola.cli --model vllm:deepseek-r1

# Qwen API (default)
python -m paola.cli --model qwen-plus

# Claude API
python -m paola.cli --model claude-sonnet-4

# OpenAI API
python -m paola.cli --model gpt-4
```

---

## Troubleshooting

### "Connection refused" error

The vLLM server isn't running or isn't reachable.

1. Check if job is running: `squeue -u $USER`
2. Check logs: `tail -f logs/vllm_server_*.log`
3. Verify node name and port in `VLLM_API_BASE`

### Job pending too long

GPU resources may be scarce.

```bash
# Check queue
squeue -p gpuidle

# Try a different partition
# Edit submit_vllm_job.sh to use different partition
```

### "CUDA out of memory" error

Reduce memory usage in `submit_vllm_job.sh`:

```bash
--gpu-memory-utilization 0.85 \
--max-model-len 4096 \
```

### Model loading takes too long

The 70B model takes 2-5 minutes to load. This is normal.

---

## For Colleagues: Quick Reference

### First-Time Setup

```bash
# 1. Clone or access the repo
cd /scratch/longchen/AgenticOptimization-dev

# 2. Activate environment
conda activate ml

# 3. Test with API model (no GPU needed)
python -m paola.cli --model qwen-plus
```

### Using Local LLM

```bash
# 1. Submit server job
sbatch scripts/submit_vllm_job.sh

# 2. Wait and get node name
squeue -u $USER  # Look for the node (e.g., gpu047)

# 3. Connect Paola (in new terminal)
export VLLM_API_BASE=http://gpu047:8000/v1
python -m paola.cli --model vllm:deepseek-r1
```

### Stop the Server

```bash
# Cancel the job
scancel JOBID
```

---

## Performance Notes

| Configuration | Tokens/sec | VRAM Usage |
|---------------|------------|------------|
| 2x A30, 8K context | 15-30 | ~45GB |
| 2x A30, 4K context | 20-35 | ~40GB |
| 1x A30, 2K context | 5-10 | ~22GB |

Recommended: Use 2 GPUs for the 70B model.

---

## Support

- Check logs: `logs/vllm_server_*.log`
- Paola docs: `docs/` folder
- Issues: Contact Long Chen
