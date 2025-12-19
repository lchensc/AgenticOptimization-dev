# Local LLM Setup for Paola

This guide explains how to run Paola with local LLMs using vLLM on the cluster.

## Overview

Running a local LLM provides:
- **Zero API costs** - No per-token charges
- **Data privacy** - All processing stays on your cluster
- **No rate limits** - Unlimited requests
- **Offline capability** - Works without internet

## Available Models

| Model | Location | Size | Context | Tool Calling | Best For |
|-------|----------|------|---------|--------------|----------|
| Qwen3-32B | `/scratch/longchen/LLM/Qwen3-32B` | 59GB | 40K | ✅ Yes | **Paola / Agentic tasks** |
| DeepSeek-R1-Distill-Qwen-32B | `/scratch/longchen/LLM/DeepSeek-R1-Distill-Qwen-32B` | 45GB | 128K | ❌ No | Reasoning only |
| DeepSeek-R1-Distill-Llama-70B | `/scratch/longchen/LLM/DeepSeek-R1-Distill-Llama-70B` | 140GB | 128K | ❌ No | Reasoning only |

> ⚠️ **Important**: DeepSeek-R1 models do NOT support tool/function calling. They are reasoning models trained for chain-of-thought thinking, not agentic tasks. **Use Qwen3-32B for Paola.**

## Quick Start

### Step 1: Submit a vLLM Server Job

```bash
cd /scratch/longchen/AgenticOptimization-dev

# For Qwen3-32B (2x H100) - RECOMMENDED FOR PAOLA
sbatch scripts/submit_vllm_qwen3.sh
```

### Step 2: Check Job Status and Get Node

```bash
squeue -u $USER
# Look for the node name (e.g., gpu027)

# Check logs for "Uvicorn running"
tail -f logs/vllm_server_<jobid>.log
```

### Step 3: Run Paola with Local Model

```bash
export VLLM_API_BASE=http://<node>:8000/v1
python -m paola.cli --model vllm:qwen3-32b
```

---

## Downloading Models from HuggingFace

Use Python with `huggingface_hub` for reliable downloads:

```python
from huggingface_hub import hf_hub_download, list_repo_files

repo_id = "Qwen/Qwen3-32B"
local_dir = "/scratch/longchen/LLM/Qwen3-32B"

# List and download all files
files = list_repo_files(repo_id)
print(f"Downloading {len(files)} files...")

for filename in files:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
    )
    print(f"Downloaded: {filename}")
```

---

## vLLM Server Scripts

| Script | Model | GPUs | Tool Calling |
|--------|-------|------|--------------|
| `submit_vllm_qwen3.sh` | Qwen3-32B | 2x H100 | ✅ Yes (hermes parser) |

### Tool Calling Configuration

Qwen3-32B uses the `hermes` tool parser:

```bash
--enable-auto-tool-choice --tool-call-parser hermes
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VLLM_API_BASE` | vLLM server URL (e.g., `http://gpu027:8000/v1`) |

---

## Using with Paola

### Model Naming

Use the `vllm:` prefix to specify vLLM models:

```bash
python -m paola.cli --model vllm:qwen3-32b
```

### Programmatic Usage

```python
from paola.llm.models import initialize_llm

# Initialize local vLLM model
llm = initialize_llm("vllm:qwen3-32b", temperature=0.1)

# Use in your code
response = llm.invoke("Solve the optimization problem...")
```

---

## Performance Notes

### Model Loading Times

| Model | GPUs | Load Time |
|-------|------|-----------|
| Qwen3-32B | 2x H100 | ~5 min |

### Hardware Requirements

| Model | Min GPUs | VRAM |
|-------|----------|------|
| Qwen3-32B | 2x H100 | 160GB |

---

## Troubleshooting

### "Connection refused" error

The vLLM server isn't running. Check job status:
```bash
squeue -u $USER
tail -f logs/vllm_server_<jobid>.log
```

### Model loading slow

H100 nodes have faster NVMe for model loading.

---

## Files Reference

| File | Purpose |
|------|---------|
| `scripts/submit_vllm_qwen3.sh` | Qwen3-32B on H100 |
| `paola/llm/models.py` | LLM initialization with vLLM support |

---

## Why Not DeepSeek-R1?

DeepSeek-R1 models are **reasoning models** designed for chain-of-thought thinking. They output `<think>...</think>` reasoning blocks instead of structured tool calls.

- GitHub Issue #51: Function calling marked as "not planned"
- vLLM docs: "reasoning model doesn't support tool calling"

For agentic tasks like Paola that require tool calling, use **Qwen3-32B** instead.
