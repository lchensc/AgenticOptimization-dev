# Local LLM Setup for Paola

This guide explains how to run Paola with local LLMs using vLLM on the cluster.

## Overview

Running a local LLM provides:
- **Zero API costs** - No per-token charges
- **Data privacy** - All processing stays on your cluster
- **No rate limits** - Unlimited requests
- **Offline capability** - Works without internet

## Available Models

| Model | Location | Size | Context | Best For |
|-------|----------|------|---------|----------|
| Qwen3-32B | `/scratch/longchen/LLM/Qwen3-32B` | 59GB | 40K | General purpose |
| DeepSeek-R1-Distill-Qwen-32B | `/scratch/longchen/LLM/DeepSeek-R1-Distill-Qwen-32B` | 45GB | 128K | Reasoning tasks |
| DeepSeek-R1-Distill-Llama-70B | `/scratch/longchen/LLM/DeepSeek-R1-Distill-Llama-70B` | 140GB | 128K | Complex reasoning |

## Quick Start

### Step 1: Submit a vLLM Server Job

```bash
cd /scratch/longchen/AgenticOptimization-dev

# For Qwen3-32B (2x H100)
sbatch scripts/submit_vllm_qwen3.sh

# For DeepSeek-R1-Distill-Qwen-32B (2x H100)
sbatch scripts/submit_vllm_deepseek_qwen32b.sh

# For DeepSeek-R1-Distill-Llama-70B (16x V100 on DGX)
sbatch scripts/submit_vllm_deepseek.sh
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
# or: python -m paola.cli --model vllm:deepseek-r1-qwen32b
# or: python -m paola.cli --model vllm:deepseek-r1
```

---

## Downloading Models from HuggingFace

Use Python with `huggingface_hub` for reliable downloads:

```python
from huggingface_hub import hf_hub_download, list_repo_files

repo_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
local_dir = "/scratch/longchen/LLM/DeepSeek-R1-Distill-Qwen-32B"

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

| Script | Model | GPUs | Tool Parser |
|--------|-------|------|-------------|
| `submit_vllm_qwen3.sh` | Qwen3-32B | 2x H100 | hermes |
| `submit_vllm_deepseek_qwen32b.sh` | DeepSeek-R1 Qwen 32B | 2x H100 | hermes |
| `submit_vllm_deepseek.sh` | DeepSeek-R1 Llama 70B | 16x V100 | llama3_json |

### Tool Calling Configuration

| Model Base | Tool Parser | vLLM Flag |
|------------|-------------|-----------|
| Qwen-based | `hermes` | `--tool-call-parser hermes` |
| Llama-based | `llama3_json` | `--tool-call-parser llama3_json` |

All servers enable tool calling with: `--enable-auto-tool-choice`

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
python -m paola.cli --model vllm:deepseek-r1-qwen32b
python -m paola.cli --model vllm:deepseek-r1
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
| DeepSeek-R1 Qwen 32B | 2x H100 | ~30 min |
| DeepSeek-R1 Llama 70B | 16x V100 | ~40 min |

### Hardware Requirements

| Model | Min GPUs | VRAM |
|-------|----------|------|
| 32B models | 2x H100 | 160GB |
| 70B model | 16x V100 | 512GB |

---

## Troubleshooting

### "Connection refused" error

The vLLM server isn't running. Check job status:
```bash
squeue -u $USER
tail -f logs/vllm_server_<jobid>.log
```

### Model loading slow

H100 nodes have faster NVMe. For V100, use DGX nodes (dgx001) for faster I/O.

---

## Files Reference

| File | Purpose |
|------|---------|
| `scripts/submit_vllm_qwen3.sh` | Qwen3-32B on H100 |
| `scripts/submit_vllm_deepseek_qwen32b.sh` | DeepSeek-R1 Qwen 32B on H100 |
| `scripts/submit_vllm_deepseek.sh` | DeepSeek-R1 Llama 70B on V100 |
| `paola/llm/models.py` | LLM initialization with vLLM support |
