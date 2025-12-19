# Paola Quick Start on Cluster

## Current Setup Status

✅ Dependencies installed in conda environment `ml`
⏳ DeepSeek-R1-Distill-Llama-70B downloading (10/17 shards complete)

---

## Option 1: Use Paola with Claude/GPT (Working Now)

Since the dependencies are installed, you can use Paola **right now** with API-based models:

### Using Claude (Anthropic)
```bash
conda activate ml
export ANTHROPIC_API_KEY="your-api-key-here"
python -m paola.cli --model claude-sonnet-4
```

### Using GPT (OpenAI)
```bash
conda activate ml
export OPENAI_API_KEY="your-api-key-here"
python -m paola.cli --model gpt-4
```

---

## Option 2: Use Paola with Local DeepSeek-R1 (After Download Completes)

Once the model download finishes, you can run it locally with zero API costs:

### Step 1: Wait for model download to complete
```bash
# Check progress
/scratch/longchen/LLM/check_download_progress.sh
```

### Step 2: Start vLLM server (when download completes)
```bash
# Activate environment
conda activate ml

# Start vLLM server with both A30 GPUs
python3 -m vllm.entrypoints.openai.api_server \
    --model /scratch/longchen/LLM/DeepSeek-R1-Distill-Llama-70B \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --port 8000
```

This will start an OpenAI-compatible API server on port 8000.

### Step 3: Connect Paola to local model
```bash
# In a new terminal
conda activate ml
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=dummy  # vLLM doesn't need real key for local use

python -m paola.cli --model gpt-3.5-turbo  # Any model name works with vLLM
```

---

## Option 3: Use with Ollama (Alternative)

If you prefer Ollama over vLLM:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull DeepSeek-R1 (quantized version)
ollama pull deepseek-r1:70b

# Start Ollama server
ollama serve &

# Use with Paola
export OLLAMA_BASE_URL=http://localhost:11434
python -m paola.cli --model ollama:deepseek-r1:70b
```

---

## Troubleshooting

### Error: "Qwen requires langchain-qwq"
**Solution**: Specify a different model explicitly:
```bash
python -m paola.cli --model claude-sonnet-4
# or
python -m paola.cli --model ollama:deepseek-r1:70b
```

### Error: No API key found
**Solution**: Set the appropriate environment variable:
```bash
# For Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# For OpenAI
export OPENAI_API_KEY="sk-..."

# For Qwen (if you install langchain-qwq)
export DASHSCOPE_API_KEY="sk-..."
```

### Check model download progress
```bash
/scratch/longchen/LLM/check_download_progress.sh
```

Expected output when complete:
```
Current size: ~140GB
Shards downloaded: 17 / 17
✓ Download appears complete!
```

---

## Recommended Workflow for Testing

1. **Test Paola functionality now** with Claude or GPT to familiarize yourself
2. **Monitor download progress** in the background
3. **Once download completes**, switch to local DeepSeek-R1-70B model
4. **Compare performance** between API models and local model

---

## Model Download Status

Current progress: Use this command to check:
```bash
/scratch/longchen/LLM/check_download_progress.sh
```

Or monitor live:
```bash
tail -f /tmp/claude/-scratch-longchen/tasks/b6ec8f1.output
```

---

## Hardware Compatibility

Your 2x NVIDIA A30 GPUs (48GB total) are **perfect** for the 70B model!

- Model VRAM requirement: ~43GB
- Your available VRAM: 48GB
- Method: Tensor parallelism (model split across both GPUs)
- Expected performance: 15-30 tokens/second

See [A30_GPU_Compatibility.md](../LLM/A30_GPU_Compatibility.md) for detailed analysis.
