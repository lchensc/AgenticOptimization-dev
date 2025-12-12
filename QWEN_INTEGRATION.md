# Qwen LLM Integration

**Date**: December 10, 2025
**Status**: ✅ Complete

## Summary

Updated AOpt to use **Qwen models** as the primary LLM provider, following the same pattern as AdjointFlow's agent_v6.py.

## Changes Made

### 1. **Agent Core** (`aopt/agent/react_agent.py`)

**Added**:
- Multi-provider LLM support (Qwen, Claude, OpenAI)
- `initialize_llm()` function for model-based provider detection
- Environment variable loading via `python-dotenv`
- Model name-based provider detection:
  - Qwen: contains "qwen" or "qwq"
  - OpenAI: contains "gpt" or "openai"
  - Default: Anthropic/Claude

**Code**:
```python
from langchain_qwq import ChatQwen

def initialize_llm(llm_model: str, temperature: float = 0.0):
    """Initialize LLM based on model name."""
    if "qwen" in llm_model.lower():
        if not os.environ.get("DASHSCOPE_API_KEY"):
            raise ValueError("DASHSCOPE_API_KEY not found in .env")
        return ChatQwen(model=llm_model, temperature=temperature)
    # ... (OpenAI, Claude fallbacks)
```

**Updated**:
- `build_aopt_agent()` now accepts `llm_model` parameter
- `create_react_node()` uses `initialize_llm()` instead of hardcoded ChatAnthropic
- Removed hardcoded "claude-sonnet-4-5-20250929"

### 2. **Agent Class** (`aopt/agent/agent.py`)

**Changed**:
- Default model: `"claude-sonnet-4-5-20250929"` → `"qwen-plus"`
- Added `temperature` parameter (default: 0.0)
- Updated docstrings with Qwen model examples
- Pass `llm_model` and `temperature` to `build_aopt_agent()`

**Before**:
```python
def __init__(self, llm_model: str = "claude-sonnet-4-5-20250929", ...):
```

**After**:
```python
def __init__(
    self,
    llm_model: str = "qwen-plus",
    temperature: float = 0.0,
    ...
):
```

### 3. **Dependencies** (`requirements.txt`)

**Added**:
```txt
langchain-qwq>=0.1.0  # Qwen models (primary)
langchain-anthropic>=0.2.0  # Claude models (optional)
langchain-openai>=0.2.0  # OpenAI/GPT models (optional)
```

### 4. **Environment Setup**

**Created**: `.env.example`
```bash
# Qwen API Key (Primary - Required)
DASHSCOPE_API_KEY=your_qwen_api_key_here

# Optional keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

**Usage**:
```bash
cp .env.example .env
# Edit .env and add DASHSCOPE_API_KEY
```

### 5. **Documentation Updates**

**README.md**:
- Added "Setup Environment" section
- Updated Quick Start to show Qwen usage
- Listed supported models (Qwen primary, Claude/OpenAI optional)

**PROGRESS.md**:
- Added design decision: "Qwen models as primary LLM"
- Added design decision: "Multi-provider support"

### 6. **Tests** (`tests/test_agent.py`)

**Updated**:
- `test_agent_creation()`: Uses `"qwen-plus"` instead of `"claude-sonnet-4-5"`
- `test_agent_repr()`: Checks for "qwen" in string representation
- Added `temperature` assertion

**All tests passing**: ✅ 7 passed, 1 skipped

## Supported Models

### Qwen (Primary)
- `"qwen-flash"` - Fast, cheap
- `"qwen-turbo"` - Balanced
- `"qwen-plus"` - **Default**, best quality

**Requirements**:
- Package: `langchain-qwq>=0.1.0`
- API Key: `DASHSCOPE_API_KEY` in .env
- Get key at: https://dashscope.console.aliyun.com/

### Claude (Optional)
- `"claude-sonnet-4"`
- `"claude-3-5-sonnet-20241022"`

**Requirements**:
- Package: `langchain-anthropic>=0.2.0`
- API Key: `ANTHROPIC_API_KEY` in .env

### OpenAI (Optional)
- `"gpt-4"`
- `"gpt-3.5-turbo"`

**Requirements**:
- Package: `langchain-openai>=0.2.0`
- API Key: `OPENAI_API_KEY` in .env

## Usage Example

```python
from paola import Agent

# Qwen (default)
agent = Agent(llm_model="qwen-plus", temperature=0.0)

# Claude (if ANTHROPIC_API_KEY set)
agent = Agent(llm_model="claude-sonnet-4")

# OpenAI (if OPENAI_API_KEY set)
agent = Agent(llm_model="gpt-4")

# Run optimization
result = agent.run("Minimize drag, maintain CL >= 0.8")
```

## Compatibility with AdjointFlow

The implementation follows the **exact same pattern** as AdjointFlow's `agent_v6.py`:

| Feature | AdjointFlow V6 | AOpt |
|---------|----------------|------|
| Multi-provider support | ✅ | ✅ |
| Qwen via langchain-qwq | ✅ | ✅ |
| Model name detection | ✅ | ✅ |
| DASHSCOPE_API_KEY from env | ✅ | ✅ |
| Temperature parameter | ✅ | ✅ |
| Optional thinking mode | ✅ | ⏳ (future) |

## Testing

```bash
# Activate environment
conda activate agent

# Run agent tests
python -m pytest tests/test_agent.py -v

# Result: 7 passed, 1 skipped ✅
```

## Next Steps

1. ✅ Qwen integration complete
2. ⏳ Implement tools (Week 2)
3. ⏳ Test with actual LLM calls once tools are ready
4. ⏳ Consider adding Qwen thinking mode parameter (optional)

---

**Status**: ✅ Qwen integration complete and tested. Agent ready for tool implementation.
