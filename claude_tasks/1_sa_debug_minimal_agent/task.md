# Task: Debug and Improve Invalid Tool Calls Handling

## Task ID: 1_sa_debug_minimal_agent
## Priority: High
## Status: Assigned

---

## Objective

Investigate and improve the handling of `invalid_tool_calls` in the minimal ReAct agent. The current implementation has a hardcoded fix (`_fix_python_json`) that handles specific Qwen LLM quirks. We need to:

1. **Understand** why invalid_tool_calls occur
2. **Research** professional/standard approaches in LangChain ecosystem
3. **Design** a robust, extensible solution
4. **Implement** the improvements

---

## Context

### Project Overview

This is an **Agentic Optimization Platform** where an autonomous AI agent controls optimization processes using tool primitives. The agent uses LangChain/LangGraph with Qwen LLM as the primary model.

**Key files:**
- `debug_agent/minimal_react_agent.py` - Minimal agent with the hardcoded fix (PRIMARY FOCUS)
- `aopt/agent/react_agent.py` - Production agent (needs same fix if solution works)

### Current Problem

In `debug_agent/minimal_react_agent.py` (lines 21-80), there is a hardcoded fix for invalid tool calls:

```python
def _fix_python_json(s: str) -> str:
    """Fix Python syntax in JSON: [x]*n and [x for _ in range(n)]."""
    # ... handles:
    # Pattern 1: [content] * N  -> expanded array
    # Pattern 2: [expr for _ in range(n)] -> expanded array
```

This fix exists because **Qwen LLM sometimes generates Python syntax instead of valid JSON** in tool arguments. For example:
- Instead of `{"bounds": [[-5, 10], [-5, 10], ...]}` (10 items)
- Qwen generates `{"bounds": [[-5, 10]] * 10}` (Python multiplication)

### How Invalid Tool Calls Happen in LangChain

When `llm_with_tools.invoke(messages)` returns an `AIMessage`, it has two relevant attributes:
- `response.tool_calls` - Successfully parsed tool calls (valid JSON)
- `response.invalid_tool_calls` - Tool calls that failed JSON parsing

Invalid tool calls occur when:
1. **LLM generates invalid JSON** (like Python syntax) - Our current issue
2. **Streaming fragments** - Incomplete JSON from streaming chunks
3. **Schema mismatch** - Arguments don't match tool schema
4. **Malformed structure** - Missing required fields

---

## Your Tasks

### Task 1: Root Cause Analysis (Research)

1. **Run the minimal agent** and capture actual invalid_tool_calls:
   ```bash
   cd /home/longchen/PythonCode/gendesign/AgenticOptimization
   export DASHSCOPE_API_KEY=your_key  # (or use existing .env)
   python debug_agent/minimal_react_agent.py  # or test_qwen_rosenbrock.py
   ```

2. **Log the raw invalid_tool_calls** to understand exactly what Qwen produces:
   - Add logging to capture `response.invalid_tool_calls` before any fixing
   - Document the patterns you observe

3. **Check LangChain source code** for how `invalid_tool_calls` is populated:
   - Look at `langchain_core.messages.ai.AIMessage`
   - Understand the parsing logic

### Task 2: Research Professional Approaches

1. **LangChain official patterns** for tool error handling:
   - https://python.langchain.com/docs/how_to/tools_error/
   - https://github.com/langchain-ai/langgraph/discussions/2189

2. **LangGraph prebuilt ToolNode** error handling:
   - Does it have built-in invalid_tool_call handling?
   - Can we leverage it?

3. **Other LLM providers** - Do Claude/OpenAI have this issue? Is it Qwen-specific?

4. **Best practices** for robust tool calling with LangChain:
   - Retry mechanisms
   - Fallback strategies
   - Error message feedback to LLM

### Task 3: Design a Robust Solution

Design an `InvalidToolCallHandler` class or strategy pattern:

```python
# Example design (you may propose different)
class InvalidToolCallHandler:
    """Handles invalid tool calls from LLM."""

    def __init__(self, strategies: list[FixStrategy] = None):
        self.strategies = strategies or [
            JsonSyntaxFixer(),      # Fix JSON syntax errors
            PythonToJsonFixer(),    # Convert Python syntax to JSON
            PartialJsonFixer(),     # Handle incomplete JSON from streaming
            SchemaValidator(),      # Validate against tool schema
        ]

    def handle(self, invalid_call: dict) -> Optional[dict]:
        """Try to fix invalid call, return valid call or None."""
        for strategy in self.strategies:
            try:
                fixed = strategy.fix(invalid_call)
                if fixed:
                    return fixed
            except Exception:
                continue
        return None  # Could not fix

    def create_error_message(self, invalid_call: dict) -> ToolMessage:
        """Create informative error message for LLM feedback."""
        ...
```

Consider:
1. **Extensibility** - Easy to add new fix strategies
2. **Observability** - Log what was fixed and how
3. **Feedback to LLM** - Help LLM learn from errors
4. **Graceful degradation** - What if fix fails?

### Task 4: Implementation

1. **Implement the solution** in a new file or module
2. **Integrate with minimal_react_agent.py** for testing
3. **Add unit tests** for the handler
4. **Document** the approach

---

## Files to Study

| File | Purpose |
|------|---------|
| `debug_agent/minimal_react_agent.py` | Current hardcoded fix (lines 21-80) |
| `aopt/agent/react_agent.py` | Production agent (no invalid_tool_calls handling yet) |
| `debug_agent/test_qwen_rosenbrock.py` | Test script that triggers the issue |
| `aopt/tools/*.py` | Tool definitions (to understand expected schemas) |

---

## Expected Deliverables

1. **Analysis Report**: Document what causes invalid_tool_calls in our system
2. **Design Document**: Proposed solution architecture
3. **Implementation**: Code for robust invalid_tool_call handling
4. **Tests**: Unit tests for the handler
5. **Integration**: Updated minimal_react_agent.py using the new handler

---

## Success Criteria

1. Agent can handle Qwen's Python-syntax-in-JSON quirk robustly
2. Solution is extensible for other fix strategies
3. Failed fixes provide helpful feedback to LLM
4. No hardcoded string manipulation scattered in agent code
5. Clear logging of what was fixed

---

## Notes from Research

From LangChain ecosystem research:

1. **LangChain has `InvalidToolCall` type** in `langchain_core.messages`
2. **Known issues with streaming** where tool_call arguments fragment
3. **LangGraph ToolNode** has error handling that was changed in v1.0.1
4. **Official pattern** suggests catching exceptions and creating `ToolMessage` with error content

Key insight: The professional approach is to:
1. Try to recover invalid calls when possible
2. Send error feedback to LLM in `ToolMessage` so it can retry correctly
3. Use structured error handling, not scattered try-except

---

## Getting Started

```bash
# Navigate to project
cd /home/longchen/PythonCode/gendesign/AgenticOptimization

# Activate environment (if using conda)
conda activate agent

# Install dependencies (if needed)
pip install -r requirements.txt

# Run tests to verify everything works
pytest tests/ -v

# Run the minimal agent to observe invalid_tool_calls
python debug_agent/minimal_react_agent.py
```

---

## Questions to Answer

1. What specific patterns does Qwen produce that cause invalid_tool_calls?
2. Is this issue Qwen-specific or could it happen with Claude/OpenAI too?
3. Should we use LangGraph's built-in ToolNode error handling?
4. How should we feedback errors to the LLM for self-correction?
5. What's the right abstraction level for the fix handler?

---

*Task assigned by: Manager Agent*
*Date: December 11, 2025*
