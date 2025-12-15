# Bug Fix: Conversational Agent `.invoke()` Method

**Date**: 2025-12-14
**Issue**: `'function' object has no attribute 'invoke'`
**Status**: ✅ FIXED

---

## Problem

When trying to use the conversational agent in the CLI:

```
paola> create a 2D rosenbrock problem from registered evaluators
Agent error: 'function' object has no attribute 'invoke'
```

### Root Cause

The conversational agent returned a plain Python function, but the CLI expected an object with a `.invoke()` method (like LangGraph's compiled graph).

**CLI code** (`paola/cli/repl.py:233`):
```python
final_state = self.agent.invoke(state, config=config)
```

**Conversational agent** (`paola/agent/conversational_agent.py:227`):
```python
def build_conversational_agent(...):
    def invoke_agent(state: dict) -> dict:
        ...
    return invoke_agent  # ← Plain function, no .invoke() method
```

**ReAct agent** (for comparison):
```python
def build_optimization_agent(...):
    workflow = StateGraph(AgentState)
    ...
    return workflow.compile()  # ← LangGraph graph with .invoke() method
```

---

## Solution

Created a wrapper class `ConversationalAgentExecutor` that provides the same API as LangGraph's compiled graph.

### Implementation

**File**: `paola/agent/conversational_agent.py`

```python
class ConversationalAgentExecutor:
    """
    Wrapper for conversational agent function to provide LangGraph-compatible API.

    This allows the conversational agent to be used with the same interface as
    the LangGraph-based ReAct agent (which has .invoke() method).
    """

    def __init__(self, agent_func):
        """
        Initialize executor.

        Args:
            agent_func: The conversational agent function
        """
        self.agent_func = agent_func

    def invoke(self, state: dict, config: dict = None) -> dict:
        """
        Invoke the agent (LangGraph-compatible API).

        Args:
            state: Agent state dict
            config: Optional config dict (for callbacks, etc.)

        Returns:
            Updated state dict
        """
        # Note: config contains callbacks for token tracking, but we ignore it here
        # because the agent_func already has callback_manager in closure
        return self.agent_func(state)
```

**Updated return statement**:
```python
def build_conversational_agent(...):
    ...
    def invoke_agent(state: dict) -> dict:
        ...

    return ConversationalAgentExecutor(invoke_agent)  # ← Wrapped in executor
```

---

## Additional Fixes

### Issue 2: Event Creation Parameter Errors

**Error**:
```
create_event() got an unexpected keyword argument 'tool_name'
```

**Root Cause**: The `create_event()` function signature only accepts:
- `event_type`
- `iteration`
- `data`
- `context`

But the conversational agent was passing `tool_name` as a direct parameter.

**Before** (incorrect):
```python
callback_mgr.emit(create_event(
    event_type=EventType.TOOL_CALL,
    tool_name=tool_name,  # ← Wrong! Not a valid parameter
    data={"args": tool_args}
))
```

**After** (correct):
```python
callback_mgr.emit(create_event(
    event_type=EventType.TOOL_CALL,
    iteration=iteration,
    data={"tool_name": tool_name, "args": tool_args}  # ← tool_name in data dict
))
```

Fixed in two locations:
1. Line 178-182: TOOL_CALL event
2. Line 206-210: TOOL_RESULT event

---

## Files Modified

1. **`paola/agent/conversational_agent.py`**
   - Added `ConversationalAgentExecutor` class (lines 26-56)
   - Updated `build_conversational_agent()` return statement (line 227)
   - Fixed `create_event()` calls to put `tool_name` in data dict (lines 178-182, 206-210)

2. **`test_conversational_agent.py`**
   - Updated test calls from `agent(state)` to `agent.invoke(state)`

3. **`test_cli_integration.py`** (new test file)
   - Created integration test to verify CLI works with conversational agent

---

## Testing

### Test 1: Unit Tests

**File**: `test_conversational_agent.py`

```bash
python test_conversational_agent.py
```

**Results**:
- ✅ Test 1 (List evaluators): 1 tool call, stopped after listing
- ✅ Test 2 (Create problem): 1 tool call, did NOT optimize
- ✅ All tests PASS

### Test 2: CLI Integration

**File**: `test_cli_integration.py`

```bash
python test_cli_integration.py
```

**Results**:
```
User input: 'create a 2D rosenbrock problem from registered evaluators'
--------------------------------------------------------------------------------
🔧 create_benchmark_problem...
✓ create_benchmark_problem completed

I've successfully created a 2D Rosenbrock problem...

✅ SUCCESS: Agent processed request without errors
```

### Test 3: Manual CLI Test

```bash
python -m paola.cli.main
```

```
paola> create a 2D rosenbrock problem from registered evaluators
🔧 create_benchmark_problem...
✓ create_benchmark_problem completed

I've successfully created a 2D Rosenbrock problem with the identifier 'rosenbrock_2d'.
```

**Result**: ✅ WORKS

---

## API Compatibility

Both agents now have the same interface:

```python
# Conversational agent
agent = build_conversational_agent(tools, llm_model, callback_manager, temperature)
result = agent.invoke(state, config={"callbacks": [...]})

# ReAct agent
agent = build_optimization_agent(tools, llm_model, callback_manager, temperature)
result = agent.invoke(state, config={"callbacks": [...]})
```

This allows the CLI to use either agent type without code changes:

```python
if self.agent_type == "conversational":
    self.agent = build_conversational_agent(...)
else:  # "react"
    self.agent = build_optimization_agent(...)

# Same invoke() call works for both
final_state = self.agent.invoke(state, config=config)
```

---

## Summary

**Problem**: Conversational agent returned plain function without `.invoke()` method

**Solution**: Wrapped function in `ConversationalAgentExecutor` class that provides LangGraph-compatible API

**Additional**: Fixed event creation to put `tool_name` in data dict instead of passing as parameter

**Status**: ✅ FIXED and TESTED

**Impact**: CLI now works with conversational agent (default) and maintains backward compatibility with ReAct agent

---

## Lessons Learned

### 1. API Consistency Matters

Even though the conversational agent doesn't use LangGraph internally, it should provide the same API as the ReAct agent for drop-in replacement.

**Learning**: When replacing one implementation with another, match the interface exactly.

### 2. Test Integration Points

Unit tests passed, but integration with CLI failed because the API contract wasn't met.

**Learning**: Test the integration, not just the isolated component.

### 3. Understand Event System

The `create_event()` signature requires specific parameters - can't just pass arbitrary kwargs.

**Learning**: Check function signatures before calling, especially for framework utilities.

---

## Code Statistics

**Lines added**: 43 (wrapper class + documentation)
**Lines changed**: 6 (return statement + event calls)
**Files modified**: 3
**Tests created**: 1

**Total changes**: ~50 lines to fix critical bug and ensure API compatibility
