# Task Result: Invalid Tool Calls Handler

**Task ID**: 1_sa_debug_minimal_agent
**Status**: In Progress
**Date**: December 11, 2025

---

## Phase 1: Root Cause Analysis (COMPLETED)

### What We Discovered

I ran diagnostic tests to capture actual `invalid_tool_calls` from Qwen LLM. Here's what happens:

#### The Problem

When asked to generate tool arguments with repeated list items (e.g., "10 bounds, each being [-5, 10]"), **Qwen generates Python syntax instead of valid JSON**:

**Test Case 1**: "Use test_tool with 10 bounds, each being [-5, 10]"
```json
{"bounds": [[-5, 10] for _ in range(10)]}
```
**Error**: `JSONDecodeError: Expecting ',' delimiter: line 1 column 22`

**Test Case 2**: "Call test_tool where bounds=[[-5,10]] repeated 15 times"
```json
{"bounds": [[-5, 10]] * 15}
```
**Error**: `JSONDecodeError: Expecting ',' delimiter: line 1 column 23`

#### Why This Happens

1. **LangChain's tool binding** sends the tool schema to the LLM
2. **Qwen optimizes for conciseness** - instead of writing `[[-5, 10], [-5, 10], [-5, 10], ...]`, it uses Python shorthand
3. **LangChain's JSON parser fails** because Python syntax is not valid JSON
4. **Result**: The tool call ends up in `response.invalid_tool_calls` instead of `response.tool_calls`

#### Structure of InvalidToolCall

From LangChain source (`langchain_core.messages.InvalidToolCall`):

```python
{
    'name': str,           # Tool name (e.g., "optimizer_create")
    'id': str,             # Unique call ID for threading ToolMessage responses
    'args': str | None,    # Raw arguments string (UNPARSED - still contains Python syntax)
    'error': str | None,   # Error message from JSON parser
    'type': str,           # Always "invalid_tool_call"
}
```

**Key insight**: `args` is a **string** containing the malformed JSON, not a parsed dict.

#### Confirmed Patterns

From my tests, Qwen produces these specific patterns:

1. **List multiplication**: `[item] * N`
   - Example: `[[-5, 10]] * 10`
   - Pattern: `\[([^\]]+)\]\s*\*\s*(\d+)`

2. **List comprehension**: `[item for _ in range(N)]`
   - Example: `[[-5, 10] for _ in range(10)]`
   - Pattern: `\[(.+?)\s+for\s+_\s+in\s+range\((\d+)\)\]`

3. **Valid JSON** (sometimes Qwen gets it right):
   - Example: `[[-5, 10], [-5, 10], [-5, 10]]`
   - This goes to `tool_calls`, not `invalid_tool_calls`

#### Is This Qwen-Specific?

Based on my research:
- **Claude/OpenAI**: Typically more compliant with JSON schemas, but can still produce malformed JSON under certain conditions
- **Qwen models (especially qwen-flash)**: More prone to using Python shorthand
- **Root cause**: LLM training data includes lots of Python code, so Python syntax is "natural" to them

**Conclusion**: This is primarily a Qwen issue, but a robust handler benefits all LLMs.

---

## Phase 2: Research on Professional Approaches (IN PROGRESS)

### LangChain Official Documentation

#### Pattern 1: Error Feedback to LLM

From [LangChain error handling docs](https://python.langchain.com/docs/how_to/tools_error/):

```python
# The professional approach: Let LLM self-correct
for tool_call in invalid_tool_calls:
    error_message = ToolMessage(
        content=f"Error: {tool_call['error']}\nPlease retry with valid JSON.",
        tool_call_id=tool_call['id']
    )
    messages.append(error_message)
    # LLM sees error and retries in next turn
```

**Pros**:
- LLM learns from mistakes
- No hardcoded string manipulation
- Natural ReAct loop

**Cons**:
- Requires extra LLM call (expensive)
- May not work if LLM consistently makes the same mistake
- Slower (adds iteration)

#### Pattern 2: Attempt Recovery First, Then Feedback

The hybrid approach (what we're implementing):

```python
for invalid_call in invalid_tool_calls:
    # Try to fix automatically
    fixed = handler.try_fix(invalid_call)

    if fixed:
        # Success! Use the fixed call
        tool_calls.append(fixed)
    else:
        # Failed - send error to LLM for retry
        error_msg = ToolMessage(
            content=f"Could not parse arguments: {invalid_call['error']}",
            tool_call_id=invalid_call['id']
        )
        messages.append(error_msg)
```

**Pros**:
- Best of both worlds: fix simple issues automatically, escalate complex ones
- Saves LLM calls for known patterns
- Maintains observability (log what was fixed)

**Cons**:
- More complex implementation
- Risk of "too smart" fixes that mask underlying issues

### LangGraph ToolNode Research

I examined the LangGraph prebuilt `ToolNode` to see how it handles errors:

```python
# From langgraph.prebuilt.tool_node (v1.0.1+)
class ToolNode:
    def __call__(self, state):
        for tool_call in state.tool_calls:
            try:
                result = tool.invoke(tool_call.args)
                # ... wrap in ToolMessage
            except Exception as e:
                # Send error back to LLM
                error_msg = ToolMessage(
                    content=f"Error: {str(e)}",
                    tool_call_id=tool_call.id
                )
```

**Insight**: LangGraph's official approach is "execute and catch errors", not "fix invalid JSON". This suggests:
- Invalid JSON fixing is **application-specific**, not part of LangGraph core
- Our handler is the right level of abstraction

### Best Practices Summary

From research, the professional approach is:

1. **Observability First**: Log all invalid calls for debugging
2. **Try Recovery**: Attempt automated fixes for known patterns
3. **Feedback Errors**: Send unfixable errors back to LLM
4. **Extensibility**: Use strategy pattern for different fix types
5. **Safety**: Never silently fail - either fix or error

---

## Phase 3: Design (NEXT)

### Proposed Architecture: InvalidToolCallHandler

```python
class InvalidToolCallHandler:
    """
    Handles invalid tool calls from LLM with extensible fix strategies.

    Design principles:
    - Strategy pattern for different fix types
    - Observability (log all fixes)
    - Graceful degradation (return None if unfixable)
    - Clear error messages for LLM feedback
    """

    def __init__(self, strategies: list[FixStrategy] = None):
        self.strategies = strategies or self._default_strategies()
        self.stats = {"total": 0, "fixed": 0, "failed": 0}

    def _default_strategies(self):
        return [
            PythonListMultiplicationFixer(),  # [x] * N
            PythonListComprehensionFixer(),   # [x for _ in range(N)]
            # Future: JsonSyntaxFixer(), PartialJsonFixer(), etc.
        ]

    def handle(self, invalid_call: dict) -> Optional[dict]:
        """
        Try to fix an invalid tool call.

        Returns:
            dict: Fixed tool call (ready to use) or None if unfixable
        """
        # ... (detailed design below)

    def create_error_message(self, invalid_call: dict) -> ToolMessage:
        """Create informative error message for LLM feedback."""
        # ... (detailed design below)
```

#### Strategy Interface

```python
class FixStrategy(ABC):
    """Base class for fix strategies."""

    @abstractmethod
    def can_fix(self, args_str: str) -> bool:
        """Check if this strategy can handle the args string."""
        pass

    @abstractmethod
    def fix(self, args_str: str) -> dict:
        """Fix the args string, return parsed dict."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging."""
        pass
```

#### Concrete Strategies

```python
class PythonListMultiplicationFixer(FixStrategy):
    """Fix [item] * N patterns."""

    def can_fix(self, args_str: str) -> bool:
        return bool(re.search(r'\[.*?\]\s*\*\s*\d+', args_str))

    def fix(self, args_str: str) -> dict:
        # Implementation from current _fix_python_json()
        # ...
```

```python
class PythonListComprehensionFixer(FixStrategy):
    """Fix [item for _ in range(N)] patterns."""

    def can_fix(self, args_str: str) -> bool:
        return bool(re.search(r'\[.+?\s+for\s+_\s+in\s+range\(\d+\)\]', args_str))

    def fix(self, args_str: str) -> dict:
        # Implementation from current _fix_python_json()
        # ...
```

### Integration Points

**In `minimal_react_agent.py`** (lines 131-150):

```python
# Current (hardcoded):
invalid_calls = getattr(response, 'invalid_tool_calls', None) or []
for inv in invalid_calls:
    try:
        fixed_args = _parse_tool_args(inv['args'])  # Hardcoded fix
        tool_calls.append(...)
    except Exception as e:
        ...

# After refactor (clean):
handler = InvalidToolCallHandler()  # Instantiate once at agent creation

invalid_calls = getattr(response, 'invalid_tool_calls', None) or []
for inv in invalid_calls:
    fixed_call = handler.handle(inv)
    if fixed_call:
        tool_calls.append(fixed_call)
    else:
        # Send error to LLM for retry
        messages.append(handler.create_error_message(inv))
```

---

## Next Steps

1.  Root cause analysis (DONE)
2. = Research professional approaches (IN PROGRESS - documenting now)
3. � Finalize design document
4. � Implement InvalidToolCallHandler
5. � Write unit tests
6. � Integrate with minimal_react_agent.py
7. � Test with real Qwen agent run
8. � (Optional) Integrate with production react_agent.py

---

## Files Created

- `/home/longchen/PythonCode/gendesign/AgenticOptimization/debug_agent/debug_invalid_calls.py` - Debug script
- `/home/longchen/PythonCode/gendesign/AgenticOptimization/debug_agent/invalid_calls_log.jsonl` - Captured patterns

---

## References

- LangChain InvalidToolCall: `langchain_core.messages.InvalidToolCall`
- LangChain error handling: https://python.langchain.com/docs/how_to/tools_error/
- LangGraph ToolNode: `langgraph.prebuilt.tool_node`

---

## Phase 4: Implementation (COMPLETED)

### File Structure

```
aopt/agent/
└── invalid_tool_call_handler.py  # Main handler implementation (350 lines)

tests/
└── test_invalid_tool_call_handler.py  # Unit tests (400 lines, 22 tests)

debug_agent/
├── minimal_react_agent.py  # Updated to use handler
├── debug_invalid_calls.py  # Debug script for pattern discovery
└── test_handler_integration.py  # Integration test
```

### Handler Implementation

The `InvalidToolCallHandler` class provides:

1. **Strategy Pattern**: Extensible fix strategies
   - `PythonListMultiplicationFixer` - Handles `[item] * N`
   - `PythonListComprehensionFixer` - Handles `[item for _ in range(N)]`
   - Easy to add more strategies

2. **Observability**: Full logging and statistics
   - Logs every fix attempt (if verbose=True)
   - Tracks: `total`, `fixed`, `failed`
   - Reports which strategy fixed each call

3. **Graceful Degradation**:
   - Returns `None` if unfixable
   - Creates informative error message for LLM feedback
   - Never crashes on unexpected input

4. **Safety**:
   - Tries parsing as-is first (handles false alarms)
   - Only applies strategies to truly invalid JSON
   - Each strategy is isolated (one failure doesn't affect others)

### Code Example

```python
# In minimal_react_agent.py
from aopt.agent.invalid_tool_call_handler import InvalidToolCallHandler

# Create handler (per agent run)
handler = InvalidToolCallHandler(verbose=True)

# In ReAct loop
for invalid_call in response.invalid_tool_calls:
    fixed_call = handler.handle(invalid_call)

    if fixed_call:
        # Success - add to tool_calls
        tool_calls.append(fixed_call)
    else:
        # Failed - send error to LLM
        messages.append(handler.create_error_message(invalid_call))

# At end
print(f"Stats: {handler.get_stats()}")
# {'total': 1, 'fixed': 1, 'failed': 0}
```

### Test Results

**Unit Tests**: 22 tests, all passing
```bash
$ pytest tests/test_invalid_tool_call_handler.py -v
================================ 22 passed =================================
```

Coverage:
- Individual strategy tests (multiplication, comprehension)
- Handler integration tests
- Edge cases (empty args, missing keys, zero multiplication)
- Real-world scenarios (Rosenbrock 10D from debug logs)
- Statistics tracking
- Custom strategies

**Integration Test**: Agent successfully uses handler
```bash
$ python debug_agent/test_handler_integration.py
[Iteration 1]
  (fixed invalid tool call using PythonListMultiplication)
  -> create_bounds(bounds=[...10 items])
  <- Created 10 bounds: [[-5, 10], [-5, 10], [-5, 10]]...

Invalid calls stats: {'total': 1, 'fixed': 1, 'failed': 0}

✓ Handler successfully fixed invalid tool calls!
```

---

## Phase 5: Benefits and Impact

### Improvements Over Hardcoded Fix

**Before** (`_fix_python_json` in minimal_react_agent.py):
- Hardcoded in agent logic (lines 21-80)
- No observability (silent fixes)
- No statistics tracking
- Hard to extend (must edit agent code)
- No error feedback to LLM
- Scattered error handling

**After** (`InvalidToolCallHandler`):
- Clean separation of concerns (dedicated module)
- Full observability (logs every fix)
- Statistics tracking (`total`, `fixed`, `failed`)
- Easy to extend (add new strategies)
- Error feedback to LLM for unfixable cases
- Centralized error handling
- Comprehensive unit tests

### Code Reduction

**Before**: 60 lines of hardcoded fix logic in agent
**After**: 3 lines in agent + reusable handler module

### Future Extensibility

Easy to add new fix strategies:

```python
class JsonSyntaxFixer(FixStrategy):
    """Fix common JSON syntax errors."""

    def can_fix(self, args_str: str) -> bool:
        return "'" in args_str or args_str.endswith(",")

    def fix(self, args_str: str) -> str:
        fixed = args_str.replace("'", '"')
        fixed = re.sub(r',\s*}', '}', fixed)
        return fixed
```

---

## Summary

### What Was Accomplished

1. ✅ **Root Cause Analysis**: Identified Qwen's Python syntax patterns
2. ✅ **Research**: Studied LangChain best practices for error handling
3. ✅ **Design**: Created extensible strategy pattern architecture
4. ✅ **Implementation**: Built robust handler with 350 lines
5. ✅ **Testing**: Wrote 22 unit tests, all passing
6. ✅ **Integration**: Updated minimal_react_agent.py successfully
7. ✅ **Validation**: Verified with real Qwen LLM calls

### Files Modified/Created

**Created**:
- `aopt/agent/invalid_tool_call_handler.py` (350 lines)
- `tests/test_invalid_tool_call_handler.py` (400 lines, 22 tests)
- `debug_agent/debug_invalid_calls.py` (debug script)
- `debug_agent/test_handler_integration.py` (integration test)

**Modified**:
- `debug_agent/minimal_react_agent.py` (refactored to use handler)

### Key Achievements

1. **Robust Error Handling**: Handles both known and unknown patterns
2. **Extensibility**: Strategy pattern allows easy addition of new fix types
3. **Observability**: Full logging and statistics tracking
4. **Clean Architecture**: Separation of concerns, no hardcoded logic in agent
5. **LLM Feedback**: Sends informative errors back to LLM for unfixable cases
6. **Well-Tested**: 22 unit tests covering all scenarios

---

**Status**: ✅ COMPLETED
**Date**: December 11, 2025

---

## MANAGER UPDATE: Simplified Approach (December 11, 2025)

### Problem with Previous Solution

The subagent created an over-engineered `InvalidToolCallHandler` class (~350 lines) with strategy patterns. **This is NOT the professional approach.**

### The Standard LangChain Pattern

After consulting LangChain documentation and best practices, the **correct professional approach** is:

**❌ Don't try to fix invalid tool calls**
**✅ Send error feedback to LLM and let it self-correct**

### Why This Is Better

1. **Simplicity**: 10 lines vs 350 lines
2. **Standard practice**: What LangChain/LangGraph does
3. **LLM learning**: LLM sees errors and learns
4. **Maintainability**: No complex strategies to maintain

### Implementation (Final)

```python
# Handle invalid_tool_calls - send error feedback to LLM for self-correction
invalid_calls = getattr(response, 'invalid_tool_calls', None) or []
for inv in invalid_calls:
    tool_name = inv.get('name', 'unknown')
    error_msg = inv.get('error', 'Failed to parse tool call')
    tool_id = inv.get('id', 'unknown')

    # Send error back to LLM via ToolMessage
    messages.append(ToolMessage(
        content=f"Error calling {tool_name}: {error_msg}\n"
                f"Please check the tool schema and provide valid JSON arguments.",
        tool_call_id=tool_id,
        status="error"
    ))
```

### Files Updated (Final)

1. **`debug_agent/minimal_react_agent.py`** - Lines 68-88: Simple error feedback
2. **`aopt/agent/react_agent.py`** - Lines 271-301: Same pattern with event emission

### Test Results

All 99 tests pass:
```bash
$ python -m pytest tests/ -v --tb=short -k "not llm"
================== 99 passed, 1 skipped, 8 warnings in 2.81s ===================
```

### Files to Ignore/Remove

The following files created by subagent are **not needed** and can be removed:
- `aopt/agent/invalid_tool_call_handler.py` (complex handler)
- `tests/test_invalid_tool_call_handler.py` (tests for unused handler)
- `debug_agent/debug_invalid_calls.py` (debug script)

### References

- [LangChain Tool Error Handling](https://python.langchain.com/docs/how_to/tools_error/)
- [LangGraph ToolNode](https://langchain-opentutorial.gitbook.io/langchain-opentutorial/17-langgraph/01-core-features/10-lnaggraph-toolnode)
- [Tool Message API](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.tool.ToolMessage.html)

### Key Takeaway

**Professional practice**: Don't try to outsmart the LLM. Send error feedback and let it learn.

---

**FINAL STATUS**: ✅ COMPLETED with simplified approach
**Manager**: Applied standard LangChain pattern (10 lines)
**Outcome**: Clean, maintainable, professional solution
