# qwen-plus Conversation Format Fix

## The Problem

When using qwen-plus (instead of qwen-flash), the agent failed on continuation with error:

```
Error code: 400 - An assistant message with "tool_calls" must be followed
by tool messages responding to each "tool_call_id". The following tool_call_ids
did not have response messages: message[6].role
```

## Root Cause

**qwen-plus is stricter about conversation format validation than qwen-flash.**

The original implementation embedded the system prompt in the first HumanMessage:

```python
# OLD CODE (conversational_agent.py)
if is_first_invocation:
    prompt = build_optimization_prompt(context, tools)
    last_user_msg = messages[-1].content if messages else ""
    # Replace with system context + user message
    messages[-1] = HumanMessage(content=f"{prompt}\n\nUser request: {last_user_msg}")
```

This created conversation history like:

```
[
  HumanMessage("System: ...\n\nUser request: optimize rosenbrock"),
  AIMessage(tool_calls=[...]),
  ToolMessage(...),
  AIMessage("Done"),
  HumanMessage("continue")
]
```

**Problem:**
- The long embedded prompt in HumanMessage created an unusual message structure
- qwen-plus's strict validation rejected this format
- Error appeared on continuation, not first turn (because full history is validated)

## The Fix

**Use proper SystemMessage instead of embedding prompt in HumanMessage:**

```python
# NEW CODE (conversational_agent.py)
has_system_message = any(isinstance(m, SystemMessage) for m in messages)

if not has_system_message and messages:
    # First invocation: add system message with context
    prompt = build_optimization_prompt(context, tools)
    # Insert system message at the beginning
    messages.insert(0, SystemMessage(content=prompt))
```

This creates clean conversation history:

```
[
  SystemMessage("System prompt..."),           ← Clean system message
  HumanMessage("optimize rosenbrock"),         ← Pure user request
  AIMessage(tool_calls=[...]),
  ToolMessage(...),
  AIMessage("Done"),
  HumanMessage("continue")                     ← No duplicate system
]
```

## Benefits

1. **Standard format:** Follows LangChain/OpenAI standard message pattern
2. **Compatible with strict providers:** qwen-plus accepts this format
3. **Cleaner separation:** System instructions separate from user messages
4. **No duplication:** System message added once, not embedded repeatedly
5. **Better debugging:** Message roles are clear and standard

## Message Pattern

**Valid conversation structure:**

```
[SystemMessage]                    # Optional, added once at start
[HumanMessage]                     # User request
[AIMessage(with tool_calls)]       # Agent decides to use tools
[ToolMessage(tool_call_id=X)]      # Tool response 1
[ToolMessage(tool_call_id=Y)]      # Tool response 2
[AIMessage(no tool_calls)]         # Agent final response
[HumanMessage]                     # User continues
[AIMessage]                        # Agent responds
...
```

**Requirements for strict providers:**
- SystemMessage at beginning (if used)
- AIMessage with tool_calls MUST be followed by ToolMessage for EACH tool_call_id
- ToolMessages must have matching tool_call_id
- No orphaned tool_calls
- No SystemMessages in the middle of conversation

## Testing

Created `tests/test_conversation_format.py` to verify:
- ✓ SystemMessage added once on first invocation
- ✓ No duplicate system messages on continuation
- ✓ Clean message structure compatible with strict validation
- ✓ All tool_calls have matching ToolMessage responses

## Files Changed

- `paola/agent/conversational_agent.py` - Use SystemMessage instead of embedding
- `tests/test_conversation_format.py` - Verify conversation format

## Verification

**Before fix:**
```bash
paola> optimize 10D rosenbrock
✓ Works (creates evaluator, registers)
paola> continue
✗ Error 400: Invalid conversation format
```

**After fix:**
```bash
paola> optimize 10D rosenbrock
✓ Works (creates evaluator, registers)
paola> continue
✓ Works (conversation continues normally)
```

## Related Issues

This fix also improves compatibility with:
- OpenAI API (expects clean role separation)
- Anthropic Claude API (prefers SystemMessage)
- Azure OpenAI (strict validation)
- Other providers with message format requirements

## Lesson Learned

**Don't embed system prompts in user messages.**

While it works with lenient providers (qwen-flash), it breaks with strict providers (qwen-plus). Use proper SystemMessage for:
- Clean message structure
- Better compatibility
- Easier debugging
- Standard patterns
