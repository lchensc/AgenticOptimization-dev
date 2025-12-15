# Conversational Agent Integration

**Date**: 2025-12-14
**Status**: ✅ IMPLEMENTED
**Related**: `docs/minimal_prompt_philosophy.md`

---

## Overview

PAOLA now supports **two agent architectures**:

1. **ReAct Agent** (original): Autonomous task completion with self-driven loop
2. **Conversational Agent** (new, default): Interactive request-response like Claude Code

The conversational agent addresses the problem where the ReAct agent was executing entire workflows autonomously instead of responding to specific user requests.

---

## Problem: ReAct Agent Too Autonomous

### What Was Happening

**User request**: "create a 2D rosenbrock problem from registered evaluators"

**ReAct agent behavior**:
```
🔧 foundry_list_evaluators()
🔧 create_nlp_problem(...)
🔧 start_optimization_run(...)
🔧 run_scipy_optimization(...)
🔧 finalize_optimization_run(...)
🔧 analyze_convergence(...)
💬 DONE - Successfully optimized!
```

**10+ tool calls** executing entire workflow when user only asked to create a problem!

### Root Cause

The ReAct agent uses an **autonomous loop architecture**:

```python
# paola/agent/react_agent.py
workflow.add_conditional_edges(
    "react",
    lambda state: "end" if state["done"] else "continue",
    {
        "continue": "react",  # ← Keeps looping autonomously!
        "end": END
    }
)
```

This loop continues until the agent decides the entire imagined task is "done". The agent interprets "create problem" as "complete an optimization project" and executes the full workflow.

### Why Prompts Didn't Fix It

Initial attempts tried to fix behavior with prescriptive prompts:
- ❌ "Use tool X first, then Y"
- ❌ "When user says A, do B"
- ❌ Adding workflow hints and decision trees

**None of this worked** because the issue was architectural, not prompt-based.

---

## Solution: Conversational Agent

### Architecture

The conversational agent uses a **request-response pattern** like Claude Code:

```python
# paola/agent/conversational_agent.py
def invoke_agent(state: dict) -> dict:
    """Process one user message with ReAct cycles."""

    # ReAct cycles: keep processing until final text response
    while iteration < max_iterations:
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Check if response has tool calls
        if not response.tool_calls:
            # No tool calls → final text response → STOP
            return {
                "messages": messages,
                "done": True,  # Always done after giving final response
                "iteration": iteration
            }

        # Execute tool calls
        for tool_call in response.tool_calls:
            tool_result = execute_tool(tools, tool_call)
            messages.append(tool_result)

        # Continue loop - LLM reasons about tool results
```

**Key difference**: Stops when `not response.tool_calls` (LLM gives final text response), NOT when agent decides entire imagined workflow is "done".

### Does It Still Use ReAct?

**Yes!** The conversational agent still uses **Reasoning + Acting cycles**:

```
User: "create a 2D rosenbrock problem from registered evaluators"

Agent ReAct cycles (within single response):
  Cycle 1:
    💭 Reasoning: "I need to list evaluators first to see what's available"
    🔧 Action: foundry_list_evaluators()
    👁️ Observation: [list of evaluators including rosenbrock_eval]

  Cycle 2:
    💭 Reasoning: "Found rosenbrock_eval, I'll use it to create NLP problem"
    🔧 Action: create_nlp_problem(evaluator_id="rosenbrock_eval", ...)
    👁️ Observation: {success: true, problem_id: "rosenbrock_2d_nlp"}

  Cycle 3:
    💭 Reasoning: "Problem created successfully, I can give final response"
    💬 Final Response: "Created NLP problem 'rosenbrock_2d_nlp' with..."
    🛑 STOP (no more tool calls)
```

**Difference from ReAct agent**: The cycles are scoped to the specific user request, not an entire imagined workflow.

---

## Behavior Comparison

### Example 1: List Evaluators

**User**: "list registered evaluators"

**ReAct Agent** (autonomous):
```
🔧 foundry_list_evaluators()
💬 "Found 12 evaluators. Should I create an optimization problem?"
🔧 create_nlp_problem(...) [assumes user wants this]
🔧 start_optimization_run(...) [assumes next step]
💬 "DONE - Ready to optimize!"
```

**Conversational Agent** (interactive):
```
🔧 foundry_list_evaluators()
💬 "Here are the 12 registered evaluators: [list]"
🛑 STOP - waits for next instruction
```

### Example 2: Create Problem

**User**: "create a 2D rosenbrock problem from registered evaluators"

**ReAct Agent** (autonomous):
```
🔧 foundry_list_evaluators()
🔧 create_nlp_problem(...)
🔧 start_optimization_run(...)
🔧 run_scipy_optimization(...)
🔧 analyze_convergence(...)
🔧 finalize_optimization_run(...)
💬 "DONE - Optimization completed! Final objective: 1.8e-8"
```

**Conversational Agent** (interactive):
```
🔧 foundry_list_evaluators() [checks what's available]
🔧 create_nlp_problem(evaluator_id="rosenbrock_eval", ...)
💬 "Created NLP problem 'rosenbrock_2d_nlp'. Recommended solvers: SLSQP, L-BFGS-B."
🛑 STOP - waits for next instruction
```

### Example 3: Multi-Step Request

**User**: "create and optimize a 2D rosenbrock problem with SLSQP"

**Both agents** (user explicitly requested multiple steps):
```
🔧 foundry_list_evaluators()
🔧 create_nlp_problem(...)
🔧 run_scipy_optimization(algorithm="SLSQP", ...)
💬 "Optimization completed. Final design: [0.999, 0.999], objective: 1.8e-8"
🛑 STOP
```

**Key**: Conversational agent still executes multi-step workflows when user explicitly requests them!

---

## Implementation

### Files Created/Modified

1. **`paola/agent/conversational_agent.py`** (NEW - 226 lines)
   - Main conversational agent implementation
   - ReAct cycles scoped to user requests
   - Stops after completing specific request

2. **`paola/agent/__init__.py`** (MODIFIED)
   - Exports both agent types
   - Documented differences

3. **`paola/cli/repl.py`** (MODIFIED)
   - Added `agent_type` parameter (default: "conversational")
   - Conditional agent initialization
   - Displays agent type on startup

### API

```python
# Build conversational agent (new)
from paola.agent import build_conversational_agent

agent = build_conversational_agent(
    tools=tools,
    llm_model="qwen-plus",
    callback_manager=callback_manager,
    temperature=0.0
)

# Build ReAct agent (original)
from paola.agent import build_optimization_agent

agent = build_optimization_agent(
    tools=tools,
    llm_model="qwen-plus",
    callback_manager=callback_manager,
    temperature=0.0
)
```

### CLI Usage

```python
# Use conversational agent (default)
from paola.cli.repl import AgenticOptREPL

repl = AgenticOptREPL(
    llm_model="qwen-flash",
    agent_type="conversational"  # Default
)
repl.run()

# Use ReAct agent (autonomous)
repl = AgenticOptREPL(
    llm_model="qwen-flash",
    agent_type="react"  # Opt-in to autonomous behavior
)
repl.run()
```

### CLI Output

```
PAOLA - Platform for Agentic Optimization with Learning and Analysis
The optimization platform that learns from every run

Version 0.1.0

Type your optimization goals in natural language.
Type '/help' for commands, '/exit' to quit.

Initializing agent...
✓ Agent ready! (type: conversational)

paola> create a 2D rosenbrock problem from registered evaluators
```

---

## Testing

### Test Results

**Test Script**: `test_conversational_agent.py`

#### Test 1: List Evaluators
```
User: "list registered evaluators"

Result:
  - Tool calls: 1 (foundry_list_evaluators)
  - Stopped after listing
  - Did NOT create problem or run optimization

✅ PASS: Agent executed only what was asked
```

#### Test 2: Create Problem
```
User: "create a 2D rosenbrock problem from registered evaluators"

Result:
  - Tool calls: 1 (create_benchmark_problem)
  - Created problem
  - Did NOT run optimization, analysis, or finalization
  - Stopped after creating problem

✅ PASS: Agent did NOT run optimization (only created problem as requested)
✅ PASS: Agent created the problem
```

### Expected Behavior

**Interactive Workflow** (like Claude Code):
```
User: "list evaluators"
Agent: [lists evaluators] ✓ STOP

User: "create a rosenbrock problem"
Agent: [creates problem] ✓ STOP

User: "optimize with SLSQP"
Agent: [runs optimization] ✓ STOP

User: "analyze the results"
Agent: [analyzes results] ✓ STOP
```

**Autonomous Workflow** (ReAct agent, opt-in only):
```
User: "optimize rosenbrock in 2D"
Agent: [lists evaluators] → [creates problem] → [runs optimization] → [analyzes] → DONE
```

---

## Design Principles

### 1. Trust the LLM

The conversational agent relies on LLM intelligence to:
- ✅ Understand what the user is asking for
- ✅ Determine which tools are needed
- ✅ Execute only what was requested
- ✅ Stop after completing the specific task

### 2. Minimal Prompts

System prompts are kept minimal (see `docs/minimal_prompt_philosophy.md`):
- ✅ No workflow prescriptions: "first do X, then Y"
- ✅ No decision trees: "if user says A, do B"
- ✅ No usage hints: "use this tool when..."
- ✅ Just factual descriptions of tool capabilities

### 3. User Control

The user controls the workflow pace:
- ✅ Agent responds to specific requests
- ✅ Agent stops after completing request
- ✅ User decides next action
- ✅ Interactive, like Claude Code

### 4. Still Intelligent

The agent can still:
- ✅ Use multiple tools in a single response (ReAct cycles)
- ✅ Reason about which tools to use
- ✅ Handle complex multi-step requests
- ✅ Ask questions when unclear
- ✅ Provide helpful context in responses

---

## When to Use Each Agent

### Use Conversational Agent (Default)

**For**:
- Interactive exploration
- Incremental workflow building
- Learning and experimenting
- Fine-grained control
- Claude Code-like experience

**Examples**:
- "list evaluators"
- "create a problem"
- "show run 5"
- "compare runs 3 and 7"

### Use ReAct Agent (Opt-In)

**For**:
- Fully autonomous workflows
- Batch processing
- When you trust agent to execute entire process
- Scripted/automated usage

**Examples**:
- "optimize this wing design and analyze convergence"
- "run 5 different algorithms on this problem and compare results"

**Note**: ReAct agent still needs work to be truly useful for autonomous workflows. The conversational agent is recommended for all interactive use.

---

## Future Work

### Potential Enhancements

1. **Hybrid Mode**: Allow user to toggle between conversational and autonomous within session
   ```
   paola> /mode autonomous
   paola> optimize rosenbrock
   [agent runs full workflow]
   paola> /mode interactive
   ```

2. **Explicit Workflow Requests**: Detect multi-step intentions
   ```
   User: "create, optimize, and analyze a rosenbrock problem"
   Agent: [executes all three steps, then stops]
   ```

3. **Approval Prompts**: Ask before executing expensive operations
   ```
   Agent: "This will run 100 CFD evaluations (~$10,000). Proceed? [y/n]"
   ```

4. **Session Resumption**: Continue autonomous workflows across sessions
   ```
   paola> /resume workflow_123
   Agent: [continues from checkpoint]
   ```

---

## Alignment with Claude Code Philosophy

The conversational agent aligns with Claude Code's design:

| Aspect | Claude Code | PAOLA Conversational Agent |
|--------|-------------|----------------------------|
| **Interaction** | Request → Response → STOP | Request → Response → STOP ✓ |
| **User Control** | User controls pace | User controls pace ✓ |
| **Tool Usage** | Multiple tools per response | Multiple tools per response ✓ |
| **Autonomy** | Minimal (executes what's asked) | Minimal (executes what's asked) ✓ |
| **Prompts** | Minimal, trust LLM intelligence | Minimal, trust LLM intelligence ✓ |
| **Workflow** | Interactive, step-by-step | Interactive, step-by-step ✓ |

---

## Summary

**Problem**: ReAct agent executed entire workflows autonomously instead of responding to specific user requests

**Root Cause**: Autonomous loop architecture (`while not done: ...`)

**Solution**: Conversational agent with request-response pattern (`if not tool_calls: STOP`)

**Result**:
- ✅ Agent executes only what user asks
- ✅ Agent stops after completing specific request
- ✅ User controls workflow pace
- ✅ Still uses ReAct (Reasoning + Acting) within responses
- ✅ Behaves like Claude Code (interactive, helpful, controlled)

**Status**: IMPLEMENTED and TESTED ✓

**Files**:
- `paola/agent/conversational_agent.py` (new implementation)
- `paola/agent/__init__.py` (exports both agents)
- `paola/cli/repl.py` (integrated into CLI)
- `test_conversational_agent.py` (verification tests)

**Default**: Conversational agent is now the default for all interactive CLI usage.
