# Analysis: Why Agent Chose Benchmark Over Registered Evaluator

## The Problem

**Observation:**
Despite having `rosenbrock_eval` registered in Foundry, when the user asked to "optimize the rosenbrock function in 2D using SLSQP", the agent chose `create_benchmark_problem` instead of using the registered evaluator with `create_nlp_problem`.

```
Registered Evaluators:
- rosenbrock_eval (python_function) ●
- ackley_eval (python_function) ●
- rastrigin_eval (python_function) ●
- sphere_eval (python_function) ●

User: "optimize the rosenbrock function in 2D using SLSQP"

Agent Decision:
🔧 create_benchmark_problem...  ← Used benchmark instead!
```

## Root Cause Analysis

### 1. Tool Availability (✓ Not the Issue)

The agent HAS all necessary tools:
- `foundry_list_evaluators` - to check registered evaluators
- `create_nlp_problem` - to use registered evaluators
- `create_benchmark_problem` - to create built-in benchmarks

**Conclusion:** Tools exist, not a tool availability problem.

### 2. Discoverability Problem (✓ ROOT CAUSE)

The agent's decision process when seeing "optimize rosenbrock":

```
User request: "optimize the rosenbrock function in 2D using SLSQP"
               ↓
Agent sees two options:
  Option A: create_benchmark_problem(function_name="rosenbrock")
            ↑ Direct match! Easy!

  Option B: foundry_list_evaluators() → find "rosenbrock_eval"
            → create_nlp_problem(objective_evaluator_id="rosenbrock_eval")
            ↑ Two steps, requires lookup

Agent chooses: Option A (simpler, direct)
```

**Why Option A won:**
1. **No prompt guidance** to check Foundry first
2. **Tool descriptions don't clarify** when to use which
3. **Natural path of least resistance** - benchmark matches "rosenbrock" directly

### 3. Specific Issues Found

#### Issue 1: System Prompt Lacks Guidance

**Before:**
```python
**Instructions:**
1. Explain your reasoning before calling tools
2. Tool arguments must be valid JSON
3. When registering evaluators:
   - Create standalone files...
```

**Problem:** No instruction about USING registered evaluators!

#### Issue 2: Tool Descriptions Don't Distinguish

**create_benchmark_problem:**
```
"Create and register a built-in analytical benchmark problem.

Available benchmark functions:
- rosenbrock: Classic Rosenbrock function..."
```

**Problem:** Doesn't say "only use if no registered evaluator exists"

**create_nlp_problem:**
```
"Create Nonlinear Programming (NLP) problem from registered Foundry evaluators."
```

**Problem:** Doesn't say "PREFERRED METHOD" or "check foundry first"

#### Issue 3: No Proactive Checking Pattern

The agent doesn't have a pattern of:
1. User asks for function X
2. Check if X is registered
3. Use registered OR fall back to benchmark

This pattern isn't encoded anywhere.

## The Fixes Applied

### Fix 1: Add Explicit Prompt Guidance

**Location:** `paola/agent/prompts.py:58-61`

```python
**Instructions:**
3. When creating optimization problems:
   - ALWAYS check foundry_list_evaluators first to see what's registered
   - PREFER registered evaluators (create_nlp_problem) over benchmarks
   - Only use create_benchmark_problem if no suitable evaluator exists
```

**Impact:** Agent now knows the decision order.

### Fix 2: Update Tool Descriptions

**Location:** `paola/tools/evaluator_tools.py:307-309`

```python
@tool
def create_benchmark_problem(...):
    """
    Create and register a built-in analytical benchmark problem.

    IMPORTANT: Only use this if no suitable evaluator is registered in Foundry.
    Always check foundry_list_evaluators first and prefer create_nlp_problem
    with registered evaluators.
    ...
```

**Location:** `paola/tools/evaluator_tools.py:415-416`

```python
@tool
def create_nlp_problem(...):
    """
    Create Nonlinear Programming (NLP) problem from registered Foundry evaluators.

    PREFERRED METHOD: Use this to create problems with registered evaluators.
    Check foundry_list_evaluators to see available evaluators.
    ...
```

**Impact:** Tool descriptions now guide decision-making.

## Expected Behavior After Fix

### New Decision Flow

```
User: "optimize the rosenbrock function in 2D using SLSQP"
       ↓
Agent reasoning:
  "I need to optimize rosenbrock. Let me check registered evaluators first."
       ↓
🔧 foundry_list_evaluators()
       ↓
Result: ["rosenbrock_eval", "ackley_eval", "rastrigin_eval", "sphere_eval"]
       ↓
Agent reasoning:
  "Found rosenbrock_eval! I should use create_nlp_problem with this evaluator,
   not create_benchmark_problem."
       ↓
🔧 create_nlp_problem(objective_evaluator_id="rosenbrock_eval", ...)
       ↓
Success!
```

### Test Case

**Input:**
```
User: "optimize the rosenbrock function in 2D using SLSQP"

Registered evaluators:
- rosenbrock_eval ●
```

**Expected Output (After Fix):**
```
Agent reasoning: Let me check registered evaluators first...

🔧 foundry_list_evaluators...
✓ Found: rosenbrock_eval, ackley_eval, rastrigin_eval, sphere_eval

Agent reasoning: I found rosenbrock_eval registered. I'll use create_nlp_problem
instead of create_benchmark_problem.

🔧 create_nlp_problem(objective_evaluator_id="rosenbrock_eval", bounds=[[-5, 10], [-5, 10]])
✓ create_nlp_problem completed

🔧 start_optimization_run...
🔧 run_scipy_optimization...
```

**Why this is correct:**
1. Agent checks Foundry first (instruction 3.a)
2. Finds registered evaluator (rosenbrock_eval)
3. Prefers create_nlp_problem over benchmark (instruction 3.b)

## Verification Steps

### Manual Test

1. Register an evaluator:
   ```bash
   paola> /register_eval my_function.py
   ```

2. Try optimizing it:
   ```bash
   paola> optimize my_function using SLSQP
   ```

3. Check agent's reasoning:
   - Should call `foundry_list_evaluators` first
   - Should recognize registered evaluator
   - Should use `create_nlp_problem` not `create_benchmark_problem`

### Automated Test

Create `tests/test_agent_prefers_registered.py`:

```python
"""Test that agent prefers registered evaluators over benchmarks."""

# 1. Register rosenbrock_eval
# 2. Ask agent to "optimize rosenbrock"
# 3. Verify agent calls:
#    - foundry_list_evaluators (to check)
#    - create_nlp_problem (to use registered)
#    NOT create_benchmark_problem
```

## Design Insight: Agent Autonomy vs. Guidance

### The Tension

**Too much hardcoding:**
```python
# Bad: Force agent to ALWAYS check foundry
if user_mentions_function_name:
    force_check_foundry()  # ← Removes agent autonomy
```

**Too little guidance:**
```python
# Bad: No guidance at all
# Agent has to figure out workflow from examples
# → Often chooses wrong path
```

### The Balance (Current Solution)

**Explicit instructions + Tool descriptions:**
```python
# System prompt: "ALWAYS check foundry_list_evaluators first"
# Tool docstring: "IMPORTANT: Only use if no suitable evaluator exists"
# → Agent knows the pattern but decides how to apply it
```

**Why this works:**
- Agent understands the **workflow**: check → decide → use
- Agent retains **flexibility**: can still use benchmarks when appropriate
- Agent learns **preference**: registered > benchmark

## Related Concepts

### Problem Discoverability vs. Solution Hardcoding

**Discoverability:** Agent can find registered evaluators but needs to know to look

**Solution:**
- Make looking a **default behavior** via instructions
- Make tools **self-documenting** via docstrings
- Don't hardcode **decision logic** in code

### Agentic Learning Pattern

This fix demonstrates a key pattern:

```
Human notices suboptimal behavior
  ↓
Analyze: Why did agent choose X over Y?
  ↓
Root cause: Missing guidance, not missing capability
  ↓
Fix: Add guidance to prompt + tool docs
  ↓
Agent learns the pattern without losing autonomy
```

**Not:**
```
Human notices problem
  ↓
Add deterministic if/else logic
  ↓
Agent loses autonomy, becomes scripted
```

## Summary

| Aspect | Before Fix | After Fix |
|--------|-----------|-----------|
| **Prompt guidance** | No mention of checking Foundry | Explicit: "ALWAYS check foundry_list_evaluators first" |
| **Tool clarity** | Both options equal | create_nlp_problem marked "PREFERRED" |
| **Agent workflow** | Direct match (rosenbrock → benchmark) | Check → Find → Use registered |
| **Decision basis** | Convenience (easier path) | Instruction + preference |

**Core lesson:** Agent autonomy requires **clear guidance** on **when to use which tools**, not just **availability of tools**.

The fix maintains agent autonomy while establishing the correct workflow pattern through instructions and tool documentation.
