# Minimal Prompt Philosophy - PAOLA Agent

**Date**: 2025-12-14
**Issue**: Agent was too prescriptive, executing rigid workflows instead of being interactive
**Solution**: Minimal prompts that trust LLM intelligence

---

## Problem

Agent behavior was:
1. ❌ Too autonomous - executing entire workflows without asking user
2. ❌ Too rigid - following prescribed patterns
3. ❌ Not interactive - not clarifying or asking questions
4. ❌ Not following user's specific instructions

**Example**:
```
User: "create a 2D rosenbrock NLP"
Agent:
  - Creates benchmark problem
  - Creates NLP problem
  - Starts optimization run
  - Runs optimization
  - Finalizes run
  - Says "DONE"

User never asked for all these steps!
```

---

## Root Cause

### Prescriptive Prompts

**Before** (`paola/agent/prompts/optimization.py`):
```python
"""
You are an autonomous optimization agent.

**Instructions:**
1. Explain your reasoning before calling tools
2. Tool arguments must be valid JSON

Decide next action. Use tools or respond "DONE".
"""
```

**Problems**:
- "autonomous optimization agent" → suggests it should work independently
- "Instructions" → prescribes what to do
- "Decide next action" → prescribes decision-making
- "respond DONE" → prescribes when to stop

### Prescriptive Tool Hints

**Before** (system prompt):
```python
**Problem Formulation:**
- create_nlp_problem: Create NLP from registered Foundry evaluators (check foundry_list_evaluators first)
- create_benchmark_problem: Quick built-in test functions - use when no custom evaluator needed

**NOTE: When user says "registered evaluator", use foundry_list_evaluators + create_nlp_problem**
```

**Problems**:
- "(check foundry_list_evaluators first)" → prescribes workflow
- "use when no custom evaluator needed" → prescribes when to use
- "NOTE: When user says X, use Y" → explicit decision tree

### Prescriptive Tool Descriptions

**Before** (`create_nlp_problem`):
```python
"""
IMPORTANT: This uses evaluators registered in Foundry (via foundry_store_evaluator).
Use foundry_list_evaluators() first to see available evaluators.
This is NOT for benchmark functions - use create_benchmark_problem for those.
"""
```

**Problems**:
- "Use X first" → prescribes workflow
- "This is NOT for Y" → prescribes negative rules

---

## Solution: Minimal Prompts

### Principle from CLAUDE.md

> **CRITICAL - Minimal Prompting**: Keep system prompts and tool schemas minimal. Trust the LLM's intelligence. Never add verbose guidance, formatting rules, or hand-holding without explicit permission. The agent must learn from experience, not from over-specified prompts.

### Applied Changes

#### 1. Main Agent Prompt

**After**:
```python
"""
You are PAOLA, an optimization assistant.

User request: {user_request}

Current state:
- Problem: {problem_status}
- Optimizer: {optimizer_status}
- Iteration: {iteration}
- Best objective: {best_obj}

Tools available:
{tool_list}
"""
```

**Changes**:
- ✅ "optimization assistant" (not "autonomous agent")
- ✅ "User request" (not "Goal")
- ✅ Minimal state (removed budget, cache, history, observations)
- ✅ Just facts, no instructions
- ✅ No "Decide", no "DONE", no workflows

#### 2. Tool List in System Prompt

**After**:
```python
**Problem Formulation:**
- create_benchmark_problem: Built-in analytical functions
- create_nlp_problem: NLP from registered evaluators

**Run Management:**
- start_optimization_run: Start new optimization run
...
```

**Changes**:
- ✅ Removed workflow hints: "(check X first)"
- ✅ Removed usage guidance: "use when..."
- ✅ Removed NOTE with decision tree
- ✅ Just tool name + brief description

#### 3. Tool Descriptions

**After** (`create_nlp_problem`):
```python
"""
Create Nonlinear Programming (NLP) problem from registered Foundry evaluators.

NLP standard form:
    minimize/maximize f(x)
    subject to:
      g_i(x) ≤ value
      h_j(x) = value
      x_lower ≤ x ≤ x_upper

Args:
    problem_id: Unique identifier
    objective_evaluator_id: Evaluator for objective function
    bounds: Design variable bounds
    ...

Returns:
    {success, problem_id, recommended_solvers, ...}
"""
```

**Changes**:
- ✅ Removed "IMPORTANT: Use X first"
- ✅ Removed "This is NOT for Y"
- ✅ Just describes WHAT it does, not WHEN to use it

**After** (`create_benchmark_problem`):
```python
"""
Create and register a built-in analytical benchmark problem.

Available benchmark functions:
- rosenbrock: Global minimum at (1,1,...,1)
- sphere: Global minimum at origin
...
"""
```

**Changes**:
- ✅ Removed "Use this for quick testing"
- ✅ Removed "For custom functions, use create_nlp_problem instead"
- ✅ Just describes WHAT it does

---

## What This Achieves

### 1. Trust the LLM

The LLM (qwen-plus, claude, gpt-4) is intelligent enough to:
- ✅ Understand what tools do from their descriptions
- ✅ Reason about which tool to use for a given task
- ✅ Ask user for clarification when unclear
- ✅ Follow user's specific instructions

### 2. Flexibility

Agent can now:
- ✅ Execute just what user asks (not entire workflows)
- ✅ Ask questions when unclear
- ✅ Choose different approaches based on context
- ✅ Learn from experience (not rigid patterns)

### 3. Interactive Behavior

Agent should now:
- ✅ Read user request carefully
- ✅ Execute what user asked for
- ✅ Stop and report back
- ✅ Wait for next instruction
- ✅ Ask if unsure

---

## Expected Behavior Changes

### Before (Rigid Workflow):

```
User: "create a 2D rosenbrock NLP"

Agent:
  💭 I need to create NLP with rosenbrock...
  🔧 create_benchmark_problem(...)
  🔧 create_nlp_problem(...)
  🔧 start_optimization_run(...)
  🔧 run_scipy_optimization(...)
  🔧 finalize_optimization_run(...)
  ✅ DONE - Optimized successfully!
```

### After (Interactive):

```
User: "create a 2D rosenbrock NLP"

Agent:
  💭 User wants to create an NLP problem...
  🔧 foundry_list_evaluators()

  I found rosenbrock_eval in Foundry. I'll create an NLP problem with:
  - Objective: rosenbrock_eval
  - Dimension: 2D
  - Bounds: [-5, 10] for each dimension (typical for Rosenbrock)

  🔧 create_nlp_problem(
      problem_id="rosenbrock_2d_nlp",
      objective_evaluator_id="rosenbrock_eval",
      bounds=[[-5, 10], [-5, 10]]
  )

  ✅ Created NLP problem "rosenbrock_2d_nlp"
  Recommended solvers: SLSQP, L-BFGS-B

  [Waits for user's next instruction]
```

**Key differences**:
- ✅ Only does what user asked
- ✅ Explains what it's doing
- ✅ Stops after completing the request
- ✅ Doesn't assume user wants to optimize immediately

---

## Testing

### Test 1: Minimal Request

```
User: "list registered evaluators"

Expected:
Agent:
  🔧 foundry_list_evaluators()

  Available evaluators:
  - rosenbrock_eval
  - sphere_eval

  [Done - waits for next instruction]
```

### Test 2: Ambiguous Request

```
User: "create rosenbrock problem"

Expected:
Agent:
  💭 "create rosenbrock problem" could mean:
     1. Built-in benchmark (create_benchmark_problem)
     2. From registered evaluator (create_nlp_problem)

  Let me check if rosenbrock is registered...

  🔧 foundry_list_evaluators()

  Found rosenbrock_eval in Foundry. I'll use that.

  🔧 create_nlp_problem(...)

  [Or if not found: create_benchmark_problem(...)]
```

### Test 3: Multi-Step Request

```
User: "create and optimize rosenbrock in 2D with SLSQP"

Expected:
Agent:
  💭 User wants me to create AND optimize...

  🔧 foundry_list_evaluators()
  🔧 create_nlp_problem(...)
  🔧 run_scipy_optimization(algorithm="SLSQP", ...)

  Optimization completed:
  - Final design: [0.999, 0.999]
  - Final objective: 1.8e-8

  [Done - reports back]
```

---

## Comparison: Before vs After

| Aspect | Before (Prescriptive) | After (Minimal) |
|--------|----------------------|-----------------|
| **Agent identity** | "autonomous optimization agent" | "optimization assistant" |
| **Prompt length** | ~30 lines | ~10 lines |
| **Instructions** | Explicit (1, 2, 3...) | None |
| **Tool hints** | "(check X first)", "use when..." | Tool name + brief description |
| **Tool descriptions** | "IMPORTANT: Use X first", "NOT for Y" | Just describes capabilities |
| **Workflows** | Prescribed patterns | LLM decides |
| **Behavior** | Executes full workflows | Executes user request |
| **Flexibility** | Rigid | Flexible |

---

## Files Modified

1. **`paola/agent/prompts/optimization.py`**
   - Main agent prompt: 30 lines → 10 lines
   - Tool list: Removed workflow hints and NOTEs
   - Removed budget, cache, history, observations from state

2. **`paola/tools/evaluator_tools.py`**
   - `create_nlp_problem`: Removed prescriptive guidance
   - `create_benchmark_problem`: Removed usage hints

**Total reduction**: ~50 lines of prescriptive text removed

---

## Principles Going Forward

### DO:
- ✅ Describe WHAT tools do (capabilities, requirements, outputs)
- ✅ Trust LLM intelligence
- ✅ Keep prompts minimal
- ✅ Use factual language
- ✅ Show tool schemas clearly

### DON'T:
- ❌ Prescribe WHEN to use tools ("use X when...")
- ❌ Prescribe HOW to use tools ("first do X, then Y")
- ❌ Add workflow hints ("check X first")
- ❌ Add decision trees ("if user says X, do Y")
- ❌ Over-specify behavior

### Quote from CLAUDE.md:
> "The agent must learn from experience, not from over-specified prompts."

---

**Status**: IMPLEMENTED ✅

The agent now has minimal prompts that trust LLM intelligence. It should behave more like Claude Code - interactive, flexible, following specific user instructions rather than executing rigid workflows.
