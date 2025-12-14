# Minimal Prompting Design: Remove Tools, Not Add Instructions

## Design Principle

From `CLAUDE.md`:
> **CRITICAL - Minimal Prompting**: Keep system prompts and tool schemas minimal. Trust the LLM's intelligence. Never add verbose guidance, formatting rules, or hand-holding without explicit permission. The agent must learn from experience, not from over-specified prompts.

## The Problem We Faced

Agent was using `create_benchmark_problem` instead of registered evaluators:

```
Registered: rosenbrock_eval ●
User: "optimize rosenbrock"
Agent: create_benchmark_problem(function_name="rosenbrock")  ← Wrong choice!
```

## Two Approaches to Fix This

### Approach 1: Add Verbose Instructions (❌ Rejected)

**Initial fix attempt:**
```python
**Instructions:**
3. When creating optimization problems:
   - ALWAYS check foundry_list_evaluators first to see what's registered
   - PREFER registered evaluators (create_nlp_problem) over benchmarks
   - Only use create_benchmark_problem if no suitable evaluator exists
```

**Problems:**
- Verbose (violates minimal prompting principle)
- Requires stronger LLM (works with qwen-plus, not qwen-flash)
- Adds complexity to decision-making
- Doesn't scale (what if we add more tools?)

### Approach 2: Remove the Tool (✅ Adopted)

**Better fix:**
```python
# Before: Two tools, complex guidance needed
self.tools = [
    create_benchmark_problem,  # ← Remove this
    create_nlp_problem,        # ← Keep this
]

# After: One tool, no guidance needed
self.tools = [
    create_nlp_problem,  # Only option!
]
```

**Benefits:**
- Minimal prompting (just one line about Foundry)
- Works with all LLMs (no complex decision-making)
- Cleaner tool set
- Forces agentic workflow (register → use)

## The Minimal Guidance We Kept

**System prompt:** (3 simple instructions)
```python
**Instructions:**
1. Explain your reasoning before calling tools
2. Tool arguments must be valid JSON (e.g., no Python expressions)
3. Evaluators must be registered in Foundry before use (check foundry_list_evaluators)
```

**Why instruction #3?**
- Foundry is a **new concept** specific to PAOLA
- LLMs wouldn't naturally know about this registry pattern
- Single line is sufficient to establish the mental model

**What we DON'T say:**
- ❌ Step-by-step workflow instructions
- ❌ "ALWAYS do X before Y" commands
- ❌ "PREFER this over that" guidance
- ❌ Verbose examples

## Why This Works Better

### Cognitive Load Comparison

**Approach 1 (Verbose Instructions):**
```
User: "optimize rosenbrock"
       ↓
Agent thinks:
  - Should I check Foundry first? (instruction says "ALWAYS")
  - Is there a registered evaluator? (need to call tool)
  - Should I prefer registered over benchmark? (instruction says "PREFER")
  - Or is benchmark appropriate here? (instruction says "if no suitable")
  - What counts as "suitable"?
       ↓
Decision paralysis or wrong choice with weaker LLMs
```

**Approach 2 (Remove Tool):**
```
User: "optimize rosenbrock"
       ↓
Agent thinks:
  - Need to create problem
  - Only have create_nlp_problem
  - Requires objective_evaluator_id
  - Check what's registered
       ↓
Natural workflow, clear path
```

### Forcing the Correct Pattern

**Old workflow (two paths):**
```
Register evaluator (optional)
       ↓
Optimize:
  Path A: Use benchmark → Done
  Path B: Use registered → Done
```
Agent could skip registration entirely!

**New workflow (one path):**
```
Register evaluator (required)
       ↓
Optimize:
  Use registered → Done
```
Agent must register first!

## When Benchmarks Are Still Useful

If we need benchmarks again in the future:

**Option A: Register them like any other evaluator**
```python
# System or user pre-registers common benchmarks
foundry.register_evaluator(rosenbrock_config)
foundry.register_evaluator(sphere_config)

# Agent uses them the same way
create_nlp_problem(objective_evaluator_id="rosenbrock_builtin")
```

**Option B: Make registration automatic**
```python
# When agent asks for unknown evaluator
create_nlp_problem(objective_evaluator_id="rosenbrock")
# → Foundry auto-registers if it's a known benchmark
```

Both options maintain the single-path workflow!

## Comparison with Traditional Software

### Traditional Approach: Hard-Coded Logic

```python
def create_problem(function_name):
    # Deterministic decision tree
    if function_name in BENCHMARKS:
        return create_benchmark(function_name)
    elif foundry.has_evaluator(function_name):
        return create_from_foundry(function_name)
    else:
        raise ValueError("Unknown function")
```

**Problem:** Every new scenario requires code changes

### Agentic Approach: Minimal Tool Set

```python
# Just provide the right tools
tools = [
    foundry_list_evaluators,  # Discovery
    create_nlp_problem,       # Creation
    # (no create_benchmark_problem)
]

# Agent figures out the workflow
```

**Benefit:** Workflow emerges from tool composition, not hard-coded logic

## Design Lessons

### 1. Tool Set Design > Instruction Complexity

**Worse:**
```python
tools = [tool_A, tool_B, tool_C, tool_D]
instructions = "Use A for X, B for Y, prefer C over D unless..."
```

**Better:**
```python
tools = [tool_A, tool_B]  # Removed C and D
instructions = "Use A for X, B for Y"
```

**Best:**
```python
tools = [tool_A]  # Only one option for this purpose
instructions = ""  # Self-evident
```

### 2. New Concepts Need Minimal Explanation

**Foundry is new → One line explanation:**
```
"Evaluators must be registered in Foundry before use"
```

**scipy.optimize is standard → No explanation:**
```
"Run scipy optimization"  # LLM already knows scipy
```

### 3. Remove Before Adding

When agent makes wrong choices:

**First try:** Remove wrong option
**Only if needed:** Add guidance to choose correctly

Don't default to adding instructions!

## Verification

### Before Fix
```bash
paola> optimize rosenbrock function
🔧 create_benchmark_problem(function_name="rosenbrock")  ← Wrong!
```

### After Fix
```bash
paola> optimize rosenbrock function
🔧 foundry_list_evaluators()  ← Checks registry
🔧 create_nlp_problem(objective_evaluator_id="rosenbrock_eval")  ← Correct!
```

No verbose instructions needed—agent follows natural path.

## Summary

| Aspect | Verbose Instructions | Remove Tool |
|--------|---------------------|-------------|
| **Prompt complexity** | High | Minimal |
| **LLM requirements** | Stronger model needed | Works with all |
| **Scalability** | Degrades with more tools | Maintains clarity |
| **Agent autonomy** | Constrained by rules | Natural emergence |
| **Maintenance** | Update docs when tools change | Self-documenting |

**Core principle:** When agent has two paths and chooses wrong one, don't add instructions to choose right one—remove the wrong path.

This aligns perfectly with PAOLA's philosophy:
- Trust LLM intelligence
- Minimal prompting
- Agentic autonomy
- Compositional workflows

The best instruction is no instruction—just the right tool set.
