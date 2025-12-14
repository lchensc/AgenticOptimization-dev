# Prompt Fix Summary - Agent Using Wrong Tool

**Date**: 2025-12-14
**Issue**: Agent uses `create_benchmark_problem` when user asks for "registered evaluators"

---

## Problem

Agent behavior when user says "create from registered evaluators":
1. ❌ Calls `create_benchmark_problem` (wrong - these are NOT registered evaluators)
2. ❌ Then tries `create_nlp_problem` with non-existent evaluator_id

**Root Cause**: Unclear distinction in system prompt between:
- **Benchmark problems** (built-in analytical functions)
- **Foundry evaluators** (custom user functions registered in Foundry)

---

## What Was Changed

### 1. System Prompt Reordering (`paola/agent/prompts/optimization.py`)

**Before**:
```python
**Problem Formulation:**
- create_benchmark_problem: Built-in analytical functions (rosenbrock, sphere, etc.)
- create_nlp_problem: Create NLP problem from registered evaluators
  → Check available evaluators: foundry_list_evaluators

**Evaluator Management:**
- foundry_list_evaluators: See registered custom evaluators
- foundry_get_evaluator: Get evaluator details
- Note: Prefer registered evaluators over benchmarks when available
```

**After**:
```python
**Problem Formulation:**
- create_nlp_problem: Create NLP from registered Foundry evaluators (check foundry_list_evaluators first)
- create_benchmark_problem: Quick built-in test functions (rosenbrock, sphere) - use when no custom evaluator needed

**Evaluator Management:**
- foundry_list_evaluators: List registered custom evaluators in Foundry
- foundry_get_evaluator: Get evaluator details
- NOTE: When user says "registered evaluator", use foundry_list_evaluators + create_nlp_problem
```

**Changes**:
1. ✅ Put `create_nlp_problem` FIRST (prioritization signal)
2. ✅ Made it explicit: "registered Foundry evaluators"
3. ✅ Added workflow hint: "check foundry_list_evaluators first"
4. ✅ Downgraded benchmark to: "Quick built-in test functions" (fallback)
5. ✅ Added explicit NOTE: keyword "registered evaluator" → use Foundry workflow

### 2. Tool Description Updates

#### `create_nlp_problem` (`paola/tools/evaluator_tools.py`)

**Added**:
```python
"""
Create Nonlinear Programming (NLP) problem from registered Foundry evaluators.

IMPORTANT: This uses evaluators registered in Foundry (via foundry_store_evaluator).
Use foundry_list_evaluators() first to see available evaluators.
This is NOT for benchmark functions - use create_benchmark_problem for those.
...
"""
```

**Why**: Make it crystal clear this tool is ONLY for Foundry evaluators

#### `create_benchmark_problem` (`paola/tools/evaluator_tools.py`)

**Added**:
```python
"""
Create and register a built-in analytical benchmark problem (quick test functions).

Use this for quick testing with standard functions (rosenbrock, sphere, etc.).
For custom user functions registered in Foundry, use create_nlp_problem instead.
...
"""
```

**Why**: Clarify this is for built-in functions, redirect to `create_nlp_problem` for custom

---

## Expected Agent Behavior Now

### Scenario 1: User says "create from registered evaluators"

**Expected workflow**:
1. ✅ Agent calls `foundry_list_evaluators()`
2. ✅ Finds available evaluators (e.g., "rosenbrock_eval")
3. ✅ Calls `create_nlp_problem(objective_evaluator_id="rosenbrock_eval", ...)`
4. ✅ Success

### Scenario 2: User says "create rosenbrock problem"

**Ambiguous - agent should**:
1. ✅ Check `foundry_list_evaluators()` first
2. ✅ If "rosenbrock_eval" found → use `create_nlp_problem`
3. ✅ If NOT found → use `create_benchmark_problem` (fallback)

### Scenario 3: User says "test with rosenbrock"

**Quick test - agent should**:
1. ✅ Use `create_benchmark_problem` (built-in, quick)

---

## Key Design Decisions

### 1. Ordering Matters

Tools listed first in prompt → higher priority in agent's mind
- Put `create_nlp_problem` FIRST
- Benchmark second (fallback)

### 2. Explicit Keywords

Agent responds to explicit signals:
- "registered evaluator" → Foundry workflow
- "custom function" → Foundry workflow
- "test" / "quick" → Benchmark workflow

### 3. Tool Descriptions as Documentation

LangChain shows tool descriptions to LLM → use them for guidance
- Added workflow hints
- Added explicit "NOT for X" statements
- Added cross-references between tools

### 4. Minimal But Clear

- Kept additions short (~3-4 sentences per tool)
- No verbose workflows
- Trust LLM to figure out details
- But provide clear disambiguation

---

## Testing

### Test 1: Registered Evaluator

```bash
# User says:
"create a 2D rosenbrock problem from registered evaluators"

# Expected agent actions:
1. foundry_list_evaluators()
2. create_nlp_problem(objective_evaluator_id="rosenbrock_eval", bounds=[[-5,10],[-5,10]])
```

### Test 2: Ambiguous Request

```bash
# User says:
"optimize rosenbrock in 2D"

# Expected agent actions:
1. foundry_list_evaluators()
2. If found: create_nlp_problem(...)
3. If not: create_benchmark_problem(...)
```

### Test 3: Quick Test

```bash
# User says:
"test SLSQP on rosenbrock"

# Expected agent actions:
1. create_benchmark_problem(function_name="rosenbrock", dimension=2)
2. run_scipy_optimization(algorithm="SLSQP")
```

---

## If Agent Still Confused

### Additional Options to Try:

#### Option 1: Add Example to Prompt (More Verbose)

```python
**Example Workflows:**

When user says "registered evaluator":
1. foundry_list_evaluators()
2. create_nlp_problem(objective_evaluator_id="...", ...)

When user says "test rosenbrock":
1. create_benchmark_problem(function_name="rosenbrock", ...)
```

**Tradeoff**: More rigid, less trust in LLM

#### Option 2: Rename Tools for Clarity

```python
# Rename to emphasize distinction:
create_nlp_from_foundry_evaluators(...)  # Current: create_nlp_problem
create_builtin_benchmark_problem(...)    # Current: create_benchmark_problem
```

**Tradeoff**: Longer names, API change

#### Option 3: Merge Benchmarks into Foundry

Make benchmark functions also available as Foundry evaluators:
- Pre-register rosenbrock, sphere, etc. in Foundry at startup
- Remove `create_benchmark_problem` entirely
- Single unified path: always use `create_nlp_problem`

**Tradeoff**: More setup, less clear separation of concerns

---

## Recommendation

**Current approach (prompt + tool description updates) should work.**

If still issues:
1. Try adding explicit example (Option 1)
2. Consider pre-registering benchmarks in Foundry (Option 3)
3. Last resort: Rename tools (Option 2)

---

## Files Modified

1. `paola/agent/prompts/optimization.py` - System prompt reordering
2. `paola/tools/evaluator_tools.py` - Tool description updates

**Total changes**: ~15 lines
**Philosophy**: Minimal but clear disambiguation

---

**Status**: READY FOR TESTING ✅
