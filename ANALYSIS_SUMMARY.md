# Analysis Summary: Why Agent Doesn't Use Registered Evaluators

## TL;DR

**Problem**: Agent falls back to `create_benchmark_problem` instead of using registered evaluators.

**Root Cause**: Missing infrastructure + no prompt guidance.

**Solution**:
1. Add `create_problem_from_evaluator` tool (CRITICAL - blocks everything)
2. Update prompt minimally (~150 chars)
3. Test workflow

---

## The Architectural Gap

### What We Built (Days 1-4)

```
Foundry (Persistent Storage)
├── rosenbrock_eval.json  ← Evaluator configs stored here
├── sphere_eval.json
└── ...
```

### What Agent Uses (Runtime)

```python
# Problem Registry (in-memory dict)
_PROBLEM_REGISTRY = {
    "rosenbrock_2d": <Rosenbrock object>,  # From create_benchmark_problem
    "sphere_10d": <Sphere object>
}
```

### The Missing Bridge

**NO connection between Foundry and Problem Registry!**

Even if agent wanted to use registered evaluators, there's NO TOOL to:
1. Load evaluator from Foundry
2. Create FoundryEvaluator instance
3. Register it for optimization use

---

## What the Prompt Says Now

```
**Problem Formulation:**
- create_benchmark_problem: Create benchmark optimization problem
  ↑
  Only mentions built-in benchmarks!
```

**Agent never learns**:
- Registered evaluators exist
- How to list them
- How to use them
- When to prefer them

---

## Your Concern: Infrastructure Sufficiency

> "Evaluator is only for the function and gradient calls, and we need, certainly, other optimization related objects."

**You're absolutely right!** For complete optimization, we need:

| Component | What It Is | Status |
|-----------|-----------|--------|
| **Evaluator** | f(x) and ∇f(x) | ✅ Fully implemented |
| **Design Space** | Bounds, initial point | ❌ Must specify at runtime |
| **Constraints** | g(x) ≤ 0, h(x) = 0 | ❌ Not integrated yet |
| **Problem Metadata** | Dimensions, type | ❌ Created separately |

**Current Architecture**:
- Evaluator = "How to compute f(x)"
- Problem = "Design space + constraints + metadata"
- Linked by evaluator_id

**This is actually CORRECT design!**

Why?
- Evaluator: Reusable (same function, different problems)
- Problem: Specific (rosenbrock in 2D vs 10D = different problems)
- Example: Same rosenbrock evaluator used for:
  - 2D problem, bounds [-5, 10]
  - 10D problem, bounds [-2, 2]
  - 5D problem with constraints

---

## Proposed Solution

### 1. Add Critical Tool (MUST HAVE)

```python
@tool
def create_problem_from_evaluator(
    evaluator_id: str,
    problem_id: str,
    bounds: List[List[float]],
    initial_design: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Create optimization problem from registered Foundry evaluator.

    This bridges Foundry storage → Runtime problem registry.
    """
```

**What it does**:
1. Retrieve evaluator config from Foundry
2. Create FoundryEvaluator instance
3. Register in problem registry (for optimization)
4. Create Problem metadata
5. Link evaluator to problem

**Without this tool, nothing else matters!**

### 2. Update Prompt (MINIMAL)

Add to tool list:
```python
**Problem Formulation:**
- create_benchmark_problem: Built-in analytical functions
- create_problem_from_evaluator: Use registered custom evaluators
  → Check available: foundry_list_evaluators

**Evaluator Management:**
- foundry_list_evaluators: See registered evaluators
- Prefer registered evaluators over benchmarks when available
```

**Total addition**: ~150 characters (stays minimal)

### 3. Workflow Example

```
User: "Optimize rosenbrock_eval in 2D with SLSQP"

Agent:
  1. foundry_list_evaluators()
     → Finds: rosenbrock_eval

  2. create_problem_from_evaluator(
         evaluator_id="rosenbrock_eval",
         problem_id="user_rosenbrock",
         bounds=[[-5, 10], [-5, 10]]
     )
     → Creates problem in registry

  3. run_scipy_optimization(
         problem_id="user_rosenbrock",
         algorithm="SLSQP"
     )
     → Runs optimization using registered evaluator!
```

---

## Questions for Discussion

### 1. Tool Implementation Priority

**Q**: Implement `create_problem_from_evaluator` first (critical path)?

**A**: YES - Without this tool, agent physically cannot use registered evaluators.

### 2. Bounds Storage

**Q**: Should evaluators store recommended bounds in their config?

**Current**: No, bounds specified at problem creation time

**Alternative**: Add to evaluator config:
```json
{
  "evaluator_id": "rosenbrock_eval",
  "recommended_setup": {
    "bounds": [[-5, 10], [-5, 10]],
    "default_dimension": 2
  }
}
```

**Recommendation**: Keep current approach (flexibility), add recommended_setup later (convenience).

### 3. Prompt Philosophy

**Q**: Minimal prompt vs explicit guidance?

**Options**:
- **Minimal** (current): ~150 chars, trust LLM
- **Instructive**: ~500 chars, explicit workflow
- **Contextual**: ~250 chars, hints without prescription

**Recommendation**: Contextual (balanced approach)

### 4. Default Behavior

**Q**: User says "optimize rosenbrock" - ambiguous! Built-in or registered?

**Options**:
- A: Always check registered first, fall back to benchmark
- B: Ask user to clarify
- C: Heuristic (e.g., "_eval" suffix means registered)

**Recommendation**: Option A (check registered first)

---

## Next Steps

### Immediate (Critical Path)

1. **Implement `create_problem_from_evaluator` tool**
   - File: `paola/tools/problem_tools.py`
   - ~80 lines of code
   - Add to agent's tool list in `repl.py`

2. **Update prompt**
   - File: `paola/agent/prompts/optimization.py`
   - Modify `_get_default_tool_list()`
   - Add ~150 characters

3. **Test**
   - User: "optimize rosenbrock_eval in 2D"
   - Verify agent uses registered evaluator

### Later Enhancements

4. **Add recommended_setup to evaluator config** (optional)
   - Makes problem creation easier
   - Agent can use defaults if not specified

5. **Integrate constraints** (future work)
   - Link constraint definitions to evaluators
   - Support g(x) ≤ 0, h(x) = 0

6. **Smart disambiguation** (nice-to-have)
   - Auto-suggest registered evaluators
   - "Found rosenbrock_eval, use that?"

---

## Files to Review

1. **`docs/analysis/EVALUATOR_USAGE_GAP_ANALYSIS.md`**
   - Comprehensive technical analysis
   - Architecture diagrams
   - Workflow comparisons
   - 50+ pages

2. **`docs/analysis/PROMPT_UPDATE_STRATEGY.md`**
   - Prompt philosophy discussion
   - Update strategies
   - Testing approach
   - 30+ pages

3. **`ANALYSIS_SUMMARY.md`** (this file)
   - Executive summary
   - Key decisions
   - Next steps

---

## Recommendation

**Proceed with**:
1. ✅ Implement `create_problem_from_evaluator` tool (1 hour)
2. ✅ Minimal prompt update (15 minutes)
3. ✅ Test workflow (30 minutes)

**Total time**: ~2 hours

**This unblocks registered evaluator usage immediately!**
