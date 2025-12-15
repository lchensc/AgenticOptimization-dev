# Prompt Update Strategy Discussion

**Context**: Agent doesn't use registered evaluators because the prompt doesn't tell it to.

## Current Prompt Analysis

### What the Prompt Says Now

From `paola/agent/prompts/optimization.py`:

```python
**Problem Formulation:**
- create_benchmark_problem: Create benchmark optimization problem

**Run Management:**
- start_optimization_run: Start new optimization run
...

**Analysis (Deterministic - Fast & Free):**
- analyze_convergence: Check convergence rate, stalling, improvement
...
```

**Issues**:
1. Only mentions `create_benchmark_problem`
2. No mention of Foundry or registered evaluators
3. No guidance on when to use what
4. Registration tools are listed but without context

### What Tools Are Actually Available

Looking at `paola/cli/repl.py` line 83-114:

```python
self.tools = [
    # Problem formulation
    create_benchmark_problem,

    # Run management
    start_optimization_run,
    finalize_optimization_run,
    get_active_runs,

    # Optimization
    run_scipy_optimization,

    # Analysis...
    analyze_convergence_new,
    analyze_efficiency,
    get_all_metrics,
    analyze_run_with_ai,

    # Knowledge...
    store_optimization_insight,
    retrieve_optimization_knowledge,
    list_all_knowledge,

    # Evaluator registration  ← Added in Day 4
    read_file,
    execute_python,
    foundry_store_evaluator,
    foundry_list_evaluators,
    foundry_get_evaluator,
]
```

**The tools ARE there**, but the agent doesn't know:
- What they're for
- When to use them
- How they fit into the workflow

## Prompt Update Philosophies

### Philosophy 1: Minimalistic (Current Preference)

**Principle**: Trust LLM intelligence, minimal guidance

**For Registration Prompt** (Day 3):
- < 1000 characters
- Shows target schema
- Lists tools briefly
- One example
- No verbose guidance

**Pros**:
- Clean, concise
- LLM figures out details
- Less maintenance

**Cons**:
- LLM may not discover features
- No workflow guidance
- Trial-and-error learning

### Philosophy 2: Instructive

**Principle**: Clear workflows, explicit guidance

**Example**:
```
**Problem Formulation Workflow:**

1. Check for registered evaluators:
   - foundry_list_evaluators: See what's available

2. Create problem:
   a. If user specifies registered evaluator (e.g., "rosenbrock_eval"):
      → Use create_problem_from_evaluator
   b. If user mentions standard function (e.g., "rosenbrock"):
      → Check registered evaluators first
      → Fall back to create_benchmark_problem if not found
   c. If user describes custom function:
      → Guide them through registration first

3. Run optimization:
   → start_optimization_run
   → run_scipy_optimization
   → finalize_optimization_run
```

**Pros**:
- Clear decision logic
- Faster discovery
- Fewer mistakes

**Cons**:
- Longer prompt
- More rigid
- May override LLM reasoning

### Philosophy 3: Contextual Hints

**Principle**: Provide context without being prescriptive

**Example**:
```
**Problem Formulation:**

Create optimization problems using:
- create_benchmark_problem: Built-in analytical functions (rosenbrock, sphere, etc.)
- create_problem_from_evaluator: Use registered evaluators from Foundry

Registered evaluators offer:
- Custom user functions
- PAOLA capabilities (caching, observation gates)
- Performance tracking

Check available evaluators: foundry_list_evaluators
```

**Pros**:
- Balanced
- Provides context
- LLM still makes decisions

**Cons**:
- Still longer than minimal
- May be ambiguous

## Specific Challenges

### Challenge 1: Tool Discovery

**Problem**: Agent has 15+ tools, needs to know which to use when

**Current**: Tools listed with brief descriptions (dynamic from tool.description)

**Options**:
1. **Categorize clearly** (Problem Formulation, Analysis, etc.)
2. **Priority hints** ("Prefer X over Y when...")
3. **Context-sensitive** (Show different tools based on state)

### Challenge 2: Workflow Ambiguity

**Scenario**: User says "optimize rosenbrock in 2D"

**Ambiguity**:
- Built-in benchmark function rosenbrock?
- Registered evaluator named rosenbrock?
- Custom function to register?

**Current behavior**: Defaults to benchmark (because that's what prompt mentions)

**Options**:
1. **Explicit disambiguation** in prompt
2. **Agent asks user** for clarification
3. **Smart search**: Check registered first, fall back to benchmark

### Challenge 3: Missing Bridge Tool

**Problem**: Even with perfect prompt, there's NO tool to use registered evaluators!

**Current tools**:
- `foundry_list_evaluators` - Lists registered ✅
- `foundry_get_evaluator` - Gets config ✅
- ??? - Creates problem from evaluator ❌

**Must add**:
```python
@tool
def create_problem_from_evaluator(...):
    """Load registered evaluator and create optimization problem."""
```

**Without this tool, prompt updates alone won't work!**

## Recommended Approach

### Step 1: Add Missing Infrastructure (CRITICAL)

**Before any prompt changes**, implement:

```python
@tool
def create_problem_from_evaluator(
    evaluator_id: str,
    problem_id: str,
    bounds: List[List[float]],
    initial_design: Optional[List[float]] = None,
    dimension: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create optimization problem from registered Foundry evaluator.

    Use this when the user references a registered evaluator (check with
    foundry_list_evaluators first). This loads the evaluator configuration
    from Foundry and makes it available for optimization.

    Args:
        evaluator_id: ID of registered evaluator (e.g., "rosenbrock_eval")
        problem_id: Unique ID for this problem instance
        bounds: Design variable bounds [[lower1, upper1], ...]
        initial_design: Starting point (optional, random if not provided)
        dimension: Number of dimensions (optional, inferred from bounds)

    Returns:
        {
            "success": bool,
            "problem_id": str,
            "evaluator_id": str,
            "dimension": int,
            "message": str
        }

    Example:
        # After listing evaluators and finding "rosenbrock_eval"
        result = create_problem_from_evaluator(
            evaluator_id="rosenbrock_eval",
            problem_id="user_rosenbrock_2d",
            bounds=[[-5, 10], [-5, 10]],
            initial_design=[0, 0]
        )
        # Now use problem_id="user_rosenbrock_2d" in optimization
    """
    from paola.foundry import OptimizationFoundry, FileStorage, FoundryEvaluator
    from paola.tools.evaluator_tools import register_problem
    from paola.foundry.problem import Problem
    import numpy as np
    from datetime import datetime

    try:
        # Get Foundry instance (from context or create)
        storage = FileStorage()
        foundry = OptimizationFoundry(storage=storage)

        # Retrieve evaluator config
        config = foundry.get_evaluator_config(evaluator_id)

        # Create FoundryEvaluator instance
        evaluator = FoundryEvaluator.from_config(config)

        # Register in problem registry (for runtime use)
        register_problem(problem_id, evaluator)

        # Infer dimension
        if dimension is None:
            dimension = len(bounds)

        # Create Problem metadata
        problem = Problem(
            problem_id=problem_id,
            name=config.get('name', evaluator_id),
            dimensions=dimension,
            problem_type="user_registered",
            created_at=datetime.now().isoformat(),
            metadata={
                "evaluator_id": evaluator_id,
                "source": config.get('source', {}),
                "bounds": bounds,
                "initial_design": initial_design
            }
        )

        # Store Problem metadata in Foundry
        foundry.register_problem(problem)

        # Link evaluator to problem
        foundry.link_evaluator_to_problem(evaluator_id, problem_id)

        return {
            "success": True,
            "problem_id": problem_id,
            "evaluator_id": evaluator_id,
            "dimension": dimension,
            "message": f"Created problem '{problem_id}' using evaluator '{evaluator_id}'"
        }

    except KeyError:
        return {
            "success": False,
            "error": f"Evaluator '{evaluator_id}' not found in Foundry. Use foundry_list_evaluators to see available evaluators."
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
```

### Step 2: Minimal Prompt Update

**Add one section** to the tool list (contextual hints approach):

```python
def _get_default_tool_list() -> str:
    return """
**Problem Formulation:**
- create_benchmark_problem: Built-in analytical functions (rosenbrock, sphere, etc.)
- create_problem_from_evaluator: Use registered custom evaluators
  → First check available evaluators: foundry_list_evaluators

**Evaluator Management:**  ← NEW SECTION
- foundry_list_evaluators: See registered custom evaluators
- foundry_get_evaluator: Get evaluator details
- Note: Prefer registered evaluators over benchmarks when available

**Run Management:**
...
"""
```

**Changes**:
1. Add `create_problem_from_evaluator` to "Problem Formulation"
2. Add "Evaluator Management" section
3. Add hint: "Prefer registered evaluators over benchmarks"

**Keep it under 300 characters total addition!**

### Step 3: Tool Description Update

Update the tool's own description to guide discovery:

```python
@tool
def foundry_list_evaluators(...):
    """
    List registered custom evaluators available in Foundry.

    Use this to see what custom functions have been registered by users.
    If you find a suitable evaluator, create a problem with
    create_problem_from_evaluator instead of create_benchmark_problem.

    Returns:
        {"evaluators": [...], "count": int}
    """
```

## Testing Strategy

### Test Case 1: Explicit Evaluator ID

```
User: "Optimize rosenbrock_eval in 2D with SLSQP"

Expected Agent Behavior:
1. Recognizes "_eval" suffix suggests registered evaluator
2. Calls foundry_list_evaluators() to verify
3. Finds rosenbrock_eval
4. Calls create_problem_from_evaluator(
       evaluator_id="rosenbrock_eval",
       problem_id="user_rosenbrock",
       bounds=[[-5, 10], [-5, 10]]
   )
5. Calls run_scipy_optimization(problem_id="user_rosenbrock", ...)
```

### Test Case 2: Ambiguous Name

```
User: "Optimize rosenbrock in 2D with SLSQP"

Expected Agent Behavior:
1. Sees "rosenbrock" mentioned
2. Calls foundry_list_evaluators() to check if registered
3. If found: uses create_problem_from_evaluator
4. If not: falls back to create_benchmark_problem
```

### Test Case 3: Registration First

```
User: "Register the sphere function from evaluators.py, then optimize it"

Expected Agent Behavior:
1. Calls read_file("evaluators.py")
2. Calls foundry_store_evaluator(...) with sphere config
3. Calls foundry_list_evaluators() to verify
4. Calls create_problem_from_evaluator(evaluator_id="sphere_eval", ...)
5. Calls run_scipy_optimization(...)
```

## Proposed Changes Summary

### 1. New Tool (MUST HAVE)
- `create_problem_from_evaluator` in `paola/tools/problem_tools.py`
- ~80 lines of code
- Bridges Foundry → Problem Registry

### 2. Prompt Update (MINIMAL)
- Add 1 sentence to Problem Formulation section
- Add "Evaluator Management" subsection (3 lines)
- Total: ~150 characters added

### 3. Tool Description Updates (NICE TO HAVE)
- Update `foundry_list_evaluators` description
- Update `create_benchmark_problem` description to mention alternative
- Total: ~100 characters per tool

## Discussion Questions

1. **Prompt Philosophy**: Minimal vs Instructive vs Contextual?
   - **Recommendation**: Contextual hints (balanced approach)

2. **Default Behavior**: When user says "optimize rosenbrock"?
   - **Recommendation**: Check registered first, fall back to benchmark

3. **Bounds Specification**: Where should recommended bounds be stored?
   - **Option A**: In evaluator config (requires schema update)
   - **Option B**: User specifies at problem creation (current approach)
   - **Recommendation**: Option B for now, Option A later

4. **Agent Discovery**: How does agent know to check for registered evaluators?
   - **Option A**: Prompt explicitly says so
   - **Option B**: Agent discovers through tool descriptions
   - **Recommendation**: Both (prompt hint + tool descriptions)

## Conclusion

**Critical Path**:
1. ✅ Implement `create_problem_from_evaluator` tool (blocking)
2. ✅ Add minimal prompt update (contextual hints)
3. ✅ Update tool descriptions
4. ✅ Test with example cases

**Success Criteria**:
- Agent uses registered evaluators when user specifies evaluator_id
- Agent checks for registered evaluators before falling back to benchmarks
- Prompt stays concise (< 250 chars added)
- Workflow is discoverable

**Timeline**: 1-2 hours implementation + testing
