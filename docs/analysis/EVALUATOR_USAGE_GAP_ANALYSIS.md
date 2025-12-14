# Evaluator Usage Gap Analysis

**Date**: 2025-12-14
**Issue**: Registered evaluators are not being used by the agent for optimization

## Problem Statement

After implementing evaluator registration (Days 1-4), we discovered that the agent does **not** use registered evaluators when asked to optimize. Instead, the agent falls back to `create_benchmark_problem` which only creates built-in analytical functions.

**Example**:
```
User: "Optimize rosenbrock_eval with SLSQP in 2D"

Agent does:
  ✗ create_benchmark_problem(problem_id="rosenbrock_2d", function_name="rosenbrock")
  ✗ run_scipy_optimization(problem_id="rosenbrock_2d", ...)

Agent should do:
  ✓ Use the registered rosenbrock_eval from Foundry
  ✓ Create problem using that evaluator
  ✓ Run optimization
```

## Root Cause Analysis

### Current Architecture

The system has TWO separate subsystems that don't communicate:

#### 1. **Foundry (Persistent Storage)**
```
.paola_data/evaluators/
  ├── rosenbrock_eval.json     ← Registered evaluator config
  ├── sphere_eval.json
  └── ...
```

**Contents**: JSON configurations with:
- `evaluator_id`: "rosenbrock_eval"
- `source.file_path`: "/path/to/evaluators.py"
- `source.callable_name`: "rosenbrock"
- `capabilities`: observation_gates, caching, etc.

**Access**: Via Foundry methods:
- `foundry.get_evaluator_config(evaluator_id)` → dict
- `FoundryEvaluator.from_config(config)` → evaluator instance

#### 2. **Problem Registry (In-Memory)**
```python
# In paola/tools/evaluator_tools.py
_PROBLEM_REGISTRY: Dict[str, Any] = {
    "rosenbrock_2d": <Rosenbrock object>,  # Created by create_benchmark_problem
    "sphere_10d": <Sphere object>,
}
```

**Contents**: Maps `problem_id` to actual evaluator objects

**Access**:
- `register_problem(problem_id, evaluator_object)`
- `_get_problem(problem_id)` → evaluator object
- Used by: `evaluate_function`, `run_scipy_optimization`

### The Disconnect

```
┌─────────────────────────────────────────────────────────────┐
│                     FOUNDRY (Persistent)                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │ rosenbrock_eval.json                               │    │
│  │ {                                                  │    │
│  │   "evaluator_id": "rosenbrock_eval",               │    │
│  │   "source": {                                      │    │
│  │     "file_path": "/path/to/evaluators.py",         │    │
│  │     "callable_name": "rosenbrock"                  │    │
│  │   }                                                │    │
│  │ }                                                  │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↑
                            │
                   NO CONNECTION!  ← THE GAP
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              PROBLEM REGISTRY (In-Memory)                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │ _PROBLEM_REGISTRY = {                              │    │
│  │   "rosenbrock_2d": <Rosenbrock object>,            │    │
│  │   "sphere_10d": <Sphere object>                    │    │
│  │ }                                                  │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  Used by: evaluate_function(), run_scipy_optimization()    │
└─────────────────────────────────────────────────────────────┘
```

**Missing Bridge**: No tool to:
1. Load evaluator config from Foundry
2. Create FoundryEvaluator instance
3. Register it in problem registry
4. Make it available for optimization

### What the Agent Knows

Looking at `paola/agent/prompts/optimization.py`:

```python
def _get_default_tool_list() -> str:
    return """
**Problem Formulation:**
- create_benchmark_problem: Create benchmark optimization problem
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            Only mentions built-in benchmarks!

**Run Management:**
- start_optimization_run: Start new optimization run
- finalize_optimization_run: Finalize completed run
...
"""
```

**Agent is NEVER told**:
- That evaluators can be registered in Foundry
- How to use registered evaluators
- That `foundry_list_evaluators` shows available evaluators
- That registered evaluators should be used instead of benchmarks

## Infrastructure Inventory

### What We Have ✅

1. **Evaluator Registration System**:
   - `foundry_store_evaluator(config)` - stores evaluator in Foundry
   - `foundry_list_evaluators()` - lists registered evaluators
   - `foundry_get_evaluator(evaluator_id)` - retrieves config
   - `FoundryEvaluator.from_config(config)` - creates evaluator instance

2. **Problem Registry**:
   - `register_problem(problem_id, evaluator)` - registers in-memory
   - `_get_problem(problem_id)` - retrieves from registry
   - Used by all optimization tools

3. **Optimization Tools**:
   - `run_scipy_optimization(problem_id, ...)` - needs problem in registry
   - `evaluate_function(problem_id, design)` - needs problem in registry

### What We're Missing ❌

1. **No Bridge Tool**: Tool to load evaluator from Foundry into problem registry
   ```python
   # MISSING:
   @tool
   def create_problem_from_evaluator(
       evaluator_id: str,
       problem_id: str,
       bounds: List[List[float]],
       ...
   ) -> Dict[str, Any]:
       """Load registered evaluator and create optimization problem."""
   ```

2. **No Workflow Guidance**: Agent doesn't know:
   - When to use registered evaluators vs benchmarks
   - How to check for registered evaluators
   - How to create problems from them

3. **No Prompt Updates**: System prompt doesn't mention:
   - Registered evaluators exist
   - How to list/use them
   - Preferred workflow (use registered > create benchmark)

## Complete Workflow Analysis

### Current Workflow (What Agent Does Now)

```
User: "Optimize rosenbrock function in 2D with SLSQP"
  ↓
Agent thinks: "Need to create problem..."
  ↓
Agent calls: create_benchmark_problem(
    problem_id="rosenbrock_2d",
    function_name="rosenbrock",  ← Built-in analytical function
    dimension=2
)
  ↓
Returns: {
    "success": True,
    "problem_id": "rosenbrock_2d"
}
  ↓
Registers in _PROBLEM_REGISTRY:
    _PROBLEM_REGISTRY["rosenbrock_2d"] = <Rosenbrock analytical object>
  ↓
Agent calls: run_scipy_optimization(
    problem_id="rosenbrock_2d",  ← Uses registered analytical function
    algorithm="SLSQP",
    bounds=[[-5, 10], [-5, 10]]
)
  ↓
Optimization runs using analytical Rosenbrock function
```

**Result**: Uses built-in benchmark, NOT registered evaluator

### Desired Workflow (What Should Happen)

```
User: "Optimize rosenbrock_eval in 2D with SLSQP"
  ↓
Agent thinks: "rosenbrock_eval sounds like a registered evaluator..."
  ↓
Agent calls: foundry_list_evaluators()  ← Check what's registered
  ↓
Returns: {
    "evaluators": [
        {"evaluator_id": "rosenbrock_eval", "name": "rosenbrock", ...},
        {"evaluator_id": "sphere_eval", ...}
    ]
}
  ↓
Agent thinks: "Found rosenbrock_eval! Use it..."
  ↓
Agent calls: create_problem_from_evaluator(  ← NEW TOOL NEEDED!
    evaluator_id="rosenbrock_eval",
    problem_id="user_rosenbrock_2d",
    bounds=[[-5, 10], [-5, 10]]
)
  ↓
Tool does:
    1. config = foundry.get_evaluator_config("rosenbrock_eval")
    2. evaluator = FoundryEvaluator.from_config(config)
    3. register_problem("user_rosenbrock_2d", evaluator)
    4. Create Problem metadata
    5. foundry.register_problem(Problem(...))
  ↓
Returns: {
    "success": True,
    "problem_id": "user_rosenbrock_2d",
    "evaluator_id": "rosenbrock_eval"
}
  ↓
Agent calls: run_scipy_optimization(
    problem_id="user_rosenbrock_2d",  ← Uses registered evaluator!
    algorithm="SLSQP",
    bounds=[[-5, 10], [-5, 10]]
)
  ↓
Optimization runs using registered evaluator from evaluators.py
```

**Result**: Uses registered evaluator with full PAOLA capabilities (caching, observation gates, etc.)

## Gap Summary

### 1. **Missing Tool**: `create_problem_from_evaluator`

**What it needs to do**:
- Accept: `evaluator_id`, `problem_id`, `bounds`, initial design, dimension
- Retrieve evaluator config from Foundry
- Create FoundryEvaluator instance
- Register in problem registry
- Create and store Problem metadata
- Return problem_id for use in optimization

### 2. **Missing Prompt Guidance**

**What agent needs to know**:
- Registered evaluators are available in Foundry
- How to list them (`foundry_list_evaluators`)
- When to use them (prefer registered > benchmarks)
- How to create problems from them (`create_problem_from_evaluator`)

### 3. **Missing Workflow Priority**

**Decision logic needed**:
```
IF user mentions specific evaluator_id (e.g., "rosenbrock_eval"):
    → Use create_problem_from_evaluator
ELIF user mentions standard function name (e.g., "rosenbrock"):
    → Check if registered evaluator exists first
    → If yes: use create_problem_from_evaluator
    → If no: fall back to create_benchmark_problem
```

## Proposed Solutions

### Option 1: Minimal - Add Bridge Tool Only

**Pros**:
- Small code change
- Clean separation of concerns

**Cons**:
- Agent still needs to discover tool
- No explicit guidance on when to use it

**Implementation**:
1. Create `create_problem_from_evaluator` tool
2. Add to agent's tool list
3. Agent discovers through trial/error

### Option 2: Moderate - Tool + Prompt Update

**Pros**:
- Agent knows about registered evaluators
- Clear workflow guidance
- Tool is discoverable

**Cons**:
- More invasive prompt changes
- Need to balance minimalism with clarity

**Implementation**:
1. Create `create_problem_from_evaluator` tool
2. Update system prompt to mention:
   - "Registered evaluators available in Foundry"
   - "Use foundry_list_evaluators to see available evaluators"
   - "Prefer registered evaluators over benchmarks"
3. Update tool formatting to group registration tools

### Option 3: Comprehensive - Auto-Discovery Workflow

**Pros**:
- Agent automatically uses registered evaluators
- Seamless user experience
- Smart fallback logic

**Cons**:
- More complex logic
- Potential for confusion

**Implementation**:
1. Create `create_problem_from_evaluator` tool
2. Modify `create_benchmark_problem` to:
   - First check if evaluator with that name is registered
   - If yes: suggest using registered evaluator
   - If no: proceed with benchmark
3. Update prompt with workflow guidance
4. Add examples to prompt

## User's Concern: Infrastructure Sufficiency

> "I am not sure whether the provided tool/foundry infrastructure is sufficient for the agent, to create a complete optimization. More specifically, evaluator is only for the function and gradient calls, and we need, certainly, other optimization related objects."

### Analysis: What's Needed for Complete Optimization

**Minimum requirements**:
1. **Evaluator** (function + gradient) ✅ - We have this
2. **Design Space** (bounds, initial point) ❌ - Need to specify
3. **Optimizer Configuration** (algorithm, options) ❌ - Need to specify
4. **Constraints** (if any) ❌ - Need to specify
5. **Problem Metadata** (dimensions, type) ❌ - Need to specify

**Current Status**:
- ✅ Evaluator: Fully implemented, registered in Foundry
- ❌ Design space: Passed to optimization tools, but not stored with evaluator
- ❌ Optimizer config: Specified at run-time, not stored
- ❌ Constraints: Not integrated with evaluator registration
- ❌ Problem metadata: Stored separately from evaluator

### The Real Question

**Is the evaluator alone sufficient?**
- **For function evaluation**: YES ✅
- **For complete optimization**: NO ❌

**What else is needed?**

An evaluator only defines:
- How to compute f(x)
- How to compute ∇f(x)
- What capabilities it has (caching, gates, etc.)

But for optimization, we also need:
- **x_lower, x_upper**: Bounds on design variables
- **x_0**: Initial starting point
- **n_dim**: Number of design variables
- **Constraints**: g(x) ≤ 0, h(x) = 0
- **Problem type**: Convex, nonconvex, multimodal, etc.

**Options**:

1. **Keep separate** (current approach):
   - Evaluator = function execution
   - Problem = design space + constraints
   - Linked by evaluator_id

2. **Bundle together**:
   - Store recommended bounds with evaluator
   - Store typical constraints
   - Single "problem template" object

3. **Flexible specification**:
   - Evaluator provides function only
   - Agent specifies bounds/constraints at runtime
   - More flexible, less automated

## Recommendations

### Immediate Actions (Quick Fix)

1. **Create `create_problem_from_evaluator` tool**:
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
       Create optimization problem from registered evaluator.

       Loads evaluator from Foundry and makes it available for optimization.
       """
   ```

2. **Update system prompt** to add section:
   ```
   **Registered Evaluators:**
   - foundry_list_evaluators: See available registered evaluators
   - create_problem_from_evaluator: Create problem from registered evaluator
   - Prefer registered evaluators over benchmarks when available
   ```

3. **Test workflow**:
   ```
   User: "Optimize rosenbrock_eval in 2D"
   Agent:
     1. foundry_list_evaluators() → finds rosenbrock_eval
     2. create_problem_from_evaluator(
           evaluator_id="rosenbrock_eval",
           problem_id="user_rosenbrock",
           bounds=[[-5, 10], [-5, 10]]
        )
     3. run_scipy_optimization(problem_id="user_rosenbrock", ...)
   ```

### Medium-Term (Enhanced)

1. **Add recommended bounds to evaluator config**:
   ```json
   {
     "evaluator_id": "rosenbrock_eval",
     "source": {...},
     "recommended_setup": {
       "bounds": [[-5, 10], [-5, 10]],
       "initial_point": [0, 0],
       "dimension": 2
     }
   }
   ```

2. **Smart problem creation**:
   ```python
   create_problem_from_evaluator(
       evaluator_id="rosenbrock_eval",
       # If not specified, use recommended setup from config
       bounds=None,  # → Uses [[-5, 10], [-5, 10]] from config
       dimension=None  # → Uses 2 from config
   )
   ```

### Long-Term (Full Integration)

1. **Unified Problem Template**:
   - Single object containing: evaluator + design space + constraints
   - Stored in Foundry
   - One call to instantiate complete optimization

2. **Auto-discovery and matching**:
   - Agent: "optimize rosenbrock in 2D"
   - System checks: registered evaluators with name "rosenbrock"
   - Suggests: "Found rosenbrock_eval, use that?"

## Conclusion

**The Gap**: Missing tool to bridge Foundry (persistent evaluators) and Problem Registry (runtime evaluators)

**The Solution**: Create `create_problem_from_evaluator` tool + update prompt

**User's Concern is Valid**: Evaluator alone is NOT sufficient for complete optimization. We need design space (bounds) and optionally constraints.

**Recommended Approach**: Start minimal (tool + prompt), then enhance with recommended bounds in config.
