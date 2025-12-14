# Problem Construction Design - Proposal for Review

**Context**: Before implementing `create_problem_from_evaluators`, analyze fundamental architecture

---

## Core Design Principles

### 1. **Evaluator ≠ Problem**

**Evaluator** (Atomic):
```
evaluator: x → value
- Single computational unit
- Reusable across many problems
- Has capabilities (caching, gates, gradients)
```

**Problem** (Composite):
```
Problem = {
    objectives: [evaluator_refs],
    constraints: [evaluator_refs],
    design_space: bounds,
    ...
}
- Complete optimization formulation
- References multiple evaluators
- Defines how to compose them
```

**Relationship**: **1 Evaluator : N Problems** (many-to-many)

### 2. **Agent Composition**

The agent should be able to **compose** problems from evaluators flexibly:

```
Agent: "Minimize drag, subject to lift >= 1000"
→ Composes:
   - Objective: drag_eval
   - Constraint: lift_eval >= 1000

Agent: "Now also minimize weight"
→ Recomposes:
   - Objectives: drag_eval + weight_eval (multi-objective)
   - Constraint: lift_eval >= 1000
```

**Not hardcoded, agent decides composition!**

---

## Key Architectural Question

### What formulations must we support?

**Tier 1: Essential** (Must have)
1. ✅ Single-objective unconstrained
2. ✅ Single-objective with constraints (inequality, equality)
3. ✅ Multi-objective (weighted sum)

**Tier 2: Future** (Can add later)
4. ❌ Pareto multi-objective (needs specialized solver)
5. ❌ Integer variables (needs MILP)
6. ❌ Global optimization

**Decision**: Support Tier 1 now, design allows Tier 2 later

---

## Proposed Tool Interface

### Single Flexible Tool

```python
@tool
def create_problem_from_evaluators(
    problem_id: str,
    objectives: List[Dict[str, Any]],  # Objective specifications
    bounds: List[List[float]],          # Design space
    constraints: Optional[List[Dict[str, Any]]] = None,  # Constraint specs
    initial_design: Optional[List[float]] = None
):
    """
    Create optimization problem from registered evaluators.

    Args:
        objectives: List of objective specs:
            [{
                "evaluator_id": "drag_eval",
                "sense": "minimize",  # or "maximize"
                "weight": 1.0         # for multi-objective
            }]

        constraints: List of constraint specs:
            [{
                "evaluator_id": "lift_eval",
                "type": ">=",         # ">=", "<=", "=="
                "value": 1000.0
            }]
    """
```

### Why This Design?

**Alternative A**: One tool per formulation type
```
create_unconstrained_problem(...)
create_constrained_problem(...)
create_multiobjective_problem(...)
```
❌ Too many tools, not composable

**Alternative B**: Flexible single tool (Proposed)
```
create_problem_from_evaluators(
    objectives=[...],
    constraints=[...]  # Optional
)
```
✅ One tool, handles all cases, composable

**Alternative C**: Multiple calls to build incrementally
```
create_empty_problem(...)
add_objective(...)
add_constraint(...)
```
❌ Too many calls, stateful

**Verdict**: Alternative B (Flexible single tool)

---

## Example Compositions

### Example 1: Simple Unconstrained

```python
# User: "Optimize rosenbrock_eval in 2D"

create_problem_from_evaluators(
    problem_id="rosenbrock_2d",
    objectives=[{
        "evaluator_id": "rosenbrock_eval",
        "sense": "minimize"
    }],
    bounds=[[-5, 10], [-5, 10]]
)
```

### Example 2: Constrained

```python
# User: "Minimize drag, subject to lift >= 1000"

create_problem_from_evaluators(
    problem_id="wing_design",
    objectives=[{
        "evaluator_id": "drag_eval",
        "sense": "minimize"
    }],
    constraints=[{
        "evaluator_id": "lift_eval",
        "type": ">=",
        "value": 1000.0
    }],
    bounds=[[...]]
)
```

### Example 3: Multi-Objective

```python
# User: "Minimize drag and weight, 70% drag, 30% weight"

create_problem_from_evaluators(
    problem_id="multi_obj_wing",
    objectives=[
        {"evaluator_id": "drag_eval", "sense": "minimize", "weight": 0.7},
        {"evaluator_id": "weight_eval", "sense": "minimize", "weight": 0.3}
    ],
    bounds=[[...]]
)
```

### Example 4: Complex Multi-Constraint

```python
# User: "Minimize drag and weight with lift >= 1000 and stress <= 200"

create_problem_from_evaluators(
    problem_id="complex_wing",
    objectives=[
        {"evaluator_id": "drag_eval", "sense": "minimize", "weight": 0.6},
        {"evaluator_id": "weight_eval", "sense": "minimize", "weight": 0.4}
    ],
    constraints=[
        {"evaluator_id": "lift_eval", "type": ">=", "value": 1000},
        {"evaluator_id": "stress_eval", "type": "<=", "value": 200}
    ],
    bounds=[[...]]
)
```

---

## Implementation Strategy

### Core Component: `CompositeEvaluator`

```python
class CompositeEvaluator:
    """
    Wraps multiple evaluators into single problem interface.

    Handles:
    - Single or multiple objectives (weighted sum)
    - Constraints (transformed to scipy format)
    - Caching (delegates to underlying evaluators)
    """

    def evaluate(self, x):
        """Compute objective (weighted sum if multi-obj)."""

    def get_scipy_constraints(self):
        """Generate scipy constraint dicts."""
```

**This gets registered in problem registry and used by optimization tools!**

### Backward Compatibility

**Existing tools require NO changes**:

```python
# run_scipy_optimization just calls:
problem = _get_problem(problem_id)  # Gets CompositeEvaluator

result = scipy.optimize.minimize(
    fun=problem.evaluate,  # Works!
    constraints=problem.get_scipy_constraints()  # Works!
)
```

**Fully compatible!** ✅

---

## Questions for Review

### 1. Tool Interface

**Q**: Is `create_problem_from_evaluators` with flexible schema acceptable?
- ✅ Flexible enough?
- ✅ Still simple enough?
- ✅ Schema clear to LLM?

**Alternative considerations**:
- Simpler: Only single-objective? (too limiting)
- More explicit: Separate constraint types? (more complex)

### 2. Multi-Objective Approach

**Q**: Start with weighted sum only?

**Current proposal**:
```python
objectives=[
    {"evaluator_id": "f1", "weight": 0.7},
    {"evaluator_id": "f2", "weight": 0.3}
]
→ minimize: 0.7*f1(x) + 0.3*f2(x)
```

**Alternative**: Pareto optimization (much more complex, needs specialized solver)

**Recommendation**: Start with weighted sum, add Pareto later if needed

### 3. Constraint Types

**Q**: Support all three constraint types?
- `<=` inequality
- `>=` inequality
- `==` equality

**Scipy supports all three** → Include all ✅

### 4. Design Space

**Q**: Should evaluators store "recommended bounds"?

**Example**:
```json
{
  "evaluator_id": "rosenbrock_eval",
  "recommended_setup": {
    "bounds": [[-5, 10], [-5, 10]],
    "typical_dimension": 2
  }
}
```

**Options**:
- A: Agent always specifies bounds (current proposal)
- B: Evaluator stores recommendations, agent can override
- C: Required in evaluator config

**Recommendation**: Option A now (flexible), Option B later (convenience)

### 5. Constraint Reference

**Q**: How does agent know what evaluators can be used as constraints?

**Current**: Any evaluator can be used anywhere
- As objective
- As constraint (inequality or equality)
- In multiple roles

**Alternative**: Tag evaluators as "objective" vs "constraint" type?

**Recommendation**: Keep flexible (any evaluator, any role)

---

## What This Enables

### Capability 1: Evaluator Reuse

```
rosenbrock_eval used in:
  - Problem A: 2D unconstrained
  - Problem B: 10D unconstrained
  - Problem C: 2D constrained (rosenbrock <= 10)
  - Problem D: Multi-obj (rosenbrock + sphere)
```

### Capability 2: Dynamic Reformulation

```
Agent workflow:
1. Try unconstrained → finds infeasible solution
2. Add constraints → finds feasible solution
3. Adjust weights → better trade-off
```

### Capability 3: Mixed Evaluator Types

```
Problem:
  Objective: cfd_eval (expensive simulation)
  Constraints:
    - analytical_stress <= max (cheap analytical)
    - budget <= limit (cheap analytical)
```

**All work seamlessly!**

---

## Implementation Checklist

If approved, implement:

1. **Schema classes** (~50 lines)
   - `ObjectiveSpec`
   - `ConstraintSpec`
   - `DesignSpace`

2. **CompositeEvaluator** (~150 lines)
   - Handle multi-objective (weighted sum)
   - Handle constraints (scipy format)
   - Delegate to underlying evaluators

3. **create_problem_from_evaluators tool** (~100 lines)
   - Validate inputs
   - Load evaluators from Foundry
   - Create CompositeEvaluator
   - Register in problem registry
   - Store metadata

4. **Tests** (~200 lines)
   - Unconstrained single-obj
   - Constrained single-obj
   - Multi-objective
   - Mixed compositions

5. **Prompt update** (~50 chars)
   - Add to tool list
   - Mention preference for registered evaluators

**Total**: ~4 hours

---

## Recommendation

**Proceed with this design** because:

1. ✅ **Simple**: 1 tool, clear schema
2. ✅ **Fundamental**: Supports essential formulations
3. ✅ **Composable**: Agent can mix/match evaluators
4. ✅ **Extensible**: Easy to add features later
5. ✅ **Compatible**: No breaking changes

**Key design insights**:
- Evaluator ≠ Problem (many-to-many relationship)
- Agent composes problems from evaluators (not hardcoded)
- CompositeEvaluator handles complexity (transparent to optimization tools)

---

## Request for Feedback

Please review:
1. **Tool interface**: Is the schema clear and flexible enough?
2. **Multi-objective**: Weighted sum acceptable for now?
3. **Constraint types**: Support all three (<=, >=, ==)?
4. **Design philosophy**: Simple but fundamental - achieved?

Any concerns or suggested modifications?
