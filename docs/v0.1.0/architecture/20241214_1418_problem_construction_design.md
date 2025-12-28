# Problem Construction from Evaluators - Architectural Design

**Context**: Designing how to construct optimization problems from registered evaluators

**Principle**: Simple but fundamental - support essential compositions without over-engineering

---

## Fundamental Questions

### 1. What is an "Evaluator"?

**Current Understanding**:
```python
Evaluator: x → value
- Input: Design vector x (np.ndarray)
- Output: Scalar or vector value
- Examples:
  - drag(x) → float (drag coefficient)
  - lift(x) → float (lift coefficient)
  - stress(x) → array([σ1, σ2, σ3])
```

**Evaluator is atomic**:
- Single computational unit
- Reusable across problems
- Has no knowledge of optimization context

**Evaluator has capabilities**:
- Caching
- Observation gates
- Gradient computation
- Performance tracking

### 2. What is a "Problem"?

**A problem is a complete optimization formulation**:
```
minimize/maximize f(x)
subject to:
  g(x) ≤ 0
  h(x) = 0
  x_lower ≤ x ≤ x_upper
```

**Problem components**:
1. **Objectives**: What to minimize/maximize
2. **Constraints**: What must be satisfied
3. **Design space**: Bounds, dimensionality, initial point
4. **Problem metadata**: Type, characteristics, lineage

**Problem references evaluators but doesn't contain them**:
```
Problem "wing_optimization":
  Objective: evaluator_id="drag_eval", sense="minimize"
  Constraints:
    - evaluator_id="lift_eval", type=">=", value=100
    - evaluator_id="stress_eval", type="<=", value=200
  Bounds: [[chord_min, chord_max], [span_min, span_max], ...]
```

### 3. How do they compose?

**Key insight**: **1 Evaluator : N Problems** (many-to-many)

**Examples**:

**Same evaluator, different problems**:
```
Evaluator: rosenbrock(x)

Problem A: Unconstrained 2D
  - Objective: rosenbrock(x), minimize
  - Bounds: [[-5, 10], [-5, 10]]

Problem B: Unconstrained 10D
  - Objective: rosenbrock(x), minimize
  - Bounds: [[-5, 10]] * 10

Problem C: Constrained 2D
  - Objective: rosenbrock(x), minimize
  - Constraint: rosenbrock(x) <= 10
  - Bounds: [[-5, 10], [-5, 10]]
```

**Different evaluators, same problem**:
```
Problem: Airfoil Design

Evaluators:
  - drag_eval(x) → drag coefficient
  - lift_eval(x) → lift coefficient
  - moment_eval(x) → pitching moment

Composition:
  - Objective: drag_eval, minimize
  - Constraint 1: lift_eval >= target_lift
  - Constraint 2: |moment_eval| <= max_moment
```

---

## Optimization Formulations to Support

### Type 1: Single-Objective Unconstrained

```
minimize f(x)
subject to: x_lower ≤ x ≤ x_upper
```

**Example**:
```
Problem: "sphere_opt"
  Objective: sphere_eval(x), minimize
  Bounds: [[-5, 5]] * 10
```

**Tools needed**: `run_scipy_optimization` (current, works)

### Type 2: Single-Objective Constrained

```
minimize f(x)
subject to:
  g_i(x) ≤ 0, i=1,...,m
  h_j(x) = 0, j=1,...,p
  x_lower ≤ x ≤ x_upper
```

**Example**:
```
Problem: "wing_design"
  Objective: drag_eval(x), minimize
  Constraints:
    - lift_eval(x) >= 1000  (inequality)
    - moment_eval(x) == 0   (equality)
  Bounds: [[...]]
```

**Tools needed**:
- Define constraints from evaluators ✓
- Pass to scipy.optimize.minimize ✓

### Type 3: Multi-Objective

```
minimize [f1(x), f2(x), ..., fk(x)]
subject to: constraints...
```

**Example**:
```
Problem: "multi_obj_wing"
  Objectives:
    - drag_eval(x), minimize
    - weight_eval(x), minimize
  Constraints:
    - lift_eval(x) >= 1000
  Bounds: [[...]]
```

**Tools needed**:
- Weighted sum formulation (simple) ✓
- Pareto optimization (complex) → Future

### Type 4: Analytical vs Simulation

```
Different evaluator types in same problem:
  - Analytical functions (fast, cached)
  - Simulation evaluators (slow, expensive)
  - Surrogate models
```

**Example**:
```
Problem: "turbine_blade"
  Objective: cfd_eval(x), minimize (expensive simulation)
  Constraints:
    - stress_eval(x) <= max_stress (analytical)
    - cost_eval(x) <= budget (analytical)
```

**All evaluators have same interface** → No special handling needed ✓

---

## Design Space Exploration

### Option A: One Tool Per Formulation Type

```python
@tool
def create_unconstrained_problem(
    problem_id: str,
    objective_evaluator_id: str,
    bounds: List[List[float]]
):
    """Create unconstrained single-objective problem."""

@tool
def create_constrained_problem(
    problem_id: str,
    objective_evaluator_id: str,
    constraint_evaluator_ids: List[str],
    constraint_types: List[str],  # [">=", "<=", ...]
    constraint_values: List[float],
    bounds: List[List[float]]
):
    """Create constrained single-objective problem."""

@tool
def create_multiobjective_problem(
    problem_id: str,
    objective_evaluator_ids: List[str],
    objective_weights: List[float],
    bounds: List[List[float]]
):
    """Create multi-objective problem."""
```

**Pros**:
- Clear separation
- Explicit tool for each type
- Simple per-tool logic

**Cons**:
- 3+ tools (violates minimalism)
- Inflexible (what if constraints + multi-objective?)
- Agent must choose correct tool type

**Verdict**: ❌ Too many tools, not composable

### Option B: Unified Flexible Tool

```python
@tool
def create_problem_from_evaluators(
    problem_id: str,
    objectives: List[Dict[str, Any]],  # [{evaluator_id, sense, weight}, ...]
    constraints: Optional[List[Dict[str, Any]]] = None,  # [{evaluator_id, type, value}, ...]
    bounds: List[List[float]],
    initial_design: Optional[List[float]] = None
):
    """
    Create optimization problem from registered evaluators.

    Supports:
    - Single or multiple objectives
    - With or without constraints
    - Any evaluator composition

    Args:
        problem_id: Unique problem identifier
        objectives: List of objective specifications:
            [{
                "evaluator_id": "drag_eval",
                "sense": "minimize",  # or "maximize"
                "weight": 1.0         # for weighted sum (optional)
            }]
        constraints: List of constraint specifications:
            [{
                "evaluator_id": "lift_eval",
                "type": ">=",         # ">=", "<=", "=="
                "value": 1000.0,
                "tolerance": 1e-6     # for equality (optional)
            }]
        bounds: Design variable bounds [[lower, upper], ...]
        initial_design: Starting point (optional)

    Examples:
        # Unconstrained single-objective
        create_problem_from_evaluators(
            problem_id="rosenbrock_2d",
            objectives=[{"evaluator_id": "rosenbrock_eval", "sense": "minimize"}],
            bounds=[[-5, 10], [-5, 10]]
        )

        # Constrained single-objective
        create_problem_from_evaluators(
            problem_id="wing_design",
            objectives=[{"evaluator_id": "drag_eval", "sense": "minimize"}],
            constraints=[
                {"evaluator_id": "lift_eval", "type": ">=", "value": 1000}
            ],
            bounds=[[...]]
        )

        # Multi-objective
        create_problem_from_evaluators(
            problem_id="multi_obj",
            objectives=[
                {"evaluator_id": "drag_eval", "sense": "minimize", "weight": 0.7},
                {"evaluator_id": "weight_eval", "sense": "minimize", "weight": 0.3}
            ],
            bounds=[[...]]
        )
    """
```

**Pros**:
- Single tool (minimal)
- Flexible (all formulation types)
- Composable (agent specifies what they want)
- Clear schema

**Cons**:
- More complex tool signature
- Agent must structure dicts correctly

**Verdict**: ✅ Best balance of simplicity and flexibility

### Option C: Compositional API (Multiple Calls)

```python
@tool
def create_empty_problem(problem_id: str, bounds: List[List[float]]):
    """Create empty problem with design space."""

@tool
def add_objective(problem_id: str, evaluator_id: str, sense: str):
    """Add objective to problem."""

@tool
def add_constraint(problem_id: str, evaluator_id: str, type: str, value: float):
    """Add constraint to problem."""

@tool
def finalize_problem(problem_id: str):
    """Finalize problem for optimization."""
```

**Pros**:
- Very flexible
- Simple per-tool signatures
- Explicit composition

**Cons**:
- 4 tool calls minimum
- Stateful (problem built incrementally)
- More token usage
- Harder to validate complete problem

**Verdict**: ❌ Too many calls, stateful complexity

---

## Recommended Architecture

### Schema Design

```python
from dataclasses import dataclass
from typing import List, Optional, Literal
from enum import Enum

class ObjectiveSense(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

class ConstraintType(str, Enum):
    LESS_EQUAL = "<="
    GREATER_EQUAL = ">="
    EQUAL = "=="

@dataclass
class ObjectiveSpec:
    """Specification for one objective."""
    evaluator_id: str
    name: str  # "drag", "cost", etc.
    sense: ObjectiveSense = ObjectiveSense.MINIMIZE
    weight: float = 1.0  # For weighted sum multi-objective

@dataclass
class ConstraintSpec:
    """Specification for one constraint."""
    evaluator_id: str
    name: str  # "lift_constraint", etc.
    type: ConstraintType
    value: float  # RHS value
    tolerance: float = 1e-6  # For equality constraints

@dataclass
class DesignSpace:
    """Design space specification."""
    bounds: List[List[float]]  # [[lower, upper], ...]
    initial_point: Optional[List[float]] = None
    dimension: Optional[int] = None  # Inferred from bounds if not provided

@dataclass
class ProblemDefinition:
    """Complete optimization problem definition."""
    problem_id: str
    objectives: List[ObjectiveSpec]
    design_space: DesignSpace
    constraints: Optional[List[ConstraintSpec]] = None
    problem_type: str = "user_defined"  # Inferred from structure
    metadata: dict = None  # Additional info

    def infer_problem_type(self) -> str:
        """Infer problem type from structure."""
        has_constraints = self.constraints and len(self.constraints) > 0
        multi_obj = len(self.objectives) > 1

        if multi_obj and has_constraints:
            return "multi_objective_constrained"
        elif multi_obj:
            return "multi_objective_unconstrained"
        elif has_constraints:
            return "single_objective_constrained"
        else:
            return "single_objective_unconstrained"
```

### Implementation Strategy

```python
@tool
def create_problem_from_evaluators(
    problem_id: str,
    objectives: List[Dict[str, Any]],
    bounds: List[List[float]],
    constraints: Optional[List[Dict[str, Any]]] = None,
    initial_design: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Create optimization problem from registered evaluators.

    [Full docstring as shown in Option B above]
    """
    from paola.foundry import OptimizationFoundry, FileStorage
    from paola.foundry.problem import Problem, ProblemDefinition
    from paola.tools.evaluator_tools import register_problem
    import numpy as np

    try:
        # 1. Validate inputs
        if not objectives:
            return {"success": False, "error": "At least one objective required"}

        # 2. Get Foundry instance
        storage = FileStorage()
        foundry = OptimizationFoundry(storage=storage)

        # 3. Retrieve and validate all evaluators
        evaluator_instances = {}

        for obj in objectives:
            eval_id = obj["evaluator_id"]
            config = foundry.get_evaluator_config(eval_id)
            evaluator = FoundryEvaluator.from_config(config)
            evaluator_instances[eval_id] = evaluator

        if constraints:
            for cons in constraints:
                eval_id = cons["evaluator_id"]
                if eval_id not in evaluator_instances:
                    config = foundry.get_evaluator_config(eval_id)
                    evaluator = FoundryEvaluator.from_config(config)
                    evaluator_instances[eval_id] = evaluator

        # 4. Create composite evaluator for problem
        composite = CompositeEvaluator(
            objectives=objectives,
            constraints=constraints,
            evaluator_instances=evaluator_instances
        )

        # 5. Register in problem registry (for runtime)
        register_problem(problem_id, composite)

        # 6. Create problem metadata
        dimension = len(bounds)
        problem_def = ProblemDefinition(
            problem_id=problem_id,
            objectives=[ObjectiveSpec(**obj) for obj in objectives],
            design_space=DesignSpace(
                bounds=bounds,
                initial_point=initial_design,
                dimension=dimension
            ),
            constraints=[ConstraintSpec(**c) for c in constraints] if constraints else None
        )

        problem_type = problem_def.infer_problem_type()

        problem = Problem(
            problem_id=problem_id,
            name=problem_id,
            dimensions=dimension,
            problem_type=problem_type,
            created_at=datetime.now().isoformat(),
            metadata={
                "objectives": objectives,
                "constraints": constraints,
                "bounds": bounds,
                "evaluator_ids": list(evaluator_instances.keys())
            }
        )

        # 7. Store in Foundry
        foundry.register_problem(problem)

        # 8. Link evaluators to problem
        for eval_id in evaluator_instances.keys():
            foundry.link_evaluator_to_problem(eval_id, problem_id)

        return {
            "success": True,
            "problem_id": problem_id,
            "problem_type": problem_type,
            "dimension": dimension,
            "num_objectives": len(objectives),
            "num_constraints": len(constraints) if constraints else 0,
            "evaluators_used": list(evaluator_instances.keys()),
            "message": f"Created {problem_type} problem '{problem_id}'"
        }

    except KeyError as e:
        return {
            "success": False,
            "error": f"Evaluator not found: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


class CompositeEvaluator:
    """
    Composite evaluator that handles multiple objectives and constraints.

    This is what gets registered in the problem registry and called by
    scipy.optimize.minimize and other optimization tools.
    """

    def __init__(
        self,
        objectives: List[Dict[str, Any]],
        constraints: Optional[List[Dict[str, Any]]],
        evaluator_instances: Dict[str, FoundryEvaluator]
    ):
        self.objectives = objectives
        self.constraints = constraints or []
        self.evaluators = evaluator_instances

    def evaluate(self, x: np.ndarray):
        """
        Evaluate objective function.

        For single objective: returns scalar
        For multi-objective: returns weighted sum
        """
        if len(self.objectives) == 1:
            # Single objective
            obj_spec = self.objectives[0]
            evaluator = self.evaluators[obj_spec["evaluator_id"]]
            result = evaluator.evaluate(x)
            value = result.objectives["objective"]

            # Apply sense
            if obj_spec.get("sense", "minimize") == "maximize":
                value = -value

            return value
        else:
            # Multi-objective: weighted sum
            total = 0.0
            for obj_spec in self.objectives:
                evaluator = self.evaluators[obj_spec["evaluator_id"]]
                result = evaluator.evaluate(x)
                value = result.objectives["objective"]

                if obj_spec.get("sense", "minimize") == "maximize":
                    value = -value

                weight = obj_spec.get("weight", 1.0)
                total += weight * value

            return total

    def evaluate_constraint(self, x: np.ndarray, constraint_index: int):
        """
        Evaluate specific constraint.

        Scipy expects: g(x) <= 0 form
        We transform user's specification to this form.
        """
        cons_spec = self.constraints[constraint_index]
        evaluator = self.evaluators[cons_spec["evaluator_id"]]
        result = evaluator.evaluate(x)
        value = result.objectives["objective"]

        # Transform to g(x) <= 0 form
        cons_type = cons_spec["type"]
        cons_value = cons_spec["value"]

        if cons_type == "<=":
            # g(x) <= value  →  g(x) - value <= 0
            return value - cons_value
        elif cons_type == ">=":
            # g(x) >= value  →  value - g(x) <= 0
            return cons_value - value
        elif cons_type == "==":
            # g(x) == value  →  |g(x) - value| <= tol (handled by scipy)
            return value - cons_value
        else:
            raise ValueError(f"Unknown constraint type: {cons_type}")

    def get_scipy_constraints(self):
        """
        Generate scipy constraint dicts.

        Returns list of dicts in scipy format.
        """
        scipy_constraints = []

        for i, cons_spec in enumerate(self.constraints):
            cons_type = cons_spec["type"]

            if cons_type in ["<=", ">="]:
                # Inequality constraint
                scipy_constraints.append({
                    "type": "ineq",  # Scipy expects g(x) >= 0, we return -g(x)
                    "fun": lambda x, idx=i: -self.evaluate_constraint(x, idx)
                })
            elif cons_type == "==":
                # Equality constraint
                scipy_constraints.append({
                    "type": "eq",
                    "fun": lambda x, idx=i: self.evaluate_constraint(x, idx)
                })

        return scipy_constraints
```

---

## Usage Examples

### Example 1: Simple Unconstrained

```python
# Agent workflow:
# User: "Optimize rosenbrock_eval in 2D with SLSQP"

# 1. Check registered evaluators
result = foundry_list_evaluators()
# → Finds rosenbrock_eval

# 2. Create problem
result = create_problem_from_evaluators(
    problem_id="user_rosenbrock_2d",
    objectives=[{
        "evaluator_id": "rosenbrock_eval",
        "sense": "minimize"
    }],
    bounds=[[-5, 10], [-5, 10]],
    initial_design=[0, 0]
)
# → Returns: {
#      "problem_id": "user_rosenbrock_2d",
#      "problem_type": "single_objective_unconstrained"
#    }

# 3. Run optimization
result = run_scipy_optimization(
    problem_id="user_rosenbrock_2d",
    algorithm="SLSQP",
    bounds=[[-5, 10], [-5, 10]]
)
```

### Example 2: Constrained Single-Objective

```python
# User: "Minimize drag_eval subject to lift_eval >= 1000"

# 1. Create problem with constraint
result = create_problem_from_evaluators(
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
    bounds=[[...]]  # Airfoil parameters
)
# → Returns: {
#      "problem_id": "wing_design",
#      "problem_type": "single_objective_constrained"
#    }

# 2. Run constrained optimization
result = run_scipy_optimization(
    problem_id="wing_design",
    algorithm="SLSQP",  # Supports constraints
    bounds=[[...]]
)
```

### Example 3: Multi-Objective

```python
# User: "Minimize drag and weight, 70% drag, 30% weight"

result = create_problem_from_evaluators(
    problem_id="multi_obj_wing",
    objectives=[
        {
            "evaluator_id": "drag_eval",
            "sense": "minimize",
            "weight": 0.7
        },
        {
            "evaluator_id": "weight_eval",
            "sense": "minimize",
            "weight": 0.3
        }
    ],
    bounds=[[...]]
)
# → Returns: {
#      "problem_id": "multi_obj_wing",
#      "problem_type": "multi_objective_unconstrained"
#    }
```

### Example 4: Complex Composition

```python
# User: "Minimize drag and weight, with lift >= 1000 and stress <= 200"

result = create_problem_from_evaluators(
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
# → Returns: {
#      "problem_id": "complex_wing",
#      "problem_type": "multi_objective_constrained",
#      "num_objectives": 2,
#      "num_constraints": 2
#    }
```

---

## Integration with Existing Tools

### Modification to `run_scipy_optimization`

Current signature:
```python
def run_scipy_optimization(
    problem_id: str,
    algorithm: str,
    bounds: List[List[float]],
    ...
)
```

**No changes needed!**

The `CompositeEvaluator` registered under `problem_id` has:
- `evaluate(x)` → objective value (handles multi-obj via weighted sum)
- `get_scipy_constraints()` → list of scipy constraint dicts

`run_scipy_optimization` just calls these methods:
```python
problem = _get_problem(problem_id)  # Gets CompositeEvaluator

# Get constraints if problem has them
constraints = []
if hasattr(problem, 'get_scipy_constraints'):
    constraints = problem.get_scipy_constraints()

# Run scipy
result = scipy.optimize.minimize(
    fun=problem.evaluate,
    x0=initial_design,
    method=algorithm,
    bounds=bounds,
    constraints=constraints
)
```

**Fully backward compatible!**

---

## What This Enables

### Agent Composition Capabilities

1. **Mix evaluator types**:
   ```
   Objectives: analytical_eval (fast)
   Constraints: simulation_eval (slow, expensive)
   ```

2. **Reuse evaluators across problems**:
   ```
   rosenbrock_eval used in:
     - Problem A: 2D unconstrained
     - Problem B: 10D unconstrained
     - Problem C: 2D constrained (rosenbrock <= 10)
     - Problem D: Multi-obj (rosenbrock + sphere)
   ```

3. **Dynamic problem formulation**:
   ```
   Agent: "Let me try unconstrained first..."
   [Creates unconstrained problem]

   Agent: "Solution infeasible, adding constraints..."
   [Creates constrained problem]
   ```

4. **Iterative refinement**:
   ```
   Agent: "Weight drag 90%, weight 10%"
   [Creates problem, optimizes]

   Agent: "Trade-off not good, try 70%-30%"
   [Creates new problem with different weights]
   ```

---

## Summary

### Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Tool count** | 1 tool: `create_problem_from_evaluators` | Minimal, flexible |
| **Schema** | Structured dicts (objectives, constraints) | Balance simplicity & expressiveness |
| **Composition** | `CompositeEvaluator` class | Encapsulates multi-evaluator logic |
| **Formulations** | All via same tool | Agent specifies structure, tool adapts |
| **Backward compat** | No changes to existing tools | `run_scipy_optimization` unchanged |

### What We Support

✅ Single-objective unconstrained
✅ Single-objective constrained (inequality, equality)
✅ Multi-objective (weighted sum)
✅ Mixed evaluator types
✅ Evaluator reuse
✅ Agent composition

### What We Don't Support (Future)

❌ Pareto multi-objective (need specialized solver)
❌ Integer/mixed-integer (need MILP solver)
❌ Global optimization (need specialized algorithm)
❌ Robust optimization (need uncertainty quantification)

---

## Next Steps

1. **Implement schema classes** (`ObjectiveSpec`, `ConstraintSpec`, etc.)
2. **Implement `CompositeEvaluator`** (handles multi-obj, constraints)
3. **Implement `create_problem_from_evaluators` tool**
4. **Update prompt** (minimal addition)
5. **Test composition scenarios**

**Timeline**: ~4 hours for core implementation

**This design is:**
- ✅ Simple (1 tool, clear schema)
- ✅ Fundamental (supports essential compositions)
- ✅ Extensible (easy to add features later)
- ✅ Backward compatible (no breaking changes)
