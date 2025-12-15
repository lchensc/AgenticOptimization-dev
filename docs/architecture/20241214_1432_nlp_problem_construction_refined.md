# NLP Problem Construction - Refined Design

**Based on feedback**: Focus on Nonlinear Programming (NLP) only, emphasize agent composition

---

## Key Clarifications from Feedback

### 1. **Two Variants Identified**

**Variant 1: Nonlinear Programming (NLP)** ← **IMPLEMENT NOW**
```
minimize f(x)          (nonlinear)
subject to:
  g_i(x) ≤ 0          (0 to N inequality constraints, nonlinear)
  h_j(x) = 0          (0 to N equality constraints, nonlinear)
  x_lower ≤ x ≤ x_upper
```

**Variant 2: Multi-Objective Optimization (MOO)** ← **DEFER (Phase 7+)**
```
minimize [f1(x), f2(x), ..., fk(x)]   (Pareto front)
subject to: constraints...
```
- **Not weighted sum** (that's just a scalarization of NLP)
- **Requires specialized solver** (NSGA-II, MOEA/D, etc.)
- **Defer to Phase 7** when Pymoo integration happens

### 2. **Problem Type Taxonomy**

Optimization problems have well-defined types:

| Type | Description | Variables | Objective | Constraints |
|------|-------------|-----------|-----------|-------------|
| **LP** | Linear Programming | Continuous | Linear | Linear |
| **QP** | Quadratic Programming | Continuous | Quadratic | Linear |
| **NLP** | Nonlinear Programming | Continuous | Nonlinear | Nonlinear |
| **MILP** | Mixed-Integer LP | Mixed int/cont | Linear | Linear |
| **MINLP** | Mixed-Integer NLP | Mixed int/cont | Nonlinear | Nonlinear |
| **MOO** | Multi-Objective | Any | Vector | Any |
| **SDP** | Semidefinite Programming | Matrix | Linear | Matrix constraints |

**For now**: Only **NLP** (continuous variables, nonlinear functions)

**Why problem types matter**:
- Different types require different solvers
- Agent needs to identify problem type to select appropriate solver
- Example: LP → use CPLEX/Gurobi, NLP → use SLSQP/IPOPT

### 3. **Agent Composition is Core Innovation**

From `universal_architecture.md`:

> "The agent autonomously composes optimization formulations"
> "Flexible composition - agent decides how to structure the problem"
> "Not hardcoded templates, but compositional flexibility"

**This means**:
- Agent should have **flexibility** to compose problems from evaluators
- Not restricted to predefined templates
- Can iteratively reformulate (e.g., add constraints, change objectives)
- Can experiment with different formulations

**Example of agent composition flexibility**:
```
User: "Minimize drag on airfoil, maintain lift"

Agent workflow:
  1. Start unconstrained: minimize drag(x)
  2. [Finds infeasible solution]
  3. Add constraint: minimize drag(x), s.t. lift(x) >= 1000
  4. [Still issues]
  5. Reformulate: minimize drag(x), s.t. lift(x) >= 1020 (tighten 2%)
  6. [Converges]
```

**Agent composes and recomposes as needed!**

---

## Refined Architecture: NLP Only

### Problem Schema for NLP

```python
@dataclass
class NLPProblem:
    """
    Nonlinear Programming problem specification.

    Standard form:
        minimize f(x)
        subject to:
          g_i(x) ≤ 0,  i = 1, ..., m_ineq
          h_j(x) = 0,  j = 1, ..., m_eq
          x_lower ≤ x ≤ x_upper

    Where:
    - f(x): Nonlinear objective function
    - g_i(x): Nonlinear inequality constraints
    - h_j(x): Nonlinear equality constraints
    - x: Continuous variables
    """

    problem_id: str
    problem_type: Literal["NLP"] = "NLP"  # Fixed for now

    # Design space
    dimension: int
    bounds: List[List[float]]  # [[lower, upper], ...]
    initial_point: Optional[List[float]] = None

    # Objective
    objective_evaluator_id: str
    objective_sense: Literal["minimize", "maximize"] = "minimize"

    # Constraints (optional)
    inequality_constraints: List[InequalityConstraint] = field(default_factory=list)
    equality_constraints: List[EqualityConstraint] = field(default_factory=list)

    # Metadata
    created_at: str
    description: Optional[str] = None


@dataclass
class InequalityConstraint:
    """
    Inequality constraint: g(x) ≤ value.

    Examples:
    - g(x) ≤ 0     → value=0 (standard form)
    - g(x) ≤ 100   → value=100
    - g(x) ≥ 50    → transformed to  -g(x) ≤ -50
    """
    name: str
    evaluator_id: str
    constraint_type: Literal["<=", ">="]
    value: float  # RHS value


@dataclass
class EqualityConstraint:
    """
    Equality constraint: h(x) = value.

    Examples:
    - h(x) = 0     → value=0 (standard form)
    - h(x) = 100   → value=100

    Note: Equality constraints are harder to satisfy.
    Scipy uses tolerance (typically 1e-6).
    """
    name: str
    evaluator_id: str
    value: float  # RHS value
    tolerance: float = 1e-6  # Tolerance for equality
```

### Simplified Tool Interface

```python
@tool
def create_nlp_problem(
    problem_id: str,
    objective_evaluator_id: str,
    bounds: List[List[float]],
    objective_sense: str = "minimize",
    inequality_constraints: Optional[List[Dict[str, Any]]] = None,
    equality_constraints: Optional[List[Dict[str, Any]]] = None,
    initial_point: Optional[List[float]] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create Nonlinear Programming (NLP) problem from registered evaluators.

    NLP standard form:
        minimize/maximize f(x)
        subject to:
          g_i(x) ≤ value,  i = 1, ..., m_ineq
          h_j(x) = value,  j = 1, ..., m_eq
          x_lower ≤ x ≤ x_upper

    Args:
        problem_id: Unique problem identifier
        objective_evaluator_id: Evaluator ID for objective function f(x)
        bounds: Design variable bounds [[lower, upper], ...]
        objective_sense: "minimize" or "maximize"
        inequality_constraints: List of inequality constraint specs:
            [{
                "name": "lift_constraint",
                "evaluator_id": "lift_eval",
                "type": ">=",           # ">=" or "<="
                "value": 1000.0
            }]
        equality_constraints: List of equality constraint specs:
            [{
                "name": "moment_balance",
                "evaluator_id": "moment_eval",
                "value": 0.0,
                "tolerance": 1e-6       # Optional
            }]
        initial_point: Starting point (optional, random if not provided)
        description: Human-readable problem description

    Returns:
        {
            "success": bool,
            "problem_id": str,
            "problem_type": "NLP",
            "dimension": int,
            "num_inequality_constraints": int,
            "num_equality_constraints": int,
            "evaluators_used": List[str],
            "message": str
        }

    Examples:
        # Unconstrained NLP
        create_nlp_problem(
            problem_id="rosenbrock_2d",
            objective_evaluator_id="rosenbrock_eval",
            bounds=[[-5, 10], [-5, 10]]
        )

        # Constrained NLP (inequality only)
        create_nlp_problem(
            problem_id="airfoil_design",
            objective_evaluator_id="drag_eval",
            bounds=[[0, 15], [0.1, 0.5]],  # [alpha, thickness]
            inequality_constraints=[
                {"name": "min_lift", "evaluator_id": "lift_eval", "type": ">=", "value": 1000},
                {"name": "max_stress", "evaluator_id": "stress_eval", "type": "<=", "value": 200}
            ]
        )

        # NLP with both inequality and equality
        create_nlp_problem(
            problem_id="wing_design",
            objective_evaluator_id="drag_eval",
            bounds=[[...]],
            inequality_constraints=[
                {"name": "min_lift", "evaluator_id": "lift_eval", "type": ">=", "value": 1000}
            ],
            equality_constraints=[
                {"name": "moment_balance", "evaluator_id": "moment_eval", "value": 0.0}
            ]
        )

    Note:
        - Only supports continuous variables (NLP)
        - For integer variables, use create_milp_problem (Phase 7+)
        - For multi-objective, use create_moo_problem (Phase 7+)
        - Weighted sum of objectives is NOT multi-objective, just NLP with composite objective
    """
```

---

## Implementation Strategy

### Core Component: NLPEvaluator

```python
class NLPEvaluator:
    """
    Composite evaluator for NLP problems.

    Wraps:
    - 1 objective evaluator
    - 0-N inequality constraint evaluators
    - 0-N equality constraint evaluators

    Provides scipy.optimize.minimize-compatible interface.
    """

    def __init__(
        self,
        objective_eval: FoundryEvaluator,
        objective_sense: str,
        ineq_constraints: List[InequalityConstraint],
        eq_constraints: List[EqualityConstraint],
        constraint_evaluators: Dict[str, FoundryEvaluator]
    ):
        self.objective_eval = objective_eval
        self.objective_sense = objective_sense
        self.ineq_constraints = ineq_constraints
        self.eq_constraints = eq_constraints
        self.constraint_evals = constraint_evaluators

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate objective function.

        For minimize: return f(x)
        For maximize: return -f(x)
        """
        result = self.objective_eval.evaluate(x)
        obj_value = result.objectives["objective"]

        if self.objective_sense == "maximize":
            obj_value = -obj_value

        return float(obj_value)

    def get_scipy_constraints(self) -> List[Dict]:
        """
        Generate scipy constraint dictionaries.

        Scipy expects constraints in form:
        - Inequality: g(x) >= 0  (note: >=, not <=)
        - Equality: h(x) = 0

        We transform user specifications to this form.
        """
        scipy_constraints = []

        # Inequality constraints
        for i, cons in enumerate(self.ineq_constraints):
            def cons_func(x, cons_idx=i):
                cons_spec = self.ineq_constraints[cons_idx]
                evaluator = self.constraint_evals[cons_spec.evaluator_id]
                result = evaluator.evaluate(x)
                g_x = result.objectives["objective"]

                # Transform to scipy form: g(x) >= 0
                if cons_spec.constraint_type == "<=":
                    # User: g(x) <= value
                    # Transform: value - g(x) >= 0
                    return float(cons_spec.value - g_x)
                else:  # ">="
                    # User: g(x) >= value
                    # Transform: g(x) - value >= 0
                    return float(g_x - cons_spec.value)

            scipy_constraints.append({
                "type": "ineq",
                "fun": cons_func
            })

        # Equality constraints
        for i, cons in enumerate(self.eq_constraints):
            def cons_func(x, cons_idx=i):
                cons_spec = self.eq_constraints[cons_idx]
                evaluator = self.constraint_evals[cons_spec.evaluator_id]
                result = evaluator.evaluate(x)
                h_x = result.objectives["objective"]

                # Scipy form: h(x) = 0
                # User: h(x) = value
                # Transform: h(x) - value = 0
                return float(h_x - cons_spec.value)

            scipy_constraints.append({
                "type": "eq",
                "fun": cons_func
            })

        return scipy_constraints

    def get_gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute objective gradient (for gradient-based solvers)."""
        return self.objective_eval.compute_gradient(x, method="finite_difference")

    def get_constraint_gradients(self, x: np.ndarray) -> List[np.ndarray]:
        """Compute constraint gradients (for gradient-based solvers)."""
        # Phase 7+: Implement constraint Jacobian
        # For now, scipy uses finite-difference internally
        pass
```

---

## Problem Type Taxonomy Integration

### Problem Type Detection

```python
class ProblemTypeDetector:
    """
    Detect problem type from specification.

    Helps agent select appropriate solver.
    """

    @staticmethod
    def detect_type(problem_spec: Dict) -> str:
        """
        Detect optimization problem type.

        Returns:
            One of: "LP", "QP", "NLP", "MILP", "MINLP", "MOO", "Unknown"
        """
        # Check variable types
        has_integer_vars = problem_spec.get("has_integer_variables", False)
        has_continuous_vars = problem_spec.get("has_continuous_variables", True)

        # Check objective/constraint linearity
        # (For now, all user-provided evaluators are assumed nonlinear)
        is_linear = problem_spec.get("is_linear", False)
        is_quadratic = problem_spec.get("is_quadratic", False)

        # Check number of objectives
        num_objectives = len(problem_spec.get("objectives", [1]))

        # Decision logic
        if num_objectives > 1:
            return "MOO"  # Multi-objective

        if has_integer_vars and has_continuous_vars:
            if is_linear:
                return "MILP"
            else:
                return "MINLP"

        if has_integer_vars:
            return "IP"  # Pure integer programming

        # Continuous variables only
        if is_linear:
            return "LP"
        elif is_quadratic:
            return "QP"
        else:
            return "NLP"  # Default for user evaluators


class SolverSelector:
    """
    Select appropriate solver based on problem type.

    Agent uses this to make informed decisions.
    """

    SOLVER_MAPPING = {
        "NLP": {
            "gradient_based": ["SLSQP", "IPOPT", "SNOPT"],
            "derivative_free": ["COBYLA", "BOBYQA", "Nelder-Mead"]
        },
        "LP": {
            "solvers": ["CPLEX", "Gurobi", "GLPK", "HiGHS"]
        },
        "QP": {
            "solvers": ["OSQP", "CVXOPT", "quadprog"]
        },
        "MILP": {
            "solvers": ["CPLEX", "Gurobi", "CBC", "SCIP"]
        },
        "MOO": {
            "solvers": ["NSGA-II", "NSGA-III", "MOEA/D"]
        }
    }

    @staticmethod
    def recommend_solver(problem_type: str, gradient_available: bool) -> List[str]:
        """
        Recommend solvers for problem type.

        Returns ranked list of suitable solvers.
        """
        if problem_type == "NLP":
            if gradient_available:
                return ["SLSQP", "IPOPT"]  # SciPy SLSQP available now
            else:
                return ["COBYLA", "Nelder-Mead"]

        elif problem_type == "LP":
            return ["HiGHS"]  # Open-source LP solver

        elif problem_type == "MOO":
            return ["NSGA-II"]  # Phase 7+

        else:
            return ["SLSQP"]  # Fallback to general NLP solver
```

---

## Agent Composition Examples

### Example 1: Simple Unconstrained NLP

```python
# User: "Optimize rosenbrock_eval in 2D"

# Agent workflow:
# 1. List evaluators
result = foundry_list_evaluators()
# → Finds rosenbrock_eval

# 2. Detect problem type
# - Single objective
# - No constraints
# - Continuous variables
# → Type: NLP (unconstrained)

# 3. Create NLP problem
result = create_nlp_problem(
    problem_id="rosenbrock_2d",
    objective_evaluator_id="rosenbrock_eval",
    bounds=[[-5, 10], [-5, 10]],
    objective_sense="minimize"
)
# → Returns: {
#      "problem_type": "NLP",
#      "num_inequality_constraints": 0,
#      "num_equality_constraints": 0
#    }

# 4. Select solver
# Type: NLP, unconstrained, gradient available
# → Recommends: SLSQP or L-BFGS-B

# 5. Run optimization
result = run_scipy_optimization(
    problem_id="rosenbrock_2d",
    algorithm="SLSQP",
    bounds=[[-5, 10], [-5, 10]]
)
```

### Example 2: Constrained NLP with Iteration

```python
# User: "Minimize drag, maintain lift >= 1000"

# Agent workflow - Iteration 1:
# 1. Create constrained NLP
result = create_nlp_problem(
    problem_id="wing_v1",
    objective_evaluator_id="drag_eval",
    bounds=[[0, 15], [0.1, 0.5]],
    inequality_constraints=[
        {"name": "min_lift", "evaluator_id": "lift_eval", "type": ">=", "value": 1000}
    ]
)

# 2. Run optimization
result = run_scipy_optimization(
    problem_id="wing_v1",
    algorithm="SLSQP",  # Supports constraints
    bounds=[[0, 15], [0.1, 0.5]]
)

# → Result: Constraint violated repeatedly, lift = 995

# Agent observes issue and decides to reformulate

# Agent workflow - Iteration 2:
# 1. Create tightened NLP (agent composition!)
result = create_nlp_problem(
    problem_id="wing_v2",
    objective_evaluator_id="drag_eval",
    bounds=[[0, 15], [0.1, 0.5]],
    inequality_constraints=[
        {"name": "min_lift", "evaluator_id": "lift_eval", "type": ">=", "value": 1020}  # Tightened by 2%
    ]
)

# 2. Run optimization
result = run_scipy_optimization(
    problem_id="wing_v2",
    algorithm="SLSQP",
    bounds=[[0, 15], [0.1, 0.5]]
)

# → Result: Success, lift = 1025, feasible!
```

**This demonstrates agent composition flexibility!**

---

## Comparison: What Changed from Previous Design

| Aspect | Previous Design | Refined Design (NLP Only) |
|--------|----------------|---------------------------|
| **Scope** | Single-obj + Multi-obj | NLP only (single-obj with constraints) |
| **Multi-objective** | Weighted sum supported | Recognized as NOT true multi-obj, deferred |
| **Problem types** | Not distinguished | Explicit taxonomy (LP, NLP, MILP, etc.) |
| **Tool interface** | `create_problem_from_evaluators` | `create_nlp_problem` (focused) |
| **Objectives param** | `List[Dict]` (multi-obj) | `objective_evaluator_id` (single) |
| **Constraints** | Single list | Separated: inequality + equality |
| **Complexity** | More general, more complex | Focused on NLP, simpler |
| **Agent composition** | Mentioned | **Emphasized as core innovation** |

---

## Integration with Universal Architecture

### How This Fits

From `universal_architecture.md`:

**Agent autonomy**:
- ✅ Agent composes NLP problems flexibly
- ✅ Can iterate and reformulate (add/remove constraints, tighten bounds)
- ✅ Not restricted to templates

**Problem modeling**:
- ✅ Supports various input formats (natural language, code, structured)
- ✅ Agent identifies problem type (NLP vs LP vs MOO)
- ✅ Selects appropriate solver based on type

**Optimizer expertise**:
- ✅ Problem type → solver recommendation
- ✅ NLP → SLSQP/IPOPT/COBYLA depending on gradients
- ✅ Knowledge of 30+ algorithms across libraries

**Easy integration**:
- ✅ User provides evaluators (already registered in Foundry)
- ✅ Agent creates NLP problem from evaluators
- ✅ Connects to appropriate solver (SciPy, NLopt, etc.)

---

## Summary

### Design Decisions

1. **Focus on NLP** (nonlinear programming) only for now
2. **Defer multi-objective** to Phase 7+ when Pymoo integration happens
3. **Recognize weighted sum** as NLP scalarization, not true multi-obj
4. **Add problem type taxonomy** (LP, NLP, MILP, etc.)
5. **Emphasize agent composition** as core innovation
6. **Simplify tool interface** - one tool for NLP creation

### What We Support

✅ Unconstrained NLP: minimize f(x)
✅ Constrained NLP: minimize f(x), s.t. g(x) ≤ 0, h(x) = 0
✅ Any number of inequality constraints
✅ Any number of equality constraints
✅ Agent composition and reformulation
✅ Problem type detection and solver selection

### What We Defer

❌ True multi-objective (Pareto) - Phase 7+ (Pymoo)
❌ Integer variables (MILP) - Phase 7+
❌ Linear/Quadratic specialized solvers - Phase 6+

### Ready for Implementation

**Core components**:
1. `NLPProblem` schema (~50 lines)
2. `NLPEvaluator` class (~150 lines)
3. `create_nlp_problem` tool (~120 lines)
4. `ProblemTypeDetector` (~80 lines)
5. Tests (~250 lines)

**Total**: ~650 lines, ~4 hours

**This is focused, fundamental, and aligned with PAOLA's core vision!**
