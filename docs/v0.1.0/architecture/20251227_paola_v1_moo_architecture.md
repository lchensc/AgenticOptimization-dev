# Paola v1.0: Multi-Objective Optimization Architecture

**Date**: 2025-12-27
**Status**: Ready for Implementation
**Vision**: Paola specializes in **Multi-Objective** and **Multi-Disciplinary Optimization** with agent-centric design

> This document defines Paola v1.0 architecture redesign. Key contribution: unified problem class + graph-based MOO paradigm.

---

## Part 1: Problem Class Hierarchy Discussion

### Current State

```
OptimizationProblem (ABC)
    └── NLPProblem (continuous only, single objective)
```

- `problem_family`: "continuous" | "discrete" | "mixed" (exists but unused)
- `problem_type`: "NLP" only (LP, QP, MILP, MINLP planned but not implemented)
- **No integer variable support**
- **No multi-objective support**

### Research: How Other Frameworks Do It

| Framework | Problem Hierarchy | Variable Types | SOO/MOO |
|-----------|------------------|----------------|---------|
| **pymoo** | Unified `Problem` class | `vars={"x": Real(), "y": Integer()}` | `n_obj=1` or `n_obj>1` |
| **GEKKO** | Single `Model` class | `integer=True` flag per variable | Weighted sum workaround |
| **Platypus** | `Problem(nvars, nobjs)` | `types=[Real(), Integer()]` | Native MOO |
| **scipy** | Functional (no class) | Continuous only | SOO only |

**Key Insight**: Modern frameworks use a **unified problem class** with:
1. Variable type specification per variable
2. `n_objectives` parameter (not separate SOO/MOO classes)

### Question 1: Should Paola Have Separate NLP/MINLP Classes?

**Option A: Keep Separate Classes** (Current approach)
```
OptimizationProblem
├── NLPProblem (continuous)
├── MINLPProblem (mixed-integer)
├── MOOProblem (multi-objective)
└── MOMINLPProblem (multi-objective mixed-integer)
```

**Cons**:
- Class explosion: 2 × 2 = 4 combinations minimum
- Code duplication
- Complex type routing

**Option B: Unified Problem Class** (Recommended)
```
OptimizationProblem  # Single class handles ALL problem types
    - variables: List[Variable]  # Each has type, bounds
    - objectives: List[Objective]  # n_objectives >= 1
    - constraints: List[Constraint]
```

**Pros**:
- **pymoo-inspired**: Proven pattern
- Single code path
- Agent doesn't need to know class names
- Extensible (add constraint types, not new classes)

---

## Part 2: Unified Problem Architecture (Proposed)

### Design Principle: pymoo + Agent-Centric

Following pymoo's design but enhanced for agentic use:

```python
@dataclass
class Variable:
    """Variable specification - continuous, integer, or binary."""
    name: str
    type: Literal["continuous", "integer", "binary"]
    lower: float
    upper: float
    # Optional metadata for agent reasoning
    description: Optional[str] = None
    unit: Optional[str] = None

@dataclass
class Objective:
    """Objective function specification."""
    name: str
    evaluator_id: str
    sense: Literal["minimize", "maximize"] = "minimize"
    description: Optional[str] = None

@dataclass
class Constraint:
    """Constraint specification (unified inequality/equality)."""
    name: str
    evaluator_id: str
    type: Literal["<=", ">=", "=="]
    bound: float
    description: Optional[str] = None

@dataclass
class OptimizationProblem:
    """Unified optimization problem - handles SOO/MOO, NLP/MINLP."""

    # Identity
    problem_id: int
    name: str

    # Core specification
    variables: List[Variable]
    objectives: List[Objective]
    constraints: List[Constraint] = field(default_factory=list)

    # Derived properties
    @property
    def n_variables(self) -> int:
        return len(self.variables)

    @property
    def n_objectives(self) -> int:
        return len(self.objectives)

    @property
    def n_constraints(self) -> int:
        return len(self.constraints)

    @property
    def is_multiobjective(self) -> bool:
        return self.n_objectives > 1

    @property
    def has_integers(self) -> bool:
        return any(v.type in ("integer", "binary") for v in self.variables)

    @property
    def problem_class(self) -> str:
        """Auto-detect problem class for solver selection."""
        has_int = self.has_integers
        is_moo = self.is_multiobjective

        if is_moo and has_int:
            return "MO-MINLP"
        elif is_moo:
            return "MOO"
        elif has_int:
            return "MINLP"
        else:
            return "NLP"

    # Bounds extraction for backends
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower_bounds, upper_bounds) arrays."""
        lb = np.array([v.lower for v in self.variables])
        ub = np.array([v.upper for v in self.variables])
        return lb, ub

    def get_integer_mask(self) -> np.ndarray:
        """Return boolean mask for integer/binary variables."""
        return np.array([v.type in ("integer", "binary") for v in self.variables])
```

### Why This Works

1. **Single class** handles NLP, MINLP, MOO, MO-MINLP
2. **Variable types** specified per variable (like pymoo's `vars` dict)
3. **Multiple objectives** as list (like pymoo's `n_obj`)
4. **Auto-detection** via properties for solver routing
5. **Agent-friendly**: No class selection needed

---

## Part 3: Graph-Centric Data Flow (Key Innovation)

### Why Graph-Based Architecture Matters

The graph model is **Paola's key contribution** enabling intelligent MOO:

```
OptimizationGraph
├── Node n1: Global search (TPE) → finds promising regions
├── Node n2: Local refinement (SLSQP) from n1 → best_x
├── Node n3: MOO exploration (NSGA-II) → Pareto front
└── Edges: warm_start, refine, branch, explore
```

**Benefits for Agent**:
1. **Lineage tracking**: Agent knows what was tried and what worked
2. **Warm-starting**: Continue from previous solutions
3. **Strategy exploration**: Branch to try different approaches
4. **Learning from history**: Cross-graph knowledge transfer

### Current Tool Design (Keep and Enhance)

The current design philosophy is correct: **separate problem definition from optimization execution**.

```python
# Step 1: Register evaluators (once)
register_evaluator(name="drag", source={...})
register_evaluator(name="weight", source={...})

# Step 2: Define problem (once)
create_problem(
    name="Wing Design",
    variables=[...],
    objectives=[...],
    constraints=[...],
)

# Step 3: Create graph for optimization campaign
start_graph(problem_id=1)

# Step 4: Run optimization nodes (multiple, with strategy)
run_optimization(graph_id=1, optimizer="optuna:TPE")  # Global
run_optimization(graph_id=1, optimizer="scipy:SLSQP", parent_node="n1")  # Local
run_optimization(graph_id=1, optimizer="pymoo:NSGA-II")  # MOO exploration
```

### Enhanced Agent-Centric Improvements

Keep current structure but improve **agent experience**:

1. **Auto-detection of problem class** when starting graph
2. **Solver recommendations** based on problem class
3. **Smart warm-start suggestions** after each node
4. **Unified result format** across all backends

```python
# Agent receives guidance at each step
result = run_optimization(graph_id=1, optimizer="optuna:TPE")
# Returns:
{
    "success": True,
    "node_id": "n1",
    "best_x": [...],
    "best_f": 0.123,
    # NEW: Agent guidance
    "problem_class": "MOO",  # Auto-detected
    "suggested_next": [
        "Local refinement with scipy:SLSQP (warm_start from n1)",
        "Pareto exploration with pymoo:NSGA-II",
    ],
}
```

---

## Part 4: Implementation Plan (Clean, No Backward Compatibility)

Since backward compatibility is not required, we can redesign cleanly while keeping the core graph architecture.

### User Decisions Summary

| Decision | Choice |
|----------|--------|
| Variable format | **Explicit only** - always type/lower/upper |
| Graph tracking | **Keep graphs** - key innovation for agent MOO |
| MOO evaluators | **Both supported** - single `[f1,f2]` OR separate evaluators |
| Tool structure | **Keep current** - problem definition separate from run_optimization |

### Files to Create/Replace

| Action | File | Description |
|--------|------|-------------|
| **REPLACE** | `paola/foundry/problem.py` | New unified `OptimizationProblem` |
| **DELETE** | `paola/foundry/nlp_schema.py` | Replaced by unified problem |
| **UPDATE** | `paola/foundry/evaluator.py` | Support array returns for MOO |
| **UPDATE** | `paola/tools/optimizer.py` | Add MOO routing, keep structure |
| **UPDATE** | `paola/tools/problem.py` | New `create_problem()` with unified schema |
| **UPDATE** | `paola/foundry/__init__.py` | Export new classes |
| **KEEP** | `paola/foundry/schema/graph.py` | Graph architecture unchanged |

### Phase 1: Core Problem Schema

**New `paola/foundry/problem.py`**:

```python
"""
Unified Optimization Problem Schema.

Supports: NLP, MINLP, MOO, MO-MINLP
Inspired by pymoo's unified Problem class.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class Variable:
    """Decision variable specification."""
    name: str
    type: Literal["continuous", "integer", "binary"]
    lower: float
    upper: float
    description: Optional[str] = None

    def __post_init__(self):
        if self.type == "binary":
            self.lower, self.upper = 0.0, 1.0


@dataclass
class Objective:
    """
    Objective function specification.

    Supports TWO patterns for MOO:

    Pattern A - Separate evaluators (each objective has own evaluator):
        Objective(name="drag", evaluator_id="cfd_drag")
        Objective(name="weight", evaluator_id="structural_weight")

    Pattern B - Single evaluator returning array [f1, f2, ...]:
        Objective(name="f1", evaluator_id="zdt1", index=0)
        Objective(name="f2", evaluator_id="zdt1", index=1)
    """
    name: str
    evaluator_id: str
    sense: Literal["minimize", "maximize"] = "minimize"
    index: Optional[int] = None  # For array-returning evaluators (Pattern B)
    description: Optional[str] = None


@dataclass
class Constraint:
    """Constraint specification."""
    name: str
    evaluator_id: str
    type: Literal["<=", ">=", "=="]
    bound: float
    description: Optional[str] = None


@dataclass
class OptimizationProblem:
    """
    Unified optimization problem.

    Handles all problem types through composition:
    - NLP: continuous variables, single objective
    - MINLP: mixed variables, single objective
    - MOO: continuous variables, multiple objectives
    - MO-MINLP: mixed variables, multiple objectives
    """

    problem_id: int
    name: str
    variables: List[Variable]
    objectives: List[Objective]
    constraints: List[Constraint] = field(default_factory=list)

    # Metadata
    description: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Lineage
    parent_problem_id: Optional[int] = None
    derivation_notes: Optional[str] = None

    @property
    def n_variables(self) -> int:
        return len(self.variables)

    @property
    def n_objectives(self) -> int:
        return len(self.objectives)

    @property
    def n_constraints(self) -> int:
        return len(self.constraints)

    @property
    def is_multiobjective(self) -> bool:
        return self.n_objectives > 1

    @property
    def has_integers(self) -> bool:
        return any(v.type in ("integer", "binary") for v in self.variables)

    @property
    def problem_class(self) -> str:
        """Auto-detect problem class."""
        moo = self.is_multiobjective
        mint = self.has_integers
        if moo and mint:
            return "MO-MINLP"
        elif moo:
            return "MOO"
        elif mint:
            return "MINLP"
        return "NLP"

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.array([v.lower for v in self.variables])
        ub = np.array([v.upper for v in self.variables])
        return lb, ub

    def get_integer_indices(self) -> List[int]:
        return [i for i, v in enumerate(self.variables)
                if v.type in ("integer", "binary")]

    def get_objective_senses(self) -> List[str]:
        return [o.sense for o in self.objectives]
```

### Phase 2: Evaluator Array Support

**Update `paola/foundry/evaluator.py`** `_parse_result()`:

```python
def _parse_result(self, raw_result, execution_time: float) -> EvaluationResult:
    # Auto-detect format
    if isinstance(raw_result, (list, np.ndarray)):
        # Multi-objective: [f1, f2, ...]
        arr = np.atleast_1d(raw_result)
        objectives = {f"f{i}": float(v) for i, v in enumerate(arr)}
        constraints = {}
    elif isinstance(raw_result, dict):
        objectives = {k: v for k, v in raw_result.items() if not k.startswith("g")}
        constraints = {k: v for k, v in raw_result.items() if k.startswith("g")}
    elif isinstance(raw_result, (int, float, np.number)):
        objectives = {"f0": float(raw_result)}
        constraints = {}
    else:
        raise ValueError(f"Unsupported result type: {type(raw_result)}")

    return EvaluationResult(objectives=objectives, constraints=constraints, ...)
```

### Phase 3: Update create_problem Tool

**Update `paola/tools/problem.py`** with new unified schema:

```python
@tool
def create_problem(
    name: str,
    variables: List[Dict[str, Any]],
    objectives: List[Dict[str, Any]],
    constraints: Optional[List[Dict[str, Any]]] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create optimization problem with unified schema.

    Args:
        name: Problem name
        variables: List of variable specs (explicit format required):
            {"name": "x", "type": "continuous|integer|binary", "lower": 0, "upper": 10}
        objectives: List of objective specs:
            Pattern A (separate evaluators):
                {"name": "drag", "evaluator_id": "cfd_drag", "sense": "minimize"}
            Pattern B (single array evaluator):
                {"name": "f1", "evaluator_id": "zdt1", "index": 0}
        constraints: Optional list of constraint specs:
            {"name": "lift", "evaluator_id": "lift_eval", "type": ">=", "bound": 1000}

    Returns:
        problem_id, problem_class (auto-detected), n_variables, n_objectives

    Examples:
        # SOO with continuous variables (NLP)
        create_problem(
            name="Minimize drag",
            variables=[{"name": "chord", "type": "continuous", "lower": 0.5, "upper": 2}],
            objectives=[{"name": "drag", "evaluator_id": "cfd_drag"}],
        )

        # MOO with mixed variables (MO-MINLP)
        create_problem(
            name="Wing design",
            variables=[
                {"name": "twist", "type": "continuous", "lower": -5, "upper": 5},
                {"name": "n_ribs", "type": "integer", "lower": 3, "upper": 10},
            ],
            objectives=[
                {"name": "drag", "evaluator_id": "cfd_drag"},
                {"name": "weight", "evaluator_id": "fem_weight"},
            ],
        )

        # MOO with single array evaluator
        create_problem(
            name="ZDT1",
            variables=[{"name": f"x{i}", "type": "continuous", "lower": 0, "upper": 1}
                       for i in range(30)],
            objectives=[
                {"name": "f1", "evaluator_id": "zdt1", "index": 0},
                {"name": "f2", "evaluator_id": "zdt1", "index": 1},
            ],
        )
    """
```

### Phase 4: Update run_optimization Tool

**Update `paola/tools/optimizer.py`** - keep structure, add MOO routing:

```python
@tool
def run_optimization(
    graph_id: int,
    optimizer: str,
    config: Optional[str] = None,
    max_iterations: int = 100,
    parent_node: Optional[str] = None,
    edge_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run optimization node in graph (existing structure, enhanced).

    Now supports:
    - NLP: scipy:SLSQP, scipy:L-BFGS-B, ipopt
    - MINLP: pymoo:GA, pymoo:DE (with integer handling)
    - MOO: pymoo:NSGA-II, pymoo:NSGA-III, pymoo:MOEA/D
    - MO-MINLP: pymoo:NSGA-II (with integer handling)

    Returns (enhanced with agent guidance):
        success, node_id, best_x, best_f
        problem_class: Auto-detected type
        pareto_front: For MOO problems
        suggested_next: Agent guidance for next optimization step
    """
    # ... existing graph/node logic ...

    # NEW: Detect MOO from problem
    problem = _get_problem_from_graph(graph_id)
    is_moo = problem.is_multiobjective

    # NEW: Validate algorithm/problem compatibility
    _validate_algorithm_problem_match(optimizer, problem.problem_class)

    if is_moo:
        # Multi-objective path
        def objective(x):
            return problem.evaluate_objectives(x)  # Returns array
        config_dict["n_objectives"] = problem.n_objectives
        family = "multiobjective"
    else:
        # Single-objective path
        def objective(x):
            return float(problem.evaluate(x))

    # ... existing backend call ...

    # NEW: Add agent suggestions to response
    return {
        **existing_response,
        "problem_class": problem.problem_class,
        "suggested_next": _generate_suggestions(problem, result),
    }
```

### Phase 5: Solver Recommendation Logic

```python
# Algorithm compatibility matrix
ALGORITHM_CAPABILITIES = {
    # SOO algorithms
    "scipy:SLSQP": {"soo": True, "moo": False, "integer": False},
    "scipy:L-BFGS-B": {"soo": True, "moo": False, "integer": False},
    "ipopt": {"soo": True, "moo": False, "integer": False},
    "optuna:TPE": {"soo": True, "moo": False, "integer": True},

    # Evolutionary (SOO with integer support)
    "pymoo:GA": {"soo": True, "moo": False, "integer": True},
    "pymoo:DE": {"soo": True, "moo": False, "integer": True},
    "pymoo:PSO": {"soo": True, "moo": False, "integer": False},

    # MOO algorithms
    "pymoo:NSGA-II": {"soo": False, "moo": True, "integer": True},
    "pymoo:NSGA-III": {"soo": False, "moo": True, "integer": True},
    "pymoo:MOEA/D": {"soo": False, "moo": True, "integer": True},
}

def _validate_algorithm_problem_match(algorithm: str, problem_class: str):
    """Validate algorithm is compatible with problem class."""
    caps = ALGORITHM_CAPABILITIES.get(algorithm, {})

    is_moo = problem_class in ("MOO", "MO-MINLP")
    has_int = problem_class in ("MINLP", "MO-MINLP")

    if is_moo and not caps.get("moo"):
        raise ValueError(f"{algorithm} does not support multi-objective. Use pymoo:NSGA-II")
    if not is_moo and not caps.get("soo"):
        raise ValueError(f"{algorithm} requires multi-objective problem")
    if has_int and not caps.get("integer"):
        raise ValueError(f"{algorithm} does not support integer variables. Use pymoo:GA")
```

---

## Part 5: Summary

### Finalized Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Problem hierarchy** | Single unified `OptimizationProblem` | Follows pymoo, avoids class explosion |
| **Variable format** | Explicit only | Always require `type`, `lower`, `upper` |
| **MOO evaluators** | Both patterns supported | Separate evaluators OR single array evaluator |
| **Graph architecture** | **Keep** - key innovation | Enables agent strategy, warm-start, lineage |
| **Tool structure** | Keep current philosophy | Separate problem definition from run_optimization |
| **Problem class** | Auto-detected from spec | NLP, MINLP, MOO, MO-MINLP |
| **Backward compat** | Not required | Clean slate implementation |

### Files to Modify

| File | Action | Description |
|------|--------|-------------|
| `paola/foundry/problem.py` | **NEW** | Unified `OptimizationProblem`, `Variable`, `Objective`, `Constraint` |
| `paola/foundry/nlp_schema.py` | **DELETE** | Replaced by unified problem.py |
| `paola/foundry/evaluator.py` | **UPDATE** | Support array returns in `_parse_result()` |
| `paola/tools/problem.py` | **UPDATE** | New `create_problem()` with unified schema |
| `paola/tools/optimizer.py` | **UPDATE** | Add MOO routing, algorithm validation, suggestions |
| `paola/foundry/__init__.py` | **UPDATE** | Export new classes |
| `paola/foundry/schema/graph.py` | **KEEP** | Graph architecture unchanged |

### Problem Class Auto-Detection

| Variables | Objectives | → Problem Class |
|-----------|------------|-----------------|
| All continuous | 1 | **NLP** |
| Has integer/binary | 1 | **MINLP** |
| All continuous | >1 | **MOO** |
| Has integer/binary | >1 | **MO-MINLP** |

### Algorithm Compatibility

| Algorithm | SOO | MOO | Integer |
|-----------|-----|-----|---------|
| scipy:SLSQP | ✓ | ✗ | ✗ |
| scipy:L-BFGS-B | ✓ | ✗ | ✗ |
| ipopt | ✓ | ✗ | ✗ |
| optuna:TPE | ✓ | ✗ | ✓ |
| pymoo:GA | ✓ | ✗ | ✓ |
| pymoo:DE | ✓ | ✗ | ✓ |
| pymoo:NSGA-II | ✗ | ✓ | ✓ |
| pymoo:NSGA-III | ✗ | ✓ | ✓ |

### Benefits

1. **Unified**: One problem class handles NLP, MINLP, MOO, MO-MINLP
2. **Graph-centric**: Key innovation for agent MOO strategies
3. **Flexible MOO**: Both separate and array evaluators supported
4. **pymoo-aligned**: Uses proven patterns
5. **Agent-friendly**: Auto-detection, suggestions, validation
6. **Clean**: No backward compatibility baggage
