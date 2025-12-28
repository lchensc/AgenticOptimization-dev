# Optimizer Architecture v2: Modular Multi-Objective Design

**Date**: December 27, 2025
**Status**: DRAFT - For Discussion
**Author**: Claude Code

---

## Executive Summary

The current optimizer architecture has grown organically and needs restructuring to:
1. Support **multi-objective optimization** (Pareto fronts, NSGA-II/III)
2. Add **evolutionary algorithms** (GA, DE, PSO) via pymoo
3. Improve **modularity** for easier extension
4. Reduce **code duplication** (1147 lines across 2 files)

This document proposes a clean separation of concerns with a plugin-based backend system.

---

## Current Architecture Analysis

### File Structure (Current)

```
paola/
├── optimizers/
│   └── backends.py          # 632 lines - ALL backend implementations
└── tools/
    └── optimizer.py         # 515 lines - Tool + graph integration
```

### Problems Identified

| Issue | Location | Lines | Impact |
|-------|----------|-------|--------|
| **Monolithic backends.py** | `backends.py` | 632 | Hard to add new optimizers |
| **Mixed responsibilities** | `optimizer.py` | 515 | Graph logic + optimization + result mapping |
| **Duplicated wrapper logic** | Both files | ~200 | Each backend reimplements history tracking |
| **Single-objective only** | `OptimizationResult` | - | No Pareto front support |
| **Hardcoded families** | `registry.py` | - | Family ↔ backend coupling is rigid |

### Current Data Flow

```
run_optimization()           # 363 lines of tool code
    │
    ├─ Get graph, problem    # 40 lines
    ├─ Parse optimizer spec  # 30 lines
    ├─ Determine x0          # 60 lines
    ├─ Create components     # 70 lines (gradient vs bayesian vs ...)
    │
    └─ backend.optimize()    # Call backend
        │
        ├─ Wrap objective    # Each backend does this
        ├─ Track history     # Each backend does this
        ├─ Run optimizer     # Backend-specific
        └─ Return result     # Each backend formats this
```

**Observation**: ~50% of `run_optimization` is component creation, which depends on optimizer family. This coupling makes adding new families tedious.

---

## Proposed Architecture v2

### Design Principles

1. **Single Responsibility**: Each module does one thing well
2. **Open/Closed**: Add new optimizers without modifying existing code
3. **Strategy Pattern**: Backends are interchangeable strategies
4. **Multi-Objective First**: Design for Pareto fronts, single-objective is a special case

### New File Structure

```
paola/
├── optimizers/
│   ├── __init__.py              # Public API
│   ├── base.py                  # Abstract base classes (50 lines)
│   ├── result.py                # Result types (single + multi-obj) (80 lines)
│   ├── wrapper.py               # Objective wrapper utilities (100 lines)
│   ├── registry.py              # Backend registry (60 lines)
│   │
│   ├── backends/                # One file per backend family
│   │   ├── __init__.py
│   │   ├── scipy_backend.py     # SciPy minimize (150 lines)
│   │   ├── ipopt_backend.py     # IPOPT (100 lines)
│   │   ├── optuna_backend.py    # Optuna (200 lines)
│   │   └── pymoo_backend.py     # pymoo (250 lines) ← NEW
│   │
│   └── families/                # Component schemas per family
│       ├── __init__.py
│       ├── gradient.py          # Gradient-based (SLSQP, IPOPT, etc.)
│       ├── bayesian.py          # Bayesian (Optuna TPE, GP)
│       ├── evolutionary.py      # GA, DE, PSO ← NEW
│       └── multiobjective.py    # NSGA-II/III, MOEA/D ← NEW
│
└── tools/
    └── optimizer.py             # Thin wrapper, ~150 lines
```

### Key Abstractions

#### 1. OptimizationResult (Enhanced)

```python
@dataclass
class OptimizationResult:
    """Result from any optimizer - single or multi-objective."""

    success: bool
    message: str

    # Single-objective (always present)
    best_x: np.ndarray
    best_f: float

    # Multi-objective (optional)
    pareto_front: Optional[np.ndarray] = None  # Shape: (n_solutions, n_objectives)
    pareto_set: Optional[np.ndarray] = None    # Shape: (n_solutions, n_vars)

    # Statistics
    n_evaluations: int = 0
    n_generations: int = 0  # For evolutionary
    history: List[Dict] = field(default_factory=list)

    # Raw result for advanced use
    raw_result: Any = None

    @property
    def is_multiobjective(self) -> bool:
        return self.pareto_front is not None
```

#### 2. OptimizerBackend (Base Class)

```python
class OptimizerBackend(ABC):
    """Abstract base for all optimizer backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g., 'scipy', 'pymoo')."""
        pass

    @property
    @abstractmethod
    def family(self) -> str:
        """Optimizer family for component selection."""
        pass

    @property
    def supports_multiobjective(self) -> bool:
        """Whether this backend supports multi-objective."""
        return False

    @property
    def supports_constraints(self) -> bool:
        """Whether this backend supports constraints."""
        return True

    @abstractmethod
    def is_available(self) -> bool:
        """Check if dependencies are installed."""
        pass

    @abstractmethod
    def list_methods(self) -> List[str]:
        """List available methods/algorithms."""
        pass

    @abstractmethod
    def optimize(
        self,
        problem: "OptimizationProblem",  # New unified problem type
        config: Dict[str, Any],
    ) -> OptimizationResult:
        """Run optimization to completion."""
        pass
```

#### 3. OptimizationProblem (Unified)

```python
@dataclass
class OptimizationProblem:
    """Unified problem definition for all backends."""

    # Required
    objectives: List[Callable[[np.ndarray], float]]  # List for multi-obj
    bounds: List[Tuple[float, float]]

    # Optional
    constraints: Optional[List[Constraint]] = None
    gradients: Optional[List[Callable]] = None
    x0: Optional[np.ndarray] = None

    # Metadata
    n_objectives: int = 1
    objective_names: Optional[List[str]] = None

    @property
    def dimension(self) -> int:
        return len(self.bounds)

    @property
    def is_multiobjective(self) -> bool:
        return self.n_objectives > 1
```

#### 4. ObjectiveWrapper (Shared Utility)

```python
class ObjectiveWrapper:
    """Wraps objective function(s) with tracking and caching."""

    def __init__(
        self,
        objectives: List[Callable],
        cache: Optional[EvaluationCache] = None,
    ):
        self.objectives = objectives
        self.cache = cache
        self.n_evals = 0
        self.history: List[Dict] = []

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate all objectives, return array of shape (n_objectives,)."""
        self.n_evals += 1

        # Check cache
        if self.cache:
            cached = self.cache.get(x)
            if cached is not None:
                return cached

        # Evaluate
        f = np.array([obj(x) for obj in self.objectives])

        # Record history
        self.history.append({
            "eval": self.n_evals,
            "x": x.tolist(),
            "f": f.tolist(),
        })

        # Cache result
        if self.cache:
            self.cache.store(x, f)

        return f

    def evaluate_single(self, x: np.ndarray) -> float:
        """Convenience for single-objective."""
        return float(self.evaluate(x)[0])
```

---

## pymoo Backend Design

### Why pymoo?

| Feature | pymoo | DEAP | scipy.DE | pyswarm |
|---------|-------|------|----------|---------|
| Multi-objective | NSGA-II/III, MOEA/D | Manual setup | No | No |
| Single-objective | GA, DE, PSO, CMA-ES | Yes | DE only | PSO only |
| Constraints | Native support | Manual | Limited | No |
| Parallelization | Built-in | Manual | No | No |
| Active development | Yes (2025) | Slow | Stable | No (2015) |
| Problem interface | Clean `Problem` class | Functional | Callable | Callable |

### pymoo Integration

```python
class PymooBackend(OptimizerBackend):
    """pymoo backend for evolutionary and multi-objective optimization."""

    # Single-objective algorithms
    SOO_ALGORITHMS = {
        "GA": "ga",
        "DE": "de",
        "PSO": "pso",
        "CMA-ES": "cmaes",
        "ES": "es",
        "BRKGA": "brkga",
        "NelderMead": "nelder-mead",
        "PatternSearch": "pattern-search",
    }

    # Multi-objective algorithms
    MOO_ALGORITHMS = {
        "NSGA-II": "nsga2",
        "NSGA-III": "nsga3",
        "MOEA/D": "moead",
        "R-NSGA-III": "rnsga3",
        "U-NSGA-III": "unsga3",
        "AGE-MOEA": "agemoea",
        "AGE-MOEA2": "agemoea2",
        "C-TAEA": "ctaea",
        "SMS-EMOA": "sms-emoa",
    }

    @property
    def name(self) -> str:
        return "pymoo"

    @property
    def family(self) -> str:
        return "evolutionary"  # Or "multiobjective" based on algorithm

    @property
    def supports_multiobjective(self) -> bool:
        return True

    def list_methods(self) -> List[str]:
        return list(self.SOO_ALGORITHMS.keys()) + list(self.MOO_ALGORITHMS.keys())

    def optimize(
        self,
        problem: OptimizationProblem,
        config: Dict[str, Any],
    ) -> OptimizationResult:
        from pymoo.optimize import minimize
        from pymoo.core.problem import ElementwiseProblem

        algorithm = config.get("algorithm", "NSGA-II" if problem.is_multiobjective else "GA")

        # Create pymoo Problem
        class WrappedProblem(ElementwiseProblem):
            def __init__(self, opt_problem):
                super().__init__(
                    n_var=opt_problem.dimension,
                    n_obj=opt_problem.n_objectives,
                    n_constr=len(opt_problem.constraints) if opt_problem.constraints else 0,
                    xl=np.array([b[0] for b in opt_problem.bounds]),
                    xu=np.array([b[1] for b in opt_problem.bounds]),
                )
                self.opt_problem = opt_problem
                self.wrapper = ObjectiveWrapper(opt_problem.objectives)

            def _evaluate(self, x, out, *args, **kwargs):
                out["F"] = self.wrapper.evaluate(x)
                if self.opt_problem.constraints:
                    out["G"] = np.array([c(x) for c in self.opt_problem.constraints])

        pymoo_problem = WrappedProblem(problem)

        # Get algorithm instance
        algo = self._create_algorithm(algorithm, config)

        # Run optimization
        res = minimize(
            pymoo_problem,
            algo,
            termination=("n_gen", config.get("n_generations", 100)),
            seed=config.get("seed"),
            verbose=config.get("verbose", False),
        )

        # Convert result
        if problem.is_multiobjective:
            return OptimizationResult(
                success=res.X is not None,
                message="Optimization complete",
                best_x=res.X[0] if res.X is not None else problem.x0,
                best_f=res.F[0, 0] if res.F is not None else float('inf'),
                pareto_set=res.X,
                pareto_front=res.F,
                n_evaluations=pymoo_problem.wrapper.n_evals,
                n_generations=res.algorithm.n_gen,
                history=pymoo_problem.wrapper.history,
                raw_result=res,
            )
        else:
            return OptimizationResult(
                success=res.success,
                message=str(res.message) if hasattr(res, 'message') else "Complete",
                best_x=res.X,
                best_f=float(res.F[0]) if res.F is not None else float('inf'),
                n_evaluations=pymoo_problem.wrapper.n_evals,
                n_generations=res.algorithm.n_gen,
                history=pymoo_problem.wrapper.history,
                raw_result=res,
            )

    def _create_algorithm(self, name: str, config: Dict) -> "Algorithm":
        """Create pymoo algorithm with config."""
        from pymoo.algorithms.soo.nonconvex import ga, de, pso, cmaes
        from pymoo.algorithms.moo import nsga2, nsga3, moead

        pop_size = config.get("pop_size", 100)

        if name == "GA":
            return ga.GA(pop_size=pop_size)
        elif name == "DE":
            return de.DE(pop_size=pop_size)
        elif name == "PSO":
            return pso.PSO(pop_size=pop_size)
        elif name == "CMA-ES":
            return cmaes.CMAES(x0=config.get("x0"))
        elif name == "NSGA-II":
            return nsga2.NSGA2(pop_size=pop_size)
        elif name == "NSGA-III":
            from pymoo.util.ref_dirs import get_reference_directions
            n_obj = config.get("n_objectives", 2)
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
            return nsga3.NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
        elif name == "MOEA/D":
            return moead.MOEAD(...)
        else:
            raise ValueError(f"Unknown algorithm: {name}")
```

---

## Refactored tools/optimizer.py

The tool becomes a thin wrapper:

```python
@tool
def run_optimization(
    graph_id: int,
    optimizer: str,
    config: Optional[str] = None,
    # ... other args
) -> Dict[str, Any]:
    """Run optimization, creating a new node in the graph."""

    # 1. Get graph and problem (unchanged)
    graph = _FOUNDRY.get_graph(graph_id)
    problem = _get_problem(graph.problem_id)

    # 2. Parse optimizer spec
    backend_name, method = parse_optimizer_spec(optimizer)
    backend = get_backend(backend_name)

    # 3. Build OptimizationProblem (unified)
    opt_problem = build_optimization_problem(problem, method)

    # 4. Start graph node
    node = graph.start_node(optimizer=optimizer, config=config_dict)

    # 5. Run optimization (single line!)
    result = backend.optimize(opt_problem, config_dict)

    # 6. Record to graph (using result's family)
    complete_node_from_result(graph, node, result, backend.family)

    # 7. Return response
    return format_response(result, node)
```

**Line count reduction**: 515 → ~150 lines (70% reduction)

---

## Component Family Mapping

| Backend | Algorithm | Family | Component Schema |
|---------|-----------|--------|------------------|
| scipy | SLSQP, L-BFGS-B, trust-constr | gradient | GradientProgress |
| scipy | Nelder-Mead, Powell, COBYLA | derivative_free | DerivativeFreeProgress |
| ipopt | IPOPT | gradient | GradientProgress |
| optuna | TPE, GP | bayesian | BayesianProgress |
| optuna | CMA-ES | cmaes | CMAESProgress |
| pymoo | GA, DE, PSO | evolutionary | EvolutionaryProgress |
| pymoo | NSGA-II/III | multiobjective | MultiObjectiveProgress |

### New Component Schemas

```python
@dataclass
class EvolutionaryProgress(ProgressComponent):
    """Progress for evolutionary algorithms (GA, DE, PSO)."""
    generations: List[GenerationRecord] = field(default_factory=list)

    @dataclass
    class GenerationRecord:
        generation: int
        best_fitness: float
        mean_fitness: float
        diversity: float  # Population diversity metric
        best_individual: List[float]

@dataclass
class MultiObjectiveProgress(ProgressComponent):
    """Progress for multi-objective (NSGA-II/III, MOEA/D)."""
    generations: List[MOGenerationRecord] = field(default_factory=list)

    @dataclass
    class MOGenerationRecord:
        generation: int
        n_nondominated: int  # Pareto front size
        hypervolume: Optional[float]  # HV indicator
        igd: Optional[float]  # Inverted Generational Distance
        pareto_front: List[List[float]]  # Current front
```

---

## Migration Plan

### Phase 1: Refactor Base (No New Features)
1. Create `optimizers/base.py` with new abstractions
2. Create `optimizers/result.py` with enhanced result type
3. Create `optimizers/wrapper.py` with ObjectiveWrapper
4. Create `optimizers/registry.py` with new registry

### Phase 2: Migrate Existing Backends
1. Move SciPy to `backends/scipy_backend.py`
2. Move IPOPT to `backends/ipopt_backend.py`
3. Move Optuna to `backends/optuna_backend.py`
4. Refactor `tools/optimizer.py` to use new structure

### Phase 3: Add pymoo Backend
1. Create `backends/pymoo_backend.py`
2. Add `families/evolutionary.py` components
3. Add `families/multiobjective.py` components
4. Update COMPONENT_REGISTRY

### Phase 4: Multi-Objective Tools
1. Add `create_mo_problem()` tool for multi-objective problems
2. Add `get_pareto_front()` tool to extract solutions
3. Add visualization support for Pareto fronts

---

## API Examples

### Single-Objective (Current - Unchanged)

```python
# Agent uses same interface
run_optimization(
    graph_id=1,
    optimizer="pymoo:GA",
    config='{"pop_size": 100, "n_generations": 50}'
)
```

### Multi-Objective (New)

```python
# Register multiple objectives
foundry_store_evaluator("drag", "Drag coefficient", "aero.py", "compute_drag")
foundry_store_evaluator("lift", "Lift coefficient", "aero.py", "compute_lift")

# Create multi-objective problem
create_mo_problem(
    name="Wing Design",
    objective_evaluator_ids=["drag", "lift"],  # Multiple objectives
    objective_senses=["minimize", "maximize"],  # Per-objective direction
    bounds=[[0, 1]] * 10,
)

# Run with MO algorithm
run_optimization(
    graph_id=1,
    optimizer="pymoo:NSGA-II",
    config='{"pop_size": 100, "n_generations": 100}'
)

# Get Pareto front
get_pareto_front(graph_id=1, node_id="n1")
# Returns: {
#   "pareto_set": [[0.2, 0.3, ...], [0.4, 0.1, ...], ...],
#   "pareto_front": [[0.05, 0.8], [0.07, 0.9], ...],
#   "n_solutions": 50,
#   "hypervolume": 0.85
# }
```

---

## Dependencies

```
# requirements.txt additions
pymoo>=0.6.0
```

pymoo includes:
- NumPy (already have)
- SciPy (already have)
- Matplotlib (optional, for visualization)

---

## Testing Strategy

1. **Unit tests per backend**: Each backend has isolated tests
2. **Integration tests**: Full optimization runs with Paola CLI
3. **Benchmark suite**: Compare algorithms on standard problems
   - Single-objective: Rosenbrock, Ackley, Rastrigin
   - Multi-objective: ZDT1-6, DTLZ1-7 (built into pymoo)

---

## Success Criteria

- [ ] All existing tests pass after migration
- [ ] backends.py split into 4+ files, each <250 lines
- [ ] tools/optimizer.py reduced to <200 lines
- [ ] pymoo backend working with GA, DE, PSO
- [ ] NSGA-II producing valid Pareto fronts
- [ ] 10-bar truss benchmark solvable with new backend

---

## Design Decisions (Confirmed)

1. **Pareto front storage**: Separate storage (GraphDetail tier) with reference in GraphRecord
   - GraphRecord stores: `pareto_ref: str` (reference ID), `n_pareto_solutions: int`, `hypervolume: float`
   - GraphDetail stores: Full `pareto_set` and `pareto_front` arrays
   - Rationale: Keeps GraphRecord compact for LLM context

2. **Decision making tools**: No - keep minimal
   - LLM can implement selection logic if needed
   - Avoid tool proliferation

3. **Migration approach**: Refactor first, then add pymoo
   - Phase 1-2: Restructure existing code
   - Phase 3: Add pymoo backend
   - Rationale: Cleaner foundation, easier debugging

4. **Algorithm defaults**: LLM decides via Skills system
   - No hardcoded defaults in backend
   - Build comprehensive Skills for pymoo algorithms
   - Rationale: Follows "LLM IS the Intelligence" principle

---

## Implementation Plan

### Phase 1: Base Abstractions (Day 1)

**Files to create:**

```
paola/optimizers/
├── __init__.py           # Public API exports
├── base.py               # OptimizerBackend ABC
├── result.py             # OptimizationResult (single + multi-obj)
├── wrapper.py            # ObjectiveWrapper utility
└── registry.py           # BackendRegistry class
```

**Tasks:**
- [ ] Create `OptimizationResult` with Pareto support
- [ ] Create `OptimizerBackend` ABC with `family` property
- [ ] Create `ObjectiveWrapper` with history tracking
- [ ] Create `BackendRegistry` with `register()` and `get()` methods
- [ ] Write unit tests for base classes

### Phase 2: Migrate Existing Backends (Day 2-3)

**Files to create:**

```
paola/optimizers/backends/
├── __init__.py
├── scipy_backend.py      # Extract from backends.py
├── ipopt_backend.py      # Extract from backends.py
└── optuna_backend.py     # Extract from backends.py
```

**Tasks:**
- [ ] Extract `SciPyBackend` → `scipy_backend.py`
- [ ] Extract `IPOPTBackend` → `ipopt_backend.py`
- [ ] Extract `OptunaBackend` → `optuna_backend.py`
- [ ] Refactor to use `ObjectiveWrapper`
- [ ] Register backends in `registry.py`
- [ ] Update `tools/optimizer.py` to use new structure
- [ ] Verify all existing tests pass

### Phase 3: Add pymoo Backend (Day 4-5)

**Files to create:**

```
paola/optimizers/backends/
└── pymoo_backend.py      # New backend

paola/foundry/schema/
├── evolutionary.py       # GA, DE, PSO components
└── multiobjective.py     # NSGA-II/III components

paola/skills/optimizers/pymoo/
├── skill.yaml            # Overview and capabilities
├── algorithms.yaml       # Algorithm-specific guidance
└── configurations.yaml   # Common configurations
```

**Tasks:**
- [ ] Create `PymooBackend` with SOO algorithms (GA, DE, PSO)
- [ ] Add MOO algorithms (NSGA-II, NSGA-III)
- [ ] Create `EvolutionaryProgress` component schema
- [ ] Create `MultiObjectiveProgress` component schema
- [ ] Update `COMPONENT_REGISTRY` with new families
- [ ] Create pymoo Skill with algorithm guidance
- [ ] Add pymoo to requirements.txt
- [ ] Write integration tests

### Phase 4: Refactor tools/optimizer.py (Day 6)

**Goal:** Reduce from 515 lines to ~150 lines

**Tasks:**
- [ ] Extract graph logic to helper functions
- [ ] Extract component creation to `families/` modules
- [ ] Simplify `run_optimization()` to orchestration only
- [ ] Add `create_mo_problem()` tool for multi-objective
- [ ] Update tool docstrings
- [ ] Verify all Paola CLI tests pass

---

## File-by-File Specifications

### `paola/optimizers/base.py` (~50 lines)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np

from .result import OptimizationResult


class OptimizerBackend(ABC):
    """Abstract base for optimizer backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g., 'scipy', 'pymoo')."""
        pass

    @property
    @abstractmethod
    def family(self) -> str:
        """Component family for this backend."""
        pass

    @property
    def supports_multiobjective(self) -> bool:
        return False

    @property
    def supports_constraints(self) -> bool:
        return True

    @property
    def supports_gradients(self) -> bool:
        return False

    @abstractmethod
    def is_available(self) -> bool:
        """Check if dependencies are installed."""
        pass

    @abstractmethod
    def get_methods(self) -> List[str]:
        """List available methods/algorithms."""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get backend capabilities for LLM."""
        pass

    @abstractmethod
    def optimize(
        self,
        objective: callable,
        bounds: List[List[float]],
        x0: np.ndarray,
        config: Dict[str, Any],
        constraints: Optional[List] = None,
        gradient: Optional[callable] = None,
    ) -> OptimizationResult:
        """Run optimization."""
        pass
```

### `paola/optimizers/result.py` (~80 lines)

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class OptimizationResult:
    """Result from any optimizer - single or multi-objective."""

    # Status
    success: bool
    message: str

    # Best solution (always present)
    best_x: np.ndarray
    best_f: float

    # Multi-objective (optional)
    pareto_set: Optional[np.ndarray] = None    # (n_solutions, n_vars)
    pareto_front: Optional[np.ndarray] = None  # (n_solutions, n_objectives)
    hypervolume: Optional[float] = None

    # Statistics
    n_iterations: int = 0
    n_function_evals: int = 0
    n_gradient_evals: int = 0

    # History
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Raw result for advanced use
    raw_result: Any = None

    @property
    def is_multiobjective(self) -> bool:
        return self.pareto_front is not None

    @property
    def n_pareto_solutions(self) -> int:
        if self.pareto_front is None:
            return 0
        return len(self.pareto_front)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for tool response."""
        result = {
            "success": self.success,
            "message": self.message,
            "best_x": self.best_x.tolist() if isinstance(self.best_x, np.ndarray) else list(self.best_x),
            "best_f": float(self.best_f),
            "n_iterations": self.n_iterations,
            "n_function_evals": self.n_function_evals,
            "n_gradient_evals": self.n_gradient_evals,
            "history": self.history[-20:] if len(self.history) > 20 else self.history,
        }

        if self.is_multiobjective:
            result["is_multiobjective"] = True
            result["n_pareto_solutions"] = self.n_pareto_solutions
            result["hypervolume"] = self.hypervolume
            # Don't include full Pareto set in response (stored separately)

        return result
```

### `paola/optimizers/wrapper.py` (~100 lines)

```python
from typing import Callable, List, Dict, Any, Optional
import numpy as np


class ObjectiveWrapper:
    """Wraps objective function with tracking and optional caching."""

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        cache: Optional["EvaluationCache"] = None,
    ):
        self.objective = objective
        self.cache = cache
        self.n_evals = 0
        self.history: List[Dict[str, Any]] = []
        self._best_f = float('inf')
        self._best_x = None

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate objective with tracking."""
        self.n_evals += 1

        # Check cache
        if self.cache:
            cached = self.cache.get(x)
            if cached is not None:
                return cached

        # Evaluate
        f = float(self.objective(x))

        # Track best
        if f < self._best_f:
            self._best_f = f
            self._best_x = x.copy()

        # Record history
        self.history.append({
            "iteration": self.n_evals,
            "objective": f,
            "design": x.tolist() if hasattr(x, 'tolist') else list(x),
        })

        # Cache
        if self.cache:
            self.cache.store(x, f)

        return f

    @property
    def best_x(self) -> Optional[np.ndarray]:
        return self._best_x

    @property
    def best_f(self) -> float:
        return self._best_f


class GradientWrapper:
    """Wraps gradient function with tracking."""

    def __init__(self, gradient: Callable[[np.ndarray], np.ndarray]):
        self.gradient = gradient
        self.n_evals = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.n_evals += 1
        return self.gradient(x)
```

---

## References

- [pymoo Documentation](https://pymoo.org/)
- [pymoo GitHub](https://github.com/anyoptimization/pymoo)
- [pymoo IEEE Paper](https://ieeexplore.ieee.org/document/9078759/)
- [NSGA-II Paper](https://ieeexplore.ieee.org/document/996017)
- Current Paola architecture: `docs/architecture/20251227_tool_refactoring_architecture.md`
