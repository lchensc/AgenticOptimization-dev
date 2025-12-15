# Foundry Polymorphic Schema Design

**Document ID**: 20251215_2100_foundry_polymorphic_schema_design
**Date**: December 15, 2025
**Status**: Design Proposal
**Purpose**: Design maintainable, extensible Foundry schemas for multiple optimizer types

---

## 1. Problem Statement

### Current State

The Foundry currently uses a single `RunRecord` class for all optimizer types. As we add more optimizers (SciPy, IPOPT, Optuna, CMA-ES, DE, NLopt, etc.), this approach becomes problematic.

### The Challenge

Different optimizer families have fundamentally different concepts:

| Family | Initialization | Progress | Result |
|--------|---------------|----------|--------|
| **Gradient-based** | Single point x0 | Iterations with gradient | Converged point |
| **Population-based** | Population of points | Generations | Pareto front or best |
| **Bayesian** | Prior trials (optional) | Trials with surrogate | Best trial |
| **CMA-ES** | Mean + sigma | Generations with covariance | Distribution params |

A single schema with optional fields for all these concepts:
- Becomes sparse (most fields are `None`)
- Is hard to validate (which fields should be present?)
- Grows unboundedly as new optimizers are added
- Provides poor developer experience (unclear contracts)

---

## 2. Design Principles

### 2.1 Separation of Concerns

**Common core**: Fields that ALL optimization runs share, regardless of algorithm
**Type-specific components**: Data structures specific to each optimizer family

### 2.2 Open for Extension, Closed for Modification

Adding a new optimizer family should:
- NOT require modifying the core `RunRecord` class
- Only require adding new component classes and registering them

### 2.3 Queryability

Common fields must be directly queryable across all runs:
- "Find all successful runs for problem X"
- "Find runs with final_objective < 0.01"
- "Compare wall_time across different optimizers"

### 2.4 Type Safety

Each optimizer family should have well-defined, validated schemas:
- A gradient-based run MUST have `x0`
- A population-based run MUST have `population`
- Validation happens at record creation, not query time

---

## 3. Proposed Architecture

### 3.1 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     RunRecord (Common Core)                      │
├─────────────────────────────────────────────────────────────────┤
│  Universally queryable fields:                                   │
│  - run_id, problem_id, optimizer, optimizer_family              │
│  - success, final_objective, final_design                       │
│  - n_evaluations, wall_time_seconds, created_at                 │
├─────────────────────────────────────────────────────────────────┤
│  Polymorphic components (vary by optimizer_family):             │
│  - initialization: InitializationComponent                      │
│  - progress: ProgressComponent                                  │
│  - result: ResultComponent                                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                Component Registry
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│   Gradient    │      │  Population   │      │   Bayesian    │
│    Family     │      │    Family     │      │    Family     │
├───────────────┤      ├───────────────┤      ├───────────────┤
│GradientInit   │      │PopulationInit │      │BayesianInit   │
│GradientProg   │      │PopulationProg │      │BayesianProg   │
│GradientResult │      │PopulationResult│     │BayesianResult │
└───────────────┘      └───────────────┘      └───────────────┘
```

### 3.2 Optimizer Families

| Family | Optimizers | Key Characteristics |
|--------|------------|---------------------|
| `gradient` | SciPy (SLSQP, L-BFGS-B, trust-constr), IPOPT, NLopt (LD_*) | Single starting point, iterations, gradients |
| `population` | DE, GA, NSGA-II, PSO | Population of solutions, generations |
| `bayesian` | Optuna (TPE), BO-GP | Trials, surrogate model, acquisition |
| `cmaes` | CMA-ES | Mean + covariance, special structure |
| `simplex` | Nelder-Mead, COBYLA | Simplex vertices, no gradients |

---

## 4. Schema Definitions

### 4.1 Common Core (All Runs)

```python
@dataclass
class RunRecord:
    """
    Universal run record with common queryable fields.

    All optimization runs share these fields regardless of algorithm.
    Type-specific data is stored in polymorphic components.
    """

    # === Identity ===
    run_id: int
    problem_id: str

    # === Optimizer Info ===
    optimizer: str              # Full spec: "scipy:SLSQP", "optuna:TPE"
    optimizer_family: str       # Family: "gradient", "population", "bayesian"

    # === Universal Metrics (queryable across all runs) ===
    success: bool
    final_objective: float
    final_design: List[float]   # Best design found
    n_evaluations: int          # Total function evaluations
    wall_time_seconds: float

    # === Metadata ===
    created_at: str             # ISO timestamp
    config: Dict[str, Any]      # User-provided configuration

    # === Polymorphic Components ===
    # Structure varies by optimizer_family, validated by schema registry
    initialization: 'InitializationComponent'
    progress: 'ProgressComponent'
    result: 'ResultComponent'
```

### 4.2 Component Base Classes

```python
@dataclass
class InitializationComponent(ABC):
    """Base class for initialization data."""

    # What was requested (for reproducibility)
    specification: Dict[str, Any]

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InitializationComponent':
        """Deserialize from dictionary."""
        pass


@dataclass
class ProgressComponent(ABC):
    """Base class for optimization progress data."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProgressComponent':
        pass


@dataclass
class ResultComponent(ABC):
    """Base class for detailed result data."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResultComponent':
        pass
```

### 4.3 Gradient Family Components

```python
@dataclass
class GradientInitialization(InitializationComponent):
    """Initialization for gradient-based optimizers."""

    specification: Dict[str, Any]  # {"type": "center"} or {"type": "warm_start", ...}
    x0: List[float]                # Actual starting point used

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "gradient",
            "specification": self.specification,
            "x0": self.x0
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GradientInitialization':
        return cls(
            specification=data["specification"],
            x0=data["x0"]
        )


@dataclass
class GradientIteration:
    """Single iteration record for gradient-based optimizer."""
    iteration: int
    objective: float
    design: List[float]
    gradient_norm: Optional[float] = None
    step_size: Optional[float] = None
    constraint_violation: Optional[float] = None


@dataclass
class GradientProgress(ProgressComponent):
    """Progress data for gradient-based optimizers."""

    iterations: List[GradientIteration]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "gradient",
            "iterations": [
                {
                    "iteration": it.iteration,
                    "objective": it.objective,
                    "design": it.design,
                    "gradient_norm": it.gradient_norm,
                    "step_size": it.step_size,
                    "constraint_violation": it.constraint_violation
                }
                for it in self.iterations
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GradientProgress':
        iterations = [
            GradientIteration(**it_data)
            for it_data in data["iterations"]
        ]
        return cls(iterations=iterations)


@dataclass
class GradientResult(ResultComponent):
    """Detailed result for gradient-based optimizers."""

    termination_reason: str      # "convergence", "max_iter", "failed"
    final_gradient_norm: Optional[float] = None
    final_constraint_violation: Optional[float] = None
    hessian_approximation: Optional[List[List[float]]] = None  # If available
    lagrange_multipliers: Optional[List[float]] = None         # For constrained

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "gradient",
            "termination_reason": self.termination_reason,
            "final_gradient_norm": self.final_gradient_norm,
            "final_constraint_violation": self.final_constraint_violation,
            "hessian_approximation": self.hessian_approximation,
            "lagrange_multipliers": self.lagrange_multipliers
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GradientResult':
        return cls(
            termination_reason=data["termination_reason"],
            final_gradient_norm=data.get("final_gradient_norm"),
            final_constraint_violation=data.get("final_constraint_violation"),
            hessian_approximation=data.get("hessian_approximation"),
            lagrange_multipliers=data.get("lagrange_multipliers")
        )
```

### 4.4 Population Family Components

```python
@dataclass
class PopulationInitialization(InitializationComponent):
    """Initialization for population-based optimizers."""

    specification: Dict[str, Any]  # {"type": "lhs", "size": 50}
    method: str                    # "lhs", "sobol", "random"
    population_size: int
    initial_population: List[List[float]]  # All initial individuals

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "population",
            "specification": self.specification,
            "method": self.method,
            "population_size": self.population_size,
            "initial_population": self.initial_population
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PopulationInitialization':
        return cls(
            specification=data["specification"],
            method=data["method"],
            population_size=data["population_size"],
            initial_population=data["initial_population"]
        )


@dataclass
class Generation:
    """Single generation record for population-based optimizer."""
    generation: int
    best_objective: float
    best_design: List[float]
    population_objectives: List[float]  # All objectives in generation
    diversity: Optional[float] = None   # Population diversity metric


@dataclass
class PopulationProgress(ProgressComponent):
    """Progress data for population-based optimizers."""

    generations: List[Generation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "population",
            "generations": [
                {
                    "generation": gen.generation,
                    "best_objective": gen.best_objective,
                    "best_design": gen.best_design,
                    "population_objectives": gen.population_objectives,
                    "diversity": gen.diversity
                }
                for gen in self.generations
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PopulationProgress':
        generations = [Generation(**gen_data) for gen_data in data["generations"]]
        return cls(generations=generations)


@dataclass
class PopulationResult(ResultComponent):
    """Detailed result for population-based optimizers."""

    termination_reason: str
    final_population: List[List[float]]      # Final population
    final_objectives: List[float]            # Objectives of final population
    pareto_front: Optional[List[List[float]]] = None  # For multi-objective

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "population",
            "termination_reason": self.termination_reason,
            "final_population": self.final_population,
            "final_objectives": self.final_objectives,
            "pareto_front": self.pareto_front
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PopulationResult':
        return cls(
            termination_reason=data["termination_reason"],
            final_population=data["final_population"],
            final_objectives=data["final_objectives"],
            pareto_front=data.get("pareto_front")
        )
```

### 4.5 Bayesian Family Components

```python
@dataclass
class BayesianInitialization(InitializationComponent):
    """Initialization for Bayesian optimizers."""

    specification: Dict[str, Any]  # {"type": "none"} or {"type": "warm_start", ...}
    warm_start_trials: Optional[List[Dict[str, Any]]] = None  # Prior trials if any
    n_initial_random: int = 10     # Random trials before surrogate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "bayesian",
            "specification": self.specification,
            "warm_start_trials": self.warm_start_trials,
            "n_initial_random": self.n_initial_random
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BayesianInitialization':
        return cls(
            specification=data["specification"],
            warm_start_trials=data.get("warm_start_trials"),
            n_initial_random=data.get("n_initial_random", 10)
        )


@dataclass
class Trial:
    """Single trial record for Bayesian optimizer."""
    trial_number: int
    design: List[float]
    objective: float
    state: str                    # "complete", "pruned", "failed"
    duration_seconds: Optional[float] = None


@dataclass
class BayesianProgress(ProgressComponent):
    """Progress data for Bayesian optimizers."""

    trials: List[Trial]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "bayesian",
            "trials": [
                {
                    "trial_number": t.trial_number,
                    "design": t.design,
                    "objective": t.objective,
                    "state": t.state,
                    "duration_seconds": t.duration_seconds
                }
                for t in self.trials
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BayesianProgress':
        trials = [Trial(**t_data) for t_data in data["trials"]]
        return cls(trials=trials)


@dataclass
class BayesianResult(ResultComponent):
    """Detailed result for Bayesian optimizers."""

    termination_reason: str
    best_trial_number: int
    n_complete_trials: int
    n_pruned_trials: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "bayesian",
            "termination_reason": self.termination_reason,
            "best_trial_number": self.best_trial_number,
            "n_complete_trials": self.n_complete_trials,
            "n_pruned_trials": self.n_pruned_trials
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BayesianResult':
        return cls(
            termination_reason=data["termination_reason"],
            best_trial_number=data["best_trial_number"],
            n_complete_trials=data["n_complete_trials"],
            n_pruned_trials=data["n_pruned_trials"]
        )
```

### 4.6 CMA-ES Family Components

```python
@dataclass
class CMAESInitialization(InitializationComponent):
    """Initialization for CMA-ES optimizer."""

    specification: Dict[str, Any]  # {"type": "center", "sigma": "auto"}
    mean: List[float]              # Initial mean
    sigma: float                   # Initial step size
    population_size: int           # Lambda

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "cmaes",
            "specification": self.specification,
            "mean": self.mean,
            "sigma": self.sigma,
            "population_size": self.population_size
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CMAESInitialization':
        return cls(
            specification=data["specification"],
            mean=data["mean"],
            sigma=data["sigma"],
            population_size=data["population_size"]
        )


@dataclass
class CMAESGeneration:
    """Single generation record for CMA-ES."""
    generation: int
    best_objective: float
    best_design: List[float]
    mean: List[float]              # Current mean
    sigma: float                   # Current step size
    condition_number: Optional[float] = None  # Covariance condition


@dataclass
class CMAESProgress(ProgressComponent):
    """Progress data for CMA-ES optimizer."""

    generations: List[CMAESGeneration]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "cmaes",
            "generations": [
                {
                    "generation": g.generation,
                    "best_objective": g.best_objective,
                    "best_design": g.best_design,
                    "mean": g.mean,
                    "sigma": g.sigma,
                    "condition_number": g.condition_number
                }
                for g in self.generations
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CMAESProgress':
        generations = [CMAESGeneration(**g_data) for g_data in data["generations"]]
        return cls(generations=generations)


@dataclass
class CMAESResult(ResultComponent):
    """Detailed result for CMA-ES optimizer."""

    termination_reason: str
    final_mean: List[float]
    final_sigma: float
    final_condition_number: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "cmaes",
            "termination_reason": self.termination_reason,
            "final_mean": self.final_mean,
            "final_sigma": self.final_sigma,
            "final_condition_number": self.final_condition_number
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CMAESResult':
        return cls(
            termination_reason=data["termination_reason"],
            final_mean=data["final_mean"],
            final_sigma=data["final_sigma"],
            final_condition_number=data.get("final_condition_number")
        )
```

---

## 5. Component Registry

The registry maps optimizer families to their component types and handles serialization/deserialization.

```python
from typing import Type, Dict, Tuple

@dataclass
class OptimizerFamilySchema:
    """Schema definition for an optimizer family."""
    family: str
    initialization_class: Type[InitializationComponent]
    progress_class: Type[ProgressComponent]
    result_class: Type[ResultComponent]


class ComponentRegistry:
    """
    Registry for optimizer family schemas.

    Handles:
    - Mapping optimizer names to families
    - Serialization/deserialization of components
    - Validation of run records
    """

    def __init__(self):
        self._families: Dict[str, OptimizerFamilySchema] = {}
        self._optimizer_to_family: Dict[str, str] = {}

    def register_family(
        self,
        family: str,
        initialization_class: Type[InitializationComponent],
        progress_class: Type[ProgressComponent],
        result_class: Type[ResultComponent],
        optimizers: List[str]
    ):
        """
        Register an optimizer family and its associated optimizers.

        Args:
            family: Family name (e.g., "gradient")
            initialization_class: Class for initialization component
            progress_class: Class for progress component
            result_class: Class for result component
            optimizers: List of optimizer names in this family
        """
        self._families[family] = OptimizerFamilySchema(
            family=family,
            initialization_class=initialization_class,
            progress_class=progress_class,
            result_class=result_class
        )
        for opt in optimizers:
            self._optimizer_to_family[opt] = family

    def get_family(self, optimizer: str) -> str:
        """Get family name for an optimizer."""
        # Handle "scipy:SLSQP" format
        base_optimizer = optimizer.split(":")[0]
        return self._optimizer_to_family.get(base_optimizer, "gradient")  # Default

    def get_schema(self, family: str) -> OptimizerFamilySchema:
        """Get schema for a family."""
        return self._families[family]

    def deserialize_components(
        self,
        family: str,
        init_data: Dict,
        progress_data: Dict,
        result_data: Dict
    ) -> Tuple[InitializationComponent, ProgressComponent, ResultComponent]:
        """Deserialize components from dictionaries."""
        schema = self._families[family]
        return (
            schema.initialization_class.from_dict(init_data),
            schema.progress_class.from_dict(progress_data),
            schema.result_class.from_dict(result_data)
        )


# Global registry instance
COMPONENT_REGISTRY = ComponentRegistry()

# Register built-in families
COMPONENT_REGISTRY.register_family(
    family="gradient",
    initialization_class=GradientInitialization,
    progress_class=GradientProgress,
    result_class=GradientResult,
    optimizers=["scipy", "ipopt", "nlopt"]  # NLopt LD_* methods
)

COMPONENT_REGISTRY.register_family(
    family="population",
    initialization_class=PopulationInitialization,
    progress_class=PopulationProgress,
    result_class=PopulationResult,
    optimizers=["de", "ga", "nsga2", "pso"]
)

COMPONENT_REGISTRY.register_family(
    family="bayesian",
    initialization_class=BayesianInitialization,
    progress_class=BayesianProgress,
    result_class=BayesianResult,
    optimizers=["optuna", "bo"]
)

COMPONENT_REGISTRY.register_family(
    family="cmaes",
    initialization_class=CMAESInitialization,
    progress_class=CMAESProgress,
    result_class=CMAESResult,
    optimizers=["cmaes"]
)
```

---

## 6. Storage Layer

### 6.1 Serialization

```python
def serialize_run_record(record: RunRecord) -> Dict[str, Any]:
    """Serialize RunRecord to dictionary for storage."""
    return {
        # Common fields
        "run_id": record.run_id,
        "problem_id": record.problem_id,
        "optimizer": record.optimizer,
        "optimizer_family": record.optimizer_family,
        "success": record.success,
        "final_objective": record.final_objective,
        "final_design": record.final_design,
        "n_evaluations": record.n_evaluations,
        "wall_time_seconds": record.wall_time_seconds,
        "created_at": record.created_at,
        "config": record.config,

        # Polymorphic components (serialized)
        "initialization": record.initialization.to_dict(),
        "progress": record.progress.to_dict(),
        "result": record.result.to_dict()
    }


def deserialize_run_record(data: Dict[str, Any]) -> RunRecord:
    """Deserialize RunRecord from dictionary."""
    family = data["optimizer_family"]

    init, progress, result = COMPONENT_REGISTRY.deserialize_components(
        family=family,
        init_data=data["initialization"],
        progress_data=data["progress"],
        result_data=data["result"]
    )

    return RunRecord(
        run_id=data["run_id"],
        problem_id=data["problem_id"],
        optimizer=data["optimizer"],
        optimizer_family=family,
        success=data["success"],
        final_objective=data["final_objective"],
        final_design=data["final_design"],
        n_evaluations=data["n_evaluations"],
        wall_time_seconds=data["wall_time_seconds"],
        created_at=data["created_at"],
        config=data["config"],
        initialization=init,
        progress=progress,
        result=result
    )
```

### 6.2 JSON Storage Format

```json
{
  "run_id": 42,
  "problem_id": "wing_v2",
  "optimizer": "scipy:SLSQP",
  "optimizer_family": "gradient",
  "success": true,
  "final_objective": 0.00123,
  "final_design": [0.1, 0.2, 0.3],
  "n_evaluations": 156,
  "wall_time_seconds": 12.5,
  "created_at": "2025-12-15T21:00:00",
  "config": {"ftol": 1e-6},

  "initialization": {
    "family": "gradient",
    "specification": {"type": "center"},
    "x0": [0.5, 0.5, 0.5]
  },

  "progress": {
    "family": "gradient",
    "iterations": [
      {"iteration": 1, "objective": 1.5, "design": [...], "gradient_norm": 0.8},
      {"iteration": 2, "objective": 0.9, "design": [...], "gradient_norm": 0.3},
      ...
    ]
  },

  "result": {
    "family": "gradient",
    "termination_reason": "convergence",
    "final_gradient_norm": 1e-7,
    "final_constraint_violation": 0.0
  }
}
```

---

## 7. Querying

### 7.1 Common Field Queries (Work Across All Families)

```python
class Foundry:
    def query_runs(
        self,
        problem_id: Optional[str] = None,
        optimizer: Optional[str] = None,
        optimizer_family: Optional[str] = None,
        success: Optional[bool] = None,
        min_objective: Optional[float] = None,
        max_objective: Optional[float] = None
    ) -> List[RunRecord]:
        """
        Query runs by common fields.

        Works across all optimizer families.
        """
        # Filter by common fields (always present)
        ...

    def get_best_run(self, problem_id: str) -> Optional[RunRecord]:
        """Get the most successful run for a problem."""
        runs = self.query_runs(problem_id=problem_id, success=True)
        if not runs:
            return None
        return min(runs, key=lambda r: r.final_objective)
```

### 7.2 Family-Specific Queries

```python
class Foundry:
    def get_gradient_runs(self, problem_id: str) -> List[RunRecord]:
        """Get gradient-based runs (with typed components)."""
        return self.query_runs(
            problem_id=problem_id,
            optimizer_family="gradient"
        )

    def get_convergence_history(self, run_id: int) -> List[Tuple[int, float]]:
        """
        Get iteration/objective history for a gradient run.

        Returns list of (iteration, objective) tuples.
        """
        run = self.get_run(run_id)
        if run.optimizer_family != "gradient":
            raise ValueError(f"Run {run_id} is not gradient-based")

        progress: GradientProgress = run.progress
        return [(it.iteration, it.objective) for it in progress.iterations]
```

---

## 8. Adding New Optimizer Families

To add a new optimizer family (e.g., "surrogate" for surrogate-based optimization):

### Step 1: Define Components

```python
@dataclass
class SurrogateInitialization(InitializationComponent):
    specification: Dict[str, Any]
    initial_samples: List[List[float]]
    surrogate_type: str  # "gaussian_process", "neural_network"
    # ... surrogate-specific fields

@dataclass
class SurrogateProgress(ProgressComponent):
    # ... surrogate-specific progress

@dataclass
class SurrogateResult(ResultComponent):
    # ... surrogate-specific result
```

### Step 2: Register with Registry

```python
COMPONENT_REGISTRY.register_family(
    family="surrogate",
    initialization_class=SurrogateInitialization,
    progress_class=SurrogateProgress,
    result_class=SurrogateResult,
    optimizers=["ego", "sbo"]
)
```

**No changes to RunRecord or storage layer required.**

---

## 9. Migration Path

### From Current Schema

Current runs stored without family-specific components can be migrated:

```python
def migrate_legacy_run(legacy_data: Dict) -> RunRecord:
    """Migrate legacy run record to new schema."""

    # Infer family from optimizer
    optimizer = legacy_data.get("optimizer", "scipy:SLSQP")
    family = COMPONENT_REGISTRY.get_family(optimizer)

    # Create appropriate components from legacy data
    if family == "gradient":
        init = GradientInitialization(
            specification={"type": "unknown"},  # Legacy didn't track
            x0=legacy_data.get("x0", legacy_data.get("initial_design", []))
        )
        progress = GradientProgress(
            iterations=[
                GradientIteration(
                    iteration=it.get("iteration", i),
                    objective=it["objective"],
                    design=it.get("design", [])
                )
                for i, it in enumerate(legacy_data.get("iterations", []))
            ]
        )
        result = GradientResult(
            termination_reason=legacy_data.get("termination", "unknown")
        )
    # ... handle other families

    return RunRecord(
        run_id=legacy_data["run_id"],
        problem_id=legacy_data["problem_id"],
        optimizer=optimizer,
        optimizer_family=family,
        success=legacy_data.get("success", False),
        final_objective=legacy_data.get("final_objective", float("inf")),
        final_design=legacy_data.get("final_design", []),
        n_evaluations=legacy_data.get("n_evaluations", 0),
        wall_time_seconds=legacy_data.get("wall_time", 0.0),
        created_at=legacy_data.get("created_at", ""),
        config=legacy_data.get("config", {}),
        initialization=init,
        progress=progress,
        result=result
    )
```

---

## 10. Summary

### Benefits of This Design

| Aspect | Benefit |
|--------|---------|
| **Extensibility** | Add new optimizer family without modifying core |
| **Type Safety** | Each family has well-defined, validated schema |
| **Queryability** | Common fields always queryable across all runs |
| **Maintainability** | Clear separation of concerns |
| **Storage Efficiency** | No sparse optional fields |

### Trade-offs

| Trade-off | Mitigation |
|-----------|------------|
| More complex serialization | Registry handles it centrally |
| Need to know family for type-specific queries | `optimizer_family` field makes it explicit |
| More classes to maintain | Clear organization, one file per family |

### Files to Create/Modify

```
paola/foundry/
├── schema/
│   ├── __init__.py
│   ├── base.py              # RunRecord, component ABCs
│   ├── gradient.py          # Gradient family components
│   ├── population.py        # Population family components
│   ├── bayesian.py          # Bayesian family components
│   ├── cmaes.py             # CMA-ES family components
│   └── registry.py          # ComponentRegistry
├── storage/
│   ├── serialization.py     # Serialize/deserialize logic
│   └── ...
└── foundry.py               # Main Foundry class (update queries)
```

---

## 11. Next Steps

1. **Review this design** - Discuss any concerns or modifications
2. **Implement base schema** - `RunRecord`, component ABCs, registry
3. **Implement gradient family** - Most commonly used, proves the pattern
4. **Migrate existing runs** - Update current data to new schema
5. **Implement other families** - As needed for each optimizer

---

**Document Status**: Ready for Review
**Next Action**: Discussion and approval before implementation
