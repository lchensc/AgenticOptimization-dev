# Foundry Polymorphic Schema Design

**Document ID**: 20251215_2100_foundry_polymorphic_schema_design
**Date**: December 15, 2025
**Status**: Design Proposal (Revised)
**Version**: 2.1 - Session/Run terminology clarification
**Purpose**: Design maintainable, extensible Foundry schemas for Paola v0.2.0

---

## 1. Terminology: Session vs Run

### 1.1 The Problem

The term "run" was overloaded:
- `run_optimization()` tool creates a single optimizer execution
- But Paola's vision involves multiple optimizer executions for one task

### 1.2 Solution: Clear Terminology

| Term | Definition | API |
|------|------------|-----|
| **Session** | Complete optimization task (may involve multiple optimizers) | `start_session()`, `finalize_session()` |
| **Run** | Single optimizer execution | `run_optimization()` (existing) |

### 1.3 The Paola Vision

Paola is an autonomous optimization agent that may use **multiple optimizers** from **different families** to solve a single optimization task (session).

**Traditional model:**
```
One session = One optimizer = One run
```

**Paola model:**
```
Session = Complete optimization task (orchestrated by Paola)
    └── Run 1: First optimizer
    └── Run 2: Second optimizer (warm-start from Run 1)
    └── Run 3: Third optimizer (warm-start from Run 2)
```

### 1.4 Example: Multi-Run Session

```
Session #42: "Optimize wing drag"
│
├── Run 1: Global exploration (Optuna TPE)
│   ├── Family: bayesian
│   ├── 50 trials, best obj = 0.15
│   └── Paola decides: "Found promising region, switch to gradient"
│
├── Run 2: Local refinement (SLSQP)
│   ├── Family: gradient
│   ├── Warm-start from Run 1 best
│   ├── 30 iterations, converged obj = 0.08
│   └── Paola decides: "Stuck at local minimum, try CMA-ES"
│
└── Run 3: Escape local minimum (CMA-ES)
    ├── Family: cmaes
    ├── Warm-start mean from Run 2
    ├── 20 generations, final obj = 0.05
    └── Paola decides: "Converged, done"

Overall Session: success=true, final_obj=0.05, total_evals=100
```

### 1.5 Key Insight

The **polymorphic components** (initialization, progress, result) apply at the **Run level**, not the Session level. A single Session contains multiple Runs, each with its own optimizer-family-specific data.

---

## 2. Schema Architecture

### 2.1 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                  SessionRecord (Complete Task)                   │
├─────────────────────────────────────────────────────────────────┤
│  session_id, problem_id, created_at                             │
│  success, final_objective, final_design                         │
│  total_evaluations, total_wall_time                             │
├─────────────────────────────────────────────────────────────────┤
│  runs: List[OptimizationRun]                                    │
│    ├── Run 1 (bayesian family)                                  │
│    ├── Run 2 (gradient family)                                  │
│    └── Run 3 (cmaes family)                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│               OptimizationRun (Single Optimizer)                 │
├─────────────────────────────────────────────────────────────────┤
│  run_id, optimizer, optimizer_family                            │
│  warm_start_from (reference to previous run)                    │
│  n_evaluations, wall_time, run_success                          │
├─────────────────────────────────────────────────────────────────┤
│  Polymorphic components (vary by optimizer_family):             │
│  - initialization: InitializationComponent                      │
│  - progress: ProgressComponent                                  │
│  - result: ResultComponent                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Optimizer Families

| Family | Optimizers | Key Characteristics |
|--------|------------|---------------------|
| `gradient` | SciPy (SLSQP, L-BFGS-B, trust-constr), IPOPT, NLopt (LD_*) | Single starting point, iterations, gradients |
| `population` | DE, GA, NSGA-II, PSO | Population of solutions, generations |
| `bayesian` | Optuna (TPE), BO-GP | Trials, surrogate model, acquisition |
| `cmaes` | CMA-ES | Mean + covariance, special structure |
| `simplex` | Nelder-Mead, COBYLA | Simplex vertices, derivative-free |

---

## 3. Schema Definitions

### 3.1 SessionRecord (Complete Optimization Task)

```python
@dataclass
class SessionRecord:
    """
    Complete optimization session record.

    A Session represents Paola's complete effort to solve an optimization problem.
    It may involve multiple runs using different optimizers.

    Example:
        - Run 1: Bayesian exploration to find promising region
        - Run 2: Gradient refinement to converge
        - Run 3: CMA-ES to escape local minimum
    """

    # === Session Identity ===
    session_id: int
    problem_id: str
    created_at: str              # ISO timestamp

    # === Configuration ===
    config: Dict[str, Any]       # User-provided configuration

    # === Runs (multiple optimizers) ===
    runs: List['OptimizationRun']

    # === Overall Session Outcome ===
    success: bool                # Did Paola consider this successful?
    final_objective: float       # Best objective found across all runs
    final_design: List[float]    # Best design found
    total_evaluations: int       # Sum across all runs
    total_wall_time: float       # Total time (seconds)

    # === Paola's Decision Log ===
    decisions: List['PaolaDecision']  # Why Paola switched runs, etc.

    def get_run(self, run_id: int) -> Optional['OptimizationRun']:
        """Get run by ID."""
        for run in self.runs:
            if run.run_id == run_id:
                return run
        return None

    def get_best_run(self) -> Optional['OptimizationRun']:
        """Get run that found the best objective."""
        if not self.runs:
            return None
        return min(self.runs, key=lambda r: r.best_objective)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "problem_id": self.problem_id,
            "created_at": self.created_at,
            "config": self.config,
            "runs": [r.to_dict() for r in self.runs],
            "success": self.success,
            "final_objective": self.final_objective,
            "final_design": self.final_design,
            "total_evaluations": self.total_evaluations,
            "total_wall_time": self.total_wall_time,
            "decisions": [d.to_dict() for d in self.decisions]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionRecord':
        """Deserialize from dictionary."""
        runs = [OptimizationRun.from_dict(r) for r in data["runs"]]
        decisions = [PaolaDecision.from_dict(d) for d in data.get("decisions", [])]
        return cls(
            session_id=data["session_id"],
            problem_id=data["problem_id"],
            created_at=data["created_at"],
            config=data.get("config", {}),
            runs=runs,
            success=data["success"],
            final_objective=data["final_objective"],
            final_design=data["final_design"],
            total_evaluations=data["total_evaluations"],
            total_wall_time=data["total_wall_time"],
            decisions=decisions
        )


@dataclass
class PaolaDecision:
    """
    Record of Paola's strategic decision during optimization.

    Captures why Paola switched runs, changed strategy, etc.
    Important for learning and explainability.
    """
    timestamp: str
    decision_type: str           # "start_run", "switch_optimizer", "terminate"
    reasoning: str               # Paola's reasoning (from LLM)
    from_run: Optional[int]      # Run ID before decision
    to_run: Optional[int]        # Run ID after decision (if applicable)
    metrics_at_decision: Dict[str, Any]  # Metrics that informed decision

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "decision_type": self.decision_type,
            "reasoning": self.reasoning,
            "from_run": self.from_run,
            "to_run": self.to_run,
            "metrics_at_decision": self.metrics_at_decision
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaolaDecision':
        return cls(**data)
```

### 3.2 OptimizationRun (Single Optimizer Execution)

```python
@dataclass
class OptimizationRun:
    """
    Single optimizer execution within a session.

    Each run uses one optimizer from one family.
    Runs can warm-start from previous runs.
    """

    # === Run Identity ===
    run_id: int                  # Sequential within session (1, 2, 3, ...)
    optimizer: str               # Full spec: "scipy:SLSQP", "optuna:TPE"
    optimizer_family: str        # Family: "gradient", "bayesian", etc.

    # === Warm-Start Reference ===
    warm_start_from: Optional[int]  # run_id of source run, or None

    # === Run Metrics ===
    n_evaluations: int
    wall_time: float             # Seconds
    run_success: bool            # Did this run succeed?
    best_objective: float        # Best found in this run
    best_design: List[float]     # Best design in this run

    # === Family-Specific Components (Polymorphic) ===
    initialization: 'InitializationComponent'
    progress: 'ProgressComponent'
    result: 'ResultComponent'

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "run_id": self.run_id,
            "optimizer": self.optimizer,
            "optimizer_family": self.optimizer_family,
            "warm_start_from": self.warm_start_from,
            "n_evaluations": self.n_evaluations,
            "wall_time": self.wall_time,
            "run_success": self.run_success,
            "best_objective": self.best_objective,
            "best_design": self.best_design,
            "initialization": self.initialization.to_dict(),
            "progress": self.progress.to_dict(),
            "result": self.result.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationRun':
        """Deserialize from dictionary using component registry."""
        family = data["optimizer_family"]

        init, progress, result = COMPONENT_REGISTRY.deserialize_components(
            family=family,
            init_data=data["initialization"],
            progress_data=data["progress"],
            result_data=data["result"]
        )

        return cls(
            run_id=data["run_id"],
            optimizer=data["optimizer"],
            optimizer_family=family,
            warm_start_from=data.get("warm_start_from"),
            n_evaluations=data["n_evaluations"],
            wall_time=data["wall_time"],
            run_success=data["run_success"],
            best_objective=data["best_objective"],
            best_design=data["best_design"],
            initialization=init,
            progress=progress,
            result=result
        )
```

### 3.3 Component Base Classes

```python
from abc import ABC, abstractmethod

@dataclass
class InitializationComponent(ABC):
    """Base class for run initialization data."""

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

---

## 4. Family-Specific Components

### 4.1 Gradient Family

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
        iterations = [GradientIteration(**it) for it in data["iterations"]]
        return cls(iterations=iterations)


@dataclass
class GradientResult(ResultComponent):
    """Detailed result for gradient-based optimizers."""

    termination_reason: str      # "convergence", "max_iter", "failed"
    final_gradient_norm: Optional[float] = None
    final_constraint_violation: Optional[float] = None
    lagrange_multipliers: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "gradient",
            "termination_reason": self.termination_reason,
            "final_gradient_norm": self.final_gradient_norm,
            "final_constraint_violation": self.final_constraint_violation,
            "lagrange_multipliers": self.lagrange_multipliers
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GradientResult':
        return cls(
            termination_reason=data["termination_reason"],
            final_gradient_norm=data.get("final_gradient_norm"),
            final_constraint_violation=data.get("final_constraint_violation"),
            lagrange_multipliers=data.get("lagrange_multipliers")
        )
```

### 4.2 Bayesian Family

```python
@dataclass
class BayesianInitialization(InitializationComponent):
    """Initialization for Bayesian optimizers."""

    specification: Dict[str, Any]
    warm_start_trials: Optional[List[Dict[str, Any]]] = None
    n_initial_random: int = 10

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
        trials = [Trial(**t) for t in data["trials"]]
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
        return cls(**{k: data[k] for k in ["termination_reason", "best_trial_number",
                                            "n_complete_trials", "n_pruned_trials"]})
```

### 4.3 Population Family

```python
@dataclass
class PopulationInitialization(InitializationComponent):
    """Initialization for population-based optimizers."""

    specification: Dict[str, Any]  # {"type": "lhs", "size": 50}
    method: str                    # "lhs", "sobol", "random"
    population_size: int
    initial_population: List[List[float]]

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
    mean_objective: float
    diversity: Optional[float] = None


@dataclass
class PopulationProgress(ProgressComponent):
    """Progress data for population-based optimizers."""

    generations: List[Generation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "population",
            "generations": [
                {
                    "generation": g.generation,
                    "best_objective": g.best_objective,
                    "best_design": g.best_design,
                    "mean_objective": g.mean_objective,
                    "diversity": g.diversity
                }
                for g in self.generations
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PopulationProgress':
        generations = [Generation(**g) for g in data["generations"]]
        return cls(generations=generations)


@dataclass
class PopulationResult(ResultComponent):
    """Detailed result for population-based optimizers."""

    termination_reason: str
    final_population_size: int
    final_diversity: Optional[float] = None
    pareto_front: Optional[List[List[float]]] = None  # For multi-objective

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "population",
            "termination_reason": self.termination_reason,
            "final_population_size": self.final_population_size,
            "final_diversity": self.final_diversity,
            "pareto_front": self.pareto_front
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PopulationResult':
        return cls(
            termination_reason=data["termination_reason"],
            final_population_size=data["final_population_size"],
            final_diversity=data.get("final_diversity"),
            pareto_front=data.get("pareto_front")
        )
```

### 4.4 CMA-ES Family

```python
@dataclass
class CMAESInitialization(InitializationComponent):
    """Initialization for CMA-ES optimizer."""

    specification: Dict[str, Any]
    mean: List[float]
    sigma: float
    population_size: int

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
    mean: List[float]
    sigma: float
    condition_number: Optional[float] = None


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
        generations = [CMAESGeneration(**g) for g in data["generations"]]
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

```python
from typing import Type, Dict, Tuple, List

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
    - Validation of run data
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
        """Register an optimizer family."""
        self._families[family] = OptimizerFamilySchema(
            family=family,
            initialization_class=initialization_class,
            progress_class=progress_class,
            result_class=result_class
        )
        for opt in optimizers:
            self._optimizer_to_family[opt] = family

    def get_family(self, optimizer: str) -> str:
        """Get family name for an optimizer (handles 'scipy:SLSQP' format)."""
        base = optimizer.split(":")[0]
        return self._optimizer_to_family.get(base, "gradient")

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
    optimizers=["scipy", "ipopt", "nlopt"]
)

COMPONENT_REGISTRY.register_family(
    family="bayesian",
    initialization_class=BayesianInitialization,
    progress_class=BayesianProgress,
    result_class=BayesianResult,
    optimizers=["optuna", "bo"]
)

COMPONENT_REGISTRY.register_family(
    family="population",
    initialization_class=PopulationInitialization,
    progress_class=PopulationProgress,
    result_class=PopulationResult,
    optimizers=["de", "ga", "nsga2", "pso"]
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

## 6. Storage Format

### 6.1 Single File Per Session

Each session is stored as a single JSON file containing all runs:

```
.paola_runs/
├── sessions/
│   ├── session_0001.json
│   ├── session_0002.json
│   └── session_0042.json    # Multi-run session
├── problems/
│   └── ...
└── evaluators/
    └── ...
```

### 6.2 Example: Multi-Run Session JSON

```json
{
  "session_id": 42,
  "problem_id": "wing_v2",
  "created_at": "2025-12-15T21:30:00",
  "config": {
    "goal": "minimize drag",
    "max_total_evaluations": 200
  },

  "runs": [
    {
      "run_id": 1,
      "optimizer": "optuna:TPE",
      "optimizer_family": "bayesian",
      "warm_start_from": null,
      "n_evaluations": 50,
      "wall_time": 120.5,
      "run_success": true,
      "best_objective": 0.15,
      "best_design": [0.1, 0.2, "..."],

      "initialization": {
        "family": "bayesian",
        "specification": {"type": "none"},
        "warm_start_trials": null,
        "n_initial_random": 10
      },
      "progress": {
        "family": "bayesian",
        "trials": [
          {"trial_number": 1, "design": ["..."], "objective": 0.8, "state": "complete"},
          {"trial_number": 2, "design": ["..."], "objective": 0.5, "state": "complete"}
        ]
      },
      "result": {
        "family": "bayesian",
        "termination_reason": "n_trials_reached",
        "best_trial_number": 42,
        "n_complete_trials": 50,
        "n_pruned_trials": 0
      }
    },

    {
      "run_id": 2,
      "optimizer": "scipy:SLSQP",
      "optimizer_family": "gradient",
      "warm_start_from": 1,
      "n_evaluations": 30,
      "wall_time": 45.2,
      "run_success": true,
      "best_objective": 0.08,
      "best_design": [0.12, 0.18, "..."],

      "initialization": {
        "family": "gradient",
        "specification": {"type": "warm_start", "source_run": 1},
        "x0": [0.1, 0.2, "..."]
      },
      "progress": {
        "family": "gradient",
        "iterations": [
          {"iteration": 1, "objective": 0.15, "design": ["..."], "gradient_norm": 0.5},
          {"iteration": 2, "objective": 0.12, "design": ["..."], "gradient_norm": 0.3}
        ]
      },
      "result": {
        "family": "gradient",
        "termination_reason": "convergence",
        "final_gradient_norm": 1e-6,
        "final_constraint_violation": 0.0
      }
    },

    {
      "run_id": 3,
      "optimizer": "cmaes",
      "optimizer_family": "cmaes",
      "warm_start_from": 2,
      "n_evaluations": 20,
      "wall_time": 30.1,
      "run_success": true,
      "best_objective": 0.05,
      "best_design": [0.11, 0.19, "..."],

      "initialization": {
        "family": "cmaes",
        "specification": {"type": "warm_start", "source_run": 2},
        "mean": [0.12, 0.18, "..."],
        "sigma": 0.1,
        "population_size": 10
      },
      "progress": {
        "family": "cmaes",
        "generations": [
          {"generation": 1, "best_objective": 0.08, "mean": ["..."], "sigma": 0.1},
          {"generation": 2, "best_objective": 0.06, "mean": ["..."], "sigma": 0.08}
        ]
      },
      "result": {
        "family": "cmaes",
        "termination_reason": "convergence",
        "final_mean": [0.11, 0.19, "..."],
        "final_sigma": 0.01
      }
    }
  ],

  "success": true,
  "final_objective": 0.05,
  "final_design": [0.11, 0.19, "..."],
  "total_evaluations": 100,
  "total_wall_time": 195.8,

  "decisions": [
    {
      "timestamp": "2025-12-15T21:32:00",
      "decision_type": "switch_optimizer",
      "reasoning": "Bayesian exploration found promising region (obj=0.15). Switching to gradient-based for local refinement.",
      "from_run": 1,
      "to_run": 2,
      "metrics_at_decision": {"best_obj": 0.15, "n_trials": 50}
    },
    {
      "timestamp": "2025-12-15T21:33:00",
      "decision_type": "switch_optimizer",
      "reasoning": "SLSQP converged but may be stuck at local minimum (obj=0.08). Trying CMA-ES to explore nearby region.",
      "from_run": 2,
      "to_run": 3,
      "metrics_at_decision": {"best_obj": 0.08, "gradient_norm": 1e-6}
    },
    {
      "timestamp": "2025-12-15T21:33:30",
      "decision_type": "terminate",
      "reasoning": "CMA-ES found better solution (obj=0.05) and converged. Optimization complete.",
      "from_run": 3,
      "to_run": null,
      "metrics_at_decision": {"best_obj": 0.05, "sigma": 0.01}
    }
  ]
}
```

---

## 7. Querying

### 7.1 Session-Level Queries

```python
class Foundry:
    def query_sessions(
        self,
        problem_id: Optional[str] = None,
        success: Optional[bool] = None,
        min_objective: Optional[float] = None,
        max_evaluations: Optional[int] = None
    ) -> List[SessionRecord]:
        """Query sessions by common fields."""
        ...

    def get_best_session(self, problem_id: str) -> Optional[SessionRecord]:
        """Get most successful session for a problem."""
        ...
```

### 7.2 Run-Level Queries

```python
class Foundry:
    def get_runs_by_family(
        self,
        problem_id: str,
        optimizer_family: str
    ) -> List[Tuple[SessionRecord, OptimizationRun]]:
        """Get all runs of a specific family for a problem."""
        ...

    def get_warm_start_chains(
        self,
        problem_id: str
    ) -> List[List[OptimizationRun]]:
        """Get chains of warm-started runs for learning."""
        ...
```

---

## 8. Active Session Management

### 8.1 ActiveSession Class

```python
class ActiveSession:
    """
    Handle for in-progress optimization session.

    Manages runs as they are created and completed.
    """

    def __init__(self, session_id: int, problem_id: str, storage: StorageBackend):
        self.session_id = session_id
        self.problem_id = problem_id
        self.storage = storage
        self.runs: List[OptimizationRun] = []
        self.decisions: List[PaolaDecision] = []
        self.current_run: Optional[ActiveRun] = None
        self.start_time = datetime.now()

    def start_run(
        self,
        optimizer: str,
        initialization: InitializationComponent,
        warm_start_from: Optional[int] = None
    ) -> 'ActiveRun':
        """Start a new optimization run."""
        run_id = len(self.runs) + 1
        family = COMPONENT_REGISTRY.get_family(optimizer)

        self.current_run = ActiveRun(
            run_id=run_id,
            optimizer=optimizer,
            optimizer_family=family,
            initialization=initialization,
            warm_start_from=warm_start_from
        )
        return self.current_run

    def complete_run(
        self,
        progress: ProgressComponent,
        result: ResultComponent,
        best_objective: float,
        best_design: List[float]
    ):
        """Complete current run and add to session."""
        if self.current_run is None:
            raise RuntimeError("No active run")

        run = self.current_run.finalize(
            progress=progress,
            result=result,
            best_objective=best_objective,
            best_design=best_design
        )
        self.runs.append(run)
        self.current_run = None
        self._persist()

    def record_decision(
        self,
        decision_type: str,
        reasoning: str,
        metrics: Dict[str, Any]
    ):
        """Record Paola's strategic decision."""
        decision = PaolaDecision(
            timestamp=datetime.now().isoformat(),
            decision_type=decision_type,
            reasoning=reasoning,
            from_run=len(self.runs) if self.runs else None,
            to_run=len(self.runs) + 1 if decision_type == "switch_optimizer" else None,
            metrics_at_decision=metrics
        )
        self.decisions.append(decision)

    def finalize(self, success: bool) -> SessionRecord:
        """Finalize session and return record."""
        # Compute overall metrics
        total_evals = sum(r.n_evaluations for r in self.runs)
        total_time = (datetime.now() - self.start_time).total_seconds()
        best_run = min(self.runs, key=lambda r: r.best_objective)

        record = SessionRecord(
            session_id=self.session_id,
            problem_id=self.problem_id,
            created_at=self.start_time.isoformat(),
            config={},
            runs=self.runs,
            success=success,
            final_objective=best_run.best_objective,
            final_design=best_run.best_design,
            total_evaluations=total_evals,
            total_wall_time=total_time,
            decisions=self.decisions
        )

        self.storage.save_session(record)
        return record
```

---

## 9. Adding New Optimizer Families

To add a new family (e.g., "surrogate"):

### Step 1: Define Components

```python
# paola/foundry/schema/surrogate.py

@dataclass
class SurrogateInitialization(InitializationComponent):
    # Surrogate-specific fields
    ...

@dataclass
class SurrogateProgress(ProgressComponent):
    # Surrogate-specific fields
    ...

@dataclass
class SurrogateResult(ResultComponent):
    # Surrogate-specific fields
    ...
```

### Step 2: Register

```python
COMPONENT_REGISTRY.register_family(
    family="surrogate",
    initialization_class=SurrogateInitialization,
    progress_class=SurrogateProgress,
    result_class=SurrogateResult,
    optimizers=["ego", "sbo"]
)
```

**No changes to SessionRecord, OptimizationRun, or storage layer required.**

---

## 10. API Tool Updates

### 10.1 New Session Tools

```python
# Session management (new)
start_session(problem_id, goal, config) -> session_id
finalize_session(session_id, success) -> SessionRecord
get_session_info(session_id) -> SessionRecord

# Run management (existing tool, unchanged signature)
run_optimization(problem_id, optimizer, config) -> run_result
# Now internally linked to current active session
```

### 10.2 Backward Compatibility

For simple cases (one optimizer, no session management):
- `run_optimization()` auto-creates a session if none active
- Single-run sessions work exactly like before
- No breaking changes for existing workflows

---

## 11. Summary

### Key Terminology

| Term | Definition |
|------|------------|
| **Session** | Complete optimization task orchestrated by Paola |
| **Run** | Single optimizer execution within a session |
| **Family** | Category of optimizer with shared data structures |
| **Warm-start** | Using results from previous run as initialization |
| **Decision** | Paola's strategic choice recorded for learning |

### Benefits

1. **Multi-optimizer sessions** - Paola can use different strategies in one session
2. **Warm-start chains** - Track how runs connect
3. **Decision logging** - Explainable optimization process
4. **Family-specific data** - Each optimizer stores what it needs
5. **Extensibility** - Add new families without changing core
6. **Clear terminology** - No more overloaded "run"

### Files to Create

```
paola/foundry/
├── schema/
│   ├── __init__.py
│   ├── base.py              # SessionRecord, OptimizationRun, PaolaDecision
│   ├── components.py        # Component ABCs
│   ├── gradient.py          # Gradient family
│   ├── bayesian.py          # Bayesian family
│   ├── population.py        # Population family
│   ├── cmaes.py             # CMA-ES family
│   └── registry.py          # ComponentRegistry
├── active_session.py        # ActiveSession, ActiveRun
└── ...
```

---

**Document Status**: Ready for Implementation
**Version**: 2.1 (Session/Run terminology clarification)
**Next Action**: Implement for Paola v0.2.0
