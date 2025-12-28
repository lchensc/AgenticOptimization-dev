# Optimizer Tools Integration Plan

**Date**: 2025-12-27
**Status**: Analysis Complete
**Context**: Integrating refactored optimizer backends into Paola tools

## Current State Analysis

### Refactored Backend Architecture (Complete)
```
paola/optimizers/
├── __init__.py          # Registry exports
├── base.py              # OptimizerBackend ABC
├── result.py            # OptimizationResult (new format)
├── wrapper.py           # ObjectiveWrapper, GradientWrapper
├── registry.py          # BackendRegistry
└── backends/
    ├── scipy_backend.py
    ├── ipopt_backend.py
    ├── optuna_backend.py
    └── pymoo_backend.py  # NEW: 14 algorithms
```

### Tools Layer (Needs Update)
```
paola/tools/
├── optimizer.py         # run_optimization, list_optimizers
├── evaluation.py        # evaluate_function, compute_gradient
├── graph.py             # Graph management
└── ...
```

## Issues Identified

### 1. Result Attribute Mismatch

**Current code** in `optimizer.py:326-345`:
```python
best_x = result.final_design.tolist()  # OLD
best_objective = result.final_objective  # OLD
```

**New OptimizationResult** in `result.py`:
```python
best_x: np.ndarray    # NEW
best_f: float         # NEW
```

**Fix**: Replace `final_design` → `best_x`, `final_objective` → `best_f`

### 2. Missing Family Handlers

**Current code** in `optimizer.py:220-322` only handles:
- `gradient` family
- `bayesian` family

**Missing handlers for**:
- `evolutionary` (pymoo GA, DE, PSO, ES)
- `multiobjective` (NSGA-II, NSGA-III, MOEA/D)
- `population` (legacy)
- `cmaes` (CMA-ES specific)

### 3. Missing Exports

**Current** `paola/optimizers/__init__.py`:
```python
from paola.optimizers.backends import (
    SciPyBackend,
    IPOPTBackend,
    OptunaBackend,
)  # Missing PymooBackend
```

### 4. Multi-Objective Support

The new `OptimizationResult` supports:
- `pareto_set`: (n_solutions, n_vars)
- `pareto_front`: (n_solutions, n_objectives)
- `hypervolume`: float

These need to be:
- Returned in tool response
- Stored in graph (via `pareto_ref` in MultiObjectiveResult)

## Implementation Plan

### Phase 1: Fix Compatibility Issues

#### 1.1 Update `paola/optimizers/__init__.py`
```python
from paola.optimizers.backends import (
    SciPyBackend,
    IPOPTBackend,
    OptunaBackend,
    PymooBackend,  # ADD
)
```

#### 1.2 Fix `optimizer.py` Result Attributes
```python
# Line 326-329: Replace
best_x = (
    result.best_x.tolist()
    if isinstance(result.best_x, np.ndarray)
    else list(result.best_x)
)

# Line 334, 345: Replace
best_objective=result.best_f
```

### Phase 2: Add Family Handlers

#### 2.1 Import New Schema Components
```python
from ..foundry import (
    # Existing
    COMPONENT_REGISTRY,
    EdgeType,
    GradientInitialization, GradientProgress, GradientResult,
    BayesianInitialization, BayesianProgress, BayesianResult,
    # NEW
    EvolutionaryInitialization, EvolutionaryProgress, EvolutionaryResult,
    MultiObjectiveInitialization, MultiObjectiveProgress, MultiObjectiveResult,
    PopulationInitialization, PopulationProgress, PopulationResult,
    CMAESInitialization, CMAESProgress, CMAESResult,
)
```

#### 2.2 Add Initialization Handlers (after line 236)
```python
elif family == "evolutionary":
    initialization = EvolutionaryInitialization(
        specification={"type": init_strategy, "parent_node": parent_node},
        pop_size=config_dict.get("pop_size", 100),
        seed=config_dict.get("seed"),
        initial_population=None,  # Could warm-start from parent
    )
elif family == "multiobjective":
    initialization = MultiObjectiveInitialization(
        specification={"type": init_strategy, "parent_node": parent_node},
        pop_size=config_dict.get("pop_size", 100),
        n_objectives=config_dict.get("n_objectives", 2),
        seed=config_dict.get("seed"),
    )
elif family == "cmaes":
    initialization = CMAESInitialization(
        specification={"type": init_strategy, "parent_node": parent_node},
        x0=x0.tolist(),
        sigma0=config_dict.get("sigma", 0.5),
    )
elif family == "population":
    initialization = PopulationInitialization(
        specification={"type": init_strategy, "parent_node": parent_node},
        pop_size=config_dict.get("pop_size", 50),
        seed=config_dict.get("seed"),
    )
```

#### 2.3 Add Progress/Result Handlers (after line 322)
```python
elif family == "evolutionary":
    progress = EvolutionaryProgress()
    for i, h in enumerate(result.history):
        progress.add_generation(
            generation=i + 1,
            best_fitness=h.get("best_f", h.get("f", 0.0)),
            mean_fitness=h.get("mean_f", 0.0),
            best_individual=h.get("x", []),
            diversity=h.get("diversity"),
        )
    result_component = EvolutionaryResult(
        termination_reason=result.message,
        n_generations=result.n_iterations,
        final_pop_size=config_dict.get("pop_size", 100),
    )

elif family == "multiobjective":
    progress = MultiObjectiveProgress()
    for i, h in enumerate(result.history):
        progress.add_generation(
            generation=i + 1,
            n_nondominated=h.get("n_nondominated", 0),
            hypervolume=h.get("hypervolume"),
        )
    result_component = MultiObjectiveResult(
        termination_reason=result.message,
        n_generations=result.n_iterations,
        n_pareto_solutions=result.n_pareto_solutions if result.is_multiobjective else 0,
        final_hypervolume=result.hypervolume,
        pareto_ref=None,  # Would store in GraphDetail
    )

elif family == "cmaes":
    progress = CMAESProgress()
    for i, h in enumerate(result.history):
        progress.add_generation(
            generation=i + 1,
            best_fitness=h.get("best_f", h.get("f", 0.0)),
            sigma=h.get("sigma"),
        )
    result_component = CMAESResult(
        termination_reason=result.message,
        n_generations=result.n_iterations,
        final_sigma=None,
    )

elif family == "population":
    progress = PopulationProgress()
    for i, h in enumerate(result.history):
        progress.add_generation(
            generation=i + 1,
            best_fitness=h.get("best_f", h.get("f", 0.0)),
            mean_fitness=h.get("mean_f", 0.0),
        )
    result_component = PopulationResult(
        termination_reason=result.message,
        n_generations=result.n_iterations,
        final_pop_size=config_dict.get("pop_size", 50),
    )
```

### Phase 3: Multi-Objective Support

#### 3.1 Enhance Return Value for MO
```python
response = {
    "success": result.success,
    "message": result.message,
    "node_id": completed_node.node_id,
    "best_x": best_x,
    "best_objective": float(result.best_f),
    # ... existing fields ...
}

# Add MO fields if applicable
if result.is_multiobjective:
    response["is_multiobjective"] = True
    response["n_pareto_solutions"] = result.n_pareto_solutions
    response["hypervolume"] = result.hypervolume
    # Pareto data stored in GraphDetail, referenced by pareto_ref
```

#### 3.2 Update Docstring
```python
@tool(args_schema=RunOptimizationArgs)
def run_optimization(...):
    """
    Run optimization, creating a new node in the graph.

    Args:
        ...
        optimizer: Backend:method specification
            - "scipy:SLSQP", "scipy:L-BFGS-B", "ipopt"
            - "optuna:TPE", "optuna:CMA-ES"
            - "pymoo:GA", "pymoo:DE", "pymoo:PSO"       # NEW
            - "pymoo:NSGA-II", "pymoo:NSGA-III"         # NEW (multi-objective)
        ...
    """
```

## Verification Plan

1. **Unit test**: Import all backends successfully
2. **Integration test**: Run optimization with each backend type
3. **Multi-objective test**: Verify Pareto front returned correctly
4. **Graph test**: Verify nodes recorded with correct component types

## Files to Modify

| File | Changes |
|------|---------|
| `paola/optimizers/__init__.py` | Add PymooBackend export |
| `paola/tools/optimizer.py` | Fix attributes, add family handlers |

## Estimated Effort

- Phase 1: ~10 lines changed
- Phase 2: ~80 lines added
- Phase 3: ~20 lines added
- Testing: Verify with ml conda environment
