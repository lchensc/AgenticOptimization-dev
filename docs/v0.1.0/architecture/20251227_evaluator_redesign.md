# Evaluator Architecture Redesign

**Date:** 2025-12-27
**Status:** Implemented (Simpler Fix)
**Problem:** Current `FoundryEvaluator` is hacky with too many if-else conditions and format guessing

## Implementation Summary

Instead of creating a separate evaluator module (which would break Foundry as single source of truth),
we applied a simpler fix directly to the existing infrastructure:

### Changes Made

1. **Added `n_outputs` to `OutputInterface`** (`evaluator_schema.py:126-128`):
   ```python
   n_outputs: int = Field(default=1, description="Number of output values (explicit declaration)")
   output_names: Optional[List[str]] = Field(None, description="Names for each output")
   ```

2. **Added `"array"` format type** (`evaluator_schema.py:122`):
   ```python
   format: Literal["scalar", "dict", "tuple", "array", "auto"]
   ```

3. **Updated `_parse_result`** (`evaluator.py:317-430`):
   - Uses explicit `n_outputs` when > 1 to determine format (no guessing)
   - Uses `output_names` for named outputs instead of generic `f0`, `f1`
   - Falls back to auto-detection only for legacy configs

### Why Not a Separate Module?

Initially, a separate `paola/evaluators/` module was created with:
- `EvaluatorSpec` (new schema)
- `EvaluatorRegistry` (in-memory)
- `load_evaluator()` (new loader)

This was **reverted** because:
1. Foundry is the single source of truth for all data
2. A separate registry would create dual sources of truth
3. The simpler fix (adding `n_outputs` to existing schema) achieves the same goal

## Current Issues

1. **Format guessing in `_parse_result`** - Tries to auto-detect scalar/dict/tuple/list from return value
2. **Single class does too much** - Loading, caching, observation, parsing all in one class
3. **Naming inconsistency** - Uses `design` instead of `x` for decision variables
4. **No structure known upfront** - Unlike pymoo where `n_obj`, `n_constr` are declared

## Pymoo's Clean Design (Reference)

```python
class ZDT1(Problem):
    def __init__(self):
        super().__init__(
            n_var=30,      # Known upfront
            n_obj=2,       # Known upfront
            n_constr=0,    # Known upfront
            xl=0.0, xu=1.0
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # No format guessing - just write to out
        out["F"] = np.column_stack([f1, f2])
        out["G"] = ...  # if constraints
```

Key insights:
- **Problem is an object with metadata** declared upfront
- **Single interface**: `_evaluate(x, out)` writes results
- **No format detection** - structure is known from definition

## Proposed Design

### Layer 1: Evaluator Specifications (Data)

```python
@dataclass
class EvaluatorSpec:
    """Immutable specification of an evaluator."""
    evaluator_id: str
    name: str

    # Output specification - NO GUESSING
    n_outputs: int = 1                    # How many values returned
    output_names: List[str] = None        # e.g., ["drag", "lift"]

    # Source specification
    source_type: Literal["python", "cli", "api"]
    source_config: Dict[str, Any]         # file_path, callable_name, etc.

    # Optional metadata
    cost_per_eval: float = 1.0
    description: str = ""
```

### Layer 2: Loaders (Factory Pattern)

```python
class EvaluatorLoader(ABC):
    """Abstract loader that creates callables from specs."""

    @abstractmethod
    def load(self, spec: EvaluatorSpec) -> Callable[[np.ndarray], np.ndarray]:
        """Load and return a callable that takes x and returns array of outputs."""
        pass

class PythonFunctionLoader(EvaluatorLoader):
    """Loads Python functions from files."""

    def load(self, spec: EvaluatorSpec) -> Callable:
        file_path = spec.source_config["file_path"]
        callable_name = spec.source_config["callable_name"]
        # ... import and return

class CLILoader(EvaluatorLoader):
    """Wraps CLI executables as callables."""
    # ...
```

### Layer 3: Decorators (Capabilities)

```python
class CachingEvaluator:
    """Decorator that adds caching to any evaluator."""

    def __init__(self, evaluator: Callable, cache: EvaluationCache):
        self._evaluator = evaluator
        self._cache = cache

    def __call__(self, x: np.ndarray) -> np.ndarray:
        cached = self._cache.get(x)
        if cached is not None:
            return cached
        result = self._evaluator(x)
        self._cache.store(x, result)
        return result

class ObservableEvaluator:
    """Decorator that adds observation hooks."""

    def __init__(self, evaluator: Callable, observer: EvaluationObserver):
        self._evaluator = evaluator
        self._observer = observer

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self._observer.before(x)
        result = self._evaluator(x)
        self._observer.after(x, result)
        return result
```

### Layer 4: Problem Adapter (Integration)

```python
class PaolaProblem(ElementwiseProblem):
    """
    Adapts OptimizationProblem + evaluators to pymoo interface.

    This is where MOOEvaluator lives - it knows the problem structure
    and wraps evaluators appropriately.
    """

    def __init__(self, problem: OptimizationProblem, evaluator_registry: Dict):
        self.problem = problem
        self.moo_eval = MOOEvaluator(problem, evaluator_registry.get)

        super().__init__(
            n_var=problem.n_variables,
            n_obj=problem.n_objectives,
            n_constr=problem.n_constraints,
            xl=problem.lower_bounds,
            xu=problem.upper_bounds,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # Clean interface - no format guessing
        out["F"] = self.moo_eval.evaluate_all(x)
        if self.n_constr > 0:
            out["G"] = self.moo_eval.evaluate_constraints(x)
```

## Benefits

1. **Single Responsibility** - Each class does one thing
2. **No Format Guessing** - Structure declared in EvaluatorSpec and OptimizationProblem
3. **Composable** - Decorators can be stacked: `Observable(Caching(loader.load(spec)))`
4. **Testable** - Each component can be tested independently
5. **Consistent Naming** - Use `x` throughout, matching optimization convention

## Migration Path

1. **Phase 1**: Create new `EvaluatorSpec` and loaders (parallel to existing)
2. **Phase 2**: Refactor `MOOEvaluator` to use new specs
3. **Phase 3**: Create `PaolaProblem` adapter for pymoo
4. **Phase 4**: Update tools to use new system
5. **Phase 5**: Deprecate old `FoundryEvaluator`

## Naming Convention

| Term | Usage |
|------|-------|
| `x` | Decision variables (vector) |
| `f` | Objective value(s) |
| `g` | Constraint value(s) |
| `n_var` | Number of variables |
| `n_obj` | Number of objectives |
| `xl`, `xu` | Lower/upper bounds |

Avoid: `design`, `objective`, `constraint` as variable names (use as type descriptions only)

## Open Questions

1. Should `EvaluatorSpec` be a Pydantic model or dataclass?
2. How to handle gradient evaluators in this design?
3. Should we support vectorized evaluation (batch) from the start?
