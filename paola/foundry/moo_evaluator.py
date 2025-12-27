"""
Multi-Objective Evaluator utilities.

Provides efficient handling of array-returning evaluators for MOO problems.

When multiple objectives share the same evaluator (e.g., CFD returning drag and lift),
this module ensures the evaluator is called only once per design point.

Example:
    # Problem with two objectives from same evaluator
    objectives = [
        Objective(name="drag", evaluator_id="cfd", index=0),
        Objective(name="lift", evaluator_id="cfd", index=1),
    ]

    # Create MOO evaluator
    moo_eval = MOOEvaluator(problem, get_evaluator_func)

    # Get objective functions for optimizer
    objective_funcs = moo_eval.get_objective_functions()
    # objective_funcs[0](x) returns drag, objective_funcs[1](x) returns lift
    # But cfd evaluator is only called ONCE per x value
"""

from typing import Dict, Any, Callable, List, Optional, Tuple
import numpy as np
import logging
from dataclasses import dataclass, field

from .problem import OptimizationProblem, Objective

logger = logging.getLogger(__name__)


@dataclass
class CachedEvaluation:
    """Cached evaluation result from an array-returning evaluator."""
    design_hash: str
    result: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class ArrayEvaluatorCache:
    """
    Cache for array-returning evaluators.

    Ensures each evaluator is called at most once per unique design point.
    Uses design vector hash as cache key.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of cached evaluations per evaluator
        """
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}  # evaluator_id -> {design_hash -> result}
        self._max_size = max_size

    def _design_hash(self, design: np.ndarray) -> str:
        """Create hash key from design vector."""
        # Round to avoid floating-point noise
        rounded = np.round(design, decimals=10)
        return str(rounded.tobytes())

    def get(self, evaluator_id: str, design: np.ndarray) -> Optional[np.ndarray]:
        """
        Get cached result for evaluator and design.

        Args:
            evaluator_id: Evaluator identifier
            design: Design vector

        Returns:
            Cached array result or None if not cached
        """
        if evaluator_id not in self._cache:
            return None

        design_hash = self._design_hash(design)
        return self._cache[evaluator_id].get(design_hash)

    def store(self, evaluator_id: str, design: np.ndarray, result: np.ndarray) -> None:
        """
        Store evaluation result in cache.

        Args:
            evaluator_id: Evaluator identifier
            design: Design vector
            result: Array result from evaluator
        """
        if evaluator_id not in self._cache:
            self._cache[evaluator_id] = {}

        # Evict oldest entries if at capacity
        cache = self._cache[evaluator_id]
        if len(cache) >= self._max_size:
            # Remove oldest entry (FIFO approximation)
            oldest_key = next(iter(cache))
            del cache[oldest_key]

        design_hash = self._design_hash(design)
        cache[design_hash] = result.copy()

    def clear(self, evaluator_id: Optional[str] = None) -> None:
        """
        Clear cache.

        Args:
            evaluator_id: Clear specific evaluator's cache, or all if None
        """
        if evaluator_id is None:
            self._cache.clear()
        elif evaluator_id in self._cache:
            self._cache[evaluator_id].clear()


class MOOEvaluator:
    """
    Multi-Objective Evaluator with efficient array-returning evaluator support.

    Wraps an OptimizationProblem and provides objective functions for MOO optimizers.
    Handles the case where multiple objectives share the same evaluator but use
    different indices.

    Example:
        problem = OptimizationProblem(
            objectives=[
                Objective(name="f1", evaluator_id="zdt1", index=0),
                Objective(name="f2", evaluator_id="zdt1", index=1),
            ],
            ...
        )

        moo_eval = MOOEvaluator(problem, evaluator_registry.get)
        funcs = moo_eval.get_objective_functions()

        # Both funcs call the same underlying evaluator, but efficiently
        f1_val = funcs[0](design)  # Calls zdt1, returns index 0
        f2_val = funcs[1](design)  # Uses cached result, returns index 1
    """

    def __init__(
        self,
        problem: OptimizationProblem,
        get_evaluator: Callable[[str], Callable],
        cache_size: int = 1000,
    ):
        """
        Initialize MOO evaluator.

        Args:
            problem: OptimizationProblem with objectives
            get_evaluator: Function to retrieve evaluator by ID
            cache_size: Maximum cache size per evaluator
        """
        self.problem = problem
        self.get_evaluator = get_evaluator
        self._cache = ArrayEvaluatorCache(max_size=cache_size)

        # Map evaluator_id -> list of objectives using it
        self._evaluator_objectives: Dict[str, List[Objective]] = {}
        for obj in problem.objectives:
            if obj.evaluator_id not in self._evaluator_objectives:
                self._evaluator_objectives[obj.evaluator_id] = []
            self._evaluator_objectives[obj.evaluator_id].append(obj)

        # Track if evaluator returns array or scalar
        self._evaluator_is_array: Dict[str, bool] = {}

    def _evaluate_raw(self, evaluator_id: str, design: np.ndarray) -> np.ndarray:
        """
        Evaluate raw result from evaluator with caching.

        Args:
            evaluator_id: Evaluator identifier
            design: Design vector

        Returns:
            Raw evaluator result as numpy array
        """
        # Check cache
        cached = self._cache.get(evaluator_id, design)
        if cached is not None:
            return cached

        # Get evaluator function
        evaluator = self.get_evaluator(evaluator_id)
        if evaluator is None:
            raise ValueError(f"Evaluator '{evaluator_id}' not found")

        # Evaluate
        result = evaluator(design)

        # Convert to array if needed
        if isinstance(result, np.ndarray):
            result_array = result
        elif isinstance(result, (list, tuple)):
            result_array = np.array(result)
        elif isinstance(result, (int, float, np.number)):
            result_array = np.array([float(result)])
        else:
            raise ValueError(f"Unexpected result type from evaluator: {type(result)}")

        # Store in cache
        self._cache.store(evaluator_id, design, result_array)

        return result_array

    def evaluate_objective(self, objective: Objective, design: np.ndarray) -> float:
        """
        Evaluate a single objective.

        Args:
            objective: Objective specification
            design: Design vector

        Returns:
            Objective value (float)
        """
        result = self._evaluate_raw(objective.evaluator_id, design)

        # Extract by index
        if objective.index is not None:
            if objective.index >= len(result):
                raise ValueError(
                    f"Objective '{objective.name}' requests index {objective.index}, "
                    f"but evaluator returned only {len(result)} values"
                )
            value = float(result[objective.index])
        else:
            # No index specified, use first element
            value = float(result[0])

        # Handle maximize vs minimize
        if objective.sense == "maximize":
            value = -value

        return value

    def evaluate_all(self, design: np.ndarray) -> np.ndarray:
        """
        Evaluate all objectives.

        Args:
            design: Design vector

        Returns:
            Array of objective values [f1, f2, ..., fn]
        """
        return np.array([
            self.evaluate_objective(obj, design)
            for obj in self.problem.objectives
        ])

    def get_objective_functions(self) -> List[Callable[[np.ndarray], float]]:
        """
        Get list of objective functions for MOO optimizer.

        Returns:
            List of callables, one per objective
        """
        funcs = []
        for obj in self.problem.objectives:
            # Create closure with objective captured
            def make_func(objective):
                def f(x):
                    return self.evaluate_objective(objective, x)
                return f
            funcs.append(make_func(obj))
        return funcs

    def get_constraint_functions(self) -> List[Callable[[np.ndarray], float]]:
        """
        Get list of constraint functions (g(x) <= 0 form).

        Returns:
            List of callables, one per constraint
        """
        funcs = []
        for constr in self.problem.constraints:
            def make_func(constraint):
                def g(x):
                    result = self._evaluate_raw(constraint.evaluator_id, x)
                    value = float(result[0])

                    # Convert to g(x) <= 0 form
                    if constraint.type == "<=":
                        return value - constraint.bound
                    elif constraint.type == ">=":
                        return constraint.bound - value
                    elif constraint.type == "==":
                        return abs(value - constraint.bound) - constraint.tolerance
                    else:
                        raise ValueError(f"Unknown constraint type: {constraint.type}")
                return g
            funcs.append(make_func(constr))
        return funcs

    def clear_cache(self) -> None:
        """Clear evaluation cache."""
        self._cache.clear()

    @property
    def n_objectives(self) -> int:
        """Number of objectives."""
        return len(self.problem.objectives)

    @property
    def n_constraints(self) -> int:
        """Number of constraints."""
        return len(self.problem.constraints)

    @property
    def objective_names(self) -> List[str]:
        """Names of objectives."""
        return [obj.name for obj in self.problem.objectives]

    @property
    def is_multiobjective(self) -> bool:
        """True if more than one objective."""
        return self.n_objectives > 1


def create_moo_evaluator(
    problem: OptimizationProblem,
    evaluator_registry: Dict[str, Callable],
) -> MOOEvaluator:
    """
    Create MOOEvaluator from problem and evaluator registry.

    Convenience function that wraps dictionary lookup.

    Args:
        problem: OptimizationProblem with objectives
        evaluator_registry: Dict mapping evaluator_id -> callable

    Returns:
        MOOEvaluator instance
    """
    def get_evaluator(evaluator_id: str) -> Optional[Callable]:
        return evaluator_registry.get(evaluator_id)

    return MOOEvaluator(problem, get_evaluator)
