"""
Objective and gradient wrapper utilities.

Provides consistent tracking, caching, and history recording across all backends,
eliminating duplicated wrapper logic in individual backend implementations.
"""

from typing import Callable, List, Dict, Any, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..tools.cache import EvaluationCache


class ObjectiveWrapper:
    """
    Wraps objective function with tracking and optional caching.

    Provides:
    - Evaluation counting
    - Best solution tracking
    - History recording (iteration, objective, design)
    - Optional cache integration

    Usage:
        wrapper = ObjectiveWrapper(objective_fn)
        result = optimizer.minimize(wrapper, ...)
        print(f"Evaluations: {wrapper.n_evals}, Best: {wrapper.best_f}")
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        cache: Optional["EvaluationCache"] = None,
        record_history: bool = True,
    ):
        """
        Initialize objective wrapper.

        Args:
            objective: Function f(x) -> float to minimize
            cache: Optional evaluation cache for expensive simulations
            record_history: Whether to record full history (disable for memory)
        """
        self.objective = objective
        self.cache = cache
        self.record_history = record_history

        # Tracking state
        self.n_evals = 0
        self.history: List[Dict[str, Any]] = []
        self._best_f = float("inf")
        self._best_x: Optional[np.ndarray] = None

    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate objective with tracking.

        Args:
            x: Design point

        Returns:
            Objective value f(x)
        """
        self.n_evals += 1

        # Check cache first
        if self.cache is not None:
            cached = self.cache.get(x)
            if cached is not None:
                # Still track as evaluation but don't re-evaluate
                if cached < self._best_f:
                    self._best_f = cached
                    self._best_x = x.copy()
                return cached

        # Evaluate objective
        f = float(self.objective(x))

        # Track best
        if f < self._best_f:
            self._best_f = f
            self._best_x = x.copy()

        # Record history
        if self.record_history:
            self.history.append({
                "iteration": self.n_evals,
                "objective": f,
                "design": x.tolist() if hasattr(x, "tolist") else list(x),
            })

        # Store in cache
        if self.cache is not None:
            self.cache.store(x, f)

        return f

    @property
    def best_x(self) -> Optional[np.ndarray]:
        """Best design found so far."""
        return self._best_x

    @property
    def best_f(self) -> float:
        """Best objective value found so far."""
        return self._best_f

    def get_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get evaluation history.

        Args:
            last_n: If specified, return only last N entries

        Returns:
            List of history records
        """
        if last_n is not None:
            return self.history[-last_n:]
        return self.history


class GradientWrapper:
    """
    Wraps gradient function with evaluation counting.

    Usage:
        grad_wrapper = GradientWrapper(gradient_fn)
        result = optimizer.minimize(obj_fn, jac=grad_wrapper, ...)
        print(f"Gradient evaluations: {grad_wrapper.n_evals}")
    """

    def __init__(self, gradient: Callable[[np.ndarray], np.ndarray]):
        """
        Initialize gradient wrapper.

        Args:
            gradient: Function grad_f(x) -> ndarray
        """
        self.gradient = gradient
        self.n_evals = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate gradient with counting.

        Args:
            x: Design point

        Returns:
            Gradient vector
        """
        self.n_evals += 1
        return self.gradient(x)


class MultiObjectiveWrapper:
    """
    Wraps multiple objective functions for multi-objective optimization.

    Provides:
    - Evaluation counting
    - Pareto front tracking
    - History recording

    Usage:
        wrapper = MultiObjectiveWrapper([obj1, obj2])
        # pymoo calls wrapper.evaluate(x) -> array of objectives
    """

    def __init__(
        self,
        objectives: List[Callable[[np.ndarray], float]],
        cache: Optional["EvaluationCache"] = None,
        record_history: bool = True,
    ):
        """
        Initialize multi-objective wrapper.

        Args:
            objectives: List of objective functions [f1(x), f2(x), ...]
            cache: Optional evaluation cache
            record_history: Whether to record full history
        """
        self.objectives = objectives
        self.cache = cache
        self.record_history = record_history
        self.n_objectives = len(objectives)

        # Tracking state
        self.n_evals = 0
        self.history: List[Dict[str, Any]] = []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate all objectives.

        Args:
            x: Design point

        Returns:
            Array of objective values [f1(x), f2(x), ...]
        """
        self.n_evals += 1

        # Evaluate all objectives
        f = np.array([obj(x) for obj in self.objectives])

        # Record history
        if self.record_history:
            self.history.append({
                "iteration": self.n_evals,
                "objectives": f.tolist(),
                "design": x.tolist() if hasattr(x, "tolist") else list(x),
            })

        return f
