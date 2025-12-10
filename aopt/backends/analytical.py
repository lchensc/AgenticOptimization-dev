"""
Analytical test functions for optimization benchmarking.

Provides fast, cheap evaluation functions with known optima
for testing and developing the agentic optimization platform.
"""

import numpy as np
from typing import Tuple, Optional


class AnalyticalFunction:
    """Base class for analytical test functions."""

    def __init__(self, dimension: int):
        """
        Initialize analytical function.

        Args:
            dimension: Problem dimensionality
        """
        self.dimension = dimension
        self.n_evaluations = 0
        self.n_gradient_calls = 0

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate objective function.

        Args:
            x: Design vector

        Returns:
            Objective value
        """
        raise NotImplementedError

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute gradient analytically.

        Args:
            x: Design vector

        Returns:
            Gradient vector
        """
        raise NotImplementedError

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get variable bounds.

        Returns:
            (lower_bounds, upper_bounds) tuple
        """
        raise NotImplementedError

    def get_optimum(self) -> Tuple[np.ndarray, float]:
        """
        Get known global optimum.

        Returns:
            (optimal_x, optimal_value) tuple
        """
        raise NotImplementedError

    def reset_counters(self):
        """Reset evaluation counters."""
        self.n_evaluations = 0
        self.n_gradient_calls = 0


class Rosenbrock(AnalyticalFunction):
    """
    Rosenbrock function - classic optimization benchmark.

    f(x) = sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

    Global minimum: f(1, 1, ..., 1) = 0

    Properties:
    - Narrow curved valley
    - Easy to find valley, hard to converge to minimum
    - Tests optimizer's ability to navigate curved surfaces
    """

    def __init__(self, dimension: int = 2):
        """
        Initialize Rosenbrock function.

        Args:
            dimension: Problem dimensionality (default: 2)
        """
        super().__init__(dimension)
        self.name = f"Rosenbrock-{dimension}D"

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate Rosenbrock function."""
        self.n_evaluations += 1
        x = np.asarray(x)

        # Sum of valley terms
        sum_valley = np.sum(100.0 * (x[1:] - x[:-1]**2)**2)
        # Sum of offset terms
        sum_offset = np.sum((1.0 - x[:-1])**2)

        return sum_valley + sum_offset

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute analytical gradient."""
        self.n_gradient_calls += 1
        x = np.asarray(x)
        grad = np.zeros_like(x)

        # Interior points
        grad[:-1] = -400.0 * x[:-1] * (x[1:] - x[:-1]**2) - 2.0 * (1.0 - x[:-1])
        grad[1:] += 200.0 * (x[1:] - x[:-1]**2)

        return grad

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounds: [-5, 10] for all variables."""
        lower = np.full(self.dimension, -5.0)
        upper = np.full(self.dimension, 10.0)
        return lower, upper

    def get_optimum(self) -> Tuple[np.ndarray, float]:
        """Get known optimum: x* = [1, 1, ..., 1], f* = 0."""
        x_opt = np.ones(self.dimension)
        f_opt = 0.0
        return x_opt, f_opt


class Sphere(AnalyticalFunction):
    """
    Sphere function - simplest optimization benchmark.

    f(x) = sum_{i=1}^{n} x_i^2

    Global minimum: f(0, 0, ..., 0) = 0

    Properties:
    - Convex, unimodal
    - Trivial for gradient-based methods
    - Useful for testing basic optimizer functionality
    """

    def __init__(self, dimension: int = 2):
        """
        Initialize Sphere function.

        Args:
            dimension: Problem dimensionality (default: 2)
        """
        super().__init__(dimension)
        self.name = f"Sphere-{dimension}D"

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate Sphere function."""
        self.n_evaluations += 1
        x = np.asarray(x)
        return np.sum(x**2)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute analytical gradient."""
        self.n_gradient_calls += 1
        x = np.asarray(x)
        return 2.0 * x

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounds: [-5, 5] for all variables."""
        lower = np.full(self.dimension, -5.0)
        upper = np.full(self.dimension, 5.0)
        return lower, upper

    def get_optimum(self) -> Tuple[np.ndarray, float]:
        """Get known optimum: x* = [0, 0, ..., 0], f* = 0."""
        x_opt = np.zeros(self.dimension)
        f_opt = 0.0
        return x_opt, f_opt


class ConstrainedRosenbrock(AnalyticalFunction):
    """
    Rosenbrock with constraints for testing constrained optimization.

    Minimize: f(x) = (1-x1)^2 + 100*(x2-x1^2)^2
    Subject to: x1^2 + x2^2 <= 2  (inside circle of radius sqrt(2))

    Known constrained optimum: approximately x* = [0.786, 0.618], f* = 0.046
    """

    def __init__(self):
        """Initialize constrained Rosenbrock (2D only)."""
        super().__init__(dimension=2)
        self.name = "ConstrainedRosenbrock-2D"

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate Rosenbrock function."""
        self.n_evaluations += 1
        x = np.asarray(x)
        return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute analytical gradient."""
        self.n_gradient_calls += 1
        x = np.asarray(x)
        grad = np.zeros(2)
        grad[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0]**2)
        grad[1] = 200.0 * (x[1] - x[0]**2)
        return grad

    def evaluate_constraint(self, x: np.ndarray) -> float:
        """
        Evaluate circle constraint.

        Returns:
            Constraint value (should be <= 0)
        """
        x = np.asarray(x)
        return x[0]**2 + x[1]**2 - 2.0

    def constraint_gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute constraint gradient."""
        x = np.asarray(x)
        return np.array([2.0 * x[0], 2.0 * x[1]])

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounds: [-2, 2] for both variables."""
        lower = np.array([-2.0, -2.0])
        upper = np.array([2.0, 2.0])
        return lower, upper

    def get_optimum(self) -> Tuple[np.ndarray, float]:
        """Get approximate constrained optimum."""
        x_opt = np.array([0.786, 0.618])
        f_opt = 0.046
        return x_opt, f_opt


# Factory function for easy access
def get_analytical_function(name: str, dimension: Optional[int] = None) -> AnalyticalFunction:
    """
    Get analytical test function by name.

    Args:
        name: Function name ("rosenbrock", "sphere", "constrained_rosenbrock")
        dimension: Problem dimension (ignored for constrained_rosenbrock)

    Returns:
        AnalyticalFunction instance

    Raises:
        ValueError: If function name not recognized
    """
    name_lower = name.lower()

    if name_lower == "rosenbrock":
        dim = dimension if dimension is not None else 2
        return Rosenbrock(dimension=dim)
    elif name_lower == "sphere":
        dim = dimension if dimension is not None else 2
        return Sphere(dimension=dim)
    elif name_lower == "constrained_rosenbrock":
        return ConstrainedRosenbrock()
    else:
        raise ValueError(
            f"Unknown function: {name}. "
            f"Available: 'rosenbrock', 'sphere', 'constrained_rosenbrock'"
        )
