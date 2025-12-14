"""
External evaluator functions for PAOLA testing.

This file contains benchmark optimization functions that can be registered
in the PAOLA Foundry for testing the evaluator registration system.

Usage:
    1. Register via CLI:
       paola> /register evaluators.py

    2. Or use directly:
       from evaluators import rosenbrock
       result = rosenbrock([1.0, 1.0])  # Should be 0.0 at optimum
"""

import numpy as np


def rosenbrock(x):
    """
    Rosenbrock function (Banana function).

    Classic test function for optimization algorithms. The global minimum
    is at (1, 1, ..., 1) with f(x*) = 0.

    Args:
        x: Design vector (numpy array or list), minimum dimension 2

    Returns:
        float: Objective value

    Example:
        >>> rosenbrock([1.0, 1.0])
        0.0
        >>> rosenbrock([0.0, 0.0])
        1.0
    """
    x = np.atleast_1d(x)

    if len(x) < 2:
        raise ValueError("Rosenbrock function requires at least 2 dimensions")

    result = 0.0
    for i in range(len(x) - 1):
        result += 100.0 * (x[i+1] - x[i]**2)**2 + (1.0 - x[i])**2

    return float(result)


def sphere(x):
    """
    Sphere function (sum of squares).

    Simple convex function. Global minimum at origin with f(x*) = 0.

    Args:
        x: Design vector (numpy array or list)

    Returns:
        float: Sum of squared elements

    Example:
        >>> sphere([0.0, 0.0, 0.0])
        0.0
        >>> sphere([1.0, 2.0, 3.0])
        14.0
    """
    x = np.atleast_1d(x)
    return float(np.sum(x**2))


def rastrigin(x):
    """
    Rastrigin function (highly multimodal).

    Difficult test function with many local minima. Global minimum
    at origin with f(x*) = 0.

    Args:
        x: Design vector (numpy array or list)

    Returns:
        float: Objective value

    Example:
        >>> rastrigin([0.0, 0.0])
        0.0
    """
    x = np.atleast_1d(x)
    n = len(x)
    A = 10.0
    return float(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


def ackley(x):
    """
    Ackley function (multimodal with steep ridges).

    Global minimum at origin with f(x*) = 0.

    Args:
        x: Design vector (numpy array or list)

    Returns:
        float: Objective value

    Example:
        >>> ackley([0.0, 0.0])
        0.0
    """
    x = np.atleast_1d(x)
    n = len(x)

    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))

    term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)

    return float(term1 + term2 + 20.0 + np.e)


def beale(x):
    """
    Beale function (2D test function).

    Global minimum at (3, 0.5) with f(x*) = 0.

    Args:
        x: Design vector (numpy array or list), must be 2D

    Returns:
        float: Objective value

    Example:
        >>> beale([3.0, 0.5])
        0.0
    """
    x = np.atleast_1d(x)

    if len(x) != 2:
        raise ValueError("Beale function requires exactly 2 dimensions")

    term1 = (1.5 - x[0] + x[0] * x[1])**2
    term2 = (2.25 - x[0] + x[0] * x[1]**2)**2
    term3 = (2.625 - x[0] + x[0] * x[1]**3)**2

    return float(term1 + term2 + term3)


if __name__ == "__main__":
    """Test all functions at their known optima."""
    print("Testing evaluator functions at known optima:\n")

    # Rosenbrock
    x_opt = [1.0, 1.0]
    result = rosenbrock(x_opt)
    print(f"rosenbrock({x_opt}) = {result:.10f} (expected: 0.0)")

    # Sphere
    x_opt = [0.0, 0.0, 0.0]
    result = sphere(x_opt)
    print(f"sphere({x_opt}) = {result:.10f} (expected: 0.0)")

    # Rastrigin
    x_opt = [0.0, 0.0]
    result = rastrigin(x_opt)
    print(f"rastrigin({x_opt}) = {result:.10f} (expected: 0.0)")

    # Ackley
    x_opt = [0.0, 0.0]
    result = ackley(x_opt)
    print(f"ackley({x_opt}) = {result:.10e} (expected: 0.0)")

    # Beale
    x_opt = [3.0, 0.5]
    result = beale(x_opt)
    print(f"beale({x_opt}) = {result:.10e} (expected: 0.0)")

    print("\n✓ All functions evaluated successfully")
