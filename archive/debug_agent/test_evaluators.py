"""
Test Evaluators for Agentic Architecture Demo.

This file contains evaluator functions for testing the compiled evaluator
architecture with semantic analysis and auto-generated variable extractors.
"""

import numpy as np


def rosenbrock_2d(x):
    """
    Classic 2D Rosenbrock function.

    This is the "banana function" with a narrow curved valley.
    Global minimum: f(1, 1) = 0

    Args:
        x: Design vector [x0, x1]

    Returns:
        Objective value (scalar)
    """
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def sphere_function(x):
    """
    N-dimensional sphere function.

    Simple convex bowl-shaped function.
    Global minimum: f(0, 0, ..., 0) = 0

    Args:
        x: Design vector of any dimension

    Returns:
        Sum of squares
    """
    return float(np.sum(x**2))


def rastrigin_2d(x):
    """
    2D Rastrigin function - highly multimodal.

    This function has many local minima.
    Global minimum: f(0, 0) = 0

    Args:
        x: Design vector [x0, x1]

    Returns:
        Objective value
    """
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def ackley_2d(x):
    """
    2D Ackley function - nearly flat outer region.

    This function is challenging for optimization algorithms.
    Global minimum: f(0, 0) = 0

    Args:
        x: Design vector [x0, x1]

    Returns:
        Objective value
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)

    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))

    term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)

    return term1 + term2 + a + np.exp(1)


def himmelblau(x):
    """
    Himmelblau's function - has 4 local minima.

    All four minima have the same value.
    Global minima: f(3, 2) = f(-2.805, 3.131) = f(-3.779, -3.283) = f(3.584, -1.848) = 0

    Args:
        x: Design vector [x0, x1]

    Returns:
        Objective value
    """
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
