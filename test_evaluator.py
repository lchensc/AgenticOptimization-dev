"""
Test evaluator for /register_eval command.
"""


def rosenbrock_2d(x):
    """
    2D Rosenbrock function.

    Global minimum at (1, 1) with f = 0.
    Typical bounds: [-5, 10] for each variable.
    """
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
