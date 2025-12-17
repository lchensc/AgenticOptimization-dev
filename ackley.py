import numpy as np

def evaluate(x):
    """
    10-dimensional Ackley function.
    Global minimum at x = [0, 0, ..., 0] with f(x) = 0.
    
    Args:
        x: array of shape (10,) with values in [-5, 5]
    
    Returns:
        float: function value
    """
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1/d)) - np.exp(sum2/d) + 20 + np.e