"""
Portfolio Optimization Evaluator
================================

A user-provided evaluator for portfolio optimization.
Written from first principles - no PAOLA-specific knowledge required.

Problem: Find optimal asset allocation to maximize risk-adjusted return (Sharpe Ratio)
         subject to a minimum bond allocation constraint.

Variables: 5 asset weights (stocks_us, stocks_intl, bonds_govt, bonds_corp, commodities)
Objective: Maximize Sharpe Ratio
Constraints:
  - Weights in [0, 1] (no shorting)
  - At least 20% in bonds (bonds_govt + bonds_corp >= 0.20)
"""

import numpy as np

# =============================================================================
# ASSET DATA
# =============================================================================

ASSETS = ["stocks_us", "stocks_intl", "bonds_govt", "bonds_corp", "commodities"]
N_ASSETS = len(ASSETS)

# Expected monthly returns (historical averages)
EXPECTED_RETURNS = np.array([0.008, 0.009, 0.003, 0.004, 0.006])

# Monthly volatilities
VOLATILITIES = np.array([0.045, 0.055, 0.015, 0.020, 0.060])

# Correlation matrix
CORRELATIONS = np.array([
    [1.00, 0.70, 0.20, 0.30, 0.40],
    [0.70, 1.00, 0.15, 0.25, 0.50],
    [0.20, 0.15, 1.00, 0.80, 0.10],
    [0.30, 0.25, 0.80, 1.00, 0.20],
    [0.40, 0.50, 0.10, 0.20, 1.00],
])

# Covariance matrix
COVARIANCE = np.outer(VOLATILITIES, VOLATILITIES) * CORRELATIONS

# Risk-free rate (annual)
RISK_FREE_RATE = 0.02

# Bond indices
BOND_INDICES = [2, 3]  # bonds_govt, bonds_corp


# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================

def evaluate(x: np.ndarray) -> float:
    """
    Compute negative Sharpe ratio for minimization.

    Args:
        x: Asset weights, length 5

    Returns:
        Negative Sharpe ratio (minimization convention)
    """
    x = np.asarray(x)

    # Normalize weights to sum to 1
    total = np.sum(x)
    if total <= 0:
        return 1e10  # Invalid
    weights = x / total

    # Portfolio return (annualized)
    portfolio_return = np.dot(weights, EXPECTED_RETURNS) * 12

    # Portfolio volatility (annualized)
    variance = np.dot(weights, np.dot(COVARIANCE, weights)) * 12
    volatility = np.sqrt(max(variance, 1e-10))

    # Sharpe ratio
    sharpe = (portfolio_return - RISK_FREE_RATE) / volatility

    # Return negative for minimization
    return -sharpe


# =============================================================================
# CONSTRAINT FUNCTION
# =============================================================================

def constraint_min_bonds(x: np.ndarray) -> float:
    """
    Minimum bond allocation constraint.

    Constraint: bonds_govt + bonds_corp >= 0.20

    Returns g(x) where g(x) >= 0 means feasible.

    Args:
        x: Asset weights, length 5

    Returns:
        bond_allocation - 0.20 (>= 0 when satisfied)
    """
    x = np.asarray(x)

    # Normalize weights
    total = np.sum(x)
    if total <= 0:
        return -1.0  # Infeasible
    weights = x / total

    # Bond allocation
    bond_allocation = weights[BOND_INDICES[0]] + weights[BOND_INDICES[1]]

    # Return g(x) = allocation - 0.20, feasible when >= 0
    return bond_allocation - 0.20


# =============================================================================
# PROBLEM SPECIFICATION (for reference)
# =============================================================================

DIMENSION = N_ASSETS
BOUNDS = [[0.0, 1.0]] * N_ASSETS


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test objective
    equal_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    print(f"Equal weights: objective = {evaluate(equal_weights):.4f}")
    print(f"Equal weights: constraint = {constraint_min_bonds(equal_weights):.4f}")

    # Test constraint violation
    all_stocks = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
    print(f"\nAll stocks: objective = {evaluate(all_stocks):.4f}")
    print(f"All stocks: constraint = {constraint_min_bonds(all_stocks):.4f} (should be < 0)")
