"""
Problem validation utilities.

Validates optimization problem formulations for:
- Completeness (all required fields)
- Consistency (bounds, constraints)
- Feasibility (initial points)
"""

from typing import List, Optional, Dict, Any
import numpy as np

from ..formulation.schema import OptimizationProblem, Variable, Constraint


def validate_problem(problem: OptimizationProblem) -> Dict[str, Any]:
    """
    Validate optimization problem.

    Checks:
    - At least one objective
    - At least one variable
    - Variable bounds are valid
    - Initial points are within bounds
    - Constraint types are valid

    Args:
        problem: OptimizationProblem to validate

    Returns:
        Dict with:
        - valid: bool
        - errors: List[str] (if any)
        - warnings: List[str] (if any)
    """
    errors = []
    warnings = []

    # Check objectives
    if not problem.objectives:
        errors.append("Problem must have at least one objective")

    # Check variables
    if not problem.variables:
        errors.append("Problem must have at least one variable")

    # Validate each variable
    for i, var in enumerate(problem.variables):
        # Check bounds
        if var.bounds is not None:
            lb, ub = var.bounds
            if lb is not None and ub is not None:
                if lb >= ub:
                    errors.append(
                        f"Variable {var.name}: lower_bound ({lb}) "
                        f">= upper_bound ({ub})"
                    )

            # Check initial value
            if var.initial is not None:
                if lb is not None and var.initial < lb:
                    warnings.append(
                        f"Variable {var.name}: initial value ({var.initial}) "
                        f"< lower_bound ({lb})"
                    )
                if ub is not None and var.initial > ub:
                    warnings.append(
                        f"Variable {var.name}: initial value ({var.initial}) "
                        f"> upper_bound ({ub})"
                    )

    # Validate constraints
    for i, cons in enumerate(problem.constraints):
        if cons.type not in ["equality", "inequality"]:
            errors.append(
                f"Constraint {cons.name}: invalid type '{cons.type}' "
                f"(must be 'equality' or 'inequality')"
            )

    # Check problem dimensions
    n_vars = len(problem.variables)
    n_objs = len(problem.objectives)
    n_cons = len(problem.constraints)

    if n_vars > 1000:
        warnings.append(
            f"Large problem: {n_vars} variables (may be slow)"
        )

    if n_cons > n_vars:
        warnings.append(
            f"Over-constrained: {n_cons} constraints, {n_vars} variables"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "n_variables": n_vars,
            "n_objectives": n_objs,
            "n_constraints": n_cons
        }
    }


def check_initial_feasibility(
    problem: OptimizationProblem,
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Check if initial point satisfies bounds.

    Args:
        problem: OptimizationProblem
        tolerance: Tolerance for bound checks

    Returns:
        Dict with feasibility information
    """
    violations = []

    for var in problem.variables:
        if var.initial is None or var.bounds is None:
            continue

        lb, ub = var.bounds

        if lb is not None:
            if var.initial < lb - tolerance:
                violations.append({
                    "variable": var.name,
                    "type": "lower_bound",
                    "value": var.initial,
                    "bound": lb
                })

        if ub is not None:
            if var.initial > ub + tolerance:
                violations.append({
                    "variable": var.name,
                    "type": "upper_bound",
                    "value": var.initial,
                    "bound": ub
                })

    return {
        "feasible": len(violations) == 0,
        "violations": violations,
        "n_violations": len(violations)
    }
