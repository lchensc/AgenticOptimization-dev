"""
Problem information tools for LLM-driven optimization.

v0.5.0 (v0.2.0 redesign): Code-execution model
- LLM writes Python code directly using paola.objective(), scipy, optuna, etc.
- Use paola.objective(problem_id) to get RecordingObjective callable
- See paola.api module for Recording API

This module provides:
- get_problem_info(): Get problem characteristics for optimizer selection
"""

from typing import Dict, Any
from langchain_core.tools import tool

from .schemas import (
    normalize_problem_id,
    ProblemIdType,
    GetProblemInfoArgs,
)


@tool(args_schema=GetProblemInfoArgs)
def get_problem_info(problem_id: ProblemIdType) -> Dict[str, Any]:
    """
    Get problem characteristics for optimizer selection.

    Use this to understand a problem before deciding on optimizer
    selection and configuration.

    Args:
        problem_id: Problem ID (int or str, auto-normalized)

    Returns:
        Dict with:
            success: bool
            problem_id: int
            dimension: int - Number of variables
            bounds: List - Variable bounds (truncated if >10)
            bounds_summary: str - Human-readable bounds description
            is_constrained: bool - Has constraints
            has_gradient: bool - Gradient available
            description: str - Problem description
    """
    from .evaluation import _get_problem

    try:
        # Normalize problem_id (handles str/int from LLM) - v0.4.5
        problem_id = normalize_problem_id(problem_id)
        problem = _get_problem(problem_id)

        nlp_problem = None
        if hasattr(problem, "problem"):
            nlp_problem = problem.problem

        if nlp_problem:
            bounds = nlp_problem.bounds
            n_vars = nlp_problem.dimension
            bounds_center = nlp_problem.get_bounds_center()
            bounds_width = nlp_problem.get_bounds_width()

            n_ineq = len(nlp_problem.inequality_constraints) if nlp_problem.inequality_constraints else 0
            n_eq = len(nlp_problem.equality_constraints) if nlp_problem.equality_constraints else 0

            if all(b[0] == bounds[0][0] and b[1] == bounds[0][1] for b in bounds):
                bounds_summary = f"uniform [{bounds[0][0]}, {bounds[0][1]}] for all {n_vars} variables"
            else:
                lb_min, lb_max = min(b[0] for b in bounds), max(b[0] for b in bounds)
                ub_min, ub_max = min(b[1] for b in bounds), max(b[1] for b in bounds)
                bounds_summary = f"lower: [{lb_min}, {lb_max}], upper: [{ub_min}, {ub_max}]"

            return {
                "success": True,
                "problem_id": problem_id,
                "dimension": n_vars,
                "bounds": bounds[:10] if n_vars > 10 else bounds,
                "bounds_truncated": n_vars > 10,
                "bounds_summary": bounds_summary,
                "bounds_center": bounds_center[:10] if n_vars > 10 else bounds_center,
                "bounds_width": bounds_width[:10] if n_vars > 10 else bounds_width,
                "num_inequality_constraints": n_ineq,
                "num_equality_constraints": n_eq,
                "is_constrained": n_ineq > 0 or n_eq > 0,
                "has_gradient": hasattr(problem, "gradient"),
                "domain_hint": nlp_problem.domain_hint,
                "description": nlp_problem.description or "No description",
            }

        elif hasattr(problem, "bounds"):
            bounds = problem.bounds
            n_vars = len(bounds)

            return {
                "success": True,
                "problem_id": problem_id,
                "dimension": n_vars,
                "bounds": bounds[:10] if n_vars > 10 else bounds,
                "bounds_truncated": n_vars > 10,
                "bounds_summary": f"{n_vars} variables",
                "bounds_center": [(b[0] + b[1]) / 2 for b in bounds[:10]],
                "bounds_width": [b[1] - b[0] for b in bounds[:10]],
                "num_inequality_constraints": 0,
                "num_equality_constraints": 0,
                "is_constrained": False,
                "has_gradient": hasattr(problem, "gradient"),
                "domain_hint": None,
                "description": getattr(problem, "description", "No description"),
            }

        elif hasattr(problem, "get_bounds"):
            lb, ub = problem.get_bounds()
            bounds = [[float(lb[i]), float(ub[i])] for i in range(len(lb))]
            n_vars = len(bounds)

            return {
                "success": True,
                "problem_id": problem_id,
                "dimension": n_vars,
                "bounds": bounds[:10] if n_vars > 10 else bounds,
                "bounds_truncated": n_vars > 10,
                "bounds_summary": f"{n_vars} variables",
                "bounds_center": [(b[0] + b[1]) / 2 for b in bounds[:10]],
                "bounds_width": [b[1] - b[0] for b in bounds[:10]],
                "num_inequality_constraints": 0,
                "num_equality_constraints": 0,
                "is_constrained": hasattr(problem, "constraints"),
                "has_gradient": hasattr(problem, "gradient"),
                "domain_hint": None,
                "description": getattr(problem, "name", "Analytical function"),
            }

        else:
            return {
                "success": False,
                "message": "Problem does not have bounds defined",
            }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error getting problem info: {str(e)}",
        }
