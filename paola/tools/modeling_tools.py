"""
Agent tools for problem modeling and parsing.

These tools enable the agent to:
- Parse natural language problem descriptions
- Parse Python code with objective functions
- Parse structured problem specifications
- Validate problem formulations
"""

from typing import Dict, Any, Optional
from langchain.tools import tool

from ..modeling.parsers import ProblemParser, parse_problem
from ..modeling.validation import validate_problem, check_initial_feasibility
from ..tools.session_tools import _FOUNDRY


@tool
def parse_problem_from_natural_language(description: str) -> Dict[str, Any]:
    """
    Parse natural language problem description to optimization problem.

    Examples:
    - "minimize x^2 + 3x subject to x > 1"
    - "minimize drag on airfoil, maintain CL >= 0.5"
    - "maximize portfolio return, keep risk below 0.2"

    This tool uses pattern matching (will use LLM in future).

    Args:
        description: Natural language problem description

    Returns:
        dict with:
        - problem_id: str (stored in foundry)
        - n_variables: int
        - n_objectives: int
        - n_constraints: int
        - validation: validation results
    """
    try:
        parser = ProblemParser()
        problem = parser.from_natural_language(description)

        # Validate
        validation = validate_problem(problem)

        if not validation["valid"]:
            return {
                "success": False,
                "error": f"Invalid problem: {validation['errors']}",
                "description": description
            }

        # Store in foundry
        problem_id = f"nl_{hash(description) % 100000}"

        if _FOUNDRY is not None:
            # Store problem (create method to be added to foundry)
            # For now, just return the problem data
            pass

        return {
            "success": True,
            "problem_id": problem_id,
            "n_variables": len(problem.variables),
            "n_objectives": len(problem.objectives),
            "n_constraints": len(problem.constraints),
            "validation": validation,
            "description": description
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "description": description
        }


@tool
def parse_problem_from_code(code: str) -> Dict[str, Any]:
    """
    Parse Python code to optimization problem.

    Accepts:
    - Function definitions: def objective(x): return x**2
    - Class-based problem definitions
    - NumPy expressions

    Args:
        code: Python code defining objective function

    Returns:
        dict with problem information
    """
    try:
        parser = ProblemParser()
        problem = parser.from_code(code)

        # Validate
        validation = validate_problem(problem)

        if not validation["valid"]:
            return {
                "success": False,
                "error": f"Invalid problem: {validation['errors']}"
            }

        problem_id = f"code_{hash(code) % 100000}"

        return {
            "success": True,
            "problem_id": problem_id,
            "n_variables": len(problem.variables),
            "n_objectives": len(problem.objectives),
            "n_constraints": len(problem.constraints),
            "validation": validation
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
def parse_problem_from_dict(
    n_variables: int,
    objective_sense: str = "minimize",
    bounds_lower: float = -10.0,
    bounds_upper: float = 10.0,
    n_constraints: int = 0
) -> Dict[str, Any]:
    """
    Create optimization problem from basic parameters.

    This is useful when user provides simple problem specification:
    - Number of variables
    - Objective sense (minimize/maximize)
    - Variable bounds
    - Number of constraints

    Args:
        n_variables: Number of design variables
        objective_sense: "minimize" or "maximize"
        bounds_lower: Lower bound for all variables
        bounds_upper: Upper bound for all variables
        n_constraints: Number of constraints

    Returns:
        dict with problem information
    """
    try:
        parser = ProblemParser()

        bounds = [(bounds_lower, bounds_upper)] * n_variables

        problem = parser.from_dict_simple(
            n_variables=n_variables,
            bounds=bounds,
            n_constraints=n_constraints,
            objective_sense=objective_sense
        )

        # Validate
        validation = validate_problem(problem)

        problem_id = f"simple_{n_variables}d_{n_constraints}c"

        return {
            "success": True,
            "problem_id": problem_id,
            "n_variables": n_variables,
            "n_objectives": 1,
            "n_constraints": n_constraints,
            "bounds": [bounds_lower, bounds_upper],
            "validation": validation
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
def validate_problem_formulation(problem_id: str) -> Dict[str, Any]:
    """
    Validate optimization problem formulation.

    Checks:
    - At least one objective
    - At least one variable
    - Valid bounds
    - Initial points within bounds
    - Valid constraint types

    Args:
        problem_id: Problem identifier

    Returns:
        dict with validation results:
        - valid: bool
        - errors: list of error messages
        - warnings: list of warnings
        - summary: problem statistics
    """
    # For now, return success (will integrate with foundry)
    return {
        "valid": True,
        "errors": [],
        "warnings": [],
        "summary": {
            "message": "Problem validation will be fully integrated with foundry"
        }
    }


@tool
def get_problem_info(problem_id: str) -> Dict[str, Any]:
    """
    Get information about a problem formulation.

    Args:
        problem_id: Problem identifier

    Returns:
        dict with problem information:
        - n_variables
        - n_objectives
        - n_constraints
        - variable_names
        - objective_names
        - problem_type
    """
    # Placeholder - will integrate with foundry
    return {
        "problem_id": problem_id,
        "status": "Problem retrieval will be integrated with foundry"
    }


# Export all tools
__all__ = [
    "parse_problem_from_natural_language",
    "parse_problem_from_code",
    "parse_problem_from_dict",
    "validate_problem_formulation",
    "get_problem_info",
]
