"""
Problem parsers for various input formats.

Supports:
- Natural language: LLM-powered parsing
- Python code: AST analysis
- Structured data: JSON, YAML, dict
- Existing formats: SciPy, Pyomo, CasADi (future)
"""

import ast
import json
import re
from typing import Dict, Any, Optional, List, Callable
import numpy as np

from ..formulation.schema import (
    OptimizationProblem,
    Objective,
    Variable,
    Constraint
)


class ProblemParser:
    """Parse various input formats to OptimizationProblem."""

    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize parser.

        Args:
            llm_client: LLM client for natural language parsing (optional)
        """
        self.llm_client = llm_client

    def from_natural_language(self, description: str) -> OptimizationProblem:
        """
        Parse natural language to optimization problem.

        Examples:
        - "minimize x^2 + 3x subject to x > 1"
        - "minimize drag on airfoil, maintain CL >= 0.5"
        - "maximize portfolio return, keep risk below 0.2"

        Uses LLM to extract:
        - Objectives (minimize/maximize)
        - Variables (with bounds if mentioned)
        - Constraints (equality/inequality)

        Args:
            description: Natural language problem description

        Returns:
            OptimizationProblem instance
        """
        # For now, implement simple pattern matching
        # Phase 6 Week 1: Will add LLM integration

        problem_data = self._parse_simple_math_expression(description)
        return self.from_structured(problem_data)

    def _parse_simple_math_expression(self, description: str) -> Dict[str, Any]:
        """
        Simple pattern matching for common mathematical expressions.

        This is a placeholder - will be replaced with LLM parsing.
        """
        # Extract minimize/maximize
        is_minimize = "minimize" in description.lower() or "min" in description.lower()
        sense = "minimize" if is_minimize else "maximize"

        # Extract variable names (very simple: look for single letters)
        var_pattern = r'\b([a-z])\b'
        variables = list(set(re.findall(var_pattern, description.lower())))

        # Extract constraints (simple pattern: "x > 1", "x >= 0.5", etc.)
        constraint_pattern = r'([a-z])\s*(>=|>|<=|<|=)\s*([\d.]+)'
        constraints_raw = re.findall(constraint_pattern, description.lower())

        # Build problem data
        problem_data = {
            "objectives": [{
                "name": "objective",
                "sense": sense
            }],
            "variables": [
                {"name": var, "bounds": [-10.0, 10.0], "initial": 0.0}
                for var in variables
            ],
            "constraints": []
        }

        # Add constraints
        for var, op, value in constraints_raw:
            constraint_type = "inequality"
            if op == "=":
                constraint_type = "equality"

            problem_data["constraints"].append({
                "name": f"{var}_{op}_{value}",
                "type": constraint_type,
                "expression": f"{var} {op} {value}"
            })

        return problem_data

    def from_code(self, code: str) -> OptimizationProblem:
        """
        Parse Python code to optimization problem.

        Accepts:
        - Function definitions: def objective(x): return x**2
        - Class-based problem definitions
        - NumPy expressions

        Args:
            code: Python code string

        Returns:
            OptimizationProblem instance
        """
        # Parse code using AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python code: {e}")

        # Extract function definitions
        functions = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = node

        # Look for objective function
        if "objective" in functions:
            # Analyze function to extract problem structure
            # For now, create a simple problem
            problem_data = {
                "objectives": [{
                    "name": "objective",
                    "sense": "minimize"  # Default
                }],
                "variables": [{
                    "name": "x",
                    "bounds": [-10.0, 10.0],
                    "initial": 0.0
                }],
                "constraints": []
            }
            return self.from_structured(problem_data)

        raise ValueError("No objective function found in code")

    def from_structured(self, data: Dict[str, Any]) -> OptimizationProblem:
        """
        Parse structured data (dict, JSON, YAML) to optimization problem.

        Standard schema:
        {
          "objectives": [{"name": "f", "sense": "minimize", ...}],
          "variables": [{"name": "x", "bounds": [0, 10], "initial": 5}],
          "constraints": [{"name": "c1", "type": "inequality", ...}]
        }

        Args:
            data: Dictionary with problem specification

        Returns:
            OptimizationProblem instance
        """
        # Parse objectives
        objectives = []
        for obj_data in data.get("objectives", []):
            objectives.append(Objective(**obj_data))

        # Parse variables
        variables = []
        for var_data in data.get("variables", []):
            # Handle bounds format - keep as tuple
            if "bounds" in var_data and isinstance(var_data["bounds"], list):
                var_data["bounds"] = tuple(var_data["bounds"])
            # Handle separate lower/upper bounds
            elif "lower_bound" in var_data and "upper_bound" in var_data:
                var_data["bounds"] = (var_data["lower_bound"], var_data["upper_bound"])
                var_data.pop("lower_bound", None)
                var_data.pop("upper_bound", None)
            variables.append(Variable(**var_data))

        # Parse constraints
        constraints = []
        for cons_data in data.get("constraints", []):
            constraints.append(Constraint(**cons_data))

        # Determine problem type
        n_objectives = len(objectives)
        n_constraints = len(constraints)

        if n_objectives == 1 and n_constraints == 0:
            problem_type = "nonlinear_single"
        elif n_objectives == 1 and n_constraints > 0:
            problem_type = "nonlinear_single"  # Still single objective
        elif n_objectives > 1:
            problem_type = "nonlinear_multi"
        else:
            problem_type = "nonlinear_single"

        return OptimizationProblem(
            problem_type=problem_type,
            objectives=objectives,
            variables=variables,
            constraints=constraints,
            metadata=data.get("metadata", {})
        )

    def from_scipy_format(
        self,
        bounds: List[tuple],
        constraints: Optional[List[Dict]] = None
    ) -> OptimizationProblem:
        """
        Import from SciPy minimize format.

        Args:
            bounds: List of (lower, upper) tuples
            constraints: List of constraint dicts (scipy format)

        Returns:
            OptimizationProblem instance
        """
        # Create variables from bounds
        variables = [
            Variable(
                name=f"x{i}",
                bounds=(lb, ub),
                initial=(lb + ub) / 2 if lb is not None and ub is not None else 0.0
            )
            for i, (lb, ub) in enumerate(bounds)
        ]

        # Convert scipy constraints to our format
        constraint_objects = []
        if constraints:
            for i, cons in enumerate(constraints):
                cons_type = cons.get("type", "ineq")
                constraint_type = "inequality" if cons_type == "ineq" else "equality"

                constraint_objects.append(Constraint(
                    name=f"constraint_{i}",
                    type=constraint_type,
                    expression=f"constraint_{i}"  # Placeholder
                ))

        return OptimizationProblem(
            problem_type="nonlinear_single",
            objectives=[Objective(name="objective", sense="minimize")],
            variables=variables,
            constraints=constraint_objects
        )

    def from_dict_simple(
        self,
        n_variables: int,
        bounds: Optional[List[tuple]] = None,
        n_constraints: int = 0,
        objective_sense: str = "minimize"
    ) -> OptimizationProblem:
        """
        Create simple problem from basic parameters.

        Args:
            n_variables: Number of design variables
            bounds: Optional list of (lower, upper) bounds
            n_constraints: Number of constraints
            objective_sense: "minimize" or "maximize"

        Returns:
            OptimizationProblem instance
        """
        if bounds is None:
            bounds = [(-10.0, 10.0)] * n_variables

        variables = [
            Variable(
                name=f"x{i}",
                bounds=(lb, ub),
                initial=(lb + ub) / 2
            )
            for i, (lb, ub) in enumerate(bounds)
        ]

        constraints = [
            Constraint(
                name=f"c{i}",
                type="inequality",
                expression=f"constraint_{i}"
            )
            for i in range(n_constraints)
        ]

        return OptimizationProblem(
            problem_type="nonlinear_single",
            objectives=[Objective(name="objective", sense=objective_sense)],
            variables=variables,
            constraints=constraints
        )


def parse_problem(
    description: Any,
    format: str = "auto"
) -> OptimizationProblem:
    """
    Convenience function to parse problem from any format.

    Args:
        description: Problem description (string, dict, etc.)
        format: Format hint ("natural", "code", "dict", "auto")

    Returns:
        OptimizationProblem instance
    """
    parser = ProblemParser()

    if format == "auto":
        if isinstance(description, dict):
            format = "dict"
        elif isinstance(description, str):
            # Try to determine if it's code or natural language
            if "def " in description or "class " in description:
                format = "code"
            else:
                format = "natural"

    if format == "natural":
        return parser.from_natural_language(description)
    elif format == "code":
        return parser.from_code(description)
    elif format == "dict":
        return parser.from_structured(description)
    else:
        raise ValueError(f"Unknown format: {format}")
