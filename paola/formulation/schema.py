"""
Optimization problem schema using Pydantic.

Defines the universal OptimizationProblem structure that is extensible
for future problem types (integer, stochastic, etc.).
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional


class Objective(BaseModel):
    """Single optimization objective."""

    name: str = Field(..., description="Objective name (e.g., 'drag', 'weight')")
    sense: Literal["minimize", "maximize"] = Field(..., description="Optimization direction")

    class Config:
        frozen = True  # Immutable


class Variable(BaseModel):
    """Design variable definition."""

    name: str = Field(..., description="Variable name (e.g., 'x1', 'alpha')")
    type: Literal["continuous", "integer", "binary"] = Field(
        default="continuous",
        description="Variable type"
    )
    bounds: tuple[float, float] = Field(..., description="Lower and upper bounds")
    initial: Optional[float] = Field(None, description="Initial value (optional)")

    class Config:
        frozen = True  # Immutable


class Constraint(BaseModel):
    """Optimization constraint."""

    name: str = Field(..., description="Constraint name (e.g., 'lift', 'stress')")
    type: Literal["equality", "inequality"] = Field(..., description="Constraint type")
    expression: str = Field(
        ...,
        description="Constraint expression (e.g., 'CL - 0.8 >= 0')"
    )
    bound: Optional[float] = Field(
        None,
        description="Bound value for inequality constraints"
    )

    class Config:
        frozen = True  # Immutable


class OptimizationProblem(BaseModel):
    """
    Universal optimization problem schema.

    Extensible design supports:
    - Milestone 1: nonlinear_single, nonlinear_multi
    - Future: linear, mixed_integer, stochastic, robust

    Example:
        >>> problem = OptimizationProblem(
        ...     problem_type="nonlinear_single",
        ...     objectives=[Objective(name="drag", sense="minimize")],
        ...     variables=[
        ...         Variable(name="x1", bounds=(0, 10)),
        ...         Variable(name="x2", bounds=(-5, 5))
        ...     ],
        ...     constraints=[
        ...         Constraint(name="lift", type="inequality",
        ...                    expression="CL >= 0.8", bound=0.8)
        ...     ]
        ... )
    """

    # Problem classification
    problem_type: Literal[
        "nonlinear_single",      # Milestone 1 ✓
        "nonlinear_multi",       # Milestone 1 ✓
        "linear",                # Future
        "mixed_integer",         # Future
        "stochastic",            # Future
        "robust"                 # Future
    ] = Field(..., description="Problem classification")

    # Core problem definition
    objectives: list[Objective] = Field(
        ...,
        min_items=1,
        description="One or more objectives"
    )

    variables: list[Variable] = Field(
        ...,
        min_items=1,
        description="Design variables"
    )

    constraints: list[Constraint] = Field(
        default_factory=list,
        description="Constraints (optional)"
    )

    # Problem properties (inferred or specified)
    properties: dict = Field(
        default_factory=dict,
        description="Problem properties (convex, smooth, expensive, etc.)"
    )

    # Problem ID for cache isolation
    id: Optional[str] = Field(
        None,
        description="Problem identifier for cache management"
    )

    @property
    def is_single_objective(self) -> bool:
        """Check if problem is single-objective."""
        return len(self.objectives) == 1

    @property
    def is_multi_objective(self) -> bool:
        """Check if problem is multi-objective."""
        return len(self.objectives) > 1

    @property
    def n_variables(self) -> int:
        """Number of design variables."""
        return len(self.variables)

    @property
    def n_objectives(self) -> int:
        """Number of objectives."""
        return len(self.objectives)

    @property
    def n_constraints(self) -> int:
        """Number of constraints."""
        return len(self.constraints)

    @property
    def has_constraints(self) -> bool:
        """Check if problem has constraints."""
        return len(self.constraints) > 0

    def get_bounds(self) -> tuple[list[float], list[float]]:
        """
        Get variable bounds as separate lower/upper lists.

        Returns:
            (lower_bounds, upper_bounds)
        """
        lower = [var.bounds[0] for var in self.variables]
        upper = [var.bounds[1] for var in self.variables]
        return lower, upper

    def get_initial_design(self) -> Optional[list[float]]:
        """
        Get initial design from variables.

        Returns:
            List of initial values if all specified, None otherwise
        """
        if all(var.initial is not None for var in self.variables):
            return [var.initial for var in self.variables]
        return None

    class Config:
        # Allow for future extensibility
        extra = "allow"
