"""
Pydantic schemas for tool argument validation.

Provides type-safe, self-documenting schemas for LangChain tools.
Handles automatic type coercion (e.g., str "5" -> int 5 for problem_id).
"""

from typing import Optional, List, Union, Annotated
from pydantic import BaseModel, Field, field_validator


class ProblemId(BaseModel):
    """Validated problem ID - always coerced to int.

    Accepts both string "5" and int 5, normalizes to int.
    This handles LLM JSON serialization which sends strings.
    """

    value: int

    @classmethod
    def coerce(cls, v: Union[str, int]) -> int:
        """Coerce problem_id to int."""
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                raise ValueError(f"problem_id must be numeric, got '{v}'")
        raise TypeError(f"problem_id must be str or int, got {type(v).__name__}")


# Type alias for use in function signatures
ProblemIdType = Union[str, int]


def normalize_problem_id(problem_id: ProblemIdType) -> int:
    """Normalize problem_id to int.

    This is the canonical way to handle problem_id at system boundaries.
    Use this in tools that receive problem_id from LLM.

    Args:
        problem_id: Can be str "5" or int 5

    Returns:
        int: Normalized problem ID

    Raises:
        ValueError: If problem_id cannot be converted to int
    """
    return ProblemId.coerce(problem_id)


# =============================================================================
# Tool Argument Schemas
# =============================================================================

# RunOptimizationArgs removed in v0.2.0 - LLM writes code directly


class EvaluateFunctionArgs(BaseModel):
    """Arguments for evaluate_function tool."""

    problem_id: ProblemIdType = Field(description="Problem identifier")
    design: List[float] = Field(description="Design point to evaluate")
    use_cache: bool = Field(default=True, description="Check cache first")
    compute_constraints: bool = Field(default=False, description="Also compute constraints")

    @field_validator("problem_id", mode="before")
    @classmethod
    def coerce_problem_id(cls, v):
        return normalize_problem_id(v)


class ComputeGradientArgs(BaseModel):
    """Arguments for compute_gradient tool."""

    problem_id: ProblemIdType = Field(description="Problem identifier")
    design: List[float] = Field(description="Design point for gradient")
    method: str = Field(default="analytical", description="Method: analytical, finite-difference")
    use_cache: bool = Field(default=True, description="Check cache first")
    fd_step: float = Field(default=1e-6, description="Step size for finite difference")

    @field_validator("problem_id", mode="before")
    @classmethod
    def coerce_problem_id(cls, v):
        return normalize_problem_id(v)


class GetProblemInfoArgs(BaseModel):
    """Arguments for get_problem_info tool."""

    problem_id: ProblemIdType = Field(description="Problem ID to get info for")

    @field_validator("problem_id", mode="before")
    @classmethod
    def coerce_problem_id(cls, v):
        return normalize_problem_id(v)


class DeriveProblemArgs(BaseModel):
    """Arguments for derive_problem tool."""

    parent_problem_id: ProblemIdType = Field(description="Problem ID to derive from")
    derivation_type: str = Field(description="Type: narrow_bounds, widen_bounds")
    modifications: str = Field(description="JSON with center, width_factor")
    new_name: Optional[str] = Field(default=None, description="Name for derived problem")
    reason: Optional[str] = Field(default=None, description="Why this derivation was needed")
    source_graph_id: Optional[int] = Field(default=None, description="Graph that motivated this")
    source_node_id: Optional[str] = Field(default=None, description="Node that motivated this")

    @field_validator("parent_problem_id", mode="before")
    @classmethod
    def coerce_problem_id(cls, v):
        return normalize_problem_id(v)


class GetProblemByIdArgs(BaseModel):
    """Arguments for get_problem_by_id tool."""

    problem_id: ProblemIdType = Field(description="Problem ID to retrieve")

    @field_validator("problem_id", mode="before")
    @classmethod
    def coerce_problem_id(cls, v):
        return normalize_problem_id(v)
