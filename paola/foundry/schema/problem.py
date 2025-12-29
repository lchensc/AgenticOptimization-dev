"""
Optimization Problem Schema - Base classes for problem formulations.

v0.4.1: First-class problem modeling with:
- Base OptimizationProblem class for all problem types
- Problem derivation/mutation support for graph-based optimization
- Lineage tracking for problem evolution

Problem Types:
- NLP: Nonlinear Programming (continuous, nonlinear objective/constraints)
- LP: Linear Programming (future)
- QP: Quadratic Programming (future)
- MILP: Mixed-Integer Linear Programming (future)

Problem Families:
- continuous: All variables are continuous
- discrete: All variables are discrete (integer/binary)
- mixed: Mix of continuous and discrete variables
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Literal, Tuple
from datetime import datetime
import json


class DerivationType:
    """Standard derivation types for problem mutations."""

    NARROW_BOUNDS = "narrow_bounds"       # Shrink bounds around a region
    WIDEN_BOUNDS = "widen_bounds"         # Expand search space
    RELAX_CONSTRAINTS = "relax_constraints"   # Remove/loosen constraints
    ADD_CONSTRAINTS = "add_constraints"   # Add new constraints
    SCALE = "scale"                       # Scale variables/bounds
    REDUCE_DIMENSION = "reduce_dimension"  # Fix some variables


@dataclass
class ProblemDerivation:
    """
    Specification for deriving a new problem from an existing one.

    Used to adapt search space based on optimization results.
    The derived problem maintains lineage to the parent.

    Example:
        # Narrow bounds after global search found promising region
        derivation = ProblemDerivation(
            derivation_type="narrow_bounds",
            modifications={"center": [1.2, 3.4], "width_factor": 0.3},
            reason="Focus on region found by TPE",
            source_graph_id=42,
            source_node_id="n1"
        )
    """
    derivation_type: str  # One of DerivationType constants
    modifications: Dict[str, Any]  # Type-specific modification parameters

    # Optional: reason for derivation
    reason: Optional[str] = None

    # Optional: reference to source optimization that motivated this
    source_graph_id: Optional[int] = None
    source_node_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProblemDerivation":
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class OptimizationProblem(ABC):
    """
    Base class for all optimization problem formulations.

    This is THE problem representation in Paola v0.4.1+.
    Provides common interface for different problem types (NLP, LP, QP, MILP).

    Design principle: Problems are first-class entities that can be:
    - Created, stored, and queried
    - Derived/mutated for graph-based optimization
    - Tracked with lineage for learning

    Attributes:
        problem_id: Unique identifier for this problem
        name: Human-readable name
        created_at: ISO timestamp of creation

        problem_family: "continuous", "discrete", or "mixed"
        problem_type: Specific type ("NLP", "LP", "QP", "MILP")

        n_variables: Number of decision variables
        n_constraints: Total number of constraints

        parent_problem_id: ID of parent problem (if derived)
        derivation_type: How this was derived from parent
        derivation_notes: Reason for derivation
        version: Version number within lineage (1 = root)

        description: Problem description
        domain_hint: Hint for initialization strategy
        metadata: Additional extensible metadata
    """

    # Identity - required fields first (no defaults)
    problem_id: int  # Numeric ID (like graph_id)
    name: str  # Human-readable name

    # Dimensions - required
    n_variables: int
    n_constraints: int

    # Classification - with defaults
    problem_family: Literal["continuous", "discrete", "mixed"] = "continuous"
    problem_type: str = "unknown"  # Subclasses override

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Lineage (for derivation) - all optional
    parent_problem_id: Optional[int] = None  # Numeric ID of parent
    derivation_type: Optional[str] = None
    derivation_notes: Optional[str] = None
    version: int = 1

    # Description and hints
    description: Optional[str] = None
    domain_hint: Optional[str] = None

    # Extensible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_derived(self) -> bool:
        """Whether this problem was derived from another."""
        return self.parent_problem_id is not None

    @property
    def is_root(self) -> bool:
        """Whether this is a root (non-derived) problem."""
        return self.parent_problem_id is None

    @abstractmethod
    def get_signature(self) -> Dict[str, Any]:
        """
        Get problem signature for cross-problem learning.

        Returns dict with:
        - n_dimensions: int
        - n_constraints: int
        - bounds_range: Tuple[float, float]
        - constraint_types: List[str]
        - domain_hint: Optional[str]
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationProblem":
        """Deserialize from dictionary."""
        pass

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "OptimizationProblem":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_base_dict(self) -> Dict[str, Any]:
        """Get base class fields as dict (for subclass serialization)."""
        return {
            "problem_id": self.problem_id,
            "name": self.name,
            "n_variables": self.n_variables,
            "n_constraints": self.n_constraints,
            "problem_family": self.problem_family,
            "problem_type": self.problem_type,
            "created_at": self.created_at,
            "parent_problem_id": self.parent_problem_id,
            "derivation_type": self.derivation_type,
            "derivation_notes": self.derivation_notes,
            "version": self.version,
            "description": self.description,
            "domain_hint": self.domain_hint,
            "metadata": self.metadata,
        }


# Type registry for deserialization
PROBLEM_TYPE_REGISTRY: Dict[str, type] = {}


def register_problem_type(problem_type: str, cls: type):
    """Register a problem type for deserialization."""
    PROBLEM_TYPE_REGISTRY[problem_type] = cls


def deserialize_problem(data: Dict[str, Any]) -> OptimizationProblem:
    """
    Deserialize a problem from dictionary, using correct subclass.

    Uses problem_type field to determine which class to instantiate.
    Supports both legacy schema format and new unified format.
    """
    # Check for new unified OptimizationProblem format (has "variables" field)
    if "variables" in data:
        # Use the new unified OptimizationProblem class
        from paola.foundry.problem import OptimizationProblem as UnifiedProblem
        return UnifiedProblem.from_dict(data)

    # Legacy format - use problem_type registry
    problem_type = data.get("problem_type", "NLP")

    if problem_type not in PROBLEM_TYPE_REGISTRY:
        raise ValueError(f"Unknown problem type: {problem_type}")

    cls = PROBLEM_TYPE_REGISTRY[problem_type]
    return cls.from_dict(data)
