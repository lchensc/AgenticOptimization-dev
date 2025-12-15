"""
NLP (Nonlinear Programming) problem schema.

Defines data structures for NLP problem formulation:
    minimize/maximize f(x)
    subject to:
      g_i(x) ≤ 0,  i = 1, ..., m_ineq
      h_j(x) = 0,  j = 1, ..., m_eq
      x_lower ≤ x ≤ x_upper

Where:
- f(x): Nonlinear objective function
- g_i(x): Nonlinear inequality constraints
- h_j(x): Nonlinear equality constraints
- x: Continuous decision variables
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Literal, Union
from datetime import datetime
import json

from .bounds_spec import BoundsSpec, parse_bounds_input


@dataclass
class InequalityConstraint:
    """
    Inequality constraint: g(x) ≤ value or g(x) ≥ value.

    Examples:
    - g(x) ≤ 0     → type="<=", value=0 (standard form)
    - g(x) ≤ 100   → type="<=", value=100
    - g(x) ≥ 50    → type=">=", value=50

    Note: Internally transformed to scipy format (g(x) >= 0) by NLPEvaluator.
    """
    name: str
    evaluator_id: str
    constraint_type: Literal["<=", ">="]
    value: float  # RHS value

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InequalityConstraint':
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class EqualityConstraint:
    """
    Equality constraint: h(x) = value.

    Examples:
    - h(x) = 0     → value=0 (standard form)
    - h(x) = 100   → value=100

    Note: Equality constraints are harder to satisfy than inequalities.
    Scipy uses tolerance (typically 1e-6) to check satisfaction.
    """
    name: str
    evaluator_id: str
    value: float  # RHS value
    tolerance: float = 1e-6  # Tolerance for equality check

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EqualityConstraint':
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class NLPProblem:
    """
    Nonlinear Programming (NLP) problem specification.

    Standard form:
        minimize/maximize f(x)
        subject to:
          g_i(x) ≤ 0,  i = 1, ..., m_ineq
          h_j(x) = 0,  j = 1, ..., m_eq
          x_lower ≤ x ≤ x_upper

    Where:
    - f(x): Nonlinear objective function (from registered evaluator)
    - g_i(x): Nonlinear inequality constraints (from registered evaluators)
    - h_j(x): Nonlinear equality constraints (from registered evaluators)
    - x: Continuous decision variables

    Example:
        problem = NLPProblem(
            problem_id="wing_design",
            objective_evaluator_id="drag_eval",
            objective_sense="minimize",
            dimension=2,
            bounds=[[0, 15], [0.1, 0.5]],
            inequality_constraints=[
                InequalityConstraint(
                    name="min_lift",
                    evaluator_id="lift_eval",
                    constraint_type=">=",
                    value=1000.0
                )
            ]
        )
    """

    # Required fields first
    problem_id: str
    objective_evaluator_id: str
    dimension: int
    bounds: List[List[float]]  # [[lower, upper], ...] - expanded from BoundsSpec

    # Fields with defaults
    problem_type: Literal["NLP"] = "NLP"  # Fixed for NLP
    objective_sense: Literal["minimize", "maximize"] = "minimize"
    inequality_constraints: List[InequalityConstraint] = field(default_factory=list)
    equality_constraints: List[EqualityConstraint] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: Optional[str] = None

    # The Paola Principle: Initialization is agent intelligence, not user input
    # Domain hints help the agent make better initialization decisions
    domain_hint: Optional[str] = None  # e.g., "shape_optimization", "structural", "general"

    # Store original BoundsSpec for reference (if provided)
    bounds_spec: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate problem specification."""
        # Validate bounds dimension
        if len(self.bounds) != self.dimension:
            raise ValueError(
                f"Bounds dimension ({len(self.bounds)}) doesn't match "
                f"problem dimension ({self.dimension})"
            )

        # Validate bounds format
        for i, bound in enumerate(self.bounds):
            if len(bound) != 2:
                raise ValueError(f"Bound {i} must be [lower, upper], got {bound}")
            if bound[0] >= bound[1]:
                raise ValueError(
                    f"Bound {i} has lower >= upper: [{bound[0]}, {bound[1]}]"
                )

        # Validate domain_hint if provided
        valid_domain_hints = {
            "shape_optimization",  # FFD, mesh deformation → zero init
            "structural",          # Structural design → center of bounds
            "aerodynamic",         # Aero shape → zero init
            "topology",            # Topology opt → typically uniform density
            "general"              # General → center of bounds
        }
        if self.domain_hint is not None and self.domain_hint not in valid_domain_hints:
            # Allow custom hints but log a warning (don't raise error)
            pass  # Flexible: allow unknown hints for extensibility

    @property
    def num_inequality_constraints(self) -> int:
        """Number of inequality constraints."""
        return len(self.inequality_constraints)

    @property
    def num_equality_constraints(self) -> int:
        """Number of equality constraints."""
        return len(self.equality_constraints)

    @property
    def num_constraints(self) -> int:
        """Total number of constraints."""
        return self.num_inequality_constraints + self.num_equality_constraints

    @property
    def is_unconstrained(self) -> bool:
        """Whether problem has no constraints."""
        return self.num_constraints == 0

    @property
    def is_constrained(self) -> bool:
        """Whether problem has constraints."""
        return self.num_constraints > 0

    def get_all_evaluator_ids(self) -> List[str]:
        """Get all evaluator IDs used in this problem."""
        evaluator_ids = [self.objective_evaluator_id]

        for cons in self.inequality_constraints:
            evaluator_ids.append(cons.evaluator_id)

        for cons in self.equality_constraints:
            evaluator_ids.append(cons.evaluator_id)

        return list(set(evaluator_ids))  # Remove duplicates

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = asdict(self)
        # Convert constraint objects to dicts
        data['inequality_constraints'] = [
            c.to_dict() if hasattr(c, 'to_dict') else c
            for c in self.inequality_constraints
        ]
        data['equality_constraints'] = [
            c.to_dict() if hasattr(c, 'to_dict') else c
            for c in self.equality_constraints
        ]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NLPProblem':
        """Deserialize from dictionary."""
        # Make a copy to avoid mutating input
        data = dict(data)

        # Convert constraint dicts back to objects
        if 'inequality_constraints' in data:
            data['inequality_constraints'] = [
                InequalityConstraint.from_dict(c) if isinstance(c, dict) else c
                for c in data['inequality_constraints']
            ]

        if 'equality_constraints' in data:
            data['equality_constraints'] = [
                EqualityConstraint.from_dict(c) if isinstance(c, dict) else c
                for c in data['equality_constraints']
            ]

        # Remove deprecated initial_point if present (backward compatibility)
        if 'initial_point' in data:
            del data['initial_point']

        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'NLPProblem':
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_bounds_spec(
        cls,
        problem_id: str,
        objective_evaluator_id: str,
        bounds_spec: Union[List[List[float]], Dict[str, Any], BoundsSpec],
        objective_sense: Literal["minimize", "maximize"] = "minimize",
        inequality_constraints: List[InequalityConstraint] = None,
        equality_constraints: List[EqualityConstraint] = None,
        description: Optional[str] = None,
        domain_hint: Optional[str] = None
    ) -> 'NLPProblem':
        """
        Create NLPProblem from BoundsSpec (preferred for large variable spaces).

        The Paola Principle: "Optimization complexity is Paola intelligence."
        This factory method allows compact bounds specification while
        automatically expanding them to explicit bounds.

        Args:
            problem_id: Unique problem identifier
            objective_evaluator_id: Registered evaluator for objective
            bounds_spec: One of:
                - List of [lower, upper] pairs (explicit)
                - Dict with BoundsSpec format
                - BoundsSpec object
            objective_sense: "minimize" or "maximize"
            inequality_constraints: List of inequality constraints
            equality_constraints: List of equality constraints
            description: Problem description
            domain_hint: Hint for initialization (e.g., "shape_optimization")

        Returns:
            NLPProblem with expanded bounds

        Example:
            # Uniform bounds for 100 FFD control points
            problem = NLPProblem.from_bounds_spec(
                problem_id="wing_ffd",
                objective_evaluator_id="drag_eval",
                bounds_spec={"type": "uniform", "lower": -0.05, "upper": 0.05, "dimension": 100},
                domain_hint="shape_optimization"
            )
        """
        # Parse bounds spec
        spec = parse_bounds_input(bounds_spec)

        # Expand to explicit bounds
        expanded_bounds = spec.expand()

        return cls(
            problem_id=problem_id,
            objective_evaluator_id=objective_evaluator_id,
            dimension=len(expanded_bounds),
            bounds=expanded_bounds,
            objective_sense=objective_sense,
            inequality_constraints=inequality_constraints or [],
            equality_constraints=equality_constraints or [],
            description=description,
            domain_hint=domain_hint,
            bounds_spec=spec.to_dict()
        )

    def get_bounds_center(self) -> List[float]:
        """
        Get center of bounds.

        Used by InitializationManager for gradient-based algorithms.

        Returns:
            List of center values for each variable
        """
        return [(b[0] + b[1]) / 2 for b in self.bounds]

    def get_bounds_width(self) -> List[float]:
        """
        Get width of bounds.

        Used by InitializationManager for CMA-ES sigma calculation.

        Returns:
            List of bound widths for each variable
        """
        return [b[1] - b[0] for b in self.bounds]

    def __str__(self) -> str:
        """Human-readable representation."""
        lines = [
            f"NLP Problem: {self.problem_id}",
            f"  Objective: {self.objective_sense} {self.objective_evaluator_id}",
            f"  Dimension: {self.dimension}",
        ]

        # Compact bounds display for large variable spaces
        if self.dimension <= 5:
            lines.append(f"  Bounds: {self.bounds}")
        else:
            # Show first 2 and last bound with ellipsis
            lines.append(f"  Bounds: [{self.bounds[0]}, {self.bounds[1]}, ..., {self.bounds[-1]}]")

        if self.domain_hint:
            lines.append(f"  Domain hint: {self.domain_hint}")

        if self.inequality_constraints:
            lines.append(f"  Inequality constraints: {self.num_inequality_constraints}")
            for cons in self.inequality_constraints:
                lines.append(
                    f"    {cons.name}: {cons.evaluator_id} {cons.constraint_type} {cons.value}"
                )

        if self.equality_constraints:
            lines.append(f"  Equality constraints: {self.num_equality_constraints}")
            for cons in self.equality_constraints:
                lines.append(
                    f"    {cons.name}: {cons.evaluator_id} = {cons.value}"
                )

        return "\n".join(lines)
