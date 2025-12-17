"""
NLP (Nonlinear Programming) problem schema.

Defines data structures for NLP problem formulation:
    minimize/maximize f(x)
    subject to:
      g_i(x) <= 0,  i = 1, ..., m_ineq
      h_j(x) = 0,  j = 1, ..., m_eq
      x_lower <= x <= x_upper

Where:
- f(x): Nonlinear objective function
- g_i(x): Nonlinear inequality constraints
- h_j(x): Nonlinear equality constraints
- x: Continuous decision variables

v0.4.1: Refactored to inherit from OptimizationProblem base class.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Literal, Union, Tuple
from datetime import datetime
import json

from .bounds_spec import BoundsSpec, parse_bounds_input
from .schema.problem import (
    OptimizationProblem,
    ProblemDerivation,
    DerivationType,
    register_problem_type,
)


@dataclass
class InequalityConstraint:
    """
    Inequality constraint: g(x) <= value or g(x) >= value.

    Examples:
    - g(x) <= 0     -> type="<=", value=0 (standard form)
    - g(x) <= 100   -> type="<=", value=100
    - g(x) >= 50    -> type=">=", value=50

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
    - h(x) = 0     -> value=0 (standard form)
    - h(x) = 100   -> value=100

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
class NLPProblem(OptimizationProblem):
    """
    Nonlinear Programming (NLP) problem specification.

    Inherits from OptimizationProblem base class (v0.4.1+).

    Standard form:
        minimize/maximize f(x)
        subject to:
          g_i(x) <= 0,  i = 1, ..., m_ineq
          h_j(x) = 0,  j = 1, ..., m_eq
          x_lower <= x <= x_upper

    Where:
    - f(x): Nonlinear objective function (from registered evaluator)
    - g_i(x): Nonlinear inequality constraints (from registered evaluators)
    - h_j(x): Nonlinear equality constraints (from registered evaluators)
    - x: Continuous decision variables

    Example:
        problem = NLPProblem(
            problem_id="wing_design",
            name="Wing Design Optimization",
            objective_evaluator_id="drag_eval",
            objective_sense="minimize",
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

    # Override base class defaults for NLP
    problem_family: Literal["continuous"] = "continuous"
    problem_type: Literal["NLP"] = "NLP"

    # NLP-specific required fields
    objective_evaluator_id: str = ""  # Empty default, validated in __post_init__
    bounds: List[List[float]] = field(default_factory=list)  # [[lower, upper], ...]

    # NLP-specific optional fields
    objective_sense: Literal["minimize", "maximize"] = "minimize"
    inequality_constraints: List[InequalityConstraint] = field(default_factory=list)
    equality_constraints: List[EqualityConstraint] = field(default_factory=list)

    # Store original BoundsSpec for reference (if provided)
    bounds_spec: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate problem specification."""
        # Validate objective_evaluator_id is set
        if not self.objective_evaluator_id:
            raise ValueError("objective_evaluator_id is required")

        # Compute n_variables and n_constraints from NLP-specific fields
        if self.bounds:
            # Override n_variables from bounds
            object.__setattr__(self, 'n_variables', len(self.bounds))

        # Compute n_constraints
        n_cons = len(self.inequality_constraints) + len(self.equality_constraints)
        object.__setattr__(self, 'n_constraints', n_cons)

        # Validate bounds dimension
        if self.bounds and len(self.bounds) != self.n_variables:
            raise ValueError(
                f"Bounds dimension ({len(self.bounds)}) doesn't match "
                f"problem dimension ({self.n_variables})"
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
            "shape_optimization",  # FFD, mesh deformation -> zero init
            "structural",          # Structural design -> center of bounds
            "aerodynamic",         # Aero shape -> zero init
            "topology",            # Topology opt -> typically uniform density
            "general"              # General -> center of bounds
        }
        if self.domain_hint is not None and self.domain_hint not in valid_domain_hints:
            # Allow custom hints but log a warning (don't raise error)
            pass  # Flexible: allow unknown hints for extensibility

    # Backward compatibility: dimension property
    @property
    def dimension(self) -> int:
        """Number of variables (alias for n_variables)."""
        return self.n_variables

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

    def get_signature(self) -> Dict[str, Any]:
        """
        Get problem signature for cross-problem learning.

        Returns dict compatible with ProblemSignature dataclass.
        """
        # Compute bounds range
        all_lowers = [b[0] for b in self.bounds]
        all_uppers = [b[1] for b in self.bounds]
        bounds_range = (min(all_lowers), max(all_uppers))

        # Collect constraint types
        constraint_types = []
        if self.inequality_constraints:
            constraint_types.append("inequality")
        if self.equality_constraints:
            constraint_types.append("equality")

        return {
            "n_dimensions": self.n_variables,
            "n_constraints": self.n_constraints,
            "bounds_range": bounds_range,
            "constraint_types": constraint_types,
            "domain_hint": self.domain_hint,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        # Start with base class fields
        data = self.get_base_dict()

        # Add NLP-specific fields
        data["objective_evaluator_id"] = self.objective_evaluator_id
        data["objective_sense"] = self.objective_sense
        data["bounds"] = self.bounds
        data["bounds_spec"] = self.bounds_spec

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

        # Handle legacy 'dimension' field
        if 'dimension' in data and 'n_variables' not in data:
            data['n_variables'] = data.pop('dimension')
        elif 'dimension' in data:
            del data['dimension']

        # Ensure n_variables and n_constraints are set (will be recomputed in __post_init__)
        if 'n_variables' not in data:
            data['n_variables'] = len(data.get('bounds', []))
        if 'n_constraints' not in data:
            data['n_constraints'] = (
                len(data.get('inequality_constraints', [])) +
                len(data.get('equality_constraints', []))
            )

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
        problem_id: int,
        name: str,
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
            problem_id: Numeric problem ID (from storage.get_next_problem_id())
            name: Human-readable problem name
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
                problem_id=1,
                name="Wing FFD Optimization",
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
            name=name,
            n_variables=len(expanded_bounds),
            n_constraints=0,  # Will be recomputed in __post_init__
            objective_evaluator_id=objective_evaluator_id,
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

    def derive_narrow_bounds(
        self,
        new_problem_id: int,
        new_name: str,
        center: List[float],
        width_factor: float = 0.3,
        reason: Optional[str] = None,
        source_graph_id: Optional[int] = None,
        source_node_id: Optional[str] = None,
    ) -> 'NLPProblem':
        """
        Derive a new NLP problem with narrowed bounds.

        Shrinks bounds around a center point (e.g., best solution from global search).
        The derived problem maintains lineage to this parent problem.

        Args:
            new_problem_id: Numeric ID for the derived problem (from storage.get_next_problem_id())
            new_name: Human-readable name for the derived problem
            center: Center point for new bounds (typically best_x from optimization)
            width_factor: Fraction of original width to use (0.3 = 30% of original)
            reason: Why this derivation was needed
            source_graph_id: Graph that motivated this derivation
            source_node_id: Node that motivated this derivation

        Returns:
            New NLPProblem with narrowed bounds

        Example:
            # After TPE found best_x = [1.2, 3.4]
            derived = original.derive_narrow_bounds(
                new_problem_id=2,
                new_name="Wing Design (narrowed)",
                center=[1.2, 3.4],
                width_factor=0.3,
                reason="Focus on region found by TPE"
            )
        """
        if len(center) != self.n_variables:
            raise ValueError(
                f"Center dimension ({len(center)}) doesn't match "
                f"problem dimension ({self.n_variables})"
            )

        # Compute new bounds centered on center with reduced width
        new_bounds = []
        for i, (lb, ub) in enumerate(self.bounds):
            original_width = ub - lb
            new_width = original_width * width_factor
            new_lb = max(lb, center[i] - new_width / 2)
            new_ub = min(ub, center[i] + new_width / 2)
            new_bounds.append([new_lb, new_ub])

        # Build derivation notes
        notes = reason or f"Narrowed bounds around center with width_factor={width_factor}"
        if source_graph_id and source_node_id:
            notes += f" (from graph {source_graph_id}, node {source_node_id})"

        return NLPProblem(
            problem_id=new_problem_id,
            name=new_name,
            n_variables=self.n_variables,
            n_constraints=self.n_constraints,
            objective_evaluator_id=self.objective_evaluator_id,
            bounds=new_bounds,
            objective_sense=self.objective_sense,
            inequality_constraints=self.inequality_constraints,
            equality_constraints=self.equality_constraints,
            description=self.description,
            domain_hint=self.domain_hint,
            parent_problem_id=self.problem_id,
            derivation_type=DerivationType.NARROW_BOUNDS,
            derivation_notes=notes,
            version=self.version + 1,
            metadata={
                **self.metadata,
                "derivation_spec": {
                    "center": center,
                    "width_factor": width_factor,
                    "source_graph_id": source_graph_id,
                    "source_node_id": source_node_id,
                }
            }
        )

    def derive_widen_bounds(
        self,
        new_problem_id: int,
        new_name: str,
        width_factor: float = 1.5,
        reason: Optional[str] = None,
    ) -> 'NLPProblem':
        """
        Derive a new NLP problem with widened bounds.

        Expands bounds (e.g., when solution hits boundary).

        Args:
            new_problem_id: Numeric ID for the derived problem (from storage.get_next_problem_id())
            new_name: Human-readable name for the derived problem
            width_factor: Factor to multiply original width (1.5 = 50% wider)
            reason: Why this derivation was needed

        Returns:
            New NLPProblem with widened bounds
        """
        # Compute new bounds with expanded width
        new_bounds = []
        for lb, ub in self.bounds:
            center = (lb + ub) / 2
            original_width = ub - lb
            new_width = original_width * width_factor
            new_lb = center - new_width / 2
            new_ub = center + new_width / 2
            new_bounds.append([new_lb, new_ub])

        notes = reason or f"Widened bounds with width_factor={width_factor}"

        return NLPProblem(
            problem_id=new_problem_id,
            name=new_name,
            n_variables=self.n_variables,
            n_constraints=self.n_constraints,
            objective_evaluator_id=self.objective_evaluator_id,
            bounds=new_bounds,
            objective_sense=self.objective_sense,
            inequality_constraints=self.inequality_constraints,
            equality_constraints=self.equality_constraints,
            description=self.description,
            domain_hint=self.domain_hint,
            parent_problem_id=self.problem_id,
            derivation_type=DerivationType.WIDEN_BOUNDS,
            derivation_notes=notes,
            version=self.version + 1,
            metadata={
                **self.metadata,
                "derivation_spec": {"width_factor": width_factor}
            }
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        lines = [
            f"NLP Problem: {self.problem_id}",
            f"  Objective: {self.objective_sense} {self.objective_evaluator_id}",
            f"  Variables: {self.n_variables}",
        ]

        # Compact bounds display for large variable spaces
        if self.n_variables <= 5:
            lines.append(f"  Bounds: {self.bounds}")
        else:
            # Show first 2 and last bound with ellipsis
            lines.append(f"  Bounds: [{self.bounds[0]}, {self.bounds[1]}, ..., {self.bounds[-1]}]")

        if self.domain_hint:
            lines.append(f"  Domain hint: {self.domain_hint}")

        # Lineage info
        if self.is_derived:
            lines.append(f"  Derived from: {self.parent_problem_id} ({self.derivation_type})")
            lines.append(f"  Version: {self.version}")

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


# Register NLPProblem in the type registry
register_problem_type("NLP", NLPProblem)
