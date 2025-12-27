"""
Unified Optimization Problem Schema.

Paola v1.0: Single problem class for all optimization types.
Supports NLP, MINLP, MOO, MO-MINLP through composition.

Inspired by pymoo's unified Problem class but enhanced for agentic use.

Key design decisions:
- Single OptimizationProblem class (no NLP/MOO/MINLP subclasses)
- Auto-detection of problem class via properties
- Explicit Variable/Objective/Constraint specifications
- Evaluator validation at problem creation time
- Lineage tracking for problem derivation
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Literal, Tuple, Callable
from datetime import datetime
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Variable Specification
# =============================================================================

@dataclass
class Variable:
    """
    Decision variable specification.

    Supports continuous, integer, and binary variable types.
    Each variable has explicit bounds.

    Attributes:
        name: Variable identifier (e.g., "x1", "chord", "n_ribs")
        type: Variable type - "continuous", "integer", or "binary"
        lower: Lower bound
        upper: Upper bound
        description: Optional description for agent reasoning
        unit: Optional unit (e.g., "m", "kg", "degrees")

    Examples:
        Variable(name="chord", type="continuous", lower=0.5, upper=2.0, unit="m")
        Variable(name="n_ribs", type="integer", lower=3, upper=10)
        Variable(name="use_winglet", type="binary")  # bounds auto-set to [0, 1]
    """
    name: str
    type: Literal["continuous", "integer", "binary"]
    lower: float
    upper: float
    description: Optional[str] = None
    unit: Optional[str] = None

    def __post_init__(self):
        """Validate and normalize variable specification."""
        # Binary variables have fixed bounds
        if self.type == "binary":
            self.lower = 0.0
            self.upper = 1.0

        # Validate bounds
        if self.lower >= self.upper:
            raise ValueError(
                f"Variable '{self.name}': lower ({self.lower}) must be < upper ({self.upper})"
            )

        # Integer bounds should be integers
        if self.type == "integer":
            self.lower = float(int(self.lower))
            self.upper = float(int(self.upper))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Variable":
        """Deserialize from dictionary."""
        return cls(**data)


# =============================================================================
# Objective Specification
# =============================================================================

@dataclass
class Objective:
    """
    Objective function specification.

    Supports two patterns for multi-objective optimization:

    Pattern A - Separate evaluators (each objective has its own evaluator):
        Objective(name="drag", evaluator_id="cfd_drag")
        Objective(name="weight", evaluator_id="fem_weight")

    Pattern B - Single evaluator returning array [f1, f2, ...]:
        Objective(name="f1", evaluator_id="zdt1", index=0)
        Objective(name="f2", evaluator_id="zdt1", index=1)

    Attributes:
        name: Objective name (e.g., "drag", "weight", "f1")
        evaluator_id: Registered evaluator that computes this objective
        sense: "minimize" or "maximize" (default: minimize)
        index: For array-returning evaluators, which element to use
        description: Optional description for agent reasoning

    Examples:
        # Single objective
        Objective(name="drag", evaluator_id="cfd_drag", sense="minimize")

        # Multi-objective with separate evaluators
        Objective(name="drag", evaluator_id="cfd_drag")
        Objective(name="weight", evaluator_id="fem_weight")

        # Multi-objective with array evaluator (ZDT1 returns [f1, f2])
        Objective(name="f1", evaluator_id="zdt1", index=0)
        Objective(name="f2", evaluator_id="zdt1", index=1)
    """
    name: str
    evaluator_id: str
    sense: Literal["minimize", "maximize"] = "minimize"
    index: Optional[int] = None  # For array-returning evaluators
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Objective":
        """Deserialize from dictionary."""
        return cls(**data)


# =============================================================================
# Constraint Specification
# =============================================================================

@dataclass
class Constraint:
    """
    Constraint specification.

    Unified format for inequality and equality constraints:
    - g(x) <= bound  (type="<=")
    - g(x) >= bound  (type=">=")
    - h(x) == bound  (type="==")

    Attributes:
        name: Constraint name (e.g., "min_lift", "stress_limit")
        evaluator_id: Registered evaluator that computes constraint value
        type: Constraint type - "<=", ">=", or "=="
        bound: Right-hand side value
        description: Optional description for agent reasoning
        tolerance: For equality constraints, tolerance for satisfaction

    Examples:
        # Inequality: lift >= 1000
        Constraint(name="min_lift", evaluator_id="lift_eval", type=">=", bound=1000)

        # Inequality: stress <= 250 MPa
        Constraint(name="stress_limit", evaluator_id="stress_eval", type="<=", bound=250)

        # Equality: volume == 1.0
        Constraint(name="volume", evaluator_id="vol_eval", type="==", bound=1.0)
    """
    name: str
    evaluator_id: str
    type: Literal["<=", ">=", "=="]
    bound: float
    description: Optional[str] = None
    tolerance: float = 1e-6  # For equality constraints

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Constraint":
        """Deserialize from dictionary."""
        return cls(**data)

    @property
    def is_equality(self) -> bool:
        """Whether this is an equality constraint."""
        return self.type == "=="


# =============================================================================
# Derivation Types
# =============================================================================

class DerivationType:
    """Standard derivation types for problem mutations."""
    NARROW_BOUNDS = "narrow_bounds"
    WIDEN_BOUNDS = "widen_bounds"
    RELAX_CONSTRAINTS = "relax_constraints"
    ADD_CONSTRAINTS = "add_constraints"
    SCALE = "scale"
    REDUCE_DIMENSION = "reduce_dimension"
    ADD_OBJECTIVE = "add_objective"  # New for MOO


# =============================================================================
# Unified Optimization Problem
# =============================================================================

@dataclass
class OptimizationProblem:
    """
    Unified optimization problem specification.

    Handles all problem types through composition:
    - NLP: continuous variables, single objective
    - MINLP: mixed integer variables, single objective
    - MOO: continuous variables, multiple objectives
    - MO-MINLP: mixed integer variables, multiple objectives

    The problem class is auto-detected from the specification.

    Attributes:
        problem_id: Unique numeric identifier
        name: Human-readable name
        variables: List of decision variable specifications
        objectives: List of objective specifications
        constraints: List of constraint specifications (optional)
        description: Problem description
        domain_hint: Hint for initialization (e.g., "shape_optimization")
        created_at: ISO timestamp
        parent_problem_id: For derived problems
        derivation_type: How derived from parent
        derivation_notes: Reason for derivation
        version: Version in lineage (1 = root)
        metadata: Extensible metadata

    Examples:
        # Simple NLP (continuous, single objective)
        problem = OptimizationProblem(
            problem_id=1,
            name="Minimize Drag",
            variables=[
                Variable(name="chord", type="continuous", lower=0.5, upper=2.0),
                Variable(name="twist", type="continuous", lower=-5, upper=5),
            ],
            objectives=[
                Objective(name="drag", evaluator_id="cfd_drag"),
            ],
        )

        # MOO (continuous, multiple objectives)
        problem = OptimizationProblem(
            problem_id=2,
            name="Wing Design Trade-off",
            variables=[...],
            objectives=[
                Objective(name="drag", evaluator_id="cfd_drag"),
                Objective(name="weight", evaluator_id="fem_weight"),
            ],
        )

        # MINLP (mixed integer, single objective)
        problem = OptimizationProblem(
            problem_id=3,
            name="Structural Design",
            variables=[
                Variable(name="thickness", type="continuous", lower=0.01, upper=0.1),
                Variable(name="n_ribs", type="integer", lower=3, upper=10),
            ],
            objectives=[...],
        )
    """

    # Identity
    problem_id: int
    name: str

    # Core specification
    variables: List[Variable]
    objectives: List[Objective]
    constraints: List[Constraint] = field(default_factory=list)

    # Description and hints
    description: Optional[str] = None
    domain_hint: Optional[str] = None

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Lineage
    parent_problem_id: Optional[int] = None
    derivation_type: Optional[str] = None
    derivation_notes: Optional[str] = None
    version: int = 1

    # Extensible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate problem specification."""
        # Convert dicts to dataclasses if needed
        self.variables = [
            Variable.from_dict(v) if isinstance(v, dict) else v
            for v in self.variables
        ]
        self.objectives = [
            Objective.from_dict(o) if isinstance(o, dict) else o
            for o in self.objectives
        ]
        self.constraints = [
            Constraint.from_dict(c) if isinstance(c, dict) else c
            for c in self.constraints
        ]

        # Validate we have at least one variable and one objective
        if not self.variables:
            raise ValueError("Problem must have at least one variable")
        if not self.objectives:
            raise ValueError("Problem must have at least one objective")

        # Validate variable names are unique
        var_names = [v.name for v in self.variables]
        if len(var_names) != len(set(var_names)):
            raise ValueError("Variable names must be unique")

        # Validate objective names are unique
        obj_names = [o.name for o in self.objectives]
        if len(obj_names) != len(set(obj_names)):
            raise ValueError("Objective names must be unique")

    # =========================================================================
    # Derived Properties
    # =========================================================================

    @property
    def n_variables(self) -> int:
        """Number of decision variables."""
        return len(self.variables)

    @property
    def n_objectives(self) -> int:
        """Number of objectives."""
        return len(self.objectives)

    @property
    def n_constraints(self) -> int:
        """Total number of constraints."""
        return len(self.constraints)

    @property
    def n_inequality_constraints(self) -> int:
        """Number of inequality constraints."""
        return sum(1 for c in self.constraints if not c.is_equality)

    @property
    def n_equality_constraints(self) -> int:
        """Number of equality constraints."""
        return sum(1 for c in self.constraints if c.is_equality)

    @property
    def is_multiobjective(self) -> bool:
        """Whether this is a multi-objective problem."""
        return self.n_objectives > 1

    @property
    def has_integers(self) -> bool:
        """Whether problem has integer or binary variables."""
        return any(v.type in ("integer", "binary") for v in self.variables)

    @property
    def is_constrained(self) -> bool:
        """Whether problem has constraints."""
        return self.n_constraints > 0

    @property
    def is_unconstrained(self) -> bool:
        """Whether problem has no constraints."""
        return self.n_constraints == 0

    @property
    def is_derived(self) -> bool:
        """Whether this problem was derived from another."""
        return self.parent_problem_id is not None

    @property
    def is_root(self) -> bool:
        """Whether this is a root (non-derived) problem."""
        return self.parent_problem_id is None

    @property
    def problem_family(self) -> str:
        """Variable family: continuous, discrete, or mixed."""
        types = set(v.type for v in self.variables)
        if types == {"continuous"}:
            return "continuous"
        elif types <= {"integer", "binary"}:
            return "discrete"
        else:
            return "mixed"

    @property
    def problem_class(self) -> str:
        """
        Auto-detect problem class for solver selection.

        Returns:
            "NLP": Continuous, single objective
            "MINLP": Mixed integer, single objective
            "MOO": Continuous, multi-objective
            "MO-MINLP": Mixed integer, multi-objective
        """
        is_moo = self.is_multiobjective
        has_int = self.has_integers

        if is_moo and has_int:
            return "MO-MINLP"
        elif is_moo:
            return "MOO"
        elif has_int:
            return "MINLP"
        else:
            return "NLP"

    # =========================================================================
    # Bounds and Indices
    # =========================================================================

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get variable bounds as numpy arrays.

        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays
        """
        lb = np.array([v.lower for v in self.variables])
        ub = np.array([v.upper for v in self.variables])
        return lb, ub

    def get_bounds_list(self) -> List[List[float]]:
        """Get bounds as list of [lower, upper] pairs."""
        return [[v.lower, v.upper] for v in self.variables]

    def get_integer_indices(self) -> List[int]:
        """Get indices of integer/binary variables."""
        return [i for i, v in enumerate(self.variables)
                if v.type in ("integer", "binary")]

    def get_continuous_indices(self) -> List[int]:
        """Get indices of continuous variables."""
        return [i for i, v in enumerate(self.variables)
                if v.type == "continuous"]

    def get_integer_mask(self) -> np.ndarray:
        """Get boolean mask for integer/binary variables."""
        return np.array([v.type in ("integer", "binary") for v in self.variables])

    def get_objective_senses(self) -> List[str]:
        """Get list of objective senses ("minimize" or "maximize")."""
        return [o.sense for o in self.objectives]

    def get_bounds_center(self) -> np.ndarray:
        """Get center of bounds (useful for initialization)."""
        lb, ub = self.get_bounds()
        return (lb + ub) / 2

    def get_bounds_width(self) -> np.ndarray:
        """Get width of bounds (useful for CMA-ES sigma)."""
        lb, ub = self.get_bounds()
        return ub - lb

    # =========================================================================
    # Evaluator Queries
    # =========================================================================

    def get_all_evaluator_ids(self) -> List[str]:
        """Get all unique evaluator IDs used in this problem."""
        ids = set()
        for obj in self.objectives:
            ids.add(obj.evaluator_id)
        for con in self.constraints:
            ids.add(con.evaluator_id)
        return list(ids)

    def validate_evaluators(self, available_evaluator_ids: List[str]) -> List[str]:
        """
        Validate that all referenced evaluators exist.

        Args:
            available_evaluator_ids: List of registered evaluator IDs

        Returns:
            List of missing evaluator IDs (empty if all exist)
        """
        required = set(self.get_all_evaluator_ids())
        available = set(available_evaluator_ids)
        missing = required - available
        return list(missing)

    # =========================================================================
    # Signature for Cross-Problem Learning
    # =========================================================================

    def get_signature(self) -> Dict[str, Any]:
        """
        Get problem signature for cross-problem learning.

        Returns dict with problem characteristics for matching similar problems.
        """
        lb, ub = self.get_bounds()
        return {
            "n_variables": self.n_variables,
            "n_objectives": self.n_objectives,
            "n_constraints": self.n_constraints,
            "problem_class": self.problem_class,
            "problem_family": self.problem_family,
            "bounds_range": (float(lb.min()), float(ub.max())),
            "has_equality_constraints": self.n_equality_constraints > 0,
            "domain_hint": self.domain_hint,
        }

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "problem_id": self.problem_id,
            "name": self.name,
            "variables": [v.to_dict() for v in self.variables],
            "objectives": [o.to_dict() for o in self.objectives],
            "constraints": [c.to_dict() for c in self.constraints],
            "description": self.description,
            "domain_hint": self.domain_hint,
            "created_at": self.created_at,
            "parent_problem_id": self.parent_problem_id,
            "derivation_type": self.derivation_type,
            "derivation_notes": self.derivation_notes,
            "version": self.version,
            "metadata": self.metadata,
            # Computed fields for query efficiency
            "problem_class": self.problem_class,
            "problem_family": self.problem_family,
            "n_variables": self.n_variables,
            "n_objectives": self.n_objectives,
            "n_constraints": self.n_constraints,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationProblem":
        """Deserialize from dictionary."""
        # Remove computed fields (they'll be recomputed)
        data = dict(data)
        data.pop("problem_class", None)
        data.pop("problem_family", None)
        # Keep n_* fields only if variables/objectives not present (legacy)
        if "variables" in data:
            data.pop("n_variables", None)
            data.pop("n_objectives", None)
            data.pop("n_constraints", None)

        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "OptimizationProblem":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # =========================================================================
    # Problem Derivation
    # =========================================================================

    def derive_narrow_bounds(
        self,
        new_problem_id: int,
        new_name: str,
        center: List[float],
        width_factor: float = 0.3,
        reason: Optional[str] = None,
    ) -> "OptimizationProblem":
        """
        Derive a new problem with narrowed bounds around a center point.

        Args:
            new_problem_id: ID for the derived problem
            new_name: Name for the derived problem
            center: Center point for new bounds (typically best_x from optimization)
            width_factor: Fraction of original width to use (0.3 = 30%)
            reason: Why this derivation was needed

        Returns:
            New OptimizationProblem with narrowed bounds
        """
        if len(center) != self.n_variables:
            raise ValueError(
                f"Center dimension ({len(center)}) != problem dimension ({self.n_variables})"
            )

        new_variables = []
        for i, v in enumerate(self.variables):
            original_width = v.upper - v.lower
            new_width = original_width * width_factor
            new_lb = max(v.lower, center[i] - new_width / 2)
            new_ub = min(v.upper, center[i] + new_width / 2)
            new_variables.append(Variable(
                name=v.name,
                type=v.type,
                lower=new_lb,
                upper=new_ub,
                description=v.description,
                unit=v.unit,
            ))

        return OptimizationProblem(
            problem_id=new_problem_id,
            name=new_name,
            variables=new_variables,
            objectives=self.objectives,  # Keep same objectives
            constraints=self.constraints,  # Keep same constraints
            description=self.description,
            domain_hint=self.domain_hint,
            parent_problem_id=self.problem_id,
            derivation_type=DerivationType.NARROW_BOUNDS,
            derivation_notes=reason or f"Narrowed bounds with width_factor={width_factor}",
            version=self.version + 1,
            metadata={
                **self.metadata,
                "derivation_spec": {
                    "center": center,
                    "width_factor": width_factor,
                }
            }
        )

    def derive_add_objective(
        self,
        new_problem_id: int,
        new_name: str,
        new_objective: Objective,
        reason: Optional[str] = None,
    ) -> "OptimizationProblem":
        """
        Derive a new problem with an additional objective (convert SOO to MOO).

        Args:
            new_problem_id: ID for the derived problem
            new_name: Name for the derived problem
            new_objective: Additional objective to add
            reason: Why this derivation was needed

        Returns:
            New OptimizationProblem with additional objective
        """
        return OptimizationProblem(
            problem_id=new_problem_id,
            name=new_name,
            variables=self.variables,
            objectives=self.objectives + [new_objective],
            constraints=self.constraints,
            description=self.description,
            domain_hint=self.domain_hint,
            parent_problem_id=self.problem_id,
            derivation_type=DerivationType.ADD_OBJECTIVE,
            derivation_notes=reason or f"Added objective: {new_objective.name}",
            version=self.version + 1,
            metadata=self.metadata,
        )

    # =========================================================================
    # String Representation
    # =========================================================================

    def __str__(self) -> str:
        """Human-readable representation."""
        lines = [
            f"OptimizationProblem: {self.name} (id={self.problem_id})",
            f"  Class: {self.problem_class}",
            f"  Variables: {self.n_variables} ({self.problem_family})",
        ]

        # Variable summary
        if self.n_variables <= 5:
            for v in self.variables:
                lines.append(f"    {v.name}: {v.type} [{v.lower}, {v.upper}]")
        else:
            lines.append(f"    {self.variables[0].name}, {self.variables[1].name}, ..., {self.variables[-1].name}")

        # Objectives
        lines.append(f"  Objectives: {self.n_objectives}")
        for obj in self.objectives:
            sense = "min" if obj.sense == "minimize" else "max"
            lines.append(f"    {sense} {obj.name} (eval: {obj.evaluator_id})")

        # Constraints
        if self.constraints:
            lines.append(f"  Constraints: {self.n_constraints}")
            for con in self.constraints[:3]:
                lines.append(f"    {con.name}: {con.type} {con.bound}")
            if self.n_constraints > 3:
                lines.append(f"    ... and {self.n_constraints - 3} more")

        # Lineage
        if self.is_derived:
            lines.append(f"  Derived from: {self.parent_problem_id} ({self.derivation_type})")

        return "\n".join(lines)


# =============================================================================
# Legacy Compatibility - Simple Problem class
# =============================================================================

@dataclass
class Problem:
    """
    Simple problem metadata (legacy compatibility).

    For new code, use OptimizationProblem instead.
    """
    problem_id: str
    name: str
    dimensions: int
    problem_type: str
    created_at: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Problem":
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Problem":
        return cls.from_dict(json.loads(json_str))

    def signature(self) -> Dict[str, Any]:
        return {
            "problem_type": self.problem_type,
            "dimensions": self.dimensions,
            **self.metadata.get("characteristics", {})
        }
