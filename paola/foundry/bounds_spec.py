"""
Compact bounds specification for large variable spaces.

The Paola Principle: "Optimization complexity is Paola intelligence, not user burden."

For large-scale problems (100+ variables), specifying bounds as [[lb, ub], ...] × 100
is impractical. BoundsSpec provides compact representations:

1. Uniform bounds: All variables share the same bounds
   {"type": "uniform", "lower": -0.05, "upper": 0.05, "dimension": 100}

2. Grouped bounds: Variables grouped by role
   {"type": "grouped", "groups": {
       "control_points": {"lower": -0.05, "upper": 0.05, "count": 80},
       "angles": {"lower": -15.0, "upper": 15.0, "count": 20}
   }}

3. Evaluator-derived: Evaluator provides bounds (e.g., from mesh or physics)
   {"type": "evaluator_derived", "evaluator_id": "mesh_bounds"}
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Literal
import numpy as np


@dataclass
class BoundsGroup:
    """A group of variables with shared bounds."""
    lower: float
    upper: float
    count: int
    name: Optional[str] = None

    def __post_init__(self):
        """Validate group specification."""
        if self.lower >= self.upper:
            raise ValueError(f"Invalid bounds: lower ({self.lower}) >= upper ({self.upper})")
        if self.count <= 0:
            raise ValueError(f"Invalid count: {self.count}")

    def expand(self) -> List[List[float]]:
        """Expand to explicit bounds list."""
        return [[self.lower, self.upper] for _ in range(self.count)]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "name": self.name
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], name: str = None) -> 'BoundsGroup':
        """Deserialize from dictionary."""
        return cls(
            lower=data["lower"],
            upper=data["upper"],
            count=data["count"],
            name=data.get("name", name)
        )


@dataclass
class BoundsSpec:
    """
    Compact bounds specification for large variable spaces.

    Supports three specification types:
    - uniform: All variables share the same bounds
    - grouped: Variables are grouped by role with different bounds
    - evaluator_derived: Bounds come from an evaluator (e.g., mesh geometry)

    Examples:
        # Uniform bounds for 100 FFD control points
        spec = BoundsSpec(
            spec_type="uniform",
            lower=-0.05,
            upper=0.05,
            dimension=100
        )

        # Grouped bounds for wing design
        spec = BoundsSpec(
            spec_type="grouped",
            groups={
                "control_points": BoundsGroup(-0.05, 0.05, 80),
                "twist_angles": BoundsGroup(-15.0, 15.0, 20)
            }
        )

        # Evaluator-derived bounds
        spec = BoundsSpec(
            spec_type="evaluator_derived",
            evaluator_id="mesh_bounds"
        )
    """

    spec_type: Literal["uniform", "grouped", "evaluator_derived", "explicit"]

    # For uniform bounds
    lower: Optional[float] = None
    upper: Optional[float] = None
    dimension: Optional[int] = None

    # For grouped bounds
    groups: Optional[Dict[str, BoundsGroup]] = None

    # For evaluator-derived bounds
    evaluator_id: Optional[str] = None

    # For explicit bounds (fallback, not recommended for large spaces)
    explicit_bounds: Optional[List[List[float]]] = None

    def __post_init__(self):
        """Validate specification."""
        if self.spec_type == "uniform":
            if self.lower is None or self.upper is None or self.dimension is None:
                raise ValueError(
                    "Uniform bounds require: lower, upper, and dimension"
                )
            if self.lower >= self.upper:
                raise ValueError(f"Invalid bounds: lower ({self.lower}) >= upper ({self.upper})")
            if self.dimension <= 0:
                raise ValueError(f"Invalid dimension: {self.dimension}")

        elif self.spec_type == "grouped":
            if not self.groups:
                raise ValueError("Grouped bounds require: groups dictionary")

        elif self.spec_type == "evaluator_derived":
            if not self.evaluator_id:
                raise ValueError("Evaluator-derived bounds require: evaluator_id")

        elif self.spec_type == "explicit":
            if not self.explicit_bounds:
                raise ValueError("Explicit bounds require: explicit_bounds list")

    def get_dimension(self) -> int:
        """Get total dimension of the variable space."""
        if self.spec_type == "uniform":
            return self.dimension

        elif self.spec_type == "grouped":
            return sum(g.count for g in self.groups.values())

        elif self.spec_type == "explicit":
            return len(self.explicit_bounds)

        elif self.spec_type == "evaluator_derived":
            raise ValueError("Dimension unknown for evaluator-derived bounds until resolved")

        raise ValueError(f"Unknown spec_type: {self.spec_type}")

    def expand(self, evaluator_result: Optional[List[List[float]]] = None) -> List[List[float]]:
        """
        Expand to explicit bounds list.

        Args:
            evaluator_result: For evaluator_derived type, the bounds from evaluator

        Returns:
            List of [lower, upper] for each variable
        """
        if self.spec_type == "uniform":
            return [[self.lower, self.upper] for _ in range(self.dimension)]

        elif self.spec_type == "grouped":
            bounds = []
            for group in self.groups.values():
                bounds.extend(group.expand())
            return bounds

        elif self.spec_type == "explicit":
            return self.explicit_bounds

        elif self.spec_type == "evaluator_derived":
            if evaluator_result is None:
                raise ValueError("Evaluator result required for evaluator_derived bounds")
            return evaluator_result

        raise ValueError(f"Unknown spec_type: {self.spec_type}")

    def get_center(self, evaluator_result: Optional[List[List[float]]] = None) -> np.ndarray:
        """Get center of bounds as numpy array."""
        bounds = self.expand(evaluator_result)
        return np.array([(b[0] + b[1]) / 2 for b in bounds])

    def get_width(self, evaluator_result: Optional[List[List[float]]] = None) -> np.ndarray:
        """Get width of bounds as numpy array."""
        bounds = self.expand(evaluator_result)
        return np.array([b[1] - b[0] for b in bounds])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = {"spec_type": self.spec_type}

        if self.spec_type == "uniform":
            data.update({
                "lower": self.lower,
                "upper": self.upper,
                "dimension": self.dimension
            })

        elif self.spec_type == "grouped":
            data["groups"] = {
                name: group.to_dict()
                for name, group in self.groups.items()
            }

        elif self.spec_type == "evaluator_derived":
            data["evaluator_id"] = self.evaluator_id

        elif self.spec_type == "explicit":
            data["explicit_bounds"] = self.explicit_bounds

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BoundsSpec':
        """Deserialize from dictionary."""
        spec_type = data.get("spec_type", data.get("type"))  # Support both

        if spec_type == "uniform":
            return cls(
                spec_type="uniform",
                lower=data["lower"],
                upper=data["upper"],
                dimension=data["dimension"]
            )

        elif spec_type == "grouped":
            groups = {
                name: BoundsGroup.from_dict(group_data, name)
                for name, group_data in data["groups"].items()
            }
            return cls(spec_type="grouped", groups=groups)

        elif spec_type == "evaluator_derived":
            return cls(
                spec_type="evaluator_derived",
                evaluator_id=data["evaluator_id"]
            )

        elif spec_type == "explicit":
            return cls(
                spec_type="explicit",
                explicit_bounds=data["explicit_bounds"]
            )

        raise ValueError(f"Unknown spec_type: {spec_type}")

    @classmethod
    def from_explicit(cls, bounds: List[List[float]]) -> 'BoundsSpec':
        """Create BoundsSpec from explicit bounds list."""
        return cls(spec_type="explicit", explicit_bounds=bounds)

    @classmethod
    def uniform(cls, lower: float, upper: float, dimension: int) -> 'BoundsSpec':
        """Create uniform bounds specification."""
        return cls(spec_type="uniform", lower=lower, upper=upper, dimension=dimension)

    def __str__(self) -> str:
        """Human-readable representation."""
        if self.spec_type == "uniform":
            return f"BoundsSpec(uniform: [{self.lower}, {self.upper}] × {self.dimension})"

        elif self.spec_type == "grouped":
            group_strs = [
                f"{name}: [{g.lower}, {g.upper}] × {g.count}"
                for name, g in self.groups.items()
            ]
            return f"BoundsSpec(grouped: {', '.join(group_strs)})"

        elif self.spec_type == "evaluator_derived":
            return f"BoundsSpec(evaluator_derived: {self.evaluator_id})"

        elif self.spec_type == "explicit":
            return f"BoundsSpec(explicit: {len(self.explicit_bounds)} bounds)"

        return f"BoundsSpec({self.spec_type})"


def parse_bounds_input(
    bounds_input: Union[List[List[float]], Dict[str, Any], BoundsSpec]
) -> BoundsSpec:
    """
    Parse various bounds input formats into BoundsSpec.

    Args:
        bounds_input: One of:
            - List of [lower, upper] pairs (explicit)
            - Dictionary with spec_type and parameters
            - BoundsSpec object

    Returns:
        BoundsSpec object

    Examples:
        # Explicit bounds
        spec = parse_bounds_input([[0, 1], [0, 2], [0, 3]])

        # Uniform bounds dict
        spec = parse_bounds_input({
            "type": "uniform",
            "lower": -0.05,
            "upper": 0.05,
            "dimension": 100
        })

        # Grouped bounds dict
        spec = parse_bounds_input({
            "type": "grouped",
            "groups": {
                "x": {"lower": 0, "upper": 1, "count": 50},
                "y": {"lower": -1, "upper": 1, "count": 50}
            }
        })
    """
    if isinstance(bounds_input, BoundsSpec):
        return bounds_input

    if isinstance(bounds_input, list):
        # Validate and create explicit bounds
        for i, bound in enumerate(bounds_input):
            if not isinstance(bound, (list, tuple)) or len(bound) != 2:
                raise ValueError(f"Bound {i} must be [lower, upper], got {bound}")
            if bound[0] >= bound[1]:
                raise ValueError(f"Bound {i}: lower ({bound[0]}) >= upper ({bound[1]})")
        return BoundsSpec.from_explicit(bounds_input)

    if isinstance(bounds_input, dict):
        return BoundsSpec.from_dict(bounds_input)

    raise ValueError(f"Unsupported bounds input type: {type(bounds_input)}")
