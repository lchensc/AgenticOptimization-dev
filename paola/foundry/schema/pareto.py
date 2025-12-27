"""
Pareto Front Schema for Multi-Objective Optimization.

Provides structured storage and querying of Pareto-optimal solutions.

Key classes:
- ParetoSolution: Single solution on Pareto front
- ParetoFront: Collection of solutions with query methods
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParetoSolution:
    """
    Single solution on Pareto front.

    Attributes:
        x: Design variables (decision vector)
        f: Objective values
        rank: Pareto rank (0 = non-dominated first front)
        crowding_distance: Diversity metric (higher = more isolated)
        feasible: Whether solution satisfies all constraints
        constraint_violation: Total constraint violation (0 = feasible)
        metadata: Optional additional data
    """
    x: np.ndarray
    f: np.ndarray
    rank: int = 0
    crowding_distance: float = 0.0
    feasible: bool = True
    constraint_violation: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "x": self.x.tolist() if isinstance(self.x, np.ndarray) else list(self.x),
            "f": self.f.tolist() if isinstance(self.f, np.ndarray) else list(self.f),
            "rank": self.rank,
            "crowding_distance": self.crowding_distance,
            "feasible": self.feasible,
            "constraint_violation": self.constraint_violation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParetoSolution":
        """Deserialize from dictionary."""
        return cls(
            x=np.array(data["x"]),
            f=np.array(data["f"]),
            rank=data.get("rank", 0),
            crowding_distance=data.get("crowding_distance", 0.0),
            feasible=data.get("feasible", True),
            constraint_violation=data.get("constraint_violation", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ParetoFront:
    """
    Structured Pareto front with query capabilities.

    Provides methods for:
    - Filtering by objective values
    - Finding extreme points
    - Finding knee point (best trade-off)
    - Computing hypervolume

    Attributes:
        solutions: List of Pareto-optimal solutions
        objective_names: Names of objectives (e.g., ["drag", "weight"])
        objective_senses: Optimization directions (e.g., ["minimize", "minimize"])
        hypervolume: Hypervolume indicator value
        reference_point: Reference point for hypervolume computation
        graph_id: Source graph ID
        node_id: Source node ID
        algorithm: Algorithm used (e.g., "NSGA-II")
        n_generations: Number of generations run
    """
    solutions: List[ParetoSolution]
    objective_names: List[str]
    objective_senses: List[str] = field(default_factory=lambda: ["minimize", "minimize"])
    hypervolume: Optional[float] = None
    reference_point: Optional[np.ndarray] = None
    graph_id: Optional[int] = None
    node_id: Optional[str] = None
    algorithm: Optional[str] = None
    n_generations: Optional[int] = None

    def __post_init__(self):
        """Validate and normalize."""
        # Convert solutions from dicts if needed
        self.solutions = [
            ParetoSolution.from_dict(s) if isinstance(s, dict) else s
            for s in self.solutions
        ]
        # Convert reference_point to numpy
        if self.reference_point is not None and not isinstance(self.reference_point, np.ndarray):
            self.reference_point = np.array(self.reference_point)

    @property
    def n_solutions(self) -> int:
        """Number of solutions in Pareto front."""
        return len(self.solutions)

    @property
    def n_objectives(self) -> int:
        """Number of objectives."""
        return len(self.objective_names)

    @property
    def pareto_set(self) -> np.ndarray:
        """Get design variables as (n_solutions, n_vars) array."""
        if not self.solutions:
            return np.array([])
        return np.array([s.x for s in self.solutions])

    @property
    def pareto_front_array(self) -> np.ndarray:
        """Get objective values as (n_solutions, n_obj) array."""
        if not self.solutions:
            return np.array([])
        return np.array([s.f for s in self.solutions])

    def get_objective_index(self, name: str) -> int:
        """Get index of objective by name."""
        try:
            return self.objective_names.index(name)
        except ValueError:
            raise ValueError(f"Unknown objective: {name}. Available: {self.objective_names}")

    # =========================================================================
    # Filtering Methods
    # =========================================================================

    def filter_by_objective(
        self,
        objective: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> "ParetoFront":
        """
        Filter solutions by objective value range.

        Args:
            objective: Objective name
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)

        Returns:
            New ParetoFront with filtered solutions
        """
        idx = self.get_objective_index(objective)
        filtered = []
        for sol in self.solutions:
            val = sol.f[idx]
            if min_val is not None and val < min_val:
                continue
            if max_val is not None and val > max_val:
                continue
            filtered.append(sol)

        return ParetoFront(
            solutions=filtered,
            objective_names=self.objective_names,
            objective_senses=self.objective_senses,
            graph_id=self.graph_id,
            node_id=self.node_id,
            algorithm=self.algorithm,
        )

    def filter_feasible(self) -> "ParetoFront":
        """Return only feasible solutions."""
        filtered = [s for s in self.solutions if s.feasible]
        return ParetoFront(
            solutions=filtered,
            objective_names=self.objective_names,
            objective_senses=self.objective_senses,
            graph_id=self.graph_id,
            node_id=self.node_id,
            algorithm=self.algorithm,
        )

    # =========================================================================
    # Selection Methods
    # =========================================================================

    def get_extreme(self, objective: str) -> Optional[ParetoSolution]:
        """
        Get solution with best (minimum or maximum based on sense) value for an objective.

        Args:
            objective: Objective name

        Returns:
            Solution with extreme value, or None if empty
        """
        if not self.solutions:
            return None

        idx = self.get_objective_index(objective)
        sense = self.objective_senses[idx]

        if sense == "minimize":
            return min(self.solutions, key=lambda s: s.f[idx])
        else:
            return max(self.solutions, key=lambda s: s.f[idx])

    def get_knee_point(self) -> Optional[ParetoSolution]:
        """
        Find knee point (best trade-off solution).

        Uses the maximum distance to the utopia-nadir line method.
        Works best for 2-objective problems.

        Returns:
            Knee point solution, or None if empty
        """
        if not self.solutions:
            return None

        if len(self.solutions) == 1:
            return self.solutions[0]

        # Get Pareto front array
        F = self.pareto_front_array

        # Normalize objectives to [0, 1]
        f_min = F.min(axis=0)
        f_max = F.max(axis=0)
        range_vals = f_max - f_min
        range_vals[range_vals < 1e-10] = 1.0  # Avoid division by zero
        F_norm = (F - f_min) / range_vals

        # For 2 objectives: distance to utopia-nadir line
        if F_norm.shape[1] == 2:
            # Line from (0,1) to (1,0) in normalized space
            # Distance = |x + y - 1| / sqrt(2)
            distances = np.abs(F_norm[:, 0] + F_norm[:, 1] - 1) / np.sqrt(2)
            knee_idx = int(np.argmax(distances))
        else:
            # For many objectives: use distance to ideal point
            # Weighted by inverse of range
            ideal = np.zeros(F_norm.shape[1])  # All zeros after normalization
            distances = np.linalg.norm(F_norm - ideal, axis=1)
            knee_idx = int(np.argmin(distances))

        return self.solutions[knee_idx]

    def get_closest_to(self, target: Dict[str, float]) -> Optional[ParetoSolution]:
        """
        Find solution closest to target objective values.

        Args:
            target: Dict of {objective_name: target_value}

        Returns:
            Closest solution, or None if empty
        """
        if not self.solutions:
            return None

        F = self.pareto_front_array

        # Normalize
        f_min = F.min(axis=0)
        f_max = F.max(axis=0)
        range_vals = f_max - f_min
        range_vals[range_vals < 1e-10] = 1.0

        F_norm = (F - f_min) / range_vals

        # Build target vector
        target_vec = np.zeros(self.n_objectives)
        for name, val in target.items():
            idx = self.get_objective_index(name)
            target_vec[idx] = (val - f_min[idx]) / range_vals[idx]

        # Find closest
        distances = np.linalg.norm(F_norm - target_vec, axis=1)
        closest_idx = int(np.argmin(distances))

        return self.solutions[closest_idx]

    def get_by_rank(self, rank: int = 0) -> List[ParetoSolution]:
        """Get solutions with specified Pareto rank."""
        return [s for s in self.solutions if s.rank == rank]

    def get_most_diverse(self, n: int = 1) -> List[ParetoSolution]:
        """Get n solutions with highest crowding distance (most diverse)."""
        sorted_solutions = sorted(
            self.solutions,
            key=lambda s: s.crowding_distance,
            reverse=True
        )
        return sorted_solutions[:n]

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def compute_hypervolume(self, reference_point: Optional[np.ndarray] = None) -> float:
        """
        Compute hypervolume indicator.

        Args:
            reference_point: Reference point (default: max + 10% margin)

        Returns:
            Hypervolume value
        """
        if not self.solutions:
            return 0.0

        try:
            from pymoo.indicators.hv import HV
        except ImportError:
            logger.warning("pymoo not installed, cannot compute hypervolume")
            return 0.0

        F = self.pareto_front_array

        if reference_point is None:
            if self.reference_point is not None:
                reference_point = self.reference_point
            else:
                reference_point = F.max(axis=0) * 1.1

        hv = HV(ref_point=reference_point)
        self.hypervolume = float(hv(F))
        self.reference_point = reference_point

        return self.hypervolume

    def get_spread(self) -> Dict[str, Tuple[float, float]]:
        """
        Get min/max range for each objective.

        Returns:
            Dict of {objective_name: (min, max)}
        """
        if not self.solutions:
            return {}

        F = self.pareto_front_array
        spread = {}
        for i, name in enumerate(self.objective_names):
            spread[name] = (float(F[:, i].min()), float(F[:, i].max()))

        return spread

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics for agent reasoning."""
        spread = self.get_spread()

        return {
            "n_solutions": self.n_solutions,
            "n_objectives": self.n_objectives,
            "objective_names": self.objective_names,
            "hypervolume": self.hypervolume,
            "spread": spread,
            "n_feasible": sum(1 for s in self.solutions if s.feasible),
            "algorithm": self.algorithm,
            "graph_id": self.graph_id,
            "node_id": self.node_id,
        }

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "solutions": [s.to_dict() for s in self.solutions],
            "objective_names": self.objective_names,
            "objective_senses": self.objective_senses,
            "hypervolume": self.hypervolume,
            "reference_point": (
                self.reference_point.tolist()
                if self.reference_point is not None
                else None
            ),
            "graph_id": self.graph_id,
            "node_id": self.node_id,
            "algorithm": self.algorithm,
            "n_generations": self.n_generations,
            "n_solutions": self.n_solutions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParetoFront":
        """Deserialize from dictionary."""
        data = dict(data)
        data.pop("n_solutions", None)  # Computed property

        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ParetoFront":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_arrays(
        cls,
        pareto_set: np.ndarray,
        pareto_front: np.ndarray,
        objective_names: List[str],
        objective_senses: Optional[List[str]] = None,
        **kwargs
    ) -> "ParetoFront":
        """
        Create ParetoFront from numpy arrays.

        Args:
            pareto_set: Design variables (n_solutions, n_vars)
            pareto_front: Objective values (n_solutions, n_obj)
            objective_names: Names of objectives
            objective_senses: Optimization senses (default: all minimize)
            **kwargs: Additional attributes

        Returns:
            ParetoFront instance
        """
        n_solutions = len(pareto_set)
        n_objectives = len(objective_names)

        if objective_senses is None:
            objective_senses = ["minimize"] * n_objectives

        solutions = []
        for i in range(n_solutions):
            solutions.append(ParetoSolution(
                x=pareto_set[i],
                f=pareto_front[i],
            ))

        return cls(
            solutions=solutions,
            objective_names=objective_names,
            objective_senses=objective_senses,
            **kwargs
        )

    def to_dataframe(self):
        """
        Convert to pandas DataFrame for analysis/visualization.

        Returns:
            DataFrame with columns for each variable and objective
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for to_dataframe()")

        if not self.solutions:
            return pd.DataFrame()

        # Build data
        data = []
        for i, sol in enumerate(self.solutions):
            row = {"solution_id": i}
            # Add objectives
            for j, name in enumerate(self.objective_names):
                row[name] = sol.f[j]
            # Add variables
            for j, val in enumerate(sol.x):
                row[f"x{j}"] = val
            # Add metadata
            row["rank"] = sol.rank
            row["crowding_distance"] = sol.crowding_distance
            row["feasible"] = sol.feasible
            data.append(row)

        return pd.DataFrame(data)

    def __str__(self) -> str:
        """Human-readable representation."""
        lines = [
            f"ParetoFront: {self.n_solutions} solutions, {self.n_objectives} objectives",
            f"  Objectives: {self.objective_names}",
        ]
        if self.hypervolume is not None:
            lines.append(f"  Hypervolume: {self.hypervolume:.4f}")

        spread = self.get_spread()
        for name, (lo, hi) in spread.items():
            lines.append(f"  {name}: [{lo:.4f}, {hi:.4f}]")

        if self.algorithm:
            lines.append(f"  Algorithm: {self.algorithm}")

        return "\n".join(lines)
