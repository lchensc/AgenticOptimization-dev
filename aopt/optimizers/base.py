"""
Base optimizer interface for the agentic optimization platform.

All optimizer wrappers must implement this interface to work with
the agent's optimizer tools.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass, field
import json


@dataclass
class OptimizerState:
    """
    Optimizer state for checkpointing and restart.

    Contains all information needed to restore an optimizer to
    a previous state or restart from a specific design.
    """

    # Current design and objective
    current_design: np.ndarray
    current_objective: float
    current_gradient: Optional[np.ndarray] = None
    current_constraints: Optional[Dict[str, float]] = None

    # Best design seen so far
    best_design: np.ndarray = field(default=None)
    best_objective: float = field(default=float("inf"))

    # Iteration count
    iteration: int = 0

    # Optimizer-specific state (e.g., Hessian approximation, trust region)
    optimizer_data: Dict[str, Any] = field(default_factory=dict)

    # Convergence info
    converged: bool = False
    convergence_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "current_design": self.current_design.tolist() if self.current_design is not None else None,
            "current_objective": self.current_objective,
            "current_gradient": self.current_gradient.tolist() if self.current_gradient is not None else None,
            "current_constraints": self.current_constraints,
            "best_design": self.best_design.tolist() if self.best_design is not None else None,
            "best_objective": self.best_objective,
            "iteration": self.iteration,
            "optimizer_data": self.optimizer_data,
            "converged": self.converged,
            "convergence_reason": self.convergence_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizerState":
        """Deserialize state from dictionary."""
        return cls(
            current_design=np.array(data["current_design"]) if data["current_design"] is not None else None,
            current_objective=data["current_objective"],
            current_gradient=np.array(data["current_gradient"]) if data.get("current_gradient") is not None else None,
            current_constraints=data.get("current_constraints"),
            best_design=np.array(data["best_design"]) if data.get("best_design") is not None else None,
            best_objective=data.get("best_objective", float("inf")),
            iteration=data.get("iteration", 0),
            optimizer_data=data.get("optimizer_data", {}),
            converged=data.get("converged", False),
            convergence_reason=data.get("convergence_reason"),
        )


class BaseOptimizer(ABC):
    """
    Abstract base class for all optimizers.

    Defines the interface that the agent's optimizer tools expect.
    All optimizer wrappers (scipy, pymoo, custom) must implement this.
    """

    def __init__(
        self,
        problem_id: str,
        algorithm: str,
        bounds: Tuple[np.ndarray, np.ndarray],
        initial_design: Optional[np.ndarray] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize optimizer.

        Args:
            problem_id: Unique problem identifier
            algorithm: Algorithm name (e.g., "SLSQP", "COBYLA")
            bounds: (lower_bounds, upper_bounds) tuple
            initial_design: Starting design point
            options: Algorithm-specific options
        """
        self.problem_id = problem_id
        self.algorithm = algorithm
        self.bounds = bounds
        self.initial_design = initial_design
        self.options = options or {}

        # State tracking
        self.state = OptimizerState(
            current_design=initial_design.copy() if initial_design is not None else None,
            current_objective=float("inf"),
            best_design=initial_design.copy() if initial_design is not None else None,
            best_objective=float("inf"),
        )

        # Evaluation history
        self.history: list[Dict[str, Any]] = []

    @abstractmethod
    def propose_design(self) -> np.ndarray:
        """
        Propose next design to evaluate.

        Returns:
            Design vector to evaluate

        Raises:
            StopIteration: If optimizer has converged
        """
        pass

    @abstractmethod
    def update(
        self,
        design: np.ndarray,
        objective: float,
        gradient: Optional[np.ndarray] = None,
        constraints: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Update optimizer with evaluation results.

        Args:
            design: Evaluated design
            objective: Objective value
            gradient: Gradient vector (if available)
            constraints: Constraint values (if any)

        Returns:
            Update info dict with keys:
                - converged: bool
                - reason: Optional[str]
                - improvement: float (objective change)
                - gradient_norm: Optional[float]
                - step_size: Optional[float]
        """
        pass

    @abstractmethod
    def checkpoint(self) -> Dict[str, Any]:
        """
        Create checkpoint of current optimizer state.

        Returns:
            Serializable checkpoint dictionary
        """
        pass

    @abstractmethod
    def restore(self, checkpoint: Dict[str, Any]):
        """
        Restore optimizer from checkpoint.

        Args:
            checkpoint: Checkpoint dictionary from checkpoint()
        """
        pass

    @abstractmethod
    def restart_from_design(
        self,
        design: np.ndarray,
        objective: float,
        gradient: Optional[np.ndarray] = None,
        new_options: Optional[Dict[str, Any]] = None,
    ):
        """
        Restart optimizer from specified design.

        Used for strategic restarts after constraint tightening,
        gradient method switching, etc.

        Args:
            design: Design to restart from
            objective: Objective at restart design
            gradient: Gradient at restart design (if available)
            new_options: New algorithm options (if changing settings)
        """
        pass

    def get_state(self) -> OptimizerState:
        """Get current optimizer state."""
        return self.state

    def get_best(self) -> Tuple[np.ndarray, float]:
        """
        Get best design found so far.

        Returns:
            (best_design, best_objective) tuple
        """
        return self.state.best_design.copy(), self.state.best_objective

    def get_history(self) -> list[Dict[str, Any]]:
        """Get evaluation history."""
        return self.history.copy()

    def is_converged(self) -> bool:
        """Check if optimizer has converged."""
        return self.state.converged

    def get_convergence_info(self) -> Dict[str, Any]:
        """
        Get convergence information.

        Returns:
            Dict with:
                - converged: bool
                - reason: Optional[str]
                - iterations: int
                - best_objective: float
        """
        return {
            "converged": self.state.converged,
            "reason": self.state.convergence_reason,
            "iterations": self.state.iteration,
            "best_objective": self.state.best_objective,
        }

    def _update_best(self, design: np.ndarray, objective: float):
        """Update best design if improved."""
        if objective < self.state.best_objective:
            self.state.best_design = design.copy()
            self.state.best_objective = objective

    def _record_evaluation(
        self,
        design: np.ndarray,
        objective: float,
        gradient: Optional[np.ndarray] = None,
        constraints: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record evaluation in history."""
        record = {
            "iteration": self.state.iteration,
            "design": design.copy(),
            "objective": objective,
            "gradient": gradient.copy() if gradient is not None else None,
            "constraints": constraints.copy() if constraints else None,
            "is_best": objective < self.state.best_objective,
        }
        if metadata:
            record.update(metadata)

        self.history.append(record)
