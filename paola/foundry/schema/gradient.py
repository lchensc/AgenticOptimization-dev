"""
Gradient-based optimizer family components.

Covers: SciPy (SLSQP, L-BFGS-B, trust-constr), IPOPT, NLopt gradient methods.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .components import InitializationComponent, ProgressComponent, ResultComponent


@dataclass
class GradientInitialization(InitializationComponent):
    """Initialization for gradient-based optimizers."""

    specification: Dict[str, Any]
    x0: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "gradient",
            "specification": self.specification,
            "x0": self.x0,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GradientInitialization':
        return cls(
            specification=data["specification"],
            x0=data["x0"],
        )


@dataclass
class GradientIteration:
    """Single iteration record for gradient-based optimizer."""

    iteration: int
    objective: float
    design: List[float]
    gradient_norm: Optional[float] = None
    step_size: Optional[float] = None
    constraint_violation: Optional[float] = None


@dataclass
class GradientProgress(ProgressComponent):
    """Progress data for gradient-based optimizers."""

    iterations: List[GradientIteration] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "gradient",
            "iterations": [
                {
                    "iteration": it.iteration,
                    "objective": it.objective,
                    "design": it.design,
                    "gradient_norm": it.gradient_norm,
                    "step_size": it.step_size,
                    "constraint_violation": it.constraint_violation,
                }
                for it in self.iterations
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GradientProgress':
        iterations = [
            GradientIteration(
                iteration=it["iteration"],
                objective=it["objective"],
                design=it["design"],
                gradient_norm=it.get("gradient_norm"),
                step_size=it.get("step_size"),
                constraint_violation=it.get("constraint_violation"),
            )
            for it in data["iterations"]
        ]
        return cls(iterations=iterations)

    def add_iteration(
        self,
        iteration: int,
        objective: float,
        design: List[float],
        gradient_norm: Optional[float] = None,
        step_size: Optional[float] = None,
        constraint_violation: Optional[float] = None,
    ):
        """Add an iteration record."""
        self.iterations.append(
            GradientIteration(
                iteration=iteration,
                objective=objective,
                design=design,
                gradient_norm=gradient_norm,
                step_size=step_size,
                constraint_violation=constraint_violation,
            )
        )


@dataclass
class GradientResult(ResultComponent):
    """Detailed result for gradient-based optimizers."""

    termination_reason: str
    final_gradient_norm: Optional[float] = None
    final_constraint_violation: Optional[float] = None
    lagrange_multipliers: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "gradient",
            "termination_reason": self.termination_reason,
            "final_gradient_norm": self.final_gradient_norm,
            "final_constraint_violation": self.final_constraint_violation,
            "lagrange_multipliers": self.lagrange_multipliers,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GradientResult':
        return cls(
            termination_reason=data["termination_reason"],
            final_gradient_norm=data.get("final_gradient_norm"),
            final_constraint_violation=data.get("final_constraint_violation"),
            lagrange_multipliers=data.get("lagrange_multipliers"),
        )
