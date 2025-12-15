"""
CMA-ES optimizer family components.

CMA-ES has unique structure (mean + covariance evolution) that warrants
its own family rather than grouping with general population methods.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .components import InitializationComponent, ProgressComponent, ResultComponent


@dataclass
class CMAESInitialization(InitializationComponent):
    """Initialization for CMA-ES optimizer."""

    specification: Dict[str, Any]
    mean: List[float]
    sigma: float
    population_size: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "cmaes",
            "specification": self.specification,
            "mean": self.mean,
            "sigma": self.sigma,
            "population_size": self.population_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CMAESInitialization':
        return cls(
            specification=data["specification"],
            mean=data["mean"],
            sigma=data["sigma"],
            population_size=data["population_size"],
        )


@dataclass
class CMAESGeneration:
    """Single generation record for CMA-ES."""

    generation: int
    best_objective: float
    best_design: List[float]
    mean: List[float]
    sigma: float
    condition_number: Optional[float] = None


@dataclass
class CMAESProgress(ProgressComponent):
    """Progress data for CMA-ES optimizer."""

    generations: List[CMAESGeneration] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "cmaes",
            "generations": [
                {
                    "generation": g.generation,
                    "best_objective": g.best_objective,
                    "best_design": g.best_design,
                    "mean": g.mean,
                    "sigma": g.sigma,
                    "condition_number": g.condition_number,
                }
                for g in self.generations
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CMAESProgress':
        generations = [
            CMAESGeneration(
                generation=g["generation"],
                best_objective=g["best_objective"],
                best_design=g["best_design"],
                mean=g["mean"],
                sigma=g["sigma"],
                condition_number=g.get("condition_number"),
            )
            for g in data["generations"]
        ]
        return cls(generations=generations)

    def add_generation(
        self,
        generation: int,
        best_objective: float,
        best_design: List[float],
        mean: List[float],
        sigma: float,
        condition_number: Optional[float] = None,
    ):
        """Add a generation record."""
        self.generations.append(
            CMAESGeneration(
                generation=generation,
                best_objective=best_objective,
                best_design=best_design,
                mean=mean,
                sigma=sigma,
                condition_number=condition_number,
            )
        )


@dataclass
class CMAESResult(ResultComponent):
    """Detailed result for CMA-ES optimizer."""

    termination_reason: str
    final_mean: List[float]
    final_sigma: float
    final_condition_number: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "cmaes",
            "termination_reason": self.termination_reason,
            "final_mean": self.final_mean,
            "final_sigma": self.final_sigma,
            "final_condition_number": self.final_condition_number,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CMAESResult':
        return cls(
            termination_reason=data["termination_reason"],
            final_mean=data["final_mean"],
            final_sigma=data["final_sigma"],
            final_condition_number=data.get("final_condition_number"),
        )
