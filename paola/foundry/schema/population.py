"""
Population-based optimizer family components.

Covers: Differential Evolution, Genetic Algorithms, NSGA-II, PSO.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .components import InitializationComponent, ProgressComponent, ResultComponent


@dataclass
class PopulationInitialization(InitializationComponent):
    """Initialization for population-based optimizers."""

    specification: Dict[str, Any]
    method: str  # "lhs", "sobol", "random"
    population_size: int
    initial_population: List[List[float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "population",
            "specification": self.specification,
            "method": self.method,
            "population_size": self.population_size,
            "initial_population": self.initial_population,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PopulationInitialization':
        return cls(
            specification=data["specification"],
            method=data["method"],
            population_size=data["population_size"],
            initial_population=data["initial_population"],
        )


@dataclass
class Generation:
    """Single generation record for population-based optimizer."""

    generation: int
    best_objective: float
    best_design: List[float]
    mean_objective: float
    diversity: Optional[float] = None


@dataclass
class PopulationProgress(ProgressComponent):
    """Progress data for population-based optimizers."""

    generations: List[Generation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "population",
            "generations": [
                {
                    "generation": g.generation,
                    "best_objective": g.best_objective,
                    "best_design": g.best_design,
                    "mean_objective": g.mean_objective,
                    "diversity": g.diversity,
                }
                for g in self.generations
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PopulationProgress':
        generations = [
            Generation(
                generation=g["generation"],
                best_objective=g["best_objective"],
                best_design=g["best_design"],
                mean_objective=g["mean_objective"],
                diversity=g.get("diversity"),
            )
            for g in data["generations"]
        ]
        return cls(generations=generations)

    def add_generation(
        self,
        generation: int,
        best_objective: float,
        best_design: List[float],
        mean_objective: float,
        diversity: Optional[float] = None,
    ):
        """Add a generation record."""
        self.generations.append(
            Generation(
                generation=generation,
                best_objective=best_objective,
                best_design=best_design,
                mean_objective=mean_objective,
                diversity=diversity,
            )
        )


@dataclass
class PopulationResult(ResultComponent):
    """Detailed result for population-based optimizers."""

    termination_reason: str
    final_population_size: int
    final_diversity: Optional[float] = None
    pareto_front: Optional[List[List[float]]] = None  # For multi-objective

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "population",
            "termination_reason": self.termination_reason,
            "final_population_size": self.final_population_size,
            "final_diversity": self.final_diversity,
            "pareto_front": self.pareto_front,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PopulationResult':
        return cls(
            termination_reason=data["termination_reason"],
            final_population_size=data["final_population_size"],
            final_diversity=data.get("final_diversity"),
            pareto_front=data.get("pareto_front"),
        )
