"""
Evolutionary algorithm family components.

Covers: pymoo (GA, DE, PSO, ES, BRKGA), and other evolutionary optimizers.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .components import InitializationComponent, ProgressComponent, ResultComponent


@dataclass
class EvolutionaryInitialization(InitializationComponent):
    """Initialization for evolutionary algorithms."""

    specification: Dict[str, Any]
    pop_size: int
    seed: Optional[int] = None
    initial_population: Optional[List[List[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "evolutionary",
            "specification": self.specification,
            "pop_size": self.pop_size,
            "seed": self.seed,
            "initial_population": self.initial_population,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionaryInitialization":
        return cls(
            specification=data["specification"],
            pop_size=data["pop_size"],
            seed=data.get("seed"),
            initial_population=data.get("initial_population"),
        )


@dataclass
class Generation:
    """Single generation record for evolutionary optimizer."""

    generation: int
    best_fitness: float
    mean_fitness: float
    best_individual: List[float]
    diversity: Optional[float] = None
    constraint_violation: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "best_individual": self.best_individual,
            "diversity": self.diversity,
            "constraint_violation": self.constraint_violation,
        }


@dataclass
class EvolutionaryProgress(ProgressComponent):
    """Progress data for evolutionary algorithms."""

    generations: List[Generation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "evolutionary",
            "generations": [g.to_dict() for g in self.generations],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionaryProgress":
        generations = [
            Generation(
                generation=g["generation"],
                best_fitness=g["best_fitness"],
                mean_fitness=g["mean_fitness"],
                best_individual=g["best_individual"],
                diversity=g.get("diversity"),
                constraint_violation=g.get("constraint_violation"),
            )
            for g in data["generations"]
        ]
        return cls(generations=generations)

    def add_generation(
        self,
        generation: int,
        best_fitness: float,
        mean_fitness: float,
        best_individual: List[float],
        diversity: Optional[float] = None,
        constraint_violation: Optional[float] = None,
    ):
        """Add a generation record."""
        self.generations.append(
            Generation(
                generation=generation,
                best_fitness=best_fitness,
                mean_fitness=mean_fitness,
                best_individual=best_individual,
                diversity=diversity,
                constraint_violation=constraint_violation,
            )
        )


@dataclass
class EvolutionaryResult(ResultComponent):
    """Detailed result for evolutionary optimizers."""

    termination_reason: str
    n_generations: int
    final_pop_size: int
    final_diversity: Optional[float] = None
    convergence_generation: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "evolutionary",
            "termination_reason": self.termination_reason,
            "n_generations": self.n_generations,
            "final_pop_size": self.final_pop_size,
            "final_diversity": self.final_diversity,
            "convergence_generation": self.convergence_generation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionaryResult":
        return cls(
            termination_reason=data["termination_reason"],
            n_generations=data["n_generations"],
            final_pop_size=data["final_pop_size"],
            final_diversity=data.get("final_diversity"),
            convergence_generation=data.get("convergence_generation"),
        )
