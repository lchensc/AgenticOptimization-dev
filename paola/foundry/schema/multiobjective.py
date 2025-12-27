"""
Multi-objective optimization family components.

Covers: NSGA-II, NSGA-III, MOEA/D, AGE-MOEA, SMS-EMOA from pymoo.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .components import InitializationComponent, ProgressComponent, ResultComponent


@dataclass
class MultiObjectiveInitialization(InitializationComponent):
    """Initialization for multi-objective algorithms."""

    specification: Dict[str, Any]
    pop_size: int
    n_objectives: int
    seed: Optional[int] = None
    reference_directions: Optional[List[List[float]]] = None  # For NSGA-III

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "multiobjective",
            "specification": self.specification,
            "pop_size": self.pop_size,
            "n_objectives": self.n_objectives,
            "seed": self.seed,
            "reference_directions": self.reference_directions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiObjectiveInitialization":
        return cls(
            specification=data["specification"],
            pop_size=data["pop_size"],
            n_objectives=data["n_objectives"],
            seed=data.get("seed"),
            reference_directions=data.get("reference_directions"),
        )


@dataclass
class MOGeneration:
    """Single generation record for multi-objective optimizer."""

    generation: int
    n_nondominated: int  # Size of Pareto front
    hypervolume: Optional[float] = None
    igd: Optional[float] = None  # Inverted Generational Distance
    spread: Optional[float] = None  # Diversity metric
    pareto_front_sample: Optional[List[List[float]]] = None  # Sample of front (for tracking)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "n_nondominated": self.n_nondominated,
            "hypervolume": self.hypervolume,
            "igd": self.igd,
            "spread": self.spread,
            "pareto_front_sample": self.pareto_front_sample,
        }


@dataclass
class MultiObjectiveProgress(ProgressComponent):
    """Progress data for multi-objective algorithms."""

    generations: List[MOGeneration] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "multiobjective",
            "generations": [g.to_dict() for g in self.generations],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiObjectiveProgress":
        generations = [
            MOGeneration(
                generation=g["generation"],
                n_nondominated=g["n_nondominated"],
                hypervolume=g.get("hypervolume"),
                igd=g.get("igd"),
                spread=g.get("spread"),
                pareto_front_sample=g.get("pareto_front_sample"),
            )
            for g in data["generations"]
        ]
        return cls(generations=generations)

    def add_generation(
        self,
        generation: int,
        n_nondominated: int,
        hypervolume: Optional[float] = None,
        igd: Optional[float] = None,
        spread: Optional[float] = None,
        pareto_front_sample: Optional[List[List[float]]] = None,
    ):
        """Add a generation record."""
        self.generations.append(
            MOGeneration(
                generation=generation,
                n_nondominated=n_nondominated,
                hypervolume=hypervolume,
                igd=igd,
                spread=spread,
                pareto_front_sample=pareto_front_sample,
            )
        )


@dataclass
class MultiObjectiveResult(ResultComponent):
    """Detailed result for multi-objective optimizers."""

    termination_reason: str
    n_generations: int
    n_pareto_solutions: int
    final_hypervolume: Optional[float] = None
    final_igd: Optional[float] = None
    final_spread: Optional[float] = None
    pareto_ref: Optional[str] = None  # Reference to full Pareto data in GraphDetail

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "multiobjective",
            "termination_reason": self.termination_reason,
            "n_generations": self.n_generations,
            "n_pareto_solutions": self.n_pareto_solutions,
            "final_hypervolume": self.final_hypervolume,
            "final_igd": self.final_igd,
            "final_spread": self.final_spread,
            "pareto_ref": self.pareto_ref,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiObjectiveResult":
        return cls(
            termination_reason=data["termination_reason"],
            n_generations=data["n_generations"],
            n_pareto_solutions=data["n_pareto_solutions"],
            final_hypervolume=data.get("final_hypervolume"),
            final_igd=data.get("final_igd"),
            final_spread=data.get("final_spread"),
            pareto_ref=data.get("pareto_ref"),
        )
