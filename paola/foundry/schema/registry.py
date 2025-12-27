"""
Component registry for optimizer family schemas.

Maps optimizer names to families and handles polymorphic deserialization.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Type, Optional

from .components import InitializationComponent, ProgressComponent, ResultComponent
from .gradient import GradientInitialization, GradientProgress, GradientResult
from .bayesian import BayesianInitialization, BayesianProgress, BayesianResult
from .population import PopulationInitialization, PopulationProgress, PopulationResult
from .cmaes import CMAESInitialization, CMAESProgress, CMAESResult
from .evolutionary import EvolutionaryInitialization, EvolutionaryProgress, EvolutionaryResult
from .multiobjective import MultiObjectiveInitialization, MultiObjectiveProgress, MultiObjectiveResult


@dataclass
class OptimizerFamilySchema:
    """Schema definition for an optimizer family."""

    family: str
    initialization_class: Type[InitializationComponent]
    progress_class: Type[ProgressComponent]
    result_class: Type[ResultComponent]


class ComponentRegistry:
    """
    Registry for optimizer family schemas.

    Handles:
    - Mapping optimizer names to families
    - Serialization/deserialization of components
    - Validation of run data
    """

    def __init__(self):
        self._families: Dict[str, OptimizerFamilySchema] = {}
        self._optimizer_to_family: Dict[str, str] = {}

    def register_family(
        self,
        family: str,
        initialization_class: Type[InitializationComponent],
        progress_class: Type[ProgressComponent],
        result_class: Type[ResultComponent],
        optimizers: List[str],
    ):
        """Register an optimizer family."""
        self._families[family] = OptimizerFamilySchema(
            family=family,
            initialization_class=initialization_class,
            progress_class=progress_class,
            result_class=result_class,
        )
        for opt in optimizers:
            self._optimizer_to_family[opt] = family

    def get_family(self, optimizer: str) -> str:
        """
        Get family name for an optimizer.

        Handles formats like 'scipy:SLSQP' by extracting base name.
        Returns 'gradient' as default for unknown optimizers.
        """
        base = optimizer.split(":")[0].lower()
        return self._optimizer_to_family.get(base, "gradient")

    def get_schema(self, family: str) -> Optional[OptimizerFamilySchema]:
        """Get schema for a family."""
        return self._families.get(family)

    def list_families(self) -> List[str]:
        """List all registered families."""
        return list(self._families.keys())

    def list_optimizers(self, family: Optional[str] = None) -> List[str]:
        """List registered optimizers, optionally filtered by family."""
        if family is None:
            return list(self._optimizer_to_family.keys())
        return [
            opt for opt, fam in self._optimizer_to_family.items() if fam == family
        ]

    def create_initialization(
        self, family: str, **kwargs
    ) -> InitializationComponent:
        """Create initialization component for family."""
        schema = self._families.get(family)
        if schema is None:
            raise ValueError(f"Unknown family: {family}")
        return schema.initialization_class(**kwargs)

    def create_progress(self, family: str) -> ProgressComponent:
        """Create empty progress component for family."""
        schema = self._families.get(family)
        if schema is None:
            raise ValueError(f"Unknown family: {family}")
        return schema.progress_class()

    def create_result(self, family: str, **kwargs) -> ResultComponent:
        """Create result component for family."""
        schema = self._families.get(family)
        if schema is None:
            raise ValueError(f"Unknown family: {family}")
        return schema.result_class(**kwargs)

    def deserialize_components(
        self,
        family: str,
        init_data: Dict,
        progress_data: Dict,
        result_data: Dict,
    ) -> Tuple[InitializationComponent, ProgressComponent, ResultComponent]:
        """Deserialize components from dictionaries."""
        schema = self._families.get(family)
        if schema is None:
            raise ValueError(f"Unknown family: {family}")
        return (
            schema.initialization_class.from_dict(init_data),
            schema.progress_class.from_dict(progress_data),
            schema.result_class.from_dict(result_data),
        )


# Global registry instance
COMPONENT_REGISTRY = ComponentRegistry()

# Register built-in families
COMPONENT_REGISTRY.register_family(
    family="gradient",
    initialization_class=GradientInitialization,
    progress_class=GradientProgress,
    result_class=GradientResult,
    optimizers=["scipy", "ipopt", "nlopt", "snopt"],
)

COMPONENT_REGISTRY.register_family(
    family="bayesian",
    initialization_class=BayesianInitialization,
    progress_class=BayesianProgress,
    result_class=BayesianResult,
    optimizers=["optuna", "bo", "botorch"],
)

COMPONENT_REGISTRY.register_family(
    family="population",
    initialization_class=PopulationInitialization,
    progress_class=PopulationProgress,
    result_class=PopulationResult,
    optimizers=["de", "ga", "pso"],  # Legacy population-based
)

COMPONENT_REGISTRY.register_family(
    family="cmaes",
    initialization_class=CMAESInitialization,
    progress_class=CMAESProgress,
    result_class=CMAESResult,
    optimizers=["cmaes", "cma"],
)

# pymoo single-objective evolutionary algorithms
COMPONENT_REGISTRY.register_family(
    family="evolutionary",
    initialization_class=EvolutionaryInitialization,
    progress_class=EvolutionaryProgress,
    result_class=EvolutionaryResult,
    optimizers=["pymoo", "pymoo-ga", "pymoo-de", "pymoo-pso", "pymoo-es", "pymoo-brkga"],
)

# pymoo multi-objective algorithms
COMPONENT_REGISTRY.register_family(
    family="multiobjective",
    initialization_class=MultiObjectiveInitialization,
    progress_class=MultiObjectiveProgress,
    result_class=MultiObjectiveResult,
    optimizers=["nsga2", "nsga3", "moead", "agemoea", "smsemoa", "ctaea"],
)
