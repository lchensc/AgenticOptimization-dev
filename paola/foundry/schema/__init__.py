"""
Paola Foundry Schema - Polymorphic optimization data structures.

Session/Run Hierarchy:
- SessionRecord: Complete optimization task (may involve multiple runs)
- OptimizationRun: Single optimizer execution
- PaolaDecision: Strategic decision record

Component ABCs:
- InitializationComponent: How the optimizer was initialized
- ProgressComponent: Iteration/trial/generation data
- ResultComponent: Final result details

Family-Specific Components:
- Gradient: SciPy, IPOPT, NLopt gradient methods
- Bayesian: Optuna, BO-GP
- Population: DE, GA, NSGA-II, PSO
- CMA-ES: CMA-ES optimizer

Registry:
- COMPONENT_REGISTRY: Maps optimizers to families, handles deserialization
"""

# Base classes
from .base import SessionRecord, OptimizationRun, PaolaDecision

# Component ABCs
from .components import (
    InitializationComponent,
    ProgressComponent,
    ResultComponent,
)

# Gradient family
from .gradient import (
    GradientInitialization,
    GradientIteration,
    GradientProgress,
    GradientResult,
)

# Bayesian family
from .bayesian import (
    BayesianInitialization,
    Trial,
    BayesianProgress,
    BayesianResult,
)

# Population family
from .population import (
    PopulationInitialization,
    Generation,
    PopulationProgress,
    PopulationResult,
)

# CMA-ES family
from .cmaes import (
    CMAESInitialization,
    CMAESGeneration,
    CMAESProgress,
    CMAESResult,
)

# Registry
from .registry import (
    OptimizerFamilySchema,
    ComponentRegistry,
    COMPONENT_REGISTRY,
)

__all__ = [
    # Base
    "SessionRecord",
    "OptimizationRun",
    "PaolaDecision",
    # ABCs
    "InitializationComponent",
    "ProgressComponent",
    "ResultComponent",
    # Gradient
    "GradientInitialization",
    "GradientIteration",
    "GradientProgress",
    "GradientResult",
    # Bayesian
    "BayesianInitialization",
    "Trial",
    "BayesianProgress",
    "BayesianResult",
    # Population
    "PopulationInitialization",
    "Generation",
    "PopulationProgress",
    "PopulationResult",
    # CMA-ES
    "CMAESInitialization",
    "CMAESGeneration",
    "CMAESProgress",
    "CMAESResult",
    # Registry
    "OptimizerFamilySchema",
    "ComponentRegistry",
    "COMPONENT_REGISTRY",
]
