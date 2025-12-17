"""
Paola Foundry Schema - Polymorphic optimization data structures.

v0.3.x: Graph-based architecture
- OptimizationGraph: Complete optimization task (THE data model)
- OptimizationNode: Single optimizer execution
- OptimizationEdge: Typed relationship between nodes
- GraphDecision: Strategic decision record

Two-Tier Storage (v0.3.1):
- GraphRecord (Tier 1): LLM-ready, ~1KB, strategy-focused
- GraphDetail (Tier 2): Full trajectories, 10-100KB, for visualization

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

# Graph-based schema (v0.3.0+)
from .graph import (
    EdgeType,
    OptimizationEdge,
    OptimizationNode,
    GraphDecision,
    OptimizationGraph,
)

# Two-tier storage schema (v0.3.1+)
from .graph_record import (
    ProblemSignature,
    NodeSummary,
    EdgeSummary,
    GraphRecord,
)
from .graph_detail import (
    ConvergencePoint,
    XPoint,
    NodeDetail,
    GraphDetail,
)
from .conversion import (
    split_graph,
    create_problem_signature,
)

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
    # Graph-based schema (v0.3.x)
    "EdgeType",
    "OptimizationEdge",
    "OptimizationNode",
    "GraphDecision",
    "OptimizationGraph",
    # Two-tier storage (v0.3.1+)
    "ProblemSignature",
    "NodeSummary",
    "EdgeSummary",
    "GraphRecord",
    "ConvergencePoint",
    "XPoint",
    "NodeDetail",
    "GraphDetail",
    "split_graph",
    "create_problem_signature",
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
