"""
Data Foundry - Foundation for PAOLA optimization data.

The foundry provides a single source of truth for optimization data,
managing problems, graphs, nodes, results with versioning and lineage.

v0.3.x: Graph-based architecture
- Graph = complete optimization task (may involve multiple nodes)
- Node = single optimizer execution within a graph
- Edges = relationships between nodes (warm_start, branch, etc.)
- Polymorphic components per optimizer family

The foundry module is responsible for:
- Optimization graph lifecycle (create → track nodes → finalize)
- Data persistence and retrieval with lineage tracking
- Problem definition and management
- Query interfaces for analysis and knowledge extraction

This is the data foundation layer upon which intelligence (agent),
learning (knowledge base), and analysis are built.
"""

from .foundry import OptimizationFoundry
from .problem import (
    Problem,
    Variable,
    Objective,
    Constraint,
    OptimizationProblem,
    DerivationType,
)
from .storage import StorageBackend, FileStorage, ParetoStorage

# v0.3.0 Graph schema classes
from .schema import (
    # Graph-based schema (v0.3.0+)
    EdgeType,
    OptimizationEdge,
    OptimizationNode,
    GraphDecision,
    OptimizationGraph,
    # Component ABCs
    InitializationComponent,
    ProgressComponent,
    ResultComponent,
    # Gradient family
    GradientInitialization,
    GradientIteration,
    GradientProgress,
    GradientResult,
    # Bayesian family
    BayesianInitialization,
    Trial,
    BayesianProgress,
    BayesianResult,
    # Population family
    PopulationInitialization,
    Generation,
    PopulationProgress,
    PopulationResult,
    # CMA-ES family
    CMAESInitialization,
    CMAESGeneration,
    CMAESProgress,
    CMAESResult,
    # Evolutionary family (pymoo SOO)
    EvolutionaryInitialization,
    EvolutionaryProgress,
    EvolutionaryResult,
    # Multi-objective family (pymoo MOO)
    MultiObjectiveInitialization,
    MOGeneration,
    MultiObjectiveProgress,
    MultiObjectiveResult,
    # Pareto front (v1.0 MOO)
    ParetoSolution,
    ParetoFront,
    # Registry
    COMPONENT_REGISTRY,
)

# Active graph management (v0.3.0+)
from .active_graph import ActiveGraph, ActiveNode

# Cross-process coordination (v0.2.1+)
from .journal import GraphJournal

# Evaluator system
from .evaluator import FoundryEvaluator, InterjectionRequested, EvaluationError
from .capabilities import EvaluationObserver, EvaluationCache, PerformanceTracker
from .evaluator_schema import (
    EvaluatorConfig,
    EvaluatorSource,
    EvaluatorInterface,
    EvaluatorCapabilities,
    EvaluatorPerformance,
    create_python_function_config,
    create_cli_executable_config,
)
from .evaluator_storage import EvaluatorStorage

# NLP problem construction
from .nlp_schema import NLPProblem, InequalityConstraint, EqualityConstraint
from .nlp_evaluator import NLPEvaluator
from .problem_types import ProblemTypeDetector, SolverSelector

# MOO evaluator support
from .moo_evaluator import MOOEvaluator, ArrayEvaluatorCache, create_moo_evaluator

__all__ = [
    # Core
    "OptimizationFoundry",
    "StorageBackend",
    "FileStorage",
    "ParetoStorage",
    # Unified problem schema (v1.0)
    "Variable",
    "Objective",
    "Constraint",
    "OptimizationProblem",
    "DerivationType",
    # Legacy
    "Problem",
    # v0.3.x Graph-based architecture
    "EdgeType",
    "OptimizationEdge",
    "OptimizationNode",
    "GraphDecision",
    "OptimizationGraph",
    "ActiveGraph",
    "ActiveNode",
    # Cross-process coordination
    "GraphJournal",
    # Component ABCs
    "InitializationComponent",
    "ProgressComponent",
    "ResultComponent",
    # Gradient family
    "GradientInitialization",
    "GradientIteration",
    "GradientProgress",
    "GradientResult",
    # Bayesian family
    "BayesianInitialization",
    "Trial",
    "BayesianProgress",
    "BayesianResult",
    # Population family
    "PopulationInitialization",
    "Generation",
    "PopulationProgress",
    "PopulationResult",
    # CMA-ES family
    "CMAESInitialization",
    "CMAESGeneration",
    "CMAESProgress",
    "CMAESResult",
    # Evolutionary family (pymoo SOO)
    "EvolutionaryInitialization",
    "EvolutionaryProgress",
    "EvolutionaryResult",
    # Multi-objective family (pymoo MOO)
    "MultiObjectiveInitialization",
    "MOGeneration",
    "MultiObjectiveProgress",
    "MultiObjectiveResult",
    # Pareto front (v1.0 MOO)
    "ParetoSolution",
    "ParetoFront",
    # Registry
    "COMPONENT_REGISTRY",
    # Evaluator registration
    "FoundryEvaluator",
    "InterjectionRequested",
    "EvaluationError",
    # PAOLA capabilities
    "EvaluationObserver",
    "EvaluationCache",
    "PerformanceTracker",
    # Configuration schema
    "EvaluatorConfig",
    "EvaluatorSource",
    "EvaluatorInterface",
    "EvaluatorCapabilities",
    "EvaluatorPerformance",
    "create_python_function_config",
    "create_cli_executable_config",
    # Storage
    "EvaluatorStorage",
    # NLP problem construction
    "NLPProblem",
    "InequalityConstraint",
    "EqualityConstraint",
    "NLPEvaluator",
    "ProblemTypeDetector",
    "SolverSelector",
    # MOO evaluator
    "MOOEvaluator",
    "ArrayEvaluatorCache",
    "create_moo_evaluator",
]
