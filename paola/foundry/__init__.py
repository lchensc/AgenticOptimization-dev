"""
Data Foundry - Foundation for PAOLA optimization data.

The foundry provides a single source of truth for optimization data,
managing problems, graphs, nodes, results with versioning and lineage.

v0.3.0: Graph-based architecture
- Graph = complete optimization task (may involve multiple nodes)
- Node = single optimizer execution within a graph
- Edges = relationships between nodes (warm_start, branch, etc.)

v0.2.0: Session-based architecture (legacy)
- Session = complete optimization task (may involve multiple runs)
- Run = single optimizer execution within a session
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
from .problem import Problem
from .storage import StorageBackend, FileStorage

# Deprecated: v0.1.0 Run classes (for backward compatibility with analysis modules)
# TODO: Remove in v0.3.0 after migrating analysis modules to SessionRecord
from .run import Run, RunRecord, IterationRecord

# v0.3.0 Graph schema classes
from .schema import (
    # Graph-based schema (v0.3.0+)
    EdgeType,
    OptimizationEdge,
    OptimizationNode,
    GraphDecision,
    OptimizationGraph,
    # Legacy (v0.2.0, for migration)
    SessionRecord,
    OptimizationRun,
    PaolaDecision,
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
    # Registry
    COMPONENT_REGISTRY,
)

# Active graph management (v0.3.0+)
from .active_graph import ActiveGraph, ActiveNode

# Active session management (v0.2.0 legacy)
from .active_session import ActiveSession, ActiveRun

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

__all__ = [
    # Core
    "OptimizationFoundry",
    "Problem",
    "StorageBackend",
    "FileStorage",
    # Deprecated v0.1.0 (for backward compatibility - TODO: remove in v0.3.0)
    "Run",
    "RunRecord",
    "IterationRecord",
    # v0.3.0 Graph-based architecture
    "EdgeType",
    "OptimizationEdge",
    "OptimizationNode",
    "GraphDecision",
    "OptimizationGraph",
    "ActiveGraph",
    "ActiveNode",
    # v0.2.0 Session/Run (legacy)
    "SessionRecord",
    "OptimizationRun",
    "PaolaDecision",
    "ActiveSession",
    "ActiveRun",
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
]
