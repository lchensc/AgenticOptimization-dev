"""
Data Foundry - Foundation for PAOLA optimization data.

The foundry provides a single source of truth for optimization data,
managing problems, runs, results with versioning and lineage.

Inspired by Palantir Foundry's ontology layer: creates organized,
trustworthy data that the agent and knowledge base can build upon.

The foundry module is responsible for:
- Optimization run lifecycle (create → track → finalize)
- Data persistence and retrieval with lineage tracking
- Problem definition and management
- Query interfaces for analysis and knowledge extraction

This is the data foundation layer upon which intelligence (agent),
learning (knowledge base), and analysis are built.
"""

from .foundry import OptimizationFoundry
from .run import Run, RunRecord
from .problem import Problem
from .storage import StorageBackend, FileStorage
from .evaluator import FoundryEvaluator, InterjectionRequested, EvaluationError
from .capabilities import EvaluationObserver, EvaluationCache, PerformanceTracker
from .evaluator_schema import (
    EvaluatorConfig,
    EvaluatorSource,
    EvaluatorInterface,
    EvaluatorCapabilities,
    EvaluatorPerformance,
    create_python_function_config,
    create_cli_executable_config
)
from .evaluator_storage import EvaluatorStorage
from .nlp_schema import NLPProblem, InequalityConstraint, EqualityConstraint
from .nlp_evaluator import NLPEvaluator
from .problem_types import ProblemTypeDetector, SolverSelector

__all__ = [
    "OptimizationFoundry",
    "Run",
    "RunRecord",
    "Problem",
    "StorageBackend",
    "FileStorage",
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
