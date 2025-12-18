"""
OptimizationFoundry - Data foundation for optimization.

The foundry provides a single source of truth for optimization data,
managing problems, graphs, results with versioning and lineage.

v0.4.7: Type consistency fix
- problem_id is now int throughout (was str in graph schemas)
- query_graphs() uses exact int match (no pattern matching)
- Backward compatible: from_dict() coerces legacy string values

v0.4.6: Single source of truth for problems
- Added _problem_cache for NLPEvaluator instances
- get_problem_evaluator() with cache-through loading
- Replaces distributed _PROBLEM_REGISTRY pattern

v0.3.1: Two-tier graph storage
- GraphRecord (Tier 1): LLM-ready, ~1KB, strategy-focused
- GraphDetail (Tier 2): Full trajectories, for visualization

v0.3.0: Graph-based architecture
- Graph = complete optimization task (may involve multiple nodes)
- Node = single optimizer execution

Pattern: Dependency injection (testable, explicit)
"""

from typing import Dict, Optional, List, Any, Union
import logging

logger = logging.getLogger(__name__)

from .storage import StorageBackend
from .schema import OptimizationGraph
from .schema import GraphRecord, GraphDetail, ProblemSignature
from .schema.conversion import create_problem_signature
from .active_graph import ActiveGraph
from .problem import Problem
from .evaluator_storage import EvaluatorStorage
from .evaluator_schema import EvaluatorConfig


class OptimizationFoundry:
    """
    Data foundation for optimization.

    The foundry provides a single source of truth for optimization data,
    managing problems, graphs, results with versioning and lineage.

    Design:
    - Graph = complete optimization task (may involve multiple nodes)
    - Node = single optimizer execution within a graph
    - Uses dependency injection (pass storage backend)
    - No singleton (each instance is independent)
    - Manages active graphs (in-memory)
    - Delegates persistence to storage backend

    Example:
        # Initialize foundry
        storage = FileStorage()  # uses .paola_foundry by default
        foundry = OptimizationFoundry(storage=storage)

        # Create graph
        graph = foundry.create_graph(
            problem_id="rosenbrock_10d",
            goal="Minimize Rosenbrock function"
        )

        # Start a node within graph
        node = graph.start_node(
            optimizer="scipy:SLSQP",
            config={},
            initialization=GradientInitialization(...)
        )

        # Record iterations
        node.record_iteration({"iteration": 1, "objective": 0.5, ...})

        # Complete node
        graph.complete_node(progress, result, best_obj, best_x)

        # Finalize graph
        record = foundry.finalize_graph(graph.graph_id, success=True)
    """

    def __init__(self, storage: StorageBackend):
        """
        Initialize foundry with storage backend.

        Args:
            storage: Storage backend (FileStorage, SQLiteStorage, etc.)
        """
        self.storage = storage
        self._active_graphs: Dict[int, ActiveGraph] = {}

        # Problem evaluator cache (v0.4.6 - single source of truth)
        # Maps problem_id (int) -> NLPEvaluator instance
        self._problem_cache: Dict[int, Any] = {}

        # Initialize evaluator storage
        self.evaluator_storage = EvaluatorStorage(storage)

    # =========================================================================
    # Graph Lifecycle Management (v0.3.0+)
    # =========================================================================

    def create_graph(
        self,
        problem_id: int,  # v0.4.7: Changed from str to int for type consistency
        goal: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> ActiveGraph:
        """
        Create new optimization graph.

        This creates an active graph handle that can contain multiple
        optimizer nodes. The graph is persisted when finalized.

        Args:
            problem_id: Numeric problem ID (from create_nlp_problem)
            goal: Natural language optimization goal
            config: Graph configuration

        Returns:
            ActiveGraph: Active graph handle

        Example:
            graph = foundry.create_graph(
                problem_id=7,
                goal="Minimize the Ackley function"
            )
        """
        # Get next graph ID from storage
        graph_id = self.storage.get_next_graph_id()

        # Create active graph
        graph = ActiveGraph(
            graph_id=graph_id,
            problem_id=problem_id,
            goal=goal,
            config=config,
        )

        # Register as active
        self._active_graphs[graph_id] = graph

        return graph

    def get_graph(self, graph_id: int) -> Optional[ActiveGraph]:
        """
        Get active graph by ID.

        Only returns graphs that are currently active (in-progress).
        For completed graphs, use load_graph().

        Args:
            graph_id: Graph identifier

        Returns:
            ActiveGraph if active, None otherwise
        """
        return self._active_graphs.get(graph_id)

    def finalize_graph(self, graph_id: int) -> Optional[OptimizationGraph]:
        """
        Finalize graph and persist to storage (two-tier).

        Saves both:
        - Tier 1: GraphRecord (~1KB) for LLM learning
        - Tier 2: GraphDetail (10-100KB) for visualization

        v0.4.8: Removed success parameter. All finalized graphs are "completed".
        Quality judgment is not encoded in schema - agent reasons from final_objective.

        Args:
            graph_id: Graph identifier

        Returns:
            OptimizationGraph if graph found, None otherwise
        """
        graph = self._active_graphs.get(graph_id)
        if graph is None:
            return None

        # Finalize graph to get immutable record
        record = graph.finalize()

        # Try to create problem signature from registered problem
        problem_signature = None
        problem = self.storage.load_problem(record.problem_id)
        if problem:
            # Handle NLPProblem attributes (n_variables, n_constraints, bounds)
            constraint_types = []
            if hasattr(problem, 'inequality_constraints') and problem.inequality_constraints:
                constraint_types.extend(["inequality"] * len(problem.inequality_constraints))
            if hasattr(problem, 'equality_constraints') and problem.equality_constraints:
                constraint_types.extend(["equality"] * len(problem.equality_constraints))

            problem_signature = create_problem_signature(
                n_dimensions=problem.n_variables,
                bounds=getattr(problem, 'bounds', None),
                n_constraints=problem.n_constraints,
                constraint_types=constraint_types if constraint_types else None,
                domain_hint=getattr(problem, 'domain_hint', None),
            )

        # Persist to storage (two-tier)
        self.storage.save_graph(record, problem_signature)

        # Remove from active registry
        del self._active_graphs[graph_id]

        return record

    def get_active_graphs(self) -> Dict[int, ActiveGraph]:
        """
        Get all active (in-progress) graphs.

        Returns:
            Dict mapping graph_id to ActiveGraph
        """
        return self._active_graphs.copy()

    # =========================================================================
    # Graph Storage Queries (Completed Graphs - Two-Tier)
    # =========================================================================

    def load_graph(self, graph_id: int) -> Optional[OptimizationGraph]:
        """
        Load completed graph from storage (legacy format).

        For backward compatibility. New code should use load_graph_record().

        Args:
            graph_id: Graph identifier

        Returns:
            OptimizationGraph or None if not found
        """
        return self.storage.load_graph(graph_id)

    def load_graph_record(self, graph_id: int) -> Optional[GraphRecord]:
        """
        Load Tier 1 GraphRecord (for LLM queries).

        This is the compact representation optimized for cross-graph learning.

        Args:
            graph_id: Graph identifier

        Returns:
            GraphRecord or None if not found
        """
        return self.storage.load_graph_record(graph_id)

    def load_graph_detail(self, graph_id: int) -> Optional[GraphDetail]:
        """
        Load Tier 2 GraphDetail (for visualization/deep analysis).

        Args:
            graph_id: Graph identifier

        Returns:
            GraphDetail or None if not found
        """
        return self.storage.load_graph_detail(graph_id)

    def load_all_graphs(self) -> List[OptimizationGraph]:
        """
        Load all graphs from storage (legacy format).

        DEPRECATED: Use load_all_graph_records() for better performance.

        Returns:
            List of all OptimizationGraphs, sorted by graph_id
        """
        return self.storage.load_all_graphs()

    def load_all_graph_records(self) -> List[GraphRecord]:
        """
        Load all GraphRecords (Tier 1) from storage.

        This is the preferred method for listing/querying graphs.

        Returns:
            List of all GraphRecords, sorted by graph_id
        """
        return self.storage.load_all_graph_records()

    def query_graphs(
        self,
        problem_id: Optional[int] = None,
        n_dimensions: Optional[int] = None,
        limit: int = 100,
    ) -> List[GraphRecord]:
        """
        Query graphs with filters (returns Tier 1 GraphRecords).

        This queries the compact GraphRecord format optimized for LLM learning.

        v0.4.8: Removed success filter. Quality judgment is not in schema.
        Agent should reason from final_objective values directly.

        Args:
            problem_id: Filter by exact problem ID (int)
            n_dimensions: Filter by problem dimensions
            limit: Maximum number of results

        Returns:
            List of matching GraphRecords

        Example:
            # Get all graphs for problem 7
            records = foundry.query_graphs(problem_id=7, limit=10)

            # Get graphs for similar-sized problems
            records = foundry.query_graphs(n_dimensions=50, limit=5)
        """
        records = self.storage.load_all_graph_records()

        # Apply filters
        filtered = []
        for record in records:
            # Problem ID filter (exact match)
            if problem_id is not None:
                if record.problem_id != problem_id:
                    continue

            # Dimensions filter
            if n_dimensions is not None:
                if record.problem_signature is None:
                    continue
                if record.problem_signature.n_dimensions != n_dimensions:
                    continue

            filtered.append(record)

            # Limit
            if len(filtered) >= limit:
                break

        return filtered

    # =========================================================================
    # Problem Management (v0.4.6 - Single Source of Truth)
    # =========================================================================

    def register_problem(self, problem: Problem) -> None:
        """
        Register problem definition (storage only).

        For most cases, use register_problem_evaluator() which also caches
        the evaluator for runtime use.

        Args:
            problem: Problem to register
        """
        self.storage.save_problem(problem)

    def register_problem_evaluator(
        self,
        problem: Problem,
        evaluator: Any,
    ) -> None:
        """
        Register problem definition AND cache its evaluator atomically.

        This is the preferred method for registering new problems. It ensures
        both storage and cache are updated together.

        Args:
            problem: Problem definition to persist
            evaluator: NLPEvaluator instance for runtime use

        Example:
            nlp_evaluator = NLPEvaluator.from_problem(problem, foundry)
            foundry.register_problem_evaluator(problem, nlp_evaluator)
        """
        # Persist to storage
        self.storage.save_problem(problem)

        # Cache evaluator for runtime access
        self._problem_cache[problem.problem_id] = evaluator

        logger.debug(f"Registered problem {problem.problem_id} with evaluator")

    def get_problem(self, problem_id: Union[str, int]) -> Optional[Problem]:
        """
        Get problem definition from storage.

        Args:
            problem_id: Problem identifier

        Returns:
            Problem or None if not found
        """
        # Normalize to int
        if isinstance(problem_id, str):
            try:
                problem_id = int(problem_id)
            except ValueError:
                pass  # Keep as string for legacy compatibility
        return self.storage.load_problem(problem_id)

    def get_problem_evaluator(self, problem_id: Union[str, int]) -> Optional[Any]:
        """
        Get problem evaluator with cache-through loading.

        This is the single source of truth for problem evaluators.
        If not in cache, loads from storage and creates evaluator.

        Args:
            problem_id: Problem ID (int or str, normalized to int)

        Returns:
            NLPEvaluator or None if problem not found

        Example:
            evaluator = foundry.get_problem_evaluator(7)
            if evaluator:
                result = evaluator.evaluate(x)
        """
        # Normalize to int
        if isinstance(problem_id, str):
            try:
                problem_id = int(problem_id)
            except ValueError:
                return None  # Invalid ID

        # Check cache first
        if problem_id in self._problem_cache:
            return self._problem_cache[problem_id]

        # Cache miss - load from storage and create evaluator
        problem = self.storage.load_problem(problem_id)
        if problem is None:
            return None

        # Create evaluator (lazy import to avoid circular dependency)
        try:
            from .nlp_evaluator import NLPEvaluator
            evaluator = NLPEvaluator.from_problem(problem, self)
            self._problem_cache[problem_id] = evaluator
            logger.debug(f"Loaded problem {problem_id} from storage into cache")
            return evaluator
        except Exception as e:
            logger.warning(f"Failed to create evaluator for problem {problem_id}: {e}")
            return None

    def clear_problem_cache(self) -> None:
        """
        Clear problem evaluator cache.

        Use for testing or when storage has been modified externally.
        """
        self._problem_cache.clear()
        logger.debug("Cleared problem cache")

    def get_cached_problem_ids(self) -> List[int]:
        """
        Get list of problem IDs currently in cache.

        Returns:
            List of cached problem IDs
        """
        return list(self._problem_cache.keys())

    # ===== Evaluator Management =====

    def register_evaluator(self, config: EvaluatorConfig) -> str:
        """
        Register evaluator in Foundry.

        Args:
            config: EvaluatorConfig to register

        Returns:
            evaluator_id
        """
        return self.evaluator_storage.store_evaluator(config)

    def get_evaluator_config(self, evaluator_id: str) -> Dict:
        """
        Get evaluator configuration.

        Args:
            evaluator_id: Evaluator ID

        Returns:
            Configuration dict (for FoundryEvaluator)
        """
        config = self.evaluator_storage.retrieve_evaluator(evaluator_id)
        return config.dict()

    def list_evaluators(
        self,
        evaluator_type: Optional[str] = None,
        status: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[EvaluatorConfig]:
        """
        List registered evaluators with optional filters.

        Args:
            evaluator_type: Filter by type (python_function, cli_executable)
            status: Filter by status (registered, validated, active)
            domain: Filter by domain

        Returns:
            List of EvaluatorConfig
        """
        return self.evaluator_storage.list_evaluators(
            evaluator_type=evaluator_type,
            status=status,
            domain=domain,
        )

    def update_evaluator_performance(
        self,
        evaluator_id: str,
        execution_time: float,
        success: bool,
    ):
        """
        Update evaluator performance metrics.

        Called by FoundryEvaluator after each evaluation.

        Args:
            evaluator_id: Evaluator ID
            execution_time: Time taken (seconds)
            success: Whether evaluation succeeded
        """
        self.evaluator_storage.update_performance(
            evaluator_id=evaluator_id,
            execution_time=execution_time,
            success=success,
        )

    def link_evaluator_to_session(self, evaluator_id: str, session_id: int):
        """
        Link evaluator to optimization session.

        Args:
            evaluator_id: Evaluator ID
            session_id: Session ID
        """
        self.evaluator_storage.add_run_reference(evaluator_id, str(session_id))

    def link_evaluator_to_problem(self, evaluator_id: str, problem_id: str):
        """
        Link evaluator to problem.

        Args:
            evaluator_id: Evaluator ID
            problem_id: Problem ID
        """
        self.evaluator_storage.add_problem_reference(evaluator_id, problem_id)

    def get_evaluator_statistics(self) -> Dict:
        """
        Get evaluator storage statistics.

        Returns:
            Dict with statistics
        """
        return self.evaluator_storage.get_statistics()

    # =========================================================================
    # Utilities
    # =========================================================================

    def clear_active_graphs(self) -> None:
        """
        Clear all active graphs (for testing).

        Warning: This removes graphs from registry without finalizing them.
        Only use in testing scenarios.
        """
        self._active_graphs.clear()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OptimizationFoundry("
            f"active_graphs={len(self._active_graphs)}, "
            f"storage={type(self.storage).__name__})"
        )
