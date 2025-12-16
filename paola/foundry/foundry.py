"""
OptimizationFoundry - Data foundation for optimization.

The foundry provides a single source of truth for optimization data,
managing problems, graphs, results with versioning and lineage.

v0.3.0: Graph-based architecture
- Graph = complete optimization task (may involve multiple nodes)
- Node = single optimizer execution

v0.2.0: Session-based architecture (legacy)
- Session = complete optimization task (may involve multiple runs)
- Run = single optimizer execution

Pattern: Dependency injection (testable, explicit)
"""

from typing import Dict, Optional, List
import re

from .storage import StorageBackend
from .schema import SessionRecord, OptimizationGraph
from .active_session import ActiveSession
from .active_graph import ActiveGraph
from .problem import Problem
from .evaluator_storage import EvaluatorStorage
from .evaluator_schema import EvaluatorConfig


class OptimizationFoundry:
    """
    Data foundation for optimization.

    The foundry provides a single source of truth for optimization data,
    managing problems, graphs, results with versioning and lineage.

    v0.3.0 Design:
    - Graph = complete optimization task (may involve multiple nodes)
    - Node = single optimizer execution within a graph
    - Uses dependency injection (pass storage backend)
    - No singleton (each instance is independent)
    - Manages active graphs (in-memory)
    - Delegates persistence to storage backend

    Example (v0.3.0 Graph API):
        # Initialize foundry
        storage = FileStorage(base_dir=".paola_runs")
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

    Legacy (v0.2.0 Session API still supported for migration)
    """

    def __init__(self, storage: StorageBackend):
        """
        Initialize foundry with storage backend.

        Args:
            storage: Storage backend (FileStorage, SQLiteStorage, etc.)
        """
        self.storage = storage
        self._active_graphs: Dict[int, ActiveGraph] = {}
        self._active_sessions: Dict[int, ActiveSession] = {}  # Legacy (v0.2.0)

        # Initialize evaluator storage
        self.evaluator_storage = EvaluatorStorage(storage)

    # =========================================================================
    # Graph Lifecycle Management (v0.3.0+)
    # =========================================================================

    def create_graph(
        self,
        problem_id: str,
        goal: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> ActiveGraph:
        """
        Create new optimization graph.

        This creates an active graph handle that can contain multiple
        optimizer nodes. The graph is persisted when finalized.

        Args:
            problem_id: Problem identifier
            goal: Natural language optimization goal
            config: Graph configuration

        Returns:
            ActiveGraph: Active graph handle

        Example:
            graph = foundry.create_graph(
                problem_id="rosenbrock_10d",
                goal="Minimize the Rosenbrock function"
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

    def finalize_graph(self, graph_id: int, success: bool) -> Optional[OptimizationGraph]:
        """
        Finalize graph and persist to storage.

        Args:
            graph_id: Graph identifier
            success: Whether optimization was successful

        Returns:
            OptimizationGraph if graph found, None otherwise
        """
        graph = self._active_graphs.get(graph_id)
        if graph is None:
            return None

        # Finalize graph to get immutable record
        record = graph.finalize(success)

        # Persist to storage
        self.storage.save_graph(record)

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
    # Graph Storage Queries (Completed Graphs)
    # =========================================================================

    def load_graph(self, graph_id: int) -> Optional[OptimizationGraph]:
        """
        Load completed graph from storage.

        Args:
            graph_id: Graph identifier

        Returns:
            OptimizationGraph or None if not found
        """
        return self.storage.load_graph(graph_id)

    def load_all_graphs(self) -> List[OptimizationGraph]:
        """
        Load all graphs from storage.

        Returns:
            List of all OptimizationGraphs, sorted by graph_id
        """
        return self.storage.load_all_graphs()

    def query_graphs(
        self,
        problem_id: Optional[str] = None,
        success: Optional[bool] = None,
        limit: int = 100,
    ) -> List[OptimizationGraph]:
        """
        Query graphs with filters.

        Currently does post-load filtering (loads all, then filters).
        Future: Push filtering to storage backend for efficiency.

        Args:
            problem_id: Filter by problem ID (supports wildcards)
            success: Filter by success status
            limit: Maximum number of results

        Returns:
            List of matching OptimizationGraphs

        Example:
            # Get all successful graphs on Rosenbrock
            graphs = foundry.query_graphs(
                problem_id="rosenbrock*",
                success=True,
                limit=10
            )
        """
        graphs = self.storage.load_all_graphs()

        # Apply filters
        filtered = []
        for graph in graphs:
            # Problem ID filter (support wildcards)
            if problem_id is not None:
                if '*' in problem_id:
                    # Wildcard matching
                    pattern = problem_id.replace('*', '.*')
                    if not re.match(pattern, graph.problem_id):
                        continue
                else:
                    if graph.problem_id != problem_id:
                        continue

            # Success filter
            if success is not None and graph.success != success:
                continue

            filtered.append(graph)

            # Limit
            if len(filtered) >= limit:
                break

        return filtered

    # =========================================================================
    # Session Lifecycle Management (v0.2.0 Legacy)
    # =========================================================================

    def create_session(
        self,
        problem_id: str,
        config: Optional[Dict] = None,
    ) -> ActiveSession:
        """
        Create new optimization session.

        This creates an active session handle that can contain multiple
        optimizer runs. The session is persisted when finalized.

        Args:
            problem_id: Problem identifier
            config: Session configuration (goal, constraints, etc.)

        Returns:
            ActiveSession: Active session handle

        Example:
            session = foundry.create_session(
                problem_id="rosenbrock_10d",
                config={"goal": "minimize", "max_evaluations": 1000}
            )
        """
        # Get next session ID from storage
        session_id = self.storage.get_next_session_id()

        # Create active session
        session = ActiveSession(
            session_id=session_id,
            problem_id=problem_id,
            config=config,
        )

        # Register as active
        self._active_sessions[session_id] = session

        return session

    def get_session(self, session_id: int) -> Optional[ActiveSession]:
        """
        Get active session by ID.

        Only returns sessions that are currently active (in-progress).
        For completed sessions, use load_session().

        Args:
            session_id: Session identifier

        Returns:
            ActiveSession if active, None otherwise
        """
        return self._active_sessions.get(session_id)

    def finalize_session(self, session_id: int, success: bool) -> Optional[SessionRecord]:
        """
        Finalize session and persist to storage.

        Args:
            session_id: Session identifier
            success: Whether optimization was successful

        Returns:
            SessionRecord if session found, None otherwise
        """
        session = self._active_sessions.get(session_id)
        if session is None:
            return None

        # Finalize session to get immutable record
        record = session.finalize(success)

        # Persist to storage
        self.storage.save_session(record)

        # Remove from active registry
        del self._active_sessions[session_id]

        return record

    def get_active_sessions(self) -> Dict[int, ActiveSession]:
        """
        Get all active (in-progress) sessions.

        Returns:
            Dict mapping session_id to ActiveSession
        """
        return self._active_sessions.copy()

    # ===== Storage Queries (Completed Sessions) =====

    def load_session(self, session_id: int) -> Optional[SessionRecord]:
        """
        Load completed session from storage.

        Args:
            session_id: Session identifier

        Returns:
            SessionRecord or None if not found
        """
        return self.storage.load_session(session_id)

    def load_all_sessions(self) -> List[SessionRecord]:
        """
        Load all sessions from storage.

        Returns:
            List of all SessionRecords, sorted by session_id
        """
        return self.storage.load_all_sessions()

    def query_sessions(
        self,
        problem_id: Optional[str] = None,
        success: Optional[bool] = None,
        limit: int = 100,
    ) -> List[SessionRecord]:
        """
        Query sessions with filters.

        Currently does post-load filtering (loads all, then filters).
        Future: Push filtering to storage backend for efficiency.

        Args:
            problem_id: Filter by problem ID (supports wildcards)
            success: Filter by success status
            limit: Maximum number of results

        Returns:
            List of matching SessionRecords

        Example:
            # Get all successful sessions on Rosenbrock
            sessions = foundry.query_sessions(
                problem_id="rosenbrock*",
                success=True,
                limit=10
            )
        """
        sessions = self.storage.load_all_sessions()

        # Apply filters
        filtered = []
        for session in sessions:
            # Problem ID filter (support wildcards)
            if problem_id is not None:
                if '*' in problem_id:
                    # Wildcard matching
                    pattern = problem_id.replace('*', '.*')
                    if not re.match(pattern, session.problem_id):
                        continue
                else:
                    if session.problem_id != problem_id:
                        continue

            # Success filter
            if success is not None and session.success != success:
                continue

            filtered.append(session)

            # Limit
            if len(filtered) >= limit:
                break

        return filtered

    # ===== Problem Management =====

    def register_problem(self, problem: Problem) -> None:
        """
        Register problem definition.

        Args:
            problem: Problem to register
        """
        self.storage.save_problem(problem)

    def get_problem(self, problem_id: str) -> Optional[Problem]:
        """
        Get problem definition.

        Args:
            problem_id: Problem identifier

        Returns:
            Problem or None if not found
        """
        return self.storage.load_problem(problem_id)

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

    def clear_active_sessions(self) -> None:
        """
        Clear all active sessions (for testing).

        Warning: This removes sessions from registry without finalizing them.
        Only use in testing scenarios.
        """
        self._active_sessions.clear()

    def clear_all_active(self) -> None:
        """
        Clear all active graphs and sessions (for testing).
        """
        self._active_graphs.clear()
        self._active_sessions.clear()

    # =========================================================================
    # Deprecated v0.1.0 API (for backward compatibility)
    # =========================================================================

    def load_run(self, run_id: int):
        """
        DEPRECATED: Load run by ID.

        This method is deprecated. Use load_session() instead.
        In v0.2.0, runs are contained within sessions.

        Args:
            run_id: Run identifier

        Returns:
            None (deprecated - no backward compatible storage)
        """
        import warnings
        warnings.warn(
            "load_run() is deprecated. Use load_session() instead. "
            "In v0.2.0, runs are contained within sessions.",
            DeprecationWarning,
            stacklevel=2
        )
        return None

    def get_run(self, run_id: int):
        """
        DEPRECATED: Get active run by ID.

        This method is deprecated. Use get_session() instead.
        In v0.2.0, runs are managed through sessions.

        Args:
            run_id: Run identifier

        Returns:
            None (deprecated)
        """
        import warnings
        warnings.warn(
            "get_run() is deprecated. Use get_session() instead. "
            "In v0.2.0, runs are managed through sessions.",
            DeprecationWarning,
            stacklevel=2
        )
        return None

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OptimizationFoundry("
            f"active_graphs={len(self._active_graphs)}, "
            f"active_sessions={len(self._active_sessions)}, "
            f"storage={type(self.storage).__name__})"
        )
