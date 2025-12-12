"""
OptimizationPlatform - Central API for run management.

Replaces: runs.manager.RunManager singleton
Pattern: Dependency injection (testable, explicit)
"""

from typing import Dict, Optional, List
from datetime import datetime

from .storage import StorageBackend
from .run import Run, RunRecord
from .problem import Problem


class OptimizationPlatform:
    """
    Central platform for optimization run management.

    This class is the foundation of PAOLA's data platform.
    It manages the lifecycle of optimization runs and provides
    query interfaces for analysis and knowledge extraction.

    Design:
    - Uses dependency injection (pass storage backend)
    - No singleton (each instance is independent)
    - Manages active runs (in-memory)
    - Delegates persistence to storage backend

    Example:
        # Initialize platform
        storage = FileStorage(base_dir=".paola_runs")
        platform = OptimizationPlatform(storage=storage)

        # Create and track run
        run = platform.create_run(
            problem_id="rosenbrock_10d",
            problem_name="Rosenbrock 10D",
            algorithm="SLSQP"
        )

        # Optimization loop records iterations
        run.record_iteration(design=x, objective=f)

        # Finalize
        run.finalize(result)

        # Query
        all_runs = platform.load_all_runs()
    """

    def __init__(self, storage: StorageBackend):
        """
        Initialize platform with storage backend.

        Args:
            storage: Storage backend (FileStorage, SQLiteStorage, etc.)
        """
        self.storage = storage
        self._active_runs: Dict[int, Run] = {}

    # ===== Run Lifecycle Management =====

    def create_run(
        self,
        problem_id: str,
        problem_name: str,
        algorithm: str,
        description: str = ""
    ) -> Run:
        """
        Create new optimization run.

        This creates an active run handle that tools use to track
        optimization progress. The run auto-persists to storage
        on every iteration.

        Args:
            problem_id: Problem identifier
            problem_name: Human-readable problem name
            algorithm: Optimization algorithm
            description: Optional description

        Returns:
            Run: Active run handle

        Example:
            run = platform.create_run(
                problem_id="rosenbrock_10d",
                problem_name="Rosenbrock 10D",
                algorithm="SLSQP",
                description="First attempt with default settings"
            )
        """
        # Get next run ID from storage
        run_id = self.storage.get_next_run_id()

        # Create active run
        run = Run(
            run_id=run_id,
            problem_id=problem_id,
            problem_name=problem_name,
            algorithm=algorithm,
            storage=self.storage,
            description=description
        )

        # Register as active
        self._active_runs[run_id] = run

        return run

    def get_run(self, run_id: int) -> Optional[Run]:
        """
        Get active run by ID.

        Only returns runs that are currently active (in-progress).
        For completed runs, use load_run().

        Args:
            run_id: Run identifier

        Returns:
            Run if active, None otherwise
        """
        return self._active_runs.get(run_id)

    def finalize_run(self, run_id: int) -> None:
        """
        Finalize run and remove from active registry.

        Called after run.finalize() to remove from active runs.

        Args:
            run_id: Run identifier
        """
        if run_id in self._active_runs:
            del self._active_runs[run_id]

    def get_active_runs(self) -> Dict[int, Run]:
        """
        Get all active (in-progress) runs.

        Returns:
            Dict mapping run_id to Run
        """
        return self._active_runs.copy()

    # ===== Storage Queries (Completed Runs) =====

    def load_run(self, run_id: int) -> Optional[RunRecord]:
        """
        Load completed run from storage.

        Args:
            run_id: Run identifier

        Returns:
            RunRecord or None if not found
        """
        return self.storage.load_run(run_id)

    def load_all_runs(self) -> List[RunRecord]:
        """
        Load all runs from storage.

        Returns:
            List of all RunRecords, sorted by run_id
        """
        return self.storage.load_all_runs()

    def query_runs(
        self,
        algorithm: Optional[str] = None,
        problem_id: Optional[str] = None,
        success: Optional[bool] = None,
        limit: int = 100
    ) -> List[RunRecord]:
        """
        Query runs with filters.

        Currently does post-load filtering (loads all, then filters).
        Future: Push filtering to storage backend for efficiency.

        Args:
            algorithm: Filter by algorithm name
            problem_id: Filter by problem ID (supports wildcards)
            success: Filter by success status
            limit: Maximum number of results

        Returns:
            List of matching RunRecords

        Example:
            # Get all successful SLSQP runs on Rosenbrock
            runs = platform.query_runs(
                algorithm="SLSQP",
                problem_id="rosenbrock*",
                success=True,
                limit=10
            )
        """
        runs = self.storage.load_all_runs()

        # Apply filters
        filtered = []
        for run in runs:
            # Algorithm filter
            if algorithm is not None and run.algorithm != algorithm:
                continue

            # Problem ID filter (support wildcards)
            if problem_id is not None:
                if '*' in problem_id:
                    # Wildcard matching
                    pattern = problem_id.replace('*', '.*')
                    import re
                    if not re.match(pattern, run.problem_id):
                        continue
                else:
                    if run.problem_id != problem_id:
                        continue

            # Success filter
            if success is not None and run.success != success:
                continue

            filtered.append(run)

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

    # ===== Utilities =====

    def clear_active_runs(self) -> None:
        """
        Clear all active runs (for testing).

        Warning: This removes runs from registry without finalizing them.
        Only use in testing scenarios.
        """
        self._active_runs.clear()

    def __repr__(self) -> str:
        """String representation."""
        return f"OptimizationPlatform(active_runs={len(self._active_runs)}, storage={type(self.storage).__name__})"
