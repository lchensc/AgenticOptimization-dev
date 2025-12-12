"""
Singleton manager for active optimization runs.

Provides global access to active runs from any tool.
"""

from typing import Optional, Dict
import threading

from .active_run import OptimizationRun
from ..storage import StorageBackend


class RunManager:
    """
    Singleton manager for active optimization runs.

    Thread-safe registry of active runs that tools can access.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize manager (only once)."""
        if self._initialized:
            return

        self._active_runs: Dict[int, OptimizationRun] = {}
        self._storage: Optional[StorageBackend] = None
        self._initialized = True

    def set_storage(self, storage: StorageBackend) -> None:
        """
        Set storage backend for run persistence.

        Args:
            storage: Storage backend instance
        """
        self._storage = storage

    def get_storage(self) -> Optional[StorageBackend]:
        """Get current storage backend."""
        return self._storage

    def create_run(
        self,
        problem_id: str,
        problem_name: str,
        algorithm: str,
        description: str = ""
    ) -> OptimizationRun:
        """
        Create new active run.

        Args:
            problem_id: Problem identifier
            problem_name: Human-readable problem name
            algorithm: Optimization algorithm
            description: Optional description

        Returns:
            New OptimizationRun instance

        Raises:
            RuntimeError: If storage not set
        """
        if self._storage is None:
            raise RuntimeError("Storage backend not set. Call set_storage() first.")

        # Get next run ID from storage
        run_id = self._storage.get_next_run_id()

        # Create active run
        run = OptimizationRun(
            run_id=run_id,
            problem_id=problem_id,
            problem_name=problem_name,
            algorithm=algorithm,
            storage=self._storage,
            description=description
        )

        # Register as active
        self._active_runs[run_id] = run

        return run

    def get_run(self, run_id: int) -> Optional[OptimizationRun]:
        """
        Get active run by ID.

        Args:
            run_id: Run identifier

        Returns:
            OptimizationRun if active, None otherwise
        """
        return self._active_runs.get(run_id)

    def finalize_run(self, run_id: int) -> None:
        """
        Finalize run and remove from active registry.

        Args:
            run_id: Run identifier
        """
        if run_id in self._active_runs:
            del self._active_runs[run_id]

    def get_active_runs(self) -> Dict[int, OptimizationRun]:
        """Get all active runs."""
        return self._active_runs.copy()

    def clear_active_runs(self) -> None:
        """Clear all active runs (for testing)."""
        self._active_runs.clear()

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None
