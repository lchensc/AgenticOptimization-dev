"""Abstract storage interface for optimization data."""

from abc import ABC, abstractmethod
from typing import List, Optional

# Import from platform.run, not storage.models
from ..run import RunRecord
from ..problem import Problem


class StorageBackend(ABC):
    """
    Abstract storage interface for optimization data.

    Implementations: FileStorage (JSON), SQLiteStorage (future)
    """

    @abstractmethod
    def save_run(self, run: RunRecord) -> None:
        """
        Persist optimization run.

        Args:
            run: RunRecord to save
        """
        pass

    @abstractmethod
    def load_run(self, run_id: int) -> Optional[RunRecord]:
        """
        Load optimization run by ID.

        Args:
            run_id: Run identifier

        Returns:
            RunRecord or None if not found
        """
        pass

    @abstractmethod
    def load_all_runs(self) -> List[RunRecord]:
        """
        Load all optimization runs.

        Returns:
            List of all RunRecords, sorted by run_id
        """
        pass

    @abstractmethod
    def save_problem(self, problem: Problem) -> None:
        """
        Persist problem metadata.

        Args:
            problem: Problem to save
        """
        pass

    @abstractmethod
    def load_problem(self, problem_id: str) -> Optional[Problem]:
        """
        Load problem by ID.

        Args:
            problem_id: Problem identifier

        Returns:
            Problem or None if not found
        """
        pass

    @abstractmethod
    def get_next_run_id(self) -> int:
        """
        Get next available run ID.

        This method must be atomic to prevent ID conflicts.

        Returns:
            Next run ID
        """
        pass
