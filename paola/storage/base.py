"""Abstract storage interface for optimization data."""

from abc import ABC, abstractmethod
from typing import List, Optional

from .models import OptimizationRun, Problem


class StorageBackend(ABC):
    """Abstract storage interface for optimization data."""

    @abstractmethod
    def save_run(self, run: OptimizationRun) -> None:
        """Persist optimization run."""
        pass

    @abstractmethod
    def load_run(self, run_id: int) -> Optional[OptimizationRun]:
        """Load optimization run by ID."""
        pass

    @abstractmethod
    def load_all_runs(self) -> List[OptimizationRun]:
        """Load all optimization runs."""
        pass

    @abstractmethod
    def save_problem(self, problem: Problem) -> None:
        """Persist problem metadata."""
        pass

    @abstractmethod
    def load_problem(self, problem_id: str) -> Optional[Problem]:
        """Load problem by ID."""
        pass

    @abstractmethod
    def get_next_run_id(self) -> int:
        """Get next available run ID."""
        pass
