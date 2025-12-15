"""Abstract storage interface for optimization data."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..schema import SessionRecord
from ..problem import Problem


class StorageBackend(ABC):
    """
    Abstract storage interface for optimization data.

    Implementations: FileStorage (JSON), SQLiteStorage (future)

    v0.2.0: Session-based storage (SessionRecord contains runs)
    """

    @abstractmethod
    def save_session(self, session: SessionRecord) -> None:
        """
        Persist optimization session.

        Args:
            session: SessionRecord to save
        """
        pass

    @abstractmethod
    def load_session(self, session_id: int) -> Optional[SessionRecord]:
        """
        Load optimization session by ID.

        Args:
            session_id: Session identifier

        Returns:
            SessionRecord or None if not found
        """
        pass

    @abstractmethod
    def load_all_sessions(self) -> List[SessionRecord]:
        """
        Load all optimization sessions.

        Returns:
            List of all SessionRecords, sorted by session_id
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
    def get_next_session_id(self) -> int:
        """
        Get next available session ID.

        This method must be atomic to prevent ID conflicts.

        Returns:
            Next session ID
        """
        pass
