"""
Storage backends for knowledge base.

SKELETON IMPLEMENTATION:
- Interfaces defined
- MemoryKnowledgeStorage implemented (for testing)
- FileKnowledgeStorage skeleton only

FUTURE WORK:
- File-based persistence
- Vector store integration (Chroma, FAISS)
- Indexing for fast retrieval
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path


class KnowledgeStorage(ABC):
    """Abstract storage backend for knowledge."""

    @abstractmethod
    def store(self, insight_id: str, insight: Dict[str, Any]) -> None:
        """
        Store insight.

        Args:
            insight_id: Unique identifier
            insight: Insight data dictionary
        """
        pass

    @abstractmethod
    def retrieve(self, insight_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve insight by ID.

        Args:
            insight_id: Unique identifier

        Returns:
            Insight data or None if not found
        """
        pass

    @abstractmethod
    def list_all(self) -> List[Dict[str, Any]]:
        """
        List all insights.

        Returns:
            List of all stored insights
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all insights (for testing)."""
        pass


class MemoryKnowledgeStorage(KnowledgeStorage):
    """
    In-memory knowledge storage.

    Use for:
    - Development and testing
    - Temporary sessions
    - Prototyping

    NOT for:
    - Production use (no persistence)
    - Large knowledge bases (memory constrained)
    """

    def __init__(self):
        """Initialize empty memory storage."""
        self._insights: Dict[str, Dict[str, Any]] = {}

    def store(self, insight_id: str, insight: Dict[str, Any]) -> None:
        """Store insight in memory."""
        self._insights[insight_id] = insight

    def retrieve(self, insight_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve insight from memory."""
        return self._insights.get(insight_id)

    def list_all(self) -> List[Dict[str, Any]]:
        """List all insights in memory."""
        return list(self._insights.values())

    def clear(self) -> None:
        """Clear all insights from memory."""
        self._insights.clear()

    def count(self) -> int:
        """Get count of stored insights."""
        return len(self._insights)


class FileKnowledgeStorage(KnowledgeStorage):
    """
    File-based knowledge storage.

    SKELETON ONLY - NOT IMPLEMENTED

    Future design:
    - One JSON file per insight
    - Index file for fast lookup
    - Metadata for retrieval

    Directory structure:
        .paola/knowledge/
        ├── index.json          # Insight metadata
        ├── insights/
        │   ├── insight_001.json
        │   ├── insight_002.json
        │   └── ...
    """

    def __init__(self, base_dir: str = ".paola/knowledge"):
        """
        Initialize file storage.

        Args:
            base_dir: Base directory for knowledge files
        """
        self.base_dir = Path(base_dir)
        # TODO: Create directory structure
        # TODO: Load index

    def store(self, insight_id: str, insight: Dict[str, Any]) -> None:
        """Store insight to file (NOT IMPLEMENTED)."""
        raise NotImplementedError("FileKnowledgeStorage is skeleton only")

    def retrieve(self, insight_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve insight from file (NOT IMPLEMENTED)."""
        raise NotImplementedError("FileKnowledgeStorage is skeleton only")

    def list_all(self) -> List[Dict[str, Any]]:
        """List all insights (NOT IMPLEMENTED)."""
        raise NotImplementedError("FileKnowledgeStorage is skeleton only")

    def clear(self) -> None:
        """Clear all insights (NOT IMPLEMENTED)."""
        raise NotImplementedError("FileKnowledgeStorage is skeleton only")
