"""File-based storage implementation using JSON files."""

import json
from pathlib import Path
from typing import List, Optional

from .backend import StorageBackend
from ..schema import SessionRecord, OptimizationGraph
from ..problem import Problem


class FileStorage(StorageBackend):
    """
    File-based storage using JSON files.

    v0.3.0 Directory structure:
        .paola_runs/
        ├── graphs/                     # v0.3.0+
        │   ├── graph_0001.json
        │   └── ...
        ├── sessions/                   # v0.2.0 legacy
        │   ├── session_0001.json
        │   └── ...
        ├── problems/
        │   └── problem_id.json
        └── metadata.json (tracks next_graph_id, next_session_id)
    """

    def __init__(self, base_dir: str = ".paola_runs"):
        """
        Initialize file storage.

        Args:
            base_dir: Base directory for storage (default: .paola_runs)
        """
        self.base_dir = Path(base_dir)
        self.graphs_dir = self.base_dir / "graphs"
        self.sessions_dir = self.base_dir / "sessions"
        self.problems_dir = self.base_dir / "problems"
        self.metadata_file = self.base_dir / "metadata.json"

        # Create directories if needed
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.problems_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata
        if not self.metadata_file.exists():
            self._save_metadata({"next_graph_id": 1, "next_session_id": 1})
        else:
            # Ensure next_graph_id exists for upgrades from v0.2.0
            metadata = self._load_metadata()
            if "next_graph_id" not in metadata:
                metadata["next_graph_id"] = 1
                self._save_metadata(metadata)

    # =========================================================================
    # Graph Operations (v0.3.0+)
    # =========================================================================

    def save_graph(self, graph: OptimizationGraph) -> None:
        """Save graph to JSON file."""
        file_path = self.graphs_dir / f"graph_{graph.graph_id:04d}.json"
        with open(file_path, 'w') as f:
            f.write(graph.to_json())

    def load_graph(self, graph_id: int) -> Optional[OptimizationGraph]:
        """Load graph from JSON file."""
        file_path = self.graphs_dir / f"graph_{graph_id:04d}.json"
        if not file_path.exists():
            return None

        with open(file_path, 'r') as f:
            return OptimizationGraph.from_json(f.read())

    def load_all_graphs(self) -> List[OptimizationGraph]:
        """Load all graphs, sorted by graph_id."""
        graphs = []
        for file_path in sorted(self.graphs_dir.glob("graph_*.json")):
            with open(file_path, 'r') as f:
                graphs.append(OptimizationGraph.from_json(f.read()))
        return graphs

    def get_next_graph_id(self) -> int:
        """Get and increment next graph ID."""
        metadata = self._load_metadata()
        graph_id = metadata.get("next_graph_id", 1)
        metadata["next_graph_id"] = graph_id + 1
        self._save_metadata(metadata)
        return graph_id

    # =========================================================================
    # Session Operations (v0.2.0 legacy)
    # =========================================================================

    def save_session(self, session: SessionRecord) -> None:
        """Save session to JSON file."""
        file_path = self.sessions_dir / f"session_{session.session_id:04d}.json"
        with open(file_path, 'w') as f:
            f.write(session.to_json())

    def load_session(self, session_id: int) -> Optional[SessionRecord]:
        """Load session from JSON file."""
        file_path = self.sessions_dir / f"session_{session_id:04d}.json"
        if not file_path.exists():
            return None

        with open(file_path, 'r') as f:
            return SessionRecord.from_json(f.read())

    def load_all_sessions(self) -> List[SessionRecord]:
        """Load all sessions, sorted by session_id."""
        sessions = []
        for file_path in sorted(self.sessions_dir.glob("session_*.json")):
            with open(file_path, 'r') as f:
                sessions.append(SessionRecord.from_json(f.read()))
        return sessions

    def get_next_session_id(self) -> int:
        """Get and increment next session ID."""
        metadata = self._load_metadata()
        session_id = metadata.get("next_session_id", 1)
        metadata["next_session_id"] = session_id + 1
        self._save_metadata(metadata)
        return session_id

    # =========================================================================
    # Problem Operations
    # =========================================================================

    def save_problem(self, problem: Problem) -> None:
        """Save problem metadata."""
        file_path = self.problems_dir / f"{problem.problem_id}.json"
        with open(file_path, 'w') as f:
            json.dump(problem.to_dict(), f, indent=2)

    def load_problem(self, problem_id: str) -> Optional[Problem]:
        """Load problem metadata."""
        file_path = self.problems_dir / f"{problem_id}.json"
        if not file_path.exists():
            return None

        with open(file_path, 'r') as f:
            return Problem.from_dict(json.load(f))

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _load_metadata(self) -> dict:
        """Load metadata from file."""
        with open(self.metadata_file, 'r') as f:
            return json.load(f)

    def _save_metadata(self, metadata: dict) -> None:
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
