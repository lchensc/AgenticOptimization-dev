"""File-based storage implementation using JSON files."""

import json
from pathlib import Path
from typing import List, Optional

from .backend import StorageBackend
from ..schema import SessionRecord, OptimizationGraph
from ..schema import GraphRecord, GraphDetail, ProblemSignature
from ..schema.conversion import split_graph, create_problem_signature
from ..problem import Problem


class FileStorage(StorageBackend):
    """
    File-based storage using JSON files.

    v0.3.1 Directory structure (two-tier):
        .paola_runs/
        ├── graphs/                     # Tier 1: GraphRecord (~1KB each)
        │   ├── graph_0001.json
        │   └── ...
        ├── details/                    # Tier 2: GraphDetail (10-100KB each)
        │   ├── graph_0001_detail.json
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
        self.details_dir = self.base_dir / "details"  # NEW: Tier 2
        self.sessions_dir = self.base_dir / "sessions"
        self.problems_dir = self.base_dir / "problems"
        self.metadata_file = self.base_dir / "metadata.json"

        # Create directories if needed
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        self.details_dir.mkdir(parents=True, exist_ok=True)  # NEW
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
    # Graph Operations (v0.3.1+ two-tier storage)
    # =========================================================================

    def save_graph(
        self,
        graph: OptimizationGraph,
        problem_signature: Optional[ProblemSignature] = None,
    ) -> None:
        """
        Save graph using two-tier storage.

        Args:
            graph: Full optimization graph
            problem_signature: Optional problem signature for similarity matching
        """
        # Split into Tier 1 (record) and Tier 2 (detail)
        record, detail = split_graph(graph, problem_signature)

        # Save Tier 1: GraphRecord (~1KB)
        record_path = self.graphs_dir / f"graph_{graph.graph_id:04d}.json"
        with open(record_path, 'w') as f:
            f.write(record.to_json())

        # Save Tier 2: GraphDetail (10-100KB)
        detail_path = self.details_dir / f"graph_{graph.graph_id:04d}_detail.json"
        with open(detail_path, 'w') as f:
            f.write(detail.to_json())

    def load_graph(self, graph_id: int) -> Optional[OptimizationGraph]:
        """
        Load full graph from JSON file (legacy format).

        For backward compatibility with v0.3.0 format.
        New code should use load_graph_record() and load_graph_detail().
        """
        file_path = self.graphs_dir / f"graph_{graph_id:04d}.json"
        if not file_path.exists():
            return None

        with open(file_path, 'r') as f:
            content = f.read()
            data = json.loads(content)

            # Check if it's new format (GraphRecord) or old format (OptimizationGraph)
            if "nodes" in data and isinstance(data.get("nodes"), dict):
                first_node = next(iter(data["nodes"].values()), None) if data["nodes"] else None
                # Old format has "progress" in nodes, new format has "config" directly
                if first_node and "progress" in first_node:
                    # Old format - full OptimizationGraph
                    return OptimizationGraph.from_json(content)

            # New format or empty - can't reconstruct full graph
            # Return None to indicate should use load_graph_record instead
            return None

    def load_graph_record(self, graph_id: int) -> Optional[GraphRecord]:
        """
        Load Tier 1 GraphRecord (for LLM queries).

        Args:
            graph_id: Graph ID to load

        Returns:
            GraphRecord or None if not found
        """
        file_path = self.graphs_dir / f"graph_{graph_id:04d}.json"
        if not file_path.exists():
            return None

        with open(file_path, 'r') as f:
            return GraphRecord.from_json(f.read())

    def load_graph_detail(self, graph_id: int) -> Optional[GraphDetail]:
        """
        Load Tier 2 GraphDetail (for visualization/deep analysis).

        Args:
            graph_id: Graph ID to load

        Returns:
            GraphDetail or None if not found
        """
        detail_path = self.details_dir / f"graph_{graph_id:04d}_detail.json"
        if not detail_path.exists():
            return None

        with open(detail_path, 'r') as f:
            return GraphDetail.from_json(f.read())

    def load_all_graph_records(self) -> List[GraphRecord]:
        """Load all GraphRecords (Tier 1), sorted by graph_id."""
        records = []
        for file_path in sorted(self.graphs_dir.glob("graph_*.json")):
            try:
                with open(file_path, 'r') as f:
                    records.append(GraphRecord.from_json(f.read()))
            except Exception:
                # Skip files that can't be parsed as GraphRecord
                pass
        return records

    def load_all_graphs(self) -> List[OptimizationGraph]:
        """
        Load all graphs (legacy format), sorted by graph_id.

        DEPRECATED: Use load_all_graph_records() instead.
        """
        graphs = []
        for file_path in sorted(self.graphs_dir.glob("graph_*.json")):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    data = json.loads(content)

                    # Try to parse as old format
                    first_node = next(iter(data.get("nodes", {}).values()), None)
                    if first_node and "progress" in first_node:
                        graphs.append(OptimizationGraph.from_json(content))
            except Exception:
                pass
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
