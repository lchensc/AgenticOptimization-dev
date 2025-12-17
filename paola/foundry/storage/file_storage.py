"""File-based storage implementation using JSON files."""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from .backend import StorageBackend
from ..schema import OptimizationGraph
from ..schema import GraphRecord, GraphDetail, ProblemSignature
from ..schema.problem import OptimizationProblem, deserialize_problem
from ..schema.conversion import split_graph, create_problem_signature
from ..problem import Problem


class FileStorage(StorageBackend):
    """
    File-based storage using JSON files.

    v0.4.1 Directory structure (with problem index):
        .paola_runs/
        ├── graphs/                     # Tier 1: GraphRecord (~1KB each)
        │   ├── graph_0001.json
        │   └── ...
        ├── details/                    # Tier 2: GraphDetail (10-100KB each)
        │   ├── graph_0001_detail.json
        │   └── ...
        ├── problems/                   # Problem storage with index
        │   ├── index.json              # Lineage tree and metadata
        │   ├── rosenbrock_10d.json     # Full problem definitions
        │   └── rosenbrock_10d_v2.json
        └── metadata.json (tracks next_graph_id)
    """

    def __init__(self, base_dir: str = ".paola_runs"):
        """
        Initialize file storage.

        Args:
            base_dir: Base directory for storage (default: .paola_runs)
        """
        self.base_dir = Path(base_dir)
        self.graphs_dir = self.base_dir / "graphs"
        self.details_dir = self.base_dir / "details"
        self.problems_dir = self.base_dir / "problems"
        self.metadata_file = self.base_dir / "metadata.json"
        self.problem_index_file = self.problems_dir / "index.json"

        # Create directories if needed
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        self.details_dir.mkdir(parents=True, exist_ok=True)
        self.problems_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata
        if not self.metadata_file.exists():
            self._save_metadata({"next_graph_id": 1})
        else:
            # Ensure next_graph_id exists for upgrades
            metadata = self._load_metadata()
            if "next_graph_id" not in metadata:
                metadata["next_graph_id"] = 1
                self._save_metadata(metadata)

        # Initialize problem index (v0.4.1)
        if not self.problem_index_file.exists():
            self._save_problem_index({"problems": {}})

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
    # Problem Operations (v0.4.1 - with index and lineage tracking)
    # =========================================================================

    def save_problem(self, problem: OptimizationProblem) -> None:
        """
        Save problem with index update (v0.4.1).

        Args:
            problem: OptimizationProblem (or subclass like NLPProblem)
        """
        # Save full problem definition
        file_path = self.problems_dir / f"{problem.problem_id}.json"
        with open(file_path, 'w') as f:
            json.dump(problem.to_dict(), f, indent=2)

        # Update index
        index = self._load_problem_index()

        # Build index entry
        entry = {
            "problem_id": problem.problem_id,
            "problem_type": problem.problem_type,
            "n_variables": problem.n_variables,
            "n_constraints": problem.n_constraints,
            "created_at": problem.created_at,
            "parent_problem_id": problem.parent_problem_id,
            "derivation_type": problem.derivation_type,
            "version": problem.version,
            "children": [],  # Will be updated when children are created
            "graphs_using": [],  # Will be updated when graphs reference this
        }

        # If this is a derived problem, update parent's children list
        if problem.parent_problem_id and problem.parent_problem_id in index["problems"]:
            parent_entry = index["problems"][problem.parent_problem_id]
            if problem.problem_id not in parent_entry.get("children", []):
                parent_entry.setdefault("children", []).append(problem.problem_id)

        index["problems"][problem.problem_id] = entry
        self._save_problem_index(index)

    def save_problem_legacy(self, problem: Problem) -> None:
        """Save problem metadata (legacy format for backward compatibility)."""
        file_path = self.problems_dir / f"{problem.problem_id}.json"
        with open(file_path, 'w') as f:
            json.dump(problem.to_dict(), f, indent=2)

    def load_problem(self, problem_id: str) -> Optional[OptimizationProblem]:
        """
        Load problem by ID (v0.4.1).

        Returns the correct subclass based on problem_type.

        Args:
            problem_id: Problem identifier

        Returns:
            OptimizationProblem subclass instance or None
        """
        file_path = self.problems_dir / f"{problem_id}.json"
        if not file_path.exists():
            return None

        with open(file_path, 'r') as f:
            data = json.load(f)

        # Use type registry to deserialize correct class
        return deserialize_problem(data)

    def load_problem_legacy(self, problem_id: str) -> Optional[Problem]:
        """Load problem metadata (legacy format)."""
        file_path = self.problems_dir / f"{problem_id}.json"
        if not file_path.exists():
            return None

        with open(file_path, 'r') as f:
            return Problem.from_dict(json.load(f))

    def list_problems(
        self,
        problem_type: Optional[str] = None,
        show_derived: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        List all problems from index.

        Args:
            problem_type: Filter by type ("NLP", "LP", etc.)
            show_derived: Include derived problems (default: True)

        Returns:
            List of problem index entries
        """
        index = self._load_problem_index()
        problems = []

        for problem_id, entry in index.get("problems", {}).items():
            # Filter by type
            if problem_type and entry.get("problem_type") != problem_type:
                continue

            # Filter out derived if requested
            if not show_derived and entry.get("parent_problem_id"):
                continue

            problems.append(entry)

        # Sort by created_at
        problems.sort(key=lambda p: p.get("created_at", ""))
        return problems

    def get_problem_lineage(self, problem_id: str) -> List[Dict[str, Any]]:
        """
        Get problem lineage (chain of parent problems).

        Args:
            problem_id: Problem to trace lineage for

        Returns:
            List from root to this problem (inclusive)
        """
        index = self._load_problem_index()
        lineage = []

        current_id = problem_id
        while current_id:
            entry = index.get("problems", {}).get(current_id)
            if not entry:
                break
            lineage.append(entry)
            current_id = entry.get("parent_problem_id")

        # Reverse to get root -> current order
        lineage.reverse()
        return lineage

    def get_problem_children(self, problem_id: str) -> List[str]:
        """
        Get direct children of a problem.

        Args:
            problem_id: Parent problem ID

        Returns:
            List of child problem IDs
        """
        index = self._load_problem_index()
        entry = index.get("problems", {}).get(problem_id)
        if not entry:
            return []
        return entry.get("children", [])

    def update_problem_graphs(self, problem_id: str, graph_id: int) -> None:
        """
        Record that a graph uses this problem.

        Called when a graph is finalized to track usage.

        Args:
            problem_id: Problem ID
            graph_id: Graph ID that uses this problem
        """
        index = self._load_problem_index()
        entry = index.get("problems", {}).get(problem_id)
        if entry:
            if graph_id not in entry.get("graphs_using", []):
                entry.setdefault("graphs_using", []).append(graph_id)
            self._save_problem_index(index)

    def _load_problem_index(self) -> Dict[str, Any]:
        """Load problem index from file."""
        if not self.problem_index_file.exists():
            return {"problems": {}}
        with open(self.problem_index_file, 'r') as f:
            return json.load(f)

    def _save_problem_index(self, index: Dict[str, Any]) -> None:
        """Save problem index to file."""
        with open(self.problem_index_file, 'w') as f:
            json.dump(index, f, indent=2)

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
