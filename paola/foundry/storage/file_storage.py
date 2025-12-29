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

    v0.4.2 Directory structure (unified .paola_foundry):
        .paola_foundry/
        ├── evaluators/                 # Evaluator metadata and source
        │   ├── {id}.json               # EvaluatorConfig
        │   └── {id}.py                 # Standalone source
        ├── graphs/                     # Tier 1: GraphRecord (~1KB each)
        │   └── graph_0001.json
        ├── details/                    # Tier 2: GraphDetail (10-100KB each)
        │   └── graph_0001_detail.json
        ├── problems/                   # Problem storage with index
        │   ├── index.json              # Lineage tree and metadata
        │   └── rosenbrock_10d.json     # Full problem definitions
        └── metadata.json (tracks next_graph_id)
    """

    def __init__(self, base_dir: str = ".paola_foundry"):
        """
        Initialize file storage.

        Args:
            base_dir: Base directory for storage (default: .paola_foundry)
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

        # Auto-migrate from legacy .paola_runs if needed (v0.4.2 -> v0.4.3)
        self._migrate_legacy_data()

        # Initialize metadata
        if not self.metadata_file.exists():
            self._save_metadata({"next_graph_id": 1, "next_problem_id": 1})
        else:
            # Ensure next_graph_id and next_problem_id exist for upgrades
            metadata = self._load_metadata()
            updated = False
            if "next_graph_id" not in metadata:
                metadata["next_graph_id"] = 1
                updated = True
            if "next_problem_id" not in metadata:
                metadata["next_problem_id"] = 1
                updated = True
            if updated:
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

    def save_graph_from_record(
        self,
        record: GraphRecord,
        detail: Optional[GraphDetail] = None,
    ) -> None:
        """
        Save GraphRecord directly (for journal-based finalization).

        This method bypasses the OptimizationGraph → split_graph pipeline,
        allowing direct persistence of a GraphRecord created from journal data.

        v0.2.1: Added for journal-based finalization.

        Args:
            record: GraphRecord to save
            detail: Optional GraphDetail (creates minimal if not provided)
        """
        # Save Tier 1: GraphRecord
        record_path = self.graphs_dir / f"graph_{record.graph_id:04d}.json"
        with open(record_path, 'w') as f:
            f.write(record.to_json())

        # Save Tier 2: GraphDetail (even if minimal)
        if detail is None:
            detail = GraphDetail(graph_id=record.graph_id, nodes={})
        detail_path = self.details_dir / f"graph_{record.graph_id:04d}_detail.json"
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
        Save problem with index update (v0.4.3 - numeric IDs).

        Args:
            problem: OptimizationProblem (or subclass like NLPProblem)
        """
        # Save full problem definition with numeric ID filename
        file_path = self.problems_dir / f"problem_{problem.problem_id:04d}.json"
        with open(file_path, 'w') as f:
            json.dump(problem.to_dict(), f, indent=2)

        # Update index
        index = self._load_problem_index()

        # Build index entry (keyed by numeric ID as string for JSON compatibility)
        entry = {
            "problem_id": problem.problem_id,
            "name": problem.name,  # Human-readable name
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
        parent_key = str(problem.parent_problem_id) if problem.parent_problem_id else None
        if parent_key and parent_key in index["problems"]:
            parent_entry = index["problems"][parent_key]
            if problem.problem_id not in parent_entry.get("children", []):
                parent_entry.setdefault("children", []).append(problem.problem_id)

        # Store with string key (JSON requires string keys)
        index["problems"][str(problem.problem_id)] = entry
        self._save_problem_index(index)

    def save_problem_legacy(self, problem: Problem) -> None:
        """Save problem metadata (legacy format for backward compatibility)."""
        file_path = self.problems_dir / f"{problem.problem_id}.json"
        with open(file_path, 'w') as f:
            json.dump(problem.to_dict(), f, indent=2)

    def load_problem(self, problem_id: int) -> Optional[OptimizationProblem]:
        """
        Load problem by numeric ID (v0.4.3).

        Returns the correct subclass based on problem_type.

        Args:
            problem_id: Numeric problem identifier

        Returns:
            OptimizationProblem subclass instance or None
        """
        # Try numeric format first (v0.4.3+)
        file_path = self.problems_dir / f"problem_{problem_id:04d}.json"
        if not file_path.exists():
            # Fallback to legacy string format for backward compatibility
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

    def get_problem_lineage(self, problem_id: int) -> List[Dict[str, Any]]:
        """
        Get problem lineage (chain of parent problems).

        Args:
            problem_id: Numeric problem ID to trace lineage for

        Returns:
            List from root to this problem (inclusive)
        """
        index = self._load_problem_index()
        lineage = []

        current_id = problem_id
        while current_id is not None:
            # Index uses string keys for JSON compatibility
            entry = index.get("problems", {}).get(str(current_id))
            if not entry:
                break
            lineage.append(entry)
            current_id = entry.get("parent_problem_id")

        # Reverse to get root -> current order
        lineage.reverse()
        return lineage

    def get_problem_children(self, problem_id: int) -> List[int]:
        """
        Get direct children of a problem.

        Args:
            problem_id: Parent numeric problem ID

        Returns:
            List of child problem IDs (numeric)
        """
        index = self._load_problem_index()
        entry = index.get("problems", {}).get(str(problem_id))
        if not entry:
            return []
        return entry.get("children", [])

    def update_problem_graphs(self, problem_id: int, graph_id: int) -> None:
        """
        Record that a graph uses this problem.

        Called when a graph is finalized to track usage.

        Args:
            problem_id: Numeric problem ID
            graph_id: Graph ID that uses this problem
        """
        index = self._load_problem_index()
        entry = index.get("problems", {}).get(str(problem_id))
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

    def _migrate_legacy_data(self) -> None:
        """
        Migrate data from legacy .paola_runs directory to .paola_foundry.

        This handles the v0.4.2 -> v0.4.3 transition where storage was unified.
        Only runs if .paola_foundry is empty but .paola_runs has data.
        """
        import shutil

        legacy_dir = Path(".paola_runs")
        if not legacy_dir.exists():
            return

        # Check if we have data in legacy but not in new location
        legacy_graphs = legacy_dir / "graphs"
        legacy_details = legacy_dir / "details"

        # Only migrate if new location is empty and legacy has data
        new_has_graphs = any(self.graphs_dir.glob("graph_*.json"))
        legacy_has_graphs = legacy_graphs.exists() and any(legacy_graphs.glob("graph_*.json"))

        if not new_has_graphs and legacy_has_graphs:
            print(f"[Migration] Copying graphs from {legacy_graphs} to {self.graphs_dir}")
            for f in legacy_graphs.glob("graph_*.json"):
                shutil.copy2(f, self.graphs_dir / f.name)

        new_has_details = any(self.details_dir.glob("graph_*_detail.json"))
        legacy_has_details = legacy_details.exists() and any(legacy_details.glob("graph_*_detail.json"))

        if not new_has_details and legacy_has_details:
            print(f"[Migration] Copying details from {legacy_details} to {self.details_dir}")
            for f in legacy_details.glob("graph_*_detail.json"):
                shutil.copy2(f, self.details_dir / f.name)

        # Migrate metadata (take max next_graph_id)
        legacy_metadata_file = legacy_dir / "metadata.json"
        if legacy_metadata_file.exists() and self.metadata_file.exists():
            with open(legacy_metadata_file, 'r') as f:
                legacy_meta = json.load(f)
            with open(self.metadata_file, 'r') as f:
                new_meta = json.load(f)

            # Take the max next_graph_id
            legacy_id = legacy_meta.get("next_graph_id", 1)
            new_id = new_meta.get("next_graph_id", 1)
            if legacy_id > new_id:
                new_meta["next_graph_id"] = legacy_id
                self._save_metadata(new_meta)
                print(f"[Migration] Updated next_graph_id to {legacy_id}")

    def get_next_problem_id(self) -> int:
        """Get and increment next problem ID."""
        metadata = self._load_metadata()
        problem_id = metadata.get("next_problem_id", 1)
        metadata["next_problem_id"] = problem_id + 1
        self._save_metadata(metadata)
        return problem_id
