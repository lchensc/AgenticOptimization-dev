"""
Append-only journal for cross-process graph coordination.

This module provides the GraphJournal class that enables coordination
between the CLI process and subprocesses spawned by bash("python script.py").

Design:
- One JSON entry per line (JSONL format)
- Atomic appends with fcntl file locking
- Reconstruct state by replaying entries for a graph_id
- No compaction (full history preserved)

The journal solves the subprocess isolation problem where in-memory
state (_checkpoint_summaries) is lost when the subprocess exits.
RecordingObjective already writes evaluations to JSONL; the journal
handles graph-level metadata (best_x, best_f, status).

v0.2.1: Added get_full_graph_data() for journal-based finalization.
This allows main process to finalize graphs created in subprocesses.

See: docs/v0.2.0/20251229_journal_architecture_implementation.md
"""

import json
import fcntl
import time
from pathlib import Path
from typing import Optional, Dict, Any, List


class GraphJournal:
    """
    Append-only journal for graph metadata coordination.

    Entry types:
    - graph_created: When paola.objective() creates a new graph
    - checkpoint: When paola.checkpoint() saves node progress
    - continue: When paola.continue_graph() starts a new node
    - finalized: When paola.finalize_graph() completes a graph

    Example journal entries:
        {"type":"graph_created","graph_id":108,"problem_id":7,"goal":"...","ts":...}
        {"type":"checkpoint","graph_id":108,"node_id":"n1","best_f":3.98,"ts":...}
        {"type":"finalized","graph_id":108,"ts":...}
    """

    def __init__(self, journal_path: Path):
        """
        Initialize journal.

        Args:
            journal_path: Path to journal.jsonl file
        """
        self.path = Path(journal_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: Dict[str, Any]) -> None:
        """
        Append entry atomically with file locking.

        Args:
            entry: Journal entry dict (must include "type" field)
        """
        entry["ts"] = time.time()

        with open(self.path, 'a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps(entry) + '\n')
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def get_graph_state(self, graph_id: int) -> Optional[Dict[str, Any]]:
        """
        Reconstruct graph state by replaying journal entries.

        Args:
            graph_id: Graph ID to query

        Returns:
            State dict with keys:
            - graph_id: int
            - problem_id: int
            - goal: str
            - status: "active" | "checkpointed" | "finalized"
            - nodes: Dict[node_id -> node_info]
            - best_f: float or None
            - best_x: List[float] or None
            - best_node_id: str or None

            Returns None if graph not found.
        """
        if not self.path.exists():
            return None

        state = None
        nodes = {}

        with open(self.path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                if entry.get("graph_id") != graph_id:
                    continue

                entry_type = entry.get("type")

                if entry_type == "graph_created":
                    state = {
                        "graph_id": graph_id,
                        "problem_id": entry.get("problem_id"),
                        "goal": entry.get("goal"),
                        "status": "active",
                        "nodes": {},
                        "best_f": None,
                        "best_x": None,
                        "best_node_id": None,
                    }

                elif entry_type == "checkpoint" and state:
                    node_id = entry.get("node_id")
                    node_info = {
                        "node_id": node_id,
                        "best_f": entry.get("best_f"),
                        "best_x": entry.get("best_x"),
                        "n_evaluations": entry.get("n_evals"),
                        "optimizer": entry.get("optimizer"),
                        "status": "checkpointed",
                    }
                    state["nodes"][node_id] = node_info
                    state["status"] = "checkpointed"

                    # Update graph-level best
                    if entry.get("best_f") is not None:
                        if state["best_f"] is None or entry["best_f"] < state["best_f"]:
                            state["best_f"] = entry["best_f"]
                            state["best_x"] = entry.get("best_x")
                            state["best_node_id"] = node_id

                elif entry_type == "continue" and state:
                    # Track node creation from continue_graph
                    new_node_id = entry.get("new_node_id")
                    if new_node_id:
                        state["nodes"][new_node_id] = {
                            "node_id": new_node_id,
                            "parent_node": entry.get("parent_node"),
                            "edge_type": entry.get("edge_type"),
                            "status": "active",
                        }

                elif entry_type == "finalized" and state:
                    state["status"] = "finalized"
                    state["finalized_at"] = entry.get("ts")

        return state

    def get_node_best_x(self, graph_id: int, node_id: str) -> Optional[List[float]]:
        """
        Get best_x for a specific node (for warm-start).

        Args:
            graph_id: Graph ID
            node_id: Node ID (e.g., "n1")

        Returns:
            best_x as list or None if not found
        """
        state = self.get_graph_state(graph_id)
        if state is None:
            return None

        node = state.get("nodes", {}).get(node_id)
        if node is None:
            return None

        return node.get("best_x")

    def list_active_graphs(self) -> List[int]:
        """
        List graphs that are active or checkpointed (not finalized).

        Returns:
            List of graph IDs
        """
        if not self.path.exists():
            return []

        graph_states = {}  # graph_id -> status

        with open(self.path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                gid = entry.get("graph_id")
                etype = entry.get("type")

                if etype == "graph_created":
                    graph_states[gid] = "active"
                elif etype == "checkpoint":
                    graph_states[gid] = "checkpointed"
                elif etype == "finalized":
                    graph_states[gid] = "finalized"

        return [gid for gid, status in graph_states.items()
                if status in ("active", "checkpointed")]

    def count_nodes(self, graph_id: int) -> int:
        """
        Count nodes in a graph (for generating next node ID).

        Args:
            graph_id: Graph ID

        Returns:
            Number of nodes (checkpointed + active)
        """
        state = self.get_graph_state(graph_id)
        if state is None:
            return 0
        return len(state.get("nodes", {}))

    def get_full_graph_data(self, graph_id: int) -> Optional[Dict[str, Any]]:
        """
        Get complete graph data for reconstruction into GraphRecord.

        This method collects all journal entries for a graph and returns
        the full data needed to create a minimal but valid GraphRecord
        for journal-based finalization (v0.2.1).

        Args:
            graph_id: Graph ID to query

        Returns:
            Dict with:
            - graph_id: int
            - problem_id: int
            - goal: str
            - created_at: float (timestamp from graph_created entry)
            - nodes: Dict[node_id -> full node data including edges]
            - best_f: float (best across nodes)
            - best_x: List[float]
            - total_evaluations: int
            - total_wall_time: float

            Returns None if graph not found.
        """
        if not self.path.exists():
            return None

        graph_data = None

        with open(self.path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                if entry.get("graph_id") != graph_id:
                    continue

                entry_type = entry.get("type")

                if entry_type == "graph_created":
                    graph_data = {
                        "graph_id": graph_id,
                        "problem_id": entry.get("problem_id"),
                        "goal": entry.get("goal"),
                        "created_at": entry.get("ts"),
                        "nodes": {},
                        "best_f": None,
                        "best_x": None,
                        "best_node_id": None,
                        "total_evaluations": 0,
                        "total_wall_time": 0.0,
                        "status": "active",
                    }

                elif entry_type == "checkpoint" and graph_data:
                    node_id = entry.get("node_id")
                    node_data = {
                        "node_id": node_id,
                        "problem_id": entry.get("problem_id"),
                        "best_f": entry.get("best_f"),
                        "best_x": entry.get("best_x"),
                        "n_evaluations": entry.get("n_evals", 0),
                        "n_actual_evals": entry.get("n_actual_evals", 0),
                        "optimizer": entry.get("optimizer", "unknown"),
                        "wall_time": entry.get("wall_time", 0.0),
                        "parent_node": entry.get("parent_node"),
                        "edge_type": entry.get("edge_type"),
                        "status": "completed",
                        "checkpoint_ts": entry.get("ts"),
                    }
                    graph_data["nodes"][node_id] = node_data

                    # Accumulate totals
                    graph_data["total_evaluations"] += node_data["n_evaluations"]
                    graph_data["total_wall_time"] += node_data["wall_time"]

                    # Update graph-level best
                    if node_data["best_f"] is not None:
                        if graph_data["best_f"] is None or node_data["best_f"] < graph_data["best_f"]:
                            graph_data["best_f"] = node_data["best_f"]
                            graph_data["best_x"] = node_data["best_x"]
                            graph_data["best_node_id"] = node_id

                    graph_data["status"] = "checkpointed"

                elif entry_type == "continue" and graph_data:
                    # Update node with parent relationship if not already checkpointed
                    new_node_id = entry.get("new_node_id")
                    if new_node_id and new_node_id not in graph_data["nodes"]:
                        graph_data["nodes"][new_node_id] = {
                            "node_id": new_node_id,
                            "parent_node": entry.get("parent_node"),
                            "edge_type": entry.get("edge_type"),
                            "status": "active",
                        }

                elif entry_type == "finalized" and graph_data:
                    graph_data["status"] = "finalized"
                    graph_data["finalized_at"] = entry.get("ts")

        return graph_data
