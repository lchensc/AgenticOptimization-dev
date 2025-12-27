"""
Pareto Front Storage Backend.

Provides persistent storage for Pareto fronts from multi-objective optimization.

Storage structure:
    .paola_foundry/
    └── pareto/
        └── graph_{graph_id}_node_{node_id}.json

Each file contains a complete ParetoFront with all solutions and metadata.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..schema.pareto import ParetoFront, ParetoSolution

logger = logging.getLogger(__name__)


class ParetoStorage:
    """
    Storage backend for Pareto fronts.

    Stores Pareto fronts as JSON files indexed by graph_id and node_id.
    Provides query methods for retrieving and comparing Pareto fronts.
    """

    def __init__(self, base_path: str):
        """
        Initialize Pareto storage.

        Args:
            base_path: Base path for foundry storage (e.g., ".paola_foundry")
        """
        self.base_path = Path(base_path)
        self.pareto_dir = self.base_path / "pareto"
        self.pareto_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, graph_id: int, node_id: str) -> Path:
        """Get file path for a Pareto front."""
        return self.pareto_dir / f"graph_{graph_id}_node_{node_id}.json"

    def store(self, pareto_front: ParetoFront) -> str:
        """
        Store a Pareto front.

        Args:
            pareto_front: ParetoFront to store (must have graph_id and node_id)

        Returns:
            Path to stored file

        Raises:
            ValueError: If graph_id or node_id not set
        """
        if pareto_front.graph_id is None or pareto_front.node_id is None:
            raise ValueError("ParetoFront must have graph_id and node_id set")

        path = self._get_path(pareto_front.graph_id, pareto_front.node_id)

        with open(path, "w") as f:
            json.dump(pareto_front.to_dict(), f, indent=2)

        logger.info(
            f"Stored Pareto front: graph {pareto_front.graph_id}, "
            f"node {pareto_front.node_id}, {pareto_front.n_solutions} solutions"
        )

        return str(path)

    def load(self, graph_id: int, node_id: str) -> Optional[ParetoFront]:
        """
        Load a Pareto front.

        Args:
            graph_id: Graph ID
            node_id: Node ID

        Returns:
            ParetoFront if found, None otherwise
        """
        path = self._get_path(graph_id, node_id)

        if not path.exists():
            return None

        with open(path, "r") as f:
            data = json.load(f)

        return ParetoFront.from_dict(data)

    def exists(self, graph_id: int, node_id: str) -> bool:
        """Check if a Pareto front exists."""
        return self._get_path(graph_id, node_id).exists()

    def delete(self, graph_id: int, node_id: str) -> bool:
        """
        Delete a Pareto front.

        Returns:
            True if deleted, False if not found
        """
        path = self._get_path(graph_id, node_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_for_graph(self, graph_id: int) -> List[str]:
        """
        List all Pareto fronts for a graph.

        Returns:
            List of node_ids with Pareto fronts
        """
        node_ids = []
        prefix = f"graph_{graph_id}_node_"

        for path in self.pareto_dir.glob(f"{prefix}*.json"):
            # Extract node_id from filename
            filename = path.stem  # e.g., "graph_1_node_n3"
            if filename.startswith(prefix):
                node_id = filename[len(prefix):]
                node_ids.append(node_id)

        return sorted(node_ids)

    def list_all(self) -> List[Dict[str, Any]]:
        """
        List all stored Pareto fronts.

        Returns:
            List of dicts with graph_id, node_id, n_solutions, hypervolume
        """
        results = []

        for path in self.pareto_dir.glob("graph_*_node_*.json"):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                results.append({
                    "graph_id": data.get("graph_id"),
                    "node_id": data.get("node_id"),
                    "n_solutions": data.get("n_solutions", len(data.get("solutions", []))),
                    "hypervolume": data.get("hypervolume"),
                    "algorithm": data.get("algorithm"),
                    "objective_names": data.get("objective_names"),
                })
            except Exception as e:
                logger.warning(f"Error reading {path}: {e}")

        return results

    def get_best_hypervolume(self, graph_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the Pareto front with highest hypervolume for a graph.

        Returns:
            Dict with node_id, hypervolume, n_solutions, or None if none found
        """
        node_ids = self.list_for_graph(graph_id)
        if not node_ids:
            return None

        best = None
        best_hv = float("-inf")

        for node_id in node_ids:
            pf = self.load(graph_id, node_id)
            if pf and pf.hypervolume is not None and pf.hypervolume > best_hv:
                best_hv = pf.hypervolume
                best = {
                    "node_id": node_id,
                    "hypervolume": pf.hypervolume,
                    "n_solutions": pf.n_solutions,
                }

        return best

    def query(
        self,
        graph_id: Optional[int] = None,
        min_solutions: Optional[int] = None,
        min_hypervolume: Optional[float] = None,
        algorithm: Optional[str] = None,
        limit: int = 10,
    ) -> List[ParetoFront]:
        """
        Query Pareto fronts with filters.

        Args:
            graph_id: Filter by graph ID
            min_solutions: Minimum number of solutions
            min_hypervolume: Minimum hypervolume value
            algorithm: Filter by algorithm name
            limit: Maximum results to return

        Returns:
            List of matching ParetoFront objects
        """
        results = []

        for item in self.list_all():
            # Apply filters
            if graph_id is not None and item["graph_id"] != graph_id:
                continue
            if min_solutions is not None and item["n_solutions"] < min_solutions:
                continue
            if min_hypervolume is not None:
                hv = item.get("hypervolume")
                if hv is None or hv < min_hypervolume:
                    continue
            if algorithm is not None:
                if item.get("algorithm") != algorithm:
                    continue

            # Load full Pareto front
            pf = self.load(item["graph_id"], item["node_id"])
            if pf:
                results.append(pf)

            if len(results) >= limit:
                break

        return results

    def compare(
        self,
        graph_id_1: int,
        node_id_1: str,
        graph_id_2: int,
        node_id_2: str,
    ) -> Dict[str, Any]:
        """
        Compare two Pareto fronts.

        Returns:
            Comparison dict with hypervolume difference, solution counts, etc.
        """
        pf1 = self.load(graph_id_1, node_id_1)
        pf2 = self.load(graph_id_2, node_id_2)

        if pf1 is None:
            raise ValueError(f"Pareto front not found: graph {graph_id_1}, node {node_id_1}")
        if pf2 is None:
            raise ValueError(f"Pareto front not found: graph {graph_id_2}, node {node_id_2}")

        # Compute hypervolumes if not already computed
        if pf1.hypervolume is None:
            pf1.compute_hypervolume()
        if pf2.hypervolume is None:
            pf2.compute_hypervolume()

        return {
            "front_1": {
                "graph_id": graph_id_1,
                "node_id": node_id_1,
                "n_solutions": pf1.n_solutions,
                "hypervolume": pf1.hypervolume,
                "algorithm": pf1.algorithm,
            },
            "front_2": {
                "graph_id": graph_id_2,
                "node_id": node_id_2,
                "n_solutions": pf2.n_solutions,
                "hypervolume": pf2.hypervolume,
                "algorithm": pf2.algorithm,
            },
            "hypervolume_diff": (
                (pf1.hypervolume - pf2.hypervolume)
                if pf1.hypervolume and pf2.hypervolume
                else None
            ),
            "better": (
                "front_1" if pf1.hypervolume and pf2.hypervolume and pf1.hypervolume > pf2.hypervolume
                else "front_2" if pf1.hypervolume and pf2.hypervolume and pf2.hypervolume > pf1.hypervolume
                else "equal"
            ),
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dict with total count, total solutions, storage size, etc.
        """
        all_fronts = self.list_all()

        total_solutions = sum(f["n_solutions"] for f in all_fronts)
        total_size = sum(
            self._get_path(f["graph_id"], f["node_id"]).stat().st_size
            for f in all_fronts
            if self._get_path(f["graph_id"], f["node_id"]).exists()
        )

        return {
            "n_pareto_fronts": len(all_fronts),
            "total_solutions": total_solutions,
            "storage_bytes": total_size,
            "storage_kb": total_size / 1024,
            "unique_graphs": len(set(f["graph_id"] for f in all_fronts)),
        }
