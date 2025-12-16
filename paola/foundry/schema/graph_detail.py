"""
Tier 2: GraphDetail - Full trajectory data for debugging/visualization.

This is the detailed representation for deep analysis.
Focus: "What happened during execution?"

Size: 10-100KB+ (depends on dimensions and iterations)

Contains:
- Initial points (x0)
- Best solutions (best_x)
- Convergence history (objective vs iteration)
- Full x trajectory (x at every iteration)

Used by:
- CLI /graph plot (convergence curves)
- Future get_node_detail tool (LLM deep analysis)
- Debugging and visualization
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class ConvergencePoint:
    """Single point in convergence history."""

    iteration: int
    objective: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "objective": self.objective,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConvergencePoint":
        return cls(
            iteration=data["iteration"],
            objective=data["objective"],
        )


@dataclass
class XPoint:
    """Single point in x trajectory."""

    iteration: int
    x: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "x": self.x,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "XPoint":
        return cls(
            iteration=data["iteration"],
            x=data["x"],
        )


@dataclass
class NodeDetail:
    """
    Full execution data for a node.

    Focus: What happened during optimization?
    """

    node_id: str
    x0: Optional[List[float]] = None  # Initial point
    best_x: Optional[List[float]] = None  # Best solution
    convergence_history: List[ConvergencePoint] = field(default_factory=list)
    x_history: List[XPoint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "x0": self.x0,
            "best_x": self.best_x,
            "convergence_history": [p.to_dict() for p in self.convergence_history],
            "x_history": [p.to_dict() for p in self.x_history],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeDetail":
        convergence = [
            ConvergencePoint.from_dict(p)
            for p in data.get("convergence_history", [])
        ]
        x_history = [
            XPoint.from_dict(p)
            for p in data.get("x_history", [])
        ]
        return cls(
            node_id=data["node_id"],
            x0=data.get("x0"),
            best_x=data.get("best_x"),
            convergence_history=convergence,
            x_history=x_history,
        )

    def get_objectives(self) -> List[float]:
        """Get list of objectives from convergence history."""
        return [p.objective for p in self.convergence_history]

    def get_iterations(self) -> List[int]:
        """Get list of iteration numbers from convergence history."""
        return [p.iteration for p in self.convergence_history]


@dataclass
class GraphDetail:
    """
    Tier 2: Full trajectory data.

    Stored separately from GraphRecord to keep Tier 1 compact.
    Loaded on demand for visualization and deep analysis.
    """

    graph_id: int
    nodes: Dict[str, NodeDetail] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphDetail":
        nodes = {
            nid: NodeDetail.from_dict(ndata)
            for nid, ndata in data.get("nodes", {}).items()
        }
        return cls(
            graph_id=data["graph_id"],
            nodes=nodes,
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "GraphDetail":
        """Deserialize from JSON string."""
        import json
        return cls.from_dict(json.loads(json_str))

    def get_node_detail(self, node_id: str) -> Optional[NodeDetail]:
        """Get detail for a specific node."""
        return self.nodes.get(node_id)
