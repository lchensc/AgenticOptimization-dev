"""
Tier 1: GraphRecord - LLM-ready graph representation.

This is the compact representation optimized for cross-graph learning.
Focus: "What strategy was used and did it work?"

Size: ~1-2KB per graph (minimal, strategy-focused)

Contains:
- Problem signature (for similarity matching)
- Strategy pattern and structure
- Node summaries with FULL config (no trajectories)
- Outcome and decisions

Does NOT contain:
- Convergence history (moved to Tier 2)
- x trajectory (moved to Tier 2)
- Raw iteration data
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class ProblemSignature:
    """
    Problem characteristics for similarity matching.

    Enables finding similar problems across graphs.
    """

    n_dimensions: int
    n_constraints: int
    bounds_range: Tuple[float, float]  # (min_bound, max_bound)
    constraint_types: List[str] = field(default_factory=list)  # ["equality", "inequality"]
    domain_hint: Optional[str] = None  # "rosenbrock", "ackley", etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_dimensions": self.n_dimensions,
            "n_constraints": self.n_constraints,
            "bounds_range": list(self.bounds_range),
            "constraint_types": self.constraint_types,
            "domain_hint": self.domain_hint,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProblemSignature":
        return cls(
            n_dimensions=data["n_dimensions"],
            n_constraints=data["n_constraints"],
            bounds_range=tuple(data["bounds_range"]),
            constraint_types=data.get("constraint_types", []),
            domain_hint=data.get("domain_hint"),
        )


@dataclass
class NodeSummary:
    """
    Minimal node representation for LLM learning.

    Focus: What optimizer with what config achieved what result?
    No trajectory data - that's in Tier 2.
    """

    node_id: str
    optimizer: str  # "scipy:SLSQP", "optuna:TPE"
    optimizer_family: str  # "gradient", "bayesian", "population", "cmaes"
    config: Dict[str, Any]  # FULL configuration (key for learning!)
    init_strategy: str  # "random", "warm_start", "center"
    parent_node: Optional[str]  # Edge source (for warm_start)
    edge_type: Optional[str]  # "warm_start", "refine", "restart", etc.

    status: str  # "completed", "failed"
    n_evaluations: int
    wall_time: float
    start_objective: Optional[float]  # Objective at iteration 1
    best_objective: Optional[float]  # Best objective achieved

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "optimizer": self.optimizer,
            "optimizer_family": self.optimizer_family,
            "config": self.config,
            "init_strategy": self.init_strategy,
            "parent_node": self.parent_node,
            "edge_type": self.edge_type,
            "status": self.status,
            "n_evaluations": self.n_evaluations,
            "wall_time": self.wall_time,
            "start_objective": self.start_objective,
            "best_objective": self.best_objective,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeSummary":
        return cls(
            node_id=data["node_id"],
            optimizer=data["optimizer"],
            optimizer_family=data["optimizer_family"],
            config=data.get("config", {}),
            init_strategy=data.get("init_strategy", "unknown"),
            parent_node=data.get("parent_node"),
            edge_type=data.get("edge_type"),
            status=data.get("status", "completed"),
            n_evaluations=data.get("n_evaluations", 0),
            wall_time=data.get("wall_time", 0.0),
            start_objective=data.get("start_objective"),
            best_objective=data.get("best_objective"),
        )


@dataclass
class EdgeSummary:
    """Compact edge representation."""

    source: str
    target: str
    edge_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EdgeSummary":
        return cls(
            source=data["source"],
            target=data["target"],
            edge_type=data["edge_type"],
        )


@dataclass
class GraphRecord:
    """
    Tier 1: LLM-ready graph representation.

    Focus: Strategy and outcome, not trajectory.
    ~1-2KB per graph.

    This is what the LLM sees when querying past graphs for learning.
    Contains everything needed to understand:
    - What problem was solved
    - What strategy was used (pattern, optimizers, configs)
    - What outcome was achieved
    - Why decisions were made
    """

    # Identity
    graph_id: int
    problem_id: str
    created_at: str
    goal: Optional[str] = None

    # Problem signature (enables similarity matching)
    problem_signature: Optional[ProblemSignature] = None

    # Structure
    pattern: str = "single"  # "single", "chain", "multistart", "tree", "dag"
    edges: List[EdgeSummary] = field(default_factory=list)

    # Nodes (minimal)
    nodes: Dict[str, NodeSummary] = field(default_factory=dict)

    # Outcome
    status: str = "completed"
    success: bool = False
    final_objective: Optional[float] = None
    final_x: Optional[List[float]] = None
    total_evaluations: int = 0
    total_wall_time: float = 0.0

    # Decisions (reasoning strings)
    decisions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "problem_id": self.problem_id,
            "created_at": self.created_at,
            "goal": self.goal,
            "problem_signature": self.problem_signature.to_dict() if self.problem_signature else None,
            "pattern": self.pattern,
            "edges": [e.to_dict() for e in self.edges],
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "status": self.status,
            "success": self.success,
            "final_objective": self.final_objective,
            "final_x": self.final_x,
            "total_evaluations": self.total_evaluations,
            "total_wall_time": self.total_wall_time,
            "decisions": self.decisions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphRecord":
        sig_data = data.get("problem_signature")
        signature = ProblemSignature.from_dict(sig_data) if sig_data else None

        edges = [EdgeSummary.from_dict(e) for e in data.get("edges", [])]
        nodes = {
            nid: NodeSummary.from_dict(ndata)
            for nid, ndata in data.get("nodes", {}).items()
        }

        return cls(
            graph_id=data["graph_id"],
            problem_id=data["problem_id"],
            created_at=data["created_at"],
            goal=data.get("goal"),
            problem_signature=signature,
            pattern=data.get("pattern", "single"),
            edges=edges,
            nodes=nodes,
            status=data.get("status", "completed"),
            success=data.get("success", False),
            final_objective=data.get("final_objective"),
            final_x=data.get("final_x"),
            total_evaluations=data.get("total_evaluations", 0),
            total_wall_time=data.get("total_wall_time", 0.0),
            decisions=data.get("decisions", []),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "GraphRecord":
        """Deserialize from JSON string."""
        import json
        return cls.from_dict(json.loads(json_str))

    def summary(self) -> str:
        """Get a brief summary for display."""
        n_nodes = len(self.nodes)
        status_str = "success" if self.success else "failed"
        obj_str = f"{self.final_objective:.6e}" if self.final_objective is not None else "N/A"
        return (
            f"Graph #{self.graph_id}: {self.problem_id} | "
            f"{n_nodes} nodes | {self.pattern} | "
            f"{status_str} | obj={obj_str}"
        )

    def get_strategy_sequence(self) -> List[str]:
        """Get ordered list of optimizers used."""
        # Sort nodes by creation order (assuming node_id order = creation order)
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.node_id)
        return [n.optimizer for n in sorted_nodes]
