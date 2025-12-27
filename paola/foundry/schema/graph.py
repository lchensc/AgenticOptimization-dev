"""
Graph-based schema for optimization data.

v0.3.0: Graph replaces Session as THE data model.

Core Principles:
- "Graph as substrate, not constraint"
- "Graph externalizes state, agent makes decisions"

Hierarchy:
- OptimizationGraph: Complete optimization task
- OptimizationNode: Single optimizer execution
- OptimizationEdge: Typed relationship between nodes
- GraphDecision: Strategic decision record

Terminology:
- x: Optimization variable vector (universal math convention)
- best_x: Best solution found
- final_x: Final solution of graph
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Literal


# =============================================================================
# Edge Types
# =============================================================================

class EdgeType:
    """
    Standard edge types for optimization graphs.

    These capture the semantic meaning of relationships between nodes.
    """

    # Primary types
    WARM_START = "warm_start"   # Use source's best_x as target's x0
    RESTART = "restart"         # Same config as source, new random seed
    REFINE = "refine"           # Local refinement from source's solution

    # Exploration types
    BRANCH = "branch"           # Explore different direction from source
    EXPLORE = "explore"         # Global exploration seeded by source

    # Combination types (for future use)
    MERGE = "merge"             # Target combines results from multiple sources
    SELECT = "select"           # Target selected from multiple candidates

    @classmethod
    def all_types(cls) -> List[str]:
        """Get all valid edge types."""
        return [
            cls.WARM_START,
            cls.RESTART,
            cls.REFINE,
            cls.BRANCH,
            cls.EXPLORE,
            cls.MERGE,
            cls.SELECT,
        ]

    @classmethod
    def is_valid(cls, edge_type: str) -> bool:
        """Check if edge type is valid."""
        return edge_type in cls.all_types()


# =============================================================================
# Edge
# =============================================================================

@dataclass
class OptimizationEdge:
    """
    Directed relationship between optimization nodes.

    Edge types capture the semantic meaning of the relationship.
    """

    source: str                 # Source node_id
    target: str                 # Target node_id
    edge_type: str              # Relationship type (see EdgeType)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate edge type."""
        if not EdgeType.is_valid(self.edge_type):
            raise ValueError(
                f"Invalid edge type: {self.edge_type}. "
                f"Valid types: {EdgeType.all_types()}"
            )

    @classmethod
    def create(
        cls,
        source: str,
        target: str,
        edge_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'OptimizationEdge':
        """Factory method to create an edge."""
        return cls(
            source=source,
            target=target,
            edge_type=edge_type,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationEdge':
        """Deserialize from dictionary."""
        return cls(
            source=data["source"],
            target=data["target"],
            edge_type=data["edge_type"],
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Node
# =============================================================================

@dataclass
class OptimizationNode:
    """
    A single optimization execution in the graph.

    Equivalent to the previous "Run" concept, but with:
    - String node_id (n1, n2, n3) instead of int run_id
    - Explicit status tracking
    - Optional components (populated when completed)
    """

    # Identity
    node_id: str                    # Unique within graph (e.g., "n1", "n2")

    # Optimizer configuration
    optimizer: str                  # Full spec: "scipy:SLSQP", "optuna:TPE"
    optimizer_family: str           # Family: "gradient", "bayesian", etc.
    config: Dict[str, Any] = field(default_factory=dict)

    # Execution state
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    created_at: str = ""            # ISO timestamp
    completed_at: Optional[str] = None

    # Results (populated when completed)
    n_evaluations: int = 0
    wall_time: float = 0.0
    best_objective: Optional[float] = None
    best_x: Optional[List[float]] = None  # Best solution vector

    # MOO-specific fields (v0.4.9)
    is_multiobjective: bool = False
    n_pareto_solutions: Optional[int] = None  # Number of Pareto-optimal solutions
    hypervolume: Optional[float] = None  # Hypervolume indicator
    pareto_ref: Optional[str] = None  # Reference to ParetoStorage file

    # Polymorphic components (per optimizer family)
    # These are Optional because they're populated during/after execution
    initialization: Optional[Any] = None  # InitializationComponent
    progress: Optional[Any] = None        # ProgressComponent
    result: Optional[Any] = None          # ResultComponent

    def is_completed(self) -> bool:
        """Check if node has completed execution."""
        return self.status == "completed"

    def is_successful(self) -> bool:
        """Check if node completed successfully with a result."""
        return self.status == "completed" and self.best_objective is not None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "node_id": self.node_id,
            "optimizer": self.optimizer,
            "optimizer_family": self.optimizer_family,
            "config": self.config,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "n_evaluations": self.n_evaluations,
            "wall_time": self.wall_time,
            "best_objective": self.best_objective,
            "best_x": self.best_x,
            "initialization": self.initialization.to_dict() if self.initialization else None,
            "progress": self.progress.to_dict() if self.progress else None,
            "result": self.result.to_dict() if self.result else None,
        }
        # MOO fields (only include if relevant)
        if self.is_multiobjective:
            result["is_multiobjective"] = True
            result["n_pareto_solutions"] = self.n_pareto_solutions
            result["hypervolume"] = self.hypervolume
            result["pareto_ref"] = self.pareto_ref
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationNode':
        """Deserialize from dictionary."""
        from .registry import COMPONENT_REGISTRY

        family = data["optimizer_family"]

        # Deserialize polymorphic components if present
        init_data = data.get("initialization")
        progress_data = data.get("progress")
        result_data = data.get("result")

        initialization = None
        progress = None
        result = None

        if init_data and progress_data and result_data:
            initialization, progress, result = COMPONENT_REGISTRY.deserialize_components(
                family=family,
                init_data=init_data,
                progress_data=progress_data,
                result_data=result_data,
            )

        return cls(
            node_id=data["node_id"],
            optimizer=data["optimizer"],
            optimizer_family=family,
            config=data.get("config", {}),
            status=data.get("status", "completed"),
            created_at=data.get("created_at", ""),
            completed_at=data.get("completed_at"),
            n_evaluations=data.get("n_evaluations", 0),
            wall_time=data.get("wall_time", 0.0),
            best_objective=data.get("best_objective"),
            best_x=data.get("best_x"),
            # MOO fields (v0.4.9)
            is_multiobjective=data.get("is_multiobjective", False),
            n_pareto_solutions=data.get("n_pareto_solutions"),
            hypervolume=data.get("hypervolume"),
            pareto_ref=data.get("pareto_ref"),
            initialization=initialization,
            progress=progress,
            result=result,
        )


# =============================================================================
# Decision (updated for graph)
# =============================================================================

@dataclass
class GraphDecision:
    """
    Record of Paola's strategic decision during optimization.

    Similar to PaolaDecision but references nodes instead of runs.
    """

    timestamp: str
    decision_type: str              # "add_node", "branch", "terminate", etc.
    reasoning: str                  # Natural language explanation
    from_node: Optional[str]        # Node ID before decision
    to_node: Optional[str]          # Node ID after decision
    metrics_at_decision: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp,
            "decision_type": self.decision_type,
            "reasoning": self.reasoning,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "metrics_at_decision": self.metrics_at_decision,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphDecision':
        """Deserialize from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            decision_type=data["decision_type"],
            reasoning=data["reasoning"],
            from_node=data.get("from_node"),
            to_node=data.get("to_node"),
            metrics_at_decision=data.get("metrics_at_decision", {}),
        )


# =============================================================================
# Graph
# =============================================================================

@dataclass
class OptimizationGraph:
    """
    Graph-based representation of an optimization task.

    This is THE data model for optimization in PAOLA v0.3.0+.
    Replaces the previous Session/Run list model.

    The graph captures:
    - What optimizers were run (nodes)
    - How they relate to each other (edges)
    - Why decisions were made (decisions)
    - Overall outcome (success, best result)

    Example:
        Graph #42: wing_drag

            ┌─────────┐
            │   n1    │ optuna:TPE (explore)
            └────┬────┘
                 │ warm_start
            ┌────┴────┐
            │   n2    │ scipy:SLSQP (refine)
            └────┬────┘
                 │ warm_start
            ┌────┴────┐
            │   n3    │ cmaes (escape local min)
            └─────────┘
    """

    # Identity
    graph_id: int
    problem_id: int  # v0.4.7: Changed from str to int for type consistency
    created_at: str                 # ISO timestamp
    goal: Optional[str] = None      # Natural language optimization goal

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Graph structure
    nodes: Dict[str, OptimizationNode] = field(default_factory=dict)
    edges: List[OptimizationEdge] = field(default_factory=list)

    # Outcome
    # v0.4.8: Simplified status (active/completed), removed success flag
    # Quality judgment is NOT encoded - agent reasons from final_objective
    status: Literal["active", "completed"] = "active"
    final_objective: Optional[float] = None
    final_x: Optional[List[float]] = None  # Final solution vector
    total_evaluations: int = 0
    total_wall_time: float = 0.0

    # Agent decisions
    decisions: List[GraphDecision] = field(default_factory=list)

    # =========================================================================
    # Node Operations
    # =========================================================================

    def add_node(self, node: OptimizationNode) -> None:
        """Add a node to the graph."""
        if node.node_id in self.nodes:
            raise ValueError(f"Node {node.node_id} already exists in graph")
        self.nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[OptimizationNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def get_best_node(self) -> Optional[OptimizationNode]:
        """Get node with best (minimum) objective value."""
        completed = [
            n for n in self.nodes.values()
            if n.is_successful()
        ]
        if not completed:
            return None
        return min(completed, key=lambda n: n.best_objective)

    def get_completed_nodes(self) -> List[OptimizationNode]:
        """Get all completed nodes."""
        return [n for n in self.nodes.values() if n.is_completed()]

    # =========================================================================
    # Edge Operations
    # =========================================================================

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OptimizationEdge:
        """
        Add an edge to the graph.

        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Type of relationship (see EdgeType)
            metadata: Optional edge metadata

        Returns:
            The created edge

        Raises:
            ValueError: If source or target node doesn't exist
        """
        if source not in self.nodes:
            raise ValueError(f"Source node {source} not found in graph")
        if target not in self.nodes:
            raise ValueError(f"Target node {target} not found in graph")

        edge = OptimizationEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            metadata=metadata or {},
        )
        self.edges.append(edge)
        return edge

    def get_predecessors(self, node_id: str) -> List[str]:
        """Get all node IDs that have edges pointing to this node."""
        return [e.source for e in self.edges if e.target == node_id]

    def get_successors(self, node_id: str) -> List[str]:
        """Get all node IDs that this node points to."""
        return [e.target for e in self.edges if e.source == node_id]

    def get_incoming_edges(self, node_id: str) -> List[OptimizationEdge]:
        """Get all edges pointing to this node."""
        return [e for e in self.edges if e.target == node_id]

    def get_outgoing_edges(self, node_id: str) -> List[OptimizationEdge]:
        """Get all edges from this node."""
        return [e for e in self.edges if e.source == node_id]

    def get_parent_node(self, node_id: str) -> Optional[OptimizationNode]:
        """
        Get the parent node (first predecessor) for warm-start scenarios.

        For nodes with multiple predecessors, returns the first one.
        """
        predecessors = self.get_predecessors(node_id)
        if predecessors:
            return self.nodes.get(predecessors[0])
        return None

    # =========================================================================
    # Graph Analysis
    # =========================================================================

    def get_root_nodes(self) -> List[str]:
        """Get node IDs with no incoming edges (entry points)."""
        targets = {e.target for e in self.edges}
        return [nid for nid in self.nodes if nid not in targets]

    def get_leaf_nodes(self) -> List[str]:
        """Get node IDs with no outgoing edges (terminal nodes)."""
        sources = {e.source for e in self.edges}
        return [nid for nid in self.nodes if nid not in sources]

    def detect_pattern(self) -> str:
        """
        Detect the structural pattern of the graph.

        Returns:
            "empty"      - No nodes
            "single"     - Only one node
            "multistart" - Multiple roots, no edges (parallel independent runs)
            "chain"      - Linear sequence (A → B → C)
            "tree"       - One root, branching (A → {B, C})
            "dag"        - General directed acyclic graph
        """
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)
        roots = self.get_root_nodes()

        if n_nodes == 0:
            return "empty"
        if n_nodes == 1:
            return "single"
        if n_edges == 0:
            return "multistart"

        # Check for chain: single root, each node has at most one successor
        if len(roots) == 1:
            max_out_degree = max(
                len(self.get_successors(nid))
                for nid in self.nodes
            )
            if max_out_degree <= 1:
                return "chain"
            else:
                return "tree"

        return "dag"

    def topological_sort(self) -> List[str]:
        """
        Return node IDs in topological order (respects edge dependencies).

        Useful for processing nodes in execution order.

        Returns:
            List of node IDs in topological order
        """
        # Kahn's algorithm
        in_degree = {nid: 0 for nid in self.nodes}
        for edge in self.edges:
            in_degree[edge.target] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for successor in self.get_successors(node):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        # Check for cycles (shouldn't happen in valid optimization graphs)
        if len(result) != len(self.nodes):
            raise ValueError("Graph contains a cycle - invalid optimization graph")

        return result

    def get_node_depth(self, node_id: str) -> int:
        """
        Get the depth of a node (longest path from any root).

        Returns:
            Depth (0 for root nodes)
        """
        predecessors = self.get_predecessors(node_id)
        if not predecessors:
            return 0
        return 1 + max(self.get_node_depth(p) for p in predecessors)

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "graph_id": self.graph_id,
            "problem_id": self.problem_id,
            "created_at": self.created_at,
            "goal": self.goal,
            "config": self.config,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "status": self.status,
            # v0.4.8: removed success flag - quality judgment not in schema
            "final_objective": self.final_objective,
            "final_x": self.final_x,
            "total_evaluations": self.total_evaluations,
            "total_wall_time": self.total_wall_time,
            "decisions": [d.to_dict() for d in self.decisions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationGraph':
        """Deserialize from dictionary."""
        nodes = {
            nid: OptimizationNode.from_dict(ndata)
            for nid, ndata in data.get("nodes", {}).items()
        }
        edges = [
            OptimizationEdge.from_dict(edata)
            for edata in data.get("edges", [])
        ]
        decisions = [
            GraphDecision.from_dict(ddata)
            for ddata in data.get("decisions", [])
        ]

        # v0.4.8: Map legacy status values to new simplified status
        status = data.get("status", "completed")
        if status == "failed":
            status = "completed"  # Legacy "failed" is now just "completed"

        return cls(
            graph_id=data["graph_id"],
            problem_id=data["problem_id"],
            created_at=data["created_at"],
            goal=data.get("goal"),
            config=data.get("config", {}),
            nodes=nodes,
            edges=edges,
            status=status,
            # v0.4.8: success field removed - ignore legacy values
            final_objective=data.get("final_objective"),
            final_x=data.get("final_x"),
            total_evaluations=data.get("total_evaluations", 0),
            total_wall_time=data.get("total_wall_time", 0.0),
            decisions=decisions,
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'OptimizationGraph':
        """Deserialize from JSON string."""
        import json
        return cls.from_dict(json.loads(json_str))

    # =========================================================================
    # Display Helpers
    # =========================================================================

    def summary(self) -> str:
        """Get a brief summary of the graph."""
        pattern = self.detect_pattern()
        best = self.get_best_node()
        best_str = f"{best.best_objective:.6e}" if best else "N/A"

        title = self.goal if self.goal else self.problem_id
        return (
            f"Graph #{self.graph_id}: {title} | "
            f"{len(self.nodes)} nodes | {pattern} | "
            f"best: {best_str}"
        )

    def __repr__(self) -> str:
        return (
            f"OptimizationGraph(id={self.graph_id}, "
            f"problem={self.problem_id}, "
            f"nodes={len(self.nodes)}, "
            f"edges={len(self.edges)}, "
            f"pattern={self.detect_pattern()})"
        )
