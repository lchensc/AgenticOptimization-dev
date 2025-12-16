"""
Active graph and node management for in-progress optimizations.

ActiveGraph: Handle for in-progress optimization graph
ActiveNode: Handle for in-progress single optimizer execution (node)

v0.3.0: Graph-based architecture
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional

from .schema import (
    OptimizationGraph,
    OptimizationNode,
    OptimizationEdge,
    GraphDecision,
    EdgeType,
    InitializationComponent,
    ProgressComponent,
    ResultComponent,
    COMPONENT_REGISTRY,
)


@dataclass
class ActiveNode:
    """
    Handle for in-progress single optimizer execution (node).

    Collects iteration data during optimization and finalizes to OptimizationNode.
    """

    node_id: str
    optimizer: str
    optimizer_family: str
    config: Dict[str, Any]
    initialization: InitializationComponent
    parent_node: Optional[str] = None
    edge_type: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    raw_iterations: List[Dict[str, Any]] = field(default_factory=list)
    _best_objective: float = field(default=float('inf'))
    _best_x: Optional[List[float]] = None

    def record_iteration(self, data: Dict[str, Any]):
        """
        Record an iteration/trial/generation.

        Args:
            data: Iteration data dict. Expected keys depend on family:
                - gradient: iteration, objective, x, gradient_norm, step_size
                - bayesian: trial_number, x, objective, state
                - population: generation, best_objective, best_x, mean_objective
                - cmaes: generation, best_objective, best_x, mean, sigma
        """
        self.raw_iterations.append(data)

        # Track best
        obj = data.get("objective") or data.get("best_objective")
        if obj is not None and obj < self._best_objective:
            self._best_objective = obj
            self._best_x = data.get("x") or data.get("best_x")

    def get_best(self) -> Dict[str, Any]:
        """Get current best objective and x."""
        return {
            "objective": self._best_objective,
            "x": self._best_x,
        }

    def finalize(
        self,
        progress: ProgressComponent,
        result: ResultComponent,
        best_objective: float,
        best_x: List[float],
        success: bool = True,
    ) -> OptimizationNode:
        """
        Finalize node and return immutable OptimizationNode.

        Args:
            progress: Family-specific progress component with iteration data
            result: Family-specific result component
            best_objective: Final best objective value
            best_x: Final best x vector
            success: Whether optimization succeeded

        Returns:
            Immutable OptimizationNode record
        """
        wall_time = (datetime.now() - self.start_time).total_seconds()

        return OptimizationNode(
            node_id=self.node_id,
            optimizer=self.optimizer,
            optimizer_family=self.optimizer_family,
            config=self.config,
            status="completed" if success else "failed",
            best_objective=best_objective,
            best_x=best_x,
            n_evaluations=len(self.raw_iterations),
            wall_time=wall_time,
            initialization=self.initialization,
            progress=progress,
            result=result,
        )


class ActiveGraph:
    """
    Handle for in-progress optimization graph.

    Manages nodes as they are created and completed.
    Tracks agent's strategic decisions.
    """

    def __init__(
        self,
        graph_id: int,
        problem_id: str,
        goal: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.graph_id = graph_id
        self.problem_id = problem_id
        self.goal = goal
        self.config = config or {}
        self.nodes: Dict[str, OptimizationNode] = {}
        self.edges: List[OptimizationEdge] = []
        self.decisions: List[GraphDecision] = []
        self.current_node: Optional[ActiveNode] = None
        self.start_time = datetime.now()
        self._next_node_num = 1

    def _generate_node_id(self) -> str:
        """Generate next node ID (n1, n2, n3, ...)."""
        node_id = f"n{self._next_node_num}"
        self._next_node_num += 1
        return node_id

    def start_node(
        self,
        optimizer: str,
        config: Dict[str, Any],
        initialization: InitializationComponent,
        parent_node: Optional[str] = None,
        edge_type: Optional[str] = None,
    ) -> ActiveNode:
        """
        Start a new optimization node.

        Args:
            optimizer: Optimizer spec (e.g., "scipy:SLSQP", "optuna:TPE")
            config: Optimizer configuration
            initialization: Family-specific initialization component
            parent_node: Node ID to continue from, or None for root node
            edge_type: Edge type if parent_node is specified

        Returns:
            ActiveNode handle for recording iterations

        Raises:
            RuntimeError: If another node is already active
            ValueError: If parent_node specified but not found, or edge_type missing
        """
        if self.current_node is not None:
            raise RuntimeError(
                f"Node {self.current_node.node_id} is still active. "
                "Complete it before starting a new node."
            )

        # Validate parent_node if specified
        if parent_node is not None:
            if parent_node not in self.nodes:
                raise ValueError(
                    f"Parent node '{parent_node}' not found. "
                    f"Available nodes: {list(self.nodes.keys())}"
                )
            if edge_type is None:
                raise ValueError(
                    "edge_type must be specified when parent_node is provided"
                )
            if not EdgeType.is_valid(edge_type):
                raise ValueError(
                    f"Invalid edge_type '{edge_type}'. "
                    f"Valid types: {EdgeType.all_types()}"
                )

        node_id = self._generate_node_id()
        family = COMPONENT_REGISTRY.get_family(optimizer)

        self.current_node = ActiveNode(
            node_id=node_id,
            optimizer=optimizer,
            optimizer_family=family,
            config=config,
            initialization=initialization,
            parent_node=parent_node,
            edge_type=edge_type,
        )
        return self.current_node

    def complete_node(
        self,
        progress: ProgressComponent,
        result: ResultComponent,
        best_objective: float,
        best_x: List[float],
        success: bool = True,
    ) -> OptimizationNode:
        """
        Complete current node and add to graph.

        Args:
            progress: Family-specific progress component
            result: Family-specific result component
            best_objective: Final best objective
            best_x: Final best x vector
            success: Whether optimization succeeded

        Returns:
            Completed OptimizationNode record
        """
        if self.current_node is None:
            raise RuntimeError("No active node to complete")

        node = self.current_node.finalize(
            progress=progress,
            result=result,
            best_objective=best_objective,
            best_x=best_x,
            success=success,
        )

        # Add node to graph
        self.nodes[node.node_id] = node

        # Add edge if this node has a parent
        if self.current_node.parent_node is not None:
            edge = OptimizationEdge.create(
                source=self.current_node.parent_node,
                target=node.node_id,
                edge_type=self.current_node.edge_type,
            )
            self.edges.append(edge)

        self.current_node = None
        return node

    def record_decision(
        self,
        decision_type: str,
        reasoning: str,
        from_node: Optional[str] = None,
        to_node: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Record agent's strategic decision.

        Args:
            decision_type: Type of decision (start_node, warm_start, branch, terminate)
            reasoning: Agent's reasoning for the decision
            from_node: Source node ID (if applicable)
            to_node: Target node ID (if applicable)
            metrics: Metrics that informed the decision
        """
        decision = GraphDecision(
            timestamp=datetime.now().isoformat(),
            decision_type=decision_type,
            reasoning=reasoning,
            from_node=from_node,
            to_node=to_node,
            metrics_at_decision=metrics or {},
        )
        self.decisions.append(decision)

    def get_node(self, node_id: str) -> Optional[OptimizationNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def get_best_node(self) -> Optional[OptimizationNode]:
        """Get node with best objective so far."""
        if not self.nodes:
            return None
        completed = [n for n in self.nodes.values() if n.status == "completed"]
        if not completed:
            return None
        return min(completed, key=lambda n: n.best_objective or float('inf'))

    def get_leaf_nodes(self) -> List[OptimizationNode]:
        """Get nodes with no successors (potential continuation points)."""
        # Find nodes that are targets of edges
        targets = {e.target for e in self.edges}
        sources = {e.source for e in self.edges}

        # Leaf nodes are nodes that are not sources (no outgoing edges)
        leaf_ids = set(self.nodes.keys()) - sources

        return [self.nodes[nid] for nid in leaf_ids]

    def get_root_nodes(self) -> List[OptimizationNode]:
        """Get nodes with no predecessors (starting points)."""
        targets = {e.target for e in self.edges}
        root_ids = set(self.nodes.keys()) - targets
        return [self.nodes[nid] for nid in root_ids]

    def get_graph_state(self) -> Dict[str, Any]:
        """
        Get current graph state for agent queries.

        Returns dict with:
            - nodes: List of node summaries
            - best_node: Best node summary
            - leaf_nodes: Potential continuation points
            - pattern: Current graph pattern
        """
        node_summaries = []
        for node in self.nodes.values():
            node_summaries.append({
                "node_id": node.node_id,
                "optimizer": node.optimizer,
                "status": node.status,
                "best_objective": node.best_objective,
                "n_evaluations": node.n_evaluations,
            })

        best = self.get_best_node()
        best_summary = None
        if best:
            best_summary = {
                "node_id": best.node_id,
                "optimizer": best.optimizer,
                "best_objective": best.best_objective,
                "best_x": best.best_x,
            }

        leaves = self.get_leaf_nodes()
        leaf_summaries = [
            {
                "node_id": n.node_id,
                "optimizer": n.optimizer,
                "best_objective": n.best_objective,
            }
            for n in leaves
        ]

        # Detect pattern
        pattern = self._detect_pattern()

        return {
            "graph_id": self.graph_id,
            "problem_id": self.problem_id,
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "nodes": node_summaries,
            "best_node": best_summary,
            "leaf_nodes": leaf_summaries,
            "pattern": pattern,
        }

    def _detect_pattern(self) -> str:
        """Detect graph pattern: empty, single, multistart, chain, tree, dag."""
        if not self.nodes:
            return "empty"
        if len(self.nodes) == 1:
            return "single"
        if not self.edges:
            return "multistart"

        # Check for chain (each node has at most one predecessor and one successor)
        roots = self.get_root_nodes()
        if len(roots) == 1:
            # Could be chain or tree
            # Check if any node has multiple successors
            for node_id in self.nodes:
                successors = [e.target for e in self.edges if e.source == node_id]
                if len(successors) > 1:
                    return "tree"
            return "chain"
        else:
            # Multiple roots - could be multistart or dag
            for node_id in self.nodes:
                predecessors = [e.source for e in self.edges if e.target == node_id]
                if len(predecessors) > 1:
                    return "dag"
            return "multistart"

    def finalize(self, success: bool) -> OptimizationGraph:
        """
        Finalize graph and return immutable OptimizationGraph.

        Args:
            success: Whether overall optimization was successful

        Returns:
            Immutable OptimizationGraph record
        """
        if self.current_node is not None:
            raise RuntimeError(
                f"Node {self.current_node.node_id} is still active. "
                "Complete it before finalizing graph."
            )

        # Compute overall metrics
        total_evals = sum(n.n_evaluations or 0 for n in self.nodes.values())
        total_time = (datetime.now() - self.start_time).total_seconds()

        # Get best result
        best_node = self.get_best_node()
        if best_node:
            final_objective = best_node.best_objective
            final_x = best_node.best_x
        else:
            final_objective = None
            final_x = None

        return OptimizationGraph(
            graph_id=self.graph_id,
            problem_id=self.problem_id,
            goal=self.goal,
            created_at=self.start_time.isoformat(),
            nodes=self.nodes,
            edges=self.edges,
            success=success,
            final_objective=final_objective,
            final_x=final_x,
            total_evaluations=total_evals,
            total_wall_time=total_time,
            decisions=self.decisions,
        )
