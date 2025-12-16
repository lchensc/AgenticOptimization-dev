# Optimization Graph Architecture

**Document ID**: 20251216_optimization_graph_design
**Date**: December 16, 2025
**Status**: Approved Design
**Version**: 1.0
**Purpose**: Define graph-based data model for PAOLA Foundry

---

## 1. Executive Summary

### 1.1 Core Principle

> **"Graph as substrate, not constraint"**
>
> The optimization graph is the foundational data structure for organizing optimization data.
> It records what happens, not dictates what should happen.
> Like git's DAG: users think in commits/branches, the graph tracks structure.

### 1.2 Key Decisions

| Decision | Details |
|----------|---------|
| Graph is THE data model | Replaces Session/Run list model |
| Node = Optimization Run | One optimizer execution per node |
| Edges are explicit | Relationships are first-class, typed |
| Graph externalizes state | Agent doesn't remember x0, references node IDs |
| Agent makes explicit decisions | Agent specifies which node to warm-start from |
| Experiment later | Test graph-structured prompting when ready |

### 1.3 Benefits

1. **Natural representation** - Optimization strategies ARE graphs
2. **Flexible** - Supports chain, tree, multi-start, portfolio patterns
3. **Queryable history** - "What was tried from this point?"
4. **Foundation for learning** - Graph patterns → knowledge base
5. **Future-proof** - Can expose to LLM when beneficial

---

## 2. Schema Design

### 2.1 Core Types

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from datetime import datetime

# =============================================================================
# Node: Single Optimization Execution
# =============================================================================

@dataclass
class OptimizationNode:
    """
    A single optimization execution in the graph.

    Equivalent to the previous "Run" concept, but with explicit graph identity.
    """

    # === Identity ===
    node_id: str                    # Unique within graph (e.g., "n1", "n2")

    # === Optimizer Configuration ===
    optimizer: str                  # Full spec: "scipy:SLSQP", "optuna:TPE"
    optimizer_family: str           # Family: "gradient", "bayesian", etc.
    config: Dict[str, Any]          # Optimizer-specific configuration

    # === Execution State ===
    status: Literal["pending", "running", "completed", "failed"]
    created_at: str                 # ISO timestamp
    completed_at: Optional[str]     # ISO timestamp when finished

    # === Results (populated when completed) ===
    n_evaluations: int = 0
    wall_time: float = 0.0
    best_objective: Optional[float] = None
    best_design: Optional[List[float]] = None

    # === Polymorphic Components (per optimizer family) ===
    initialization: Optional['InitializationComponent'] = None
    progress: Optional['ProgressComponent'] = None
    result: Optional['ResultComponent'] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
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
            "best_design": self.best_design,
            "initialization": self.initialization.to_dict() if self.initialization else None,
            "progress": self.progress.to_dict() if self.progress else None,
            "result": self.result.to_dict() if self.result else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationNode':
        """Deserialize from dictionary."""
        # Deserialize polymorphic components using registry
        family = data["optimizer_family"]

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
                result_data=result_data
            )

        return cls(
            node_id=data["node_id"],
            optimizer=data["optimizer"],
            optimizer_family=data["optimizer_family"],
            config=data.get("config", {}),
            status=data["status"],
            created_at=data["created_at"],
            completed_at=data.get("completed_at"),
            n_evaluations=data.get("n_evaluations", 0),
            wall_time=data.get("wall_time", 0.0),
            best_objective=data.get("best_objective"),
            best_design=data.get("best_design"),
            initialization=initialization,
            progress=progress,
            result=result,
        )


# =============================================================================
# Edge: Relationship Between Nodes
# =============================================================================

@dataclass
class OptimizationEdge:
    """
    Directed relationship between optimization nodes.

    Edge types capture the semantic meaning of the relationship.
    """

    source: str                     # Source node_id
    target: str                     # Target node_id
    edge_type: str                  # Relationship type (see EdgeType)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Edge-specific data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationEdge':
        return cls(
            source=data["source"],
            target=data["target"],
            edge_type=data["edge_type"],
            metadata=data.get("metadata", {}),
        )


class EdgeType:
    """
    Standard edge types for optimization graphs.

    These capture the semantic meaning of relationships between nodes.
    """

    # === Primary Types ===
    WARM_START = "warm_start"       # Use source's best solution as target's x0
    RESTART = "restart"             # Same config as source, new random seed
    REFINE = "refine"               # Local refinement from source's solution

    # === Exploration Types ===
    BRANCH = "branch"               # Explore different direction from source
    EXPLORE = "explore"             # Global exploration seeded by source

    # === Combination Types ===
    MERGE = "merge"                 # Target combines results from multiple sources
    SELECT = "select"               # Target selected from multiple candidates


# =============================================================================
# Graph: Complete Optimization Task
# =============================================================================

@dataclass
class OptimizationGraph:
    """
    Graph-based representation of an optimization task.

    This is THE data model for optimization in PAOLA.
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

    # === Identity ===
    graph_id: int                   # Unique identifier
    problem_id: str                 # Problem being solved
    created_at: str                 # ISO timestamp

    # === Configuration ===
    config: Dict[str, Any] = field(default_factory=dict)  # Goal, constraints, etc.

    # === Graph Structure ===
    nodes: Dict[str, OptimizationNode] = field(default_factory=dict)
    edges: List[OptimizationEdge] = field(default_factory=list)

    # === Outcome ===
    status: Literal["active", "completed", "failed"] = "active"
    success: bool = False
    final_objective: Optional[float] = None
    final_design: Optional[List[float]] = None
    total_evaluations: int = 0
    total_wall_time: float = 0.0

    # === Agent Decisions ===
    decisions: List['PaolaDecision'] = field(default_factory=list)

    # =========================================================================
    # Node Operations
    # =========================================================================

    def add_node(self, node: OptimizationNode) -> None:
        """Add a node to the graph."""
        if node.node_id in self.nodes:
            raise ValueError(f"Node {node.node_id} already exists")
        self.nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[OptimizationNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def get_best_node(self) -> Optional[OptimizationNode]:
        """Get node with best objective value."""
        completed = [n for n in self.nodes.values()
                     if n.status == "completed" and n.best_objective is not None]
        if not completed:
            return None
        return min(completed, key=lambda n: n.best_objective)

    # =========================================================================
    # Edge Operations
    # =========================================================================

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an edge to the graph."""
        if source not in self.nodes:
            raise ValueError(f"Source node {source} not found")
        if target not in self.nodes:
            raise ValueError(f"Target node {target} not found")

        edge = OptimizationEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            metadata=metadata or {}
        )
        self.edges.append(edge)

    def get_predecessors(self, node_id: str) -> List[str]:
        """Get all nodes that have edges pointing to this node."""
        return [e.source for e in self.edges if e.target == node_id]

    def get_successors(self, node_id: str) -> List[str]:
        """Get all nodes that this node points to."""
        return [e.target for e in self.edges if e.source == node_id]

    def get_incoming_edges(self, node_id: str) -> List[OptimizationEdge]:
        """Get all edges pointing to this node."""
        return [e for e in self.edges if e.target == node_id]

    def get_outgoing_edges(self, node_id: str) -> List[OptimizationEdge]:
        """Get all edges from this node."""
        return [e for e in self.edges if e.source == node_id]

    # =========================================================================
    # Graph Analysis
    # =========================================================================

    def get_root_nodes(self) -> List[str]:
        """Get nodes with no incoming edges (entry points)."""
        targets = {e.target for e in self.edges}
        return [nid for nid in self.nodes if nid not in targets]

    def get_leaf_nodes(self) -> List[str]:
        """Get nodes with no outgoing edges (terminal nodes)."""
        sources = {e.source for e in self.edges}
        return [nid for nid in self.nodes if nid not in sources]

    def detect_pattern(self) -> str:
        """
        Detect the structural pattern of the graph.

        Returns:
            "single"     - Only one node
            "chain"      - Linear sequence (A → B → C)
            "multistart" - Multiple roots, no edges
            "tree"       - One root, multiple branches
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
        if len(roots) == 1 and n_edges == n_nodes - 1:
            # Could be chain or tree
            max_out_degree = max(len(self.get_successors(nid)) for nid in self.nodes)
            if max_out_degree == 1:
                return "chain"
            else:
                return "tree"
        return "dag"

    def topological_sort(self) -> List[str]:
        """
        Return nodes in topological order (respects edge dependencies).

        Useful for processing nodes in execution order.
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

        return result

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "graph_id": self.graph_id,
            "problem_id": self.problem_id,
            "created_at": self.created_at,
            "config": self.config,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "status": self.status,
            "success": self.success,
            "final_objective": self.final_objective,
            "final_design": self.final_design,
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
            PaolaDecision.from_dict(ddata)
            for ddata in data.get("decisions", [])
        ]

        graph = cls(
            graph_id=data["graph_id"],
            problem_id=data["problem_id"],
            created_at=data["created_at"],
            config=data.get("config", {}),
            status=data.get("status", "completed"),
            success=data.get("success", False),
            final_objective=data.get("final_objective"),
            final_design=data.get("final_design"),
            total_evaluations=data.get("total_evaluations", 0),
            total_wall_time=data.get("total_wall_time", 0.0),
        )
        graph.nodes = nodes
        graph.edges = edges
        graph.decisions = decisions

        return graph


# =============================================================================
# Decision Record (unchanged from Session model)
# =============================================================================

@dataclass
class PaolaDecision:
    """
    Record of Paola's strategic decision during optimization.

    Captures why Paola made choices - important for learning.
    """
    timestamp: str
    decision_type: str           # "add_node", "branch", "terminate", etc.
    reasoning: str               # Natural language explanation
    from_node: Optional[str]     # Node ID before decision
    to_node: Optional[str]       # Node ID after decision
    metrics_at_decision: Dict[str, Any]  # Metrics that informed decision

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "decision_type": self.decision_type,
            "reasoning": self.reasoning,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "metrics_at_decision": self.metrics_at_decision,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaolaDecision':
        return cls(**data)
```

### 2.2 Node ID Convention

```python
# Simple auto-increment within graph
"n1", "n2", "n3", ...

# Or semantic naming (future option)
"explore_1", "refine_1", "escape_1", ...
```

**Recommendation**: Start with simple `n1, n2, n3` for clarity.

---

## 3. Storage Design

### 3.1 File Structure

```
.paola_runs/
├── graphs/                      # Renamed from sessions/
│   ├── graph_0001.json
│   ├── graph_0002.json
│   └── graph_0042.json
├── problems/
│   └── ...
└── evaluators/
    └── ...
```

### 3.2 JSON Format

```json
{
  "graph_id": 42,
  "problem_id": "wing_drag",
  "created_at": "2025-12-16T10:30:00",
  "config": {
    "goal": "minimize drag while maintaining lift"
  },

  "nodes": {
    "n1": {
      "node_id": "n1",
      "optimizer": "optuna:TPE",
      "optimizer_family": "bayesian",
      "config": {"n_trials": 50},
      "status": "completed",
      "created_at": "2025-12-16T10:30:00",
      "completed_at": "2025-12-16T10:32:00",
      "n_evaluations": 50,
      "wall_time": 120.0,
      "best_objective": 0.15,
      "best_design": [0.1, 0.2, 0.3],
      "initialization": {"family": "bayesian", "...": "..."},
      "progress": {"family": "bayesian", "trials": ["..."]},
      "result": {"family": "bayesian", "...": "..."}
    },
    "n2": {
      "node_id": "n2",
      "optimizer": "scipy:SLSQP",
      "optimizer_family": "gradient",
      "config": {},
      "status": "completed",
      "...": "..."
    },
    "n3": {
      "node_id": "n3",
      "optimizer": "cmaes",
      "optimizer_family": "cmaes",
      "...": "..."
    }
  },

  "edges": [
    {"source": "n1", "target": "n2", "edge_type": "warm_start", "metadata": {}},
    {"source": "n2", "target": "n3", "edge_type": "warm_start", "metadata": {}}
  ],

  "status": "completed",
  "success": true,
  "final_objective": 0.05,
  "final_design": [0.11, 0.19, 0.28],
  "total_evaluations": 100,
  "total_wall_time": 195.8,

  "decisions": [
    {
      "timestamp": "2025-12-16T10:32:00",
      "decision_type": "add_node",
      "reasoning": "Optuna found promising region, switching to gradient refinement",
      "from_node": "n1",
      "to_node": "n2",
      "metrics_at_decision": {"best_obj": 0.15, "n_trials": 50}
    }
  ]
}
```

---

## 4. Foundry API

### 4.1 OptimizationFoundry Updates

```python
class OptimizationFoundry:
    """
    Data foundation for optimization graphs.

    v0.3.0: Graph-based architecture (replaces Session model)
    """

    # === Graph Lifecycle ===

    def create_graph(
        self,
        problem_id: str,
        config: Optional[Dict] = None,
    ) -> ActiveGraph:
        """Create new optimization graph."""
        ...

    def get_graph(self, graph_id: int) -> Optional[ActiveGraph]:
        """Get active graph by ID."""
        ...

    def finalize_graph(self, graph_id: int, success: bool) -> Optional[OptimizationGraph]:
        """Finalize graph and persist to storage."""
        ...

    # === Storage Queries ===

    def load_graph(self, graph_id: int) -> Optional[OptimizationGraph]:
        """Load completed graph from storage."""
        ...

    def load_all_graphs(self) -> List[OptimizationGraph]:
        """Load all graphs from storage."""
        ...

    # === Convenience Methods ===

    def get_node(self, graph_id: int, node_id: str) -> Optional[OptimizationNode]:
        """Get specific node from graph."""
        graph = self.load_graph(graph_id)
        if graph:
            return graph.get_node(node_id)
        return None
```

### 4.2 ActiveGraph (In-Memory Handle)

```python
@dataclass
class ActiveGraph:
    """
    Handle for in-progress optimization graph.

    Manages nodes as they are created and completed.
    """

    graph_id: int
    problem_id: str
    config: Dict[str, Any]

    nodes: Dict[str, OptimizationNode]
    edges: List[OptimizationEdge]
    decisions: List[PaolaDecision]

    current_node: Optional[ActiveNode] = None
    start_time: datetime = field(default_factory=datetime.now)

    _next_node_num: int = 1

    def create_node(
        self,
        optimizer: str,
        config: Optional[Dict] = None,
        parent_node: Optional[str] = None,
        edge_type: str = EdgeType.WARM_START,
    ) -> 'ActiveNode':
        """
        Create a new optimization node.

        If parent_node is specified, creates an edge from parent to new node.

        Args:
            optimizer: Optimizer spec (e.g., "scipy:SLSQP")
            config: Optimizer configuration
            parent_node: Optional parent node to connect from
            edge_type: Type of edge if parent specified

        Returns:
            ActiveNode handle for the new node
        """
        node_id = f"n{self._next_node_num}"
        self._next_node_num += 1

        family = COMPONENT_REGISTRY.get_family(optimizer)

        node = OptimizationNode(
            node_id=node_id,
            optimizer=optimizer,
            optimizer_family=family,
            config=config or {},
            status="pending",
            created_at=datetime.now().isoformat(),
        )

        self.nodes[node_id] = node

        # Add edge if parent specified
        if parent_node:
            self.add_edge(parent_node, node_id, edge_type)

        self.current_node = ActiveNode(node=node, graph=self)
        return self.current_node

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Add an edge between nodes."""
        edge = OptimizationEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            metadata=metadata or {},
        )
        self.edges.append(edge)

    def record_decision(
        self,
        decision_type: str,
        reasoning: str,
        from_node: Optional[str] = None,
        to_node: Optional[str] = None,
        metrics: Optional[Dict] = None,
    ) -> None:
        """Record an agent decision."""
        decision = PaolaDecision(
            timestamp=datetime.now().isoformat(),
            decision_type=decision_type,
            reasoning=reasoning,
            from_node=from_node,
            to_node=to_node,
            metrics_at_decision=metrics or {},
        )
        self.decisions.append(decision)

    def finalize(self, success: bool) -> OptimizationGraph:
        """Finalize graph and return immutable record."""
        # Compute aggregates
        total_evals = sum(n.n_evaluations for n in self.nodes.values())
        total_time = (datetime.now() - self.start_time).total_seconds()

        # Find best node
        best_node = None
        best_obj = float('inf')
        for node in self.nodes.values():
            if node.status == "completed" and node.best_objective is not None:
                if node.best_objective < best_obj:
                    best_obj = node.best_objective
                    best_node = node

        return OptimizationGraph(
            graph_id=self.graph_id,
            problem_id=self.problem_id,
            created_at=self.start_time.isoformat(),
            config=self.config,
            nodes=self.nodes,
            edges=self.edges,
            status="completed",
            success=success,
            final_objective=best_node.best_objective if best_node else None,
            final_design=best_node.best_design if best_node else None,
            total_evaluations=total_evals,
            total_wall_time=total_time,
            decisions=self.decisions,
        )
```

---

## 5. Tool API (Explicit Graph Operations)

### 5.1 Design Principle

> **Graph externalizes state, agent makes decisions.**

The graph helps the agent by:
- **Tracking state** - Agent doesn't need to remember x0 values, just node IDs
- **Recording history** - What was tried, from where
- **Enabling explicit decisions** - Agent specifies exactly which node to continue from

The agent must **explicitly specify** the parent node when warm-starting. The system does NOT automatically select "best" - that's the agent's decision.

```python
# Agent queries graph state:
get_graph_info(graph_id=1)
# Returns: nodes with their objectives, structure

# Agent decides (explicit):
"Node n2 has the best result (0.08). I'll warm-start SLSQP from n2."

# Agent specifies exactly:
run_optimization(graph_id=1, optimizer="scipy:SLSQP", parent_node="n2")
# System creates node n4, adds edge n2→n4
```

### 5.2 Tool Interface

```python
@tool
def run_optimization(
    graph_id: int,
    optimizer: str,
    config: Optional[str] = None,
    init_strategy: str = "center",
    parent_node: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute optimization within a graph.

    Creates a new node, optionally connected to a parent node.

    Args:
        graph_id: Active graph ID
        optimizer: Optimizer spec (e.g., "scipy:SLSQP", "optuna:TPE")
        config: JSON string with optimizer options
        init_strategy: "center", "random", or "warm_start"
        parent_node: Node ID to warm-start from (required if init_strategy="warm_start")

    Returns:
        node_id: ID of the created node
        best_objective: Best value found
        best_design: Best solution found
        n_evaluations: Number of function evaluations

    Example:
        # Fresh start
        run_optimization(graph_id=1, optimizer="optuna:TPE", init_strategy="random")

        # Warm-start from previous node
        run_optimization(
            graph_id=1,
            optimizer="scipy:SLSQP",
            init_strategy="warm_start",
            parent_node="n1"
        )
    """
    ...

@tool
def start_graph(problem_id: str, goal: str) -> Dict[str, Any]:
    """
    Start a new optimization graph.

    Args:
        problem_id: Problem to optimize
        goal: Natural language description of optimization goal

    Returns:
        graph_id: ID of the created graph
    """
    ...

@tool
def finalize_graph(graph_id: int, success: bool, notes: str = "") -> Dict[str, Any]:
    """
    Finalize an optimization graph.

    Args:
        graph_id: Graph to finalize
        success: Whether optimization achieved its goal
        notes: Optional notes about the outcome

    Returns:
        final_objective: Best objective found
        total_evaluations: Total function evaluations
        n_nodes: Number of optimization runs
    """
    ...

@tool
def get_graph_info(graph_id: int) -> Dict[str, Any]:
    """
    Get information about an optimization graph.

    Returns:
        status: Current graph status
        nodes: List of nodes with their status and results
        best_node: ID of node with best objective
        best_objective: Best objective value found
        pattern: Detected graph pattern (chain, tree, etc.)
    """
    ...
```

### 5.3 Agent-Graph Interaction

```
Agent: "I'll optimize the Ackley function with multi-start L-BFGS-B"

→ start_graph(problem_id="ackley_10d", goal="minimize with multi-start")
← graph_id: 1

→ run_optimization(graph_id=1, optimizer="scipy:L-BFGS-B", init_strategy="random")
← node_id: "n1", best_objective: 6.63

→ run_optimization(graph_id=1, optimizer="scipy:L-BFGS-B", init_strategy="random")
← node_id: "n2", best_objective: 4.03

→ run_optimization(graph_id=1, optimizer="scipy:L-BFGS-B", init_strategy="random")
← node_id: "n3", best_objective: 5.21

→ get_graph_info(graph_id=1)
← nodes: [
    {id: "n1", optimizer: "scipy:L-BFGS-B", best_objective: 6.63},
    {id: "n2", optimizer: "scipy:L-BFGS-B", best_objective: 4.03},
    {id: "n3", optimizer: "scipy:L-BFGS-B", best_objective: 5.21},
  ]
  pattern: "multistart"

Agent reasons: "Looking at the graph info:
  - n1: 6.63 (worst)
  - n2: 4.03 (best)
  - n3: 5.21
  I'll warm-start SLSQP from n2 since it found the best region."

→ run_optimization(graph_id=1, optimizer="scipy:SLSQP", parent_node="n2")
← node_id: "n4", best_objective: 3.98, parent: "n2"

→ finalize_graph(graph_id=1, success=True, notes="Multi-start + refinement achieved 3.98")
```

**Key observations**:
1. Agent queries graph state via `get_graph_info()`
2. Agent **explicitly decides** which node to continue from (n2)
3. Agent doesn't need to remember x0 values - just references node ID
4. Graph records the decision (edge n2→n4)

---

## 6. CLI Design

### 6.1 Command Mapping

| Command | Description |
|---------|-------------|
| `/graphs` | List all optimization graphs |
| `/show <id>` | Show graph overview with structure |
| `/show <id>.<node>` | Show specific node details |
| `/plot <id>` | Plot graph convergence |
| `/plot <id>.<node>` | Plot single node |
| `/graph <id>` | ASCII visualization of graph structure |

### 6.2 ID Notation

```
<graph_id>.<node_id>

Examples:
  42        → Graph 42
  42.n1     → Node n1 of Graph 42
  42.n2     → Node n2 of Graph 42
```

### 6.3 Display Examples

**`/graphs` - List all graphs**

```
╭──────────────────────────────────────────────────────────────────╮
│                     Optimization Graphs                          │
├──────┬─────────────────────┬───────┬────────┬───────────┬───────┤
│  ID  │ Problem             │ Nodes │ Pattern│ Best      │ Evals │
├──────┼─────────────────────┼───────┼────────┼───────────┼───────┤
│   1  │ rosenbrock_10d      │   1   │ single │ 1.23e-06  │    45 │
│   6  │ ackley_10d          │   5   │ multi  │ 4.03e+00  │    98 │
│  42  │ wing_drag           │   3   │ chain  │ 5.00e-02  │   100 │
╰──────┴─────────────────────┴───────┴────────┴───────────┴───────╯
```

**`/show 42` - Graph overview**

```
╭─────────────────────────────────────────────────────────────────╮
│ Graph #42: wing_drag                                            │
│━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
│                                                                 │
│ Status:       ✓ Complete                                        │
│ Pattern:      Chain (sequential warm-start)                     │
│ Best:         5.00e-02 (node n3)                                │
│ Total Evals:  100                                               │
│ Total Time:   195.8s                                            │
│                                                                 │
│ Structure:                                                      │
│                                                                 │
│     [n1] optuna:TPE                                             │
│      │   50 evals → 0.15                                        │
│      │                                                          │
│      ↓ warm_start                                               │
│                                                                 │
│     [n2] scipy:SLSQP                                            │
│      │   30 evals → 0.08                                        │
│      │                                                          │
│      ↓ warm_start                                               │
│                                                                 │
│     [n3] cmaes                    ★ best                        │
│          20 evals → 0.05                                        │
│                                                                 │
│ Decisions:                                                      │
│   n1→n2: "Found promising region, switch to gradient"           │
│   n2→n3: "Stuck at local min, try CMA-ES"                       │
│                                                                 │
│ Tip: /show 42.n2 for node details, /plot 42 for convergence    │
╰─────────────────────────────────────────────────────────────────╯
```

**`/show 42.n2` - Node detail**

```
╭─────────────────────────────────────────────────────────────────╮
│ Node n2 of Graph #42                                            │
│━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
│                                                                 │
│ Optimizer:    scipy:SLSQP (gradient family)                     │
│ Status:       ✓ Completed                                       │
│                                                                 │
│ Connections:                                                    │
│   ← n1 (warm_start)                                             │
│   → n3 (warm_start)                                             │
│                                                                 │
│ Performance:                                                    │
│   Best Objective:  8.00e-02                                     │
│   Evaluations:     30                                           │
│   Wall Time:       45.2s                                        │
│                                                                 │
│ Initialization:                                                 │
│   Strategy:  warm_start from n1                                 │
│   x0: [0.10, 0.20, 0.30, ...]                                   │
│                                                                 │
│ Convergence:                                                    │
│   Iterations:  28                                               │
│   Initial:     1.50e-01                                         │
│   Final:       8.00e-02                                         │
│   Improvement: 46.7%                                            │
│   Termination: gradient_norm < tol                              │
│                                                                 │
│ Navigation: /plot 42.n2 | /show 42 (back to graph)             │
╰─────────────────────────────────────────────────────────────────╯
```

**`/graph 42` - ASCII structure**

```
╭─────────────────────────────────────────────────────────────────╮
│ Graph Structure: #42 wing_drag                                  │
│                                                                 │
│         ┌───────────────┐                                       │
│         │      n1       │                                       │
│         │  optuna:TPE   │                                       │
│         │   → 0.15      │                                       │
│         └───────┬───────┘                                       │
│                 │                                               │
│                 │ warm_start                                    │
│                 ▼                                               │
│         ┌───────────────┐                                       │
│         │      n2       │                                       │
│         │ scipy:SLSQP   │                                       │
│         │   → 0.08      │                                       │
│         └───────┬───────┘                                       │
│                 │                                               │
│                 │ warm_start                                    │
│                 ▼                                               │
│         ┌───────────────┐                                       │
│         │      n3       │  ★                                    │
│         │    cmaes      │                                       │
│         │   → 0.05      │                                       │
│         └───────────────┘                                       │
│                                                                 │
│ Legend: ★ = best result                                         │
╰─────────────────────────────────────────────────────────────────╯
```

---

## 7. Migration Strategy

### 7.1 Session → Graph Conversion

```python
def migrate_session_to_graph(session: SessionRecord) -> OptimizationGraph:
    """
    Convert v0.2.0 SessionRecord to v0.3.0 OptimizationGraph.

    Mapping:
    - session_id → graph_id
    - runs[i] → nodes["n{i+1}"]
    - warm_start_from → edge with type "warm_start"
    """
    graph = OptimizationGraph(
        graph_id=session.session_id,
        problem_id=session.problem_id,
        created_at=session.created_at,
        config=session.config,
        status="completed",
        success=session.success,
        final_objective=session.final_objective,
        final_design=session.final_design,
        total_evaluations=session.total_evaluations,
        total_wall_time=session.total_wall_time,
    )

    # Convert runs to nodes
    for run in session.runs:
        node = OptimizationNode(
            node_id=f"n{run.run_id}",
            optimizer=run.optimizer,
            optimizer_family=run.optimizer_family,
            config={},
            status="completed",
            created_at=session.created_at,
            n_evaluations=run.n_evaluations,
            wall_time=run.wall_time,
            best_objective=run.best_objective,
            best_design=run.best_design,
            initialization=run.initialization,
            progress=run.progress,
            result=run.result,
        )
        graph.nodes[node.node_id] = node

        # Convert warm_start_from to edge
        if run.warm_start_from is not None:
            edge = OptimizationEdge(
                source=f"n{run.warm_start_from}",
                target=f"n{run.run_id}",
                edge_type=EdgeType.WARM_START,
            )
            graph.edges.append(edge)

    # Convert decisions
    for d in session.decisions:
        decision = PaolaDecision(
            timestamp=d.timestamp,
            decision_type=d.decision_type,
            reasoning=d.reasoning,
            from_node=f"n{d.from_run}" if d.from_run else None,
            to_node=f"n{d.to_run}" if d.to_run else None,
            metrics_at_decision=d.metrics_at_decision,
        )
        graph.decisions.append(decision)

    return graph
```

### 7.2 Migration Command

```python
def migrate_storage(storage: StorageBackend):
    """Migrate all sessions to graphs."""
    sessions_dir = storage.base_dir / "sessions"
    graphs_dir = storage.base_dir / "graphs"

    graphs_dir.mkdir(exist_ok=True)

    for session_file in sessions_dir.glob("session_*.json"):
        session = storage.load_session_from_file(session_file)
        graph = migrate_session_to_graph(session)
        storage.save_graph(graph)

    # Optionally archive old sessions
    archive_dir = storage.base_dir / "archive" / "sessions_v0.2"
    shutil.move(sessions_dir, archive_dir)
```

---

## 8. Terminology Update

| v0.2.0 Term | v0.3.0 Term | Notes |
|-------------|-------------|-------|
| Session | Graph | Container for optimization task |
| Run | Node | Single optimizer execution |
| session_id | graph_id | Unique identifier |
| run_id | node_id | Identifier within graph |
| warm_start_from | Edge (warm_start) | Explicit relationship |
| `/sessions` | `/graphs` | CLI command |
| `/show 6.2` | `/show 6.n2` | Node notation |

---

## 9. Implementation Plan

### Priority Order (User Decision)
1. **Foundry schema + storage first** - Core data model
2. Tools second - Agent interface
3. CLI third - User interface
4. Documentation last

### Phase 1: Schema (Foundation)

**Files to create:**
- `paola/foundry/schema/graph.py` - OptimizationGraph, OptimizationNode, OptimizationEdge

**Tasks:**
- [ ] Create `OptimizationNode` dataclass
- [ ] Create `OptimizationEdge` dataclass
- [ ] Create `EdgeType` constants
- [ ] Create `OptimizationGraph` dataclass with graph operations
- [ ] Implement `to_dict()` / `from_dict()` serialization
- [ ] Implement pattern detection (`detect_pattern()`)
- [ ] Implement graph traversal (`get_predecessors()`, `get_successors()`, `topological_sort()`)
- [ ] Unit tests: `tests/test_graph_schema.py`

### Phase 2: Active Graph + Storage

**Files to create/modify:**
- `paola/foundry/active_graph.py` - ActiveGraph, ActiveNode
- `paola/foundry/storage/file_storage.py` - Add graph methods

**Tasks:**
- [ ] Create `ActiveGraph` class (in-memory handle)
- [ ] Create `ActiveNode` class (run handle)
- [ ] Implement `create_node()`, `add_edge()`, `record_decision()`, `finalize()`
- [ ] Add `StorageBackend.save_graph()`, `load_graph()`, `load_all_graphs()`
- [ ] Implement `FileStorage` graph methods
- [ ] Create migration utility `migrate_session_to_graph()`
- [ ] Unit tests: `tests/test_active_graph.py`, `tests/test_graph_storage.py`

### Phase 3: Foundry API

**Files to modify:**
- `paola/foundry/foundry.py` - Update API

**Tasks:**
- [ ] Add `create_graph()` method
- [ ] Add `get_graph()` method (active graphs)
- [ ] Add `finalize_graph()` method
- [ ] Add `load_graph()`, `load_all_graphs()` methods
- [ ] Add `get_node()` convenience method
- [ ] Deprecate session methods (warnings)
- [ ] Unit tests: `tests/test_foundry_graph.py`

### Phase 4: Tools

**Files to modify:**
- `paola/tools/session_tools.py` → rename to `graph_tools.py`
- `paola/tools/optimization_tools.py` - Update for graphs

**Tasks:**
- [ ] Rename `start_session` → `start_graph`
- [ ] Rename `finalize_session` → `finalize_graph`
- [ ] Rename `get_session_info` → `get_graph_info`
- [ ] Update `run_optimization()` with explicit `parent_node` parameter
- [ ] Update tool docstrings for agent understanding
- [ ] Integration tests: `tests/test_graph_tools.py`

### Phase 5: CLI

**Files to modify:**
- `paola/cli/repl.py` - Command routing
- `paola/cli/commands.py` - Display logic

**Tasks:**
- [ ] Replace `/sessions` → `/graphs`
- [ ] Update `/show` for graph display with structure visualization
- [ ] Update `/show <id>.<node>` for node detail
- [ ] Add `/graph <id>` for ASCII visualization
- [ ] Update `/plot` for pattern-aware plotting
- [ ] Update ID parsing for `graph_id.node_id` notation
- [ ] CLI tests

### Phase 6: Cleanup & Documentation

**Tasks:**
- [ ] Run migration on existing sessions
- [ ] Update CLAUDE.md
- [ ] Update CLI help text
- [ ] Archive old session code
- [ ] Final integration testing

---

## 10. Future Extensions

### 10.1 Expose Graph to LLM (Experiment)

When ready to test if graph-structured thinking helps:

```python
@tool
def get_graph_structure(graph_id: int) -> Dict[str, Any]:
    """
    Get the full graph structure for planning.

    Returns nodes, edges, and patterns to help plan next steps.
    """
    ...

@tool
def plan_graph_extension(
    graph_id: int,
    strategy: str,  # "branch", "refine", "multistart"
) -> Dict[str, Any]:
    """
    Plan graph extension based on strategy.

    Returns suggested nodes and edges to add.
    """
    ...
```

### 10.2 Design-Level Nodes (Domain-Specific)

For aerodynamic optimization:
```python
@dataclass
class DesignNode:
    """A design point in the optimization graph."""
    node_id: str
    design: List[float]
    evaluations: List[EvaluationResult]  # Multiple evals per design
    derived_from: Optional[str]  # Parent design
```

### 10.3 Graph Templates

```python
GRAPH_TEMPLATES = {
    "global_then_local": {
        "description": "Global exploration followed by local refinement",
        "structure": "n1(bayesian) → n2(gradient)",
    },
    "multistart": {
        "description": "Multiple independent starts",
        "structure": "{n1, n2, n3}(gradient) → select_best",
    },
    "portfolio": {
        "description": "Try multiple optimizers in parallel",
        "structure": "{n1(bayesian), n2(gradient), n3(cmaes)} → select_best",
    },
}
```

---

## 11. Summary

### Core Principle
> **"Graph as substrate, not constraint"**
>
> **"Graph externalizes state, agent makes decisions"**

### Key Decisions
1. **Graph is THE data model** for optimization in PAOLA
2. **Node = Optimization Run** (simple n1, n2, n3 IDs)
3. **Edges are explicit and typed** (warm_start, branch, refine, etc.)
4. **Graph externalizes state** - Agent references node IDs, not variable values
5. **Agent makes explicit decisions** - Agent specifies which node to continue from
6. **Clean break** - `/graphs` replaces `/sessions` (no backward compatibility alias)
7. **Experiment later** with graph-structured prompting

### Benefits
1. Natural representation of optimization strategies
2. Flexible: supports all patterns (chain, tree, multistart, dag)
3. Queryable history and structure
4. Agent doesn't need to remember x0 values - just node IDs
5. Foundation for learning (graph patterns → knowledge)
6. Future-proof for LLM experimentation

### Implementation Priority
1. **Schema + Storage first** (Phase 1-2)
2. Foundry API (Phase 3)
3. Tools (Phase 4)
4. CLI (Phase 5)
5. Documentation (Phase 6)
