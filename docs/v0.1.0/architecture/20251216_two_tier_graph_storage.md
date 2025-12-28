# Two-Tier Graph Storage Design

**Date**: 2024-12-16
**Status**: Design

## Problem Statement

Current graph storage is bloated with per-iteration design vectors:
- 50D problem × 200 iterations = 10,000 floats per node
- Files reach 100-300KB for moderate graphs
- LLM needs strategy patterns, not raw trajectories

Missing information that LLM actually needs:
- Full optimizer configuration (tolerances, settings)
- Problem signature (for similarity matching)
- Convergence curves (for learning efficiency patterns)

## Design Decisions

1. **Keep x_history** in Tier 2 (for debugging/visualization)
2. **convergence_history in Tier 2** (LLM can query when needed via future tool)
3. **Full config detail in Tier 1** (LLM observes, reasons, acts, learns from config choices)
4. **Tier 1 focuses on strategy and outcome** - not trajectory details

## Architecture

### Tier 1: GraphRecord (LLM-Ready)

Stored in: `.paola_runs/graphs/graph_XXXX.json`
Size: ~1-2KB per graph (minimal, strategy-focused)

**Focus: "What strategy was used and did it work?"**

```
GraphRecord
├── Identity
│   ├── graph_id: int
│   ├── problem_id: str
│   ├── created_at: str
│   └── goal: Optional[str]
│
├── Problem Signature (NEW - enables similarity matching)
│   ├── n_dimensions: int
│   ├── n_constraints: int
│   ├── bounds_range: [min, max]  # Characteristic scale
│   ├── constraint_types: ["equality", "inequality"]
│   └── domain_hint: Optional[str]  # "rosenbrock", "ackley"
│
├── Structure
│   ├── pattern: str  # "single", "chain", "multistart", "tree", "dag"
│   └── edges: List[{source, target, edge_type}]
│
├── Nodes (Minimal - strategy info only)
│   └── Dict[node_id, NodeSummary]
│       ├── node_id: str
│       ├── optimizer: str  # "scipy:SLSQP"
│       ├── optimizer_family: str
│       ├── config: Dict  # FULL configuration
│       ├── init_strategy: str
│       ├── parent_node: Optional[str]
│       ├── edge_type: Optional[str]
│       ├── status: str
│       ├── n_evaluations: int
│       ├── wall_time: float
│       ├── start_objective: float  # Before optimization
│       └── best_objective: float   # After optimization
│       # NOTE: No convergence_history here - moved to Tier 2
│
├── Outcome
│   ├── status: str
│   ├── success: bool
│   ├── final_objective: float
│   ├── final_x: List[float]  # Just the final solution
│   ├── total_evaluations: int
│   └── total_wall_time: float
│
└── Decisions
    └── List[{timestamp, decision_type, reasoning, from_node, to_node}]
```

### Tier 2: GraphDetail (Debug/Visualization/Deep Analysis)

Stored in: `.paola_runs/details/graph_XXXX_detail.json`
Size: 10-100KB+ (depends on dimensions and iterations)

**Focus: "What happened during execution?"**

```
GraphDetail
├── graph_id: int
└── nodes: Dict[node_id, NodeDetail]
    └── NodeDetail
        ├── node_id: str
        ├── x0: List[float]  # Initial point
        ├── best_x: List[float]  # Best solution
        ├── convergence_history: List[{iter, obj}]  # Objective trajectory
        └── x_history: List[{iter, x}]  # Full design trajectory
```

**Future Tool** (for LLM to query Tier 2 when needed):

```python
@tool
def get_node_detail(graph_id: int, node_id: str) -> Dict:
    """
    Get detailed execution history for a node.

    Use when you need to understand HOW a node converged,
    not just the final result. Useful for:
    - Diagnosing convergence issues
    - Understanding optimizer behavior
    - Comparing convergence rates

    Returns:
        x0: Initial point
        best_x: Best solution found
        convergence_history: [{iter, obj}, ...]
    """
```

## Schema Changes

### New Classes

```python
# paola/foundry/schema/graph_record.py

@dataclass
class ProblemSignature:
    """Problem characteristics for similarity matching."""
    n_dimensions: int
    n_constraints: int
    bounds_range: Tuple[float, float]  # (min_bound, max_bound)
    constraint_types: List[str]  # ["equality", "inequality"]
    domain_hint: Optional[str] = None

@dataclass
class NodeSummary:
    """Minimal node representation for LLM learning.

    Focus: What optimizer with what config achieved what result?
    No trajectory data - that's in Tier 2.
    """
    node_id: str
    optimizer: str
    optimizer_family: str
    config: Dict[str, Any]  # FULL configuration (key for learning!)
    init_strategy: str
    parent_node: Optional[str]
    edge_type: Optional[str]

    status: str
    n_evaluations: int
    wall_time: float
    start_objective: float  # Objective at iteration 1
    best_objective: float   # Best objective achieved
    # NOTE: No convergence_history - moved to Tier 2

@dataclass
class EdgeSummary:
    """Compact edge representation."""
    source: str
    target: str
    edge_type: str

@dataclass
class GraphRecord:
    """Tier 1: LLM-ready graph representation.

    Focus: Strategy and outcome, not trajectory.
    ~1-2KB per graph.
    """
    # Identity
    graph_id: int
    problem_id: str
    created_at: str
    goal: Optional[str]

    # Problem signature (enables similarity matching)
    problem_signature: ProblemSignature

    # Structure
    pattern: str
    edges: List[EdgeSummary]

    # Nodes (minimal)
    nodes: Dict[str, NodeSummary]

    # Outcome
    status: str
    success: bool
    final_objective: Optional[float]
    final_x: Optional[List[float]]
    total_evaluations: int
    total_wall_time: float

    # Decisions (reasoning strings only)
    decisions: List[Dict[str, Any]]
```

```python
# paola/foundry/schema/graph_detail.py

@dataclass
class ConvergencePoint:
    """Single point in convergence history."""
    iteration: int
    objective: float

@dataclass
class XPoint:
    """Single point in x trajectory."""
    iteration: int
    x: List[float]

@dataclass
class NodeDetail:
    """Full execution data for a node.

    Focus: What happened during optimization?
    """
    node_id: str
    x0: List[float]                           # Initial point
    best_x: List[float]                       # Best solution
    convergence_history: List[ConvergencePoint]  # Objective trajectory
    x_history: List[XPoint]                   # Full design trajectory

@dataclass
class GraphDetail:
    """Tier 2: Full trajectory data."""
    graph_id: int
    nodes: Dict[str, NodeDetail]
```

## Conversion

### OptimizationGraph → GraphRecord + GraphDetail

```python
def split_graph(graph: OptimizationGraph, problem_info: ProblemInfo) -> Tuple[GraphRecord, GraphDetail]:
    """Convert full graph to two-tier representation."""

    # Build problem signature from problem_info
    signature = ProblemSignature(
        n_dimensions=problem_info.dimensions,
        n_constraints=len(problem_info.constraints),
        bounds_range=(min(problem_info.bounds), max(problem_info.bounds)),
        constraint_types=[c.type for c in problem_info.constraints],
        domain_hint=problem_info.domain_hint,
    )

    # Build node summaries and details
    node_summaries = {}
    node_details = {}

    for node_id, node in graph.nodes.items():
        # Extract convergence history and x_history for Tier 2
        convergence = []
        x_history = []
        start_objective = None

        if node.progress:
            for iter_data in node.progress.iterations:
                convergence.append(ConvergencePoint(
                    iteration=iter_data.iteration,
                    objective=iter_data.objective,
                ))
                x_history.append(XPoint(
                    iteration=iter_data.iteration,
                    x=iter_data.design,
                ))
            if convergence:
                start_objective = convergence[0].objective

        # Get parent node and edge type from edges
        parent_node = None
        edge_type = None
        for edge in graph.edges:
            if edge.target == node_id:
                parent_node = edge.source
                edge_type = edge.edge_type
                break

        # Node summary (Tier 1) - minimal, no trajectory
        node_summaries[node_id] = NodeSummary(
            node_id=node_id,
            optimizer=node.optimizer,
            optimizer_family=node.optimizer_family,
            config=node.config,  # FULL config - key for learning!
            init_strategy=node.initialization.specification.get("type", "unknown") if node.initialization else "unknown",
            parent_node=parent_node,
            edge_type=edge_type,
            status=node.status,
            n_evaluations=node.n_evaluations,
            wall_time=node.wall_time,
            start_objective=start_objective,
            best_objective=node.best_objective,
            # NOTE: No convergence_history - moved to Tier 2
        )

        # Node detail (Tier 2) - full trajectory
        node_details[node_id] = NodeDetail(
            node_id=node_id,
            x0=node.initialization.x0 if node.initialization else None,
            best_x=node.best_x,
            convergence_history=convergence,  # Moved from Tier 1
            x_history=x_history,
        )

    # Build record (Tier 1)
    record = GraphRecord(
        graph_id=graph.graph_id,
        problem_id=graph.problem_id,
        created_at=graph.created_at,
        goal=graph.goal,
        problem_signature=signature,
        pattern=graph.detect_pattern(),
        edges=[EdgeSummary(e.source, e.target, e.edge_type) for e in graph.edges],
        nodes=node_summaries,
        status=graph.status,
        success=graph.success,
        final_objective=graph.final_objective,
        final_x=graph.final_x,
        total_evaluations=graph.total_evaluations,
        total_wall_time=graph.total_wall_time,
        decisions=[d.to_dict() for d in graph.decisions],
    )

    # Build detail (Tier 2)
    detail = GraphDetail(
        graph_id=graph.graph_id,
        nodes=node_details,
    )

    return record, detail
```

## Storage Changes

### FileStorage Updates

```python
class FileStorage:
    def __init__(self, base_path: Path):
        self.graphs_dir = base_path / "graphs"
        self.details_dir = base_path / "details"  # NEW

    def save_graph(self, graph: OptimizationGraph, problem_info: ProblemInfo):
        """Save graph as two-tier storage."""
        record, detail = split_graph(graph, problem_info)

        # Tier 1: GraphRecord
        record_path = self.graphs_dir / f"graph_{graph.graph_id:04d}.json"
        with open(record_path, 'w') as f:
            json.dump(record.to_dict(), f, indent=2)

        # Tier 2: GraphDetail
        detail_path = self.details_dir / f"graph_{graph.graph_id:04d}_detail.json"
        with open(detail_path, 'w') as f:
            json.dump(detail.to_dict(), f, indent=2)

    def load_graph_record(self, graph_id: int) -> Optional[GraphRecord]:
        """Load Tier 1 only (for LLM queries)."""
        path = self.graphs_dir / f"graph_{graph_id:04d}.json"
        if path.exists():
            return GraphRecord.from_dict(json.load(open(path)))
        return None

    def load_graph_detail(self, graph_id: int) -> Optional[GraphDetail]:
        """Load Tier 2 (for visualization)."""
        path = self.details_dir / f"graph_{graph_id:04d}_detail.json"
        if path.exists():
            return GraphDetail.from_dict(json.load(open(path)))
        return None
```

## Tool Changes

### query_past_graphs (NEW)

```python
@tool
def query_past_graphs(
    problem_pattern: Optional[str] = None,
    n_dimensions: Optional[int] = None,
    success: Optional[bool] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    """
    Query past optimization graphs for learning.

    Use this to find what strategies worked for similar problems.
    Returns compact summaries optimized for reasoning.

    Args:
        problem_pattern: Problem ID pattern (e.g., "ackley*", "rosenbrock*")
        n_dimensions: Filter by problem dimensions
        success: Filter by success status
        limit: Maximum results to return

    Returns:
        List of graph summaries with:
        - graph_id, problem_id
        - problem_signature (dimensions, constraints)
        - pattern (chain, multistart, etc.)
        - strategy sequence (optimizers used, in order)
        - outcome (success, final_objective, efficiency)

    Example:
        # Find what worked for similar high-dimensional problems
        results = query_past_graphs(n_dimensions=50, success=True)

        # Results show: "Graph #42 used TPE→L-BFGS-B chain, achieved 0.001"
        # Agent reasons: "I should try TPE→L-BFGS-B for this 50D problem"
    """
```

### get_graph_state (No Change)

The existing `get_graph_state()` reads from ActiveGraph (in-memory), not from storage. No changes needed.

## Example: Before vs After

### Before (Current)

```json
// graph_0006.json - 289KB, 8840 lines
{
  "nodes": {
    "n1": {
      "progress": {
        "iterations": [
          {"iteration": 1, "design": [50 floats...], "objective": 21.28},
          {"iteration": 2, "design": [50 floats...], "objective": 21.23},
          // ... 200 more iterations with full design vectors
        ]
      }
    }
  }
}
```

### After (Tier 1) - LLM-Ready

```json
// graph_0006.json - ~1KB (strategy-focused, no trajectories)
{
  "graph_id": 6,
  "problem_id": "ackley_50d",
  "problem_signature": {
    "n_dimensions": 50,
    "n_constraints": 0,
    "bounds_range": [-32.768, 32.768],
    "domain_hint": "ackley"
  },
  "pattern": "chain",
  "nodes": {
    "n1": {
      "optimizer": "optuna:TPE",
      "optimizer_family": "bayesian",
      "config": {"n_trials": 100, "sampler": "TPE"},
      "init_strategy": "random",
      "parent_node": null,
      "edge_type": null,
      "status": "completed",
      "n_evaluations": 100,
      "wall_time": 5.2,
      "start_objective": 21.28,
      "best_objective": 15.2
    },
    "n2": {
      "optimizer": "scipy:L-BFGS-B",
      "optimizer_family": "gradient",
      "config": {"maxiter": 200, "ftol": 1e-8, "gtol": 1e-6},
      "init_strategy": "warm_start",
      "parent_node": "n1",
      "edge_type": "warm_start",
      "status": "completed",
      "n_evaluations": 150,
      "wall_time": 3.1,
      "start_objective": 15.2,
      "best_objective": 0.001
    }
  },
  "edges": [
    {"source": "n1", "target": "n2", "edge_type": "warm_start"}
  ],
  "success": true,
  "final_objective": 0.001,
  "total_evaluations": 250,
  "total_wall_time": 8.3,
  "decisions": [
    {"reasoning": "TPE found promising region, switching to gradient-based refinement"}
  ]
}
```

### After (Tier 2) - Detailed Trajectories

```json
// graph_0006_detail.json - ~100KB (all trajectory data)
{
  "graph_id": 6,
  "nodes": {
    "n1": {
      "node_id": "n1",
      "x0": [50 floats...],
      "best_x": [50 floats...],
      "convergence_history": [
        {"iteration": 1, "objective": 21.28},
        {"iteration": 2, "objective": 21.23},
        {"iteration": 10, "objective": 19.5},
        {"iteration": 50, "objective": 16.8},
        {"iteration": 100, "objective": 15.2}
      ],
      "x_history": [
        {"iteration": 1, "x": [50 floats...]},
        {"iteration": 2, "x": [50 floats...]},
        // ...
      ]
    },
    "n2": {
      "node_id": "n2",
      "x0": [50 floats...],
      "best_x": [50 floats...],
      "convergence_history": [...],
      "x_history": [...]
    }
  }
}
```

## Migration Strategy

1. New graphs use two-tier storage automatically
2. Old graphs remain readable (backward compatible)
3. Optional: migration script to split existing graphs

## Impact Summary

| Aspect | Before | After |
|--------|--------|-------|
| Tier 1 file size | 100-300KB | **~1KB** |
| LLM context usage | Massive, unusable | **Compact, learnable** |
| Full config available | No | **Yes** |
| Problem signature | No | **Yes** |
| Convergence curves | Hidden in x data | Tier 2 (queryable) |
| x trajectory | In main file | Tier 2 (separate) |
| get_graph_state() | No change | No change |

## What LLM Learns From Tier 1

From a single ~1KB GraphRecord, the LLM can learn:

1. **Problem characteristics**: "50D Ackley with bounds [-32, 32]"
2. **Strategy pattern**: "Chain: TPE → L-BFGS-B with warm_start"
3. **Configuration choices**: "TPE used 100 trials, L-BFGS-B used ftol=1e-8"
4. **Efficiency**: "100 + 150 = 250 evals, reduced 21.28 → 0.001"
5. **Decision rationale**: "Switched to gradient after finding promising region"

This is exactly what the LLM needs to make informed decisions for similar problems.

## Next Steps

1. Implement GraphRecord and GraphDetail schema classes
2. Update FileStorage for two-tier writes
3. Modify finalize_graph to use two-tier storage
4. Add `query_past_graphs` tool (Tier 1 only)
5. Add `get_node_detail` tool (Tier 2, for future deep analysis)
6. Update CLI `/graph plot` to use Tier 2
7. Optional: Migration script for existing graphs
