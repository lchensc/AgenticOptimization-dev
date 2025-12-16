"""
Unit tests for graph-based optimization schema.

Tests:
- EdgeType validation
- OptimizationEdge creation and serialization
- OptimizationNode creation and serialization
- GraphDecision creation and serialization
- OptimizationGraph operations (nodes, edges, traversal, patterns)
"""

import pytest
from datetime import datetime

from paola.foundry.schema import (
    EdgeType,
    OptimizationEdge,
    OptimizationNode,
    GraphDecision,
    OptimizationGraph,
)


# =============================================================================
# EdgeType Tests
# =============================================================================

class TestEdgeType:
    """Tests for EdgeType constants and validation."""

    def test_all_types_returns_list(self):
        """all_types() should return a list of strings."""
        types = EdgeType.all_types()
        assert isinstance(types, list)
        assert len(types) > 0
        assert all(isinstance(t, str) for t in types)

    def test_standard_types_exist(self):
        """Standard edge types should be defined."""
        assert EdgeType.WARM_START == "warm_start"
        assert EdgeType.RESTART == "restart"
        assert EdgeType.REFINE == "refine"
        assert EdgeType.BRANCH == "branch"
        assert EdgeType.EXPLORE == "explore"

    def test_is_valid_accepts_valid_types(self):
        """is_valid() should accept all standard types."""
        for edge_type in EdgeType.all_types():
            assert EdgeType.is_valid(edge_type) is True

    def test_is_valid_rejects_invalid_types(self):
        """is_valid() should reject invalid types."""
        assert EdgeType.is_valid("invalid_type") is False
        assert EdgeType.is_valid("") is False
        assert EdgeType.is_valid("WARM_START") is False  # Case sensitive


# =============================================================================
# OptimizationEdge Tests
# =============================================================================

class TestOptimizationEdge:
    """Tests for OptimizationEdge dataclass."""

    def test_create_edge(self):
        """Should create edge with valid type."""
        edge = OptimizationEdge(
            source="n1",
            target="n2",
            edge_type=EdgeType.WARM_START,
        )
        assert edge.source == "n1"
        assert edge.target == "n2"
        assert edge.edge_type == "warm_start"
        assert edge.metadata == {}

    def test_create_edge_with_metadata(self):
        """Should create edge with metadata."""
        edge = OptimizationEdge(
            source="n1",
            target="n2",
            edge_type=EdgeType.BRANCH,
            metadata={"reason": "explore alternative"},
        )
        assert edge.metadata == {"reason": "explore alternative"}

    def test_create_edge_invalid_type_raises(self):
        """Should raise ValueError for invalid edge type."""
        with pytest.raises(ValueError, match="Invalid edge type"):
            OptimizationEdge(
                source="n1",
                target="n2",
                edge_type="invalid_type",
            )

    def test_edge_to_dict(self):
        """Should serialize edge to dictionary."""
        edge = OptimizationEdge(
            source="n1",
            target="n2",
            edge_type=EdgeType.REFINE,
            metadata={"step": 1},
        )
        d = edge.to_dict()
        assert d == {
            "source": "n1",
            "target": "n2",
            "edge_type": "refine",
            "metadata": {"step": 1},
        }

    def test_edge_from_dict(self):
        """Should deserialize edge from dictionary."""
        d = {
            "source": "n1",
            "target": "n2",
            "edge_type": "warm_start",
            "metadata": {"x0_source": "best"},
        }
        edge = OptimizationEdge.from_dict(d)
        assert edge.source == "n1"
        assert edge.target == "n2"
        assert edge.edge_type == "warm_start"
        assert edge.metadata == {"x0_source": "best"}

    def test_edge_roundtrip(self):
        """Should survive serialization roundtrip."""
        original = OptimizationEdge(
            source="n1",
            target="n2",
            edge_type=EdgeType.EXPLORE,
            metadata={"seed": 42},
        )
        restored = OptimizationEdge.from_dict(original.to_dict())
        assert restored.source == original.source
        assert restored.target == original.target
        assert restored.edge_type == original.edge_type
        assert restored.metadata == original.metadata


# =============================================================================
# OptimizationNode Tests
# =============================================================================

class TestOptimizationNode:
    """Tests for OptimizationNode dataclass."""

    def test_create_node_minimal(self):
        """Should create node with minimal required fields."""
        node = OptimizationNode(
            node_id="n1",
            optimizer="scipy:SLSQP",
            optimizer_family="gradient",
        )
        assert node.node_id == "n1"
        assert node.optimizer == "scipy:SLSQP"
        assert node.optimizer_family == "gradient"
        assert node.status == "pending"
        assert node.best_objective is None
        assert node.best_x is None

    def test_create_node_completed(self):
        """Should create completed node with results."""
        node = OptimizationNode(
            node_id="n1",
            optimizer="scipy:SLSQP",
            optimizer_family="gradient",
            status="completed",
            n_evaluations=50,
            wall_time=1.5,
            best_objective=0.001,
            best_x=[1.0, 2.0, 3.0],
        )
        assert node.status == "completed"
        assert node.n_evaluations == 50
        assert node.best_objective == 0.001
        assert node.best_x == [1.0, 2.0, 3.0]

    def test_is_completed(self):
        """is_completed() should check status."""
        node_pending = OptimizationNode(
            node_id="n1", optimizer="scipy:SLSQP", optimizer_family="gradient"
        )
        node_completed = OptimizationNode(
            node_id="n2", optimizer="scipy:SLSQP", optimizer_family="gradient",
            status="completed"
        )
        assert node_pending.is_completed() is False
        assert node_completed.is_completed() is True

    def test_is_successful(self):
        """is_successful() should check status and result."""
        node_pending = OptimizationNode(
            node_id="n1", optimizer="scipy:SLSQP", optimizer_family="gradient"
        )
        node_completed_no_result = OptimizationNode(
            node_id="n2", optimizer="scipy:SLSQP", optimizer_family="gradient",
            status="completed"
        )
        node_successful = OptimizationNode(
            node_id="n3", optimizer="scipy:SLSQP", optimizer_family="gradient",
            status="completed", best_objective=0.1
        )
        assert node_pending.is_successful() is False
        assert node_completed_no_result.is_successful() is False
        assert node_successful.is_successful() is True

    def test_node_to_dict(self):
        """Should serialize node to dictionary."""
        node = OptimizationNode(
            node_id="n1",
            optimizer="scipy:SLSQP",
            optimizer_family="gradient",
            config={"maxiter": 100},
            status="completed",
            created_at="2025-12-16T10:00:00",
            n_evaluations=50,
            wall_time=1.5,
            best_objective=0.001,
            best_x=[1.0, 2.0],
        )
        d = node.to_dict()
        assert d["node_id"] == "n1"
        assert d["optimizer"] == "scipy:SLSQP"
        assert d["optimizer_family"] == "gradient"
        assert d["config"] == {"maxiter": 100}
        assert d["status"] == "completed"
        assert d["best_objective"] == 0.001
        assert d["best_x"] == [1.0, 2.0]

    def test_node_from_dict(self):
        """Should deserialize node from dictionary."""
        d = {
            "node_id": "n1",
            "optimizer": "optuna:TPE",
            "optimizer_family": "bayesian",
            "config": {"n_trials": 50},
            "status": "completed",
            "created_at": "2025-12-16T10:00:00",
            "n_evaluations": 50,
            "wall_time": 10.0,
            "best_objective": 0.05,
            "best_x": [0.1, 0.2, 0.3],
        }
        node = OptimizationNode.from_dict(d)
        assert node.node_id == "n1"
        assert node.optimizer == "optuna:TPE"
        assert node.optimizer_family == "bayesian"
        assert node.best_objective == 0.05
        assert node.best_x == [0.1, 0.2, 0.3]

    def test_node_roundtrip(self):
        """Should survive serialization roundtrip."""
        original = OptimizationNode(
            node_id="n1",
            optimizer="scipy:L-BFGS-B",
            optimizer_family="gradient",
            config={"maxiter": 200},
            status="completed",
            created_at="2025-12-16T10:00:00",
            completed_at="2025-12-16T10:01:00",
            n_evaluations=100,
            wall_time=60.0,
            best_objective=1e-6,
            best_x=[1.0, 1.0, 1.0],
        )
        restored = OptimizationNode.from_dict(original.to_dict())
        assert restored.node_id == original.node_id
        assert restored.optimizer == original.optimizer
        assert restored.status == original.status
        assert restored.best_objective == original.best_objective
        assert restored.best_x == original.best_x


# =============================================================================
# GraphDecision Tests
# =============================================================================

class TestGraphDecision:
    """Tests for GraphDecision dataclass."""

    def test_create_decision(self):
        """Should create decision record."""
        decision = GraphDecision(
            timestamp="2025-12-16T10:00:00",
            decision_type="add_node",
            reasoning="Starting with global exploration",
            from_node=None,
            to_node="n1",
            metrics_at_decision={},
        )
        assert decision.decision_type == "add_node"
        assert decision.to_node == "n1"

    def test_decision_to_dict(self):
        """Should serialize decision to dictionary."""
        decision = GraphDecision(
            timestamp="2025-12-16T10:05:00",
            decision_type="branch",
            reasoning="Found promising region, switching to gradient",
            from_node="n1",
            to_node="n2",
            metrics_at_decision={"best_obj": 0.15},
        )
        d = decision.to_dict()
        assert d["decision_type"] == "branch"
        assert d["from_node"] == "n1"
        assert d["to_node"] == "n2"
        assert d["metrics_at_decision"] == {"best_obj": 0.15}

    def test_decision_roundtrip(self):
        """Should survive serialization roundtrip."""
        original = GraphDecision(
            timestamp="2025-12-16T10:00:00",
            decision_type="terminate",
            reasoning="Converged successfully",
            from_node="n3",
            to_node=None,
            metrics_at_decision={"final_obj": 0.001, "gradient_norm": 1e-8},
        )
        restored = GraphDecision.from_dict(original.to_dict())
        assert restored.decision_type == original.decision_type
        assert restored.reasoning == original.reasoning
        assert restored.metrics_at_decision == original.metrics_at_decision


# =============================================================================
# OptimizationGraph Tests
# =============================================================================

class TestOptimizationGraph:
    """Tests for OptimizationGraph dataclass."""

    @pytest.fixture
    def empty_graph(self):
        """Create an empty graph."""
        return OptimizationGraph(
            graph_id=1,
            problem_id="test_problem",
            created_at="2025-12-16T10:00:00",
        )

    @pytest.fixture
    def chain_graph(self):
        """Create a chain graph: n1 → n2 → n3."""
        graph = OptimizationGraph(
            graph_id=1,
            problem_id="test_problem",
            created_at="2025-12-16T10:00:00",
        )

        # Add nodes
        graph.add_node(OptimizationNode(
            node_id="n1", optimizer="optuna:TPE", optimizer_family="bayesian",
            status="completed", best_objective=0.15, best_x=[0.1, 0.2],
        ))
        graph.add_node(OptimizationNode(
            node_id="n2", optimizer="scipy:SLSQP", optimizer_family="gradient",
            status="completed", best_objective=0.08, best_x=[0.11, 0.19],
        ))
        graph.add_node(OptimizationNode(
            node_id="n3", optimizer="cmaes", optimizer_family="cmaes",
            status="completed", best_objective=0.05, best_x=[0.1, 0.2],
        ))

        # Add edges
        graph.edges.append(OptimizationEdge("n1", "n2", EdgeType.WARM_START))
        graph.edges.append(OptimizationEdge("n2", "n3", EdgeType.WARM_START))

        return graph

    @pytest.fixture
    def multistart_graph(self):
        """Create a multistart graph: {n1, n2, n3} independent."""
        graph = OptimizationGraph(
            graph_id=2,
            problem_id="test_problem",
            created_at="2025-12-16T10:00:00",
        )

        for i, obj in enumerate([0.5, 0.3, 0.7], start=1):
            graph.add_node(OptimizationNode(
                node_id=f"n{i}", optimizer="scipy:L-BFGS-B", optimizer_family="gradient",
                status="completed", best_objective=obj, best_x=[float(i)],
            ))

        return graph

    @pytest.fixture
    def tree_graph(self):
        """Create a tree graph: n1 → {n2, n3}."""
        graph = OptimizationGraph(
            graph_id=3,
            problem_id="test_problem",
            created_at="2025-12-16T10:00:00",
        )

        graph.add_node(OptimizationNode(
            node_id="n1", optimizer="optuna:TPE", optimizer_family="bayesian",
            status="completed", best_objective=0.2, best_x=[0.5],
        ))
        graph.add_node(OptimizationNode(
            node_id="n2", optimizer="scipy:SLSQP", optimizer_family="gradient",
            status="completed", best_objective=0.1, best_x=[0.4],
        ))
        graph.add_node(OptimizationNode(
            node_id="n3", optimizer="ipopt", optimizer_family="gradient",
            status="completed", best_objective=0.15, best_x=[0.45],
        ))

        graph.edges.append(OptimizationEdge("n1", "n2", EdgeType.WARM_START))
        graph.edges.append(OptimizationEdge("n1", "n3", EdgeType.BRANCH))

        return graph

    # === Node Operations ===

    def test_add_node(self, empty_graph):
        """Should add node to graph."""
        node = OptimizationNode(
            node_id="n1", optimizer="scipy:SLSQP", optimizer_family="gradient"
        )
        empty_graph.add_node(node)
        assert "n1" in empty_graph.nodes
        assert empty_graph.get_node("n1") == node

    def test_add_duplicate_node_raises(self, empty_graph):
        """Should raise error when adding duplicate node."""
        node = OptimizationNode(
            node_id="n1", optimizer="scipy:SLSQP", optimizer_family="gradient"
        )
        empty_graph.add_node(node)
        with pytest.raises(ValueError, match="already exists"):
            empty_graph.add_node(node)

    def test_get_node_not_found(self, empty_graph):
        """Should return None for non-existent node."""
        assert empty_graph.get_node("n999") is None

    def test_get_best_node(self, chain_graph):
        """Should return node with minimum objective."""
        best = chain_graph.get_best_node()
        assert best.node_id == "n3"
        assert best.best_objective == 0.05

    def test_get_best_node_empty_graph(self, empty_graph):
        """Should return None for empty graph."""
        assert empty_graph.get_best_node() is None

    def test_get_completed_nodes(self, chain_graph):
        """Should return all completed nodes."""
        completed = chain_graph.get_completed_nodes()
        assert len(completed) == 3

    # === Edge Operations ===

    def test_add_edge(self, empty_graph):
        """Should add edge between existing nodes."""
        empty_graph.add_node(OptimizationNode(
            node_id="n1", optimizer="scipy:SLSQP", optimizer_family="gradient"
        ))
        empty_graph.add_node(OptimizationNode(
            node_id="n2", optimizer="scipy:SLSQP", optimizer_family="gradient"
        ))

        edge = empty_graph.add_edge("n1", "n2", EdgeType.WARM_START)
        assert edge.source == "n1"
        assert edge.target == "n2"
        assert len(empty_graph.edges) == 1

    def test_add_edge_invalid_source_raises(self, empty_graph):
        """Should raise error for non-existent source node."""
        empty_graph.add_node(OptimizationNode(
            node_id="n2", optimizer="scipy:SLSQP", optimizer_family="gradient"
        ))
        with pytest.raises(ValueError, match="Source node n1 not found"):
            empty_graph.add_edge("n1", "n2", EdgeType.WARM_START)

    def test_add_edge_invalid_target_raises(self, empty_graph):
        """Should raise error for non-existent target node."""
        empty_graph.add_node(OptimizationNode(
            node_id="n1", optimizer="scipy:SLSQP", optimizer_family="gradient"
        ))
        with pytest.raises(ValueError, match="Target node n2 not found"):
            empty_graph.add_edge("n1", "n2", EdgeType.WARM_START)

    def test_get_predecessors(self, chain_graph):
        """Should return predecessor nodes."""
        assert chain_graph.get_predecessors("n1") == []
        assert chain_graph.get_predecessors("n2") == ["n1"]
        assert chain_graph.get_predecessors("n3") == ["n2"]

    def test_get_successors(self, chain_graph):
        """Should return successor nodes."""
        assert chain_graph.get_successors("n1") == ["n2"]
        assert chain_graph.get_successors("n2") == ["n3"]
        assert chain_graph.get_successors("n3") == []

    def test_get_parent_node(self, chain_graph):
        """Should return parent node for warm-start."""
        parent = chain_graph.get_parent_node("n2")
        assert parent.node_id == "n1"

        assert chain_graph.get_parent_node("n1") is None

    # === Graph Analysis ===

    def test_get_root_nodes_chain(self, chain_graph):
        """Should identify root nodes in chain."""
        roots = chain_graph.get_root_nodes()
        assert roots == ["n1"]

    def test_get_root_nodes_multistart(self, multistart_graph):
        """Should identify all nodes as roots in multistart."""
        roots = multistart_graph.get_root_nodes()
        assert set(roots) == {"n1", "n2", "n3"}

    def test_get_leaf_nodes_chain(self, chain_graph):
        """Should identify leaf nodes in chain."""
        leaves = chain_graph.get_leaf_nodes()
        assert leaves == ["n3"]

    def test_get_leaf_nodes_tree(self, tree_graph):
        """Should identify leaf nodes in tree."""
        leaves = tree_graph.get_leaf_nodes()
        assert set(leaves) == {"n2", "n3"}

    def test_detect_pattern_empty(self, empty_graph):
        """Should detect empty pattern."""
        assert empty_graph.detect_pattern() == "empty"

    def test_detect_pattern_single(self, empty_graph):
        """Should detect single pattern."""
        empty_graph.add_node(OptimizationNode(
            node_id="n1", optimizer="scipy:SLSQP", optimizer_family="gradient"
        ))
        assert empty_graph.detect_pattern() == "single"

    def test_detect_pattern_multistart(self, multistart_graph):
        """Should detect multistart pattern."""
        assert multistart_graph.detect_pattern() == "multistart"

    def test_detect_pattern_chain(self, chain_graph):
        """Should detect chain pattern."""
        assert chain_graph.detect_pattern() == "chain"

    def test_detect_pattern_tree(self, tree_graph):
        """Should detect tree pattern."""
        assert tree_graph.detect_pattern() == "tree"

    def test_topological_sort_chain(self, chain_graph):
        """Should return nodes in topological order for chain."""
        order = chain_graph.topological_sort()
        assert order == ["n1", "n2", "n3"]

    def test_topological_sort_multistart(self, multistart_graph):
        """Should return all nodes for multistart (order doesn't matter)."""
        order = multistart_graph.topological_sort()
        assert set(order) == {"n1", "n2", "n3"}

    def test_get_node_depth(self, chain_graph):
        """Should compute node depth correctly."""
        assert chain_graph.get_node_depth("n1") == 0
        assert chain_graph.get_node_depth("n2") == 1
        assert chain_graph.get_node_depth("n3") == 2

    # === Serialization ===

    def test_graph_to_dict(self, chain_graph):
        """Should serialize graph to dictionary."""
        d = chain_graph.to_dict()
        assert d["graph_id"] == 1
        assert d["problem_id"] == "test_problem"
        assert len(d["nodes"]) == 3
        assert len(d["edges"]) == 2
        assert "n1" in d["nodes"]

    def test_graph_from_dict(self, chain_graph):
        """Should deserialize graph from dictionary."""
        d = chain_graph.to_dict()
        restored = OptimizationGraph.from_dict(d)

        assert restored.graph_id == chain_graph.graph_id
        assert restored.problem_id == chain_graph.problem_id
        assert len(restored.nodes) == len(chain_graph.nodes)
        assert len(restored.edges) == len(chain_graph.edges)
        assert restored.detect_pattern() == chain_graph.detect_pattern()

    def test_graph_roundtrip(self, chain_graph):
        """Should survive full serialization roundtrip."""
        chain_graph.decisions.append(GraphDecision(
            timestamp="2025-12-16T10:00:00",
            decision_type="add_node",
            reasoning="Starting exploration",
            from_node=None,
            to_node="n1",
        ))
        chain_graph.success = True
        chain_graph.final_objective = 0.05
        chain_graph.final_x = [0.1, 0.2]

        restored = OptimizationGraph.from_dict(chain_graph.to_dict())

        assert restored.success == chain_graph.success
        assert restored.final_objective == chain_graph.final_objective
        assert restored.final_x == chain_graph.final_x
        assert len(restored.decisions) == 1
        assert restored.decisions[0].decision_type == "add_node"

    def test_graph_json_roundtrip(self, chain_graph):
        """Should survive JSON serialization roundtrip."""
        json_str = chain_graph.to_json()
        restored = OptimizationGraph.from_json(json_str)

        assert restored.graph_id == chain_graph.graph_id
        assert len(restored.nodes) == len(chain_graph.nodes)

    # === Display ===

    def test_summary(self, chain_graph):
        """Should generate readable summary."""
        summary = chain_graph.summary()
        assert "Graph #1" in summary
        assert "test_problem" in summary
        assert "3 nodes" in summary
        assert "chain" in summary

    def test_repr(self, chain_graph):
        """Should have readable repr."""
        r = repr(chain_graph)
        assert "OptimizationGraph" in r
        assert "id=1" in r
        assert "nodes=3" in r
