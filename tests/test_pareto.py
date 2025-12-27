"""
Tests for Pareto front schema and storage.
"""

import pytest
import numpy as np
import tempfile
import os

from paola.foundry.schema.pareto import ParetoSolution, ParetoFront
from paola.foundry.storage.pareto_storage import ParetoStorage


class TestParetoSolution:
    """Tests for ParetoSolution dataclass."""

    def test_create_solution(self):
        sol = ParetoSolution(
            x=np.array([1.0, 2.0]),
            f=np.array([0.5, 0.3]),
        )
        assert len(sol.x) == 2
        assert len(sol.f) == 2
        assert sol.rank == 0
        assert sol.feasible

    def test_serialization(self):
        sol = ParetoSolution(
            x=np.array([1.0, 2.0]),
            f=np.array([0.5, 0.3]),
            rank=1,
            crowding_distance=0.5,
        )
        d = sol.to_dict()
        sol2 = ParetoSolution.from_dict(d)

        np.testing.assert_array_equal(sol.x, sol2.x)
        np.testing.assert_array_equal(sol.f, sol2.f)
        assert sol2.rank == 1
        assert sol2.crowding_distance == 0.5


class TestParetoFront:
    """Tests for ParetoFront class."""

    @pytest.fixture
    def sample_front(self):
        """Create a sample 2-objective Pareto front."""
        solutions = [
            ParetoSolution(x=np.array([0.0, 0.0]), f=np.array([0.0, 1.0])),
            ParetoSolution(x=np.array([0.5, 0.5]), f=np.array([0.25, 0.5])),
            ParetoSolution(x=np.array([0.7, 0.7]), f=np.array([0.4, 0.3])),
            ParetoSolution(x=np.array([1.0, 1.0]), f=np.array([1.0, 0.0])),
        ]
        return ParetoFront(
            solutions=solutions,
            objective_names=["f1", "f2"],
            objective_senses=["minimize", "minimize"],
            graph_id=1,
            node_id="n1",
            algorithm="NSGA-II",
        )

    def test_properties(self, sample_front):
        assert sample_front.n_solutions == 4
        assert sample_front.n_objectives == 2
        assert sample_front.objective_names == ["f1", "f2"]

    def test_pareto_arrays(self, sample_front):
        ps = sample_front.pareto_set
        pf = sample_front.pareto_front_array

        assert ps.shape == (4, 2)
        assert pf.shape == (4, 2)

    def test_filter_by_objective(self, sample_front):
        # Filter f1 <= 0.5 (includes f1=0.0, 0.25, 0.4)
        filtered = sample_front.filter_by_objective("f1", max_val=0.5)
        assert filtered.n_solutions == 3
        for sol in filtered.solutions:
            assert sol.f[0] <= 0.5

    def test_get_extreme(self, sample_front):
        # Get minimum f1
        sol = sample_front.get_extreme("f1")
        assert sol.f[0] == 0.0  # First solution has f1=0

        # Get minimum f2
        sol = sample_front.get_extreme("f2")
        assert sol.f[1] == 0.0  # Last solution has f2=0

    def test_get_knee_point(self, sample_front):
        knee = sample_front.get_knee_point()
        assert knee is not None
        # Knee should be in the middle of the front
        assert 0.0 < knee.f[0] < 1.0
        assert 0.0 < knee.f[1] < 1.0

    def test_get_closest_to(self, sample_front):
        # Find solution closest to (0.3, 0.4)
        sol = sample_front.get_closest_to({"f1": 0.3, "f2": 0.4})
        assert sol is not None
        # Should be the second solution (0.25, 0.5)
        np.testing.assert_array_almost_equal(sol.f, [0.25, 0.5])

    def test_get_spread(self, sample_front):
        spread = sample_front.get_spread()
        assert "f1" in spread
        assert "f2" in spread
        assert spread["f1"] == (0.0, 1.0)
        assert spread["f2"] == (0.0, 1.0)

    def test_serialization(self, sample_front):
        d = sample_front.to_dict()
        pf2 = ParetoFront.from_dict(d)

        assert pf2.n_solutions == sample_front.n_solutions
        assert pf2.objective_names == sample_front.objective_names
        assert pf2.graph_id == sample_front.graph_id

    def test_from_arrays(self):
        pareto_set = np.array([[0.0, 0.0], [1.0, 1.0]])
        pareto_front = np.array([[0.0, 1.0], [1.0, 0.0]])

        pf = ParetoFront.from_arrays(
            pareto_set=pareto_set,
            pareto_front=pareto_front,
            objective_names=["f1", "f2"],
            graph_id=1,
            node_id="n1",
        )

        assert pf.n_solutions == 2
        assert pf.n_objectives == 2


class TestParetoStorage:
    """Tests for ParetoStorage class."""

    @pytest.fixture
    def storage(self):
        """Create temporary storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ParetoStorage(tmpdir)

    @pytest.fixture
    def sample_front(self):
        """Create a sample Pareto front."""
        solutions = [
            ParetoSolution(x=np.array([0.0, 0.0]), f=np.array([0.0, 1.0])),
            ParetoSolution(x=np.array([1.0, 1.0]), f=np.array([1.0, 0.0])),
        ]
        return ParetoFront(
            solutions=solutions,
            objective_names=["f1", "f2"],
            graph_id=1,
            node_id="n1",
            algorithm="NSGA-II",
            hypervolume=0.5,
        )

    def test_store_and_load(self, storage, sample_front):
        path = storage.store(sample_front)
        assert os.path.exists(path)

        loaded = storage.load(1, "n1")
        assert loaded is not None
        assert loaded.n_solutions == 2
        assert loaded.hypervolume == 0.5

    def test_exists(self, storage, sample_front):
        assert not storage.exists(1, "n1")
        storage.store(sample_front)
        assert storage.exists(1, "n1")

    def test_delete(self, storage, sample_front):
        storage.store(sample_front)
        assert storage.exists(1, "n1")

        result = storage.delete(1, "n1")
        assert result
        assert not storage.exists(1, "n1")

    def test_list_for_graph(self, storage, sample_front):
        # Store multiple fronts for same graph
        storage.store(sample_front)

        front2 = ParetoFront(
            solutions=sample_front.solutions,
            objective_names=["f1", "f2"],
            graph_id=1,
            node_id="n2",
        )
        storage.store(front2)

        nodes = storage.list_for_graph(1)
        assert set(nodes) == {"n1", "n2"}

    def test_list_all(self, storage, sample_front):
        storage.store(sample_front)

        all_fronts = storage.list_all()
        assert len(all_fronts) == 1
        assert all_fronts[0]["graph_id"] == 1
        assert all_fronts[0]["node_id"] == "n1"

    def test_get_best_hypervolume(self, storage):
        # Store two fronts with different hypervolumes
        front1 = ParetoFront(
            solutions=[ParetoSolution(x=np.array([0.0]), f=np.array([0.5]))],
            objective_names=["f1"],
            graph_id=1,
            node_id="n1",
            hypervolume=0.3,
        )
        front2 = ParetoFront(
            solutions=[ParetoSolution(x=np.array([0.0]), f=np.array([0.5]))],
            objective_names=["f1"],
            graph_id=1,
            node_id="n2",
            hypervolume=0.8,
        )

        storage.store(front1)
        storage.store(front2)

        best = storage.get_best_hypervolume(1)
        assert best is not None
        assert best["node_id"] == "n2"
        assert best["hypervolume"] == 0.8

    def test_get_stats(self, storage, sample_front):
        storage.store(sample_front)

        stats = storage.get_stats()
        assert stats["n_pareto_fronts"] == 1
        assert stats["total_solutions"] == 2
        assert stats["unique_graphs"] == 1


class TestParetoFrontHypervolume:
    """Tests for hypervolume computation."""

    def test_compute_hypervolume(self):
        """Test hypervolume with known value."""
        # Simple case: two solutions forming a square
        solutions = [
            ParetoSolution(x=np.array([0.0]), f=np.array([0.0, 1.0])),
            ParetoSolution(x=np.array([1.0]), f=np.array([1.0, 0.0])),
        ]
        pf = ParetoFront(
            solutions=solutions,
            objective_names=["f1", "f2"],
        )

        try:
            hv = pf.compute_hypervolume(reference_point=np.array([2.0, 2.0]))
            # Hypervolume should be positive
            assert hv > 0
        except ImportError:
            pytest.skip("pymoo not installed")
