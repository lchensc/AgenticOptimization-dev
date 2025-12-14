"""
Test the new run architecture.

Verifies that:
1. RunManager can be initialized with storage
2. Runs can be created and tracked
3. Runs auto-persist to storage
4. CLI can read runs from storage
"""

from paola.runs import RunManager, OptimizationRun as ActiveRun
from paola.storage import FileStorage
from paola.tools.evaluator_tools import create_benchmark_problem, register_problem
from paola.tools.run_tools import start_optimization_run
from paola.backends.analytical import get_analytical_function
import numpy as np
from scipy.optimize import OptimizeResult
import tempfile
import shutil

def test_run_architecture():
    """Test the complete run architecture."""

    # Create temporary storage
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp directory: {temp_dir}")

    try:
        storage = FileStorage(temp_dir)

        # Initialize RunManager with storage
        manager = RunManager()
        manager.set_storage(storage)
        print("✓ RunManager initialized with storage")

        # Create a problem
        problem = get_analytical_function("rosenbrock", 2)
        from paola.tools.evaluator_tools import register_problem
        register_problem("rosenbrock_2d", problem)
        print("✓ Problem registered")

        # Create a run via tool (simulating agent)
        result = start_optimization_run.invoke({
            "problem_id": "rosenbrock_2d",
            "algorithm": "SLSQP",
            "description": "Test run"
        })

        assert result["success"], f"Failed to create run: {result['message']}"
        run_id = result["run_id"]
        print(f"✓ Run created: #{run_id}")

        # Get the active run
        run = manager.get_run(run_id)
        assert run is not None, "Run not found in manager"
        print(f"✓ Run retrieved from manager: {run}")

        # Record some iterations
        for i in range(5):
            design = np.array([i * 0.1, i * 0.1])
            objective = float(i * i)
            run.record_iteration(design, objective)
        print(f"✓ Recorded 5 iterations")

        # Finalize with a fake result
        fake_result = OptimizeResult(
            x=np.array([1.0, 1.0]),
            fun=0.0,
            success=True,
            message="Optimization terminated successfully",
            nfev=10,
            nit=5
        )
        run.finalize(fake_result)
        print(f"✓ Run finalized")

        # Verify run was saved to storage
        loaded_run = storage.load_run(run_id)
        assert loaded_run is not None, "Run not found in storage"
        assert loaded_run.run_id == run_id
        assert loaded_run.algorithm == "SLSQP"
        assert loaded_run.success == True
        print(f"✓ Run persisted to storage and loaded successfully")
        print(f"  - Run ID: {loaded_run.run_id}")
        print(f"  - Problem: {loaded_run.problem_name}")
        print(f"  - Algorithm: {loaded_run.algorithm}")
        print(f"  - Objective: {loaded_run.objective_value}")
        print(f"  - Success: {loaded_run.success}")
        print(f"  - Evaluations: {loaded_run.n_evaluations}")

        # Test that CLI can read the run
        all_runs = storage.load_all_runs()
        assert len(all_runs) == 1, f"Expected 1 run, found {len(all_runs)}"
        assert all_runs[0].run_id == run_id
        print(f"✓ CLI can query runs from storage")

        print("\n✅ All tests passed!")
        print("\nArchitecture verified:")
        print("  ✓ Agent creates runs explicitly via tools")
        print("  ✓ Runs auto-persist to storage")
        print("  ✓ Storage is independent of agent/CLI")
        print("  ✓ CLI queries storage for display")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        RunManager.reset()  # Reset singleton for next test
        print(f"\n✓ Cleaned up temp directory")

if __name__ == "__main__":
    test_run_architecture()
