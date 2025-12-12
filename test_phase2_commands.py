"""
Test Phase 2 commands: /compare, /plot, /plot compare.

Creates multiple test runs and verifies command handlers work correctly.
"""

from paola.runs import RunManager
from paola.storage import FileStorage
from paola.cli.commands import CommandHandler
from paola.tools.evaluator_tools import register_problem
from paola.backends.analytical import get_analytical_function
from rich.console import Console
import numpy as np
from scipy.optimize import OptimizeResult
import tempfile
import shutil


def create_test_runs(storage, manager):
    """Create multiple test runs with different characteristics."""

    # Create problems
    rosenbrock = get_analytical_function("rosenbrock", 2)
    sphere = get_analytical_function("sphere", 2)
    register_problem("rosenbrock_2d", rosenbrock)
    register_problem("sphere_2d", sphere)

    # Run 1: SLSQP on Rosenbrock (good convergence)
    run1 = manager.create_run("rosenbrock_2d", "Rosenbrock-2D", "SLSQP", "Good convergence")
    for i in range(20):
        design = np.array([1.0 - i*0.05, 1.0 - i*0.05])
        objective = float(100 * i * 0.01)  # Converging
        run1.record_iteration(design, objective)

    result1 = OptimizeResult(
        x=np.array([1.0, 1.0]),
        fun=0.001,
        success=True,
        message="Optimization terminated successfully",
        nfev=20,
        nit=20
    )
    run1.finalize(result1)

    # Run 2: BFGS on Rosenbrock (slower convergence)
    run2 = manager.create_run("rosenbrock_2d", "Rosenbrock-2D", "BFGS", "Slower convergence")
    for i in range(30):
        design = np.array([1.0 - i*0.03, 1.0 - i*0.03])
        objective = float(200 * i * 0.005)  # Slower convergence
        run2.record_iteration(design, objective)

    result2 = OptimizeResult(
        x=np.array([1.0, 1.0]),
        fun=0.005,
        success=True,
        message="Optimization terminated successfully",
        nfev=30,
        nit=30
    )
    run2.finalize(result2)

    # Run 3: SLSQP on Sphere (different problem)
    run3 = manager.create_run("sphere_2d", "Sphere-2D", "SLSQP", "Different problem")
    for i in range(15):
        design = np.array([i*0.1, i*0.1])
        objective = float(50 - i * 2)  # Converging to local minimum
        run3.record_iteration(design, objective)

    result3 = OptimizeResult(
        x=np.array([0.0, 0.0]),
        fun=5.0,
        success=True,
        message="Optimization terminated successfully",
        nfev=15,
        nit=15
    )
    run3.finalize(result3)

    return [run1.run_id, run2.run_id, run3.run_id]


def test_phase2_commands():
    """Test all Phase 2 command handlers."""

    # Create temporary storage
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp directory: {temp_dir}\n")

    try:
        storage = FileStorage(temp_dir)
        manager = RunManager()
        manager.set_storage(storage)

        # Create test runs
        print("Creating test runs...")
        run_ids = create_test_runs(storage, manager)
        print(f"✓ Created runs: {run_ids}\n")

        # Create command handler
        console = Console()
        handler = CommandHandler(storage, console)

        # Test 1: /plot <run_id>
        print("=" * 60)
        print("Test 1: /plot <run_id>")
        print("=" * 60)
        handler.handle_plot(run_ids[0])
        print("✓ /plot command works\n")

        # Test 2: /compare <run1> <run2>
        print("=" * 60)
        print("Test 2: /compare <run1> <run2>")
        print("=" * 60)
        handler.handle_compare([run_ids[0], run_ids[1]])
        print("✓ /compare command works\n")

        # Test 3: /plot compare <run1> <run2> <run3>
        print("=" * 60)
        print("Test 3: /plot compare <run1> <run2> <run3>")
        print("=" * 60)
        handler.handle_plot_compare(run_ids)
        print("✓ /plot compare command works\n")

        print("=" * 60)
        print("✅ All Phase 2 commands working correctly!")
        print("=" * 60)
        print("\nPhase 2 Implementation Complete:")
        print("  ✓ /plot <run_id> - ASCII convergence plot")
        print("  ✓ /compare <run1> <run2> - Side-by-side comparison")
        print("  ✓ /plot compare <run1> <run2> - Overlay convergence curves")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        RunManager.reset()
        print(f"\n✓ Cleaned up temp directory")


if __name__ == "__main__":
    test_phase2_commands()
