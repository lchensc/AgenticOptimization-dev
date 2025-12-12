#!/usr/bin/env python
"""Test script to verify Phase 1 refactoring works end-to-end."""

import sys
import tempfile
import shutil
from pathlib import Path

def test_platform_basics():
    """Test basic platform operations."""
    print("Testing platform basics...")

    from paola.platform import OptimizationPlatform, FileStorage, Run, RunRecord, Problem
    import numpy as np

    # Create temp directory for testing
    temp_dir = tempfile.mkdtemp(prefix="paola_test_")

    try:
        # Initialize platform
        storage = FileStorage(base_dir=temp_dir)
        platform = OptimizationPlatform(storage=storage)
        print(f"✓ Platform initialized: {platform}")

        # Create a run
        run = platform.create_run(
            problem_id="test_problem",
            problem_name="Test Problem",
            algorithm="SLSQP",
            description="Test run"
        )
        print(f"✓ Run created: {run}")
        assert run.run_id == 1, "First run should have ID 1"

        # Record some iterations
        for i in range(5):
            design = np.random.rand(10)
            objective = np.sum(design**2)
            run.record_iteration(design=design, objective=objective)
        print(f"✓ Recorded {len(run.iterations)} iterations")

        # Check persistence (file should exist)
        run_file = Path(temp_dir) / "runs" / "run_001.json"
        assert run_file.exists(), "Run file should be persisted"
        print(f"✓ Run persisted to: {run_file}")

        # Create mock result
        class MockResult:
            def __init__(self):
                self.fun = 0.123
                self.x = np.array([0.1] * 10)
                self.success = True
                self.message = "Optimization terminated successfully"
                self.nfev = 50
                self.nit = 10
                self.njev = 10

        # Finalize run
        result = MockResult()
        run.finalize(result)
        print("✓ Run finalized")

        # Remove from active
        platform.finalize_run(run.run_id)
        assert platform.get_run(1) is None, "Run should be removed from active"
        print("✓ Run removed from active registry")

        # Load from storage
        loaded_run = platform.load_run(1)
        assert loaded_run is not None, "Should be able to load run"
        assert loaded_run.run_id == 1
        assert loaded_run.success == True
        assert loaded_run.objective_value == 0.123
        print(f"✓ Loaded run from storage: {loaded_run.run_id}")

        # Query all runs
        all_runs = platform.load_all_runs()
        assert len(all_runs) == 1
        print(f"✓ Loaded all runs: {len(all_runs)} runs")

        print("\n✅ All platform tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"✓ Cleaned up temp directory: {temp_dir}")


def test_tools_integration():
    """Test that tools can use the platform."""
    print("\nTesting tools integration...")

    from paola.platform import OptimizationPlatform, FileStorage
    from paola.tools.run_tools import set_platform, start_optimization_run, get_active_runs
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp(prefix="paola_test_")

    try:
        # Initialize platform
        storage = FileStorage(base_dir=temp_dir)
        platform = OptimizationPlatform(storage=storage)

        # Set global platform for tools
        set_platform(platform)
        print("✓ Platform set for tools")

        # Call tool (use .invoke() for LangChain tools)
        result = start_optimization_run.invoke({
            "problem_id": "test_problem_2",
            "algorithm": "SLSQP",
            "description": "Tool test"
        })

        assert result["success"] == True, "Tool should succeed"
        assert result["run_id"] == 1, "Should get run_id 1"
        print(f"✓ Tool created run: {result['run_id']}")

        # Get active runs (use .invoke() for LangChain tools)
        active_result = get_active_runs.invoke({})
        assert active_result["success"] == True
        assert active_result["count"] == 1
        print(f"✓ Active runs: {active_result['count']}")

        print("\n✅ All tool integration tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"✓ Cleaned up temp directory: {temp_dir}")


def test_cli_initialization():
    """Test that CLI can initialize with new platform."""
    print("\nTesting CLI initialization...")

    try:
        from paola.cli import AgenticOptREPL
        from paola.platform import FileStorage
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp(prefix="paola_test_")

        try:
            # Create REPL (don't run it)
            storage = FileStorage(base_dir=temp_dir)
            repl = AgenticOptREPL(llm_model="qwen-flash", storage=storage)

            assert repl.platform is not None, "REPL should have platform"
            assert repl.command_handler is not None, "REPL should have command handler"
            print("✓ REPL initialized with platform")

            print("\n✅ CLI initialization test passed!")
            return True

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Phase 1 Refactoring Verification Tests")
    print("="*60)
    print()

    results = []

    results.append(("Platform Basics", test_platform_basics()))
    results.append(("Tools Integration", test_tools_integration()))
    results.append(("CLI Initialization", test_cli_initialization()))

    print()
    print("="*60)
    print("Test Summary")
    print("="*60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")

    all_passed = all(r[1] for r in results)

    print()
    if all_passed:
        print("🎉 All tests passed! Phase 1 refactoring successful!")
        return 0
    else:
        print("❌ Some tests failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
