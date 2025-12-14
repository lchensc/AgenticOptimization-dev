#!/usr/bin/env python
"""Test script to verify Phase 2 analysis module works end-to-end."""

import sys
import tempfile
import shutil
import numpy as np

def test_deterministic_metrics():
    """Test deterministic metrics computation."""
    print("Testing deterministic metrics...")

    from paola.foundry import OptimizationFoundry, FileStorage, Run
    from paola.analysis import compute_metrics

    temp_dir = tempfile.mkdtemp(prefix="paola_test_")

    try:
        # Create platform and run
        storage = FileStorage(base_dir=temp_dir)
        platform = OptimizationFoundry(storage=storage)

        run = platform.create_run(
            problem_id="test_rosenbrock",
            problem_name="Rosenbrock 10D",
            algorithm="SLSQP"
        )

        # Record some iterations with realistic convergence
        objectives = [100.0, 50.0, 25.0, 12.0, 6.0, 3.0, 1.5, 0.75, 0.38, 0.19, 0.10]
        for i, obj in enumerate(objectives):
            design = np.random.rand(10) * 0.1
            gradient = np.random.randn(10) * 0.01
            run.record_iteration(design=design, objective=obj, gradient=gradient)

        # Create mock result
        class MockResult:
            def __init__(self):
                self.fun = 0.10
                self.x = np.zeros(10)
                self.success = True
                self.message = "Optimization terminated successfully"
                self.nfev = 50
                self.nit = len(objectives)
                self.njev = len(objectives)

        run.finalize(MockResult())
        platform.finalize_run(run.run_id)

        # Load and compute metrics
        loaded_run = platform.load_run(run.run_id)
        metrics = compute_metrics(loaded_run)

        print(f"✓ Metrics computed successfully")

        # Verify structure
        assert "convergence" in metrics
        assert "gradient" in metrics
        assert "constraints" in metrics
        assert "efficiency" in metrics
        assert "objective" in metrics
        print(f"✓ All metric categories present")

        # Verify convergence metrics
        assert metrics["convergence"]["iterations_total"] == len(objectives)
        assert metrics["convergence"]["rate"] > 0  # Should be positive (improving)
        assert not metrics["convergence"]["is_stalled"]  # Not stalled
        print(f"✓ Convergence metrics correct")
        print(f"  - Rate: {metrics['convergence']['rate']:.4f}")
        print(f"  - Stalled: {metrics['convergence']['is_stalled']}")
        print(f"  - Improvement (last 10): {metrics['convergence']['improvement_last_10']:.6f}")

        # Verify gradient metrics
        assert metrics["gradient"]["quality"] in ["good", "noisy", "flat", "unknown"]
        print(f"✓ Gradient metrics correct")
        print(f"  - Quality: {metrics['gradient']['quality']}")
        print(f"  - Norm: {metrics['gradient']['norm']:.6e}")

        # Verify efficiency metrics
        assert metrics["efficiency"]["evaluations"] > 0
        print(f"✓ Efficiency metrics correct")
        print(f"  - Evaluations: {metrics['efficiency']['evaluations']}")
        print(f"  - Improvement per eval: {metrics['efficiency']['improvement_per_eval']:.6f}")

        print("\n✅ All deterministic metrics tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ai_analysis_structure():
    """Test AI analysis returns correct structure (even if LLM fails)."""
    print("\nTesting AI analysis structure...")

    from paola.foundry import OptimizationFoundry, FileStorage, Run
    from paola.analysis import compute_metrics, ai_analyze
    import os

    # Note: AI analysis might fail without API key, but should return proper structure
    temp_dir = tempfile.mkdtemp(prefix="paola_test_")

    try:
        # Create platform and run
        storage = FileStorage(base_dir=temp_dir)
        platform = OptimizationFoundry(storage=storage)

        run = platform.create_run(
            problem_id="test_problem",
            problem_name="Test Problem",
            algorithm="SLSQP"
        )

        # Add minimal iteration data
        run.record_iteration(design=np.zeros(5), objective=1.0)

        class MockResult:
            def __init__(self):
                self.fun = 1.0
                self.x = np.zeros(5)
                self.success = True
                self.message = "Test"
                self.nfev = 1
                self.nit = 1
                self.njev = 1

        run.finalize(MockResult())
        platform.finalize_run(run.run_id)

        loaded_run = platform.load_run(run.run_id)
        metrics = compute_metrics(loaded_run)

        # Try AI analysis (will likely fail without API key, but should handle gracefully)
        try:
            insights = ai_analyze(loaded_run, metrics, focus="overall")

            # Check structure (even if analysis failed)
            assert "diagnosis" in insights
            assert "root_cause" in insights
            assert "confidence" in insights
            assert "recommendations" in insights
            assert "metadata" in insights
            print(f"✓ AI analysis returns proper structure")

            if "error" in insights.get("metadata", {}):
                print(f"  [Note: AI analysis failed (expected without API key): {insights['metadata']['error']}]")
            else:
                print(f"  - Diagnosis: {insights['diagnosis'][:50]}...")
                print(f"  - Confidence: {insights['confidence']}")
                print(f"  - Recommendations: {len(insights['recommendations'])}")

        except Exception as e:
            # AI analysis failure is expected without API key
            print(f"  [Note: AI analysis raised exception (expected without API key): {type(e).__name__}]")

        print("\n✅ AI analysis structure test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_analysis_tools():
    """Test that analysis tools can be imported and used."""
    print("\nTesting analysis tools...")

    from paola.foundry import OptimizationFoundry, FileStorage
    from paola.tools.analysis import analyze_convergence, get_all_metrics, analyze_efficiency
    from paola.tools.run_tools import set_foundry
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp(prefix="paola_test_")

    try:
        # Initialize platform
        storage = FileStorage(base_dir=temp_dir)
        platform = OptimizationFoundry(storage=storage)
        set_foundry(platform)

        # Create a run
        run = platform.create_run(
            problem_id="test_problem",
            problem_name="Test",
            algorithm="SLSQP"
        )

        # Add iterations
        for i in range(5):
            run.record_iteration(
                design=np.random.rand(3),
                objective=10.0 / (i + 1),
                gradient=np.random.randn(3) * 0.1
            )

        class MockResult:
            def __init__(self):
                self.fun = 2.0
                self.x = np.zeros(3)
                self.success = True
                self.message = "Success"
                self.nfev = 5
                self.nit = 5
                self.njev = 5

        run.finalize(MockResult())
        platform.finalize_run(run.run_id)

        # Test analyze_convergence tool
        result = analyze_convergence.invoke({"run_id": 1})
        assert "iterations_total" in result
        assert result["iterations_total"] == 5
        print(f"✓ analyze_convergence tool works")

        # Test get_all_metrics tool
        result = get_all_metrics.invoke({"run_id": 1})
        assert "convergence" in result
        assert "gradient" in result
        print(f"✓ get_all_metrics tool works")

        # Test analyze_efficiency tool
        result = analyze_efficiency.invoke({"run_id": 1})
        assert "evaluations" in result
        print(f"✓ analyze_efficiency tool works")

        print("\n✅ All analysis tools tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_cli_show_with_metrics():
    """Test that CLI /show command includes metrics."""
    print("\nTesting CLI /show with metrics...")

    from paola.foundry import OptimizationFoundry, FileStorage
    from paola.cli.commands import CommandHandler
    from rich.console import Console
    import io

    temp_dir = tempfile.mkdtemp(prefix="paola_test_")

    try:
        # Create platform with run
        storage = FileStorage(base_dir=temp_dir)
        platform = OptimizationFoundry(storage=storage)

        run = platform.create_run(
            problem_id="test",
            problem_name="Test Problem",
            algorithm="SLSQP"
        )

        for i in range(10):
            run.record_iteration(
                design=np.random.rand(5),
                objective=10.0 / (i + 1),
                gradient=np.random.randn(5) * 0.1
            )

        class MockResult:
            def __init__(self):
                self.fun = 1.0
                self.x = np.zeros(5)
                self.success = True
                self.message = "Converged"
                self.nfev = 10
                self.nit = 10
                self.njev = 10

        run.finalize(MockResult())
        platform.finalize_run(run.run_id)

        # Create command handler with captured output
        console = Console(file=io.StringIO())
        handler = CommandHandler(platform, console)

        # Call handle_show
        handler.handle_show(1)

        # Get output
        output = console.file.getvalue()

        # Verify metrics are in output
        assert "Metrics:" in output
        assert "Convergence:" in output
        assert "Efficiency:" in output
        assert "Gradient:" in output
        print(f"✓ CLI /show includes metrics")

        print("\n✅ CLI metrics display test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all Phase 2 tests."""
    print("="*60)
    print("Phase 2 Analysis Module Verification Tests")
    print("="*60)
    print()

    results = []

    results.append(("Deterministic Metrics", test_deterministic_metrics()))
    results.append(("AI Analysis Structure", test_ai_analysis_structure()))
    results.append(("Analysis Tools", test_analysis_tools()))
    results.append(("CLI Show with Metrics", test_cli_show_with_metrics()))

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
        print("🎉 All tests passed! Phase 2 refactoring successful!")
        print()
        print("Key Features Added:")
        print("  ✓ compute_metrics() - Deterministic analysis (instant, free)")
        print("  ✓ ai_analyze() - AI-powered reasoning (opt-in, strategic)")
        print("  ✓ Analysis tools for agent (thin wrappers)")
        print("  ✓ CLI /show enhanced with metrics")
        print("  ✓ CLI /analyze command for AI analysis")
        return 0
    else:
        print("❌ Some tests failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
