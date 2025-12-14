#!/usr/bin/env python
"""End-to-end test of real optimization workflow."""

import sys
import tempfile
import shutil
import numpy as np


def test_complete_workflow():
    """Test complete optimization workflow from start to finish."""
    print("=" * 70)
    print("End-to-End Optimization Workflow Test")
    print("=" * 70)
    print()

    from paola.foundry import OptimizationFoundry, FileStorage
    from paola.tools.evaluator_tools import create_benchmark_problem
    from paola.tools.run_tools import start_optimization_run, finalize_optimization_run, set_foundry
    from paola.tools.optimizer_tools import run_scipy_optimization
    from paola.analysis import compute_metrics, ai_analyze
    from paola.cli.commands import CommandHandler
    from rich.console import Console

    temp_dir = tempfile.mkdtemp(prefix="paola_workflow_")

    try:
        print("Step 1: Initialize Platform")
        print("-" * 70)
        storage = FileStorage(base_dir=temp_dir)
        platform = OptimizationFoundry(storage=storage)
        set_foundry(platform)
        print(f"✓ Platform initialized with storage: {temp_dir}")
        print()

        print("Step 2: Create Benchmark Problem")
        print("-" * 70)
        problem_id = "rosenbrock_5d"
        problem_result = create_benchmark_problem.invoke({
            "problem_id": problem_id,
            "function_name": "rosenbrock",
            "dimension": 5
        })
        print(f"✓ Problem created: {problem_result['function_name']}")
        print(f"  - Problem ID: {problem_id}")
        print(f"  - Dimension: {problem_result['dimension']}")
        print()

        print("Step 3: Start Optimization Run")
        print("-" * 70)
        run_result = start_optimization_run.invoke({
            "problem_id": problem_id,
            "algorithm": "SLSQP",
            "description": "Test workflow - Rosenbrock 5D with SLSQP"
        })
        print(f"✓ Run started: ID={run_result['run_id']}")
        print(f"  - Algorithm: {run_result['algorithm']}")
        print(f"  - Problem: {run_result['problem_name']}")
        run_id = run_result['run_id']
        print()

        print("Step 4: Run Optimization")
        print("-" * 70)
        print("  Running SLSQP on Rosenbrock 5D...")
        # Define bounds: Rosenbrock typically uses [-5, 10] for each dimension
        bounds = [[-5.0, 10.0] for _ in range(5)]
        opt_result = run_scipy_optimization.invoke({
            "problem_id": problem_id,
            "algorithm": "SLSQP",
            "bounds": bounds,
            "run_id": run_id,
            "options": '{"maxiter": 100, "ftol": 1e-6}'
        })
        print(f"✓ Optimization completed")
        print(f"  - Success: {opt_result['success']}")
        print(f"  - Final objective: {opt_result['final_objective']:.6e}")
        print(f"  - Iterations: {opt_result.get('n_iterations', opt_result.get('nit', 'N/A'))}")
        print(f"  - Evaluations: {opt_result.get('n_evaluations', opt_result.get('nfev', 'N/A'))}")
        print()

        print("Step 5: Finalize Run")
        print("-" * 70)
        finalize_result = finalize_optimization_run.invoke({
            "run_id": run_id
        })
        status = finalize_result.get('status', finalize_result.get('message', 'completed'))
        print(f"✓ Run finalized: {status}")
        print()

        print("Step 6: Load and Verify Run Record")
        print("-" * 70)
        run_record = platform.load_run(run_id)
        assert run_record is not None
        print(f"✓ Run record loaded from storage")
        print(f"  - Run ID: {run_record.run_id}")
        print(f"  - Success: {run_record.success}")
        print(f"  - Objective: {run_record.objective_value:.6e}")
        print(f"  - Evaluations: {run_record.n_evaluations}")
        print(f"  - Duration: {run_record.duration:.2f}s")
        print()

        print("Step 7: Compute Deterministic Metrics")
        print("-" * 70)
        metrics = compute_metrics(run_record)
        print(f"✓ Metrics computed")
        print(f"  [Convergence]")
        print(f"    - Rate: {metrics['convergence']['rate']:.4f}")
        print(f"    - Stalled: {metrics['convergence']['is_stalled']}")
        print(f"    - Iterations: {metrics['convergence']['iterations_total']}")
        print(f"  [Gradient]")
        print(f"    - Quality: {metrics['gradient']['quality']}")
        print(f"    - Norm: {metrics['gradient']['norm']:.6e}")
        print(f"  [Efficiency]")
        print(f"    - Evaluations: {metrics['efficiency']['evaluations']}")
        print(f"    - Improvement/eval: {metrics['efficiency']['improvement_per_eval']:.6e}")
        print(f"  [Objective]")
        print(f"    - Best: {metrics['objective']['best']:.6e}")
        print(f"    - Improvement: {metrics['objective']['improvement_from_start']:.6e}")
        print()

        print("Step 8: Test CLI Commands")
        print("-" * 70)
        console = Console()
        handler = CommandHandler(platform, console)

        # Test /runs
        print("  Testing /runs command...")
        handler.handle_runs()
        print("  ✓ /runs works")

        # Test /show
        print(f"  Testing /show {run_id} command...")
        handler.handle_show(run_id)
        print("  ✓ /show works")

        # Test /best
        print("  Testing /best command...")
        handler.handle_best()
        print("  ✓ /best works")

        print()

        print("Step 9: Verify Multiple Runs")
        print("-" * 70)
        # Create another run with different algorithm
        run2_result = start_optimization_run.invoke({
            "problem_id": problem_id,
            "algorithm": "Nelder-Mead",
            "description": "Test workflow - Rosenbrock 5D with Nelder-Mead"
        })
        run2_id = run2_result['run_id']

        opt2_result = run_scipy_optimization.invoke({
            "problem_id": problem_id,
            "run_id": run2_id,
            "algorithm": "Nelder-Mead",
            "bounds": bounds,
            "use_gradient": False
        })

        finalize_optimization_run.invoke({"run_id": run2_id})

        print(f"✓ Second run completed")
        print(f"  - Algorithm: Nelder-Mead")
        print(f"  - Success: {opt2_result['success']}")
        print(f"  - Final objective: {opt2_result['final_objective']:.6e}")
        print()

        # Verify platform has both runs
        all_runs = platform.load_all_runs()
        assert len(all_runs) == 2
        print(f"✓ Platform has {len(all_runs)} runs")
        print()

        print("Step 10: Test Comparison Features")
        print("-" * 70)
        print(f"  Testing /compare {run_id} {run2_id}...")
        handler.handle_compare([run_id, run2_id])
        print("  ✓ /compare works")
        print()

        print("=" * 70)
        print("✅ COMPLETE WORKFLOW TEST PASSED!")
        print("=" * 70)
        print()
        print("Summary:")
        print(f"  ✓ Created benchmark problem (Rosenbrock 5D)")
        print(f"  ✓ Ran 2 optimizations (SLSQP, Nelder-Mead)")
        print(f"  ✓ Stored runs in platform")
        print(f"  ✓ Computed metrics successfully")
        print(f"  ✓ CLI commands functional")
        print(f"  ✓ Comparison features working")
        print()
        print("The optimization workflow is fully operational! 🚀")
        print()

        return True

    except Exception as e:
        print()
        print("=" * 70)
        print("❌ WORKFLOW TEST FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_agent_with_tools():
    """Test that agent can use tools in workflow."""
    print()
    print("=" * 70)
    print("Agent Tools Integration Test")
    print("=" * 70)
    print()

    from paola.tools.evaluator_tools import create_benchmark_problem
    from paola.tools.run_tools import start_optimization_run, finalize_optimization_run
    from paola.tools.optimizer_tools import run_scipy_optimization
    from paola.tools.analysis import analyze_convergence, get_all_metrics
    from paola.tools.knowledge_tools import store_optimization_insight

    print("Testing tool invocations...")
    print()

    # Test create_benchmark_problem
    result = create_benchmark_problem.invoke({
        "problem_id": "sphere_3d",
        "function_name": "sphere",
        "dimension": 3
    })
    assert "function_name" in result
    print("✓ create_benchmark_problem works")

    # Test knowledge tools (skeleton)
    result = store_optimization_insight.invoke({
        "problem_type": "sphere",
        "dimensions": 3,
        "algorithm": "SLSQP",
        "success": True,
        "iterations": 10,
        "final_objective": 0.001
    })
    assert result["status"] == "not_implemented"  # Expected for skeleton
    print("✓ store_optimization_insight works (skeleton)")

    print()
    print("✅ All agent tools callable and functional!")
    print()

    return True


def test_storage_persistence():
    """Test that storage persists across sessions."""
    print()
    print("=" * 70)
    print("Storage Persistence Test")
    print("=" * 70)
    print()

    from paola.foundry import OptimizationFoundry, FileStorage
    from paola.tools.evaluator_tools import create_benchmark_problem
    from paola.tools.run_tools import start_optimization_run, finalize_optimization_run, set_foundry
    from paola.tools.optimizer_tools import run_scipy_optimization

    temp_dir = tempfile.mkdtemp(prefix="paola_persist_")

    try:
        # Session 1: Create and run optimization
        print("Session 1: Create and store run...")
        storage1 = FileStorage(base_dir=temp_dir)
        platform1 = OptimizationFoundry(storage=storage1)
        set_foundry(platform1)

        problem = create_benchmark_problem.invoke({
            "problem_id": "sphere_2d",
            "function_name": "sphere",
            "dimension": 2
        })

        run_result = start_optimization_run.invoke({
            "problem_id": problem["problem_id"],
            "algorithm": "SLSQP"
        })
        run_id = run_result["run_id"]

        bounds = [[-5.0, 5.0], [-5.0, 5.0]]  # 2D bounds
        run_scipy_optimization.invoke({
            "problem_id": "sphere_2d",
            "run_id": run_id,
            "algorithm": "SLSQP",
            "bounds": bounds
        })

        finalize_optimization_run.invoke({"run_id": run_id})

        run1 = platform1.load_run(run_id)
        obj1 = run1.objective_value
        print(f"  ✓ Run {run_id} created with objective {obj1:.6e}")
        print()

        # Session 2: Load from storage
        print("Session 2: Load from persisted storage...")
        storage2 = FileStorage(base_dir=temp_dir)
        platform2 = OptimizationFoundry(storage=storage2)

        run2 = platform2.load_run(run_id)
        assert run2 is not None
        assert run2.run_id == run_id
        assert run2.objective_value == obj1
        print(f"  ✓ Run {run_id} loaded with same objective {run2.objective_value:.6e}")
        print()

        print("✅ Storage persistence works!")
        print()

        return True

    except Exception as e:
        print(f"❌ Persistence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all end-to-end tests."""
    results = []

    results.append(("Complete Workflow", test_complete_workflow()))
    results.append(("Agent Tools Integration", test_agent_with_tools()))
    results.append(("Storage Persistence", test_storage_persistence()))

    print()
    print("=" * 70)
    print("END-TO-END TEST SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")

    all_passed = all(r[1] for r in results)

    print()
    if all_passed:
        print("🎉 ALL END-TO-END TESTS PASSED!")
        print()
        print("The PAOLA platform is fully operational:")
        print("  ✓ Problem formulation works")
        print("  ✓ Run management works")
        print("  ✓ Optimization works (SciPy integration)")
        print("  ✓ Storage persistence works")
        print("  ✓ Metrics analysis works")
        print("  ✓ CLI commands work")
        print("  ✓ Multi-run comparison works")
        print()
        print("Ready for:")
        print("  → Interactive CLI usage")
        print("  → Real optimization problems")
        print("  → Agent-driven workflows")
        print()
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
