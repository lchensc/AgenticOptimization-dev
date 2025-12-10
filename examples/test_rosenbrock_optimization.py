"""
Integration test: Test tools work correctly.

This script tests that the optimization tools function properly:
1. Cache tools work (evaluate_function, cache_stats)
2. Optimizer can be created and run via scipy
3. Analytical backends provide correct gradients

NOTE: This does NOT test iteration-level agent control yet.
For that, we need the LLM agent to make decisions at each step.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aopt.backends import Rosenbrock
from aopt.tools import (
    register_problem,
    optimizer_create,
    optimizer_propose,
    optimizer_update,
    evaluate_function,
    compute_gradient,
    cache_stats,
    cache_clear,
    clear_optimizer_registry,
)


def test_rosenbrock_optimization():
    """Test full optimization workflow on 2D Rosenbrock."""
    print("=" * 70)
    print("End-to-End Test: 2D Rosenbrock Optimization")
    print("=" * 70)

    # 1. Setup problem
    print("\n1. Setting up problem...")
    problem_id = "rosenbrock_2d"
    problem = Rosenbrock(dimension=2)
    register_problem(problem_id, problem)
    print(f"   ✓ Registered {problem.name}")
    print(f"   ✓ Known optimum: x* = [1, 1], f* = 0")

    # 2. Create optimizer
    print("\n2. Creating optimizer...")
    result = optimizer_create.invoke({
        "optimizer_id": "test_opt",
        "problem_id": problem_id,
        "algorithm": "SLSQP",
        "bounds": [[-5.0, 10.0], [-5.0, 10.0]],
        "initial_design": [-1.0, 1.0],
        "options": '{"maxiter": 50, "ftol": 1e-9}'
    })
    print(f"   {result['message']}")
    assert result['success'], f"Failed to create optimizer: {result.get('message')}"

    # 3. Optimization loop
    print("\n3. Running optimization loop...")
    max_iterations = 50
    converged = False

    for iteration in range(max_iterations):
        # Propose design
        proposal = optimizer_propose.invoke({"optimizer_id": "test_opt"})
        if not proposal['success']:
            if proposal.get('converged'):
                print(f"\n   ✓ Optimizer converged at iteration {iteration}")
                converged = True
                break
            else:
                print(f"\n   ✗ Error: {proposal['message']}")
                break

        design = proposal['design']

        # Evaluate objective
        eval_result = evaluate_function.invoke({
            "problem_id": problem_id,
            "design": design,
            "use_cache": True
        })
        assert eval_result['success'], f"Evaluation failed: {eval_result['message']}"

        objective = eval_result['objective']
        cache_hit = eval_result['cache_hit']

        # Compute gradient
        grad_result = compute_gradient.invoke({
            "problem_id": problem_id,
            "design": design,
            "method": "analytical",
            "use_cache": True
        })
        assert grad_result['success'], f"Gradient failed: {grad_result['message']}"

        gradient = grad_result['gradient']
        grad_norm = grad_result['gradient_norm']

        # Update optimizer
        update_result = optimizer_update.invoke({
            "optimizer_id": "test_opt",
            "design": design,
            "objective": objective,
            "gradient": gradient
        })
        assert update_result['success'], f"Update failed: {update_result['message']}"

        # Print progress
        cache_indicator = "[CACHE]" if cache_hit else ""
        print(
            f"   Iter {iteration:2d}: "
            f"x = [{design[0]:7.4f}, {design[1]:7.4f}], "
            f"f = {objective:10.6f}, "
            f"|∇f| = {grad_norm:.2e} "
            f"{cache_indicator}"
        )

        # Check convergence
        if update_result['converged']:
            print(f"\n   ✓ Converged: {update_result['reason']}")
            converged = True
            break

    # 4. Verify result
    print("\n4. Verifying result...")
    if not converged:
        print("   ✗ Did not converge within max iterations")
        return False

    # Get final design from last proposal
    final_design = np.array(design)
    final_objective = objective

    # Compare to known optimum
    x_opt, f_opt = problem.get_optimum()
    error_x = np.linalg.norm(final_design - x_opt)
    error_f = abs(final_objective - f_opt)

    print(f"   Final design:   x = [{final_design[0]:.6f}, {final_design[1]:.6f}]")
    print(f"   Final objective:f = {final_objective:.6e}")
    print(f"   Error in x:     ||x - x*|| = {error_x:.2e}")
    print(f"   Error in f:     |f - f*| = {error_f:.2e}")

    # 5. Cache statistics
    print("\n5. Cache statistics...")
    stats = cache_stats()
    print(f"   Cache entries: {stats['total_entries']}")
    print(f"   Hit rate: {stats['hit_rate']:.1%}")
    print(f"   Cost saved: {stats['total_cost_saved']:.1f} units")

    # 6. Evaluate success
    print("\n6. Test result...")
    success = error_x < 1e-3 and error_f < 1e-6
    if success:
        print("   ✓ TEST PASSED - Converged to known optimum!")
    else:
        print("   ✗ TEST FAILED - Did not reach optimum")
        print(f"      (tolerance: ||x - x*|| < 1e-3, |f - f*| < 1e-6)")

    print("\n" + "=" * 70)
    return success


def test_cache_efficiency():
    """Test that cache prevents redundant evaluations."""
    print("\n" + "=" * 70)
    print("Cache Efficiency Test")
    print("=" * 70)

    # Clear previous test
    cache_clear()
    clear_optimizer_registry()

    print("\n1. Evaluating same design 3 times...")
    problem_id = "rosenbrock_2d"
    problem = Rosenbrock(dimension=2)
    register_problem(problem_id, problem)

    design = [0.5, 0.5]

    # First evaluation - should compute
    result1 = evaluate_function.invoke({"problem_id": problem_id, "design": design, "use_cache": True})
    print(f"   Evaluation 1: cache_hit = {result1['cache_hit']}, cost = {result1['cost']}")
    assert not result1['cache_hit'], "First evaluation should not be cache hit"

    # Second evaluation - should hit cache
    result2 = evaluate_function.invoke({"problem_id": problem_id, "design": design, "use_cache": True})
    print(f"   Evaluation 2: cache_hit = {result2['cache_hit']}, cost = {result2['cost']}")
    assert result2['cache_hit'], "Second evaluation should be cache hit"

    # Third evaluation - should hit cache
    result3 = evaluate_function.invoke({"problem_id": problem_id, "design": design, "use_cache": True})
    print(f"   Evaluation 3: cache_hit = {result3['cache_hit']}, cost = {result3['cost']}")
    assert result3['cache_hit'], "Third evaluation should be cache hit"

    print("\n2. Cache statistics...")
    stats = cache_stats()
    print(f"   Total entries: {stats['total_entries']}")
    print(f"   Hit rate: {stats['hit_rate']:.1%}")
    print(f"   Cost saved: {stats['total_cost_saved']:.1f} units")

    assert stats['total_entries'] == 1, "Should have 1 unique entry"
    # Note: hit_rate tracking not yet implemented in cache_stats

    print("\n   ✓ Cache working correctly!")
    print("=" * 70)


if __name__ == "__main__":
    print("\nRunning integration tests...\n")

    # Test 1: Cache efficiency
    test_cache_efficiency()

    # Test 2: Full optimization
    success = test_rosenbrock_optimization()

    if success:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Tests failed")
        sys.exit(1)
