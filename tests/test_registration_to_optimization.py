"""
Test complete workflow: Registration → Optimization.

This demonstrates the full pipeline:
1. Agent creates standalone evaluator file
2. Agent registers evaluator in Foundry
3. User creates NLP problem using registered evaluator
4. User runs optimization with the problem
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import shutil
import numpy as np

print("=" * 80)
print("Testing Complete Workflow: Registration → Optimization")
print("=" * 80)

# Create temporary directory for evaluator files
temp_dir = tempfile.mkdtemp()
print(f"\nUsing temporary directory: {temp_dir}")

try:
    # ========================================================================
    # PART 1: Agent Registration Workflow
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 1: Agent Registers Evaluator")
    print("=" * 80)

    from paola.tools.registration_tools import write_file, execute_python, foundry_store_evaluator

    # Step 1: Agent creates standalone evaluator file
    print("\n1. Agent creates standalone evaluator file...")

    evaluator_code = '''"""
Rosenbrock 2D function evaluator.

Global minimum at (1, 1) with f = 0.
"""

import numpy as np


def rosenbrock_2d(x):
    """2D Rosenbrock function."""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def evaluate(x):
    """
    Standard evaluator interface.

    Args:
        x: numpy array of shape (2,)

    Returns:
        float: Objective value
    """
    x = np.atleast_1d(x)

    # Validate dimension
    if len(x) != 2:
        raise ValueError(f"Expected 2D input, got {len(x)}D")

    # Evaluate
    result = rosenbrock_2d(x)

    return float(result)


if __name__ == "__main__":
    """Test the evaluator"""
    # Test at known optimum
    x_opt = np.array([1.0, 1.0])
    f_opt = evaluate(x_opt)
    print(f"At optimum (1, 1): f = {f_opt:.6f}")

    # Test at another point
    x_test = np.array([0.0, 0.0])
    f_test = evaluate(x_test)
    print(f"At test point (0, 0): f = {f_test:.6f}")

    print("✓ Evaluator test passed")
'''

    eval_file_path = os.path.join(temp_dir, "rosenbrock_opt.py")
    write_result = write_file.invoke({
        "file_path": eval_file_path,
        "content": evaluator_code
    })

    if not write_result["success"]:
        print(f"   ✗ Failed to create file: {write_result['error']}")
        sys.exit(1)

    print(f"   ✓ Created: {eval_file_path}")

    # Step 2: Agent tests the evaluator
    print("\n2. Agent tests the evaluator...")

    test_result = execute_python.invoke({
        "code": f"exec(open('{eval_file_path}').read())"
    })

    if not test_result["success"]:
        print(f"   ✗ Test failed!")
        print(f"   stdout: {test_result.get('stdout', '')}")
        print(f"   stderr: {test_result.get('stderr', '')}")
        sys.exit(1)

    print(f"   ✓ Test passed!")

    # Step 3: Agent registers evaluator in Foundry
    print("\n3. Agent registers evaluator in Foundry...")

    register_result = foundry_store_evaluator.invoke({
        "evaluator_id": "rosenbrock_opt",
        "name": "Rosenbrock 2D Optimizer",
        "file_path": eval_file_path,
        "callable_name": "evaluate",
        "description": "2D Rosenbrock function for optimization testing"
    })

    if not register_result["success"]:
        print(f"   ✗ Registration failed: {register_result['error']}")
        sys.exit(1)

    print(f"   ✓ {register_result['message']}")

    # ========================================================================
    # PART 2: User Creates Problem and Runs Optimization
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 2: User Creates Problem & Runs Optimization")
    print("=" * 80)

    from paola.tools.evaluator_tools import create_nlp_problem
    from paola.tools.optimizer_tools import run_scipy_optimization
    from paola.tools.run_tools import set_foundry
    from paola.foundry import OptimizationFoundry, FileStorage

    # Initialize foundry for run tools
    storage = FileStorage(base_dir=".paola_data")
    foundry = OptimizationFoundry(storage=storage)
    set_foundry(foundry)

    # Step 4: User creates NLP problem using registered evaluator
    print("\n4. User creates NLP problem...")

    problem_result = create_nlp_problem.invoke({
        "problem_id": "rosenbrock_test",
        "objective_evaluator_id": "rosenbrock_opt",
        "bounds": [[-5.0, 10.0], [-5.0, 10.0]],
        "objective_sense": "minimize",
        "initial_point": [-1.0, 1.0]
    })

    if not problem_result["success"]:
        print(f"   ✗ Problem creation failed: {problem_result['message']}")
        sys.exit(1)

    print(f"   ✓ {problem_result['message']}")

    # Step 5: User runs optimization
    print("\n5. User runs optimization with SLSQP...")

    from paola.tools.run_tools import start_optimization_run

    # Create optimization run
    run_result = start_optimization_run.invoke({
        "problem_id": "rosenbrock_test",
        "algorithm": "SLSQP",
        "max_iterations": 100
    })

    if not run_result["success"]:
        print(f"   ✗ Run creation failed: {run_result['message']}")
        sys.exit(1)

    run_id = run_result["run_id"]
    print(f"   ✓ Created run {run_id}")

    # Run optimization
    opt_result = run_scipy_optimization.invoke({
        "run_id": run_id,
        "problem_id": "rosenbrock_test",
        "algorithm": "SLSQP",
        "initial_guess": [-1.0, 1.0],
        "bounds": [[-5.0, 10.0], [-5.0, 10.0]],
        "max_iterations": 100
    })

    if not opt_result["success"]:
        print(f"   ✗ Optimization failed: {opt_result['message']}")
        sys.exit(1)

    print(f"   ✓ Optimization completed!")

    # Print available result fields
    for key in ['num_iterations', 'final_objective', 'best_design', 'converged']:
        if key in opt_result:
            print(f"      {key}: {opt_result[key]}")

    # Verify we found the optimum (1, 1)
    final_x = np.array(opt_result.get('best_design', opt_result.get('final_design', [0, 0])))
    expected_x = np.array([1.0, 1.0])
    error = np.linalg.norm(final_x - expected_x)

    print(f"\n   Verification:")
    print(f"      Expected optimum: [1.0, 1.0]")
    print(f"      Found optimum:    [{final_x[0]:.4f}, {final_x[1]:.4f}]")
    print(f"      Error:            {error:.6f}")

    if error > 1e-3:
        print(f"   ✗ Optimization did not converge to correct solution!")
        sys.exit(1)

    print(f"   ✓ Optimization converged to correct solution!")

    # ========================================================================
    # SUCCESS
    # ========================================================================
    print("\n" + "=" * 80)
    print("✓ COMPLETE WORKFLOW TEST PASSED")
    print("=" * 80)
    print("\nWorkflow verified:")
    print("  1. ✓ Agent created standalone evaluator file")
    print("  2. ✓ Agent tested evaluator")
    print("  3. ✓ Agent registered evaluator in Foundry")
    print("  4. ✓ User created NLP problem from registered evaluator")
    print("  5. ✓ User ran optimization and found correct solution")
    print("\nThis demonstrates the complete pipeline:")
    print("  Agent Registration → Foundry Storage → Problem Creation → Optimization")
    print("")

finally:
    # Cleanup
    shutil.rmtree(temp_dir)
    print(f"Cleaned up: {temp_dir}")
