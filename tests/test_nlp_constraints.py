"""
Test NLP constraint handling.

Verifies that inequality constraints are properly enforced during optimization.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paola.tools.evaluator_tools import create_nlp_problem
from paola.tools.optimizer_tools import run_scipy_optimization
from paola.tools.registration_tools import foundry_store_evaluator
from paola.tools.run_tools import set_foundry
from paola.foundry import FileStorage, OptimizationFoundry
import tempfile
import shutil

print("=" * 80)
print("Testing NLP Constraint Handling")
print("=" * 80)

# Create temporary storage
temp_dir = tempfile.mkdtemp()
print(f"\nUsing temporary directory: {temp_dir}")

try:
    # Initialize foundry
    storage = FileStorage(base_dir=temp_dir)
    foundry = OptimizationFoundry(storage=storage)
    set_foundry(foundry)

    # Register Rosenbrock evaluator
    print("\n1. Registering Rosenbrock evaluator...")

    rosenbrock_code = '''
def rosenbrock_2d(x):
    """2D Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
'''

    result = foundry_store_evaluator.invoke({
        "evaluator_id": "rosenbrock_eval",
        "function_name": "rosenbrock_2d",
        "code": rosenbrock_code,
        "config": {
            "description": "2D Rosenbrock function",
            "dimension": 2,
            "outputs": ["objective"],
            "gradient_method": "finite_difference"
        }
    })

    if result["success"]:
        print(f"   ✓ Registered: {result['evaluator_id']}")
    else:
        print(f"   ✗ Failed: {result['message']}")
        sys.exit(1)

    # Create constrained NLP problem: minimize rosenbrock s.t. x[0] >= 2
    print("\n2. Creating constrained NLP problem (x[0] >= 2)...")

    result = create_nlp_problem.invoke({
        "problem_id": "constrained_rosenbrock",
        "objective_evaluator_id": "rosenbrock_eval",
        "bounds": [[-5, 10], [-5, 10]],
        "inequality_constraints": [
            {
                "name": "x0_min",
                "evaluator_id": "rosenbrock_eval",  # Using same evaluator, but will use x[0]
                "type": ">=",
                "value": 2.0
            }
        ]
    })

    # Wait, this won't work - we need an evaluator that returns x[0]
    # Let me create a proper constraint evaluator

    print("   Creating constraint evaluator (returns x[0])...")

    constraint_code = '''
def x0_evaluator(x):
    """Returns first component of design vector."""
    return x[0]
'''

    result = foundry_store_evaluator.invoke({
        "evaluator_id": "x0_eval",
        "function_name": "x0_evaluator",
        "code": constraint_code,
        "config": {
            "description": "Returns x[0]",
            "dimension": 2,
            "outputs": ["objective"],
            "gradient_method": "finite_difference"
        }
    })

    if result["success"]:
        print(f"   ✓ Registered constraint evaluator: {result['evaluator_id']}")
    else:
        print(f"   ✗ Failed: {result['message']}")
        sys.exit(1)

    # Now create the NLP problem properly
    result = create_nlp_problem.invoke({
        "problem_id": "constrained_rosenbrock",
        "objective_evaluator_id": "rosenbrock_eval",
        "bounds": [[-5, 10], [-5, 10]],
        "inequality_constraints": [
            {
                "name": "x0_min",
                "evaluator_id": "x0_eval",
                "type": ">=",
                "value": 2.0
            }
        ]
    })

    if result["success"]:
        print(f"   ✓ Created problem: {result['problem_id']}")
        print(f"   Constraints: {result['num_inequality_constraints']} inequality")
    else:
        print(f"   ✗ Failed: {result['message']}")
        sys.exit(1)

    # Run optimization with SLSQP
    print("\n3. Running SLSQP optimization...")

    result = run_scipy_optimization.invoke({
        "problem_id": "constrained_rosenbrock",
        "algorithm": "SLSQP",
        "bounds": [[-5, 10], [-5, 10]],
        "initial_design": [0.0, 0.0],
        "options": '{"maxiter": 200}'
    })

    if result["success"]:
        print(f"   ✓ Optimization completed")
        print(f"   Final design: {result['final_design']}")
        print(f"   Final objective: {result['final_objective']:.6f}")
        print(f"   Iterations: {result['n_iterations']}")

        # Check constraint satisfaction
        x_opt = result['final_design']
        constraint_value = x_opt[0]

        print(f"\n4. Verifying constraint satisfaction...")
        print(f"   Constraint: x[0] >= 2.0")
        print(f"   Actual x[0]: {constraint_value:.6f}")

        if constraint_value >= 1.99:  # Allow small tolerance
            print(f"   ✓ Constraint satisfied!")

            # Expected solution near x[0] = 2.0, x[1] ≈ 4.0 (since x[1] ≈ x[0]^2 for Rosenbrock)
            print(f"\n   Expected solution: near x[0] = 2.0, x[1] ≈ 4.0")
            print(f"   Actual solution: x[0] = {x_opt[0]:.4f}, x[1] = {x_opt[1]:.4f}")

            # Unconstrained minimum is at (1, 1)
            print(f"\n   Note: Unconstrained minimum is at (1, 1) with f = 0")
            print(f"         Constraint prevents reaching this point")

        else:
            print(f"   ✗ Constraint VIOLATED! x[0] = {constraint_value:.6f} < 2.0")
            sys.exit(1)

    else:
        print(f"   ✗ Optimization failed: {result['message']}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)

finally:
    # Cleanup
    shutil.rmtree(temp_dir)
    print(f"\nCleaned up: {temp_dir}")
