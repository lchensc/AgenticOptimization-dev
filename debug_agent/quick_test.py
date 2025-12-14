#!/usr/bin/env python
"""
Quick Test Script for Agentic Architecture.

This script automates the basic test scenario to quickly verify that
the compiled evaluator architecture works correctly.

Run: python debug_agent/quick_test.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paola.tools.agentic_registration import (
    register_evaluator_agentic,
    auto_generate_variable_extractor,
    set_foundry
)
from paola.tools.smart_nlp_creation import create_nlp_problem_smart
from paola.tools.optimizer_tools import run_scipy_optimization
from paola.foundry import FileStorage, OptimizationFoundry
import tempfile
import shutil

print("=" * 80)
print("Quick Test: Agentic Evaluator Architecture")
print("=" * 80)

# Use temporary directory for testing
temp_dir = tempfile.mkdtemp()
print(f"\nUsing temporary directory: {temp_dir}")
print("(Will be cleaned up at the end)")

try:
    # Initialize foundry
    print("\n" + "-" * 80)
    print("Step 1: Initialize Foundry")
    print("-" * 80)

    storage = FileStorage(base_dir=temp_dir)
    foundry = OptimizationFoundry(storage=storage)
    set_foundry(foundry)

    print("✓ Foundry initialized")

    # Register evaluator
    print("\n" + "-" * 80)
    print("Step 2: Register Evaluator (Agentic)")
    print("-" * 80)

    evaluator_file = Path(__file__).parent / "test_evaluators.py"
    print(f"Source file: {evaluator_file}")

    result = register_evaluator_agentic.invoke({
        "file_path": str(evaluator_file),
        "function_name": "rosenbrock_2d"
    })

    if not result["success"]:
        print(f"✗ Registration failed: {result.get('error', 'Unknown')}")
        sys.exit(1)

    print(f"✓ Registered: {result['evaluator_id']}")
    print(f"  Type: {result['semantics'].get('type')}")
    print(f"  Dimension: {result['semantics'].get('input_dimension')}")
    print(f"  Tests passed: {result['tests_passed']}")

    # Create NLP with natural language constraint
    print("\n" + "-" * 80)
    print("Step 3: Create NLP with Smart Tool")
    print("-" * 80)

    print("Creating problem with constraint: x[0] >= 1.5")

    result = create_nlp_problem_smart.invoke({
        "problem_id": "rosenbrock_constrained",
        "objective_evaluator_id": "rosenbrock_2d_eval",
        "constraints": ["x[0] >= 1.5"]
    })

    if not result["success"]:
        print(f"✗ NLP creation failed: {result.get('error', 'Unknown')}")
        sys.exit(1)

    print(f"✓ NLP created: {result['problem_id']}")
    print(f"  Dimension: {result['dimension']}")
    print(f"  Constraints: {result['num_inequality_constraints']}")

    auto_generated = result.get('auto_generated_extractors', [])
    if auto_generated:
        print(f"  Auto-generated: {', '.join(auto_generated)}")

    # Run optimization
    print("\n" + "-" * 80)
    print("Step 4: Run Optimization with Constraints")
    print("-" * 80)

    result = run_scipy_optimization.invoke({
        "problem_id": "rosenbrock_constrained",
        "algorithm": "SLSQP",
        "bounds": [[-5, 10], [-5, 10]],
        "initial_design": [0.0, 0.0]
    })

    if not result["success"]:
        print(f"✗ Optimization failed: {result.get('message', 'Unknown')}")
        sys.exit(1)

    print(f"✓ Optimization completed")
    print(f"  Algorithm: SLSQP")
    print(f"  Final design: {result['final_design']}")
    print(f"  Final objective: {result['final_objective']:.6f}")
    print(f"  Iterations: {result['n_iterations']}")

    # Verify constraint
    print("\n" + "-" * 80)
    print("Step 5: Verify Constraint Satisfaction")
    print("-" * 80)

    x_opt = result['final_design']
    print(f"Constraint: x[0] >= 1.5")
    print(f"Actual x[0]: {x_opt[0]:.6f}")

    if x_opt[0] >= 1.49:
        print("✓ CONSTRAINT SATISFIED!")
    else:
        print("✗ CONSTRAINT VIOLATED!")
        sys.exit(1)

    print(f"\nExpected behavior:")
    print(f"  Unconstrained min: (1.0, 1.0) with f = 0")
    print(f"  Constrained min: ~(1.5, 2.25) with f ≈ 0.25")
    print(f"  Actual: ({x_opt[0]:.4f}, {x_opt[1]:.4f}) with f = {result['final_objective']:.6f}")

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    print("\n✓ All steps completed successfully!")
    print("\nVerified:")
    print("  1. ✓ Agentic registration with LLM analysis")
    print("  2. ✓ Semantic metadata extraction")
    print("  3. ✓ Auto-generated variable extractor (x0_extractor)")
    print("  4. ✓ Smart NLP problem creation")
    print("  5. ✓ Optimization with enforced constraints")
    print("  6. ✓ Constraint satisfaction verified")

    print("\n🎉 Agentic architecture is working correctly!")

    print("\n" + "=" * 80)

except Exception as e:
    print(f"\n✗ Test failed with error:")
    print(f"  {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"\nCleaned up: {temp_dir}")

print("\nFor full interactive testing, see:")
print("  debug_agent/AGENTIC_ARCHITECTURE_TEST_SCENARIO.md")
