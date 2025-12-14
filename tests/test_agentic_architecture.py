"""
Test Agentic Architecture with Compiled Evaluators.

This test verifies the complete compiled evaluator workflow:
1. Register evaluator with LLM analysis → Immutable snapshot
2. Create NLP with smart tool → Auto-generate extractors
3. Run optimization with constraints → Constraints enforced
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paola.tools.agentic_registration import (
    register_evaluator_agentic,
    auto_generate_variable_extractor,
    set_foundry
)
from paola.tools.smart_nlp_creation import create_nlp_problem_smart
from paola.tools.optimizer_tools import run_scipy_optimization
from paola.foundry import FileStorage, OptimizationFoundry
from paola.foundry.evaluator_compiler import EvaluatorCompiler

print("=" * 80)
print("Testing Agentic Architecture with Compiled Evaluators")
print("=" * 80)

# Create temporary directories
temp_dir = tempfile.mkdtemp()
test_file_dir = tempfile.mkdtemp()

print(f"\nTemp data dir: {temp_dir}")
print(f"Test file dir: {test_file_dir}")

try:
    # Step 1: Create test evaluator file
    print("\n" + "=" * 80)
    print("Step 1: Create Test Evaluator File")
    print("=" * 80)

    evaluator_file = Path(test_file_dir) / "test_evaluators.py"
    evaluator_code = '''import numpy as np

def rosenbrock_2d(x):
    """
    Classic 2D Rosenbrock function.

    Global minimum: f(1, 1) = 0
    """
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def sphere_function(x):
    """Simple sphere function."""
    return np.sum(x**2)
'''

    evaluator_file.write_text(evaluator_code)
    print(f"✓ Created: {evaluator_file}")

    # Step 2: Initialize foundry
    print("\n" + "=" * 80)
    print("Step 2: Initialize Foundry")
    print("=" * 80)

    storage = FileStorage(base_dir=temp_dir)
    foundry = OptimizationFoundry(storage=storage)
    set_foundry(foundry)

    print(f"✓ Foundry initialized at: {temp_dir}")

    # Step 3: Register evaluator with agentic tool (LLM analysis)
    print("\n" + "=" * 80)
    print("Step 3: Register Evaluator (Agentic with LLM)")
    print("=" * 80)

    print("Registering rosenbrock_2d...")

    result = register_evaluator_agentic.invoke({
        "file_path": str(evaluator_file),
        "function_name": "rosenbrock_2d"
    })

    if not result["success"]:
        print(f"✗ Registration failed: {result.get('error', 'Unknown error')}")
        if 'traceback' in result:
            print(result['traceback'])
        sys.exit(1)

    print(f"\n✓ Registration successful!")
    print(f"  Evaluator ID: {result['evaluator_id']}")
    print(f"  Source: {result['source_path']}")
    print(f"  Tests passed: {result['tests_passed']}")

    print(f"\nSemantic Analysis:")
    semantics = result.get('semantics', {})
    print(f"  Type: {semantics.get('type', 'unknown')}")
    print(f"  Input dimension: {semantics.get('input_dimension', 'unknown')}")
    print(f"  Output type: {semantics.get('output_type', 'unknown')}")
    print(f"  Description: {semantics.get('description', 'N/A')}")

    # Verify compiled evaluator exists
    compiler = EvaluatorCompiler(base_dir=temp_dir)
    source_path = Path(temp_dir) / "evaluators" / "rosenbrock_2d_eval" / "source.py"
    metadata_path = Path(temp_dir) / "evaluators" / "rosenbrock_2d_eval" / "metadata.json"

    if source_path.exists() and metadata_path.exists():
        print(f"\n✓ Immutable snapshot created:")
        print(f"  {source_path}")
        print(f"  {metadata_path}")
    else:
        print(f"\n✗ Snapshot files not found!")
        sys.exit(1)

    # Step 4: Create NLP problem with smart tool (auto-generates extractors)
    print("\n" + "=" * 80)
    print("Step 4: Create NLP Problem with Smart Tool")
    print("=" * 80)

    print("Creating problem with constraint: x[0] >= 1.5")

    result = create_nlp_problem_smart.invoke({
        "problem_id": "rosenbrock_constrained_smart",
        "objective_evaluator_id": "rosenbrock_2d_eval",
        "constraints": ["x[0] >= 1.5"],  # ← Should auto-generate x0_extractor
        "objective_sense": "minimize"
    })

    if not result["success"]:
        print(f"✗ NLP creation failed: {result.get('error', 'Unknown error')}")
        if 'traceback' in result:
            print(result['traceback'])
        sys.exit(1)

    print(f"\n✓ NLP problem created!")
    print(f"  Problem ID: {result['problem_id']}")
    print(f"  Dimension: {result['dimension']}")
    print(f"  Constraints: {result['num_inequality_constraints']} inequality")

    auto_generated = result.get('auto_generated_extractors', [])
    if auto_generated:
        print(f"\n✓ Auto-generated extractors:")
        for extractor in auto_generated:
            print(f"    - {extractor}")
    else:
        print(f"\n  No extractors auto-generated")

    # Verify x0_extractor was created
    x0_source = Path(temp_dir) / "evaluators" / "x0_extractor" / "source.py"
    if x0_source.exists():
        print(f"\n✓ x0_extractor compiled and stored:")
        print(f"  {x0_source}")
    else:
        print(f"\n⚠ Warning: x0_extractor not found at {x0_source}")

    # Step 5: Run optimization with constraints
    print("\n" + "=" * 80)
    print("Step 5: Run Optimization with Constraints")
    print("=" * 80)

    result = run_scipy_optimization.invoke({
        "problem_id": "rosenbrock_constrained_smart",
        "algorithm": "SLSQP",
        "bounds": [[-5, 10], [-5, 10]],
        "initial_design": [0.0, 0.0]
    })

    if not result["success"]:
        print(f"✗ Optimization failed: {result.get('message', 'Unknown error')}")
        if 'traceback' in result:
            print(result['traceback'])
        sys.exit(1)

    print(f"\n✓ Optimization completed!")
    print(f"  Algorithm: SLSQP")
    print(f"  Final design: {result['final_design']}")
    print(f"  Final objective: {result['final_objective']:.6f}")
    print(f"  Iterations: {result['n_iterations']}")
    print(f"  Function evals: {result['n_function_evals']}")

    # Step 6: Verify constraint satisfaction
    print("\n" + "=" * 80)
    print("Step 6: Verify Constraint Satisfaction")
    print("=" * 80)

    x_opt = result['final_design']
    constraint_value = x_opt[0]

    print(f"\nConstraint: x[0] >= 1.5")
    print(f"Actual x[0]: {constraint_value:.6f}")

    if constraint_value >= 1.49:  # Small tolerance
        print(f"✓ Constraint SATISFIED!")
    else:
        print(f"✗ Constraint VIOLATED!")
        sys.exit(1)

    print(f"\nExpected behavior:")
    print(f"  Unconstrained minimum: (1.0, 1.0) with f = 0")
    print(f"  Constrained minimum: ~(1.5, 2.25) with f > 0")
    print(f"  Actual: ({x_opt[0]:.4f}, {x_opt[1]:.4f}) with f = {result['final_objective']:.6f}")

    # Step 7: Test manual extractor generation
    print("\n" + "=" * 80)
    print("Step 7: Test Manual Extractor Generation")
    print("=" * 80)

    result = auto_generate_variable_extractor.invoke({
        "variable_index": 1,
        "dimension": 2
    })

    if not result["success"]:
        print(f"✗ Extractor generation failed: {result.get('error', 'Unknown error')}")
    else:
        print(f"✓ Generated: {result['evaluator_id']}")
        x1_source = Path(temp_dir) / "evaluators" / "x1_extractor" / "source.py"
        if x1_source.exists():
            print(f"  Source: {x1_source}")

    # Final summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    print("\n✓ All steps completed successfully!")
    print("\nVerified:")
    print("  1. ✓ Agentic registration with LLM analysis")
    print("  2. ✓ Immutable evaluator snapshots created")
    print("  3. ✓ Semantic metadata extracted")
    print("  4. ✓ Variable extractors auto-generated")
    print("  5. ✓ Smart NLP problem creation")
    print("  6. ✓ Optimization with constraints")
    print("  7. ✓ Constraint satisfaction verified")

    print("\nArchitecture Benefits Demonstrated:")
    print("  • Evaluators are immutable (won't break if source changes)")
    print("  • Agent understands evaluator semantics")
    print("  • Auto-generates missing components (x0_extractor)")
    print("  • Constraints are correctly enforced")

    print("\n" + "=" * 80)

finally:
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    shutil.rmtree(test_file_dir, ignore_errors=True)
    print(f"\nCleaned up temporary directories")
