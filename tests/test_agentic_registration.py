"""
Test the agentic evaluator registration workflow.

This simulates what the agent would do when given the task:
"Register evaluators from test_evaluator.py"
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paola.tools.registration_tools import read_file, write_file, execute_python
import tempfile
import shutil

print("=" * 80)
print("Testing Agentic Evaluator Registration")
print("=" * 80)

# Create temporary directory for output
temp_dir = tempfile.mkdtemp()
print(f"\nUsing temporary directory: {temp_dir}")

try:
    # Step 1: Agent reads the source file
    print("\n1. Agent reads source file (test_evaluator.py)...")
    result = read_file.invoke({"file_path": "test_evaluator.py"})

    if not result["success"]:
        print(f"   ✗ Failed to read file: {result['error']}")
        sys.exit(1)

    print(f"   ✓ Read file ({len(result['contents'])} bytes)")
    print(f"\n   File contents:")
    print("   " + "\n   ".join(result["contents"].split("\n")[:15]))

    # Step 2: Agent analyzes and creates standalone evaluator
    print("\n2. Agent creates standalone evaluator file...")

    # Agent would use LLM reasoning here to understand the function
    # For this test, we simulate what the agent would generate

    evaluator_code = '''"""
Evaluator: rosenbrock_eval
Auto-generated standalone evaluator
Original: test_evaluator.py::rosenbrock_2d
"""

import numpy as np


def rosenbrock_2d(x):
    """
    2D Rosenbrock function.

    Global minimum at (1, 1) with f = 0.
    Typical bounds: [-5, 10] for each variable.
    """
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

    # Write the evaluator file
    eval_file_path = os.path.join(temp_dir, "rosenbrock_eval.py")
    write_result = write_file.invoke({
        "file_path": eval_file_path,
        "content": evaluator_code
    })

    if not write_result["success"]:
        print(f"   ✗ Failed to write file: {write_result['error']}")
        sys.exit(1)

    print(f"   ✓ Created evaluator file: {eval_file_path}")
    print(f"   ✓ Wrote {write_result['bytes_written']} bytes")

    # Step 3: Agent tests the evaluator
    print("\n3. Agent tests the evaluator...")

    test_result = execute_python.invoke({
        "code": f"exec(open('{eval_file_path}').read())"
    })

    if not test_result["success"]:
        print(f"   ✗ Test failed!")
        print(f"   stdout: {test_result.get('stdout', '')}")
        print(f"   stderr: {test_result.get('stderr', '')}")
        sys.exit(1)

    print(f"   ✓ Test passed!")
    print(f"   Output:")
    for line in test_result["stdout"].split("\n"):
        if line.strip():
            print(f"      {line}")

    # Step 4: Verify the evaluator can be imported and used
    print("\n4. Verifying evaluator works standalone...")

    import importlib.util
    spec = importlib.util.spec_from_file_location("rosenbrock_eval", eval_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import numpy as np
    x_test = np.array([2.0, 4.0])
    result = module.evaluate(x_test)
    print(f"   ✓ evaluate([2.0, 4.0]) = {result:.6f}")

    # Step 5: Agent registers evaluator in Foundry
    print("\n5. Agent registers evaluator in Foundry...")

    from paola.tools.registration_tools import foundry_store_evaluator

    register_result = foundry_store_evaluator.invoke({
        "evaluator_id": "rosenbrock_eval",
        "name": "Rosenbrock 2D",
        "file_path": eval_file_path,
        "callable_name": "evaluate",
        "description": "2D Rosenbrock function with global minimum at (1, 1)"
    })

    if not register_result["success"]:
        print(f"   ✗ Registration failed: {register_result['error']}")
        sys.exit(1)

    print(f"   ✓ Registered: {register_result['message']}")

    # Step 6: Verify registration by listing evaluators
    print("\n6. Verifying registration...")

    from paola.tools.registration_tools import foundry_list_evaluators

    list_result = foundry_list_evaluators.invoke({})

    if list_result["success"]:
        evaluator_ids = [e["evaluator_id"] for e in list_result["evaluators"]]
        if "rosenbrock_eval" in evaluator_ids:
            print(f"   ✓ Found 'rosenbrock_eval' in registered evaluators")
        else:
            print(f"   ✗ 'rosenbrock_eval' not found in: {evaluator_ids}")
            sys.exit(1)
    else:
        print(f"   ✗ Failed to list evaluators: {list_result['error']}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("✓ Agentic Registration Workflow Test PASSED")
    print("=" * 80)
    print("\nThe agent successfully:")
    print("  1. Read the source file")
    print("  2. Created a standalone evaluator with:")
    print("     - All necessary imports (numpy)")
    print("     - Function code copied")
    print("     - Standard evaluate(x) interface")
    print("     - Input validation")
    print("     - Self-test capability")
    print("  3. Tested the evaluator")
    print("  4. Verified it works standalone")
    print("  5. Registered evaluator in Foundry")
    print("  6. Verified registration succeeded")
    print("\n✓ Ready for integration with CLI /register_eval command!")

finally:
    # Cleanup
    shutil.rmtree(temp_dir)
    print(f"\nCleaned up: {temp_dir}")
