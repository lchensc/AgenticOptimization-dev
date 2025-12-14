"""
Test basic evaluator compiler functionality (no LLM required).

Verifies:
1. Code extraction and compilation
2. Variable extractor generation
3. Loading compiled evaluators
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paola.foundry.evaluator_compiler import EvaluatorCompiler

print("=" * 80)
print("Testing Evaluator Compiler (Basic)")
print("=" * 80)

# Create temporary directories
temp_dir = tempfile.mkdtemp()
test_file_dir = tempfile.mkdtemp()

print(f"\nTemp data dir: {temp_dir}")
print(f"Test file dir: {test_file_dir}")

try:
    # Create test evaluator file
    print("\n1. Creating test evaluator file...")
    evaluator_file = Path(test_file_dir) / "test.py"
    evaluator_code = '''import numpy as np

def simple_function(x):
    """Simple test function: sum of squares."""
    return float(np.sum(x**2))

def rosenbrock_2d(x):
    """2D Rosenbrock function."""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
'''

    evaluator_file.write_text(evaluator_code)
    print(f"  ✓ Created: {evaluator_file}")

    # Initialize compiler
    print("\n2. Initializing compiler...")
    compiler = EvaluatorCompiler(base_dir=temp_dir)
    print(f"  ✓ Compiler ready")

    # Test 1: Compile simple function
    print("\n3. Compiling simple_function...")
    result = compiler.compile_function(
        source_file=evaluator_file,
        function_name="simple_function",
        evaluator_id="simple_eval"
    )

    if not result["success"]:
        print(f"  ✗ Failed: {result.get('error')}")
        sys.exit(1)

    print(f"  ✓ Compiled successfully")
    print(f"    Source: {result['source_path']}")
    print(f"    Metadata: {result['metadata_path']}")

    # Verify files exist
    source_path = Path(result['source_path'])
    if source_path.exists():
        print(f"  ✓ Source file exists")
        print(f"\n  Source code:")
        print("  " + "\n  ".join(source_path.read_text().split('\n')[:10]) + "\n  ...")
    else:
        print(f"  ✗ Source file not found!")
        sys.exit(1)

    # Test 2: Load and execute compiled evaluator
    print("\n4. Loading compiled evaluator...")
    func = compiler.load_evaluator("simple_eval")

    if func is None:
        print(f"  ✗ Failed to load evaluator")
        sys.exit(1)

    print(f"  ✓ Loaded successfully")

    # Test execution
    print("\n5. Testing execution...")
    test_input = np.array([1.0, 2.0, 3.0])
    result = func(test_input)
    expected = 1.0**2 + 2.0**2 + 3.0**2  # 14.0

    print(f"  Input: {test_input}")
    print(f"  Output: {result}")
    print(f"  Expected: {expected}")

    if abs(result - expected) < 1e-6:
        print(f"  ✓ Execution correct!")
    else:
        print(f"  ✗ Execution incorrect!")
        sys.exit(1)

    # Test 3: Generate variable extractor
    print("\n6. Generating variable extractor...")
    result = compiler.generate_variable_extractor(
        variable_index=0,
        dimension=2
    )

    if not result["success"]:
        print(f"  ✗ Failed: {result.get('error')}")
        sys.exit(1)

    print(f"  ✓ Generated: {result['evaluator_id']}")
    print(f"    Source: {result['source_path']}")

    # Load and test extractor
    print("\n7. Testing variable extractor...")
    extractor = compiler.load_evaluator("x0_extractor")

    if extractor is None:
        print(f"  ✗ Failed to load extractor")
        sys.exit(1)

    test_input = np.array([1.5, 2.5])
    result = extractor(test_input)
    expected = 1.5

    print(f"  Input: {test_input}")
    print(f"  Output: {result}")
    print(f"  Expected: {expected}")

    if abs(result - expected) < 1e-6:
        print(f"  ✓ Extractor works correctly!")
    else:
        print(f"  ✗ Extractor incorrect!")
        sys.exit(1)

    # Test 4: List evaluators
    print("\n8. Listing evaluators...")
    evaluators = compiler.list_evaluators()
    print(f"  Found {len(evaluators)} evaluators:")
    for eval_info in evaluators:
        print(f"    - {eval_info['evaluator_id']}")

    # Test 5: Get metadata
    print("\n9. Reading metadata...")
    metadata = compiler.get_metadata("simple_eval")
    if metadata:
        print(f"  ✓ Metadata loaded")
        print(f"    Origin: {metadata['origin']['original_file']}")
        print(f"    Function: {metadata['origin']['function_name']}")
        print(f"    Created: {metadata['created_at']}")
    else:
        print(f"  ✗ Metadata not found")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Summary: All Basic Tests Passed!")
    print("=" * 80)
    print("\nVerified:")
    print("  ✓ Code extraction and compilation")
    print("  ✓ Immutable snapshot creation")
    print("  ✓ Evaluator loading and execution")
    print("  ✓ Variable extractor generation")
    print("  ✓ Metadata storage and retrieval")
    print("\n" + "=" * 80)

finally:
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    shutil.rmtree(test_file_dir, ignore_errors=True)
    print(f"\nCleaned up temporary directories")
