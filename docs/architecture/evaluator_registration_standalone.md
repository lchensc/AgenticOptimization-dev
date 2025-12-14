# Evaluator Registration: Standalone Design

**Date**: 2025-12-14
**Principle**: Generated evaluator files must be completely standalone

---

## The Two Critical Requirements

### 1. No Variable Extractors - Pure Example

**User provides**: Just the objective function, nothing else
```python
# evaluators.py
def rosenbrock_2d(x):
    """
    2D Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2

    Global minimum at (1, 1) with f = 0.
    """
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
```

That's it! No x0/x1 accessors, just the function to optimize.

### 2. Generated Files Must Be Standalone

**Problem with import-based approach**:
```python
# .paola_data/evaluators/rosenbrock_eval.py (BAD - NOT STANDALONE)
import sys
sys.path.insert(0, '/path/to/user/code')
from evaluators import rosenbrock_2d  # ← Breaks if user moves/deletes file!

def evaluate(x):
    return float(rosenbrock_2d(x))
```

**Issues**:
- ❌ Depends on user's file location
- ❌ Breaks if user moves/renames evaluators.py
- ❌ Not portable (can't share .paola_data/)
- ❌ Import path issues

**Standalone approach**:
```python
# .paola_data/evaluators/rosenbrock_eval.py (GOOD - STANDALONE)
import numpy as np

# Function code copied directly (no external dependencies!)
def rosenbrock_2d(x):
    """
    2D Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2

    Global minimum at (1, 1) with f = 0.
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

    # Dimension validation
    if len(x) != 2:
        raise ValueError(f"rosenbrock_eval expects 2D input, got {len(x)}D")

    # Call function
    result = rosenbrock_2d(x)

    return float(result)


# Metadata
METADATA = {
    "evaluator_id": "rosenbrock_eval",
    "name": "Rosenbrock 2D",
    "dimension": 2,
    "output_type": "scalar",
    "purpose": "objective",
    "registered_at": "2025-12-14",
    "original_source": "evaluators.py::rosenbrock_2d"
}
```

**Benefits**:
- ✅ **Completely standalone** - no external dependencies
- ✅ User can delete original file - wrapper still works
- ✅ **Portable** - can copy .paola_data/ anywhere
- ✅ **Robust** - no import path issues
- ✅ **Version control** - frozen snapshot of function at registration time

---

## Complete Workflow Example

### User's File (Simple!)

```python
# evaluators.py
def rosenbrock_2d(x):
    """
    2D Rosenbrock function.
    Global minimum at (1, 1) with f = 0.
    """
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
```

### Step 1: User Command

```bash
paola> /register_eval evaluators.py
```

### Step 2: Agent Reads and Analyzes

```
Agent: Reading evaluators.py...

Agent: Found 1 function:
  • rosenbrock_2d(x)
    - Takes array input
    - Returns scalar
    - Docstring mentions: 2D, minimum at (1,1)
    - Appears to be objective function

Auto-detection:
  • Type: objective
  • Dimension: 2
  • Evaluator ID: rosenbrock_eval

Confirm? (yes/edit/cancel)
```

### Step 3: User Confirms

```
User: yes
```

### Step 4: Agent Creates Standalone Wrapper

**Agent extracts function code**:
```python
# Agent uses AST parsing to extract function
import ast

tree = ast.parse(file_contents)
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "rosenbrock_2d":
        function_code = ast.get_source_segment(file_contents, node)
        # function_code = "def rosenbrock_2d(x):\n    ..."
```

**Agent generates standalone file**:

`.paola_data/evaluators/rosenbrock_eval.py`:
```python
"""
Evaluator: rosenbrock_eval
Auto-generated: 2025-12-14 10:30:00
Original source: evaluators.py::rosenbrock_2d

This is a standalone evaluator - no external dependencies.
"""

import numpy as np


# ========== Original Function (copied) ==========

def rosenbrock_2d(x):
    """
    2D Rosenbrock function.
    Global minimum at (1, 1) with f = 0.
    """
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


# ========== Standard Evaluator Interface ==========

def evaluate(x):
    """
    Standard evaluator interface for PAOLA.

    Input: numpy array of shape (2,)
    Output: scalar float

    This function wraps rosenbrock_2d with validation.
    """
    x = np.atleast_1d(x)

    # Validate dimension
    expected_dim = 2
    if len(x) != expected_dim:
        raise ValueError(
            f"rosenbrock_eval expects {expected_dim}D input, "
            f"got {len(x)}D: {x}"
        )

    # Evaluate
    result = rosenbrock_2d(x)

    # Ensure scalar output
    if not np.isscalar(result):
        raise ValueError(f"Expected scalar output, got {type(result)}: {result}")

    return float(result)


# ========== Metadata ==========

METADATA = {
    "evaluator_id": "rosenbrock_eval",
    "name": "Rosenbrock 2D",
    "description": "2D Rosenbrock function. Global minimum at (1, 1) with f = 0.",
    "dimension": 2,
    "output_type": "scalar",
    "purpose": "objective",
    "registered_at": "2025-12-14T10:30:00",
    "original_source": {
        "file": "evaluators.py",
        "function": "rosenbrock_2d"
    }
}


# ========== Testing (optional) ==========

if __name__ == "__main__":
    """Test the evaluator."""

    # Test 1: Known minimum
    x_opt = np.array([1.0, 1.0])
    f_opt = evaluate(x_opt)
    print(f"At optimum (1, 1): f = {f_opt:.6f} (expected: 0.0)")
    assert abs(f_opt) < 1e-10, f"Expected ~0.0, got {f_opt}"

    # Test 2: Random point
    x_test = np.array([0.0, 0.0])
    f_test = evaluate(x_test)
    print(f"At test point (0, 0): f = {f_test:.6f}")

    # Test 3: Dimension validation
    try:
        x_wrong = np.array([1.0])  # 1D instead of 2D
        evaluate(x_wrong)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Dimension validation works: {e}")

    print("\n✓ All tests passed!")
```

**Key features**:
1. **Function code copied** (lines 11-16)
2. **No imports from user code** - completely standalone
3. **Standard interface** `evaluate(x)` (lines 21-42)
4. **Metadata** for agent queries (lines 47-58)
5. **Self-test** capability (lines 63-83)

### Step 5: Agent Updates Registry

`.paola_data/foundry/evaluators.json`:
```json
{
  "evaluators": [
    {
      "evaluator_id": "rosenbrock_eval",
      "name": "Rosenbrock 2D",
      "description": "2D Rosenbrock function. Global minimum at (1, 1) with f = 0.",
      "source": {
        "type": "standalone_python_file",
        "file_path": ".paola_data/evaluators/rosenbrock_eval.py",
        "callable_name": "evaluate"
      },
      "interface": {
        "input": {
          "type": "numpy_array",
          "dimension": 2
        },
        "output": {
          "format": "scalar"
        }
      },
      "metadata": {
        "purpose": "objective",
        "original_source": "evaluators.py::rosenbrock_2d",
        "registered_at": "2025-12-14T10:30:00"
      }
    }
  ]
}
```

### Step 6: Agent Confirms

```
✓ Registered evaluator: rosenbrock_eval

Created standalone file:
  .paola_data/evaluators/rosenbrock_eval.py

Updated registry:
  .paola_data/foundry/evaluators.json

You can now use 'rosenbrock_eval' in optimization problems.

Test the evaluator:
  python .paola_data/evaluators/rosenbrock_eval.py
```

---

## Using the Evaluator

### Direct Test (Standalone!)

```bash
$ python .paola_data/evaluators/rosenbrock_eval.py

At optimum (1, 1): f = 0.000000 (expected: 0.0)
At test point (0, 0): f = 1.000000
Dimension validation works: rosenbrock_eval expects 2D input, got 1D: [1.]

✓ All tests passed!
```

**Works without original evaluators.py!** ✓

### In Optimization

```bash
paola> Optimize rosenbrock_eval with SLSQP

Agent: Creating optimization run...
Agent: Loading evaluator from .paola_data/evaluators/rosenbrock_eval.py
Agent: Running SLSQP...

✓ Optimization complete!
  • Solution: x = [1.0000, 1.0000]
  • Objective: f = 0.0000
  • Iterations: 87
```

**Works even if user deleted evaluators.py!** ✓

---

## Implementation: Code Extraction

### Agent Tool to Extract Function Code

```python
import ast
import inspect

def extract_function_code(file_path: str, function_name: str) -> str:
    """
    Extract complete function code from Python file.

    Args:
        file_path: Path to Python file
        function_name: Name of function to extract

    Returns:
        Complete function code as string

    Example:
        code = extract_function_code("evaluators.py", "rosenbrock_2d")
        # Returns: "def rosenbrock_2d(x):\n    ..."
    """
    # Read file
    with open(file_path, 'r') as f:
        source = f.read()

    # Parse AST
    tree = ast.parse(source)

    # Find function
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # Extract source segment
            function_code = ast.get_source_segment(source, node)
            return function_code

    raise ValueError(f"Function '{function_name}' not found in {file_path}")
```

### Standalone Wrapper Template

```python
STANDALONE_WRAPPER_TEMPLATE = '''"""
Evaluator: {evaluator_id}
Auto-generated: {timestamp}
Original source: {source_file}::{function_name}

This is a standalone evaluator - no external dependencies.
"""

import numpy as np


# ========== Original Function (copied) ==========

{function_code}


# ========== Standard Evaluator Interface ==========

def evaluate(x):
    """
    Standard evaluator interface for PAOLA.

    Input: numpy array of shape ({dimension},)
    Output: scalar float
    """
    x = np.atleast_1d(x)

    # Validate dimension
    if len(x) != {dimension}:
        raise ValueError(
            f"{evaluator_id} expects {dimension}D input, got {{len(x)}}D: {{x}}"
        )

    # Evaluate
    result = {function_name}(x)

    # Ensure scalar output
    if not np.isscalar(result):
        raise ValueError(f"Expected scalar, got {{type(result)}}: {{result}}")

    return float(result)


# ========== Metadata ==========

METADATA = {{
    "evaluator_id": "{evaluator_id}",
    "name": "{name}",
    "description": "{description}",
    "dimension": {dimension},
    "output_type": "scalar",
    "purpose": "{purpose}",
    "registered_at": "{timestamp}",
    "original_source": {{
        "file": "{source_file}",
        "function": "{function_name}"
    }}
}}


# ========== Testing ==========

if __name__ == "__main__":
    """Test the evaluator."""
    import sys

    # Test with sample input
    x_test = np.random.rand({dimension})
    print(f"Testing with x = {{x_test}}")

    try:
        result = evaluate(x_test)
        print(f"Result: f(x) = {{result:.6f}}")
        print("✓ Evaluator works!")
    except Exception as e:
        print(f"✗ Error: {{e}}")
        sys.exit(1)
'''
```

### Agent Registration Function

```python
@tool
def register_standalone_evaluator(
    file_path: str,
    function_name: str,
    evaluator_id: str,
    dimension: int,
    purpose: str = "objective"
) -> Dict[str, Any]:
    """
    Register evaluator by creating standalone wrapper file.

    Args:
        file_path: Original Python file
        function_name: Function to register
        evaluator_id: ID for evaluator (e.g., "rosenbrock_eval")
        dimension: Input dimension
        purpose: "objective" or "constraint"

    Returns:
        {
            "success": True,
            "evaluator_id": "rosenbrock_eval",
            "wrapper_file": ".paola_data/evaluators/rosenbrock_eval.py"
        }
    """
    # 1. Extract function code
    function_code = extract_function_code(file_path, function_name)

    # 2. Generate standalone wrapper
    from datetime import datetime

    wrapper_code = STANDALONE_WRAPPER_TEMPLATE.format(
        evaluator_id=evaluator_id,
        timestamp=datetime.now().isoformat(),
        source_file=file_path,
        function_name=function_name,
        function_code=function_code,
        dimension=dimension,
        name=evaluator_id.replace("_eval", "").replace("_", " ").title(),
        description=f"Evaluator for {function_name}",
        purpose=purpose
    )

    # 3. Write wrapper file
    wrapper_path = f".paola_data/evaluators/{evaluator_id}.py"
    os.makedirs(".paola_data/evaluators", exist_ok=True)

    with open(wrapper_path, 'w') as f:
        f.write(wrapper_code)

    # 4. Test the wrapper
    test_result = execute_python(code=f"python {wrapper_path}")

    if not test_result["success"]:
        return {
            "success": False,
            "error": f"Wrapper test failed: {test_result['stderr']}"
        }

    # 5. Update Foundry registry
    foundry.register_evaluator({
        "evaluator_id": evaluator_id,
        "source": {
            "type": "standalone_python_file",
            "file_path": wrapper_path
        },
        "interface": {
            "input": {"dimension": dimension},
            "output": {"format": "scalar"}
        },
        "metadata": {
            "purpose": purpose,
            "original_source": f"{file_path}::{function_name}"
        }
    })

    return {
        "success": True,
        "evaluator_id": evaluator_id,
        "wrapper_file": wrapper_path,
        "test_output": test_result["stdout"]
    }
```

---

## Advantages of Standalone Design

### 1. Robustness
```
User deletes evaluators.py
→ .paola_data/evaluators/rosenbrock_eval.py still works! ✓

User renames evaluators.py
→ No effect on registered evaluator ✓

User moves project
→ .paola_data/ is portable ✓
```

### 2. Version Control
```
User registers rosenbrock_2d version 1
→ Frozen in .paola_data/evaluators/rosenbrock_eval.py

User modifies rosenbrock_2d in evaluators.py
→ Registered evaluator unchanged (stable)

User wants new version
→ Register again as rosenbrock_v2_eval
```

### 3. Portability
```
User can share .paola_data/ with colleague
→ Everything works standalone ✓

User can archive optimization results
→ Include .paola_data/ for reproducibility ✓
```

### 4. Debugging
```
Issue with evaluator?
→ Inspect .paola_data/evaluators/rosenbrock_eval.py directly
→ Run python rosenbrock_eval.py to test
→ No hidden dependencies ✓
```

---

## Summary

### Key Changes from Previous Design

| Aspect | Previous | Standalone (Correct) |
|--------|----------|---------------------|
| Dependencies | Imports user's file | No external imports |
| Robustness | Breaks if file moved | Always works |
| Portability | Requires user's code | Fully portable |
| Version control | Dynamic (latest code) | Frozen snapshot |
| Testing | Needs user's file | Self-testable |

### The Standalone Principle

**Every generated evaluator file must:**
1. Contain complete function code (no imports from user code)
2. Work independently of original file
3. Be testable on its own: `python rosenbrock_eval.py`
4. Be portable: copy `.paola_data/` anywhere

### Simple Example Revisited

**User provides**:
```python
# evaluators.py
def rosenbrock_2d(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
```

**Agent creates**:
```python
# .paola_data/evaluators/rosenbrock_eval.py
import numpy as np

# Function code COPIED (not imported!)
def rosenbrock_2d(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def evaluate(x):
    x = np.atleast_1d(x)
    if len(x) != 2:
        raise ValueError(f"Expected 2D, got {len(x)}D")
    return float(rosenbrock_2d(x))

METADATA = {...}
```

**Result**: Completely standalone, robust, portable! ✓

This is the **correct architecture** for evaluator registration.
