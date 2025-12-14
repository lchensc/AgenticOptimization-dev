# Evaluator Registration: Semantic Twin Design

**Date**: 2025-12-14
**Principle**: Agent creates semantic twin that preserves functionality and dependencies

---

## The Critical Insight

### ❌ Wrong: Blind Code Copying

```python
# User's evaluators.py
import numpy as np
from scipy.optimize import rosen

def my_rosenbrock(x):
    """Uses scipy's Rosenbrock implementation"""
    return rosen(x)
```

**Bad agent behavior** (blind copy):
```python
# .paola_data/evaluators/my_rosenbrock_eval.py

# Agent just copies function code - WRONG!
def my_rosenbrock(x):
    """Uses scipy's Rosenbrock implementation"""
    return rosen(x)  # ← NameError: 'rosen' not defined!

def evaluate(x):
    return my_rosenbrock(x)  # ← Breaks!
```

**Problem**: Loses external dependencies (scipy import), doesn't work!

### ✅ Correct: Semantic Twin Creation

**Smart agent behavior** (semantic understanding):

```python
# .paola_data/evaluators/my_rosenbrock_eval.py
"""
Evaluator: my_rosenbrock_eval
Auto-generated: 2025-12-14
Original: evaluators.py::my_rosenbrock

Semantic twin created by analyzing original function.
Preserves all functionality and dependencies.
"""

import numpy as np
from scipy.optimize import rosen  # ← Agent understands this is needed!


# ========== Semantic Twin ==========

def my_rosenbrock(x):
    """Uses scipy's Rosenbrock implementation"""
    return rosen(x)


# ========== Standard Interface ==========

def evaluate(x):
    """Standard evaluator interface."""
    x = np.atleast_1d(x)
    result = my_rosenbrock(x)
    return float(result)


METADATA = {...}
```

**Agent's reasoning**:
```
1. Read original function
2. Identify: calls rosen() from scipy.optimize
3. Understand: Need to import scipy.optimize.rosen
4. Create twin: Include import + function
5. Result: Works standalone! ✓
```

---

## Semantic Twin Creation Process

### Example 1: External Library Dependency

**User's file**:
```python
# evaluators.py
import numpy as np
from scipy.optimize import rosen

def scipy_rosenbrock(x):
    """Rosenbrock using scipy built-in"""
    return rosen(x)
```

**Agent's analysis** (LLM reasoning):
```
Function: scipy_rosenbrock(x)
Returns: rosen(x)

Analysis:
- Calls external function: rosen
- rosen is from: scipy.optimize (imported at top)
- This is a standard library function

Semantic twin needs:
1. Import: from scipy.optimize import rosen
2. Function: scipy_rosenbrock(x) that calls rosen(x)
3. Wrapper: evaluate(x) with validation
```

**Generated twin**:
```python
"""
Semantic twin of scipy_rosenbrock
Preserves dependency on scipy.optimize.rosen
"""
import numpy as np
from scipy.optimize import rosen  # Dependency preserved


def scipy_rosenbrock(x):
    """Rosenbrock using scipy built-in"""
    return rosen(x)


def evaluate(x):
    x = np.atleast_1d(x)
    result = scipy_rosenbrock(x)
    return float(result)
```

**Test**:
```bash
$ python .paola_data/evaluators/scipy_rosenbrock_eval.py
✓ Works! (scipy imported correctly)
```

---

### Example 2: Helper Function Dependencies

**User's file**:
```python
# evaluators.py
import numpy as np

def compute_penalty(x, bounds):
    """Helper: penalty for out-of-bounds"""
    penalty = 0.0
    for i, (lower, upper) in enumerate(bounds):
        if x[i] < lower:
            penalty += (lower - x[i])**2
        elif x[i] > upper:
            penalty += (x[i] - upper)**2
    return penalty


def penalized_rosenbrock(x):
    """Rosenbrock with penalty for bounds [0, 2]"""
    bounds = [(0, 2), (0, 2)]
    obj = (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    penalty = compute_penalty(x, bounds)
    return obj + 1000 * penalty
```

**Agent's analysis**:
```
Function: penalized_rosenbrock(x)
Dependencies:
- Calls: compute_penalty(x, bounds) - HELPER FUNCTION
- compute_penalty is defined in same file
- Also uses numpy

Semantic twin needs:
1. Import: numpy
2. Helper function: compute_penalty (MUST INCLUDE!)
3. Main function: penalized_rosenbrock
4. Wrapper: evaluate
```

**Generated twin**:
```python
"""
Semantic twin of penalized_rosenbrock
Includes helper function: compute_penalty
"""
import numpy as np


# ========== Helper Functions ==========

def compute_penalty(x, bounds):
    """Helper: penalty for out-of-bounds"""
    penalty = 0.0
    for i, (lower, upper) in enumerate(bounds):
        if x[i] < lower:
            penalty += (lower - x[i])**2
        elif x[i] > upper:
            penalty += (x[i] - upper)**2
    return penalty


# ========== Main Function ==========

def penalized_rosenbrock(x):
    """Rosenbrock with penalty for bounds [0, 2]"""
    bounds = [(0, 2), (0, 2)]
    obj = (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    penalty = compute_penalty(x, bounds)
    return obj + 1000 * penalty


# ========== Standard Interface ==========

def evaluate(x):
    x = np.atleast_1d(x)
    result = penalized_rosenbrock(x)
    return float(result)
```

**Agent included helper function!** ✓

---

### Example 3: Complex Dependencies

**User's file**:
```python
# evaluators.py
import numpy as np
from scipy.linalg import norm
from my_custom_lib import special_transform  # User's custom module


def transformed_objective(x):
    """Objective with custom transformation"""
    # Apply user's custom transformation
    x_transformed = special_transform(x, mode='standard')

    # Compute objective using scipy
    obj = norm(x_transformed - np.ones_like(x))**2

    return obj
```

**Agent's analysis**:
```
Function: transformed_objective(x)

Dependencies:
1. numpy (standard library) ✓
2. scipy.linalg.norm (standard library) ✓
3. my_custom_lib.special_transform (CUSTOM MODULE) ⚠️

Issue: Cannot import my_custom_lib in .paola_data/evaluators/

Options:
A. Ask user to explain special_transform
B. Copy special_transform code if available
C. Warn user about external dependency

Decision: Agent asks user for clarification
```

**Agent asks user**:
```
⚠️  Warning: Function 'transformed_objective' depends on custom module:
    • my_custom_lib.special_transform

This module won't be available in .paola_data/evaluators/

Options:
1. Provide the code for special_transform (I'll include it)
2. Replace with equivalent standard library function
3. Keep dependency and set sys.path (less robust)

What would you like to do?
```

**User provides code**:
```python
def special_transform(x, mode='standard'):
    """My custom transformation"""
    if mode == 'standard':
        return x / np.linalg.norm(x)
    else:
        return x
```

**Generated twin**:
```python
"""
Semantic twin of transformed_objective
Custom dependency 'special_transform' included from user
"""
import numpy as np
from scipy.linalg import norm


# ========== Custom Dependencies (from my_custom_lib) ==========

def special_transform(x, mode='standard'):
    """
    User's custom transformation.
    Original: my_custom_lib.special_transform
    """
    if mode == 'standard':
        return x / np.linalg.norm(x)
    else:
        return x


# ========== Main Function ==========

def transformed_objective(x):
    """Objective with custom transformation"""
    x_transformed = special_transform(x, mode='standard')
    obj = norm(x_transformed - np.ones_like(x))**2
    return obj


# ========== Standard Interface ==========

def evaluate(x):
    x = np.atleast_1d(x)
    result = transformed_objective(x)
    return float(result)
```

**Result**: Works standalone with custom dependency! ✓

---

## Agent's Semantic Analysis Workflow

### Step 1: Parse Function and Identify Dependencies

```python
# Agent uses LLM to analyze
llm_prompt = f"""
Analyze this Python function:

{function_code}

Identify:
1. All imported modules/functions used
2. All helper functions called (defined elsewhere in file)
3. Any custom modules that need special handling
4. The semantic purpose of the function

Return JSON:
{{
  "imports": ["numpy", "scipy.optimize.rosen"],
  "helper_functions": ["compute_penalty"],
  "custom_modules": ["my_custom_lib"],
  "purpose": "Computes penalized Rosenbrock with bounds",
  "dimension": 2
}}
"""

analysis = llm.invoke(llm_prompt)
```

### Step 2: Gather All Dependencies

```python
# Agent gathers code for all dependencies

dependencies = {
    "imports": [],
    "helper_functions": [],
    "custom_code": []
}

# Standard library imports
for import_name in analysis["imports"]:
    if is_standard_library(import_name):
        dependencies["imports"].append(import_name)
    else:
        # Custom module - need to handle specially
        agent_asks_user(import_name)

# Helper functions
for helper_name in analysis["helper_functions"]:
    helper_code = extract_function_code(file_path, helper_name)
    dependencies["helper_functions"].append(helper_code)
```

### Step 3: Generate Semantic Twin

```python
twin_code = f'''
"""
Semantic twin of {function_name}
Generated: {timestamp}
Original: {source_file}::{function_name}
"""

# ========== Imports ==========
{generate_import_statements(dependencies["imports"])}


# ========== Helper Functions ==========
{join_functions(dependencies["helper_functions"])}


# ========== Custom Dependencies ==========
{dependencies["custom_code"]}


# ========== Main Function ==========
{function_code}


# ========== Standard Interface ==========
def evaluate(x):
    """Standard evaluator interface."""
    x = np.atleast_1d(x)

    # Dimension validation
    if len(x) != {dimension}:
        raise ValueError(f"Expected {dimension}D, got {{len(x)}}D")

    # Evaluate
    result = {function_name}(x)

    # Ensure scalar
    return float(result)


# ========== Metadata ==========
METADATA = {{
    "evaluator_id": "{evaluator_id}",
    "dimension": {dimension},
    "dependencies": {analysis["imports"]},
    "semantic_purpose": "{analysis["purpose"]}"
}}
'''
```

### Step 4: Test Semantic Twin

```python
# Agent tests the generated twin
test_code = f"""
import sys
import numpy as np

# Load the twin
exec(open('.paola_data/evaluators/{evaluator_id}.py').read())

# Test with random input
x_test = np.random.rand({dimension})
result = evaluate(x_test)

print(f"Test input: {{x_test}}")
print(f"Test output: {{result}}")
print("✓ Twin works!")
"""

test_result = execute_python(test_code)

if "✓ Twin works!" not in test_result["stdout"]:
    # Twin doesn't work - agent debugging
    agent_debug_and_fix()
```

---

## Agent-User Collaboration Examples

### Scenario 1: Standard Dependencies

```
Agent: Analyzing scipy_rosenbrock...
Agent: Found dependency: scipy.optimize.rosen (standard library)
Agent: Creating semantic twin with scipy dependency...
Agent: ✓ Twin created and tested successfully!
```

**No user input needed** - agent handles standard libraries.

### Scenario 2: Custom Module Detected

```
Agent: Analyzing transformed_objective...
Agent: ⚠️  Found custom module: my_custom_lib.special_transform

This is a custom dependency. I have 3 options:
1. You provide the code for special_transform (I'll include it)
2. Replace with equivalent standard function (if you tell me what it does)
3. Keep import and add sys.path (less portable)

Which option? Or explain what special_transform does?

User: It normalizes x to unit length

Agent: Got it! I'll replace with standard numpy implementation:
  special_transform(x, 'standard') → x / np.linalg.norm(x)

Confirm? (yes/provide code instead)

User: yes

Agent: ✓ Created semantic twin with equivalent standard implementation
```

**Agent understands and adapts** - uses semantic understanding!

### Scenario 3: Complex Helper Functions

```
Agent: Analyzing penalized_rosenbrock...
Agent: Found helper function: compute_penalty
Agent: Extracting compute_penalty from evaluators.py...
Agent: ✓ Including helper in semantic twin
Agent: Testing twin...
Agent: ✓ Twin works! Helper function included.
```

**Agent automatically includes helpers** - no user input needed.

---

## Key Principles

### 1. Semantic Understanding Over Blind Copying

**Agent must understand**:
- What the function computes
- What dependencies it needs
- How to preserve semantics in new context

**Not just copy-paste!**

### 2. Preserve All Dependencies

**Standard libraries**: Include imports automatically
**Helper functions**: Extract and include
**Custom modules**: Ask user for code or equivalent

### 3. Create Working Twin

**Must be testable**:
```bash
python .paola_data/evaluators/my_eval.py
# Should work standalone!
```

### 4. Handle Edge Cases Intelligently

**Missing dependencies**: Ask user
**Complex imports**: Simplify if possible
**Custom code**: Include with attribution

---

## Implementation: LLM-Powered Dependency Analysis

```python
@tool
def create_semantic_twin(
    file_path: str,
    function_name: str,
    evaluator_id: str
) -> Dict[str, Any]:
    """
    Create semantic twin of function with all dependencies.

    Agent uses LLM to:
    1. Analyze function semantically
    2. Identify all dependencies
    3. Gather dependency code
    4. Generate working twin
    5. Test twin
    """

    # Read original file
    source_code = read_file(file_path)
    function_code = extract_function_code(source_code, function_name)

    # LLM analyzes dependencies
    analysis = llm_analyze_dependencies(function_code, source_code)

    # Gather dependencies
    dependencies = gather_dependencies(
        imports=analysis["imports"],
        helpers=analysis["helper_functions"],
        file_path=file_path
    )

    # Handle custom modules (may ask user)
    custom_deps = handle_custom_modules(
        modules=analysis["custom_modules"],
        interactive=True
    )

    # Generate twin
    twin_code = generate_twin(
        function_code=function_code,
        dependencies=dependencies,
        custom_code=custom_deps,
        metadata=analysis
    )

    # Write and test
    twin_path = f".paola_data/evaluators/{evaluator_id}.py"
    write_file(twin_path, twin_code)

    test_result = test_evaluator(twin_path)

    if not test_result["success"]:
        # Agent debugging
        fixed_twin = agent_debug_twin(twin_code, test_result["error"])
        write_file(twin_path, fixed_twin)
        test_result = test_evaluator(twin_path)

    return {
        "success": test_result["success"],
        "evaluator_id": evaluator_id,
        "twin_path": twin_path,
        "dependencies_included": dependencies.keys()
    }
```

---

## Summary

### ❌ Wrong Approach (Blind Copy)
```
Agent: Copy function code → Paste into wrapper → Done
Result: Missing imports, missing helpers → Breaks!
```

### ✅ Correct Approach (Semantic Twin)
```
Agent: Analyze function semantically
     → Identify ALL dependencies (imports, helpers, custom code)
     → Gather dependency code
     → Generate complete twin with all dependencies
     → Test twin standalone
     → Fix issues if any
Result: Working standalone evaluator! ✓
```

### The Key Difference

**Blind copy**: Mechanical, fragile, doesn't understand code
**Semantic twin**: Intelligent, robust, understands semantics

**Example**:
```python
# User's function
from scipy.optimize import rosen
def f(x): return rosen(x)

# Blind copy (WRONG)
def f(x): return rosen(x)  # NameError!

# Semantic twin (CORRECT)
from scipy.optimize import rosen
def f(x): return rosen(x)  # Works! ✓
```

The agent must **understand dependencies and preserve them** in the semantic twin!

This is the robust, intelligent way to create standalone evaluators.
