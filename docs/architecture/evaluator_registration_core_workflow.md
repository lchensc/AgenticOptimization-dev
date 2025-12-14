# Evaluator Registration: Core Workflow (First Principles)

**Date**: 2025-12-14
**Goal**: Define the minimal, essential evaluator registration workflow

---

## The Fundamental Requirements

### What We Actually Need

**User has**: `evaluators.py` containing Python functions
```python
# evaluators.py
def rosenbrock_2d(x):
    """2D Rosenbrock function"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def get_x0(x):
    """Returns first component of design vector"""
    return x[0]

def sphere_2d(x):
    """2D sphere function"""
    return x[0]**2 + x[1]**2
```

**User wants**: Register these functions so they can be used in optimization

**System needs**:
1. Understand what each function does (objective? constraint? dimension?)
2. Create executable evaluator wrappers in `.paola_data/evaluators/`
3. Register metadata in Foundry for agent to query later

---

## Core Workflow (Minimal)

### Step 1: User Command
```bash
paola> /register_eval evaluators.py
```

**What this means**: "Register evaluation functions from this file"

### Step 2: Agent Reads File
```python
# Agent uses read_file tool
file_contents = read_file("evaluators.py")

# Agent sees:
"""
def rosenbrock_2d(x):
    ...
def get_x0(x):
    ...
def sphere_2d(x):
    ...
"""
```

### Step 3: Agent Communicates with User

**Agent's LLM reasoning**:
```
I see 3 functions in evaluators.py:
1. rosenbrock_2d(x) - Returns (1-x[0])² + 100(x[1]-x[0]²)²
   → Looks like objective function, 2D input
2. get_x0(x) - Returns x[0]
   → Looks like variable accessor, probably for constraints
3. sphere_2d(x) - Returns x[0]² + x[1]²
   → Looks like objective function, 2D input

Need to confirm with user and get any missing info.
```

**Agent asks**:
```
I found 3 functions:
  1. rosenbrock_2d - Appears to be objective function (2D → scalar)
  2. get_x0 - Appears to return design variable x[0] (for constraints?)
  3. sphere_2d - Appears to be objective function (2D → scalar)

Which functions do you want to register?
(all/select/cancel)
```

**User responds**: `all`

**Agent asks** (for each function):
```
For rosenbrock_2d:
  • Purpose: objective / constraint / other? [auto-detected: objective]
  • Dimension: [auto-detected: 2]
  • Evaluator ID: [suggested: rosenbrock_eval]

Accept auto-detection? (yes/edit/skip)
```

**User responds**: `yes`

**Agent asks**:
```
For get_x0:
  • Purpose: [auto-detected: constraint/variable accessor]
  • Dimension: [auto-detected: any]
  • Evaluator ID: [suggested: x0_eval]

Accept auto-detection? (yes/edit/skip)
```

**User responds**: `yes`

(Same for sphere_2d...)

### Step 4: Agent Creates Evaluator Files

**Agent creates**: `.paola_data/evaluators/rosenbrock_eval.py`

```python
"""
Auto-generated evaluator wrapper for rosenbrock_2d
Generated: 2025-12-14
Source: evaluators.py
"""

import sys
import numpy as np

# Import original function
sys.path.insert(0, '/path/to')
from evaluators import rosenbrock_2d

def evaluate(x):
    """
    Wrapper for rosenbrock_2d.

    Input: numpy array of shape (2,)
    Output: scalar float
    """
    x = np.atleast_1d(x)

    # Validate dimension
    if len(x) != 2:
        raise ValueError(f"Expected 2D input, got {len(x)}D")

    # Call original function
    result = rosenbrock_2d(x)

    # Ensure scalar output
    return float(result)


# Metadata for agent
METADATA = {
    "evaluator_id": "rosenbrock_eval",
    "source_file": "evaluators.py",
    "source_function": "rosenbrock_2d",
    "dimension": 2,
    "output_type": "scalar",
    "purpose": "objective",
    "registered_at": "2025-12-14T10:30:00"
}
```

**Agent creates**: `.paola_data/evaluators/x0_eval.py`

```python
"""
Auto-generated evaluator wrapper for get_x0
Generated: 2025-12-14
Source: evaluators.py
"""

import sys
import numpy as np

sys.path.insert(0, '/path/to')
from evaluators import get_x0

def evaluate(x):
    """
    Wrapper for get_x0.

    Input: numpy array (any dimension)
    Output: scalar float (value of x[0])
    """
    x = np.atleast_1d(x)
    result = get_x0(x)
    return float(result)


METADATA = {
    "evaluator_id": "x0_eval",
    "source_file": "evaluators.py",
    "source_function": "get_x0",
    "dimension": None,  # Works for any dimension
    "output_type": "scalar",
    "purpose": "constraint",
    "registered_at": "2025-12-14T10:30:00"
}
```

### Step 5: Agent Registers in Foundry

**Agent stores** in `.paola_data/foundry/evaluators.json`:

```json
{
  "evaluators": [
    {
      "evaluator_id": "rosenbrock_eval",
      "name": "Rosenbrock 2D",
      "source": {
        "type": "python_file",
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
        "original_source": "evaluators.py::rosenbrock_2d"
      }
    },
    {
      "evaluator_id": "x0_eval",
      "name": "Variable x[0] Accessor",
      "source": {
        "type": "python_file",
        "file_path": ".paola_data/evaluators/x0_eval.py",
        "callable_name": "evaluate"
      },
      "interface": {
        "input": {
          "type": "numpy_array",
          "dimension": null
        },
        "output": {
          "format": "scalar"
        }
      },
      "metadata": {
        "purpose": "constraint",
        "original_source": "evaluators.py::get_x0"
      }
    }
  ]
}
```

### Step 6: Agent Confirms

```
✓ Registered 3 evaluators:
  • rosenbrock_eval (objective, 2D)
  • x0_eval (constraint, any dimension)
  • sphere_eval (objective, 2D)

Evaluator files created in: .paola_data/evaluators/
Registry updated in: .paola_data/foundry/evaluators.json

You can now use these in optimization problems.
```

---

## Key Design Decisions

### 1. Agent-User Collaboration (Not Fully Automatic)

**Why**: User knows their functions better than LLM can infer
- LLM makes intelligent suggestions (auto-detection)
- User confirms or corrects
- Result: High accuracy, user stays in control

**Example**:
```
Agent: "get_x0 appears to return x[0]. Is this for constraint formulation?"
User: "Yes"
Agent: "OK, registering as constraint evaluator 'x0_eval'"
```

### 2. Wrapper Files in .paola_data/evaluators/

**Why**:
- **Isolation**: User's original file stays untouched
- **Standardization**: All evaluators have same interface (`evaluate(x)`)
- **Portability**: Evaluators can be moved/shared
- **Validation**: Wrappers add dimension checking, type conversion

**Structure**:
```
.paola_data/
├── evaluators/
│   ├── rosenbrock_eval.py    # Wrapper for rosenbrock_2d
│   ├── x0_eval.py             # Wrapper for get_x0
│   └── sphere_eval.py         # Wrapper for sphere_2d
└── foundry/
    └── evaluators.json        # Registry
```

### 3. Standard Evaluator Interface

**All evaluators** expose the same interface:

```python
def evaluate(x: np.ndarray) -> float:
    """
    Standard evaluator interface.

    Args:
        x: Design vector (numpy array)

    Returns:
        float: Evaluation result
    """
    pass
```

**Benefits**:
- Agent doesn't need to know how to call each function
- Easy to swap evaluators
- Consistent error handling

### 4. Metadata in Two Places

**In wrapper file** (`.paola_data/evaluators/rosenbrock_eval.py`):
```python
METADATA = {
    "evaluator_id": "rosenbrock_eval",
    "dimension": 2,
    "purpose": "objective"
}
```

**In Foundry registry** (`.paola_data/foundry/evaluators.json`):
```json
{
  "evaluator_id": "rosenbrock_eval",
  "source": {...},
  "interface": {...}
}
```

**Why both**:
- Wrapper metadata: For standalone use, introspection
- Foundry registry: For agent queries, searching, filtering

---

## The "No Variable Extractor Needed" Insight

### User's Key Point

> "Forget about the variable extractor. That is like due to a weak LLM."

**You're absolutely right!** Here's why:

### Scenario: User wants constraint `x[0] >= 1.5`

**Option A (Complex - Automatic Generation)**:
```
Agent: Detects need for x[0]
Agent: Generates x0_extractor code automatically
Agent: Tests, registers, uses
Problem: Complex, fragile, assumes dimension
```

**Option B (Simple - User-Provided)**:
```
User provides evaluators.py:

def rosenbrock_2d(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

def get_x0(x):
    return x[0]

def get_x1(x):
    return x[1]
```

```
Agent: Registers all functions
User: "Create NLP with rosenbrock_2d as objective and get_x0 >= 1.5"
Agent: Uses get_x0 as constraint evaluator
✓ DONE - No generation needed!
```

### Why This is Better

1. **User provides what they need**: If they want constraints on variables, they write simple accessors
2. **No guessing**: Agent doesn't need to infer or generate
3. **More general**: User can write ANY constraint function, not just `x[i]`
4. **Simpler code**: No automatic generation complexity

### Example: User's Complete File

```python
# evaluators.py - User provides everything needed

# Objective functions
def rosenbrock_2d(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def sphere_2d(x):
    return x[0]**2 + x[1]**2

# Constraint functions
def get_x0(x):
    """Returns first variable (for constraints like x[0] >= 1.5)"""
    return x[0]

def get_x1(x):
    """Returns second variable (for constraints like x[1] <= 2.0)"""
    return x[1]

def sum_xy(x):
    """Returns x[0] + x[1] (for constraints like x+y <= 10)"""
    return x[0] + x[1]
```

**Agent registers all 5 functions**, user can now formulate any constraint!

**Advantages**:
- ✅ User controls exactly what functions exist
- ✅ User can write domain-specific constraints
- ✅ No automatic generation complexity
- ✅ Works for any constraint, not just variable bounds

---

## Complete Workflow Example

### User Session

```bash
paola> /register_eval evaluators.py

Agent: Reading evaluators.py...

Agent: I found 5 functions:
  1. rosenbrock_2d - Objective function (2D → scalar)
  2. sphere_2d - Objective function (2D → scalar)
  3. get_x0 - Returns x[0] (constraint/variable accessor)
  4. get_x1 - Returns x[1] (constraint/variable accessor)
  5. sum_xy - Returns x[0]+x[1] (custom constraint)

Which functions to register? (all/select numbers/cancel)

User: all

Agent: Analyzing each function...

For rosenbrock_2d:
  • Type: objective
  • Dimension: 2
  • ID: rosenbrock_eval
  ✓ Auto-detected

For sphere_2d:
  • Type: objective
  • Dimension: 2
  • ID: sphere_eval
  ✓ Auto-detected

For get_x0:
  • Type: constraint
  • Dimension: any (works for any problem dimension)
  • ID: x0_eval
  ✓ Auto-detected

For get_x1:
  • Type: constraint
  • Dimension: any
  • ID: x1_eval
  ✓ Auto-detected

For sum_xy:
  • Type: constraint (custom)
  • Dimension: 2
  • ID: sum_xy_eval
  ✓ Auto-detected

Confirm registration? (yes/edit/cancel)

User: yes

Agent: Creating evaluator wrappers...
  ✓ .paola_data/evaluators/rosenbrock_eval.py
  ✓ .paola_data/evaluators/sphere_eval.py
  ✓ .paola_data/evaluators/x0_eval.py
  ✓ .paola_data/evaluators/x1_eval.py
  ✓ .paola_data/evaluators/sum_xy_eval.py

Agent: Updating Foundry registry...
  ✓ 5 evaluators registered

✓ Registration complete!

You can now use:
  • rosenbrock_eval, sphere_eval as objectives
  • x0_eval, x1_eval, sum_xy_eval as constraints

Try: "Optimize rosenbrock_eval with constraint x0_eval >= 1.5"
```

### Now User Can Create Constrained Problem

```bash
paola> Create NLP with rosenbrock_eval as objective and x0_eval >= 1.5

Agent: Creating NLP problem...

Agent uses:
  - Objective: rosenbrock_eval ✓ (registered)
  - Constraint: x0_eval >= 1.5 ✓ (registered)

Agent: Problem created!
  • ID: rosenbrock_constrained
  • Objective: minimize rosenbrock_eval
  • Constraints: x0_eval >= 1.5
  • Recommended solvers: SLSQP, trust-constr

Ready to optimize!

paola> Solve with SLSQP

Agent: Running SLSQP...
Applying 1 inequality constraint
✓ Optimization complete!
  • Solution: x = [1.5, 2.25]  ← Satisfies x[0] >= 1.5 ✓
  • Objective: 2.5625
```

**It works!** No automatic generation needed!

---

## Implementation Requirements

### 1. New Command: `/register_eval`

```python
# In paola/cli/repl.py

elif cmd == '/register_eval':
    if len(cmd_parts) < 2:
        self.console.print("[red]Usage: /register_eval <file.py>[/red]")
    else:
        # This is AGENT-DRIVEN, not deterministic!
        self._process_with_agent(f"Register evaluators from {cmd_parts[1]}")
```

**Key difference from `/register`**:
- `/register`: Deterministic, user prompts, no LLM
- `/register_eval`: Agent-driven, LLM analyzes, user confirms

### 2. Agent Tool: `register_evaluators`

```python
@tool
def register_evaluators(
    file_path: str,
    functions: List[str],
    confirmations: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    Register multiple evaluators from a file.

    Args:
        file_path: Path to Python file with functions
        functions: List of function names to register
        confirmations: Dict with user confirmations for each function
            {
                "rosenbrock_2d": {
                    "evaluator_id": "rosenbrock_eval",
                    "purpose": "objective",
                    "dimension": 2
                },
                ...
            }

    Returns:
        {
            "success": True,
            "registered": ["rosenbrock_eval", "x0_eval", ...],
            "files_created": [".paola_data/evaluators/rosenbrock_eval.py", ...]
        }
    """
    # 1. Read original file
    # 2. For each function, create wrapper in .paola_data/evaluators/
    # 3. Update Foundry registry
    # 4. Return results
```

### 3. Wrapper Template

```python
# Template for generated wrapper files
WRAPPER_TEMPLATE = '''
"""
Auto-generated evaluator wrapper for {function_name}
Generated: {timestamp}
Source: {source_file}
"""

import sys
import numpy as np

# Import original function
sys.path.insert(0, '{source_dir}')
from {module_name} import {function_name}

def evaluate(x):
    """
    Wrapper for {function_name}.

    Input: numpy array of shape ({dimension},)
    Output: scalar float
    """
    x = np.atleast_1d(x)

    {validation}

    # Call original function
    result = {function_name}(x)

    # Ensure scalar output
    return float(result)


# Metadata
METADATA = {{
    "evaluator_id": "{evaluator_id}",
    "source_file": "{source_file}",
    "source_function": "{function_name}",
    "dimension": {dimension},
    "output_type": "scalar",
    "purpose": "{purpose}",
    "registered_at": "{timestamp}"
}}
'''
```

### 4. Foundry Storage Schema

```json
{
  "evaluators": [
    {
      "evaluator_id": "rosenbrock_eval",
      "name": "Rosenbrock 2D",
      "source": {
        "type": "python_file",
        "file_path": ".paola_data/evaluators/rosenbrock_eval.py",
        "original_file": "evaluators.py",
        "original_function": "rosenbrock_2d"
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
        "registered_at": "2025-12-14T10:30:00"
      }
    }
  ]
}
```

---

## Advantages of This Approach

### 1. Simplicity
- No automatic code generation complexity
- User provides what they need
- Agent just wraps and registers

### 2. User Control
- User writes functions they understand
- Agent confirms understanding
- User corrects if needed

### 3. Generality
- Works for ANY function, not just variable accessors
- User can write domain-specific constraints
- No limits on complexity

### 4. Robustness
- No LLM inference errors
- User validates everything
- Clear separation: user creates, agent registers

### 5. Maintainability
- Wrapper files are simple, templated
- Easy to debug (just Python files)
- Can inspect .paola_data/evaluators/ directory

---

## Summary

**Core Workflow (4 Steps)**:

1. **User provides**: `evaluators.py` with functions
2. **User commands**: `/register_eval evaluators.py`
3. **Agent-user collaborate**:
   - Agent reads file, detects functions
   - Agent suggests purpose/dimension/ID
   - User confirms or corrects
4. **Agent registers**:
   - Creates wrappers in `.paola_data/evaluators/`
   - Updates Foundry registry
   - Reports success

**Key Insight**: **No automatic variable extractor generation needed!**
- User provides constraint functions (get_x0, get_x1, etc.) if needed
- Agent just registers what user provides
- Simpler, more robust, more general

**Result**: User can now create NLP with constraints like `x0_eval >= 1.5` and it works correctly!

---

## Next Steps

1. Implement `/register_eval` command
2. Implement `register_evaluators` tool
3. Create wrapper template generator
4. Update Foundry to store evaluator metadata
5. Test with user's workflow

This is the **minimal, essential** registration system that enables agentic optimization.
