# Evaluator Registration: Pure Agentic Approach

**Date**: 2025-12-14
**Principle**: Let the agent figure it out - just give task + tools!

---

## The Insight

### ❌ What I Was Doing (Overengineered)

```python
def create_semantic_twin():
    # Hardcoded dependency analysis
    analysis = llm_analyze_dependencies(function_code)

    # Hardcoded dependency gathering
    imports = extract_imports(analysis)
    helpers = extract_helpers(analysis)

    # Hardcoded template generation
    twin_code = TEMPLATE.format(
        imports=imports,
        helpers=helpers,
        function=function_code
    )

    # Hardcoded testing
    test_result = test_evaluator(twin_code)

    # Hardcoded fixing
    if not test_result:
        fixed = fix_twin(twin_code, error)
```

**Problem**: Deterministic logic, rigid templates, trying to anticipate every case!

### ✅ What You're Saying (Pure Agentic)

```python
# Just give the agent a task and tools!

agent.invoke({
    "task": """
    Create a standalone evaluator file for function 'rosenbrock_2d'
    from 'evaluators.py' in '.paola_data/evaluators/rosenbrock_eval.py'

    Requirements:
    - Must work standalone (include all dependencies)
    - Standard interface: evaluate(x) → float
    - Must be testable
    """,

    "tools": [
        read_file,
        write_file,
        execute_python
    ]
})
```

**Agent figures out everything using ReAct reasoning!**

---

## Pure Agentic Workflow

### The Task (Simple!)

```
User: /register_eval evaluators.py

Agent receives task:
"Register all evaluator functions from evaluators.py as standalone
evaluators in .paola_data/evaluators/"
```

### The Tools (Minimal!)

```python
# Tool 1: Read files
@tool
def read_file(file_path: str) -> str:
    """Read file contents."""
    with open(file_path) as f:
        return f.read()

# Tool 2: Write files
@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to file."""
    with open(file_path, 'w') as f:
        f.write(content)
    return f"Written to {file_path}"

# Tool 3: Execute Python
@tool
def execute_python(code: str) -> dict:
    """Execute Python code and return output."""
    # (existing implementation)
```

**That's it!** No dependency analysis, no template generation, no hardcoded logic!

### Agent's ReAct Reasoning

```
Agent Thought: I need to register evaluators from evaluators.py

Agent Action: read_file("evaluators.py")

Observation:
def rosenbrock_2d(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

Agent Thought:
I see one function: rosenbrock_2d
- Takes array x
- Returns scalar
- Uses x[0] and x[1], so 2D
- No external imports needed
- Just uses basic operations

I need to create a standalone evaluator file with:
1. The function itself
2. Standard evaluate() wrapper
3. Validation for 2D input

Let me write it:

Agent Action: write_file(
    ".paola_data/evaluators/rosenbrock_eval.py",
    """
import numpy as np

def rosenbrock_2d(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def evaluate(x):
    x = np.atleast_1d(x)
    if len(x) != 2:
        raise ValueError(f"Expected 2D, got {len(x)}D")
    return float(rosenbrock_2d(x))

if __name__ == "__main__":
    x = np.array([1.0, 1.0])
    print(f"Test: f({x}) = {evaluate(x)}")
    """
)

Observation: Written to .paola_data/evaluators/rosenbrock_eval.py

Agent Thought: Now I should test if it works

Agent Action: execute_python(
    "python .paola_data/evaluators/rosenbrock_eval.py"
)

Observation:
Test: f([1. 1.]) = 0.0

Agent Thought:
Perfect! The evaluator works.
f(1,1) = 0 is correct for Rosenbrock.

Agent Response: ✓ Registered rosenbrock_eval successfully!
```

**No hardcoded logic - just pure reasoning!**

---

## Example: Complex Dependencies

### User's File

```python
# evaluators.py
from scipy.optimize import rosen

def scipy_rosenbrock(x):
    return rosen(x)
```

### Agent's ReAct Reasoning

```
Agent Action: read_file("evaluators.py")

Observation:
from scipy.optimize import rosen

def scipy_rosenbrock(x):
    return rosen(x)

Agent Thought:
This function uses scipy.optimize.rosen - external dependency!
The standalone file needs to import this.

Let me create the twin:

Agent Action: write_file(
    ".paola_data/evaluators/scipy_rosenbrock_eval.py",
    """
import numpy as np
from scipy.optimize import rosen

def scipy_rosenbrock(x):
    return rosen(x)

def evaluate(x):
    x = np.atleast_1d(x)
    return float(scipy_rosenbrock(x))

if __name__ == "__main__":
    x = np.array([1.0, 1.0])
    print(f"Test: {evaluate(x)}")
    """
)

Agent Thought: Test it

Agent Action: execute_python(
    "python .paola_data/evaluators/scipy_rosenbrock_eval.py"
)

Observation: Test: 0.0

Agent Thought: Works! scipy was imported correctly.
```

**Agent figured out the scipy dependency on its own!**

---

## Example: Helper Functions

### User's File

```python
# evaluators.py
def compute_penalty(x, bounds):
    penalty = 0.0
    for i, (lb, ub) in enumerate(bounds):
        if x[i] < lb: penalty += (lb - x[i])**2
        if x[i] > ub: penalty += (x[i] - ub)**2
    return penalty

def penalized_rosenbrock(x):
    bounds = [(0, 2), (0, 2)]
    obj = (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    return obj + 1000 * compute_penalty(x, bounds)
```

### Agent's ReAct Reasoning

```
Agent Action: read_file("evaluators.py")

Observation: [sees both functions]

Agent Thought:
penalized_rosenbrock calls compute_penalty.
compute_penalty is a helper function defined in same file.
I need to include BOTH in the standalone file!

Agent Action: write_file(
    ".paola_data/evaluators/penalized_rosenbrock_eval.py",
    """
import numpy as np

# Helper function
def compute_penalty(x, bounds):
    penalty = 0.0
    for i, (lb, ub) in enumerate(bounds):
        if x[i] < lb: penalty += (lb - x[i])**2
        if x[i] > ub: penalty += (x[i] - ub)**2
    return penalty

# Main function
def penalized_rosenbrock(x):
    bounds = [(0, 2), (0, 2)]
    obj = (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    return obj + 1000 * compute_penalty(x, bounds)

# Standard interface
def evaluate(x):
    x = np.atleast_1d(x)
    if len(x) != 2:
        raise ValueError(f"Expected 2D")
    return float(penalized_rosenbrock(x))
    """
)

Agent Action: execute_python("python ...")

Observation: [test output]

Agent Thought: Works! Both functions included correctly.
```

**Agent understood the helper dependency and included it!**

---

## Example: Agent Handles Errors

### First Attempt (With Bug)

```
Agent Action: write_file(
    "rosenbrock_eval.py",
    """
def rosenbrock_2d(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def evaluate(x):
    return rosenbrock_2d(x)  # BUG: No validation!
    """
)

Agent Action: execute_python("python rosenbrock_eval.py")

Observation:
AttributeError: 'float' object has no attribute '__getitem__'

Agent Thought:
Oops! The function expects an array but I didn't convert it.
Let me fix:

Agent Action: write_file(
    "rosenbrock_eval.py",
    """
import numpy as np  # ADDED

def rosenbrock_2d(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def evaluate(x):
    x = np.atleast_1d(x)  # FIXED
    if len(x) != 2:
        raise ValueError("Expected 2D")
    return float(rosenbrock_2d(x))
    """
)

Agent Action: execute_python("python rosenbrock_eval.py")

Observation: Test passed!

Agent Thought: Fixed! Now it works.
```

**Agent debugs and fixes issues on its own!**

---

## The Complete Implementation

### CLI Command

```python
# paola/cli/repl.py

elif cmd == '/register_eval':
    if len(cmd_parts) < 2:
        self.console.print("[red]Usage: /register_eval <file.py>[/red]")
    else:
        # Just send to agent as natural language task!
        task = f"Register all evaluator functions from {cmd_parts[1]} as standalone evaluators in .paola_data/evaluators/"
        self._process_with_agent(task)
```

### Agent Has These Tools

```python
# Agent's tool list (existing + minimal additions)
self.tools = [
    # File operations (EXISTING)
    read_file,
    write_file,
    execute_python,

    # Foundry operations (for metadata)
    foundry_store_evaluator,
    foundry_list_evaluators,

    # That's it! No special registration logic needed!
]
```

### Agent System Prompt (Simple!)

```python
REGISTRATION_GUIDANCE = """
When registering evaluators:

1. Read the source file to understand functions
2. For each function, create a standalone file in .paola_data/evaluators/:
   - Include all dependencies (imports, helpers)
   - Provide standard evaluate(x) interface
   - Add validation
3. Test each file by running it
4. If errors, debug and fix
5. Register in Foundry with metadata

The files must work standalone - no dependencies on user's code!
"""
```

**That's it!** No templates, no hardcoded logic!

---

## Why This Is Better

### My Overengineered Approach
```python
# 500+ lines of:
- Dependency analysis functions
- Import extraction logic
- Helper detection algorithms
- Template generation
- Hardcoded test patterns
- Error handling for every case
```

**Problem**: Rigid, can't handle unexpected cases!

### Pure Agentic Approach
```python
# 20 lines of:
- Task description
- Tools: read, write, execute
- Let agent figure it out
```

**Benefit**: Flexible, handles any case through reasoning!

---

## Comparison Table

| Aspect | Hardcoded Logic | Pure Agentic |
|--------|----------------|--------------|
| **Code lines** | 500+ | ~20 |
| **Handles edge cases** | Must anticipate | Reasons through |
| **Handles new cases** | Need code update | Just works |
| **Debugging** | Hardcoded fixes | Agent adapts |
| **Maintainability** | Complex | Simple |
| **Intelligence** | Deterministic | LLM reasoning |

---

## Example Session

```bash
paola> /register_eval evaluators.py

💭 I need to register evaluators from evaluators.py.
   Let me first read the file.

🔧 read_file("evaluators.py")

💭 I found 1 function: rosenbrock_2d(x)
   - Takes array input
   - Returns scalar
   - 2D function
   - No external dependencies

   I'll create a standalone file with:
   - The function
   - Standard evaluate() wrapper
   - Input validation
   - Test code

🔧 write_file(".paola_data/evaluators/rosenbrock_eval.py", ...)

💭 Now let me test if it works

🔧 execute_python("python .paola_data/evaluators/rosenbrock_eval.py")

✓ Test passed! f(1,1) = 0.0

💭 Perfect! Now I'll register it in Foundry

🔧 foundry_store_evaluator(config={...})

✓ Registered rosenbrock_eval successfully!

You can now use 'rosenbrock_eval' in optimization problems.
```

**All through natural reasoning - no hardcoded logic!**

---

## Summary

**The Fundamental Principle**:

> Don't hardcode the logic - give the agent the task and tools, let it figure out the solution!

**What the agent needs**:
1. **Task**: "Create standalone evaluators from this file"
2. **Tools**: read_file, write_file, execute_python
3. **Guidance**: "Files must be standalone, include dependencies, test them"

**Agent does everything through ReAct**:
- Reads source file
- Understands dependencies
- Creates standalone twin
- Tests it
- Fixes errors
- Registers in Foundry

**Result**: Simple, flexible, handles any case through LLM intelligence!

This is true **agentic design** - minimal hardcoding, maximum reasoning!

Should I implement this pure agentic approach?
