# Complete Evaluator Registration Workflow

## Overview

This document explains how evaluators flow from registration to optimization in PAOLA.

## The Complete Pipeline

```
Agent Registration → Foundry Storage → Problem Creation → Optimization
       ↓                    ↓                  ↓                ↓
    .py file          metadata.json      NLPProblem      scipy.optimize
```

## Step-by-Step Workflow

### 1. Agent Creates Standalone Evaluator File

**What happens:**
- Agent reads user's source file (e.g., `evaluators.py`)
- Agent creates standalone .py file in `.paola_data/evaluators/`
- File must work independently with all dependencies

**Example:**
```python
# .paola_data/evaluators/rosenbrock.py

import numpy as np  # ← Agent includes dependencies

def rosenbrock_2d(x):
    """Original function."""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def evaluate(x):
    """Standard interface for PAOLA."""
    x = np.atleast_1d(x)
    if len(x) != 2:
        raise ValueError(f"Expected 2D input, got {len(x)}D")
    return float(rosenbrock_2d(x))

if __name__ == "__main__":
    # Self-test code
    ...
```

**Agent tools used:**
- `read_file()` - Read source file
- `write_file()` - Create standalone evaluator
- `execute_python()` - Test the evaluator

### 2. Agent Registers Evaluator in Foundry

**What happens:**
- Agent calls `foundry_store_evaluator()` with metadata
- Foundry stores configuration in `.paola_data/evaluators/{id}.json`
- Configuration points to the .py file

**Agent tool call:**
```python
foundry_store_evaluator(
    evaluator_id="rosenbrock_eval",
    name="Rosenbrock 2D",
    file_path=".paola_data/evaluators/rosenbrock.py",
    callable_name="evaluate",
    description="2D Rosenbrock function"
)
```

**What gets stored:**
```json
{
  "evaluator_id": "rosenbrock_eval",
  "name": "Rosenbrock 2D",
  "source": {
    "type": "python_function",
    "file_path": ".paola_data/evaluators/rosenbrock.py",
    "callable_name": "evaluate"
  },
  "interface": { ... },
  "capabilities": { ... },
  "lineage": {
    "registered_at": "2025-01-XX...",
    "registered_by": "agent"
  }
}
```

### 3. User Creates NLP Problem Using Registered Evaluator

**What happens:**
- User (or agent) calls `create_nlp_problem()` with `evaluator_id`
- Function loads evaluator config from Foundry
- Creates `NLPEvaluator` that wraps the registered evaluator

**Example:**
```python
create_nlp_problem(
    problem_id="rosenbrock_opt",
    objective_evaluator_id="rosenbrock_eval",  # ← References registered evaluator
    bounds=[[-5, 10], [-5, 10]],
    objective_sense="minimize"
)
```

**What happens internally:**
```python
# In create_nlp_problem():
# 1. Verify evaluator exists
foundry.get_evaluator_config("rosenbrock_eval")  # ✓ Found

# 2. Create NLP specification
nlp_problem = NLPProblem(
    objective_evaluator_id="rosenbrock_eval",
    ...
)

# 3. Create composite evaluator
nlp_evaluator = NLPEvaluator.from_problem(nlp_problem, foundry)

# 4. Register in problem registry
register_problem("rosenbrock_opt", nlp_evaluator)
```

### 4. NLPEvaluator Loads the Python Function

**What happens:**
- `NLPEvaluator.from_problem()` creates `FoundryEvaluator`
- `FoundryEvaluator.__init__()` loads evaluator config
- `FoundryEvaluator._load_python_function()` imports the module

**Code flow:**
```python
# In NLPEvaluator.from_problem():
objective_eval = FoundryEvaluator(
    evaluator_id="rosenbrock_eval",
    foundry=foundry
)

# In FoundryEvaluator.__init__():
self.config = foundry.get_evaluator_config("rosenbrock_eval")
self._user_callable = self._load_user_function()

# In FoundryEvaluator._load_python_function():
file_path = ".paola_data/evaluators/rosenbrock.py"
callable_name = "evaluate"

# Import module dynamically
spec = importlib.util.spec_from_file_location("user_module", file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Get the callable
return getattr(module, callable_name)  # ← Returns the evaluate() function
```

### 5. Optimization Runs Using the Evaluator

**What happens:**
- `run_scipy_optimization()` gets problem from registry
- Problem is `NLPEvaluator` wrapping `FoundryEvaluator`
- scipy calls `nlp_evaluator.evaluate(x)` → `foundry_evaluator.evaluate(x)` → user's `evaluate(x)`

**Call chain:**
```python
# scipy.optimize.minimize calls:
objective_with_history(x)
  ↓
nlp_evaluator.evaluate(x)  # NLPEvaluator
  ↓
foundry_evaluator.evaluate(x)  # FoundryEvaluator
  ↓
self._user_callable(x)  # Your evaluate() function from .py file
  ↓
rosenbrock_2d(x)  # Your actual function
```

## Key Design Points

### Why Standalone Files?

Agent creates complete, standalone .py files instead of just storing references because:

1. **Semantic Twin**: Agent understands dependencies and includes them
2. **Testable**: File can be run directly for testing
3. **Portable**: No hidden dependencies on source file
4. **Versioned**: Each file is a complete snapshot

### Why Foundry Metadata?

The JSON configuration enables:

1. **Discovery**: List all registered evaluators
2. **Lineage**: Track which runs use which evaluators
3. **Capabilities**: Store PAOLA features (caching, observation gates)
4. **Performance**: Track execution time, success rate
5. **Interface**: Document input/output types

### Why Dynamic Loading?

`FoundryEvaluator` imports functions dynamically because:

1. **No Code Generation**: No wrapper code to maintain
2. **Configuration-Driven**: Everything controlled by config
3. **Extensible**: Can add new evaluator types without code changes
4. **Lazy Loading**: Only load when needed

## Fixing the Registration Issue

### The Problem

Original `foundry_store_evaluator()` expected complex nested dict:

```python
@tool
def foundry_store_evaluator(
    config: Dict[str, Any],  # ← What structure???
    test_result: Dict[str, Any] = {"success": True}
) -> Dict[str, Any]:
    ...
```

Agent didn't know what structure `config` should have, leading to repeated failures.

### The Solution

Simplified API with explicit parameters:

```python
@tool
def foundry_store_evaluator(
    evaluator_id: str,      # ← Clear parameters
    name: str,
    file_path: str,
    callable_name: str = "evaluate",
    description: str = None
) -> Dict[str, Any]:
    """
    Register Python function evaluator in Foundry.

    Example:
        foundry_store_evaluator(
            evaluator_id="rosenbrock_eval",
            name="Rosenbrock 2D",
            file_path=".paola_data/evaluators/rosenbrock.py",
            callable_name="evaluate"
        )
    """
    # Build config internally using convenience function
    config = create_python_function_config(
        evaluator_id=evaluator_id,
        name=name,
        file_path=file_path,
        callable_name=callable_name,
        ...
    )
    ...
```

Now agent can easily call with simple parameters!

## Verification

See `tests/test_registration_to_optimization.py` for complete end-to-end test:

```bash
$ python tests/test_registration_to_optimization.py

✓ COMPLETE WORKFLOW TEST PASSED

Workflow verified:
  1. ✓ Agent created standalone evaluator file
  2. ✓ Agent tested evaluator
  3. ✓ Agent registered evaluator in Foundry
  4. ✓ User created NLP problem from registered evaluator
  5. ✓ User ran optimization and found correct solution

This demonstrates the complete pipeline:
  Agent Registration → Foundry Storage → Problem Creation → Optimization
```

## Summary

The registration workflow enables:

1. **Agent Autonomy**: Agent can register evaluators without hardcoded logic
2. **Semantic Understanding**: Agent creates proper standalone files
3. **Foundry Integration**: Metadata enables discovery and tracking
4. **Seamless Usage**: Registered evaluators work transparently in optimization

This is the foundation for the agent's ability to understand optimization problems by controlling evaluator registration!
