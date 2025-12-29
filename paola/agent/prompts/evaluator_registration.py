"""
Minimalistic system prompt for evaluator registration.

Trusts LLM intelligence - no verbose guidance or hand-holding.
"""

EVALUATOR_REGISTRATION_PROMPT = """
Register user's evaluator in Foundry.

Your task: Generate JSON configuration (not Python code).

Target configuration:
{
  "evaluator_id": "unique_id",
  "name": "descriptive_name",
  "source": {
    "type": "python_function",
    "file_path": "path/to/file.py",
    "callable_name": "function_name"
  },
  "interface": {
    "output": {"format": "auto"}
  },
  "capabilities": {
    "observation_gates": true,
    "caching": true
  }
}

Tools available:
- read_file(path) - Read user's code
- write_file(path, content) - Write test script
- bash(command) - Run test (e.g., "python test_eval.py")
- foundry_store_evaluator(config) - Store in Foundry

Process: read → generate config → test → store

Example:
User: "Function rosenbrock in funcs.py"
→ read_file("funcs.py")
→ Generate config with source.callable_name="rosenbrock"
→ Test with FoundryEvaluator
→ Store if successful
"""

# Single example for reference (trust LLM to generalize)
REGISTRATION_EXAMPLE = """
User code (funcs.py):
```python
def rosenbrock(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
```

Configuration:
{
  "evaluator_id": "rosenbrock_eval",
  "name": "rosenbrock",
  "source": {
    "type": "python_function",
    "file_path": "funcs.py",
    "callable_name": "rosenbrock"
  },
  "interface": {
    "output": {"format": "scalar"}
  },
  "capabilities": {
    "observation_gates": true,
    "caching": true
  },
  "performance": {
    "cost_per_eval": 1.0
  }
}

Test:
```python
from paola.foundry import FoundryEvaluator
import numpy as np

evaluator = FoundryEvaluator.from_config(config)
result = evaluator.evaluate(np.array([1.0, 1.0]))
assert result.objectives["objective"] == 0.0
```
"""
