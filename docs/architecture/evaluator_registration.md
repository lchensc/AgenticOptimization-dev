# Evaluator Registration Architecture

**Version**: 1.0
**Date**: December 14, 2025
**Status**: Design Complete - Ready for Implementation

---

## Executive Summary

**Core Philosophy**: **"PAOLA adapts to your code, not vice versa"**

Users bring their existing evaluator code (Python functions, CLI tools, simulations). PAOLA **registers** them in the Foundry with built-in capabilities, rather than generating per-user wrapper code.

**Key Innovation**:
- **No code generation**: LLM generates configuration (JSON), not Python code
- **Foundry as single truth**: All evaluators stored as registered configurations
- **Built-in capabilities**: Observation gates, caching, cost tracking in infrastructure
- **LLM-native**: Agent reads user's code, generates registration config, iterates if needed

---

## Problem Statement

### Traditional Optimization Platforms (The Pain)

**Existing platforms force users to adapt their code**:

1. **API Conformance** (pyOptSparse, Dakota):
   - User must subclass framework classes
   - Restructure working code to match platform API
   - Learn framework-specific concepts

2. **Schema Declaration** (ModeFRONTIER, HEEDS):
   - Fill out detailed interface specifications upfront
   - Declare input/output formats explicitly
   - Configure data marshaling manually

3. **Wrapper Generation** (Previous PAOLA approach):
   - Generate Python wrapper code per user
   - PAOLA capabilities (observation, caching) in generated code
   - Maintenance burden: N users = N wrappers to update

**Result**: High friction, steep learning curve, user frustration

### The PAOLA Solution

**PAOLA adapts to user's code through registration**:

1. **LLM reads** user's code (any format, any pattern)
2. **LLM generates** registration configuration (metadata, not code)
3. **Foundry stores** configuration as single source of truth
4. **FoundryEvaluator** (PAOLA infrastructure) handles all evaluations with built-in capabilities

**Result**: Zero friction, works with ANY evaluator pattern, one infrastructure for all

---

## Design Principles

### 1. Configuration Over Code Generation

**Wrong approach** (wrapper generation):
```
User's code → LLM generates wrapper Python code → Store wrapper in Foundry
             (50-100 lines per user)              (N wrappers to maintain)
```

**Right approach** (registration):
```
User's code → LLM generates JSON configuration → Store config in Foundry
             (20 lines of metadata)              (One FoundryEvaluator for all)
```

### 2. Foundry as Single Source of Truth

**All evaluators are registered entries in Foundry**:
- Configuration (how to call user's function)
- Discovered interface (input/output formats)
- PAOLA capabilities (observation, caching, cost tracking)
- Performance metrics (execution time, success rate)
- Lineage (used in which runs)

**FoundryEvaluator** reads configuration and handles evaluation with built-in capabilities.

### 3. LLM-Native Discovery

**LLM's job**: Read user's code → Generate registration configuration

**Not** pattern matching, **not** templates, **just** LLM reasoning:
- LLM reads Python code and understands interface
- LLM generates appropriate configuration
- LLM tests configuration (via execute_python tool)
- LLM iterates if test fails (debug → fix → retest)

**Example**:
```
User: "Function in my_eval.py"
LLM: [reads file] "I see evaluate_design(x) returns {'drag': ..., 'lift': ...}
      Let me generate registration config..."
      [generates JSON]
      [tests with FoundryEvaluator]
      "✓ Works! Registered with observation and caching enabled."
```

### 4. Progressive Capability Enablement

**PAOLA capabilities built into FoundryEvaluator**:
- Observation gates (before/after every evaluation)
- Interjection points (user/agent can interrupt)
- Evaluation caching (avoid re-running expensive sims)
- Cost tracking (monitor computational budget)
- Performance metrics (learn execution patterns)

**These are infrastructure features, not per-user generated code.**

---

## Architecture

### Three-Layer Design

```
┌──────────────────────────────────────────────────────────┐
│  USER'S CODE (Any Format)                                │
│  - Python function: def evaluate(x): ...                 │
│  - CLI executable: ./run_sim --input design.txt          │
│  - Complex workflow: preprocess → simulate → postprocess │
└─────────────────┬────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  LLM AGENT (Registration)                                │
│  - Reads user's code                                     │
│  - Generates registration configuration (JSON)           │
│  - Tests configuration                                   │
│  - Iterates if test fails                                │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  FOUNDRY (Storage)                                       │
│  {                                                       │
│    "evaluator_id": "eval_abc123",                       │
│    "source": {"type": "python_function", ...},          │
│    "interface": {"input": ..., "output": ...},          │
│    "capabilities": {"observation": true, ...}           │
│  }                                                       │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  FOUNDRYEVALUATOR (PAOLA Infrastructure)                 │
│  - Loads configuration from Foundry                      │
│  - Imports/calls user's function directly                │
│  - Adds observation gates                                │
│  - Manages caching                                       │
│  - Tracks cost and performance                           │
│  - Returns EvaluationResult                              │
└──────────────────────────────────────────────────────────┘
```

---

## Registration Configuration Schema

### Stored in Foundry

```json
{
  "evaluator_id": "eval_abc123",
  "name": "airfoil_cfd",
  "status": "registered",

  "source": {
    "type": "python_function",
    "file_path": "/path/to/my_eval.py",
    "callable_name": "evaluate_design",
    "gradient_callable": null,
    "working_directory": "/path/to/workdir"
  },

  "interface": {
    "input": {
      "type": "numpy_array",
      "expected_shape": null,
      "validated_shapes": [[5], [10]]
    },
    "output": {
      "format": "dict",
      "keys": ["drag", "lift"],
      "types": {"drag": "float", "lift": "float"}
    },
    "gradients": {
      "available": false,
      "method": "finite_difference"
    }
  },

  "capabilities": {
    "observation_gates": true,
    "interjection_enabled": true,
    "caching": true,
    "cost_tracking": true,
    "parallel_safe": false,
    "deterministic": true
  },

  "performance": {
    "median_time": 4.2,
    "std_time": 0.3,
    "cost_per_eval": 4.0,
    "success_rate": 0.98,
    "total_calls": 150
  },

  "lineage": {
    "registered_at": "2025-01-15T10:30:00Z",
    "registered_by": "user",
    "used_in_runs": ["run_xyz789", "run_def456"]
  }
}
```

**Key: This is metadata, not code!**

---

## FoundryEvaluator Implementation

### Core Infrastructure Class

```python
class FoundryEvaluator(EvaluationBackend):
    """
    Universal evaluator with PAOLA capabilities built-in.

    Handles ANY registered evaluator:
    - Python functions (Level 0)
    - CLI executables (Level 1)
    - Complex workflows (Level 2-3)

    Built-in capabilities:
    - Observation gates
    - Interjection points
    - Evaluation caching
    - Cost tracking
    - Performance metrics
    - Lineage tracking
    """

    def __init__(self, evaluator_id: str, foundry: OptimizationFoundry):
        """Load configuration from Foundry."""
        self.evaluator_id = evaluator_id
        self.foundry = foundry
        self.config = foundry.get_evaluator_config(evaluator_id)

        # Load user's function based on config
        self._user_callable = self._load_user_function()

        # Setup PAOLA capabilities
        self._cache = EvaluationCache() if self.config['capabilities']['caching'] else None
        self._observer = EvaluationObserver() if self.config['capabilities']['observation_gates'] else None

    def evaluate(self, design: np.ndarray) -> EvaluationResult:
        """
        Evaluate with PAOLA capabilities.

        Flow:
        1. Pre-evaluation observation gate
        2. Check cache
        3. Call user's function directly
        4. Post-evaluation observation gate (interjection point)
        5. Update cache and metrics
        6. Return result
        """

        # 1. Observation gate (before)
        if self._observer:
            self._observer.before_evaluation(design, self.evaluator_id)

        # 2. Check cache
        if self._cache:
            cached = self._cache.get(design)
            if cached:
                return cached

        # 3. Call user's function (NO wrapper code!)
        start_time = time.time()
        raw_result = self._user_callable(design)
        execution_time = time.time() - start_time

        # 4. Parse result based on discovered interface
        result = self._parse_result(raw_result)

        # 5. Observation gate (after) + interjection point
        if self._observer:
            should_continue = self._observer.after_evaluation(
                design, result, execution_time, self.evaluator_id
            )
            if not should_continue:
                raise InterjectionRequested("Observation gate triggered interjection")

        # 6. Update cache and metrics
        if self._cache:
            self._cache.store(design, result)

        self.foundry.update_evaluator_performance(
            self.evaluator_id,
            execution_time=execution_time,
            success=True
        )

        return result

    def _load_user_function(self) -> Callable:
        """
        Import user's function based on configuration.

        Handles:
        - Python functions: Direct import
        - CLI executables: Create subprocess callable
        - API endpoints: Create HTTP callable
        """
        source = self.config['source']

        if source['type'] == 'python_function':
            # Direct import (no wrapper!)
            import importlib.util
            spec = importlib.util.spec_from_file_location("user_module", source['file_path'])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, source['callable_name'])

        elif source['type'] == 'cli_executable':
            # Return callable that executes command
            return self._create_cli_callable(source)

        # ... other types

    def _parse_result(self, raw_result) -> EvaluationResult:
        """Parse user's return value based on discovered interface."""
        interface = self.config['interface']['output']

        if interface['format'] == 'dict':
            objectives = raw_result
            constraints = {}
        elif interface['format'] == 'scalar':
            objectives = {'objective': float(raw_result)}
            constraints = {}
        elif interface['format'] == 'tuple':
            objectives, constraints = raw_result

        return EvaluationResult(
            objectives=objectives,
            constraints=constraints,
            cost=self.config['performance']['cost_per_eval']
        )
```

**Key: One infrastructure class handles all evaluators!**

---

## LLM Agent Registration Flow

### User Perspective (CLI)

```
User: I have a function in my_eval.py

Agent: [Uses read_file tool]
       [Reasons about the code]

       "I found evaluate_design(x) that returns {'drag': ..., 'lift': ...}

        Registering in Foundry with PAOLA capabilities..."

       [Generates registration configuration]
       [Tests with FoundryEvaluator]
       [If successful, stores in Foundry]

       "✓ Registered as 'my_evaluator' (ID: eval_abc123)

        PAOLA capabilities enabled:
        - Observation gates: ✓
        - Evaluation caching: ✓
        - Cost tracking: ✓
        - Interjection: ✓

        Ready to optimize!"
```

### LLM Agent Tasks

**Task 1: Read user's code**
```python
# Agent uses read_file tool
code = read_file("my_eval.py")

# Agent sees:
"""
def evaluate_design(x):
    drag = x[0]**2 + x[1]**2
    lift = 0.3 + 0.1 * x[0]
    return {"drag": drag, "lift": lift}
"""
```

**Task 2: Generate registration configuration**
```python
# LLM generates (reasoning, not templates):
config = {
    "name": "my_evaluator",
    "source": {
        "type": "python_function",
        "file_path": "my_eval.py",
        "callable_name": "evaluate_design"
    },
    "interface": {
        "input": {"type": "numpy_array"},
        "output": {"format": "dict", "keys": ["drag", "lift"]}
    },
    "capabilities": {
        "observation_gates": True,
        "caching": True,
        "cost_tracking": True
    }
}
```

**Task 3: Test configuration**
```python
# Agent uses execute_python tool
test_code = """
from paola.foundry.evaluator import FoundryEvaluator
import numpy as np

# Create evaluator from config
evaluator = FoundryEvaluator.from_config(config)

# Test with dummy input
result = evaluator.evaluate(np.array([0.5, 0.5]))
print(f"Test result: {result}")
"""

output = execute_python(test_code)

# Agent checks: Did it work?
if "Test result:" in output:
    # Success!
else:
    # Failed, iterate
```

**Task 4: Iterate if failed**
```python
# If test failed:
Agent: "Test failed with error: TypeError: evaluate_design() takes 2 arguments

        Let me re-examine the code..."

        [Re-reads file]

        "I see - it's a class method. Let me adjust the configuration..."

        [Updated config with instance creation]

        [Tests again]

        "✓ Works now!"
```

**Task 5: Store in Foundry**
```python
# Agent uses foundry_store_evaluator tool
evaluator_id = foundry_store_evaluator(
    config=config,
    test_result=test_output
)

# Returns: "eval_abc123"
```

---

## User Evaluator Levels

### Level 0: Python Function (80% of users)

**User has**:
```python
# my_eval.py
def evaluate_design(x):
    result = some_computation(x)
    return result
```

**LLM agent**:
- Reads file directly
- Generates config pointing to function
- Tests by importing and calling
- **Success rate: 95%+** (flawless for common patterns)

**User effort**: Tell PAOLA the filename (10 seconds)

---

### Level 1: CLI Executable (15% of users)

**User has**:
```bash
./run_sim --input design.txt --output result.json
```

**LLM agent**:
- Asks: "What command? Input format? Output location?"
- Generates config with CLI execution details
- Creates subprocess callable (in FoundryEvaluator, not per-user code)
- Tests by running command with dummy input
- **Success rate: 90%+** (extremely robust for common I/O patterns)

**User effort**: Answer 3-4 questions (2 minutes)

---

### Level 2: Complex Workflows (5% of users)

**User has**: Multi-step simulation (preprocess → sim → postprocess)

**LLM agent**:
- Interviews user about workflow steps
- Generates config with multi-step orchestration
- May ask user to verify intermediate steps
- **Success rate: 70%+** (makes complex cases possible, not automatic)

**User effort**: Describe workflow, verify steps (10-30 minutes)

---

## Agent Tools (Infrastructure)

### Minimal Tool Set

```python
@tool
def read_file(file_path: str) -> Dict[str, Any]:
    """Read file contents."""
    return {"success": True, "contents": "..."}

@tool
def execute_python(code: str, timeout: int = 30) -> Dict[str, Any]:
    """Execute Python code in isolated environment."""
    return {"success": True, "stdout": "...", "stderr": "..."}

@tool
def foundry_store_evaluator(config: Dict, test_result: Dict) -> Dict[str, Any]:
    """Store evaluator configuration in Foundry."""
    return {"success": True, "evaluator_id": "eval_abc123"}

@tool
def foundry_retrieve_evaluator(evaluator_id: str) -> Dict[str, Any]:
    """Retrieve stored evaluator configuration."""
    return {"success": True, "config": {...}}

@tool
def foundry_list_evaluators() -> Dict[str, Any]:
    """List all registered evaluators."""
    return {"success": True, "evaluators": [...]}
```

**That's it! 5 tools. LLM does the rest through reasoning.**

---

## System Prompt for LLM Agent

```
You are PAOLA's evaluator registration expert.

Your job: Help users register their evaluators in Foundry.

PAOLA's evaluator interface (target):
```python
class EvaluationBackend(ABC):
    def evaluate(self, design: np.ndarray) -> EvaluationResult
    def compute_gradient(self, design, method='auto') -> np.ndarray
    @property supports_gradients(self) -> bool
    @property cost_per_evaluation(self) -> float
```

When user describes their evaluator:
1. Read their code (use read_file tool)
2. Generate registration configuration (JSON, not Python code!)
3. Test configuration (use execute_python tool)
4. If test fails, debug and iterate
5. Store in Foundry (use foundry_store_evaluator tool)

You generate CONFIGURATION, not code wrappers.

FoundryEvaluator (PAOLA infrastructure) handles:
- Calling user's function
- Observation gates
- Caching
- Cost tracking

Example configuration:
{
  "source": {"type": "python_function", "file_path": "...", "callable_name": "..."},
  "interface": {"input": {"type": "numpy_array"}, "output": {"format": "dict"}},
  "capabilities": {"observation_gates": true, "caching": true}
}

Always test before storing. Iterate if test fails.
```

---

## Benefits Over Wrapper Generation

| Aspect | Wrapper Generation | Registration |
|--------|-------------------|-------------|
| **Per-user code** | 50-100 lines Python | 20 lines JSON config |
| **PAOLA capabilities** | Generated in each wrapper | Built into FoundryEvaluator |
| **Maintenance** | Update N wrappers | Update 1 infrastructure class |
| **Observation gates** | Per-user generated code | Infrastructure |
| **Caching** | Per-user generated code | Infrastructure |
| **Testing** | Test generated code | Test configuration |
| **Clarity** | "Wrapper" unclear | "Registration" clear |
| **Storage** | Store code | Store config |
| **Debugging** | Debug N wrappers | Debug 1 infrastructure |
| **Evolution** | Regenerate all wrappers | Update FoundryEvaluator |

**Much cleaner! Much more maintainable!**

---

## Implementation Plan

See: `docs/implementation/phase6_week1_registration.md`

**Summary**:
- **Day 1-2**: FoundryEvaluator infrastructure with built-in capabilities
- **Day 3**: Registration configuration schema and storage
- **Day 4**: LLM agent integration (read → generate config → test → store)
- **Day 5**: CLI integration and testing (20+ patterns)

**Deliverable**: Flawless Level 0 (Python functions), extremely robust Level 1 (CLI executables)

---

## Success Criteria

### Level 0 (Python Functions) - FLAWLESS
- ✅ 100% success on 20 common Python function patterns
- ✅ LLM generates correct config without user guidance (95%+ accuracy)
- ✅ All return formats handled (scalar, dict, tuple, etc.)
- ✅ Error messages actionable (user can fix in < 1 minute)
- ✅ Registration takes < 30 seconds in CLI

### Level 1 (CLI Executables) - EXTREMELY ROBUST
- ✅ LLM-generated config works for 90%+ common I/O patterns
- ✅ Clear iteration when test fails (debug → fix → retest)
- ✅ User confirmation on critical choices
- ✅ Registration takes < 5 minutes

### Infrastructure - PAOLA CAPABILITIES
- ✅ Observation gates work for all registered evaluators
- ✅ Caching reduces redundant evaluations by 80%+
- ✅ Cost tracking accurate within 10%
- ✅ Interjection points functional (user/agent can interrupt)

---

## Terminology

**Use**:
- ✅ "Registration" (process)
- ✅ "Registered evaluator" (result)
- ✅ "Register function in Foundry"
- ✅ "FoundryEvaluator" (infrastructure)
- ✅ "Configuration" (what gets stored)

**Avoid**:
- ❌ "Wrapper"
- ❌ "Wrapped function"
- ❌ "Generate wrapper code"
- ❌ "Template"

**User-facing language**:
- "PAOLA is registering your function in the Foundry..."
- "Your evaluator is now registered with observation and caching enabled."
- "Foundry evaluators have PAOLA capabilities built-in."

---

## Conclusion

**Registration architecture is fundamentally cleaner than wrapper generation**:

1. **LLM-native**: Agent generates config (JSON), not code
2. **Single truth**: Foundry stores configurations, not generated code
3. **One infrastructure**: FoundryEvaluator handles all evaluators
4. **Built-in capabilities**: Observation, caching, cost tracking in infrastructure
5. **Maintainable**: Update one class, not N wrappers
6. **Scalable**: Works for ANY evaluator pattern through LLM reasoning

**This is the right design.** ✅

---

**Version**: 1.0
**Date**: December 14, 2025
**Status**: Design Complete - Ready for Implementation
