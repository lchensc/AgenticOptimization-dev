# Phase 6 Week 1: Evaluator Registration Implementation Plan

**Goal**: Flawless Level 0 (Python functions), Extremely Robust Level 1 (CLI executables)

**Duration**: 5 days

**Date**: December 14, 2025

---

## Overview

Implement LLM-native evaluator registration system based on configuration (not code generation).

**Core deliverables**:
1. FoundryEvaluator infrastructure with PAOLA capabilities
2. Registration configuration schema and storage
3. LLM agent for registration (read → generate config → test → iterate)
4. CLI integration with natural language interface
5. Comprehensive testing (20+ Python patterns, 10+ CLI patterns)

---

## Day 1: FoundryEvaluator Infrastructure

### Morning: Core Class Structure

**File**: `paola/foundry/evaluator.py` (NEW)

**Implementation**:

```python
class FoundryEvaluator(EvaluationBackend):
    """
    Universal evaluator with PAOLA capabilities built-in.

    Handles ANY registered evaluator through configuration.
    """

    def __init__(self, evaluator_id: str, foundry: OptimizationFoundry):
        """Load configuration and setup capabilities."""

    def evaluate(self, design: np.ndarray) -> EvaluationResult:
        """
        Evaluate with full PAOLA capabilities:
        1. Pre-evaluation observation gate
        2. Cache check
        3. Call user's function
        4. Post-evaluation observation gate + interjection point
        5. Update cache and metrics
        """

    def _load_user_function(self) -> Callable:
        """
        Import user's function based on configuration.

        Handles:
        - Python functions (direct import)
        - CLI executables (subprocess callable)
        - Future: API endpoints, workflows
        """

    def _parse_result(self, raw_result) -> EvaluationResult:
        """Parse based on discovered interface in config."""

    def compute_gradient(self, design, method='auto'):
        """User gradients or finite difference."""
```

**Tests**: `test_foundry_evaluator.py`
- Create evaluator from config
- Call simple Python function
- Verify result parsing
- Test error handling

**Deliverable**: FoundryEvaluator can call Python functions ✅

---

### Afternoon: PAOLA Capabilities

**File**: `paola/foundry/capabilities.py` (NEW)

**Implementation**:

```python
class EvaluationObserver:
    """Observation gates for monitoring evaluations."""

    def before_evaluation(self, design, evaluator_id):
        """Called before every evaluation."""
        # Log design
        # Check for issues (out of bounds, similar to failed previous)

    def after_evaluation(self, design, result, time, evaluator_id) -> bool:
        """
        Called after every evaluation.

        Returns:
            True: Continue normally
            False: Trigger interjection
        """
        # Log result and time
        # Check for anomalies
        # Return whether to continue or request interjection

class EvaluationCache:
    """Cache for expensive evaluations."""

    def get(self, design: np.ndarray) -> Optional[EvaluationResult]:
        """Check cache for design (with tolerance)."""

    def store(self, design: np.ndarray, result: EvaluationResult):
        """Store result in cache."""

class InterjectionRequested(Exception):
    """Exception raised when observation gate triggers interjection."""
```

**Integration**: Add to FoundryEvaluator

**Tests**:
- Observer logs evaluations
- Cache hit/miss
- Interjection triggering

**Deliverable**: PAOLA capabilities working ✅

---

## Day 2: Configuration Schema and Storage

### Morning: Configuration Schema

**File**: `paola/foundry/schemas.py` (NEW or add to existing)

**Implementation**:

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class EvaluatorSource(BaseModel):
    """Where the evaluator comes from."""
    type: str = Field(..., description="python_function, cli_executable, api_endpoint")
    file_path: Optional[str] = None
    callable_name: Optional[str] = None
    gradient_callable: Optional[str] = None
    working_directory: Optional[str] = None
    # For CLI executables:
    command: Optional[str] = None
    input_file: Optional[str] = None
    output_file: Optional[str] = None
    input_format: Optional[str] = None  # text, json, csv
    output_format: Optional[str] = None

class EvaluatorInterface(BaseModel):
    """Discovered interface specification."""
    input: Dict[str, Any]  # type, shape, validated shapes
    output: Dict[str, Any]  # format, keys, types
    gradients: Dict[str, Any]  # available, method

class EvaluatorCapabilities(BaseModel):
    """PAOLA capabilities for this evaluator."""
    observation_gates: bool = True
    interjection_enabled: bool = True
    caching: bool = True
    cost_tracking: bool = True
    parallel_safe: bool = False
    deterministic: bool = True

class EvaluatorPerformance(BaseModel):
    """Performance metrics (learned over time)."""
    median_time: Optional[float] = None
    std_time: Optional[float] = None
    cost_per_eval: float = 1.0
    success_rate: float = 1.0
    total_calls: int = 0

class EvaluatorConfig(BaseModel):
    """Complete evaluator configuration."""
    evaluator_id: str
    name: str
    status: str = "registered"  # registered, validated, active, failed
    source: EvaluatorSource
    interface: EvaluatorInterface
    capabilities: EvaluatorCapabilities
    performance: EvaluatorPerformance
    lineage: Dict[str, Any] = {}
```

**Tests**:
- Pydantic validation
- Serialize/deserialize
- Schema evolution

**Deliverable**: Configuration schema validated ✅

---

### Afternoon: Foundry Storage

**File**: `paola/foundry/evaluator_storage.py` (NEW)

**Implementation**:

```python
class EvaluatorStorage:
    """Storage and retrieval of evaluator configurations."""

    def __init__(self, foundry: OptimizationFoundry):
        """Initialize with foundry instance."""

    def store_evaluator(self, config: EvaluatorConfig) -> str:
        """
        Store evaluator configuration.

        Returns: evaluator_id
        """

    def retrieve_evaluator(self, evaluator_id: str) -> EvaluatorConfig:
        """Retrieve configuration by ID."""

    def list_evaluators(
        self,
        evaluator_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[EvaluatorConfig]:
        """List evaluators with filters."""

    def update_performance(
        self,
        evaluator_id: str,
        execution_time: float,
        success: bool
    ):
        """Update performance statistics after evaluation."""
```

**Integration**: Connect to Foundry database

**Tests**:
- Store and retrieve
- List with filters
- Performance updates

**Deliverable**: Storage working ✅

---

## Day 3: LLM Agent Registration

### Morning: Agent Tools

**File**: `paola/tools/evaluator_tools.py` (NEW)

**Implementation**:

```python
@tool
def read_file(file_path: str) -> Dict[str, Any]:
    """
    Read file contents.

    Returns:
        {
            "success": True,
            "contents": "...",
            "file_type": "python"
        }
    """

@tool
def execute_python(code: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute Python code in isolated environment.

    Returns:
        {
            "success": True/False,
            "stdout": "...",
            "stderr": "...",
            "error": "..." (if failed)
        }
    """

@tool
def foundry_store_evaluator(
    config: Dict[str, Any],
    test_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Store evaluator configuration in Foundry.

    Returns:
        {
            "success": True,
            "evaluator_id": "eval_abc123"
        }
    """

@tool
def foundry_retrieve_evaluator(evaluator_id: str) -> Dict[str, Any]:
    """Retrieve stored evaluator configuration."""

@tool
def foundry_list_evaluators() -> Dict[str, Any]:
    """List all registered evaluators."""
```

**Tests**:
- Each tool works independently
- Error handling
- Timeout handling

**Deliverable**: Agent tools functional ✅

---

### Afternoon: Agent System Prompt

**File**: `paola/agent/prompts/evaluator_registration.py` (NEW)

**Implementation**:

```python
EVALUATOR_REGISTRATION_PROMPT = """
You are PAOLA's evaluator registration expert.

Your job: Help users register their evaluators in the Foundry.

PAOLA's evaluator interface (target for all registered evaluators):

```python
class EvaluationBackend(ABC):
    @abstractmethod
    def evaluate(self, design: np.ndarray) -> EvaluationResult:
        \"\"\"Evaluate objective and constraints.\"\"\"

    @abstractmethod
    def compute_gradient(self, design, method='auto') -> np.ndarray:
        \"\"\"Compute gradient.\"\"\"

    @property
    def supports_gradients(self) -> bool:
        \"\"\"Whether gradients are available.\"\"\"

    @property
    def cost_per_evaluation(self) -> float:
        \"\"\"Computational cost.\"\"\"
```

When user describes their evaluator:
1. Read their code using read_file tool
2. Generate registration configuration (JSON, NOT Python code!)
3. Test configuration using execute_python tool
4. If test fails, debug and iterate
5. Store in Foundry using foundry_store_evaluator tool

You generate CONFIGURATION, not wrapper code!

Configuration schema:
{
  "name": "evaluator_name",
  "source": {
    "type": "python_function",
    "file_path": "path/to/file.py",
    "callable_name": "function_name"
  },
  "interface": {
    "input": {"type": "numpy_array"},
    "output": {"format": "dict", "keys": ["objective_name"]}
  },
  "capabilities": {
    "observation_gates": true,
    "caching": true,
    "cost_tracking": true
  }
}

FoundryEvaluator (PAOLA infrastructure) handles:
- Calling user's function directly
- Observation gates (before/after every evaluation)
- Caching (avoid redundant expensive evaluations)
- Cost tracking
- Interjection points (user/agent can interrupt)

Example registration flow:

User: "I have a function in my_eval.py"

You:
1. read_file("my_eval.py")
2. Examine code, identify evaluator function
3. Generate configuration based on function signature and return type
4. Test with execute_python:
   ```python
   from paola.foundry.evaluator import FoundryEvaluator
   import numpy as np

   config = {...}
   evaluator = FoundryEvaluator.from_config(config)
   result = evaluator.evaluate(np.array([0.5, 0.5]))
   print(f"Test: {result}")
   ```
5. If test succeeds, foundry_store_evaluator(config, test_result)
6. If test fails, debug and iterate

Always test before storing. Iterate until it works.

Common patterns to handle:
- Function returns scalar: wrap as {"objective": value}
- Function returns dict: use as objectives
- Function returns tuple: (objectives, constraints)
- Class method: need to instantiate class first
- Multiple functions in file: ask user which one
"""

EVALUATOR_REGISTRATION_EXAMPLES = """
Example 1: Simple function returning scalar
--------------------------------------------
User code:
```python
def rosenbrock(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
```

Configuration:
{
  "source": {"type": "python_function", "file_path": "funcs.py", "callable_name": "rosenbrock"},
  "interface": {"output": {"format": "scalar"}},
  "capabilities": {"observation_gates": true, "caching": true}
}

Example 2: Function returning dict
-----------------------------------
User code:
```python
def cfd_sim(x):
    drag = x[0]**2 + x[1]**2
    lift = 0.3 + 0.1*x[0]
    return {"drag": drag, "lift": lift}
```

Configuration:
{
  "source": {"type": "python_function", "file_path": "sim.py", "callable_name": "cfd_sim"},
  "interface": {"output": {"format": "dict", "keys": ["drag", "lift"]}},
  "capabilities": {"observation_gates": true, "caching": true}
}

Example 3: CLI executable
--------------------------
User description: "Run ./run_sim, reads design from input.txt, writes result to output.json"

Configuration:
{
  "source": {
    "type": "cli_executable",
    "command": "./run_sim",
    "input_file": "input.txt",
    "input_format": "text",
    "output_file": "output.json",
    "output_format": "json"
  },
  "interface": {"output": {"format": "json"}},
  "capabilities": {"observation_gates": true, "caching": true}
}
"""
```

**Deliverable**: System prompt ready ✅

---

## Day 4: CLI Integration

### Morning: Registration Flow in CLI

**File**: `paola/cli/evaluator_registration.py` (NEW)

**Implementation**:

```python
class EvaluatorRegistrationHandler:
    """Handle evaluator registration in CLI."""

    def __init__(self, agent, foundry):
        self.agent = agent
        self.foundry = foundry

    async def register_evaluator(self, user_message: str):
        """
        Handle evaluator registration from user message.

        Flow:
        1. Agent reads user's code
        2. Generates configuration
        3. Tests configuration
        4. Iterates if needed
        5. Stores in Foundry
        6. Confirms with user
        """

        # Inject registration prompt into agent context
        prompt = EVALUATOR_REGISTRATION_PROMPT + EVALUATOR_REGISTRATION_EXAMPLES

        # Agent processes with registration tools
        response = await self.agent.run(
            user_message,
            system_prompt=prompt,
            tools=[
                read_file,
                execute_python,
                foundry_store_evaluator,
                foundry_list_evaluators
            ]
        )

        return response
```

**Integration**: Add to CLI command handling

**Tests**:
- User says "I have a function in my_eval.py"
- Agent registers correctly
- Confirmation shown to user

**Deliverable**: CLI registration working ✅

---

### Afternoon: Error Handling and Iteration

**Implementation**:

```python
class RegistrationErrorHandler:
    """Handle registration failures and iteration."""

    def handle_test_failure(self, error: str, config: Dict) -> Dict:
        """
        Analyze test failure and suggest fixes.

        Common issues:
        - Wrong callable name
        - Function is class method
        - Wrong return format
        - Missing dependencies
        """

    def iterate_config(self, original_config: Dict, error: str) -> Dict:
        """Generate updated config based on error."""
```

**Tests**:
- Test fails initially
- Agent iterates and fixes
- Eventually succeeds

**Deliverable**: Robust error handling ✅

---

## Day 5: Comprehensive Testing

### Morning: Level 0 Testing (Python Functions)

**File**: `test_evaluator_registration_level0.py` (NEW)

**20 test cases**:

```python
def test_simple_scalar_return():
    """Function returns single number."""

def test_dict_return():
    """Function returns {"objective": value}."""

def test_tuple_return():
    """Function returns (objectives, constraints)."""

def test_numpy_scalar():
    """Function returns np.float64."""

def test_list_return():
    """Function returns [value]."""

def test_class_method():
    """Function is class method."""

def test_multiple_objectives():
    """Function returns {"obj1": ..., "obj2": ...}."""

def test_with_constraints():
    """Function returns objectives and constraints."""

def test_with_gradients():
    """Function has gradient companion."""

def test_nested_imports():
    """Function imports other modules."""

def test_global_state():
    """Function uses global variables."""

def test_file_io():
    """Function reads/writes files."""

def test_long_execution():
    """Function takes time to run."""

def test_stochastic():
    """Function is non-deterministic."""

def test_error_handling():
    """Function can raise exceptions."""

def test_boundary_values():
    """Function handles edge cases."""

def test_vectorized():
    """Function expects vectorized input."""

def test_multiprocess_safe():
    """Function can run in parallel."""

def test_large_output():
    """Function returns large data."""

def test_special_types():
    """Function returns custom types."""
```

**Success criteria**: 20/20 pass ✅

---

### Afternoon: Level 1 Testing (CLI Executables)

**File**: `test_evaluator_registration_level1.py` (NEW)

**10 test cases**:

```python
def test_text_io():
    """Command reads/writes text files."""

def test_json_io():
    """Command reads/writes JSON."""

def test_csv_io():
    """Command reads/writes CSV."""

def test_stdout_output():
    """Command prints to stdout."""

def test_multiple_files():
    """Command outputs multiple files."""

def test_working_directory():
    """Command needs specific working dir."""

def test_environment_variables():
    """Command needs env vars."""

def test_command_arguments():
    """Command takes additional arguments."""

def test_error_codes():
    """Command uses exit codes."""

def test_long_running():
    """Command takes significant time."""
```

**Success criteria**: 9/10 pass (90%+) ✅

---

### End of Day: Documentation

**Files to create/update**:

1. `docs/user_guide/evaluator_registration.md` - User-facing guide
2. `docs/examples/register_python_function.md` - Example walkthrough
3. `docs/examples/register_cli_executable.md` - Example walkthrough
4. Update `README.md` with registration info

---

## Deliverables Checklist

- [ ] FoundryEvaluator infrastructure (Day 1)
- [ ] PAOLA capabilities (observation, caching, cost tracking) (Day 1)
- [ ] Configuration schema (Day 2)
- [ ] Foundry storage (Day 2)
- [ ] Agent tools (Day 3)
- [ ] System prompt and examples (Day 3)
- [ ] CLI integration (Day 4)
- [ ] Error handling and iteration (Day 4)
- [ ] Level 0 tests (20 cases) (Day 5)
- [ ] Level 1 tests (10 cases) (Day 5)
- [ ] Documentation (Day 5)

---

## Success Metrics

### Level 0 (Python Functions)
- ✅ 100% success on 20 test cases
- ✅ < 30 seconds registration time
- ✅ LLM generates correct config without user guidance (95%+)
- ✅ Clear error messages if test fails

### Level 1 (CLI Executables)
- ✅ 90%+ success on 10 test cases
- ✅ < 5 minutes registration time
- ✅ LLM iterates successfully when test fails
- ✅ User confirmation on critical choices

### Infrastructure
- ✅ Observation gates log all evaluations
- ✅ Caching reduces redundant calls by 80%+
- ✅ Cost tracking accurate within 10%
- ✅ No memory leaks or performance degradation

---

## Code Volume Estimate

| Component | Lines of Code | Complexity |
|-----------|--------------|------------|
| FoundryEvaluator | 200 | Medium |
| Capabilities (Observer, Cache) | 150 | Low |
| Configuration schema | 100 | Low |
| Evaluator storage | 100 | Low |
| Agent tools | 150 | Low |
| System prompt | 100 | Low |
| CLI integration | 100 | Medium |
| Tests | 500 | Low |
| **Total** | **~1,400 lines** | |

**Comparison**: ~1,400 lines registration vs ~1,000+ lines per-user wrapper generation

**Key difference**: Registration is ONE-TIME infrastructure investment, wrapper generation is N × 1000 lines

---

## Next Steps (Week 2)

After Week 1 completion:

1. **Optimizer integration** - Connect registered evaluators to optimizers
2. **Multi-fidelity support** - Register evaluators with multiple fidelity levels
3. **Advanced capabilities** - Parallel evaluation, cost budgeting
4. **Real-world testing** - Test with actual user evaluators

---

## Notes

- Focus on FLAWLESS Level 0 - this is 80% of users
- LLM iteration is critical - don't expect perfection on first try
- User confirmation on ambiguous choices
- Keep configuration simple - can extend later
- Observation gates are foundation for future agent intelligence

---

**Ready to implement!** 🚀
