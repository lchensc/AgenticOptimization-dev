# LLM-Powered Evaluator Registration - Detailed Design

**Date**: 2025-12-14

---

## 1. Tool Naming Strategy: `register_evaluator`

### The Question

Why `register_evaluator` instead of `register_evaluator_with_llm`?

### Answer: Two Different Entry Points

There are **two distinct workflows** for registration:

#### Workflow A: User Manual Registration (CLI command)
```bash
paola> /register evaluators.py
  Function name: rosenbrock_2d
  Evaluator name: Rosenbrock
  Evaluator ID: rosenbrock_eval
```

**Purpose**: User-driven, interactive registration via CLI
**Implementation**: Current `handle_register()` in commands.py
**No LLM needed**: Deterministic, user provides all info

#### Workflow B: Agent Autonomous Registration (Agent tool)
```python
# Agent decides to register an evaluator
agent uses tool: register_evaluator

register_evaluator(file_path="evaluators.py", function_name="rosenbrock_2d")
# LLM analyzes code → extracts metadata → stores with semantic understanding
```

**Purpose**: Agent-driven, autonomous registration during optimization
**Implementation**: New tool for agent
**LLM required**: Semantic analysis, metadata extraction

### Naming Proposal

**Option 1: Two Separate Tools**
```python
# For agent use
@tool
def register_evaluator(
    file_path: str,
    function_name: str
) -> Dict[str, Any]:
    """
    Register evaluator with LLM semantic analysis (agent use).

    Agent workflow:
    1. Read file
    2. LLM analyzes code
    3. Extract dimension, variables, I/O structure
    4. Test configuration
    5. Store with full metadata
    """
    pass

# For CLI use (existing)
def handle_register(file_path: str):
    """Manual registration via CLI (user interactive)."""
    pass
```

**Option 2: Single Tool with Mode**
```python
@tool
def register_evaluator(
    file_path: str,
    function_name: Optional[str] = None,
    mode: str = "auto"
) -> Dict[str, Any]:
    """
    Register evaluator.

    Args:
        file_path: Path to evaluator file
        function_name: Function name (auto-detect if None)
        mode: "auto" (LLM if available), "llm" (force LLM), "manual" (prompt user)
    """
    if mode == "llm" or (mode == "auto" and function_name):
        return _register_with_llm(file_path, function_name)
    else:
        return _register_manual(file_path)
```

### Recommended: Option 1

**Reasoning**:
- Clear separation of concerns
- Agent always uses LLM-powered registration
- CLI can remain simple and fast
- Tool schema is simpler (no mode confusion)

**Final naming**:
```python
# Agent tool (NEW)
register_evaluator(file_path, function_name)  # LLM-powered

# CLI handler (EXISTING)
handle_register(file_path)  # Manual/interactive
```

The agent tool is simply called `register_evaluator` because **for the agent, LLM use is the default and only way**.

---

## 2. Automatic Variable Extractor Generation

### What Are Variable Extractors?

Variable extractors are **simple evaluators that return design variable values**.

#### Example Problem

User wants to optimize `rosenbrock(x, y)` with constraint `x >= 1.5`.

**Issue**:
- Objective evaluator: `rosenbrock_eval(x)` → returns `(1-x[0])² + 100(x[1]-x[0]²)²`
- Constraint needs: evaluator that returns `x[0]`
- **No such evaluator exists!**

**Solution**:
Create a variable extractor:

```python
def x0_extractor(x):
    """Returns first design variable."""
    return x[0]
```

### Why Automatic Generation?

**Without automatic generation**:
```
User: "Optimize rosenbrock with x >= 1.5"

Agent:
1. Need constraint evaluator for x[0]
2. Check Foundry → x0_extractor not found
3. ❌ STUCK: Ask user to create x0_extractor manually
   OR
4. ❌ WRONG: Use rosenbrock_eval as constraint (your test case!)
```

**With automatic generation**:
```
User: "Optimize rosenbrock with x >= 1.5"

Agent:
1. Need constraint evaluator for x[0]
2. Check Foundry → x0_extractor not found
3. Check rosenbrock_eval metadata:
   {
     "dimension": 2,
     "variable_names": ["x0", "x1"]
   }
4. ✓ Generate x0_extractor on-the-fly!
5. ✓ Register it automatically
6. ✓ Use in constraint
7. ✓ SUCCESS
```

### How It Works

#### Step 1: Detect Need for Variable Extractor

When agent creates NLP problem with constraint:

```python
# Agent reasoning
"User wants constraint: x >= 1.5"
"This means: x[0] >= 1.5"
"Need evaluator that returns x[0]"
"Naming convention: x0_extractor or var_0_extractor"
```

#### Step 2: Check if Extractor Exists

```python
# Agent checks Foundry
result = foundry_list_evaluators()

if "x0_extractor" not in [e["evaluator_id"] for e in result["evaluators"]]:
    # Need to create it
    pass
```

#### Step 3: Get Problem Dimension

```python
# Agent queries objective evaluator metadata
obj_metadata = foundry_get_evaluator("rosenbrock_eval")

dimension = obj_metadata["interface"]["input"]["dimension"]  # 2
# Now agent knows the problem is 2D
```

#### Step 4: Generate Variable Extractor Code

```python
# Agent generates code
extractor_code = f'''
def x0_extractor(x):
    """
    Variable extractor: returns x[0] (first design variable).

    Auto-generated for constraint formulation.

    Input: Design vector x of dimension {dimension}
    Output: Scalar value x[0]
    """
    import numpy as np
    x = np.atleast_1d(x)
    return float(x[0])
'''
```

#### Step 5: Register the Extractor

```python
# Agent uses execute_python to test
test_result = execute_python(code=f'''
{extractor_code}

import numpy as np
result = x0_extractor(np.array([1.5, 2.0]))
print(f"Test: x0_extractor([1.5, 2.0]) = {{result}}")
assert result == 1.5, "Test failed"
print("SUCCESS")
''')

# If test passes, register
if "SUCCESS" in test_result["stdout"]:
    register_evaluator(
        file_path=None,  # Code string, not file
        function_name="x0_extractor",
        code_string=extractor_code,  # NEW parameter
        metadata={
            "dimension": dimension,
            "extractor_index": 0,
            "auto_generated": True
        }
    )
```

#### Step 6: Use in Constraint

```python
# Now agent can create NLP with correct constraint
create_nlp_problem(
    problem_id="rosenbrock_constrained",
    objective_evaluator_id="rosenbrock_eval",
    bounds=[[-5, 10], [-5, 10]],
    inequality_constraints=[{
        "name": "x0_min",
        "evaluator_id": "x0_extractor",  # ← AUTO-GENERATED!
        "type": ">=",
        "value": 1.5
    }]
)
```

### Complete Variable Extractor Set

For a 2D problem, agent can generate:

```python
# x0_extractor (index 0)
def x0_extractor(x):
    return float(x[0])

# x1_extractor (index 1)
def x1_extractor(x):
    return float(x[1])

# sum_extractor (helper for constraints like x0 + x1 <= 10)
def sum_extractor(x):
    return float(x[0] + x[1])

# norm_extractor (helper for constraints like ||x|| <= 5)
def norm_extractor(x):
    import numpy as np
    return float(np.linalg.norm(x))
```

### When Generation Happens

**Trigger**: Agent needs a variable extractor that doesn't exist

**Scenarios**:
1. **User specifies constraint by variable name**:
   - "x >= 1.5" → generate x0_extractor
   - "y <= 2.0" → generate x1_extractor (if y is 2nd variable)

2. **User specifies constraint by index**:
   - "x[0] >= 1.5" → generate x0_extractor
   - "x[1] <= 2.0" → generate x1_extractor

3. **User specifies constraint by combined expression**:
   - "x + y <= 5" → generate sum_extractor
   - "x² + y² <= 10" → generate norm_squared_extractor

### Implementation: Tool Enhancement

```python
@tool
def generate_variable_extractor(
    variable_index: int,
    problem_dimension: int,
    evaluator_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate and register a variable extractor.

    Args:
        variable_index: Index of variable to extract (0-based)
        problem_dimension: Total problem dimension
        evaluator_id: Optional custom ID (default: f"x{variable_index}_extractor")

    Returns:
        {
            "success": True,
            "evaluator_id": "x0_extractor",
            "code": "def x0_extractor(x): ..."
        }

    Example:
        # Generate extractor for first variable
        generate_variable_extractor(
            variable_index=0,
            problem_dimension=2
        )
        # Returns extractor that does: return x[0]
    """
    if evaluator_id is None:
        evaluator_id = f"x{variable_index}_extractor"

    # Generate code
    code = f'''
def {evaluator_id.replace("_extractor", "")}(x):
    """Variable extractor: returns x[{variable_index}]."""
    import numpy as np
    x = np.atleast_1d(x)
    if len(x) != {problem_dimension}:
        raise ValueError(f"Expected dimension {problem_dimension}, got {{len(x)}}")
    return float(x[{variable_index}])
'''

    # Test
    test_code = f'''
{code}
import numpy as np
test_x = np.random.rand({problem_dimension})
result = {evaluator_id.replace("_extractor", "")}(test_x)
assert result == test_x[{variable_index}]
print("SUCCESS")
'''

    test_result = execute_python(code=test_code)

    if "SUCCESS" not in test_result.get("stdout", ""):
        return {"success": False, "error": "Test failed"}

    # Register
    config = {
        "evaluator_id": evaluator_id,
        "name": f"Variable x[{variable_index}] Extractor",
        "source": {
            "type": "python_code_string",
            "code": code,
            "callable_name": evaluator_id.replace("_extractor", "")
        },
        "interface": {
            "input": {
                "type": "numpy_array",
                "dimension": problem_dimension
            },
            "output": {
                "format": "scalar",
                "physical_meaning": f"design_variable_{variable_index}"
            }
        },
        "metadata": {
            "auto_generated": True,
            "extractor_type": "variable",
            "variable_index": variable_index
        }
    }

    store_result = foundry_store_evaluator(config=config, test_result=test_result)

    return {
        "success": True,
        "evaluator_id": evaluator_id,
        "code": code
    }
```

### Agent Workflow with Auto-Generation

```python
# Agent's internal reasoning during create_nlp_problem

def agent_create_constraint(constraint_spec: str, objective_eval_id: str):
    """
    Agent intelligently creates constraint.

    Args:
        constraint_spec: "x >= 1.5" or "x[0] >= 1.5" or "chord >= 2.0"
        objective_eval_id: "rosenbrock_eval"
    """
    # Parse constraint
    if "x >=" in constraint_spec or "x[0] >=" in constraint_spec:
        # Need variable extractor
        variable_index = 0
        constraint_value = 1.5

        # Get problem dimension from objective evaluator
        obj_metadata = foundry_get_evaluator(objective_eval_id)
        dimension = obj_metadata["interface"]["input"]["dimension"]

        # Check if x0_extractor exists
        if not evaluator_exists("x0_extractor"):
            # Generate it automatically
            generate_variable_extractor(
                variable_index=0,
                problem_dimension=dimension
            )

        # Return constraint spec
        return {
            "name": "x0_min",
            "evaluator_id": "x0_extractor",
            "type": ">=",
            "value": constraint_value
        }
```

---

## 3. Semantic Metadata

### What Is Semantic Metadata?

**Semantic metadata** = Information about what the evaluator *means*, not just how to call it.

**Deterministic registration** (current):
```python
{
    "evaluator_id": "rosenbrock_eval",
    "source": {
        "file_path": "/path/to/evaluators.py",
        "callable_name": "rosenbrock_2d"
    },
    "interface": {
        "output": {"format": "auto"}  # ← NO MEANING!
    }
}
```

**LLM-powered registration** (proposed):
```python
{
    "evaluator_id": "rosenbrock_eval",
    "source": {
        "file_path": "/path/to/evaluators.py",
        "callable_name": "rosenbrock_2d"
    },
    "interface": {
        "input": {
            "type": "numpy_array",
            "dimension": 2,                        # ← SEMANTIC: dimension
            "variable_names": ["x", "y"],          # ← SEMANTIC: variable names
            "variable_types": ["continuous", "continuous"],  # ← SEMANTIC: types
            "physical_units": [None, None],        # ← SEMANTIC: units (if known)
            "recommended_bounds": [[-5, 10], [-5, 10]]  # ← SEMANTIC: typical bounds
        },
        "output": {
            "format": "scalar",                    # ← SEMANTIC: output structure
            "physical_meaning": "objective_value", # ← SEMANTIC: what it represents
            "typical_range": [0, 1000],            # ← SEMANTIC: expected values
            "minimization_optimal": True           # ← SEMANTIC: direction
        }
    },
    "metadata": {
        "description": "2D Rosenbrock function, classic optimization benchmark",
        "problem_class": "continuous_unconstrained",
        "difficulty": "medium",
        "known_optimum": {
            "x_opt": [1.0, 1.0],
            "f_opt": 0.0
        },
        "properties": [
            "nonlinear",
            "non-convex",
            "smooth",
            "deterministic"
        ]
    }
}
```

### Schema Structure

```python
@dataclass
class SemanticEvaluatorMetadata:
    """Complete semantic metadata for an evaluator."""

    # Basic identification
    evaluator_id: str
    name: str
    description: str

    # Source information
    source: SourceConfig  # file_path, callable_name, etc.

    # INPUT semantics
    input_semantics: InputSemantics = field(default_factory=lambda: InputSemantics(
        type="numpy_array",
        dimension=None,              # ← How many variables?
        variable_names=None,         # ← What are they called? ["x", "y", "chord", ...]
        variable_types=None,         # ← continuous, discrete, categorical
        physical_units=None,         # ← meters, kg, dimensionless
        recommended_bounds=None,     # ← Typical bounds from code/docs
        variable_descriptions=None   # ← What each variable represents
    ))

    # OUTPUT semantics
    output_semantics: OutputSemantics = field(default_factory=lambda: OutputSemantics(
        format="scalar",             # ← scalar, vector, dict
        output_keys=None,            # ← If dict: ["drag", "lift", "stress"]
        physical_meaning=None,       # ← objective, constraint, metric
        physical_units=None,         # ← Newtons, meters, dimensionless
        typical_range=None,          # ← Expected value range
        minimization_optimal=None    # ← True if lower is better
    ))

    # SEMANTIC understanding
    semantics: EvaluatorSemantics = field(default_factory=lambda: EvaluatorSemantics(
        problem_class=None,          # ← benchmark, physics, black_box
        domain=None,                 # ← aerodynamics, structures, general
        difficulty=None,             # ← easy, medium, hard
        known_optimum=None,          # ← If benchmark: known solution
        properties=[]                # ← smooth, deterministic, expensive, etc.
    ))
```

### Example: Rosenbrock Function

**LLM analyzes this code**:
```python
def rosenbrock_2d(x):
    """
    2D Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2

    Classic optimization benchmark with narrow curved valley.
    Global minimum at (1, 1) with f = 0.
    Typically bounded: -5 <= x, y <= 10
    """
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
```

**LLM extracts**:
```python
{
    "evaluator_id": "rosenbrock_eval",
    "name": "Rosenbrock 2D",
    "description": "Classic optimization benchmark with narrow curved valley",

    "input_semantics": {
        "type": "numpy_array",
        "dimension": 2,                          # ← From signature
        "variable_names": ["x", "y"],            # ← From docstring
        "variable_types": ["continuous", "continuous"],
        "physical_units": [None, None],          # ← Dimensionless
        "recommended_bounds": [[-5, 10], [-5, 10]]  # ← From docstring
    },

    "output_semantics": {
        "format": "scalar",                      # ← Returns single float
        "physical_meaning": "objective_value",   # ← It's an objective
        "minimization_optimal": True             # ← Lower is better
    },

    "semantics": {
        "problem_class": "benchmark",
        "domain": "general_optimization",
        "difficulty": "medium",
        "known_optimum": {
            "x_opt": [1.0, 1.0],                 # ← From docstring
            "f_opt": 0.0
        },
        "properties": [
            "nonlinear",
            "smooth",
            "non-convex",
            "deterministic"
        ]
    }
}
```

### Example: Engineering Evaluator

**LLM analyzes this code**:
```python
def calculate_wing_drag(design):
    """
    Calculate drag force on transonic wing.

    Design variables:
    - chord: Wing chord length (meters, 2-15m typical)
    - thickness_ratio: Thickness-to-chord ratio (0.08-0.15 typical)

    Returns drag force in Newtons.
    Lower drag is better for efficiency.

    Typical range: 1000-5000 N
    """
    chord, thickness_ratio = design
    # CFD simulation...
    return drag_force
```

**LLM extracts**:
```python
{
    "evaluator_id": "wing_drag_eval",
    "name": "Wing Drag Calculator",
    "description": "Calculate drag force on transonic wing using CFD",

    "input_semantics": {
        "type": "numpy_array",
        "dimension": 2,
        "variable_names": ["chord", "thickness_ratio"],  # ← FROM DOCSTRING!
        "variable_types": ["continuous", "continuous"],
        "physical_units": ["meters", "dimensionless"],   # ← FROM DOCSTRING!
        "recommended_bounds": [[2, 15], [0.08, 0.15]],   # ← FROM DOCSTRING!
        "variable_descriptions": [
            "Wing chord length",
            "Thickness-to-chord ratio"
        ]
    },

    "output_semantics": {
        "format": "scalar",
        "physical_meaning": "drag_force",
        "physical_units": "Newtons",                     # ← FROM DOCSTRING!
        "typical_range": [1000, 5000],                   # ← FROM DOCSTRING!
        "minimization_optimal": True                     # ← FROM DOCSTRING!
    },

    "semantics": {
        "problem_class": "physics_based",
        "domain": "aerodynamics",
        "properties": ["expensive", "smooth", "deterministic"]
    }
}
```

### How Agent Uses Semantic Metadata

#### Use Case 1: Constraint Creation (Your Test Case!)

```python
# User: "Optimize wing_drag with chord >= 5m"

# Agent reasoning:
obj_metadata = get_evaluator("wing_drag_eval")
variable_names = obj_metadata["input_semantics"]["variable_names"]
# ["chord", "thickness_ratio"]

# User said "chord >= 5m"
# "chord" is variable 0
variable_index = variable_names.index("chord")  # 0

# Check if chord_extractor exists
if not evaluator_exists("chord_extractor"):
    # Generate it
    generate_variable_extractor(
        variable_index=0,
        problem_dimension=2,
        evaluator_id="chord_extractor"
    )

# Create constraint
constraint = {
    "name": "min_chord",
    "evaluator_id": "chord_extractor",
    "type": ">=",
    "value": 5.0
}
```

**Without semantic metadata**: Agent doesn't know which variable is "chord" → can't create constraint!

#### Use Case 2: Bounds Suggestion

```python
# User: "Optimize wing_drag"

# Agent reasoning:
metadata = get_evaluator("wing_drag_eval")
bounds = metadata["input_semantics"]["recommended_bounds"]
# [[2, 15], [0.08, 0.15]]

# Agent suggests:
"I recommend these bounds based on the evaluator documentation:
  - chord: [2, 15] meters
  - thickness_ratio: [0.08, 0.15]

Use these bounds? (yes/no)"
```

#### Use Case 3: Problem Compatibility Check

```python
# User: "Use drag_eval as objective and lift_eval as constraint"

# Agent reasoning:
drag_metadata = get_evaluator("drag_eval")
lift_metadata = get_evaluator("lift_eval")

drag_dim = drag_metadata["input_semantics"]["dimension"]  # 2
drag_vars = drag_metadata["input_semantics"]["variable_names"]  # ["chord", "thickness"]

lift_dim = lift_metadata["input_semantics"]["dimension"]  # 2
lift_vars = lift_metadata["input_semantics"]["variable_names"]  # ["chord", "thickness"]

# ✓ Compatible: same dimension and variable names
```

#### Use Case 4: Smart Initial Point

```python
# User: "Optimize rosenbrock"

# Agent reasoning:
metadata = get_evaluator("rosenbrock_eval")
bounds = metadata["input_semantics"]["recommended_bounds"]  # [[-5, 10], [-5, 10]]

# Don't use random initial point at extreme bounds
# Use middle of recommended range
initial_point = [(b[0] + b[1]) / 2 for b in bounds]  # [2.5, 2.5]
```

### How Semantic Metadata is Extracted

**LLM Prompt** (simplified):
```
Analyze this evaluator function and extract semantic metadata.

Code:
{code}

Extract JSON with this schema:
{{
  "dimension": <int>,
  "variable_names": [<strings>],
  "variable_types": ["continuous"|"discrete"|"categorical"],
  "physical_units": [<strings or null>],
  "recommended_bounds": [[lower, upper], ...],
  "output_format": "scalar"|"vector"|"dict",
  "physical_meaning": <string>,
  "minimization_optimal": <bool>,
  "description": <string>,
  "known_optimum": {{x_opt, f_opt}} or null
}}

Look for:
- Function signature for dimension
- Docstring for variable names, units, bounds
- Return statement for output structure
- Comments about optimal direction
```

**LLM Response** for rosenbrock:
```json
{
  "dimension": 2,
  "variable_names": ["x", "y"],
  "variable_types": ["continuous", "continuous"],
  "physical_units": [null, null],
  "recommended_bounds": [[-5, 10], [-5, 10]],
  "output_format": "scalar",
  "physical_meaning": "objective_value",
  "minimization_optimal": true,
  "description": "Classic optimization benchmark with narrow curved valley",
  "known_optimum": {"x_opt": [1.0, 1.0], "f_opt": 0.0}
}
```

---

## Summary

### 1. Tool Naming: `register_evaluator`

**Decision**: Simple name without `_with_llm` suffix
**Reason**: For agent tools, LLM use is the default (and only) way
**Implementation**: Separate tools for agent (`register_evaluator`) vs CLI (`handle_register`)

### 2. Automatic Variable Extractor Generation

**What**: Auto-create evaluators like `x0_extractor(x) → x[0]` when needed for constraints
**Why**: Enables agent to formulate `x >= 1.5` constraints without user manually creating extractors
**How**:
- Agent detects need (constraint on variable)
- Gets dimension from objective evaluator metadata
- Generates extractor code programmatically
- Tests and registers automatically
**Tool**: `generate_variable_extractor(variable_index, problem_dimension)`

### 3. Semantic Metadata

**What**: Rich information about evaluator meaning (not just calling interface)
**Includes**:
- **Input**: dimension, variable names, types, units, recommended bounds
- **Output**: format, physical meaning, units, typical range, optimal direction
- **Semantics**: problem class, domain, difficulty, known optimum, properties

**Why**: Enables agent to:
- Create constraints by variable name ("chord >= 5m" → auto-generate chord_extractor)
- Suggest sensible bounds
- Check evaluator compatibility
- Make intelligent optimization decisions

**How Extracted**: LLM analyzes code + docstrings → extracts semantic info → stores in metadata

---

This architecture enables the agent to autonomously handle your test case:
```
User: "Optimize rosenbrock with x >= 1.5"

Agent:
1. Get rosenbrock_eval metadata → dimension=2, variables=[x, y]
2. Parse "x >= 1.5" → need x0_extractor
3. Generate x0_extractor automatically
4. Create NLP with correct constraint
5. ✓ SUCCESS
```
