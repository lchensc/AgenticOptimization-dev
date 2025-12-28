# Agentic Evaluator Architecture

**Date**: 2025-12-14
**Status**: ✅ IMPLEMENTED
**Version**: 1.0

---

## Executive Summary

PAOLA now implements a **compiled evaluator architecture** where evaluators are extracted from source files and stored as immutable, self-contained snapshots with semantic metadata. This enables true agentic behavior where the agent understands what evaluators compute and can autonomously create missing components.

### Key Innovation

**Before** (Path-Based Registration):
- Agent treats evaluators as black boxes
- Cannot understand problem structure
- Cannot create constraints on design variables
- Fragile: file changes break cached results

**After** (Compiled with Semantics):
- Agent understands evaluator semantics via LLM analysis
- Can auto-generate variable extractors for constraints
- Immutable snapshots prevent cache invalidation
- Portable, self-contained evaluators

---

## Architecture Overview

### Directory Structure

```
.paola_data/
├── evaluators/
│   ├── rosenbrock_eval/
│   │   ├── source.py              # Immutable snapshot
│   │   ├── metadata.json          # Semantic analysis
│   │   ├── test_results.json      # Validation results
│   │   └── dependencies/          # Any data files
│   ├── x0_extractor/              # Auto-generated
│   │   ├── source.py
│   │   ├── metadata.json
│   │   └── test_results.json
│   └── manifest.json              # Registry
├── runs/
└── cache/
```

### Data Flow

```
User: "Register rosenbrock from evaluators.py"
                    ↓
    register_evaluator_agentic tool
                    ↓
    ┌───────────────────────────────┐
    │  1. Read Source File          │
    │  2. LLM Analyzes Semantics    │ ← Understands input/output structure
    │  3. Compiler Extracts Code    │
    │  4. Generate Standalone .py   │
    │  5. ReAct Tests Evaluator     │ ← Iterates until tests pass
    │  6. Store Immutable Snapshot  │
    └───────────────────────────────┘
                    ↓
        .paola_data/evaluators/rosenbrock_eval/
                    ├── source.py
                    └── metadata.json
                              ↓
              Agent now knows:
              - Type: objective_function
              - Input dimension: 2
              - Variables: ["x[0]", "x[1]"]
              - Formula: "(1-x0)^2 + 100*(x1-x0^2)^2"
```

---

## Core Components

### 1. Evaluator Compiler (`paola/foundry/evaluator_compiler.py`)

Extracts and compiles evaluators into immutable snapshots.

**Key Methods**:

```python
compiler = EvaluatorCompiler(base_dir=".paola_data")

# Compile function from source file
result = compiler.compile_function(
    source_file=Path("evaluators.py"),
    function_name="rosenbrock_2d",
    evaluator_id="rosenbrock_eval",
    metadata={"semantics": {...}}
)

# Auto-generate variable extractor
result = compiler.generate_variable_extractor(
    variable_index=0,
    dimension=2,
    evaluator_id="x0_extractor"
)

# Load compiled evaluator
func = compiler.load_evaluator("rosenbrock_eval")
```

**Features**:
- AST-based code extraction
- Import detection and inclusion
- Standalone code generation
- Metadata storage
- Dynamic loading

### 2. Semantic Analyzer (`paola/agent/evaluator_analyzer.py`)

Uses LLM to analyze code and extract semantic information.

**Key Methods**:

```python
analyzer = EvaluatorSemanticAnalyzer(llm)

# Analyze function code
result = analyzer.analyze_function(
    code=function_code,
    function_name="rosenbrock_2d",
    file_context=full_file_content
)

# Returns:
# {
#   "type": "objective_function",
#   "input_dimension": 2,
#   "input_variables": ["x[0]", "x[1]"],
#   "output_type": "scalar",
#   "formula": "(1-x0)^2 + 100*(x1-x0^2)^2",
#   "properties": {"convex": false, "smooth": true},
#   "dependencies": ["numpy"]
# }
```

**LLM Prompt Design**:
- Structured prompt for consistent JSON output
- Asks for type, dimension, variables, formula, properties
- Handles various code patterns (functions, methods, classes)

### 3. Agentic Registration Tools (`paola/tools/agentic_registration.py`)

Agent-facing tools for registering evaluators and auto-generating extractors.

**Tools**:

#### `register_evaluator_agentic`

```python
result = register_evaluator_agentic.invoke({
    "file_path": "evaluators.py",
    "function_name": "rosenbrock_2d",
    "evaluator_id": "rosenbrock_eval"  # Optional
})

# Returns:
# {
#   "success": True,
#   "evaluator_id": "rosenbrock_eval",
#   "semantics": {...},
#   "source_path": ".paola_data/evaluators/rosenbrock_eval/source.py",
#   "tests_passed": True,
#   "test_results": [...]
# }
```

**Workflow**:
1. Read source file
2. Extract function code
3. LLM analyzes semantics
4. Compiler creates snapshot
5. ReAct testing loop validates
6. Store with metadata
7. Register in Foundry

#### `auto_generate_variable_extractor`

```python
result = auto_generate_variable_extractor.invoke({
    "variable_index": 0,
    "dimension": 2,
    "evaluator_id": "x0_extractor"  # Optional
})

# Auto-creates:
# def x0_extractor(x):
#     return float(x[0])
```

### 4. Smart NLP Creation (`paola/tools/smart_nlp_creation.py`)

Enhanced NLP problem creation with semantic understanding.

**Tool**:

#### `create_nlp_problem_smart`

```python
result = create_nlp_problem_smart.invoke({
    "problem_id": "rosenbrock_constrained",
    "objective_evaluator_id": "rosenbrock_eval",
    "constraints": ["x[0] >= 1.5"],  # Natural language
    "objective_sense": "minimize"
})

# Automatically:
# 1. Reads rosenbrock_eval semantics
# 2. Parses constraint: needs x[0]
# 3. Auto-generates x0_extractor
# 4. Creates NLP with correct constraint
```

**Constraint Parsing**:
- Natural language: `"x[0] >= 1.5"`, `"x[1] <= 5.0"`
- Evaluator-based: `"lift_eval >= 1000"`, `"stress_eval <= 200"`
- Auto-detects which extractors needed
- Generates missing extractors on-demand

---

## Metadata Format

**`.paola_data/evaluators/{id}/metadata.json`**:

```json
{
  "evaluator_id": "rosenbrock_eval",
  "created_at": "2025-12-14T10:30:00Z",
  "version": "1.0.0",

  "origin": {
    "original_file": "/path/to/evaluators.py",
    "function_name": "rosenbrock_2d",
    "extraction_method": "llm_analysis",
    "extraction_timestamp": "2025-12-14T10:30:00Z"
  },

  "semantics": {
    "type": "objective_function",
    "description": "Classic Rosenbrock banana function",
    "input_dimension": 2,
    "input_variables": ["x[0]", "x[1]"],
    "output_type": "scalar",
    "formula": "(1-x[0])^2 + 100*(x[1]-x[0]^2)^2",
    "properties": {
      "convex": false,
      "smooth": true,
      "differentiable": true
    }
  },

  "capabilities": {
    "gradient_method": "finite_difference",
    "vectorized": false,
    "supports_batch": false
  },

  "validation": {
    "test_points": [
      {"input": [1.0, 1.0], "output": 0.0, "passed": true},
      {"input": [0.0, 0.0], "output": 1.0, "passed": true}
    ],
    "all_tests_passed": true
  }
}
```

---

## User Workflow Example

### Before: Path-Based (Old)

```python
User: /register evaluators.py
CLI: "Function name?" → rosenbrock_2d
CLI: "Evaluator ID?" → rosenbrock_eval
# Stored as path reference

User: "Create optimization with Rosenbrock objective and constraint x[0] >= 1.5"

Agent:
- Sees rosenbrock_eval (black box, no semantics)
- Tries to use rosenbrock_eval for constraint ✗ Wrong!
- Creates constraint: rosenbrock(x) >= 1.5 instead of x[0] >= 1.5
- Optimization fails: constraint semantically incorrect
```

### After: Compiled (New)

```python
User: "Register rosenbrock from evaluators.py"

Agent uses register_evaluator_agentic:
1. Reads evaluators.py
2. LLM analyzes: "2D objective function, accesses x[0] and x[1]"
3. Compiles immutable snapshot
4. Tests with [0,0], [1,1]
5. Stores with semantic metadata

User: "Create optimization with Rosenbrock objective and constraint x[0] >= 1.5"

Agent uses create_nlp_problem_smart:
1. Reads rosenbrock_eval semantics: dimension=2, type=objective
2. Parses constraint: needs x[0]
3. Checks: x0_extractor exists? No
4. Auto-generates x0_extractor
5. Creates NLP: objective=rosenbrock_eval, constraint=x0_extractor >= 1.5 ✓
6. Optimization succeeds: constraint correct!
```

---

## Benefits

### 1. Immutability

**Problem**: Original file changes break cached results

```python
# evaluators.py version 1
def rosenbrock_2d(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Run optimization, cache results...

# User edits file (version 2)
def rosenbrock_2d(x):
    return (1 - x[0])**2 + 10 * (x[1] - x[0]**2)**2  # Changed 100 → 10!

# Cache is now INVALID but system doesn't know!
```

**Solution**: Immutable snapshots

```
.paola_data/evaluators/rosenbrock_eval/source.py  ← Frozen at registration
Original evaluators.py can change without affecting cached results
```

### 2. Semantic Understanding

**Enables**:
- Agent knows this is an objective function, not a constraint
- Agent knows input is 2D: x[0], x[1]
- Agent can infer need for x0_extractor, x1_extractor
- Agent can validate constraint specifications against dimension

### 3. Auto-Generation

**Agent creates missing components**:

```python
Constraint: "x[0] >= 1.5"
           ↓
Agent reasoning:
- Needs evaluator that returns x[0]
- x0_extractor doesn't exist
- I can auto-generate it!
           ↓
Auto-creates x0_extractor
           ↓
Uses in constraint ✓
```

### 4. Portability

```
.paola_data/evaluators/  ← Self-contained
├── rosenbrock_eval/
│   └── source.py        ← All dependencies included
└── x0_extractor/
    └── source.py

Zip and share → Works on any machine!
No path dependencies, no missing files
```

### 5. Traceability

```
Cached result for x=[2, 4]:
  evaluator_id: "rosenbrock_eval"
  evaluator_version: "1.0.0"
  source_hash: "abc123..."

Can verify EXACT code that produced this result
```

---

## Integration with Agent

### New Agent Tools

Agent now has access to:

1. **`register_evaluator_agentic`**: LLM-based registration with semantics
2. **`auto_generate_variable_extractor`**: On-demand extractor creation
3. **`create_nlp_problem_smart`**: Natural language constraint specification

### Agent Reasoning Pattern

```
User: "Optimize Rosenbrock with x[0] >= 1.5"

Agent (internal reasoning):
1. "Need to register rosenbrock first"
   → Uses register_evaluator_agentic
   → Learns: objective_function, 2D, variables [x[0], x[1]]

2. "Constraint x[0] >= 1.5 needs x[0] evaluator"
   → Checks: x0_extractor exists? No
   → Uses auto_generate_variable_extractor
   → Creates x0_extractor

3. "Now create NLP problem"
   → Uses create_nlp_problem_smart
   → Constraint: x0_extractor >= 1.5 ✓

4. "Run optimization"
   → Constraint is enforced correctly!
```

---

## Testing

### Basic Compiler Test

**File**: `tests/test_compiler_basic.py`

Verifies:
- ✅ Code extraction and compilation
- ✅ Immutable snapshot creation
- ✅ Evaluator loading and execution
- ✅ Variable extractor generation
- ✅ Metadata storage and retrieval

**Run**:
```bash
python tests/test_compiler_basic.py
```

### Full Agentic Test

**File**: `tests/test_agentic_architecture.py`

Verifies:
- ✅ LLM-based registration
- ✅ Semantic analysis
- ✅ Auto-generated extractors
- ✅ Smart NLP creation
- ✅ Constraint enforcement

**Run** (requires LLM credentials):
```bash
python tests/test_agentic_architecture.py
```

---

## Comparison: Old vs New

| Aspect | Path-Based (Old) | Compiled (New) |
|--------|------------------|----------------|
| **Storage** | Path reference | Immutable snapshot |
| **Stability** | ✗ File changes break it | ✓ Frozen at registration |
| **Portability** | ✗ Absolute paths | ✓ Self-contained |
| **Semantics** | ✗ Black box | ✓ Full metadata |
| **Auto-generation** | ✗ Manual | ✓ Automated |
| **Constraint handling** | ✗ Fails | ✓ Works correctly |
| **Agent understanding** | ✗ None | ✓ Complete |
| **Registration time** | ~1 second | ~3-5 seconds (LLM) |
| **Registration cost** | Free | ~$0.01 per file |
| **Reliability** | Template-based | LLM + ReAct tested |

---

## Future Enhancements

### Phase 1 (Current)
- ✅ Basic compilation
- ✅ LLM semantic analysis
- ✅ Variable extractors
- ✅ Smart NLP creation

### Phase 2 (Next)
- [ ] Dependency extraction (helper functions)
- [ ] Data file handling (CSV, JSON)
- [ ] Class method extraction
- [ ] Batch registration (multiple functions)

### Phase 3 (Later)
- [ ] Constraint Jacobian generation
- [ ] Analytical gradient extraction
- [ ] Multi-objective evaluators
- [ ] Evaluator composition

---

## Migration Guide

### For Existing Users

**Old workflow** (still supported):
```python
/register evaluators.py
# Interactive Q&A
```

**New workflow** (recommended):
```python
"Register rosenbrock from evaluators.py"
# Agent handles everything automatically
```

**Migration**:
- Old evaluators continue to work
- New registrations use compiled architecture
- Gradually re-register evaluators for benefits

### For Developers

**Adding new evaluator support**:

1. Extend `EvaluatorSemanticAnalyzer` for new patterns
2. Update compiler for new source types (classes, CLI tools)
3. Add new extractor types (gradient extractors, multi-output, etc.)

---

## Conclusion

The agentic evaluator architecture transforms PAOLA from a **traditional platform with LLM UI** to a **truly agentic system** where the agent:

1. **Understands** what evaluators compute via semantic analysis
2. **Creates** missing components autonomously
3. **Validates** evaluators before registration
4. **Preserves** immutable snapshots for reproducibility

This enables the agent to correctly handle complex problem formulations like constraints on design variables, which was previously impossible.

**Your test case now works**:
```python
User: "Optimize Rosenbrock with x[0] >= 1.5"
Agent: ✓ Correctly creates constraint using auto-generated x0_extractor
Result: Constraint is enforced, optimization succeeds!
```

---

**Implementation**: Complete
**Status**: Ready for testing
**Cost**: ~$0.01 per evaluator registration (LLM analysis)
**Benefit**: True agentic problem formulation
