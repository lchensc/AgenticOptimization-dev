# Agentic Architecture - Test Scenario

**Date**: 2025-12-14
**Purpose**: Hands-on testing of compiled evaluator architecture with semantic analysis
**Location**: `debug_agent/`

---

## Prerequisites

1. PAOLA CLI is installed and working
2. LLM credentials configured (for semantic analysis)
3. All recent changes committed and pulled

---

## Test Scenario Overview

This scenario demonstrates the complete agentic evaluator workflow:

1. **Agentic Registration**: Register evaluator with LLM semantic analysis
2. **Auto-Generated Extractors**: Create constraints on design variables
3. **Smart NLP Creation**: Natural language constraint specification
4. **Constraint Enforcement**: Verify constraints are correctly enforced
5. **Immutability**: Show that evaluators are frozen snapshots

---

## Scenario 1: Basic Agentic Registration

### Step 1: Start PAOLA CLI

```bash
cd /home/longchen/PythonCode/gendesign/AgenticOptimization
python paola_cli.py
```

### Step 2: Register Evaluator with Agentic Tool

**User input**:
```
paola> Register the rosenbrock_2d function from debug_agent/test_evaluators.py
```

**Expected agent behavior**:
- Agent uses `register_evaluator_agentic` tool
- Reads source file
- LLM analyzes code and extracts semantics
- Creates immutable snapshot in `.paola_data/evaluators/rosenbrock_2d_eval/`
- Runs test cases to validate
- Stores with semantic metadata

**Expected output**:
```
💭 I'll register the rosenbrock_2d evaluator...
🔧 register_evaluator_agentic...
✓ register_evaluator_agentic completed

✓ Evaluator 'rosenbrock_2d_eval' compiled and tested successfully
  Type: objective_function
  Input dimension: 2
  Tests: 3/3 passed
```

### Step 3: Verify Compiled Evaluator

**User input**:
```
paola> /evals
```

**Expected output**:
```
╭─ Registered Evaluators ─────────────────────────────────────╮
│                                                              │
│  rosenbrock_2d_eval                                          │
│    Classic 2D Rosenbrock function                            │
│    Created: 2025-12-14                                       │
│    Type: objective_function                                  │
│                                                              │
╰──────────────────────────────────────────────────────────────╯
```

### Step 4: Inspect Compiled Files

**Shell command** (separate terminal):
```bash
ls -la .paola_data/evaluators/rosenbrock_2d_eval/
cat .paola_data/evaluators/rosenbrock_2d_eval/source.py
cat .paola_data/evaluators/rosenbrock_2d_eval/metadata.json
```

**Expected**:
- `source.py`: Standalone function with imports
- `metadata.json`: Semantic information from LLM

**Key metadata fields**:
```json
{
  "semantics": {
    "type": "objective_function",
    "input_dimension": 2,
    "input_variables": ["x[0]", "x[1]"],
    "output_type": "scalar",
    "formula": "(1-x0)^2 + 100*(x1-x0^2)^2"
  }
}
```

---

## Scenario 2: Auto-Generated Variable Extractors

### Step 1: Create NLP with Natural Language Constraint

**User input**:
```
paola> Create an optimization problem with rosenbrock_2d as objective and constraint x[0] >= 1.5
```

**Expected agent behavior**:
- Agent uses `create_nlp_problem_smart` tool
- Reads rosenbrock_2d_eval semantics: dimension=2
- Parses constraint: `x[0] >= 1.5` → needs x0_extractor
- Checks if x0_extractor exists → No
- Auto-generates x0_extractor using `auto_generate_variable_extractor`
- Creates NLP problem with correct constraint

**Expected output**:
```
💭 I'll create the NLP problem with the constraint...
🔧 create_nlp_problem_smart...
✓ create_nlp_problem_smart completed

✓ Created NLP problem 'rosenbrock_constrained'
  Objective: minimize rosenbrock_2d_eval
  Dimension: 2
  Inequality constraints: 1
  Auto-generated extractors: x0_extractor
```

### Step 2: Verify Auto-Generated Extractor

**User input**:
```
paola> /evals
```

**Expected output** (should now show both evaluators):
```
╭─ Registered Evaluators ─────────────────────────────────────╮
│                                                              │
│  rosenbrock_2d_eval                                          │
│    Classic 2D Rosenbrock function                            │
│                                                              │
│  x0_extractor                                                │
│    Extracts x[0] from design vector                          │
│    Auto-generated                                            │
│                                                              │
╰──────────────────────────────────────────────────────────────╯
```

### Step 3: Inspect Auto-Generated Extractor

**Shell command**:
```bash
cat .paola_data/evaluators/x0_extractor/source.py
cat .paola_data/evaluators/x0_extractor/metadata.json
```

**Expected source.py**:
```python
"""
Evaluator: x0_extractor
Auto-generated variable extractor for x[0]
...
"""

import numpy as np

def x0_extractor(x):
    """
    Extract component 0 from design vector.

    Args:
        x: Design vector (dimension >= 1)

    Returns:
        x[0] as float
    """
    x = np.atleast_1d(x)
    return float(x[0])
```

---

## Scenario 3: Optimization with Constraints

### Step 1: Run Optimization

**User input**:
```
paola> Solve the rosenbrock_constrained problem with SLSQP starting from [0, 0]
```

**Expected agent behavior**:
- Agent uses `start_optimization_run` to create tracked run
- Agent uses `run_scipy_optimization` with:
  - problem_id: "rosenbrock_constrained"
  - algorithm: "SLSQP"
  - initial_design: [0, 0]
- Constraint is extracted and passed to scipy
- Optimization respects x[0] >= 1.5

**Expected output**:
```
💭 I'll run the optimization...
🔧 start_optimization_run...
✓ start_optimization_run completed

🔧 run_scipy_optimization...
Applying 1 inequality and 0 equality constraints
✓ run_scipy_optimization completed

✓ Optimization completed
  Final design: [1.5000, 2.2500]
  Final objective: 0.2500
  Iterations: 15
```

### Step 2: Verify Constraint Satisfaction

**Analysis**:
- Unconstrained Rosenbrock minimum: (1.0, 1.0) with f = 0
- Constraint: x[0] >= 1.5
- Expected constrained minimum: ~(1.5, 2.25) with f ≈ 0.25
- **Key test**: x[0] should be >= 1.5 in final solution

**Verification**:
```
Final x[0] = 1.5000 >= 1.5 ✓ CONSTRAINT SATISFIED!
```

### Step 3: Compare with Unconstrained

**User input**:
```
paola> Now solve unconstrained Rosenbrock with SLSQP from [0, 0]
```

**Expected**: Should find (1.0, 1.0) with f ≈ 0

**Comparison**:
| Case | x[0] | x[1] | f(x) | Constraint Satisfied |
|------|------|------|------|---------------------|
| Unconstrained | 1.0 | 1.0 | 0.0 | ✗ Violates x[0] >= 1.5 |
| Constrained | 1.5 | 2.25 | 0.25 | ✓ Satisfies x[0] >= 1.5 |

---

## Scenario 4: Multiple Constraints

### Step 1: Create Problem with Multiple Constraints

**User input**:
```
paola> Create optimization problem with rosenbrock_2d objective and constraints: x[0] >= 1.5 and x[1] <= 3.0
```

**Expected agent behavior**:
- Parses two constraints
- Auto-generates x0_extractor (already exists)
- Auto-generates x1_extractor (new)
- Creates NLP with both constraints

**Expected output**:
```
✓ Auto-generated extractors: x1_extractor
  (x0_extractor already exists)
```

### Step 2: Solve and Verify

**User input**:
```
paola> Solve this problem with SLSQP
```

**Expected**:
- Final solution should satisfy:
  - x[0] >= 1.5 ✓
  - x[1] <= 3.0 ✓

---

## Scenario 5: Immutability Test

### Step 1: Modify Original File

**Action**: Edit `debug_agent/test_evaluators.py` and change Rosenbrock:

```python
def rosenbrock_2d(x):
    """Changed function!"""
    return (1 - x[0])**2 + 10 * (x[1] - x[0]**2)**2  # Changed 100 → 10
```

### Step 2: Run Optimization Again

**User input**:
```
paola> Solve rosenbrock_constrained with SLSQP
```

**Expected behavior**:
- Uses compiled snapshot from `.paola_data/evaluators/rosenbrock_2d_eval/source.py`
- Original file change has NO EFFECT
- Results are identical to previous run

**Verification**:
```bash
# Check that compiled version still has 100, not 10
grep "100 \*" .paola_data/evaluators/rosenbrock_2d_eval/source.py
# Should find: "100 * (x[1] - x[0]**2)**2"
```

### Step 3: Re-register to Pick Up Changes

**User input**:
```
paola> Register rosenbrock_2d from debug_agent/test_evaluators.py as rosenbrock_2d_v2
```

**Expected**:
- Creates new evaluator: `rosenbrock_2d_v2_eval`
- This one has the modified code (10 instead of 100)
- Original `rosenbrock_2d_eval` remains unchanged

---

## Scenario 6: Complex Evaluator (Ackley Function)

### Step 1: Register Ackley Function

**User input**:
```
paola> Register ackley_2d from debug_agent/test_evaluators.py
```

**Expected**:
- LLM analyzes complex function with exponentials
- Identifies mathematical properties
- Creates compiled snapshot

### Step 2: Review Semantic Analysis

**User input**:
```
paola> /eval ackley_2d_eval
```

**Expected metadata**:
```json
{
  "semantics": {
    "type": "objective_function",
    "description": "2D Ackley function - nearly flat outer region",
    "input_dimension": 2,
    "output_type": "scalar",
    "properties": {
      "convex": false,
      "smooth": true,
      "differentiable": true
    },
    "dependencies": ["numpy"]
  }
}
```

### Step 3: Optimize with Constraints

**User input**:
```
paola> Optimize ackley_2d with constraint x[0] >= -2.0, starting from [4, 4]
```

**Expected**:
- Auto-generates x0_extractor (if not exists)
- Solves constrained problem
- Should find solution near x ≈ [0, 0] (global minimum)
- But respects constraint x[0] >= -2.0

---

## Scenario 7: Error Handling

### Test 1: Non-Existent Function

**User input**:
```
paola> Register nonexistent_function from debug_agent/test_evaluators.py
```

**Expected**:
```
✗ Function 'nonexistent_function' not found in debug_agent/test_evaluators.py
```

### Test 2: Invalid Constraint

**User input**:
```
paola> Create optimization with rosenbrock_2d and constraint x[5] >= 1.0
```

**Expected**:
```
✗ Constraint references x[5] but dimension is 2
```

### Test 3: Wrong Evaluator Type

**User input**:
```
paola> Create optimization with x0_extractor as objective
```

**Expected**:
```
✗ 'x0_extractor' is a variable extractor, not an objective function
```

---

## Verification Checklist

After running all scenarios, verify:

- [ ] Agentic registration created compiled evaluators
- [ ] Semantic metadata is accurate
- [ ] Variable extractors auto-generated correctly
- [ ] Constraints are enforced during optimization
- [ ] Immutability: original file changes don't affect compiled evaluators
- [ ] Multiple constraints work together
- [ ] Complex functions (Ackley) analyzed correctly
- [ ] Error handling is appropriate

---

## Directory Structure After Testing

```
.paola_data/
├── evaluators/
│   ├── rosenbrock_2d_eval/
│   │   ├── source.py
│   │   └── metadata.json
│   ├── ackley_2d_eval/
│   │   ├── source.py
│   │   └── metadata.json
│   ├── x0_extractor/
│   │   ├── source.py
│   │   └── metadata.json
│   └── x1_extractor/
│       ├── source.py
│       └── metadata.json
├── runs/
│   └── ... (optimization run data)
└── cache/
```

---

## Expected Costs

- Evaluator registration (LLM analysis): ~$0.01 per function
- Total for this scenario: ~$0.05
- Worth it for semantic understanding and auto-generation!

---

## Troubleshooting

### Issue: LLM analysis fails

**Solution**: Check LLM credentials in `.env`

### Issue: Import errors in compiled evaluators

**Solution**: Verify imports are at top of source file

### Issue: Constraint not enforced

**Solution**: Check that SLSQP is used (supports constraints)

### Issue: Auto-generation fails

**Solution**: Verify dimension is correctly inferred from semantics

---

## Success Criteria

✅ All scenarios complete without errors
✅ Constraints are correctly enforced
✅ Variable extractors auto-generated
✅ Semantic metadata is accurate
✅ Immutability preserved
✅ Agent demonstrates true semantic understanding

---

## Next Steps After Testing

1. Try with your own evaluator functions
2. Test with complex constraints (inequalities + equalities)
3. Experiment with different optimization algorithms
4. Compare performance: old vs new architecture
5. Share feedback on semantic analysis accuracy

---

**Happy Testing! 🚀**
