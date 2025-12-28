# Evaluator Registration: Deterministic vs LLM-Powered

**Date**: 2025-12-14
**Question**: Should evaluator registration be deterministic (current) or LLM-powered (original design)?

---

## The Fundamental Question

When a user registers an evaluator like `rosenbrock_2d(x)`, should the system:

**Option A (Current)**: Parse the code deterministically and build config mechanically
**Option B (Original Design)**: Use LLM to understand the code semantically and generate config

This is a **fundamental architectural decision** about agent autonomy.

---

## Current Implementation (Deterministic)

### Code Location
**File**: `/paola/cli/commands.py:562-696` (`handle_register`)

### How It Works

```python
# User provides manually:
function_name = input("  Function name: ").strip()
evaluator_name = input("  Evaluator name: ").strip()
evaluator_id = input("  Evaluator ID: ").strip()

# Build config deterministically
config = {
    "evaluator_id": evaluator_id,
    "name": evaluator_name,
    "source": {
        "type": "python_function",
        "file_path": str(path.absolute()),
        "callable_name": function_name
    },
    "interface": {
        "output": {"format": "auto"}  # ← GENERIC, no semantic understanding
    },
    "capabilities": {
        "observation_gates": True,
        "caching": True
    },
    "performance": {
        "cost_per_eval": 1.0
    }
}
```

### What Gets Stored

For `rosenbrock_2d(x)`:
```python
{
    "evaluator_id": "rosenbrock_eval",
    "interface": {
        "output": {"format": "auto"}  # ← Agent doesn't know dimension, variable count, etc.
    }
}
```

### Pros
✅ Fast and predictable
✅ No LLM costs
✅ Deterministic behavior
✅ Simple implementation

### Cons
❌ **Agent has NO semantic understanding**
❌ Doesn't know problem dimension
❌ Doesn't know design variable count
❌ Doesn't understand input/output structure
❌ **Cannot autonomously create constraint evaluators**

---

## Original Design (LLM-Powered)

### Code Location
**Doc**: `/docs/architecture/evaluator_registration.md:410-484`

### How It Works

```python
# Agent workflow (from design doc):

# 1. Agent reads the file
file_contents = read_file(file_path)

# 2. LLM analyzes code semantically
prompt = f"""
Analyze this evaluator function:

{file_contents}

Generate a configuration that describes:
- Dimension of input
- Names of design variables
- Output structure (scalar, vector, dict)
- Physical meaning if clear
"""

# 3. LLM generates config with semantic understanding
config = {
    "evaluator_id": "rosenbrock_eval",
    "name": "Rosenbrock 2D",
    "source": {...},
    "interface": {
        "input": {
            "type": "numpy_array",
            "dimension": 2,              # ← Agent KNOWS dimension!
            "variable_names": ["x0", "x1"]  # ← Agent KNOWS variable names!
        },
        "output": {
            "format": "scalar",           # ← Agent KNOWS output structure!
            "physical_meaning": "objective_value"
        }
    },
    "metadata": {
        "description": "2D Rosenbrock function, classic optimization benchmark",
        "problem_type": "continuous",
        "recommended_bounds": [[-5, 10], [-5, 10]]
    }
}
```

### What Gets Stored

For `rosenbrock_2d(x)`:
```python
{
    "evaluator_id": "rosenbrock_eval",
    "interface": {
        "input": {
            "dimension": 2,                 # ← SEMANTIC UNDERSTANDING!
            "variable_names": ["x0", "x1"]  # ← KNOWS the variables!
        }
    }
}
```

### Pros
✅ **Agent has semantic understanding**
✅ Knows problem dimension automatically
✅ Understands design variables
✅ Can infer input/output structure
✅ **Can autonomously create constraint evaluators**
✅ Better error messages
✅ Smarter optimization decisions

### Cons
❌ Costs LLM tokens (~$0.001-0.01 per registration)
❌ Slower (2-5 seconds vs instant)
❌ Less predictable (LLM might misunderstand)
❌ More complex implementation

---

## The Critical Insight: Constraint Creation

### User's Observation

> "If we give paola agent the autonomy of reading and registering the Evals to Foundry, then she has clearer insight in the optimization problem (including the design variables). In our test task, she might then be able to come up with the correct inequality constraint x>=1.5."

This is **absolutely correct**. Here's why:

### Scenario: User wants `x[0] >= 1.5` constraint

**With Deterministic Registration (Current)**:

```
User: "Create optimization with rosenbrock as objective and constraint x >= 1.5"

Agent thinks:
1. I need objective evaluator: rosenbrock_eval ✓
2. I need constraint evaluator for x[0] >= 1.5
3. Check registered evaluators...
4. No evaluator returns x[0]!
5. I need to create one... but how?
6. I don't know the dimension of the problem
7. I don't know variable names
8. ❌ STUCK: Try using bounds instead (WRONG)
   OR
9. ❌ STUCK: Ask user to create x0_extractor manually
```

**With LLM-Powered Registration (Original Design)**:

```
User: "Create optimization with rosenbrock as objective and constraint x >= 1.5"

Agent thinks:
1. I need objective evaluator: rosenbrock_eval ✓
2. Retrieve rosenbrock_eval metadata:
   {
     "dimension": 2,
     "variable_names": ["x0", "x1"]
   }
3. I need constraint evaluator for x[0] >= 1.5
4. Check registered evaluators...
5. No evaluator returns x[0]
6. ✓ I KNOW the problem has dimension 2 and variable x0!
7. ✓ I can create x0_extractor automatically:

   def x0_extractor(x):
       """Returns first design variable x[0]"""
       return x[0]

8. Register x0_extractor to Foundry
9. Use it in constraint: x0_extractor >= 1.5
10. ✓ SUCCESS!
```

### Key Difference

- **Deterministic**: Agent is blind to problem structure → gets stuck
- **LLM-Powered**: Agent understands problem structure → creates solution autonomously

---

## Test Case Analysis

### Your Test (with Deterministic Registration)

```
paola> create optimization with rosenbrock and constraint x >= 1.5

Agent actions:
1. ✓ Get rosenbrock_eval
2. ✓ Create NLP problem
3. ❌ Use rosenbrock_eval as constraint (WRONG!)
   inequality_constraints=[{
       "evaluator_id": "rosenbrock_eval",  # ← Should be x0_extractor!
       "type": ">=",
       "value": 1.5
   }]
4. Result: Constraint "rosenbrock(x) >= 1.5" instead of "x[0] >= 1.5"
5. ❌ WRONG CONSTRAINT!
```

**Root cause**: Agent doesn't know the problem has 2 variables [x0, x1], so it can't create x0_extractor.

### Same Test (with LLM-Powered Registration)

```
paola> create optimization with rosenbrock and constraint x >= 1.5

Agent actions:
1. ✓ Get rosenbrock_eval → sees dimension=2, variables=[x0, x1]
2. ✓ Agent understands: "User wants x[0] >= 1.5, not rosenbrock(x) >= 1.5"
3. ✓ Check for x0_extractor → not found
4. ✓ Create x0_extractor on-the-fly:

   code = '''
   def x0_extractor(x):
       """Returns first design variable"""
       return x[0]
   '''
   execute_python(code)
   foundry_store_evaluator({
       "evaluator_id": "x0_extractor",
       "dimension": 2,
       "outputs": ["scalar"]
   })

5. ✓ Create NLP with correct constraint:
   inequality_constraints=[{
       "evaluator_id": "x0_extractor",  # ← CORRECT!
       "type": ">=",
       "value": 1.5
   }]
6. ✓ CORRECT CONSTRAINT!
```

**Success factor**: Agent has semantic understanding of problem structure.

---

## Implementation Complexity

### Current (Deterministic)
- Lines of code: ~135 lines (commands.py:562-696)
- Complexity: Low
- Dependencies: None

### LLM-Powered
- Lines of code: ~200-300 lines (estimated)
- Complexity: Medium
- Dependencies: LLM API calls, retry logic

### Migration Path

Could implement **hybrid approach**:

```python
def handle_register(self, file_path: str, mode: str = "auto"):
    """
    Register evaluator with configurable mode.

    Args:
        file_path: Path to evaluator file
        mode: Registration mode
            - "auto": Use LLM if agent is active, otherwise deterministic
            - "llm": Force LLM-powered (agentic)
            - "manual": Force deterministic (current)
    """
    if mode == "llm" or (mode == "auto" and self.agent_active):
        return self._register_with_llm(file_path)
    else:
        return self._register_manual(file_path)
```

---

## Performance Comparison

### Deterministic
- Registration time: ~1 second (I/O only)
- Cost: $0
- Accuracy: 100% for what it does (but limited)

### LLM-Powered
- Registration time: 3-5 seconds (LLM call + I/O)
- Cost: $0.001-0.01 per registration
- Accuracy: 90-95% (LLM might misunderstand complex code)

### Cost Analysis

For typical user with 10 evaluators:
- Deterministic: $0
- LLM-Powered: $0.01-0.10 total

**This is negligible** compared to optimization costs ($1-100 per run).

---

## Strategic Value: Agentic Platform Vision

From `CLAUDE.md`:

> "PAOLA: The optimization platform that learns from every run"
>
> "The first optimization platform where an AI agent continuously observes optimization progress, detects feasibility and convergence issues, **autonomously adapts strategy**, accumulates knowledge..."

### Key Question: What Enables Autonomy?

**Semantic understanding of problem structure.**

Without knowing:
- Problem dimension
- Design variable names/count
- Input/output structure

The agent **cannot** autonomously:
- Create constraint evaluators
- Reformulate problems
- Suggest bounds
- Adapt strategies intelligently

### Example: Autonomous Problem Reformulation

**User**: "Optimize airfoil drag with lift >= 1000"

**With LLM Registration**:
```
Agent:
1. Retrieves drag_eval metadata:
   - dimension: 2 (chord, thickness)
   - variable_names: ["chord_length", "thickness_ratio"]

2. Retrieves lift_eval metadata:
   - dimension: 2 (same variables)
   - outputs: ["lift_force"]

3. ✓ Creates NLP:
   - Objective: minimize drag_eval
   - Constraint: lift_eval >= 1000

4. ✓ AUTONOMOUS SUCCESS
```

**Without LLM Registration**:
```
Agent:
1. Retrieves drag_eval → no dimension info
2. Retrieves lift_eval → no dimension info
3. ❌ Cannot verify compatibility
4. ❌ Creates problem blindly, might fail at runtime
5. ❌ NEEDS USER INTERVENTION
```

---

## Recommendation

### Short Answer
**Implement LLM-powered registration** for agentic operation.

### Why

1. **Aligns with Platform Vision**: PAOLA is fundamentally agentic - agent needs semantic understanding
2. **Enables Autonomy**: Agent can create constraint evaluators, reformulate problems automatically
3. **Solves Real User Pain**: Your test case shows this is not theoretical - it's a real blocker
4. **Negligible Cost**: $0.001-0.01 per registration vs $1-100 per optimization run
5. **Better UX**: Agent handles complexity instead of user

### Migration Strategy

**Phase 1** (Immediate - 1 day):
- Add LLM-powered registration alongside current method
- Use for agent-driven registration (via tools)
- Keep manual registration as fallback

**Phase 2** (1 week):
- Add automatic variable extractor generation
- Agent creates x0, x1, ..., xN extractors when needed

**Phase 3** (2 weeks):
- Knowledge accumulation: Store learned evaluator patterns
- Smarter registration over time

### Implementation Approach

```python
# In tools/registration_tools.py

@tool
def register_evaluator_with_llm(
    file_path: str,
    user_hints: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Register evaluator using LLM to understand code semantically.

    The agent reads the file, uses LLM to extract:
    - Dimension
    - Variable names
    - Input/output structure
    - Physical meaning

    Returns full configuration with semantic metadata.
    """
    # Read file
    code = read_file(file_path)

    # LLM analyzes
    analysis = llm.invoke(f"""
    Analyze this evaluator function:

    {code}

    Extract:
    1. Input dimension
    2. Variable names
    3. Output structure (scalar/vector/dict)
    4. Physical meaning

    Return JSON configuration.
    """)

    # Generate config with semantic understanding
    config = build_config_from_analysis(analysis)

    # Test
    test_result = test_evaluator(config)

    # Store with full metadata
    return foundry_store_evaluator(config, test_result)
```

---

## Conclusion

The user's insight is **correct and fundamental**:

> "If we give paola agent the autonomy of reading and registering the Evals to Foundry, then she has clearer insight in the optimization problem (including the design variables). In our test task, she might then be able to come up with the correct inequality constraint x>=1.5."

**The current deterministic registration blocks agent autonomy.**

To achieve the vision of PAOLA as an agentic platform that autonomously composes strategies, the agent **must have semantic understanding** of evaluators.

**Recommendation**: Implement LLM-powered registration as the default for agent-driven workflows.

---

## Action Items

- [ ] Implement `register_evaluator_with_llm` tool (1 day)
- [ ] Add automatic variable extractor generation (2 days)
- [ ] Update agent prompts to use LLM registration (1 day)
- [ ] Test with constraint creation workflow (0.5 days)
- [ ] Document semantic metadata schema (0.5 days)

**Total estimate**: 5 days to full LLM-powered registration

This aligns with the **original architecture design** and unlocks true agent autonomy.
