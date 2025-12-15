# Decision: Registration vs Wrapper Architecture

**Date**: December 14, 2025
**Status**: Decided - Registration Architecture
**Impact**: Phase 6 implementation approach

---

## Context

During Phase 6 design discussion, two approaches emerged for integrating user evaluators:

1. **Wrapper Generation**: LLM generates Python wrapper code per user
2. **Registration**: LLM generates JSON configuration, infrastructure handles execution

---

## Decision

**Use Registration Architecture** ✅

---

## Rationale

### Why NOT Wrapper Generation

**Problems with wrapper approach**:

1. **Per-user code maintenance**
   - Generate 50-100 lines of Python per user
   - Update all wrappers when adding PAOLA capabilities
   - N users = N wrappers to maintain

2. **PAOLA capabilities scattered**
   - Observation gates in generated code
   - Caching in generated code
   - Cost tracking in generated code
   - Hard to ensure consistency

3. **Testing complexity**
   - Must test generated code quality
   - Edge cases in code generation
   - Template maintenance burden

4. **Unclear terminology**
   - "Wrapper" confuses users
   - "Generate wrapper code" sounds heavy

5. **Not aligned with philosophy**
   - Users still adapt to our code generation patterns
   - Not truly "PAOLA adapts to your code"

---

### Why Registration

**Benefits of registration approach**:

1. **Configuration over code**
   - 20 lines JSON config vs 50-100 lines Python
   - LLM generates metadata, not executable code
   - Easier to validate, store, evolve

2. **Single source of truth**
   - Foundry stores configurations
   - FoundryEvaluator is ONE infrastructure class
   - Update one place, all evaluators benefit

3. **Built-in capabilities**
   - Observation gates in FoundryEvaluator
   - Caching in FoundryEvaluator
   - Cost tracking in FoundryEvaluator
   - Consistent behavior across all evaluators

4. **Maintainability**
   - Add new capability → update FoundryEvaluator (1 class)
   - NOT update N wrappers
   - Much cleaner evolution path

5. **Clear terminology**
   - "Registration" is clear and accurate
   - "Foundry as single truth" reinforces architecture
   - Users understand they're registering, not wrapping

6. **LLM-native**
   - LLM generates JSON (natural for LLMs)
   - LLM tests by executing Python (infrastructure)
   - LLM iterates on configuration (debug → fix → test)

7. **Scalable**
   - Works for ANY evaluator type through config
   - Python functions, CLI tools, APIs, workflows
   - Same infrastructure handles all

---

## What Changes

### Old Approach (Discarded)

**Files that would have been created**:
- ❌ `paola/backends/template_generator.py` - Generate wrapper code
- ❌ `paola/backends/executable_wrapper.py` - Template library
- ❌ `paola/backends/wrapper_validation.py` - Validate generated code
- ❌ Template files (JSON, text, CSV patterns)

**Estimated code**: ~1,000 lines of template/generation logic

---

### New Approach (Implemented)

**Files to create**:
- ✅ `paola/foundry/evaluator.py` - FoundryEvaluator (infrastructure)
- ✅ `paola/foundry/capabilities.py` - Observer, Cache, etc.
- ✅ `paola/foundry/schemas.py` - Configuration schema (Pydantic)
- ✅ `paola/foundry/evaluator_storage.py` - Store/retrieve configs
- ✅ `paola/tools/evaluator_tools.py` - Agent tools (read_file, execute_python, etc.)
- ✅ `paola/agent/prompts/evaluator_registration.py` - System prompt

**Estimated code**: ~800 lines of infrastructure (write once, use forever)

---

## What Stays from Previous Work

### Keep (Still Valid)

1. **`paola/backends/base.py`** ✅
   - EvaluationBackend interface
   - EvaluationResult dataclass
   - Unchanged

2. **`paola/backends/analytical.py`** ✅
   - Analytical test functions
   - Unchanged

3. **`paola/modeling/parsers.py`** ✅
   - Problem parsing
   - Unchanged

4. **`paola/modeling/validation.py`** ✅
   - Problem validation
   - Unchanged

5. **`paola/formulation/schema.py`** ✅
   - OptimizationProblem, Variable, Objective, Constraint
   - Unchanged

6. **`test_phase6_modeling.py`** ✅
   - Tests for problem modeling
   - Still valid

---

### Modify (Needs Update)

1. **`paola/backends/user_function.py`** 🔄
   - **Current**: Simple wrapper class for Python callables
   - **Status**: Keep as utility, but NOT primary interface
   - **Plan**: Use internally by FoundryEvaluator for Python function type
   - **Note**: The flexible return format handling is still useful!

2. **`paola/backends/__init__.py`** 🔄
   - **Update**: Export FoundryEvaluator as primary
   - **Keep**: UserFunctionBackend as internal utility

---

### Remove (Not Needed)

Nothing to remove - we stopped before implementing wrapper generation! ✅

---

## Architecture Comparison

### Wrapper Generation (Old)

```
User's Python function
    ↓
LLM reads code
    ↓
LLM generates Python wrapper code (50-100 lines)
    ├─ Import user's function
    ├─ Add observation gates
    ├─ Add caching
    ├─ Add cost tracking
    └─ Return EvaluationResult
    ↓
Store wrapper code in Foundry
    ↓
Execute wrapper when evaluating
```

**Issues**:
- N users = N wrappers
- PAOLA capabilities in generated code
- Hard to update all wrappers

---

### Registration (New)

```
User's Python function
    ↓
LLM reads code
    ↓
LLM generates JSON configuration (20 lines)
    {
      "source": {"file_path": "...", "callable_name": "..."},
      "interface": {"output": {"format": "dict"}},
      "capabilities": {"observation_gates": true, "caching": true}
    }
    ↓
Store configuration in Foundry
    ↓
FoundryEvaluator (PAOLA infrastructure)
    ├─ Loads configuration
    ├─ Imports user's function directly
    ├─ Adds observation gates (built-in)
    ├─ Adds caching (built-in)
    ├─ Adds cost tracking (built-in)
    └─ Returns EvaluationResult
```

**Benefits**:
- N users = N configs (lightweight)
- PAOLA capabilities in ONE infrastructure class
- Easy to update: modify FoundryEvaluator

---

## Implementation Impact

### Phase 6 Week 1 Plan

**No changes to timeline**: Still 5 days

**Changes to deliverables**:
- Day 1: FoundryEvaluator (instead of wrapper templates)
- Day 2: Configuration schema (instead of template library)
- Day 3: LLM agent for config generation (instead of code generation)
- Day 4-5: Same (CLI integration, testing)

**Code volume**: ~1,400 lines (registration) vs ~2,000 lines (wrapper generation)

**Complexity**: Lower (configuration simpler than code generation)

---

## Migration Path

### For Current Codebase

**Step 1**: Keep existing code
- `paola/backends/base.py` - No changes
- `paola/backends/user_function.py` - Keep as utility
- `paola/backends/analytical.py` - No changes
- `paola/modeling/` - No changes

**Step 2**: Implement new architecture
- `paola/foundry/evaluator.py` - NEW
- `paola/foundry/capabilities.py` - NEW
- `paola/foundry/schemas.py` - NEW (or add to existing)
- `paola/foundry/evaluator_storage.py` - NEW

**Step 3**: Update imports
- Primary import: `from paola.foundry.evaluator import FoundryEvaluator`
- Internal use: `UserFunctionBackend` used by FoundryEvaluator

**No breaking changes** - additive only! ✅

---

## User-Facing Changes

### Terminology

**Before** (wrapper):
- "PAOLA generates a wrapper for your function"
- "Wrapper code is stored in Foundry"
- "Update wrapper if function changes"

**After** (registration):
- "PAOLA registers your function in Foundry" ✅
- "Configuration is stored in Foundry" ✅
- "Re-register if function changes" ✅

### User Experience

**Before** (wrapper):
```
User: "I have a function in my_eval.py"
Agent: "Generating wrapper code..."
       [Shows 50 lines of generated Python]
       "Review this wrapper code?"
User: "Uh... looks complicated..."
```

**After** (registration):
```
User: "I have a function in my_eval.py"
Agent: "Registering in Foundry with PAOLA capabilities..."
       "✓ Registered! Observation gates, caching, cost tracking enabled."
User: "That was easy!" ✅
```

---

## Lessons Learned

1. **Think in terms of data, not code**
   - Configuration is data (JSON)
   - Infrastructure is code (FoundryEvaluator)
   - Don't generate code when data suffices

2. **Single source of truth matters**
   - One infrastructure class >> N generated wrappers
   - Updates propagate instantly
   - Consistency guaranteed

3. **LLM-native doesn't mean code generation**
   - LLMs are great at reading code
   - LLMs are great at generating configs
   - LLMs don't need to generate wrapper code

4. **Listen to user feedback**
   - "Wrapper" was confusing terminology
   - "Registration" is clear and accurate
   - User clarity leads to better design

5. **Philosophy guides architecture**
   - "PAOLA adapts to your code" → registration
   - Not "Generate code that wraps your code" → wrapper
   - Stay true to core philosophy

---

## Conclusion

**Registration architecture is fundamentally better than wrapper generation.**

- Simpler implementation
- Easier maintenance
- Clearer user experience
- More aligned with philosophy
- More scalable

**This decision improves PAOLA's architecture.** ✅

---

## References

- Architecture document: `docs/architecture/evaluator_registration.md`
- Implementation plan: `docs/implementation/phase6_week1_registration.md`
- Original discussion: Session on December 14, 2025

---

**Decision: APPROVED** ✅
**Status: Proceeding with registration architecture**
