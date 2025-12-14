# Evaluator Registration Design - Summary

**Date**: December 14, 2025
**Status**: Design Complete, Ready for Implementation

---

## What We Designed

**New architecture for evaluator integration based on REGISTRATION, not wrapper code generation.**

### Core Innovation

**Philosophy**: "PAOLA adapts to your code, not vice versa"

**Implementation**:
- Users bring their evaluator (Python function, CLI tool, etc.)
- LLM agent **registers** it in Foundry (generates JSON config, not code)
- **FoundryEvaluator** (PAOLA infrastructure) handles all evaluations with built-in capabilities
- Foundry is single source of truth

---

## Key Documents Created

### 1. Architecture Document
**Location**: `docs/architecture/evaluator_registration.md`

**Contents**:
- Problem statement (why existing platforms are hard to use)
- Design principles (configuration over code generation)
- Architecture (3-layer: User code → LLM agent → Foundry → FoundryEvaluator)
- Configuration schema (JSON metadata, not Python code)
- FoundryEvaluator implementation (infrastructure with PAOLA capabilities)
- LLM agent flow (read → generate config → test → iterate → store)
- User evaluator levels (Level 0: Python functions, Level 1: CLI executables)
- Benefits over wrapper generation

**Key sections**:
- Executive Summary
- Registration Configuration Schema
- FoundryEvaluator Implementation
- LLM-Native Discovery
- Success Criteria

---

### 2. Implementation Plan
**Location**: `docs/implementation/phase6_week1_registration.md`

**Contents**:
- 5-day detailed implementation plan
- Day-by-day breakdown:
  - Day 1: FoundryEvaluator infrastructure
  - Day 2: Configuration schema and storage
  - Day 3: LLM agent registration
  - Day 4: CLI integration
  - Day 5: Comprehensive testing (20+ Level 0, 10+ Level 1 cases)
- Success metrics
- Code volume estimates
- Testing strategy

**Deliverables**:
- Flawless Level 0 (Python functions): 100% success on 20 patterns
- Extremely robust Level 1 (CLI executables): 90%+ success on 10 patterns
- Built-in PAOLA capabilities (observation, caching, cost tracking)

---

### 3. Decision Document
**Location**: `docs/decisions/registration_vs_wrapper_architecture.md`

**Contents**:
- Why we chose registration over wrapper generation
- Comparison of approaches
- What changes vs what stays
- Migration path
- Lessons learned

**Key decisions**:
- Configuration (JSON) over code generation (Python)
- Foundry as single source of truth
- FoundryEvaluator as universal infrastructure
- LLM generates metadata, not executable code

---

## Architecture Overview

```
┌──────────────────────────────────────┐
│  USER'S CODE                         │
│  - Python function                   │
│  - CLI executable                    │
│  - Workflow                          │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  LLM AGENT                           │
│  - Reads user's code                 │
│  - Generates JSON configuration      │
│  - Tests configuration               │
│  - Iterates if needed                │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  FOUNDRY (Single Source of Truth)    │
│  {                                   │
│    "source": {...},                  │
│    "interface": {...},               │
│    "capabilities": {...}             │
│  }                                   │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  FOUNDRYEVALUATOR (Infrastructure)   │
│  - Loads config                      │
│  - Calls user's function directly    │
│  - Adds observation gates            │
│  - Manages caching                   │
│  - Tracks cost                       │
└──────────────────────────────────────┘
```

---

## What Gets Stored in Foundry

**Not Python wrapper code, but JSON configuration:**

```json
{
  "evaluator_id": "eval_abc123",
  "name": "airfoil_cfd",
  "source": {
    "type": "python_function",
    "file_path": "/path/to/my_eval.py",
    "callable_name": "evaluate_design"
  },
  "interface": {
    "input": {"type": "numpy_array"},
    "output": {"format": "dict", "keys": ["drag", "lift"]}
  },
  "capabilities": {
    "observation_gates": true,
    "caching": true,
    "cost_tracking": true
  },
  "performance": {
    "median_time": 4.2,
    "cost_per_eval": 4.0
  }
}
```

**~20 lines of metadata vs 50-100 lines of generated Python code!**

---

## User Experience (CLI)

```
User: I have a function in my_eval.py

Agent: [Reads file]
       [Generates configuration]
       [Tests it]

       ✓ Registered as 'my_evaluator' (ID: eval_abc123)

       PAOLA capabilities enabled:
       - Observation gates: ✓
       - Evaluation caching: ✓
       - Cost tracking: ✓

       Ready to optimize!

User: Evaluate at x = [2.0, 2.0]

Agent: [Retrieves eval_abc123 from Foundry]
       [Calls via FoundryEvaluator]

       Result: drag = 8.0, lift = 0.5
```

**Simple, fast, no code generation!**

---

## Key Benefits

### 1. Simpler Implementation
- ~1,400 lines (registration) vs ~2,000+ lines (wrapper generation)
- Configuration validation easier than code validation
- JSON schema simpler than code templates

### 2. Better Maintenance
- Update FoundryEvaluator (1 class) to add capabilities
- NOT update N generated wrappers
- Single source of truth in Foundry

### 3. Clearer User Experience
- "Registration" is clear terminology
- "Wrapper" was confusing
- Users understand what's happening

### 4. More Scalable
- Works for ANY evaluator type through configuration
- LLM reasons about patterns, no hardcoded templates
- Easy to add new evaluator types

### 5. LLM-Native
- LLM generates JSON (natural for LLMs)
- LLM tests by executing Python
- LLM iterates when test fails
- No code generation fragility

---

## Current Codebase Status

### Files That Stay (Unchanged)
✅ `paola/backends/base.py` - EvaluationBackend, EvaluationResult
✅ `paola/backends/analytical.py` - Test functions
✅ `paola/backends/user_function.py` - Keep as utility (used internally)
✅ `paola/modeling/parsers.py` - Problem parsing
✅ `paola/modeling/validation.py` - Problem validation
✅ `paola/formulation/schema.py` - OptimizationProblem, Variable, etc.
✅ `test_phase6_modeling.py` - Tests still valid

### Files to Create (Phase 6 Week 1)
📝 `paola/foundry/evaluator.py` - FoundryEvaluator infrastructure
📝 `paola/foundry/capabilities.py` - Observer, Cache
📝 `paola/foundry/schemas.py` - Configuration schema
📝 `paola/foundry/evaluator_storage.py` - Storage layer
📝 `paola/tools/evaluator_tools.py` - Agent tools
📝 `paola/agent/prompts/evaluator_registration.py` - System prompt
📝 Tests for all above

### Files NOT Created (No Wrapper Generation)
❌ No `template_generator.py`
❌ No wrapper templates
❌ No per-user generated code

**Clean slate for registration implementation!**

---

## Success Criteria

### Level 0 (Python Functions) - FLAWLESS
- ✅ 100% success on 20 common function patterns
- ✅ Registration in < 30 seconds
- ✅ LLM generates correct config without user guidance (95%+)
- ✅ Clear error messages if test fails

### Level 1 (CLI Executables) - EXTREMELY ROBUST
- ✅ 90%+ success on 10 common I/O patterns
- ✅ Registration in < 5 minutes
- ✅ LLM iterates successfully when needed
- ✅ User confirms critical choices

### Infrastructure - PAOLA CAPABILITIES
- ✅ Observation gates work for all evaluators
- ✅ Caching reduces redundant calls by 80%+
- ✅ Cost tracking accurate within 10%
- ✅ Interjection points functional

---

## Implementation Timeline

**Phase 6 Week 1**: 5 days

| Day | Focus | Deliverable |
|-----|-------|------------|
| 1 | FoundryEvaluator + Capabilities | Infrastructure working |
| 2 | Config schema + Storage | Configs stored in Foundry |
| 3 | LLM agent + Tools | Registration flow working |
| 4 | CLI integration | User can register via CLI |
| 5 | Comprehensive testing | 20+ Level 0, 10+ Level 1 tests pass |

**Ready to start implementation Monday!** 🚀

---

## Next Steps

1. **Review documents**:
   - Architecture: `docs/architecture/evaluator_registration.md`
   - Implementation plan: `docs/implementation/phase6_week1_registration.md`
   - Decision rationale: `docs/decisions/registration_vs_wrapper_architecture.md`

2. **Approve design**:
   - Confirm registration approach
   - Review success criteria
   - Adjust timeline if needed

3. **Begin implementation**:
   - Day 1: FoundryEvaluator infrastructure
   - Follow 5-day plan

---

## Questions for Review

1. **Architecture**: Is the registration approach (config over code) the right design?
2. **Configuration schema**: Is the JSON structure comprehensive enough?
3. **Success criteria**: Are the metrics (100% Level 0, 90% Level 1) realistic?
4. **Timeline**: Is 5 days sufficient for implementation?
5. **Scope**: Should we add anything to Week 1 deliverables?

---

## Terminology Reference

**Use**:
- ✅ "Registration" (process)
- ✅ "Registered evaluator" (result)
- ✅ "Register in Foundry"
- ✅ "FoundryEvaluator" (infrastructure)
- ✅ "Configuration" (what's stored)

**Avoid**:
- ❌ "Wrapper"
- ❌ "Wrapped function"
- ❌ "Generate wrapper code"
- ❌ "Template"

---

**Design complete! Ready for your review and approval.** ✅
