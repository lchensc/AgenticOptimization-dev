# Paola v0.4.0 Refactoring Plan

**Date**: 2025-12-17
**Current Version**: v0.3.1
**Target Version**: v0.4.0

## Overview

This document outlines the refactoring plan to simplify Paola's codebase by:
1. Removing legacy/unused code
2. Consolidating overlapping modules
3. Simplifying folder structure

---

## Current State Analysis

### Package Statistics
- **Total Python files**: 87
- **Total lines**: ~18,500
- **Folders**: 21 directories

### Current Structure
```
paola/
├── agent/           # LangGraph agents (conversational, react)
├── analysis/        # Metrics computation
├── backends/        # Evaluation backends (analytical functions)
├── callbacks/       # Event system for real-time monitoring
├── cli/             # Interactive CLI
├── formulation/     # EMPTY - just schema.py (problem definition)
├── foundry/         # Data foundation layer (27 files!)
│   ├── schema/      # Polymorphic components
│   └── storage/     # File storage backend
├── knowledge/       # Skeleton - not active
├── llm/             # Token tracking
├── modeling/        # Problem parsing (not integrated)
├── optimizers/      # Optimizer backends + gate
├── skills/          # Optimizer expertise (YAML)
├── tools/           # 15 tool files!
└── utils/           # Empty
```

---

## Issues Identified

### 1. Legacy/Unused Code

| File/Module | Status | Reason |
|------------|--------|--------|
| `tools/run_tools.py` | Legacy | Session-based run management, replaced by graph_tools |
| `tools/modeling_tools.py` | Legacy | Has `get_problem_info` that duplicates optimization_tools |
| `tools/smart_nlp_creation.py` | Experimental | Only used in debug_agent, not integrated |
| `tools/agentic_registration.py` | Experimental | Only used in debug_agent, not integrated |
| `modeling/` folder | Unused | Problem parsing not integrated into main workflow |
| `formulation/` folder | Unused | Just schema.py, not used anywhere |
| `utils/__init__.py` | Empty | No content |
| `knowledge/` folder | Skeleton | Interfaces only, no implementation |
| `optimizers/gate.py` | Legacy | Gate-based iteration control, not used in v0.3.x |
| `tools/gate_control_tools.py` | Legacy | Tools for gate, not used in v0.3.x |

### 2. Overlapping Functionality

| Issue | Location | Overlap |
|-------|----------|---------|
| `get_problem_info` | optimization_tools.py AND modeling_tools.py | Duplicate function |
| Problem schema | formulation/schema.py AND foundry/nlp_schema.py | Both define problems |
| Evaluator code | 5+ files in foundry/ | evaluator.py, evaluator_schema.py, evaluator_compiler.py, evaluator_storage.py, nlp_evaluator.py |

### 3. Complex Folder Structure

**Foundry folder is too complex (27 files)**:
```
foundry/
├── active_graph.py      # Active graph tracking
├── bounds_spec.py       # Bounds parsing
├── capabilities.py      # Optimizer capabilities
├── evaluator_compiler.py # Evaluator compilation
├── evaluator.py         # Evaluator class
├── evaluator_schema.py  # Evaluator schema
├── evaluator_storage.py # Evaluator storage
├── foundry.py           # Main class
├── nlp_evaluator.py     # NLP evaluator
├── nlp_schema.py        # NLP schema
├── problem.py           # Problem class
├── problem_types.py     # Problem types
├── schema/              # 11 more files
└── storage/             # 3 more files
```

**Tools folder has 15 files** (some not used):
```
tools/
├── __init__.py              # 132 lines of exports
├── agentic_registration.py  # NOT used in CLI
├── analysis.py              # Used
├── cache_agent_tools.py     # Used
├── cache_tools.py           # Used
├── evaluator_tools.py       # Used
├── gate_control_tools.py    # NOT used in v0.3.x
├── graph_tools.py           # Used
├── knowledge_tools.py       # Skeleton
├── modeling_tools.py        # NOT used in CLI
├── observation_tools.py     # Used in tests only
├── optimization_tools.py    # Used
├── registration_tools.py    # Used
├── run_tools.py             # Legacy (session-based)
└── smart_nlp_creation.py    # NOT used in CLI
```

---

## Refactoring Plan

### Phase 1: Remove Clearly Unused Code

**Files to delete:**
- `paola/utils/__init__.py` (empty)
- `paola/formulation/` folder (unused)
- `paola/tools/run_tools.py` (legacy session-based, replaced by graph_tools)
- `paola/tools/modeling_tools.py` (duplicates optimization_tools)

**Update __init__.py exports accordingly**

### Phase 2: Archive Experimental Code

Move to `archive/` folder (keep for reference but remove from package):
- `paola/tools/agentic_registration.py` → `archive/tools/`
- `paola/tools/smart_nlp_creation.py` → `archive/tools/`
- `paola/modeling/` folder → `archive/modeling/`

### Phase 3: Consolidate Gate Infrastructure

**Decision needed**: Keep or remove gate-based iteration control?

**Option A: Remove (Recommended for now)**
- Delete `paola/optimizers/gate.py`
- Delete `paola/tools/gate_control_tools.py`
- Update `paola/optimizers/__init__.py`
- Move to archive for future reference

**Rationale**: v0.3.x uses run-to-completion backends. Gate was for per-iteration control which is not currently used. Can be re-added in v0.5.x if needed.

**Option B: Keep but don't export**
- Remove from `__init__.py` exports
- Keep files for future use

### Phase 4: Consolidate Foundry Module

Reorganize foundry into clearer submodules:

```
foundry/
├── __init__.py          # Clean exports
├── foundry.py           # Main class
├── active_graph.py      # Graph management
├── evaluator/           # NEW subfolder
│   ├── __init__.py
│   ├── core.py          # Merge evaluator.py + nlp_evaluator.py
│   ├── schema.py        # Merge evaluator_schema.py + nlp_schema.py
│   ├── compiler.py      # evaluator_compiler.py
│   └── storage.py       # evaluator_storage.py
├── problem/             # NEW subfolder
│   ├── __init__.py
│   ├── schema.py        # nlp_schema.py (problem part)
│   ├── types.py         # problem_types.py
│   └── bounds.py        # bounds_spec.py
├── graph/               # RENAME from schema/
│   ├── __init__.py
│   ├── base.py          # graph.py
│   ├── record.py        # graph_record.py
│   ├── detail.py        # graph_detail.py
│   ├── components.py    # optimizer family components
│   └── conversion.py
└── storage/             # Keep as-is
```

### Phase 5: Simplify Tools Module

**Consolidate tools into logical groups:**

```
tools/
├── __init__.py          # Clean exports
├── core/                # Core optimization workflow
│   ├── __init__.py
│   ├── graph.py         # graph_tools.py
│   ├── optimization.py  # optimization_tools.py
│   └── evaluator.py     # evaluator_tools.py
├── analysis/            # Analysis tools
│   ├── __init__.py
│   ├── metrics.py       # analysis.py
│   └── observation.py   # observation_tools.py
├── persistence/         # Data tools
│   ├── __init__.py
│   ├── cache.py         # cache_tools.py + cache_agent_tools.py
│   ├── registration.py  # registration_tools.py
│   └── knowledge.py     # knowledge_tools.py
```

### Phase 6: Clean Up Knowledge Module

**Option A: Keep skeleton**
- Mark clearly as "skeleton - Phase 3"
- Remove from main exports
- Keep for future RAG integration

**Option B: Delete**
- Remove entirely
- Re-implement when ready

**Recommendation**: Option A (keep skeleton)

---

## Implementation Order

### Step 1: Delete Unused Files (Low Risk)
1. Delete `utils/__init__.py`
2. Delete `formulation/` folder
3. Delete `tools/run_tools.py`
4. Delete `tools/modeling_tools.py`
5. Update all `__init__.py` files
6. Run tests

### Step 2: Archive Experimental (Low Risk)
1. Create `archive/` folder
2. Move `tools/agentic_registration.py`
3. Move `tools/smart_nlp_creation.py`
4. Move `modeling/` folder
5. Update imports in debug_agent/
6. Run tests

### Step 3: Gate Decision (Medium Risk)
1. Decide keep/remove
2. If remove: delete gate.py and gate_control_tools.py
3. Update imports
4. Update tests (may break some)
5. Run tests

### Step 4: Foundry Reorganization (Higher Risk)
1. Create new subfolders
2. Move files with preserved imports
3. Update all imports
4. Run tests
5. Fix broken imports

### Step 5: Tools Reorganization (Higher Risk)
1. Create new subfolders
2. Move files with preserved imports
3. Update all imports
4. Run tests
5. Fix broken imports

---

## Test Impact

### Tests likely to need updates:
- `tests/test_run_architecture.py` - uses run_tools
- `tests/test_phase1_refactoring.py` - uses run_tools
- `tests/test_phase2_analysis.py` - uses run_tools
- `tests/test_phase4_cli.py` - uses run_tools
- `tests/test_conversational_agent.py` - uses run_tools
- `tests/test_end_to_end_workflow.py` - uses run_tools
- `tests/test_nlp_constraints.py` - uses run_tools
- `tests/test_registration_to_optimization.py` - uses run_tools
- `tests/test_gate_control_tools.py` - uses gate
- `tests/test_integration.py` - uses gate
- `tests/test_agentic_architecture.py` - uses agentic_registration
- `tests/test_observation_tools.py` - uses observation_tools

### Tests should still work:
- `tests/test_active_graph.py`
- `tests/test_graph_schema.py`
- `tests/test_cache_tools.py`
- `tests/test_callbacks.py`
- `tests/test_token_tracking.py`
- `tests/test_compiler_basic.py`
- `tests/test_evaluator_registration.py`

---

## Documentation Impact

### Files to update:
- `CLAUDE.md` - Update structure section
- `README.md` - Update examples if needed
- `docs/progress/20251217_current_state_and_roadmap.md` - Mark as superseded

---

## Expected Outcomes

### Before:
- 87 Python files
- 21 directories
- ~18,500 lines
- Complex imports

### After (estimated):
- ~60 Python files (-30%)
- ~15 directories (-30%)
- ~15,000 lines (-20%)
- Clearer module boundaries

---

## Decision Points

Before proceeding, confirm:

1. **Gate infrastructure**: Remove or keep?
   - Recommendation: Remove (archive for later)

2. **Knowledge skeleton**: Delete or keep?
   - Recommendation: Keep but mark as skeleton

3. **Test strategy**: Fix failing tests or delete obsolete tests?
   - Recommendation: Delete tests for removed code

4. **Import compatibility**: Maintain backward-compatible imports?
   - Recommendation: No - clean break at v0.4.0

---

## Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| 1: Delete unused | Low | Easy to revert from git |
| 2: Archive experimental | Low | Files preserved |
| 3: Gate decision | Medium | Tests may fail |
| 4: Foundry reorg | High | Many import changes |
| 5: Tools reorg | High | Many import changes |

---

## Next Steps

1. Review this plan
2. Make decision points
3. Create git branch `refactor/v0.4.0`
4. Execute phases incrementally
5. Run tests after each phase
6. Update documentation
7. Merge to main
