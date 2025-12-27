# Paola Tools Cleanup Analysis

**Date**: December 27, 2025
**Issue**: Tools folder contains mix of new modular architecture and old monolithic files

---

## Current State Analysis

### Files Created in Refactoring (NEW)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `problem.py` | 465 | Problem formulation (create_nlp_problem, derive_problem, list_problems, get_problem_lineage) | ✅ **CLEAN** |
| `evaluator.py` | 177 | Evaluator registration (foundry_store_evaluator, foundry_list_evaluators, foundry_get_evaluator) | ✅ **CLEAN** |
| `file_tools.py` | 129 | File operations (read_file, write_file, execute_python) | ✅ **CLEAN** |

### Files Modified in Refactoring (REFACTORED)

| File | Before | After | Change | Status |
|------|--------|-------|--------|--------|
| `evaluator_tools.py` | 948 | 398 | -550 (problem formulation moved out) | ✅ **CLEAN** |
| `registration_tools.py` | 331 | 26 | -305 (now just re-exports) | ✅ **CLEAN** |

### Files Untouched (OLD MONOLITHIC STRUCTURE)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `optimization_tools.py` | 518 | run_optimization, get_problem_info, list_optimizers | ⚠️ **OLD** |
| `graph_tools.py` | 519 | start_graph, get_graph_state, finalize_graph, query_past_graphs | ⚠️ **OLD** |
| `observation_tools.py` | 733 | analyze_convergence, detect_pattern, check_feasibility | ⚠️ **OLD** |
| `cache_tools.py` | 329 | cache_get, cache_store, run_db_log | ⚠️ **OLD** |
| `cache_agent_tools.py` | 89 | cache_clear, cache_stats, run_db_query | ⚠️ **OLD** |
| `analysis.py` | 431 | analyze_convergence_new, analyze_efficiency (deprecated?) | ⚠️ **OLD** |
| `schemas.py` | 172 | Pydantic validation schemas | ✅ **SUPPORT** |

---

## Problem: Inconsistent Architecture

The refactoring was **partial**:
- ✅ Problem formulation tools: **Modularized** (problem.py, evaluator.py, file_tools.py)
- ⚠️ Optimization/graph/observation tools: **Still monolithic**

This creates confusion:
1. **Why is `problem.py` separate but `optimization_tools.py` isn't?**
2. **What's the actual architectural principle?** (some modular, some not)
3. **Is the refactoring complete or abandoned halfway?**

---

## Proposed Cleanup Strategy

### Option 1: Complete the Modularization (Most Consistent)

Break remaining monolithic files into logical modules:

**From `optimization_tools.py` (518 lines)**:
- Keep as-is OR split into:
  - `optimizer.py` - run_optimization, list_optimizers
  - `problem_info.py` - get_problem_info (seems like it belongs with problem.py?)

**From `graph_tools.py` (519 lines)**:
- Keep as-is (already logical - all about graphs)
- BUT: `set_foundry`, `get_foundry` are utilities, not graph tools
  - Move to `foundry_utils.py`?

**From `observation_tools.py` (733 lines)**:
- Keep as-is OR split into:
  - `convergence.py` - analyze_convergence, detect_pattern
  - `feasibility.py` - check_feasibility
  - `gradient.py` - get_gradient_quality

**From `cache_tools.py` + `cache_agent_tools.py`**:
- Merge into single `cache.py` (358 lines total)
- Or keep separate (one for internal, one for LLM?)

**From `analysis.py`**:
- ⚠️ **DEPRECATED?** Contains old analysis functions that overlap with observation_tools.py
- Candidate for deletion if unused

### Option 2: Revert to Monolithic (Simplest)

**Undo the refactoring**:
- Move `problem.py` functions back to `evaluator_tools.py`
- Move `evaluator.py` functions back to `registration_tools.py`
- Delete `file_tools.py`, move functions back to `registration_tools.py`

**Rationale**: Consistency - all tools in original structure
**Con**: Loses the benefits of modularization

### Option 3: Keep Hybrid (Current State)

**Do nothing**, document that:
- Core formulation tools are modular (problem, evaluator, file)
- Execution tools are monolithic (optimization, graph, observation, cache)

**Rationale**: Don't change what's not broken
**Con**: Inconsistent architecture, harder to understand

---

## Recommended Approach: **Option 1 - Complete the Modularization**

### Reasoning

1. **Consistency**: All tools follow same modular pattern
2. **Maintainability**: Smaller files easier to understand and modify
3. **LLM tool selection**: Clearer separation improves tool choice accuracy
4. **Momentum**: Already started refactoring, finish it properly

### Detailed Plan

#### Phase 1: Analyze Dependencies

Check what imports what to avoid circular dependencies:
```bash
grep -r "from paola.tools" paola/tools/*.py
```

#### Phase 2: Reorganize by Logical Grouping

**New structure**:
```
paola/tools/
├── problem.py              # ✅ EXISTS - Problem formulation
├── evaluator.py            # ✅ EXISTS - Evaluator registration
├── file_tools.py           # ✅ EXISTS - File operations
├── optimizer.py            # 🆕 NEW - Optimization execution (from optimization_tools.py)
├── graph.py                # 🆕 NEW - Graph management (from graph_tools.py, rename)
├── analysis.py             # 🔄 REFACTOR - Merge observation_tools.py + old analysis.py
├── cache.py                # 🔄 REFACTOR - Merge cache_tools.py + cache_agent_tools.py
├── foundry_utils.py        # 🆕 NEW - Foundry utilities (set_foundry, get_foundry)
├── schemas.py              # ✅ KEEP - Validation schemas
├── __init__.py             # 🔄 UPDATE - New imports
└── _deprecated/            # 📦 ARCHIVE - Old files for reference
    ├── optimization_tools.py
    ├── graph_tools.py
    ├── observation_tools.py
    ├── cache_tools.py
    └── cache_agent_tools.py
```

#### Phase 3: Create New Modules

**1. `optimizer.py`** (from `optimization_tools.py`):
```python
"""
Optimizer execution tools.

Tools for running optimizers:
- run_optimization: Execute optimization run
- list_optimizers: List available optimizers
"""

@tool
def run_optimization(...):
    """Run optimization, creating a new node in the graph."""
    ...

@tool
def list_optimizers():
    """List available optimization algorithms."""
    ...
```

**2. `graph.py`** (rename from `graph_tools.py`, remove foundry utils):
```python
"""
Optimization graph management.

Tools for graph lifecycle:
- start_graph: Create new optimization graph
- get_graph_state: Get current graph state
- finalize_graph: Close graph and persist
- query_past_graphs: Search historical graphs
- get_past_graph: Retrieve specific graph
"""
# Remove: set_foundry, get_foundry (move to foundry_utils.py)
```

**3. `analysis.py`** (merge observation_tools.py + old analysis.py):
```python
"""
Optimization analysis and observation.

Tools for analyzing optimization results:
- analyze_convergence: Analyze convergence behavior
- detect_pattern: Detect optimization patterns
- check_feasibility: Check constraint satisfaction
- get_gradient_quality: Assess gradient quality
- analyze_efficiency: Analyze computational efficiency
"""
# Merge both analysis files, remove duplicates
```

**4. `cache.py`** (merge cache_tools.py + cache_agent_tools.py):
```python
"""
Evaluation caching.

Tools for caching expensive evaluations:
- cache_get: Retrieve cached evaluation
- cache_store: Store evaluation result
- cache_clear: Clear evaluation cache
- cache_stats: Get cache statistics
- run_db_query: Query evaluation database
- run_db_log: Log database queries
"""
```

**5. `foundry_utils.py`** (extract from graph_tools.py):
```python
"""
Foundry utilities.

Internal utilities for Foundry management:
- set_foundry: Set global Foundry instance
- get_foundry: Get global Foundry instance
"""
# These are utilities, not LLM-facing tools
# Not exported in __all__, only for internal use
```

#### Phase 4: Update __init__.py

```python
"""
Agent tools for Paola.

Modular architecture:
- problem.py: Problem formulation
- evaluator.py: Evaluator registration
- file_tools.py: File operations
- optimizer.py: Optimization execution
- graph.py: Graph management
- analysis.py: Result analysis
- cache.py: Evaluation caching
- foundry_utils.py: Internal utilities
- schemas.py: Validation schemas
"""

from paola.tools.problem import (
    create_nlp_problem,
    derive_problem,
    list_problems,
    get_problem_lineage,
)

from paola.tools.evaluator import (
    foundry_store_evaluator,
    foundry_list_evaluators,
    foundry_get_evaluator,
)

from paola.tools.file_tools import (
    read_file,
    write_file,
    execute_python,
)

from paola.tools.optimizer import (
    run_optimization,
    list_optimizers,
)

from paola.tools.graph import (
    start_graph,
    get_graph_state,
    finalize_graph,
    query_past_graphs,
    get_past_graph,
)

from paola.tools.analysis import (
    analyze_convergence,
    detect_pattern,
    check_feasibility,
    get_gradient_quality,
    compute_improvement_statistics,
    analyze_efficiency,
)

from paola.tools.cache import (
    cache_get,
    cache_store,
    cache_clear,
    cache_stats,
    run_db_query,
    run_db_log,
)

# Internal utilities
from paola.tools.foundry_utils import (
    set_foundry,
    get_foundry,
)

# Validation schemas
from paola.tools.schemas import (
    normalize_problem_id,
    ProblemIdType,
)

# Backward compatibility
from paola.tools.evaluator_tools import (
    evaluate_function,
    compute_gradient,
    create_benchmark_problem,
)
```

#### Phase 5: Archive Old Files

Move to `_deprecated/` for reference:
- `optimization_tools.py` → `_deprecated/optimization_tools.py`
- `graph_tools.py` → `_deprecated/graph_tools.py`
- `observation_tools.py` → `_deprecated/observation_tools.py`
- `cache_tools.py` → `_deprecated/cache_tools.py`
- `cache_agent_tools.py` → `_deprecated/cache_agent_tools.py`

#### Phase 6: Update Imports

Search and replace all imports:
```bash
# Find all files importing old modules
grep -r "from paola.tools.optimization_tools" paola/
grep -r "from paola.tools.graph_tools" paola/
grep -r "from paola.tools.observation_tools" paola/
grep -r "from paola.tools.cache_tools" paola/

# Update to new modules
# optimization_tools → optimizer
# graph_tools → graph
# observation_tools → analysis
# cache_tools → cache
```

---

## Alternative: Minimal Cleanup (If Full Refactoring Too Risky)

### Just rename files to match documentation

1. `optimization_tools.py` → `optimizer.py` (no content change)
2. `graph_tools.py` → `graph.py` (no content change)
3. `observation_tools.py` → `analysis.py` (no content change)
4. Merge `cache_tools.py` + `cache_agent_tools.py` → `cache.py`
5. Update __init__.py imports
6. Update all references

**Benefit**: Architectural consistency without restructuring code
**Risk**: Lower (just renames + one merge)

---

## Questions for Decision

1. **How important is architectural consistency?**
   - High → Full modularization (Option 1)
   - Medium → File renaming only
   - Low → Keep as-is (Option 3)

2. **What's the timeline?**
   - Urgent → Minimal cleanup (renaming)
   - Normal → Full modularization

3. **What's being actively developed?**
   - If benchmark work is priority → Defer cleanup
   - If architecture cleanup is priority → Full refactoring

---

## Recommendation

**Start with minimal cleanup (file renaming)**:

1. Rename files to match intended architecture:
   - `optimization_tools.py` → `optimizer.py`
   - `graph_tools.py` → `graph.py`
   - `observation_tools.py` → `analysis.py`
   - Merge cache files → `cache.py`

2. Update imports in:
   - `__init__.py`
   - `repl.py`
   - Any other files importing these modules

3. Test that everything still works

4. **Then decide** if further splitting is needed

**Rationale**:
- Lower risk than full restructuring
- Achieves architectural consistency
- Can iterate to finer-grained modules later if needed
- Unblocks benchmark work without major disruption

---

## Next Steps

1. **Get user decision**: Which approach?
2. **If renaming**: Execute rename + update imports
3. **If full refactor**: Create new modules phase by phase
4. **Test**: Ensure all imports work
5. **Commit**: Document architectural change
6. **Update docs**: Fix architecture documentation

