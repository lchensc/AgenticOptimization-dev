# Journal Analysis and Execution Model Redesign (v0.2.1)

**Date**: 2025-12-29
**Status**: Implemented and Verified

## Problem Statement

The v0.2.0 Recording API introduced subprocess isolation for code execution (via `execute_python`), but this created a critical issue: the main CLI process could not finalize graphs created in subprocesses because the `ActiveGraph` object was lost when the subprocess exited.

### Observed Behavior

```
Subprocess (execute_python):
  paola.objective(33) → creates graph #115 in subprocess._active_graphs
  paola.checkpoint() → logs to journal ✓
  [subprocess exits] → graph object garbage-collected ✗

Main Process (CLI):
  get_graph_state(115) → reads journal ✓ ("checkpointed")
  finalize_graph(115) → looks for foundry._active_graphs[115] → NOT FOUND ✗
```

**Error**: `"Graph 115 not found or already finalized"`

## Analysis

### What Works (Journal Phase 1)

| Feature | Status | Evidence |
|---------|--------|----------|
| Journal writes from subprocess | ✓ | Graph 115/116/117 logged `graph_created`, `checkpoint` |
| Cross-process visibility | ✓ | `get_graph_state(115)` returns `"checkpointed"` |
| Warm-start data preserved | ✓ | `best_x`, `best_f` correctly logged and retrievable |
| Node counting for continue | ✓ | `journal.count_nodes()` works |

### What Was Broken

| Feature | Status | Evidence |
|---------|--------|----------|
| `finalize_graph()` | ✗ | `"Graph not found or already finalized"` |
| `get_past_graph()` | ✗ | No finalized record exists |

### Root Cause

The journal provided **read-only visibility** but could not complete the graph lifecycle because:
1. `finalize_graph()` required an `ActiveGraph` object in memory
2. The `ActiveGraph` only existed in the subprocess's memory
3. When subprocess exited, the object was garbage-collected
4. Main process had no way to reconstruct the graph from journal data

## Solution: Journal-Based Finalization (v0.2.1)

### Design

Enable `finalize_graph(graph_id)` to work without an in-memory `ActiveGraph` by reconstructing a `GraphRecord` directly from journal entries.

### Implementation Summary

| File | Change | LOC |
|------|--------|-----|
| `paola/api.py` | Enhanced checkpoint journal entry with full node data | ~10 |
| `paola/recording/objective.py` | Added `parent_node`, `edge_type` tracking | ~10 |
| `paola/foundry/journal.py` | Added `get_full_graph_data()` method | ~60 |
| `paola/foundry/foundry.py` | Added `finalize_graph_from_journal()` method | ~100 |
| `paola/foundry/storage/file_storage.py` | Added `save_graph_from_record()` method | ~20 |
| `paola/tools/graph.py` | Added journal fallback in `finalize_graph` tool | ~15 |

### Key Changes

#### 1. Enhanced Checkpoint Journal Entry

```python
# paola/api.py - checkpoint() function
journal.append({
    "type": "checkpoint",
    "graph_id": graph_id,
    "node_id": node_id,
    "problem_id": f.problem_id,
    "best_f": summary.get("best_f"),
    "best_x": summary.get("best_x"),
    "n_evals": summary.get("n_evaluations"),
    "n_actual_evals": summary.get("n_actual_evals"),
    "optimizer": optimizer or "unknown",
    "wall_time": summary.get("wall_time", 0.0),
    "parent_node": getattr(f, '_parent_node', None),
    "edge_type": getattr(f, '_edge_type', None),
})
```

#### 2. Journal Reconstruction Method

```python
# paola/foundry/journal.py
def get_full_graph_data(self, graph_id: int) -> Optional[Dict[str, Any]]:
    """Get complete graph data for reconstruction into GraphRecord."""
    # Returns:
    # - graph_id, problem_id, goal, created_at
    # - nodes: Dict[node_id -> full node data with edges]
    # - best_f, best_x, total_evaluations, total_wall_time
```

#### 3. Journal-Based Finalization

```python
# paola/foundry/foundry.py
def finalize_graph_from_journal(self, graph_id: int) -> Optional[GraphRecord]:
    """Finalize graph using journal data when not in _active_graphs."""
    # 1. Get full graph data from journal
    # 2. Build NodeSummary objects from journal data
    # 3. Infer optimizer_family from optimizer string
    # 4. Detect pattern from nodes and edges
    # 5. Create GraphRecord with problem signature
    # 6. Save to storage (both tiers)
    # 7. Mark as finalized in journal
```

#### 4. Tool Fallback

```python
# paola/tools/graph.py - finalize_graph tool
graph = _FOUNDRY.get_graph(graph_id)

if graph is not None:
    # Graph is in active memory - use normal finalization
    record = _FOUNDRY.finalize_graph(graph_id)
else:
    # Journal fallback (v0.2.1)
    record = _FOUNDRY.finalize_graph_from_journal(graph_id)
```

## Part 2: Bash Tool Addition

### Motivation

Replace `execute_python` with a more general-purpose Bash tool:
1. **Empowers agent flexibility**: Run scripts, tests, git operations, package management
2. **Separation of concerns**: Bash for general scripting, paola API for optimization
3. **Claude Code pattern**: Proven design from production systems

### Implementation

Created `paola/tools/bash_tools.py`:

```python
@tool
def bash(
    command: str,
    timeout: int = 120,
    description: str = "",
) -> Dict[str, Any]:
    """Execute bash command."""
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=os.getcwd(),
    )
    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }
```

### Workflow Changes

**Before (v0.2.0)**:
```python
execute_python(code="import paola; f = paola.objective(...)")
```

**After (v0.2.1)**:
```python
write_file("opt_script.py", content=SCRIPT)
bash("python opt_script.py")
```

## Verification

Tested with comprehensive subprocess test:

```
============================================================
Test: Journal-based Finalization
============================================================

--- Running subprocess ---
Graph ID: 118
Node ID: n1
Running SLSQP from x0=[0.5 0.5]...
Result: x=[-1.11e-16 -1.11e-16], f=2.47e-32
Checkpoint complete: best_f=2.47e-32

--- Main process: analyzing graph 118 ---
Active graph in memory: None

Calling finalize_graph_from_journal...

✓ Successfully finalized graph 118
  Pattern: single
  Best f: 2.47e-32
  Total evals: 7
  Nodes: ['n1']
  ✓ Graph record saved to graph_0118.json
  ✓ Graph record can be loaded back

============================================================
✓ All tests passed!
============================================================
```

## Summary

### What Was Fixed
- `finalize_graph()` now works for subprocess-created graphs via journal fallback
- Graph lifecycle is complete: create → optimize → checkpoint → finalize → query

### What Was Added
- `bash` tool for general scripting (replaces `execute_python`)
- `get_full_graph_data()` for journal reconstruction
- `finalize_graph_from_journal()` for journal-based finalization
- `save_graph_from_record()` for direct GraphRecord persistence

### What Was Removed
- `execute_python` tool (replaced by `bash` + `write_file`)

### Architecture Benefits
1. **Subprocess isolation preserved**: Code runs in separate process
2. **Cross-process coordination works**: Journal enables visibility and finalization
3. **Agent empowered**: Bash tool provides general scripting capabilities
4. **Multi-turn optimization enabled**: Agent can `continue_graph()` from CLI after subprocess completes
