# Journal Architecture Implementation Plan

**Date**: 2025-12-29
**Version**: v0.2.1
**Status**: Implementation Ready

---

## Executive Summary

This document describes the implementation of a **Journal-based cross-process coordination layer** for Paola's v0.2.0 code-execution architecture. The Journal solves the subprocess isolation problem where graph metadata (checkpoint summaries) is lost when `execute_python` subprocesses exit.

---

## 1. Problem Statement

### 1.1 Symptom

```
CLI: execute_python(code_with_paola_checkpoint)
Subprocess: Returns {"graph_id": 108, "best_f": 3.98, ...}
CLI: get_graph_state(108)
Result: "Graph 108 not found"
```

### 1.2 Root Cause Analysis

The `execute_python` tool uses `subprocess.run()` (see `paola/tools/file_tools.py:99-104`):

```python
result = subprocess.run(
    [sys.executable, temp_file],
    capture_output=True,
    text=True,
    timeout=timeout
)
```

This creates a **fresh Python process** with separate memory. The `_checkpoint_summaries` dict in `paola/api.py:87` is **in-memory only**:

```python
# Store checkpoint summaries for warm-start support
# Key: (graph_id, node_id), Value: summary dict with best_x
_checkpoint_summaries: Dict[tuple, Dict[str, Any]] = {}
```

When the subprocess exits, this dict is garbage-collected. The CLI process has no visibility.

### 1.3 Why v0.1.0 Didn't Have This Problem

In v0.1.0, `run_optimization` was a LangChain `@tool` function that ran **in the same process** as the CLI. State was shared in-memory. v0.2.0's code-execution model is more flexible but introduces subprocess isolation.

---

## 2. Architecture Analysis

### 2.1 Current Recording Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CURRENT ARCHITECTURE                              │
│                                                                          │
│  Layer 1: RecordingObjective                                            │
│     └─ __call__(x) → writes to evaluations.jsonl (FILE-BASED ✓)        │
│     └─ Uses fcntl.flock() for atomic writes                             │
│     └─ paola/recording/objective.py:249-262                             │
│                                                                          │
│  Layer 2: Script Storage                                                 │
│     └─ checkpoint() saves script to scripts/graph_XXXX/nX.py (FILE ✓)  │
│     └─ paola/api.py:197-199                                             │
│                                                                          │
│  Layer 3: Graph Metadata ← THIS IS BROKEN                               │
│     └─ _checkpoint_summaries dict (IN-MEMORY → LOST ON EXIT ✗)         │
│     └─ paola/api.py:87                                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: RecordingObjective already writes evaluations to JSONL with file locking. The problem is **graph-level metadata** (`best_x`, `best_f`, `status`, `graph_id`) exists only in-memory.

### 2.2 Research Findings

Cross-process coordination patterns from successful projects:

| Project | Approach | Trade-offs |
|---------|----------|------------|
| **Optuna** | JournalStorage with append-only log | Atomic, works on NFS, simple |
| **MLflow** | SQLite + REST API | SQLite has locking issues in distributed |
| **DVC** | Git-like with .dvc pointer files | Complex for real-time coordination |
| **Sacred** | File observer pattern | Similar to Journal |

**Key Insight from Optuna**: SQLite causes "database is locked" errors in distributed/subprocess scenarios. Optuna's `JournalStorage` uses an **append-only journal file** with atomic operations.

---

## 3. Solution: Graph Journal

### 3.1 Design Principles

1. **Append-only**: No in-place updates, only appends (crash-safe)
2. **Atomic writes**: Use `fcntl.flock()` for file locking (like RecordingObjective)
3. **Reconstruct by replay**: Read journal entries, filter by `graph_id`, rebuild state
4. **Keep always**: No compaction or cleanup (full history preserved)

### 3.2 Storage Layout

```
.paola_foundry/
├── journal.jsonl                 # NEW: Graph lifecycle events
│   {"type":"graph_created","graph_id":108,"problem_id":7,"ts":...}
│   {"type":"checkpoint","graph_id":108,"node_id":"n1","best_f":3.98,...}
│   {"type":"finalized","graph_id":108,"success":true,...}
│
├── cache/graph_0108/
│   └── evaluations.jsonl         # EXISTING: Fine-grained eval history
│
├── scripts/graph_0108/
│   └── n1.py                     # EXISTING: Script for reproducibility
│
├── graphs/                       # Finalized GraphRecords (Tier 1)
├── details/                      # GraphDetail (Tier 2)
└── metadata.json
```

### 3.3 Journal Entry Types

| Type | When | Fields |
|------|------|--------|
| `graph_created` | `paola.objective()` | `graph_id`, `problem_id`, `goal` |
| `checkpoint` | `paola.checkpoint()` | `graph_id`, `node_id`, `best_f`, `best_x`, `n_evals` |
| `continue` | `paola.continue_graph()` | `graph_id`, `parent_node`, `edge_type`, `new_node_id` |
| `finalized` | `paola.finalize_graph()` | `graph_id`, `success`, `notes` |

---

## 4. Implementation Plan

### 4.1 File: `paola/foundry/journal.py` (NEW)

**Purpose**: Append-only journal for cross-process graph coordination.

```python
"""Append-only journal for cross-process graph coordination."""
import json
import fcntl
import time
from pathlib import Path
from typing import Optional, Dict, Any, List


class GraphJournal:
    """
    Append-only journal for graph metadata coordination.

    Design:
    - One JSON entry per line (JSONL format)
    - Atomic appends with fcntl file locking
    - Reconstruct state by replaying entries for a graph_id
    - No compaction (full history preserved)
    """

    def __init__(self, journal_path: Path):
        """
        Initialize journal.

        Args:
            journal_path: Path to journal.jsonl file
        """
        self.path = Path(journal_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: Dict[str, Any]) -> None:
        """
        Append entry atomically with file locking.

        Args:
            entry: Journal entry dict (must include "type" field)
        """
        entry["ts"] = time.time()

        with open(self.path, 'a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps(entry) + '\n')
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def get_graph_state(self, graph_id: int) -> Optional[Dict[str, Any]]:
        """
        Reconstruct graph state by replaying journal entries.

        Args:
            graph_id: Graph ID to query

        Returns:
            State dict or None if graph not found
        """
        if not self.path.exists():
            return None

        state = None
        nodes = {}  # node_id -> node_info

        with open(self.path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                if entry.get("graph_id") != graph_id:
                    continue

                entry_type = entry.get("type")

                if entry_type == "graph_created":
                    state = {
                        "graph_id": graph_id,
                        "problem_id": entry.get("problem_id"),
                        "goal": entry.get("goal"),
                        "status": "active",
                        "nodes": {},
                        "best_f": None,
                        "best_x": None,
                        "best_node_id": None,
                    }

                elif entry_type == "checkpoint" and state:
                    node_id = entry.get("node_id")
                    node_info = {
                        "node_id": node_id,
                        "best_f": entry.get("best_f"),
                        "best_x": entry.get("best_x"),
                        "n_evaluations": entry.get("n_evals"),
                        "optimizer": entry.get("optimizer"),
                    }
                    state["nodes"][node_id] = node_info
                    state["status"] = "checkpointed"

                    # Update graph-level best
                    if entry.get("best_f") is not None:
                        if state["best_f"] is None or entry["best_f"] < state["best_f"]:
                            state["best_f"] = entry["best_f"]
                            state["best_x"] = entry.get("best_x")
                            state["best_node_id"] = node_id

                elif entry_type == "continue" and state:
                    # Track node creation from continue_graph
                    new_node_id = entry.get("new_node_id")
                    if new_node_id:
                        state["nodes"][new_node_id] = {
                            "node_id": new_node_id,
                            "parent_node": entry.get("parent_node"),
                            "edge_type": entry.get("edge_type"),
                            "status": "active",
                        }

                elif entry_type == "finalized" and state:
                    state["status"] = "finalized"
                    state["finalized_at"] = entry.get("ts")

        return state

    def get_node_best_x(self, graph_id: int, node_id: str) -> Optional[List[float]]:
        """
        Get best_x for a specific node (for warm-start).

        Args:
            graph_id: Graph ID
            node_id: Node ID (e.g., "n1")

        Returns:
            best_x as list or None if not found
        """
        state = self.get_graph_state(graph_id)
        if state is None:
            return None

        node = state.get("nodes", {}).get(node_id)
        if node is None:
            return None

        return node.get("best_x")

    def list_active_graphs(self) -> List[int]:
        """
        List graphs that are active or checkpointed (not finalized).

        Returns:
            List of graph IDs
        """
        if not self.path.exists():
            return []

        graph_states = {}  # graph_id -> status

        with open(self.path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                gid = entry.get("graph_id")
                etype = entry.get("type")

                if etype == "graph_created":
                    graph_states[gid] = "active"
                elif etype == "checkpoint":
                    graph_states[gid] = "checkpointed"
                elif etype == "finalized":
                    graph_states[gid] = "finalized"

        return [gid for gid, status in graph_states.items()
                if status in ("active", "checkpointed")]

    def count_nodes(self, graph_id: int) -> int:
        """
        Count nodes in a graph (for generating next node ID).

        Args:
            graph_id: Graph ID

        Returns:
            Number of nodes (checkpointed + active)
        """
        state = self.get_graph_state(graph_id)
        if state is None:
            return 0
        return len(state.get("nodes", {}))
```

### 4.2 File: `paola/api.py` (MODIFY)

**Changes**:
1. Add journal accessor `_get_journal()`
2. Modify `objective()` to log `graph_created`
3. Modify `checkpoint()` to log checkpoint entry
4. Modify `continue_graph()` to log continue entry and read parent's best_x from journal
5. Modify `finalize_graph()` to log finalization

#### 4.2.1 Add Journal Accessor (after line 55)

```python
# Module-level journal instance (lazy initialization)
_journal: Optional["GraphJournal"] = None


def _get_journal() -> "GraphJournal":
    """Get or create the module-level Journal instance."""
    global _journal
    if _journal is None:
        from paola.foundry.journal import GraphJournal
        journal_path = Path(_foundry_base_dir) / "journal.jsonl"
        _journal = GraphJournal(journal_path)
    return _journal
```

#### 4.2.2 Modify `objective()` (after line 149, before return)

```python
    # Log graph creation to journal
    journal = _get_journal()
    journal.append({
        "type": "graph_created",
        "graph_id": graph_id,
        "problem_id": problem_id,
        "goal": goal,
    })

    return recording_obj
```

#### 4.2.3 Modify `checkpoint()` (after line 238, before return)

```python
    # Log to journal for cross-process visibility
    journal = _get_journal()
    journal.append({
        "type": "checkpoint",
        "graph_id": graph_id,
        "node_id": node_id,
        "best_f": summary.get("best_f"),
        "best_x": summary.get("best_x"),
        "n_evals": summary.get("n_evaluations"),
        "optimizer": optimizer,
    })

    return summary
```

#### 4.2.4 Modify `continue_graph()` (lines 274-343)

Add journal lookup for parent's best_x as fallback:

```python
def continue_graph(
    graph_id: int,
    parent_node: str,
    edge_type: str = "warm_start",
) -> RecordingObjective:
    foundry = _get_foundry()
    journal = _get_journal()

    # First, try to get parent's best_x from checkpoint summaries (same-session)
    parent_best_x = None
    checkpoint_key = (graph_id, parent_node)
    if checkpoint_key in _checkpoint_summaries:
        parent_summary = _checkpoint_summaries[checkpoint_key]
        if parent_summary.get("best_x"):
            parent_best_x = np.array(parent_summary["best_x"])

    # Second, try journal (cross-process fallback)
    if parent_best_x is None:
        journal_best_x = journal.get_node_best_x(graph_id, parent_node)
        if journal_best_x is not None:
            parent_best_x = np.array(journal_best_x)

    # ... rest of the function ...

    # Determine next node ID using journal count
    journal_node_count = journal.count_nodes(graph_id)
    checkpoint_nodes = {k[1] for k in _checkpoint_summaries.keys() if k[0] == graph_id}
    existing_count = max(journal_node_count, len(checkpoint_nodes))
    node_id = f"n{existing_count + 1}"

    # ... create RecordingObjective ...

    # Log continue to journal
    journal.append({
        "type": "continue",
        "graph_id": graph_id,
        "parent_node": parent_node,
        "edge_type": edge_type,
        "new_node_id": node_id,
    })

    return recording_obj
```

#### 4.2.5 Modify `finalize_graph()` (after line 402, before return)

```python
    # Log finalization to journal
    journal = _get_journal()
    journal.append({
        "type": "finalized",
        "graph_id": graph_id,
        "notes": notes,
    })

    return result
```

### 4.3 File: `paola/tools/graph.py` (MODIFY)

**Change**: Add journal lookup in `get_graph_state()` as fallback after active graphs and before returning "not found".

#### Modify `get_graph_state()` (insert after line 113, before line 162)

```python
        # Check journal for checkpointed graphs (cross-process)
        from paola.foundry.journal import GraphJournal
        journal_path = _FOUNDRY.storage.base_path / "journal.jsonl"
        journal = GraphJournal(journal_path)
        journal_state = journal.get_graph_state(graph_id)

        if journal_state is not None and journal_state.get("status") != "finalized":
            # Build node summaries from journal state
            node_summaries = []
            for nid, node_info in journal_state.get("nodes", {}).items():
                node_summaries.append({
                    "node_id": nid,
                    "optimizer": node_info.get("optimizer", "unknown"),
                    "status": node_info.get("status", "checkpointed"),
                    "best_objective": node_info.get("best_f"),
                    "n_evaluations": node_info.get("n_evaluations"),
                })

            best_summary = None
            if journal_state.get("best_node_id"):
                best_node = journal_state["nodes"].get(journal_state["best_node_id"])
                if best_node:
                    best_summary = {
                        "node_id": journal_state["best_node_id"],
                        "optimizer": best_node.get("optimizer", "unknown"),
                        "best_objective": journal_state["best_f"],
                        "best_x": journal_state["best_x"],
                    }

            return {
                "success": True,
                "status": journal_state["status"],
                "graph_id": graph_id,
                "problem_id": journal_state.get("problem_id"),
                "goal": journal_state.get("goal"),
                "n_nodes": len(journal_state.get("nodes", {})),
                "nodes": node_summaries,
                "best_node": best_summary,
                "leaf_nodes": node_summaries,  # All nodes are potential leaves
                "message": f"Graph #{graph_id} is {journal_state['status']} (from journal).",
            }
```

### 4.4 File: `paola/foundry/__init__.py` (MODIFY)

Add export for GraphJournal:

```python
from .journal import GraphJournal
```

---

## 5. Data Flow After Implementation

```
Subprocess (execute_python)                  journal.jsonl
───────────────────────────                  ─────────────
paola.objective(7)
  → journal.append({"type":"graph_created",...})  ──→  Line 1

minimize(f, x0, ...)
  → evaluations.jsonl                        (unchanged)

paola.checkpoint(f, script, ...)
  → journal.append({"type":"checkpoint",...})  ──→  Line 2
  → scripts/graph_XXXX/nX.py                 (unchanged)
  → print(json.dumps(summary))

[Subprocess exits - _checkpoint_summaries lost, but journal persists]

                                             CLI Process
                                             ───────────
                                             get_graph_state(108)
                                               ← journal.get_graph_state(108)
                                               ← Replay entries, filter by graph_id
                                               → Returns checkpointed state ✓
```

---

## 6. What to Keep vs Replace

| Component | Action | Reason |
|-----------|--------|--------|
| `RecordingObjective` | **KEEP** | Already file-based (JSONL), crash-safe |
| `evaluations.jsonl` | **KEEP** | Fine-grained eval history, works |
| `scripts/` storage | **KEEP** | Script reproducibility, works |
| `_checkpoint_summaries` dict | **KEEP + SUPPLEMENT** | Keep for same-process speed, add journal fallback |
| `_active_objectives` dict | **KEEP** | Same-process tracking still useful |
| **Journal** | **ADD** | Cross-process coordination for graph metadata |
| Instrumentation | **KEEP (future)** | Orthogonal - for optimizer verification |
| OpenTelemetry | **DEFER** | Orthogonal - for observability |

---

## 7. Implementation Order

| Step | File | Change | LOC |
|------|------|--------|-----|
| 1 | `paola/foundry/journal.py` | Create GraphJournal class | ~150 |
| 2 | `paola/foundry/__init__.py` | Export GraphJournal | 1 |
| 3 | `paola/api.py` | Add `_get_journal()` accessor | ~10 |
| 4 | `paola/api.py` | Modify `objective()` to log creation | ~5 |
| 5 | `paola/api.py` | Modify `checkpoint()` to log checkpoint | ~10 |
| 6 | `paola/api.py` | Modify `continue_graph()` with journal fallback | ~20 |
| 7 | `paola/api.py` | Modify `finalize_graph()` to log finalization | ~5 |
| 8 | `paola/tools/graph.py` | Add journal lookup in `get_graph_state()` | ~35 |
| 9 | Test | Manual CLI test with execute_python | - |

**Total estimated changes**: ~240 lines

---

## 8. Testing Plan

### 8.1 Unit Tests

```python
# tests/test_journal.py

def test_journal_append_and_read():
    """Test basic append and read."""
    journal = GraphJournal(tmp_path / "journal.jsonl")
    journal.append({"type": "graph_created", "graph_id": 1, "problem_id": 7})
    journal.append({"type": "checkpoint", "graph_id": 1, "node_id": "n1", "best_f": 3.98})

    state = journal.get_graph_state(1)
    assert state["graph_id"] == 1
    assert state["status"] == "checkpointed"
    assert state["nodes"]["n1"]["best_f"] == 3.98


def test_journal_multiple_graphs():
    """Test isolation between graphs."""
    journal = GraphJournal(tmp_path / "journal.jsonl")
    journal.append({"type": "graph_created", "graph_id": 1, "problem_id": 7})
    journal.append({"type": "graph_created", "graph_id": 2, "problem_id": 8})

    state1 = journal.get_graph_state(1)
    state2 = journal.get_graph_state(2)

    assert state1["problem_id"] == 7
    assert state2["problem_id"] == 8


def test_journal_finalization():
    """Test finalization marks graph as done."""
    journal = GraphJournal(tmp_path / "journal.jsonl")
    journal.append({"type": "graph_created", "graph_id": 1, "problem_id": 7})
    journal.append({"type": "finalized", "graph_id": 1})

    state = journal.get_graph_state(1)
    assert state["status"] == "finalized"

    active = journal.list_active_graphs()
    assert 1 not in active
```

### 8.2 Integration Test (Manual)

```python
# In CLI:
# 1. Create problem
create_nlp_problem(...)  # problem_id=32

# 2. Execute optimization code
execute_python('''
import paola
from scipy.optimize import minimize
import json

f = paola.objective(problem_id=32, goal="Test journal")
result = minimize(f, [0.0, 0.0], method="SLSQP", bounds=[(-5,5), (-5,5)])
summary = paola.checkpoint(f, script="...", reasoning="Test")
print(json.dumps(summary))
''')

# 3. Query graph state (should work now!)
get_graph_state(graph_id=108)  # Should return checkpointed state
```

---

## 9. Success Criteria

After implementation:

1. **Subprocess creates graph** → Journal records `graph_created`
2. **Subprocess checkpoints** → Journal records `checkpoint` with best_x, best_f
3. **Subprocess exits** → In-memory `_checkpoint_summaries` lost (expected)
4. **CLI calls `get_graph_state()`** → Reads journal, returns checkpointed state
5. **CLI calls `continue_graph()`** → Reads journal for parent's best_x
6. **CLI calls `finalize_graph()`** → Journal records finalization

---

## 10. Future Considerations

### 10.1 Bash Tool for Agent

Following Claude Code's pattern, Paola should have a Bash tool for flexible scripting:

```python
@tool
def bash(command: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute bash command.

    Agent can use this for:
    - Testing scripts: `python scripts/my_slsqp.py --test`
    - Inspecting files: `ls -la scripts/`
    - Git operations: `git status`
    - Package management: `pip list | grep scipy`
    """
    result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
    return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
```

This separates:
- **Code development** (write file, bash test) - agent's discretion
- **Optimization execution** (execute_python with paola) - recorded to journal

### 10.2 Journal Compaction (NOT Recommended)

The design explicitly chooses to **keep full history**:
- Journal files are small (one line per operation)
- Full history enables debugging and audit
- Replay-based state reconstruction is fast enough

If journal grows very large (100K+ entries), consider:
- Index file for faster lookup
- Per-graph journal files (journal/graph_0001.jsonl)

---

## 11. References

- Optuna JournalStorage: https://optuna.readthedocs.io/en/stable/reference/storages.html
- RecordingObjective implementation: `paola/recording/objective.py`
- Current API: `paola/api.py`
- Graph tools: `paola/tools/graph.py`
