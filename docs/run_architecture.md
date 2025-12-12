# Run-Based Architecture

## Overview

The AgenticOpt platform uses a clean, professional architecture where the **agent explicitly manages optimization runs** through dedicated tools. This ensures complete independence between components.

## Architecture Principles

### 1. **Agent Controls Tracking**
The agent explicitly creates and manages optimization runs using tools:
- `start_optimization_run` - Create a tracked run
- `run_scipy_optimization(..., run_id=X)` - Link optimization to run
- `finalize_optimization_run` - Add final notes

### 2. **Runs Auto-Persist**
`OptimizationRun` objects automatically save to storage as they progress:
- Every iteration recorded → auto-save
- Finalization → final save
- Storage-independent: works with FileStorage, SQLite, etc.

### 3. **Complete Independence**
```
Agent Tools (creates runs)
    ↓
OptimizationRun (auto-persists)
    ↓
Storage Layer (FileStorage)
    ↑
CLI/Notebooks (reads storage)
```

No coupling between layers - each can be used independently.

## Component Details

### Active Run (`aopt/runs/active_run.py`)

```python
class OptimizationRun:
    """Active optimization run managed by agent."""

    def __init__(self, run_id, problem_id, algorithm, storage):
        self.start_time = datetime.now()
        self.iterations = []
        self.result = None

    def record_iteration(self, design, objective):
        """Record iteration and auto-save."""
        self.iterations.append({...})
        self._persist()  # Automatic

    def finalize(self, result):
        """Finalize with scipy result."""
        self.result = result
        self._persist()
```

### Run Manager (`aopt/runs/manager.py`)

```python
class RunManager:
    """Singleton managing active runs."""

    def create_run(self, problem_id, algorithm):
        """Create new tracked run."""
        run_id = storage.get_next_run_id()
        run = OptimizationRun(...)
        self._active_runs[run_id] = run
        return run

    def get_run(self, run_id):
        """Get active run by ID."""
        return self._active_runs.get(run_id)
```

### Agent Tools (`aopt/tools/run_tools.py`)

```python
@tool
def start_optimization_run(
    problem_id: str,
    algorithm: str
) -> Dict[str, Any]:
    """
    Create tracked optimization run.
    Returns run_id to use in subsequent tools.
    """
    manager = RunManager()
    run = manager.create_run(problem_id, algorithm)
    return {"run_id": run.run_id, ...}

@tool
def run_scipy_optimization(
    problem_id: str,
    algorithm: str,
    bounds: List[List[float]],
    run_id: Optional[int] = None
):
    """
    Run optimization and optionally record to run.
    If run_id provided, results auto-saved.
    """
    # ... run optimization ...

    if run_id:
        run = RunManager().get_run(run_id)
        run.finalize(result)
```

## Usage Flow

### In Agent Conversation

```
User: "Optimize 10D Rosenbrock with SLSQP"

Agent thinks:
1. First create problem
2. Then create tracked run
3. Then run optimization with run_id
4. Results auto-saved!

Agent executes:

create_benchmark_problem(
    problem_id="rosenbrock_10d",
    function_name="rosenbrock",
    dimension=10
)

start_optimization_run(
    problem_id="rosenbrock_10d",
    algorithm="SLSQP"
)
→ Returns: {"run_id": 1}

run_scipy_optimization(
    problem_id="rosenbrock_10d",
    algorithm="SLSQP",
    bounds=[[-5, 10]] * 10,
    run_id=1  ← Links to run
)
→ Results auto-saved to run #1
```

### In CLI

```bash
$ python paola_cli.py
paola> optimize 10D Rosenbrock with SLSQP
💭 Creating problem and run...
🔧 create_benchmark_problem...
✓ create_benchmark_problem completed
🔧 start_optimization_run...
✓ start_optimization_run completed
🔧 run_scipy_optimization...
✓ run_scipy_optimization completed

paola> /runs
┌────┬──────────────┬───────────┬────────┬─────────────┬────────┬───────┐
│ ID │ Problem      │ Algorithm │ Status │ Best Value  │ Evals  │ Time  │
├────┼──────────────┼───────────┼────────┼─────────────┼────────┼───────┤
│  1 │ Rosenbrock   │ SLSQP     │ ✓      │  0.023456   │ 142    │ 2.3s  │
└────┴──────────────┴───────────┴────────┴─────────────┴────────┴───────┘

paola> /show 1
Run #1: SLSQP on Rosenbrock-10D
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status:       ✓ Complete
Objective:    0.023456
Evaluations:  142
...
```

### In Jupyter Notebook

```python
from aopt.storage import FileStorage

# Read same data as CLI
storage = FileStorage('.aopt_runs')
runs = storage.load_all_runs()

for run in runs:
    print(f"Run {run.run_id}: {run.algorithm} on {run.problem_name}")
    print(f"  Objective: {run.objective_value}")
    print(f"  Success: {run.success}")
```

## Benefits

### ✅ Tools Independent of CLI
Tools don't know about CLI - they just accept `run_id`.
Can use tools in notebooks, scripts, API without CLI.

### ✅ Storage Independent of Execution
Storage persists to `.aopt_runs/` directory.
Survives CLI restarts, accessible from anywhere.

### ✅ Agent Has Explicit Control
Agent decides when to create runs, what to track.
No hidden magic, full autonomy.

### ✅ Clean Code Structure
Each layer has single responsibility:
- `aopt/runs/` - Active run management
- `aopt/storage/` - Persistence
- `aopt/tools/` - Agent capabilities
- `aopt/cli/` - User interface

### ✅ Professional & Maintainable
Standard design patterns, no tight coupling.
Easy to test, extend, modify.

## Comparison: Old vs New

### ❌ Old (Callback-Based)

```python
# CLI injected callback_manager into tools
def execute_tool(tool, args, callback_manager):
    args['callback_manager'] = callback_manager  # ❌ Coupling
    return tool.invoke(args)

# Tools emitted events
@tool
def run_optimization(..., callback_manager=None):
    result = optimize(...)
    if callback_manager:  # ❌ Tool knows about CLI
        callback_manager.emit(OPTIMIZATION_COMPLETE)
```

**Problems:**
- Tools coupled to CLI infrastructure
- Can't use tools without callback system
- Hard to test in isolation

### ✅ New (Run-Based)

```python
# Agent explicitly creates run
@tool
def start_optimization_run(problem_id, algorithm):
    run = RunManager().create_run(...)
    return {"run_id": run.run_id}

# Tools accept optional run_id
@tool
def run_optimization(..., run_id=None):
    result = optimize(...)
    if run_id:  # ✅ Tool independent
        run = RunManager().get_run(run_id)
        run.finalize(result)
```

**Benefits:**
- Tools fully independent
- Agent explicitly manages tracking
- Clean separation of concerns

## Testing

```bash
# Test end-to-end architecture
python test_run_architecture.py

✓ RunManager initialized with storage
✓ Problem registered
✓ Run created: #1
✓ Run retrieved from manager
✓ Recorded 5 iterations
✓ Run finalized
✓ Run persisted to storage
✓ CLI can query runs from storage

✅ All tests passed!
```

## Extension Points

### Adding New Storage Backend

```python
class SQLiteStorage(StorageBackend):
    """SQLite-based storage."""

    def save_run(self, run):
        # Save to database

    def load_run(self, run_id):
        # Load from database
```

### Adding New UI

```python
# Web API
@app.get("/runs")
def get_runs():
    storage = FileStorage('.aopt_runs')
    return storage.load_all_runs()

# Jupyter widget
from aopt.storage import FileStorage
from ipywidgets import Table

storage = FileStorage()
runs = storage.load_all_runs()
display(Table(runs))
```

### Adding Run Analytics

```python
@tool
def analyze_run_convergence(run_id: int):
    """Analyze convergence of completed run."""
    storage = RunManager().get_storage()
    run = storage.load_run(run_id)

    # Analyze result_data["iterations"]
    return convergence_analysis
```

## Summary

The run-based architecture provides:
1. **Explicit agent control** over tracking
2. **Automatic persistence** to storage
3. **Complete independence** between layers
4. **Professional code structure** with clean separation

This is the correct way to build a multi-layered system with independent components.
