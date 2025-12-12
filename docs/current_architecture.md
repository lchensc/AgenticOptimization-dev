# PAOLA Current Architecture Analysis

**Status**: Phase 2 Complete (CLI + Run Tracking)
**Generated**: 2025-12-12

## Executive Summary

PAOLA has evolved from design documents into a working system with 5 major architectural layers:

1. **Agent Layer** - ReAct agent with LangGraph
2. **Tool Layer** - Agent primitives for optimization control
3. **Run Management Layer** - Active run tracking (NEW in Phase 2)
4. **Storage Layer** - Persistent run history (NEW in Phase 2)
5. **CLI Layer** - Interactive REPL interface (NEW in Phase 2)

## Architectural Layers

### Layer 1: Agent Core (`paola/agent/`)

**Purpose**: Autonomous ReAct agent that controls optimization strategy

**Key Components**:
- `react_agent.py` - LangGraph-based ReAct loop
- `agent.py` - Base agent abstractions

**Architecture Pattern**:
```
User Goal → Agent → [Observe → Reason → Act] → Tools → Results
                ↑___________________________________|
                        Continuous Loop
```

**Key Design Decisions**:
1. **Message History Retention**: Uses `Annotated[list, operator.add]` to accumulate full conversation history
   - **Critical**: Prevents agent from "forgetting" past decisions
   - Fixed in architecture review (was previously losing context)

2. **Full Autonomy**: Agent decides when to stop (no fixed iteration count)
   - Checks for "DONE" or "CONVERGED" in agent response
   - No prescribed control flow

3. **Multi-LLM Support**:
   - Qwen (qwen-flash, qwen-plus, qwen-max) via DASHSCOPE_API_KEY
   - Anthropic (Claude) via ANTHROPIC_API_KEY
   - OpenAI (GPT-4) via OPENAI_API_KEY
   - Runtime model switching in CLI

4. **Event Emission**: Real-time callbacks at every decision point
   - AGENT_STEP, REASONING, TOOL_CALL, TOOL_RESULT, etc.
   - Enables streaming display in CLI

**Strengths**:
- Clean separation: agent logic vs tool implementations
- Full autonomy preserved (no hardcoded strategies)
- Robust message threading for LangChain compatibility

**Potential Issues**:
- Prompt building is verbose (522 lines in `react_agent.py`)
- Context update logic incomplete (line 516: `# TODO: Implement context update logic`)

---

### Layer 2: Tool Primitives (`paola/tools/`)

**Purpose**: Atomic operations that agent composes into strategies

**Tool Categories**:

1. **Evaluator Tools** (`evaluator_tools.py`)
   - `create_benchmark_problem` - Define optimization problems
   - Evaluation and gradient computation

2. **Optimizer Tools** (`optimizer_tools.py`)
   - `run_scipy_optimization` - Execute scipy optimizer
   - Supports SLSQP, L-BFGS-B, COBYLA, etc.

3. **Run Management Tools** (`run_tools.py`) - **NEW**
   - `start_optimization_run` - Create tracked run
   - `finalize_optimization_run` - Close run and add notes
   - `get_active_runs` - List in-progress runs

4. **Observation Tools** (`observation_tools.py`)
   - `analyze_convergence` - Detect stalling/divergence
   - Pattern detection, gradient quality checks

5. **Gate Control Tools** (`gate_control_tools.py`)
   - For iteration-by-iteration blocking mode (Phase 3)
   - Agent approves each optimizer step

6. **Cache Tools** (`cache_tools.py`)
   - Evaluation cache for expensive simulations
   - Cache hit rate tracking

**Architecture Pattern**:
```
Agent calls tool → Tool uses RunManager singleton → Updates active run → Auto-persists to storage
```

**Key Design Decisions**:

1. **Explicit Run Creation**: Agent must call `start_optimization_run` before optimization
   - Returns `run_id` that links all subsequent operations
   - Ensures intentional run tracking (no implicit state)

2. **Auto-Persistence**: Every iteration automatically saves to storage
   - Active run calls `run._persist()` after each iteration
   - No manual save required

3. **Singleton RunManager**: Global registry of active runs
   - Thread-safe singleton pattern
   - Tools access via `RunManager()` (always returns same instance)

**Strengths**:
- Tools are simple, focused, composable
- Run tracking is transparent to optimization algorithms
- Cache integration reduces redundant evaluations

**Potential Issues**:
- RunManager singleton creates global state (testing complexity)
- No rollback mechanism if tool fails mid-operation
- Limited validation on tool arguments (agent can pass invalid data)

---

### Layer 3: Run Management (`paola/runs/`)

**Purpose**: Track optimization runs as first-class objects

**Key Components**:

1. **RunManager** (`manager.py`) - Singleton registry
   - Creates runs with auto-incrementing IDs
   - Maintains active run registry
   - Connects to storage backend

2. **OptimizationRun** (`active_run.py`) - Active run object
   - Records iterations as optimization progresses
   - Auto-persists to storage on every update
   - Finalizes with scipy `OptimizeResult`

**Data Flow**:
```
Agent calls start_optimization_run
    ↓
RunManager.create_run()
    ↓
Gets next ID from Storage.get_next_run_id()
    ↓
Creates OptimizationRun(run_id, storage=...)
    ↓
Registers in _active_runs dict
    ↓
Returns run object to tool
    ↓
Tool returns run_id to agent
    ↓
Agent uses run_id in run_scipy_optimization
    ↓
Optimization records iterations via run.record_iteration()
    ↓
Each iteration calls run._persist() → storage.save_run()
```

**Key Design Decisions**:

1. **Active Objects**: Runs are not passive data - they actively manage state
   - Methods: `record_iteration()`, `finalize()`, `get_current_best()`
   - Encapsulates persistence logic

2. **Dual Model Pattern**:
   - `active_run.OptimizationRun` - In-memory active object
   - `storage.models.OptimizationRun` - Serializable dataclass
   - Conversion via `_to_model()` method

3. **Iteration Tracking**: Full history stored in memory and persisted
   - Design vector, objective, gradient, constraints per iteration
   - Enables convergence analysis and plotting

**Strengths**:
- Clear separation: active management vs persistent storage
- Auto-persistence eliminates manual save points
- Run state survives crashes (persisted every iteration)

**Potential Issues**:
- Singleton pattern makes parallel runs difficult
- No transaction semantics (partial iterations persist)
- Memory growth with long optimization runs (all iterations in RAM)

---

### Layer 4: Storage Backend (`paola/storage/`)

**Purpose**: Persistent storage for optimization history

**Key Components**:

1. **StorageBackend** (`base.py`) - Abstract interface
   - Defines contract: `save_run()`, `load_run()`, `load_all_runs()`, etc.
   - Enables swapping implementations (file, database, S3)

2. **FileStorage** (`file_storage.py`) - File-based implementation
   - JSON files in `.paola_runs/` directory
   - Structure:
     ```
     .paola_runs/
     ├── runs/
     │   ├── run_001.json
     │   ├── run_002.json
     │   └── ...
     ├── problems/
     │   └── rosenbrock_10d.json
     └── metadata.json (tracks next_run_id)
     ```

3. **Data Models** (`models.py`) - Serializable dataclasses
   - `OptimizationRun` - Run metadata + result
   - `Problem` - Problem definition

**Architecture Pattern**:
```
OptimizationRun (active) --_to_model()--> OptimizationRun (dataclass) --to_json()--> File
                                                                                        ↓
File --from_json()--> OptimizationRun (dataclass) --CLI reads--> Display
```

**Key Design Decisions**:

1. **JSON Storage**: Human-readable, git-friendly, no DB dependencies
   - Trade-off: Slower than binary, no indexing/queries

2. **Atomic ID Generation**: `get_next_run_id()` increments counter in metadata.json
   - Ensures unique IDs across sessions
   - No race conditions (file-based locking via OS)

3. **Full Result Storage**: Stores complete scipy `OptimizeResult` + all iterations
   - Enables post-analysis without re-running
   - Trade-off: Large files for long runs

**Strengths**:
- No external dependencies (just filesystem)
- Easy to backup/version control
- Human-readable for debugging

**Potential Issues**:
- No concurrent access handling (race conditions possible)
- Linear scan for queries (no indexing)
- File size grows unbounded (no compression)
- No migration strategy if schema changes

---

### Layer 5: CLI Interface (`paola/cli/`)

**Purpose**: Interactive REPL for conversational optimization

**Key Components**:

1. **REPL** (`repl.py`) - Main interface
   - prompt_toolkit for rich input (history, auto-suggest)
   - Rich console for formatted output
   - Handles: user input, agent invocation, command routing

2. **CommandHandler** (`commands.py`) - Slash command logic
   - Pure presentation layer (reads from storage)
   - Commands: `/runs`, `/show`, `/plot`, `/compare`, `/best`
   - ASCII plotting with asciichartpy

3. **CLICallback** (`callback.py`) - Real-time event display
   - Subscribes to agent events
   - Streams tool calls, reasoning, results as they happen

**User Interaction Flow**:
```
User types: "optimize 10D Rosenbrock with SLSQP"
    ↓
REPL._process_with_agent()
    ↓
Agent invoked with conversation history
    ↓
Agent emits events → CLICallback displays in real-time
    ↓
Agent calls tools → Results streamed
    ↓
Final response displayed
    ↓
User types: "/plot 1" → CommandHandler reads storage → Displays ASCII plot
```

**Key Design Decisions**:

1. **Separation of Concerns**:
   - REPL: Manages agent + user interaction
   - CommandHandler: Pure presentation (stateless, reads storage)
   - No mixing of agent state and display logic

2. **Two Interaction Modes**:
   - Natural language → Agent (non-deterministic, uses tools)
   - Slash commands → Direct queries (deterministic, fast)

3. **Streaming Display**: Events displayed as they occur
   - No "waiting..." spinners
   - User sees agent reasoning in real-time

4. **Model Selection**: Runtime LLM switching
   - `/models` command shows available models
   - Re-initializes agent with new model
   - Conversation history preserved

**Strengths**:
- Clean separation: agent control vs deterministic queries
- Real-time feedback prevents "black box" feeling
- Persistent storage enables post-session analysis

**Potential Issues**:
- No error recovery if agent crashes mid-optimization
- Plot downsampling loses information for long runs
- No pagination for large run lists
- ASCII plots limited vs matplotlib (but normalize for terminal width)

---

## Cross-Cutting Concerns

### 1. Callback System (`paola/callbacks/`)

**Purpose**: Event streaming for real-time monitoring

**Architecture**:
```
Agent emits event → CallbackManager → [Callback1, Callback2, ...] → Display/Log/File
```

**Event Types** (18 total):
- Agent lifecycle: AGENT_START, AGENT_STEP, AGENT_DONE
- Tool execution: TOOL_CALL, TOOL_RESULT, TOOL_ERROR
- Optimization: ITERATION_START, EVALUATION, CACHE_HIT
- Convergence: CONVERGENCE_CHECK, PATTERN_DETECTED
- Adaptation: RESTART, ADAPTATION_START

**Callbacks**:
- `RichConsoleCallback` - Terminal display (CLI)
- `FileLoggerCallback` - JSON event log
- `CaptureCallback` - For testing/replay

**Strengths**:
- Decouples agent from display
- Easy to add new callbacks (monitoring, logging, etc.)
- Error isolation (one callback failure doesn't break others)

**Potential Issues**:
- No event replay/time-travel debugging
- Callbacks are fire-and-forget (no acknowledgment)

---

### 2. Data Flow Architecture

**Key Insight**: PAOLA has **two separate data paths**:

**Path 1: Agent Loop (Non-Deterministic)**
```
User Goal → Agent → Tools → RunManager → Active Run → Storage (persist)
```
- Agent controls flow
- Non-deterministic reasoning
- Creates/updates data

**Path 2: Query Commands (Deterministic)**
```
User Command → CommandHandler → Storage (read) → Display
```
- Direct storage reads
- Deterministic presentation
- No agent involved

**Why This Matters**:
- Agent and queries never conflict (one writes, one reads)
- Storage is single source of truth
- CLI can inspect runs while agent is running (future: concurrent runs)

---

## Architecture Strengths

### 1. **Agent Autonomy Preserved**
- Agent has full control (no hardcoded loops)
- Tools are primitives, not prescriptive workflows
- Matches PAOLA vision from design docs

### 2. **Persistent Learning Foundation**
- All runs persisted automatically
- Storage abstraction enables future backends (SQL, S3, etc.)
- Ready for Phase 3 knowledge base (RAG over past runs)

### 3. **Clean Separation of Concerns**
- Agent ≠ Storage ≠ Display
- Each layer has single responsibility
- Easy to test/extend independently

### 4. **Real-Time Observability**
- Callback system shows agent reasoning live
- No "black box" optimization
- Users understand agent decisions

### 5. **Multi-Run Analysis Ready**
- Storage schema supports comparison tools
- CLI has `/compare` and `/plot compare` commands
- Foundation for Phase 3 multi-run learning

---

## Architecture Issues & Design Discussion Points

### Issue 1: **Singleton RunManager Creates Testing Complexity**

**Current State**:
```python
manager = RunManager()  # Always returns same instance
```

**Problem**:
- Global state persists across tests
- Must call `RunManager.reset()` in test teardown
- Parallel agent execution difficult

**Options for Discussion**:
1. **Keep singleton, improve testing utilities**
   - Provide `@with_clean_run_manager` decorator
   - Accept limitation: one active optimization per process

2. **Dependency injection pattern**
   ```python
   agent = build_agent(tools, run_manager=RunManager())
   tools = build_tools(run_manager=run_manager)
   ```
   - Pro: Testable, explicit dependencies
   - Con: More verbose, breaks current tool signatures

3. **Context manager pattern**
   ```python
   with OptimizationSession(storage) as session:
       agent = session.create_agent(...)
       # All tools share session.run_manager
   ```
   - Pro: Clear lifecycle, supports nesting
   - Con: Requires refactoring CLI and tools

**Question for Discussion**: Which approach fits PAOLA's long-term vision?

---

### Issue 2: **Incomplete Context Update Logic**

**Current State** (react_agent.py:516):
```python
def update_context(context: dict, tool_results: list) -> dict:
    """Update agent context with tool results."""
    new_context = context.copy()
    # TODO: Implement context update logic based on tool results
    return new_context
```

**Problem**:
- Agent context doesn't accumulate observations
- Tool results not reflected in next iteration's prompt
- Agent may "forget" recent discoveries

**Options for Discussion**:
1. **Automatic extraction from tool results**
   ```python
   if 'objective' in result:
       context['best_objectives'] = min(context.get('best_objectives', inf), result['objective'])
   ```
   - Pro: No manual tracking
   - Con: Assumes result structure

2. **Tool-specific context updaters**
   ```python
   CONTEXT_UPDATERS = {
       'run_scipy_optimization': update_optimization_context,
       'analyze_convergence': update_convergence_context,
   }
   ```
   - Pro: Explicit, testable
   - Con: Maintenance burden

3. **Rely on message history only**
   - Remove context dict entirely
   - Agent re-reads history each iteration
   - Pro: Simpler
   - Con: Long prompts, token costs

**Question for Discussion**: How important is structured context vs message history?

---

### Issue 3: **Storage Schema Evolution Strategy Missing**

**Current State**:
- JSON files with fixed schema
- No version field in stored runs
- No migration tooling

**Problem**:
- Adding new fields breaks old runs
- No backward compatibility plan

**Options for Discussion**:
1. **Schema versioning**
   ```python
   {
       "schema_version": "2.0",
       "run_id": 1,
       ...
   }
   ```
   - Migration scripts: `migrate_v1_to_v2.py`
   - Pro: Explicit, traceable
   - Con: Maintenance overhead

2. **Append-only design**
   - Never modify schema, only add fields
   - Old readers ignore new fields
   - Pro: Simple
   - Con: Accumulates cruft

3. **Accept breaking changes**
   - Phase 2 is pre-release
   - Document that storage format may change
   - Pro: Fast iteration
   - Con: Lost data between versions

**Question for Discussion**: How stable should storage be at this phase?

---

### Issue 4: **No Transaction Semantics in Run Tracking**

**Current State**:
```python
run.record_iteration(design, obj)  # Immediately persists
# If next operation fails, partial state persisted
```

**Problem**:
- Crash during optimization leaves incomplete run
- No rollback if finalization fails

**Options for Discussion**:
1. **Batch persistence**
   ```python
   with run.transaction():
       run.record_iteration(...)
       run.record_iteration(...)
   # Only persists on __exit__
   ```
   - Pro: Atomic updates
   - Con: Data loss if crash before commit

2. **Write-ahead log**
   ```python
   run.record_iteration(...)  # Writes to .paola_runs/wal/run_1.log
   run.finalize()             # Merges log into run_001.json
   ```
   - Pro: Crash recovery
   - Con: Complexity

3. **Accept inconsistency**
   - Mark runs as "in-progress" vs "finalized"
   - CLI shows status
   - Pro: Simple
   - Con: Incomplete data

**Question for Discussion**: How critical is crash recovery for Phase 2?

---

### Issue 5: **Limited Query Capabilities**

**Current State**:
```python
runs = storage.load_all_runs()  # Linear scan
best_run = min(runs, key=lambda r: r.objective_value)
```

**Problem**:
- No filtering (e.g., "all SLSQP runs on Rosenbrock")
- No sorting (e.g., "top 10 by convergence speed")
- Loads entire run history into memory

**Options for Discussion**:
1. **Add query methods to StorageBackend**
   ```python
   storage.query_runs(algorithm="SLSQP", problem_id="rosenbrock*", limit=10)
   ```
   - Pro: Clean API
   - Con: File-based storage still slow

2. **Switch to SQLite**
   ```python
   class SQLiteStorage(StorageBackend):
       # Native indexing, SQL queries
   ```
   - Pro: Fast, scalable
   - Con: Binary format, harder to inspect

3. **Index file**
   ```python
   .paola_runs/index.json:
   {
       "by_algorithm": {"SLSQP": [1, 3, 5], "BFGS": [2, 4]},
       "by_problem": {"rosenbrock_10d": [1, 2, 3]}
   }
   ```
   - Pro: Fast lookups, still JSON
   - Con: Index can desync from data

**Question for Discussion**: What query patterns are most important for Phase 3?

---

### Issue 6: **CLI Plot Downsampling Loses Information**

**Current State** (commands.py:118):
```python
if len(objectives) > max_chart_width:
    # Downsample to 60 points for terminal display
    step = len(objectives) / max_chart_width
    objectives_to_plot = [objectives[int(i * step)] for i in range(60)]
```

**Problem**:
- 1000-iteration run shows only 60 points
- Misses convergence details

**Options for Discussion**:
1. **Min-max downsampling**
   ```python
   # For each bin, show both min and max to preserve spikes
   ```
   - Pro: Preserves outliers
   - Con: More complex

2. **Adaptive downsampling**
   ```python
   # More points where objective changes rapidly
   ```
   - Pro: Preserves interesting regions
   - Con: Complex algorithm

3. **Offer matplotlib export**
   ```bash
   /plot 1 --save plot_001.png
   ```
   - Pro: Full fidelity
   - Con: Requires matplotlib, file management

4. **Zoom/pan interface**
   ```bash
   /plot 1 --range 0:100  # Show iterations 0-100 only
   ```
   - Pro: Interactive exploration
   - Con: Multiple commands needed

**Question for Discussion**: Is ASCII plotting enough, or should we add matplotlib?

---

## Recommendations for Discussion

### Priority 1: **Finalize Context Update Logic**
- **Why**: Impacts agent effectiveness immediately
- **Proposal**: Implement structured context with tool-specific updaters
- **Effort**: 1-2 days

### Priority 2: **Add Schema Versioning**
- **Why**: Prevents data loss as we iterate
- **Proposal**: Add `schema_version` field, document that format may change in Phase 2
- **Effort**: 1 day

### Priority 3: **Evaluate RunManager Singleton**
- **Why**: Affects testing and future parallelism
- **Proposal**: Keep singleton for Phase 2, revisit for Phase 3 when multi-run parallelism needed
- **Effort**: 0 days (accept current design)

### Priority 4: **Design Query Interface for Phase 3**
- **Why**: Enables knowledge base (RAG over past runs)
- **Proposal**: Sketch query API, defer implementation to Phase 3
- **Effort**: Planning only

### Lower Priority:
- Transaction semantics - Accept incomplete runs for now
- Plot downsampling - ASCII is sufficient, defer matplotlib
- Storage backend - FileStorage adequate for Phase 2

---

## Open Questions for Architectural Discussion

1. **Context vs History**: Should agent rely on structured context dict or just message history?
   - Structured context is explicit but requires maintenance
   - Message history is automatic but verbose/costly

2. **Storage Evolution**: File-based JSON or migrate to SQLite?
   - JSON is simple, human-readable, git-friendly
   - SQLite enables queries, scales better

3. **RunManager Lifecycle**: Singleton, dependency injection, or context manager?
   - Singleton is simple but couples code
   - DI is testable but verbose
   - Context manager is explicit but requires refactoring

4. **Event Replay**: Should we support time-travel debugging?
   - Store events to file for replay
   - Useful for debugging agent decisions
   - Adds complexity

5. **Multi-Run Parallelism**: Will Phase 3 need concurrent optimization runs?
   - If yes, singleton RunManager won't work
   - If no, current design is fine

6. **CLI vs API**: Should PAOLA have a programmatic API for notebooks/scripts?
   - CLI is for interactive use
   - Notebooks might want direct API: `paola.optimize(problem, algorithm)`
   - Can coexist with CLI

---

## Summary

PAOLA's current architecture successfully implements:
- ✅ Agentic control (no fixed loops)
- ✅ Persistent run tracking
- ✅ Real-time observability
- ✅ Multi-run analysis foundation

Key architectural decisions to discuss:
1. Context update strategy
2. Storage schema evolution
3. RunManager singleton pattern
4. Query interface design

The architecture is clean, modular, and true to the vision. Main focus for discussion should be **forward compatibility** as we move toward Phase 3 (knowledge base, multi-run learning).
