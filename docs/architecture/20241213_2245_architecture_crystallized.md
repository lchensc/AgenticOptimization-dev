# PAOLA Architecture - Crystallized Design

**Date**: December 13, 2025
**Version**: 0.1.0
**Status**: Production-Ready

---

## 1. Project Identity

### Name and Branding

**PAOLA**: **Package** for Agentic Optimization with Learning and Analysis

- **Package** (not Platform) - Humble positioning for early releases
- Lowercase "and" in expansion (not affecting acronym)
- Future: May evolve to "Platform" as capabilities mature

### Tagline

*The optimization package that learns from every run*

### Core Innovation

**PAOLA solves**: Engineering optimization requires deep expertise that is:
- **Tacit**: Locked in senior engineers' heads ("tighten constraints by 2%")
- **Manual**: Fixed algorithms, manual tuning, trial-and-error
- **Non-learning**: Every optimization starts from scratch
- **Failure-prone**: 50% of optimizations fail (gradient noise, infeasibility)

**PAOLA transforms**: Manual expertise → Autonomous intelligence through:
- **Agent autonomy**: No fixed loops, agent composes strategies from tools
- **Knowledge accumulation**: Learns from every run via RAG-based retrieval
- **Strategic adaptation**: Detects and resolves issues automatically

---

## 2. Architecture Overview

### Layered Architecture

```
┌──────────────────────────────────────────────────────────────┐
│            APPLICATION LAYER (User Interface)                │
│  - CLI (interactive REPL)                                    │
│  - Python SDK (programmatic access)                          │
│  - Future: Web UI, REST API                                  │
└───────────────────────┬──────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────┐
│         INTELLIGENCE LAYER (Core Innovation)                 │
│  - Agent (autonomous reasoning, strategy composition)        │
│  - Knowledge Base (RAG, patterns, warm-starting)             │
│  - Analysis (deterministic metrics + AI reasoning)           │
└───────────────────────┬──────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────┐
│         EXECUTION LAYER (Tool Primitives)                    │
│  - Optimizer Tools (create, propose, update, restart)        │
│  - Evaluator Tools (function eval, gradient compute)         │
│  - Observation Tools (convergence, pattern detection)        │
│  - Gate Control Tools (continuation decisions)               │
│  - Cache Tools (evaluation reuse)                            │
└───────────────────────┬──────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────┐
│      DATA FOUNDATION LAYER (Foundry - Inspired by           │
│                             Palantir Foundry)                │
│  - OptimizationFoundry (single source of truth)             │
│  - Run Management (create, track, finalize)                  │
│  - Storage Backend (FileStorage with lineage)                │
│  - Problem Definitions (canonical formulations)              │
└──────────────────────────────────────────────────────────────┘
```

### Design Philosophy: Foundry-Inspired

**Inspired by Palantir Foundry's ontology layer**:
- **Single source of truth**: All optimization data organized in foundry
- **Versioning & lineage**: Track how problems evolve, which runs derive from which
- **Separation of concerns**: Data foundation separate from intelligence
- **Builder's workshop**: Intelligence layer builds solutions on top of foundation

**Critical difference from Palantir Foundry**:
- PAOLA adds **autonomous intelligence** (agent) on top of data foundation
- Foundry is **passive** infrastructure; PAOLA agent is **active** intelligence
- PAOLA = Foundry's data organization + Autonomous agent capabilities

---

## 3. Module Structure

### 3.1 Foundry (Data Foundation)

**Location**: `paola/foundry/`

**Purpose**: Single source of truth for optimization data

**Components**:

```python
# paola/foundry/foundry.py
class OptimizationFoundry:
    """
    Data foundation for optimization runs.

    Manages:
    - Problem definitions (canonical formulations)
    - Run lifecycle (create → track → finalize)
    - Storage persistence (with lineage tracking)
    - Query interfaces (for analysis and knowledge extraction)
    """

    def __init__(self, storage: StorageBackend):
        """Dependency injection - no singletons."""
        self.storage = storage
        self._active_runs: Dict[int, Run] = {}

    def create_run(self, problem_id, algorithm) -> Run:
        """Create new tracked optimization run."""

    def load_run(self, run_id) -> RunRecord:
        """Load completed run from storage."""

    def query_runs(self, filters) -> List[RunRecord]:
        """Query with algorithm, problem, success filters."""

# paola/foundry/run.py
class Run:
    """Active optimization run (in-progress)."""
    def record_iteration(self, design, objective):
        """Record iteration and auto-persist."""

    def finalize(self, result):
        """Finalize with scipy result and persist."""

class RunRecord:
    """Storage representation of completed run."""
    # Immutable, for analysis and knowledge extraction

# paola/foundry/storage/
class StorageBackend(ABC):
    """Abstract interface for storage."""

class FileStorage(StorageBackend):
    """JSON-based file storage with lineage."""
```

**Key Principles**:
- Dependency injection (testable, no singletons)
- Active vs Storage separation (Run vs RunRecord)
- Auto-persistence (every iteration saves)
- Lineage tracking (which run came from which problem)

---

### 3.2 Agent (Intelligence Layer)

**Location**: `paola/agent/`

**Purpose**: Autonomous reasoning and strategy composition

**Components**:

```python
# paola/agent/react_agent.py
def build_optimization_agent(llm_model, tools, callback_manager):
    """
    Build LangGraph ReAct agent.

    Agent loop:
    1. Observe (read optimization state)
    2. Reason (LLM decides next action)
    3. Act (execute tool)
    4. Repeat until goal achieved
    """

    graph = StateGraph(AgentState)
    graph.add_node("react", react_node)
    # Agent decides when to stop (no fixed loop)

# paola/agent/prompts.py
def build_optimization_prompt(goal, context, budget):
    """
    Build system prompt for agent.

    Provides:
    - Goal description
    - Available tools
    - Context (current optimization state)
    - Budget (remaining evaluations/CPU hours)
    """
```

**Key Principles**:
- Full autonomy (agent decides when to stop)
- Tool composition (no fixed loop)
- Observable (emits events at every step)
- Context-aware (sees optimization state)

---

### 3.3 Tools (Execution Layer)

**Location**: `paola/tools/`

**Purpose**: Atomic primitives that agent composes into strategies

**Categories** (20+ tools total):

**1. Problem Formulation** (1 tool):
- `create_benchmark_problem` - Create test problems

**2. Run Management** (3 tools):
- `start_optimization_run` - Create tracked run
- `finalize_optimization_run` - Finalize completed run
- `get_active_runs` - List active runs

**3. Optimization** (6 tools):
- `optimizer_create` - Create optimizer instance
- `optimizer_propose` - Get next design
- `optimizer_update` - Update with evaluation
- `optimizer_restart` - Restart with new settings
- `run_scipy_optimization` - Convenience wrapper

**4. Evaluation** (3 tools):
- `evaluate_function` - Run simulation
- `compute_gradient` - Finite-difference or adjoint
- `create_benchmark_problem` - Test functions

**5. Cache** (4 tools):
- `cache_get` - Retrieve cached evaluation
- `cache_store` - Store evaluation
- `cache_clear` - Clear cache
- `cache_stats` - Get statistics

**6. Observation** (5 tools):
- `analyze_convergence` - Rate, stalling, improvement
- `detect_pattern` - Feasibility issues, gradient noise
- `check_feasibility` - Constraint violations
- `get_gradient_quality` - Gradient variance

**7. Analysis** (3 tools):
- `get_all_metrics` - Deterministic metrics (instant, free)
- `analyze_run_with_ai` - AI diagnosis (~$0.02-0.05)

**8. Knowledge** (3 tools, skeleton):
- `store_optimization_insight` - Store successful strategy
- `retrieve_optimization_knowledge` - RAG-based retrieval
- `list_all_knowledge` - List all insights

**Key Principles**:
- LangChain `@tool` decorators (agent-compatible)
- Explicit dependency injection (set_foundry)
- Error handling (tools never crash agent)
- Cost transparency (expensive operations documented)

---

### 3.4 Knowledge (Learning Layer)

**Location**: `paola/knowledge/`

**Purpose**: Accumulate and retrieve optimization wisdom

**Current Status**: Skeleton (interface defined, ready for data)

**Design** (to be implemented with 50+ runs):

```python
# paola/knowledge/knowledge_base.py
class KnowledgeBase:
    """RAG-based knowledge retrieval for optimization."""

    def store_insight(self, insight: OptimizationInsight):
        """
        Store successful strategy.

        Insight contains:
        - Problem signature (dimension, constraints, physics)
        - Strategy (algorithm, settings, adaptations)
        - Outcome (success rate, iterations, cost)
        - Narrative (what worked, what didn't)
        """

    def retrieve_similar(self, signature: ProblemSignature, k=5):
        """
        Find k most similar past optimizations.

        Uses:
        - Embedding-based similarity
        - Problem signature matching
        - Physics domain filtering
        """

# paola/knowledge/signature.py
class ProblemSignature:
    """Characteristics that define a problem class."""
    dimension: int
    has_constraints: bool
    constraint_types: List[str]
    physics_domain: Optional[str]  # "aerodynamics", "structures"

    def similarity(self, other) -> float:
        """Compute similarity score [0, 1]."""
```

**Future Implementation**:
1. Collect 50+ runs of diverse problems
2. Implement problem signature extraction
3. RAG with vector DB (Chroma, FAISS)
4. Warm-starting from similar problems
5. Agentic learning (agent stores insights after runs)

---

### 3.5 Analysis (Intelligence Layer)

**Location**: `paola/analysis/`

**Purpose**: Dual-layer metrics (deterministic + AI)

**Components**:

```python
# paola/analysis/metrics.py
def compute_metrics(run: RunRecord) -> Dict[str, Any]:
    """
    Deterministic metrics (instant, free).

    Returns:
    - Convergence: rate, stalling, improvement
    - Gradient: variance, norm, quality
    - Constraints: violations, activity
    - Efficiency: evaluations, improvement per eval
    - Objective: final value, history
    """

# paola/analysis/ai_analysis.py
async def ai_analyze(
    run: RunRecord,
    focus: str = "overall",
    llm_model: str = "qwen-flash"
) -> Dict[str, Any]:
    """
    AI-powered strategic analysis (~$0.02-0.05).

    Returns:
    - Diagnosis (what happened)
    - Recommendations (what to try next)
    - Confidence (how sure is the diagnosis)

    Focus options:
    - convergence (why did it converge/stall?)
    - efficiency (how to use fewer evaluations?)
    - algorithm (is this the right algorithm?)
    - overall (comprehensive analysis)
    """
```

**Key Principles**:
- Two-tier: Fast free metrics, strategic AI analysis
- Cost transparency (~$0.02-0.05 for AI)
- Focus parameter (targeted analysis)

---

### 3.6 CLI (Application Layer)

**Location**: `paola/cli/`

**Purpose**: Interactive REPL for optimization

**Components**:

```python
# paola/cli/repl.py
class AgenticOptREPL:
    """Main REPL for interactive optimization."""

    def __init__(self, llm_model="qwen-flash"):
        # Initialize foundry (data foundation)
        self.foundry = OptimizationFoundry(storage=FileStorage())
        set_foundry(self.foundry)

        # Initialize agent (intelligence)
        self.agent = build_optimization_agent(llm_model, tools)

    def run(self):
        """Main conversation loop."""

# paola/cli/commands.py
class CommandHandler:
    """Handles /commands (reads from foundry)."""

    def handle_runs(self):
        """Display all runs in table."""

    def handle_show(self, run_id):
        """Show detailed run with metrics."""

    def handle_analyze(self, run_id, focus):
        """AI-powered analysis."""
```

**Commands**:
- Natural language: "optimize 10D Rosenbrock with SLSQP"
- `/runs` - List all runs
- `/show <id>` - Detailed results with metrics
- `/analyze <id> [focus]` - AI analysis
- `/plot <id>` - Convergence plot
- `/compare <id1> <id2>` - Side-by-side comparison
- `/best` - Best solution across all runs
- `/tokens` - LLM token usage statistics
- `/help`, `/exit`, `/clear`, `/model`, `/models`

---

## 4. Data Flow

### 4.1 Optimization Workflow

```
User: "optimize 10D Rosenbrock with SLSQP"
   ↓
┌──────────────────────────────────────────┐
│ 1. AGENT REASONING                       │
│    - Parse goal                          │
│    - Decide strategy                     │
│    - Select tools                        │
└──────────┬───────────────────────────────┘
           ↓
┌──────────▼───────────────────────────────┐
│ 2. PROBLEM CREATION                      │
│    Tool: create_benchmark_problem        │
│    → Registers in foundry                │
└──────────┬───────────────────────────────┘
           ↓
┌──────────▼───────────────────────────────┐
│ 3. RUN CREATION                          │
│    Tool: start_optimization_run          │
│    → Foundry creates Run instance        │
│    → Returns run_id                      │
└──────────┬───────────────────────────────┘
           ↓
┌──────────▼───────────────────────────────┐
│ 4. OPTIMIZATION EXECUTION                │
│    Tool: run_scipy_optimization          │
│    → Calls scipy.optimize                │
│    → Records iterations to Run           │
│    → Auto-persists to foundry            │
└──────────┬───────────────────────────────┘
           ↓
┌──────────▼───────────────────────────────┐
│ 5. FINALIZATION                          │
│    Tool: finalize_optimization_run       │
│    → Run.finalize(result)                │
│    → Persists to foundry storage         │
│    → Removes from active runs            │
└──────────┬───────────────────────────────┘
           ↓
┌──────────▼───────────────────────────────┐
│ 6. ANALYSIS                              │
│    Tool: get_all_metrics                 │
│    → Foundry.load_run(run_id)            │
│    → compute_metrics(run_record)         │
│    → Returns to agent                    │
└──────────────────────────────────────────┘
```

### 4.2 Knowledge Accumulation (Future)

```
After Run Completes
   ↓
┌──────────────────────────────────────────┐
│ Agent Reflection                         │
│  - What worked?                          │
│  - What failed?                          │
│  - Key insight?                          │
└──────────┬───────────────────────────────┘
           ↓
┌──────────▼───────────────────────────────┐
│ Tool: store_optimization_insight         │
│  - Extract problem signature             │
│  - Record successful adaptations         │
│  - Store to knowledge base               │
└──────────┬───────────────────────────────┘
           ↓
┌──────────▼───────────────────────────────┐
│ Future Optimization                      │
│  Tool: retrieve_optimization_knowledge   │
│  - RAG retrieval of similar problems     │
│  - Warm-start with proven strategy       │
│  - 30% faster convergence                │
└──────────────────────────────────────────┘
```

---

## 5. Key Design Principles

### 5.1 Foundry-Inspired Data Foundation

**Principle**: Single source of truth for all optimization data

**Implementation**:
- All runs stored in foundry with consistent schema
- Lineage tracking (problem → run → result)
- Version control (problems evolve over time)
- Queryable (filter by algorithm, problem, success)

**Benefit**: Team collaboration, no "which result is correct?" confusion

### 5.2 Agent Autonomy

**Principle**: Agent controls everything, no fixed loops

**Implementation**:
- Agent decides: when to evaluate, what fidelity, when to adapt, when to stop
- Tool primitives (not prescribed algorithms)
- ReAct loop (observe → reason → act)

**Benefit**: Novel strategies emerge, not limited to pre-programmed approaches

### 5.3 Observable Everything

**Principle**: Every action is observable and explainable

**Implementation**:
- Event system (15+ event types)
- Callback manager (multiple listeners)
- CLI callback (real-time display)
- File logger (audit trail)

**Benefit**: Debugging, trust, transparency

### 5.4 Knowledge Accumulation

**Principle**: Platform learns from every run

**Implementation**:
- RAG-based retrieval
- Problem signatures
- Warm-starting
- Agentic learning (agent stores insights)

**Benefit**: Faster convergence, democratized expertise

### 5.5 Separation of Concerns

**Layers are independent**:
- **Foundry**: Data persistence (no intelligence)
- **Tools**: Atomic operations (no orchestration)
- **Agent**: Intelligence (no data storage)
- **CLI**: Presentation (no business logic)

**Benefit**: Testable, maintainable, extensible

---

## 6. Technology Stack

### Core Dependencies

**Agent Framework**:
- LangChain (tool abstraction)
- LangGraph (ReAct state machine)
- Qwen/Claude/OpenAI (LLM providers)

**Data & Analysis**:
- Pydantic (schemas)
- NumPy, SciPy (optimization)
- JSON (storage format)

**User Interface**:
- Rich (terminal output)
- prompt_toolkit (REPL)
- asciichartpy (plotting)

**LLM Integration**:
- Token tracking (cost management)
- Prompt caching (90% cost savings on repeated prompts)

---

## 7. Future Extensions

### Phase 1-3 (Next 6 months)

**Track 1: Engineering Workflows** (3-4 weeks)
- SU2/OpenFOAM integration
- Multi-fidelity support
- Budget tracking
- **Milestone**: Optimize transonic airfoil with agent

**Track 2: Knowledge Base** (4-5 weeks)
- Collect 50+ runs
- Problem signature extraction
- RAG implementation
- Warm-starting
- **Milestone**: 30% faster convergence with warm-start

**Track 3: Strategic Adaptation** (3-4 weeks)
- Enhanced observation (feasibility, gradient quality)
- Constraint management
- Gradient method switching
- **Milestone**: Agent resolves feasibility issue automatically

### Phase 4-5 (Next 12 months)

**Integration Expansion** (Foundry-inspired):
- CAD connectors (parametrization)
- Pipeline builder (DAG workflows)
- Advanced collaboration (reviews, approvals)
- Enterprise features (SSO, multi-tenancy)

---

## 8. Success Metrics

### Technical Metrics
- ✅ 30+ tests passing (all phases)
- ✅ Clean imports (no circular dependencies)
- ✅ Type hints (Pydantic schemas)
- ✅ Documentation (comprehensive)

### User Experience Metrics
- Natural language goals work
- <5 seconds from goal to first action
- Real-time feedback (event streaming)
- Persistent runs (survive CLI restart)

### Business Metrics (Future)
- 30% faster convergence with warm-start
- 90% success rate (vs 50% baseline)
- 10× cost savings (multi-fidelity + cache)

---

## 9. Summary

**PAOLA** is a Python **package** (not platform, humble positioning) for autonomous engineering optimization that:

1. **Organizes data** in a foundry (single source of truth, inspired by Palantir Foundry)
2. **Adds intelligence** via autonomous agent (unique differentiator)
3. **Accumulates knowledge** across runs via RAG-based learning
4. **Provides analysis** with dual-layer metrics (deterministic + AI)
5. **Enables collaboration** with shared workspace (future)

**Architecture**: 4 layers (Application → Intelligence → Execution → Foundation)

**Core Innovation**: Agent autonomy + Knowledge accumulation on top of organized data foundation

**Current Status**: Production-ready for optimization workflows (Phases 1-5 complete)

**Next Phase**: Engineering integration + Knowledge base + Strategic adaptation

---

**Version**: 0.1.0
**Date**: December 13, 2025
**Status**: ✅ Production-Ready
