# PAOLA Refactoring Blueprint

**Document Version**: 1.0
**Date**: 2025-12-12
**Status**: Pre-Refactoring Design Document

## Executive Summary

PAOLA has reached Phase 2 completion with working CLI and run tracking, but the architecture has become complex with 5 intersecting layers. This document defines the **target architecture** for refactoring into 4 clean, purpose-driven modules that align with PAOLA's core value proposition: **organizational learning through AI-powered optimization**.

**Current Problems**:
- ❌ `runs/` + `storage/` awkwardly separated (two "OptimizationRun" classes)
- ❌ Analysis scattered (observation tools + CLI commands duplicate logic)
- ❌ No knowledge module (learning not designed for)
- ❌ RunManager singleton creates testing complexity

**Target Architecture**:
```
4 Clean Modules:
1. Data Platform    - Foundation (run management + persistence)
2. Analysis         - Intelligence (deterministic + AI reasoning)
3. Knowledge        - Learning (RAG-based knowledge accumulation)
4. Agent + Tools    - Orchestration (kept simple and clean)
```

**Timeline**: 6-8 days refactoring + 2-3 weeks polish/testing

---

## Part 1: Architectural Vision

### Core Value Proposition

**Traditional platforms**: Agent + optimization tools = **commodity**

**PAOLA's differentiator**: **Organizational memory** that enables knowledge accumulation

The platform's added value comes from:
1. **Excellent data management** - Every run perfectly captured
2. **AI-powered analysis** - Strategic reasoning, not just metrics
3. **Knowledge accumulation** - Learning from past runs (RAG)
4. **Agent orchestration** - Autonomous formulation → optimization → react

### First Principles: Module Design

Each module has a **single responsibility**:

| Module | Responsibility | Dependencies | Users |
|--------|---------------|--------------|-------|
| **Data Platform** | Run lifecycle + persistence | None (foundation) | All other modules |
| **Analysis** | Metrics + reasoning | Data Platform | Agent, CLI, Knowledge |
| **Knowledge** | Learning + retrieval | Data Platform, Analysis | Agent |
| **Agent + Tools** | Orchestration | All modules | CLI, API |

**Dependency Flow** (no circular dependencies):
```
Data Platform (foundation)
    ↑
    ├── Analysis (reads data, computes insights)
    ├── Knowledge (reads data + analysis, stores learnings)
    └── Agent + Tools (orchestrates everything)
```

---

## Part 2: Module Specifications

## Module 1: Data Platform (`paola/platform/`)

### Responsibility
- Manage optimization run lifecycle (create → track → finalize)
- Persist all optimization data (runs, problems, results)
- Provide query interface for retrieval
- Ensure data integrity and consistency

### Structure
```
paola/platform/
├── __init__.py
│   └── OptimizationPlatform (main API)
│
├── platform.py           # OptimizationPlatform class
├── run.py               # Run and RunRecord classes
├── problem.py           # Problem definition
│
├── storage/
│   ├── __init__.py
│   ├── backend.py       # Abstract StorageBackend
│   ├── file_storage.py  # JSON-based implementation
│   └── sql_storage.py   # Future: SQLite implementation
│
└── models.py            # Data schemas (dataclasses)
```

### Key Classes

#### **OptimizationPlatform**
```python
class OptimizationPlatform:
    """
    Central platform for optimization run management.

    Replaces: RunManager singleton
    Pattern: Dependency injection (testable, explicit)
    """

    def __init__(self, storage: StorageBackend):
        """
        Initialize platform with storage backend.

        Args:
            storage: Storage backend (FileStorage, SQLiteStorage, etc.)
        """
        self.storage = storage
        self._active_runs: Dict[int, Run] = {}

    # Run lifecycle
    def create_run(
        self,
        problem_id: str,
        problem_name: str,
        algorithm: str,
        description: str = ""
    ) -> Run:
        """
        Create new optimization run.

        Returns:
            Run: Active run handle for tracking progress
        """

    def get_run(self, run_id: int) -> Optional[Run]:
        """Get active run by ID."""

    def finalize_run(self, run_id: int) -> None:
        """Finalize run and remove from active registry."""

    # Queries
    def load_run(self, run_id: int) -> Optional[RunRecord]:
        """Load completed run from storage."""

    def load_all_runs(self) -> List[RunRecord]:
        """Load all runs from storage."""

    def query_runs(
        self,
        algorithm: Optional[str] = None,
        problem_id: Optional[str] = None,
        success: Optional[bool] = None,
        limit: int = 100
    ) -> List[RunRecord]:
        """Query runs with filters (future: for knowledge retrieval)."""

    # Problem management
    def register_problem(self, problem: Problem) -> None:
        """Register problem definition."""

    def get_problem(self, problem_id: str) -> Optional[Problem]:
        """Get problem definition."""
```

#### **Run** (Active Handle)
```python
class Run:
    """
    Active optimization run handle.

    Provides methods for tracking progress during optimization.
    Auto-persists to storage on every update.

    Replaces: runs.active_run.OptimizationRun
    """

    def __init__(
        self,
        run_id: int,
        problem_id: str,
        problem_name: str,
        algorithm: str,
        storage: StorageBackend
    ):
        self.run_id = run_id
        self.problem_id = problem_id
        self.problem_name = problem_name
        self.algorithm = algorithm
        self.storage = storage

        # Tracking state
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.iterations: List[IterationRecord] = []
        self.result: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}

        # AI insights cache
        self.ai_insights: Optional[Dict[str, Any]] = None

        self.finalized = False

    def record_iteration(
        self,
        design: np.ndarray,
        objective: float,
        gradient: Optional[np.ndarray] = None,
        constraints: Optional[Dict[str, float]] = None
    ) -> None:
        """Record optimization iteration and auto-persist."""

    def finalize(self, result: Any, metadata: Optional[Dict] = None) -> None:
        """Finalize run with scipy OptimizeResult."""

    def add_ai_insights(self, insights: Dict[str, Any]) -> None:
        """Cache AI analysis insights with run."""

    def get_current_best(self) -> Optional[Dict[str, Any]]:
        """Get current best objective and design."""

    def to_record(self) -> RunRecord:
        """Convert to immutable storage record."""
```

#### **RunRecord** (Storage Model)
```python
@dataclass
class RunRecord:
    """
    Immutable optimization run record for storage.

    Replaces: storage.models.OptimizationRun
    Separation: Active (Run) vs Data (RunRecord)
    """
    run_id: int
    problem_id: str
    problem_name: str
    algorithm: str
    objective_value: float
    success: bool
    n_evaluations: int
    timestamp: str  # ISO format
    duration: float
    result_data: Dict[str, Any]  # Full result + iterations
    ai_insights: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]: ...
    def to_json(self) -> str: ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunRecord': ...

    @classmethod
    def from_json(cls, json_str: str) -> 'RunRecord': ...
```

### Migration from Current Code

**Before** (Current):
```python
# Singleton pattern
manager = RunManager()  # Global state
run = manager.create_run(...)
```

**After** (Target):
```python
# Dependency injection
platform = OptimizationPlatform(storage=FileStorage())
run = platform.create_run(...)
```

**Changes Required**:
1. Rename `runs.active_run.OptimizationRun` → `platform.Run`
2. Rename `storage.models.OptimizationRun` → `platform.RunRecord`
3. Replace `RunManager()` calls with `platform` parameter
4. Update all tools to accept `platform` in initialization

---

## Module 2: Analysis (`paola/analysis/`)

### Responsibility
- Compute deterministic metrics (fast, free, reproducible)
- Provide AI-powered strategic reasoning (opt-in, costs money)
- Enable both programmatic (agent) and interactive (CLI) use
- Support knowledge extraction

### Structure
```
paola/analysis/
├── __init__.py
│   ├── compute_metrics()      # Main deterministic entry point
│   └── ai_analyze()           # Main AI entry point
│
├── metrics.py                 # DETERMINISTIC METRICS
│   ├── compute_metrics()
│   ├── compute_convergence_metrics()
│   ├── compute_gradient_metrics()
│   ├── compute_constraint_metrics()
│   └── compute_efficiency_metrics()
│
├── convergence.py             # DETERMINISTIC ALGORITHMS
│   ├── detect_stalling()
│   ├── detect_divergence()
│   └── estimate_remaining_iterations()
│
├── patterns.py                # DETERMINISTIC PATTERN DETECTION
│   ├── detect_constraint_violations()
│   ├── detect_gradient_noise()
│   ├── detect_cycling()
│   └── detect_trust_region_issues()
│
├── comparison.py              # DETERMINISTIC COMPARISON
│   ├── compare_runs()
│   ├── rank_algorithms()
│   └── compute_relative_efficiency()
│
├── visualization.py           # PLOTTING
│   ├── plot_convergence()
│   ├── plot_comparison()
│   └── plot_constraint_satisfaction()
│
└── ai_analysis.py            # AI REASONING LAYER
    ├── ai_analyze()
    ├── _build_analysis_prompt()
    ├── _format_metrics_for_llm()
    └── _parse_analysis_response()
```

### Key Design: Two-Layer Architecture

#### **Layer 1: Deterministic Core** (Always available)

```python
# paola/analysis/metrics.py

def compute_metrics(run: RunRecord) -> Dict[str, Any]:
    """
    Compute all deterministic metrics for a run.

    Foundation for both display and AI analysis.
    Fast (milliseconds), free, reproducible.

    Args:
        run: Optimization run record

    Returns:
        {
            "convergence": {
                "rate": float,              # (f_n - f_n-1) / f_n-1
                "is_stalled": bool,         # Rate < threshold for N iters
                "improvement_last_10": float,
                "iterations_total": int,
                "estimated_remaining": int,  # Extrapolation
            },
            "gradient": {
                "norm": float,
                "variance": float,          # Variance over last N gradients
                "quality": "good" | "noisy" | "flat",
                "method": "analytical" | "finite_difference",
            },
            "constraints": {
                "violations": [
                    {"name": str, "value": float, "bound": float, "margin": float}
                ],
                "active_count": int,
                "lagrange_multipliers": {name: float},
                "feasibility_trend": "improving" | "degrading" | "stable",
            },
            "efficiency": {
                "evaluations": int,
                "cache_hit_rate": float,
                "cost_per_improvement": float,  # Evals / improvement
                "wasted_evaluations": int,      # Evals with no improvement
            },
            "objective": {
                "current": float,
                "best": float,
                "worst": float,
                "improvement_from_start": float,
                "improvement_rate": float,
            },
            "algorithm_health": {
                "trust_region_size": float,     # For SLSQP, BFGS
                "step_size": float,
                "qp_solver_success_rate": float,
                "backtracking_frequency": float,
            }
        }

    Usage:
        # CLI: Fast display
        metrics = compute_metrics(run)
        display_table(metrics)

        # Agent: Monitor during optimization
        metrics = compute_metrics(run)
        if metrics["convergence"]["is_stalled"]:
            # Trigger adaptation
    """
```

#### **Layer 2: AI Reasoning** (Opt-in, strategic)

```python
# paola/analysis/ai_analysis.py

from typing import Literal

AnalysisFocus = Literal[
    "convergence",      # Why converging slowly/fast?
    "feasibility",      # Why violating constraints?
    "efficiency",       # Why so many evaluations?
    "algorithm",        # Should we switch algorithms?
    "overall",          # Holistic diagnosis
]

def ai_analyze(
    run: RunRecord,
    deterministic_metrics: Dict[str, Any],
    focus: AnalysisFocus = "overall",
    llm_model: str = "qwen-plus",
    force_reanalysis: bool = False
) -> Dict[str, Any]:
    """
    AI-powered strategic analysis of optimization run.

    Uses LLM to reason over deterministic metrics and provide
    strategic diagnosis + actionable recommendations.

    Cost: ~$0.02-0.05 per analysis
    Latency: 5-10 seconds
    Caching: Results stored with run (avoid redundant analysis)

    Args:
        run: Optimization run record
        deterministic_metrics: Pre-computed metrics from compute_metrics()
        focus: What aspect to analyze
        llm_model: LLM to use for reasoning
        force_reanalysis: Ignore cached insights

    Returns:
        {
            "diagnosis": str,           # What's happening (2-3 sentences)
            "root_cause": str,          # Why it's happening (1-2 sentences)
            "confidence": "low" | "medium" | "high",
            "evidence": [str],          # Supporting evidence from metrics
            "recommendations": [
                {
                    "action": str,      # Tool name (e.g., "constraint_adjust")
                    "args": dict,       # Tool arguments
                    "rationale": str,   # Why this helps
                    "priority": int,    # Execution order (1=first)
                    "expected_impact": str,  # What we expect to change
                }
            ],
            "metadata": {
                "model": str,
                "timestamp": str,
                "focus": str,
                "cost_estimate": float,
            }
        }

    Example:
        # Agent workflow
        metrics = compute_metrics(run)

        if metrics["convergence"]["is_stalled"]:
            # Deterministic signals issue, ask AI for strategy
            insights = ai_analyze(run, metrics, focus="convergence")

            # Execute recommendations
            for rec in insights["recommendations"]:
                if rec["action"] == "constraint_adjust":
                    constraint_adjust_bounds(**rec["args"])

    Design Principles:
        1. Deterministic as foundation (AI builds on metrics)
        2. Opt-in (cost control)
        3. Cached (avoid redundant analysis)
        4. Structured output (agent can execute recommendations)
    """

    # Check cache
    if not force_reanalysis and run.ai_insights:
        cached = run.ai_insights
        if cached.get("focus") == focus and not _is_stale(cached):
            return cached

    # Build structured prompt
    prompt = _build_analysis_prompt(run, deterministic_metrics, focus)

    # Call LLM
    llm = initialize_llm(llm_model)
    response = llm.invoke([HumanMessage(content=prompt)])

    # Parse structured response (JSON)
    insights = _parse_analysis_response(response.content)

    # Add metadata
    insights["metadata"] = {
        "model": llm_model,
        "timestamp": datetime.now().isoformat(),
        "focus": focus,
        "cost_estimate": _estimate_cost(llm_model, prompt),
    }

    return insights


def _build_analysis_prompt(
    run: RunRecord,
    metrics: Dict[str, Any],
    focus: AnalysisFocus
) -> str:
    """
    Build structured prompt for AI analysis.

    Prompt structure:
    1. Problem context (objectives, constraints, variables)
    2. Algorithm information
    3. Current status (iterations, objective values)
    4. Deterministic metrics (formatted for readability)
    5. Recent iteration details (last 10 iterations)
    6. Analysis focus
    7. Output format (JSON schema)

    Returns:
        Structured prompt string
    """

    metrics_formatted = _format_metrics_for_llm(metrics)
    recent_iters = run.result_data.get("iterations", [])[-10:]

    prompt = f"""You are an expert optimization analyst. Analyze this run and provide strategic recommendations.

PROBLEM FORMULATION:
- Objectives: {_format_objectives(run)}
- Variables: {run.problem_name} dimensions
- Constraints: {_format_constraints(run)}

ALGORITHM: {run.algorithm}

CURRENT STATUS:
- Total iterations: {run.n_evaluations}
- Current objective: {metrics['objective']['current']:.6f}
- Best objective: {metrics['objective']['best']:.6f}
- Improvement from start: {metrics['objective']['improvement_from_start']:.2%}

DETERMINISTIC METRICS:
{metrics_formatted}

RECENT ITERATION DETAILS (last 10):
{_format_recent_iterations(recent_iters)}

ANALYSIS FOCUS: {focus}

YOUR TASK:
1. Diagnose what's happening in this optimization
2. Identify root cause of current behavior
3. Provide concrete, actionable recommendations

Output MUST be valid JSON:
{{
    "diagnosis": "Brief explanation of current state (2-3 sentences)",
    "root_cause": "Why this is happening (1-2 sentences)",
    "confidence": "low|medium|high",
    "evidence": [
        "Metric or pattern that supports diagnosis",
        "Another piece of evidence"
    ],
    "recommendations": [
        {{
            "action": "constraint_adjust|optimizer_restart|algorithm_switch|gradient_method_change",
            "args": {{"param": "value"}},
            "rationale": "Why this action will help",
            "priority": 1,
            "expected_impact": "What should change"
        }}
    ]
}}

IMPORTANT:
- Base diagnosis on provided metrics (evidence-driven)
- Recommendations must be specific (e.g., "tighten CL to 0.52" not "improve constraints")
- action field must be a tool name that agent can call
- Provide 1-3 recommendations ordered by priority
"""
    return prompt
```

### Tool Exposure

```python
# paola/tools/analysis.py

@tool
def analyze_convergence(run_id: int, platform: OptimizationPlatform) -> Dict[str, Any]:
    """
    Deterministic convergence analysis (fast, free).

    Use this for quick checks during optimization.
    """
    from paola.analysis import compute_metrics

    run = platform.load_run(run_id)
    metrics = compute_metrics(run)
    return metrics["convergence"]


@tool
def analyze_run_with_ai(
    run_id: int,
    platform: OptimizationPlatform,
    focus: str = "overall"
) -> Dict[str, Any]:
    """
    AI-powered strategic analysis (costs ~$0.02-0.05).

    Use when deterministic metrics show issues and you need
    strategic advice (should I restart? switch algorithms?).

    Returns structured recommendations that agent can execute.
    """
    from paola.analysis import compute_metrics, ai_analyze

    run = platform.load_run(run_id)
    metrics = compute_metrics(run)
    insights = ai_analyze(run, metrics, focus=focus)

    return insights
```

### CLI Integration

```python
# paola/cli/commands.py

def handle_show(self, run_id: int):
    """Show deterministic metrics (instant, free)."""
    from paola.analysis import compute_metrics

    run = self.platform.load_run(run_id)
    metrics = compute_metrics(run)

    self._display_metrics(run, metrics)


def handle_analyze(self, run_id: int, focus: str = "overall"):
    """AI-powered analysis (costs money, opt-in)."""
    from paola.analysis import compute_metrics, ai_analyze

    run = self.platform.load_run(run_id)

    # Show deterministic metrics first (instant preview)
    self.console.print("\n[cyan]Computing metrics...[/cyan]")
    metrics = compute_metrics(run)
    self._display_metrics(run, metrics)

    # Check cache
    if run.ai_insights:
        age = _compute_age(run.ai_insights["metadata"]["timestamp"])
        self.console.print(f"\n[dim]Found cached AI analysis ({age})[/dim]")
        if self._confirm("Use cached?"):
            self._display_ai_insights(run.ai_insights)
            return

    # Confirm cost
    if not self._confirm("⚠ AI analysis costs ~$0.02-0.05. Continue?"):
        return

    # Run AI analysis
    with self.console.status("[dim]Analyzing with AI...[/dim]"):
        insights = ai_analyze(run, metrics, focus=focus)

    self._display_ai_insights(insights)
```

---

## Module 3: Knowledge (`paola/knowledge/`)

### Responsibility
- Store agent-curated insights from past optimizations
- Retrieve relevant knowledge for new optimizations (RAG)
- Enable warm-starting and strategy reuse
- Support organizational learning

### Structure
```
paola/knowledge/
├── __init__.py
│   └── KnowledgeBase (main API)
│
├── knowledge_base.py        # KnowledgeBase class
├── embedding.py             # Problem signature → vector
├── retrieval.py             # Similarity search
├── extraction.py            # Insight extraction prompts
│
└── storage/
    ├── memory_store.py      # In-memory (for testing)
    ├── file_store.py        # JSON-based
    └── vector_store.py      # Future: Chromadb/FAISS
```

### Key Classes

#### **KnowledgeBase**
```python
class KnowledgeBase:
    """
    RAG-based knowledge accumulation from past optimizations.

    Agent writes insights after analyzing successful runs.
    Agent retrieves insights when starting similar optimizations.
    """

    def __init__(self, storage: KnowledgeStorage, embedding_model: str = "simple"):
        """
        Initialize knowledge base.

        Args:
            storage: Knowledge storage backend
            embedding_model: "simple" (hand-crafted) or "semantic" (sentence-transformers)
        """
        self.storage = storage
        self.embedder = create_embedder(embedding_model)

    def store_insight(
        self,
        problem_signature: Dict[str, Any],
        insight: str,
        recommendations: List[Dict[str, Any]],
        evidence_runs: List[int],
        confidence: str = "medium"
    ) -> str:
        """
        Store agent-curated insight.

        Args:
            problem_signature: Problem characteristics
                {
                    "problem_type": "nonlinear",
                    "smooth": True,
                    "convex": False,
                    "dimension": 10,
                    "constraints": {"count": 5, "types": ["inequality"]},
                    "objective_type": "minimize"
                }
            insight: What the agent learned (textual)
            recommendations: What worked well
            evidence_runs: Run IDs that support this insight
            confidence: How confident in this insight

        Returns:
            insight_id: UUID for this insight

        Example:
            kb.store_insight(
                problem_signature={
                    "problem_type": "nonlinear",
                    "smooth": True,
                    "dimension": 10,
                },
                insight="SLSQP converges 2x faster with tightened constraints",
                recommendations=[
                    {
                        "action": "constraint_adjust",
                        "args": {"tighten_by": 0.04},
                        "success_rate": 0.85
                    }
                ],
                evidence_runs=[12, 15, 18],
                confidence="high"
            )
        """

    def retrieve_insights(
        self,
        problem_signature: Dict[str, Any],
        top_k: int = 5,
        min_confidence: str = "low"
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant insights for a problem.

        Uses problem signature similarity (embedding-based).

        Args:
            problem_signature: Target problem characteristics
            top_k: Number of insights to retrieve
            min_confidence: Minimum confidence level

        Returns:
            [
                {
                    "insight_id": str,
                    "insight": str,
                    "recommendations": [...],
                    "evidence_runs": [...],
                    "confidence": str,
                    "similarity": float,  # 0-1
                    "problem_signature": {...},
                }
            ]

        Example:
            # Agent starting new optimization
            insights = kb.retrieve_insights(
                problem_signature={
                    "problem_type": "nonlinear",
                    "smooth": True,
                    "dimension": 10,
                },
                top_k=3
            )

            # Agent uses insights to warm-start
            for insight in insights:
                if insight["similarity"] > 0.8:
                    # High confidence, apply recommendations
                    apply_recommendations(insight["recommendations"])
        """

    def get_insight(self, insight_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve specific insight by ID."""

    def list_insights(
        self,
        problem_type: Optional[str] = None,
        min_confidence: str = "low",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List insights with optional filters."""
```

#### **Embedding Strategy**

```python
# paola/knowledge/embedding.py

def create_embedder(model: str) -> ProblemEmbedder:
    """
    Create problem signature embedder.

    Args:
        model: "simple" or "semantic"

    Returns:
        ProblemEmbedder instance
    """
    if model == "simple":
        return HandCraftedEmbedder()
    elif model == "semantic":
        return SemanticEmbedder()
    else:
        raise ValueError(f"Unknown embedding model: {model}")


class HandCraftedEmbedder:
    """
    Hand-crafted feature embeddings (Phase 1).

    Simple, interpretable, no external dependencies.
    """

    def embed(self, problem_signature: Dict[str, Any]) -> np.ndarray:
        """
        Convert problem signature to feature vector.

        Features:
        - Dimension (normalized)
        - Problem type (one-hot: linear, quadratic, nonlinear)
        - Smoothness (binary)
        - Convexity (binary)
        - Constraint count (normalized)
        - Constraint types (one-hot: equality, inequality)

        Returns:
            Feature vector (length: ~10-15)
        """

    def similarity(self, sig1: Dict, sig2: Dict) -> float:
        """Cosine similarity between problem signatures."""


class SemanticEmbedder:
    """
    Semantic embeddings using sentence-transformers (Phase 2).

    Embeds textual description of problem + numerical features.
    More expressive but requires model download.
    """

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed(self, problem_signature: Dict[str, Any]) -> np.ndarray:
        """
        Hybrid embedding: textual + numerical.

        1. Convert signature to text description
        2. Embed text with sentence-transformers
        3. Concatenate with hand-crafted features

        Returns:
            Hybrid vector (length: ~400)
        """
```

### Tool Exposure

```python
# paola/tools/knowledge.py

@tool
def store_optimization_insight(
    problem_signature: Dict[str, Any],
    insight: str,
    recommendations: List[Dict[str, Any]],
    evidence_runs: List[int],
    knowledge_base: KnowledgeBase
) -> Dict[str, Any]:
    """
    Store what you learned from this optimization.

    Use this after successful optimization to record insights
    for future similar problems.

    Example:
        # Agent after successful run
        store_optimization_insight(
            problem_signature={
                "problem_type": "nonlinear",
                "smooth": True,
                "dimension": 10,
            },
            insight="Tightening CL constraint by 4% prevented boundary stalling",
            recommendations=[
                {
                    "action": "constraint_adjust",
                    "args": {"constraint_id": "CL", "tighten_by": 0.04}
                }
            ],
            evidence_runs=[12]
        )
    """


@tool
def retrieve_optimization_knowledge(
    problem_signature: Dict[str, Any],
    knowledge_base: KnowledgeBase,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Retrieve insights from similar past optimizations.

    Use this when starting new optimization to warm-start
    with proven strategies.

    Returns:
        {
            "insights": [
                {
                    "insight": str,
                    "recommendations": [...],
                    "similarity": float,
                    "evidence_runs": [...]
                }
            ],
            "count": int
        }

    Example:
        # Agent starting new optimization
        knowledge = retrieve_optimization_knowledge(
            problem_signature={
                "problem_type": "nonlinear",
                "smooth": True,
                "dimension": 10,
            },
            top_k=3
        )

        # Agent reasons about applicability
        if knowledge["insights"]:
            # Consider applying recommendations
    """
```

### CLI Integration

```python
# paola/cli/commands.py

def handle_knowledge(self):
    """Show stored knowledge."""
    insights = self.knowledge_base.list_insights(limit=20)

    if not insights:
        self.console.print("\n[dim]No knowledge stored yet[/dim]\n")
        return

    table = Table(title="Knowledge Base")
    table.add_column("ID", style="cyan")
    table.add_column("Insight")
    table.add_column("Problem Type")
    table.add_column("Confidence")
    table.add_column("Evidence")

    for insight in insights:
        table.add_row(
            insight["insight_id"][:8],
            insight["insight"][:50] + "...",
            insight["problem_signature"].get("problem_type", "N/A"),
            insight["confidence"],
            str(len(insight["evidence_runs"]))
        )

    self.console.print(table)


def handle_knowledge_show(self, insight_id: str):
    """Show detailed insight."""
    insight = self.knowledge_base.get_insight(insight_id)

    if not insight:
        self.console.print(f"[red]Insight {insight_id} not found[/red]")
        return

    # Display full insight with recommendations
    self._display_insight(insight)
```

---

## Module 4: Agent + Tools (`paola/agent/`, `paola/tools/`)

### Responsibility
- Orchestrate optimization workflow (formulation → solve → analyze → react)
- Provide clean ReAct implementation
- Expose platform capabilities as tools
- Keep logic minimal (delegate to modules)

### Structure
```
paola/agent/
├── __init__.py
├── react_agent.py       # ReAct loop (cleaned up)
└── prompts.py          # Separated prompt building

paola/tools/
├── __init__.py
│   └── build_tools()   # Tool construction
│
├── formulation.py      # Problem definition tools
├── optimization.py     # Run optimization tools
├── analysis.py         # Wrap paola.analysis
└── knowledge.py        # Wrap paola.knowledge
```

### Tool Design Principle: Thin Wrappers

Tools should have MINIMAL logic - just wrap modules:

```python
# paola/tools/optimization.py

@tool
def run_optimization(
    problem_id: str,
    algorithm: str,
    run_id: int,
    platform: OptimizationPlatform
) -> Dict[str, Any]:
    """
    Run optimization algorithm.

    This is a THIN WRAPPER - delegates to scipy.
    No business logic here.
    """
    # Get run handle
    run = platform.get_run(run_id)

    # Get problem
    problem = platform.get_problem(problem_id)

    # Run scipy optimizer
    result = scipy.optimize.minimize(
        fun=problem.objective,
        x0=problem.initial_guess,
        method=algorithm,
        # Callback records iterations
        callback=lambda xk: run.record_iteration(xk, problem.objective(xk))
    )

    # Finalize run
    run.finalize(result)

    return {
        "success": result.success,
        "objective": result.fun,
        "iterations": result.nit,
        "message": result.message
    }
```

### Agent Initialization (Dependency Injection)

```python
# Main entry point

def main():
    # Initialize platform (foundation)
    storage = FileStorage(base_dir=".paola_runs")
    platform = OptimizationPlatform(storage=storage)

    # Initialize knowledge base
    knowledge_storage = FileKnowledgeStorage(base_dir=".paola_knowledge")
    knowledge_base = KnowledgeBase(storage=knowledge_storage, embedding_model="simple")

    # Build tools (pass dependencies)
    tools = build_tools(platform=platform, knowledge_base=knowledge_base)

    # Build agent
    callback_manager = CallbackManager()
    callback_manager.register(CLICallback())

    agent = build_aopt_agent(
        tools=tools,
        llm_model="qwen-plus",
        callback_manager=callback_manager
    )

    # Initialize CLI
    cli = CLI(agent=agent, platform=platform, knowledge_base=knowledge_base)
    cli.run()
```

---

## Part 3: Implementation Roadmap

### Phase 1: Data Platform Refactoring (2-3 days)

**Goal**: Unify `runs/` + `storage/` → `platform/`

**Tasks**:
1. Create `paola/platform/` module structure
2. Implement `OptimizationPlatform` class
3. Refactor classes:
   - `Run` (active handle)
   - `RunRecord` (storage model)
   - `Problem` (problem definition)
4. Migrate `FileStorage` to `platform/storage/`
5. Update all imports across codebase
6. Remove `runs/` directory
7. Update tests

**Success Criteria**:
- ✅ All existing tests pass
- ✅ No `RunManager` singleton (dependency injection)
- ✅ Clear separation: Run (active) vs RunRecord (data)

**Files Changed**: ~20-30 files

---

### Phase 2: Analysis Module Extraction (1-2 days)

**Goal**: Centralize analysis logic, add AI reasoning

**Tasks**:
1. Create `paola/analysis/` module structure
2. Extract functions from `observation_tools.py`:
   - `analyze_convergence` → `analysis/convergence.py`
   - Pattern detection → `analysis/patterns.py`
3. Implement `compute_metrics()` (unified deterministic)
4. Implement `ai_analyze()` (AI reasoning layer)
5. Create thin wrapper tools in `tools/analysis.py`
6. Update CLI commands to use `paola.analysis`
7. Remove duplicate logic from `observation_tools.py`

**Success Criteria**:
- ✅ CLI `/show` uses `compute_metrics()`
- ✅ New CLI `/analyze` uses `ai_analyze()`
- ✅ Agent can call both deterministic and AI analysis
- ✅ No duplicated analysis logic

**Files Changed**: ~15-20 files

---

### Phase 3: Knowledge Module Implementation (2-3 days)

**Goal**: Enable agent to store/retrieve insights

**Tasks**:
1. Create `paola/knowledge/` module structure
2. Implement `KnowledgeBase` class (dict-based storage initially)
3. Implement `HandCraftedEmbedder` (simple feature vectors)
4. Implement file-based knowledge storage (JSON)
5. Create tools:
   - `store_optimization_insight`
   - `retrieve_optimization_knowledge`
6. Add CLI commands:
   - `/knowledge` - list insights
   - `/knowledge show <id>` - detailed view
7. Add agent prompt guidance for knowledge usage

**Success Criteria**:
- ✅ Agent can store insights (manually triggered)
- ✅ Agent can retrieve insights (similarity-based)
- ✅ CLI can inspect knowledge base
- ✅ Knowledge persists across sessions

**Files Changed**: ~10-15 files

---

### Phase 4: Agent Polish (1 day)

**Goal**: Clean up agent implementation

**Tasks**:
1. Complete context update logic (react_agent.py:516)
2. Move prompt building to `prompts.py`
3. Simplify ReAct loop (remove redundant code)
4. Improve error messages
5. Add agent reasoning examples to prompts

**Success Criteria**:
- ✅ No TODO comments in agent code
- ✅ Prompts separated from logic
- ✅ Agent context properly maintained

**Files Changed**: 2-3 files

---

### Phase 5: Integration & Testing (2-3 days)

**Goal**: Ensure everything works together

**Tasks**:
1. Update all tests for new architecture
2. Add module-level tests:
   - `test_platform.py`
   - `test_analysis.py`
   - `test_knowledge.py`
3. Integration test: Full optimization workflow
4. Performance testing (1000-iteration run)
5. Update documentation:
   - Architecture diagrams
   - API reference
   - Examples

**Success Criteria**:
- ✅ All tests pass
- ✅ Full workflow works end-to-end
- ✅ Documentation updated

---

## Part 4: Migration Guide

### For Existing Code

**Before Refactoring**:
```python
# Tool implementation
from ..runs import RunManager

@tool
def start_optimization_run(...):
    manager = RunManager()  # Singleton
    run = manager.create_run(...)
    return {"run_id": run.run_id}
```

**After Refactoring**:
```python
# Tool implementation
from paola.platform import OptimizationPlatform

def create_start_optimization_run_tool(platform: OptimizationPlatform):
    @tool
    def start_optimization_run(...):
        run = platform.create_run(...)  # Dependency injection
        return {"run_id": run.run_id}

    return start_optimization_run
```

### For CLI

**Before**:
```python
# CLI initialization
class AgenticOptREPL:
    def __init__(self, llm_model: str = "qwen-flash", storage: StorageBackend = None):
        self.storage = storage or FileStorage()
        run_manager = RunManager()
        run_manager.set_storage(self.storage)
```

**After**:
```python
# CLI initialization
class AgenticOptREPL:
    def __init__(
        self,
        agent: Any,
        platform: OptimizationPlatform,
        knowledge_base: KnowledgeBase,
        llm_model: str = "qwen-flash"
    ):
        self.agent = agent
        self.platform = platform
        self.knowledge_base = knowledge_base
        self.llm_model = llm_model
```

---

## Part 5: Testing Strategy

### Unit Tests (Module-Level)

**Data Platform**:
```python
# tests/test_platform.py

def test_platform_create_run():
    storage = MemoryStorage()  # In-memory for testing
    platform = OptimizationPlatform(storage)

    run = platform.create_run(
        problem_id="test_problem",
        problem_name="Test",
        algorithm="SLSQP"
    )

    assert run.run_id == 1
    assert run.algorithm == "SLSQP"

def test_platform_run_persistence():
    storage = FileStorage(base_dir="/tmp/test_paola")
    platform = OptimizationPlatform(storage)

    run = platform.create_run(...)
    run.record_iteration(x, obj)
    run.finalize(result)

    # Load from storage
    loaded = platform.load_run(run.run_id)
    assert loaded.objective_value == result.fun
```

**Analysis Module**:
```python
# tests/test_analysis.py

def test_compute_metrics():
    run = create_mock_run(iterations=50)
    metrics = compute_metrics(run)

    assert "convergence" in metrics
    assert "gradient" in metrics
    assert metrics["convergence"]["iterations_total"] == 50

def test_ai_analyze_caching():
    run = create_mock_run()
    metrics = compute_metrics(run)

    # First call
    insights1 = ai_analyze(run, metrics)

    # Second call should use cache
    insights2 = ai_analyze(run, metrics)

    assert insights1 == insights2  # Same result from cache
```

**Knowledge Module**:
```python
# tests/test_knowledge.py

def test_knowledge_store_retrieve():
    kb = KnowledgeBase(storage=MemoryKnowledgeStorage())

    # Store insight
    insight_id = kb.store_insight(
        problem_signature={"smooth": True, "dim": 10},
        insight="Test insight",
        recommendations=[],
        evidence_runs=[1]
    )

    # Retrieve
    insights = kb.retrieve_insights(
        problem_signature={"smooth": True, "dim": 10},
        top_k=1
    )

    assert len(insights) == 1
    assert insights[0]["insight"] == "Test insight"
```

### Integration Tests

```python
# tests/test_integration.py

def test_full_optimization_workflow():
    """
    End-to-end test: formulation → optimization → analysis → knowledge.
    """
    # Setup
    platform = OptimizationPlatform(storage=MemoryStorage())
    kb = KnowledgeBase(storage=MemoryKnowledgeStorage())
    tools = build_tools(platform=platform, knowledge_base=kb)
    agent = build_aopt_agent(tools=tools, llm_model="qwen-flash")

    # Agent workflow
    result = agent.invoke({
        "messages": [HumanMessage("Optimize 5D Rosenbrock with SLSQP")],
        "context": {},
        "done": False,
        "iteration": 0
    })

    # Verify run created
    runs = platform.load_all_runs()
    assert len(runs) == 1
    assert runs[0].success

    # Verify analysis available
    metrics = compute_metrics(runs[0])
    assert metrics["convergence"]["rate"] < 0.1  # Converged

    # Verify knowledge stored (if agent decided to)
    insights = kb.list_insights()
    # May or may not have insights depending on agent decision
```

---

## Part 6: Success Criteria

### Refactoring Complete When:

**Architecture**:
- ✅ 4 clean modules: Platform, Analysis, Knowledge, Agent+Tools
- ✅ No circular dependencies
- ✅ Clear separation of concerns
- ✅ No singleton pattern (dependency injection)

**Functionality**:
- ✅ All Phase 2 features still work (CLI, run tracking, plotting)
- ✅ Deterministic analysis available (instant, free)
- ✅ AI analysis available (opt-in, costs money)
- ✅ Knowledge base functional (store/retrieve)
- ✅ Agent can autonomously optimize + learn

**Quality**:
- ✅ All tests pass (>90% coverage)
- ✅ No duplicate logic
- ✅ Documentation updated
- ✅ Performance acceptable (<100ms for deterministic analysis)

**Extensibility**:
- ✅ Easy to add new storage backend (SQLite)
- ✅ Easy to add new analysis metrics
- ✅ Easy to swap embedding models
- ✅ Ready for Phase 3 (multi-run learning)

---

## Part 7: Risk Assessment

### Low Risk (Standard Refactoring)

**Data Platform**:
- Risk: Breaking existing run storage
- Mitigation: Keep same JSON schema, add migration script
- Rollback: Can deserialize old format

**Analysis Extraction**:
- Risk: Missing edge cases in metric computation
- Mitigation: Extensive unit tests, compare with old implementation
- Rollback: Keep old `observation_tools.py` temporarily

### Medium Risk (New Features)

**AI Analysis**:
- Risk: LLM produces invalid JSON
- Mitigation: Robust parsing with fallback, validation
- Rollback: Always have deterministic analysis as backup

**Knowledge Base**:
- Risk: Poor similarity matching (retrieves irrelevant insights)
- Mitigation: Start with hand-crafted features (interpretable), add metrics
- Rollback: Agent can ignore retrieved knowledge

### Mitigation Strategy

1. **Incremental Migration**: Refactor one module at a time
2. **Parallel Implementation**: Keep old code temporarily during migration
3. **Extensive Testing**: Unit + integration tests for each phase
4. **Versioning**: Git branches for each phase
5. **Rollback Plan**: Each phase can be reverted independently

---

## Part 8: Appendix

### A. Key API Examples

**Platform API**:
```python
platform = OptimizationPlatform(storage=FileStorage())

# Create run
run = platform.create_run("rosenbrock_10d", "Rosenbrock 10D", "SLSQP")

# Track progress
run.record_iteration(design=x, objective=f)

# Finalize
run.finalize(result)

# Query
all_runs = platform.load_all_runs()
slsqp_runs = platform.query_runs(algorithm="SLSQP")
```

**Analysis API**:
```python
from paola.analysis import compute_metrics, ai_analyze

# Deterministic (instant)
metrics = compute_metrics(run)
print(f"Convergence rate: {metrics['convergence']['rate']}")

# AI (costs money)
insights = ai_analyze(run, metrics, focus="convergence")
print(insights["diagnosis"])
for rec in insights["recommendations"]:
    execute_tool(rec["action"], rec["args"])
```

**Knowledge API**:
```python
from paola.knowledge import KnowledgeBase

kb = KnowledgeBase(storage=FileKnowledgeStorage())

# Store
kb.store_insight(
    problem_signature={"smooth": True, "dim": 10},
    insight="SLSQP with tightened constraints converged 2x faster",
    recommendations=[{"action": "constraint_adjust", "args": {...}}],
    evidence_runs=[12, 15]
)

# Retrieve
insights = kb.retrieve_insights(
    problem_signature={"smooth": True, "dim": 10},
    top_k=3
)
```

### B. File Size Estimates

**After Refactoring**:
```
paola/platform/      ~800 lines
paola/analysis/      ~1200 lines (600 deterministic + 600 AI)
paola/knowledge/     ~600 lines
paola/agent/         ~400 lines (cleaned up)
paola/tools/         ~400 lines (thin wrappers)
paola/cli/           ~600 lines
paola/callbacks/     ~300 lines

Total: ~4300 lines (vs current ~3500 lines)
```

Growth is from new features (AI analysis, knowledge), not bloat.

### C. Performance Targets

**Deterministic Analysis**:
- Target: <100ms for 1000-iteration run
- Method: Pure NumPy/Python, no LLM calls

**AI Analysis**:
- Target: 5-10 seconds (LLM latency)
- Caching: <10ms for cached insights

**Knowledge Retrieval**:
- Target: <100ms for 100 insights (in-memory)
- Future: <500ms for 10,000 insights (vector DB)

---

## Summary

This refactoring transforms PAOLA from a working prototype into a **production-ready platform** with:

1. **Solid Foundation**: Data Platform manages all run lifecycle
2. **Intelligent Analysis**: Deterministic metrics + AI reasoning
3. **Organizational Learning**: Knowledge accumulation via RAG
4. **Clean Orchestration**: Agent delegates to modules

**Timeline**: 6-8 days refactoring + 2-3 weeks polish

**Outcome**: First release ready with all 4 core capabilities:
- ✅ Agent-driven optimization workflow
- ✅ Excellent data management
- ✅ AI-powered analysis and learning
- ✅ Polished CLI interface

Ready to begin implementation?
