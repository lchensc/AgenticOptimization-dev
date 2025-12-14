# PAOLA Next Phase Development Plan

**Date**: December 13, 2025
**Current Version**: 0.1.0 (Phases 1-5 Complete)
**Status**: Production-Ready for Basic Workflows

---

## Executive Summary

PAOLA has achieved a **solid foundation** with all core infrastructure complete:
- ✅ Data foundation (Foundry)
- ✅ Agent autonomy (ReAct agent with 12 tools)
- ✅ Analysis (deterministic + AI)
- ✅ Interactive CLI
- ✅ Knowledge skeleton (ready for data)

**Next phase**: Transform from "basic workflow automation" to "intelligent optimization system" by implementing the **3 strategic tracks** from our architecture:

1. **Track 1: Engineering Workflows** - Real-world integration (CFD/FEA)
2. **Track 2: Knowledge Accumulation** - Learning from experience
3. **Track 3: Strategic Adaptation** - Autonomous problem-solving

---

## Part 1: Current State Analysis

### What Works Now (Phases 1-5)

| Component | Status | Capabilities |
|-----------|--------|--------------|
| **Foundry** | ✅ Production | Single source of truth, Run/RunRecord, FileStorage, lineage tracking |
| **Agent** | ✅ Production | ReAct loop, 12 tools, LangGraph-based, multi-model (Qwen/Claude/OpenAI) |
| **Tools** | ✅ Production | Problem creation, run management, SciPy optimization, metrics, knowledge placeholders |
| **Analysis** | ✅ Production | 5 metric categories (convergence, gradient, constraints, efficiency, objective), AI diagnosis |
| **Knowledge** | ✅ Skeleton | Interface defined, MemoryKnowledgeStorage, agent tools (placeholders) |
| **CLI** | ✅ Production | Interactive REPL, 12+ commands, Rich output, model switching |
| **Testing** | ✅ Production | 30+ tests across 5 suites, all passing |

### Verified Workflows

**Test Results** (from test_end_to_end_workflow.py):
- ✅ Rosenbrock 5D optimization with SLSQP: converged to 4.5e-07 in 47 iterations
- ✅ Multi-algorithm comparison (SLSQP vs Nelder-Mead)
- ✅ Storage persistence and loading
- ✅ Metrics computation across all categories
- ✅ CLI commands (/runs, /show, /compare)

**What Users Can Do Today**:
```bash
# Launch CLI
python -m paola.cli

# Natural language optimization
paola> optimize 10D Rosenbrock with SLSQP

# Analyze results
paola> /show 1
paola> /analyze 1 convergence

# Compare algorithms
paola> compare SLSQP and BFGS on this problem
```

---

## Part 2: Gap Analysis (Current vs. Architecture Vision)

### From architecture_crystallized.md Section 7: Future Extensions

| Track | Vision | Current State | Gap |
|-------|--------|---------------|-----|
| **Track 1: Engineering Workflows** | SU2/OpenFOAM integration, multi-fidelity, budget tracking | Only analytical functions (Rosenbrock, Sphere) | No real workflows |
| **Track 2: Knowledge Base** | RAG with vector DB, warm-starting, problem signatures, 50+ runs | Skeleton interface only | No RAG, no data |
| **Track 3: Strategic Adaptation** | Constraint management, gradient switching, Bayesian control | Basic metrics only | No adaptation tools |

### Specific Missing Capabilities (from design docs)

**From CLAUDE.md and agent_controlled_optimization.md**:

1. **Strategic Adaptation Mechanisms**:
   - ❌ Constraint feasibility management (tightening bounds)
   - ❌ Gradient method switching (adjoint ↔ finite-difference)
   - ❌ Bayesian exploration control (acquisition function tuning)
   - ❌ Convergence verification (high-fidelity validation)

2. **Tool Primitives** (from Section 7 of agent_controlled_optimization.md):
   - ❌ `optimizer_checkpoint() / restore()` - Optimizer state saving
   - ❌ `workflow_execute()` - CFD/FEA simulation
   - ❌ `gradient_compute()` - Method switching
   - ❌ `constraint_adjust_bounds()` - Dynamic constraint adjustment
   - ❌ `budget_remaining()` - Cost tracking
   - ❌ `knowledge_apply()` - Warm-starting from RAG

3. **Observation Tools** (Section 15 of agent_controlled_optimization.md):
   - ❌ `detect_infeasibility_pattern()` - Repeated constraint violations
   - ❌ `detect_gradient_noise()` - Variance-based detection
   - ❌ `check_constraint_activity()` - Active set analysis
   - ❌ `estimate_convergence_horizon()` - Remaining iterations

4. **Evaluation Cache**:
   - ❌ Cache integration with real workflows (only analytical functions now)
   - ❌ Design-key hashing with tolerance
   - ❌ Cost savings tracking

5. **Knowledge Base**:
   - ❌ Problem signature extraction
   - ❌ RAG-based retrieval (vector DB: Chroma/FAISS)
   - ❌ Warm-starting from similar problems
   - ❌ Agentic learning (agent stores insights automatically)
   - ❌ Knowledge corpus (need 50+ runs)

---

## Part 3: First Principles Analysis

### Core Value Proposition (from agentic_optimization_vision.md)

**PAOLA's unique value** comes from 3 innovations:
1. **Agent autonomy** - No fixed loops, agent composes strategies
2. **Knowledge accumulation** - Learns from every run via RAG
3. **Strategic adaptation** - Detects and resolves issues automatically

**Current state**: Only #1 is implemented (basic autonomy with tools)
**Missing**: #2 (no learning) and #3 (no adaptation)

### Priority Ordering from First Principles

**Question**: What enables the most value, fastest?

**Analysis**:

**Option A: Engineering Workflows First (Track 1)**
- **Pros**: Demonstrates real-world applicability, immediate user value
- **Cons**: Complex (CFD setup, mesh generation, job scheduling), high time investment (4-5 weeks), doesn't unlock unique value propositions #2 and #3

**Option B: Knowledge Base First (Track 2)**
- **Pros**: Unlocks value proposition #2 (learning), enables warm-starting
- **Cons**: Requires 50+ runs for meaningful RAG, chicken-and-egg problem (need workflows to generate data)

**Option C: Strategic Adaptation First (Track 3)**
- **Pros**: Unlocks value proposition #3 (intelligent adaptation), works with current analytical functions, demonstrates agent intelligence, relatively quick (3-4 weeks)
- **Cons**: Full value requires real workflows (to show constraint management, gradient noise)

**First Principles Conclusion**:

**Priority 1: Track 3 (Strategic Adaptation)** - Because:
1. Demonstrates unique value (intelligence) without needing complex workflows
2. Can be validated on analytical functions (Rosenbrock with modified constraints)
3. Builds agent capabilities that apply to ALL future problems
4. Relatively quick implementation (3-4 weeks)
5. Unblocks agent's decision-making power

**Priority 2: Track 1 (Engineering Workflows)** - Because:
6. Needed for real-world validation
7. Generates data for knowledge base
8. Demonstrates applicability

**Priority 3: Track 2 (Knowledge Base)** - Because:
9. Requires data from Track 1 (50+ runs)
10. Requires mature workflows to accumulate meaningful patterns
11. Can leverage existing runs retroactively

**Revised Ordering**: 3 → 1 → 2 (Adaptation → Engineering → Knowledge)

This ordering:
- Gets to unique value fastest (agent intelligence)
- Validates on simple problems first (lower risk)
- Generates data while building workflows (efficiency)
- Implements knowledge base when there's real data (practicality)

---

## Part 4: Development Plan - Phase 6-8

### Phase 6: Strategic Adaptation (Track 3)
**Duration**: 3-4 weeks
**Goal**: Agent autonomously detects and resolves optimization issues

#### 6.1 Enhanced Observation Tools (Week 1)

**New Tools** (4 tools):
```python
@tool
def detect_infeasibility_pattern(run_id: int) -> Dict[str, Any]:
    """
    Detect repeated constraint violations.

    Returns:
    - violated_constraints: List[str]
    - violation_count: int
    - severity: "low|medium|high"
    - recommendation: "tighten|relax|switch_algorithm"
    """

@tool
def detect_gradient_noise(run_id: int, window: int = 10) -> Dict[str, Any]:
    """
    Detect noisy gradients from variance.

    Returns:
    - gradient_variance: float
    - is_noisy: bool
    - confidence: float
    - recommendation: "switch_to_fd|increase_step|adjoint_ok"
    """

@tool
def check_constraint_activity(run_id: int) -> Dict[str, Any]:
    """
    Analyze active constraint set.

    Returns:
    - active_constraints: List[str]
    - near_active: List[str]  # Within 5% of bound
    - binding_iterations: Dict[str, int]  # How long constraint has been active
    """

@tool
def estimate_convergence_horizon(run_id: int) -> Dict[str, Any]:
    """
    Predict remaining iterations to convergence.

    Returns:
    - estimated_remaining: int
    - confidence: float
    - is_stalled: bool
    - recommended_action: str
    """
```

**Implementation**:
- Add to `paola/tools/observation_tools.py`
- Use existing metrics from `paola/analysis/metrics.py`
- Add statistical analysis (variance, moving averages)

#### 6.2 Constraint Management Tools (Week 2)

**New Tools** (3 tools):
```python
@tool
def constraint_tighten(
    problem_id: str,
    constraint_id: str,
    tighten_by: float = 0.02
) -> Dict[str, Any]:
    """
    Tighten constraint bounds to force feasibility.

    Example: CL >= 0.5 becomes CL >= 0.51 (2% tighter)

    Returns:
    - new_bound: float
    - old_bound: float
    - problem_id_new: str  # New problem ID with tightened constraint
    """

@tool
def constraint_relax(
    problem_id: str,
    constraint_id: str,
    relax_by: float = 0.05
) -> Dict[str, Any]:
    """
    Relax constraint bounds when over-constrained.
    """

@tool
def optimizer_restart_with_constraints(
    run_id: int,
    constraint_adjustments: Dict[str, float]
) -> Dict[str, Any]:
    """
    Restart optimization from current best with adjusted constraints.

    Uses current best design as initial point.
    Creates new run with lineage to parent run.
    """
```

**Implementation**:
- Extend `paola/foundry/problem.py` to support constraint modification
- Add problem versioning (rosenbrock_v1 → rosenbrock_v2)
- Track lineage (run 2 derives from run 1 with tightened CL)

#### 6.3 Gradient Method Switching (Week 3)

**New Tools** (2 tools):
```python
@tool
def gradient_compute_fd(
    problem_id: str,
    design: List[float],
    step_size: float = 1e-6
) -> Dict[str, Any]:
    """
    Compute gradient using finite-difference.

    Returns:
    - gradient: List[float]
    - method: "finite_difference"
    - cost: float  # CPU hours
    """

@tool
def gradient_compute_adjoint(
    problem_id: str,
    design: List[float]
) -> Dict[str, Any]:
    """
    Compute gradient using adjoint method.

    Returns:
    - gradient: List[float]
    - method: "adjoint"
    - cost: float  # CPU hours
    """
```

**Note**: For Phase 6, these will use finite-difference on analytical functions. Real adjoint integration comes in Phase 7 with CFD.

**Implementation**:
- Add gradient tools to `paola/tools/optimizer_tools.py`
- Integrate with SciPy optimizers (override gradient function)
- Add gradient history tracking to Run

#### 6.4 Agent Prompts for Adaptation (Week 4)

**Update** `paola/agent/prompts.py`:

Add strategic reasoning guidance:
```python
STRATEGIC_ADAPTATION_GUIDANCE = """
When optimization encounters issues, use these patterns:

1. Repeated Constraint Violations:
   - Use detect_infeasibility_pattern()
   - If CL constraint violated 5+ times:
     → constraint_tighten(constraint_id="CL", tighten_by=0.02)
     → optimizer_restart_with_constraints()

2. Noisy Gradients:
   - Use detect_gradient_noise()
   - If gradient variance > 0.1:
     → Switch to gradient_compute_fd() with larger step
     → Or restart with gradient-free method (Nelder-Mead)

3. Stalled Convergence:
   - Use estimate_convergence_horizon()
   - If stalled (no improvement for 20 iterations):
     → optimizer_restart() with different initial point
     → Or switch algorithm (SLSQP → BFGS)

4. Active Constraints:
   - Use check_constraint_activity()
   - If constraint active for 30+ iterations:
     → Likely at optimum, verify with high-fidelity
"""
```

**Testing**:
- Create test problems that trigger each pattern
- Verify agent autonomously detects and resolves
- Example test: "Optimize Rosenbrock with CL >= 0.5 (impossible constraint)"

#### 6.5 Validation and Testing

**Test Cases**:
1. **Infeasible Constraint**: Rosenbrock with impossible constraint, agent tightens and converges
2. **Gradient Noise**: Add noise to Rosenbrock gradient, agent switches to FD
3. **Stalled Convergence**: Bad initial point, agent restarts with better point
4. **Active Constraint**: Multi-constraint problem, agent identifies binding constraints

**Success Metrics**:
- ✅ Agent resolves infeasibility without human intervention
- ✅ Agent switches gradient methods when needed
- ✅ Agent restarts intelligently (not randomly)
- ✅ All adaptations logged to foundry with lineage

---

### Phase 7: Engineering Workflows (Track 1)
**Duration**: 4-5 weeks
**Goal**: Real CFD/FEA integration with multi-fidelity

#### 7.1 Workflow Abstraction (Week 1)

**Design**:
```python
# paola/workflows/base.py
class WorkflowBackend(ABC):
    """Abstract interface for simulation workflows."""

    @abstractmethod
    def execute(
        self,
        design: np.ndarray,
        fidelity: str = "medium"
    ) -> WorkflowResult:
        """Execute simulation and return objectives, constraints."""

    @abstractmethod
    def compute_gradient(
        self,
        design: np.ndarray,
        method: str = "adjoint"
    ) -> np.ndarray:
        """Compute gradient (adjoint or finite-difference)."""
```

**Fidelity Levels**:
- `low`: Coarse mesh, fast (10 min)
- `medium`: Standard mesh, accurate (2 hours)
- `high`: Fine mesh, verification (10 hours)

#### 7.2 SU2 Integration (Weeks 2-3)

**Components**:
1. **Mesh parameterization** (FFD boxes or CST)
2. **SU2 job execution** (subprocess with timeout)
3. **Result parsing** (forces.csv, adjoint gradients)
4. **Error handling** (divergence, mesh failure)

**Example Workflow**:
```python
# paola/workflows/su2_workflow.py
class SU2AirfoilWorkflow(WorkflowBackend):
    """Transonic airfoil optimization."""

    def execute(self, design, fidelity="medium"):
        # 1. Generate mesh (FFD deformation)
        mesh = self.parameterization.apply(design)

        # 2. Run SU2
        result = run_su2(mesh, config=self.get_config(fidelity))

        # 3. Parse results
        return WorkflowResult(
            objectives={"drag": result.CD},
            constraints={"lift": result.CL - 0.5}
        )

    def compute_gradient(self, design, method="adjoint"):
        if method == "adjoint":
            return self._run_su2_adjoint(design)
        else:
            return self._finite_difference(design)
```

**Testing**:
- RAE2822 airfoil (transonic, well-documented)
- Verify: CL, CD match literature
- Verify: Adjoint gradient matches finite-difference

#### 7.3 Multi-Fidelity Support (Week 4)

**Agent Tools**:
```python
@tool
def workflow_execute(
    problem_id: str,
    design: List[float],
    fidelity: str = "medium"
) -> Dict[str, Any]:
    """
    Execute workflow at specified fidelity.

    Fidelity:
    - low: Fast, approximate (10x cheaper)
    - medium: Standard, accurate
    - high: Slow, verification (10x more expensive)
    """

@tool
def recommend_fidelity(run_id: int) -> Dict[str, Any]:
    """
    Recommend fidelity based on optimization stage.

    Returns:
    - recommended: "low|medium|high"
    - rationale: str

    Strategy:
    - Early exploration (iterations < 20): low
    - Refinement (20-50): medium
    - Verification (converged): high
    """
```

**Agent Strategy**:
- Start with low-fidelity (cheap exploration)
- Switch to medium when improving
- Verify final design with high-fidelity

#### 7.4 Budget Tracking (Week 5)

**Implementation**:
```python
# paola/foundry/budget.py
class Budget:
    """Track computational budget."""

    def __init__(self, total_cpu_hours: float):
        self.total = total_cpu_hours
        self.used = 0.0
        self.history = []

    def charge(self, cost: float, description: str):
        """Charge budget and log."""
        self.used += cost
        self.history.append({
            "timestamp": datetime.now(),
            "cost": cost,
            "description": description,
            "remaining": self.remaining()
        })

    def remaining(self) -> float:
        return self.total - self.used
```

**Agent Tool**:
```python
@tool
def budget_remaining() -> Dict[str, Any]:
    """
    Get remaining computational budget.

    Returns:
    - remaining_cpu_hours: float
    - percent_used: float
    - recommendation: str  # "continue|switch_to_low_fidelity|terminate"
    """
```

**Testing**:
- Set budget to 50 CPU hours
- Verify agent uses low-fidelity when budget < 20%
- Verify agent terminates gracefully when budget exhausted

---

### Phase 8: Knowledge Accumulation (Track 2)
**Duration**: 4-5 weeks
**Goal**: Platform learns from 50+ optimization runs

#### 8.1 Data Collection (Weeks 1-2)

**Prerequisite**: Have Phases 6-7 complete (adaptation + workflows)

**Strategy**:
1. Run 50+ optimizations with variety:
   - 10× Analytical (Rosenbrock, Sphere, Rastrigin, different dimensions)
   - 20× Airfoil (transonic, subsonic, different CL targets)
   - 10× Multi-objective (CL/CD tradeoff)
   - 10× Constrained (geometric, performance)

2. Capture metadata for each run:
   - Problem signature (dimension, constraints, physics)
   - Algorithm used (SLSQP, BFGS, Nelder-Mead)
   - Adaptations made (constraint tightening, gradient switching)
   - Outcome (success, iterations, cost)
   - Agent's reflection (what worked, what didn't)

**Implementation**:
```python
# paola/knowledge/insight.py
class OptimizationInsight:
    """Structured knowledge from successful run."""

    problem_signature: ProblemSignature
    strategy: OptimizationStrategy
    outcome: OptimizationOutcome
    narrative: str  # Agent's reflection

    created_at: datetime
    confidence: float  # How confident is this insight?
```

#### 8.2 Problem Signature Extraction (Week 3)

**Design**:
```python
# paola/knowledge/signature.py
class ProblemSignature:
    """Characteristics that define problem class."""

    dimension: int
    objective_count: int
    constraint_count: int
    constraint_types: List[str]  # ["equality", "inequality", "bounds"]

    physics_domain: Optional[str]  # "aerodynamics", "structures", "heat_transfer"
    regime: Optional[str]  # "transonic", "subsonic", "supersonic"

    problem_class: str  # "smooth_unconstrained", "nonlinear_constrained", etc.

    def similarity(self, other: "ProblemSignature") -> float:
        """
        Compute similarity score [0, 1].

        Weights:
        - Dimension match: 0.3
        - Constraint structure: 0.3
        - Physics domain: 0.2
        - Problem class: 0.2
        """
```

**Automatic Extraction**:
```python
def extract_signature(problem: Problem) -> ProblemSignature:
    """Extract signature from problem definition."""

    # Analyze mathematical structure
    dimension = len(problem.variables)

    # Classify problem (smooth, noisy, multi-modal, etc.)
    problem_class = classify_problem_structure(problem)

    # Extract physics (if specified in metadata)
    physics_domain = problem.metadata.get("physics_domain")

    return ProblemSignature(...)
```

#### 8.3 RAG Implementation (Week 4)

**Technology**:
- **Vector DB**: Chroma (lightweight, Python-native)
- **Embeddings**: OpenAI text-embedding-3-small or Qwen embeddings
- **Retrieval**: Top-k similarity search

**Implementation**:
```python
# paola/knowledge/rag.py
class RAGKnowledgeBase:
    """RAG-based knowledge retrieval."""

    def __init__(self, vector_db_path: str = ".paola_knowledge"):
        self.db = chromadb.PersistentClient(path=vector_db_path)
        self.collection = self.db.get_or_create_collection(
            name="optimization_insights",
            embedding_function=OpenAIEmbeddingFunction()
        )

    def store_insight(self, insight: OptimizationInsight):
        """
        Store insight with embeddings.

        Embedding created from:
        - Problem signature (dimension, constraints, physics)
        - Strategy description (algorithm, adaptations)
        - Narrative (agent's reflection)
        """

        # Create embedding text
        text = f"""
        Problem: {insight.problem_signature.physics_domain},
                 {insight.problem_signature.dimension}D,
                 {insight.problem_signature.constraint_count} constraints
        Strategy: {insight.strategy.algorithm},
                  adaptations: {insight.strategy.adaptations}
        Outcome: {insight.outcome.success},
                 {insight.outcome.iterations} iterations,
                 final objective {insight.outcome.final_value}
        Insight: {insight.narrative}
        """

        self.collection.add(
            documents=[text],
            metadatas=[insight.to_dict()],
            ids=[insight.id]
        )

    def retrieve_similar(
        self,
        signature: ProblemSignature,
        k: int = 5
    ) -> List[OptimizationInsight]:
        """
        Retrieve k most similar past optimizations.

        Returns insights ranked by similarity.
        """

        # Create query from signature
        query = signature.to_embedding_text()

        # Retrieve
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )

        # Convert to insights
        return [OptimizationInsight.from_dict(m) for m in results["metadatas"][0]]
```

#### 8.4 Warm-Starting (Week 5)

**Agent Tool**:
```python
@tool
def retrieve_optimization_knowledge(
    problem_id: str,
    k: int = 3
) -> Dict[str, Any]:
    """
    Retrieve knowledge from similar past optimizations.

    Returns top k insights with:
    - Similar problem characteristics
    - Successful strategies
    - Agent reflections

    Use this BEFORE starting optimization to warm-start.
    """

@tool
def apply_warm_start(
    problem_id: str,
    insight_id: str
) -> Dict[str, Any]:
    """
    Apply warm-start from retrieved insight.

    Actions:
    - Use proven algorithm from insight
    - Apply constraint adjustments proactively
    - Set initial point from similar problem
    """
```

**Agent Workflow**:
```
1. User: "Optimize transonic airfoil for minimum drag, CL >= 0.5"

2. Agent:
   a) create_benchmark_problem() or formulate_problem()
   b) retrieve_optimization_knowledge()  # NEW: Check knowledge base

3. Knowledge base returns:
   - "Similar problem (RAE2822, CL >= 0.5) succeeded with:"
     * Algorithm: SLSQP
     * Adaptation: Tightened CL constraint by 2% (agent learned this!)
     * Initial point: [0.5, 0.3, ...]
     * Outcome: 45 iterations, converged

4. Agent:
   c) apply_warm_start(insight_id="rae2822_run_42")
   d) start_optimization_run()
   e) run_scipy_optimization(
        algorithm="SLSQP",  # From knowledge
        initial_point=[0.5, 0.3, ...]  # From knowledge
        constraint_adjustments={"CL": 0.02}  # From knowledge
      )

5. Result: Converges in 30 iterations (33% faster!) ✓
```

**Testing**:
- Run same problem 5× without knowledge base (baseline)
- Run same problem 5× with knowledge base (warm-started)
- Verify: 30-40% fewer iterations on average

#### 8.5 Agentic Learning

**Agent Reflection After Each Run**:
```python
# In agent prompts (paola/agent/prompts.py)
POST_OPTIMIZATION_REFLECTION = """
After optimization completes, reflect on what you learned:

1. What worked well?
   - Which algorithm was effective?
   - Which adaptations helped?
   - What was the key insight?

2. What didn't work?
   - Which approaches failed?
   - What would you avoid next time?

3. Key insight (1 sentence):
   - Distill the most important lesson from this optimization

Use store_optimization_insight() to save this for future optimizations.
"""
```

**Example Agent Reflection**:
```
Agent: "I learned that for transonic airfoil optimization with CL constraints,
       the optimizer tends to undershoot CL by ~2%. Proactively tightening
       the CL constraint to 0.51 (instead of 0.5) led to faster convergence
       (30 iterations vs 45 baseline). This pattern likely applies to all
       lift-constrained aerodynamic optimizations."

[Stores to knowledge base with embedding]

Future optimization:
User: "Optimize another airfoil with CL >= 0.6"
Agent: [Retrieves past insight] "I'll tighten CL to 0.612 based on past experience"
```

---

## Part 5: Implementation Strategy

### Development Philosophy

**From CLAUDE.md Design Principles**:
1. **Agent Autonomy First** - Every new tool should enable agent decisions, not prescribe behavior
2. **Observable Everything** - All adaptations must emit events and log to foundry
3. **Strategic Restarts** - Adaptations are informed, not random experiments
4. **Cache Everything** - Real workflows are expensive, cache all evaluations
5. **Learn Continuously** - Every run should contribute to knowledge base

### Incremental Validation

**Each phase milestone**:
1. **Implement**: New tools + agent prompts
2. **Test**: Automated tests + manual CLI verification
3. **Validate**: Does agent use tools correctly? Does it solve problems autonomously?
4. **Document**: Update architecture docs with lessons learned
5. **Merge**: Only merge when all tests pass

### Risk Mitigation

**Potential Risks**:

1. **Risk**: Agent doesn't use new tools
   - **Mitigation**: Add explicit examples in prompts, test with capture callbacks

2. **Risk**: CFD integration is complex and fragile
   - **Mitigation**: Start with well-documented test case (RAE2822), containerize SU2

3. **Risk**: Knowledge base has insufficient data (< 50 runs)
   - **Mitigation**: Phase 8 AFTER Phase 7, generate data systematically

4. **Risk**: RAG embeddings don't capture optimization patterns well
   - **Mitigation**: Experiment with embedding strategies, add problem signature metadata

---

## Part 6: Success Metrics

### Phase 6 (Strategic Adaptation)
- ✅ Agent resolves infeasible constraint without human intervention (5/5 test cases)
- ✅ Agent switches gradient methods when detecting noise (variance > 0.1)
- ✅ Agent restarts intelligently when stalled (no improvement 20+ iterations)
- ✅ All adaptations logged with lineage to foundry

### Phase 7 (Engineering Workflows)
- ✅ RAE2822 optimization converges to known optimum (literature validation)
- ✅ Multi-fidelity strategy reduces cost by 5× (vs all-medium fidelity)
- ✅ Adjoint gradients match finite-difference within 5%
- ✅ Budget tracking prevents runaway costs

### Phase 8 (Knowledge Accumulation)
- ✅ 50+ runs in knowledge base (diverse problems)
- ✅ Warm-starting reduces iterations by 30% on similar problems
- ✅ Agent reflection generates meaningful insights (human-validated quality)
- ✅ RAG retrieval finds relevant past optimizations (top-3 similarity > 0.7)

### Overall Success (Phases 6-8)
- ✅ End-to-end workflow: User goal → Agent optimizes → Knowledge stored → Warm-start next time
- ✅ Agent demonstrates all 3 value propositions:
  1. Autonomy (composes strategies from tools)
  2. Learning (warm-starts from past runs)
  3. Adaptation (resolves issues automatically)

---

## Part 7: Timeline and Resources

### Estimated Timeline (10-12 weeks total)

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 6 (Adaptation) | 3-4 weeks | Agent autonomously resolves optimization issues |
| Phase 7 (Engineering) | 4-5 weeks | Real CFD workflows with multi-fidelity |
| Phase 8 (Knowledge) | 4-5 weeks | 50+ runs, RAG, warm-starting |

**Note**: Phases can overlap partially (e.g., start collecting data in Phase 6-7 for Phase 8)

### Resource Requirements

**Compute**:
- Phase 6: Local machine (analytical functions)
- Phase 7: HPC cluster for CFD (10-50 CPU hours per optimization)
- Phase 8: Embedding API costs ($0.10 per 1M tokens, ~$5-10 total)

**Data**:
- Phase 8: 50+ optimization runs (generate in Phases 6-7)

**External Dependencies**:
- SU2 (open-source CFD, already available)
- Chroma (vector DB, pip install)
- OpenAI/Qwen embeddings (API)

---

## Part 8: Revised Architecture Vision

### Updated Module Structure (Post-Phases 6-8)

```
paola/
├── foundry/              # ✅ Phase 1 (DONE)
│   ├── foundry.py
│   ├── run.py
│   ├── problem.py
│   ├── budget.py         # ← NEW (Phase 7)
│   └── storage/
├── agent/                # ✅ Phase 4 (DONE)
│   ├── react_agent.py
│   └── prompts.py        # ← UPDATED (Phases 6-8: adaptation guidance)
├── tools/                # ← EXPANDED
│   ├── run_tools.py      # ✅ DONE
│   ├── optimizer_tools.py # ✅ DONE
│   ├── evaluator_tools.py # ✅ DONE
│   ├── analysis.py       # ✅ DONE
│   ├── knowledge_tools.py # ✅ Skeleton → REAL (Phase 8)
│   ├── observation_tools.py # ← EXPANDED (Phase 6: 4 new tools)
│   ├── constraint_tools.py  # ← NEW (Phase 6)
│   ├── gradient_tools.py    # ← NEW (Phase 6)
│   ├── workflow_tools.py    # ← NEW (Phase 7)
│   └── budget_tools.py      # ← NEW (Phase 7)
├── workflows/            # ← NEW (Phase 7)
│   ├── base.py           # Abstract WorkflowBackend
│   ├── su2_workflow.py   # SU2 integration
│   └── analytical.py     # Move existing analytical functions here
├── knowledge/            # ✅ Skeleton → FULL (Phase 8)
│   ├── knowledge_base.py # ← UPDATED (RAG implementation)
│   ├── rag.py            # ← NEW (Chroma integration)
│   ├── signature.py      # ← NEW (Problem signature extraction)
│   ├── insight.py        # ← NEW (OptimizationInsight schema)
│   └── storage.py        # ← UPDATED (RAGKnowledgeStorage)
├── analysis/             # ✅ Phase 2 (DONE)
├── cli/                  # ✅ Phases 2-5 (DONE)
├── callbacks/            # ✅ Phase 1 (DONE)
└── backends/             # ✅ Phase 1 → MOVED to workflows/analytical.py
```

### Tool Count Evolution

| Phase | Tool Count | New Tools |
|-------|------------|-----------|
| Phase 5 (Current) | 12 tools | - |
| Phase 6 (Adaptation) | **21 tools** | +9 (observation, constraint, gradient) |
| Phase 7 (Engineering) | **24 tools** | +3 (workflow, budget, fidelity) |
| Phase 8 (Knowledge) | **27 tools** | +3 (knowledge retrieval updated to use RAG) |

---

## Part 9: Next Immediate Actions

### Week 1: Phase 6 Kickoff

**Day 1-2**: Enhanced Observation Tools
1. Create `paola/tools/observation_tools.py` (new file)
2. Implement 4 new tools:
   - `detect_infeasibility_pattern()`
   - `detect_gradient_noise()`
   - `check_constraint_activity()`
   - `estimate_convergence_horizon()`
3. Write tests (test_phase6_observation.py)

**Day 3-4**: Test on Analytical Functions
1. Create test problems that trigger each pattern:
   - Rosenbrock with impossible constraint (CL >= 10 when max is 5)
   - Rosenbrock with noisy gradient (add Gaussian noise)
   - Bad initial point that causes stalling
2. Verify detection tools work correctly

**Day 5**: Integrate with Agent
1. Update `paola/agent/prompts.py` with observation guidance
2. Test: Agent calls observation tools during optimization
3. Verify: Agent receives correct diagnostic information

---

## Part 10: Long-Term Vision (12+ months)

### Beyond Phase 8

**Phase 9: Advanced RAG** (2-3 months)
- Multi-modal knowledge (store CFD images, mesh quality)
- Agentic knowledge curation (agent reviews and refines insights)
- Knowledge graph (relationships between problems, algorithms, adaptations)

**Phase 10: Collaboration Features** (2-3 months)
- Multi-user workspace (shared foundry)
- Review & approval workflow
- Knowledge sharing across teams

**Phase 11: Enterprise Integration** (3-4 months)
- CAD connectors (CATIA, NX, SolidWorks)
- Pipeline builder (DAG workflows)
- SSO, multi-tenancy, RBAC

**Phase 12: Research Extensions** (ongoing)
- Multi-objective optimization (NSGA-II, MOEA/D)
- Robust optimization (uncertainty quantification)
- Topology optimization
- Deep learning surrogates

---

## Summary

**Current State**: Production-ready foundation (Phases 1-5 complete) ✅

**Next Phase Plan**:
1. **Phase 6 (3-4 weeks)**: Strategic Adaptation - Agent autonomously resolves issues
2. **Phase 7 (4-5 weeks)**: Engineering Workflows - Real CFD with multi-fidelity
3. **Phase 8 (4-5 weeks)**: Knowledge Accumulation - RAG-based learning from 50+ runs

**Ordering Rationale** (from first principles):
- Adaptation FIRST: Demonstrates unique intelligence value, works on simple problems
- Engineering SECOND: Generates real data, validates on complex problems
- Knowledge THIRD: Requires data from engineering workflows

**Success Criteria**:
- Agent demonstrates all 3 core value propositions (autonomy, learning, adaptation)
- 30% faster convergence with warm-starting
- 90% success rate on engineering problems (vs 50% baseline)
- Real users adopt PAOLA for production workflows

**Timeline**: 10-12 weeks to complete Phases 6-8

**Ready to start**: Week 1 begins with Phase 6 observation tools ✓

---

**Next Step**: Review this plan, get approval, then implement Phase 6 Week 1 (observation tools) 🚀
