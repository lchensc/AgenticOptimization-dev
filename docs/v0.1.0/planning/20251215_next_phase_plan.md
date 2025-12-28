# PAOLA Next Phase Implementation Plan

**Date**: December 15, 2025
**Current Version**: v0.2.0 (Session-based architecture)
**Purpose**: Plan the next implementation steps after completing v0.2.0

---

## 1. Current State Assessment

### What's Complete (v0.2.0)

| Component | Status | Notes |
|-----------|--------|-------|
| Session-based architecture | ✅ Complete | Session/Run separation, polymorphic components |
| Optimizer backends | ✅ Complete | SciPy, IPOPT, Optuna with unified interface |
| Evaluator registration | ✅ Complete | FoundryEvaluator, NLPEvaluator, storage |
| NLP problem construction | ✅ Complete | create_nlp_problem, constraints, bounds |
| Tools | ✅ Complete | Session, optimization, evaluator, observation, gate |
| CLI | ✅ Complete | Conversational interface, /sessions, /plot, /analyze |
| Analysis | ✅ Complete | Metrics computation, AI-powered analysis |
| Agent | ✅ Complete | ReAct + Conversational, multi-provider LLM |

### What's Missing

| Component | Status | Gap |
|-----------|--------|-----|
| Knowledge Base | 🟡 Skeleton | `retrieve_insights()` not implemented, no RAG |
| Strategic Multi-Run | 🟡 Architecture only | Agent doesn't auto-switch optimizers |
| Multi-Run Learning | 🔴 Not started | No learning from warm-start chains |
| Integration Tests | 🟡 Partial | Need end-to-end workflow tests |

---

## 2. Strategic Options

### Option A: Knowledge Base with RAG (Phase 3)
**Goal**: Enable organizational learning from past optimizations

- Implement embedding-based retrieval
- Connect agent to knowledge for warm-starting
- File-based persistence for insights

**Value**: "Platform that learns from every run"

### Option B: Strategic Multi-Run Agent Behavior
**Goal**: Agent autonomously composes multi-optimizer strategies

- Detect convergence patterns (stalled, local minimum)
- Auto-switch optimizers based on diagnosis
- Warm-start across optimizer families

**Value**: Core PAOLA differentiator - intelligent strategy composition

### Option C: Testing & Hardening
**Goal**: Production-ready robustness

- End-to-end integration tests
- Error handling and recovery
- Performance benchmarks

**Value**: Reliability for real deployments

---

## 3. Recommended Next Phase: Strategic Multi-Run Agent

**Why this first?**

1. **Core differentiator** - This IS the PAOLA vision (agent composes strategies)
2. **Architecture ready** - Session infrastructure exists, just needs behavior
3. **Visible impact** - Users see intelligent optimization, not just running SLSQP
4. **Foundation for learning** - Need to capture "what works" before building knowledge base

### Phase 3.1: Convergence Detection & Pattern Recognition

**Goal**: Agent can diagnose optimization state

**Implementation**:

1. **Convergence Observer Tool** (`tools/convergence_tools.py`)
   ```python
   @tool
   def diagnose_optimization_state(session_id: int) -> Dict[str, Any]:
       """
       Analyze current optimization state within session.

       Returns:
           state: "converging" | "stalled" | "diverging" | "early"
           confidence: 0.0-1.0
           evidence: List of observations
           suggestions: List of recommended actions
       """
   ```

2. **Stall Detection**
   - No improvement in last N iterations
   - Gradient norm stable but not small
   - Oscillating objective values

3. **Local Minimum Detection**
   - Small gradient norm
   - All nearby perturbations worse
   - Constraint binding pattern suggests boundary minimum

4. **Convergence Detection**
   - Gradient norm < tolerance
   - Objective improvement < tolerance
   - Constraint satisfaction < tolerance

### Phase 3.2: Optimizer Switching Logic

**Goal**: Agent decides when to switch optimizers

**Implementation**:

1. **Decision Framework** (not hardcoded rules!)
   - LLM receives: current state diagnosis, available optimizers, problem characteristics
   - LLM decides: continue, switch optimizer, or terminate
   - Decision is recorded in `PaolaDecision`

2. **Switching Scenarios**:

   | Diagnosis | Recommended Action |
   |-----------|-------------------|
   | Stalled at local minimum | Try CMA-ES or global optimizer |
   | Converged but uncertain | Run different optimizer to verify |
   | Bayesian found good region | Switch to gradient for refinement |
   | High dimension struggling | Try quasi-Newton (L-BFGS-B) |
   | Constraints violated | Try IPOPT or penalty method |

3. **System Prompt Addition** (minimal, per The Paola Principle):
   ```
   You can run multiple optimizers in a session. If optimization stalls:
   - Use diagnose_optimization_state to understand why
   - Consider switching optimizers with warm_start from best design
   - Record your reasoning - this helps future optimizations
   ```

### Phase 3.3: Cross-Family Warm-Starting

**Goal**: Seamlessly warm-start between optimizer families

**Implementation**:

1. **Warm-Start Adapters**:
   - Gradient → Bayesian: Use best design as seeded trial
   - Bayesian → Gradient: Use best trial as x0
   - Gradient → CMA-ES: Use best design as mean, estimate sigma from recent step sizes
   - CMA-ES → Gradient: Use final mean as x0

2. **Initialization Strategy Extension**:
   ```python
   init_strategy: str
   # Existing: "center", "random", "warm_start"
   # New: "warm_start_cross_family"  # Adapts between families
   ```

3. **Adaptation Logic in `run_optimization`**:
   ```python
   if init_strategy == "warm_start" and session.runs:
       best_run = session.get_best_run()
       source_family = best_run.optimizer_family
       target_family = COMPONENT_REGISTRY.get_family(optimizer)

       if source_family != target_family:
           x0 = adapt_warm_start(best_run, target_family)
       else:
           x0 = np.array(best_run.best_design)
   ```

### Phase 3.4: Decision Recording & Analysis

**Goal**: Capture strategic decisions for learning

**Implementation**:

1. **Enhanced `PaolaDecision`**:
   - Already defined in schema
   - Need tool to record decisions: `record_optimization_decision()`

2. **Decision Types**:
   - `start_session`: Why this problem formulation
   - `select_optimizer`: Why this optimizer
   - `switch_optimizer`: Why switching (most important!)
   - `terminate`: Why stopping now

3. **Metrics at Decision**:
   - Current best objective
   - Improvement rate (last 10 iterations)
   - Gradient norm (if available)
   - Constraint violation
   - Wall time used

---

## 4. Implementation Plan

### Week 1: Convergence Detection

**Files to create/modify**:
- `paola/tools/convergence_tools.py` (new)
- `paola/tools/__init__.py` (add exports)

**Deliverables**:
- `diagnose_optimization_state` tool
- Unit tests for stall/convergence detection

### Week 2: Optimizer Switching

**Files to modify**:
- `paola/tools/session_tools.py` (add decision recording)
- `paola/tools/optimization_tools.py` (cross-family warm-start)
- `paola/agent/prompts/optimization.py` (minimal additions)

**Deliverables**:
- Cross-family warm-start working
- Agent can switch optimizers with warm-start

### Week 3: Integration & Testing

**Files to create**:
- `tests/test_multi_run_session.py`
- `tests/test_strategic_agent.py`

**Deliverables**:
- End-to-end multi-run session test
- Agent switching test (Optuna → SLSQP → CMA-ES)

### Week 4: CLI & Demo

**Files to modify**:
- `paola/cli/commands.py` (show decisions in /show)
- `paola/cli/repl.py` (if needed)

**Deliverables**:
- `/show <session_id>` shows strategic decisions
- Demo script showing multi-optimizer session

---

## 5. Success Criteria

### Minimum Viable

1. ✅ Agent can diagnose "stalled" optimization
2. ✅ Agent switches from gradient to CMA-ES on stall
3. ✅ Warm-start works across optimizer families
4. ✅ Decisions recorded in session

### Stretch Goals

1. ✅ Agent verifies convergence with second optimizer
2. ✅ Agent explains decisions in natural language
3. ✅ Multi-run session shows improvement over single-run

---

## 6. Dependencies & Risks

### Dependencies
- None - all infrastructure exists

### Risks
1. **LLM decision quality** - May need prompt tuning
   - Mitigation: Start with simple patterns, iterate

2. **Cross-family warm-start quality** - May lose progress
   - Mitigation: Test on benchmarks first

3. **Agent verbosity** - May over-switch
   - Mitigation: Add minimum iteration requirement before switching

---

## 7. After Phase 3

Once strategic multi-run is working, the natural next step is **Knowledge Base** (Phase 4):

- Capture successful strategies in knowledge base
- Retrieve similar problems for warm-starting
- Learn which optimizer combinations work for which problem types

This builds directly on Phase 3's decision recording.

---

## 8. Alternative Paths

If Phase 3 proves too complex, consider:

### Phase 3-Lite: Manual Multi-Run
- User explicitly requests optimizer switches
- Agent handles warm-start mechanics
- No automatic diagnosis

This is lower value but lower risk.

---

**Document Status**: Approved for Implementation
**Next Action**: Start with convergence detection tools
