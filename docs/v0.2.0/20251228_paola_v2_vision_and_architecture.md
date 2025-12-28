# Paola v0.2.0: Vision and Architecture

**Date**: 2025-12-28
**Status**: Design Document
**Purpose**: First-principles redesign of Paola as an AI-centric optimization platform

---

## Executive Summary

Paola v0.2.0 represents a fundamental rethinking of the platform based on the core insight:

> **Paola = Human Optimization Expert + Unified Database**

This document captures the analysis of what makes Paola unique compared to general coding agents, and the architectural principles for v0.2.0.

---

## Part 1: The Core Innovation

### What Paola Is NOT

Paola is NOT "an LLM that writes optimization scripts."

Any general coding agent (Claude, GPT-4, Cursor) can:
- Write `scipy.optimize.minimize()` code
- Read optimizer documentation
- Debug errors
- Try different methods

This is **commodity functionality**.

### What Paola IS

Paola IS "an optimization campaign manager with persistent memory and active monitoring."

**The unique value:**

| Capability | General Coding Agent | Paola |
|------------|---------------------|-------|
| Write scipy code | ✓ | ✓ |
| Cache expensive evaluations | ✗ | ✓ |
| Track multi-run campaigns | ✗ | ✓ |
| Warm-start from past runs | ✗ | ✓ |
| Learn from past campaigns | ✗ | ✓ |
| Monitor and adapt mid-run | ✗ | ✓ |
| Budget-aware decisions | ✗ | ✓ |
| Know when to switch strategies | ✗ | ✓ |

---

## Part 2: The Five Pillars of Paola's Uniqueness

### Pillar 1: Campaign Management (Not Scripts)

**General agent thinking:**
```
write script → run → done
```

**Paola thinking:**
```
plan campaign → run node → observe → decide → run node → ... → finalize
```

Real optimization isn't "run once and done." It's:
1. Try global exploration → found promising region
2. Warm-start local refinement → improved
3. Stuck at local min → try escape strategy
4. Compare branches → select best
5. Final polish → done

The **Graph structure** enables this. It tracks:
- Multi-run progress
- Warm-starting from any node
- Branching and comparing
- Resume and continue

**This is NOT over-engineering. This is the core differentiator.**

### Pillar 2: Evaluation Economy

**General agent thinking:**
```
Evaluation is free, retry is cheap
```

**Paola thinking:**
```
Every evaluation costs $100-$10,000 (CFD, FEA)
Never waste one
```

Engineering optimization is fundamentally different:
- Each function evaluation might cost $100-$10,000
- You can't just "run it again" to debug
- Caching is CRITICAL, not optional

**Mandatory features:**
- Every evaluation cached and logged
- Budget-aware decisions
- Cost-benefit reasoning built in

### Pillar 3: Organizational Memory

**General agent thinking:**
```
Fresh context every session
```

**Paola thinking:**
```
"I remember that a problem like this succeeded with IPOPT + warm-start"
```

An optimization expert doesn't start fresh each time. They remember:
- "Last time I had a 100D aerodynamic problem, IPOPT with scaling worked"
- "SLSQP tends to fail on problems with tight constraints"
- "For this client's problems, we always need to scale by 1e-3"

The **Foundry** provides persistent, queryable memory that:
- Stores past campaigns with full context
- Enables similarity-based retrieval
- Accumulates knowledge over time

### Pillar 4: Active Monitoring

**General agent thinking:**
```
Fire code → wait → get result
```

**Paola thinking:**
```
Fire → Monitor → Detect stall → Diagnose → Adapt → Continue
```

During optimization, Paola should:
- Watch convergence in real-time
- Detect problems (stall, divergence, oscillation)
- Diagnose root causes
- Adapt strategy mid-campaign
- Explain what's happening and why

### Pillar 5: Expert Judgment

**General agent thinking:**
```
"Here's scipy code that minimizes your function"
```

**Paola thinking:**
```
"Given the problem characteristics (100D, constrained, aerodynamic)
 and past experience (graph_42 succeeded with IPOPT),
 I'll use IPOPT with adaptive mu strategy.
 If stuck, I'll warm-start CMA-ES to escape local minimum.
 This pattern succeeded 4/5 times on similar problems."
```

The "last mile" of optimization (getting from obj=0.1 to obj=0.05) requires:
- Trying multiple refinement strategies
- Warm-starting from different points
- Understanding WHY you're stuck
- Knowing optimizer-specific tricks

---

## Part 3: Architectural Decisions for v0.2.0

### Decision 1: Code Execution Over Tool Abstraction

**v0.1.0 approach (rejected for optimizer calls):**
```
LLM → run_optimization(optimizer="scipy:SLSQP", config={...}) → structured result
```

**v0.2.0 approach:**
```
LLM → writes Python optimization code → executes → sees real output/errors
```

**Rationale:**
- LLMs are trained on scipy, optuna, cyipopt code
- The tools layer is an abstraction that HIDES what LLMs already know
- Tools constrain flexibility
- LLMs can debug their own code naturally

**What tools should provide (infrastructure only):**
- Evaluation caching (expensive simulations)
- Campaign tracking (Graph management)
- Knowledge storage (Foundry)
- Active monitoring hooks

**What tools should NOT provide:**
- Optimizer call abstraction (let LLM write code)
- Result formatting (let LLM see raw output)

### Decision 2: Foundry as Semantic Knowledge Base

**v0.1.0 Foundry (log-oriented):**
```
GraphRecord:
  nodes: [{optimizer, config, best_objective, n_evals}]
  pattern: "chain"
  final_objective: 0.05
```

Stores WHAT happened, not WHY.

**v0.2.0 Foundry (knowledge-oriented):**

**Layer 1: Problem Signatures**
```python
ProblemSignature:
  # Structural
  n_dimensions: int
  n_constraints: int
  bounds_structure: str  # "uniform", "varied", "mixed"

  # Semantic
  domain: str           # "aerodynamic", "structural", "thermal"
  characteristics: []   # ["multimodal", "noisy", "expensive", "smooth"]

  # For similarity search
  embedding: vector

  similarity(other) -> float  # 0.0 to 1.0
```

**Layer 2: Strategy Patterns (Transferable)**
```python
StrategyPattern:
  name: str                    # "global-then-local", "multistart"
  applicable_when: []          # ["multimodal", "unknown_landscape"]
  not_applicable_when: []      # ["very_expensive", "smooth_convex"]

  # Learned from experience
  success_rate: float
  typical_improvement: float
```

**Layer 3: Causal Annotations (WHY)**
```python
Experience:
  problem_signature: ProblemSignature
  strategy: StrategyPattern
  outcome: "success" | "partial" | "failure"

  # The reasoning (CRITICAL)
  why_this_strategy: str       # Agent's reasoning at decision time
  why_it_worked_or_failed: str # Post-hoc analysis
  key_insight: str             # Transferable lesson
```

**Layer 4: Configuration Templates (Proven)**
```python
ConfigTemplate:
  optimizer: str
  config: dict
  applicable_for: ProblemSignature (pattern)
  success_count: int
  source_experiences: []
```

### Decision 3: AI-Centric Design Principles

**Principle 1: Code is the Native Interface**
LLMs are trained on code. Let them work in code.

**Principle 2: Minimal Abstraction, Maximum Infrastructure**
Don't abstract optimizer calls. DO provide:
- Evaluation caching
- Knowledge storage
- Campaign tracking
- Active monitoring

**Principle 3: Semantic Memory Over Logs**
Store knowledge useful for reasoning, not just records.

**Principle 4: Self-Improving**
The system should get smarter without human intervention:
- Patterns extracted from experiences
- Success rates updated
- Templates refined
- Failure modes catalogued

---

## Part 4: v0.2.0 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PAOLA'S UNIQUE VALUE                         │
│                   (What general agents lack)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  CAMPAIGN MANAGEMENT (Graph)                            │   │
│  │  - Track multi-run progress                             │   │
│  │  - Warm-start, branch, compare                          │   │
│  │  - Resume anytime                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  EVALUATION ECONOMY (Cache)                             │   │
│  │  - Every eval cached and logged                         │   │
│  │  - Budget-aware decisions                               │   │
│  │  - Never waste expensive simulations                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ORGANIZATIONAL MEMORY (Foundry)                        │   │
│  │  - Problem signatures with similarity                   │   │
│  │  - Strategy patterns (transferable)                     │   │
│  │  - Causal annotations (WHY)                            │   │
│  │  - Configuration templates (proven)                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ACTIVE MONITORING                                       │   │
│  │  - Real-time progress observation                       │   │
│  │  - Stall/divergence detection                           │   │
│  │  - Mid-run adaptation                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │         CODE EXECUTION LAYER            │
        │   (LLM writes optimization code)        │
        │   This part is NOT the differentiator   │
        └─────────────────────────────────────────┘
```

---

## Part 5: What Changes from v0.1.0

| Aspect | v0.1.0 | v0.2.0 |
|--------|--------|--------|
| Optimizer calls | Tools (abstracted) | LLM writes Python |
| Database | Execution logs | Semantic knowledge base |
| Learning | Storage only | Pattern extraction + refinement |
| Monitoring | Passive logging | Active intervention |
| Flexibility | Constrained by tool API | Unlimited (it's code) |
| Memory | Query by ID | Query by similarity |

---

## Part 6: Implementation Roadmap

### Phase 1: Infrastructure Refactoring
- Simplify tools to infrastructure-only (cache, storage, monitoring)
- Remove optimizer abstraction layer
- Enable LLM code execution with caching hooks

### Phase 2: Foundry Redesign
- Add ProblemSignature with embeddings
- Implement similarity search
- Add causal annotation storage
- Design configuration template system

### Phase 3: Campaign Protocol
- Implement 6-phase reasoning protocol
- Add mandatory history query
- Require evidence-based decisions
- Capture agent reasoning in Experience records

### Phase 4: Active Monitoring
- Real-time convergence watching
- Stall/divergence detection
- Mid-run adaptation triggers
- Diagnostic explanations

### Phase 5: Learning System
- Pattern extraction from experiences
- Success rate tracking by problem type
- Configuration template refinement
- Failure mode cataloguing

---

## Summary

**Paola v0.2.0 Vision:**

1. **Campaign management** over scripts — Graph tracks multi-run strategies
2. **Evaluation economy** — Never waste expensive evals
3. **Organizational memory** — Semantic knowledge base that learns
4. **Active monitoring** — Observe and adapt mid-run
5. **Code-native execution** — LLM writes optimization code directly

**The core principle:**

> A general coding agent can write `scipy.optimize.minimize()`.
> Only Paola can run a 5-node campaign, warm-starting from past experience,
> monitoring for stalls, adapting mid-run, and accumulating knowledge for next time.

This is what makes Paola = Expert + Database.

---

## Part 7: Code Execution Design (Detailed)

### The Challenge

Moving from tool abstraction to code execution raises three critical questions:

1. **Problem Definition**: How do we handle heterogeneous external evaluators (ANSYS, SU2, GEANT4, ML training)?
2. **Caching**: How do we inject caching hooks into external evaluators?
3. **Recording**: How do we capture structured information when LLM writes free-form code?

### 7.1 Problem as Contract, Evaluator as Registered Entity

**The separation is key:**
- **Problem** = mathematical structure (dimensions, bounds, constraints)
- **Evaluator** = computational implementation (how to call external code)

```python
# PROBLEM: Defines the mathematical contract
class Problem:
    id: int
    name: str
    n_dimensions: int
    bounds: List[Tuple[float, float]]
    constraints: List[Constraint]
    evaluator_id: str          # Reference to registered evaluator
    domain_hint: str           # "aerodynamic", "structural", etc.
    signature: ProblemSignature  # For similarity search

# EVALUATOR: Registered separately, handles heterogeneity
class Evaluator:
    id: str
    type: "python" | "script" | "api" | "hpc"
    callable: Callable | str   # Function or command
    n_inputs: int
    n_outputs: int
    has_gradient: bool
    cost_estimate: float       # seconds per evaluation
```

**Why this separation?**
- Problem class stays clean and unified
- Evaluator registration handles heterogeneous external codes
- Caching is injected at the evaluator level
- Problem metadata enables Foundry storage and similarity search

### 7.2 Evaluator Registration for Different Types

**Python function:**
```python
paola.register_evaluator(
    id="my_func",
    type="python",
    callable=my_python_function,
    n_inputs=50,
    n_outputs=1,
    has_gradient=True
)
```

**External script:**
```python
paola.register_evaluator(
    id="su2_cfd",
    type="script",
    command="python run_su2.py --input {input_file} --output {output_file}",
    input_format="json",
    output_format="json",
    cost_estimate=3600  # 1 hour per eval
)
```

**HPC job:**
```python
paola.register_evaluator(
    id="ansys_fea",
    type="hpc",
    submit_command="sbatch run_ansys.sh {input_file}",
    poll_interval=60,
    output_parser=parse_ansys_result,
    cost_estimate=14400  # 4 hours per eval
)
```

### 7.3 Caching via Wrapped Evaluators

Paola provides cached wrappers automatically:

```python
class CachedEvaluator:
    def __init__(self, evaluator, problem_id, cache):
        self.evaluator = evaluator
        self.problem_id = problem_id
        self.cache = cache

    def __call__(self, x):
        # 1. Check cache
        key = self._hash(x)
        if cached := self.cache.get(self.problem_id, key):
            return cached  # Free! No expensive computation

        # 2. Call actual evaluator (expensive)
        result = self.evaluator(x)

        # 3. Store in cache
        self.cache.store(self.problem_id, key, result, cost=self.evaluator.cost)
        return result

# LLM uses:
objective = paola.get_objective(problem_id=7)  # Returns CachedEvaluator
```

### 7.4 Structured Hooks + Free Code

**The key insight**: We need structured information capture, but don't want to constrain code flexibility.

**Solution**: Define mandatory hooks that capture critical information. Between hooks, LLM has complete freedom.

```python
# ============================================================
# MANDATORY STRUCTURED HOOKS (capture critical information)
# ============================================================

# 1. START NODE - captures intent
node = paola.start_node(
    graph_id=42,
    optimizer="scipy:SLSQP",
    config={'ftol': 1e-6, 'maxiter': 200},
    reasoning="Problem is 50D constrained. Graph 38 succeeded with SLSQP."
)

# 2. GET OBJECTIVE - enables caching
objective = paola.get_objective(problem_id=7)

# ============================================================
# FREE CODE (LLM has complete freedom here)
# ============================================================

x0 = np.zeros(50)  # Any initialization
result = scipy.optimize.minimize(
    objective,
    x0,
    method='SLSQP',
    bounds=[(-5, 5)] * 50,
    options={'ftol': 1e-6, 'maxiter': 200},
    callback=node.callback  # Optional: progress tracking
)

# ============================================================
# MANDATORY STRUCTURED HOOK (capture result)
# ============================================================

# 3. COMPLETE NODE - captures outcome
node.complete(result)
```

### 7.5 What Gets Recorded

| Hook | Information Captured |
|------|---------------------|
| `start_node()` | graph_id, node_id, optimizer, config, reasoning, timestamp |
| `get_objective()` | All evaluations via cache (x, f, cost, timestamp) |
| `node.callback()` | Iteration history, convergence progress |
| `node.complete()` | Final result, best_x, best_objective, wall_time, termination |
| **Code itself** | Stored verbatim for reproduction |

### 7.6 The Contract

**The hooks are NOT optional.** The agent MUST use:

1. `paola.start_node()` — Before optimization (captures intent and config)
2. `paola.get_objective()` — For objective function (enables caching)
3. `node.complete()` — After optimization (captures result)

**Between these hooks, LLM has complete freedom to write any optimization code.**

### 7.7 Full Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROBLEM (Contract)                           │
│  id, name, n_dimensions, bounds, constraints, evaluator_id     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATOR (Registered)                       │
│  id, type, callable/command, n_inputs, n_outputs, cost         │
│                                                                 │
│  Types: python | script | api | hpc                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CACHED EVALUATOR (Wrapper)                   │
│  Wraps evaluator with: cache check → call → cache store        │
│                                                                 │
│  paola.get_objective(problem_id) → CachedEvaluator             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM-GENERATED CODE                           │
│                                                                 │
│  node = paola.start_node(...)     # ← Structured hook          │
│  objective = paola.get_objective(...)  # ← Caching hook        │
│                                                                 │
│  # -------- FREE CODE ZONE --------                            │
│  result = scipy.optimize.minimize(objective, x0, ...)          │
│  # -------- END FREE CODE ---------                            │
│                                                                 │
│  node.complete(result)            # ← Structured hook          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GRAPH (Recorded in Foundry)                  │
│                                                                 │
│  Node record contains:                                         │
│    - optimizer: "scipy:SLSQP"                                  │
│    - config: {'ftol': 1e-6}                                    │
│    - reasoning: "Based on graph_38..."                         │
│    - iterations: [...]                                          │
│    - evaluations: [...] (from cache)                           │
│    - result: {best_x, best_objective, ...}                     │
│    - code: "..." (verbatim for reproduction)                   │
└─────────────────────────────────────────────────────────────────┘
```

### 7.8 Trade-off Analysis

| Aspect | Pure Tools (v0.1.0) | Structured Hooks (v0.2.0) |
|--------|---------------------|---------------------------|
| Information capture | Automatic (all JSON) | Via hooks (must use them) |
| Flexibility | Constrained by tool API | Full Python freedom |
| Caching | Built into tool | Via `get_objective()` |
| Reasoning capture | Tool parameter | `start_node(reasoning=...)` |
| Code storage | N/A | Store verbatim |
| Reproducibility | Replay tool calls | Re-execute code |
| Debugging | Abstracted errors | Real Python errors |

**The trade-off**: We lose automatic structured capture, but gain full code flexibility. The hooks are the bridge that makes this work.
