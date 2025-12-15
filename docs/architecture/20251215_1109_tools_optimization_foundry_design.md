# Tools - Optimization - Foundry Architecture Design

**Status**: In Progress
**Date**: December 15, 2025
**Version**: 0.2.0

---

## 1. Problem Statement

The current tool architecture has several issues:

### 1.1 Information Duplication

```python
# Current: Bounds specified TWICE
create_nlp_problem(
    problem_id="wing_design",
    bounds=[[0, 15], [0.1, 0.5]],  # Here
    ...
)

run_scipy_optimization(
    problem_id="wing_design",
    bounds=[[0, 15], [0.1, 0.5]],  # And here again!
    ...
)
```

From mathematical optimization first principles, an NLP is:

```
minimize   f(x)
subject to g(x) ≤ 0
           h(x) = 0
           x_L ≤ x ≤ x_U    ← Bounds are part of the PROBLEM
```

**Bounds are intrinsic to the problem formulation, not the optimizer invocation.**

### 1.2 Tool Responsibility Confusion

Current tools mix concerns:
- `run_scipy_optimization` does: bounds parsing + algorithm config + execution + recording
- No clear separation between problem definition and optimization session

### 1.3 Large Variable Space Scalability

With 100+ variables, tool calls become unwieldy:
```python
run_scipy_optimization(
    bounds=[[-32.768, 32.768]] * 100,  # 200 numbers in tool call!
    initial_design=[0.0] * 100,         # Another 100 numbers!
)
```

---

## 2. The Paola Principle

### 2.1 Core Insight

**Initialization complexity is exactly what Paola should abstract away.**

Users come to Paola because they:
- Have an optimization problem to solve
- Don't want to become optimization experts
- Don't know (and shouldn't need to know) that SLSQP needs x0, CMA-ES needs mean+sigma, Optuna needs nothing

**Paola's value proposition**: "I know which optimizer needs what initialization, and I'll handle it for you."

### 2.2 What Users Care About

From first principles, users care about:

1. **Defining their problem** (what to optimize)
   - Objective function
   - Constraints
   - Variable bounds (the feasible region)

2. **Getting a good solution** (the result)

Users do **NOT** care about:
- Whether SLSQP needs an initial point
- What sigma means in CMA-ES
- Whether to use LHS or random sampling for NSGA-II
- How IPOPT's `mu_init` affects convergence

### 2.3 The Fundamental Shift

| Aspect | Traditional Approach | Paola Approach |
|--------|---------------------|----------------|
| Initial point | User specifies | Agent decides |
| Algorithm-specific config | User learns each | Agent knows all |
| Warm-starting | User manages | Agent checks history |
| Domain knowledge | User provides | Agent infers |
| Large vectors in API | Required | Never exposed |

**Initialization is agent intelligence, not user input.**

---

## 3. Design Principles

### 3.1 Single Source of Truth

**Problem owns mathematical data only.** No algorithm-specific details.

```
NLP Problem (in Foundry)
├── Dimension
├── Bounds (x_L, x_U)
├── Objective (evaluator_id, sense)
├── Constraints (evaluator_ids, types, values)
└── Domain hint (optional: "shape_optimization", "hyperparameter_tuning")

NOT in Problem:
├── Initial point (agent decides)
├── Algorithm parameters (in config)
└── Initialization strategy (agent decides)
```

### 3.2 Separation of Concerns

```
Problem Formulation  → What to optimize (mathematical)
Optimizer Config     → Algorithm parameters (technical)
Agent Intelligence   → How to initialize (expert knowledge)
Optimization Run     → Execute and record
```

### 3.3 Immutability (for now)

Problems are immutable once created. To change bounds or constraints:
- Create a new problem (possibly derived from existing)
- Reference the new problem in optimization

**Future**: Will revisit for adaptive problem modification during optimization.

---

## 4. Paola's Initialization Intelligence

### 4.1 Agent's Internal Capabilities

When `run_optimization` is called, Paola reasons about initialization:

```
┌─────────────────────────────────────────────────────────────────┐
│           PAOLA'S INITIALIZATION INTELLIGENCE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: Check for warm-start opportunities                     │
│    → Query: "Are there previous runs for this problem?"         │
│    → If yes: Use best solution from most successful run         │
│                                                                 │
│  Step 2: Infer domain from problem structure                    │
│    → FFD control points? → Shape optimization → zero init       │
│    → Neural network weights? → Xavier/He-like initialization    │
│    → General bounded NLP? → Center of bounds                    │
│                                                                 │
│  Step 3: Match algorithm requirements                           │
│    → Gradient-based (SLSQP, IPOPT): single point needed         │
│    → Population-based (NSGA-II): generate LHS population        │
│    → Bayesian (Optuna): let sampler handle (no init needed)     │
│    → CMA-ES: compute mean and sigma from bounds                 │
│                                                                 │
│  Step 4: Apply expert heuristics                                │
│    → Constrained problem? Prefer feasible starting point        │
│    → Previous run failed? Try different initialization          │
│    → Multi-modal suspected? Consider multiple restarts          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Algorithm-Specific Initialization Rules

| Algorithm Class | Paola's Default Strategy |
|-----------------|-------------------------|
| **Gradient-based** (SLSQP, L-BFGS-B, IPOPT) | Center of bounds, or warm-start if history exists |
| **Shape optimization** (any algorithm) | Zero (baseline geometry) |
| **Population-based** (NSGA-II, DE) | Latin Hypercube Sampling within bounds |
| **CMA-ES** | Mean = center of bounds, sigma = 0.25 × bound width |
| **Bayesian** (Optuna TPE) | Let sampler handle (random initial exploration) |
| **Interior-point** (IPOPT) | Center, with appropriate mu_init |

### 4.3 Warm-Start Logic

```
Paola's warm-start decision tree:

1. Query foundry.get_runs(problem_id)
2. If successful runs exist:
   a. Get best solution from most successful run
   b. Verify solution is within current bounds (may have changed)
   c. Use as initial point for gradient-based methods
   d. Use as seed for population-based methods
3. If no history or previous runs failed:
   a. Fall back to algorithm-specific default
   b. Log reasoning: "No warm-start available, using center of bounds"
```

### 4.4 This Solves the Large Variable Space Problem

**For bounds**: In problem definition (compact specification supported).

**For initial point**: User never specifies it.

Paola internally computes:
- `np.zeros(100)` for shape optimization
- `(lb + ub) / 2` for general problems
- `best_from_run_42` for warm-start

The 100-dimensional vector is **never in a tool call argument**.

---

## 5. Tool Hierarchy

### 5.1 Problem Formulation Tools (Foundry-centric)

These tools define WHAT to optimize. **No initialization parameters.**

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROBLEM FORMULATION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  create_nlp_problem                                             │
│    - problem_id: str                                            │
│    - objective_evaluator_id: str                                │
│    - objective_sense: "minimize" | "maximize"                   │
│    - bounds: BoundsSpec (see Section 6)                         │
│    - inequality_constraints: [{name, evaluator_id, type, value}]│
│    - equality_constraints: [{name, evaluator_id, value}]        │
│    - domain_hint: Optional["shape_optimization" | "general"]    │
│    - description: str                                           │
│    → Creates NLPProblem in Foundry                              │
│    → NO initial_point parameter (Paola decides)                 │
│                                                                 │
│  get_problem_info                                               │
│    - problem_id: str                                            │
│    → Returns complete problem specification                     │
│                                                                 │
│  derive_problem                                                 │
│    - source_problem_id: str                                     │
│    - new_problem_id: str                                        │
│    - bounds: Optional[BoundsSpec]  (override)                   │
│    - constraints: Optional[...]    (override)                   │
│    → Creates new problem derived from existing                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Paola's Configuration Intelligence

**The Paola Principle Extended**: Configuration complexity is agent intelligence, not user burden.

Research shows optimizer packages have overwhelming complexity:
- IPOPT: ~250 options across 22 categories
- SNOPT: ~100 options (scaling-sensitive)
- CMA-ES: ~50 options (sigma, restarts, etc.)
- Even "simple" SciPy: ~50 options across methods

**Most users only touch 3-5 options.** The knowledge of *which* options matter for *which* problems is Paola's core competence.

```
┌─────────────────────────────────────────────────────────────────┐
│           PAOLA'S CONFIGURATION INTELLIGENCE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Level 1: Problem Analysis                                      │
│    → Dimensions? Constraints? Gradients available?              │
│    → Smooth or noisy? Convex or multi-modal?                    │
│    → Computational budget?                                      │
│                                                                 │
│  Level 2: Algorithm Selection                                   │
│    → Match problem to algorithm family                          │
│    → Large constrained + gradients → IPOPT                      │
│    → Black-box expensive → Bayesian (Optuna)                    │
│    → Multi-modal → CMA-ES with restarts                         │
│                                                                 │
│  Level 3: Option Configuration                                  │
│    → Set problem-appropriate defaults                           │
│    → Apply scaling based on variable ranges                     │
│    → Configure termination for budget                           │
│                                                                 │
│  Level 4: Runtime Adaptation                                    │
│    → Monitor convergence behavior                               │
│    → Detect stagnation, infeasibility, numerical issues         │
│    → Adjust options or restart with different config            │
│                                                                 │
│  Level 5: Knowledge Accumulation                                │
│    → Record what worked for this problem type                   │
│    → Apply learnings to future similar problems                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 User-Facing Intent vs Internal Configuration

**Users express intent**, Paola handles details:

| User Says | Paola Does |
|-----------|------------|
| `optimizer="auto"` | Full algorithm selection + configuration |
| `optimizer="gradient-based"` | Selects SLSQP/IPOPT/SNOPT based on problem |
| `optimizer="global"` | Selects CMA-ES/DE/Optuna based on budget |
| `priority="robustness"` | Conservative tolerances, more iterations |
| `priority="speed"` | Relaxed tolerances, early stopping |

### 5.4 Optimizer Configuration Tools (Expert Escape Hatch)

These tools are **optional** - for experts who know exactly what they want.
Most users should rely on Paola's automatic configuration.

```
┌─────────────────────────────────────────────────────────────────┐
│           OPTIMIZER CONFIGURATION (Expert Escape Hatch)         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  config_scipy                                                   │
│    - config_id: str                                             │
│    - algorithm: "SLSQP" | "L-BFGS-B" | "COBYLA" | ...           │
│    - maxiter: int = 200                                         │
│    - ftol: float = 1e-9                                         │
│    → Expert override - bypasses Paola's auto-config             │
│                                                                 │
│  config_ipopt                                                   │
│    - config_id: str                                             │
│    - max_iter: int = 3000                                       │
│    - tol: float = 1e-8                                          │
│    - linear_solver: str = "mumps"                               │
│    - mu_strategy: str = "adaptive"                              │
│    → Expert override for IPOPT's 250 options                    │
│                                                                 │
│  get_algorithm_info                                             │
│    - algorithm: str                                             │
│    → Returns algorithm characteristics for expert users         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.5 Optimization Execution Tools (Run-centric)

These tools EXECUTE optimization. **Paola handles initialization and configuration.**

```
┌─────────────────────────────────────────────────────────────────┐
│                   OPTIMIZATION EXECUTION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  run_optimization                                               │
│    - problem_id: str              ← References Foundry problem  │
│    - optimizer: str = "auto"      ← "auto", "gradient-based",   │
│                                     "global", "scipy:SLSQP",    │
│                                     or config_id                │
│    - priority: str = "balanced"   ← "speed", "robustness",      │
│                                     "accuracy", "balanced"      │
│    - options: str (JSON)          ← Expert override (optional)  │
│    - description: str             ← Run description             │
│    → Paola selects algorithm (if "auto")                        │
│    → Paola configures based on problem + priority               │
│    → Paola determines initialization                            │
│    → Creates Run, executes, handles failures, records           │
│    → Returns run_id and optimization result                     │
│                                                                 │
│  get_run_info                                                   │
│    - run_id: int                                                │
│    → Returns run status, iterations, best solution              │
│                                                                 │
│  list_runs                                                      │
│    - problem_id: str (optional)                                 │
│    - algorithm: str (optional)                                  │
│    → Returns list of runs with summary                          │
│                                                                 │
│  get_best_solution                                              │
│    - problem_id: str                                            │
│    → Returns best solution across all runs for this problem     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Compact Bounds Specification

For large variable spaces (100+ variables), bounds need compact specification.

### 6.1 BoundsSpec Format

```python
# Option 1: Explicit list (small problems)
bounds = [[-5, 10], [-5, 10], [0, 1]]

# Option 2: Uniform bounds (common case)
bounds = {
    "type": "uniform",
    "lower": -1.0,
    "upper": 1.0,
    "dimension": 100
}

# Option 3: Per-group bounds (structured problems)
bounds = {
    "type": "groups",
    "groups": [
        {"name": "shape", "lower": -0.1, "upper": 0.1, "count": 50},
        {"name": "twist", "lower": -5.0, "upper": 5.0, "count": 10},
        {"name": "thickness", "lower": 0.01, "upper": 0.2, "count": 40}
    ]
}

# Option 4: From evaluator metadata
bounds = "from_evaluator"  # Infer from evaluator's input_bounds

# Option 5: Template reference (pre-registered)
bounds = {"template": "ffd_wing_100"}
```

### 6.2 Bounds Template Registration

```python
# Register reusable bounds template
register_bounds_template(
    template_id="ffd_wing_100",
    bounds={
        "type": "groups",
        "groups": [
            {"name": "upper_surface", "lower": -0.05, "upper": 0.05, "count": 50},
            {"name": "lower_surface", "lower": -0.05, "upper": 0.05, "count": 50}
        ]
    }
)

# Use in problem creation
create_nlp_problem(
    problem_id="wing_design",
    objective_evaluator_id="su2_drag",
    bounds={"template": "ffd_wing_100"}
)
```

---

## 7. Data Flow with Agent Reasoning

### 7.1 Complete Workflow Example

```
User: "Optimize wing_design using SLSQP"

Step 1: Paola loads problem from Foundry
─────────────────────────────────────────────
problem = foundry.get_problem("wing_design")
# Returns:
#   dimension: 100
#   bounds: [[-0.05, 0.05]] * 100
#   objective_evaluator_id: "su2_drag"
#   domain_hint: "shape_optimization"

Step 2: Paola determines initialization strategy
─────────────────────────────────────────────
# Agent reasoning:
# 1. Check for warm-start: foundry.get_runs("wing_design")
#    → Found run_42 with final_objective = 0.0234 ✓
# 2. Decision: warm-start from run_42's best design
#
# OR if no history:
# 1. Check domain_hint: "shape_optimization"
# 2. Decision: initialize at zero (baseline shape)
#
# OR if general problem:
# 1. No warm-start, no domain hint
# 2. Decision: initialize at center of bounds

x0 = paola.determine_initialization(problem, algorithm="SLSQP")
# Returns: np.array([...])  # 100-dimensional vector

Step 3: Paola executes optimization
─────────────────────────────────────────────
result = scipy.minimize(
    fun=evaluator.evaluate,
    x0=x0,                    # Paola-determined
    method="SLSQP",
    bounds=problem.bounds,    # From problem
    ...
)

Step 4: Paola records results
─────────────────────────────────────────────
foundry.record_run(
    problem_id="wing_design",
    algorithm="SLSQP",
    initialization_strategy="warm_start:run_42",  # Logged for transparency
    result=result
)
```

### 7.2 Agent Reasoning Transparency

Paola logs initialization decisions for user understanding:

```
💭 Determining initialization for wing_design...
   • Algorithm: SLSQP (gradient-based, requires initial point)
   • Checking run history: Found 3 previous runs
   • Best run: run_42 (objective: 0.0234, converged)
   • Decision: Warm-start from run_42's best solution

🔧 Running SLSQP optimization...
```

---

## 8. Unified Optimizer Interface

### 8.1 Base Class

```python
# paola/optimizers/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

class OptimizerBackend(ABC):
    """
    Unified interface for all optimization backends.

    Key principle: Optimizer receives problem and evaluator.
    Initialization is handled by Paola before calling optimize().
    """

    @abstractmethod
    def optimize(
        self,
        problem: NLPProblem,
        evaluator: NLPEvaluator,
        x0: np.ndarray,             # Paola-determined initial point
        options: Dict[str, Any]
    ) -> OptimizationResult:
        """Run optimization from given starting point."""
        pass

    @classmethod
    @abstractmethod
    def algorithm_name(cls) -> str:
        """Return algorithm identifier."""
        pass

    @classmethod
    def requires_initial_point(cls) -> bool:
        """Whether algorithm needs explicit x0."""
        return True  # Most do; Optuna overrides to False

    @classmethod
    def supports_population(cls) -> bool:
        """Whether algorithm uses population-based search."""
        return False

    @classmethod
    def default_options(cls) -> Dict[str, Any]:
        """Default algorithm configuration."""
        return {}
```

### 8.2 Initialization Manager

```python
# paola/agent/initialization.py
class InitializationManager:
    """
    Paola's initialization intelligence.

    Determines optimal initialization strategy based on:
    - Algorithm requirements
    - Problem characteristics
    - Run history (warm-starting)
    - Domain knowledge
    """

    def __init__(self, foundry: OptimizationFoundry):
        self.foundry = foundry

    def determine_initialization(
        self,
        problem: NLPProblem,
        algorithm: str,
        algorithm_class: str  # "gradient", "population", "bayesian", "cmaes"
    ) -> InitializationResult:
        """
        Determine initialization for given problem and algorithm.

        Returns:
            InitializationResult with:
            - x0: initial point (or None for Bayesian)
            - population: initial population (for evolutionary)
            - sigma: step size (for CMA-ES)
            - strategy: str describing what was chosen
            - reasoning: str explaining why
        """
        # Step 1: Check for warm-start opportunities
        warm_start = self._check_warm_start(problem.problem_id)
        if warm_start is not None:
            return InitializationResult(
                x0=warm_start.best_design,
                strategy="warm_start",
                reasoning=f"Using best solution from run_{warm_start.run_id}"
            )

        # Step 2: Check domain hint
        if problem.domain_hint == "shape_optimization":
            return InitializationResult(
                x0=np.zeros(problem.dimension),
                strategy="zero",
                reasoning="Shape optimization: starting from baseline (zero deformation)"
            )

        # Step 3: Algorithm-specific defaults
        if algorithm_class == "gradient":
            return self._gradient_default(problem)
        elif algorithm_class == "population":
            return self._population_default(problem)
        elif algorithm_class == "cmaes":
            return self._cmaes_default(problem)
        elif algorithm_class == "bayesian":
            return InitializationResult(
                x0=None,
                strategy="sampler",
                reasoning="Bayesian optimization: sampler handles initialization"
            )

    def _gradient_default(self, problem: NLPProblem) -> InitializationResult:
        """Default for gradient-based: center of bounds."""
        bounds = np.array(problem.bounds)
        x0 = (bounds[:, 0] + bounds[:, 1]) / 2
        return InitializationResult(
            x0=x0,
            strategy="center",
            reasoning="Gradient-based optimizer: starting at center of bounds"
        )

    def _population_default(self, problem: NLPProblem) -> InitializationResult:
        """Default for population-based: LHS."""
        from scipy.stats import qmc
        bounds = np.array(problem.bounds)
        sampler = qmc.LatinHypercube(d=problem.dimension)
        samples = sampler.random(n=100)  # Population size
        population = qmc.scale(samples, bounds[:, 0], bounds[:, 1])
        return InitializationResult(
            population=population,
            strategy="lhs",
            reasoning="Population-based optimizer: using Latin Hypercube Sampling"
        )

    def _cmaes_default(self, problem: NLPProblem) -> InitializationResult:
        """Default for CMA-ES: center + sigma from bounds."""
        bounds = np.array(problem.bounds)
        mean = (bounds[:, 0] + bounds[:, 1]) / 2
        sigma = np.mean(bounds[:, 1] - bounds[:, 0]) * 0.25
        return InitializationResult(
            x0=mean,
            sigma=sigma,
            strategy="cmaes_default",
            reasoning=f"CMA-ES: mean at center, sigma={sigma:.4f} (1/4 of avg bound width)"
        )

    def _check_warm_start(self, problem_id: str) -> Optional[WarmStartInfo]:
        """Check if warm-start is available from previous runs."""
        runs = self.foundry.get_runs(problem_id=problem_id, status="completed")
        if not runs:
            return None

        # Find best successful run
        best_run = min(runs, key=lambda r: r.best_objective)
        if best_run.best_objective < float('inf'):
            return WarmStartInfo(
                run_id=best_run.run_id,
                best_design=best_run.best_design,
                best_objective=best_run.best_objective
            )
        return None
```

---

## 9. Migration Path

### 9.1 Current → Proposed

| Current Tool | Proposed Tool(s) | Changes |
|--------------|------------------|---------|
| `create_nlp_problem` | `create_nlp_problem` | Remove `initial_point`, add `domain_hint`, add compact bounds |
| `run_scipy_optimization` | `run_optimization` | Remove `bounds`, `initial_design`; Paola handles init |
| N/A | `config_scipy` | NEW: Store optimizer config |
| N/A | `config_nlopt` | NEW: Store optimizer config |
| `set_initial_point` | REMOVED | Paola handles initialization |
| `optimizer_create` | Deprecated | Use `run_optimization` |

### 9.2 Key Removals

**Removed from tools** (now agent intelligence):
- `initial_point` parameter in `create_nlp_problem`
- `initial_design` parameter in `run_optimization`
- `set_initial_point` tool entirely
- All initialization strategy choices from user

---

## 10. Resolved Design Decisions

| Decision | Resolution | Rationale |
|----------|------------|-----------|
| Bounds ownership | Problem definition only | Mathematical property |
| Initial point | Agent intelligence | The Paola Principle |
| **Algorithm selection** | **Agent intelligence** | Problem characteristics drive choice |
| **Option configuration** | **Agent intelligence** | 250 IPOPT options → user sees "priority" |
| Problem mutability | Immutable (for now) | Simplicity |
| Tool granularity | Intent-based + expert escape hatch | User simplicity with expert override |
| Large variable spaces | Compact bounds spec + no init in API | Scalability |
| Warm-starting | Automatic by agent | Expert knowledge |
| **Convergence failures** | **Agent handles** | Diagnose, adjust, retry |

---

## 11. Implementation Priority

### Phase 1: Core Refactoring
1. Implement `InitializationManager` class
2. Update `create_nlp_problem` (remove initial_point, add domain_hint)
3. Implement compact `BoundsSpec` parsing
4. Create unified `run_optimization` tool

### Phase 2: Optimizer Configs
5. Implement `config_scipy`, `config_nlopt`, `config_evolutionary`
6. Implement `OptimizerFactory`
7. Store configs in Foundry

### Phase 3: Advanced Features
8. Implement bounds templates
9. Add warm-start logging and visualization
10. Multi-restart support for multi-modal problems

---

## Appendix A: Final Tool Signatures

```python
# ═══════════════════════════════════════════════════════════════
# PROBLEM FORMULATION (Mathematical - User Specifies)
# ═══════════════════════════════════════════════════════════════

@tool
def create_nlp_problem(
    problem_id: str,
    objective_evaluator_id: str,
    bounds: Union[List[List[float]], Dict[str, Any]],  # Compact spec supported
    objective_sense: str = "minimize",
    inequality_constraints: Optional[List[Dict]] = None,
    equality_constraints: Optional[List[Dict]] = None,
    domain_hint: Optional[str] = None,  # "shape_optimization", "hyperparameter_tuning"
    description: Optional[str] = None
    # NO initial_point - Paola decides
    # NO algorithm options - Paola decides
) -> Dict[str, Any]: ...

@tool
def get_problem_info(problem_id: str) -> Dict[str, Any]: ...


# ═══════════════════════════════════════════════════════════════
# OPTIMIZATION EXECUTION (Intent-Based - Paola Handles Details)
# ═══════════════════════════════════════════════════════════════

@tool
def run_optimization(
    problem_id: str,
    optimizer: str = "auto",       # "auto", "gradient-based", "global",
                                   # "scipy:SLSQP", or config_id
    priority: str = "balanced",    # "speed", "robustness", "accuracy", "balanced"
    options: Optional[str] = None, # Expert JSON override (optional)
    description: str = ""
    # NO initial_point - Paola decides
    # NO algorithm config - Paola decides (unless options override)
) -> Dict[str, Any]: ...

@tool
def get_run_info(run_id: int) -> Dict[str, Any]: ...

@tool
def list_runs(
    problem_id: Optional[str] = None,
    algorithm: Optional[str] = None
) -> Dict[str, Any]: ...

@tool
def get_best_solution(problem_id: str) -> Dict[str, Any]: ...


# ═══════════════════════════════════════════════════════════════
# EXPERT ESCAPE HATCH (Optional - For Users Who Know Exactly What They Want)
# ═══════════════════════════════════════════════════════════════

@tool
def config_scipy(
    config_id: str,
    algorithm: str,
    maxiter: int = 200,
    ftol: float = 1e-9,
    gtol: float = 1e-6
    # ... other scipy options
) -> Dict[str, Any]: ...
# Bypasses Paola's auto-configuration

@tool
def config_ipopt(
    config_id: str,
    max_iter: int = 3000,
    tol: float = 1e-8,
    linear_solver: str = "mumps",
    mu_strategy: str = "adaptive"
    # ... subset of IPOPT's 250 options
) -> Dict[str, Any]: ...
# Bypasses Paola's auto-configuration

@tool
def get_algorithm_info(algorithm: str) -> Dict[str, Any]: ...
# Returns capabilities, options, requirements for experts
```

---

## Appendix B: The Paola Principle (Complete)

> **"Optimization complexity is agent intelligence, not user burden."**

### What Users Specify

| User Responsibility | Example |
|--------------------|---------|
| **WHAT to optimize** | Problem: objective, constraints, bounds |
| **Intent** | `priority="robustness"` or `optimizer="global"` |

### What Paola Handles

| Paola's Responsibility | Complexity Hidden |
|----------------------|-------------------|
| **Initialization** | Algorithm-specific requirements (x0, sigma, population) |
| **Algorithm selection** | Match problem characteristics to optimizer |
| **Option configuration** | 250 IPOPT options, SNOPT scaling, CMA-ES restarts |
| **Convergence handling** | Detect failures, diagnose, adjust, retry |
| **Warm-starting** | Check history, apply best previous solution |
| **Knowledge accumulation** | Learn from successful runs |

### The Three-Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│  User Layer: Intent                                 │
│    "Optimize my wing robustly"                      │
├─────────────────────────────────────────────────────┤
│  Paola Layer: Expert Knowledge                      │
│    Algorithm selection + Configuration +            │
│    Initialization + Failure handling                │
├─────────────────────────────────────────────────────┤
│  Optimizer Layer: Execution                         │
│    IPOPT/SNOPT/CMA-ES with 100+ options each        │
└─────────────────────────────────────────────────────┘
```

### Benefits

1. **Users focus on engineering, not optimization**
2. **Expert knowledge applied automatically**
3. **No large vectors in API** (bounds via compact spec, init by Paola)
4. **Failures handled gracefully** (diagnose, adjust, retry)
5. **Continuous improvement** (Paola learns from experience)
6. **Expert escape hatch preserved** (for users who know exactly what they want)

---

**Document Version History**

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2025-12-15 | Initial design discussion |
| 0.2.0 | 2025-12-15 | Added Paola Principle for initialization, removed initial_point from tools, added compact bounds spec |
| 0.3.0 | 2025-12-15 | Extended Paola Principle to configuration, added intent-based optimizer parameter, config tools as expert escape hatch |
