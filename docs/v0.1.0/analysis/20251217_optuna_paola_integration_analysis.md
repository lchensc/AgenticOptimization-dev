# Optuna + Paola Integration Analysis

**Date**: 2025-12-17
**Updated**: 2025-12-17 (Second analysis based on broader Paola scope)
**Purpose**: First-principles analysis of how Paola can become an expert Optuna user

## Executive Summary

Paola's scope is **all optimization problems**, not just engineering design. This analysis examines:
1. How Paola can operate Optuna like a human expert
2. The added value of agent-driven Optuna usage
3. Connection to Paola's graph architecture and learning capabilities

**Key Finding**: Paola can provide expert-level Optuna operation for everything EXCEPT writing the objective function (which requires domain knowledge). The graph architecture enables multi-stage strategies that neither tool can do alone.

---

## 1. Understanding Optuna's Core Innovation

### Define-by-Run vs Define-and-Run

**Traditional (Define-and-Run)**:
```python
# Must specify ENTIRE search space upfront
search_space = {
    "learning_rate": [0.001, 0.01, 0.1],
    "n_layers": [2, 3, 4, 5],
    "dropout": [0.1, 0.2, 0.3],
}
optimizer.search(objective, search_space)
```

**Optuna (Define-by-Run)**:
```python
def objective(trial):
    # Search space defined INSIDE the function
    model_type = trial.suggest_categorical("model", ["MLP", "CNN", "Transformer"])

    if model_type == "MLP":
        n_layers = trial.suggest_int("n_layers", 2, 10)
        # Only MLP has this parameter
    elif model_type == "CNN":
        n_filters = trial.suggest_int("n_filters", 16, 128)
        # Only CNN has this parameter

    # Conditional search space emerges naturally
    return train_and_evaluate(model_type, ...)
```

**Why This Matters**: The search space can be *conditional* and *dynamic*. This is critical for NAS where different architectures have different hyperparameters.

### Optuna's Key Components

| Component | Purpose | Key Insight |
|-----------|---------|-------------|
| **Study** | Optimization session | Persistent, resumable, stores all trials |
| **Trial** | Single evaluation | Gets parameters via suggest_* API |
| **Sampler** | Algorithm for suggesting parameters | TPE, CMA-ES, GP, NSGA-II, etc. |
| **Pruner** | Early stopping unpromising trials | Saves compute on bad configurations |
| **Storage** | Persistence backend | SQLite, PostgreSQL, in-memory |

---

## 2. Current Paola Optuna Integration (Limitations)

Looking at `paola/optimizers/backends.py`, the current integration:

```python
class OptunaBackend:
    SAMPLERS = ["TPE", "CMA-ES", "Random", "Grid"]  # Only 4 of 11

    def optimize(self, objective, bounds, x0, config, ...):
        # Fixed search space from bounds
        for i, (lb, ub) in enumerate(bounds):
            x.append(trial.suggest_float(f"x{i}", lb, ub))
```

**Current Limitations**:

| Feature | Status | Impact |
|---------|--------|--------|
| Define-by-run | ❌ Not supported | Can't do conditional search spaces |
| Samplers | 4 of 11 | Missing GP, NSGA-II/III, QMC, AutoSampler |
| Pruning | ❌ Not supported | Wastes compute on bad trials |
| Multi-objective | ❌ Not supported | Can't optimize Pareto fronts |
| Constraints | ❌ Not supported | Optuna has constrained TPE |
| Study persistence | ❌ Not supported | Can't resume studies |
| Categorical params | ❌ Not supported | Only continuous bounds |

**Root Issue**: Paola treats Optuna like SciPy - fixed bounds, continuous variables. This misses Optuna's core value proposition.

---

## 3. What Makes an "Expert" Optuna User?

An expert Optuna user knows:

### 3.1 Sampler Selection

| Problem Characteristics | Best Sampler | Why |
|------------------------|--------------|-----|
| General, mixed types, <1000 trials | **TPE** | Handles categorical well |
| Continuous only, >1000 trials | **CMA-ES** | Better for continuous optimization |
| Need to explore uniformly first | **QMC** | Low-discrepancy sequences |
| Multi-objective | **NSGA-II/III** | Pareto front optimization |
| Small discrete space | **Grid** | Exhaustive search |
| Unsure | **AutoSampler** | Automatic selection |
| Expensive + need constraints | **GP** | Bayesian with constraints |

### 3.2 Pruning Strategy

| Scenario | Best Pruner | Configuration |
|----------|-------------|---------------|
| Training curves available | **MedianPruner** | Prune below median at each step |
| Large hyperband-style search | **HyperbandPruner** | Successive halving |
| Need statistical rigor | **WilcoxonPruner** | Significance test |
| Don't want early pruning | **PatientPruner** | Wrap with patience |

### 3.3 Search Space Design

Expert knowledge:
- **Categorical for discrete choices** (optimizer type, activation function)
- **Log-uniform for learning rates** (`suggest_float(..., log=True)`)
- **Conditional parameters** (only tune CNN filters if model is CNN)
- **Step size for integers** (`suggest_int(..., step=8)` for GPU-friendly sizes)

### 3.4 Study Management

- When to create new study vs continue existing
- How many startup trials before TPE kicks in (default: 10)
- Parallel execution with `n_jobs` and `constant_liar`

---

## 4. How Paola's Graph Architecture Enhances Optuna

### 4.1 Multi-Stage Search Space Refinement (Key Innovation!)

This is where Paola adds unique value. Traditional Optuna: one study, fixed space.
Paola: multiple nodes, evolving search space.

```
Graph #1: "Optimize neural network hyperparameters"
│
├── n1: Broad exploration (100 trials, TPE)
│   ├── Search space: model=[MLP,CNN,Transformer], lr=[1e-5,1e-1], ...
│   ├── Results: MLP wins, lr around 1e-3, depth 4-6 best
│   └── Agent analysis: "MLP consistently best, narrow search"
│
├── n2: Focused MLP search (edge_type="refine", 50 trials)
│   ├── Search space: model=MLP (fixed), lr=[5e-4,5e-3], depth=[4,7]
│   ├── Results: Best at lr=1e-3, depth=5, dropout=0.3
│   └── Agent analysis: "Converging, add regularization params"
│
└── n3: Fine-tuning (edge_type="refine", 30 trials)
    ├── Search space: narrow ranges + weight_decay, batch_norm
    └── Results: Final best configuration
```

**Value**: The agent reasons about results and *evolves the search space* - something Optuna alone cannot do.

### 4.2 Cross-Study Learning

```
query_past_graphs(problem_pattern="*neural*", success=True)

Result: "Graph #42 optimized similar problem with:
- Started with TPE, 100 trials
- Switched to CMA-ES for final refinement
- Key insight: log-scale for learning rate was critical"

Agent: "I'll use this strategy for the new problem"
```

### 4.3 Sampler Switching Within Graph

```
Graph node 1: TPE (good for initial exploration with categorical)
    ↓ Agent decides: "Found good region, switch to CMA-ES"
Graph node 2: CMA-ES (better for continuous refinement)
```

### 4.4 Pruning Strategy Decisions

The agent can:
- Analyze pruned trials to understand failure patterns
- Adjust pruning aggressiveness based on compute budget
- Decide when pruning helps vs hurts (e.g., noisy objectives)

---

## 5. What Paola CAN and CANNOT Do

### Paola CAN:

| Capability | How |
|------------|-----|
| Select optimal sampler | Based on problem characteristics + past experience |
| Configure sampler options | n_startup_trials, multivariate, seed |
| Choose pruning strategy | Based on evaluation cost and available intermediates |
| Evolve search space across nodes | Narrow ranges, fix good values, add new params |
| Learn from past studies | Cross-graph queries |
| Manage multi-objective trade-offs | Help interpret Pareto fronts |

### Paola CANNOT:

| Limitation | Why | Workaround |
|------------|-----|------------|
| Write define-by-run objective | Requires domain knowledge (what params exist for CNN vs MLP) | User provides objective function |
| Know problem-specific param ranges | Domain expertise needed | User specifies or Paola asks |
| Execute training code | Paola is optimizer, not executor | User provides evaluator |

### The Division of Labor

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER PROVIDES                            │
│  - Objective function (define-by-run or wrapper)                │
│  - Domain constraints (what parameters make sense)              │
│  - Evaluation budget                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        PAOLA HANDLES                             │
│  - Sampler selection (TPE vs CMA-ES vs GP vs ...)              │
│  - Pruner configuration (when useful)                           │
│  - Search space evolution (narrow, expand, fix)                 │
│  - Cross-study learning (what worked before)                    │
│  - Multi-stage optimization (graph nodes)                       │
│  - Result analysis and next-step recommendations                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Required Changes to Paola

### 6.1 Backend Enhancements

```python
class OptunaBackend:
    SAMPLERS = [
        "TPE", "CMA-ES", "Random", "Grid", "QMC",
        "GP", "NSGA-II", "NSGA-III", "Auto"
    ]

    PRUNERS = [
        "Median", "Hyperband", "SuccessiveHalving",
        "Percentile", "Threshold", "None"
    ]

    def optimize(self, objective, bounds, x0, config, ...):
        # Support more config options
        sampler = self._create_sampler(config)
        pruner = self._create_pruner(config)

        study = optuna.create_study(
            direction=config.get("direction", "minimize"),
            sampler=sampler,
            pruner=pruner,
            storage=config.get("storage"),  # For persistence
            study_name=config.get("study_name"),
            load_if_exists=config.get("resume", False),
        )
```

### 6.2 New Config Options for Skill

```yaml
# Sampler options
sampler: TPE  # or CMA-ES, GP, NSGA-II, Auto, etc.
sampler_options:
  n_startup_trials: 10  # Random trials before TPE kicks in
  multivariate: true    # Consider parameter correlations
  constant_liar: true   # For parallel execution

# Pruner options
pruner: Median  # or Hyperband, None, etc.
pruner_options:
  n_startup_trials: 5
  n_warmup_steps: 10

# Study options
n_trials: 100
direction: minimize  # or maximize, or list for multi-objective
study_name: "my_study"
storage: "sqlite:///optuna.db"  # For persistence
resume: true  # Load existing study if exists
```

### 6.3 Multi-Objective Support

```python
# Multi-objective study
study = optuna.create_study(
    directions=["minimize", "minimize"],  # e.g., loss and latency
    sampler=optuna.samplers.NSGAIISampler()
)

# Returns Pareto front, not single best
pareto_trials = study.best_trials
```

### 6.4 Search Space Evolution (Graph Integration)

The key innovation - agent can modify search space between nodes:

```python
# Node 1 result analysis
node1_results = analyze_study(study)
# Agent sees: "learning_rate 1e-3 to 1e-2 always best"

# Node 2 with narrowed space
node2_config = {
    "search_space": {
        "learning_rate": {"type": "float", "low": 1e-3, "high": 1e-2, "log": True},
        # Narrowed from [1e-5, 1e-1]
    }
}
```

---

## 7. Paola as Optuna Expert: Detailed Analysis

### What Does a Human Optuna Expert Do?

| Phase | Expert Actions | Domain Knowledge Required? |
|-------|----------------|---------------------------|
| **1. Problem Assessment** | HPO? NAS? Black-box? Evaluation cost? | Partial |
| **2. Search Space Design** | Parameters, ranges, log-scale, conditionals | **Yes - high** |
| **3. Sampler Selection** | TPE vs CMA-ES vs GP based on characteristics | No - algorithmic |
| **4. Pruning Config** | Choose pruner, warmup, patience | No - algorithmic |
| **5. Study Execution** | n_trials, parallelism, storage | No - operational |
| **6. Result Analysis** | Parameter importance, winning regions | No - analytical |
| **7. Strategy Iteration** | Narrow space, switch sampler, add params | Partial |

### What Paola CAN Do

| Expert Capability | How Paola Does It | Implementation |
|-------------------|-------------------|----------------|
| **Sampler selection** | Match problem characteristics to sampler strengths | Skill knowledge |
| **Sampler configuration** | Set n_startup_trials, multivariate, constant_liar | Config generation |
| **Pruning strategy** | Analyze if intermediates available, select pruner | Skill knowledge |
| **Result analysis** | Query study, identify important parameters | Reasoning |
| **Multi-stage strategy** | Explore → narrow → refine via graph nodes | Graph architecture |
| **Cross-problem learning** | "Similar problem used CMA-ES successfully" | query_past_graphs |
| **Best practice enforcement** | Log-scale for LR, step sizes for GPU dims | Skill knowledge |
| **Stopping decisions** | Detect diminishing returns | Reasoning |

### What Paola CANNOT Do (User Must Provide)

| Limitation | Reason | User's Responsibility |
|------------|--------|----------------------|
| **Write objective function** | Domain-specific code | User implements |
| **Know what parameters exist** | Model/system specific | User defines search space structure |
| **Set exact ranges** | Requires domain experience | User provides rough ranges |

### Division of Responsibility

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER PROVIDES                               │
│                                                                  │
│  1. Objective function (the "what to optimize")                 │
│  2. Search space STRUCTURE (what parameters exist)              │
│  3. Rough bounds (Paola can suggest refinements)                │
│  4. Budget constraints (max trials, max time)                   │
│  5. Problem description (for Paola to reason about)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PAOLA HANDLES                               │
│                                                                  │
│  1. Sampler selection and configuration                         │
│  2. Pruning strategy (when useful, which pruner)                │
│  3. Multi-stage optimization (explore → narrow → refine)        │
│  4. Result analysis and interpretation                          │
│  5. Cross-study learning from past optimizations                │
│  6. Stopping decisions (diminishing returns)                    │
│  7. Strategy adaptation based on results                        │
└─────────────────────────────────────────────────────────────────┘
```

### Added Value by User Type

| User Type | Without Paola | With Paola Expert | Value |
|-----------|---------------|-------------------|-------|
| **Novice** | Uses defaults, random choices | Expert-level sampler/pruner selection | **Very High** |
| **Intermediate** | Trial-and-error, manual analysis | Systematic strategy, automated analysis | **High** |
| **Expert** | Manual multi-stage, no memory | Automated execution + organizational learning | **Moderate-High** |
| **Organization** | Knowledge lost when people leave | Accumulated learning across projects | **Very High** |

---

## 8. Connection to Graph Architecture

### Multi-Stage Optuna Strategy (Graph Nodes)

```
Graph: "Hyperparameter optimization for image classifier"

n1: Broad exploration
    ├── Sampler: TPE (good for categorical + continuous)
    ├── Config: {n_trials: 50, n_startup_trials: 10}
    ├── Search space: model=[ResNet,VGG,EfficientNet], lr=[1e-5,1e-1], ...
    └── Agent analysis: "ResNet consistently best, lr optimal around 1e-3"

n2: Focused search (edge: refine)
    ├── Sampler: TPE (continue with learned distribution)
    ├── Config: {n_trials: 30}
    ├── Search space: model=ResNet (fixed), lr=[5e-4,5e-3] (narrowed)
    └── Agent analysis: "depth=50 best, now tune regularization"

n3: Fine-tuning (edge: refine)
    ├── Sampler: CMA-ES (pure continuous now)
    ├── Config: {n_trials: 20}
    ├── Search space: Add weight_decay, dropout; narrow other ranges
    └── Result: Final best configuration
```

### Cross-Study Learning

```python
# Before starting new HPO problem
past_results = query_past_graphs(
    problem_pattern="*image_classification*",
    success=True
)

# Agent learns:
# - "TPE with n_startup=10 worked well for similar size"
# - "Switching to CMA-ES for final refinement improved results"
# - "Log-scale for learning rate was critical"
```

### Unique Value: Strategy Evolution

Neither Optuna nor manual usage provides:
1. **Systematic narrowing** based on analysis (not just TPE's soft adaptation)
2. **Sampler switching** mid-optimization based on results
3. **Cross-problem memory** of what strategies worked
4. **Documented reasoning** for why each decision was made

---

## 9. Skill Content Requirements

The Optuna skill should enable Paola to operate as an expert:

### Core Knowledge (skill.yaml)
- When to use Optuna vs IPOPT/SciPy
- Sampler selection decision tree
- Pruning applicability criteria

### Samplers (options.samplers)
| Sampler | When to Use | Key Options |
|---------|-------------|-------------|
| TPE | Default for mixed spaces, <1000 trials | n_startup_trials, multivariate |
| CMA-ES | Continuous-only, >1000 trials | sigma0, restart_strategy |
| GP | Expensive evals, need constraints | n_startup_trials |
| NSGA-II/III | Multi-objective | population_size |
| Random/QMC | Initial exploration, baseline | seed |
| Auto | Unsure | (automatic) |

### Pruners (options.pruners)
| Pruner | When to Use | Key Options |
|--------|-------------|-------------|
| Median | Training with epochs | n_startup_trials, n_warmup_steps |
| Hyperband | Large budget, aggressive pruning | min/max_resource, reduction_factor |
| None | Single expensive evaluation | - |

### Paola Integration (paola section)
- Graph node strategies for Optuna
- When to switch samplers
- How to interpret and act on results
- Cross-study learning patterns

---

## 10. Recommendations

### Phase 1: Foundation (Immediate)
1. Create comprehensive Optuna skill with expert knowledge
2. Enhance backend with more samplers (GP, NSGA-II, Auto)
3. Add pruner support to backend

### Phase 2: Expert Operation (Short-term)
1. Sampler selection logic based on problem description
2. Result analysis and interpretation
3. Multi-stage graph strategies for Optuna

### Phase 3: Learning (Medium-term)
1. Cross-study learning: "what worked for similar problems"
2. Organizational knowledge accumulation
3. Auto-generated "learned" Optuna strategies

### Phase 4: Advanced (Future)
1. Multi-objective support with Pareto analysis
2. Distributed/parallel execution
3. Integration with MLOps tools

---

## 11. Summary

### Key Findings

1. **Paola CAN operate Optuna like an expert** for algorithmic decisions (sampler, pruner, strategy)

2. **User still owns domain knowledge** (objective function, parameter structure)

3. **Graph architecture enables unique value**:
   - Multi-stage strategies neither tool can do alone
   - Cross-study learning and organizational memory
   - Documented reasoning for reproducibility

4. **Added value is significant** especially for novice/intermediate users and organizations

### The Paola Principle Applied to Optuna

> "Optuna complexity is Paola intelligence, not user burden"

User says: "Optimize my neural network hyperparameters"
Paola handles: TPE vs CMA-ES? How many startup trials? When to prune? When to narrow? When to stop?

### Investment Recommendation

**High priority** - Optuna expertise is valuable for:
- Expanding Paola beyond engineering to ML/HPO domain
- Demonstrating "expert operation" capability
- Leveraging graph architecture for multi-stage strategies
- Building organizational learning features
