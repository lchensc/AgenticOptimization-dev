# Future PAOLA Graph Extensions: Brainstorming

**Date**: 2025-12-16
**Status**: Brainstorming (not scheduled for near-term development)

## Overview

The PAOLA graph structure (v0.3.x) provides a flexible foundation for tracking optimization processes. This document explores two potential future extensions that leverage the graph architecture for high-impact applications.

**Current State (v0.3.x)**:
- Graph = complete optimization task
- Node = single optimizer execution (hundreds of iterations)
- Agent decides between optimizer runs

**Core Insight**: The graph structure can be applied at different granularities and domains beyond the current "optimizer-run" level.

---

## Direction A: Per-Iterate Engineering Design Graphs

### Motivation

Engineering simulations are expensive:
- CFD evaluation: 4-10 CPU hours ($400-$1000)
- Structural FEA: 1-4 CPU hours ($100-$400)
- Adjoint gradient: 6+ CPU hours ($600)

At this cost scale, the current granularity (node = optimizer run with 100s of iterations) is too coarse. Engineers want control after **every** evaluation, not after the optimizer finishes.

### Proposed Granularity

| Current (v0.3.x) | Proposed Engineering Graphs |
|------------------|----------------------------|
| Node = optimizer run | Node = single evaluation |
| Agent decides between runs | Agent decides after EVERY eval |
| Good for cheap evaluations | Essential for expensive simulations |

### Example Graph

```
Graph #1: "Optimize airfoil drag"
│
├── n1: LHS sample 1     → Cd=0.050  (6 CPU-hrs)
├── n2: LHS sample 2     → Cd=0.080  (6 CPU-hrs)
├── n3: LHS sample 3     → Cd=0.030★ (6 CPU-hrs)
│
├── n4: Surrogate suggestion from n3 → Cd=0.025★ (6 CPU-hrs)
│   └── Agent: "Surrogate working well, continue"
│
├── n5: Adjoint gradient from n4 → Cd=0.022★ (8 CPU-hrs)
│   └── Agent: "Gradient effective but expensive.
│               Try surrogate before next gradient."
│
└── n6: Surrogate refinement from n5 → Cd=0.021★ (6 CPU-hrs)
    └── Agent: "Within tolerance. Done."

Total: 6 evaluations, ~38 CPU-hrs, ~$3,800
(vs. blind optimizer: 50+ evals, ~$20,000+)
```

### Key Benefits

1. **Surgical control**: Stop after any bad evaluation
2. **Explicit design lineage**: Track how design A evolved to design B
3. **Restart from ANY point**: Not just optimizer's best, but any evaluated design
4. **Surrogate integration**: Each node carries surrogate prediction vs actual value
5. **Cost awareness**: Agent can reason about compute budget

### Schema Extensions Needed

```python
@dataclass
class EvaluationNode:
    node_id: str

    # Design point
    design_vector: List[float]

    # Evaluation results
    objective: float
    constraints: Optional[Dict[str, float]]

    # Cost tracking
    compute_time: float  # CPU-hours
    wall_time: float

    # Source of this design point
    source: Literal["lhs", "surrogate", "gradient", "random", "user"]

    # Surrogate info (if applicable)
    surrogate_prediction: Optional[float]
    prediction_uncertainty: Optional[float]
    prediction_error: Optional[float]  # |actual - predicted|

# Extended edge types
class EngineeringEdgeType(Enum):
    SURROGATE_SUGGESTION = "surrogate_suggestion"
    GRADIENT_STEP = "gradient_step"
    TRUST_REGION_STEP = "trust_region_step"
    INFILL_CRITERION = "infill_criterion"  # EI, UCB, etc.
    LHS_SAMPLE = "lhs_sample"
```

### Research Questions

1. **Granularity threshold**: At what eval cost does per-iterate control pay off? (Hypothesis: >1 CPU-hr/eval)
2. **Agent cognitive load**: Can LLM effectively reason about 50+ individual evaluations?
3. **Surrogate-agent interaction**: Should agent choose surrogate model type, or just when to trust it?
4. **Batch decisions**: When to request multiple evals in parallel vs sequential?

### Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Impact | **High** | Direct cost savings ($10,000s per optimization) |
| Differentiation | **High** | No existing tool offers this |
| Feasibility | **Medium** | Requires surrogate model integration |
| Alignment with PAOLA | **High** | Natural extension of current graph model |

**Recommendation**: Strong candidate for future development. Aligns with PAOLA's "expensive simulation" value proposition.

---

## Direction B: Dynamic Search Space NAS

### Motivation

Optuna requires users to predefine conditional search spaces:

```python
def objective(trial):
    # User must define ALL of this upfront
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    if optimizer == "Adam":
        lr = trial.suggest_float("lr", 1e-4, 1e-2)
    else:
        lr = trial.suggest_float("lr", 1e-3, 1e-1)
        momentum = trial.suggest_float("momentum", 0.5, 0.99)
```

User burden:
- Must know all options to try
- Must know reasonable ranges for each
- Must know conditional dependencies
- Must define everything **before** any trial runs

### Proposed Enhancement

PAOLA generates search spaces dynamically; Optuna executes them.

```
┌─────────────────────────────────────────────────────────────────┐
│  PAOLA: Decides WHAT to search    │  Optuna: Searches it        │
│  (Dynamic, reasoning-based)       │  (Algorithmic, rigorous)    │
└─────────────────────────────────────────────────────────────────┘
```

### Example Flow

**Phase 1: Initial Search Space (AI-generated)**
```
User: "Optimize neural network for tabular classification"

PAOLA analyzes: "Tabular + moderate size → MLP or TabNet likely good"

PAOLA generates search space:
{
  "model": ["MLP", "TabNet"],
  "if MLP": {"n_layers": [2,5], "hidden_dim": [64,256], "dropout": [0.1,0.5]},
  "if TabNet": {"n_steps": [3,10], "relaxation": [1.0,2.0]}
}

→ Optuna runs 20 trials
```

**Phase 2: Refined Search Space (AI adapts)**
```
PAOLA observes:
- MLP avg=0.85, best=0.88
- TabNet avg=0.81, best=0.83
- Best MLPs: n_layers=4-5, dropout=0.3-0.4

PAOLA reasons: "MLP winning. Deeper better. Should explore activations."

PAOLA generates REFINED space:
{
  "model": "MLP",  # Fixed
  "n_layers": [4, 8],  # Extended
  "hidden_dim": [128, 512],  # Shifted up
  "dropout": [0.25, 0.45],  # Narrowed
  "activation": ["relu", "gelu", "silu"],  # NEW dimension
  "batch_norm": [true, false]  # NEW dimension
}

→ Optuna runs 20 more trials
```

**Phase 3: Final Tuning**
```
PAOLA observes: GELU + batch_norm combo optimal, n_layers=6 sweet spot

PAOLA generates FINAL space:
{
  "n_layers": [5, 7],
  "hidden_dim": [256, 384],
  "activation": "gelu",  # Fixed
  "batch_norm": true,  # Fixed
  "learning_rate": [1e-4, 1e-3],  # NEW: training params
  "weight_decay": [1e-5, 1e-3]  # NEW
}

→ Optuna runs 20 trials → Best model found
```

### Graph Representation

```
Graph #1: "Optimize tabular classifier"
│
├── n1: Broad exploration
│   ├── search_space: {MLP vs TabNet, broad ranges}
│   ├── optuna_study: 20 trials, TPE sampler
│   ├── best_accuracy: 0.88
│   └── agent_analysis: "MLP winning, deeper better"
│
├── n2: Focus on MLP (edge_type="refine")
│   ├── search_space: {MLP only, new dimensions}
│   ├── optuna_study: 20 trials
│   ├── best_accuracy: 0.91
│   └── agent_analysis: "GELU+BN optimal"
│
└── n3: Final tuning (edge_type="refine")
    ├── search_space: {narrow ranges, training params}
    ├── optuna_study: 20 trials
    └── best_accuracy: 0.93
```

### Key Differences from Standard Optuna

| Aspect | Standard Optuna | PAOLA + Optuna |
|--------|-----------------|----------------|
| Search space | Fixed, human-defined | Dynamic, AI-generated |
| Adaptation | None (or manual restart) | Automatic based on results |
| New dimensions | Must restart study | AI adds mid-search |
| Narrowing | Requires human analysis | AI narrows automatically |
| Domain knowledge | Encoded in space definition | AI applies on-the-fly |

### Schema Extensions Needed

```python
@dataclass
class SearchSpaceNode:
    node_id: str

    # AI-generated search space
    search_space: Dict[str, Any]  # Optuna-compatible spec
    reasoning: str  # Why this space was chosen

    # Optuna execution
    n_trials: int
    sampler: str  # "TPE", "CMA-ES"
    pruner: Optional[str]

    # Results summary
    best_trial: Dict[str, Any]
    top_k_trials: List[Dict[str, Any]]

    # AI analysis
    analysis: str
    suggested_refinements: List[str]

class SearchSpaceEdgeType(Enum):
    REFINE = "refine"  # Narrow ranges based on evidence
    EXPAND = "expand"  # Add new dimensions
    PIVOT = "pivot"    # Switch focus entirely
    ZOOM = "zoom"      # Deep dive into region
```

### Research Questions

1. **When does dynamic space beat fixed?**: Simple problems may not benefit
2. **Exploration-exploitation**: How aggressive should space narrowing be?
3. **Dimension discovery**: How does AI know what NEW dimensions to add?
4. **Integration depth**: Use Optuna as black-box, or deeper integration?

### Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Impact | **Medium-High** | Large NAS/HPO user base |
| Differentiation | **Medium** | Optuna + human iteration exists |
| Feasibility | **High** | Optuna integration straightforward |
| Alignment with PAOLA | **Medium** | Different domain from engineering opt |

**Recommendation**: Interesting direction but faces competition from established tools (Optuna, Ray Tune). The value-add is automating the "human in the loop" for search space refinement.

---

## Comparison Summary

| Aspect | Direction A (Engineering) | Direction B (NAS) |
|--------|--------------------------|-------------------|
| Node represents | Single evaluation | Search space + Optuna study |
| Evaluation cost | Very high (hours) | Moderate (minutes-hours) |
| Graph size | Tens of nodes | Few nodes (3-5 typically) |
| Agent value | Cost avoidance | Smart space generation |
| Key innovation | Per-eval agent control | Dynamic search spaces |
| Competition | Low (novel approach) | High (Optuna, Ray Tune) |

---

## Shared Infrastructure Needs

Both directions would benefit from:

1. **Flexible node polymorphism**: Current `NodeSummary` is optimizer-focused
2. **Richer edge semantics**: Domain-specific edge types
3. **Intermediate state storage**: Surrogate models, Optuna studies
4. **Graph compression**: Summarization for large graphs (Direction A especially)

---

## Overall Recommendation

**Priority**: Direction A > Direction B

**Rationale**:
- Direction A aligns with PAOLA's core value proposition (expensive simulations)
- Direction A has clearer differentiation (no existing tools offer this)
- Direction A has more concrete cost justification
- Direction B is valuable but faces established competition

**Note**: These are brainstorming ideas for potential future extensions, not scheduled for near-term development.
