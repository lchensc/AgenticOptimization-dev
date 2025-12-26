# Research Analysis: Adaptive Problem Formulation in Optimization Literature

## Executive Summary

This document presents a comprehensive literature review and first-principles analysis of approaches that adapt not just solvers/parameters, but also the **problem formulation itself** during optimization. The key finding is:

> **The literature treats problem formulation and solver selection SEPARATELY. No existing framework systematically adapts BOTH in an integrated, graph-based approach like Paola.**

---

## 1. Literature Review: Existing Approaches

### 1.1 Algorithm Selection Problem (ASP) - Rice 1976

**Framework**: Four components - Problem space, Feature space, Algorithm space, Performance space

**Key Systems**:
- **SATzilla**: Portfolio-based selection for SAT using empirical hardness models
- **ISAC**: Instance-Specific Algorithm Configuration using GGA + stochastic programming
- **Instance Space Analysis (ISA)**: 2D visualization of algorithm performance across problem features

**Limitation**: Selects algorithm AFTER problem is formulated. Does not adapt formulation.

### 1.2 Algorithm Configuration (SMAC, ParamILS, GGA)

**SMAC** (Sequential Model-based Algorithm Configuration):
- Random forests as surrogate models
- Learns configurations per-instance using features
- Handles continuous, integer, categorical parameters

**ParamILS**: Iterated local search for parameter tuning

**GGA**: AND-OR trees for complex parameter dependencies

**Limitation**: Tunes parameters for a FIXED problem formulation.

### 1.3 Hyper-Heuristics

**Definition**: "Heuristics to choose heuristics" - meta-level selection/generation

**Types**:
- Selection hyper-heuristics (choose from heuristic bank)
- Generation hyper-heuristics (create new heuristics)

**Dynamic Algorithm Configuration (DAC)**: Uses RL to learn dynamic policies for online adaptation

**Limitation**: Adapts heuristics, not problem formulation.

### 1.4 Learning-Based Optimization

**Neural Combinatorial Optimization**:
- Pointer Networks (Vinyals 2015): Attention for variable-size outputs
- RL for TSP (Bello 2016): Policy gradient with tour length reward

**Graph Neural Networks for MIP**:
- Gasse 2019: GCN for branch-and-bound variable selection
- Bipartite variable-constraint graph representation
- Imitation learning from strong branching

**Limitation**: Learns policies for FIXED problem structures.

### 1.5 Decomposition Methods

**Benders Decomposition**: Divide variables, generate cuts iteratively

**Lagrangian Relaxation**: Transfer constraints to objective with multipliers

**Column Generation**: Generate variables on-demand

**Variable Neighborhood Decomposition Search (VNDS)**: Two-level VNS with decomposition

**Limitation**: These ARE reformulation strategies, but applied manually, not adaptively.

### 1.6 Multi-Fidelity Optimization

**Approaches**:
- Coarse-to-fine surrogates
- Variable fidelity models (cheap low-fidelity + expensive high-fidelity)
- Trust region methods with adaptive surrogate management

**Limitation**: Adapts model fidelity, not constraint/variable structure.

### 1.7 Constraint Handling Evolution

**Progression**: Penalty → Barrier → Augmented Lagrangian

**Constraint Aggregation**: KS function, p-norm for combining constraints

**Constraint Relaxation Hierarchies**: Allow search across infeasible regions with guidance

**Limitation**: These are techniques, not adaptive frameworks.

---

## 2. The Gap in Literature

### 2.1 What Exists (Separate Adaptation)

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRADITIONAL: SEPARATE ADAPTATION              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   User → [Formulation] → [Solver Selection] → [Configuration]   │
│           (FIXED)         (ASP/portfolio)     (SMAC/manual)      │
│                                                                  │
│   No feedback loop. No coordinated adaptation.                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 What's Missing (Unified Adaptation)

No framework in the literature provides:
1. **Integrated formulation + solver adaptation** in a single system
2. **Graph-based strategy tracking** with explicit edge semantics
3. **Cross-session learning** from formulation-solver-outcome triples
4. **Agent reasoning** about when/how to reformulate

### 2.3 Closest Existing Work

| Approach | Adapts Formulation? | Adapts Solver? | Integrated? |
|----------|---------------------|----------------|-------------|
| ASP | ❌ | ✅ | N/A |
| SMAC | ❌ | ✅ (params) | N/A |
| Decomposition | ✅ (manually) | ❌ | ❌ |
| Multi-fidelity | ✅ (fidelity only) | ❌ | ❌ |
| Hyper-heuristics | ❌ | ✅ | N/A |
| **Paola** | ✅ | ✅ | ✅ |

---

## 3. First-Principles Analysis

### 3.1 Taxonomy of Adaptation Dimensions

| Dimension | What Can Be Adapted | Granularity | Mechanism |
|-----------|---------------------|-------------|-----------|
| **Formulation** | | | |
| - Bounds | Variable search space | Per-phase | Paola: `derive_problem()` |
| - Constraints | Active constraint set | Per-phase | Agent reasoning |
| - Variables | Which to include | Per-phase | Decomposition |
| - Objective | Surrogate vs original | Per-phase | Multi-fidelity |
| **Solver** | | | |
| - Algorithm family | Gradient/Bayesian/Pop | Per-run | ASP, Paola agent |
| - Method | Specific implementation | Per-run | Paola: `optimizer="scipy:SLSQP"` |
| **Parameters** | | | |
| - Tolerances | Convergence criteria | Per-run | SMAC, Paola skills |
| - Iteration limits | Budget allocation | Per-run | Manual, agent |
| **Strategy** | | | |
| - Pattern | Chain, tree, multistart | Per-problem | Paola graph |
| - Warm-starting | Use previous solution | Per-phase | Paola edges |

### 3.2 Granularity Spectrum

```
Per-Eval → Per-Iter → Per-Phase → Per-Problem → Cross-Problem
 (finest)              (PAOLA      (PAOLA        (PAOLA
                        NODES)      GRAPHS)       SKILLS)
```

**Key Insight**: Existing approaches adapt at per-problem granularity. Paola enables per-phase adaptation through graph nodes.

### 3.3 Trade-offs

| Trade-off | Exploration | Exploitation | Paola Balance |
|-----------|-------------|--------------|---------------|
| Optimizer selection | Try diverse | Use known-good | `query_past_graphs()` + explore |
| Configuration | Wide sweeps | Use defaults | Skills provide defaults |
| Search space | Wide bounds | Narrow focus | Graph: TPE(wide) → SLSQP(narrow) |
| Strategy | Try patterns | Proven chains | Learned skills |

### 3.4 Computational Overhead

| Approach | Overhead | Justification |
|----------|----------|---------------|
| ASP | Low (features) | Pre-computed |
| SMAC | High (100s trials) | Amortized |
| Neural | Very High (training) | Repeated instances |
| **Paola (LLM)** | Medium (~$0.10/call) | CFD costs $400-1000 (0.01-0.025% overhead) |

---

## 4. Paola's Unique Contributions

### 4.1 What the Graph Structure Enables

| Capability | How Graph Enables | Traditional Alternative |
|------------|-------------------|------------------------|
| State externalization | Node IDs, not x0 values | Track internally |
| Strategy patterns | Graph topology | Implicit in code |
| Warm-start semantics | Explicit edge types | Implicit x0 passing |
| Decision tracking | `GraphDecision.reasoning` | Lost |
| Cross-run learning | `GraphRecord` | Only final result |
| Problem evolution | `derive_problem()` lineage | No tracking |

### 4.2 LLM Reasoning vs Learned Policies

| Aspect | Learned Policy | LLM Reasoning |
|--------|----------------|---------------|
| Knowledge source | Training data | World knowledge + skills |
| Generalization | In-distribution | Novel problems OK |
| Interpretability | Black-box | Natural language |
| Adaptation speed | Retrain | Immediate |
| Reasoning depth | Fixed policy | Chain-of-thought |

### 4.3 The Paola Principle

> "Optimization complexity is Paola intelligence, not user burden."

| Complexity | Traditional | Paola |
|------------|-------------|-------|
| IPOPT 250+ options | User learns | Skills package |
| Warm-start config | User sets | Edge type → auto |
| Convergence failure | User diagnoses | Agent adapts |
| Multi-start strategy | User implements | Graph pattern |

### 4.4 Two-Tier Storage for Learning

```
Tier 1: GraphRecord (~1KB) - For LLM learning
├── Problem signature (dims, constraints, bounds)
├── Strategy pattern (chain, multistart, tree)
├── Node summaries with FULL config
├── Outcomes and decisions

Tier 2: GraphDetail (10-100KB) - For visualization
├── Convergence histories
├── Full x trajectories
```

---

## 5. Comparison Matrix

| Dimension | ASP | SMAC | Hyper-H | Neural | Decomp | **Paola** |
|-----------|-----|------|---------|--------|--------|-----------|
| **Formulation** | | | | | | |
| Bounds | ❌ | ❌ | ❌ | ❌ | Manual | ✅ derive_problem() |
| Constraints | ❌ | ❌ | ❌ | ❌ | Manual | ✅ Agent reasoning |
| **Solver** | | | | | | |
| Selection | ✅ Portfolio | ❌ Fixed | ✅ Bank | ✅ Learned | ❌ | ✅ Per-node |
| Config | ❌ | ✅ Tuned | ❌ | ✅ Implicit | ❌ | ✅ Skills+agent |
| **Strategy** | | | | | | |
| Multi-phase | ❌ | ❌ | ✅ Sequential | ❌ | ✅ Fixed | ✅ Graph nodes |
| Warm-start | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ Explicit edges |
| **Learning** | | | | | | |
| From past | Features | Surrogate | Experience | Weights | ❌ | ✅ Two-tier |
| Org memory | ❌ | ❌ | ❌ | Model | ❌ | ✅ Skills |
| **Interpret** | Features | Low | Medium | None | Math | ✅ NL reasoning |

---

## 6. Positioning in Literature

```
┌─────────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION APPROACHES                       │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│   ASP        │ Algorithm    │ Hyper-       │ Neural            │
│   (1976)     │ Config       │ Heuristics   │ Combinatorial     │
│              │ (2011+)      │ (2000+)      │ (2017+)           │
├──────────────┴──────────────┴──────────────┴───────────────────┤
│                                                                 │
│  GAP: No unified framework for formulation + solver + config    │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                          ↓                                      │
│                       PAOLA                                     │
│   Graph-based multi-node optimization with LLM agent            │
│                                                                 │
│   • Formulation: derive_problem() with lineage                  │
│   • Solver: Agent selects per-node                              │
│   • Config: Skills + agent reasoning                            │
│   • Strategy: Graph patterns (chain, tree, multistart)          │
│   • Learning: Two-tier storage + learned skills                 │
│                                                                 │
│   "Optimization complexity is Paola intelligence,               │
│    not user burden."                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Key Literature Sources

### Algorithm Selection & Configuration
- Rice, J.R. (1976). "The Algorithm Selection Problem" - Foundational framework
- Xu et al. (2008). "SATzilla: Portfolio-based Algorithm Selection for SAT"
- Hutter et al. (2011). "Sequential Model-based Optimization for General Algorithm Configuration" (SMAC)
- Kadioglu et al. (2010). "ISAC: Instance-Specific Algorithm Configuration"

### Learning-Based Optimization
- Vinyals et al. (2015). "Pointer Networks" - Attention for variable-size outputs
- Bello et al. (2016). "Neural Combinatorial Optimization with RL"
- Gasse et al. (2019). "Exact Combinatorial Optimization with GCN"
- Bengio, Lodi, Prouvost (2021). "Machine Learning for Combinatorial Optimization: A Methodological Tour d'Horizon"

### Decomposition & Reformulation
- Benders (1962). Benders Decomposition
- Dantzig-Wolfe (1960). Column Generation
- Geoffrion (1974). Lagrangian Relaxation

### Landscape Analysis
- Smith-Miles & Lopes (2012). "Measuring Instance Difficulty for Combinatorial Optimization"
- Mersmann et al. (2011). "Exploratory Landscape Analysis"

### Hyper-Heuristics
- Burke et al. (2013). "Hyper-heuristics: A Survey of the State of the Art"
- Adriaensen et al. (2022). "Automated Dynamic Algorithm Configuration"

---

## 8. Conclusions

### 8.1 Literature Gap Confirmed

The optimization literature extensively covers:
- Algorithm selection (given fixed problem)
- Algorithm configuration (given fixed problem)
- Decomposition strategies (applied manually)
- Learning policies (for fixed problem structures)

**But lacks**: Unified frameworks that adaptively combine formulation changes with solver/configuration adaptation.

### 8.2 Paola's Novel Contribution

Paola fills this gap through:
1. **Graph-based architecture** externalizing strategy state
2. **Edge semantics** (warm_start, refine, branch, explore) capturing relationships
3. **LLM reasoning** enabling interpretable, generalizable decisions
4. **Two-tier storage** enabling cross-session learning
5. **Skills infrastructure** packaging expert knowledge progressively

### 8.3 Research Opportunity

Paola's approach represents a novel paradigm that could be formalized and studied:
- **Theoretical**: What are optimal graph structures for different problem classes?
- **Empirical**: How does LLM reasoning compare to learned policies?
- **Practical**: What skills/patterns transfer across domains?

---

## 9. Related Research

For multi-objective optimization (MOO) extensions of this work, see:
- [Graph-Based Multi-Objective Optimization: A Novel Paradigm](../research/20251220_graph_based_moo_paradigm.md)

---

## 10. Recommendation

This research analysis should be documented in the project for:
1. **Academic positioning** - Paola fills a genuine gap in the literature
2. **Design validation** - The graph-based approach is well-founded
3. **Future development** - Identified areas for extension
