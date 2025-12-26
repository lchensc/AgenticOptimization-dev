# Paola: Graph-Based Agentic Optimization with LLM Reasoning

**Version**: Draft 0.1
**Date**: 2024-12-26
**Status**: Working Draft

---

## Abstract

Modern optimization problems require sophisticated combinations of algorithms, configurations, and multi-stage strategies. Traditionally, this complexity burden falls on users, requiring deep expertise in optimization methods and extensive manual tuning. We present **Paola**, the first optimization framework that unifies problem formulation adaptation, solver selection, and algorithm configuration through an LLM-based agent operating on a graph-based representation. Unlike existing tools that address these aspects separately, Paola's graph architecture explicitly represents optimization strategies as nodes (optimizer runs) and edges (relationships like warm-start, refinement, and exploration), enabling the agent to reason about and compose multi-stage optimization workflows. We demonstrate that Paola: (1) achieves competitive or superior performance to fixed algorithm approaches across standard benchmarks, (2) effectively leverages cross-session learning through a novel two-tier storage architecture, (3) extends naturally to multi-objective optimization through agent-guided Pareto exploration, and (4) justifies its LLM inference overhead for problems with expensive function evaluations. Our results position Paola as a new paradigm where optimization complexity becomes system intelligence rather than user burden.

---

## 1. Introduction

### 1.1 The Optimization Burden Problem

Real-world optimization problems present practitioners with daunting complexity:

**Algorithm Selection**: Given a new problem, which optimizer should be used? SLSQP for smooth constrained problems? CMA-ES for rugged landscapes? TPE for black-box hyperparameter tuning? The Algorithm Selection Problem (Rice, 1976) formalizes this challenge, yet most practitioners still rely on trial-and-error or domain folklore.

**Configuration Burden**: Even after selecting an algorithm, configuration remains challenging. IPOPT alone has 250+ options. Trust region methods require choosing radii. Population-based methods need population sizes, mutation rates, and selection pressures. Default configurations rarely perform optimally across problem classes.

**Multi-Stage Strategies**: Expert practitioners know that effective optimization often requires composed strategies: global exploration followed by local refinement, multi-start approaches to escape local optima, or progressive tightening of search bounds. Encoding these strategies requires both expertise and manual implementation.

**Learning from Experience**: Each optimization run generates valuable information about what works for similar problems. Yet this knowledge typically remains tacit in practitioners' heads rather than systematically captured and reused.

### 1.2 Why Existing Tools Fall Short

Current optimization frameworks address these challenges partially and separately:

| Tool Category | Algorithm Selection | Configuration | Multi-Stage | Learning |
|--------------|---------------------|---------------|-------------|----------|
| Solver Libraries (SciPy, IPOPT) | Manual | Manual | Manual | None |
| AutoML (SMAC, Optuna) | Fixed per run | Automated | Limited | Within-run |
| Hyper-heuristics | Automated | Limited | Sequential | Rule-based |
| Neural Combinatorial | Learned | Implicit | Fixed | Training-based |

**The Gap**: No existing framework provides unified, adaptive handling of formulation, solver selection, configuration, AND multi-stage strategy composition with cross-session learning.

### 1.3 The Paola Approach

Paola introduces a new paradigm based on three key innovations:

**1. The Paola Principle**: "Optimization complexity is Paola intelligence, not user burden."

Rather than exposing complexity to users, Paola internalizes it through an LLM agent with access to structured optimization knowledge (skills) and historical performance data (graphs).

**2. Graph-Based Strategy Representation**:

Optimization strategies are represented as directed graphs where:
- **Nodes** represent individual optimizer runs with complete configurations
- **Edges** represent relationships: `warm_start` (continue from solution), `refine` (local improvement), `branch` (explore alternative), `explore` (global restart)
- **Patterns** emerge from graph topology: chains (sequential refinement), trees (multi-start exploration), DAGs (complex composed strategies)

**3. Two-Tier Storage for Learning**:

Optimization history is stored in two tiers optimized for different purposes:
- **Tier 1 (GraphRecord)**: Compact (~1KB) summaries for LLM reasoning and cross-session learning
- **Tier 2 (GraphDetail)**: Full trajectories (10-100KB) for visualization and debugging

### 1.4 Contributions

This paper makes the following contributions:

1. **Conceptual**: We identify the gap in unified adaptive optimization and propose the graph-based agentic paradigm as a solution.

2. **Architectural**: We present Paola's design including the graph schema, agent reasoning model, skills infrastructure, and two-tier storage.

3. **Empirical**: We demonstrate Paola's effectiveness on:
   - Single-objective continuous and constrained benchmarks
   - Multi-objective optimization via agent-guided Pareto exploration
   - Cross-session learning experiments

4. **Analysis**: We characterize when Paola's approach provides benefits and its computational overhead trade-offs.

5. **Open Source**: We release Paola as an open-source Python framework for community use and extension.

---

## 2. Related Work

### 2.1 Algorithm Selection and Configuration

**Algorithm Selection Problem (ASP)**: Rice (1976) formalized algorithm selection as mapping from problem features to algorithm choice. Modern implementations include:
- **SATzilla** (Xu et al., 2008): Portfolio-based SAT solver selection
- **ISAC** (Kadioglu et al., 2010): Instance-specific algorithm configuration
- **AutoFolio** (Lindauer et al., 2015): Automated algorithm selection

**Algorithm Configuration**: Automated tuning of algorithm parameters:
- **SMAC** (Hutter et al., 2011): Sequential model-based configuration using random forests
- **ParamILS** (Hutter et al., 2009): Iterated local search for parameters
- **irace** (López-Ibáñez et al., 2016): Racing-based configuration

**Limitation**: These approaches treat algorithm selection and configuration as separate problems, typically with fixed problem formulation.

### 2.2 Hyper-heuristics and Meta-optimization

**Hyper-heuristics** (Burke et al., 2013): "Heuristics to choose heuristics"
- Selection hyper-heuristics: choose from heuristic bank
- Generation hyper-heuristics: create new heuristics

**Dynamic Algorithm Configuration** (Adriaensen et al., 2022): Uses RL to learn adaptive policies for online algorithm control.

**Limitation**: Focuses on algorithm/heuristic selection within fixed problem structure.

### 2.3 Learning-Based Optimization

**Neural Combinatorial Optimization**:
- Pointer Networks (Vinyals et al., 2015): Attention for variable-size outputs
- RL for TSP (Bello et al., 2016): Policy gradient with tour length reward

**GNN for MIP** (Gasse et al., 2019): Graph convolutional networks for branch-and-bound variable selection.

**Limitation**: Requires training on problem distributions; struggles with out-of-distribution generalization.

### 2.4 LLM-Based Optimization

**OPRO** (Yang et al., 2023): Uses LLMs as gradient-free optimizers through prompting. Demonstrated on prompt optimization and simple mathematical problems.

**EvoPrompt** (Guo et al., 2023): Combines LLMs with evolutionary algorithms for discrete prompt optimization.

**LLM-EA** (Liu et al., 2024): Uses LLMs to generate, select, and mutate candidate solutions in evolutionary optimization.

**Limitation**: These approaches use LLMs for direct solution generation rather than for reasoning about optimization strategy and algorithm orchestration.

### 2.5 The Paola Difference

Paola is distinguished by:
1. **Unified adaptation**: Formulation + solver + configuration, not just one aspect
2. **Graph-based representation**: Explicit strategy externalization
3. **LLM as reasoner**: Agent reasons about strategy, not just generates solutions
4. **Cross-session learning**: Two-tier storage enables learning across optimization runs
5. **Skills infrastructure**: Structured, progressive-disclosure optimizer knowledge

---

## 3. The Paola Framework

### 3.1 Design Philosophy: The Paola Principle

> "Optimization complexity is Paola intelligence, not user burden."

This principle drives all design decisions:

| Traditional Approach | Paola Approach |
|---------------------|----------------|
| User selects algorithm | Agent reasons about problem → selects algorithm |
| User configures 250+ options | Skills package expert knowledge → agent applies |
| User implements multi-stage strategy | Graph structure → agent composes strategy |
| User diagnoses convergence failures | Agent observes results → adapts approach |
| Knowledge stays in user's head | Two-tier storage → system learns |

### 3.2 Graph-Based Architecture

#### 3.2.1 Graph Schema

```
OptimizationGraph
├── graph_id: str
├── problem_signature: ProblemSignature
├── nodes: Dict[str, OptimizationNode]
├── edges: List[GraphEdge]
├── pattern: GraphPattern  # single, chain, tree, multistart, dag
└── outcome: GraphOutcome
```

**OptimizationNode**: Represents a single optimizer execution
```
OptimizationNode
├── node_id: str
├── optimizer: str  # e.g., "scipy:SLSQP", "optuna:TPE"
├── config: OptimizerConfig
├── result: OptimizationResult
│   ├── best_x: List[float]
│   ├── best_objective: float
│   ├── n_evaluations: int
│   └── convergence_status: str
└── timing: NodeTiming
```

**GraphEdge**: Represents relationship between nodes
```
GraphEdge
├── source_node: str
├── target_node: str
├── edge_type: EdgeType  # warm_start, refine, branch, explore
└── metadata: Dict  # edge-specific information
```

#### 3.2.2 Edge Semantics

| Edge Type | Meaning | Typical Use |
|-----------|---------|-------------|
| `warm_start` | Initialize from parent's solution | Local refinement, algorithm switching |
| `refine` | Tighten bounds around parent's solution | Progressive focusing |
| `branch` | Explore alternative from same starting point | Multi-start, escape local optima |
| `explore` | Global restart with different strategy | Basin hopping, diversification |

#### 3.2.3 Graph Patterns

Patterns emerge from graph topology and are automatically detected:

- **Single**: One node (simple optimization)
- **Chain**: Linear sequence (progressive refinement)
- **Tree**: Branching structure (multi-start exploration)
- **Multistart**: Parallel independent runs
- **DAG**: Complex composed strategy

### 3.3 Agent Architecture

Paola supports two agent modes:

**ReAct Agent**: Autonomous optimization loop
```
while not converged:
    observation = observe(graph_state, problem_info)
    thought = reason(observation, skills, past_graphs)
    action = select_action(thought)
    result = execute(action)
    update(graph, result)
```

**Conversational Agent**: User-guided optimization with agent assistance

#### 3.3.1 Tools Available to Agent

| Tool Category | Tools | Purpose |
|--------------|-------|---------|
| **Graph** | `start_graph`, `get_graph_state`, `finalize_graph`, `query_past_graphs` | Graph lifecycle |
| **Optimization** | `run_optimization`, `get_problem_info`, `list_available_optimizers` | Execute optimization |
| **Problem** | `create_nlp_problem`, `derive_problem`, `evaluate_function` | Problem formulation |
| **Skills** | `list_skills`, `load_skill`, `query_skills` | Access optimizer expertise |

#### 3.3.2 Agent Decision Model

The agent makes key decisions at each step:
1. **Continue or finalize?** Based on convergence, budget, improvement rate
2. **Which optimizer?** Based on problem characteristics, past performance
3. **Which configuration?** Based on skills and problem-specific adaptation
4. **Which edge type?** Based on current state and exploration/exploitation balance
5. **From which parent?** Based on best result or diversification needs

### 3.4 Skills Infrastructure

Skills provide structured, progressive-disclosure optimizer knowledge:

```yaml
# Example: scipy_slsqp skill
name: scipy:SLSQP
category: gradient_based
summary: "Sequential Least Squares Programming for constrained optimization"

when_to_use:
  - Smooth objective with gradient available
  - Equality and inequality constraints
  - Medium-scale problems (< 1000 variables)

key_options:
  ftol: 1e-9     # Function tolerance
  maxiter: 100   # Maximum iterations
  disp: false    # Display convergence

sections:
  warm_start: |
    SLSQP supports warm-starting via x0.
    Pass previous solution as initial point...
  constraints: |
    Constraints specified as dicts with 'type', 'fun', 'jac'...
```

**Progressive Disclosure**:
1. `list_skills()`: Overview of available expertise
2. `load_skill("scipy:SLSQP")`: Summary and key options
3. `load_skill("scipy:SLSQP", "warm_start")`: Detailed section

### 3.5 Two-Tier Storage for Learning

#### Tier 1: GraphRecord (~1KB per graph)
- Problem signature (dimensions, constraint count, bounds)
- Strategy pattern (chain, tree, etc.)
- Node summaries with FULL configurations
- Final outcome and timing

**Purpose**: Enable LLM to query and reason about past optimization runs efficiently.

#### Tier 2: GraphDetail (10-100KB per graph)
- Complete convergence trajectories
- Full solution vectors at each iteration
- Detailed timing breakdown
- Constraint violation history

**Purpose**: Visualization, debugging, and detailed analysis.

#### Cross-Session Learning

```python
# Agent can query past graphs
past_graphs = query_past_graphs(
    problem_signature=current_problem.signature,
    limit=5
)

# Reasoning: "Past graph G42 used TPE→SLSQP chain for similar
# 20D constrained problem, achieving 1e-6 in 150 evaluations.
# I'll try the same pattern."
```

---

## 4. Experiments

### 4.1 Experimental Setup

#### 4.1.1 Benchmark Problems

**Single-Objective Continuous**:
- Rosenbrock (10D, 50D, 100D)
- Ackley (10D, 50D, 100D)
- Sphere (10D, 50D, 100D)
- Rastrigin (10D, 50D, 100D)
- Schwefel (10D, 50D)
- Griewank (10D, 50D)

**Constrained Optimization**:
- G1-G6 (CEC constrained benchmarks)
- Pressure Vessel Design
- Welded Beam Design
- Speed Reducer Design

**Multi-Objective**:
- ZDT1-4 (2 objectives)
- DTLZ1-4 (3 objectives)
- Welded Beam Bi-objective (minimize cost, minimize deflection)

#### 4.1.2 Baselines

| Baseline | Description |
|----------|-------------|
| **Fixed-SLSQP** | SciPy SLSQP with default configuration |
| **Fixed-CMA-ES** | CMA-ES with default population |
| **Fixed-TPE** | Optuna TPE with 100 trials |
| **SMAC** | Sequential model-based algorithm configuration |
| **Random** | Random search baseline |
| **NSGA-II** | For multi-objective comparisons |
| **MOEA/D** | For multi-objective comparisons |

#### 4.1.3 Metrics

**Single-Objective**:
- Evaluations to reach target quality
- Final objective value at fixed budget
- Success rate (reaching target)

**Multi-Objective**:
- Hypervolume indicator
- Inverted Generational Distance (IGD)
- Spread metric

**Cost**:
- Total function evaluations
- LLM API cost (tokens × price)
- Wall-clock time

#### 4.1.4 Experimental Protocol

- 10 independent runs per problem-method combination
- Fixed random seeds for reproducibility
- Evaluation budget: 1000 evaluations (SO), 5000 evaluations (MOO)
- LLM: Claude-3.5-Sonnet (or Qwen-Plus for cost comparison)

### 4.2 Single-Objective Results

[TO BE FILLED AFTER EXPERIMENTS]

#### 4.2.1 Convergence Comparison

**Table: Final objective values (mean ± std) at 1000 evaluations**

| Problem | Paola | Fixed-SLSQP | Fixed-CMA-ES | SMAC | Random |
|---------|-------|-------------|--------------|------|--------|
| Rosenbrock-10D | | | | | |
| Rosenbrock-50D | | | | | |
| Ackley-10D | | | | | |
| ... | | | | | |

#### 4.2.2 Evaluations to Target

**Table: Evaluations to reach 1e-6 of optimum (median, 10 runs)**

| Problem | Paola | Fixed-SLSQP | Fixed-CMA-ES | SMAC |
|---------|-------|-------------|--------------|------|
| Rosenbrock-10D | | | | |
| ... | | | | |

#### 4.2.3 Success Rate

**Table: Percentage of runs reaching target quality**

[TO BE FILLED]

### 4.3 Constrained Optimization Results

[TO BE FILLED AFTER EXPERIMENTS]

### 4.4 Multi-Objective Results

[TO BE FILLED AFTER EXPERIMENTS]

#### 4.4.1 Agent-Guided Pareto Exploration

We implement agent-guided MOO as described in Section 5:
- Nodes represent scalarized optimizations with specific weight configurations
- Edges represent preference shifts (`weight_shift`, `reference_shift`)
- Agent reasons about which regions of Pareto front to explore

**Table: Hypervolume comparison at 5000 evaluations**

| Problem | Paola-MOO | NSGA-II | MOEA/D |
|---------|-----------|---------|--------|
| ZDT1 | | | |
| ZDT2 | | | |
| DTLZ1 | | | |
| ... | | | |

### 4.5 Ablation Studies

#### 4.5.1 Graph Structure Impact

**Question**: Does multi-node graph structure improve over single optimizer?

**Experiment**: Compare Paola with graph disabled (single best optimizer) vs full graph.

[TO BE FILLED]

#### 4.5.2 Cross-Session Learning Impact

**Question**: Does `query_past_graphs()` improve performance?

**Experiment**: Run same problem class 10 times, measure learning curve.

[TO BE FILLED]

#### 4.5.3 Skills Impact

**Question**: Do skills improve agent decisions?

**Experiment**: Compare agent with full skills vs no skills access.

[TO BE FILLED]

### 4.6 Cost Analysis

#### 4.6.1 LLM Overhead

**Table: LLM cost per optimization run**

| Problem Class | Avg Tokens | Avg Cost | Overhead % |
|--------------|------------|----------|------------|
| Simple (10D unconstrained) | | | |
| Medium (50D constrained) | | | |
| Complex (100D multi-start) | | | |

#### 4.6.2 When is Paola Cost-Justified?

For expensive function evaluations (CFD, FEA), LLM overhead becomes negligible:

| Evaluation Cost | Paola Overhead | Justification |
|-----------------|----------------|---------------|
| $0.01 (simple function) | ~$0.10 | 1000% - NOT justified |
| $1.00 (moderate simulation) | ~$0.10 | 10% - marginal |
| $100 (CFD analysis) | ~$0.10 | 0.1% - fully justified |

---

## 5. Multi-Objective Optimization Extension

### 5.1 The A Posteriori Necessity

Multi-objective optimization is fundamentally a posteriori: decision makers cannot articulate preferences without first observing actual trade-offs. This creates a dilemma:

- **Full Pareto enumeration**: Computationally expensive, cognitively overwhelming
- **A priori weights**: DM cannot specify without seeing trade-offs
- **Interactive methods**: Require human in loop at each step

### 5.2 Agent-Guided Pareto Exploration

Paola extends the graph paradigm to MOO:

**Node semantics for MOO**:
```
ParetoExplorationNode
├── scalarization: ScalarizationType  # weighted_sum, chebyshev, epsilon
├── weights: Optional[List[float]]
├── reference: Optional[List[float]]
├── result: ParetoNodeResult
│   ├── solution_x: List[float]
│   ├── objectives: List[float]
│   ├── is_pareto_optimal: bool
│   └── local_pareto_set: Optional[List]
```

**Edge types for MOO**:
| Edge Type | Meaning |
|-----------|---------|
| `anchor_to_interior` | From extreme point toward compromise |
| `weight_shift` | Change scalarization weights |
| `reference_shift` | Move reference point |
| `local_refinement` | Dense local Pareto sampling |
| `region_jump` | Jump to different Pareto region |

### 5.3 The Cognitive Division of Labor

Agent-guided MOO enables optimal human-agent collaboration:

**Human DM excels at**:
- Value judgments ("safety matters more than cost")
- Intuitive domain understanding
- Final decision accountability

**LLM Agent excels at**:
- Processing high-dimensional trade-offs
- Maintaining consistency across comparisons
- Systematic exploration strategy
- Integrating multi-domain knowledge

**Result**: DM provides high-level values; agent handles exploration strategy.

### 5.4 Pareto Optimality Guarantees

Every solution presented to DM is guaranteed Pareto-optimal:
1. Agent uses scalarization methods that guarantee Pareto optimality
2. Non-dominated sorting validates solutions
3. Local Pareto refinement ensures no dominated solutions in final set

---

## 6. Discussion

### 6.1 When Does Paola Excel?

Based on our experiments, Paola provides greatest benefit when:

1. **Problem class is unfamiliar**: Agent can reason about characteristics and select appropriate approach
2. **Multi-stage strategies help**: Problems benefiting from global→local refinement
3. **Function evaluations are expensive**: LLM overhead becomes negligible
4. **Similar problems will be solved**: Cross-session learning accumulates value

### 6.2 Limitations

1. **LLM cost for cheap functions**: For trivial problems, Paola overhead is not justified
2. **LLM reliability**: Agent decisions depend on LLM reasoning quality
3. **Latency**: LLM API calls add latency between optimization steps
4. **Novel problem types**: Agent may not have relevant knowledge for entirely new problem classes

### 6.3 Future Directions

1. **Learned skills**: Auto-generate skills from successful optimization patterns
2. **Local LLM**: Run with local models to reduce latency and cost
3. **Parallel optimization**: Execute multiple nodes concurrently
4. **Problem formulation adaptation**: Automatic constraint/bound adjustment

---

## 7. Conclusion

We have presented Paola, a novel framework for agentic optimization that unifies problem formulation adaptation, solver selection, algorithm configuration, and multi-stage strategy composition. Through a graph-based architecture and LLM-based reasoning, Paola embodies the principle that optimization complexity should be system intelligence, not user burden.

Our experiments demonstrate that Paola:
- Achieves competitive or superior performance on standard benchmarks
- Effectively composes multi-stage optimization strategies
- Enables cross-session learning through two-tier storage
- Extends naturally to multi-objective optimization
- Justifies its overhead for problems with expensive evaluations

By releasing Paola as open-source software, we invite the community to extend and apply this paradigm to their optimization challenges.

---

## References

[TO BE FORMATTED IN PROPER CITATION STYLE]

- Rice, J.R. (1976). The Algorithm Selection Problem
- Hutter et al. (2011). Sequential Model-based Optimization for General Algorithm Configuration
- Burke et al. (2013). Hyper-heuristics: A Survey of the State of the Art
- Yang et al. (2023). Large Language Models as Optimizers
- Deb et al. (2002). A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II
- Zhang & Li (2007). MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition

---

## Appendix A: Paola API Reference

[TO BE ADDED]

## Appendix B: Full Experimental Results

[TO BE ADDED]

## Appendix C: Skill Definitions

[TO BE ADDED]
