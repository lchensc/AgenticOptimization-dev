# Paola Skills Infrastructure Design

**Date**: 2025-12-17
**Version**: 0.4.0 (proposed)
**Status**: Design Document

## 1. Executive Summary

This document defines the Skills infrastructure for Paola - a system for packaging and delivering domain expertise to the optimization agent. Skills enable Paola to become an expert in using different optimizers (IPOPT, SciPy, Optuna) and solving domain-specific problems (aerodynamics, structures, MDO).

### Key Design Decisions

1. **LLM-Agnostic**: Works with any LLM that supports function calling (Claude, GPT-4, Qwen, Gemini, open-source)
2. **Tool-Based Progressive Disclosure**: Knowledge loaded via explicit tool calls, not harness magic
3. **YAML-First Format**: Structured data with embedded markdown for prose
4. **Multi-Category**: Optimizers, domains, patterns, and learned skills
5. **Learning-Ready**: Architecture supports accumulating expertise from experience

## 2. Motivation

### 2.1 The Problem

Paola's goal is to be an expert optimization assistant. This requires:

1. **Optimizer Expertise**: Knowing IPOPT's 100+ options, when to use each, how to configure for different scenarios
2. **Domain Expertise**: Understanding aerodynamic shape optimization, structural constraints, MDO coupling
3. **Pattern Expertise**: Warm-starting strategies, multi-fidelity approaches, constraint handling
4. **Learned Expertise**: What worked for similar problems in the past

Current approaches are inadequate:

| Approach | Problem |
|----------|---------|
| Tool descriptions | Always loaded, can't fit 100+ options per optimizer |
| Hardcoded config tools | Limits agent to what we expose |
| Prompts | Static, no progressive disclosure |
| LLM training data | Stale, can't include organization-specific knowledge |

### 2.2 The Solution: Skills

Skills are **external knowledge packages** that:
- Load on-demand (token efficient)
- Are maintained as files (not code)
- Work with any capable LLM
- Support progressive disclosure
- Can be extended with learned knowledge

### 2.3 Relationship to Claude Skills

Claude Skills (Anthropic) pioneered this pattern but are Claude-specific:

| Claude Skills | Paola Skills |
|--------------|--------------|
| Harness auto-detects relevant skills | Agent calls `list_skills()` tool |
| Harness injects skill into context | Agent calls `load_skill()` tool |
| SKILL.md format | skill.yaml format |
| Claude only | Any LLM with tool support |

Paola Skills adopt the same philosophy (progressive disclosure, external knowledge) but implement it via tools for LLM-agnostic operation.

### 2.4 What Makes Paola Skills Unique

Paola Skills are NOT just a Claude Skills copycat. The file format may be similar, but Paola Skills are deeply integrated with an optimization-specialized system:

```
┌─────────────────────────────────────────────────────────────────┐
│                     CLAUDE CODE + SKILLS                         │
├─────────────────────────────────────────────────────────────────┤
│  Skills: Static knowledge packages                               │
│  ├── "How to create PDFs"                                       │
│  ├── "How to use Excel"                                         │
│  └── "How to configure IPOPT" (could be added)                  │
│                                                                  │
│  Execution: General-purpose coding, run scripts                  │
│  Memory: None (each session starts fresh)                        │
│  Learning: None (skills are static files)                        │
│  Specialization: None (general-purpose assistant)                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     PAOLA + SKILLS                               │
├─────────────────────────────────────────────────────────────────┤
│  Skills: Living knowledge system for optimization                │
│  ├── Optimizer skills with graph-aware configurations           │
│  ├── Domain skills (aerodynamics, structures, MDO)              │
│  ├── Pattern skills (warm-starting, multi-fidelity)             │
│  └── LEARNED skills (auto-generated from experience)            │
│                                                                  │
│  + GRAPH ARCHITECTURE (unique to Paola)                          │
│  ├── Multi-node optimization strategies                          │
│  ├── Warm-starting between nodes via edges                       │
│  ├── Skills provide edge-aware configurations                    │
│  └── Strategy patterns: chain, tree, multistart                  │
│                                                                  │
│  + LEARNING FROM EXPERIENCE (unique to Paola)                    │
│  ├── Two-tier storage captures full config + outcome            │
│  ├── query_past_graphs() finds similar problems                 │
│  ├── Successful strategies → learned/ skills                    │
│  └── Organizational knowledge accumulates over time              │
│                                                                  │
│  + EVALUATION MANAGEMENT (unique to Paola)                       │
│  ├── Expensive simulation caching                                │
│  ├── Skills understand evaluation costs                          │
│  └── Guidance on minimizing function evaluations                 │
│                                                                  │
│  Specialization: Optimization expert (not general-purpose)       │
└─────────────────────────────────────────────────────────────────┘
```

**Key Differentiators:**

| Aspect | Claude Skills | Paola Skills |
|--------|--------------|--------------|
| **Purpose** | General task knowledge | Optimization expertise |
| **Integration** | Standalone packages | Integrated with graphs, evaluators, learning |
| **Memory** | None (static files) | Cross-session learning from graphs |
| **Growth** | Manual updates | Auto-generated learned/ skills |
| **Configurations** | Generic instructions | Graph-edge-aware optimizer configs |
| **Cost awareness** | None | Evaluation cost guidance |
| **Specialization** | General-purpose | Optimization-first design |

**Why Use Paola Instead of Claude Code?**

Claude Code could add an "optimization skill" that teaches it about IPOPT options. But:

1. **Claude Code lacks graphs**: No concept of multi-node optimization strategies
2. **Claude Code lacks memory**: Can't learn from past optimizations
3. **Claude Code lacks evaluation management**: Doesn't understand expensive simulations
4. **Claude Code is general-purpose**: Paola is specialized, thus deeper expertise

The moat is not the skill format. The moat is **the optimization-specialized system that skills plug into**.

## 3. Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     ANY LLM                                      │
│           (Claude, GPT-4, Qwen, Gemini, Llama, ...)             │
│                 With function/tool calling                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Standard tool calls
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PAOLA AGENT LAYER                            │
│              (LangGraph/LangChain orchestration)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SKILLS TOOLS                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐  │
│  │list_skills()│ │load_skill() │ │query_skills()│ │get_config()│ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SKILLS ENGINE                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────────┐│
│  │   Loader    │ │   Index     │ │   Search (embeddings)       ││
│  └─────────────┘ └─────────────┘ └─────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SKILL FILES                                  │
│            (YAML/Markdown, version controlled)                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Progressive Disclosure Levels

```
┌─────────────────────────────────────────────────────────────────┐
│ LEVEL 1: Discovery (~50 tokens per skill)                       │
│ ─────────────────────────────────────────────────────────────── │
│ list_skills() returns:                                          │
│ - name, description, when_to_use                                │
│ - Enough for agent to decide relevance                          │
│ - Indexed at startup, always fast                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Agent: "I need IPOPT expertise"
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ LEVEL 2: Overview (~500-1000 tokens)                            │
│ ─────────────────────────────────────────────────────────────── │
│ load_skill("ipopt") returns:                                    │
│ - Full overview with capabilities                               │
│ - Integration guidance with Paola                               │
│ - List of available sections                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Agent: "I need warm-start options"
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ LEVEL 3: Deep Knowledge (~200-2000 tokens per section)          │
│ ─────────────────────────────────────────────────────────────── │
│ load_skill("ipopt", "options.warm_start") returns:              │
│ - Detailed option reference                                     │
│ - When to use each option                                       │
│ - Example values                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Integration with Paola Graph Architecture

Skills integrate with Paola's graph-based optimization:

```
┌─────────────────────────────────────────────────────────────────┐
│                     OPTIMIZATION GRAPH                           │
│                                                                  │
│   n1 (SLSQP) ──warm_start──► n2 (IPOPT) ──refine──► n3 (IPOPT) │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Agent needs to configure n2
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Agent reasoning:                                                 │
│ 1. list_skills("optimizers") → sees ipopt skill                 │
│ 2. load_skill("ipopt") → learns about warm-start support        │
│ 3. load_skill("ipopt", "configurations") → gets warm_start cfg  │
│ 4. run_optimization(config=warm_start_config)                   │
└─────────────────────────────────────────────────────────────────┘
```

## 4. Directory Structure

```
paola/
├── skills/                           # Skills root
│   ├── __init__.py                  # Skills engine
│   ├── tools.py                     # Skill tools (@tool decorated)
│   ├── loader.py                    # Skill loading logic
│   ├── index.py                     # Skill indexing and search
│   │
│   ├── registry.yaml                # Global skill index
│   │
│   ├── optimizers/                  # Category: Optimizer expertise
│   │   ├── ipopt/
│   │   │   ├── skill.yaml          # Main skill definition
│   │   │   ├── options.yaml        # Full option reference
│   │   │   ├── configurations.yaml # Pre-built configs
│   │   │   └── troubleshooting.md  # Problem-solving guide
│   │   ├── scipy/
│   │   │   ├── skill.yaml
│   │   │   └── methods/
│   │   │       ├── slsqp.yaml
│   │   │       ├── lbfgs.yaml
│   │   │       └── trust_constr.yaml
│   │   └── optuna/
│   │       ├── skill.yaml
│   │       └── samplers.yaml
│   │
│   ├── domains/                     # Category: Domain expertise
│   │   ├── aerodynamics/
│   │   │   ├── skill.yaml
│   │   │   ├── shape_optimization.md
│   │   │   └── cfd_integration.md
│   │   ├── structures/
│   │   │   ├── skill.yaml
│   │   │   └── topology_optimization.md
│   │   └── mdo/
│   │       ├── skill.yaml
│   │       └── coupling_strategies.md
│   │
│   ├── patterns/                    # Category: Optimization patterns
│   │   ├── warm_starting/
│   │   │   └── skill.yaml
│   │   ├── multi_fidelity/
│   │   │   └── skill.yaml
│   │   └── constraint_handling/
│   │       └── skill.yaml
│   │
│   └── learned/                     # Category: Learned from experience
│       ├── strategies/             # Successful optimization strategies
│       └── failures/               # Failure patterns to avoid
```

## 5. File Format Specifications

### 5.1 skill.yaml (Main Skill Definition)

```yaml
# === METADATA (Level 1 - Always indexed) ===
name: ipopt                          # Unique identifier
version: "1.0.0"                     # Skill version
category: optimizers                 # Category for organization
author: paola-team                   # Optional

description: >
  IPOPT (Interior Point OPTimizer) for large-scale nonlinear optimization.
  Best for problems with many constraints, supports warm-starting.

when_to_use:
  - Large-scale problems (1000+ variables)
  - Problems with equality and inequality constraints
  - When warm-starting from previous solutions
  - When high accuracy is required

when_not_to_use:
  - Simple bound-constrained problems (use L-BFGS-B instead)
  - Black-box functions without gradients (use Optuna instead)
  - Very small problems (SciPy has less overhead)

keywords:
  - interior-point
  - constrained
  - large-scale
  - warm-start
  - NLP

# === OVERVIEW (Level 2 - Loaded on skill activation) ===
overview: |
  ## IPOPT Overview

  IPOPT implements a primal-dual interior-point method with a filter
  line-search for nonlinear programming.

  ### Key Capabilities

  - **Constraint Handling**: Equality and inequality constraints via barrier method
  - **Warm-Starting**: Critical for Paola's graph-based multi-node optimization
  - **Linear Solvers**: MUMPS (default), MA57, MA97, Pardiso for large sparse systems
  - **Scaling**: Automatic gradient-based scaling or user-provided
  - **Hessian Options**: Exact, limited-memory BFGS, or approximation

  ### Integration with Paola

  When using `run_optimization(optimizer="ipopt", ...)`:

  1. **Warm-starting**: If using `init_strategy="warm_start"`, you MUST also set
     `warm_start_init_point: "yes"` in the config. The graph edge provides x0,
     but IPOPT needs this flag to initialize multipliers properly.

  2. **No Hessian**: If your evaluator doesn't provide Hessian, set
     `hessian_approximation: "limited-memory"` and consider increasing
     `limited_memory_max_history` to 10-20.

  3. **Large-scale**: For 1000+ variables, ensure `linear_solver: "mumps"`
     and consider `nlp_scaling_method: "gradient-based"`.

  ### Option Categories

  Detailed options available via `load_skill("ipopt", "options.<category>")`:

  | Category | Key Options | Use Case |
  |----------|-------------|----------|
  | termination | tol, max_iter, acceptable_tol | Convergence control |
  | warm_start | warm_start_init_point, warm_start_bound_push | Continuing from parent |
  | scaling | nlp_scaling_method, obj_scaling_factor | Problem conditioning |
  | hessian | hessian_approximation, limited_memory_max_history | Second-order info |
  | barrier | mu_strategy, mu_init | Interior-point method |
  | linear_solver | linear_solver, linear_system_scaling | Matrix factorization |
  | output | print_level, output_file | Debugging |

# === RESOURCES (Level 3 - Loaded on demand) ===
resources:
  options: options.yaml              # Full option reference
  configurations: configurations.yaml # Pre-built configs
  troubleshooting: troubleshooting.md # Problem-solving

# === RELATIONSHIPS ===
related_skills:
  - scipy                            # Alternative optimizer
  - warm_starting                    # Pattern skill
  - constraint_handling              # Pattern skill

# === PAOLA INTEGRATION (Unique to Paola Skills) ===
paola:
  # Optimizer mapping
  optimizer_name: ipopt              # Maps to run_optimization(optimizer=...)
  backend: IPOPTBackend              # Backend class
  requires_gradient: preferred       # required, preferred, optional, not_used
  supports_constraints: true
  supports_warm_start: true

  # Graph-aware configurations (unique to Paola)
  graph_integration:
    edge_configurations:
      warm_start:
        description: "Config when edge_type='warm_start'"
        required_options:
          warm_start_init_point: "yes"
        recommended_options:
          warm_start_bound_push: 1.0e-6
          mu_strategy: "adaptive"
      refine:
        description: "Config when edge_type='refine'"
        required_options:
          warm_start_init_point: "yes"
        recommended_options:
          tol: 1.0e-10
          max_iter: 1000
      explore:
        description: "Config for independent exploration"
        recommended_options:
          tol: 1.0e-4
          max_iter: 100

    typical_patterns:
      - pattern: chain
        description: "Sequential refinement: global → local"
        example: "optuna:TPE → ipopt (warm_start) → ipopt (refine)"
      - pattern: multistart
        description: "Multiple independent starts"
        example: "ipopt from x1, ipopt from x2, ipopt from x3"

  # Evaluation cost awareness (unique to Paola)
  evaluation_guidance:
    typical_cost: medium             # low, medium, high, very_high
    gradient_cost: similar           # free, cheap, similar, expensive, unavailable
    hessian_cost: unavailable        # free, cheap, similar, expensive, unavailable
    recommended_settings:
      no_gradient:
        hessian_approximation: "limited-memory"
        limited_memory_max_history: 10
      expensive_evals:
        description: "Minimize evaluations for expensive simulations"
        mu_strategy: "adaptive"
        tol: 1.0e-6

  # Learning hooks (unique to Paola)
  learning:
    track_options:                   # Options to capture for learning
      - tol
      - max_iter
      - mu_strategy
      - hessian_approximation
      - warm_start_init_point
    success_indicators:              # What indicates success for this optimizer
      - convergence_achieved: true
      - constraint_violation: "<1e-6"
    failure_patterns:                # Common failure modes to learn from
      - pattern: "max_iter_reached"
        likely_cause: "Poor scaling or difficult problem"
        suggested_action: "Try adaptive mu_strategy, increase limited_memory_max_history"
      - pattern: "restoration_failed"
        likely_cause: "Infeasible problem or bad starting point"
        suggested_action: "Check constraints, try infeasible_start config"
```

### 5.2 options.yaml (Option Reference)

```yaml
# paola/skills/optimizers/ipopt/options.yaml
# Full IPOPT option reference

termination:
  tol:
    type: float
    default: 1.0e-8
    range: [0, .inf]
    description: >
      Desired convergence tolerance (relative). The algorithm terminates
      when the scaled NLP error becomes smaller than this value.
    when_to_adjust: >
      - Tighten to 1e-10 for high-accuracy engineering design
      - Loosen to 1e-4 for quick exploration or noisy functions
    example_values:
      high_accuracy: 1.0e-10
      standard: 1.0e-8
      exploration: 1.0e-4

  max_iter:
    type: int
    default: 3000
    range: [0, .inf]
    description: Maximum number of iterations.
    when_to_adjust: >
      - Increase for complex problems with many local minima
      - Decrease for quick feasibility checks
    example_values:
      quick: 100
      standard: 3000
      thorough: 10000

  acceptable_tol:
    type: float
    default: 1.0e-6
    range: [0, .inf]
    description: >
      Acceptable convergence tolerance. If algorithm cannot achieve tol,
      it may terminate with this looser tolerance after acceptable_iter
      iterations at acceptable level.
    when_to_adjust: >
      Set between tol and 1e-4 to allow graceful termination for
      difficult problems.

  max_wall_time:
    type: float
    default: 1.0e20
    range: [0, .inf]
    description: Maximum wall-clock time in seconds.
    when_to_adjust: Use for time-limited optimization runs.

warm_start:
  warm_start_init_point:
    type: string
    options: ["yes", "no"]
    default: "no"
    description: >
      Warm-start initialization. When "yes", IPOPT expects initial values
      for primal variables, bound multipliers, and constraint multipliers.
    when_to_use: >
      CRITICAL: Set to "yes" when using Paola's init_strategy="warm_start".
      The graph edge provides x0, but this flag tells IPOPT to also
      initialize dual variables properly.
    paola_integration: >
      When edge_type is "warm_start" or "refine", this should be "yes".
    related_options:
      - warm_start_bound_push
      - warm_start_bound_frac
      - warm_start_mult_bound_push

  warm_start_bound_push:
    type: float
    default: 0.001
    range: [0, 0.5]
    description: >
      Minimum distance from initial point to bounds during warm start.
    when_to_adjust: >
      - Decrease to 1e-6 if parent solution is well within bounds
      - Increase if getting bound violations at start

  warm_start_mult_init_max:
    type: float
    default: 1.0e6
    range: [0, .inf]
    description: Maximum initial value for equality constraint multipliers.
    when_to_adjust: >
      Increase if warm-start fails with large multiplier values from parent.

scaling:
  nlp_scaling_method:
    type: string
    options: ["none", "user-scaling", "gradient-based", "equilibration-based"]
    default: "gradient-based"
    description: Method for scaling the NLP.
    when_to_use: >
      - "gradient-based": Good default for most problems
      - "user-scaling": When you provide scaling factors
      - "equilibration-based": For poorly scaled problems
      - "none": When problem is already well-scaled

  obj_scaling_factor:
    type: float
    default: 1.0
    range: [-.inf, .inf]
    description: Scaling factor for the objective function.
    when_to_adjust: >
      Set to -1 for maximization. Adjust magnitude if objective is
      very large or very small compared to constraints.

  nlp_scaling_max_gradient:
    type: float
    default: 100.0
    range: [0, .inf]
    description: Maximum gradient after scaling.
    when_to_adjust: >
      Increase if seeing "scaled gradient too large" warnings.

hessian:
  hessian_approximation:
    type: string
    options: ["exact", "limited-memory"]
    default: "exact"
    description: How to obtain second-order information.
    when_to_use: >
      - "exact": When evaluator provides Hessian (fastest convergence)
      - "limited-memory": When no Hessian available (most common in Paola)
    paola_integration: >
      Most Paola evaluators don't provide Hessians. Use "limited-memory"
      as default unless your evaluator explicitly supports Hessian.

  limited_memory_max_history:
    type: int
    default: 6
    range: [0, .inf]
    description: Number of past iterations for L-BFGS approximation.
    when_to_adjust: >
      - Increase to 10-20 for better Hessian approximation
      - Decrease if memory is constrained
    recommendation: 10-20 for engineering optimization

barrier:
  mu_strategy:
    type: string
    options: ["monotone", "adaptive"]
    default: "monotone"
    description: Update strategy for barrier parameter.
    when_to_use: >
      - "adaptive": More robust, better for difficult problems
      - "monotone": Faster when problem is well-behaved
    recommendation: >
      Start with "adaptive" for robustness. Switch to "monotone" if
      convergence is smooth and you want speed.

  mu_init:
    type: float
    default: 0.1
    range: [0, .inf]
    description: Initial value of the barrier parameter.
    when_to_adjust: >
      - Decrease for problems starting near solution
      - Increase for infeasible starting points

linear_solver:
  linear_solver:
    type: string
    options: ["mumps", "ma27", "ma57", "ma77", "ma86", "ma97", "pardiso", "pardisomkl"]
    default: "mumps"
    description: Linear solver for the KKT system.
    when_to_use: >
      - "mumps": Good default, freely available
      - "ma57": Faster for many problems (requires HSL)
      - "ma97": Best for very large problems (requires HSL)
      - "pardiso": Good parallel performance (requires license)
    paola_integration: >
      MUMPS is included with cyipopt. HSL solvers require separate
      installation but can be 2-5x faster.

  linear_system_scaling:
    type: string
    options: ["none", "mc19", "slack-based"]
    default: "mc19"
    description: Scaling method for linear system.

output:
  print_level:
    type: int
    default: 5
    range: [0, 12]
    description: Output verbosity level.
    when_to_adjust: >
      - 0: No output (production)
      - 5: Standard progress (default)
      - 8-12: Detailed debugging
    paola_integration: Use 0 for clean Paola output.

  output_file:
    type: string
    default: ""
    description: File for detailed output.
    when_to_adjust: Set for debugging problematic optimizations.
```

### 5.3 configurations.yaml (Pre-built Configs)

```yaml
# paola/skills/optimizers/ipopt/configurations.yaml
# Pre-built configurations for common scenarios

default:
  description: Sensible defaults for general use
  options:
    print_level: 0
    hessian_approximation: "limited-memory"
    limited_memory_max_history: 10
    mu_strategy: "adaptive"

high_accuracy:
  description: For precision-critical engineering design
  when_to_use: >
    Final design optimization, validation runs, when small improvements matter
  options:
    tol: 1.0e-10
    acceptable_tol: 1.0e-8
    max_iter: 5000
    hessian_approximation: "limited-memory"
    limited_memory_max_history: 20
    mu_strategy: "adaptive"
    print_level: 0

warm_start_from_parent:
  description: Continue from parent node's solution
  when_to_use: >
    Graph edges with edge_type="warm_start" or "refine".
    Parent node provides x0; this config ensures proper initialization.
  options:
    warm_start_init_point: "yes"
    warm_start_bound_push: 1.0e-6
    warm_start_bound_frac: 1.0e-6
    warm_start_mult_init_max: 1.0e8
    hessian_approximation: "limited-memory"
    mu_strategy: "adaptive"
    print_level: 0
  paola_note: >
    Use with run_optimization(..., init_strategy="warm_start",
    parent_node="n1", edge_type="warm_start")

large_scale:
  description: For problems with 1000+ variables
  when_to_use: Large-scale optimization, shape optimization, topology
  options:
    hessian_approximation: "limited-memory"
    limited_memory_max_history: 20
    linear_solver: "mumps"
    nlp_scaling_method: "gradient-based"
    mu_strategy: "adaptive"
    max_iter: 5000
    print_level: 0

quick_exploration:
  description: Fast initial search with loose tolerances
  when_to_use: >
    First node in optimization graph, design space exploration,
    when you want quick feedback before detailed optimization
  options:
    tol: 1.0e-4
    acceptable_tol: 1.0e-3
    max_iter: 100
    hessian_approximation: "limited-memory"
    mu_strategy: "monotone"
    print_level: 0

infeasible_start:
  description: When starting from infeasible point
  when_to_use: >
    Starting point violates constraints, or expect infeasibility
  options:
    mu_init: 1.0
    mu_strategy: "adaptive"
    hessian_approximation: "limited-memory"
    constr_viol_tol: 1.0e-4
    print_level: 0

robust_convergence:
  description: Maximum robustness for difficult problems
  when_to_use: >
    When standard settings fail to converge, highly nonlinear problems
  options:
    mu_strategy: "adaptive"
    mu_init: 0.1
    hessian_approximation: "limited-memory"
    limited_memory_max_history: 20
    max_iter: 10000
    acceptable_iter: 15
    acceptable_tol: 1.0e-4
    print_level: 0
```

### 5.4 registry.yaml (Global Index)

```yaml
# paola/skills/registry.yaml
# Global skill index for fast discovery

version: "1.0"
generated: "2025-12-17"

categories:
  optimizers:
    description: Expertise for optimization backends
    skills:
      - ipopt
      - scipy
      - optuna

  domains:
    description: Domain-specific optimization expertise
    skills:
      - aerodynamics
      - structures
      - mdo

  patterns:
    description: Optimization patterns and strategies
    skills:
      - warm_starting
      - multi_fidelity
      - constraint_handling

  learned:
    description: Knowledge learned from past optimizations
    skills: []  # Dynamically populated

# Quick reference for skill discovery
skills:
  ipopt:
    path: optimizers/ipopt
    description: Interior-point optimizer for large-scale NLP
    keywords: [interior-point, constrained, large-scale, warm-start]

  scipy:
    path: optimizers/scipy
    description: SciPy optimization methods (SLSQP, L-BFGS-B, etc.)
    keywords: [gradient, bound-constrained, small-scale]

  optuna:
    path: optimizers/optuna
    description: Bayesian optimization with TPE, CMA-ES samplers
    keywords: [bayesian, black-box, derivative-free, hyperparameter]

  aerodynamics:
    path: domains/aerodynamics
    description: Aerodynamic shape optimization expertise
    keywords: [CFD, drag, lift, airfoil, wing]

  structures:
    path: domains/structures
    description: Structural optimization expertise
    keywords: [topology, stress, compliance, FEA]

  mdo:
    path: domains/mdo
    description: Multi-disciplinary design optimization
    keywords: [coupling, aerostructures, system-level]

  warm_starting:
    path: patterns/warm_starting
    description: Strategies for warm-starting optimization
    keywords: [continuation, refinement, graph-edge]
```

## 6. Tool Interface Specification

### 6.1 list_skills

```python
@tool
def list_skills(category: str = None) -> str:
    """
    List available skills with their descriptions.

    This is the entry point for discovering what expertise Paola has.
    Call this first to see what skills are available, then use
    load_skill() to get detailed knowledge.

    Args:
        category: Optional filter
            - "optimizers": Optimizer expertise (IPOPT, SciPy, Optuna)
            - "domains": Domain expertise (aerodynamics, structures, MDO)
            - "patterns": Optimization patterns (warm-starting, multi-fidelity)
            - "learned": Knowledge from past optimizations
            - None: List all skills

    Returns:
        Formatted list of skills with:
        - name: Skill identifier
        - description: What the skill provides
        - when_to_use: When this skill is relevant

    Example:
        list_skills()                    # All skills
        list_skills("optimizers")        # Just optimizer skills
        list_skills("domains")           # Just domain skills
    """
```

### 6.2 load_skill

```python
@tool
def load_skill(skill_name: str, section: str = "overview") -> str:
    """
    Load detailed knowledge from a skill.

    Skills use progressive disclosure - start with "overview" and
    drill down to specific sections as needed.

    Args:
        skill_name: Skill to load (e.g., "ipopt", "aerodynamics")
        section: Which section to load
            - "overview": Main guidance and capabilities (default)
            - "options": Full option/parameter reference
            - "options.<category>": Specific category (e.g., "options.warm_start")
            - "configurations": Pre-built configurations
            - "troubleshooting": Problem-solving guidance
            - "examples": Usage examples

    Returns:
        Skill content formatted for LLM consumption

    Example:
        load_skill("ipopt")                         # Overview
        load_skill("ipopt", "options")              # All options
        load_skill("ipopt", "options.warm_start")   # Just warm-start options
        load_skill("ipopt", "configurations")       # Pre-built configs
        load_skill("ipopt", "troubleshooting")      # Problem-solving
    """
```

### 6.3 query_skills

```python
@tool
def query_skills(query: str, limit: int = 3) -> str:
    """
    Search across all skills for relevant knowledge.

    Use this when you're not sure which skill has the information,
    or to find related knowledge across multiple skills.

    Args:
        query: Natural language query
        limit: Maximum number of results (default: 3)

    Returns:
        Relevant skill sections matching the query, ranked by relevance

    Example:
        query_skills("how to handle infeasible starting point")
        query_skills("warm-start configuration")
        query_skills("aerodynamic shape optimization constraints")
    """
```

### 6.4 get_skill_configuration

```python
@tool
def get_skill_configuration(skill_name: str, config_name: str) -> str:
    """
    Get a pre-built configuration from a skill.

    Skills include tested configurations for common scenarios.
    The returned config can be used directly with run_optimization().

    Args:
        skill_name: Skill name (e.g., "ipopt", "scipy")
        config_name: Configuration name (e.g., "warm_start_from_parent")

    Returns:
        JSON configuration string ready for run_optimization()

    Example:
        config = get_skill_configuration("ipopt", "warm_start_from_parent")
        run_optimization(
            graph_id="g1",
            optimizer="ipopt",
            config=config,
            init_strategy="warm_start",
            parent_node="n1",
            edge_type="warm_start"
        )
    """
```

## 7. Skills Engine Implementation

### 7.1 Module Structure

```python
# paola/skills/__init__.py

from .tools import (
    list_skills,
    load_skill,
    query_skills,
    get_skill_configuration,
)
from .loader import SkillLoader
from .index import SkillIndex

__all__ = [
    "list_skills",
    "load_skill",
    "query_skills",
    "get_skill_configuration",
    "SkillLoader",
    "SkillIndex",
]
```

### 7.2 SkillLoader

```python
# paola/skills/loader.py

class SkillLoader:
    """Loads and parses skill files."""

    def __init__(self, skills_root: Path):
        self.skills_root = skills_root
        self._cache = {}  # Cache loaded skills

    def load_metadata(self, skill_name: str) -> dict:
        """Load Level 1 metadata (name, description, when_to_use)."""

    def load_overview(self, skill_name: str) -> str:
        """Load Level 2 overview content."""

    def load_section(self, skill_name: str, section: str) -> str:
        """Load Level 3 specific section."""

    def load_configuration(self, skill_name: str, config_name: str) -> dict:
        """Load a pre-built configuration."""

    def list_configurations(self, skill_name: str) -> list[str]:
        """List available configurations for a skill."""
```

### 7.3 SkillIndex

```python
# paola/skills/index.py

class SkillIndex:
    """Indexes skills for discovery and search."""

    def __init__(self, skills_root: Path):
        self.skills_root = skills_root
        self._index = None
        self._embeddings = None  # For semantic search

    def build_index(self):
        """Build/rebuild the skill index from registry.yaml and skill files."""

    def list_skills(self, category: str = None) -> list[dict]:
        """List skills, optionally filtered by category."""

    def search(self, query: str, limit: int = 3) -> list[dict]:
        """Semantic search across skill content."""

    def get_skill_path(self, skill_name: str) -> Path:
        """Get filesystem path for a skill."""
```

## 8. Integration Points

### 8.1 With run_optimization

Skills provide configurations that integrate with `run_optimization`:

```python
# Agent workflow
config = get_skill_configuration("ipopt", "warm_start_from_parent")

run_optimization(
    graph_id="g1",
    optimizer="ipopt",
    config=config,  # From skill
    init_strategy="warm_start",
    parent_node="n1",
    edge_type="warm_start"
)
```

### 8.2 With Graph Storage (Learning)

Successful graph outcomes can generate learned skills:

```python
# After graph finalization
if graph.success and graph.outcome_quality > threshold:
    # Extract strategy as learned skill
    learned_skill = extract_strategy_skill(graph)
    save_to_learned_skills(learned_skill)
```

### 8.3 With Knowledge Base (RAG)

Skills can be indexed for RAG retrieval:

```python
# query_skills uses embedding search
results = query_skills("warm-start for constrained optimization")
# Returns relevant sections from multiple skills
```

### 8.4 With Two-Tier Graph Storage

Skills leverage the two-tier storage architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│ TIER 1: GraphRecord (~1KB) - What LLM learns from               │
├─────────────────────────────────────────────────────────────────┤
│ - Problem signature (n_vars, n_constraints, bounds)             │
│ - Strategy pattern (chain, multistart, tree)                    │
│ - Node summaries with FULL optimizer config ← Skills use this   │
│ - Start/best objectives per node                                │
│ - Outcome and decisions                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ query_past_graphs() retrieves
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Skills query past graphs to find:                                │
│ - What optimizer configs worked for similar problems            │
│ - What graph patterns succeeded (chain vs multistart)           │
│ - Common failure modes and how they were resolved               │
└─────────────────────────────────────────────────────────────────┘
```

## 9. Learning Architecture (Unique to Paola)

The `learned/` skill category is what truly differentiates Paola from Claude Skills. These are not static files - they are **automatically generated from successful optimizations**.

### 9.1 Learning Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     OPTIMIZATION EXECUTION                       │
│                                                                  │
│   Graph created → Nodes executed → Graph finalized               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ On successful finalization
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LEARNING EXTRACTION                          │
│                                                                  │
│   1. Check success criteria (from skill's learning.success_indicators) │
│   2. Extract problem signature                                   │
│   3. Extract strategy pattern (nodes, edges, configs)           │
│   4. Extract outcome metrics                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ If meets learning threshold
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LEARNED SKILL GENERATION                     │
│                                                                  │
│   Generate skill.yaml in learned/strategies/                    │
│   - problem_signature for matching                              │
│   - strategy for replication                                    │
│   - outcome for confidence                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Future optimizations
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LEARNED SKILL RETRIEVAL                      │
│                                                                  │
│   query_skills("10-variable constrained problem")               │
│   → Returns learned skill from similar past problem             │
│   → Agent can replicate successful strategy                     │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Learned Skill Schema

```yaml
# paola/skills/learned/strategies/aero_shape_opt_001.yaml
# Auto-generated from graph_0042 on 2025-12-15

name: aero_shape_opt_001
category: learned
type: strategy                       # strategy | failure_pattern | configuration

# Source tracking
source:
  graph_id: graph_0042
  created: "2025-12-15T14:30:00Z"
  user: engineering_team
  confidence: high                   # Based on outcome quality

# Problem matching (for retrieval)
problem_signature:
  n_variables: 24
  n_constraints: 12
  has_equality_constraints: true
  has_inequality_constraints: true
  bound_constrained: true
  domain_hint: aerodynamics
  keywords:
    - shape
    - drag
    - cfd

# The learned strategy
strategy:
  pattern: chain
  total_evaluations: 347
  nodes:
    - id: n1
      optimizer: optuna:TPE
      config:
        n_trials: 50
        sampler: TPE
      purpose: global_exploration
      evaluations: 50
      outcome: "Found promising region"

    - id: n2
      optimizer: ipopt
      config:
        warm_start_init_point: "yes"
        hessian_approximation: "limited-memory"
        limited_memory_max_history: 15
        mu_strategy: "adaptive"
        tol: 1.0e-6
      purpose: local_refinement
      parent: n1
      edge_type: warm_start
      evaluations: 297
      outcome: "Converged to optimum"

  edges:
    - from: n1
      to: n2
      type: warm_start
      decision: "TPE found good region, switch to gradient-based"

# Outcome metrics
outcome:
  success: true
  initial_objective: 0.0892
  final_objective: 0.0234
  improvement: 73.8%
  constraint_violation: 1.2e-8
  wall_time_seconds: 3420

# Usage guidance
when_to_use: |
  This strategy worked well for:
  - Aerodynamic shape optimization with ~20-30 variables
  - Problems with equality and inequality constraints
  - When CFD evaluations are expensive (use TPE for global first)

when_not_to_use: |
  - Very small problems (skip TPE, go direct to IPOPT)
  - Problems without gradients (can't use IPOPT refinement)
```

### 9.3 Failure Pattern Learning

Paola also learns from failures to avoid repeating mistakes:

```yaml
# paola/skills/learned/failures/ipopt_restoration_fail_001.yaml

name: ipopt_restoration_fail_001
category: learned
type: failure_pattern

source:
  graph_id: graph_0039
  created: "2025-12-14T10:15:00Z"

# Pattern identification
failure:
  optimizer: ipopt
  error_type: restoration_failed
  error_message: "Restoration phase failed"

  context:
    n_variables: 15
    n_constraints: 8
    starting_point: "random"
    config:
      warm_start_init_point: "no"
      mu_init: 0.1

# What was tried and didn't work
attempted_fixes:
  - action: "Increased max_iter"
    result: "Still failed"
  - action: "Changed mu_strategy to adaptive"
    result: "Still failed"

# What eventually worked
resolution:
  action: "Used infeasible_start configuration"
  config_changes:
    mu_init: 1.0
    constr_viol_tol: 1.0e-4
  result: "Converged successfully"

# Guidance for future
prevention: |
  When starting from random point with many constraints:
  - Use infeasible_start configuration
  - Or first run a feasibility-finding phase
  - Consider starting from a known feasible point
```

### 9.4 Learning Integration with Skills Tools

```python
@tool
def query_learned_strategies(
    n_variables: int = None,
    n_constraints: int = None,
    domain: str = None,
    keywords: list[str] = None,
    limit: int = 3
) -> str:
    """
    Find learned strategies from past successful optimizations.

    This searches the learned/ skill category for strategies
    that worked on similar problems.

    Args:
        n_variables: Approximate number of variables
        n_constraints: Approximate number of constraints
        domain: Domain hint (aerodynamics, structures, mdo)
        keywords: Problem keywords to match
        limit: Maximum results

    Returns:
        Matching learned strategies with configs and outcomes
    """

@tool
def apply_learned_strategy(
    strategy_name: str,
    graph_id: str
) -> str:
    """
    Apply a learned strategy to a new graph.

    This creates nodes and edges based on a previously
    successful strategy, adapting to the current problem.

    Args:
        strategy_name: Name of learned strategy skill
        graph_id: Graph to apply strategy to

    Returns:
        Created nodes and edges, ready for execution
    """
```

## 10. Skill Categories

### 10.1 Optimizer Skills

| Skill | Description | Key Content |
|-------|-------------|-------------|
| ipopt | Interior-point NLP | 100+ options, warm-start, scaling |
| scipy | SciPy methods | SLSQP, L-BFGS-B, trust-constr, COBYLA |
| optuna | Bayesian optimization | TPE, CMA-ES, samplers, pruners |

### 10.2 Domain Skills

| Skill | Description | Key Content |
|-------|-------------|-------------|
| aerodynamics | Aero shape optimization | CFD coupling, drag objectives, shape params |
| structures | Structural optimization | Topology, stress constraints, FEA coupling |
| mdo | Multi-disciplinary | Coupling strategies, convergence, decomposition |

### 10.3 Pattern Skills

| Skill | Description | Key Content |
|-------|-------------|-------------|
| warm_starting | Continuation strategies | When to warm-start, config patterns |
| multi_fidelity | Multi-fidelity optimization | Surrogate models, fidelity switching |
| constraint_handling | Constraint strategies | Penalty, barrier, relaxation |

### 10.4 Learned Skills (Unique to Paola)

Dynamically generated from successful optimizations (see Section 9 for details):

```yaml
# Example learned skill
name: wing_drag_minimization_v1
source: graph_0042
problem_signature:
  n_variables: 24
  n_constraints: 12
  domain: aerodynamics
strategy:
  pattern: chain
  nodes:
    - optimizer: optuna:TPE
      config: quick_exploration
      purpose: global_search
    - optimizer: ipopt
      config: warm_start_from_parent
      purpose: local_refinement
outcome:
  success: true
  final_objective: 0.0234
  total_evaluations: 347
```

## 11. Future Extensions

### 11.1 Skill Composition

Skills that combine other skills:

```yaml
name: aerostructural_optimization
type: composite
components:
  - aerodynamics
  - structures
  - mdo
```

### 11.2 Skill Versioning

Track skill evolution:

```yaml
version: "2.0.0"
changelog:
  - version: "2.0.0"
    date: "2025-12-17"
    changes: "Added warm-start configurations"
  - version: "1.0.0"
    date: "2025-11-01"
    changes: "Initial release"
```

### 11.3 Organization-Specific Skills

Support for custom skill directories:

```python
SkillLoader(
    skills_root=[
        "paola/skills",           # Built-in
        "~/.paola/skills",        # User
        "/org/paola/skills",      # Organization
    ]
)
```

### 11.4 Skill Validation

Schema validation for skill files:

```python
def validate_skill(skill_path: Path) -> list[str]:
    """Validate skill against schema, return warnings/errors."""
```

## 12. Implementation Roadmap

### Phase 1: Core Infrastructure
- [ ] Skill file format (skill.yaml, options.yaml, configurations.yaml)
- [ ] SkillLoader implementation
- [ ] Basic SkillIndex (registry-based)
- [ ] Four skill tools (list_skills, load_skill, query_skills, get_skill_configuration)
- [ ] Full option passthrough in optimizer backends (remove hardcoded mappings)

### Phase 2: Optimizer Skills
- [ ] IPOPT skill (complete with all options, graph integration, learning hooks)
- [ ] SciPy skill (all methods: SLSQP, L-BFGS-B, trust-constr, COBYLA)
- [ ] Optuna skill (TPE, CMA-ES, samplers)

### Phase 3: Learning Architecture (Unique to Paola)
- [ ] Learning pipeline: graph finalization → skill extraction
- [ ] Learned skill schema and storage
- [ ] query_learned_strategies() tool
- [ ] apply_learned_strategy() tool
- [ ] Failure pattern learning
- [ ] Integration with two-tier graph storage

### Phase 4: Search & RAG
- [ ] Embedding-based search in query_skills
- [ ] Cross-skill semantic search
- [ ] Problem signature matching for learned skills

### Phase 5: Domain Skills
- [ ] Aerodynamics skill (shape optimization, CFD coupling)
- [ ] Structures skill (topology optimization, FEA)
- [ ] MDO skill (coupling strategies, decomposition)

### Phase 6: Advanced Features
- [ ] Skill composition (composite skills)
- [ ] Organization-specific skill directories
- [ ] Skill versioning and changelog
- [ ] Skill validation against schema

## 13. Summary: Why Paola Skills Matter

Paola Skills are not just a Claude Skills copycat. They are the foundation of Paola's unique value proposition:

```
┌─────────────────────────────────────────────────────────────────┐
│                     PAOLA'S COMPETITIVE MOAT                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. SKILLS FORMAT                                                │
│     └── Similar to Claude Skills (progressive disclosure)        │
│     └── BUT: LLM-agnostic (works with any model)                │
│                                                                  │
│  2. GRAPH INTEGRATION (Unique)                                   │
│     └── Skills know about nodes, edges, warm-starting           │
│     └── Edge-aware configurations                                │
│     └── Strategy patterns (chain, multistart, tree)             │
│                                                                  │
│  3. LEARNING ARCHITECTURE (Unique)                               │
│     └── Auto-generated learned/ skills from successful graphs   │
│     └── Failure pattern learning                                 │
│     └── Organizational knowledge accumulation                    │
│     └── Cross-session memory                                     │
│                                                                  │
│  4. EVALUATION AWARENESS (Unique)                                │
│     └── Skills understand simulation costs                       │
│     └── Guidance on minimizing expensive evaluations            │
│     └── Integration with Foundry caching                         │
│                                                                  │
│  5. OPTIMIZATION SPECIALIZATION (Unique)                         │
│     └── Deep optimizer expertise (100+ IPOPT options)           │
│     └── Domain skills (aerodynamics, structures, MDO)           │
│     └── Pattern skills (warm-starting, multi-fidelity)          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**The Bottom Line:**

Claude Code + Skills = General-purpose assistant that can do optimization

Paola + Skills = **Optimization expert** that:
- Knows every optimizer option and when to use it
- Runs multi-node optimization strategies (graphs)
- Learns from every optimization to get better
- Accumulates organizational expertise over time
- Works with any capable LLM

The moat is not the skill format. The moat is the **optimization-specialized system** that skills plug into.

## 14. References

- [Claude Skills Announcement](https://claude.com/blog/skills)
- [Agent Skills Engineering Blog](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [IPOPT Options Reference](https://coin-or.github.io/Ipopt/OPTIONS.html)
- [cyipopt Documentation](https://cyipopt.readthedocs.io/en/stable/)
- [Paola Graph Architecture](./20251215_graph_based_optimization_design.md)
- [Paola Two-Tier Storage](./20251215_two_tier_graph_storage.md)
