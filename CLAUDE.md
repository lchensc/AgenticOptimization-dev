# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains **PAOLA (Platform for Agentic Optimization with Learning and Analysis)** - a next-generation engineering optimization system where an autonomous AI agent controls the optimization process, accumulates knowledge from past optimizations, and analyzes multiple runs to achieve reliable convergence.

### Core Innovation

Unlike traditional optimization platforms (HEEDS, ModeFRONTIER, Dakota, pyOptSparse, FADO) that use fixed control loops with user-configured algorithms, PAOLA provides four key innovations:

1. **Agentic Control**: Agent autonomously composes strategies using tool primitives (not fixed loops)
2. **The Paola Principle**: Optimization complexity is agent intelligence, not user burden
3. **Organizational Learning**: Knowledge base with RAG-based retrieval for warm-starting similar problems
4. **Multi-Run Analysis**: Compares multiple optimization strategies to select best approaches

**Traditional approach**: Platform prescribes the loop, user configures it (250+ IPOPT options!), no memory between runs

**PAOLA approach**: Agent controls everything, handles all configuration complexity, learns from past optimizations

The agent continuously observes optimization progress, reasons about numerical health and feasibility patterns, autonomously adapts the strategy (constraint bounds, gradient methods, exploration control), and accumulates knowledge that improves future optimizations.

### The Paola Principle

> **"Optimization complexity is Paola intelligence, not user burden."**

This principle defines PAOLA's approach to the overwhelming complexity of optimization:

| What Users Specify | What Paola Handles |
|-------------------|-------------------|
| Problem definition (objective, constraints, bounds) | Algorithm selection |
| Intent (`priority="robustness"`) | Option configuration (250+ IPOPT options) |
| | Initialization (x0, sigma, population) |
| | Convergence failure handling |
| | Warm-starting from history |

**Example**: IPOPT has ~250 options across 22 categories. SNOPT is scaling-sensitive. CMA-ES needs sigma tuning. Most users only touch 3-5 options. Paola knows which options matter for which problems - this expert knowledge is Paola's core competence.

## Repository Structure

```
AgenticOptimization/
├── paola/                                 # Main package (currently 'aopt', rename planned)
│   ├── agent/                            # ReAct agent implementation
│   ├── tools/                            # Tool primitives for agent
│   ├── storage/                          # Run storage backend
│   ├── cli/                              # Interactive CLI (Phase 2)
│   ├── knowledge/                        # Knowledge base + RAG (TODO: Phase 3)
│   └── analysis/                         # Multi-run analysis (TODO: Phase 3)
├── docs/
│   ├── agent_controlled_optimization.md  # Detailed technical design
│   ├── agentic_optimization_vision.md    # Vision and value proposition
│   ├── cli_architecture.md               # CLI design (Phase 2)
│   └── run_architecture.md               # Run-based architecture (Phase 2)
└── CLAUDE.md                             # This file
```

## Key Concepts

### Agent Autonomy vs. Fixed Loops

**Traditional platforms** (all existing software):
```python
# Fixed loop prescribed by platform
for iteration in range(max_iterations):
    design = optimizer.propose()
    result = evaluate(design)
    optimizer.update(result)
    if convergence_criteria_met():
        break
```

**Agentic platform** (this innovation):
```python
# Agent decides everything autonomously
agent = OptimizationAgent(llm="qwen-plus")
goal = "Minimize drag on transonic wing, maintain CL >= 0.5"
tools = platform.get_tools()  # optimizer_*, workflow_*, cache_*, etc.
agent.optimize(goal, tools)
# Agent composes its own strategy, no fixed loop
```

### Tool Architecture (Three-Layer Design)

Tools are organized in three layers reflecting the Paola Principle:

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

**Problem Formulation Tools** (User specifies WHAT):
- `create_nlp_problem(problem_id, objective, bounds, constraints, domain_hint)` - Define problem mathematically
- `get_problem_info(problem_id)` - Retrieve problem specification

**Optimization Execution Tools** (Intent-based):
- `run_optimization(problem_id, optimizer="auto", priority="balanced")` - Run with Paola handling details
  - `optimizer`: "auto", "gradient-based", "global", or specific like "scipy:SLSQP"
  - `priority`: "speed", "robustness", "accuracy", "balanced"
- `get_run_info(run_id)` - Get run status and results
- `get_best_solution(problem_id)` - Best solution across all runs

**Expert Escape Hatch** (Optional, for users who know exactly what they want):
- `config_scipy(config_id, algorithm, ...)` - Bypass Paola's auto-config
- `config_ipopt(config_id, ...)` - Direct IPOPT configuration

**Evaluator Tools**:
- `foundry_store_evaluator(evaluator_id, source_code)` - Register objective/constraint functions
- `foundry_list_evaluators()` - List available evaluators

**Analysis Tools** (Phase 2/3):
- `analyze_runs(run_ids)` - Compare multiple optimization strategies
- `get_all_metrics(run_id)` - Convergence, gradient, constraint metrics

### Strategic Adaptation Mechanisms

The agent makes informed decisions to adapt strategy based on continuous observation:

1. **Constraint Feasibility Management**
   - Detects repeated constraint violations (e.g., CL = 0.49 when CL >= 0.5 required)
   - Tightens constraints to force optimizer into feasible region
   - Proven pattern for handling optimizer stuck at infeasible boundaries

2. **Gradient Method Switching**
   - Monitors gradient variance/noise levels
   - Switches adjoint ↔ finite-difference when numerical noise detected
   - Proven pattern from AdjointFlow V6 (handles shock oscillations)

3. **Bayesian Exploration Control**
   - Adjusts acquisition functions (EI, UCB, PI) for exploration vs exploitation
   - Uses seed management for deterministic replay
   - Warm-starts surrogate models when changing strategies

4. **Convergence Verification**
   - Automatically runs high-fidelity verification of converged designs
   - Detects false convergence from gradient noise
   - Validates feasibility before terminating

### Evaluation Cache (Critical for Efficiency)

Engineering simulations are 10,000× more expensive than optimizer iterations:
- CFD evaluation: 4-10 CPU hours → $400-$1000
- Gradient (adjoint): 6 CPU hours → $600
- Optimizer iteration: 0.001 hours → $0.10

The cache prevents re-running expensive simulations when the optimizer revisits designs during line search or trust region adjustments.

### Agent Observation Metrics

The agent monitors every iteration:
- **Convergence health**: improvement_rate, gradient_norm, step_size
- **Numerical health**: gradient_variance, condition_number, constraint_activity
- **Optimizer health**: trust_region_size, qp_solver_success, lagrange_multipliers
- **Resource usage**: budget_used, evaluations_count

## Design Documents

### docs/architecture/tools_optimization_foundry_design.md (Primary)

**The definitive design document** for PAOLA's tool architecture:
- The Paola Principle: "Optimization complexity is agent intelligence, not user burden"
- Three-layer architecture (User Intent → Paola Intelligence → Optimizer Execution)
- Intent-based `run_optimization(optimizer="auto", priority="robustness")`
- Paola's Configuration Intelligence (algorithm selection, option configuration)
- Paola's Initialization Intelligence (x0, sigma, population handling)
- Compact bounds specification for large variable spaces
- Expert escape hatch for direct configuration

### docs/architecture/optimizer_initialization_research.md

Research on initialization handling across optimizers:
- Survey of IPOPT, SNOPT, SciPy, NLopt, CMA-ES, Optuna, pymoo
- Why initialization is agent intelligence, not user input
- Algorithm-specific defaults (gradient→center, shape_opt→zero, etc.)

### docs/architecture/optimizer_configuration_research.md

Research on configuration complexity:
- IPOPT: 250 options across 22 categories
- SNOPT: scaling sensitivity
- Why most users only touch 3-5 options
- How Paola applies expert knowledge automatically

### agent_controlled_optimization.md (50KB)

Comprehensive technical design covering:
- Core innovation: Agent autonomy vs prescribed loops
- Comparison with all major existing platforms (HEEDS, ModeFRONTIER, Dakota, pyOptSparse, FADO, Tosca)
- Practical adaptation mechanisms and the "replay constraint"
- Agent decision patterns and reasoning examples
- Complete workflow example showing agent in action

### agentic_optimization_vision.md (7KB)

High-level vision and value propositions:
- 7 core value propositions from first principles analysis
- Paradigm shift from "operator configures loops" to "director sets goals"
- Knowledge accumulation and learning organization concept

## Implementation Status

**Current state**: Design/documentation phase (no code yet)

**Planned implementation phases**:
1. **Phase 1** (2 months): Core mechanisms - evaluation cache, optimizer wrapper with checkpointing, basic agent with observation
2. **Phase 2** (2 months): Agent intelligence - full observation metrics, constraint management, gradient switching, Bayesian control
3. **Phase 3** (2 months): Validation & extension - real engineering test cases, multiple optimizers, knowledge base, production hardening

## Development Principles

When implementing this platform:

1. **The Paola Principle**: Optimization complexity is agent intelligence, not user burden. Users specify intent ("robustness"), Paola handles the 250 IPOPT options.
2. **Agent Autonomy First**: Never add a "hook" or "callback" - the agent IS the controller
3. **Intent Over Options**: Expose high-level intents (priority, optimizer family), not low-level knobs. Expert escape hatch for those who need it.
4. **Tools Not Control Flow**: Platform provides primitives, agent composes strategy
5. **Observable Everything**: Every action must be observable and explainable
6. **Strategic Restarts**: Adaptations are informed restarts from better positions, not unpredictable experiments
7. **Cache Everything**: Simulations are expensive, cache all evaluations
8. **Learn Continuously**: Every optimization adds to the knowledge base
9. **CRITICAL - Minimal Prompting**: Keep system prompts and tool schemas minimal. Trust the LLM's intelligence. Never add verbose guidance, formatting rules, or hand-holding without explicit permission. The agent must learn from experience, not from over-specified prompts.

## Key Terminology

- **The Paola Principle**: Core design philosophy - "Optimization complexity is Paola intelligence, not user burden"
- **Intent-based tools**: Tools that accept user intent (e.g., `priority="robustness"`) rather than low-level configuration
- **Expert escape hatch**: Optional tools for users who need direct control over optimizer configuration
- **Configuration intelligence**: Paola's ability to select and configure optimizer options based on problem characteristics
- **Initialization intelligence**: Paola's ability to determine optimal starting points based on algorithm type, domain, and history
- **Fixed loop**: Traditional optimization paradigm where platform controls iteration
- **Tool primitives**: Atomic operations that agent composes into strategies
- **Evaluation cache**: Storage for expensive simulation results (design → objective, gradient, constraints)
- **Strategic restart**: Informed decision to restart optimizer from current best with modified settings
- **Replay constraint**: Limitation that changing optimizer settings (like scaling) changes trajectory, preventing free experimentation
- **Gradient variance**: Metric for detecting numerical noise in gradients
- **Constraint feasibility management**: Agent's ability to detect and fix repeated constraint violations
- **Compositional strategy**: Agent-invented optimization approach combining multiple tools/algorithms
- **Agentic Learning**: Autonomous accumulation of strategic knowledge through observation and experience across optimization runs (not machine learning)
- **Knowledge base**: RAG-based storage of problem signatures, successful setups, and expert knowledge
- **Multi-run analysis**: Comparing multiple optimization strategies to identify best practices
- **Warm-starting**: Using retrieved knowledge from similar past problems to accelerate convergence
- **Problem signature**: Characteristics that define a problem class (dimensions, constraints, physics, regime)
- **Domain hint**: Optional problem metadata (e.g., "shape_optimization") that guides Paola's decisions

## Architectural Patterns

### Agent-Tool Interaction

```python
# Agent's autonomous reasoning loop (not prescribed)
while not done:
    observation = observe()       # Monitor optimization health
    decision = reason(obs)        # LLM reasoning about what to do
    action = choose_tool(dec)     # Select tool based on decision
    execute(action)               # Use tool primitive
```

### Knowledge Accumulation

```python
# Platform learns patterns across all optimizations
# Example: "Airfoil CL constraints typically undershoot by 2%"
# Later optimization automatically applies this knowledge
# → Tighten CL constraints proactively
```

### Multi-Strategy Composition

```python
# Agent invents strategy never programmed into platform
# Example: Bayesian(10 samples) → SLSQP(2 parallel starts)
#          → Constraint tightening → SLSQP restart → High-fidelity verification
# Total: 45 evaluations, agent decided sequence dynamically
```

## Value Proposition Summary

**PAOLA: The optimization platform that learns from every run**

> **"Optimization complexity is Paola intelligence, not user burden."**

"The first optimization platform where an AI agent handles all optimization complexity - algorithm selection, configuration, initialization, and failure recovery - while continuously learning from past optimizations."

**For Engineers**:
- Say `priority="robustness"`, not configure 250 IPOPT options
- Natural language goals instead of algorithm configuration
- Platform learns from your past optimizations
- Automatic warm-starting from similar problems

**For Companies**:
- 90% success rate (vs 50%), 2-3× faster convergence
- Organizational knowledge accumulation (expert knowledge persists)
- Multi-run analysis reveals best practices
- Democratize optimization expertise

**Technical Moat**:
- **The Paola Principle**: Expert knowledge encoded in agent
- Agent autonomy (no fixed loops)
- Configuration intelligence (250+ options → "robustness")
- Initialization intelligence (algorithm-aware, history-aware)
- Knowledge base with RAG retrieval
- Strategic adaptation within runs
- Evaluation cache for efficiency
