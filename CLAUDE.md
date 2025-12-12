# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains **PAOLA (Platform for Agentic Optimization with Learning and Analysis)** - a next-generation engineering optimization system where an autonomous AI agent controls the optimization process, accumulates knowledge from past optimizations, and analyzes multiple runs to achieve reliable convergence.

### Core Innovation

Unlike traditional optimization platforms (HEEDS, ModeFRONTIER, Dakota, pyOptSparse, FADO) that use fixed control loops with user-configured algorithms, PAOLA provides three key innovations:

1. **Agentic Control**: Agent autonomously composes strategies using tool primitives (not fixed loops)
2. **Organizational Learning**: Knowledge base with RAG-based retrieval for warm-starting similar problems
3. **Multi-Run Analysis**: Compares multiple optimization strategies to select best approaches

**Traditional approach**: Platform prescribes the loop, user configures it, no memory between runs

**PAOLA approach**: Agent controls everything, learns from past optimizations, analyzes multiple strategies

The agent continuously observes optimization progress, reasons about numerical health and feasibility patterns, autonomously adapts the strategy (constraint bounds, gradient methods, exploration control), and accumulates knowledge that improves future optimizations.

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

### Tool Primitives Architecture

The platform provides atomic tools that the agent composes into strategies:

**Optimizer Tools**:
- `optimizer_create(algorithm, problem)` - Create optimizer instance
- `optimizer_propose_design()` - Get next design candidate
- `optimizer_update(design, objective, gradient)` - Update optimizer state
- `optimizer_restart_from_checkpoint(new_settings)` - Restart with new configuration
- `optimizer_checkpoint() / restore()` - Save/restore optimizer state

**Evaluation Tools**:
- `workflow_execute(template, design, fidelity)` - Run CFD/FEA simulation
- `workflow_get_template(name)` - Get workflow configuration

**Data Tools**:
- `cache_get(design) / cache_store(design, results)` - Evaluation cache
- `database_query(filter)` - Query optimization history
- `database_find_similar(design)` - Find similar past designs

**Learning Tools** (Phase 3):
- `knowledge_store(problem_signature, successful_setup)` - Store expert knowledge
- `knowledge_retrieve(problem_signature)` - RAG-based retrieval of similar problems
- `knowledge_apply(retrieved_knowledge)` - Warm-start with proven strategies

**Analysis Tools** (Phase 2/3):
- `analyze_runs(run_ids)` - Compare multiple optimization strategies
- `plot_convergence(run_ids)` - Visualize convergence history
- `recommend_strategy(problem, past_results)` - Select best approach based on analysis

**Utility Tools**:
- `gradient_compute(design, method)` - Switch between adjoint/finite-difference
- `constraint_adjust_bounds(constraint_id, new_bound)` - Tighten/relax constraints
- `budget_remaining()` - Check remaining computational budget
- `time_elapsed()` - Track optimization time

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

### agent_controlled_optimization.md (50KB)

Comprehensive technical design covering:
- Core innovation: Agent autonomy vs prescribed loops
- Comparison with all major existing platforms (HEEDS, ModeFRONTIER, Dakota, pyOptSparse, FADO, Tosca)
- Tool primitives architecture
- Practical adaptation mechanisms and the "replay constraint"
- Agent decision patterns and reasoning examples
- Complete workflow example showing agent in action
- Implementation architecture and phases

**Key sections**:
- Section 4: Detailed comparison with existing software
- Section 7: Tool primitives design
- Section 8: Practical adaptation mechanisms (constraint replay understanding)
- Section 15: Specific agent adaptation strategies with examples
- Section 17: Complete workflow showing agent reasoning and actions

### agentic_optimization_vision.md (7KB)

High-level vision and value propositions:
- 7 core value propositions from first principles analysis
- Paradigm shift from "operator configures loops" to "director sets goals"
- Knowledge accumulation and learning organization concept
- Multi-fidelity intelligence and compositional problem formulation
- Failure mode prevention through predictive analytics

## Implementation Status

**Current state**: Design/documentation phase (no code yet)

**Planned implementation phases**:
1. **Phase 1** (2 months): Core mechanisms - evaluation cache, optimizer wrapper with checkpointing, basic agent with observation
2. **Phase 2** (2 months): Agent intelligence - full observation metrics, constraint management, gradient switching, Bayesian control
3. **Phase 3** (2 months): Validation & extension - real engineering test cases, multiple optimizers, knowledge base, production hardening

## Development Principles

When implementing this platform:

1. **Agent Autonomy First**: Never add a "hook" or "callback" - the agent IS the controller
2. **Tools Not Control Flow**: Platform provides primitives, agent composes strategy
3. **Observable Everything**: Every action must be observable and explainable
4. **Strategic Restarts**: Adaptations are informed restarts from better positions, not unpredictable experiments
5. **Cache Everything**: Simulations are expensive, cache all evaluations
6. **Learn Continuously**: Every optimization adds to the knowledge base
7. **CRITICAL - Minimal Prompting**: Keep system prompts and tool schemas minimal. Trust the LLM's intelligence. Never add verbose guidance, formatting rules, or hand-holding without explicit permission. The agent must learn from experience, not from over-specified prompts.

## Key Terminology

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

"The first optimization platform where an AI agent continuously observes optimization progress, detects feasibility and convergence issues, autonomously adapts strategy, accumulates knowledge from past optimizations, and analyzes multiple runs to achieve reliable convergence."

**For Engineers**:
- Natural language goals instead of algorithm configuration
- Platform learns from your past optimizations
- Automatic warm-starting from similar problems

**For Companies**:
- 90% success rate (vs 50%), 2-3× faster convergence
- Organizational knowledge accumulation (expert knowledge persists)
- Multi-run analysis reveals best practices

**Technical Moat**:
- Agent autonomy (no fixed loops)
- Knowledge base with RAG retrieval
- Multi-run analysis and comparison
- Strategic adaptation within runs
- Evaluation cache for efficiency
- Proven patterns from AdjointFlow
