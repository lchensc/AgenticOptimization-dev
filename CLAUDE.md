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

### The Paola Principle

> **"Optimization complexity is Paola intelligence, not user burden."**

This principle defines PAOLA's approach to the overwhelming complexity of optimization:

| What Users Specify | What Paola Handles |
|-------------------|-------------------|
| Problem definition (objective, constraints, bounds) | Algorithm selection |
| Natural language goals | Option configuration (250+ IPOPT options) |
| | Initialization (x0, sigma, population) |
| | Convergence failure handling |
| | Warm-starting from history |

**Example**: IPOPT has ~250 options across 22 categories. SNOPT is scaling-sensitive. CMA-ES needs sigma tuning. Most users only touch 3-5 options. Paola knows which options matter for which problems - this expert knowledge is Paola's core competence.

## Repository Structure

```
AgenticOptimization/
├── paola/                                 # Main package
│   ├── agent/                            # LangGraph agents (conversational, react)
│   ├── tools/                            # LangChain @tool functions
│   │   ├── session_tools.py              # Session management (v0.2.0)
│   │   ├── optimization_tools.py         # run_optimization, get_problem_info
│   │   ├── evaluator_tools.py            # create_nlp_problem, evaluate_function
│   │   ├── config_tools.py               # Expert escape hatch (config_scipy, etc.)
│   │   └── analysis.py                   # Metrics and AI analysis
│   ├── foundry/                          # Data foundation layer
│   │   ├── schema/                       # Polymorphic components per optimizer family
│   │   ├── storage/                      # FileStorage backend
│   │   ├── active_session.py             # In-progress session/run tracking
│   │   └── foundry.py                    # OptimizationFoundry main class
│   ├── optimizers/                       # Optimizer backends (SciPy, IPOPT, Optuna)
│   ├── cli/                              # Interactive CLI (repl.py, commands.py)
│   ├── knowledge/                        # Knowledge base (skeleton - Phase 3)
│   └── analysis/                         # Metrics computation
├── docs/                                  # Documentation (timestamped)
│   ├── architecture/                     # Design documents
│   └── ...                               # Other categories
├── tests/                                 # Test suite
└── CLAUDE.md                             # This file
```

## Documentation Conventions

- **Timestamp prefix**: `YYYYMMDD_HHMM_name.md` (e.g., `20251215_1530_feature_design.md`)
- **Subfolders**: `architecture/`, `implementation/`, `bugfix/`, `analysis/`, `decisions/`, `planning/`, `progress/`, `archive/`

## Key Concepts

### v0.2.0 Session-Based Architecture

**Session** = Complete optimization task (may involve multiple runs with different optimizers)
**Run** = Single optimizer execution within a session

```
Session #42: "Optimize wing drag"
│
├── Run 1: Global exploration (Optuna TPE)
│   └── Paola decides: "Found promising region, switch to gradient"
│
├── Run 2: Local refinement (SLSQP)
│   └── Paola decides: "Stuck at local minimum, try CMA-ES"
│
└── Run 3: Escape local minimum (CMA-ES)
    └── Paola decides: "Converged, done"

Session: success=true, final_obj=0.05, total_evals=100
```

### Optimizer Families (Polymorphic Components)

Each run uses family-specific data structures:

| Family | Optimizers | Key Data |
|--------|------------|----------|
| `gradient` | SciPy (SLSQP, L-BFGS-B), IPOPT | iterations, gradient_norm, step_size |
| `bayesian` | Optuna (TPE) | trials, acquisition values |
| `population` | DE, GA, PSO | generations, population |
| `cmaes` | CMA-ES | mean, covariance, generations |

### Tool Architecture (LLM-Driven)

The LLM agent IS the intelligence. It has been trained on IPOPT docs, SciPy reference, Optuna tutorials, optimization theory. We don't re-implement this knowledge in Python - we let the LLM reason.

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM AGENT (Intelligence)                      │
│  Trained knowledge of: IPOPT, Optuna, NLopt, SciPy, CMA-ES     │
│  Core tasks: analyze problem → select optimizer → configure     │
└─────────────────────────────────────────────────────────────────┘
              │               │               │
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ INFORMATION     │ │ EXECUTION       │ │ SESSION         │
│ TOOLS           │ │ TOOLS           │ │ TOOLS           │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ get_problem_info│ │ run_optimization│ │ start_session   │
│ list_optimizers │ │ create_nlp_prob │ │ finalize_session│
│ get_opt_options │ │                 │ │ get_session_info│
└─────────────────┘ └─────────────────┘ └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OPTIMIZER BACKENDS                            │
│  SciPyBackend: SLSQP, L-BFGS-B, trust-constr, COBYLA           │
│  IPOPTBackend: Interior-point with 250+ options                 │
│  OptunaBackend: TPE, CMA-ES, Random samplers                    │
└─────────────────────────────────────────────────────────────────┘
```

### Core Tools (v0.2.0)

**Session Management**:
- `start_session(problem_id, goal)` - Create new optimization session
- `finalize_session(session_id, success)` - Complete and persist session
- `get_session_info(session_id)` - Get session status

**Optimization Execution**:
- `run_optimization(session_id, optimizer, config, init_strategy)` - Execute optimization
  - `optimizer`: "scipy:SLSQP", "scipy:L-BFGS-B", "ipopt", "optuna:TPE"
  - `init_strategy`: "center", "random", "warm_start"
  - `config`: JSON string with optimizer-specific options

**Problem Formulation**:
- `create_nlp_problem(problem_id, objective_evaluator_id, bounds, constraints)` - Define NLP
- `get_problem_info(problem_id)` - Get problem characteristics for LLM reasoning

**Expert Escape Hatch** (optional):
- `config_scipy(...)`, `config_ipopt(...)` - Direct configuration for experts

### Evaluation Cache (Critical for Efficiency)

Engineering simulations are 10,000× more expensive than optimizer iterations:
- CFD evaluation: 4-10 CPU hours → $400-$1000
- Gradient (adjoint): 6 CPU hours → $600
- Optimizer iteration: 0.001 hours → $0.10

The cache prevents re-running expensive simulations when the optimizer revisits designs.

## Design Documents

### docs/architecture/20251215_2100_foundry_polymorphic_schema_design.md

**Session-based architecture design** for v0.2.0:
- Session vs Run terminology
- Polymorphic components per optimizer family
- Clean API without backward compatibility

### docs/architecture/20251215_1630_llm_driven_optimization_redesign.md

**LLM-driven architecture**:
- Why hardcoded if-else is wrong (not real intelligence)
- LLM trained knowledge IS the intelligence
- Information/Execution/Session tool split

### docs/architecture/20251215_1109_tools_optimization_foundry_design.md

**Tool architecture design**:
- The Paola Principle implementation
- Compact bounds specification
- Expert escape hatch pattern

## Implementation Status

**Current state**: v0.2.0 - Session-based architecture implemented

**Working features**:
- CLI with conversational interface (`python -m paola.cli`)
- Session management (start, run, finalize)
- Multiple optimizer backends (SciPy, IPOPT, Optuna)
- Evaluator registration system
- Polymorphic run storage per optimizer family

**In progress**:
- Knowledge base with RAG retrieval (skeleton implemented)
- Multi-run analysis
- Strategic adaptation within sessions

## Development Principles

When implementing this platform:

1. **The Paola Principle**: Optimization complexity is agent intelligence, not user burden
2. **LLM IS the Intelligence**: Don't hardcode optimizer selection logic - let LLM reason
3. **Session-Based**: Every optimization runs within a session for multi-run support
4. **Tools Not Control Flow**: Platform provides primitives, agent composes strategy
5. **Observable Everything**: Every action must be observable and explainable
6. **Cache Everything**: Simulations are expensive, cache all evaluations
7. **CRITICAL - Minimal Prompting**: Keep system prompts minimal. Trust the LLM's intelligence. Never add verbose guidance without explicit permission

## Key Terminology

- **Session**: Complete optimization task (may involve multiple runs)
- **Run**: Single optimizer execution within a session
- **Optimizer Family**: Category of optimizer (gradient, bayesian, population, cmaes)
- **Polymorphic Components**: Family-specific data structures (iterations vs trials vs generations)
- **Warm-start**: Using previous run's result as starting point
- **The Paola Principle**: "Optimization complexity is Paola intelligence, not user burden"
- **Expert Escape Hatch**: Optional tools for direct optimizer configuration
- **Evaluation Cache**: Storage for expensive simulation results
- **Domain Hint**: Optional problem metadata (e.g., "shape_optimization")

## CLI Commands

```
/sessions           - List all optimization sessions
/show <id>          - Show session details
/plot <id>          - Plot convergence
/compare <id1> <id2> - Compare sessions
/analyze <id>       - AI-powered analysis
/evals              - List registered evaluators
/help               - Show all commands
```

## Value Proposition Summary

**PAOLA: The optimization platform that learns from every run**

> **"Optimization complexity is Paola intelligence, not user burden."**

"The first optimization platform where an AI agent handles all optimization complexity - algorithm selection, configuration, initialization, and failure recovery - while continuously learning from past optimizations."

**For Engineers**:
- Natural language goals instead of algorithm configuration
- Platform learns from your past optimizations
- Automatic warm-starting from similar problems

**For Companies**:
- Organizational knowledge accumulation
- Multi-run analysis reveals best practices
- Democratize optimization expertise

**Technical Moat**:
- LLM-driven intelligence (not hardcoded rules)
- Session-based multi-run optimization
- Polymorphic storage per optimizer family
- Knowledge base with RAG retrieval
