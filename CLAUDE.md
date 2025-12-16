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
│   │   ├── graph_tools.py                # Graph management (v0.3.0)
│   │   ├── session_tools.py              # Session management (v0.2.0 legacy)
│   │   ├── optimization_tools.py         # run_optimization, get_problem_info
│   │   ├── evaluator_tools.py            # create_nlp_problem, evaluate_function
│   │   ├── config_tools.py               # Expert escape hatch (config_scipy, etc.)
│   │   └── analysis.py                   # Metrics and AI analysis
│   ├── foundry/                          # Data foundation layer
│   │   ├── schema/                       # Polymorphic components per optimizer family
│   │   ├── storage/                      # FileStorage backend
│   │   ├── active_graph.py               # In-progress graph/node tracking (v0.3.0)
│   │   ├── active_session.py             # In-progress session/run tracking (v0.2.0)
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

### v0.3.0 Graph-Based Architecture (Current)

**Graph** = Complete optimization task (may involve multiple nodes with different strategies)
**Node** = Single optimizer execution within a graph
**Edge** = Relationship between nodes (warm_start, restart, refine, branch, explore)

Design principle: **"Graph externalizes state, agent makes decisions."**
- The graph helps the agent track state (node IDs, not x0 values)
- The agent explicitly decides which node to continue from
- The system does NOT automatically select "best" - that's the agent's decision

```
Graph #1: "Optimize wing drag"
│
├── n1: Global exploration (Optuna TPE)
│   └── Agent decides: "Found promising region, switch to gradient"
│
├── n2: Local refinement from n1 (SLSQP) [warm_start edge]
│   └── Agent decides: "Stuck at local minimum, try CMA-ES"
│
└── n3: Escape local minimum from n2 (CMA-ES) [refine edge]
    └── Agent decides: "Converged, done"

Graph: success=true, pattern="chain", final_obj=0.05, total_evals=100
```

**Edge Types:**
- `warm_start`: Use parent's best_x as starting point
- `restart`: Fresh start with knowledge of parent result
- `refine`: Tighten tolerances to polish solution
- `branch`: Explore alternative from same starting point
- `explore`: Independent exploration (no dependency)

**Graph Patterns:**
- `single`: One node only
- `multistart`: Multiple roots, no edges
- `chain`: Sequential refinement (n1 → n2 → n3)
- `tree`: Branching from common ancestors
- `dag`: General directed acyclic graph

### v0.2.0 Session-Based Architecture (Legacy)

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
│ INFORMATION     │ │ EXECUTION       │ │ GRAPH           │
│ TOOLS           │ │ TOOLS           │ │ TOOLS           │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ get_problem_info│ │ run_optimization│ │ start_graph     │
│ list_optimizers │ │ create_nlp_prob │ │ get_graph_state │
│ get_opt_options │ │                 │ │ finalize_graph  │
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

### Core Tools (v0.3.0)

**Graph Management** (7 tools total):
- `start_graph(problem_id, goal)` - Create new optimization graph
- `get_graph_state(graph_id)` - Get graph state for agent decision-making
- `finalize_graph(graph_id, success, notes)` - Complete and persist graph

**Optimization Execution**:
- `run_optimization(graph_id, optimizer, config, init_strategy, parent_node, edge_type)` - Execute optimization
  - `graph_id`: Graph to add node to
  - `optimizer`: "scipy:SLSQP", "scipy:L-BFGS-B", "ipopt", "optuna:TPE"
  - `init_strategy`: "center", "random", "warm_start"
  - `parent_node`: Node ID to continue from (e.g., "n1")
  - `edge_type`: Relationship type (warm_start, restart, refine, branch, explore)
  - `config`: JSON string with optimizer-specific options

**Information Tools**:
- `get_problem_info(problem_id)` - Get problem characteristics for LLM reasoning
- `list_optimizers()` - List available optimizer backends
- `get_optimizer_options(optimizer)` - Get optimizer configuration options

**Problem Formulation**:
- `create_nlp_problem(problem_id, objective_evaluator_id, bounds, constraints)` - Define NLP

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

**Current state**: v0.3.0 - Graph-based architecture implemented

**Working features**:
- CLI with conversational interface (`python -m paola.cli`)
- Graph management (start, run, get_state, finalize)
- Multiple optimizer backends (SciPy, IPOPT, Optuna)
- Evaluator registration system
- Polymorphic node storage per optimizer family
- Graph patterns (single, multistart, chain, tree, dag)
- Edge types for node relationships (warm_start, restart, refine, branch, explore)
- Agent explicitly specifies parent_node and edge_type

**Legacy support**:
- Session-based API (v0.2.0) still available for backward compatibility

**In progress**:
- Knowledge base with RAG retrieval (skeleton implemented)
- Multi-run analysis
- Strategic adaptation within graphs

## Development Principles

When implementing this platform:

1. **The Paola Principle**: Optimization complexity is agent intelligence, not user burden
2. **LLM IS the Intelligence**: Don't hardcode optimizer selection logic - let LLM reason
3. **Graph-Based**: Every optimization runs within a graph for multi-node support
4. **Graph Externalizes State**: Agent makes decisions, graph tracks node IDs
5. **Tools Not Control Flow**: Platform provides primitives, agent composes strategy
6. **Observable Everything**: Every action must be observable and explainable
7. **Cache Everything**: Simulations are expensive, cache all evaluations
8. **CRITICAL - Minimal Prompting**: Keep system prompts minimal. Trust the LLM's intelligence. Never add verbose guidance without explicit permission

## Key Terminology

- **Graph**: Complete optimization task (may involve multiple nodes)
- **Node**: Single optimizer execution within a graph
- **Edge**: Relationship between nodes (warm_start, restart, refine, branch, explore)
- **Pattern**: Graph structure (single, multistart, chain, tree, dag)
- **Optimizer Family**: Category of optimizer (gradient, bayesian, population, cmaes)
- **Polymorphic Components**: Family-specific data structures (iterations vs trials vs generations)
- **Warm-start**: Using parent node's best_x as starting point
- **The Paola Principle**: "Optimization complexity is Paola intelligence, not user burden"
- **Expert Escape Hatch**: Optional tools for direct optimizer configuration
- **Evaluation Cache**: Storage for expensive simulation results
- **Domain Hint**: Optional problem metadata (e.g., "shape_optimization")

Legacy terms (v0.2.0):
- **Session**: Complete optimization task (equivalent to Graph)
- **Run**: Single optimizer execution (equivalent to Node)

## CLI Commands

```
# Graph Commands (v0.3.0)
/graphs              - List all optimization graphs
/graph show <id>     - Show detailed graph information
/graph plot <id>     - Plot convergence history
/graph compare <id1> <id2> - Compare graphs side-by-side
/graph best          - Show best solution across all graphs

# General Commands (auto-detect graph vs session)
/show <id>           - Show details (prefers graph)
/plot <id>           - Plot convergence
/compare <id1> <id2> - Compare
/best                - Show best solution

# Legacy Session Commands (v0.2.0)
/sessions            - List all optimization sessions

# Other Commands
/evals               - List registered evaluators
/analyze <id>        - AI-powered analysis
/help                - Show all commands
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
- Graph-based multi-node optimization with explicit agent decisions
- Polymorphic storage per optimizer family
- Knowledge base with RAG retrieval
