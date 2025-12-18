# Paola

**Package for Agentic Optimization with Learning and Analysis**

*The optimization package that learns from every run*

Version 0.4.7

---

An AI-driven optimization platform where an autonomous agent controls the entire optimization process - algorithm selection, configuration, initialization, and failure recovery - while learning from past optimizations.

## The Paola Principle

> **"Optimization complexity is Paola intelligence, not user burden."**

Traditional optimization requires users to configure 250+ IPOPT options, tune CMA-ES sigma values, or select between dozens of SciPy methods. Paola handles all this complexity through LLM reasoning, leaving users to simply describe their goals.

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API key (create .env file)
echo "DASHSCOPE_API_KEY=your_key_here" > .env
# Get key at: https://dashscope.console.aliyun.com/

# Or use Anthropic/OpenAI
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

### 2. Launch the CLI

```bash
python -m paola.cli
```

### 3. Your First Optimization

```
paola> optimize a 10D Rosenbrock function

[Agent creates problem, starts graph, runs optimization]

Graph #1 completed successfully!
  Strategy: scipy:SLSQP
  Final objective: 4.517e-07
  Iterations: 47

paola> /graphs
[Lists all optimization graphs]

paola> /graph show 1
[Detailed graph with nodes, edges, convergence]
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Graph-Based Architecture** | Multi-node optimization with chains, trees, warm-starts |
| **LLM-Driven Intelligence** | Agent selects algorithms, configures options, handles failures |
| **Skills Infrastructure** | Progressive-disclosure optimizer expertise (IPOPT, SciPy, Optuna) |
| **Two-Tier Storage** | Compact GraphRecords for learning + full details for debugging |
| **Foundry Single Source** | Unified data layer with cache-through loading |
| **Multiple Backends** | SciPy (8 methods), IPOPT (250+ options), Optuna (8 samplers) |

## Architecture

```
paola/
├── agent/              # LangGraph-based ReAct agent
│   ├── react_agent.py  # Autonomous agent loop
│   └── prompts/        # Minimal system prompts
├── foundry/            # Data foundation layer
│   ├── foundry.py      # Single source of truth
│   ├── schema/         # GraphRecord, GraphDetail, polymorphic components
│   └── storage/        # FileStorage backend
├── tools/              # LangChain @tool functions
│   ├── graph_tools.py  # start_graph, finalize_graph, query_past_graphs
│   ├── optimization_tools.py  # run_optimization, get_problem_info
│   ├── evaluator_tools.py     # create_nlp_problem, evaluate_function
│   └── observation_tools.py   # analyze_convergence, detect_pattern
├── skills/             # Optimizer expertise (IPOPT, SciPy, Optuna)
│   ├── optimizers/     # YAML skill definitions
│   └── tools.py        # list_skills, load_skill, query_skills
├── optimizers/         # Backend implementations
│   └── backends.py     # SciPyBackend, IPOPTBackend, OptunaBackend
└── cli/                # Interactive CLI
    ├── repl.py         # Main REPL loop
    └── commands.py     # /graphs, /show, /skills, etc.
```

## Graph-Based Optimization

Paola uses **graphs** to represent optimization tasks:

```
Graph #1: "Optimize wing drag"
│
├── n1: Global exploration (Optuna TPE)
│   └── Agent: "Found promising region, switch to gradient"
│
├── n2: Local refinement from n1 (SLSQP) [warm_start edge]
│   └── Agent: "Stuck at local minimum, try CMA-ES"
│
└── n3: Escape local minimum (CMA-ES) [refine edge]
    └── Agent: "Converged, done"

Result: success=true, pattern="chain", final_obj=0.05
```

**Graph** = Complete optimization task (may involve multiple nodes)
**Node** = Single optimizer execution
**Edge** = Relationship (warm_start, restart, refine, branch, explore)

## CLI Commands

### Natural Language
```
paola> optimize a 50D Ackley function with Optuna TPE
paola> refine the result with SLSQP
paola> what strategies have worked for similar problems?
```

### Graph Commands
```
/graphs              - List all optimization graphs
/graph show <id>     - Show detailed graph information
/graph plot <id>     - Plot convergence history
/graph compare <id1> <id2>  - Compare graphs side-by-side
/graph best          - Show best solution across all graphs
```

### Skills Commands
```
/skills              - List all optimizer skills
/skill <name>        - Show skill details (e.g., /skill ipopt)
```

### Other Commands
```
/evals               - List registered evaluators
/analyze <id>        - AI-powered analysis
/help                - Show all commands
/exit                - Exit CLI
```

## Agent Tools

**Graph Management** (4 tools):
- `start_graph` - Create new optimization graph
- `get_graph_state` - Query graph state for agent decisions
- `finalize_graph` - Complete and persist graph
- `query_past_graphs` - Query historical graphs for learning

**Optimization** (3 tools):
- `run_optimization` - Execute optimizer with full config control
- `get_problem_info` - Get problem characteristics for LLM reasoning
- `list_available_optimizers` - List backends and availability

**Problem Formulation** (4 tools):
- `create_nlp_problem` - Define NLP with objective, bounds, constraints
- `derive_problem` - Create derived problem (narrow bounds, etc.)
- `evaluate_function` - Evaluate objective at a point
- `compute_gradient` - Compute gradient (analytical or FD)

**Skills** (3 tools):
- `list_skills` - Discover optimizer expertise
- `load_skill` - Load detailed configuration knowledge
- `query_skills` - Search across skills

## Optimizer Backends

| Backend | Methods | Notes |
|---------|---------|-------|
| **SciPy** | SLSQP, L-BFGS-B, trust-constr, COBYLA, Nelder-Mead, Powell, BFGS, CG | Gradient and derivative-free |
| **IPOPT** | Interior-point | 250+ configurable options |
| **Optuna** | TPE, CMA-ES, GP, NSGA-II, QMC, Random, Grid | Black-box optimization |

## Skills Infrastructure

Skills provide progressive-disclosure optimizer expertise:

```
paola> /skill ipopt

IPOPT Skill
-----------
Interior-point method for large-scale nonlinear optimization.

When to use:
- Constrained nonlinear problems
- Large-scale (1000+ variables)
- Need warm-starting capability

Integration:
- Warm-start: Set warm_start_init_point: 'yes'
- Scaling: IPOPT auto-scales, but manual may help
...
```

Load specific sections: `load_skill("ipopt", "options.warm_start")`

## Data Storage

Paola uses two-tier storage for efficiency:

**Tier 1: GraphRecord** (~1KB) - For LLM learning
- Problem signature (dimensions, constraints, bounds)
- Strategy pattern (chain, multistart, tree)
- Node summaries with full optimizer config
- Success/failure outcomes

**Tier 2: GraphDetail** (10-100KB) - For visualization
- Full convergence trajectories
- Complete x vectors
- Detailed timing

```
.paola_foundry/
├── graphs/         # Tier 1: GraphRecord
├── details/        # Tier 2: GraphDetail
├── problems/       # Problem definitions
├── evaluators/     # Registered evaluators
└── metadata.json   # Next IDs, etc.
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Specific test suites
pytest tests/test_foundry.py -v
pytest tests/test_cli_integration.py -v
```

## Supported LLMs

**Qwen** (recommended, cost-effective):
- `qwen-flash`, `qwen-plus`, `qwen-max`, `qwen-turbo`

**Anthropic**:
- `claude-sonnet-4`, `claude-3-5-sonnet`

**OpenAI**:
- `gpt-4`, `gpt-3.5-turbo`

Configure in `.env` and switch with `/models` in CLI.

## Development

See `CLAUDE.md` for detailed development principles:

1. **The Paola Principle**: Optimization complexity is agent intelligence
2. **LLM IS the Intelligence**: Don't hardcode optimizer selection
3. **Graph-Based**: Every optimization runs within a graph
4. **Tools Not Control Flow**: Platform provides primitives
5. **Observable Everything**: Every action must be explainable
6. **Cache Everything**: Simulations are expensive
7. **Minimal Prompting**: Trust the LLM's trained knowledge
8. **Foundry is Single Source of Truth**: All data access through Foundry
9. **Expert Optimizer Usage**: Skills teach correct API usage

## Roadmap

**v0.4.7** (Current):
- Type consistency fix: problem_id is int throughout
- Simplified query_graphs (exact match instead of patterns)
- Backward compatible with legacy data

**v0.4.6**:
- Foundry as single source of truth
- Cache-through problem loading
- Pydantic validation for type safety

**v0.5.0** (Planned):
- Cross-graph learning tools
- Pattern extraction from successful graphs
- Aggregate statistics

**v0.6.0** (Future):
- RAG integration for semantic search
- Learned skills auto-generation

## License

TBD

## Citation

Paper in preparation: "Paola: Package for Agentic Optimization with Learning and Analysis"

---

**Paola**: The optimization platform that learns from every run

**Launch**: `python -m paola.cli`
