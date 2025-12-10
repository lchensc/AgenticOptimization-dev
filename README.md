# AOpt - Agentic Optimization Platform

**AI-Centric Optimization for Engineering and Science**

The first optimization platform where an autonomous AI agent controls the optimization process, composing strategies from tool primitives rather than following fixed loops.

## Status

🚧 **In Development** - Milestone 1 (Week 1) in progress

## Quick Start

### 1. Setup Environment

```bash
# Copy .env.example to .env
cp .env.example .env

# Edit .env and add your DASHSCOPE_API_KEY (Qwen)
# Get key at: https://dashscope.console.aliyun.com/
```

### 2. Run Optimization

```python
from aopt import Agent

# Create agent (uses Qwen by default)
agent = Agent(llm_model="qwen-plus", verbose=True)

# Run optimization
result = agent.run("""
    Minimize drag on transonic airfoil
    Maintain CL >= 0.8
""")
```

**Supported Models**:
- **Qwen** (primary): `"qwen-flash"`, `"qwen-turbo"`, `"qwen-plus"` (default)
- **Claude** (optional): `"claude-sonnet-4"`, `"claude-3-5-sonnet-20241022"`
- **OpenAI** (optional): `"gpt-4"`, `"gpt-3.5-turbo"`

## What's Implemented (Week 1)

### ✅ Core Infrastructure
- **Repository structure**: Full `aopt/` package layout
- **Pydantic schemas**: `OptimizationProblem`, `Objective`, `Variable`, `Constraint`
- **Callback system**: Real-time event streaming architecture
- **Cache tools**: Evaluation cache to prevent re-computation
- **Tests**: 30 passing tests with >90% coverage

### ✅ Callback System (Real-time Streaming)
```python
from aopt.callbacks import EventCapture, RichConsoleCallback, FileLogger

# Built-in rich console
agent = Agent(verbose=True)  # Beautiful terminal output

# Custom callbacks
capture = EventCapture()
agent.register_callback(capture)
agent.register_callback(FileLogger("run.log"))

# Multiple callbacks work simultaneously
agent.run("Minimize Rosenbrock")
assert capture.count(EventType.CACHE_HIT) > 0
```

**Features**:
- 15+ event types (AGENT_START, TOOL_CALL, CACHE_HIT, CONVERGENCE_CHECK, etc.)
- Error isolation (callback failures don't break optimization)
- EventCapture for testing
- FileLogger for debugging
- RichConsoleCallback for beautiful terminal output

### ✅ Evaluation Cache
```python
from aopt.tools.cache_tools import cache_get, cache_store

# First evaluation - cache miss
result = evaluate(design, problem_id)
cache_store(design, problem_id, objectives, gradient, cost=10.0)

# Second evaluation - cache hit (saves 10 CPU hours!)
cached = cache_get(design, problem_id)
assert cached["hit"]
assert cached["cost"] == 10.0  # Original cost, not re-incurred
```

**Critical for efficiency**:
- Engineering simulations: 10,000× more expensive than optimizer iterations
- Prevents re-evaluation during line searches
- Problem-isolated caching
- Tolerance-based design matching

### ✅ Run Database
```python
from aopt.tools.cache_tools import run_db_log, run_db_query

# Log every decision
run_db_log(
    optimizer_id="opt_001",
    iteration=5,
    design=[1.0, 2.0],
    objectives=[0.0245],
    action="evaluate",
    reasoning="Proposed by SLSQP line search"
)

# Query history
entries = run_db_query("opt_001", limit=10)
```

## Architecture

```
aopt/
├── agent/              # ReAct agent (TODO: Week 1)
├── callbacks/          # ✅ Event streaming system
├── formulation/        # ✅ Problem schemas
├── tools/              # ✅ Cache tools, optimizer tools (partial)
├── optimizers/         # TODO: Week 2
├── backends/           # TODO: Week 2
└── utils/              # TODO
```

## Testing

```bash
# Activate environment
conda activate agent

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=aopt --cov-report=html
```

**Current test results**: ✅ 30 passed in 0.32s

## Development Roadmap

### Week 1 (Current)
- [x] Repository structure
- [x] Pydantic schemas
- [x] Callback system
- [x] Cache tools
- [x] Tests for core infrastructure
- [ ] ReAct agent skeleton (in progress)
- [ ] Formulation tools

### Week 2
- [ ] Optimizer tools (create, propose, update, restart)
- [ ] Scipy optimizer integration (SLSQP, COBYLA, L-BFGS-B)
- [ ] Evaluator tools with cache integration
- [ ] Analytical backend (Rosenbrock, etc.)
- [ ] End-to-end test: Agent solves 2D Rosenbrock

### Week 3
- [ ] Observer tools (history, convergence, patterns)
- [ ] Adapter tools (modify constraints, switch gradient method)
- [ ] Safe optimizer restart with cache reuse
- [ ] Constrained optimization test

### Week 4
- [ ] Pymoo integration (NSGA-II)
- [ ] Multi-objective support
- [ ] Complete Milestone 1

## Key Innovations

1. **Intelligent Formulation**: Agent converts natural language → structured problems
2. **Full Autonomy**: No fixed loops, agent decides everything
3. **Compositional**: 18 tool primitives for strategy composition
4. **Adaptive**: Observes, detects patterns, modifies strategy mid-run
5. **Efficient**: Evaluation cache prevents expensive re-computation
6. **Safe**: Optimizer restarts from best design with cache reuse
7. **Observable**: Real-time event streaming via callbacks
8. **Extensible**: Easy to add new problem types, optimizers, tools
9. **Explainable**: Every decision logged with reasoning

## Documentation

- `docs/architecture_v3_final.md` - Complete system architecture
- `docs/callback_streaming_architecture.md` - Event streaming design
- `docs/architecture_v3_high_severity_fixes.md` - Critical fixes applied

## Dependencies

See `requirements.txt` for full list. Core dependencies:
- LangChain + LangGraph (agent framework)
- Pydantic (schemas)
- Rich (terminal output)
- Scipy (optimization)
- Pymoo (multi-objective)

## License

TBD

## Citation

Paper in preparation: "Agentic Optimization for Engineering Design"

---

**Status**: Week 1 in progress. Core infrastructure complete, agent implementation next.
