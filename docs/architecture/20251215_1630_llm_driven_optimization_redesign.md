# LLM-Driven Optimization Architecture Redesign

**Document ID**: 20251215_1630_llm_driven_optimization_redesign
**Date**: December 15, 2025
**Status**: Design (Revised)
**Purpose**: Correct architectural flaw - optimizer configuration is LLM intelligence

---

## 1. The Core Problem

### 1.1 What We Built (WRONG)

```python
# paola/agent/configuration.py - HARDCODED DETERMINISM
class ConfigurationManager:
    def select_algorithm(self, problem, priority):
        if is_constrained:
            if priority == "robustness":
                return "SLSQP"  # Fixed rule, not intelligence
```

This is just **moving if-else from user code to our code** - no actual intelligence.

### 1.2 The Paola Principle Correctly Applied

**"Optimization complexity is Paola intelligence"** means:

| Aspect | Wrong (Hardcoded) | Correct (LLM Intelligence) |
|--------|-------------------|---------------------------|
| Algorithm selection | Python if-else | LLM reasons based on problem |
| Configuration | Lookup table | LLM applies optimizer knowledge |
| Adaptation | None | LLM observes results, adjusts |
| New optimizer | Add Python code | LLM already knows from training |

The LLM has been trained on:
- [IPOPT documentation](https://coin-or.github.io/Ipopt/OPTIONS.html) (~250 options)
- [Optuna documentation](https://optuna.readthedocs.io/)
- [NLopt algorithms](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/)
- [SciPy optimize reference](https://docs.scipy.org/doc/scipy/reference/optimize.html)
- Optimization theory and best practices

**This knowledge IS the intelligence. We don't need to re-implement it in Python.**

---

## 2. Correct Architecture

### 2.1 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM AGENT (Intelligence)                      │
│                                                                 │
│  Trained knowledge of: IPOPT, Optuna, NLopt, SciPy, CMA-ES     │
│  Future: RAG from knowledge base of past runs                   │
│                                                                 │
│  Core intelligence:                                             │
│    1. Analyze problem characteristics                           │
│    2. Select appropriate optimizer                              │
│    3. Configure optimizer options                               │
│    4. Interpret results and adapt strategy                      │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ INFORMATION     │ │ EXECUTION       │ │ ANALYSIS        │
│ TOOLS           │ │ TOOL            │ │ TOOLS           │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ get_problem_info│ │ run_optimization│ │ get_run_history │
│ list_optimizers │ │   (central hub) │ │ analyze_results │
└─────────────────┘ └─────────────────┘ └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OPTIMIZER BACKENDS                            │
├─────────────────────────────────────────────────────────────────┤
│  SciPyBackend: SLSQP, L-BFGS-B, trust-constr, COBYLA, ...      │
│  IPOPTBackend: Interior-point with 250+ options                 │
│  OptunaBackend: TPE, CMA-ES, Random samplers                    │
│  NLoptBackend: LD_SLSQP, LN_COBYLA, GN_DIRECT, ... (future)    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 The Central Execution Tool

```python
@tool
def run_optimization(
    problem_id: str,
    optimizer: str,
    config: Optional[str] = None,
    max_iterations: int = 100
) -> Dict[str, Any]:
    """
    Execute optimization with LLM-specified configuration.

    Args:
        problem_id: Problem to optimize (from create_nlp_problem)
        optimizer: Optimizer specification:
            - "scipy:SLSQP" - SciPy with SLSQP method
            - "scipy:L-BFGS-B" - SciPy with L-BFGS-B
            - "ipopt" - IPOPT interior-point optimizer
            - "optuna:TPE" - Optuna with TPE sampler
            - "optuna:CMA-ES" - Optuna with CMA-ES sampler
        config: JSON string with optimizer-specific options.
            LLM constructs this based on its knowledge.
            Examples:
            - SciPy: '{"ftol": 1e-6, "gtol": 1e-5}'
            - IPOPT: '{"tol": 1e-6, "mu_strategy": "adaptive"}'
            - Optuna: '{"n_trials": 100}'
        max_iterations: Maximum iterations (common parameter)

    Returns:
        - success: bool
        - final_objective: float
        - final_design: List[float]
        - n_iterations: int
        - n_function_evals: int
        - convergence_info: Dict
        - optimizer_message: str
    """
```

### 2.3 Information Tools

```python
@tool
def get_problem_info(problem_id: str) -> Dict[str, Any]:
    """
    Get problem characteristics for LLM reasoning.

    Returns:
        - dimension: int
        - bounds: compact representation
        - bounds_center: List[float]
        - bounds_width: List[float]
        - num_inequality_constraints: int
        - num_equality_constraints: int
        - domain_hint: Optional[str]
        - description: str

    LLM uses this to reason about optimizer selection.
    """

@tool
def list_available_optimizers() -> Dict[str, Any]:
    """
    List available optimizer backends.

    Returns:
        - scipy: {available: true, methods: ["SLSQP", "L-BFGS-B", ...]}
        - ipopt: {available: true/false, version: "..."}
        - optuna: {available: true/false, samplers: ["TPE", "CMA-ES", ...]}
        - nlopt: {available: true/false, algorithms: [...]}

    LLM uses this to know what's installed.
    """

@tool
def get_run_history(problem_id: str) -> Dict[str, Any]:
    """
    Get history of optimization runs for learning.

    Returns:
        - runs: List of past runs with results
        - best_run: Most successful run
        - insights: Any stored observations

    LLM uses this to learn from past attempts.
    """
```

---

## 3. Optimizer Backend Architecture

### 3.1 Backend Interface

Each optimizer backend implements a common interface:

```python
class OptimizerBackend(ABC):
    """Base class for optimizer backends."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is installed."""

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Return capabilities and available methods."""

    @abstractmethod
    def optimize(
        self,
        problem: NLPProblem,
        evaluator: Callable,
        config: Dict[str, Any],
        max_iterations: int
    ) -> OptimizationResult:
        """Execute optimization."""
```

### 3.2 SciPy Backend (Existing)

```python
class SciPyBackend(OptimizerBackend):
    """SciPy optimization backend."""

    METHODS = {
        "SLSQP": {"gradient": True, "constraints": True, "bounds": True},
        "L-BFGS-B": {"gradient": True, "constraints": False, "bounds": True},
        "trust-constr": {"gradient": True, "constraints": True, "bounds": True},
        "COBYLA": {"gradient": False, "constraints": True, "bounds": False},
        "Nelder-Mead": {"gradient": False, "constraints": False, "bounds": False},
    }

    def optimize(self, problem, evaluator, config, max_iterations):
        method = config.get("method", "SLSQP")
        options = config.get("options", {})
        options["maxiter"] = max_iterations

        # Get initial point (for now: center of bounds)
        x0 = problem.get_bounds_center()

        result = scipy.optimize.minimize(
            fun=evaluator,
            x0=x0,
            method=method,
            bounds=problem.bounds,
            constraints=problem.get_scipy_constraints(),
            options=options
        )
        return self._convert_result(result)
```

### 3.3 IPOPT Backend (New)

```python
class IPOPTBackend(OptimizerBackend):
    """IPOPT interior-point optimizer backend."""

    def is_available(self) -> bool:
        try:
            import cyipopt
            return True
        except ImportError:
            return False

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "IPOPT",
            "type": "interior-point",
            "supports_constraints": True,
            "supports_bounds": True,
            "requires_gradients": True,  # Can use finite diff
            "key_options": [
                "tol", "max_iter", "mu_strategy", "mu_init",
                "linear_solver", "nlp_scaling_method"
            ]
        }

    def optimize(self, problem, evaluator, config, max_iterations):
        import cyipopt

        # Use cyipopt's scipy-compatible interface
        from cyipopt import minimize_ipopt

        x0 = problem.get_bounds_center()
        bounds = [(b[0], b[1]) for b in problem.bounds]

        # Extract IPOPT options from config
        options = config.get("options", {})
        options["max_iter"] = max_iterations

        result = minimize_ipopt(
            fun=evaluator,
            x0=x0,
            bounds=bounds,
            constraints=problem.get_scipy_constraints(),
            options=options
        )
        return self._convert_result(result)
```

### 3.4 Optuna Backend (New)

```python
class OptunaBackend(OptimizerBackend):
    """Optuna Bayesian optimization backend."""

    SAMPLERS = {
        "TPE": "Tree-structured Parzen Estimator",
        "CMA-ES": "Covariance Matrix Adaptation",
        "Random": "Random sampling",
        "Grid": "Grid search"
    }

    def is_available(self) -> bool:
        try:
            import optuna
            return True
        except ImportError:
            return False

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "Optuna",
            "type": "bayesian",
            "supports_constraints": False,  # Basic version
            "supports_bounds": True,
            "requires_gradients": False,
            "samplers": list(self.SAMPLERS.keys()),
            "key_options": ["n_trials", "sampler", "seed"]
        }

    def optimize(self, problem, evaluator, config, max_iterations):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler_name = config.get("sampler", "TPE")
        n_trials = config.get("n_trials", max_iterations)

        # Create sampler
        if sampler_name == "TPE":
            sampler = optuna.samplers.TPESampler()
        elif sampler_name == "CMA-ES":
            sampler = optuna.samplers.CmaEsSampler()
        elif sampler_name == "Random":
            sampler = optuna.samplers.RandomSampler()
        else:
            sampler = optuna.samplers.TPESampler()

        # Define objective
        def objective(trial):
            x = []
            for i, (lb, ub) in enumerate(problem.bounds):
                x.append(trial.suggest_float(f"x{i}", lb, ub))
            return evaluator(x)

        # Run optimization
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        return self._convert_result(study)
```

---

## 4. LLM Intelligence Examples

### 4.1 Benchmark: Rosenbrock Function

```
User: "Solve the Rosenbrock 10D problem"

LLM: "Let me first understand the problem."
LLM calls: get_problem_info("rosenbrock_10d")

Tool returns:
{
    "dimension": 10,
    "bounds": [[-5, 10], [-5, 10], ...],
    "num_inequality_constraints": 0,
    "domain_hint": null,
    "description": "Rosenbrock function, global min at (1,1,...,1)"
}

LLM reasons:
"Rosenbrock is a classic benchmark. Key characteristics:
- Unimodal but with a narrow curved valley
- Global minimum at (1,1,...,1) with f=0
- Gradient-based methods work well but can be slow in the valley
- 10D is moderate scale

For this problem, L-BFGS-B should work well - it's designed for
large-scale bound-constrained problems and handles the valley well.
Let me try it with default settings first."

LLM calls: run_optimization(
    problem_id="rosenbrock_10d",
    optimizer="scipy:L-BFGS-B",
    config='{"ftol": 1e-8}',
    max_iterations=200
)

Result: final_obj=1e-12, converged at (1.0, 1.0, ...)
LLM: "Excellent! L-BFGS-B found the global minimum efficiently."
```

### 4.2 Benchmark: Ackley Function (Multi-modal)

```
User: "Optimize the Ackley 10D problem"

LLM calls: get_problem_info("ackley_10d")

Tool returns:
{
    "dimension": 10,
    "bounds": [[-5, 5], ...],
    "description": "Ackley function, many local minima, global min at origin"
}

LLM reasons:
"Ackley is a challenging multi-modal benchmark:
- Many local minima surrounding global minimum
- Global minimum at origin with f(0)=0
- Gradient methods often get stuck in local minima

Let me first try SLSQP to see where it converges..."

LLM calls: run_optimization(
    problem_id="ackley_10d",
    optimizer="scipy:SLSQP",
    max_iterations=100
)

Result: final_obj=4.5, converged=true

LLM reasons:
"As expected, SLSQP converged to a local minimum (4.5 vs global 0).
For multi-modal problems, I should use a global optimizer.

Optuna with TPE is good for black-box optimization and can escape
local minima through its sampling strategy. Let me try it."

LLM calls: run_optimization(
    problem_id="ackley_10d",
    optimizer="optuna:TPE",
    config='{"n_trials": 500}',
    max_iterations=500
)

Result: final_obj=0.02

LLM: "Much better! Optuna found a point near the global minimum.
The result 0.02 is close to the true minimum of 0."
```

### 4.3 Constrained Problem with IPOPT

```
User: "Solve the constrained optimization problem 'constrained_test'"

LLM calls: get_problem_info("constrained_test")

Tool returns:
{
    "dimension": 50,
    "num_inequality_constraints": 3,
    "num_equality_constraints": 1,
    "description": "Large constrained NLP"
}

LLM reasons:
"This is a constrained problem with:
- 50 variables (medium scale)
- 3 inequality + 1 equality constraints
- Needs an optimizer that handles general constraints well

IPOPT is ideal for this - it's designed for large-scale
constrained nonlinear optimization. Key options:
- mu_strategy='adaptive' for robustness
- tol=1e-6 for standard accuracy"

LLM calls: list_available_optimizers()

Tool returns: {"ipopt": {"available": true, ...}}

LLM calls: run_optimization(
    problem_id="constrained_test",
    optimizer="ipopt",
    config='{"tol": 1e-6, "mu_strategy": "adaptive", "max_iter": 500}',
    max_iterations=500
)
```

---

## 5. Key Design Decisions

### 5.1 Configuration is LLM Intelligence

The LLM decides configuration based on:
- **Problem characteristics** (from `get_problem_info`)
- **Trained knowledge** of optimizer documentation
- **Past experience** (from `get_run_history`, future)

We do NOT hardcode:
- "If constrained, use SLSQP"
- "If priority=robust, set tol=1e-6"
- Algorithm classification tables

### 5.2 Initialization Unchanged (For Now)

Per user guidance, leave initialization logic as-is:
- Center of bounds for gradient methods
- Zero for shape optimization (domain_hint)
- Future: LLM can also reason about initialization

### 5.3 run_optimization is the Hub

Single execution tool that:
- Accepts optimizer specification and config
- Routes to appropriate backend
- Returns standardized results

LLM orchestrates the workflow:
1. Query problem info
2. Reason about optimizer selection
3. Construct config
4. Execute
5. Analyze results
6. Adapt if needed

---

## 6. Files to Change

### 6.1 Remove

```
DELETE: paola/agent/configuration.py     # Hardcoded intelligence
DELETE: paola/agent/initialization.py    # Keep logic, remove class (move to backends)
```

### 6.2 Create

```
CREATE: paola/optimizers/__init__.py
CREATE: paola/optimizers/base.py         # OptimizerBackend ABC
CREATE: paola/optimizers/scipy_backend.py
CREATE: paola/optimizers/ipopt_backend.py
CREATE: paola/optimizers/optuna_backend.py
```

### 6.3 Refactor

```
REFACTOR: paola/tools/optimization_tools.py
  - Keep: run_optimization (refactor to use backends)
  - Add: list_available_optimizers
  - Refactor: get_optimization_strategy → get_problem_info
```

### 6.4 Keep

```
KEEP: paola/foundry/bounds_spec.py       # Correct
KEEP: paola/foundry/nlp_schema.py        # Correct
KEEP: paola/tools/evaluator_tools.py     # Correct
KEEP: paola/tools/config_tools.py        # Useful for LLM to explore options
```

---

## 7. Testing Strategy

### 7.1 Benchmark Problems

Start with well-established benchmarks:
- **Rosenbrock**: Unimodal, tests gradient methods
- **Ackley**: Multi-modal, tests global optimizers
- **Rastrigin**: Highly multi-modal, challenging
- **Sphere**: Simple, sanity check
- **Beale**: 2D with known minimum

### 7.2 Test Scenarios

1. **Single optimizer**: Does run_optimization work correctly?
2. **Multiple optimizers**: Can LLM compare results?
3. **Configuration reasoning**: Does LLM adapt config based on problem?
4. **Failure recovery**: Does LLM try different approach after failure?

---

## 8. Summary

| Aspect | Previous (Wrong) | Revised (Correct) |
|--------|-----------------|-------------------|
| Algorithm selection | `ConfigurationManager.select_algorithm()` | LLM reasons from problem info |
| Configuration | Hardcoded priority mapping | LLM applies optimizer knowledge |
| Execution | `run_optimization` calls hardcoded logic | `run_optimization` executes LLM decisions |
| Multiple backends | Only SciPy | SciPy, IPOPT, Optuna, NLopt |
| Testing | Wing optimization | Benchmarks (Rosenbrock, Ackley, etc.) |

**The Paola Principle:**
- Tools provide information TO the LLM
- LLM reasons using trained knowledge
- Tools execute decisions FROM the LLM
- Configuration intelligence is in the LLM, not in Python

---

## Sources

- [cyipopt documentation](https://cyipopt.readthedocs.io/en/stable/)
- [IPOPT options](https://coin-or.github.io/Ipopt/OPTIONS.html)
- [Optuna documentation](https://optuna.readthedocs.io/)
- [NLopt Python reference](https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/)
- [SciPy optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html)
