# PAOLA Universal Architecture

**Package for Agentic Optimization with Learning and Analysis**

**Version**: 0.2.0 (Universal Vision - Revised)
**Date**: December 14, 2025
**Status**: Design Complete - Ready for Implementation

---

## Executive Summary

**PAOLA is an AI-powered optimization agent that makes solving optimization problems easy.**

### The Problem

You have an optimization problem to solve, but:
- Don't know which optimizer to use (30+ algorithms available, which is best?)
- Don't want to spend weeks learning optimization software and configuration
- Have a formulation and evaluator code, but need to connect them to an optimizer
- OR have a problem description that needs mathematical modeling

### PAOLA's Solution

**What you bring** (typical first user):
1. **Mathematical formulation** - Objectives, variables, constraints (you already have this)
2. **Digital evaluator** - Function that computes objective and gradient (you already have this)
3. OR just a problem description - PAOLA can help model it into optimization formulation

**What PAOLA provides**:
1. **Expert optimizer selection** - Knows which of 30+ algorithms to use for your problem
2. **Automatic integration** - Connects your formulation + evaluator + optimizer seamlessly
3. **Autonomous solving** - Runs optimization, adapts when issues arise, guides to solution
4. **Steering capability** - You can interrupt and guide if needed (like Claude Code)
5. **Continuous learning** - Gets better with every optimization

**No optimization expertise required** - You don't need to be an optimization expert to get good results.

---

## Part 1: Core Capabilities

PAOLA's value comes from four core capabilities:

### 1. Optimizer Expertise (Primary Value)

**The challenge**: 30+ optimization algorithms exist across 6+ libraries. Which one should you use? How to configure it?

**PAOLA's expertise**:
- **Deep knowledge** of algorithm characteristics:
  - SciPy (10+ algorithms): SLSQP, L-BFGS-B, COBYLA, Nelder-Mead, BFGS, Powell, TNC, CG
  - NLopt (20+ algorithms): BOBYQA, MMA, DIRECT, CRS2, NEWUOA, etc.
  - Optuna (Bayesian): TPE, CMA-ES, Grid Search, Random
  - Pymoo (Multi-objective): NSGA-II, NSGA-III, MOEA/D
  - DEAP (Evolutionary): GA, GP, ES, PSO
  - CasADi (Symbolic): IPOPT, SNOPT, WORHP

- **Intelligent selection** based on:
  - Problem structure (smooth, noisy, constrained, multi-objective)
  - Gradient availability (analytical, adjoint, finite-difference, none)
  - Computational budget (cheap, expensive, very expensive)
  - Dimensionality (low, medium, high)

- **Automatic configuration**:
  - Algorithm hyperparameters (tolerances, step sizes, population sizes)
  - Convergence criteria
  - Constraint handling methods

- **Adaptive strategies**:
  - Switches algorithms when stuck
  - Adjusts parameters when detecting issues
  - Reformulates problem when needed

**Value**: "You don't need to be an optimization expert - PAOLA brings the expertise"

---

### 2. Problem Modeling (LLM-Powered Capability)

**The challenge**: Translating real-world problems into mathematical optimization formulations.

**PAOLA's modeling capability**:

**Input formats** (flexible):
- **Natural language**: "minimize x² + 3x subject to x > 1"
- **Problem description**: "minimize drag on transonic airfoil while maintaining CL ≥ 0.5"
- **Python code**: Function definitions, classes, existing implementations
- **Structured data**: JSON, YAML, pandas DataFrames
- **Existing formulations**: Direct import from SciPy, Pyomo, CasADi formats

**Modeling assistance**:
- **LLM-powered translation**: Real-world description → mathematical formulation
- **Variable identification**: Extract design variables from problem context
- **Objective formulation**: Single or multi-objective, with appropriate formulation
- **Constraint extraction**: Equality, inequality, bounds from requirements
- **Problem classification**: Smooth, noisy, multi-modal, black-box

**Reformulation capabilities**:
- Variable scaling (normalize to [0, 1])
- Objective scaling (normalize to O(1))
- Constraint handling (log barrier, penalty, augmented Lagrangian)
- Multi-objective strategies (weighted sum, Pareto optimization)

**Realistic positioning**:
- ✅ **LLMs ARE capable** at mathematical modeling - don't shy away from this strength
- ✅ **PAOLA CAN help** users formulate problems from natural descriptions
- ❌ **NOT claiming** to be better than domain experts at complex practical problems
- ✅ **Recognition**: Most first users already have formulations ready

**Value**: "If you need help formulating your problem, PAOLA can assist - but most users bring formulations ready"

---

### 3. Easy Integration (Practical Value)

**The typical user scenario**:
- ✅ You have: Mathematical formulation (objectives, variables, constraints)
- ✅ You have: Digital evaluator (Python function, CFD script, simulation code)
- ❌ You don't have: Optimizer implementation and configuration
- ❌ You don't have: Easy way to connect formulation + evaluator + optimizer

**PAOLA's integration value**:

**Minimal setup** - Three steps to start optimizing:
```python
# Step 1: Describe your problem (or import existing formulation)
problem = paola.Problem(
    objectives=[Objective("drag", sense="minimize")],
    variables=[Variable("alpha", bounds=(0, 10))],
    constraints=[Constraint("lift >= 0.5")]
)

# Step 2: Provide your evaluator (you already have this)
def my_evaluator(design):
    # Your existing CFD code, ML training, or any function
    drag, lift = run_my_simulation(design)
    return {"drag": drag}, {"lift": lift}

# Step 3: Let PAOLA solve it
result = paola.optimize(problem, evaluator=my_evaluator)
# PAOLA selects optimizer, configures, runs, adapts, solves
```

**Backend interfacing** (engineering design optimization):
- **Physics engine connections**: CFD (SU2, OpenFOAM), FEA (FEniCS, Abaqus), multiphysics
- **Adjoint gradient integration**: Connect optimizer to adjoint solvers seamlessly
- **Multi-fidelity support**: Low/medium/high fidelity evaluations
- **Job scheduling**: HPC cluster integration, parallel evaluations
- **Cost tracking**: CPU hours, wall time, budget management

**User-provided evaluators** (most common case):
- Python function
- Shell script calling external software
- REST API endpoint
- Cloud function
- Existing simulation workflow

**Value**: "You have the pieces (formulation + evaluator), PAOLA connects them and solves"

---

### 4. Learning Infrastructure (Continuous Improvement)

**The foundry** - Single source of truth for optimization data:
- **Stores all optimizations**: Formulations, runs, results, decisions
- **Lineage tracking**: Which runs derive from which, adaptations made
- **Query interface**: Find similar problems, compare strategies
- **Versioning**: Problem evolution over time

**Knowledge base** - Accumulates expertise:
- **RAG-based retrieval**: Find similar past optimizations
- **Problem signatures**: Mathematical structure + computational characteristics
- **Strategy patterns**: What worked for similar problems
- **Cross-domain learning**: Patterns transfer across domains

**Continuous improvement**:
- Every optimization contributes to knowledge base
- Agent learns: "Problems with structure X succeed with algorithm Y"
- Warm-starting: New problems benefit from past experience
- Performance: 30-40% faster convergence over time

**Value**: "PAOLA gets smarter with every optimization you run"

---

## Part 2: Architecture Overview

### Layered Architecture

```
┌──────────────────────────────────────────────────────────────┐
│               APPLICATION LAYER                              │
│  - CLI (interactive REPL) ✅ Implemented                     │
│  - Python SDK (programmatic access) ✅ Implemented           │
│  - Future: Web UI, REST API, Jupyter integration             │
└───────────────────────┬──────────────────────────────────────┘
                        ↓
┌───────────────────────▼──────────────────────────────────────┐
│         AGENT LAYER (Expert Autonomous Worker)               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ ReAct Agent ✅ Implemented                           │   │
│  │  - Autonomous reasoning loop (Observe → Reason → Act)│   │
│  │  - Optimizer expertise (selection, configuration)    │   │
│  │  - Modeling assistance (LLM-powered)                 │   │
│  │  - Adaptive strategies (constraint mgmt, switching)  │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Steering Interface (Phase 9+)                        │   │
│  │  - User can interrupt autonomous workflow            │   │
│  │  - Guide decisions (like Claude Code)                │   │
│  │  - Collaborative problem-solving                     │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────┬──────────────────────────────────────┘
                        ↓
┌───────────────────────▼──────────────────────────────────────┐
│           CAPABILITY LAYER (Core Components)                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. OPTIMIZER EXPERTISE (30+ algorithms)              │   │
│  │    - SciPy ✅, NLopt, Optuna, Pymoo, DEAP, CasADi    │   │
│  │    - Intelligent selection and configuration         │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 2. PROBLEM MODELING (LLM-powered)                    │   │
│  │    - Parsers (natural language, code, structured)    │   │
│  │    - Mathematical modeling (real-world → formulation)│   │
│  │    - Reformulation (scaling, penalty, transformations)│  │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 3. BACKEND INTERFACING (Physics engines)             │   │
│  │    - Analytical ✅, CFD (SU2), FEA, ML training       │   │
│  │    - User-provided evaluators (most common)          │   │
│  │    - Adjoint gradient integration                    │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────┬──────────────────────────────────────┘
                        ↓
┌───────────────────────▼──────────────────────────────────────┐
│      INFRASTRUCTURE LAYER (Learning & Data)                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Foundry (Data Foundation) ✅ Implemented             │   │
│  │  - Single source of truth for all optimizations      │   │
│  │  - Run management, lineage tracking, versioning      │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Knowledge Base ✅ Skeleton, Phase 8: Full RAG        │   │
│  │  - Experience accumulation across all optimizations  │   │
│  │  - Problem signatures, strategy patterns             │   │
│  │  - Cross-domain learning and transfer                │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Analysis ✅ Implemented                              │   │
│  │  - Deterministic metrics (instant, free)             │   │
│  │  - AI-powered diagnosis (strategic, ~$0.02)          │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### Agent Execution Model

**PRIMARY MODE: Autonomous Worker**
```
User: "Minimize drag on airfoil, CL >= 0.5, use my CFD evaluator"
  ↓
Agent autonomously:
  1. Analyzes problem structure (smooth, constrained, expensive)
  2. Retrieves knowledge (similar airfoil problems succeeded with SLSQP)
  3. Selects optimizer (SLSQP recommended, fallback: COBYLA)
  4. Connects formulation + user evaluator + SLSQP
  5. Runs optimization (monitors progress, adapts if issues)
  6. Returns solution
  ↓
User: [Gets solution, PAOLA handled everything]
```

**SECONDARY MODE: User Steering** (Phase 9+, like Claude Code)
```
User: "Minimize drag on airfoil, CL >= 0.5"
  ↓
Agent: "I recommend SLSQP (gradient-based, good for smooth constrained problems)"
  ↓
User: [Interrupts] "Actually, use Optuna instead, I want Bayesian optimization"
  ↓
Agent: "Switching to Optuna with TPE sampler..."
  ↓
Agent: [After 20 iterations] "CL constraint violated repeatedly, should I tighten?"
  ↓
User: "Yes, tighten by 2%"
  ↓
Agent: "Tightening CL >= 0.5 to CL >= 0.51, restarting..."
  ↓
[Collaborative solving continues]
```

**Flexibility**: Start autonomous, user can steer anytime

---

## Part 3: Module Structure

### 3.1 Agent (Expert Autonomous Worker)

**Location**: `paola/agent/`

**Purpose**: Expert optimization consultant with autonomous execution

```python
# paola/agent/react_agent.py
def build_optimization_agent(llm_model, tools, callback_manager):
    """
    Build ReAct agent with optimization expertise.

    Agent capabilities:
    1. Optimizer selection (knows 30+ algorithms)
    2. Problem modeling (LLM-powered)
    3. Adaptive strategies (constraint mgmt, switching)
    4. Autonomous execution (observe → reason → act)
    5. Steering interface (user can guide)
    """

# paola/agent/prompts.py
OPTIMIZER_EXPERTISE_PROMPT = """
You are an expert optimization consultant with deep knowledge of:
- 30+ optimization algorithms across 6 libraries
- When to use gradient-based vs derivative-free methods
- How to handle constraints (penalty, barrier, augmented Lagrangian)
- Multi-objective optimization strategies
- Computational budget management

Given a problem, you autonomously:
1. Analyze structure (smooth, noisy, constrained, etc.)
2. Select best optimizer based on characteristics
3. Configure algorithm parameters
4. Connect formulation + evaluator + optimizer
5. Monitor progress and adapt when needed

You work autonomously, but user can interrupt and steer.
"""

# paola/agent/steering.py (Phase 9+)
class SteeringInterface:
    """
    User steering capability (like Claude Code).

    User can:
    - Interrupt autonomous workflow
    - Override optimizer selection
    - Guide adaptation decisions
    - Request explanations
    """
```

**Status**:
- ✅ `react_agent.py` - Implemented (autonomous loop)
- ✅ `prompts.py` - Implemented (needs expansion with optimizer expertise)
- ⏳ `steering.py` - Phase 9+ (collaborative mode)

---

### 3.2 Optimizers (Expert Knowledge of 30+ Algorithms)

**Location**: `paola/optimizers/`

**Purpose**: Unified interface to all major optimization libraries

```python
# paola/optimizers/base.py
class OptimizerBackend(ABC):
    """Universal optimizer interface."""

    @abstractmethod
    def optimize(
        self,
        problem: Problem,
        evaluator: Callable,
        initial_design: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """Run optimization."""

    @property
    def algorithm_name(self) -> str:
        """Algorithm identifier (e.g., 'SLSQP', 'Optuna-TPE')."""

    @property
    def requires_gradients(self) -> bool:
        """Does this algorithm need gradients?"""

    @property
    def supports_constraints(self) -> bool:
        """Can handle constraints?"""

    @property
    def supports_multi_objective(self) -> bool:
        """Can handle multiple objectives?"""

    @property
    def recommended_for(self) -> List[str]:
        """Problem classes this algorithm excels at."""

# paola/optimizers/scipy_optimizer.py
class ScipyOptimizer(OptimizerBackend):
    """SciPy library wrapper."""

    ALGORITHMS = {
        "SLSQP": {
            "requires_gradients": True,
            "supports_constraints": True,
            "recommended_for": ["smooth_constrained", "small_medium"]
        },
        "L-BFGS-B": {
            "requires_gradients": True,
            "supports_constraints": False,  # Only bounds
            "recommended_for": ["smooth_unconstrained", "large_scale"]
        },
        "COBYLA": {
            "requires_gradients": False,
            "supports_constraints": True,
            "recommended_for": ["noisy", "derivative_free"]
        },
        # ... 7+ more algorithms
    }

# paola/optimizers/nlopt_optimizer.py (Phase 6)
class NLoptOptimizer(OptimizerBackend):
    """NLopt library wrapper (20+ algorithms)."""

    ALGORITHMS = {
        "LN_COBYLA": {"gradients": False, "constraints": True},
        "LN_BOBYQA": {"gradients": False, "constraints": False},
        "LD_SLSQP": {"gradients": True, "constraints": True},
        "LD_MMA": {"gradients": True, "constraints": True},
        "GN_DIRECT": {"gradients": False, "global": True},
        # ... 15+ more algorithms
    }

# paola/optimizers/optuna_optimizer.py (Phase 6)
class OptunaOptimizer(OptimizerBackend):
    """Optuna Bayesian optimization."""

    SAMPLERS = ["TPE", "CMA-ES", "Grid", "Random"]
    # Recommended for: expensive black-box, hyperparameter tuning

# paola/optimizers/pymoo_optimizer.py (Phase 7)
class PymooOptimizer(OptimizerBackend):
    """Pymoo multi-objective optimization."""

    ALGORITHMS = ["NSGA2", "NSGA3", "MOEAD", "AGEMOEA"]
    # Recommended for: multi-objective Pareto optimization

# paola/optimizers/selector.py
class OptimizerSelector:
    """
    Expert knowledge for optimizer selection.

    Decision logic:
    - Smooth + gradients + constraints → SLSQP
    - Smooth + gradients + large scale → L-BFGS-B
    - Noisy / no gradients → COBYLA, Nelder-Mead, Optuna
    - Multi-objective → NSGA-II, MOEA/D
    - Expensive black-box → Bayesian (Optuna TPE)
    - Global search → DIRECT, CRS2, Genetic algorithms
    """

    def recommend(
        self,
        problem: Problem,
        gradient_available: bool,
        evaluation_cost: str  # "cheap", "expensive", "very_expensive"
    ) -> List[str]:
        """Return ranked list of recommended algorithms."""
```

**Status**:
- ✅ `scipy_optimizer.py` - Implemented (10+ algorithms)
- ⏳ `nlopt_optimizer.py` - Phase 6
- ⏳ `optuna_optimizer.py` - Phase 6
- ⏳ `pymoo_optimizer.py` - Phase 7
- ⏳ `selector.py` - Phase 6 (expert selection logic)

---

### 3.3 Modeling (LLM-Powered Problem Formulation)

**Location**: `paola/modeling/`

**Purpose**: Mathematical modeling of real-world problems

```python
# paola/modeling/parsers.py
class ProblemParser:
    """Parse various input formats to OptimizationProblem."""

    def from_natural_language(self, description: str) -> Problem:
        """
        LLM-powered parsing of natural language.

        Examples:
        - "minimize x^2 + 3x subject to x > 1"
        - "minimize drag on airfoil, maintain CL >= 0.5"
        - "maximize portfolio return, keep risk below 0.2"

        Uses LLM to extract:
        - Objectives (minimize/maximize)
        - Variables (with bounds)
        - Constraints (equality/inequality)
        """

    def from_code(self, code: str) -> Problem:
        """
        Parse Python code to optimization problem.

        Accepts:
        - Function definitions
        - Class-based problem definitions
        - NumPy expressions
        """

    def from_structured(self, data: Dict) -> Problem:
        """
        Parse structured data (JSON, YAML, dict).

        Standard schema:
        {
          "objectives": [{"name": "f", "sense": "minimize", ...}],
          "variables": [{"name": "x", "bounds": [0, 10], ...}],
          "constraints": [{"type": "inequality", ...}]
        }
        """

    def from_existing_format(self, source: str, format: str) -> Problem:
        """
        Import from existing optimization frameworks.

        Formats:
        - scipy: SciPy minimize format
        - pyomo: Pyomo ConcreteModel
        - casadi: CasADi Opti stack
        """

# paola/modeling/modeling.py
class MathematicalModeling:
    """
    LLM-powered mathematical modeling assistance.

    Capabilities:
    - Real-world description → optimization formulation
    - Variable identification from context
    - Objective function formulation
    - Constraint extraction
    - Problem classification
    """

    def model_problem(
        self,
        description: str,
        domain: Optional[str] = None
    ) -> Problem:
        """
        Model real-world problem into optimization formulation.

        LLM-powered translation with domain knowledge:
        - Engineering: Identify design variables, physics-based objectives
        - ML: Hyperparameters, architecture search, training objectives
        - Finance: Asset weights, risk/return formulation
        - Operations: Scheduling variables, resource constraints

        Note: Don't claim superiority over domain experts,
              but LLMs ARE capable at mathematical modeling.
        """

# paola/modeling/reformulation.py
class ProblemReformulation:
    """Problem transformations to improve optimization."""

    def scale_variables(self, problem: Problem) -> Problem:
        """Normalize variables to [0, 1]."""

    def scale_objective(self, problem: Problem) -> Problem:
        """Normalize objective to O(1)."""

    def constraint_to_penalty(self, problem: Problem) -> Problem:
        """Convert constraints to penalty terms."""

    def constraint_to_barrier(self, problem: Problem) -> Problem:
        """Convert inequality constraints to log barrier."""

    def augmented_lagrangian(self, problem: Problem) -> Problem:
        """Lagrangian penalty formulation."""

# paola/modeling/schema.py (✅ Already implemented)
class OptimizationProblem:
    """Pydantic schema for optimization problems."""

    problem_type: str
    objectives: List[Objective]
    variables: List[Variable]
    constraints: List[Constraint]
    metadata: Dict[str, Any]
```

**Status**:
- ✅ `schema.py` - Implemented (Pydantic schemas)
- ⏳ `parsers.py` - Phase 6 (natural language, code, structured)
- ⏳ `modeling.py` - Phase 6 (LLM-powered modeling)
- ⏳ `reformulation.py` - Phase 6 (transformations)

---

### 3.4 Backends (Physics Engine Interfacing)

**Location**: `paola/backends/`

**Purpose**: Connect optimizer to evaluation engines

**Note**: This is what "easy to interface" means - connecting optimizer to physics engines (CFD, FEA, etc.)

```python
# paola/backends/base.py
class EvaluationBackend(ABC):
    """
    Universal interface for evaluation backends.

    Backends types:
    1. User-provided functions (most common first user case)
    2. Analytical test functions
    3. Physics engines (CFD, FEA)
    4. ML training
    5. External APIs
    """

    @abstractmethod
    def evaluate(self, design: np.ndarray) -> EvaluationResult:
        """
        Evaluate objective and constraints.

        Returns:
            EvaluationResult with:
            - objectives: Dict[str, float]
            - constraints: Dict[str, float]
            - cost: float  # Computational cost (CPU hours)
        """

    @abstractmethod
    def compute_gradient(
        self,
        design: np.ndarray,
        method: str = "auto"
    ) -> np.ndarray:
        """
        Compute gradient.

        Methods:
        - "auto": Use best available
        - "adjoint": Adjoint solver (CFD/FEA)
        - "finite_difference": Numerical approximation
        - "symbolic": Symbolic differentiation
        """

# paola/backends/user_function.py
class UserFunctionBackend(EvaluationBackend):
    """
    User-provided evaluator (most common case).

    User brings:
    - Python function
    - Shell script
    - REST API endpoint
    - Existing simulation code

    PAOLA connects it to optimizer.
    """

    def __init__(self, user_function: Callable):
        self.user_function = user_function

    def evaluate(self, design):
        # Call user's function
        return self.user_function(design)

# paola/backends/analytical.py (✅ Implemented)
class AnalyticalBackend(EvaluationBackend):
    """Mathematical test functions."""

    FUNCTIONS = {
        "rosenbrock": rosenbrock_function,
        "sphere": sphere_function,
        "rastrigin": rastrigin_function,
        # 20+ standard test functions
    }

# paola/backends/cfd.py (Phase 7)
class SU2Backend(EvaluationBackend):
    """
    SU2 CFD solver integration (flagship showcase).

    Physics engine interfacing:
    - Mesh parameterization (FFD, CST)
    - SU2 execution (subprocess, HPC cluster)
    - Result parsing (forces, moments)
    - Adjoint gradient integration (seamless connection)
    - Multi-fidelity (coarse/medium/fine mesh)
    """

    def evaluate(self, design, fidelity="medium"):
        # 1. Parameterize mesh
        mesh = self.parameterize_mesh(design)

        # 2. Run SU2
        result = self.run_su2(mesh, fidelity=fidelity)

        # 3. Parse results
        return EvaluationResult(
            objectives={"drag": result.CD},
            constraints={"lift": result.CL - self.target_CL}
        )

    def compute_gradient(self, design, method="adjoint"):
        if method == "adjoint":
            # Run SU2 adjoint solver (seamless integration)
            return self._run_su2_adjoint(design)
        else:
            return self._finite_difference(design)

# paola/backends/ml.py (Phase 7)
class MLBackend(EvaluationBackend):
    """ML hyperparameter tuning backend."""

    def evaluate(self, hyperparameters):
        # Train model, return validation loss
        model = self.create_model(hyperparameters)
        model.fit(self.train_data)
        loss = model.evaluate(self.val_data)
        return EvaluationResult(objectives={"val_loss": loss})
```

**Status**:
- ✅ `analytical.py` - Implemented (20+ test functions)
- ⏳ `user_function.py` - Phase 6 (wrapper for user functions)
- ⏳ `cfd.py` - Phase 7 (SU2 integration)
- ⏳ `ml.py` - Phase 7 (PyTorch/TensorFlow)

---

### 3.5 Foundry (Data Foundation)

**Location**: `paola/foundry/`

**Purpose**: Single source of truth for all optimization data

```python
# paola/foundry/foundry.py (✅ Implemented)
class OptimizationFoundry:
    """
    Data foundation for optimization runs.

    Stores:
    - Problem definitions
    - Run records (iterations, results)
    - Lineage (which runs derive from which)
    - Decisions (optimizer selections, adaptations)
    """

    def create_run(
        self,
        problem_id: str,
        algorithm: str,
        metadata: Dict[str, Any]
    ) -> Run:
        """Create tracked optimization run."""

    def query_runs(
        self,
        algorithm: Optional[str] = None,
        success: Optional[bool] = None
    ) -> List[RunRecord]:
        """Query runs with filters."""

# paola/foundry/run.py (✅ Implemented)
class Run:
    """Active optimization run."""

    def record_iteration(self, design, objective, gradient):
        """Record iteration and auto-persist."""

    def finalize(self, result):
        """Finalize with result and persist."""

class RunRecord:
    """Storage representation of completed run (immutable)."""
```

**Status**: ✅ Fully implemented (Phases 1-5)

---

### 3.6 Knowledge (Experience Accumulation)

**Location**: `paola/knowledge/`

**Purpose**: Learn from every optimization

```python
# paola/knowledge/signature.py
class ProblemSignature:
    """Characteristics for similarity matching."""

    # Mathematical structure
    dimension: int
    constraint_count: int
    problem_class: str  # "smooth_constrained", "noisy", etc.

    # Computational characteristics
    gradient_available: bool
    evaluation_cost: str  # "cheap", "expensive", "very_expensive"

    def similarity(self, other) -> float:
        """Compute similarity score [0, 1]."""

# paola/knowledge/rag.py (Phase 8)
class RAGKnowledgeBase:
    """RAG-based knowledge retrieval."""

    def store_insight(self, insight: OptimizationInsight):
        """
        Store successful optimization experience.

        Insight includes:
        - Problem signature
        - Optimizer used (and why it worked)
        - Adaptations made
        - Outcome
        """

    def retrieve_similar(
        self,
        signature: ProblemSignature,
        k: int = 5
    ) -> List[OptimizationInsight]:
        """Retrieve k most similar past optimizations."""
```

**Status**:
- ✅ Skeleton implemented (Phases 3-5)
- ⏳ Full RAG implementation (Phase 8)

---

### 3.7 Tools (Agent Primitives)

**Location**: `paola/tools/`

**Purpose**: 35+ tools that agent uses to solve problems

**Categories**:
1. **Problem Formulation** (5 tools) - Parse natural language, code, structured formats
2. **Optimizer Selection** (6 tools) - Recommend, compare, tune algorithms
3. **Backend Integration** (4 tools) - Connect evaluators, multi-fidelity
4. **Observation** (8 tools) - Detect issues, analyze convergence
5. **Adaptation** (7 tools) - Constraint management, reformulation, switching
6. **Budget Management** (2 tools) - Track cost, estimate remaining
7. **Knowledge** (3 tools) - Store/retrieve insights

**Status**:
- ✅ 12 tools implemented (Phases 1-5)
- ⏳ 23 tools to implement (Phases 6-8)

---

## Part 4: Implementation Plan

### Phase 6: Universal Adaptation & Optimizer Integration (4-5 weeks)

**Week 1: Problem Modeling & Parsing**
- Natural language parser (LLM-based)
- Code parser (AST analysis)
- Structured parser (JSON/YAML)
- User function wrapper

**Week 2: Optimizer Integration**
- NLopt wrapper (20+ algorithms)
- Optuna wrapper (Bayesian optimization)
- Optimizer selector (expert decision logic)
- `recommend_optimizers()` tool

**Weeks 3-4: Observation & Adaptation**
- 4 observation tools (infeasibility, gradient noise, constraint activity, convergence)
- 3 constraint management tools (tighten, relax, restart)
- 2 gradient tools (FD, adjoint prep)

**Week 5: Testing & Integration**
- Agent prompts (optimizer expertise)
- End-to-end tests (agent selects optimizers correctly)
- User function integration tests

**Deliverable**: Agent autonomously selects optimizers, parses problems, adapts intelligently

---

### Phase 7: Backend Interfacing & Multi-Fidelity (4-5 weeks)

**Week 1: Backend Framework**
- `EvaluationBackend` interface refinement
- Multi-fidelity support (low/medium/high)
- Cost tracking (CPU hours, budget)

**Week 2: User Function Backend**
- Wrapper for user-provided evaluators
- Gradient estimation (finite-difference)
- Error handling and validation

**Week 3: CFD Backend (SU2)**
- Mesh parameterization (FFD or CST)
- SU2 execution (subprocess, HPC)
- Result parsing
- Adjoint gradient integration

**Week 4: ML Backend**
- PyTorch model training wrapper
- Hyperparameter encoding/decoding
- Training cost tracking

**Week 5: Multi-Fidelity & Budget**
- Fidelity recommendation logic
- Budget management
- Agent fidelity strategy

**Deliverable**: PAOLA interfaces analytical, user functions, CFD, and ML backends

---

### Phase 8: Knowledge Accumulation (4-5 weeks)

**Weeks 1-2: Data Collection**
- Run 50+ optimizations:
  - 15× Analytical (various dimensions, constraints)
  - 20× User functions (from test users)
  - 15× CFD/ML (if backends ready)

**Week 3: Problem Signatures**
- Extract signatures from all 50+ runs
- Similarity computation
- Clustering analysis

**Week 4: RAG Implementation**
- Chroma vector database
- Embedding strategy
- Retrieval and ranking

**Week 5: Warm-Starting**
- `retrieve_optimization_knowledge()` - RAG retrieval
- Agent uses retrieved knowledge
- Measure convergence improvement (target: 30%)

**Deliverable**: Knowledge base with 50+ runs, warm-starting working

---

### Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 6 (Adaptation & Optimizers) | 4-5 weeks | Agent selects from 30+ algorithms, parses problems, adapts |
| Phase 7 (Backend Interfacing) | 4-5 weeks | User functions, CFD, ML backends working |
| Phase 8 (Knowledge) | 4-5 weeks | 50+ runs, RAG, warm-starting |
| **Total** | **12-15 weeks** | **Fully capable universal optimization agent** |

---

## Part 5: Value Propositions Summary

**PAOLA makes solving optimization problems easy:**

### For Users Who Have Formulations + Evaluators (Typical First User)

**You bring**:
- Mathematical formulation (objectives, variables, constraints) ✓ You have this
- Digital evaluator (function, CFD script, simulation code) ✓ You have this

**PAOLA provides**:
1. **Expert optimizer selection** - Knows which of 30+ algorithms to use
2. **Automatic integration** - Connects formulation + evaluator + optimizer
3. **Minimal setup** - 3 lines of code to start optimizing
4. **Autonomous solving** - Runs, monitors, adapts, solves
5. **Continuous learning** - Gets better with experience

**No expertise required**: You don't need to be an optimization expert

---

### For Users Who Need Modeling Help

**You bring**:
- Problem description (natural language, rough idea)

**PAOLA provides**:
1. **LLM-powered modeling** - Translates description → mathematical formulation
2. **Problem structuring** - Identifies variables, objectives, constraints
3. **Don't shy away** - LLMs ARE capable at mathematical modeling
4. **Realistic** - Not claiming to be better than domain experts
5. **Assistance available** - Help when you need it

**Value**: "If you need formulation help, PAOLA can assist"

---

### For Everyone

1. **Universal domain support** - Analytical, engineering, ML, finance, operations
2. **30+ optimizers** - All major libraries integrated (SciPy, NLopt, Optuna, Pymoo, DEAP, CasADi)
3. **Autonomous with steering** - Works autonomously, you can guide if needed
4. **Professional infrastructure** - Foundry for data, knowledge base for learning
5. **Gets smarter** - Every optimization improves future recommendations

---

## Part 6: Comparison with Existing Tools

### Domain-Specific Tools

| Tool | What They Do | Limitation |
|------|-------------|-----------|
| **HEEDS** | Engineering design optimization | Single domain, no learning, fixed algorithms |
| **ModeFRONTIER** | Engineering optimization platform | Single domain, manual setup, no agent |
| **Optuna** | ML hyperparameter tuning | Single domain (ML), user selects everything |
| **Pyomo** | Mathematical programming (OR) | Single domain, requires optimization expertise |

**PAOLA's difference**: Universal domains, autonomous agent, continuous learning

---

### Algorithm Libraries

| Library | What They Do | Limitation |
|---------|-------------|-----------|
| **SciPy** | 10+ optimization algorithms | User must select/configure, no guidance |
| **NLopt** | 20+ algorithms (C library) | User must know which to use, manual setup |
| **Pymoo** | Multi-objective optimization | User must understand MOO, configure everything |

**PAOLA's difference**: Integrates ALL libraries, agent selects/configures automatically, no expertise required

---

### PAOLA's Unique Position

**First tool that**:
1. Integrates 30+ algorithms across all major libraries
2. Has autonomous agent with optimizer expertise
3. Works across all domains (universal)
4. Learns continuously (knowledge accumulation)
5. Makes optimization accessible (no expertise required)

---

## Part 7: Success Metrics

### Phase 6 Success Metrics

- ✅ Agent correctly selects optimizer for 20+ diverse problems (80%+ accuracy)
- ✅ Natural language parsing works for common problem descriptions (90%+ success)
- ✅ Agent detects and resolves infeasibility/gradient noise (5/5 test cases)
- ✅ NLopt integration: 20+ algorithms accessible

### Phase 7 Success Metrics

- ✅ User function backend: Easy 3-line setup for user evaluators
- ✅ CFD backend: RAE2822 optimization converges to literature values (±5%)
- ✅ Multi-fidelity reduces cost by 5× (vs all-medium)
- ✅ ML backend: Tunes PyTorch model better than random search

### Phase 8 Success Metrics

- ✅ Knowledge base: 50+ runs collected
- ✅ Warm-starting: 30% fewer iterations on similar problems
- ✅ RAG retrieval: Top-3 similarity > 0.7
- ✅ Agent autonomously uses knowledge (without prompting)

### Overall Success (End of Phase 8)

- ✅ End-to-end workflow: User provides formulation + evaluator → PAOLA solves
- ✅ 90% success rate on diverse problems (vs 50% baseline with manual selection)
- ✅ User testimonial: "I didn't know which optimizer to use, PAOLA figured it out"
- ✅ Learning verified: Optimization #100 is 30% faster than optimization #1

---

## Summary

**PAOLA v0.2.0: AI-powered optimization agent that makes solving optimization problems easy**

### Core Capabilities

1. **Optimizer Expertise** - Knows which of 30+ algorithms to use and how to configure them
2. **Problem Modeling** - LLM-powered formulation assistance (don't shy away, but don't over-promise)
3. **Easy Integration** - You bring formulation + evaluator, PAOLA connects them with optimizer
4. **Learning Infrastructure** - Gets smarter with every optimization (foundry + knowledge base)

### Execution Model

- **PRIMARY**: Autonomous worker (you describe, PAOLA solves)
- **SECONDARY**: User steering (interrupt and guide, like Claude Code)

### Typical User Journey

```
You have:
  ✓ Mathematical formulation (objectives, variables, constraints)
  ✓ Digital evaluator (Python function, CFD script, etc.)

You don't have:
  ✗ Knowledge of which optimizer to use
  ✗ Optimizer implementation
  ✗ Easy way to connect everything

PAOLA provides:
  → Expert optimizer selection (30+ algorithms)
  → Automatic integration (3 lines of code)
  → Autonomous solving (monitor, adapt, solve)
  → Continuous learning (gets better over time)

Result:
  ✓ Optimal solution
  ✓ No optimization expertise required
```

### Timeline

- **Phase 6** (4-5 weeks): Optimizer integration, problem modeling, adaptation
- **Phase 7** (4-5 weeks): Backend interfacing (user functions, CFD, ML)
- **Phase 8** (4-5 weeks): Knowledge accumulation (50+ runs, RAG)
- **Total**: 12-15 weeks to fully capable universal optimization agent

### Next Step

**Phase 6 Week 1**: Implement problem modeling and parsing tools

---

**Version**: 0.2.0 (Universal Vision - Revised)
**Date**: December 14, 2025
**Status**: Design Complete ✅
**Ready for**: Implementation Phase 6
