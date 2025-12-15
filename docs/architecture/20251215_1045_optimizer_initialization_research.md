# Optimizer Initialization and Bounds Handling Research

**Status**: Complete
**Date**: December 15, 2025
**Purpose**: Inform PAOLA's design decisions for handling large variable spaces

---

## Executive Summary

This document surveys how different optimization software handles:
1. Initial point specification (required vs optional, defaults)
2. Variable bounds specification
3. Large-scale problem handling

### Key Findings

| Optimizer Type | Initial Point | Default Strategy | Bounds Required |
|---------------|---------------|------------------|-----------------|
| **Gradient-based (scipy, SLSQP)** | Required | None (must specify) | Optional |
| **Interior-point (IPOPT)** | Required | Automatic push to interior | Required |
| **SQP (SNOPT)** | Required | User-specified | Required |
| **Evolutionary (NSGA-II)** | Population | Random/LHS within bounds | Required |
| **CMA-ES** | Mean + Sigma | User-specified mean | Optional |
| **Bayesian (Optuna)** | None | Random sampling | Required (search space) |
| **Commercial (GUROBI)** | Optional | Solver-determined | Yes |

**Design Implication for PAOLA**: Bounds are universally required; initial point handling varies significantly by algorithm class.

---

## 1. Gradient-Based Optimizers

### 1.1 SciPy minimize

**Source**: [SciPy minimize documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)

**Initial Point (`x0`)**:
- **Required**: Yes, `x0` is a required positional argument
- **No default**: Must be explicitly provided
- **Type**: Array of real elements, shape `(n,)`

**Bounds**:
- Optional for most methods
- Format: `[(lb, ub), ...]` or `Bounds` object
- If initial guess is within bounds, all subsequent evaluations stay within bounds

**Key Insight**: SciPy requires explicit initial point with no automatic generation.

### 1.2 SNOPT

**Source**: [SNOPT: An SQP Algorithm for Large-Scale Constrained Optimization](https://web.stanford.edu/group/SOL/papers/SNOPT-SIGEST.pdf)

**Initial Point**:
- Required
- "Without help, SNOPT will not be able to find an optimal solution"
- Nonlinear functions evaluated only at points satisfying bounds

**Bounds**:
- Required for constrained problems
- Supports `-INF`/`+INF` for infinite bounds
- Fixed variables: equal lower and upper bounds
- Free variables: both bounds infinite

**Large-Scale Handling**:
- Efficient for problems with thousands of constraints and variables
- Uses sparse matrix techniques
- Limited-memory quasi-Newton approximation

### 1.3 NLopt

**Source**: [NLopt Documentation](https://nlopt.readthedocs.io/en/latest/NLopt_Reference/)

**Initial Point**:
- Required: Input `x` array provides initial guess
- On return: `x` contains optimized values

**Bounds**:
- **Guaranteed enforcement**: "NLopt guarantees that all intermediate steps will satisfy the bound constraints"
- Initial step size derived heuristically from bounds
- For BOBYQA: supports unequal initial-step sizes (rescales internally)

**Key Insight**: NLopt uses bounds to derive initial step sizes for derivative-free methods.

---

## 2. Interior-Point Methods

### 2.1 IPOPT

**Sources**:
- [IPOPT Options](https://coin-or.github.io/Ipopt/OPTIONS.html)
- [pyOptSparse IPOPT](https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/optimizers/IPOPT.html)

**Initial Point**:
- Required
- **Automatic adjustment**: IPOPT moves initial values close to bounds into interior
- Controlled by `bound_push` and `bound_frac` options

**Initialization Options**:
```
bound_mult_init_method:
  - "constant": All multipliers = bound_mult_init_val
  - "mu-based": mu_init / slack (good if starting near optimal)

warm_start_init_point:
  - yes: Pass primal and dual values
  - no: Start fresh

mu_init: Initial barrier parameter (default 0.1)
  - Smaller value (1e-5) if starting near feasible
```

**Automatic Initialization**:
- Can solve least-squares for primal variables
- Useful when user "doesn't know anything about starting point"

**Scaling**:
- Default: scales based on gradient at initial point
- Problematic for warm-starting (scaling depends on start point)

**Key Insight**: IPOPT has sophisticated automatic initialization but benefits from good starting points.

### 2.2 GUROBI

**Sources**: [Gurobi Help Center](https://support.gurobi.com/hc/en-us/community/posts/13300150517521-set-the-initial-values-of-variables-to-warmstart)

**Initial Point**:
- **Optional** for LP/QP
- Methods: `Start` attribute, `.mst` file, `.sol` file

**Warm Start**:
- LP: Automatic if constraint modified (dual simplex)
- MIP: `VarHintVal` for heuristic guidance
- "Not necessary to provide a value for each variable"

**Key Options**:
```
LPWarmStart = 2: Retain presolve benefits with warm start
MIPfocus = 2: When start point is likely optimal (focus on proving)
```

**Key Insight**: GUROBI allows partial initialization - not all variables need values.

---

## 3. Evolutionary Algorithms

### 3.1 NSGA-II (pymoo)

**Sources**:
- [pymoo Sampling](https://pymoo.org/operators/sampling.html)
- [pymoo Biased Initialization](https://pymoo.org/customization/initialization.html)

**Initial Population**:
- Three options:
  1. `Sampling` implementation (random method)
  2. `Population` object (pre-defined solutions)
  3. NumPy array `(n_individuals, n_var)`

**Default Sampling Methods**:
- `FloatRandomSampling`: Default for continuous
- Latin Hypercube Sampling (LHS): Better coverage

**Bounds**:
- **Required** for global optimization
- "Essential part since entire design space within bounds is searched"
- Bounds too large → slow convergence
- Bounds too small → may miss global optimum

**Research Finding** ([SpringerLink](https://link.springer.com/chapter/10.1007/978-3-540-85646-7_12)):
- "Initial population plays an important role in convergence"
- "Well-distributed sampling increases robustness"
- "Avoids premature convergence"

**Key Insight**: Evolutionary algorithms need bounds; LHS improves convergence over random.

### 3.2 CMA-ES

**Sources**:
- [CMA-ES Wikipedia](https://en.wikipedia.org/wiki/CMA-ES)
- [CMA-ES Tutorial](https://arxiv.org/pdf/1604.00772)

**Required Parameters**:
1. **Initial mean `m`**: Starting point in R^n
2. **Initial sigma `σ`**: Step-size (standard deviation)
3. Covariance matrix `C = I` (identity, automatic)

**Sigma Selection**:
- "Should be about 1/4th of search domain width"
- Variables should be scaled to similar sensitivity

**Bounds**:
- Optional but supported: `enforce_bounds=[[-2, 2], [-1, 1]]`
- Best for dimensions 3-100

**Key Insight**: CMA-ES requires mean and sigma, not explicit x0.

---

## 4. Bayesian Optimization

### 4.1 Optuna

**Sources**:
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Neptune.ai Tutorial](https://neptune.ai/blog/how-to-optimize-hyperparameter-search)

**Initial Values**:
- **Not required**: Sampler generates initial points
- TPE (default): Random initial samples, then model-guided

**Search Space**:
- Defined via `suggest_*` methods (define-by-run)
- `trial.suggest_float("x", -5, 5)` defines bounds implicitly

**Initialization Process**:
1. Random initial samples
2. Build surrogate model
3. Acquisition function guides next point
4. Repeat until budget exhausted

**Population Size**:
- Automatic by default
- Override with `population_size` parameter

**Key Insight**: Optuna doesn't need initial point; search space definition includes bounds.

---

## 5. Commercial Platforms

### 5.1 ModeFRONTIER

**Sources**: [ModeFRONTIER Training](https://hopsan.github.io/tutorials/tutorial_modefrontier.pdf)

**Variable Definition**:
- Each variable (including bounds) is a node in workflow
- Planner interface for bounds/domain settings

**Initialization**:
- Random initial distribution
- **Autonomous mode**: Auto-setup without user knowledge
- pilOPT: Self-adaptive, one-click optimization

**Bounds Reduction**:
- Can reduce bounds to "optimal cluster" during optimization
- Improves efficiency in subsequent phases

### 5.2 HEEDS

**Sources**: [Siemens HEEDS](https://plm.sw.siemens.com/en-US/simcenter/integration-solutions/heeds/)

**SHERPA Algorithm**:
- Hybrid adaptive method
- Self-tuning based on problem characteristics
- Switches strategies automatically

**Key Insight**: Commercial tools emphasize autonomous/adaptive initialization.

---

## 6. Engineering Framework: Dakota

**Source**: [Dakota Documentation](https://snl-dakota.github.io/docs/6.20.0/users/usingdakota/reference/variables.html)

**Variable Specification**:
```dakota
variables
  continuous_design = 2
    initial_point    0.9    1.1
    upper_bounds     5.8    2.9
    lower_bounds     0.5   -2.9
    descriptors      'radius' 'location'
```

**Defaults**:
- `initial_point`: **Defaults to 0** if not specified
- `upper_bounds`: DBL_MAX (platform max)
- `lower_bounds`: -DBL_MAX

**Range Variables**:
- Doubly-bounded: Initialize to mean/midpoint
- Semi-bounded: Initialize to 0 if in range
- Uncertain variables: Initialize to mean

**Key Insight**: Dakota defaults to 0 or midpoint - practical engineering default.

---

## 7. Engineering Framework: pyOptSparse

**Source**: [pyOptSparse Documentation](https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/guide.html)

**Variable Definition**:
```python
optProb.addVarGroup(
    "xvars",
    nVars=3,
    varType="c",
    lower=[0, 0, 0],
    upper=[42, 42, 42],
    value=10  # Initial value (scalar applies to all)
)
```

**Initial Value**:
- Scalar: Same value for all variables
- Array: Per-variable values

**Hot Start**:
- History file enables restarts
- Cold start: Use final design from previous run

**Large-Scale Design**:
- Paper: "A Python framework for large-scale constrained nonlinear optimization of sparse systems"
- Supports thousands of variables efficiently

---

## 8. Domain-Specific: CFD Shape Optimization

**Sources**:
- [NASA FFD Paper](https://fun3d.larc.nasa.gov/AIAA-2004-4630-FFD.pdf)
- [SU2 Tutorial](https://su2code.github.io/tutorials/Inviscid_3D_Constrained_ONERAM6/)

**FFD (Free-Form Deformation)**:
- Control points parameterize deformation
- **Initial values = 0**: Baseline geometry has zero deformation

**Common Practice**:
```
Design variables: FFD control point movements
Initial values: All zeros (baseline shape)
Bounds: Symmetric, e.g., [-0.05, 0.05] meters
```

**Key Insight**: In shape optimization, **zero is the natural initial point** (no deformation from baseline).

---

## 9. Summary by Algorithm Class

### 9.1 Gradient-Based (Local)

| Aspect | Handling |
|--------|----------|
| Initial Point | **Required** (no default) |
| Bounds | Optional but recommended |
| Large Scale | Sparse linear algebra (IPOPT, SNOPT) |
| Best Practice | Start near expected optimum |

### 9.2 Population-Based (Global)

| Aspect | Handling |
|--------|----------|
| Initial Point | Population, not single point |
| Bounds | **Required** (defines search space) |
| Sampling | Random, LHS, or user-provided |
| Large Scale | Parallelization, surrogate-assisted |

### 9.3 Bayesian (Sample-Efficient)

| Aspect | Handling |
|--------|----------|
| Initial Point | **Not required** (random start) |
| Bounds | **Required** (search space definition) |
| Large Scale | Problematic (>100 dims challenging) |
| Best Practice | Good bounds, not initial point |

---

## 10. The Paola Principle: Final Recommendations

### 10.1 Core Insight

The research above reveals significant complexity in initialization across optimizer types:
- Gradient-based optimizers require explicit x0
- CMA-ES needs mean + sigma
- Bayesian methods need nothing (sampler handles it)
- Population-based methods need sampling strategies

**This complexity is exactly what Paola should abstract away.**

### 10.2 The Paola Principle

> **"Initialization is agent intelligence, not user input."**

Users come to Paola because they:
- Have an optimization problem to solve
- Don't want to become optimization experts
- Don't know (and shouldn't need to know) algorithm-specific initialization requirements

**Paola's value proposition**: "I know which optimizer needs what initialization, and I'll handle it for you."

### 10.3 Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Bounds | In problem definition | Mathematical property of NLP |
| Initial point | **Agent decides** | Expert knowledge, not user burden |
| Warm-starting | **Agent checks automatically** | Leverage run history |
| Domain knowledge | **Agent infers** | Shape opt → zero; general → center |

### 10.4 What Users Specify vs What Paola Handles

**Users specify (in problem definition):**
- Objective function (evaluator)
- Constraints
- Bounds (feasible region)
- Domain hint (optional: "shape_optimization")

**Paola handles internally:**
- Whether algorithm needs x0, population, or nothing
- Computing center of bounds
- Computing sigma for CMA-ES
- Checking for warm-start opportunities
- Latin Hypercube Sampling for evolutionary algorithms
- All algorithm-specific initialization logic

### 10.5 Paola's Initialization Decision Tree

```
1. Check for warm-start opportunities
   → Previous successful runs exist? → Use best solution

2. Check domain hint
   → "shape_optimization"? → Initialize at zero (baseline)

3. Apply algorithm-specific defaults
   → Gradient-based? → Center of bounds
   → Population-based? → LHS within bounds
   → CMA-ES? → Center + sigma = 0.25 × bound width
   → Bayesian? → Let sampler handle (no init needed)
```

### 10.6 Benefits

1. **No large vectors in tool calls** - init never in API
2. **Expert knowledge applied automatically** - research-backed strategies
3. **Intelligent warm-starting** - agent checks history
4. **Simpler user interface** - users focus on problem, not algorithm details
5. **Future-proof** - Paola can learn better strategies over time

### 10.7 Tool Signature Impact

```python
# BEFORE (user specifies initialization)
create_nlp_problem(
    problem_id="wing",
    bounds=[...],
    initial_point="center"  # User must know this
)

# AFTER (Paola handles initialization)
create_nlp_problem(
    problem_id="wing",
    bounds=[...],
    domain_hint="shape_optimization"  # Optional hint
    # NO initial_point - Paola decides
)
```

See `tools_optimization_foundry_design.md` for complete architecture.

---

## Sources

### Commercial Software
- [Gurobi Warm Start](https://support.gurobi.com/hc/en-us/community/posts/13300150517521-set-the-initial-values-of-variables-to-warmstart)
- [ModeFRONTIER Capabilities](https://engineering.esteco.com/modefrontier/modefrontier-capabilities/)

### Open-Source Libraries
- [SciPy minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
- [NLopt Documentation](https://nlopt.readthedocs.io/en/latest/NLopt_Reference/)
- [IPOPT Options](https://coin-or.github.io/Ipopt/OPTIONS.html)
- [pymoo Sampling](https://pymoo.org/operators/sampling.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Dakota Variables](https://snl-dakota.github.io/docs/6.20.0/users/usingdakota/reference/variables.html)
- [pyOptSparse Guide](https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/guide.html)

### Academic Papers
- [SNOPT Paper](https://web.stanford.edu/group/SOL/papers/SNOPT-SIGEST.pdf)
- [CMA-ES Tutorial](https://arxiv.org/pdf/1604.00772)
- [NASA FFD Paper](https://fun3d.larc.nasa.gov/AIAA-2004-4630-FFD.pdf)
- [Initial Population in MOGAs](https://link.springer.com/chapter/10.1007/978-3-540-85646-7_12)

### Tutorials
- [SU2 Shape Optimization](https://su2code.github.io/tutorials/Inviscid_3D_Constrained_ONERAM6/)
- [Neptune.ai Optuna Guide](https://neptune.ai/blog/how-to-optimize-hyperparameter-search)
