# Optimizer Configuration Options: Research and Paola's Role

**Status**: Complete
**Date**: December 15, 2025
**Purpose**: Analyze optimizer configuration complexity and define Paola's expert role

---

## Executive Summary

This document surveys the configuration options across major optimization packages and analyzes how Paola, as an expert optimization assistant, should handle this complexity.

### Key Finding

**Configuration complexity is overwhelming:**

| Optimizer | Total Options | Typical User Touches | Expert Might Touch |
|-----------|---------------|---------------------|-------------------|
| IPOPT | ~250 | 3-5 | 50-100 |
| SNOPT | ~100 | 3-5 | 30-50 |
| SciPy (all methods) | ~50 | 2-3 | 10-20 |
| NLopt | ~30 | 3-5 | 15-20 |
| CMA-ES (pycma) | ~50 | 2-3 | 15-25 |
| Optuna | ~40 | 1-2 | 10-15 |
| pymoo/NSGA-II | ~30 | 3-5 | 15-20 |

**Implication**: Even optimization experts only use a fraction of available options. The knowledge of *which* options matter for *which* problems is exactly Paola's core competence.

---

## 1. IPOPT: The Complexity Benchmark

**Source**: [IPOPT Options Reference](https://coin-or.github.io/Ipopt/OPTIONS.html)

### 1.1 Option Categories (22 categories, ~250 options)

| Category | # Options | Purpose |
|----------|-----------|---------|
| **Termination** | ~15 | When to stop (`tol`, `max_iter`, `acceptable_tol`) |
| **Output** | ~13 | Logging (`print_level`, `output_file`) |
| **NLP** | ~15 | Problem handling (`jacobian_approximation`) |
| **NLP Scaling** | ~6 | Numerical stability (`nlp_scaling_method`) |
| **Initialization** | ~8 | Starting point (`bound_push`, `least_square_init`) |
| **Warm Start** | ~8 | Reusing solutions (`warm_start_init_point`) |
| **Barrier Parameter** | ~25+ | Interior point core (`mu_strategy`, `mu_init`) |
| **Line Search** | ~35+ | Step selection (`line_search_method`, `alpha_red_factor`) |
| **Linear Solver** | ~4 | Backend selection (`linear_solver`: MA27, MA57, MUMPS, PARDISO) |
| **Step Calculation** | ~20+ | Direction computation (`mehrotra_algorithm`) |
| **Restoration Phase** | ~10 | Feasibility recovery (`expect_infeasible_problem`) |
| **Hessian Approximation** | ~10 | Second derivatives (`hessian_approximation`, `limited_memory_max_history`) |
| **Derivative Checker** | ~5 | Validation (`derivative_test`) |
| **MA27/MA57/PARDISO/MUMPS** | ~60+ | Solver-specific tuning |

### 1.2 User Tiers

**Beginner (3-5 options)**:
```python
options = {
    'tol': 1e-6,
    'max_iter': 1000,
    'print_level': 5
}
```

**Intermediate (15-25 options)**:
```python
options = {
    'tol': 1e-8,
    'max_iter': 3000,
    'linear_solver': 'ma57',
    'nlp_scaling_method': 'gradient-based',
    'mu_strategy': 'adaptive',
    'warm_start_init_point': 'yes',
    'hessian_approximation': 'limited-memory',
    # ...
}
```

**Expert (50+ options)**: Tuning barrier parameters, line search coefficients, linear solver pivoting tolerances, restoration phase behavior...

### 1.3 Troubleshooting Options

When IPOPT fails to converge, experts adjust:

| Problem | Options to Adjust |
|---------|------------------|
| Local infeasibility | `expect_infeasible_problem`, `required_infeasibility_reduction` |
| Slow convergence | `mu_strategy`, `mu_init`, `acceptable_iter` |
| Numerical issues | `nlp_scaling_method`, `check_derivatives_for_naninf` |
| Restoration failures | `max_resto_iter`, `start_with_resto` |
| Poor Hessian | `hessian_approximation`, `limited_memory_max_history` |

---

## 2. SciPy minimize

**Source**: [SciPy minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)

### 2.1 Method-Specific Options

| Method | Key Options | Constraints | Gradients |
|--------|-------------|-------------|-----------|
| **SLSQP** | `ftol`, `maxiter`, `disp` | Yes (eq/ineq) | Optional |
| **L-BFGS-B** | `maxcor`, `ftol`, `gtol`, `maxiter` | Bounds only | Yes |
| **trust-constr** | `gtol`, `xtol`, `barrier_tol`, `verbose` | Yes (all types) | Optional |
| **COBYLA** | `rhobeg`, `tol`, `maxiter` | Ineq only | No |
| **Nelder-Mead** | `xatol`, `fatol`, `maxiter` | No | No |
| **BFGS** | `gtol`, `norm`, `maxiter` | No | Optional |
| **Newton-CG** | `xtol`, `maxiter` | No | Yes |

### 2.2 Complexity Comparison

```
SLSQP:        ~5 options   (simple wrapper around Fortran)
L-BFGS-B:     ~8 options   (limited-memory quasi-Newton)
trust-constr: ~15 options  (most sophisticated, large-scale)
```

### 2.3 Hidden Complexity

Most users write:
```python
result = minimize(fun, x0, method='SLSQP', bounds=bounds)
```

But effective use requires understanding:
- When to switch from SLSQP to trust-constr
- How `ftol` vs `gtol` affect convergence
- When to provide analytical gradients vs finite-difference
- How to handle ill-conditioned problems

---

## 3. NLopt

**Source**: [NLopt Reference](https://nlopt.readthedocs.io/en/latest/NLopt_Reference/)

### 3.1 Termination Criteria

| Option | Description | Default |
|--------|-------------|---------|
| `ftol_rel` | Relative function tolerance | 1e-4 |
| `ftol_abs` | Absolute function tolerance | 0 |
| `xtol_rel` | Relative parameter tolerance | 1e-4 |
| `xtol_abs` | Absolute parameter tolerance (per-dim) | 0 |
| `stopval` | Stop when objective reaches value | - |
| `maxeval` | Maximum evaluations | - |
| `maxtime` | Maximum time (seconds) | - |

### 3.2 Algorithm-Specific Options

| Algorithm | Special Options |
|-----------|----------------|
| L-BFGS | `vector_storage` (memory for gradients) |
| BOBYQA | Initial step size from bounds |
| MLSL | `local_optimizer` (subsidiary optimizer) |
| AUGLAG | `local_optimizer`, constraint handling |

### 3.3 The 40+ Algorithm Problem

NLopt provides 40+ algorithms. Users must choose:
- Global vs Local
- Gradient-based vs Derivative-free
- Constrained vs Unconstrained

This selection itself is expert knowledge.

---

## 4. Optuna (Bayesian Optimization)

**Source**: [Optuna Documentation](https://optuna.readthedocs.io/)

### 4.1 Sampler Options

**TPESampler** (default):
| Option | Description | Default |
|--------|-------------|---------|
| `n_startup_trials` | Random trials before TPE | 10 |
| `multivariate` | Joint distribution modeling | False |
| `group` | Search space decomposition | False |
| `consider_prior` | Use Gaussian prior | True |
| `prior_weight` | Prior strength | 1.0 |

**CmaEsSampler**:
| Option | Description | Default |
|--------|-------------|---------|
| `sigma0` | Initial step size | - |
| `n_startup_trials` | Random before CMA | 1 |
| `restart_strategy` | None, 'ipop', 'bipop' | None |
| `popsize` | Population size | Auto |
| `use_separable_cma` | Diagonal covariance | False |

### 4.2 Pruner Options

| Pruner | Key Options |
|--------|-------------|
| MedianPruner | `n_startup_trials`, `n_warmup_steps`, `interval_steps` |
| HyperbandPruner | `min_resource`, `max_resource`, `reduction_factor` |
| ThresholdPruner | `lower`, `upper` |

### 4.3 Best Combinations (Research-Backed)

| Sampler | Best Pruner |
|---------|-------------|
| RandomSampler | MedianPruner |
| TPESampler | HyperbandPruner |

---

## 5. CMA-ES (pycma)

**Source**: [pycma Documentation](https://github.com/CMA-ES/pycma)

### 5.1 Core Parameters

| Parameter | Description | Best Practice |
|-----------|-------------|---------------|
| `sigma0` | Initial step size | ~1/4 of search domain width |
| `popsize` | Population size | 5 to 5n (n = dimensions) |
| `CMA_diagonal` | Iterations with diagonal covariance | 0 or ~100 |

### 5.2 Restart Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| None | No restart | Simple problems |
| IPOP | Increasing population | Multi-modal, need more exploration |
| BIPOP | Alternating small/large | Complex landscapes |

### 5.3 Advanced Options

```python
# Expert-level configuration
options = {
    'CMA_diagonal': 100,      # Faster for high-dim
    'CMA_active': True,       # Negative update
    'CMA_mu': None,           # Parents = popsize // 2
    'CMA_cmean': 1,           # Learning rate for mean
    'seed': 1234,             # Reproducibility
    'tolx': 1e-11,            # Parameter tolerance
    'tolfun': 1e-11,          # Function tolerance
    'verb_time': 0,           # Timing verbosity
}
```

---

## 6. pymoo (Evolutionary Algorithms)

**Source**: [pymoo Documentation](https://pymoo.org/)

### 6.1 NSGA-II Default Configuration

```python
NSGA2(
    pop_size=100,
    sampling=FloatRandomSampling(),
    selection=TournamentSelection(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)
```

### 6.2 Operator Options

**Crossover Operators**:
| Operator | Parameters | Use Case |
|----------|------------|----------|
| SBX | `prob`, `eta` | Continuous variables |
| TwoPointCrossover | - | Binary |
| UniformCrossover | `prob` | Mixed |

**Mutation Operators**:
| Operator | Parameters | Use Case |
|----------|------------|----------|
| PM (Polynomial) | `eta`, `prob` | Continuous |
| BitflipMutation | `prob` | Binary |

### 6.3 Hidden Complexity

Users must understand:
- `eta` (distribution index): Lower = more spread, Higher = closer to parents
- `prob`: Crossover probability (typically 0.9)
- Population size vs number of generations trade-off
- When to use steady-state (`n_offsprings=1`) vs generational

---

## 7. SNOPT

**Source**: [SNOPT Documentation](https://web.stanford.edu/group/SOL/guides/sndoc7.pdf)

### 7.1 Key Option Categories

| Category | Options | Examples |
|----------|---------|----------|
| Iteration Limits | Major/Minor iterations | `Major iterations limit 500` |
| Tolerances | Feasibility, optimality | `Major feasibility tolerance 1e-7` |
| Scaling | Problem conditioning | `Scale option 2` |
| QP Solver | Subproblem handling | `QPSolver Cholesky` |
| Print Control | Output verbosity | `Major print level 1` |

### 7.2 Scaling Sensitivity

> "SNOPT is quite sensitive to scaling and care must be taken to provide acceptable values. If an optimization problem is not appropriately scaled, optimization may fail or take an unnecessarily long time."

Options:
- `Scale option`: 0 (none), 1 (linear only), 2 (all)
- `Scale tolerance`: Controls iterative scaling (default 0.9)

---

## 8. The Knowledge Gap

### 8.1 What Users Know vs What They Need

| User Level | Knows | Needs to Know |
|------------|-------|---------------|
| **Beginner** | `tol`, `maxiter` | Algorithm selection, when defaults fail |
| **Intermediate** | Algorithm families, basic tuning | Problem-specific adjustments |
| **Expert** | Deep algorithm internals | New problem types |
| **Developer** | Everything | Nothing (they wrote it) |

### 8.2 The Expert Knowledge Problem

**Experts know** (but rarely document):
- IPOPT's `mu_init=1e-5` helps when starting near-feasible
- SNOPT needs good scaling or it fails silently
- CMA-ES `sigma0` should be ~1/4 of search width
- L-BFGS-B's `maxcor` matters for ill-conditioned problems
- Trust-constr is better than SLSQP for large-scale
- NSGA-II's `eta` affects exploration/exploitation

**This knowledge is:**
- Scattered across papers, forums, personal experience
- Problem-dependent
- Often discovered through trial and error
- Not encoded in documentation

---

## 9. Problem Characteristics → Option Selection

### 9.1 Algorithm Selection Matrix

| Problem Characteristic | Recommended Algorithm | Key Options |
|-----------------------|----------------------|-------------|
| Smooth, unconstrained, small | L-BFGS-B | `gtol=1e-8` |
| Smooth, constrained, small | SLSQP | `ftol=1e-9` |
| Smooth, constrained, large | IPOPT, trust-constr | scaling, linear solver |
| Noisy gradients | COBYLA, Nelder-Mead | `rhobeg` |
| Black-box, expensive | Optuna (TPE) | `n_startup_trials` |
| Multi-modal, moderate budget | CMA-ES (BIPOP) | `sigma0`, `restarts` |
| Multi-objective | NSGA-II/III | `pop_size`, operators |
| Integer/Mixed | Optuna, GA | sampler type |

### 9.2 Convergence Problem → Option Adjustment

| Symptom | Likely Cause | Option Adjustment |
|---------|--------------|-------------------|
| Stagnation | Local minimum | Restart strategy, multi-start |
| Infeasibility | Bad scaling | `nlp_scaling_method`, `scale_option` |
| Slow convergence | Conservative step | `mu_strategy`, `alpha_red_factor` |
| Oscillation | Poor Hessian | `hessian_approximation`, `limited_memory` |
| Numerical errors | Ill-conditioning | `bound_push`, `check_derivatives` |

---

## 10. Paola's Role: First-Principles Analysis

### 10.1 The Core Insight

**Option configuration is expert knowledge that should be automated.**

The pattern is identical to initialization:
- Massive complexity exists (250+ IPOPT options)
- Most users only need 3-5 options
- The right options depend on problem characteristics
- Expert knowledge is scattered and implicit
- Trial-and-error wastes computational budget

### 10.2 What Users Actually Want

Users want to say:
> "Optimize my wing design problem"

NOT:
> "Run IPOPT with mu_strategy=adaptive, nlp_scaling_method=gradient-based, linear_solver=ma57, limited_memory_max_history=10, ..."

### 10.3 The Paola Principle Extended

> **"Configuration is agent intelligence, not user burden."**

| Aspect | Traditional | Paola |
|--------|-------------|-------|
| Algorithm selection | User chooses | Paola recommends based on problem |
| Option defaults | Generic | Problem-adapted |
| Convergence issues | User debugs | Paola diagnoses and adjusts |
| Expert knowledge | In user's head | In Paola's reasoning |

### 10.4 Paola's Configuration Intelligence

```
┌─────────────────────────────────────────────────────────────────┐
│           PAOLA'S CONFIGURATION INTELLIGENCE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Level 1: Problem Analysis                                      │
│    → Dimensions? Constraints? Gradients available?              │
│    → Smooth or noisy? Convex or multi-modal?                    │
│    → Budget (evaluations allowed)?                              │
│                                                                 │
│  Level 2: Algorithm Selection                                   │
│    → Match problem characteristics to algorithm families        │
│    → Consider trade-offs (speed vs robustness)                  │
│    → Apply knowledge from similar past problems                 │
│                                                                 │
│  Level 3: Option Configuration                                  │
│    → Set problem-appropriate defaults                           │
│    → Apply scaling based on variable ranges                     │
│    → Configure termination criteria for budget                  │
│                                                                 │
│  Level 4: Runtime Adaptation                                    │
│    → Monitor convergence behavior                               │
│    → Detect stagnation, infeasibility, numerical issues         │
│    → Adjust options or restart with different config            │
│                                                                 │
│  Level 5: Learning                                              │
│    → Record what worked for this problem type                   │
│    → Build knowledge base for future problems                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 10.5 User-Facing Simplicity

**What users see** (high-level intents):

```python
run_optimization(
    problem_id="wing_design",
    # High-level intents, not low-level knobs:
    priority="robustness"  # or "speed", "accuracy"
)
```

**What Paola does internally**:

```
Paola's reasoning:

1. Problem analysis:
   - 100 variables (large-scale)
   - Nonlinear constraints
   - Gradients available (adjoint)
   - Previous runs show ill-conditioning

2. Algorithm selection:
   - Large-scale + constraints + gradients → IPOPT
   - Ill-conditioning history → careful scaling

3. Configuration:
   {
     'linear_solver': 'ma57',        # Good for large sparse
     'nlp_scaling_method': 'gradient-based',
     'mu_strategy': 'adaptive',
     'tol': 1e-6,
     'max_iter': 500,
     'print_level': 3
   }

4. Reasoning logged:
   "Selected IPOPT for large-scale constrained NLP.
    Using MA57 solver and gradient-based scaling
    based on previous ill-conditioning issues."
```

### 10.6 Handling Convergence Failures

When optimization fails, Paola reasons:

```
Optimization failed: "Restoration phase failed"

Paola's diagnosis:
1. Check constraint violation at failure point
   → Constraint 'lift >= 1000' violated: lift = 980

2. Possible causes:
   a) Problem is infeasible
   b) Poor scaling makes feasible region unreachable
   c) Starting point too far from feasible region

3. Remediation attempts:
   a) Retry with expect_infeasible_problem=yes
   b) Retry with relaxed nlp_scaling
   c) Retry with different starting point

4. If still fails:
   "The lift constraint may be too tight. Consider:
    - Relaxing to lift >= 950
    - Or using a different baseline geometry"
```

---

## 11. Design Implications for Paola

### 11.1 Tool Interface Changes

**Current approach** (exposes complexity):
```python
config_ipopt(
    config_id="my_config",
    tol=1e-6,
    max_iter=500,
    mu_strategy="adaptive",
    linear_solver="ma57",
    nlp_scaling_method="gradient-based",
    # ... 20 more options
)
```

**Paola approach** (exposes intent):
```python
# Option A: Let Paola handle everything
run_optimization(
    problem_id="wing_design",
    optimizer="auto"  # Paola selects and configures
)

# Option B: Specify algorithm family, Paola configures
run_optimization(
    problem_id="wing_design",
    optimizer="gradient-based",
    priority="robustness"
)

# Option C: Expert override (escape hatch)
run_optimization(
    problem_id="wing_design",
    optimizer="ipopt",
    options='{"mu_init": 1e-4}'  # Expert knows what they want
)
```

### 11.2 Configuration Layers

```
┌────────────────────────────────────────────┐
│  Layer 1: User Intent                      │
│    "I want a robust solution"              │
├────────────────────────────────────────────┤
│  Layer 2: Paola's Problem Analysis         │
│    Large-scale, constrained, gradients     │
├────────────────────────────────────────────┤
│  Layer 3: Paola's Algorithm Selection      │
│    IPOPT (best for this class)             │
├────────────────────────────────────────────┤
│  Layer 4: Paola's Configuration            │
│    Problem-appropriate options             │
├────────────────────────────────────────────┤
│  Layer 5: Expert Override (optional)       │
│    Specific options if user knows better   │
└────────────────────────────────────────────┘
```

### 11.3 Knowledge Accumulation

Paola learns from experience:

```
After successful optimization:
  - Problem type: aerodynamic shape, 100 variables
  - Algorithm used: IPOPT
  - Key options: mu_strategy=adaptive, linear_solver=ma57
  - Iterations: 127
  - Result: converged to tol=1e-6

Stored insight:
  "Aerodynamic shape problems with ~100 variables
   converge well with IPOPT + adaptive mu + MA57.
   Expect ~100-150 iterations."
```

---

## 12. Summary: The Paola Principle for Configuration

### 12.1 Core Principles

1. **Configuration complexity is hidden from users**
   - 250 IPOPT options → user sees "optimizer=robust"

2. **Problem characteristics drive configuration**
   - Paola analyzes problem → selects algorithm → configures options

3. **Expert knowledge is encoded in agent**
   - "SNOPT needs good scaling" → Paola checks and applies

4. **Runtime adaptation handles failures**
   - Convergence issue → Paola diagnoses → adjusts → retries

5. **Learning improves over time**
   - Successful configs remembered for similar problems

### 12.2 What This Enables

| Capability | Without Paola | With Paola |
|------------|---------------|------------|
| Algorithm selection | User must learn all options | Automatic |
| Option tuning | Trial and error | Problem-appropriate |
| Failure handling | User debugs | Automated diagnosis |
| Scaling issues | Often missed | Detected and fixed |
| Expert knowledge | Requires experience | Built-in |

### 12.3 The Ultimate Vision

User says:
> "Minimize drag on my wing while maintaining lift ≥ 1000 N"

Paola:
1. Analyzes problem (100 FFD variables, nonlinear constraints)
2. Selects IPOPT (best for this class)
3. Configures with problem-appropriate options
4. Detects convergence stagnation at iter 50
5. Adjusts mu_strategy and restarts
6. Converges successfully
7. Records insight for future wing problems

**The user never sees a single optimizer option.**

---

## Sources

### Optimizer Documentation
- [IPOPT Options](https://coin-or.github.io/Ipopt/OPTIONS.html)
- [IPOPT Output](https://coin-or.github.io/Ipopt/OUTPUT.html)
- [SciPy minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
- [NLopt Reference](https://nlopt.readthedocs.io/en/latest/NLopt_Reference/)
- [NLopt Algorithms](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/)
- [Optuna Samplers](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
- [pymoo NSGA-II](https://pymoo.org/algorithms/moo/nsga2.html)
- [SNOPT Options](https://ccom.ucsd.edu/~optimizers/docs/snopt/options.html)
- [pycma API](https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.html)
- [pyOptSparse Guide](https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/guide.html)

### Best Practices
- [IPOPT Troubleshooting (Julia Discourse)](https://discourse.julialang.org/t/local-infeasibility-ipopt/70026)
- [Hyperparameter Optimization Survey](https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1484)
- [AWS SageMaker HPO Best Practices](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-considerations.html)
