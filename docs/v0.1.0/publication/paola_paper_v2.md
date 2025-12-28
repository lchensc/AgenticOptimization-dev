# PAOLA: Platform for Agentic Optimization with Learning and Analysis

**A Paradigm-Introducing Technical Report**

**Version**: 2.0
**Date**: 2025-12-26
**Status**: Draft

---

## Abstract

Optimization problems pervade science and engineering, yet solving them effectively requires specialized expertise that most domain practitioners lack. We present **PAOLA**, the first AI agent that uses optimization algorithms as an expert would. Unlike existing tools that require users to select and configure optimizers, PAOLA formulates optimization problems from natural language descriptions, selects appropriate algorithms based on problem characteristics, and can be taught domain-specific knowledge through a Skills system. We demonstrate PAOLA's capabilities through three case studies: portfolio optimization (showing problem formulation), structural design (showing expert algorithm selection), and ML hyperparameter tuning (showing teachability). Our core philosophy—"Bring your problem, not your PhD in optimization"—positions PAOLA as accessible optimization expertise for domain experts who are not optimization experts.

**Keywords**: Agentic optimization, LLM agents, optimization expertise, Skills-based learning

---

## 1. Introduction

### 1.1 The Optimization Expertise Barrier

Every day, engineers, scientists, and researchers face optimization problems:
- A mechanical engineer minimizing material cost while meeting structural constraints
- A portfolio manager maximizing risk-adjusted returns
- A data scientist tuning hyperparameters for model performance

These are optimization problems. Powerful algorithms exist to solve them: SLSQP for smooth constrained problems, CMA-ES for black-box landscapes, TPE for hyperparameter spaces. Yet most practitioners cannot effectively use these tools because they lack optimization expertise.

**The traditional workflow** requires the user to:
1. Translate their domain problem into mathematical formulation
2. Select an appropriate algorithm from hundreds of options
3. Configure algorithm parameters (IPOPT alone has 250+ options)
4. Interpret results and adapt strategy when optimization fails

This expertise barrier prevents domain experts from leveraging the full power of optimization methods. They either use suboptimal default configurations or spend significant time learning optimization theory.

### 1.2 The PAOLA Principle

We propose a paradigm shift: **optimization expertise should be a service, not a prerequisite**.

**Traditional approach:**
```
Human → [has expertise] → selects algorithm → configures → runs → interprets
         ^^^^^^^^^^^^
         This is the barrier
```

**PAOLA approach:**
```
Human → [describes problem] → PAOLA agent → [reasons like expert] → solves
                                             ^^^^^^^^^^^^^^^^
                                             This is the innovation
```

We call this the **PAOLA Principle**: *"Bring your problem, not your PhD in optimization."*

### 1.3 What PAOLA Is and Is Not

**PAOLA IS:**
- An AI that formulates optimization problems from natural language
- An AI that selects and uses optimizers as an expert would
- An AI that can be taught domain-specific knowledge via Skills

**PAOLA IS NOT:**
- A new optimization algorithm competing on benchmarks
- A replacement for domain expertise (users still understand their problems)
- Limited to specific domains (extensible via Skills)

### 1.4 Contributions

This paper introduces the **agentic optimization** paradigm through:

1. **Three Pillars of PAOLA**: Problem formulation expert, optimizer polyglot, teachable via Skills

2. **Case studies** demonstrating each pillar in action on accessible problems

3. **Honest discussion** of when PAOLA is appropriate and its limitations

4. **Open-source release** for community use and extension

---

## 2. The PAOLA Paradigm

PAOLA is built on three pillars that together enable optimization expertise as a service.

### 2.1 Pillar 1: Problem Formulation Expert

Domain experts describe problems in their natural language:

> "I want to minimize material cost for a pressure vessel, but it needs to hold 1200 psi and fit in a 2m × 1m space"

PAOLA translates this to a mathematical optimization problem:
- **Objective**: Minimize material volume (proxy for cost)
- **Constraints**: Pressure capacity ≥ 1200 psi, dimensions ≤ 2m × 1m
- **Variables**: Wall thickness, radius, length

The user did not need to specify:
- Mathematical formulation (min f(x) s.t. g(x) ≤ 0)
- Variable bounds or scaling
- Constraint handling approach

### 2.2 Pillar 2: Optimizer Polyglot

PAOLA reasons about which optimizer to use based on problem characteristics:

| Problem Characteristic | PAOLA's Selection |
|----------------------|-------------------|
| Smooth, constrained, small-scale | scipy:SLSQP |
| Large-scale with many constraints | IPOPT |
| Black-box, multimodal | CMA-ES |
| Mixed continuous/categorical | optuna:TPE |

When the first optimizer gets stuck (e.g., converges to local minimum), PAOLA reasons about alternatives:

```
[OBSERVE] SLSQP converged at f=12.3 after 50 iterations
[REASON]  Rapid initial progress, then stalled - likely local minimum
[DECIDE]  Problem appears multimodal, switching to global method
[SELECT]  CMA-ES for global exploration, then warm-start SLSQP
```

This is expert-like reasoning, externalized and interpretable.

### 2.3 Pillar 3: Teachable via Skills

Domain experts can teach PAOLA specialized knowledge through **Skills**—structured documents that encode:
- When to apply the knowledge (triggers)
- Domain-specific parameter ranges
- Best practices and pitfalls

Example Skill for ML hyperparameter tuning:
```yaml
name: ml_hyperparameter_tuning
when_to_use:
  - "hyperparameter optimization"
  - "model tuning"
domain_knowledge:
  learning_rate: [1e-5, 1e-1]  # log scale
  batch_size: [16, 256]        # powers of 2
  optimizer_recommendation: optuna:TPE
```

When a user asks to tune their ML model, PAOLA loads this Skill and applies the expert knowledge. The domain expert's knowledge becomes PAOLA's knowledge.

### 2.4 Architecture Overview

PAOLA's architecture enables these three pillars through:

1. **Foundry**: Single source of truth for problems, evaluators, and optimization history. Enables cross-session learning.

2. **Skills System**: Progressive disclosure of expert knowledge. PAOLA loads detailed knowledge only when needed.

3. **LLM Agent**: Uses trained knowledge of optimization theory plus context-specific Skills to reason about strategy.

4. **Optimizer Backends**: Unified interface to multiple optimization libraries (SciPy, IPOPT, Optuna, CMA-ES).

The graph-based strategy representation (nodes = optimizer runs, edges = relationships) serves as an implementation detail that makes optimization strategies observable and learnable—but it is not the core innovation.

---

## 3. Case Studies

We demonstrate PAOLA's three pillars through accessible case studies that users can reproduce.

### 3.1 Case Study 1: Portfolio Optimization (Formulation Focus)

**User's request:**
> "I want to maximize returns on my investment portfolio, but I can't lose more than 10% in any month, and I need at least 20% in bonds for stability."

**What PAOLA does:**
1. Identifies optimization problem type: maximize risk-adjusted return
2. Translates constraints to mathematical form
3. Formulates as: maximize Sharpe ratio subject to bond allocation ≥ 20%
4. Selects optimizer: L-BFGS-B for bound-constrained

**What user DIDN'T specify:**
- Sharpe ratio formulation
- Covariance matrix computation
- Optimizer selection
- Tolerance parameters

**Result:** PAOLA finds allocation with Sharpe ratio 0.546, bond allocation 47%.

**Key insight:** The expertise barrier is removed. A financial analyst without optimization training can now solve portfolio optimization.

### 3.2 Case Study 2: Cantilever Beam Design (Expert Selection Focus)

**User's request:**
> "Design a cantilever beam that minimizes weight but can support 1000N without exceeding stress limits."

**What PAOLA reasons:**
```
[ANALYZE] Problem has 2 continuous variables
[ANALYZE] Problem has inequality constraints (stress, deflection)
[ANALYZE] Objective is smooth (mass = ρ × width × height × length)
[DECIDE]  Select gradient-based constrained optimizer
[SELECT]  scipy:SLSQP - efficient SQP method for small constrained problems
[CONFIG]  ftol=1e-9 (standard engineering tolerance)
```

**Result:** PAOLA finds design with mass 15.6 kg, stress at 3.6% of yield, deflection at 9.1% of limit.

**Key insight:** PAOLA selects optimizers as an expert would. If SLSQP failed, it would switch to global methods.

### 3.3 Case Study 3: ML Hyperparameter Tuning (Teachability Focus)

**Setup:** A domain expert creates a Skill for ML tuning:
```yaml
name: ml_hyperparameter_tuning
domain_knowledge:
  learning_rate: [1e-5, 1e-1]  # log scale
  n_estimators: [50, 500]
  max_depth: [3, 15]
  optimizer: optuna:TPE
  n_trials: 100
```

**User's request:**
> "I have a Random Forest model for classification. Can you find the best hyperparameters?"

**What PAOLA does:**
1. Detects "hyperparameter" trigger → loads Skill
2. Applies learned ranges from Skill
3. Uses TPE sampler (from Skill recommendation)
4. Runs optimization with Skill-informed configuration

**Key insight:** Domain expertise is encoded once, benefits all users. The Skill author's knowledge becomes PAOLA's knowledge.

---

## 4. The PAOLA Experience

### 4.1 What Users Don't Need to Know Anymore

| Traditional Requirement | PAOLA Handles |
|------------------------|---------------|
| Algorithm selection | Automatic based on problem characteristics |
| Parameter configuration | Skill-informed defaults |
| Constraint handling | Formulation translation |
| Gradient computation | Finite differences or analytical |
| Multi-stage strategies | Agent reasoning |

### 4.2 What Users Still Control

- Problem definition (they are the domain experts)
- Objective priorities (what to optimize)
- Final decision-making (PAOLA proposes, human disposes)
- Teaching PAOLA new domains (via Skills)

### 4.3 Interaction Modes

1. **Conversational**: Natural language problem description
2. **Programmatic**: Python API for integration
3. **Teaching**: Skill creation for domain extension

---

## 5. Discussion

### 5.1 When PAOLA Is Appropriate

PAOLA provides most value when:
- User has a domain problem but lacks optimization expertise
- Problem involves expensive function evaluations (LLM overhead amortized)
- Multiple optimization runs are expected (learning improves over time)
- Domain knowledge can be encoded in Skills

### 5.2 When Traditional Approaches Are Better

Consider alternatives when:
- Problem is well-understood benchmark (just use known best algorithm)
- Single-shot optimization with tight latency requirements
- User is an optimization expert who wants fine control
- Function evaluations are extremely cheap (LLM overhead not justified)

### 5.3 Limitations and Future Work

**Current limitations:**
- LLM inference adds overhead (justified for expensive evaluations)
- Skills require expert creation (barrier to extension)
- Graph structure learning is nascent

**Future directions:**
- Automatic Skill generation from optimization traces
- Multi-objective Pareto exploration
- Integration with simulation-based optimization workflows

---

## 6. Related Work

PAOLA differs fundamentally from existing systems:

| System | Focus | PAOLA's Differentiation |
|--------|-------|------------------------|
| **AlphaEvolve** (DeepMind) | Discovers new algorithms | PAOLA uses existing algorithms as expert |
| **OptiChat** | Explains optimization models | PAOLA formulates AND solves |
| **ChemCrow** | Chemistry-specific tools | PAOLA is domain-agnostic, teachable |
| **LLAMBO/LLINBO** | BO enhancement | PAOLA integrates all optimizer types |
| **AutoML-Agent** | ML pipeline automation | PAOLA handles any optimization |

Research has shown that pure LLM optimizers lack feedback sensitivity (EMNLP 2025). PAOLA is a **hybrid**: LLM reasoning combined with rigorous optimization backends.

---

## 7. Conclusion

PAOLA introduces the **agentic optimization** paradigm: AI that uses optimization algorithms as an expert would. Through three pillars—problem formulation expert, optimizer polyglot, and teachable via Skills—PAOLA makes optimization expertise accessible to domain experts who are not optimization experts.

The core innovation is not a new algorithm or benchmark result. It is a shift in who bears the optimization complexity burden: from user prerequisite to system intelligence.

We release PAOLA as open source, inviting the community to:
- Use PAOLA for their optimization problems
- Create Skills for their domains
- Extend PAOLA's capabilities

**The PAOLA Principle:** *Bring your problem, not your PhD in optimization.*

---

## Acknowledgments

[To be added]

---

## References

[Selected references to be added during finalization]

---

## Appendix A: Getting Started

Installation:
```bash
pip install paola
```

Quick example:
```python
from paola import optimize

result = optimize(
    "Minimize material cost for a pressure vessel "
    "that holds 1200 psi and fits in 2m × 1m space"
)
print(result.solution)
```

For detailed documentation, visit: [URL to be added]

---

## Appendix B: Reproducing Case Studies

The case studies in this paper are **real PAOLA sessions**, not simulations. To reproduce them:

### Prerequisites

1. **Install PAOLA** and its dependencies
2. **Start the LLM server** (vLLM with Qwen3-32B or equivalent)
3. **Start PAOLA CLI**: `python -m paola.cli --model vllm:qwen3-32b`

### Case Study Evaluator Files

We provide three evaluator files that domain experts would write:

| Case Study | Evaluator File | Pillar Demonstrated |
|------------|----------------|---------------------|
| Portfolio | `examples/evaluators/portfolio_evaluator.py` | Problem Formulation |
| Beam Design | `examples/evaluators/cantilever_beam_evaluator.py` | Optimizer Selection |
| ML Tuning | `examples/evaluators/ml_hyperparameter_evaluator.py` | Teachability |

### Running a Session

```bash
# 1. Start PAOLA CLI
python -m paola.cli --model vllm:qwen3-32b

# 2. Register the evaluator
> /register_eval examples/evaluators/portfolio_evaluator.py

# 3. Describe your problem in natural language
> I want to maximize my portfolio's Sharpe ratio with at least 20% in bonds

# 4. PAOLA (the LLM agent) will:
#    - Read and understand the evaluator
#    - Formulate the optimization problem
#    - Select appropriate optimizer
#    - Run optimization
#    - Report results
```

### Session Artifacts

After running a session, PAOLA saves:
- **Graphs**: `.paola_foundry/graphs/graph_XXXX.json` - Complete optimization strategy
- **Problems**: `.paola_foundry/problems/problem_XXXX.json` - Problem formulation
- **History**: `.paola_history` - Conversation transcript

These artifacts provide the evidence for the case studies in this paper.
