# PAOLA Case Study Evaluators

This directory contains evaluator files for the three PAOLA case studies. These are real evaluators that demonstrate how domain experts provide their problems to PAOLA.

## Prerequisites

1. **Start the vLLM server** (for LLM inference):
   ```bash
   cd /scratch/longchen/AgenticOptimization-dev
   sbatch scripts/submit_vllm_qwen3.sh

   # Wait for job to start, check node name
   squeue -u $USER
   tail -f logs/vllm_server_<jobid>.log
   # Look for "Node: gpuXXX" and "Uvicorn running"
   ```

2. **Set environment variable**:
   ```bash
   export VLLM_API_BASE=http://gpuXXX:8000/v1  # Replace with actual node
   ```

## Case Study 1: Portfolio Optimization

**Demonstrates**: Problem Formulation (natural language → mathematical optimization)

### Run the session:

```bash
# Start PAOLA
python -m paola.cli --model vllm:qwen3-32b

# In PAOLA CLI:
> /register_eval examples/evaluators/portfolio_evaluator.py

> I want to maximize my portfolio's risk-adjusted return (Sharpe ratio).
> I have 5 assets: US stocks, international stocks, government bonds,
> corporate bonds, and commodities. I need at least 20% allocation to
> bonds for stability. No short selling allowed.
```

### What PAOLA will do:
1. Read the evaluator file to understand the problem
2. Identify this as a constrained optimization problem
3. Formulate bounds (0-1 for each weight), constraints (bond allocation ≥ 20%)
4. Select appropriate optimizer (likely SLSQP or L-BFGS-B)
5. Run optimization and report optimal allocation

### Expected result:
- Optimal allocation with Sharpe ratio ~0.5-0.8
- Bond allocation ≥ 20% (constraint satisfied)
- Weights sum to 1

---

## Case Study 2: Cantilever Beam Design

**Demonstrates**: Optimizer Polyglot (expert-like algorithm selection)

### Run the session:

```bash
# Start PAOLA
python -m paola.cli --model vllm:qwen3-32b

# In PAOLA CLI:
> /register_eval examples/evaluators/cantilever_beam_evaluator.py

> Design a cantilever beam that minimizes weight. The beam is made of
> aluminum, 1 meter long, and must support a 1000N load at the tip.
> The tip deflection cannot exceed 10mm, and the stress must stay
> below the yield stress with a safety factor of 1.5.
```

### What PAOLA will do:
1. Recognize this as a constrained engineering problem
2. Reason about optimizer selection:
   - "This is a 2-variable problem with constraints"
   - "Smooth objective (mass), smooth constraints (stress, deflection)"
   - "SLSQP is appropriate for small-scale constrained problems"
3. Run optimization
4. If stuck, may switch to different optimizer

### Expected result:
- Optimal beam dimensions (width ~15-25mm, height ~60-80mm)
- Mass ~1-3 kg
- Stress ratio < 1.0 (within allowable)
- Deflection < 10mm

---

## Case Study 3: ML Hyperparameter Tuning

**Demonstrates**: Teachability via Skills (domain knowledge injection)

### First, ensure the ML tuning Skill exists:

Check `paola/skills/domains/ml_tuning/skill.yaml` or create one with:
- Recommended optimizer: optuna:TPE
- Typical hyperparameter ranges
- Best practices for HPO

### Run the session:

```bash
# Start PAOLA
python -m paola.cli --model vllm:qwen3-32b

# In PAOLA CLI:
> /register_eval examples/evaluators/ml_hyperparameter_evaluator.py

> I have a Random Forest model for classification. Find the best
> hyperparameters to maximize validation accuracy. The hyperparameters
> are learning_rate, n_estimators, max_depth, and min_samples_split.
```

### What PAOLA will do:
1. Recognize "hyperparameter" keyword, load ML tuning Skill (if available)
2. Apply Skill knowledge for parameter ranges
3. Select TPE sampler (recommended for HPO)
4. Run optimization with 50-100 trials
5. Report best hyperparameters and accuracy

### Expected result:
- Hyperparameters close to optimal
- Validation accuracy ~90-95%

---

## Testing Evaluators Standalone

Each evaluator can be run standalone to verify it works:

```bash
cd /scratch/longchen/AgenticOptimization-dev

# Test portfolio evaluator
python examples/evaluators/portfolio_evaluator.py

# Test beam evaluator
python examples/evaluators/cantilever_beam_evaluator.py

# Test ML evaluator
python examples/evaluators/ml_hyperparameter_evaluator.py
```

---

## What Makes This "Real"

Unlike simulated examples, these sessions involve:

1. **Real LLM reasoning**: The Qwen3-32B model generates actual reasoning about the problem
2. **Real optimizer execution**: SciPy, Optuna, or other backends actually run
3. **Real persistence**: Results saved in `.paola_foundry/` for cross-session learning
4. **Real tool calls**: The agent uses tools to read files, register evaluators, run optimization

The user provides their domain knowledge (the evaluator function), and PAOLA provides optimization expertise (algorithm selection, configuration, strategy).

---

## Recording Sessions for the Paper

To record a session for documentation:

```bash
# PAOLA automatically saves:
# - Graph structure: .paola_foundry/graphs/
# - Problems: .paola_foundry/problems/
# - Session history: .paola_history

# For detailed logging, set environment variable:
export PAOLA_LOG_LEVEL=DEBUG
```

After running a session, the graph file will contain:
- All nodes (optimizer runs)
- Edges (relationships between runs)
- Final objectives
- Total evaluations and wall time

This provides the evidence for the paper's case studies.
