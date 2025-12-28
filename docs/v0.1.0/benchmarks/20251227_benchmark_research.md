# Benchmark Problems Research for Paola Testing

**Date**: December 27, 2025
**Purpose**: Find benchmark optimization problems with natural language descriptions and known solutions to test Paola's ability to formulate and solve optimization problems from user intent.

## Research Objective

Find benchmark examples that:
1. Have **natural language problem descriptions** (not pre-formulated mathematical problems)
2. Have **known optimal solutions** for validation
3. Are similar to Paola's target domains: **portfolio optimization**, **engineering design**, **ML hyperparameter tuning**
4. Allow testing Paola's intelligence in problem formulation, constraint handling, and optimizer selection

---

## Selected Benchmarks

### 1. Portfolio Optimization: OR-Library Benchmark ⭐

**Source**: [OR-Library Portfolio Data](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/portinfo.html)

**What It Offers**:
- 5 benchmark datasets with 31-225 assets
- Historical return and covariance data from real markets
- Known optimal solutions for mean-variance optimization
- Realistic constraints: cardinality (max # assets), diversification (min allocation per asset)

**Datasets**:
| Dataset | # Assets | Market | Constraints |
|---------|----------|---------|-------------|
| Port1 | 31 | Hang Seng (Hong Kong) | Cardinality, bounds |
| Port2 | 85 | Hang Seng | Cardinality, bounds |
| Port3 | 89 | DAX 100 (Germany) | Cardinality, bounds |
| Port4 | 98 | FTSE 100 (UK) | Cardinality, bounds |
| Port5 | 225 | Nikkei 225 (Japan) | Cardinality, bounds |

**Example Natural Language Problem** (Port2 - 85 assets):
> "Given historical returns for 85 assets from the Hang Seng index, construct a portfolio that maximizes risk-adjusted return (Sharpe ratio). You must invest in at most 20 different assets, with each selected asset receiving at least 1% of total capital. The goal is to balance return and risk while maintaining diversification."

**Why Good for Paola**:
- ✅ Tests LLM's ability to formulate Markowitz mean-variance problem from description
- ✅ Realistic financial constraints (not just mathematical bounds)
- ✅ Known optimal solutions to validate against
- ✅ Directly comparable to current portfolio example
- ✅ Multiple difficulty levels (31 to 225 assets)

**Known Solutions** (from Chang et al. 2000):
- Port1: Various solutions documented in literature
- Port2-5: Optimal Sharpe ratios and allocations available

**References**:
- Chang, T.-J., Meade, N., Beasley, J.E. and Sharaiha, Y.M. (2000). "Heuristics for cardinality constrained portfolio optimisation", Computers & Operations Research 27, 1271-1302.

---

### 2. Structural Engineering: 10-Bar Truss Benchmark ⭐⭐

**Source**: [10-Bar Planar Truss](https://xloptimizer.com/projects/mechanics/10-bar-planar-truss), MATLAB Benchmark Suite

**Problem Description**:
Standard 10-bar planar truss cantilever structure widely used in structural optimization literature.

**Geometry**:
- 6 nodes (joints)
- 10 members (bars)
- Planar (2D) cantilever configuration
- Fixed support on left side
- Loads applied on right side

**Design Variables**:
- Cross-sectional area of each of 10 members: A₁, A₂, ..., A₁₀
- Bounds: 0.1 to 35 in²

**Constraints**:
- **Stress limit**: |σᵢ| ≤ 25 ksi for all members i
- **Displacement limit**: |dⱼ| ≤ 2 inches for all nodes j (x and y directions)

**Material Properties**:
- Young's modulus: E = 10,000 ksi (68.9 GPa)
- Density: ρ = 0.1 lb/in³

**Load Cases**:
- **LC1**: P₁ = 100 kips ↓, P₂ = 0 kips → **Optimal: 5,060.855 lbs**
- **LC2**: P₁ = 150 kips ↓, P₂ = 50 kips ↓ → **Optimal: 4,676.932 lbs**

**Example Natural Language Problem** (Load Case 1):
> "Design a 10-bar truss structure to minimize weight while supporting a 100 kip downward load at the top-right node. The structure must satisfy stress limits (no member can exceed 25 ksi in tension or compression) and displacement limits (all nodes must stay within 2 inches of their original position). Each member's cross-sectional area can range from 0.1 to 35 square inches. The material is steel with Young's modulus of 10,000 ksi and density of 0.1 lb/in³."

**Why Good for Paola**:
- ✅ Well-defined engineering problem with realistic constraints
- ✅ Known optimal solution for validation
- ✅ Tests constraint handling (inequality constraints on stress and displacement)
- ✅ Can extend to harder variants (25-bar, 72-bar, 120-bar)
- ✅ Requires FEA simulation (computational evaluator)

**Python Implementation**: `slientruss3d` package
- Installation: `pip install slientruss3d`
- Features: Direct stiffness FEA, constraint checking, fast (<0.05s for 942-bar)
- Clean API matching Paola's evaluator pattern

**References**:
- Camp, C.V. & Farshchin, M. (2014). "Design of space trusses using modified teaching-learning based optimization", Engineering Structures 62-63, 87-97.
- [NASTRAN CoFE 10-Bar Example](https://vtpasquale.github.io/NASTRAN_CoFE/2._Examples/b._Optimization/1._10-Bar_Truss_Sizing/)

---

### 3. ML Hyperparameter Tuning: OpenML-CC18 ⭐

**Source**: [OpenML-CC18 Benchmark Suite](https://www.openml.org/s/99), [AutoML Benchmark](https://openml.github.io/automlbenchmark/)

**What It Offers**:
- 72 curated classification datasets
- Mid-sized: 500 to 100,000 samples
- Up to 5,000 features
- Class imbalance ratio > 0.05
- Known best hyperparameters from SOTA AutoML systems

**Suggested Starting Dataset**: **airlines**
- Classification task
- Well-studied in AutoML literature
- **AutoGluon SOTA**: 69.4% accuracy
- Suitable for Random Forest hyperparameter tuning

**Example Natural Language Problem** (airlines dataset):
> "Tune a Random Forest classifier on the 'airlines' dataset to maximize classification accuracy. The hyperparameters to optimize are: n_estimators (number of trees, range 10-500), max_depth (maximum tree depth, range 3-20), and min_samples_split (minimum samples to split, range 2-50). The current baseline with default parameters achieves 65.8% accuracy. Your goal is to find hyperparameters that beat the state-of-the-art AutoGluon result of 69.4%."

**Hyperparameters to Optimize** (Random Forest example):
- `n_estimators`: [10, 500]
- `max_depth`: [3, 20]
- `min_samples_split`: [2, 50]
- `max_features`: ['sqrt', 'log2', None]
- `min_samples_leaf`: [1, 10]

**Why Good for Paola**:
- ✅ Practical ML problem familiar to data scientists
- ✅ Can use Bayesian optimization (TPE, GP)
- ✅ Known SOTA results to beat (AutoGluon, AutoSklearn)
- ✅ Tests Paola's ability to set up sequential optimization
- ✅ Evaluator = cross-validation function (relatively fast)

**Known SOTA Results**:
- **AutoGluon 1.0**: Best overall AutoML system on OpenML-CC18
  - 63% first place finishes across 72 datasets
  - 95%+ win-rate vs traditional models
  - 82-94% win-rate vs other AutoML systems
- **AutoSklearn**: Strong baseline, optimized max_features
- **Tuned Random Forest**: Significant improvement over defaults

**Python Access**:
```python
import openml
dataset = openml.datasets.get_dataset('airlines')
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
)
```

**References**:
- Bischl, B. et al. (2021). "OpenML Benchmarking Suites", NeurIPS Datasets and Benchmarks Track.
- [AutoML Benchmark Results](https://openml.github.io/automlbenchmark/visualization)

---

## Additional Strong Candidates

### 4. Welded Beam Design (Engineering)

**Problem**: Minimize manufacturing cost of a welded beam structure

**Design Variables** (4):
- Weld thickness (h)
- Weld length (l)
- Beam thickness (t)
- Beam width (b)

**Constraints** (7):
- Shear stress limit
- Bending stress limit
- Buckling load limit
- End deflection limit
- Side constraint

**Known Optimal**: Cost ≈ $1.72

**Natural Language**:
> "Design a welded beam to minimize manufacturing cost while meeting structural requirements. The beam must withstand shear and bending stresses, avoid buckling, and limit deflection."

---

### 5. Pressure Vessel Design (Engineering)

**Problem**: Minimize manufacturing cost of a cylindrical pressure vessel

**Design Variables** (4):
- Shell thickness (Ts)
- Head thickness (Th)
- Inner radius (R)
- Cylinder length (L)

**Constraints**:
- Minimum thickness requirements
- Pressure safety limit
- Minimum volume requirement

**Known Optimal**: Available in literature

---

## Implementation Recommendation

**Priority Order** (based on user selection: "Quick implementation, start with Engineering"):

### Phase 1: 10-Bar Truss ✅ (SELECTED)

**Why first**:
- ✅ Quick setup with `slientruss3d` package
- ✅ No data download needed
- ✅ Tests constraint handling well
- ✅ Different domain from existing portfolio example
- ✅ Clear known solution for validation

**Implementation**:
1. Install `slientruss3d>=2.0.3`
2. Create `examples/evaluators/truss_10bar_evaluator.py`
3. Implement 3 functions: `evaluate()`, `constraint_stress()`, `constraint_displacement()`
4. Test with natural language prompt in Paola CLI

**Success Criteria**:
- [ ] Paola correctly identifies 10 design variables (areas)
- [ ] Paola sets up 2 inequality constraints (stress, displacement)
- [ ] Paola chooses constrained optimizer (SLSQP or trust-constr)
- [ ] Solution within 5% of 5,060.855 lbs
- [ ] All constraints satisfied

---

### Phase 2: OR-Library Portfolio (Port5)

**After 10-bar truss validation**:
1. Download Port5 data (225 assets) from OR-Library
2. Parse covariance matrix and expected returns
3. Similar to existing portfolio example but with cardinality constraints
4. Compare against known optimal Sharpe ratio

---

### Phase 3: OpenML Hyperparameter Tuning (airlines)

**Final benchmark**:
1. Use `openml` Python package to load airlines dataset
2. Set up Random Forest with cross-validation evaluator
3. Optimize hyperparameters with Bayesian optimization
4. Compare against AutoGluon's 69.4% accuracy

---

## Research Sources

### Portfolio Optimization
- [OR-Library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/portinfo.html)
- [Benchmark Portfolio Optimization Datasets](https://people.cs.nott.ac.uk/pszrq/POdata.htm)
- Chang et al. (2000), Computers & Operations Research

### Engineering Design
- [10-Bar Truss - xloptimizer](https://xloptimizer.com/projects/mechanics/10-bar-planar-truss)
- [NASTRAN CoFE Examples](https://vtpasquale.github.io/NASTRAN_CoFE/)
- [MATLAB Benchmark Truss Problems](https://www.mathworks.com/matlabcentral/fileexchange/76228)
- [slientruss3d Package](https://github.com/leo27945875/Python_Stable_3D_Truss_Analysis)
- Camp & Farshchin (2014), Engineering Structures

### ML Hyperparameter Tuning
- [OpenML-CC18 Benchmark](https://www.openml.org/s/99)
- [AutoML Benchmark](https://openml.github.io/automlbenchmark/)
- [AutoML Benchmark Visualization](https://openml.github.io/automlbenchmark/visualization)
- Bischl et al. (2021), NeurIPS Datasets and Benchmarks

### General Optimization Benchmarks
- [BBOB-Constrained Suite](https://hub.optuna.org/benchmarks/bbob_constrained/) - 54 constrained functions
- [PyCUTEst](https://github.com/jfowkes/pycutest) - 1000+ problems
- [SciPy Global Optimization](http://infinity77.net/go_2021/) - 184-235 multivariate problems
- [Hock-Schittkowski Problems](https://apmonitor.com/wiki/index.php/Apps/HockSchittkowski) - 115 classic NLP problems

---

## Validation Metrics

For each benchmark, we will measure:

1. **Formulation Accuracy**
   - Did Paola correctly identify design variables?
   - Did Paola correctly set up constraints?
   - Did Paola choose appropriate objective sense (min/max)?

2. **Solution Quality**
   - Optimality gap: `|found - optimal| / optimal × 100%`
   - Constraint satisfaction: All constraints met?
   - Comparison to known solution

3. **Reasoning Quality**
   - Did Paola choose appropriate optimizer type?
   - Did Paola use multi-start for non-convex problems?
   - Did Paola handle constraints properly?

4. **Process Quality**
   - Did Paola read and understand the evaluator code?
   - Did Paola register evaluators correctly?
   - Did Paola create problem formulation correctly?

---

## Conclusion

These three benchmarks (10-bar truss, OR-Library portfolio, OpenML hyperparameter) provide comprehensive coverage of Paola's target domains with:
- ✅ Natural language problem descriptions
- ✅ Known optimal solutions for validation
- ✅ Realistic constraints and objectives
- ✅ Quick implementation (using existing packages)
- ✅ Progressive difficulty (engineering → finance → ML)

The benchmarks will demonstrate Paola's ability to:
1. Understand user intent from natural language
2. Formulate optimization problems correctly
3. Handle constraints appropriately
4. Choose suitable optimizers
5. Find near-optimal solutions

This moves beyond classical test functions (Rosenbrock, Ackley) to real-world problem formulation from user descriptions.
