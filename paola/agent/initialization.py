"""
InitializationManager: Agent intelligence for initialization decisions.

The Paola Principle: "Initialization is agent intelligence, not user input."

Different optimization algorithms require different initialization strategies:
- Gradient-based (SLSQP, L-BFGS-B, IPOPT): Single initial point (x0)
- CMA-ES: Mean vector + initial sigma (step size)
- Population-based (NSGA-II, GA): Population sampling (LHS, random)
- Bayesian (Optuna, BO): Let sampler handle (no init needed)

This complexity should be handled by Paola, not exposed to users.

Decision Tree:
1. Check for warm-start opportunities (previous successful runs)
2. Check domain hint ("shape_optimization" → zero, "general" → center)
3. Apply algorithm-specific defaults

Sources:
- SciPy minimize: x0 required, no default
- IPOPT: x0 required, auto-pushes to interior
- CMA-ES: mean + sigma, sigma ≈ 0.25 × bound width
- NSGA-II: Population via LHS or random sampling
- Optuna: Sampler generates initial points
"""

from typing import List, Optional, Dict, Any, Union, Literal
import numpy as np
import logging

from ..foundry.nlp_schema import NLPProblem

logger = logging.getLogger(__name__)


# Algorithm classification for initialization strategy
GRADIENT_BASED_ALGORITHMS = {
    # SciPy methods
    "SLSQP", "L-BFGS-B", "BFGS", "CG", "Newton-CG", "TNC", "COBYLA", "trust-constr",
    # NLopt methods
    "LD_SLSQP", "LD_LBFGS", "LD_MMA", "LD_CCSAQ", "LD_TNEWTON",
    # IPOPT
    "IPOPT", "ipopt",
    # SNOPT
    "SNOPT", "snopt",
}

POPULATION_BASED_ALGORITHMS = {
    "NSGA-II", "NSGA-III", "GA", "DE", "PSO",
    "nsga2", "nsga3", "genetic", "differential_evolution", "particle_swarm",
}

CMA_ES_ALGORITHMS = {
    "CMA-ES", "cma-es", "cmaes", "CMA",
}

BAYESIAN_ALGORITHMS = {
    "TPE", "tpe", "BO", "bayesian", "Optuna", "optuna",
    "GP", "gaussian_process",
}


class InitializationManager:
    """
    Agent intelligence for computing initial points.

    The Paola Principle: Users don't need to know that SLSQP needs x0,
    CMA-ES needs mean+sigma, or Bayesian methods need nothing.
    Paola handles this complexity automatically.

    Example:
        manager = InitializationManager()

        # For SLSQP on a wing optimization problem
        x0 = manager.compute_initial_point(
            problem=wing_problem,
            algorithm="SLSQP",
            run_history=previous_runs  # For warm-starting
        )

        # For CMA-ES
        mean, sigma = manager.compute_cmaes_params(problem)

        # For NSGA-II
        population = manager.generate_population(problem, size=50)
    """

    def __init__(self, run_storage=None):
        """
        Initialize manager.

        Args:
            run_storage: Optional run storage for warm-start queries
        """
        self.run_storage = run_storage

    def compute_initial_point(
        self,
        problem: NLPProblem,
        algorithm: str,
        run_history: Optional[List[Dict[str, Any]]] = None,
        force_strategy: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Compute initial point for an algorithm.

        Decision tree:
        1. If Bayesian algorithm → return None (sampler handles it)
        2. Check warm-start opportunities
        3. Check domain hint
        4. Apply algorithm-specific default

        Args:
            problem: NLP problem with bounds and domain_hint
            algorithm: Algorithm name (e.g., "SLSQP", "CMA-ES")
            run_history: Previous runs for warm-starting
            force_strategy: Override strategy ("zero", "center", "warm_start")

        Returns:
            Initial point as numpy array, or None for Bayesian methods
        """
        # Bayesian methods don't need initialization
        if algorithm in BAYESIAN_ALGORITHMS:
            logger.info(f"Algorithm {algorithm} is Bayesian - no initialization needed")
            return None

        # Force specific strategy if requested
        if force_strategy:
            return self._apply_strategy(problem, force_strategy, run_history)

        # 1. Check for warm-start opportunities
        if run_history:
            warm_start = self._find_warm_start(problem, run_history)
            if warm_start is not None:
                logger.info(f"Using warm-start from previous run")
                return warm_start

        # 2. Check domain hint
        if problem.domain_hint:
            return self._apply_domain_hint(problem)

        # 3. Apply algorithm-specific default
        return self._apply_algorithm_default(problem, algorithm)

    def _apply_strategy(
        self,
        problem: NLPProblem,
        strategy: str,
        run_history: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """Apply a specific initialization strategy."""
        if strategy == "zero":
            return np.zeros(problem.dimension)

        elif strategy == "center":
            return np.array(problem.get_bounds_center())

        elif strategy == "warm_start" and run_history:
            warm_start = self._find_warm_start(problem, run_history)
            if warm_start is not None:
                return warm_start
            # Fallback to center if no warm-start available
            return np.array(problem.get_bounds_center())

        elif strategy == "random":
            # Random within bounds
            bounds = np.array(problem.bounds)
            return np.random.uniform(bounds[:, 0], bounds[:, 1])

        else:
            logger.warning(f"Unknown strategy '{strategy}', using center of bounds")
            return np.array(problem.get_bounds_center())

    def _find_warm_start(
        self,
        problem: NLPProblem,
        run_history: List[Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """
        Find best warm-start point from run history.

        Looks for:
        1. Successful runs on the same problem
        2. Best objective value among matching runs
        3. Feasible solution preferred

        Args:
            problem: Current problem
            run_history: List of previous run records

        Returns:
            Best solution from history, or None if no suitable found
        """
        matching_runs = []

        for run in run_history:
            # Check if same problem (by ID or similar dimension)
            run_problem_id = run.get("problem_id")
            run_dimension = run.get("dimension")

            if run_problem_id == problem.problem_id:
                # Same problem - good match
                matching_runs.append(run)
            elif run_dimension == problem.dimension:
                # Same dimension - might be related
                # Could add more sophisticated matching here
                pass

        if not matching_runs:
            return None

        # Find best solution among matching runs
        best_run = None
        best_objective = float('inf')

        for run in matching_runs:
            result = run.get("result", {})
            final_x = result.get("final_x") or result.get("x")
            final_obj = result.get("final_objective") or result.get("fun")
            success = result.get("success", False)

            if final_x is not None and success:
                # Prefer successful runs
                if final_obj is not None and final_obj < best_objective:
                    best_objective = final_obj
                    best_run = run

        if best_run:
            result = best_run.get("result", {})
            final_x = result.get("final_x") or result.get("x")
            return np.array(final_x)

        return None

    def _apply_domain_hint(self, problem: NLPProblem) -> np.ndarray:
        """
        Apply initialization based on domain hint.

        Domain-specific knowledge:
        - shape_optimization: Zero (baseline geometry has no deformation)
        - aerodynamic: Zero (same as shape_optimization)
        - structural: Center of bounds (material properties, thicknesses)
        - topology: Uniform density (typically 0.5 or based on volume fraction)
        - general: Center of bounds
        """
        hint = problem.domain_hint

        if hint in ("shape_optimization", "aerodynamic"):
            # FFD, mesh deformation: zero = baseline shape
            logger.info(f"Domain hint '{hint}': initializing at zero (baseline)")
            return np.zeros(problem.dimension)

        elif hint == "topology":
            # Topology optimization: uniform intermediate density
            # Often 0.5 or based on target volume fraction
            logger.info(f"Domain hint '{hint}': initializing at 0.5 (uniform density)")
            return np.full(problem.dimension, 0.5)

        elif hint in ("structural", "general"):
            # Center of bounds is a safe default
            logger.info(f"Domain hint '{hint}': initializing at center of bounds")
            return np.array(problem.get_bounds_center())

        else:
            # Unknown hint - use center as fallback
            logger.info(f"Unknown domain hint '{hint}': using center of bounds")
            return np.array(problem.get_bounds_center())

    def _apply_algorithm_default(
        self,
        problem: NLPProblem,
        algorithm: str
    ) -> np.ndarray:
        """
        Apply algorithm-specific default initialization.

        - Gradient-based: Center of bounds (safe for line search)
        - CMA-ES: Center of bounds (mean vector)
        - Population-based: Random/LHS (handled by generate_population)
        """
        if algorithm in GRADIENT_BASED_ALGORITHMS:
            logger.info(f"Gradient-based algorithm {algorithm}: center of bounds")
            return np.array(problem.get_bounds_center())

        elif algorithm in CMA_ES_ALGORITHMS:
            logger.info(f"CMA-ES algorithm: center of bounds as mean")
            return np.array(problem.get_bounds_center())

        elif algorithm in POPULATION_BASED_ALGORITHMS:
            # For population-based, this returns ONE point (best for single-start mode)
            # Use generate_population for actual population
            logger.info(f"Population-based algorithm: random point within bounds")
            bounds = np.array(problem.bounds)
            return np.random.uniform(bounds[:, 0], bounds[:, 1])

        else:
            # Unknown algorithm - use center as safe default
            logger.warning(f"Unknown algorithm '{algorithm}': using center of bounds")
            return np.array(problem.get_bounds_center())

    def compute_cmaes_params(
        self,
        problem: NLPProblem,
        sigma_fraction: float = 0.25
    ) -> tuple:
        """
        Compute CMA-ES parameters (mean and sigma).

        CMA-ES needs:
        - mean: Initial search center (use bounds center)
        - sigma: Initial step size (typically 0.25 × bound width)

        Args:
            problem: NLP problem
            sigma_fraction: Fraction of bound width for sigma (default 0.25)

        Returns:
            (mean, sigma) tuple
        """
        mean = np.array(problem.get_bounds_center())
        widths = np.array(problem.get_bounds_width())

        # Sigma should be about 0.25 × bound width (standard recommendation)
        sigma = sigma_fraction * np.mean(widths)

        logger.info(f"CMA-ES params: mean=center, sigma={sigma:.4f}")
        return mean, sigma

    def generate_population(
        self,
        problem: NLPProblem,
        size: int = 50,
        method: str = "lhs"
    ) -> np.ndarray:
        """
        Generate initial population for evolutionary algorithms.

        Args:
            problem: NLP problem
            size: Population size
            method: Sampling method ("lhs", "random", "sobol")

        Returns:
            Population array of shape (size, dimension)
        """
        bounds = np.array(problem.bounds)
        lower = bounds[:, 0]
        upper = bounds[:, 1]

        if method == "lhs":
            # Latin Hypercube Sampling - better coverage than random
            population = self._latin_hypercube_sample(problem.dimension, size)
            # Scale to bounds
            population = lower + population * (upper - lower)

        elif method == "random":
            # Uniform random sampling
            population = np.random.uniform(
                lower, upper, size=(size, problem.dimension)
            )

        elif method == "sobol":
            # Sobol sequence (if available)
            try:
                from scipy.stats import qmc
                sampler = qmc.Sobol(d=problem.dimension, scramble=True)
                population = sampler.random(n=size)
                # Scale to bounds
                population = lower + population * (upper - lower)
            except ImportError:
                logger.warning("Sobol requires scipy.stats.qmc, falling back to LHS")
                population = self._latin_hypercube_sample(problem.dimension, size)
                population = lower + population * (upper - lower)

        else:
            logger.warning(f"Unknown sampling method '{method}', using LHS")
            population = self._latin_hypercube_sample(problem.dimension, size)
            population = lower + population * (upper - lower)

        logger.info(f"Generated population: {size} samples via {method}")
        return population

    def _latin_hypercube_sample(self, dimension: int, size: int) -> np.ndarray:
        """
        Generate Latin Hypercube samples in [0, 1]^dimension.

        Args:
            dimension: Number of dimensions
            size: Number of samples

        Returns:
            Array of shape (size, dimension) with values in [0, 1]
        """
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=dimension)
            return sampler.random(n=size)
        except ImportError:
            # Fallback to simple LHS implementation
            result = np.zeros((size, dimension))
            for j in range(dimension):
                # Create evenly spaced strata
                strata = np.arange(size) / size
                # Add random offset within each stratum
                result[:, j] = strata + np.random.uniform(0, 1/size, size)
                # Shuffle to break correlations
                np.random.shuffle(result[:, j])
            return result

    def get_initialization_strategy(
        self,
        algorithm: str,
        domain_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get initialization strategy description for an algorithm.

        Useful for explaining to the agent what initialization approach
        will be used.

        Args:
            algorithm: Algorithm name
            domain_hint: Optional domain hint

        Returns:
            Dictionary describing the strategy
        """
        if algorithm in BAYESIAN_ALGORITHMS:
            return {
                "type": "sampler_handles",
                "description": "Bayesian optimization handles initialization internally",
                "user_input_needed": False
            }

        elif algorithm in CMA_ES_ALGORITHMS:
            return {
                "type": "mean_sigma",
                "description": "CMA-ES uses mean (center of bounds) and sigma (0.25 × width)",
                "user_input_needed": False
            }

        elif algorithm in POPULATION_BASED_ALGORITHMS:
            return {
                "type": "population_sampling",
                "description": "Population-based methods use LHS sampling within bounds",
                "user_input_needed": False,
                "default_method": "lhs"
            }

        elif algorithm in GRADIENT_BASED_ALGORITHMS:
            if domain_hint in ("shape_optimization", "aerodynamic"):
                return {
                    "type": "domain_specific",
                    "description": f"Shape optimization: initialize at zero (baseline)",
                    "user_input_needed": False,
                    "initial_point": "zero"
                }
            else:
                return {
                    "type": "bounds_center",
                    "description": "Gradient-based methods use center of bounds",
                    "user_input_needed": False,
                    "initial_point": "center"
                }

        else:
            return {
                "type": "unknown_algorithm",
                "description": f"Unknown algorithm '{algorithm}', will use center of bounds",
                "user_input_needed": False
            }
