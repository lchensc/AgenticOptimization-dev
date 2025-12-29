"""
pymoo optimization backend.

Wraps pymoo for evolutionary and multi-objective optimization:
- Single-objective: GA, DE, PSO, CMA-ES, ES, BRKGA
- Multi-objective: NSGA-II, NSGA-III, MOEA/D, AGE-MOEA, SMS-EMOA

Reference: https://pymoo.org/
"""

from typing import Dict, Any, List, Optional, Callable
import numpy as np
import logging

from ..base import OptimizerBackend
from ..result import OptimizationResult

logger = logging.getLogger(__name__)


class PymooBackend(OptimizerBackend):
    """
    pymoo optimization backend for evolutionary algorithms.

    Supports both single-objective and multi-objective optimization
    with a wide range of algorithms from pymoo.

    Single-objective algorithms:
    - GA: Genetic Algorithm
    - DE: Differential Evolution
    - PSO: Particle Swarm Optimization
    - CMA-ES: Covariance Matrix Adaptation
    - ES: Evolution Strategy
    - BRKGA: Biased Random Key GA

    Multi-objective algorithms:
    - NSGA-II: Non-dominated Sorting GA II
    - NSGA-III: Reference-point based NSGA
    - MOEA/D: Multi-objective EA with Decomposition
    - AGE-MOEA: Adaptive Geometry Estimation MOEA
    - SMS-EMOA: S-metric Selection EMOA
    """

    # Single-objective algorithms
    SOO_ALGORITHMS = {
        "GA": "Genetic Algorithm",
        "DE": "Differential Evolution",
        "PSO": "Particle Swarm Optimization",
        "CMA-ES": "Covariance Matrix Adaptation ES",
        "ES": "Evolution Strategy",
        "BRKGA": "Biased Random Key GA",
        "NelderMead": "Nelder-Mead Simplex",
        "PatternSearch": "Pattern Search",
    }

    # Multi-objective algorithms
    MOO_ALGORITHMS = {
        "NSGA-II": "Non-dominated Sorting GA II",
        "NSGA-III": "Reference-point NSGA-III",
        "MOEA/D": "MOEA with Decomposition",
        "AGE-MOEA": "Adaptive Geometry MOEA",
        "AGE-MOEA2": "AGE-MOEA Version 2",
        "SMS-EMOA": "S-metric Selection EMOA",
        "R-NSGA-III": "Reference-point NSGA-III",
        "U-NSGA-III": "Unified NSGA-III",
        "C-TAEA": "Constrained Two-Archive EA",
    }

    @property
    def name(self) -> str:
        return "pymoo"

    @property
    def family(self) -> str:
        return "evolutionary"

    @property
    def supports_multiobjective(self) -> bool:
        return True

    @property
    def supports_constraints(self) -> bool:
        return True

    @property
    def supports_gradients(self) -> bool:
        return False  # Evolutionary algorithms are derivative-free

    def is_available(self) -> bool:
        try:
            import pymoo
            return True
        except ImportError:
            return False

    def get_methods(self) -> List[str]:
        return list(self.SOO_ALGORITHMS.keys()) + list(self.MOO_ALGORITHMS.keys())

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "pymoo",
            "soo_algorithms": list(self.SOO_ALGORITHMS.keys()),
            "moo_algorithms": list(self.MOO_ALGORITHMS.keys()),
            "skill": "pymoo",  # Use load_skill("pymoo") for detailed options
        }

    def _create_algorithm(
        self,
        algorithm_name: str,
        config: Dict[str, Any],
        n_var: int,
        n_obj: int = 1,
    ):
        """
        Create pymoo algorithm instance.

        Args:
            algorithm_name: Algorithm name (e.g., "GA", "NSGA-II")
            config: Algorithm configuration
            n_var: Number of design variables
            n_obj: Number of objectives

        Returns:
            pymoo Algorithm instance
        """
        pop_size = config.get("pop_size", 100)

        # Single-objective algorithms
        if algorithm_name == "GA":
            from pymoo.algorithms.soo.nonconvex.ga import GA
            return GA(pop_size=pop_size)

        elif algorithm_name == "DE":
            from pymoo.algorithms.soo.nonconvex.de import DE
            return DE(
                pop_size=pop_size,
                variant=config.get("variant", "DE/rand/1/bin"),
                CR=config.get("CR", 0.9),
                F=config.get("F", 0.8),
            )

        elif algorithm_name == "PSO":
            from pymoo.algorithms.soo.nonconvex.pso import PSO
            return PSO(
                pop_size=pop_size,
                w=config.get("w", 0.9),
                c1=config.get("c1", 2.0),
                c2=config.get("c2", 2.0),
            )

        elif algorithm_name == "CMA-ES":
            from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
            x0 = config.get("x0")
            sigma = config.get("sigma", 0.5)
            return CMAES(x0=x0, sigma=sigma)

        elif algorithm_name == "ES":
            from pymoo.algorithms.soo.nonconvex.es import ES
            return ES(n_offsprings=config.get("n_offsprings", pop_size))

        elif algorithm_name == "BRKGA":
            from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
            return BRKGA(
                n_elites=config.get("n_elites", int(0.2 * pop_size)),
                n_mutants=config.get("n_mutants", int(0.1 * pop_size)),
                bias=config.get("bias", 0.7),
            )

        elif algorithm_name == "NelderMead":
            from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
            return NelderMead()

        elif algorithm_name == "PatternSearch":
            from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
            return PatternSearch()

        # Multi-objective algorithms
        elif algorithm_name == "NSGA-II":
            from pymoo.algorithms.moo.nsga2 import NSGA2
            return NSGA2(pop_size=pop_size)

        elif algorithm_name in ["NSGA-III", "R-NSGA-III", "U-NSGA-III"]:
            from pymoo.algorithms.moo.nsga3 import NSGA3
            from pymoo.util.ref_dirs import get_reference_directions

            # Get reference directions for many-objective
            n_partitions = config.get("n_partitions", 12)
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
            return NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)

        elif algorithm_name == "MOEA/D":
            from pymoo.algorithms.moo.moead import MOEAD
            from pymoo.util.ref_dirs import get_reference_directions

            n_partitions = config.get("n_partitions", 12)
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
            return MOEAD(ref_dirs=ref_dirs)

        elif algorithm_name == "AGE-MOEA":
            from pymoo.algorithms.moo.age import AGEMOEA
            return AGEMOEA(pop_size=pop_size)

        elif algorithm_name == "AGE-MOEA2":
            from pymoo.algorithms.moo.age2 import AGEMOEA2
            return AGEMOEA2(pop_size=pop_size)

        elif algorithm_name == "SMS-EMOA":
            from pymoo.algorithms.moo.sms import SMSEMOA
            return SMSEMOA(pop_size=pop_size)

        elif algorithm_name == "C-TAEA":
            from pymoo.algorithms.moo.ctaea import CTAEA
            from pymoo.util.ref_dirs import get_reference_directions

            n_partitions = config.get("n_partitions", 12)
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
            return CTAEA(ref_dirs=ref_dirs)

        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}. Available: {self.get_methods()}")

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: List[List[float]],
        x0: np.ndarray,
        config: Dict[str, Any],
        constraints: Optional[List[Dict[str, Any]]] = None,
        gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> OptimizationResult:
        """
        Run pymoo optimization.

        Args:
            objective: Function f(x) -> float (or list of functions for multi-objective)
            bounds: Variable bounds [[lb, ub], ...]
            x0: Initial design point
            config: Algorithm configuration including:
                - algorithm: Algorithm name (default: "GA")
                - n_gen: Number of generations (default: 100)
                - pop_size: Population size (default: 100)
                - seed: Random seed
                - objectives: List of objective functions for multi-objective
            constraints: Optional constraint functions
            gradient: Not used (evolutionary algorithms are derivative-free)

        Returns:
            OptimizationResult with solution and statistics
        """
        try:
            from pymoo.core.problem import ElementwiseProblem
            from pymoo.optimize import minimize
            from pymoo.termination import get_termination
        except ImportError:
            return OptimizationResult.from_failure(
                message="pymoo not available. Install with: pip install pymoo",
                x0=x0,
            )

        # Extract configuration
        # "method" or "algorithm" specifies the pymoo algorithm (e.g., "NSGA-II", "GA")
        algorithm_name = config.get("algorithm", config.get("method", "GA"))
        n_gen = config.get("n_gen", config.get("n_generations", config.get("max_iterations", 100)))
        seed = config.get("seed")

        # Determine number of objectives
        # Agent specifies n_obj in config for MOO (preferred method)
        # Fall back to counting objectives list for backward compatibility
        if "n_obj" in config:
            n_obj = config["n_obj"]
            # Single objective function that returns array for MOO
            objectives = [objective]
        else:
            # Legacy: multiple objective functions in list
            objectives = config.get("objectives", [objective])
            if not isinstance(objectives, list):
                objectives = [objectives]
            n_obj = len(objectives)

        is_moo = n_obj > 1

        # Use appropriate algorithm for MOO
        if is_moo and algorithm_name in self.SOO_ALGORITHMS:
            algorithm_name = "NSGA-II"  # Default MOO algorithm
            logger.info(f"Switched to {algorithm_name} for multi-objective optimization")

        n_var = len(bounds)
        xl = np.array([b[0] for b in bounds])
        xu = np.array([b[1] for b in bounds])

        # Handle constraints
        n_constr = 0
        constraint_funcs = []
        if constraints:
            for c in constraints:
                if "func" in c:
                    constraint_funcs.append(c["func"])
                    n_constr += 1

        # Tracking
        history = []
        n_evals = [0]

        # Check if using single array-returning objective (from config n_obj)
        use_array_objective = "n_obj" in config and n_obj > 1

        # Create pymoo Problem
        class PaolaProblem(ElementwiseProblem):
            def __init__(self):
                super().__init__(
                    n_var=n_var,
                    n_obj=n_obj,
                    n_constr=n_constr,
                    xl=xl,
                    xu=xu,
                )

            def _evaluate(self, x, out, *args, **kwargs):
                n_evals[0] += 1

                # Evaluate objectives
                if use_array_objective:
                    # Single objective function returning array (MOO via config n_obj)
                    f_result = objectives[0](x)
                    f_values = np.asarray(f_result).flatten().tolist()
                else:
                    # Multiple objective functions or single SOO function
                    f_values = [float(obj(x)) for obj in objectives]

                out["F"] = f_values if is_moo else f_values[0]

                # Record history
                history.append({
                    "iteration": n_evals[0],
                    "objectives": f_values,
                    "design": x.tolist(),
                })

                # Evaluate constraints
                if constraint_funcs:
                    g_values = [float(g(x)) for g in constraint_funcs]
                    out["G"] = g_values

        try:
            # Create problem and algorithm
            problem = PaolaProblem()
            algorithm = self._create_algorithm(algorithm_name, config, n_var, n_obj)

            # Set termination
            termination = get_termination("n_gen", n_gen)

            # Run optimization
            res = minimize(
                problem,
                algorithm,
                termination,
                seed=seed,
                verbose=config.get("verbose", False),
                save_history=False,  # We track our own history
            )

            # Extract results
            if res.X is None:
                return OptimizationResult.from_failure(
                    message="Optimization did not find a valid solution",
                    x0=x0,
                    n_evals=n_evals[0],
                    history=history,
                )

            if is_moo:
                # Multi-objective result
                pareto_set = res.X if res.X.ndim > 1 else res.X.reshape(1, -1)
                pareto_front = res.F if res.F.ndim > 1 else res.F.reshape(1, -1)

                # Compute hypervolume if possible
                hypervolume = None
                try:
                    from pymoo.indicators.hv import HV
                    ref_point = config.get("ref_point", np.max(pareto_front, axis=0) * 1.1)
                    hv = HV(ref_point=ref_point)
                    hypervolume = float(hv(pareto_front))
                except Exception as e:
                    logger.debug(f"Could not compute hypervolume: {e}")

                return OptimizationResult(
                    success=True,
                    message=f"Found {len(pareto_set)} Pareto-optimal solutions",
                    best_x=pareto_set[0],  # First solution as representative
                    best_f=float(pareto_front[0, 0]),  # First objective of first solution
                    pareto_set=pareto_set,
                    pareto_front=pareto_front,
                    hypervolume=hypervolume,
                    n_iterations=res.algorithm.n_gen if hasattr(res.algorithm, "n_gen") else n_gen,
                    n_function_evals=n_evals[0],
                    n_gradient_evals=0,
                    history=history,
                    raw_result=res,
                )
            else:
                # Single-objective result
                best_x = res.X
                best_f = float(res.F) if np.isscalar(res.F) else float(res.F[0])

                return OptimizationResult(
                    success=True,
                    message=f"Optimization complete after {n_gen} generations",
                    best_x=best_x,
                    best_f=best_f,
                    n_iterations=res.algorithm.n_gen if hasattr(res.algorithm, "n_gen") else n_gen,
                    n_function_evals=n_evals[0],
                    n_gradient_evals=0,
                    history=history,
                    raw_result=res,
                )

        except Exception as e:
            logger.error(f"pymoo optimization failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return OptimizationResult.from_failure(
                message=f"pymoo optimization failed: {str(e)}",
                x0=x0,
                n_evals=n_evals[0],
                history=history,
            )

    def optimize_multiobjective(
        self,
        objectives: List[Callable[[np.ndarray], float]],
        bounds: List[List[float]],
        config: Dict[str, Any],
        constraints: Optional[List[Callable[[np.ndarray], float]]] = None,
    ) -> OptimizationResult:
        """
        Convenience method for multi-objective optimization.

        Args:
            objectives: List of objective functions [f1(x), f2(x), ...]
            bounds: Variable bounds [[lb, ub], ...]
            config: Algorithm configuration
            constraints: Optional constraint functions (g(x) <= 0)

        Returns:
            OptimizationResult with Pareto front
        """
        n_var = len(bounds)
        x0 = np.array([(b[0] + b[1]) / 2 for b in bounds])

        # Add objectives to config
        config = config.copy()
        config["objectives"] = objectives

        # Convert constraints to expected format
        constraint_dicts = None
        if constraints:
            constraint_dicts = [{"func": c} for c in constraints]

        return self.optimize(
            objective=objectives[0],  # Primary objective
            bounds=bounds,
            x0=x0,
            config=config,
            constraints=constraint_dicts,
        )
