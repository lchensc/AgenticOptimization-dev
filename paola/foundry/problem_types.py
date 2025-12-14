"""
Problem type detection and solver selection.

Helps agent identify problem types (LP, QP, NLP, MILP, etc.) and
select appropriate solvers based on problem characteristics.
"""

from typing import Dict, Any, List, Optional
from .nlp_schema import NLPProblem


class ProblemTypeDetector:
    """
    Detect optimization problem type from specification.

    Problem type taxonomy:
    - LP: Linear Programming (linear objective, linear constraints)
    - QP: Quadratic Programming (quadratic objective, linear constraints)
    - NLP: Nonlinear Programming (nonlinear objective/constraints)
    - MILP: Mixed-Integer Linear Programming
    - MINLP: Mixed-Integer Nonlinear Programming
    - MOO: Multi-Objective Optimization (vector objectives)
    - SDP: Semidefinite Programming (matrix constraints)

    Example:
        detector = ProblemTypeDetector()
        problem_type = detector.detect_from_nlp_problem(nlp_problem)
        # Returns: "NLP"
    """

    @staticmethod
    def detect_from_nlp_problem(problem: NLPProblem) -> str:
        """
        Detect problem type from NLP problem specification.

        Args:
            problem: NLP problem specification

        Returns:
            Problem type string: "NLP" (for now, always NLP)

        Note: Currently focused on NLP only. Future versions will detect:
        - LP (linear evaluators)
        - QP (quadratic evaluators)
        - MILP/MINLP (integer variables)
        """
        # For now, all registered evaluators are assumed nonlinear
        # Future: Analyze evaluator metadata to detect linearity
        return "NLP"

    @staticmethod
    def detect_from_specification(spec: Dict[str, Any]) -> str:
        """
        Detect problem type from general specification dict.

        Args:
            spec: Problem specification with characteristics

        Returns:
            One of: "LP", "QP", "NLP", "MILP", "MINLP", "MOO", "Unknown"

        Example:
            spec = {
                "has_integer_variables": False,
                "is_linear": False,
                "num_objectives": 1
            }
            problem_type = ProblemTypeDetector.detect_from_specification(spec)
            # Returns: "NLP"
        """
        # Check variable types
        has_integer_vars = spec.get("has_integer_variables", False)
        has_continuous_vars = spec.get("has_continuous_variables", True)

        # Check objective/constraint linearity
        # (For now, all user-provided evaluators are assumed nonlinear)
        is_linear = spec.get("is_linear", False)
        is_quadratic = spec.get("is_quadratic", False)

        # Check number of objectives
        num_objectives = spec.get("num_objectives", 1)

        # Decision logic
        if num_objectives > 1:
            return "MOO"  # Multi-objective

        if has_integer_vars and has_continuous_vars:
            # Mixed-integer
            if is_linear:
                return "MILP"
            else:
                return "MINLP"

        if has_integer_vars:
            # Pure integer
            if is_linear:
                return "ILP"  # Integer Linear Programming
            else:
                return "IP"  # Integer Programming (nonlinear)

        # Continuous variables only
        if is_linear:
            return "LP"
        elif is_quadratic:
            return "QP"
        else:
            return "NLP"  # Default for user evaluators


class SolverSelector:
    """
    Select appropriate solver based on problem type and characteristics.

    Provides expert knowledge about which solvers work best for
    different problem types.

    Example:
        selector = SolverSelector()
        solvers = selector.recommend_solver("NLP", gradient_available=True)
        # Returns: ["SLSQP", "IPOPT"]
    """

    # Solver mappings by problem type
    SOLVER_MAPPING = {
        "NLP": {
            "gradient_based": ["SLSQP", "IPOPT", "SNOPT"],
            "derivative_free": ["COBYLA", "BOBYQA", "Nelder-Mead"],
            "description": "Nonlinear Programming - continuous variables, nonlinear functions"
        },
        "LP": {
            "solvers": ["HiGHS", "CPLEX", "Gurobi", "GLPK"],
            "description": "Linear Programming - continuous variables, linear functions"
        },
        "QP": {
            "solvers": ["OSQP", "CVXOPT", "quadprog"],
            "description": "Quadratic Programming - continuous variables, quadratic objective"
        },
        "MILP": {
            "solvers": ["CPLEX", "Gurobi", "CBC", "SCIP"],
            "description": "Mixed-Integer Linear Programming - integer+continuous, linear"
        },
        "MINLP": {
            "solvers": ["Bonmin", "SCIP", "BARON"],
            "description": "Mixed-Integer Nonlinear Programming - integer+continuous, nonlinear"
        },
        "MOO": {
            "solvers": ["NSGA-II", "NSGA-III", "MOEA/D"],
            "description": "Multi-Objective Optimization - Pareto front discovery"
        }
    }

    # Available solvers in current implementation
    AVAILABLE_SOLVERS = {
        "scipy": [
            "SLSQP",       # NLP with constraints, gradient-based
            "COBYLA",      # NLP with constraints, derivative-free
            "L-BFGS-B",    # NLP with bounds, gradient-based
            "TNC",         # NLP with bounds, gradient-based
            "Nelder-Mead", # Unconstrained, derivative-free
            "Powell",      # Unconstrained, derivative-free
            "CG",          # Unconstrained, gradient-based
            "BFGS",        # Unconstrained, gradient-based
            "trust-constr" # NLP with constraints, gradient-based
        ]
    }

    @staticmethod
    def recommend_solver(
        problem_type: str,
        gradient_available: bool = True,
        has_constraints: bool = False
    ) -> List[str]:
        """
        Recommend solvers for problem type and characteristics.

        Returns ranked list of suitable solvers (best first).

        Args:
            problem_type: One of "LP", "QP", "NLP", "MILP", "MINLP", "MOO"
            gradient_available: Whether gradients can be computed
            has_constraints: Whether problem has constraints

        Returns:
            List of recommended solver names (ranked by suitability)

        Example:
            # NLP with constraints and gradients
            solvers = SolverSelector.recommend_solver(
                problem_type="NLP",
                gradient_available=True,
                has_constraints=True
            )
            # Returns: ["SLSQP", "trust-constr"]
        """
        if problem_type == "NLP":
            return SolverSelector._recommend_nlp_solver(
                gradient_available=gradient_available,
                has_constraints=has_constraints
            )

        elif problem_type == "LP":
            # Phase 6+: Linear programming solvers
            return ["HiGHS"]  # Open-source LP solver

        elif problem_type == "QP":
            # Phase 6+: Quadratic programming solvers
            return ["OSQP"]  # Open-source QP solver

        elif problem_type == "MILP":
            # Phase 7+: Mixed-integer linear
            return ["CBC", "SCIP"]

        elif problem_type == "MINLP":
            # Phase 7+: Mixed-integer nonlinear
            return ["Bonmin", "SCIP"]

        elif problem_type == "MOO":
            # Phase 7+: Multi-objective (Pymoo integration)
            return ["NSGA-II", "NSGA-III"]

        else:
            # Unknown type - fall back to general NLP solver
            return ["SLSQP"]

    @staticmethod
    def _recommend_nlp_solver(
        gradient_available: bool,
        has_constraints: bool
    ) -> List[str]:
        """
        Recommend NLP solver based on characteristics.

        Args:
            gradient_available: Whether gradients can be computed
            has_constraints: Whether problem has constraints

        Returns:
            Ranked list of NLP solvers
        """
        if has_constraints:
            if gradient_available:
                # Constrained, gradient-based
                return ["SLSQP", "trust-constr"]
            else:
                # Constrained, derivative-free
                return ["COBYLA"]
        else:
            # Unconstrained
            if gradient_available:
                # Gradient-based for unconstrained
                return ["L-BFGS-B", "BFGS", "CG"]
            else:
                # Derivative-free for unconstrained
                return ["Nelder-Mead", "Powell"]

    @staticmethod
    def is_solver_available(solver_name: str) -> bool:
        """
        Check if solver is available in current implementation.

        Args:
            solver_name: Solver name (e.g., "SLSQP")

        Returns:
            True if solver is available, False otherwise
        """
        for library_solvers in SolverSelector.AVAILABLE_SOLVERS.values():
            if solver_name in library_solvers:
                return True
        return False

    @staticmethod
    def get_solver_description(problem_type: str) -> Optional[str]:
        """
        Get description of problem type.

        Args:
            problem_type: Problem type string

        Returns:
            Description string or None
        """
        mapping = SolverSelector.SOLVER_MAPPING.get(problem_type, {})
        return mapping.get("description")

    @staticmethod
    def validate_solver_for_problem(
        solver_name: str,
        problem_type: str,
        has_constraints: bool = False
    ) -> Dict[str, Any]:
        """
        Validate if solver is appropriate for problem type.

        Args:
            solver_name: Solver name
            problem_type: Problem type
            has_constraints: Whether problem has constraints

        Returns:
            Dict with:
                - valid: bool
                - warning: Optional[str]
                - recommendation: Optional[str]
        """
        # Check if solver exists
        if not SolverSelector.is_solver_available(solver_name):
            return {
                "valid": False,
                "warning": f"Solver '{solver_name}' not available",
                "recommendation": f"Use: {SolverSelector.recommend_solver(problem_type)[0]}"
            }

        # Specific validations
        if problem_type == "NLP":
            # Check constraint support
            derivative_free_solvers = ["COBYLA", "Nelder-Mead", "Powell"]
            constrained_solvers = ["SLSQP", "COBYLA", "trust-constr"]

            if has_constraints and solver_name not in constrained_solvers:
                return {
                    "valid": False,
                    "warning": f"Solver '{solver_name}' doesn't support constraints",
                    "recommendation": "Use SLSQP or COBYLA for constrained problems"
                }

        return {"valid": True}
