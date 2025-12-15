"""
ConfigurationManager: Agent intelligence for algorithm selection and configuration.

The Paola Principle: "Configuration complexity is Paola intelligence, not user burden."

Optimizer configuration is a vast knowledge space:
- IPOPT: 250+ options (barrier parameters, scaling, linear solvers)
- SNOPT: 100+ options (major/minor iterations, scaling)
- SciPy: Method-specific options
- NLopt: Algorithm families with different capabilities

Most users only touch 3-5 options. Paola should:
1. Select appropriate algorithm based on problem characteristics
2. Configure sensible defaults based on optimization priority
3. Hide complexity while allowing expert override

Priority-based configuration:
- robustness: Conservative tolerances, more iterations, safer defaults
- speed: Relaxed tolerances, early stopping, aggressive settings
- accuracy: Tight tolerances, high precision, thorough convergence
- balanced: Middle ground (default)
"""

from typing import Dict, Any, Optional, List, Literal
import logging

from ..foundry.nlp_schema import NLPProblem

logger = logging.getLogger(__name__)


# Priority type for type hints
PriorityType = Literal["robustness", "speed", "accuracy", "balanced"]


class ConfigurationManager:
    """
    Agent intelligence for algorithm selection and configuration.

    The Paola Principle: Users specify intent ("fast", "robust", "accurate"),
    Paola translates to algorithm-specific options.

    Example:
        manager = ConfigurationManager()

        # Auto-select algorithm based on problem
        algorithm = manager.select_algorithm(problem, priority="robustness")

        # Get configuration for selected algorithm
        config = manager.configure_algorithm(algorithm, problem, priority="robustness")
    """

    def __init__(self):
        """Initialize configuration manager."""
        pass

    def select_algorithm(
        self,
        problem: NLPProblem,
        priority: PriorityType = "balanced",
        available_backends: Optional[List[str]] = None
    ) -> str:
        """
        Select appropriate algorithm based on problem characteristics.

        Decision factors:
        - Dimension: Large problems → sparse methods
        - Constraints: Constrained → SLSQP, IPOPT; Unconstrained → L-BFGS-B
        - Priority: Robustness → SLSQP; Speed → L-BFGS-B; Accuracy → trust-constr
        - Available backends: What's installed

        Args:
            problem: NLP problem to solve
            priority: Optimization priority
            available_backends: List of available backends (default: ["scipy"])

        Returns:
            Algorithm name (e.g., "SLSQP", "L-BFGS-B")
        """
        if available_backends is None:
            available_backends = ["scipy"]  # SciPy is always available

        is_constrained = problem.is_constrained
        is_large = problem.dimension > 100
        has_equality = problem.num_equality_constraints > 0

        # Decision tree
        if is_constrained:
            if priority == "robustness":
                # SLSQP is robust for constrained problems
                return "SLSQP"
            elif priority == "speed":
                # SLSQP is also fast for moderate problems
                return "SLSQP"
            elif priority == "accuracy":
                # trust-constr provides high accuracy
                return "trust-constr"
            else:  # balanced
                return "SLSQP"

        else:  # Unconstrained
            if priority == "robustness":
                return "L-BFGS-B"
            elif priority == "speed":
                # L-BFGS-B is fast for unconstrained
                return "L-BFGS-B"
            elif priority == "accuracy":
                return "trust-constr"
            else:  # balanced
                return "L-BFGS-B"

    def configure_algorithm(
        self,
        algorithm: str,
        problem: NLPProblem,
        priority: PriorityType = "balanced",
        max_iterations: Optional[int] = None,
        custom_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate algorithm-specific configuration.

        Args:
            algorithm: Algorithm name
            problem: NLP problem
            priority: Optimization priority
            max_iterations: Override max iterations
            custom_options: Custom options to merge (expert override)

        Returns:
            Configuration dictionary for the algorithm
        """
        # Get base configuration for algorithm and priority
        config = self._get_base_config(algorithm, priority, problem.dimension)

        # Override max iterations if specified
        if max_iterations is not None:
            config = self._set_max_iterations(config, algorithm, max_iterations)

        # Merge custom options (expert override)
        if custom_options:
            config = self._merge_options(config, custom_options)

        logger.info(f"Configured {algorithm} with priority='{priority}'")
        return config

    def _get_base_config(
        self,
        algorithm: str,
        priority: PriorityType,
        dimension: int
    ) -> Dict[str, Any]:
        """Get base configuration for algorithm and priority."""
        # Algorithm-specific configuration methods
        config_methods = {
            "SLSQP": self._config_slsqp,
            "L-BFGS-B": self._config_lbfgsb,
            "trust-constr": self._config_trust_constr,
            "BFGS": self._config_bfgs,
            "CG": self._config_cg,
            "TNC": self._config_tnc,
            "COBYLA": self._config_cobyla,
        }

        if algorithm in config_methods:
            return config_methods[algorithm](priority, dimension)
        else:
            # Unknown algorithm - return generic config
            logger.warning(f"Unknown algorithm '{algorithm}', using generic config")
            return self._config_generic(priority, dimension)

    def _config_slsqp(self, priority: PriorityType, dimension: int) -> Dict[str, Any]:
        """Configure SLSQP based on priority."""
        configs = {
            "robustness": {
                "method": "SLSQP",
                "options": {
                    "maxiter": 200,
                    "ftol": 1e-6,
                    "eps": 1e-8,
                    "disp": False
                }
            },
            "speed": {
                "method": "SLSQP",
                "options": {
                    "maxiter": 100,
                    "ftol": 1e-4,
                    "eps": 1e-6,
                    "disp": False
                }
            },
            "accuracy": {
                "method": "SLSQP",
                "options": {
                    "maxiter": 500,
                    "ftol": 1e-9,
                    "eps": 1e-10,
                    "disp": False
                }
            },
            "balanced": {
                "method": "SLSQP",
                "options": {
                    "maxiter": 150,
                    "ftol": 1e-6,
                    "eps": 1e-8,
                    "disp": False
                }
            }
        }
        return configs.get(priority, configs["balanced"])

    def _config_lbfgsb(self, priority: PriorityType, dimension: int) -> Dict[str, Any]:
        """Configure L-BFGS-B based on priority."""
        # Memory parameter scales with problem size
        m = min(10, max(5, dimension // 10))

        configs = {
            "robustness": {
                "method": "L-BFGS-B",
                "options": {
                    "maxiter": 200,
                    "maxfun": 15000,
                    "ftol": 1e-9,
                    "gtol": 1e-6,
                    "maxcor": m,
                    "disp": False
                }
            },
            "speed": {
                "method": "L-BFGS-B",
                "options": {
                    "maxiter": 100,
                    "maxfun": 5000,
                    "ftol": 1e-6,
                    "gtol": 1e-4,
                    "maxcor": m,
                    "disp": False
                }
            },
            "accuracy": {
                "method": "L-BFGS-B",
                "options": {
                    "maxiter": 500,
                    "maxfun": 30000,
                    "ftol": 1e-12,
                    "gtol": 1e-8,
                    "maxcor": m,
                    "disp": False
                }
            },
            "balanced": {
                "method": "L-BFGS-B",
                "options": {
                    "maxiter": 150,
                    "maxfun": 10000,
                    "ftol": 1e-9,
                    "gtol": 1e-5,
                    "maxcor": m,
                    "disp": False
                }
            }
        }
        return configs.get(priority, configs["balanced"])

    def _config_trust_constr(self, priority: PriorityType, dimension: int) -> Dict[str, Any]:
        """Configure trust-constr based on priority."""
        configs = {
            "robustness": {
                "method": "trust-constr",
                "options": {
                    "maxiter": 200,
                    "gtol": 1e-6,
                    "xtol": 1e-8,
                    "barrier_tol": 1e-6,
                    "disp": False
                }
            },
            "speed": {
                "method": "trust-constr",
                "options": {
                    "maxiter": 100,
                    "gtol": 1e-4,
                    "xtol": 1e-6,
                    "barrier_tol": 1e-4,
                    "disp": False
                }
            },
            "accuracy": {
                "method": "trust-constr",
                "options": {
                    "maxiter": 500,
                    "gtol": 1e-9,
                    "xtol": 1e-12,
                    "barrier_tol": 1e-9,
                    "disp": False
                }
            },
            "balanced": {
                "method": "trust-constr",
                "options": {
                    "maxiter": 150,
                    "gtol": 1e-6,
                    "xtol": 1e-8,
                    "barrier_tol": 1e-6,
                    "disp": False
                }
            }
        }
        return configs.get(priority, configs["balanced"])

    def _config_bfgs(self, priority: PriorityType, dimension: int) -> Dict[str, Any]:
        """Configure BFGS based on priority."""
        configs = {
            "robustness": {
                "method": "BFGS",
                "options": {
                    "maxiter": 200,
                    "gtol": 1e-6,
                    "disp": False
                }
            },
            "speed": {
                "method": "BFGS",
                "options": {
                    "maxiter": 100,
                    "gtol": 1e-4,
                    "disp": False
                }
            },
            "accuracy": {
                "method": "BFGS",
                "options": {
                    "maxiter": 500,
                    "gtol": 1e-9,
                    "disp": False
                }
            },
            "balanced": {
                "method": "BFGS",
                "options": {
                    "maxiter": 150,
                    "gtol": 1e-5,
                    "disp": False
                }
            }
        }
        return configs.get(priority, configs["balanced"])

    def _config_cg(self, priority: PriorityType, dimension: int) -> Dict[str, Any]:
        """Configure CG (Conjugate Gradient) based on priority."""
        configs = {
            "robustness": {
                "method": "CG",
                "options": {
                    "maxiter": 200,
                    "gtol": 1e-6,
                    "disp": False
                }
            },
            "speed": {
                "method": "CG",
                "options": {
                    "maxiter": 100,
                    "gtol": 1e-4,
                    "disp": False
                }
            },
            "accuracy": {
                "method": "CG",
                "options": {
                    "maxiter": 500,
                    "gtol": 1e-9,
                    "disp": False
                }
            },
            "balanced": {
                "method": "CG",
                "options": {
                    "maxiter": 150,
                    "gtol": 1e-5,
                    "disp": False
                }
            }
        }
        return configs.get(priority, configs["balanced"])

    def _config_tnc(self, priority: PriorityType, dimension: int) -> Dict[str, Any]:
        """Configure TNC (Truncated Newton) based on priority."""
        configs = {
            "robustness": {
                "method": "TNC",
                "options": {
                    "maxiter": 200,
                    "ftol": 1e-9,
                    "gtol": 1e-6,
                    "disp": False
                }
            },
            "speed": {
                "method": "TNC",
                "options": {
                    "maxiter": 100,
                    "ftol": 1e-6,
                    "gtol": 1e-4,
                    "disp": False
                }
            },
            "accuracy": {
                "method": "TNC",
                "options": {
                    "maxiter": 500,
                    "ftol": 1e-12,
                    "gtol": 1e-9,
                    "disp": False
                }
            },
            "balanced": {
                "method": "TNC",
                "options": {
                    "maxiter": 150,
                    "ftol": 1e-9,
                    "gtol": 1e-5,
                    "disp": False
                }
            }
        }
        return configs.get(priority, configs["balanced"])

    def _config_cobyla(self, priority: PriorityType, dimension: int) -> Dict[str, Any]:
        """Configure COBYLA based on priority."""
        # COBYLA is derivative-free, good for noisy problems
        configs = {
            "robustness": {
                "method": "COBYLA",
                "options": {
                    "maxiter": 1000,
                    "rhobeg": 0.5,
                    "tol": 1e-6,
                    "disp": False
                }
            },
            "speed": {
                "method": "COBYLA",
                "options": {
                    "maxiter": 500,
                    "rhobeg": 1.0,
                    "tol": 1e-4,
                    "disp": False
                }
            },
            "accuracy": {
                "method": "COBYLA",
                "options": {
                    "maxiter": 2000,
                    "rhobeg": 0.25,
                    "tol": 1e-8,
                    "disp": False
                }
            },
            "balanced": {
                "method": "COBYLA",
                "options": {
                    "maxiter": 1000,
                    "rhobeg": 0.5,
                    "tol": 1e-6,
                    "disp": False
                }
            }
        }
        return configs.get(priority, configs["balanced"])

    def _config_generic(self, priority: PriorityType, dimension: int) -> Dict[str, Any]:
        """Generic configuration for unknown algorithms."""
        iterations = {
            "robustness": 200,
            "speed": 100,
            "accuracy": 500,
            "balanced": 150
        }
        tolerances = {
            "robustness": 1e-6,
            "speed": 1e-4,
            "accuracy": 1e-9,
            "balanced": 1e-6
        }
        return {
            "options": {
                "maxiter": iterations.get(priority, 150),
                "tol": tolerances.get(priority, 1e-6),
                "disp": False
            }
        }

    def _set_max_iterations(
        self,
        config: Dict[str, Any],
        algorithm: str,
        max_iterations: int
    ) -> Dict[str, Any]:
        """Set max iterations in config."""
        config = dict(config)  # Copy to avoid mutation
        if "options" not in config:
            config["options"] = {}
        config["options"] = dict(config["options"])
        config["options"]["maxiter"] = max_iterations
        return config

    def _merge_options(
        self,
        config: Dict[str, Any],
        custom_options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge custom options into config."""
        config = dict(config)
        if "options" in custom_options:
            config["options"] = {**config.get("options", {}), **custom_options["options"]}
        # Merge top-level keys
        for key, value in custom_options.items():
            if key != "options":
                config[key] = value
        return config

    def get_algorithm_info(self, algorithm: str) -> Dict[str, Any]:
        """
        Get information about an algorithm.

        Args:
            algorithm: Algorithm name

        Returns:
            Dictionary with algorithm information
        """
        info = {
            "SLSQP": {
                "name": "Sequential Least Squares Programming",
                "type": "gradient-based",
                "supports_bounds": True,
                "supports_constraints": True,
                "derivative_free": False,
                "best_for": "Small to medium constrained problems"
            },
            "L-BFGS-B": {
                "name": "Limited-memory BFGS with Box constraints",
                "type": "gradient-based",
                "supports_bounds": True,
                "supports_constraints": False,
                "derivative_free": False,
                "best_for": "Large unconstrained or bound-constrained problems"
            },
            "trust-constr": {
                "name": "Trust Region Constrained Algorithm",
                "type": "gradient-based",
                "supports_bounds": True,
                "supports_constraints": True,
                "derivative_free": False,
                "best_for": "High-accuracy constrained optimization"
            },
            "BFGS": {
                "name": "Broyden-Fletcher-Goldfarb-Shanno",
                "type": "gradient-based",
                "supports_bounds": False,
                "supports_constraints": False,
                "derivative_free": False,
                "best_for": "Unconstrained smooth problems"
            },
            "COBYLA": {
                "name": "Constrained Optimization BY Linear Approximations",
                "type": "derivative-free",
                "supports_bounds": False,
                "supports_constraints": True,
                "derivative_free": True,
                "best_for": "Noisy or non-differentiable constrained problems"
            }
        }
        return info.get(algorithm, {"name": algorithm, "type": "unknown"})

    def get_priority_description(self, priority: PriorityType) -> str:
        """Get description of what a priority means."""
        descriptions = {
            "robustness": "Conservative settings that prioritize convergence reliability",
            "speed": "Relaxed tolerances and early stopping for faster results",
            "accuracy": "Tight tolerances for high-precision solutions",
            "balanced": "Middle ground between speed and accuracy (default)"
        }
        return descriptions.get(priority, "Unknown priority")
