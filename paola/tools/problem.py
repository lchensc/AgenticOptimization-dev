"""
Problem formulation tools.

Tools for creating and managing optimization problems:
- create_nlp_problem: Create NLP problem from evaluators
- derive_problem: Derive new problem with modified bounds
- list_problems: List all problems
- get_problem_lineage: Get problem derivation history
"""

from typing import Optional, Dict, Any, List
from langchain_core.tools import tool

from paola.tools.schemas import (
    normalize_problem_id,
    ProblemIdType,
    DeriveProblemArgs,
)


@tool
def create_nlp_problem(
    name: str,
    objective_evaluator_id: str,
    bounds: Any,  # Accept both explicit list OR compact BoundsSpec dict
    objective_sense: str = "minimize",
    inequality_constraints: Optional[List[Dict[str, Any]]] = None,
    equality_constraints: Optional[List[Dict[str, Any]]] = None,
    domain_hint: Optional[str] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create NLP optimization problem.

    Args:
        name: Problem name
        objective_evaluator_id: Registered evaluator ID for objective f(x)
        bounds: [[lo, hi], ...] or {"type": "uniform", "lower": lo, "upper": hi, "dimension": n}
        objective_sense: "minimize" or "maximize"
        inequality_constraints: [{"name": str, "evaluator_id": str, "type": ">=" or "<=", "value": float}]
        equality_constraints: [{"name": str, "evaluator_id": str, "value": float}]
        domain_hint: "shape_optimization" or "general"
        description: Problem description

    Example:
        # Constrained optimization (register constraint evaluator first)
        create_nlp_problem(
            name="Portfolio",
            objective_evaluator_id="sharpe_eval",
            bounds=[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
            inequality_constraints=[
                {"name": "min_bonds", "evaluator_id": "bond_constraint", "type": ">=", "value": 0.0}
            ]
        )
    """
    try:
        from paola.foundry import (
            NLPProblem,
            InequalityConstraint,
            EqualityConstraint,
            NLPEvaluator,
            SolverSelector,
        )
        from paola.tools.graph_tools import get_foundry

        # Use global Foundry (single source of truth - v0.4.6)
        foundry = get_foundry()
        if foundry is None:
            return {
                "success": False,
                "message": "Foundry not initialized. CLI should call set_foundry() at startup.",
            }

        # Get next numeric problem_id (v0.4.3)
        problem_id = foundry.storage.get_next_problem_id()

        # Verify objective evaluator exists
        try:
            foundry.get_evaluator_config(objective_evaluator_id)
        except KeyError:
            available = foundry.list_evaluators()
            return {
                "success": False,
                "message": (
                    f"Objective evaluator '{objective_evaluator_id}' not found in Foundry.\n"
                    f"Available evaluators: {[e['evaluator_id'] for e in available]}\n"
                    f"Use foundry_list_evaluators to see all registered evaluators."
                )
            }

        # Parse inequality constraints
        ineq_constraints_objs = []
        if inequality_constraints:
            for cons_dict in inequality_constraints:
                # Verify constraint evaluator exists
                cons_eval_id = cons_dict["evaluator_id"]
                try:
                    foundry.get_evaluator_config(cons_eval_id)
                except KeyError:
                    return {
                        "success": False,
                        "message": f"Constraint evaluator '{cons_eval_id}' not found in Foundry."
                    }

                ineq_constraints_objs.append(InequalityConstraint(
                    name=cons_dict["name"],
                    evaluator_id=cons_dict["evaluator_id"],
                    constraint_type=cons_dict["type"],
                    value=float(cons_dict["value"])
                ))

        # Parse equality constraints
        eq_constraints_objs = []
        if equality_constraints:
            for cons_dict in equality_constraints:
                # Verify constraint evaluator exists
                cons_eval_id = cons_dict["evaluator_id"]
                try:
                    foundry.get_evaluator_config(cons_eval_id)
                except KeyError:
                    return {
                        "success": False,
                        "message": f"Constraint evaluator '{cons_eval_id}' not found in Foundry."
                    }

                eq_constraints_objs.append(EqualityConstraint(
                    name=cons_dict["name"],
                    evaluator_id=cons_dict["evaluator_id"],
                    value=float(cons_dict["value"]),
                    tolerance=cons_dict.get("tolerance", 1e-6)
                ))

        # Parse bounds - support both compact format (dict) and explicit list
        from paola.foundry.bounds_spec import parse_bounds_input

        if isinstance(bounds, dict):
            # Compact format: {"type": "uniform", "lower": -5, "upper": 10, "dimension": 50}
            bounds_spec = parse_bounds_input(bounds)
            explicit_bounds = bounds_spec.expand()
            dimension = bounds_spec.get_dimension()
        elif isinstance(bounds, list):
            # Explicit format: [[-5, 10], [-5, 10], ...]
            explicit_bounds = bounds
            dimension = len(bounds)
        else:
            return {
                "success": False,
                "message": (
                    f"Invalid bounds format. Expected dict (compact) or list (explicit).\n"
                    f"Compact format: {{'type': 'uniform', 'lower': -5, 'upper': 10, 'dimension': 50}}\n"
                    f"Explicit format: [[-5, 10], [-5, 10], ...]"
                )
            }

        # Create NLPProblem specification (v0.4.3 - uses numeric IDs)
        nlp_problem = NLPProblem(
            problem_id=problem_id,
            name=name,
            n_variables=dimension,
            n_constraints=0,
            objective_evaluator_id=objective_evaluator_id,
            objective_sense=objective_sense,
            bounds=explicit_bounds,
            inequality_constraints=ineq_constraints_objs,
            equality_constraints=eq_constraints_objs,
            description=description or name,
            domain_hint=domain_hint,
            bounds_spec=bounds if isinstance(bounds, dict) else None
        )

        # Create NLPEvaluator (composite evaluator)
        nlp_evaluator = NLPEvaluator.from_problem(nlp_problem, foundry)

        # Register problem atomically (storage + cache) - v0.4.6 single source of truth
        foundry.register_problem_evaluator(nlp_problem, nlp_evaluator)

        # Recommend solvers
        has_constraints = nlp_problem.is_constrained
        recommended_solvers = SolverSelector.recommend_solver(
            problem_type="NLP",
            gradient_available=True,
            has_constraints=has_constraints
        )

        # Build message with domain hint if provided
        hint_msg = f"  Domain hint: {domain_hint}\n" if domain_hint else ""

        return {
            "success": True,
            "problem_id": problem_id,
            "name": name,
            "problem_type": "NLP",
            "dimension": dimension,
            "num_inequality_constraints": nlp_problem.num_inequality_constraints,
            "num_equality_constraints": nlp_problem.num_equality_constraints,
            "evaluators_used": nlp_problem.get_all_evaluator_ids(),
            "recommended_solvers": recommended_solvers,
            "domain_hint": domain_hint,
            "message": (
                f"Created NLP problem #{problem_id} '{name}':\n"
                f"  Objective: {objective_sense} {objective_evaluator_id}\n"
                f"  Dimension: {dimension}\n"
                f"{hint_msg}"
                f"  Inequality constraints: {nlp_problem.num_inequality_constraints}\n"
                f"  Equality constraints: {nlp_problem.num_equality_constraints}\n"
                f"  Recommended solvers: {', '.join(recommended_solvers[:2])}\n"
                f"  Note: Initial point will be computed by Paola based on domain and algorithm"
            )
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "message": f"Error creating NLP problem: {str(e)}\n{traceback.format_exc()}",
        }


@tool(args_schema=DeriveProblemArgs)
def derive_problem(
    parent_problem_id: ProblemIdType,
    derivation_type: str,
    modifications: str,  # JSON string
    new_name: Optional[str] = None,
    reason: Optional[str] = None,
    source_graph_id: Optional[int] = None,
    source_node_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Derive new problem from existing one.

    Args:
        parent_problem_id: Problem ID to derive from
        derivation_type: "narrow_bounds" or "widen_bounds"
        modifications: JSON with parameters, e.g. '{"center": [...], "width_factor": 0.3}'
        new_name: Name for derived problem
        reason: Why this derivation
        source_graph_id: Graph that motivated derivation
        source_node_id: Node that motivated derivation

    Returns:
        success, problem_id, parent_problem_id

    Example:
        derive_problem(1, "narrow_bounds", '{"center": [1.2, 3.4], "width_factor": 0.3}')
    """
    try:
        # Normalize problem_id (handles str/int from LLM)
        parent_problem_id = normalize_problem_id(parent_problem_id)
        import json as json_module
        from paola.foundry.nlp_schema import NLPProblem
        from paola.foundry.nlp_evaluator import NLPEvaluator
        from paola.tools.graph_tools import get_foundry

        # Use global Foundry (single source of truth - v0.4.6)
        foundry = get_foundry()
        if foundry is None:
            return {
                "success": False,
                "message": "Foundry not initialized. CLI should call set_foundry() at startup.",
            }

        # Parse modifications
        try:
            mods = json_module.loads(modifications)
        except json_module.JSONDecodeError as e:
            return {
                "success": False,
                "message": f"Invalid JSON in modifications: {e}"
            }

        # Load parent problem from Foundry storage
        parent = foundry.storage.load_problem(parent_problem_id)

        if parent is None:
            return {
                "success": False,
                "message": f"Parent problem #{parent_problem_id} not found in storage."
            }

        # Check that it's an NLPProblem
        if not isinstance(parent, NLPProblem):
            return {
                "success": False,
                "message": f"Parent problem #{parent_problem_id} is not an NLPProblem. "
                           f"Derivation only supported for NLP problems currently."
            }

        # Get next numeric ID for derived problem (v0.4.3)
        new_problem_id = foundry.storage.get_next_problem_id()

        # Generate name if not provided
        derived_name = new_name or f"{parent.name} ({derivation_type})"

        # Apply derivation
        if derivation_type == "narrow_bounds":
            center = mods.get("center")
            if center is None:
                return {
                    "success": False,
                    "message": "narrow_bounds derivation requires 'center' in modifications"
                }
            width_factor = mods.get("width_factor", 0.3)

            derived = parent.derive_narrow_bounds(
                new_problem_id=new_problem_id,
                new_name=derived_name,
                center=center,
                width_factor=width_factor,
                reason=reason,
                source_graph_id=source_graph_id,
                source_node_id=source_node_id,
            )

        elif derivation_type == "widen_bounds":
            width_factor = mods.get("width_factor", 1.5)

            derived = parent.derive_widen_bounds(
                new_problem_id=new_problem_id,
                new_name=derived_name,
                width_factor=width_factor,
                reason=reason,
            )

        else:
            return {
                "success": False,
                "message": f"Unknown derivation_type: {derivation_type}. "
                           f"Supported: narrow_bounds, widen_bounds"
            }

        # Create NLPEvaluator and register atomically (v0.4.6 single source of truth)
        nlp_evaluator = NLPEvaluator.from_problem(derived, foundry)
        foundry.register_problem_evaluator(derived, nlp_evaluator)

        return {
            "success": True,
            "problem_id": derived.problem_id,
            "name": derived.name,
            "parent_problem_id": derived.parent_problem_id,
            "derivation_type": derived.derivation_type,
            "n_variables": derived.n_variables,
            "version": derived.version,
            "message": (
                f"Derived problem #{derived.problem_id} '{derived.name}' from #{parent_problem_id}:\n"
                f"  Derivation: {derivation_type}\n"
                f"  Variables: {derived.n_variables}\n"
                f"  Version: {derived.version}\n"
                f"  Reason: {reason or 'Not specified'}"
            )
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "message": f"Error deriving problem: {str(e)}\n{traceback.format_exc()}",
        }


@tool
def list_problems(
    problem_type: Optional[str] = None,
    show_derived: bool = True,
) -> Dict[str, Any]:
    """
    List all registered optimization problems.

    Args:
        problem_type: Filter by type ("NLP", "LP", etc.). None for all.
        show_derived: Include derived problems (default: True)

    Returns:
        success, problems, count

    Example:
        list_problems(problem_type="NLP")
    """
    try:
        from paola.tools.graph_tools import get_foundry

        foundry = get_foundry()
        if foundry is None:
            return {
                "success": False,
                "message": "Foundry not initialized. CLI should call set_foundry() at startup.",
            }

        problems = foundry.storage.list_problems(
            problem_type=problem_type,
            show_derived=show_derived
        )

        return {
            "success": True,
            "problems": problems,
            "count": len(problems),
            "message": f"Found {len(problems)} problem(s)"
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "message": f"Error listing problems: {str(e)}\n{traceback.format_exc()}",
        }


@tool
def get_problem_lineage(problem_id: int) -> Dict[str, Any]:
    """
    Get the derivation lineage of a problem.

    Args:
        problem_id: Numeric problem ID to trace lineage for

    Returns:
        success, problem_id, lineage, children

    Example:
        get_problem_lineage(3)
    """
    try:
        from paola.tools.graph_tools import get_foundry

        foundry = get_foundry()
        if foundry is None:
            return {
                "success": False,
                "message": "Foundry not initialized. CLI should call set_foundry() at startup.",
            }

        # Get lineage
        lineage = foundry.storage.get_problem_lineage(problem_id)
        if not lineage:
            return {
                "success": False,
                "message": f"Problem #{problem_id} not found in storage."
            }

        # Get children
        children = foundry.storage.get_problem_children(problem_id)

        # Build chain string
        chain_parts = [f"#{p['problem_id']}" for p in lineage]
        chain_str = ' -> '.join(chain_parts)

        return {
            "success": True,
            "problem_id": problem_id,
            "lineage": lineage,
            "children": children,
            "message": (
                f"Lineage for problem #{problem_id}:\n"
                f"  Chain: {chain_str}\n"
                f"  Children: {children if children else 'None'}"
            )
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "message": f"Error getting lineage: {str(e)}\n{traceback.format_exc()}",
        }
