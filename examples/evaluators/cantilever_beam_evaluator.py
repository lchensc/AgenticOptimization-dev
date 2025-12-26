"""
Cantilever Beam Design Evaluator
================================

This file defines a structural optimization problem for use with PAOLA.

To use with PAOLA:
    1. Start PAOLA CLI: python -m paola.cli
    2. Register this evaluator: /register_eval examples/evaluators/cantilever_beam_evaluator.py
    3. Ask PAOLA to solve: "Design a beam that minimizes weight and supports 1000N load"

The evaluator computes beam mass and constraint violations given dimensions.
PAOLA will formulate the optimization problem and select appropriate solvers.
"""

import numpy as np

# =============================================================================
# MATERIAL PROPERTIES (Aluminum 6061-T6)
# =============================================================================

MATERIAL_NAME = "Aluminum 6061-T6"
DENSITY = 2700.0        # kg/m^3
YIELD_STRESS = 276e6    # Pa (276 MPa)
ELASTIC_MODULUS = 68.9e9  # Pa (68.9 GPa)


# =============================================================================
# DESIGN REQUIREMENTS
# =============================================================================

APPLIED_LOAD = 1000.0    # N (force at tip)
BEAM_LENGTH = 1.0        # m
MAX_DEFLECTION = 0.01    # m (10 mm max tip deflection)
SAFETY_FACTOR = 1.5      # stress must be below yield/SF


# =============================================================================
# PROBLEM SPECIFICATION
# =============================================================================

# Design variables: [width, height] in meters
BOUNDS = [
    [0.010, 0.100],  # width: 10mm to 100mm
    [0.010, 0.200],  # height: 10mm to 200mm
]

DIMENSION = 2

DESCRIPTION = """
Cantilever Beam Design Problem
------------------------------
Objective: Minimize beam mass
Variables:
  - width (b): 10-100 mm
  - height (h): 10-200 mm
Constraints:
  - Bending stress <= Yield stress / Safety factor
  - Tip deflection <= 10 mm
Material: Aluminum 6061-T6
Loading: 1000N point load at tip
"""


# =============================================================================
# PHYSICS CALCULATIONS
# =============================================================================

def compute_beam_physics(width: float, height: float) -> dict:
    """
    Compute beam physics given cross-section dimensions.

    Beam theory for rectangular cross-section cantilever:
    - Second moment of area: I = b * h^3 / 12
    - Bending moment at root: M = P * L
    - Max stress: sigma = M * (h/2) / I
    - Max deflection: delta = P * L^3 / (3 * E * I)
    """
    L = BEAM_LENGTH
    P = APPLIED_LOAD
    E = ELASTIC_MODULUS
    rho = DENSITY

    # Second moment of area
    I = (width * height**3) / 12.0

    # Volume and mass
    volume = width * height * L
    mass = volume * rho

    # Bending stress at root (maximum)
    M = P * L  # bending moment at fixed end
    if I > 0:
        stress = M * (height / 2.0) / I
    else:
        stress = float('inf')

    # Tip deflection
    if I > 0 and E > 0:
        deflection = (P * L**3) / (3.0 * E * I)
    else:
        deflection = float('inf')

    return {
        "mass": mass,
        "stress": stress,
        "deflection": deflection,
        "second_moment": I,
        "volume": volume,
    }


# =============================================================================
# EVALUATOR FUNCTION (what PAOLA calls)
# =============================================================================

def evaluate(design: np.ndarray) -> float:
    """
    Evaluate cantilever beam design.

    This function is called by PAOLA's optimizer during optimization.

    Args:
        design: Array [width, height] in meters

    Returns:
        Mass in kg (with constraint penalties if violated)
    """
    width = float(design[0])
    height = float(design[1])

    # Compute physics
    physics = compute_beam_physics(width, height)

    # Base objective: mass
    objective = physics["mass"]

    # Constraint 1: Stress must be below allowable
    allowable_stress = YIELD_STRESS / SAFETY_FACTOR
    stress_violation = max(0, physics["stress"] - allowable_stress)
    if stress_violation > 0:
        # Quadratic penalty
        objective += 100.0 * (stress_violation / allowable_stress) ** 2

    # Constraint 2: Deflection must be below maximum
    deflection_violation = max(0, physics["deflection"] - MAX_DEFLECTION)
    if deflection_violation > 0:
        # Quadratic penalty
        objective += 100.0 * (deflection_violation / MAX_DEFLECTION) ** 2

    return objective


def get_metrics(design: np.ndarray) -> dict:
    """
    Get detailed design metrics for interpretation.

    This is a helper function to understand the solution.
    Not called during optimization.
    """
    width = float(design[0])
    height = float(design[1])
    physics = compute_beam_physics(width, height)

    allowable_stress = YIELD_STRESS / SAFETY_FACTOR

    return {
        "width_mm": width * 1000,
        "height_mm": height * 1000,
        "mass_kg": physics["mass"],
        "stress_mpa": physics["stress"] / 1e6,
        "allowable_stress_mpa": allowable_stress / 1e6,
        "stress_ratio": physics["stress"] / allowable_stress,
        "deflection_mm": physics["deflection"] * 1000,
        "max_deflection_mm": MAX_DEFLECTION * 1000,
        "deflection_ratio": physics["deflection"] / MAX_DEFLECTION,
        "stress_ok": physics["stress"] <= allowable_stress,
        "deflection_ok": physics["deflection"] <= MAX_DEFLECTION,
        "feasible": (physics["stress"] <= allowable_stress and
                     physics["deflection"] <= MAX_DEFLECTION),
    }


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("Cantilever Beam Evaluator Test")
    print("=" * 50)
    print(f"Material: {MATERIAL_NAME}")
    print(f"Load: {APPLIED_LOAD} N at tip")
    print(f"Length: {BEAM_LENGTH} m")

    # Test with initial guess (center of bounds)
    initial = np.array([0.055, 0.105])  # center of bounds
    obj = evaluate(initial)
    metrics = get_metrics(initial)

    print(f"\nInitial design (center of bounds):")
    print(f"  Width:  {metrics['width_mm']:.1f} mm")
    print(f"  Height: {metrics['height_mm']:.1f} mm")
    print(f"  Mass:   {metrics['mass_kg']:.3f} kg")
    print(f"  Stress: {metrics['stress_mpa']:.1f} MPa ({metrics['stress_ratio']*100:.1f}% of allowable)")
    print(f"  Deflection: {metrics['deflection_mm']:.2f} mm ({metrics['deflection_ratio']*100:.1f}% of max)")
    print(f"  Feasible: {metrics['feasible']}")

    # Test with minimal design (likely infeasible)
    minimal = np.array([0.010, 0.010])  # smallest allowed
    obj_min = evaluate(minimal)
    metrics_min = get_metrics(minimal)

    print(f"\nMinimal design (10mm x 10mm):")
    print(f"  Mass:   {metrics_min['mass_kg']:.4f} kg")
    print(f"  Stress: {metrics_min['stress_mpa']:.1f} MPa (allowable: {metrics_min['allowable_stress_mpa']:.1f})")
    print(f"  Deflection: {metrics_min['deflection_mm']:.2f} mm (max: {metrics_min['max_deflection_mm']:.1f})")
    print(f"  Feasible: {metrics_min['feasible']}")
    print(f"  Objective with penalties: {obj_min:.3f}")
