"""
ML Hyperparameter Tuning Evaluator
==================================

This file defines a hyperparameter optimization problem for use with PAOLA.

To use with PAOLA:
    1. Start PAOLA CLI: python -m paola.cli
    2. Register this evaluator: /register_eval examples/evaluators/ml_hyperparameter_evaluator.py
    3. Ask PAOLA to solve: "Find the best hyperparameters for my Random Forest"

The evaluator trains a model and returns validation accuracy.
PAOLA will formulate the optimization problem and select appropriate solvers.

Note: This uses simulated training for demonstration.
      Replace simulate_training() with actual model training for real use.
"""

import numpy as np

# =============================================================================
# HYPERPARAMETER SPACE
# =============================================================================

# Define the hyperparameter ranges
# PAOLA will learn these from the problem specification or Skills
HYPERPARAMETERS = {
    "learning_rate": {
        "type": "log_uniform",
        "low": 1e-5,
        "high": 1e-1,
        "description": "Learning rate for gradient-based optimization"
    },
    "n_estimators": {
        "type": "int",
        "low": 50,
        "high": 500,
        "description": "Number of trees in the ensemble"
    },
    "max_depth": {
        "type": "int",
        "low": 3,
        "high": 15,
        "description": "Maximum depth of each tree"
    },
    "min_samples_split": {
        "type": "int",
        "low": 2,
        "high": 20,
        "description": "Minimum samples to split a node"
    },
}

# Ground truth optimal (for simulation)
# In real use, this is what we're trying to find
OPTIMAL_HYPERPARAMS = {
    "learning_rate": 0.01,
    "n_estimators": 150,
    "max_depth": 6,
    "min_samples_split": 5,
}


# =============================================================================
# PROBLEM SPECIFICATION
# =============================================================================

# Bounds for normalized search space [0, 1]^4
BOUNDS = [[0.0, 1.0]] * 4

DIMENSION = 4

DESCRIPTION = """
ML Hyperparameter Tuning Problem
--------------------------------
Objective: Maximize validation accuracy (equivalently, minimize negative accuracy)
Variables:
  - learning_rate: 1e-5 to 1e-1 (log scale)
  - n_estimators: 50 to 500 (integer)
  - max_depth: 3 to 15 (integer)
  - min_samples_split: 2 to 20 (integer)

This is a black-box optimization problem suitable for:
  - Bayesian optimization (TPE, GP)
  - Evolutionary algorithms
  - Random search

Recommended optimizer: optuna:TPE (good for mixed-type HPO)
"""


# =============================================================================
# HYPERPARAMETER ENCODING/DECODING
# =============================================================================

def decode_hyperparams(design: np.ndarray) -> dict:
    """
    Decode normalized design [0,1]^4 to actual hyperparameters.
    """
    return {
        "learning_rate": 10 ** (-5 + design[0] * 4),  # log scale: 1e-5 to 1e-1
        "n_estimators": int(50 + design[1] * 450),     # 50 to 500
        "max_depth": int(3 + design[2] * 12),          # 3 to 15
        "min_samples_split": int(2 + design[3] * 18),  # 2 to 20
    }


def encode_hyperparams(params: dict) -> np.ndarray:
    """
    Encode actual hyperparameters to normalized design [0,1]^4.
    """
    return np.array([
        (np.log10(params["learning_rate"]) + 5) / 4,
        (params["n_estimators"] - 50) / 450,
        (params["max_depth"] - 3) / 12,
        (params["min_samples_split"] - 2) / 18,
    ])


# =============================================================================
# MODEL TRAINING SIMULATION
# =============================================================================

def simulate_training(params: dict, noise_std: float = 0.02) -> float:
    """
    Simulate model training and return validation accuracy.

    In a real scenario, replace this with actual model training:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        return model.score(X_val, y_val)

    For demonstration, we simulate with a known optimal.
    """
    # Distance from optimal in each hyperparameter
    lr_error = abs(np.log10(params["learning_rate"]) -
                   np.log10(OPTIMAL_HYPERPARAMS["learning_rate"]))
    n_est_error = abs(params["n_estimators"] -
                      OPTIMAL_HYPERPARAMS["n_estimators"]) / 100
    depth_error = abs(params["max_depth"] -
                      OPTIMAL_HYPERPARAMS["max_depth"]) / 10
    split_error = abs(params["min_samples_split"] -
                      OPTIMAL_HYPERPARAMS["min_samples_split"]) / 10

    # Accuracy decreases with distance from optimal
    base_accuracy = 0.95
    total_error = lr_error + n_est_error + depth_error + split_error
    penalty = 0.1 * total_error

    # Add some noise (realistic)
    noise = np.random.normal(0, noise_std)

    accuracy = max(0.5, min(1.0, base_accuracy - penalty + noise))
    return accuracy


# =============================================================================
# EVALUATOR FUNCTION (what PAOLA calls)
# =============================================================================

def evaluate(design: np.ndarray) -> float:
    """
    Evaluate ML hyperparameters.

    This function is called by PAOLA's optimizer during optimization.

    Args:
        design: Normalized array [0,1]^4 encoding hyperparameters

    Returns:
        Negative validation accuracy (PAOLA minimizes, so we negate to maximize)
    """
    # Decode to actual hyperparameters
    params = decode_hyperparams(design)

    # Train and evaluate (simulated)
    accuracy = simulate_training(params)

    # Return negative accuracy (minimize negative = maximize accuracy)
    return -accuracy


def get_metrics(design: np.ndarray) -> dict:
    """
    Get detailed metrics for interpretation.

    This is a helper function to understand the solution.
    Not called during optimization.
    """
    params = decode_hyperparams(design)
    accuracy = simulate_training(params, noise_std=0)  # No noise for display

    return {
        "hyperparameters": params,
        "validation_accuracy": accuracy,
        "validation_accuracy_pct": accuracy * 100,
        "comparison_to_optimal": {
            k: {"found": params[k], "optimal": OPTIMAL_HYPERPARAMS[k]}
            for k in params
        },
    }


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("ML Hyperparameter Evaluator Test")
    print("=" * 50)
    print(DESCRIPTION)

    # Test with random hyperparameters
    np.random.seed(42)
    random_design = np.random.rand(4)
    obj = evaluate(random_design)
    metrics = get_metrics(random_design)

    print(f"\nRandom hyperparameters:")
    for name, value in metrics["hyperparameters"].items():
        optimal = OPTIMAL_HYPERPARAMS[name]
        print(f"  {name}: {value} (optimal: {optimal})")
    print(f"  Validation Accuracy: {metrics['validation_accuracy_pct']:.2f}%")
    print(f"  Objective (negative acc): {obj:.4f}")

    # Test with optimal hyperparameters
    optimal_design = encode_hyperparams(OPTIMAL_HYPERPARAMS)
    obj_opt = evaluate(optimal_design)
    metrics_opt = get_metrics(optimal_design)

    print(f"\nOptimal hyperparameters:")
    for name, value in metrics_opt["hyperparameters"].items():
        print(f"  {name}: {value}")
    print(f"  Validation Accuracy: {metrics_opt['validation_accuracy_pct']:.2f}%")
    print(f"  Objective (negative acc): {obj_opt:.4f}")

    # Test with default hyperparameters (center of bounds)
    default_design = np.array([0.5, 0.5, 0.5, 0.5])
    obj_default = evaluate(default_design)
    metrics_default = get_metrics(default_design)

    print(f"\nDefault hyperparameters (center of bounds):")
    for name, value in metrics_default["hyperparameters"].items():
        print(f"  {name}: {value}")
    print(f"  Validation Accuracy: {metrics_default['validation_accuracy_pct']:.2f}%")
