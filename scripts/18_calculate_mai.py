"""
TREE GROWTH ANALYTICS
STEP 12A - MAI CALCULATION

Purpose:
- Load the prediction engine from STEP 11
- Generate RF and PINNs predictions
- Calculate MAI from predicted volume
- Do NOT calculate THA yet

MAI definition:
    MAI = predicted_volume_m3_tree / age_years

where:
    age_years = age_months / 12

Unit:
    m3/tree/year
"""

from pathlib import Path
import sys
import importlib.util


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT PREDICTION ENGINE
# ============================================================

PREDICTION_ENGINE_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "17_prediction_engine.py"
)

spec = importlib.util.spec_from_file_location(
    "prediction_engine",
    PREDICTION_ENGINE_PATH,
)

if spec is None or spec.loader is None:
    raise ImportError(
        f"Cannot load prediction engine from: "
        f"{PREDICTION_ENGINE_PATH}"
    )

prediction_engine = importlib.util.module_from_spec(spec)

spec.loader.exec_module(prediction_engine)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES = [
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha",
]


# ============================================================
# TEST INPUT
# ============================================================

# Same smoke-test input used in STEP 11.
# This input is only for validating the calculation engine.

TEST_INPUT = {
    "age_months": 60.0,
    "inventory_year": 2020.0,
    "clone_factor": 1.0,
    "planting_year": 2015.0,
    "disease_score": 0.10,
    "trees_per_ha": 1300.0,
}


# ============================================================
# MAI CALCULATION
# ============================================================

def calculate_mai(
    predicted_volume_m3_tree: float,
    age_months: float,
) -> float:

    if age_months <= 0:
        raise ValueError(
            "age_months must be greater than 0 "
            "to calculate MAI."
        )

    age_years = age_months / 12.0

    mai = (
        predicted_volume_m3_tree
        / age_years
    )

    return float(mai)


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_mai_input(data: dict) -> None:

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in data
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    if data["age_months"] <= 0:
        raise ValueError(
            "age_months must be greater than 0."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("TREE GROWTH ANALYTICS")
    print("STEP 12A - MAI CALCULATION")
    print("=" * 75)

    # --------------------------------------------------------
    # 1. Validate input
    # --------------------------------------------------------

    print("\n[1/4] Validating input...")

    validate_mai_input(
        TEST_INPUT
    )

    print("  Features : OK")
    print("  Age      : OK")

    # --------------------------------------------------------
    # 2. Load models
    # --------------------------------------------------------

    print("\n[2/4] Loading prediction models...")

    rf_models = (
        prediction_engine
        .load_random_forest_models()
    )

    pinns_model, pinns_checkpoint = (
        prediction_engine
        .load_pinns_model()
    )

    print("  Random Forest : LOADED")
    print("  PINNs         : LOADED")

    # --------------------------------------------------------
    # 3. Run prediction
    # --------------------------------------------------------

    print("\n[3/4] Running prediction...")

    # Convert the input dictionary into
    # the 2D feature array expected by
    # the prediction functions.

    X = (
        prediction_engine
        .validate_input(
            TEST_INPUT
        )
    )

    rf_prediction = (
        prediction_engine
        .predict_random_forest(
            rf_models,
            X,
        )
    )

    pinns_prediction = (
        prediction_engine
        .predict_pinns(
            pinns_model,
            pinns_checkpoint,
            X,
        )
    )

    # --------------------------------------------------------
    # 4. Calculate MAI
    # --------------------------------------------------------

    print("\n[4/4] Calculating MAI...")

    rf_mai = calculate_mai(
        predicted_volume_m3_tree=(
            rf_prediction[
                "volume_m3_tree"
            ]
        ),
        age_months=(
            TEST_INPUT[
                "age_months"
            ]
        ),
    )

    pinns_mai = calculate_mai(
        predicted_volume_m3_tree=(
            pinns_prediction[
                "volume_m3_tree"
            ]
        ),
        age_months=(
            TEST_INPUT[
                "age_months"
            ]
        ),
    )

    age_years = (
        TEST_INPUT["age_months"]
        / 12.0
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("MAI CALCULATION RESULT")
    print("=" * 75)

    print("\nINPUT")

    print(
        f"  age_months          : "
        f"{TEST_INPUT['age_months']:.1f}"
    )

    print(
        f"  age_years           : "
        f"{age_years:.2f}"
    )

    print(
        f"  inventory_year      : "
        f"{TEST_INPUT['inventory_year']:.1f}"
    )

    print(
        f"  clone_factor        : "
        f"{TEST_INPUT['clone_factor']:.4f}"
    )

    print(
        f"  planting_year       : "
        f"{TEST_INPUT['planting_year']:.1f}"
    )

    print(
        f"  disease_score       : "
        f"{TEST_INPUT['disease_score']:.4f}"
    )

    print(
        f"  trees_per_ha        : "
        f"{TEST_INPUT['trees_per_ha']:.1f}"
    )

    # --------------------------------------------------------
    # RANDOM FOREST RESULT
    # --------------------------------------------------------

    print("\nRANDOM FOREST")

    print(
        f"  volume_m3_tree     : "
        f"{rf_prediction['volume_m3_tree']:.6f}"
    )

    print(
        f"  MAI_m3_tree_year   : "
        f"{rf_mai:.6f}"
    )

    # --------------------------------------------------------
    # PINNs RESULT
    # --------------------------------------------------------

    print("\nPINNs")

    print(
        f"  volume_m3_tree     : "
        f"{pinns_prediction['volume_m3_tree']:.6f}"
    )

    print(
        f"  MAI_m3_tree_year   : "
        f"{pinns_mai:.6f}"
    )

    # --------------------------------------------------------
    # BASIC SANITY CHECK
    # --------------------------------------------------------

    if rf_mai < 0:
        raise ValueError(
            "RF MAI is negative."
        )

    if pinns_mai < 0:
        raise ValueError(
            "PINNs MAI is negative."
        )

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("STEP 12A TEST COMPLETED")
    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()