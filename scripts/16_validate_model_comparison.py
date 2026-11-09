from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# TREE GROWTH ANALYTICS
# STEP 10 - VALIDATE MODEL COMPARISON
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

METRICS_PATH = (
    RESULTS_DIR
    / "model_comparison_metrics.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIR
    / "model_comparison_predictions.csv"
)


EXPECTED_TARGETS = [
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]


EXPECTED_METRIC_COLUMNS = [
    "target",
    "rf_MAE",
    "pinns_MAE",
    "rf_RMSE",
    "pinns_RMSE",
    "rf_R2",
    "pinns_R2",
    "rf_MAPE_percent",
    "pinns_MAPE_percent",
]


EXPECTED_PREDICTION_COLUMNS = [
    "compartment_id",
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha",

    "actual_dbh_cm",
    "actual_height_m",
    "actual_volume_m3_tree",

    "predicted_rf_dbh_cm",
    "predicted_rf_height_m",
    "predicted_rf_volume_m3_tree",

    "predicted_pinns_dbh_cm",
    "predicted_pinns_height_m",
    "predicted_pinns_volume_m3_tree",

    "difference_dbh_rf_minus_pinns",
    "difference_height_rf_minus_pinns",
    "difference_volume_rf_minus_pinns",
]


# ============================================================
# START
# ============================================================

print("=" * 75)
print("TREE GROWTH ANALYTICS")
print("STEP 10 - VALIDATE MODEL COMPARISON")
print("=" * 75)


# ============================================================
# 1. LOAD FILES
# ============================================================

print("\n[1/6] Loading comparison files...")

metrics = pd.read_csv(
    METRICS_PATH
)

predictions = pd.read_csv(
    PREDICTIONS_PATH
)

print(
    f"Metrics rows      : "
    f"{len(metrics):,}"
)

print(
    f"Prediction rows   : "
    f"{len(predictions):,}"
)

print(
    f"Prediction columns: "
    f"{len(predictions.columns)}"
)


# ============================================================
# 2. VALIDATE METRICS STRUCTURE
# ============================================================

print("\n[2/6] Validating metrics structure...")

missing_metric_columns = [
    column
    for column in EXPECTED_METRIC_COLUMNS
    if column not in metrics.columns
]

if missing_metric_columns:

    raise ValueError(
        "Missing metric columns: "
        f"{missing_metric_columns}"
    )


if set(metrics["target"]) != set(
    EXPECTED_TARGETS
):

    raise ValueError(
        "Target list pada comparison "
        "tidak sesuai."
    )


if len(metrics) != 3:

    raise ValueError(
        "Comparison metrics harus memiliki "
        "tepat 3 target."
    )


print("Metric columns : OK")
print("Metric targets : OK")
print("Metric rows    : OK")


# ============================================================
# 3. VALIDATE PREDICTION STRUCTURE
# ============================================================

print("\n[3/6] Validating prediction structure...")

missing_prediction_columns = [
    column
    for column in EXPECTED_PREDICTION_COLUMNS
    if column not in predictions.columns
]

if missing_prediction_columns:

    raise ValueError(
        "Missing prediction columns: "
        f"{missing_prediction_columns}"
    )


if len(predictions) != 121498:

    raise ValueError(
        "Jumlah prediction rows tidak sesuai. "
        f"Expected=121,498, "
        f"Actual={len(predictions):,}"
    )


print("Prediction columns : OK")
print("Prediction rows    : OK")


# ============================================================
# 4. VALIDATE NUMERICAL VALUES
# ============================================================

print("\n[4/6] Validating numerical values...")

numeric_columns = [
    column
    for column in EXPECTED_PREDICTION_COLUMNS
    if column != "compartment_id"
]


for column in numeric_columns:

    if not pd.api.types.is_numeric_dtype(
        predictions[column]
    ):

        raise ValueError(
            f"Column {column} bukan numeric."
        )


    if not np.isfinite(
        predictions[column].to_numpy()
    ).all():

        raise ValueError(
            f"Column {column} "
            "mengandung NaN atau infinite."
        )


print("Numeric values : OK")
print("NaN / Inf      : NONE")


# ============================================================
# 5. VALIDATE RF - PINNs DIFFERENCE
# ============================================================

print("\n[5/6] Validating model differences...")

dbh_difference = (
    predictions["predicted_rf_dbh_cm"]
    -
    predictions["predicted_pinns_dbh_cm"]
)

height_difference = (
    predictions["predicted_rf_height_m"]
    -
    predictions["predicted_pinns_height_m"]
)

volume_difference = (
    predictions["predicted_rf_volume_m3_tree"]
    -
    predictions["predicted_pinns_volume_m3_tree"]
)


if not np.allclose(
    predictions[
        "difference_dbh_rf_minus_pinns"
    ],
    dbh_difference,
    rtol=1e-6,
    atol=1e-6,
):

    raise ValueError(
        "Difference DBH tidak sesuai."
    )


if not np.allclose(
    predictions[
        "difference_height_rf_minus_pinns"
    ],
    height_difference,
    rtol=1e-6,
    atol=1e-6,
):

    raise ValueError(
        "Difference Height tidak sesuai."
    )


if not np.allclose(
    predictions[
        "difference_volume_rf_minus_pinns"
    ],
    volume_difference,
    rtol=1e-6,
    atol=1e-6,
):

    raise ValueError(
        "Difference Volume tidak sesuai."
    )


print("DBH difference    : OK")
print("Height difference : OK")
print("Volume difference : OK")


# ============================================================
# 6. SUMMARY
# ============================================================

print("\n[6/6] Validation summary...")

print("\nMODEL COMPARISON METRICS")
print("=" * 75)

for _, row in metrics.iterrows():

    print(
        f"\nTarget: {row['target']}"
    )

    print(
        f"  RF    -> "
        f"MAE={row['rf_MAE']:.6f} | "
        f"RMSE={row['rf_RMSE']:.6f} | "
        f"R2={row['rf_R2']:.6f} | "
        f"MAPE={row['rf_MAPE_percent']:.4f}%"
    )

    print(
        f"  PINNs -> "
        f"MAE={row['pinns_MAE']:.6f} | "
        f"RMSE={row['pinns_RMSE']:.6f} | "
        f"R2={row['pinns_R2']:.6f} | "
        f"MAPE={row['pinns_MAPE_percent']:.4f}%"
    )


print("\n" + "=" * 75)
print("STEP 10 VALIDATION COMPLETED")
print("=" * 75)

print("\nValidated files:")

print(
    "  - results/model_comparison_metrics.csv"
)

print(
    "  - results/model_comparison_predictions.csv"
)

print("\nStatus : READY FOR DASHBOARD")
print("=" * 75)