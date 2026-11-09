from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# TREE GROWTH ANALYTICS
# STEP 9 - RANDOM FOREST VS PINNs COMPARISON
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)


RF_METRICS_PATH = (
    RESULTS_DIR
    / "random_forest_metrics.csv"
)

PINNS_METRICS_PATH = (
    RESULTS_DIR
    / "pinns_metrics.csv"
)

RF_PREDICTIONS_PATH = (
    RESULTS_DIR
    / "random_forest_predictions.csv"
)

PINNS_PREDICTIONS_PATH = (
    RESULTS_DIR
    / "pinns_predictions.csv"
)


COMPARISON_METRICS_PATH = (
    RESULTS_DIR
    / "model_comparison_metrics.csv"
)

COMPARISON_PREDICTIONS_PATH = (
    RESULTS_DIR
    / "model_comparison_predictions.csv"
)


TARGETS = [
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]


# ============================================================
# START
# ============================================================

print("=" * 75)
print("TREE GROWTH ANALYTICS")
print("STEP 9 - RANDOM FOREST VS PINNs")
print("=" * 75)


# ============================================================
# 1. LOAD METRICS
# ============================================================

print("\n[1/5] Loading model metrics...")

rf_metrics = pd.read_csv(
    RF_METRICS_PATH
)

pinns_metrics = pd.read_csv(
    PINNS_METRICS_PATH
)

print(
    f"Random Forest metric rows : "
    f"{len(rf_metrics)}"
)

print(
    f"PINNs metric rows          : "
    f"{len(pinns_metrics)}"
)


# ============================================================
# 2. VALIDATE METRICS
# ============================================================

print("\n[2/5] Validating metrics...")

required_metric_columns = {
    "model",
    "target",
    "MAE",
    "RMSE",
    "R2",
    "MAPE_percent",
}


missing_rf = (
    required_metric_columns
    - set(rf_metrics.columns)
)

missing_pinns = (
    required_metric_columns
    - set(pinns_metrics.columns)
)


if missing_rf:
    raise ValueError(
        "Random Forest metrics missing columns: "
        f"{sorted(missing_rf)}"
    )


if missing_pinns:
    raise ValueError(
        "PINNs metrics missing columns: "
        f"{sorted(missing_pinns)}"
    )


rf_targets = set(
    rf_metrics["target"]
)

pinns_targets = set(
    pinns_metrics["target"]
)


if rf_targets != set(TARGETS):

    raise ValueError(
        "Random Forest targets tidak sesuai."
    )


if pinns_targets != set(TARGETS):

    raise ValueError(
        "PINNs targets tidak sesuai."
    )


print("Metric structure : OK")
print("Targets          : OK")


# ============================================================
# 3. CREATE COMPARISON TABLE
# ============================================================

print("\n[3/5] Creating model comparison table...")


comparison_rows = []


for target in TARGETS:

    rf_row = rf_metrics[
        rf_metrics["target"]
        == target
    ].iloc[0]

    pinns_row = pinns_metrics[
        pinns_metrics["target"]
        == target
    ].iloc[0]


    comparison_rows.append(
    {
        "target": target,

        "rf_MAE":
            rf_row["MAE"],

        "pinns_MAE":
            pinns_row["MAE"],

        "rf_RMSE":
            rf_row["RMSE"],

        "pinns_RMSE":
            pinns_row["RMSE"],

        "rf_R2":
            rf_row["R2"],

        "pinns_R2":
            pinns_row["R2"],

        "rf_MAPE_percent":
            rf_row["MAPE_percent"],

        "pinns_MAPE_percent":
            pinns_row["MAPE_percent"],
    }
)


comparison_df = pd.DataFrame(
    comparison_rows
)


# ============================================================
# 4. VALIDATE TEST ROW COUNT
# ============================================================

print("\n[4/5] Validating test set consistency...")


for _, row in comparison_df.iterrows():

    if (print(
    "Metric comparison : READY"
)
    ):

        raise ValueError(
            f"Test row mismatch for "
            f"{row['target']}: "
            f"RF={row['test_rows_rf']} "
            f"PINNs={row['test_rows_pinns']}"
        )




# ============================================================
# 5. SAVE COMPARISON
# ============================================================

comparison_df.to_csv(
    COMPARISON_METRICS_PATH,
    index=False,
)


print("\n[5/5] Saving comparison...")

print(
    "\nMODEL COMPARISON"
)

print("=" * 75)


for _, row in comparison_df.iterrows():

    print(
        f"\nTarget : {row['target']}"
    )

    print(
        f"  MAE"
    )

    print(
        f"    Random Forest : "
        f"{row['rf_MAE']:.6f}"
    )

    print(
        f"    PINNs         : "
        f"{row['pinns_MAE']:.6f}"
    )

    print(
        f"  RMSE"
    )

    print(
        f"    Random Forest : "
        f"{row['rf_RMSE']:.6f}"
    )

    print(
        f"    PINNs         : "
        f"{row['pinns_RMSE']:.6f}"
    )

    print(
        f"  R²"
    )

    print(
        f"    Random Forest : "
        f"{row['rf_R2']:.6f}"
    )

    print(
        f"    PINNs         : "
        f"{row['pinns_R2']:.6f}"
    )

    print(
        f"  MAPE"
    )

    print(
        f"    Random Forest : "
        f"{row['rf_MAPE_percent']:.4f}%"
    )

    print(
        f"    PINNs         : "
        f"{row['pinns_MAPE_percent']:.4f}%"
    )


# ============================================================
# PREDICTION COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("Preparing prediction comparison...")
print("=" * 75)


rf_predictions = pd.read_csv(
    RF_PREDICTIONS_PATH
)

pinns_predictions = pd.read_csv(
    PINNS_PREDICTIONS_PATH
)


print(
    f"RF prediction rows    : "
    f"{len(rf_predictions):,}"
)

print(
    f"PINNs prediction rows : "
    f"{len(pinns_predictions):,}"
)


if len(rf_predictions) != len(
    pinns_predictions
):

    raise ValueError(
        "Jumlah prediction rows RF "
        "dan PINNs berbeda."
    )


# ------------------------------------------------------------
# IMPORTANT:
# Both prediction files originate from the SAME model_test.csv.
#
# We therefore preserve the row order and validate actual
# values before combining predictions.
# ------------------------------------------------------------

actual_pairs = [
    (
        "actual_dbh_cm",
        "actual_dbh_cm",
    ),
    (
        "actual_height_m",
        "actual_height_m",
    ),
    (
        "actual_volume_m3_tree",
        "actual_volume_m3_tree",
    ),
]


for rf_column, pinns_column in actual_pairs:

    if rf_column not in rf_predictions.columns:

        raise ValueError(
            f"Column {rf_column} "
            "tidak ditemukan pada RF predictions."
        )

    if pinns_column not in pinns_predictions.columns:

        raise ValueError(
            f"Column {pinns_column} "
            "tidak ditemukan pada PINNs predictions."
        )


    same_values = np.allclose(
        rf_predictions[rf_column].values,
        pinns_predictions[pinns_column].values,
        rtol=1e-6,
        atol=1e-6,
    )


    if not same_values:

        raise ValueError(
            f"Actual values berbeda untuk "
            f"{rf_column}."
        )


print(
    "Actual test values RF vs PINNs : MATCH"
)


# ============================================================
# CREATE COMBINED PREDICTION TABLE
# ============================================================

prediction_comparison = pd.DataFrame()


# ------------------------------------------------------------
# Preserve identifiers available in PINNs prediction output
# ------------------------------------------------------------

base_columns = [
    "compartment_id",
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha",
]


for column in base_columns:

    if column in pinns_predictions.columns:

        prediction_comparison[
            column
        ] = pinns_predictions[
            column
        ]


# ------------------------------------------------------------
# Actual values
# ------------------------------------------------------------

prediction_comparison[
    "actual_dbh_cm"
] = pinns_predictions[
    "actual_dbh_cm"
]

prediction_comparison[
    "actual_height_m"
] = pinns_predictions[
    "actual_height_m"
]

prediction_comparison[
    "actual_volume_m3_tree"
] = pinns_predictions[
    "actual_volume_m3_tree"
]


# ------------------------------------------------------------
# RF predictions
# ------------------------------------------------------------

rf_prediction_columns = {
    "dbh_cm":
        "predicted_dbh_cm",

    "height_m":
        "predicted_height_m",

    "volume_m3_tree":
        "predicted_volume_m3_tree",
}

for target, source_column in rf_prediction_columns.items():

    if source_column in rf_predictions.columns:

        if target == "dbh_cm":
            output_column = "predicted_rf_dbh_cm"

        elif target == "height_m":
            output_column = "predicted_rf_height_m"

        elif target == "volume_m3_tree":
            output_column = "predicted_rf_volume_m3_tree"

        prediction_comparison[
            output_column
        ] = rf_predictions[
            source_column
        ]

    else:

        raise ValueError(
            f"RF prediction column "
            f"{source_column} tidak ditemukan."
        )


for target, output_column in (
    rf_prediction_columns.items()
):

    if output_column in rf_predictions.columns:

        prediction_comparison[
            output_column
        ] = rf_predictions[
            output_column
        ]

    else:

        raise ValueError(
            f"RF prediction column "
            f"{output_column} tidak ditemukan."
        )


# ------------------------------------------------------------
# PINNs predictions
# ------------------------------------------------------------

pinns_prediction_columns = {
    "dbh_cm":
        "predicted_pinns_dbh_cm",

    "height_m":
        "predicted_pinns_height_m",

    "volume_m3_tree":
        "predicted_pinns_volume_m3_tree",
}


for target, output_column in (
    pinns_prediction_columns.items()
):

    if output_column in pinns_predictions.columns:

        prediction_comparison[
            output_column
        ] = pinns_predictions[
            output_column
        ]

    else:

        raise ValueError(
            f"PINNs prediction column "
            f"{output_column} tidak ditemukan."
        )


# ============================================================
# MODEL DIFFERENCE
# ============================================================

prediction_comparison[
    "difference_dbh_rf_minus_pinns"
] = (
    prediction_comparison[
        "predicted_rf_dbh_cm"
    ]
    -
    prediction_comparison[
        "predicted_pinns_dbh_cm"
    ]
)


prediction_comparison[
    "difference_height_rf_minus_pinns"
] = (
    prediction_comparison[
        "predicted_rf_height_m"
    ]
    -
    prediction_comparison[
        "predicted_pinns_height_m"
    ]
)


prediction_comparison[
    "difference_volume_rf_minus_pinns"
] = (
    prediction_comparison[
        "predicted_rf_volume_m3_tree"
    ]
    -
    prediction_comparison[
        "predicted_pinns_volume_m3_tree"
    ]
)


# ============================================================
# SAVE PREDICTION COMPARISON
# ============================================================

prediction_comparison.to_csv(
    COMPARISON_PREDICTIONS_PATH,
    index=False,
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print(
    "\nPrediction comparison rows : "
    f"{len(prediction_comparison):,}"
)

print(
    "Prediction comparison columns : "
    f"{len(prediction_comparison.columns)}"
)


print("\nSaved:")

print(
    "  - results/model_comparison_metrics.csv"
)

print(
    "  - results/model_comparison_predictions.csv"
)


print("\n" + "=" * 75)
print("STEP 9 COMPLETED")
print("=" * 75)