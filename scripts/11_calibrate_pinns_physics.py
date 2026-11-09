from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# STEP 7D-2A
# CALIBRATE EMPIRICAL PHYSICS CONSTANT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "results" / "model_train.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
print("STEP 7D-2A - PINNs PHYSICS CALIBRATION")
print("=" * 75)


# ============================================================
# 1. LOAD TRAINING DATA ONLY
# ============================================================

print("\n[1/6] Loading TRAINING data only...")

df = pd.read_csv(TRAIN_PATH)

print(f"Training rows : {len(df):,}")


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

print("\n[2/6] Checking required columns...")

required_columns = [
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print("Required columns are available.")


# ============================================================
# 3. CALCULATE GEOMETRIC PROXY
# ============================================================

print("\n[3/6] Calculating DBH² × Height...")

df["dbh_height_proxy"] = (
    df["dbh_cm"] ** 2
    * df["height_m"]
)

# Only valid positive values
valid = df[
    (df["dbh_height_proxy"] > 0)
    &
    (df["volume_m3_tree"] > 0)
].copy()

print(
    f"Valid rows for calibration : "
    f"{len(valid):,}"
)


# ============================================================
# 4. CALIBRATE C
# ============================================================

print("\n[4/6] Calibrating empirical volume constant...")

# Relationship:
#
# V ≈ C × DBH² × Height
#
# Therefore:
#
# C = V / (DBH² × Height)

valid["C_observed"] = (
    valid["volume_m3_tree"]
    / valid["dbh_height_proxy"]
)


# Robust central estimate
C_median = float(
    valid["C_observed"].median()
)

C_mean = float(
    valid["C_observed"].mean()
)

C_std = float(
    valid["C_observed"].std()
)

C_min = float(
    valid["C_observed"].min()
)

C_max = float(
    valid["C_observed"].max()
)


print("\nEmpirical C statistics:")

print(f"Mean   : {C_mean:.12f}")
print(f"Median : {C_median:.12f}")
print(f"Std    : {C_std:.12f}")
print(f"Min    : {C_min:.12f}")
print(f"Max    : {C_max:.12f}")


# ============================================================
# 5. CALCULATE PHYSICS RECONSTRUCTION ERROR
# ============================================================

print("\n[5/6] Evaluating physics reconstruction on TRAINING data...")

valid["volume_physics_mean"] = (
    C_mean
    * valid["dbh_height_proxy"]
)

valid["volume_physics_median"] = (
    C_median
    * valid["dbh_height_proxy"]
)


def calculate_metrics(
    actual,
    predicted,
):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mae = np.mean(
        np.abs(actual - predicted)
    )

    rmse = np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )

    ss_res = np.sum(
        (actual - predicted) ** 2
    )

    ss_tot = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    r2 = (
        1 - ss_res / ss_tot
        if ss_tot != 0
        else np.nan
    )

    nonzero = actual != 0

    if nonzero.sum() > 0:
        mape = np.mean(
            np.abs(
                (
                    actual[nonzero]
                    - predicted[nonzero]
                )
                / actual[nonzero]
            )
        ) * 100
    else:
        mape = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE_percent": mape,
    }


mean_metrics = calculate_metrics(
    valid["volume_m3_tree"],
    valid["volume_physics_mean"],
)

median_metrics = calculate_metrics(
    valid["volume_m3_tree"],
    valid["volume_physics_median"],
)


print("\nUsing MEAN C:")

print(
    f"MAE  : {mean_metrics['MAE']:.8f}"
)

print(
    f"RMSE : {mean_metrics['RMSE']:.8f}"
)

print(
    f"R²   : {mean_metrics['R2']:.8f}"
)

print(
    f"MAPE : {mean_metrics['MAPE_percent']:.6f}%"
)


print("\nUsing MEDIAN C:")

print(
    f"MAE  : {median_metrics['MAE']:.8f}"
)

print(
    f"RMSE : {median_metrics['RMSE']:.8f}"
)

print(
    f"R²   : {median_metrics['R2']:.8f}"
)

print(
    f"MAPE : {median_metrics['MAPE_percent']:.6f}%"
)


# ============================================================
# 6. SAVE CONFIGURATION
# ============================================================

print("\n[6/6] Saving PINNs physics configuration...")


# Use median as robust calibration constant.
selected_C = C_median


physics_config = {
    "physics_type": "empirical_volume_consistency",
    "relationship": "volume_m3_tree ~= C * dbh_cm^2 * height_m",

    "C": selected_C,

    "C_statistics": {
        "mean": C_mean,
        "median": C_median,
        "std": C_std,
        "min": C_min,
        "max": C_max,
    },

    "calibration_source": "model_train.csv",
    "calibration_rows": int(len(valid)),

    "features": FEATURES,

    "targets": TARGETS,

    "physics_loss": {
        "type": "MSE",
        "formula": (
            "MSE(volume_pred, "
            "C * dbh_pred^2 * height_pred)"
        ),
    },

    "important_note": (
        "C is an empirical dataset-calibrated constant. "
        "It is not claimed to be a universal biological constant."
    ),
}


config_path = (
    RESULTS_DIR
    / "pinns_physics_config.json"
)

with open(
    config_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        physics_config,
        file,
        indent=4,
    )


metrics_path = (
    RESULTS_DIR
    / "pinns_physics_calibration_metrics.csv"
)

metrics_df = pd.DataFrame(
    [
        {
            "calibration": "mean_C",
            "C": C_mean,
            **mean_metrics,
        },
        {
            "calibration": "median_C",
            "C": C_median,
            **median_metrics,
        },
    ]
)

metrics_df.to_csv(
    metrics_path,
    index=False,
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("STEP 7D-2A COMPLETED")
print("=" * 75)

print(
    f"\nSelected C (median) : "
    f"{selected_C:.12f}"
)

print(
    f"\nSaved:"
)

print(
    f"  - {config_path}"
)

print(
    f"  - {metrics_path}"
)

print("\nPhysics relationship:")
print(
    "Volume ≈ C × DBH² × Height"
)

print(
    "\nC was calibrated using TRAINING DATA ONLY."
)

print(
    "The test dataset was NOT used during calibration."
)

print("=" * 75)