from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# STEP 43
# RANDOM FOREST TEST EVALUATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PIPELINE_DIR = (
    ROOT
    / "data"
    / "model_ready"
    / "pipeline"
)

MODEL_DIR = (
    ROOT
    / "results"
    / "random_forest"
    / "models"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "random_forest"
)

TRAINING_REPORT_PATH = (
    RESULT_DIR
    / "training_report.json"
)

RESULTS_PATH = (
    RESULT_DIR
    / "test_metrics.csv"
)

TRAJECTORY_PATH = (
    RESULT_DIR
    / "actual_vs_predicted_by_age.csv"
)

SUMMARY_PATH = (
    RESULT_DIR
    / "evaluation_summary.json"
)


# ============================================================
# OFFICIAL FEATURES
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
# MODEL DEFINITIONS
# ============================================================

MODELS = {
    "RF_DBH": {
        "model_path": MODEL_DIR / "RF_DBH.joblib",
        "data_path": PIPELINE_DIR / "test_dbh_height.csv",
        "target": "dbh_cm",
        "min_age": 6,
        "max_age": 66,
    },

    "RF_HEIGHT": {
        "model_path": MODEL_DIR / "RF_HEIGHT.joblib",
        "data_path": PIPELINE_DIR / "test_dbh_height.csv",
        "target": "height_m",
        "min_age": 6,
        "max_age": 66,
    },

    "RF_VOLUME": {
        "model_path": MODEL_DIR / "RF_VOLUME.joblib",
        "data_path": PIPELINE_DIR / "test_volume_p18.csv",
        "target": "volume_m3_tree",
        "min_age": 18,
        "max_age": 66,
    },
}


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    print()
    print("=" * 80)
    print("STEP 43 STATUS: FAIL")
    print("=" * 80)
    print(message)
    sys.exit(1)


def safe_mape(y_true, y_pred):
    """
    MAPE with zero targets excluded.

    This is especially relevant for volume because very small
    values at young ages can make ordinary MAPE unstable.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0

    if mask.sum() == 0:
        return np.nan

    return np.mean(
        np.abs(
            (y_true[mask] - y_pred[mask])
            / y_true[mask]
        )
    ) * 100.0


def check_finite(values, name):
    values = np.asarray(values, dtype=float)

    if not np.isfinite(values).all():
        fail(
            f"{name} contains NaN or infinite values."
        )


# ============================================================
# START
# ============================================================

print("=" * 80)
print("STEP 43 - RANDOM FOREST TEST EVALUATION")
print("=" * 80)

print()
print("IMPORTANT:")
print("Evaluation uses TEST data only.")
print("No training rows are used for final metrics.")
print("No model is retrained in this step.")
print()


# ============================================================
# [1/8] CHECK FILES
# ============================================================

print("[1/8] Checking model and test files...")

for model_name, config in MODELS.items():

    if not config["model_path"].exists():
        fail(
            f"Model not found for {model_name}:\n"
            f"{config['model_path']}"
        )

    if not config["data_path"].exists():
        fail(
            f"Test dataset not found for {model_name}:\n"
            f"{config['data_path']}"
        )

print("All RF models found: PASS")
print("All official test datasets found: PASS")


# ============================================================
# [2/8] LOAD MODELS AND TEST DATA
# ============================================================

print("\n[2/8] Loading models and test datasets...")

loaded_models = {}
loaded_data = {}

for model_name, config in MODELS.items():

    model = joblib.load(
        config["model_path"]
    )

    df = pd.read_csv(
        config["data_path"]
    )

    loaded_models[model_name] = model
    loaded_data[model_name] = df

    print(
        f"{model_name:<12} "
        f"rows={len(df):,} "
        f"target={config['target']}"
    )

print("Models/test data loaded: PASS")


# ============================================================
# [3/8] VERIFY TEST STRUCTURE
# ============================================================

print("\n[3/8] Verifying test structure...")

for model_name, config in MODELS.items():

    df = loaded_data[model_name]

    required = FEATURES + [
        config["target"],
        "age_months",
        "compartment_id",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        fail(
            f"{model_name} test data missing:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    if df["age_months"].min() != config["min_age"]:
        fail(
            f"{model_name}: unexpected minimum age."
        )

    if df["age_months"].max() != config["max_age"]:
        fail(
            f"{model_name}: unexpected maximum age."
        )

    values = df[
        FEATURES + [config["target"]]
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        fail(
            f"{model_name}: NaN/Inf detected."
        )

print("Test structure: PASS")


# ============================================================
# [4/8] GENERATE TEST PREDICTIONS
# ============================================================

print("\n[4/8] Generating test predictions...")

predictions = {}

for model_name, config in MODELS.items():

    df = loaded_data[model_name]
    model = loaded_models[model_name]

    X = df[
        FEATURES
    ].to_numpy(dtype=float)

    y_true = df[
        config["target"]
    ].to_numpy(dtype=float)

    y_pred = model.predict(X)

    check_finite(
        y_pred,
        f"{model_name} predictions"
    )

    predictions[model_name] = {
        "y_true": y_true,
        "y_pred": y_pred,
    }

    print(
        f"{model_name:<12} "
        f"predictions={len(y_pred):,}"
    )

print("Test predictions generated: PASS")


# ============================================================
# [5/8] CALCULATE METRICS
# ============================================================

print("\n[5/8] Calculating test metrics...")

metric_rows = []

for model_name, config in MODELS.items():

    y_true = predictions[
        model_name
    ]["y_true"]

    y_pred = predictions[
        model_name
    ]["y_pred"]

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    mape = safe_mape(
        y_true,
        y_pred
    )

    metric_rows.append({
        "model": model_name,
        "target": config["target"],
        "age_range": (
            f"P{config['min_age']}"
            f"-P{config['max_age']}"
        ),
        "test_rows": len(y_true),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE_percent": mape,
    })

metrics_df = pd.DataFrame(
    metric_rows
)

print()
print(
    metrics_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("Metrics calculated: PASS")


# ============================================================
# [6/8] BUILD ACTUAL VS PREDICTED AGE TRAJECTORY
# ============================================================

print(
    "\n[6/8] Building actual-vs-predicted age trajectories..."
)

trajectory_rows = []

for model_name, config in MODELS.items():

    df = loaded_data[model_name].copy()

    y_pred = predictions[
        model_name
    ]["y_pred"]

    target = config["target"]

    df["prediction"] = y_pred

    grouped = (
        df.groupby(
            "age_months",
            as_index=False
        )
        .agg(
            actual_mean=(
                target,
                "mean"
            ),
            predicted_mean=(
                "prediction",
                "mean"
            ),
            actual_std=(
                target,
                "std"
            ),
            predicted_std=(
                "prediction",
                "std"
            ),
            n=(
                target,
                "count"
            ),
        )
    )

    grouped["model"] = model_name
    grouped["target"] = target

    grouped["absolute_error"] = (
        grouped["actual_mean"]
        - grouped["predicted_mean"]
    ).abs()

    grouped["age_years"] = (
        grouped["age_months"] / 12.0
    )

    trajectory_rows.append(
        grouped[
            [
                "model",
                "target",
                "age_months",
                "age_years",
                "actual_mean",
                "predicted_mean",
                "actual_std",
                "predicted_std",
                "absolute_error",
                "n",
            ]
        ]
    )

trajectory_df = pd.concat(
    trajectory_rows,
    ignore_index=True
)

trajectory_df.to_csv(
    TRAJECTORY_PATH,
    index=False
)

print(
    f"Saved trajectory:"
)
print(
    TRAJECTORY_PATH
)

print("Actual-vs-predicted trajectory: PASS")


# ============================================================
# [7/8] PHI COMPARISON
# ============================================================

print("\n[7/8] Checking P66 / PHI predictions...")

phi_rows = []

for model_name, config in MODELS.items():

    df = loaded_data[model_name].copy()

    df["prediction"] = predictions[
        model_name
    ]["y_pred"]

    phi = df[
        df["age_months"] == 66
    ]

    if len(phi) == 0:
        fail(
            f"{model_name}: no P66 rows found."
        )

    target = config["target"]

    phi_rows.append({
        "model": model_name,
        "target": target,
        "phi_age_months": 66,
        "actual_mean": phi[target].mean(),
        "predicted_mean": phi["prediction"].mean(),
        "MAE_at_P66": mean_absolute_error(
            phi[target],
            phi["prediction"]
        ),
        "n": len(phi),
    })

phi_df = pd.DataFrame(
    phi_rows
)

print()
print(
    phi_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("P66/PHI evaluation: PASS")


# ============================================================
# [8/8] SAVE RESULTS
# ============================================================

print("\n[8/8] Saving evaluation results...")

metrics_df.to_csv(
    RESULTS_PATH,
    index=False
)

evaluation_summary = {
    "step": "43",
    "evaluation_source": {
        "dbh_height": str(
            (PIPELINE_DIR / "test_dbh_height.csv")
            .relative_to(ROOT)
        ),
        "volume": str(
            (PIPELINE_DIR / "test_volume_p18.csv")
            .relative_to(ROOT)
        ),
    },
    "models": {
        model_name: {
            "model_path": str(
                config["model_path"]
                .relative_to(ROOT)
            ),
            "target": config["target"],
            "age_range": (
                f"P{config['min_age']}"
                f"-P{config['max_age']}"
            ),
        }
        for model_name, config in MODELS.items()
    },
    "metrics": metric_rows,
    "phi_evaluation": (
        phi_df.to_dict(
            orient="records"
        )
    ),
    "training_data_used_for_metrics": False,
    "retraining_performed": False,
    "trajectory_generated": True,
}

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        evaluation_summary,
        f,
        indent=2,
        ensure_ascii=False
    )

print(
    f"Saved metrics:"
)
print(
    RESULTS_PATH
)

print(
    f"Saved summary:"
)
print(
    SUMMARY_PATH
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("STEP 43 STATUS: PASS")
print("=" * 80)

print()
print("RANDOM FOREST TEST EVALUATION COMPLETED.")

print()
print("Metrics:")
print("  - MAE")
print("  - RMSE")
print("  - R²")
print("  - MAPE")

print()
print("Evaluation ranges:")
print("  RF_DBH     → P6-P66")
print("  RF_HEIGHT  → P6-P66")
print("  RF_VOLUME  → P18-P66")

print()
print("Trajectory:")
print("  Actual mean vs predicted mean by age")
print("  P66/PHI comparison included")

print()
print("Saved:")
print(f"  {RESULTS_PATH}")
print(f"  {TRAJECTORY_PATH}")
print(f"  {SUMMARY_PATH}")

print()
print("NO MODEL WAS RETRAINED.")

print("=" * 80)