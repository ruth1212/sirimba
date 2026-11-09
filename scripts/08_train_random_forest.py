from pathlib import Path
import time
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# 1. PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = RESULTS_DIR / "random_forest_models"

TRAIN_PATH = RESULTS_DIR / "model_train.csv"
TEST_PATH = RESULTS_DIR / "model_test.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

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
# 3. HELPER FUNCTIONS
# ============================================================

def calculate_mape(y_true, y_pred):
    """
    MAPE dengan perlindungan terhadap nilai aktual = 0.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0

    if mask.sum() == 0:
        return np.nan

    return np.mean(
        np.abs(
            (y_true[mask] - y_pred[mask])
            / y_true[mask]
        )
    ) * 100


def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "R2": r2_score(y_true, y_pred),
        "MAPE_percent": calculate_mape(y_true, y_pred),
    }


# ============================================================
# 4. START
# ============================================================

print("=" * 75)
print("TREE GROWTH ANALYTICS")
print("STEP 7B - RANDOM FOREST REGRESSOR")
print("=" * 75)


# ============================================================
# 5. LOAD TRAIN / TEST
# ============================================================

print("\n[1/6] Loading prepared train/test data...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Train rows : {len(train_df):,}")
print(f"Test rows  : {len(test_df):,}")


# ============================================================
# 6. PREPARE X / Y
# ============================================================

print("\n[2/6] Preparing features and targets...")

X_train = train_df[FEATURES]
X_test = test_df[FEATURES]

print("\nFeatures:")
for feature in FEATURES:
    print(f"  - {feature}")


# ============================================================
# 7. TRAIN RANDOM FOREST
# ============================================================

print("\n[3/6] Training Random Forest models...")

rf_parameters = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

all_metrics = []
all_predictions = pd.DataFrame(
    {
        "compartment_id": test_df["compartment_id"],
    }
)

model_info = {
    "model": "RandomForestRegressor",
    "random_state": RANDOM_STATE,
    "features": FEATURES,
    "targets": TARGETS,
    "parameters": rf_parameters,
}


for target in TARGETS:

    print("\n" + "-" * 75)
    print(f"Training target: {target}")
    print("-" * 75)

    y_train = train_df[target]
    y_test = test_df[target]

    start_time = time.time()

    model = RandomForestRegressor(
        **rf_parameters
    )

    model.fit(
        X_train,
        y_train,
    )

    training_time = time.time() - start_time

    print(
        f"Training completed in "
        f"{training_time:.2f} seconds."
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    print("Generating predictions...")

    y_pred = model.predict(X_test)


    # ========================================================
    # METRICS
    # ========================================================

    metrics = calculate_metrics(
        y_test,
        y_pred,
    )

    print("\nEvaluation:")
    print(f"MAE  : {metrics['MAE']:.6f}")
    print(f"RMSE : {metrics['RMSE']:.6f}")
    print(f"R²   : {metrics['R2']:.6f}")
    print(
        f"MAPE : "
        f"{metrics['MAPE_percent']:.4f}%"
    )

    all_metrics.append(
        {
            "model": "Random Forest",
            "target": target,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
            "MAPE_percent": metrics["MAPE_percent"],
            "training_time_seconds": training_time,
        }
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = (
        MODEL_DIR
        / f"random_forest_{target}.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(f"Model saved: {model_path}")


    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    all_predictions[
        f"actual_{target}"
    ] = y_test.to_numpy()

    all_predictions[
        f"predicted_{target}"
    ] = y_pred


# ============================================================
# 8. SAVE METRICS
# ============================================================

print("\n[4/6] Saving evaluation metrics...")

metrics_df = pd.DataFrame(all_metrics)

metrics_path = (
    RESULTS_DIR
    / "random_forest_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False,
)

print(f"Saved: {metrics_path}")


# ============================================================
# 9. SAVE PREDICTIONS
# ============================================================

print("\n[5/6] Saving test predictions...")

prediction_path = (
    RESULTS_DIR
    / "random_forest_predictions.csv"
)

all_predictions.to_csv(
    prediction_path,
    index=False,
)

print(f"Saved: {prediction_path}")


# ============================================================
# 10. SAVE MODEL CONFIG
# ============================================================

config_path = (
    RESULTS_DIR
    / "random_forest_config.json"
)

with open(
    config_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        model_info,
        f,
        indent=4,
    )

print(f"Saved: {config_path}")


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n[6/6] Final Random Forest Summary")

print("\n" + "=" * 75)
print("RANDOM FOREST RESULTS")
print("=" * 75)

print(
    metrics_df[
        [
            "target",
            "MAE",
            "RMSE",
            "R2",
            "MAPE_percent",
        ]
    ].to_string(index=False)
)

print("\nFiles created:")

print(f"  - {metrics_path}")
print(f"  - {prediction_path}")
print(f"  - {config_path}")

print("\nModels:")

for target in TARGETS:
    print(
        f"  - random_forest_{target}.joblib"
    )

print("\n" + "=" * 75)
print("STEP 7B COMPLETED")
print("=" * 75)