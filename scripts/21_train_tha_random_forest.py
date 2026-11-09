from pathlib import Path
import json
import time

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
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "results" / "tha_model_train.csv"
TEST_PATH = PROJECT_ROOT / "results" / "tha_model_test.csv"

MODEL_DIR = PROJECT_ROOT / "results" / "tha_random_forest_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = PROJECT_ROOT / "results" / "tha_random_forest_metrics.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "results" / "tha_random_forest_predictions.csv"
CONFIG_PATH = PROJECT_ROOT / "results" / "tha_random_forest_config.json"


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

TARGET = "tha_m"

GROUP_COLUMN = "compartment_id"

RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
MAX_FEATURES = 1.0


# ============================================================
# HELPER
# ============================================================

def calculate_mape(y_true, y_pred):
    """
    MAPE excluding zero actual values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0

    if not np.any(mask):
        return np.nan

    return np.mean(
        np.abs(
            (y_true[mask] - y_pred[mask])
            / y_true[mask]
        )
    ) * 100.0


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 13B — THA RANDOM FOREST BASELINE")
print("=" * 70)

start_time = time.time()


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading THA train/test data...")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(f"Train rows: {len(train):,}")
print(f"Test rows : {len(test):,}")


# ============================================================
# VALIDATE
# ============================================================

required_columns = (
    FEATURES
    + [TARGET, GROUP_COLUMN]
)

missing_train = [
    c for c in required_columns
    if c not in train.columns
]

missing_test = [
    c for c in required_columns
    if c not in test.columns
]

if missing_train:
    raise ValueError(
        f"Missing train columns: {missing_train}"
    )

if missing_test:
    raise ValueError(
        f"Missing test columns: {missing_test}"
    )


# ============================================================
# GROUP LEAKAGE CHECK
# ============================================================

train_groups = set(
    train[GROUP_COLUMN].unique()
)

test_groups = set(
    test[GROUP_COLUMN].unique()
)

overlap = train_groups.intersection(test_groups)

print("\nGroup validation:")
print(f"Train compartments: {len(train_groups)}")
print(f"Test compartments : {len(test_groups)}")
print(f"Overlap            : {len(overlap)}")

if overlap:
    raise ValueError(
        f"Group leakage detected: {sorted(overlap)}"
    )


# ============================================================
# PREPARE X / Y
# ============================================================

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

print("\nFeatures:")
for feature in FEATURES:
    print(f"- {feature}")

print(f"\nTarget: {TARGET}")


# ============================================================
# MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

print(f"\nTrees       : {N_ESTIMATORS}")
print(f"Max depth   : {MAX_DEPTH}")
print(f"Min split   : {MIN_SAMPLES_SPLIT}")
print(f"Min leaf    : {MIN_SAMPLES_LEAF}")
print(f"Max features: {MAX_FEATURES}")
print(f"Random state: {RANDOM_STATE}")

model = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    max_features=MAX_FEATURES,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

print("\nTraining...")

model.fit(
    X_train,
    y_train,
)

training_time = time.time() - start_time

print(
    f"Training completed in "
    f"{training_time:.2f} seconds."
)


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating test predictions...")

predictions = model.predict(X_test)


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions,
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions,
    )
)

r2 = r2_score(
    y_test,
    predictions,
)

mape = calculate_mape(
    y_test,
    predictions,
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("THA RANDOM FOREST RESULTS")
print("=" * 70)

print(f"\nMAE  : {mae:.6f} m")
print(f"RMSE : {rmse:.6f} m")
print(f"R2   : {r2:.6f}")
print(f"MAPE : {mape:.4f}%")


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR
    / "random_forest_tha_m.joblib"
)

joblib.dump(
    model,
    model_path,
)

print(f"\nModel saved:")
print(model_path)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame(
    [
        {
            "model": "Random Forest",
            "target": TARGET,
            "test_rows": len(test),
            "train_rows": len(train),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape_percent": mape,
            "training_time_seconds": training_time,
        }
    ]
)

metrics.to_csv(
    METRICS_PATH,
    index=False,
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_output = pd.DataFrame(
    {
        "compartment_id": test[GROUP_COLUMN].values,

        "age_months": test["age_months"].values,
        "inventory_year": test["inventory_year"].values,
        "clone_factor": test["clone_factor"].values,
        "planting_year": test["planting_year"].values,
        "disease_score": test["disease_score"].values,
        "trees_per_ha": test["trees_per_ha"].values,

        "actual_tha_m": y_test.values,
        "predicted_tha_m": predictions,
    }
)

prediction_output.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


# ============================================================
# SAVE CONFIG
# ============================================================

config = {
    "model": "RandomForestRegressor",
    "target": TARGET,
    "features": FEATURES,
    "group_column": GROUP_COLUMN,

    "random_state": RANDOM_STATE,
    "n_estimators": N_ESTIMATORS,
    "max_depth": MAX_DEPTH,
    "min_samples_split": MIN_SAMPLES_SPLIT,
    "min_samples_leaf": MIN_SAMPLES_LEAF,
    "max_features": MAX_FEATURES,

    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "train_groups": int(len(train_groups)),
    "test_groups": int(len(test_groups)),

    "metrics": {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape_percent": float(mape),
    },

    "training_time_seconds": float(
        training_time
    ),

    "tha_definition": (
        "THA (Top Height) = tinggi dominan tegakan, "
        "yaitu ukuran tinggi kelompok pohon-pohon "
        "dominan/tertinggi."
    ),

    "tha_numerical_method": (
        "Numerical source method was not determined "
        "from the available dataset documentation."
    ),
}

with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        config,
        f,
        indent=2,
    )


# ============================================================
# FINAL
# ============================================================

print("\nSaved:")
print(f"- {METRICS_PATH}")
print(f"- {PREDICTIONS_PATH}")
print(f"- {CONFIG_PATH}")

print("\n" + "=" * 70)
print("STEP 13B COMPLETED")
print("=" * 70)

print(
    "\nTHA Random Forest baseline is ready."
)

print(
    "\nNext: integrate THA into the prediction engine "
    "and build the dashboard API."
)

print("=" * 70)