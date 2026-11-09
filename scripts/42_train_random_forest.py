from pathlib import Path
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# STEP 42
# RANDOM FOREST TRAINING
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PIPELINE_DIR = (
    ROOT
    / "data"
    / "model_ready"
    / "pipeline"
)

TRAIN_DH_PATH = (
    PIPELINE_DIR
    / "train_dbh_height.csv"
)

TRAIN_VOLUME_PATH = (
    PIPELINE_DIR
    / "train_volume_p18.csv"
)

RF_DIR = (
    ROOT
    / "results"
    / "random_forest"
)

MODEL_DIR = RF_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
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
# TARGET CONFIGURATION
# ============================================================

MODEL_CONFIG = {
    "RF_DBH": {
        "data_path": TRAIN_DH_PATH,
        "target": "dbh_cm",
        "age_range": "P6-P66",
    },

    "RF_HEIGHT": {
        "data_path": TRAIN_DH_PATH,
        "target": "height_m",
        "age_range": "P6-P66",
    },

    "RF_VOLUME": {
        "data_path": TRAIN_VOLUME_PATH,
        "target": "volume_m3_tree",
        "age_range": "P18-P66",
    },
}


# ============================================================
# RF PARAMETERS
# ============================================================

RF_PARAMETERS = {
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================
# OUTPUT
# ============================================================

TRAINING_REPORT_PATH = (
    RF_DIR
    / "training_report.json"
)


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    print()
    print("=" * 80)
    print("STEP 42 STATUS: FAIL")
    print("=" * 80)
    print(message)
    sys.exit(1)


def check_finite(df, columns, name):
    values = df[columns].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        fail(
            f"{name} contains NaN or infinite values."
        )


def train_model(
    model_name,
    config
):
    print()
    print("-" * 80)
    print(f"TRAINING {model_name}")
    print("-" * 80)

    data_path = config["data_path"]
    target = config["target"]
    age_range = config["age_range"]

    print(f"Target      : {target}")
    print(f"Age range   : {age_range}")
    print(f"Dataset     : {data_path}")

    if not data_path.exists():
        fail(
            f"Training dataset not found:\n{data_path}"
        )

    df = pd.read_csv(data_path)

    print(f"Rows        : {len(df):,}")

    required = FEATURES + [target]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        fail(
            f"{model_name} missing columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    check_finite(
        df,
        required,
        model_name
    )

    X = df[FEATURES].to_numpy(
        dtype=float
    )

    y = df[target].to_numpy(
        dtype=float
    )

    print(
        f"Feature matrix: {X.shape}"
    )

    print(
        f"Target shape  : {y.shape}"
    )

    print(
        f"Target min    : {y.min():.8f}"
    )

    print(
        f"Target max    : {y.max():.8f}"
    )

    print()
    print("Creating RandomForestRegressor...")

    model = RandomForestRegressor(
        **RF_PARAMETERS
    )

    print("Starting training...")
    print("This may take some time.")
    print()

    start_time = time.perf_counter()

    model.fit(
        X,
        y
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print()
    print(
        f"Training completed in "
        f"{elapsed:.2f} seconds."
    )

    model_path = (
        MODEL_DIR
        / f"{model_name}.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"Model saved:"
    )
    print(
        model_path
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    importance_path = (
        RF_DIR
        / f"{model_name}_feature_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False
    )

    print()
    print("Feature importance:")
    print(
        importance.to_string(
            index=False
        )
    )

    print()
    print(
        f"Feature importance saved:"
    )
    print(
        importance_path
    )

    return {
        "model_name": model_name,
        "target": target,
        "age_range": age_range,
        "training_rows": int(len(df)),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "parameters": RF_PARAMETERS,
        "training_seconds": round(
            elapsed,
            6
        ),
        "model_path": str(
            model_path.relative_to(ROOT)
        ),
        "feature_importance_path": str(
            importance_path.relative_to(ROOT)
        ),
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("STEP 42 - RANDOM FOREST TRAINING")
print("=" * 80)

print()
print("IMPORTANT:")
print("This script trains NEW Random Forest models.")
print("No old RF model/checkpoint is loaded.")
print()
print("Features:")
for feature in FEATURES:
    print(f"  - {feature}")

print()
print("Parameters:")
for key, value in RF_PARAMETERS.items():
    print(f"  {key}: {value}")

print()
print("Models to train:")
for model_name, config in MODEL_CONFIG.items():
    print(
        f"  {model_name:<12} "
        f"target={config['target']:<18} "
        f"range={config['age_range']}"
    )


# ============================================================
# TRAIN ALL THREE
# ============================================================

training_results = []

overall_start = time.perf_counter()

for model_name, config in MODEL_CONFIG.items():

    result = train_model(
        model_name,
        config
    )

    training_results.append(
        result
    )


overall_elapsed = (
    time.perf_counter()
    - overall_start
)


# ============================================================
# SAVE TRAINING REPORT
# ============================================================

report = {
    "step": "42",
    "algorithm": "RandomForestRegressor",
    "features": FEATURES,
    "parameters": RF_PARAMETERS,
    "models": training_results,
    "total_training_seconds": round(
        overall_elapsed,
        6
    ),
    "training_completed": True,
    "evaluation_completed": False,
}


with open(
    TRAINING_REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("STEP 42 STATUS: PASS")
print("=" * 80)

print()
print("RANDOM FOREST TRAINING COMPLETED.")

print()
print("Models:")

for result in training_results:

    print(
        f"  {result['model_name']:<12} "
        f"→ {result['model_path']}"
    )

print()
print(
    f"Total training time: "
    f"{overall_elapsed:.2f} seconds"
)

print()
print(
    f"Training report:"
)
print(
    TRAINING_REPORT_PATH
)

print()
print("IMPORTANT:")
print("Training is complete.")
print("Final test evaluation has NOT been performed yet.")
print("Actual-vs-predicted trajectory has NOT been generated yet.")

print()
print("NEXT STEP:")
print("STEP 43 - RANDOM FOREST TEST EVALUATION")

print("=" * 80)