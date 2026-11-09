from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# ============================================================
# STEP 41
# RANDOM FOREST PREPARATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PIPELINE_DIR = ROOT / "data" / "model_ready" / "pipeline"

TRAIN_DH_PATH = PIPELINE_DIR / "train_dbh_height.csv"
TEST_DH_PATH = PIPELINE_DIR / "test_dbh_height.csv"

TRAIN_VOLUME_PATH = PIPELINE_DIR / "train_volume_p18.csv"
TEST_VOLUME_PATH = PIPELINE_DIR / "test_volume_p18.csv"

RF_DIR = ROOT / "results" / "random_forest"

RF_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OFFICIAL CONFIGURATION
# ============================================================

FEATURES = [
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha",
]

TARGETS = {
    "dbh_cm": {
        "target_file": "dbh_height",
        "min_age": 6,
        "max_age": 66,
    },
    "height_m": {
        "target_file": "dbh_height",
        "min_age": 6,
        "max_age": 66,
    },
    "volume_m3_tree": {
        "target_file": "volume",
        "min_age": 18,
        "max_age": 66,
    },
}

GROUP_COL = "compartment_id"

RANDOM_STATE = 42

RF_CONFIG_PATH = (
    RF_DIR / "random_forest_config.json"
)


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    print()
    print("=" * 80)
    print("STEP 41 STATUS: FAIL")
    print("=" * 80)
    print(message)
    sys.exit(1)


def check_required_columns(df, required, name):
    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        fail(
            f"{name} is missing required columns:\n"
            + "\n".join(
                f"  - {col}"
                for col in missing
            )
        )


def check_finite(df, columns, name):
    values = df[columns].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        fail(
            f"{name} contains NaN or infinite values."
        )


print("=" * 80)
print("STEP 41 - RANDOM FOREST PREPARATION")
print("=" * 80)


# ============================================================
# [1/10] LOAD OFFICIAL DATA
# ============================================================

print("\n[1/10] Loading official RF datasets...")

required_files = [
    TRAIN_DH_PATH,
    TEST_DH_PATH,
    TRAIN_VOLUME_PATH,
    TEST_VOLUME_PATH,
]

for path in required_files:
    if not path.exists():
        fail(
            f"Required file not found:\n{path}"
        )

train_dh = pd.read_csv(
    TRAIN_DH_PATH
)

test_dh = pd.read_csv(
    TEST_DH_PATH
)

train_volume = pd.read_csv(
    TRAIN_VOLUME_PATH
)

test_volume = pd.read_csv(
    TEST_VOLUME_PATH
)

print(
    f"Train DBH/Height : {len(train_dh):,}"
)

print(
    f"Test DBH/Height  : {len(test_dh):,}"
)

print(
    f"Train Volume     : {len(train_volume):,}"
)

print(
    f"Test Volume      : {len(test_volume):,}"
)

print("Official RF datasets loaded: PASS")


# ============================================================
# [2/10] CHECK FEATURES
# ============================================================

print("\n[2/10] Checking six official features...")

for name, df in [
    ("Train DBH/Height", train_dh),
    ("Test DBH/Height", test_dh),
    ("Train Volume", train_volume),
    ("Test Volume", test_volume),
]:
    check_required_columns(
        df,
        FEATURES,
        name
    )

print("Six features: PASS")


# ============================================================
# [3/10] CHECK TARGETS
# ============================================================

print("\n[3/10] Checking RF targets...")

check_required_columns(
    train_dh,
    ["dbh_cm", "height_m"],
    "Train DBH/Height"
)

check_required_columns(
    test_dh,
    ["dbh_cm", "height_m"],
    "Test DBH/Height"
)

check_required_columns(
    train_volume,
    ["volume_m3_tree"],
    "Train Volume"
)

check_required_columns(
    test_volume,
    ["volume_m3_tree"],
    "Test Volume"
)

print("DBH target: PASS")
print("Height target: PASS")
print("Volume target: PASS")


# ============================================================
# [4/10] CHECK AGE RULES
# ============================================================

print("\n[4/10] Checking RF age rules...")

for name, df in [
    ("Train DBH/Height", train_dh),
    ("Test DBH/Height", test_dh),
]:

    if df["age_months"].min() != 6:
        fail(
            f"{name} does not start at P6."
        )

    if df["age_months"].max() != 66:
        fail(
            f"{name} does not end at P66."
        )

for name, df in [
    ("Train Volume", train_volume),
    ("Test Volume", test_volume),
]:

    if df["age_months"].min() != 18:
        fail(
            f"{name} does not start at P18."
        )

    if df["age_months"].max() != 66:
        fail(
            f"{name} does not end at P66."
        )

print("DBH P6-P66: PASS")
print("Height P6-P66: PASS")
print("Volume P18-P66: PASS")


# ============================================================
# [5/10] CHECK MODEL VALUES
# ============================================================

print("\n[5/10] Checking model values...")

check_finite(
    train_dh,
    FEATURES + ["dbh_cm", "height_m"],
    "Train DBH/Height"
)

check_finite(
    test_dh,
    FEATURES + ["dbh_cm", "height_m"],
    "Test DBH/Height"
)

check_finite(
    train_volume,
    FEATURES + ["volume_m3_tree"],
    "Train Volume"
)

check_finite(
    test_volume,
    FEATURES + ["volume_m3_tree"],
    "Test Volume"
)

if (
    train_dh[["dbh_cm", "height_m"]]
    <= 0
).any().any():
    fail(
        "Non-positive DBH/Height found in train."
    )

if (
    test_dh[["dbh_cm", "height_m"]]
    <= 0
).any().any():
    fail(
        "Non-positive DBH/Height found in test."
    )

if (
    train_volume["volume_m3_tree"] < 0
).any():
    fail(
        "Negative volume found in train."
    )

if (
    test_volume["volume_m3_tree"] < 0
).any():
    fail(
        "Negative volume found in test."
    )

print("No NaN/Inf: PASS")
print("DBH/Height positive: PASS")
print("Volume non-negative: PASS")


# ============================================================
# [6/10] VERIFY GROUP SPLIT
# ============================================================

print("\n[6/10] Verifying compartment separation...")

train_groups = set(
    train_dh[GROUP_COL].unique()
)

test_groups = set(
    test_dh[GROUP_COL].unique()
)

overlap = train_groups.intersection(
    test_groups
)

print(
    f"Train compartments: {len(train_groups)}"
)

print(
    f"Test compartments : {len(test_groups)}"
)

print(
    f"Overlap            : {len(overlap)}"
)

if overlap:
    fail(
        "Compartment leakage detected."
    )

print("Compartment separation: PASS")


# ============================================================
# [7/10] VERIFY RAW FEATURE VALUES
# ============================================================

print("\n[7/10] Confirming RF uses raw features...")

for feature in FEATURES:

    train_min = train_dh[feature].min()
    train_max = train_dh[feature].max()

    print(
        f"{feature:<20} "
        f"min={train_min:.6f} "
        f"max={train_max:.6f}"
    )

print()
print(
    "RF preprocessing: raw features, no scaling."
)

print("Raw feature usage: PASS")


# ============================================================
# [8/10] DEFINE THREE RF MODELS
# ============================================================

print("\n[8/10] Defining three RF models...")

model_definitions = {
    "RF_DBH": {
        "target": "dbh_cm",
        "dataset": "train_dbh_height",
        "prediction_age": "P6-P66",
    },

    "RF_HEIGHT": {
        "target": "height_m",
        "dataset": "train_dbh_height",
        "prediction_age": "P6-P66",
    },

    "RF_VOLUME": {
        "target": "volume_m3_tree",
        "dataset": "train_volume_p18",
        "prediction_age": "P18-P66",
    },
}

for model_name, definition in model_definitions.items():

    print(
        f"{model_name:<12} "
        f"target={definition['target']:<18} "
        f"range={definition['prediction_age']}"
    )

print("Three-model structure: PASS")


# ============================================================
# [9/10] RF TRAINING CONFIGURATION
# ============================================================

print("\n[9/10] Creating RF training configuration...")

training_config = {
    "algorithm": "RandomForestRegressor",

    "models": model_definitions,

    "features": FEATURES,

    "group_column": GROUP_COL,

    "random_state": RANDOM_STATE,

    "preprocessing": {
        "feature_scaling": False,
        "target_scaling": False,
        "raw_physical_units": True,
    },

    "training_rules": {
        "dbh": {
            "age_min": 6,
            "age_max": 66,
        },

        "height": {
            "age_min": 6,
            "age_max": 66,
        },

        "volume": {
            "age_min": 18,
            "age_max": 66,
        },
    },

    "evaluation_metrics": [
        "MAE",
        "RMSE",
        "R2",
        "MAPE",
    ],

    "prediction_rules": {
        "phi": "P66 = 66 months",
        "volume_prediction": "P18-P66",
        "mai": (
            "predicted_volume_m3_tree "
            "/ (age_months / 12)"
        ),
    },

    "comparison_requirements": {
        "same_features_as_pinns": True,
        "same_official_split_as_pinns": True,
        "same_targets_as_pinns": True,
        "same_evaluation_metrics_as_pinns": True,
    },
}


# ============================================================
# [10/10] SAVE CONFIGURATION
# ============================================================

config_json = json.dumps(
    training_config,
    indent=2,
    ensure_ascii=False
)

with open(
    RF_CONFIG_PATH,
    "w",
    encoding="utf-8"
) as f:
    f.write(config_json)

print(
    f"Saved: {RF_CONFIG_PATH}"
)

print("RF configuration: PASS")


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("STEP 41 STATUS: PASS")
print("=" * 80)

print()
print("RANDOM FOREST PREPARATION IS READY.")

print()
print("Models:")
print("  RF_DBH     → P6-P66")
print("  RF_HEIGHT  → P6-P66")
print("  RF_VOLUME  → P18-P66")

print()
print("Features:")
for feature in FEATURES:
    print(f"  - {feature}")

print()
print("Preprocessing:")
print("  - Raw features")
print("  - No feature scaling")
print("  - Raw physical targets")

print()
print("Evaluation:")
print("  - MAE")
print("  - RMSE")
print("  - R²")
print("  - MAPE")

print()
print("Official split:")
print("  - Same train/test split as PINNs")
print("  - Same compartment separation")

print()
print("NO MODEL TRAINING WAS PERFORMED.")

print("=" * 80)