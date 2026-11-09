from pathlib import Path
import json
import pandas as pd


# ============================================================
# STEP 45 - PREPARE PINNs CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "model_ready" / "train.csv"
TEST_PATH = ROOT / "data" / "model_ready" / "test.csv"

PIPELINE_DIR = ROOT / "data" / "model_ready" / "pipeline"
PREPROCESSING_DIR = PIPELINE_DIR / "preprocessing"

RESULT_DIR = ROOT / "results" / "pinns"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


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

GROUP_COLUMNS = [
    "compartment_id",
]

DBH_HEIGHT_MIN_AGE = 6
DBH_HEIGHT_MAX_AGE = 66

VOLUME_MIN_AGE = 18
VOLUME_MAX_AGE = 66

PHI_AGE = 66


# ============================================================
# 1. CHECK SOURCE DATA
# ============================================================

print("=" * 80)
print("STEP 45 - PINNs CONFIGURATION PREPARATION")
print("=" * 80)

print("\n[1/7] Loading official train/test data...")

if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Missing train data: {TRAIN_PATH}")

if not TEST_PATH.exists():
    raise FileNotFoundError(f"Missing test data: {TEST_PATH}")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(f"Train rows: {len(train):,}")
print(f"Test rows : {len(test):,}")


# ============================================================
# 2. CHECK FEATURES
# ============================================================

print("\n[2/7] Checking official PINNs features...")

missing_features = [
    col for col in FEATURES
    if col not in train.columns or col not in test.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )

print("Features:")
for feature in FEATURES:
    print(f"  - {feature}")

print("Feature set: PASS")


# ============================================================
# 3. CHECK TARGETS
# ============================================================

print("\n[3/7] Checking targets...")

missing_targets = [
    col for col in TARGETS
    if col not in train.columns or col not in test.columns
]

if missing_targets:
    raise ValueError(
        f"Missing required targets: {missing_targets}"
    )

print("Targets:")
for target in TARGETS:
    print(f"  - {target}")

print("Target set: PASS")


# ============================================================
# 4. CHECK AGE RULES
# ============================================================

print("\n[4/7] Checking age rules...")

train_ages = sorted(train["age_months"].unique().tolist())
test_ages = sorted(test["age_months"].unique().tolist())

print(f"Train ages: {train_ages}")
print(f"Test ages : {test_ages}")

expected_ages = list(range(6, 67, 6))

if train_ages != expected_ages:
    raise ValueError(
        f"Unexpected train ages.\n"
        f"Expected: {expected_ages}\n"
        f"Found   : {train_ages}"
    )

if test_ages != expected_ages:
    raise ValueError(
        f"Unexpected test ages.\n"
        f"Expected: {expected_ages}\n"
        f"Found   : {test_ages}"
    )

if train["age_months"].max() > PHI_AGE:
    raise ValueError("Train data contains ages above P66.")

if test["age_months"].max() > PHI_AGE:
    raise ValueError("Test data contains ages above P66.")

print("P6-P66 rule: PASS")
print("PHI = P66 = 66 months: PASS")


# ============================================================
# 5. CHECK TARGET TRAINING WINDOWS
# ============================================================

print("\n[5/7] Checking target training windows...")

train_dbh_height = train[
    train["age_months"].between(
        DBH_HEIGHT_MIN_AGE,
        DBH_HEIGHT_MAX_AGE
    )
]

test_dbh_height = test[
    test["age_months"].between(
        DBH_HEIGHT_MIN_AGE,
        DBH_HEIGHT_MAX_AGE
    )
]

train_volume = train[
    train["age_months"].between(
        VOLUME_MIN_AGE,
        VOLUME_MAX_AGE
    )
]

test_volume = test[
    test["age_months"].between(
        VOLUME_MIN_AGE,
        VOLUME_MAX_AGE
    )
]

print(
    f"DBH/Height train rows : {len(train_dbh_height):,}"
)
print(
    f"DBH/Height test rows  : {len(test_dbh_height):,}"
)

print(
    f"Volume train rows     : {len(train_volume):,}"
)
print(
    f"Volume test rows      : {len(test_volume):,}"
)

if train_volume["age_months"].min() != VOLUME_MIN_AGE:
    raise ValueError("Volume training does not start at P18.")

if test_volume["age_months"].min() != VOLUME_MIN_AGE:
    raise ValueError("Volume testing does not start at P18.")

print("DBH/Height P6-P66: PASS")
print("Volume P18-P66: PASS")


# ============================================================
# 6. CHECK PREPROCESSING ARTIFACTS
# ============================================================

print("\n[6/7] Checking preprocessing artifacts...")

required_files = [
    PREPROCESSING_DIR / "feature_scaler.joblib",
    PREPROCESSING_DIR / "target_scaler.joblib",
    PREPROCESSING_DIR / "preprocessing_config.json",
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing preprocessing artifact: {path}"
        )
    print(f"FOUND: {path.name}")

print("Preprocessing artifacts: PASS")


# ============================================================
# 7. SAVE PINNs CONFIGURATION
# ============================================================

print("\n[7/7] Saving PINNs configuration...")

config = {
    "project": "TREE GROWTH ANALYTICS - PINNs",

    "source_data": {
        "train": str(TRAIN_PATH.relative_to(ROOT)),
        "test": str(TEST_PATH.relative_to(ROOT)),
    },

    "features": FEATURES,

    "targets": TARGETS,

    "group_columns": GROUP_COLUMNS,

    "age": {
        "minimum": 6,
        "maximum": 66,
        "step_months": 6,
        "phi_age_months": 66,
        "phi_label": "P66",
    },

    "target_windows": {
        "dbh_cm": {
            "min_age_months": 6,
            "max_age_months": 66,
        },
        "height_m": {
            "min_age_months": 6,
            "max_age_months": 66,
        },
        "volume_m3_tree": {
            "min_age_months": 18,
            "max_age_months": 66,
        },
    },

    "physics": {
        "enabled": True,
        "min_age_months": 18,
        "max_age_months": 66,
        "relation": "V_physics = C * DBH^2 * Height",
        "physical_units": True,
    },

    "architecture": {
        "input_size": 6,
        "hidden_layers": [64, 64, 64],
        "activation": "Tanh",
        "output_size": 3,
        "volume_output": "Softplus",
    },

    "training": {
        "new_model_from_scratch": True,
        "reuse_old_checkpoint": False,
        "random_seed": 42,
    },

    "metrics": [
        "MAE",
        "RMSE",
        "R2",
        "MAPE",
    ],

    "comparison_rule": {
        "same_train_test_split_as_random_forest": True,
        "same_features_as_random_forest": True,
        "same_targets_as_random_forest": True,
    },
}

config_path = RESULT_DIR / "pinns_config.json"

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print(f"Saved: {config_path}")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("STEP 45 STATUS: PASS")
print("=" * 80)

print("""
PINNs configuration prepared successfully.

Rules locked:
  - 6 input features
  - DBH P6-P66
  - Height P6-P66
  - Volume P18-P66
  - Physics P18-P66
  - PHI = P66
  - New PINNs model from scratch
  - Old PINNs checkpoint NOT reused
  - Same train/test split as Random Forest
  - Same evaluation metrics
""")

print(f"Configuration:")
print(config_path)
print("=" * 80)