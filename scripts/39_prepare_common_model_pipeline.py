from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# ============================================================
# STEP 39
# COMMON MODEL PIPELINE PREPARATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "model_ready" / "train.csv"
TEST_PATH = ROOT / "data" / "model_ready" / "test.csv"

OUTPUT_DIR = ROOT / "data" / "model_ready" / "pipeline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OFFICIAL MODEL DEFINITION
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

GROUP_COL = "compartment_id"

DBH_HEIGHT_MIN_AGE = 6
VOLUME_MIN_AGE = 18
MAX_AGE = 66


# ============================================================
# OUTPUT FILES
# ============================================================

TRAIN_MAIN_PATH = OUTPUT_DIR / "train_dbh_height.csv"
TEST_MAIN_PATH = OUTPUT_DIR / "test_dbh_height.csv"

TRAIN_VOLUME_PATH = OUTPUT_DIR / "train_volume_p18.csv"
TEST_VOLUME_PATH = OUTPUT_DIR / "test_volume_p18.csv"

CONFIG_PATH = OUTPUT_DIR / "pipeline_config.json"


def fail(message):
    print()
    print("=" * 80)
    print("STEP 39 STATUS: FAIL")
    print("=" * 80)
    print(message)
    sys.exit(1)


def check_columns(df, required, name):
    missing = [col for col in required if col not in df.columns]

    if missing:
        fail(
            f"{name} is missing required columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )


print("=" * 80)
print("STEP 39 - COMMON MODEL PIPELINE PREPARATION")
print("=" * 80)


# ============================================================
# [1/9] LOAD OFFICIAL TRAIN/TEST
# ============================================================

print("\n[1/9] Loading official train/test datasets...")

if not TRAIN_PATH.exists():
    fail(f"Train file not found:\n{TRAIN_PATH}")

if not TEST_PATH.exists():
    fail(f"Test file not found:\n{TEST_PATH}")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(f"Train rows : {len(train):,}")
print(f"Test rows  : {len(test):,}")

print("Official train/test source: PASS")


# ============================================================
# [2/9] CHECK OFFICIAL COLUMNS
# ============================================================

print("\n[2/9] Checking official model columns...")

required_columns = FEATURES + TARGETS + [GROUP_COL, "age_months"]

check_columns(train, required_columns, "Train")
check_columns(test, required_columns, "Test")

print("Features: PASS")
print("Targets : PASS")
print("Group   : PASS")


# ============================================================
# [3/9] CHECK AGE RANGE
# ============================================================

print("\n[3/9] Checking age range...")

for name, df in [("Train", train), ("Test", test)]:

    min_age = df["age_months"].min()
    max_age = df["age_months"].max()

    print(
        f"{name}: min age={min_age}, max age={max_age}"
    )

    if min_age != DBH_HEIGHT_MIN_AGE:
        fail(
            f"{name} does not start at P6."
        )

    if max_age != MAX_AGE:
        fail(
            f"{name} does not end at P66."
        )

print("P6-P66 range: PASS")


# ============================================================
# [4/9] CREATE DBH/HEIGHT DATA
# ============================================================

print("\n[4/9] Preparing DBH + Height modeling data...")

train_dbh_height = train[
    FEATURES
    + ["dbh_cm", "height_m", GROUP_COL]
].copy()

test_dbh_height = test[
    FEATURES
    + ["dbh_cm", "height_m", GROUP_COL]
].copy()

if train_dbh_height.isna().any().any():
    fail("Missing values detected in DBH/Height train data.")

if test_dbh_height.isna().any().any():
    fail("Missing values detected in DBH/Height test data.")

if not np.isfinite(
    train_dbh_height[FEATURES + ["dbh_cm", "height_m"]]
    .to_numpy()
).all():
    fail("Infinite values detected in DBH/Height train data.")

if not np.isfinite(
    test_dbh_height[FEATURES + ["dbh_cm", "height_m"]]
    .to_numpy()
).all():
    fail("Infinite values detected in DBH/Height test data.")

print(f"Train DBH/Height rows: {len(train_dbh_height):,}")
print(f"Test DBH/Height rows : {len(test_dbh_height):,}")

print("DBH/Height data: PASS")


# ============================================================
# [5/9] CREATE VOLUME P18+ DATA
# ============================================================

print("\n[5/9] Preparing Volume modeling data...")

train_volume = train[
    train["age_months"] >= VOLUME_MIN_AGE
].copy()

test_volume = test[
    test["age_months"] >= VOLUME_MIN_AGE
].copy()

train_volume = train_volume[
    FEATURES
    + ["volume_m3_tree", GROUP_COL]
].copy()

test_volume = test_volume[
    FEATURES
    + ["volume_m3_tree", GROUP_COL]
].copy()

print(
    f"Volume minimum age: P{VOLUME_MIN_AGE}"
)

print(
    f"Train Volume rows: {len(train_volume):,}"
)

print(
    f"Test Volume rows : {len(test_volume):,}"
)

if train_volume["age_months"].min() != VOLUME_MIN_AGE:
    fail("Train volume does not start at P18.")

if test_volume["age_months"].min() != VOLUME_MIN_AGE:
    fail("Test volume does not start at P18.")

if train_volume["age_months"].max() != MAX_AGE:
    fail("Train volume does not end at P66.")

if test_volume["age_months"].max() != MAX_AGE:
    fail("Test volume does not end at P66.")

if train_volume["volume_m3_tree"].min() < 0:
    fail("Negative volume found in train.")

if test_volume["volume_m3_tree"].min() < 0:
    fail("Negative volume found in test.")

print("Volume P18-P66 rule: PASS")


# ============================================================
# [6/9] VERIFY AGE DISTRIBUTION FOR VOLUME
# ============================================================

print("\n[6/9] Checking Volume age distribution...")

train_volume_counts = (
    train_volume["age_months"]
    .value_counts()
    .sort_index()
)

test_volume_counts = (
    test_volume["age_months"]
    .value_counts()
    .sort_index()
)

print("\nTrain Volume age counts:")
print(train_volume_counts.to_string())

print("\nTest Volume age counts:")
print(test_volume_counts.to_string())

expected_volume_ages = [
    18,
    24,
    30,
    36,
    42,
    48,
    54,
    60,
    66,
]

actual_train_volume_ages = sorted(
    train_volume["age_months"].unique().tolist()
)

actual_test_volume_ages = sorted(
    test_volume["age_months"].unique().tolist()
)

if actual_train_volume_ages != expected_volume_ages:
    fail(
        "Unexpected train Volume age structure."
    )

if actual_test_volume_ages != expected_volume_ages:
    fail(
        "Unexpected test Volume age structure."
    )

print("Volume age structure: PASS")


# ============================================================
# [7/9] VERIFY SAME FEATURES
# ============================================================

print("\n[7/9] Verifying six-feature consistency...")

for name, df in [
    ("Train DBH/Height", train_dbh_height),
    ("Test DBH/Height", test_dbh_height),
    ("Train Volume", train_volume),
    ("Test Volume", test_volume),
]:

    actual_features = [
        col for col in FEATURES
        if col in df.columns
    ]

    if actual_features != FEATURES:
        fail(
            f"{name} does not use the official six features."
        )

print("Six-feature consistency: PASS")


# ============================================================
# [8/9] SAVE MODEL DATASETS
# ============================================================

print("\n[8/9] Saving common model datasets...")

train_dbh_height.to_csv(
    TRAIN_MAIN_PATH,
    index=False
)

test_dbh_height.to_csv(
    TEST_MAIN_PATH,
    index=False
)

train_volume.to_csv(
    TRAIN_VOLUME_PATH,
    index=False
)

test_volume.to_csv(
    TEST_VOLUME_PATH,
    index=False
)

print(f"Saved: {TRAIN_MAIN_PATH}")
print(f"Saved: {TEST_MAIN_PATH}")
print(f"Saved: {TRAIN_VOLUME_PATH}")
print(f"Saved: {TEST_VOLUME_PATH}")

print("Model datasets saved: PASS")


# ============================================================
# [9/9] SAVE PIPELINE CONFIG
# ============================================================

print("\n[9/9] Saving pipeline configuration...")

config = {
    "project": "TREE GROWTH ANALYTICS",
    "step": "39",
    "source_train": str(TRAIN_PATH.relative_to(ROOT)),
    "source_test": str(TEST_PATH.relative_to(ROOT)),
    "features": FEATURES,
    "targets": TARGETS,
    "group_column": GROUP_COL,
    "age_range": {
        "minimum": DBH_HEIGHT_MIN_AGE,
        "maximum": MAX_AGE,
    },
    "prediction_rules": {
        "dbh": "P6-P66",
        "height": "P6-P66",
        "volume": "P18-P66",
        "phi": "P66 = 66 months",
        "mai": "predicted_volume_m3_tree / (age_months / 12)",
    },
    "volume_training_rule": {
        "minimum_age_months": VOLUME_MIN_AGE,
        "maximum_age_months": MAX_AGE,
    },
    "model_comparison": {
        "same_features": True,
        "same_targets": True,
        "same_official_train_test_split": True,
        "same_group_aware_split": True,
    },
}

with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        config,
        f,
        indent=2,
        ensure_ascii=False
    )

print(f"Saved: {CONFIG_PATH}")
print("Pipeline configuration: PASS")


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("STEP 39 STATUS: PASS")
print("=" * 80)

print()
print("COMMON MODEL PIPELINE IS READY.")
print()
print("DBH / Height:")
print("  Training/evaluation range : P6-P66")
print()
print("Volume:")
print("  Training/evaluation range : P18-P66")
print()
print("PHI:")
print("  P66 = 66 months")
print()
print("Features:")
for feature in FEATURES:
    print(f"  - {feature}")

print()
print("Targets:")
for target in TARGETS:
    print(f"  - {target}")

print()
print("Random Forest and PINNs now share:")
print("  - Official train/test split")
print("  - Six official features")
print("  - Three official targets")
print("  - PHI = P66")
print("  - Volume starts at P18")
print()
print("NO MODEL TRAINING WAS PERFORMED.")
print("=" * 80)