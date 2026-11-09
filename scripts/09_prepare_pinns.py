from pathlib import Path
import sys

import pandas as pd


# ============================================================
# STEP 7C - PINNs ENVIRONMENT & DATA VALIDATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "results" / "model_train.csv"
TEST_PATH = PROJECT_ROOT / "results" / "model_test.csv"


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


print("=" * 75)
print("TREE GROWTH ANALYTICS")
print("STEP 7C - PINNs PREPARATION")
print("=" * 75)


# ============================================================
# 1. CHECK PYTORCH
# ============================================================

print("\n[1/5] Checking PyTorch...")

try:
    import torch
except ImportError:
    print("\nPyTorch is NOT installed in this virtual environment.")
    print("\nInstall it with:")
    print("python -m pip install torch")
    sys.exit(1)

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device     : {torch.cuda.get_device_name(0)}")
else:
    print("Device          : CPU")


# ============================================================
# 2. LOAD TRAIN / TEST
# ============================================================

print("\n[2/5] Loading the same train/test datasets used by RF...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Train rows : {len(train_df):,}")
print(f"Test rows  : {len(test_df):,}")


# ============================================================
# 3. VALIDATE FEATURES
# ============================================================

print("\n[3/5] Validating PINNs input features...")

required = FEATURES + TARGETS

missing_train = [
    col for col in required
    if col not in train_df.columns
]

missing_test = [
    col for col in required
    if col not in test_df.columns
]

if missing_train or missing_test:
    raise ValueError(
        f"Missing train columns: {missing_train}\n"
        f"Missing test columns : {missing_test}"
    )

print("All PINNs features and targets are available.")

print("\nPINNs input features:")
for feature in FEATURES:
    print(f"  - {feature}")

print("\nPINNs prediction targets:")
for target in TARGETS:
    print(f"  - {target}")


# ============================================================
# 4. CHECK NUMERIC VALUES
# ============================================================

print("\n[4/5] Checking numeric values...")

for column in FEATURES + TARGETS:

    if not pd.api.types.is_numeric_dtype(train_df[column]):
        raise TypeError(
            f"Column '{column}' must be numeric for PINNs."
        )

    if not pd.api.types.is_numeric_dtype(test_df[column]):
        raise TypeError(
            f"Column '{column}' must be numeric for PINNs."
        )

    if not train_df[column].notna().all():
        raise ValueError(
            f"Missing values detected in train column '{column}'."
        )

    if not test_df[column].notna().all():
        raise ValueError(
            f"Missing values detected in test column '{column}'."
        )

print("All selected inputs and targets are numeric and complete.")


# ============================================================
# 5. CHECK SAME SPLIT AS RANDOM FOREST
# ============================================================

print("\n[5/5] Checking group separation...")

train_groups = set(train_df["compartment_id"].unique())
test_groups = set(test_df["compartment_id"].unique())

overlap = train_groups.intersection(test_groups)

print(f"Train groups : {len(train_groups):,}")
print(f"Test groups  : {len(test_groups):,}")
print(f"Overlap      : {len(overlap):,}")

if overlap:
    raise ValueError(
        "Group overlap detected. PINNs cannot continue."
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("STEP 7C VALIDATION PASSED")
print("=" * 75)

print("\nIMPORTANT:")
print("1. PINNs will use exactly the same six features as Random Forest.")
print("2. PINNs will use exactly the same train/test split.")
print("3. PINNs will predict the same three targets.")
print("4. No old Random Forest model or .joblib file is used for PINNs.")
print("5. Physics loss must be defined explicitly before training.")

print("=" * 75)