from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# 1. PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

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

GROUP_COLUMN = "compartment_id"


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("=" * 70)
print("TREE GROWTH ANALYTICS")
print("STEP 7A - MODELING DATA PREPARATION")
print("=" * 70)

print("\n[1/7] Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded successfully.")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

print("\n[2/7] Removing duplicate rows...")

before = len(df)

df = df.drop_duplicates().reset_index(drop=True)

after = len(df)

print(f"Rows before : {before:,}")
print(f"Rows after  : {after:,}")
print(f"Removed     : {before - after:,}")


# ============================================================
# 5. VALIDATE REQUIRED COLUMNS
# ============================================================

print("\n[3/7] Validating required columns...")

required_columns = FEATURES + TARGETS + [GROUP_COLUMN]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Required columns are missing:\n"
        + "\n".join(f"- {col}" for col in missing_columns)
    )

print("All required columns are available.")


# ============================================================
# 6. SELECT MODELING DATA
# ============================================================

print("\n[4/7] Selecting modeling columns...")

model_columns = FEATURES + TARGETS + [GROUP_COLUMN]

model_df = df[model_columns].copy()

print(f"Modeling rows    : {len(model_df):,}")
print(f"Modeling columns : {len(model_df.columns)}")

print("\nFeatures:")
for feature in FEATURES:
    print(f"  - {feature}")

print("\nTargets:")
for target in TARGETS:
    print(f"  - {target}")


# ============================================================
# 7. CHECK MISSING VALUES
# ============================================================

print("\n[5/7] Checking missing values...")

missing = model_df.isnull().sum()

total_missing = int(missing.sum())

if total_missing > 0:
    print("\nMissing values found:")
    print(missing[missing > 0])

    raise ValueError(
        "Modeling data contains missing values. "
        "Fix the dataset before continuing."
    )

print("No missing values found.")


# ============================================================
# 8. GROUP-AWARE TRAIN / TEST SPLIT
# ============================================================

print("\n[6/7] Creating group-aware train/test split...")

unique_groups = model_df[GROUP_COLUMN].nunique()

print(f"Unique {GROUP_COLUMN}: {unique_groups:,}")

if unique_groups < 2:
    raise ValueError(
        f"At least 2 unique {GROUP_COLUMN} groups are required."
    )

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_idx, test_idx = next(
    splitter.split(
        model_df,
        groups=model_df[GROUP_COLUMN],
    )
)

train_df = model_df.iloc[train_idx].copy()
test_df = model_df.iloc[test_idx].copy()


# ============================================================
# 9. VERIFY GROUP SEPARATION
# ============================================================

train_groups = set(train_df[GROUP_COLUMN].unique())
test_groups = set(test_df[GROUP_COLUMN].unique())

overlap_groups = train_groups.intersection(test_groups)

if overlap_groups:
    raise ValueError(
        "DATA LEAKAGE DETECTED: "
        f"{len(overlap_groups)} groups appear in both train and test."
    )

print("\nGroup separation verified.")
print(f"Train groups : {len(train_groups):,}")
print(f"Test groups  : {len(test_groups):,}")
print(f"Overlap      : {len(overlap_groups):,}")


# ============================================================
# 10. SAVE TRAIN / TEST DATA
# ============================================================

print("\n[7/7] Saving modeling datasets...")

train_path = RESULTS_DIR / "model_train.csv"
test_path = RESULTS_DIR / "model_test.csv"

train_df.to_csv(
    train_path,
    index=False,
)

test_df.to_csv(
    test_path,
    index=False,
)


# ============================================================
# 11. SAVE MODEL CONFIGURATION
# ============================================================

config_path = RESULTS_DIR / "model_features.txt"

config_text = f"""
TREE GROWTH ANALYTICS - MODEL CONFIGURATION
============================================

Random state:
{RANDOM_STATE}

Test size:
{TEST_SIZE}

Group column:
{GROUP_COLUMN}

Features:
{chr(10).join(f"- {x}" for x in FEATURES)}

Targets:
{chr(10).join(f"- {x}" for x in TARGETS)}

Train rows:
{len(train_df):,}

Test rows:
{len(test_df):,}

Train groups:
{len(train_groups):,}

Test groups:
{len(test_groups):,}

Group overlap:
{len(overlap_groups):,}

IMPORTANT:
Random Forest and PINNs must use these exact same
train/test datasets for fair comparison.
"""

config_path.write_text(
    config_text.strip(),
    encoding="utf-8",
)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 7A COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"\nTrain rows : {len(train_df):,}")
print(f"Test rows  : {len(test_df):,}")

print(f"\nTrain groups : {len(train_groups):,}")
print(f"Test groups  : {len(test_groups):,}")

print("\nSaved files:")

print(f"  - {train_path}")
print(f"  - {test_path}")
print(f"  - {config_path}")

print("\nNext stage:")
print("  Random Forest Regressor")
print("=" * 70)