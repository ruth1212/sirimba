from pathlib import Path

import json
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# STEP 37
# MODEL PREPARATION & GROUP-AWARE SPLIT
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "dataset_phi66_final.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "model_ready"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FINAL MODEL SCHEMA
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

GROUP_KEY_COLS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]


print("=" * 80)
print("STEP 37 - MODEL PREPARATION")
print("=" * 80)


# ============================================================
# 1. LOAD FINAL DATASET
# ============================================================

print("\n[1/9] Loading final validated dataset...")

df = pd.read_csv(INPUT_FILE)

print(
    f"Rows    : {len(df):,}"
)

print(
    f"Columns : {len(df.columns):,}"
)


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

print("\n[2/9] Checking modeling columns...")

required = (
    FEATURES
    + TARGETS
    + GROUP_KEY_COLS
)

missing_columns = [
    col
    for col in required
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )

print("Required columns: PASS")


# ============================================================
# 3. CREATE MODEL DATAFRAME
# ============================================================

print("\n[3/9] Creating model dataframe...")

model_columns = (
    FEATURES
    + TARGETS
    + GROUP_KEY_COLS
)

model_df = (
    df[model_columns]
    .copy()
)

print(
    f"Model rows: {len(model_df):,}"
)


# ============================================================
# 4. CHECK DUPLICATES
# ============================================================

print("\n[4/9] Checking exact duplicates...")

duplicates = int(
    model_df.duplicated().sum()
)

print(
    f"Exact duplicate rows: {duplicates:,}"
)

if duplicates > 0:
    raise ValueError(
        "Duplicate rows detected. "
        "Stop before model split."
    )

print("Duplicate check: PASS")


# ============================================================
# 5. GROUP INFORMATION
# ============================================================

print("\n[5/9] Inspecting groups...")

groups = (
    model_df[GROUP_COL]
    .unique()
)

n_groups = len(groups)

print(
    f"Unique compartments: {n_groups:,}"
)

print(
    f"Train/Test split will be based on "
    f"compartment_id."
)


# ============================================================
# 6. GROUP-AWARE TRAIN / TEST SPLIT
# ============================================================

print(
    "\n[6/9] Creating group-aware train/test split..."
)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(
        model_df,
        groups=model_df[GROUP_COL],
    )
)

train_df = (
    model_df
    .iloc[train_idx]
    .copy()
)

test_df = (
    model_df
    .iloc[test_idx]
    .copy()
)


# ============================================================
# 7. VERIFY NO GROUP LEAKAGE
# ============================================================

print(
    "\n[7/9] Verifying group separation..."
)

train_groups = set(
    train_df[GROUP_COL]
)

test_groups = set(
    test_df[GROUP_COL]
)

overlap = (
    train_groups
    & test_groups
)

print(
    f"Train rows : {len(train_df):,}"
)

print(
    f"Test rows  : {len(test_df):,}"
)

print(
    f"Train groups: {len(train_groups):,}"
)

print(
    f"Test groups : {len(test_groups):,}"
)

print(
    f"Overlapping compartments: "
    f"{len(overlap):,}"
)

if overlap:
    raise ValueError(
        "GROUP LEAKAGE DETECTED."
    )

print(
    "Group separation: PASS"
)


# ============================================================
# 8. SAVE SAME SPLIT FOR BOTH MODELS
# ============================================================

print(
    "\n[8/9] Saving shared model datasets..."
)

train_file = (
    OUTPUT_DIR
    / "train.csv"
)

test_file = (
    OUTPUT_DIR
    / "test.csv"
)

train_df.to_csv(
    train_file,
    index=False
)

test_df.to_csv(
    test_file,
    index=False
)


# Save metadata
metadata = {
    "source_dataset": str(
        INPUT_FILE
    ),
    "features": FEATURES,
    "targets": TARGETS,
    "group_column": GROUP_COL,
    "group_key_columns": GROUP_KEY_COLS,
    "split_method": "GroupShuffleSplit",
    "test_size": 0.20,
    "random_state": 42,
    "total_rows": int(len(model_df)),
    "train_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "total_groups": int(n_groups),
    "train_groups": int(len(train_groups)),
    "test_groups": int(len(test_groups)),
    "group_overlap": int(len(overlap)),
}


metadata_file = (
    OUTPUT_DIR
    / "split_metadata.json"
)

with open(
    metadata_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


print(
    f"Saved:\n"
    f"  {train_file}\n"
    f"  {test_file}\n"
    f"  {metadata_file}"
)


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print(
    "\n[9/9] Final preparation summary..."
)

print(
    "\nFEATURES:"
)

for feature in FEATURES:
    print(
        f"  - {feature}"
    )

print(
    "\nTARGETS:"
)

for target in TARGETS:
    print(
        f"  - {target}"
    )

print(
    "\nSPLIT:"
)

print(
    f"  Train rows  : {len(train_df):,}"
)

print(
    f"  Test rows   : {len(test_df):,}"
)

print(
    f"  Train groups: {len(train_groups):,}"
)

print(
    f"  Test groups : {len(test_groups):,}"
)

print(
    f"  Overlap     : {len(overlap):,}"
)


# ============================================================
# FINAL CHECKS
# ============================================================

checks = {
    "Rows preserved":
        len(model_df) == 218700,

    "Train + test":
        len(train_df) + len(test_df)
        == len(model_df),

    "No group overlap":
        len(overlap) == 0,

    "Features present":
        all(
            col in model_df.columns
            for col in FEATURES
        ),

    "Targets present":
        all(
            col in model_df.columns
            for col in TARGETS
        ),
}


print("\nFINAL CHECKS:")

for name, result in checks.items():

    print(
        f"{name:<24}: "
        f"{'PASS' if result else 'FAIL'}"
    )


print("\n" + "=" * 80)

if all(checks.values()):

    print(
        "STEP 37 STATUS: PASS"
    )

    print(
        "\nShared model-ready dataset is prepared."
    )

    print(
        "Random Forest and PINNs MUST use "
        "the same train.csv and test.csv."
    )

else:

    print(
        "STEP 37 STATUS: REVIEW REQUIRED"
    )

print("=" * 80)