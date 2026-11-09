from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# STEP 37E
# OFFICIAL MODEL DATA PREPARATION
#
# This split is shared by:
#   1. Random Forest
#   2. PINNs
#
# DO NOT create separate train/test splits later.
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
# OFFICIAL TARGETS
# ============================================================

TARGETS = [
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]


# ============================================================
# GROUPING
# ============================================================

GROUP_COL = "compartment_id"

GROUP_KEY_COLS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]


# ============================================================
# EXPECTED DATASET
# ============================================================

EXPECTED_ROWS = 218700
EXPECTED_COLUMNS = 31

EXPECTED_AGES = list(
    range(6, 67, 6)
)

TEST_SIZE = 0.20
RANDOM_STATE = 42


print("=" * 80)
print("STEP 37E - OFFICIAL MODEL DATA PREPARATION")
print("=" * 80)


# ============================================================
# 1. LOAD
# ============================================================

print("\n[1/10] Loading final validated dataset...")

df = pd.read_csv(INPUT_FILE)

print(
    f"Rows    : {len(df):,}"
)

print(
    f"Columns : {len(df.columns):,}"
)


# ============================================================
# 2. BASIC DATASET CHECK
# ============================================================

print(
    "\n[2/10] Checking final dataset structure..."
)

if len(df) != EXPECTED_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_ROWS:,} rows, "
        f"found {len(df):,}."
    )

if len(df.columns) != EXPECTED_COLUMNS:
    raise ValueError(
        f"Expected {EXPECTED_COLUMNS} columns, "
        f"found {len(df.columns)}."
    )


ages = sorted(
    df["age_months"]
    .unique()
    .tolist()
)

if ages != EXPECTED_AGES:
    raise ValueError(
        f"Unexpected age structure: {ages}"
    )

rows_above_phi = int(
    (df["age_months"] > 66).sum()
)

if rows_above_phi != 0:
    raise ValueError(
        "Rows above P66 detected."
    )

print(
    "Dataset structure: PASS"
)


# ============================================================
# 3. MODEL COLUMNS
# ============================================================

print(
    "\n[3/10] Checking official model columns..."
)

required_columns = (
    FEATURES
    + TARGETS
    + GROUP_KEY_COLS
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing columns:\n"
        + "\n".join(missing_columns)
    )

print(
    "Features: PASS"
)

print(
    "Targets: PASS"
)

print(
    "Group columns: PASS"
)


# ============================================================
# 4. MISSING / INFINITE
# ============================================================

print(
    "\n[4/10] Checking model fields..."
)

model_fields = FEATURES + TARGETS

missing_count = int(
    df[model_fields]
    .isna()
    .sum()
    .sum()
)

if missing_count != 0:
    raise ValueError(
        f"Missing model values: "
        f"{missing_count:,}"
    )

numeric_model = (
    df[model_fields]
    .select_dtypes(
        include="number"
    )
)

infinite_count = int(
    pd.Series(
        numeric_model
        .to_numpy()
        .ravel()
    )
    .isin(
        [float("inf"), float("-inf")]
    )
    .sum()
)

if infinite_count != 0:
    raise ValueError(
        f"Infinite model values: "
        f"{infinite_count:,}"
    )

print(
    "No missing model values: PASS"
)

print(
    "No infinite model values: PASS"
)


# ============================================================
# 5. GROUP STRUCTURE
# ============================================================

print(
    "\n[5/10] Checking group structure..."
)

group_count = (
    df[GROUP_KEY_COLS]
    .drop_duplicates()
    .shape[0]
)

compartment_count = (
    df[GROUP_COL]
    .nunique()
)

print(
    f"Growth groups : {group_count:,}"
)

print(
    f"Compartments  : {compartment_count:,}"
)

group_age_counts = (
    df.groupby(
        GROUP_KEY_COLS
        + ["age_months"]
    )
    .size()
)

print(
    f"Group-age combinations: "
    f"{len(group_age_counts):,}"
)

unique_counts = sorted(
    group_age_counts
    .unique()
    .tolist()
)

print(
    f"Rows per group-age: "
    f"{unique_counts}"
)

if unique_counts != [30]:
    raise ValueError(
        "Group-age structure is not "
        "uniformly 30 rows."
    )

print(
    "Group-age structure: PASS"
)


# ============================================================
# 6. EXACT DUPLICATES
# ============================================================

print(
    "\n[6/10] Recording exact duplicates..."
)

duplicate_count = int(
    df.duplicated().sum()
)

print(
    f"Exact duplicate rows: "
    f"{duplicate_count:,}"
)

print(
    "Duplicate rows are retained."
)

print(
    "Reason: group-age invariant is "
    "30 observations and duplicates have "
    "already passed group-leakage audit."
)


# ============================================================
# 7. GROUP-AWARE SPLIT
# ============================================================

print(
    "\n[7/10] Creating official group-aware split..."
)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_idx, test_idx = next(
    splitter.split(
        df,
        groups=df[GROUP_COL],
    )
)

train_df = (
    df.iloc[train_idx]
    .copy()
)

test_df = (
    df.iloc[test_idx]
    .copy()
)


# ============================================================
# 8. VERIFY SPLIT
# ============================================================

print(
    "\n[8/10] Verifying official split..."
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
    f"Train rows        : {len(train_df):,}"
)

print(
    f"Test rows         : {len(test_df):,}"
)

print(
    f"Train compartments: {len(train_groups):,}"
)

print(
    f"Test compartments : {len(test_groups):,}"
)

print(
    f"Group overlap     : {len(overlap):,}"
)

if overlap:
    raise ValueError(
        "GROUP LEAKAGE DETECTED."
    )

print(
    "Group separation: PASS"
)


# ============================================================
# 9. SAVE OFFICIAL SPLIT
# ============================================================

print(
    "\n[9/10] Saving official train/test datasets..."
)

train_file = (
    OUTPUT_DIR
    / "train.csv"
)

test_file = (
    OUTPUT_DIR
    / "test.csv"
)

metadata_file = (
    OUTPUT_DIR
    / "split_metadata.json"
)

train_df.to_csv(
    train_file,
    index=False
)

test_df.to_csv(
    test_file,
    index=False
)


metadata = {
    "source_dataset": str(
        INPUT_FILE
    ),
    "features": FEATURES,
    "targets": TARGETS,
    "group_column": GROUP_COL,
    "group_key_columns": GROUP_KEY_COLS,
    "split_method": "GroupShuffleSplit",
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "total_rows": int(len(df)),
    "train_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "total_compartments": int(
        compartment_count
    ),
    "train_compartments": int(
        len(train_groups)
    ),
    "test_compartments": int(
        len(test_groups)
    ),
    "group_overlap": int(
        len(overlap)
    ),
    "exact_duplicate_rows": int(
        duplicate_count
    ),
}


with open(
    metadata_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


print(
    f"Saved train:\n{train_file}"
)

print(
    f"Saved test:\n{test_file}"
)

print(
    f"Saved metadata:\n{metadata_file}"
)


# ============================================================
# 10. FINAL CHECKS
# ============================================================

print(
    "\n[10/10] Final official split checks..."
)

checks = {
    "Total rows":
        len(df) == EXPECTED_ROWS,

    "Train + test":
        (
            len(train_df)
            + len(test_df)
            == EXPECTED_ROWS
        ),

    "No group overlap":
        len(overlap) == 0,

    "Train groups":
        len(train_groups) == 12,

    "Test groups":
        len(test_groups) == 3,

    "Features":
        all(
            feature in df.columns
            for feature in FEATURES
        ),

    "Targets":
        all(
            target in df.columns
            for target in TARGETS
        ),
}


for name, result in checks.items():

    print(
        f"{name:<22}: "
        f"{'PASS' if result else 'FAIL'}"
    )


all_pass = all(
    checks.values()
)


print("\n" + "=" * 80)

if all_pass:

    print(
        "STEP 37E STATUS: PASS"
    )

    print(
        "\nOFFICIAL MODEL SPLIT IS LOCKED."
    )

    print(
        "\nRandom Forest and PINNs "
        "must use:"
    )

    print(
        "  data/model_ready/train.csv"
    )

    print(
        "  data/model_ready/test.csv"
    )

else:

    print(
        "STEP 37E STATUS: REVIEW REQUIRED"
    )

print("=" * 80)