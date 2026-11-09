from pathlib import Path

import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# STEP 37D
# AUDIT DUPLICATES VS GROUP-AWARE SPLIT
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "dataset_phi66_final.csv"
)

GROUP_COL = "compartment_id"


print("=" * 80)
print("STEP 37D - DUPLICATE / GROUP LEAKAGE AUDIT")
print("=" * 80)


# ============================================================
# 1. LOAD
# ============================================================

print("\n[1/6] Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(
    f"Rows: {len(df):,}"
)


# ============================================================
# 2. DUPLICATES
# ============================================================

print(
    "\n[2/6] Inspecting exact duplicates..."
)

duplicate_mask = df.duplicated(
    keep=False
)

duplicates = df.loc[
    duplicate_mask
].copy()

duplicate_count = int(
    df.duplicated().sum()
)

print(
    f"Duplicate rows excluding first: "
    f"{duplicate_count:,}"
)

print(
    f"Rows participating: "
    f"{len(duplicates):,}"
)


# ============================================================
# 3. DUPLICATE COMPARTMENTS
# ============================================================

duplicate_compartments = sorted(
    duplicates[
        GROUP_COL
    ]
    .unique()
    .tolist()
)

print(
    "\n[3/6] Duplicate compartments:"
)

print(
    duplicate_compartments
)

print(
    f"Number of affected compartments: "
    f"{len(duplicate_compartments):,}"
)


# ============================================================
# 4. GROUP-AWARE SPLIT
# ============================================================

print(
    "\n[4/6] Creating group-aware split..."
)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(
        df,
        groups=df[GROUP_COL],
    )
)

train_df = df.iloc[
    train_idx
].copy()

test_df = df.iloc[
    test_idx
].copy()

train_groups = set(
    train_df[GROUP_COL]
)

test_groups = set(
    test_df[GROUP_COL]
)

print(
    f"Train rows: {len(train_df):,}"
)

print(
    f"Test rows : {len(test_df):,}"
)

print(
    f"Train compartments: "
    f"{len(train_groups):,}"
)

print(
    f"Test compartments: "
    f"{len(test_groups):,}"
)


# ============================================================
# 5. CHECK DUPLICATE SPLIT LOCATION
# ============================================================

print(
    "\n[5/6] Checking duplicate placement..."
)

duplicate_train = [
    c
    for c in duplicate_compartments
    if c in train_groups
]

duplicate_test = [
    c
    for c in duplicate_compartments
    if c in test_groups
]

duplicate_both = (
    set(duplicate_train)
    & set(duplicate_test)
)

print(
    f"Duplicate compartments in train: "
    f"{duplicate_train}"
)

print(
    f"Duplicate compartments in test : "
    f"{duplicate_test}"
)

print(
    f"Duplicate compartments in both : "
    f"{duplicate_both}"
)


# ============================================================
# 6. FINAL
# ============================================================

print(
    "\n[6/6] Final leakage audit..."
)

no_overlap = (
    len(
        train_groups
        & test_groups
    )
    == 0
)

duplicates_not_split = (
    len(duplicate_both) == 0
)

print(
    "Train/test compartment overlap:",
    "PASS" if no_overlap else "FAIL"
)

print(
    "Duplicate compartment split leakage:",
    "PASS"
    if duplicates_not_split
    else "FAIL"
)


print("\n" + "=" * 80)

if (
    no_overlap
    and duplicates_not_split
):

    print(
        "STEP 37D STATUS: PASS"
    )

    print(
        "\nExact duplicates remain under "
        "the same compartment and cannot "
        "leak across train/test."
    )

else:

    print(
        "STEP 37D STATUS: REVIEW REQUIRED"
    )

print("=" * 80)

print(
    "\nDataset was NOT modified."
)