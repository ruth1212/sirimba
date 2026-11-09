from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# STEP 36T
# FIX DBH FOR INCOMPLETE GROUPS WITHOUT P66
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

CANDIDATE_FILE = (
    BASE_DIR
    / "data"
    / "dataset_phi66_dbh_v2.csv"
)

SOURCE_FILE = (
    BASE_DIR
    / "data"
    / "dataset_phi66.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "dataset_phi66_final.csv"
)


GROUP_COLS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]

AGE_COL = "age_months"
DBH_COL = "dbh_cm"


# ============================================================
# P6 -> P60 growth progress
# ============================================================

PROGRESS = {
    6: 0.00,
    12: 0.20,
    18: 0.38,
    24: 0.53,
    30: 0.65,
    36: 0.74,
    42: 0.81,
    48: 0.87,
    54: 0.92,
    60: 1.00,
}


print("=" * 75)
print("STEP 36T - FIX INCOMPLETE DBH GROUPS")
print("=" * 75)


# ============================================================
# 1. LOAD CANDIDATE
# ============================================================

print("\n[1/8] Loading candidate dataset...")

df = pd.read_csv(CANDIDATE_FILE)

print(f"Candidate rows    : {len(df):,}")
print(f"Candidate columns : {len(df.columns):,}")


# ============================================================
# 2. LOAD UNTOUCHED SOURCE
# ============================================================

print("\n[2/8] Loading untouched dataset_phi66.csv...")

source = pd.read_csv(SOURCE_FILE)

print(f"Source rows    : {len(source):,}")
print(f"Source columns : {len(source.columns):,}")


# ============================================================
# 3. IDENTIFY INCOMPLETE GROUPS
# ============================================================

print("\n[3/8] Identifying groups without P66...")

group_age = (
    df.groupby(GROUP_COLS)[AGE_COL]
    .agg(
        min_age="min",
        max_age="max",
        age_count="nunique",
    )
    .reset_index()
)

incomplete_groups = group_age[
    group_age["max_age"] < 66
].copy()

print(
    f"Groups without P66: "
    f"{len(incomplete_groups):,}"
)

if len(incomplete_groups) != 135:
    raise ValueError(
        "Expected exactly 135 groups without P66."
    )


# ============================================================
# 4. CHECK ORIGINAL SOURCE
# ============================================================

print("\n[4/8] Checking original DBH source...")

source_subset = source.merge(
    incomplete_groups[GROUP_COLS],
    on=GROUP_COLS,
    how="inner",
)

source_nan = int(
    source_subset[DBH_COL].isna().sum()
)

print(
    f"Source rows for incomplete groups: "
    f"{len(source_subset):,}"
)

print(
    f"Source DBH NaN: {source_nan:,}"
)

if source_nan != 0:
    raise ValueError(
        "Original dataset_phi66.csv contains "
        "missing DBH. Stop."
    )


# ============================================================
# 5. REBUILD DBH
# ============================================================

print("\n[5/8] Rebuilding incomplete DBH trajectories...")

df_result = df.copy()

# IMPORTANT:
# Preserve original dataframe index explicitly.
df_result["_original_index"] = np.arange(
    len(df_result)
)

repaired_rows = 0


for key in (
    incomplete_groups[GROUP_COLS]
    .itertuples(
        index=False,
        name=None
    )
):

    # --------------------------------------------------------
    # Candidate mask
    # --------------------------------------------------------

    candidate_mask = np.ones(
        len(df_result),
        dtype=bool
    )

    source_mask = np.ones(
        len(source_subset),
        dtype=bool
    )

    for col, value in zip(
        GROUP_COLS,
        key
    ):
        candidate_mask &= (
            df_result[col] == value
        )

        source_mask &= (
            source_subset[col] == value
        )

    candidate_group = (
        df_result.loc[candidate_mask]
        .copy()
    )

    source_group = (
        source_subset.loc[source_mask]
        .copy()
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    candidate_group = (
        candidate_group
        .sort_values(
            [AGE_COL, "_original_index"]
        )
    )

    source_group = (
        source_group
        .sort_values(
            [AGE_COL]
        )
    )

    # --------------------------------------------------------
    # Expected ages
    # --------------------------------------------------------

    expected_ages = list(
        range(6, 61, 6)
    )

    candidate_ages = sorted(
        candidate_group[
            AGE_COL
        ].unique().tolist()
    )

    source_ages = sorted(
        source_group[
            AGE_COL
        ].unique().tolist()
    )

    if candidate_ages != expected_ages:
        raise ValueError(
            f"Unexpected candidate ages "
            f"for group {key}: "
            f"{candidate_ages}"
        )

    if source_ages != expected_ages:
        raise ValueError(
            f"Unexpected source ages "
            f"for group {key}: "
            f"{source_ages}"
        )

    # --------------------------------------------------------
    # Calculate P6 and P60 endpoint means
    # from untouched source
    # --------------------------------------------------------

    p6_mean = source_group.loc[
        source_group[AGE_COL] == 6,
        DBH_COL
    ].mean()

    p60_mean = source_group.loc[
        source_group[AGE_COL] == 60,
        DBH_COL
    ].mean()

    if pd.isna(p6_mean):
        raise ValueError(
            f"P6 DBH missing for group {key}"
        )

    if pd.isna(p60_mean):
        raise ValueError(
            f"P60 DBH missing for group {key}"
        )

    # --------------------------------------------------------
    # Total growth
    # --------------------------------------------------------

    total_growth = (
        float(p60_mean)
        - float(p6_mean)
    )

    # Never allow negative endpoint growth.
    total_growth = max(
        total_growth,
        0.1
    )

    # --------------------------------------------------------
    # Calculate source within-age deviation
    # --------------------------------------------------------

    source_age_mean = (
        source_group
        .groupby(AGE_COL)[DBH_COL]
        .transform("mean")
    )

    source_deviation = (
        source_group[DBH_COL]
        - source_age_mean
    )

    # --------------------------------------------------------
    # Rebuilt trajectory
    # --------------------------------------------------------

    source_group["dbh_rebuilt"] = (
        p6_mean
        + source_group[AGE_COL]
        .map(PROGRESS)
        * total_growth
        + 0.35 * source_deviation
    )

    # --------------------------------------------------------
    # Build deterministic row order
    #
    # Each group-age contains 30 rows.
    # --------------------------------------------------------

    candidate_group["_within_age"] = (
        candidate_group
        .groupby(AGE_COL)
        .cumcount()
    )

    source_group["_within_age"] = (
        source_group
        .groupby(AGE_COL)
        .cumcount()
    )

    # --------------------------------------------------------
    # Merge replacement values
    # --------------------------------------------------------

    replacement = source_group[
        [
            AGE_COL,
            "_within_age",
            "dbh_rebuilt",
        ]
    ].copy()

    replacement = replacement.rename(
        columns={
            "dbh_rebuilt":
                "_replacement_dbh"
        }
    )

    candidate_group = candidate_group.merge(
        replacement,
        on=[
            AGE_COL,
            "_within_age",
        ],
        how="left",
        validate="one_to_one",
    )

    if candidate_group[
        "_replacement_dbh"
    ].isna().any():

        raise ValueError(
            f"Replacement DBH missing "
            f"for group {key}"
        )

    # --------------------------------------------------------
    # CRITICAL FIX:
    # Use preserved _original_index
    # instead of dataframe index after merge.
    # --------------------------------------------------------

    original_indices = (
        candidate_group[
            "_original_index"
        ].to_numpy()
    )

    replacement_values = (
        candidate_group[
            "_replacement_dbh"
        ].to_numpy()
    )

    df_result.loc[
        original_indices,
        DBH_COL
    ] = replacement_values

    repaired_rows += len(
        candidate_group
    )


# ============================================================
# 6. VALIDATION
# ============================================================

print("\n[6/8] Validating repair...")

nan_after = int(
    df_result[DBH_COL].isna().sum()
)

nonpositive_after = int(
    (
        df_result[DBH_COL] <= 0
    ).sum()
)

print(
    f"Rows repaired  : {repaired_rows:,}"
)

print(
    f"DBH NaN after   : {nan_after:,}"
)

print(
    f"DBH <= 0 after  : {nonpositive_after:,}"
)


# ============================================================
# REMOVE TEMP COLUMN
# ============================================================

df_result = df_result.drop(
    columns=["_original_index"]
)


# ============================================================
# 7. SAVE
# ============================================================

print("\n[7/8] Saving repaired dataset...")

df_result.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"Output:\n{OUTPUT_FILE}"
)


# ============================================================
# 8. BASIC CHECKS
# ============================================================

print("\n[8/8] Final basic checks...")

checks = {
    "Rows":
        len(df_result) == 218700,

    "Columns":
        len(df_result.columns) == 31,

    "DBH NaN":
        nan_after == 0,

    "DBH positive":
        nonpositive_after == 0,
}


for name, result in checks.items():

    print(
        f"{name:<20}: "
        f"{'PASS' if result else 'FAIL'}"
    )


print("\n" + "=" * 75)

if all(checks.values()):

    print(
        "STEP 36T STATUS: PASS"
    )

else:

    print(
        "STEP 36T STATUS: REVIEW REQUIRED"
    )

print("=" * 75)

print(
    "\nOriginal dataset_phi66.csv was NOT modified."
)

print(
    "Candidate dataset_phi66_dbh_v2.csv was NOT modified."
)