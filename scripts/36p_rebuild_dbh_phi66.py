from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# STEP 36P
# REBUILD DBH LOGIC FOR PHI66 DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "dataset_phi66.csv"
OUTPUT_FILE = BASE_DIR / "data" / "dataset_phi66_dbh_v2.csv"

GROUP_COLS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]

AGE_COL = "age_months"
DBH_COL = "dbh_cm"

EXPECTED_AGES = list(range(6, 67, 6))


print("=" * 75)
print("STEP 36P - REBUILD DBH LOGIC PHI66")
print("=" * 75)


# ============================================================
# 1. LOAD
# ============================================================

print("\n[1/8] Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# ============================================================
# 2. VALIDATE
# ============================================================

print("\n[2/8] Validating dataset...")

required = GROUP_COLS + [AGE_COL, DBH_COL]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

ages = sorted(df[AGE_COL].unique().tolist())

if ages != EXPECTED_AGES:
    raise ValueError(
        f"Unexpected ages: {ages}"
    )

print("Schema: PASS")
print("Age structure: PASS")


# ============================================================
# 3. GROUP-AGE DATA
# ============================================================

print("\n[3/8] Creating group-age DBH trajectory...")

group_age = (
    df.groupby(
        GROUP_COLS + [AGE_COL],
        as_index=False
    )
    .agg(
        dbh_mean=(DBH_COL, "mean"),
        dbh_median=(DBH_COL, "median"),
        dbh_std=(DBH_COL, "std"),
        observation_count=(DBH_COL, "size"),
    )
    .sort_values(
        GROUP_COLS + [AGE_COL]
    )
)

print(
    f"Growth groups : "
    f"{group_age[GROUP_COLS].drop_duplicates().shape[0]:,}"
)

print(
    f"Group-age rows: "
    f"{len(group_age):,}"
)


# ============================================================
# 4. AGE PROGRESS
# ============================================================

print("\n[4/8] Defining biological DBH growth progress...")

# Fraction of total DBH growth achieved at each age.
#
# Fast early growth
# Moderate middle growth
# Slow growth approaching PHI
#
# PHI = P66
#
# These values are only used to determine the SHAPE.
# The actual magnitude comes from each growth group's
# observed P6 and P66 values.

DBH_PROGRESS = {
    6:  0.00,
    12: 0.20,
    18: 0.38,
    24: 0.53,
    30: 0.65,
    36: 0.74,
    42: 0.81,
    48: 0.87,
    54: 0.92,
    60: 0.965,
    66: 1.00,
}

progress = pd.Series(
    DBH_PROGRESS,
    dtype=float
)


# ============================================================
# 5. BUILD GROUP PARAMETERS
# ============================================================

print("\n[5/8] Calculating group-specific DBH trajectories...")

p6 = (
    group_age[
        group_age[AGE_COL] == 6
    ][
        GROUP_COLS
        + ["dbh_mean"]
    ]
    .rename(
        columns={
            "dbh_mean": "dbh_p6"
        }
    )
)

p66 = (
    group_age[
        group_age[AGE_COL] == 66
    ][
        GROUP_COLS
        + ["dbh_mean"]
    ]
    .rename(
        columns={
            "dbh_mean": "dbh_p66"
        }
    )
)

params = p6.merge(
    p66,
    on=GROUP_COLS,
    how="inner"
)

print(
    f"Groups with P6 and P66: "
    f"{len(params):,}"
)


# ============================================================
# 6. BUILD MONOTONIC SHAPE
# ============================================================

print("\n[6/8] Rebuilding DBH trajectories...")

params["total_growth"] = (
    params["dbh_p66"]
    - params["dbh_p6"]
)

# Safety check:
# if a group's P66 is somehow lower than P6,
# create a small positive growth amount rather than
# allowing a decreasing biological trajectory.

params["total_growth"] = (
    params["total_growth"]
    .clip(lower=0.10)
)


def get_progress(age):
    ages = np.array(
        sorted(DBH_PROGRESS.keys()),
        dtype=float
    )

    values = np.array(
        [
            DBH_PROGRESS[int(a)]
            for a in ages
        ],
        dtype=float
    )

    return float(
        np.interp(
            age,
            ages,
            values
        )
    )


group_age["growth_progress"] = (
    group_age[AGE_COL]
    .apply(get_progress)
)

group_age = group_age.merge(
    params[
        GROUP_COLS
        + [
            "dbh_p6",
            "dbh_p66",
            "total_growth",
        ]
    ],
    on=GROUP_COLS,
    how="left"
)


# ============================================================
# 7. CREATE NEW GROUP-AGE DBH
# ============================================================

group_age["dbh_new_mean"] = (
    group_age["dbh_p6"]
    + (
        group_age["total_growth"]
        * group_age["growth_progress"]
    )
)


# ------------------------------------------------------------
# Force exact endpoints
# ------------------------------------------------------------

group_age.loc[
    group_age[AGE_COL] == 6,
    "dbh_new_mean"
] = group_age.loc[
    group_age[AGE_COL] == 6,
    "dbh_p6"
]

group_age.loc[
    group_age[AGE_COL] == 66,
    "dbh_new_mean"
] = group_age.loc[
    group_age[AGE_COL] == 66,
    "dbh_p66"
]


# ============================================================
# 8. RESTORE INDIVIDUAL ROW VARIATION
# ============================================================

print("\n[7/8] Restoring within-group-age variation...")

# Read raw DBH only to calculate each row's original
# deviation around its group-age mean.

raw = pd.read_csv(
    INPUT_FILE,
    usecols=GROUP_COLS + [AGE_COL, DBH_COL]
)

raw_group_mean = (
    raw.groupby(
        GROUP_COLS + [AGE_COL],
        as_index=False
    )
    .agg(
        original_group_mean=(
            DBH_COL,
            "mean"
        )
    )
)

raw = raw.merge(
    raw_group_mean,
    on=GROUP_COLS + [AGE_COL],
    how="left"
)

raw["original_deviation"] = (
    raw[DBH_COL]
    - raw["original_group_mean"]
)


# Merge new group-age mean into raw-order dataframe

df_new = df.drop(
    columns=[DBH_COL]
).copy()

df_new = df_new.merge(
    group_age[
        GROUP_COLS
        + [
            AGE_COL,
            "dbh_new_mean",
        ]
    ],
    on=GROUP_COLS + [AGE_COL],
    how="left"
)

# Restore row-level variation.
#
# Only 35% of the original within-age deviation is retained.
# This prevents the 30 observations from becoming identical
# while keeping the new trajectory dominant.

df_new["_original_deviation"] = (
    raw["original_deviation"].values
)

df_new[DBH_COL] = (
    df_new["dbh_new_mean"]
    + 0.35
    * df_new["_original_deviation"]
)

df_new[DBH_COL] = (
    df_new[DBH_COL]
    .clip(lower=0.1)
)


# ============================================================
# RESTORE ORIGINAL COLUMN ORDER
# ============================================================

original_columns = pd.read_csv(
    INPUT_FILE,
    nrows=0
).columns.tolist()

df_new = df_new[
    original_columns
]


# ============================================================
# SAVE
# ============================================================

print("\n[8/8] Saving DBH V2 dataset...")

df_new.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"Output: {OUTPUT_FILE}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("DBH V2 PRELIMINARY VALIDATION")
print("=" * 75)

print(
    f"Rows    : {len(df_new):,}"
)

print(
    f"Columns : {len(df_new.columns):,}"
)


# Group-age validation

check = (
    df_new
    .groupby(
        GROUP_COLS + [AGE_COL],
        as_index=False
    )
    .agg(
        dbh_mean=(DBH_COL, "mean")
    )
    .sort_values(
        GROUP_COLS + [AGE_COL]
    )
)

check["previous_dbh"] = (
    check
    .groupby(GROUP_COLS)["dbh_mean"]
    .shift(1)
)

check["dbh_change"] = (
    check["dbh_mean"]
    - check["previous_dbh"]
)

valid = (
    check["previous_dbh"]
    .notna()
)

decreases = (
    check.loc[
        valid,
        "dbh_change"
    ] < -1e-9
).sum()

stagnations = (
    check.loc[
        valid,
        "dbh_change"
    ].abs() <= 1e-9
).sum()

increases = (
    check.loc[
        valid,
        "dbh_change"
    ] > 1e-9
).sum()


print(
    f"Valid transitions : {valid.sum():,}"
)

print(
    f"Increases         : {increases:,}"
)

print(
    f"Stagnations       : {stagnations:,}"
)

print(
    f"Decreases         : {decreases:,}"
)


# ============================================================
# AGE PROFILE
# ============================================================

profile = (
    check
    .groupby(AGE_COL)["dbh_mean"]
    .mean()
    .sort_index()
)

profile_delta = profile.diff()

print("\nDBH AGE PROFILE")
print("-" * 75)

for age in profile.index:

    label = (
        "PHI"
        if age == 66
        else f"P{int(age)}"
    )

    delta = profile_delta.loc[age]

    delta_text = (
        "N/A"
        if pd.isna(delta)
        else f"{delta:+.6f}"
    )

    print(
        f"{label:>5} | "
        f"mean={profile.loc[age]:.6f} | "
        f"delta={delta_text}"
    )


# ============================================================
# STAGE SUMMARY
# ============================================================

def stage_rate(start_age, end_age):

    subset = profile[
        (profile.index >= start_age)
        & (profile.index <= end_age)
    ]

    changes = (
        subset.diff()
        .dropna()
    )

    if changes.empty:
        return np.nan

    return changes.mean()


early = stage_rate(6, 30)
middle = stage_rate(36, 60)
phi_increment = profile_delta.loc[66]


print("\nSTAGE SUMMARY")
print("-" * 75)

print(
    f"EARLY   P6-P30  : {early:.8f}"
)

print(
    f"MIDDLE  P36-P60 : {middle:.8f}"
)

print(
    f"PHI increment    : {phi_increment:.8f}"
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 75)
print("FINAL CHECK")
print("=" * 75)

print(
    "DBH no decrease :",
    "PASS"
    if decreases == 0
    else "FAIL"
)

print(
    "Early > Middle  :",
    "PASS"
    if early > middle
    else "FAIL"
)

print(
    "PHI increment   :",
    "PASS"
    if phi_increment > 0
    else "FAIL"
)

print("\nOriginal dataset_phi66.csv was NOT modified.")

print("=" * 75)