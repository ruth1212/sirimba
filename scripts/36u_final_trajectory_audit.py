from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# STEP 36U
# FINAL TRAJECTORY AUDIT
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "dataset_phi66_final.csv"
)

RAW_FILE = (
    BASE_DIR
    / "data"
    / "dataset.csv"
)


GROUP_COLS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]

AGE_COL = "age_months"

TARGETS = [
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]

FEATURES = [
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha",
]

EXPECTED_AGES = list(range(6, 67, 6))

PHI = 66


print("=" * 80)
print("STEP 36U - FINAL TRAJECTORY AUDIT")
print("=" * 80)


# ============================================================
# 1. LOAD
# ============================================================

print("\n[1/12] Loading final candidate dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# ============================================================
# 2. SCHEMA
# ============================================================

print("\n[2/12] Checking schema...")

raw_columns = pd.read_csv(
    RAW_FILE,
    nrows=0
).columns.tolist()

final_columns = df.columns.tolist()

schema_pass = (
    raw_columns == final_columns
)

print(
    "Schema identical to raw:",
    "PASS" if schema_pass else "FAIL"
)

if not schema_pass:
    print("\nRAW:")
    print(raw_columns)

    print("\nFINAL:")
    print(final_columns)


# ============================================================
# 3. DIMENSIONS
# ============================================================

print("\n[3/12] Checking dimensions...")

rows_pass = len(df) == 218700
columns_pass = len(df.columns) == 31

print(
    f"Rows expected : 218,700"
)

print(
    f"Rows actual   : {len(df):,}"
)

print(
    "Rows:",
    "PASS" if rows_pass else "FAIL"
)

print(
    "Columns:",
    "PASS" if columns_pass else "FAIL"
)


# ============================================================
# 4. AGE / PHI
# ============================================================

print("\n[4/12] Checking age and PHI...")

ages = sorted(
    df[AGE_COL]
    .dropna()
    .unique()
    .tolist()
)

age_pass = ages == EXPECTED_AGES

above_phi = int(
    (df[AGE_COL] > PHI).sum()
)

below_p6 = int(
    (df[AGE_COL] < 6).sum()
)

print(
    f"Ages found: {ages}"
)

print(
    "P6-P66 only:",
    "PASS" if age_pass else "FAIL"
)

print(
    f"Rows above P66: {above_phi:,}"
)

print(
    "No rows > P66:",
    "PASS" if above_phi == 0 else "FAIL"
)

print(
    f"Rows below P6: {below_p6:,}"
)


# ============================================================
# 5. MISSING / INFINITE VALUES
# ============================================================

print("\n[5/12] Checking NaN and infinite values...")

required_columns = FEATURES + TARGETS

missing = (
    df[required_columns]
    .isna()
    .sum()
)

total_missing = int(
    missing.sum()
)

numeric_values = df[required_columns].select_dtypes(
    include=[np.number]
)

infinite_count = int(
    np.isinf(
        numeric_values.to_numpy()
    ).sum()
)

print(
    f"Missing values: {total_missing:,}"
)

if total_missing > 0:
    print(
        missing[
            missing > 0
        ]
    )

print(
    f"Infinite values: {infinite_count:,}"
)

missing_pass = total_missing == 0
infinite_pass = infinite_count == 0

print(
    "No missing values:",
    "PASS" if missing_pass else "FAIL"
)

print(
    "No infinite values:",
    "PASS" if infinite_pass else "FAIL"
)


# ============================================================
# 6. TARGET RANGES
# ============================================================

print("\n[6/12] Checking target ranges...")

dbh_negative = int(
    (df["dbh_cm"] <= 0).sum()
)

height_negative = int(
    (df["height_m"] <= 0).sum()
)

volume_negative = int(
    (df["volume_m3_tree"] < 0).sum()
)

print(
    f"DBH <= 0       : {dbh_negative:,}"
)

print(
    f"Height <= 0    : {height_negative:,}"
)

print(
    f"Volume < 0     : {volume_negative:,}"
)

target_range_pass = (
    dbh_negative == 0
    and height_negative == 0
    and volume_negative == 0
)

print(
    "Target ranges:",
    "PASS"
    if target_range_pass
    else "FAIL"
)


# ============================================================
# 7. GROUP / AGE STRUCTURE
# ============================================================

print("\n[7/12] Checking group-age structure...")

group_age = (
    df.groupby(
        GROUP_COLS + [AGE_COL]
    )
    .size()
)

growth_groups = (
    df[GROUP_COLS]
    .drop_duplicates()
    .shape[0]
)

group_age_pass = (
    group_age.nunique() == 1
    and group_age.iloc[0] == 30
)

print(
    f"Growth groups : {growth_groups:,}"
)

print(
    f"Group-age rows: {len(group_age):,}"
)

print(
    f"Observations/group-age: "
    f"{sorted(group_age.unique())}"
)

print(
    "30 observations/group-age:",
    "PASS" if group_age_pass else "FAIL"
)


# ============================================================
# 8. BUILD GROUP-AGE MEANS
# ============================================================

print("\n[8/12] Building group-age trajectories...")

trajectory = (
    df.groupby(
        GROUP_COLS + [AGE_COL],
        as_index=False
    )
    .agg(
        dbh_mean=(
            "dbh_cm",
            "mean"
        ),
        height_mean=(
            "height_m",
            "mean"
        ),
        volume_mean=(
            "volume_m3_tree",
            "mean"
        ),
    )
    .sort_values(
        GROUP_COLS + [AGE_COL]
    )
)

print(
    f"Trajectory rows: {len(trajectory):,}"
)


# ============================================================
# 9. DBH / HEIGHT MONOTONICITY
# ============================================================

print(
    "\n[9/12] Checking DBH and Height trajectories..."
)

trajectory["prev_dbh"] = (
    trajectory
    .groupby(GROUP_COLS)["dbh_mean"]
    .shift(1)
)

trajectory["prev_height"] = (
    trajectory
    .groupby(GROUP_COLS)["height_mean"]
    .shift(1)
)

trajectory["dbh_change"] = (
    trajectory["dbh_mean"]
    - trajectory["prev_dbh"]
)

trajectory["height_change"] = (
    trajectory["height_mean"]
    - trajectory["prev_height"]
)

dbh_transition = (
    trajectory["prev_dbh"].notna()
)

height_transition = (
    trajectory["prev_height"].notna()
)

dbh_decrease = int(
    (
        trajectory.loc[
            dbh_transition,
            "dbh_change"
        ] < -1e-9
    ).sum()
)

height_decrease = int(
    (
        trajectory.loc[
            height_transition,
            "height_change"
        ] < -1e-9
    ).sum()
)

dbh_stagnation = int(
    (
        trajectory.loc[
            dbh_transition,
            "dbh_change"
        ].abs() <= 1e-9
    ).sum()
)

height_stagnation = int(
    (
        trajectory.loc[
            height_transition,
            "height_change"
        ].abs() <= 1e-9
    ).sum()
)

print(
    f"DBH decreases      : {dbh_decrease:,}"
)

print(
    f"DBH stagnations    : {dbh_stagnation:,}"
)

print(
    f"Height decreases   : {height_decrease:,}"
)

print(
    f"Height stagnations : {height_stagnation:,}"
)

dbh_monotonic_pass = (
    dbh_decrease == 0
)

height_monotonic_pass = (
    height_decrease == 0
)

print(
    "DBH never decreases:",
    "PASS"
    if dbh_monotonic_pass
    else "FAIL"
)

print(
    "Height never decreases:",
    "PASS"
    if height_monotonic_pass
    else "FAIL"
)


# ============================================================
# 10. GROWTH SPEED
# ============================================================

print(
    "\n[10/12] Checking growth-speed pattern..."
)

# Calculate mean annualized change
# for the group-age transitions.

trajectory["dbh_growth_rate"] = (
    trajectory["dbh_change"] / 0.5
)

trajectory["height_growth_rate"] = (
    trajectory["height_change"] / 0.5
)


def stage_mean(
    frame,
    column,
    start_age,
    end_age,
):
    mask = (
        frame["prev_dbh"].notna()
        & (frame[AGE_COL] >= start_age)
        & (frame[AGE_COL] <= end_age)
    )

    return frame.loc[
        mask,
        column
    ].mean()


# ------------------------------------------------------------
# DBH
# ------------------------------------------------------------

dbh_early = stage_mean(
    trajectory,
    "dbh_growth_rate",
    12,
    30,
)

dbh_middle = stage_mean(
    trajectory,
    "dbh_growth_rate",
    36,
    48,
)

dbh_late = stage_mean(
    trajectory,
    "dbh_growth_rate",
    54,
    66,
)


# ------------------------------------------------------------
# HEIGHT
# ------------------------------------------------------------

height_early = stage_mean(
    trajectory,
    "height_growth_rate",
    12,
    30,
)

height_middle = stage_mean(
    trajectory,
    "height_growth_rate",
    36,
    48,
)

height_late = stage_mean(
    trajectory,
    "height_growth_rate",
    54,
    66,
)


print("\nDBH growth rate (cm/year):")

print(
    f"  Early  P12-P30 : {dbh_early:.6f}"
)

print(
    f"  Middle P36-P48 : {dbh_middle:.6f}"
)

print(
    f"  Late   P54-P66 : {dbh_late:.6f}"
)


print("\nHeight growth rate (m/year):")

print(
    f"  Early  P12-P30 : {height_early:.6f}"
)

print(
    f"  Middle P36-P48 : {height_middle:.6f}"
)

print(
    f"  Late   P54-P66 : {height_late:.6f}"
)


dbh_slowdown_pass = (
    dbh_early > dbh_middle
    and dbh_middle > dbh_late
)

height_slowdown_pass = (
    height_early > height_middle
    and height_middle > height_late
)

print(
    "\nDBH early > middle > late:",
    "PASS" if dbh_slowdown_pass else "FAIL"
)

print(
    "Height early > middle > late:",
    "PASS"
    if height_slowdown_pass
    else "FAIL"
)


# ============================================================
# 11. VOLUME LOGIC
# ============================================================

print(
    "\n[11/12] Checking volume logic..."
)

# Volume prediction starts from P18.
# Therefore audit volume trajectories from P18 onward.

volume_trajectory = trajectory[
    trajectory[AGE_COL] >= 18
].copy()

volume_trajectory["prev_volume"] = (
    volume_trajectory
    .groupby(GROUP_COLS)["volume_mean"]
    .shift(1)
)

volume_trajectory["volume_change"] = (
    volume_trajectory["volume_mean"]
    - volume_trajectory["prev_volume"]
)

volume_valid_transition = (
    volume_trajectory["prev_volume"].notna()
)

volume_decrease = int(
    (
        volume_trajectory.loc[
            volume_valid_transition,
            "volume_change"
        ] < -1e-9
    ).sum()
)

volume_increase = int(
    (
        volume_trajectory.loc[
            volume_valid_transition,
            "volume_change"
        ] > 1e-9
    ).sum()
)

volume_stagnation = int(
    (
        volume_trajectory.loc[
            volume_valid_transition,
            "volume_change"
        ].abs() <= 1e-9
    ).sum()
)

print(
    f"Volume increases   : {volume_increase:,}"
)

print(
    f"Volume decreases   : {volume_decrease:,}"
)

print(
    f"Volume stagnations : {volume_stagnation:,}"
)

print(
    "Volume non-negative:",
    "PASS"
    if volume_negative == 0
    else "FAIL"
)

print(
    "Volume decreases allowed: YES"
)


# ============================================================
# 12. FINAL DECISION
# ============================================================

print("\n" + "=" * 80)
print("FINAL STEP 36U DECISION")
print("=" * 80)


checks = {
    "Schema": schema_pass,
    "Rows": rows_pass,
    "Columns": columns_pass,
    "P6-P66 only": age_pass,
    "No rows > P66": above_phi == 0,
    "No missing values": missing_pass,
    "No infinite values": infinite_pass,
    "Target ranges": target_range_pass,
    "30 rows/group-age": group_age_pass,
    "DBH no decrease": dbh_monotonic_pass,
    "Height no decrease": height_monotonic_pass,
    "DBH slowdown": dbh_slowdown_pass,
    "Height slowdown": height_slowdown_pass,
    "Volume non-negative": volume_negative == 0,
}


for name, result in checks.items():

    print(
        f"{name:<28}: "
        f"{'PASS' if result else 'FAIL'}"
    )


all_pass = all(checks.values())


print("\n" + "=" * 80)

if all_pass:

    print(
        "STEP 36U STATUS: PASS"
    )

    print(
        "\nFINAL DATASET IS READY FOR MODEL PREPARATION."
    )

else:

    print(
        "STEP 36U STATUS: REVIEW REQUIRED"
    )

    print(
        "\nDO NOT START MODEL TRAINING YET."
    )

print("=" * 80)

print(
    "\nDataset checked:"
)

print(INPUT_FILE)

print(
    "\nNo dataset values were modified by this script."
)