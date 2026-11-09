from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 36S
# FINAL DATASET INTEGRITY CHECK
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "dataset_phi66_dbh_v2.csv"

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

MODEL_FEATURES = [
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha",
]

EXPECTED_AGES = list(range(6, 67, 6))
PHI = 66


print("=" * 75)
print("STEP 36S - FINAL DATASET INTEGRITY CHECK")
print("=" * 75)


# ============================================================
# 1. LOAD
# ============================================================

print("\n[1/9] Loading final candidate dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# ============================================================
# 2. SCHEMA
# ============================================================

print("\n[2/9] Checking schema...")

raw_file = BASE_DIR / "data" / "dataset.csv"

raw_columns = pd.read_csv(
    raw_file,
    nrows=0
).columns.tolist()

final_columns = df.columns.tolist()

schema_same = (
    raw_columns == final_columns
)

print(
    "Same column names/order as raw:",
    "PASS" if schema_same else "FAIL"
)

if not schema_same:
    print("\nRaw columns:")
    print(raw_columns)

    print("\nFinal columns:")
    print(final_columns)


# ============================================================
# 3. ROW / COLUMN COUNT
# ============================================================

print("\n[3/9] Checking dimensions...")

print(
    f"Rows expected : 218,700"
)

print(
    f"Rows actual   : {len(df):,}"
)

print(
    "Row count:",
    "PASS"
    if len(df) == 218700
    else "FAIL"
)

print(
    "Column count:",
    "PASS"
    if len(df.columns) == 31
    else "FAIL"
)


# ============================================================
# 4. AGE STRUCTURE
# ============================================================

print("\n[4/9] Checking age structure...")

ages = sorted(
    df[AGE_COL]
    .dropna()
    .unique()
    .tolist()
)

print(
    f"Age range: "
    f"P{int(min(ages))} - P{int(max(ages))}"
)

print(
    f"Ages found: {ages}"
)

print(
    "Age structure:",
    "PASS"
    if ages == EXPECTED_AGES
    else "FAIL"
)

rows_above_phi = (
    df[AGE_COL] > PHI
).sum()

print(
    f"Rows above P66: {rows_above_phi:,}"
)

print(
    "PHI boundary:",
    "PASS"
    if rows_above_phi == 0
    else "FAIL"
)


# ============================================================
# 5. MISSING VALUES
# ============================================================

print("\n[5/9] Checking missing values...")

required_for_model = (
    MODEL_FEATURES
    + TARGETS
)

missing_summary = (
    df[required_for_model]
    .isna()
    .sum()
)

total_missing = int(
    missing_summary.sum()
)

print(
    f"Total missing values "
    f"in model fields: {total_missing:,}"
)

if total_missing > 0:
    print(
        missing_summary[
            missing_summary > 0
        ]
    )

print(
    "Model fields complete:",
    "PASS"
    if total_missing == 0
    else "FAIL"
)


# ============================================================
# 6. TARGET VALUE VALIDATION
# ============================================================

print("\n[6/9] Checking target values...")

for target in TARGETS:

    series = df[target]

    print(f"\n{target}")

    print(
        f"  min    : {series.min():.8f}"
    )

    print(
        f"  max    : {series.max():.8f}"
    )

    print(
        f"  mean   : {series.mean():.8f}"
    )

    print(
        f"  median : {series.median():.8f}"
    )

    print(
        f"  negative: "
        f"{(series < 0).sum():,}"
    )


# DBH
dbh_valid = (
    df["dbh_cm"] > 0
).all()

# Height
height_valid = (
    df["height_m"] > 0
).all()

# Volume
volume_valid = (
    df["volume_m3_tree"] >= 0
).all()

print("\nTarget positivity checks:")

print(
    "DBH > 0:",
    "PASS" if dbh_valid else "FAIL"
)

print(
    "Height > 0:",
    "PASS" if height_valid else "FAIL"
)

print(
    "Volume >= 0:",
    "PASS" if volume_valid else "FAIL"
)


# ============================================================
# 7. GROUP-AGE STRUCTURE
# ============================================================

print("\n[7/9] Checking group-age structure...")

group_age = (
    df.groupby(
        GROUP_COLS + [AGE_COL]
    )
    .size()
)

group_count = (
    df[GROUP_COLS]
    .drop_duplicates()
    .shape[0]
)

print(
    f"Growth groups : {group_count:,}"
)

print(
    f"Group-age rows: {len(group_age):,}"
)

print(
    "\nObservations per group-age:"
)

print(
    group_age
    .value_counts()
    .sort_index()
)

group_age_uniform = (
    group_age.nunique() == 1
    and group_age.iloc[0] == 30
)

print(
    "\n30 observations/group-age:",
    "PASS"
    if group_age_uniform
    else "FAIL"
)


# ============================================================
# 8. DBH / HEIGHT TRAJECTORY
# ============================================================

print("\n[8/9] Checking DBH and Height trajectories...")

trajectory = (
    df.groupby(
        GROUP_COLS + [AGE_COL],
        as_index=False
    )
    .agg(
        dbh_mean=("dbh_cm", "mean"),
        height_mean=("height_m", "mean"),
        volume_mean=("volume_m3_tree", "mean"),
    )
    .sort_values(
        GROUP_COLS + [AGE_COL]
    )
)

trajectory["previous_dbh"] = (
    trajectory
    .groupby(GROUP_COLS)["dbh_mean"]
    .shift(1)
)

trajectory["previous_height"] = (
    trajectory
    .groupby(GROUP_COLS)["height_mean"]
    .shift(1)
)

trajectory["dbh_change"] = (
    trajectory["dbh_mean"]
    - trajectory["previous_dbh"]
)

trajectory["height_change"] = (
    trajectory["height_mean"]
    - trajectory["previous_height"]
)

valid_dbh = (
    trajectory["previous_dbh"].notna()
)

valid_height = (
    trajectory["previous_height"].notna()
)

dbh_decreases = (
    trajectory.loc[
        valid_dbh,
        "dbh_change"
    ] < -1e-9
).sum()

height_decreases = (
    trajectory.loc[
        valid_height,
        "height_change"
    ] < -1e-9
).sum()

print(
    f"DBH trajectory decreases    : "
    f"{dbh_decreases:,}"
)

print(
    f"Height trajectory decreases  : "
    f"{height_decreases:,}"
)

print(
    "DBH no decrease:",
    "PASS"
    if dbh_decreases == 0
    else "FAIL"
)

print(
    "Height no decrease:",
    "PASS"
    if height_decreases == 0
    else "FAIL"
)


# ============================================================
# 9. FINAL DECISION
# ============================================================

print("\n" + "=" * 75)
print("FINAL DATASET INTEGRITY DECISION")
print("=" * 75)

checks = {
    "Schema": schema_same,
    "Rows": len(df) == 218700,
    "Columns": len(df.columns) == 31,
    "Age structure": ages == EXPECTED_AGES,
    "No rows > PHI": rows_above_phi == 0,
    "Model fields complete": total_missing == 0,
    "DBH positive": dbh_valid,
    "Height positive": height_valid,
    "Volume non-negative": volume_valid,
    "30 rows/group-age": group_age_uniform,
    "DBH no decrease": dbh_decreases == 0,
    "Height no decrease": height_decreases == 0,
}

for name, result in checks.items():

    print(
        f"{name:<28}: "
        f"{'PASS' if result else 'FAIL'}"
    )


all_pass = all(checks.values())

print("\n" + "=" * 75)

if all_pass:
    print("FINAL DATASET STATUS: PASS")
    print("dataset_phi66_dbh_v2.csv is READY for model preparation.")
else:
    print("FINAL DATASET STATUS: REVIEW REQUIRED")

print("=" * 75)

print("\nDataset was NOT modified.")
print(
    f"Checked file: {INPUT_FILE}"
)