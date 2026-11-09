from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 36O
# AUDIT DBH ON FINAL PHI66 DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "dataset_phi66.csv"

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
print("STEP 36O - AUDIT DBH ON FINAL PHI66 DATASET")
print("=" * 75)


# ============================================================
# 1. LOAD
# ============================================================

print("\n[1/7] Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# ============================================================
# 2. BASIC VALIDATION
# ============================================================

print("\n[2/7] Validating DBH data...")

required = GROUP_COLS + [AGE_COL, DBH_COL]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

print("Required columns: PASS")

print(
    f"DBH minimum : {df[DBH_COL].min():.6f}"
)

print(
    f"DBH maximum : {df[DBH_COL].max():.6f}"
)

print(
    f"DBH mean    : {df[DBH_COL].mean():.6f}"
)


# ============================================================
# 3. GROUP-AGE AGGREGATION
# ============================================================

print("\n[3/7] Creating group-age DBH trajectory...")

trajectory = (
    df.groupby(
        GROUP_COLS + [AGE_COL],
        as_index=False
    )
    .agg(
        dbh_mean=(DBH_COL, "mean"),
        dbh_median=(DBH_COL, "median"),
        dbh_std=(DBH_COL, "std"),
        dbh_min=(DBH_COL, "min"),
        dbh_max=(DBH_COL, "max"),
        observation_count=(DBH_COL, "size"),
    )
    .sort_values(
        GROUP_COLS + [AGE_COL]
    )
)

print(
    f"Growth groups : "
    f"{trajectory[GROUP_COLS].drop_duplicates().shape[0]:,}"
)

print(
    f"Group-age rows: "
    f"{len(trajectory):,}"
)

print(
    "\nObservations/group-age:"
)

print(
    trajectory["observation_count"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 4. AGE STRUCTURE
# ============================================================

print("\n[4/7] Checking age structure...")

ages = sorted(
    trajectory[AGE_COL]
    .unique()
    .tolist()
)

print("Ages found:")

for age in ages:

    label = (
        "PHI"
        if age == 66
        else f"P{int(age)}"
    )

    rows = (
        trajectory[AGE_COL] == age
    ).sum()

    print(
        f"{label:>4} : "
        f"{rows:,} group-age rows"
    )

if ages != EXPECTED_AGES:
    raise ValueError(
        f"Unexpected ages: {ages}"
    )


# ============================================================
# 5. GROUP-LEVEL TRANSITIONS
# ============================================================

print("\n[5/7] Checking DBH transitions...")

trajectory["previous_age"] = (
    trajectory
    .groupby(GROUP_COLS)[AGE_COL]
    .shift(1)
)

trajectory["previous_dbh"] = (
    trajectory
    .groupby(GROUP_COLS)["dbh_mean"]
    .shift(1)
)

trajectory["dbh_change"] = (
    trajectory["dbh_mean"]
    - trajectory["previous_dbh"]
)

valid = (
    trajectory["previous_dbh"]
    .notna()
)

increases = trajectory[
    valid
    & (trajectory["dbh_change"] > 1e-9)
]

stagnations = trajectory[
    valid
    & (
        trajectory["dbh_change"].abs()
        <= 1e-9
    )
]

decreases = trajectory[
    valid
    & (trajectory["dbh_change"] < -1e-9)
]

print(
    f"Valid transitions : {valid.sum():,}"
)

print(
    f"Increases         : {len(increases):,}"
)

print(
    f"Stagnations       : {len(stagnations):,}"
)

print(
    f"Decreases         : {len(decreases):,}"
)


# ============================================================
# 6. GLOBAL DBH PROFILE
# ============================================================

print("\n[6/7] Building DBH age profile...")

profile = (
    trajectory
    .groupby(AGE_COL)["dbh_mean"]
    .agg(
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
    )
    .sort_index()
)

profile["delta"] = profile["mean"].diff()

print("\nGLOBAL DBH AGE PROFILE")
print("-" * 75)

for age, row in profile.iterrows():

    label = (
        "PHI"
        if age == 66
        else f"P{int(age)}"
    )

    if pd.isna(row["delta"]):
        delta = "N/A"
    else:
        delta = f"{row['delta']:+.6f}"

    print(
        f"{label:>5} | "
        f"mean={row['mean']:.6f} | "
        f"median={row['median']:.6f} | "
        f"delta={delta}"
    )


# ============================================================
# 7. GROWTH STAGE AUDIT
# ============================================================

print("\n[7/7] Auditing DBH growth stages...")


def stage_rate(start_age, end_age):

    subset = profile[
        (profile.index >= start_age)
        & (profile.index <= end_age)
    ]

    changes = (
        subset["delta"]
        .dropna()
    )

    if changes.empty:
        return np.nan

    return changes.mean()


early = stage_rate(6, 30)
middle = stage_rate(36, 60)
late = stage_rate(66, 66)


print("\nSTAGE SUMMARY")
print("-" * 75)

print(
    f"EARLY   P6-P30  : "
    f"{early:.8f}"
)

print(
    f"MIDDLE  P36-P60 : "
    f"{middle:.8f}"
)

# P66 is the harvest endpoint, so we do not
# calculate a P66-P300 late stage.
#
# Instead, inspect the final PHI increment.

phi_increment = profile.loc[66, "delta"]

print(
    f"PHI increment   : "
    f"{phi_increment:.8f}"
)


# ============================================================
# CRITICAL TRANSITIONS
# ============================================================

print("\nCRITICAL DBH TRANSITIONS")
print("-" * 75)

critical_pairs = [
    (6, 12),
    (12, 18),
    (18, 24),
    (24, 30),
    (30, 36),
    (36, 42),
    (42, 48),
    (48, 54),
    (54, 60),
    (60, 66),
]


def transition_stats(age_a, age_b):

    a = trajectory[
        trajectory[AGE_COL] == age_a
    ][GROUP_COLS + ["dbh_mean"]]

    b = trajectory[
        trajectory[AGE_COL] == age_b
    ][GROUP_COLS + ["dbh_mean"]]

    merged = a.merge(
        b,
        on=GROUP_COLS,
        suffixes=("_a", "_b"),
    )

    if merged.empty:
        return None

    merged["change"] = (
        merged["dbh_mean_b"]
        - merged["dbh_mean_a"]
    )

    return merged


for age_a, age_b in critical_pairs:

    result = transition_stats(
        age_a,
        age_b
    )

    if result is None:
        continue

    print(
        f"\nP{age_a} -> "
        f"P{age_b}"
    )

    print(
        f"  groups      : "
        f"{len(result):,}"
    )

    print(
        f"  mean change : "
        f"{result['change'].mean():+.6f}"
    )

    print(
        f"  median      : "
        f"{result['change'].median():+.6f}"
    )

    print(
        f"  min         : "
        f"{result['change'].min():+.6f}"
    )

    print(
        f"  max         : "
        f"{result['change'].max():+.6f}"
    )

    print(
        f"  decreases   : "
        f"{(result['change'] < -1e-9).sum():,}"
    )


# ============================================================
# FINAL DECISION
# ============================================================

print("\n" + "=" * 75)
print("FINAL DBH AUDIT")
print("=" * 75)

no_decrease = (
    len(decreases) == 0
)

early_gt_middle = (
    early > middle
)

phi_positive = (
    phi_increment > 0
)

print(
    "DBH never decreases :",
    "PASS"
    if no_decrease
    else "FAIL"
)

print(
    "Early > Middle      :",
    "PASS"
    if early_gt_middle
    else "FAIL"
)

print(
    "PHI increment > 0   :",
    "PASS"
    if phi_positive
    else "FAIL"
)

print(
    "\nDataset was NOT modified."
)

print("=" * 75)