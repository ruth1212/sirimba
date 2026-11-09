from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# STEP 36M
# FINAL AUDIT - HEIGHT GROWTH LOGIC V2
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "dataset_growth_logic_v2.csv"

GROUP_COLS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]

AGE_COL = "age_months"
HEIGHT_COL = "height_m"


print("=" * 75)
print("STEP 36M - FINAL AUDIT HEIGHT GROWTH LOGIC V2")
print("=" * 75)


# ============================================================
# 1. LOAD
# ============================================================

print("\n[1/7] Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# ============================================================
# 2. GROUP-AGE AGGREGATION
# ============================================================

print("\n[2/7] Creating group-age trajectory...")

trajectory = (
    df.groupby(GROUP_COLS + [AGE_COL], as_index=False)
      .agg(
          height_mean=(HEIGHT_COL, "mean"),
          height_median=(HEIGHT_COL, "median"),
          height_std=(HEIGHT_COL, "std"),
          height_min=(HEIGHT_COL, "min"),
          height_max=(HEIGHT_COL, "max"),
          observation_count=(HEIGHT_COL, "size"),
      )
      .sort_values(GROUP_COLS + [AGE_COL])
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
    f"Rows/group-age distribution:\n"
    f"{trajectory['observation_count'].value_counts().sort_index()}"
)


# ============================================================
# 3. PER-GROUP TRANSITION AUDIT
# ============================================================

print("\n[3/7] Auditing age-to-age transitions...")

trajectory["previous_age"] = (
    trajectory.groupby(GROUP_COLS)[AGE_COL]
    .shift(1)
)

trajectory["previous_height"] = (
    trajectory.groupby(GROUP_COLS)["height_mean"]
    .shift(1)
)

trajectory["height_change"] = (
    trajectory["height_mean"]
    - trajectory["previous_height"]
)

valid = trajectory["previous_height"].notna()

decreases = trajectory.loc[
    valid & (trajectory["height_change"] < -1e-9)
]

stagnations = trajectory.loc[
    valid & (trajectory["height_change"].abs() <= 1e-9)
]

increases = trajectory.loc[
    valid & (trajectory["height_change"] > 1e-9)
]

print(f"Valid transitions : {valid.sum():,}")
print(f"Increases         : {len(increases):,}")
print(f"Stagnations       : {len(stagnations):,}")
print(f"Decreases         : {len(decreases):,}")


# ============================================================
# 4. GLOBAL AGE PROFILE
# ============================================================

print("\n[4/7] Building global age profile...")

profile = (
    trajectory
    .groupby(AGE_COL)["height_mean"]
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

print("\nGLOBAL AGE PROFILE")
print("-" * 75)

for age, row in profile.iterrows():

    label = "PHI" if age == 66 else f"P{int(age)}"

    if pd.isna(row["delta"]):
        delta_text = "N/A"
    else:
        delta_text = f"{row['delta']:+.6f}"

    print(
        f"{label:>5} | "
        f"mean={row['mean']:.6f} | "
        f"median={row['median']:.6f} | "
        f"delta={delta_text}"
    )


# ============================================================
# 5. IMPORTANT AGE TRANSITIONS
# ============================================================

print("\n[5/7] Inspecting critical transitions...")

critical_pairs = [
    (54, 60),
    (60, 66),
    (66, 72),
    (72, 78),
    (120, 126),
    (174, 180),
    (180, 186),
    (234, 240),
    (240, 246),
    (294, 300),
]


def transition_stats(age_a, age_b):

    a = trajectory[
        trajectory[AGE_COL] == age_a
    ][GROUP_COLS + ["height_mean"]].copy()

    b = trajectory[
        trajectory[AGE_COL] == age_b
    ][GROUP_COLS + ["height_mean"]].copy()

    merged = a.merge(
        b,
        on=GROUP_COLS,
        suffixes=("_a", "_b"),
    )

    if merged.empty:
        return None

    merged["change"] = (
        merged["height_mean_b"]
        - merged["height_mean_a"]
    )

    return merged


for age_a, age_b in critical_pairs:

    result = transition_stats(age_a, age_b)

    if result is None:
        print(f"\nP{age_a} -> P{age_b}: NO DATA")
        continue

    print(f"\nP{age_a} -> P{age_b}")
    print(
        f"  groups          : {len(result):,}"
    )
    print(
        f"  mean change     : "
        f"{result['change'].mean():+.6f}"
    )
    print(
        f"  median change   : "
        f"{result['change'].median():+.6f}"
    )
    print(
        f"  min change      : "
        f"{result['change'].min():+.6f}"
    )
    print(
        f"  max change      : "
        f"{result['change'].max():+.6f}"
    )
    print(
        f"  decreases       : "
        f"{(result['change'] < -1e-9).sum():,}"
    )
    print(
        f"  stagnations     : "
        f"{(result['change'].abs() <= 1e-9).sum():,}"
    )


# ============================================================
# 6. GROWTH STAGE AUDIT
# ============================================================

print("\n[6/7] Auditing growth stages...")


def stage_rate(start_age, end_age):

    subset = profile[
        (profile.index >= start_age)
        & (profile.index <= end_age)
    ]

    changes = subset["delta"].dropna()

    if len(changes) == 0:
        return np.nan

    return changes.mean()


early = stage_rate(6, 30)
middle = stage_rate(36, 60)
late = stage_rate(66, 300)


print("\nSTAGE SUMMARY")
print("-" * 75)
print(f"EARLY   P6-P30   : {early:.8f}")
print(f"MIDDLE  P36-P60  : {middle:.8f}")
print(f"LATE    P66-P300 : {late:.8f}")


# ============================================================
# 7. FINAL LOGIC DECISION
# ============================================================

print("\n" + "=" * 75)
print("FINAL LOGIC CHECK")
print("=" * 75)


no_decrease = len(decreases) == 0

fast_to_moderate = (
    early > middle
)

moderate_to_slow = (
    middle > late
)


# Critical global transitions
global_decrease_count = (
    profile["delta"].dropna() < -1e-9
).sum()


print(
    f"No decrease within growth groups : "
    f"{'PASS' if no_decrease else 'FAIL'}"
)

print(
    f"Fast > Moderate                  : "
    f"{'PASS' if fast_to_moderate else 'FAIL'}"
)

print(
    f"Moderate > Slow                  : "
    f"{'PASS' if moderate_to_slow else 'FAIL'}"
)

print(
    f"Global profile decreases         : "
    f"{global_decrease_count}"
)


# PHI
phi = trajectory[
    trajectory[AGE_COL] == 66
]["height_mean"]

print(
    f"\nPHI (66 months) mean : "
    f"{phi.mean():.6f} m"
)

print(
    f"PHI (66 months) median : "
    f"{phi.median():.6f} m"
)


# ============================================================
# FINAL DECISION
# ============================================================

print("\n" + "=" * 75)

if (
    no_decrease
    and fast_to_moderate
    and moderate_to_slow
    and global_decrease_count == 0
):
    print("FINAL DATASET HEIGHT LOGIC: PASS")
else:
    print("FINAL DATASET HEIGHT LOGIC: REVIEW REQUIRED")

print("=" * 75)

print("\nIMPORTANT:")
print("This script ONLY audits the dataset.")
print("The dataset was NOT modified.")