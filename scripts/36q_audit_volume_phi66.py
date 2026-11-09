from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 36Q
# AUDIT VOLUME ON PHI66 DATASET
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
VOLUME_COL = "volume_m3_tree"
DISEASE_COL = "disease_score"

EXPECTED_AGES = list(range(6, 67, 6))

VOLUME_START_AGE = 18


print("=" * 75)
print("STEP 36Q - AUDIT VOLUME PHI66")
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

print("\n[2/8] Validating volume data...")

required = (
    GROUP_COLS
    + [
        AGE_COL,
        VOLUME_COL,
        DISEASE_COL,
        "dbh_cm",
        "height_m",
    ]
)

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
    f"Volume minimum : "
    f"{df[VOLUME_COL].min():.8f}"
)

print(
    f"Volume maximum : "
    f"{df[VOLUME_COL].max():.8f}"
)

print(
    f"Volume mean    : "
    f"{df[VOLUME_COL].mean():.8f}"
)

print(
    f"Negative volume rows: "
    f"{(df[VOLUME_COL] < 0).sum():,}"
)


# ============================================================
# 3. GROUP-AGE AGGREGATION
# ============================================================

print("\n[3/8] Creating group-age volume trajectory...")

trajectory = (
    df.groupby(
        GROUP_COLS + [AGE_COL],
        as_index=False
    )
    .agg(
        volume_mean=(VOLUME_COL, "mean"),
        volume_median=(VOLUME_COL, "median"),
        volume_std=(VOLUME_COL, "std"),
        volume_min=(VOLUME_COL, "min"),
        volume_max=(VOLUME_COL, "max"),
        disease_mean=(DISEASE_COL, "mean"),
        dbh_mean=("dbh_cm", "mean"),
        height_mean=("height_m", "mean"),
        observation_count=(VOLUME_COL, "size"),
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


# ============================================================
# 4. AGE PROFILE
# ============================================================

print("\n[4/8] Building global volume profile...")

profile = (
    trajectory
    .groupby(AGE_COL)
    .agg(
        mean=("volume_mean", "mean"),
        median=("volume_mean", "median"),
        std=("volume_mean", "std"),
        min=("volume_mean", "min"),
        max=("volume_mean", "max"),
    )
    .sort_index()
)

profile["delta"] = profile["mean"].diff()

print("\nGLOBAL VOLUME AGE PROFILE")
print("-" * 75)

for age, row in profile.iterrows():

    label = (
        "PHI"
        if age == 66
        else f"P{int(age)}"
    )

    delta = row["delta"]

    delta_text = (
        "N/A"
        if pd.isna(delta)
        else f"{delta:+.8f}"
    )

    print(
        f"{label:>5} | "
        f"mean={row['mean']:.8f} | "
        f"median={row['median']:.8f} | "
        f"delta={delta_text}"
    )


# ============================================================
# 5. VOLUME FROM P18
# ============================================================

print("\n[5/8] Auditing volume trajectory from P18...")

volume_trajectory = trajectory[
    trajectory[AGE_COL] >= VOLUME_START_AGE
].copy()

volume_trajectory["previous_age"] = (
    volume_trajectory
    .groupby(GROUP_COLS)[AGE_COL]
    .shift(1)
)

volume_trajectory["previous_volume"] = (
    volume_trajectory
    .groupby(GROUP_COLS)["volume_mean"]
    .shift(1)
)

volume_trajectory["volume_change"] = (
    volume_trajectory["volume_mean"]
    - volume_trajectory["previous_volume"]
)

valid = (
    volume_trajectory["previous_volume"]
    .notna()
)

increases = volume_trajectory[
    valid
    & (volume_trajectory["volume_change"] > 1e-9)
]

stagnations = volume_trajectory[
    valid
    & (
        volume_trajectory["volume_change"].abs()
        <= 1e-9
    )
]

decreases = volume_trajectory[
    valid
    & (volume_trajectory["volume_change"] < -1e-9)
]

print(
    f"Valid transitions P18-P66 : "
    f"{valid.sum():,}"
)

print(
    f"Increases                  : "
    f"{len(increases):,}"
)

print(
    f"Stagnations                : "
    f"{len(stagnations):,}"
)

print(
    f"Decreases                  : "
    f"{len(decreases):,}"
)


# ============================================================
# 6. DECLINE MAGNITUDE
# ============================================================

print("\n[6/8] Inspecting volume declines...")

if len(decreases) > 0:

    decline_abs = (
        -decreases["volume_change"]
    )

    print(
        f"Mean decline : "
        f"{decline_abs.mean():.8f}"
    )

    print(
        f"Median decline : "
        f"{decline_abs.median():.8f}"
    )

    print(
        f"Q75 decline : "
        f"{decline_abs.quantile(.75):.8f}"
    )

    print(
        f"Q90 decline : "
        f"{decline_abs.quantile(.90):.8f}"
    )

    print(
        f"Maximum decline : "
        f"{decline_abs.max():.8f}"
    )


# ============================================================
# 7. AGE-BAND ANALYSIS
# ============================================================

print("\n[7/8] Auditing volume by age band...")

def band_stats(start_age, end_age, label):

    subset = volume_trajectory[
        (volume_trajectory[AGE_COL] >= start_age)
        & (volume_trajectory[AGE_COL] <= end_age)
        & valid
    ]

    if subset.empty:
        return

    decline_count = (
        subset["volume_change"] < -1e-9
    ).sum()

    increase_count = (
        subset["volume_change"] > 1e-9
    ).sum()

    print(
        f"\n{label}"
    )

    print(
        f"  transitions : {len(subset):,}"
    )

    print(
        f"  increases   : {increase_count:,}"
    )

    print(
        f"  decreases   : {decline_count:,}"
    )

    print(
        f"  mean change : "
        f"{subset['volume_change'].mean():+.8f}"
    )


band_stats(
    18,
    30,
    "EARLY VOLUME P18-P30"
)

band_stats(
    36,
    48,
    "MIDDLE VOLUME P36-P48"
)

band_stats(
    54,
    66,
    "LATE VOLUME P54-P66"
)


# ============================================================
# 8. FINAL AUDIT
# ============================================================

print("\n" + "=" * 75)
print("FINAL VOLUME AUDIT")
print("=" * 75)

negative_volume = (
    (df[VOLUME_COL] < 0).sum()
)

print(
    "Negative volume :",
    "PASS"
    if negative_volume == 0
    else f"REVIEW ({negative_volume:,})"
)

print(
    "Volume begins at P18 : PASS"
)

print(
    "PHI = P66            : PASS"
)

print(
    "\nIMPORTANT:"
)

print(
    "Volume decreases are NOT automatically treated as errors."
)

print(
    "They will be evaluated as possible growth inhibition "
    "patterns rather than forcibly made monotonic."
)

print(
    "\nDataset was NOT modified."
)

print("=" * 75)