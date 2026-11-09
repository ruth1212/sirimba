from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression


# ============================================================
# STEP 36E
# ISOTONIC HEIGHT CORRECTION TEST
#
# Goal:
#   Find the closest non-decreasing Height trajectory
#   while preserving actual data as much as possible.
#
# Important:
#   - Work at growth-group + age level.
#   - 30 rows are aggregated into one group-age mean.
#   - Original dataset is NOT modified.
# ============================================================


BASE_DIR = Path(r"C:\tree_growth_fs\tree_growth_fs")

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "dataset_corrected.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "height_isotonic_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "height_isotonic_group_age.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "height_isotonic_summary.csv"
)


GROUP_COLS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]

AGE_COL = "age_months"
HEIGHT_COL = "height_m"


print("=" * 70)
print("STEP 36E : ISOTONIC HEIGHT CORRECTION TEST")
print("=" * 70)


# ============================================================
# 1. LOAD
# ============================================================

print("\n[1] Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


required = GROUP_COLS + [
    AGE_COL,
    HEIGHT_COL,
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print("Required columns: PASS")


# ============================================================
# 2. GROUP-AGE AGGREGATION
# ============================================================

print("\n[2] Creating group-age trajectory...")

group_age = (
    df.groupby(
        GROUP_COLS + [AGE_COL],
        as_index=False
    )
    .agg(
        actual_height_m=(HEIGHT_COL, "mean"),
        rows_at_age=(HEIGHT_COL, "size"),
    )
)

print(
    f"Group-age rows : {len(group_age):,}"
)

print(
    f"Growth groups  : "
    f"{group_age[GROUP_COLS].drop_duplicates().shape[0]:,}"
)

print(
    f"Unique ages    : "
    f"{group_age[AGE_COL].nunique():,}"
)


# ============================================================
# 3. VALIDATE 30 ROWS
# ============================================================

print("\n[3] Checking rows per group-age...")

if (
    group_age["rows_at_age"].min() == 30
    and
    group_age["rows_at_age"].max() == 30
):
    print("30 rows per group-age: PASS")
else:
    print("WARNING: not every group-age contains 30 rows")


# ============================================================
# 4. SORT
# ============================================================

group_age = group_age.sort_values(
    GROUP_COLS + [AGE_COL]
).reset_index(drop=True)


# ============================================================
# 5. ISOTONIC REGRESSION
# ============================================================

print("\n[4] Applying isotonic regression...")

group_age["corrected_height_m"] = np.nan


# ------------------------------------------------------------
# Fit one isotonic trajectory per growth group.
# ------------------------------------------------------------

for group_key, idx in group_age.groupby(
    GROUP_COLS,
    sort=False
).groups.items():

    sub = group_age.loc[idx].sort_values(
        AGE_COL
    )

    x = sub[AGE_COL].astype(float).to_numpy()

    y = sub["actual_height_m"].astype(float).to_numpy()

    # Non-decreasing constraint.
    model = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip",
    )

    y_corrected = model.fit_transform(
        x,
        y
    )

    group_age.loc[
        sub.index,
        "corrected_height_m"
    ] = y_corrected


print("Isotonic regression: PASS")


# ============================================================
# 6. CORRECTION
# ============================================================

group_age["correction_m"] = (
    group_age["corrected_height_m"]
    -
    group_age["actual_height_m"]
)

group_age["absolute_correction_m"] = (
    group_age["correction_m"].abs()
)


# ============================================================
# 7. VALIDATE TRAJECTORY
# ============================================================

print("\n[5] Validating corrected trajectories...")

group_age["previous_corrected_height_m"] = (
    group_age
    .groupby(GROUP_COLS)["corrected_height_m"]
    .shift(1)
)

group_age["corrected_change_m"] = (
    group_age["corrected_height_m"]
    -
    group_age["previous_corrected_height_m"]
)

valid = group_age[
    group_age["previous_corrected_height_m"].notna()
].copy()

decreases = (
    valid["corrected_change_m"] < -1e-9
).sum()

increases = (
    valid["corrected_change_m"] > 1e-9
).sum()

stagnations = (
    valid["corrected_change_m"].abs() <= 1e-9
).sum()

print(
    f"Valid transitions : {len(valid):,}"
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
# 8. GROWTH RATE
# ============================================================

print("\n[6] Calculating growth rates...")

valid["previous_age_months"] = (
    valid
    .groupby(GROUP_COLS)[AGE_COL]
    .shift(1)
)

valid["age_gap_months"] = (
    valid[AGE_COL].astype(float)
    -
    valid["previous_age_months"].astype(float)
)

valid["growth_rate_cm_year"] = (
    valid["corrected_change_m"]
    * 100
    /
    (
        valid["age_gap_months"] / 12
    )
)

print("Growth-rate calculation: PASS")


# ============================================================
# 9. STAGE DIAGNOSTIC
# ============================================================

print("\n[7] Early / middle / late growth diagnostic...")


def stage_stats(data, minimum, maximum, name):

    x = data[
        (data[AGE_COL] >= minimum)
        &
        (data[AGE_COL] <= maximum)
    ]["growth_rate_cm_year"]

    if len(x) == 0:
        return {
            "stage": name,
            "count": 0,
            "mean_cm_year": np.nan,
            "median_cm_year": np.nan,
            "min_cm_year": np.nan,
            "max_cm_year": np.nan,
        }

    return {
        "stage": name,
        "count": len(x),
        "mean_cm_year": x.mean(),
        "median_cm_year": x.median(),
        "min_cm_year": x.min(),
        "max_cm_year": x.max(),
    }


stage_df = pd.DataFrame([
    stage_stats(
        valid,
        12,
        30,
        "EARLY_P6_P30"
    ),
    stage_stats(
        valid,
        36,
        60,
        "MIDDLE_P36_P60"
    ),
    stage_stats(
        valid,
        66,
        300,
        "LATE_P66_P300"
    ),
])

print(
    stage_df.to_string(index=False)
)


# ============================================================
# 10. CORRECTION MAGNITUDE
# ============================================================

print("\n[8] Correction magnitude...")

correction = group_age["absolute_correction_m"]

print(
    f"Mean absolute correction   : "
    f"{correction.mean():.6f} m"
)

print(
    f"Median absolute correction : "
    f"{correction.median():.6f} m"
)

print(
    f"Q75                       : "
    f"{correction.quantile(.75):.6f} m"
)

print(
    f"Q90                       : "
    f"{correction.quantile(.90):.6f} m"
)

print(
    f"Q95                       : "
    f"{correction.quantile(.95):.6f} m"
)

print(
    f"Q99                       : "
    f"{correction.quantile(.99):.6f} m"
)

print(
    f"Maximum                   : "
    f"{correction.max():.6f} m"
)


# ============================================================
# 11. CORRECTION DISTRIBUTION
# ============================================================

bins = [
    -0.000001,
    0,
    0.5,
    1,
    2,
    np.inf,
]

labels = [
    "0 m",
    ">0 to 0.5 m",
    ">0.5 to 1 m",
    ">1 to 2 m",
    ">2 m",
]

group_age["correction_bin"] = pd.cut(
    correction,
    bins=bins,
    labels=labels,
    include_lowest=True,
)

distribution = (
    group_age["correction_bin"]
    .value_counts()
    .sort_index()
)

print("\nCorrection distribution:")

for label, count in distribution.items():

    pct = (
        count
        /
        len(group_age)
        *
        100
    )

    print(
        f"{str(label):15s}"
        f"{count:7,d}"
        f" ({pct:6.2f}%)"
    )


# ============================================================
# 12. PHI = 66
# ============================================================

print("\n[9] PHI = 66 validation...")

phi = group_age[
    group_age[AGE_COL] == 66
]

print(
    f"PHI rows : {len(phi):,}"
)

print(
    f"Actual PHI mean     : "
    f"{phi['actual_height_m'].mean():.6f} m"
)

print(
    f"Corrected PHI mean  : "
    f"{phi['corrected_height_m'].mean():.6f} m"
)

print(
    f"PHI mean abs change : "
    f"{phi['absolute_correction_m'].mean():.6f} m"
)

print(
    f"PHI max abs change  : "
    f"{phi['absolute_correction_m'].max():.6f} m"
)


# ============================================================
# 13. LATE STAGNATION
# ============================================================

print("\n[10] Late-age stagnation diagnostic...")

late = valid[
    valid[AGE_COL] >= 66
]

near_stagnant = (
    late["growth_rate_cm_year"].abs()
    <= 0.05
).sum()

print(
    f"Late transitions       : {len(late):,}"
)

print(
    f"Near-stagnant <=0.05   : "
    f"{near_stagnant:,}"
)

if len(late) > 0:

    print(
        f"Percentage near-stagnant: "
        f"{near_stagnant / len(late) * 100:.2f}%"
    )


# ============================================================
# 14. PLATEAU LENGTH
# ============================================================

print("\n[11] Measuring corrected plateaus...")

group_age["is_stagnant"] = (
    group_age["corrected_height_m"]
    .diff()
    .abs()
    <= 1e-9
)

# Recalculate within each growth group.
group_age["corrected_change_m"] = (
    group_age["corrected_height_m"]
    -
    group_age
    .groupby(GROUP_COLS)["corrected_height_m"]
    .shift(1)
)

group_age["is_stagnant"] = (
    group_age["corrected_change_m"]
    .abs()
    <= 1e-9
)

# Count consecutive stagnant transitions per group.
max_plateau_transitions = 0

for _, sub in group_age.groupby(
    GROUP_COLS,
    sort=False
):

    stagnant = (
        sub["is_stagnant"]
        .fillna(False)
        .astype(int)
    )

    current = 0

    for value in stagnant:

        if value == 1:
            current += 1
            max_plateau_transitions = max(
                max_plateau_transitions,
                current
            )
        else:
            current = 0

print(
    f"Maximum consecutive stagnant transitions: "
    f"{max_plateau_transitions}"
)

print(
    "Note: one transition = 6 months."
)


# ============================================================
# 15. AGE PROFILE
# ============================================================

print("\n[12] Corrected age profile...")

age_profile = (
    group_age
    .groupby(AGE_COL)
    .agg(
        actual_mean_height_m=(
            "actual_height_m",
            "mean"
        ),
        corrected_mean_height_m=(
            "corrected_height_m",
            "mean"
        ),
        mean_absolute_correction_m=(
            "absolute_correction_m",
            "mean"
        ),
    )
    .reset_index()
)

age_profile["corrected_change_m"] = (
    age_profile["corrected_mean_height_m"]
    .diff()
)

print(
    age_profile.to_string(index=False)
)


# ============================================================
# 16. SAVE
# ============================================================

print("\n[13] Saving prototype results...")

group_age.to_csv(
    OUTPUT_FILE,
    index=False
)

stage_file = (
    OUTPUT_DIR
    / "height_isotonic_stage_summary.csv"
)

stage_df.to_csv(
    stage_file,
    index=False
)

age_file = (
    OUTPUT_DIR
    / "height_isotonic_age_profile.csv"
)

age_profile.to_csv(
    age_file,
    index=False
)

summary = pd.DataFrame([
    {
        "metric": "group_age_rows",
        "value": len(group_age),
    },
    {
        "metric": "valid_transitions",
        "value": len(valid),
    },
    {
        "metric": "increases",
        "value": int(increases),
    },
    {
        "metric": "stagnations",
        "value": int(stagnations),
    },
    {
        "metric": "decreases",
        "value": int(decreases),
    },
    {
        "metric": "mean_absolute_correction_m",
        "value": correction.mean(),
    },
    {
        "metric": "median_absolute_correction_m",
        "value": correction.median(),
    },
    {
        "metric": "q95_absolute_correction_m",
        "value": correction.quantile(.95),
    },
    {
        "metric": "max_absolute_correction_m",
        "value": correction.max(),
    },
    {
        "metric": "phi_mean_absolute_correction_m",
        "value": phi[
            "absolute_correction_m"
        ].mean(),
    },
    {
        "metric": "phi_max_absolute_correction_m",
        "value": phi[
            "absolute_correction_m"
        ].max(),
    },
    {
        "metric": "late_near_stagnant_pct",
        "value": (
            near_stagnant
            /
            len(late)
            *
            100
            if len(late) > 0
            else np.nan
        ),
    },
    {
        "metric": "max_consecutive_stagnant_transitions",
        "value": max_plateau_transitions,
    },
])

summary.to_csv(
    SUMMARY_FILE,
    index=False
)

print(
    f"Saved: {OUTPUT_FILE}"
)

print(
    f"Saved: {SUMMARY_FILE}"
)

print(
    f"Saved: {stage_file}"
)

print(
    f"Saved: {age_file}"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("STEP 36E FINISHED")
print("=" * 70)

if decreases == 0:
    print("NO-DECREASE CHECK : PASS")
else:
    print("NO-DECREASE CHECK : FAIL")

print(
    "ORIGINAL DATASET MODIFIED : NO"
)

print("=" * 70)