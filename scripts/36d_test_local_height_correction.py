from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 36D
# LOCAL HEIGHT CORRECTION
#
# Method:
#   Correct only local decreases.
#
# Rule:
#   If current group-age mean height < previous corrected height,
#   current corrected height = previous corrected height.
#
# Important difference from cumulative maximum:
#   We do NOT use future values.
#   We do NOT globally propagate a future maximum backward.
#
# Prototype only.
# Original dataset is NOT modified.
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
    / "height_local_correction_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "height_local_group_age.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "height_local_summary.csv"
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
print("STEP 36D : LOCAL HEIGHT CORRECTION TEST")
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
        f"Missing columns: {missing}"
    )

print("Required columns: PASS")


# ============================================================
# 2. GROUP-AGE MEAN
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
# 3. VERIFY 30 ROWS
# ============================================================

print("\n[3] Checking group-age row count...")

if (
    group_age["rows_at_age"].min() == 30
    and
    group_age["rows_at_age"].max() == 30
):
    print("30 rows per group-age: PASS")
else:
    print(
        "WARNING: group-age row count is not always 30"
    )


# ============================================================
# 4. SORT
# ============================================================

group_age = group_age.sort_values(
    GROUP_COLS + [AGE_COL]
).reset_index(drop=True)


# ============================================================
# 5. PREVIOUS ACTUAL
# ============================================================

print("\n[4] Building local trajectory...")

group_age["previous_actual_height_m"] = (
    group_age
    .groupby(GROUP_COLS)["actual_height_m"]
    .shift(1)
)


# ============================================================
# 6. LOCAL CORRECTION
# ============================================================

print("\n[5] Applying LOCAL correction...")

# Start with actual values.
group_age["corrected_height_m"] = (
    group_age["actual_height_m"]
)

# A decrease means:
#
# current actual < previous actual
#
# Correct ONLY this point to previous actual.
#
# We intentionally do not use cummax().
decrease_mask = (
    group_age["previous_actual_height_m"].notna()
    &
    (
        group_age["actual_height_m"]
        <
        group_age["previous_actual_height_m"]
    )
)

group_age.loc[
    decrease_mask,
    "corrected_height_m"
] = (
    group_age.loc[
        decrease_mask,
        "previous_actual_height_m"
    ]
)

group_age["correction_m"] = (
    group_age["corrected_height_m"]
    -
    group_age["actual_height_m"]
)

print(
    f"Local decreases corrected : "
    f"{decrease_mask.sum():,}"
)


# ============================================================
# 7. CHECK FOR REMAINING DECREASES
# ============================================================

print("\n[6] Validating corrected trajectory...")

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

remaining_decreases = (
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
    f"Decreases         : {remaining_decreases:,}"
)


# ============================================================
# 8. GROWTH RATE
# ============================================================

print("\n[7] Calculating growth rate...")

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
# 9. STAGE SUMMARY
# ============================================================

print("\n[8] Early / middle / late diagnostic...")


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

print("\n[9] Correction magnitude...")

correction = group_age["correction_m"]

print(
    f"Mean correction   : "
    f"{correction.mean():.6f} m"
)

print(
    f"Median correction : "
    f"{correction.median():.6f} m"
)

print(
    f"Q75               : "
    f"{correction.quantile(.75):.6f} m"
)

print(
    f"Q90               : "
    f"{correction.quantile(.90):.6f} m"
)

print(
    f"Q95               : "
    f"{correction.quantile(.95):.6f} m"
)

print(
    f"Q99               : "
    f"{correction.quantile(.99):.6f} m"
)

print(
    f"Maximum           : "
    f"{correction.max():.6f} m"
)


# ============================================================
# 11. CORRECTION DISTRIBUTION
# ============================================================

bins = [
    -np.inf,
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
    right=True
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

print("\n[10] PHI = 66 validation...")

phi = group_age[
    group_age[AGE_COL] == 66
]

print(
    f"PHI rows : {len(phi):,}"
)

print(
    f"Actual PHI mean    : "
    f"{phi['actual_height_m'].mean():.6f} m"
)

print(
    f"Corrected PHI mean : "
    f"{phi['corrected_height_m'].mean():.6f} m"
)

print(
    f"PHI mean correction: "
    f"{phi['correction_m'].mean():.6f} m"
)

print(
    f"PHI max correction : "
    f"{phi['correction_m'].max():.6f} m"
)


# ============================================================
# 13. LATE STAGNATION
# ============================================================

print("\n[11] Late-age stagnation diagnostic...")

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
# 14. PHI TRANSITION
# ============================================================

print("\n[12] P54 -> PHI -> P72 diagnostic...")

transition = group_age[
    group_age[AGE_COL].isin([54, 60, 66, 72])
][
    GROUP_COLS
    + [
        AGE_COL,
        "actual_height_m",
        "corrected_height_m",
        "correction_m",
    ]
]

pivot = transition.pivot(
    index=GROUP_COLS,
    columns=AGE_COL,
    values=[
        "actual_height_m",
        "corrected_height_m",
        "correction_m",
    ]
).reset_index()

pivot.columns = [
    "_".join(
        str(x)
        for x in col
        if str(x) != ""
    )
    if isinstance(col, tuple)
    else str(col)
    for col in pivot.columns
]

print(
    f"Trajectory patterns inspected: "
    f"{len(pivot):,}"
)


# ============================================================
# 15. SAVE
# ============================================================

print("\n[13] Saving prototype...")

group_age.to_csv(
    OUTPUT_FILE,
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
        "metric": "local_decreases_corrected",
        "value": int(decrease_mask.sum()),
    },
    {
        "metric": "remaining_decreases",
        "value": int(remaining_decreases),
    },
    {
        "metric": "mean_correction_m",
        "value": correction.mean(),
    },
    {
        "metric": "median_correction_m",
        "value": correction.median(),
    },
    {
        "metric": "q95_correction_m",
        "value": correction.quantile(.95),
    },
    {
        "metric": "max_correction_m",
        "value": correction.max(),
    },
    {
        "metric": "phi_mean_correction_m",
        "value": phi["correction_m"].mean(),
    },
    {
        "metric": "phi_max_correction_m",
        "value": phi["correction_m"].max(),
    },
    {
        "metric": "late_near_stagnant_pct",
        "value": (
            near_stagnant / len(late) * 100
            if len(late) > 0
            else np.nan
        ),
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


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("STEP 36D FINISHED")
print("=" * 70)

if remaining_decreases == 0:
    print("NO-DECREASE CHECK : PASS")
else:
    print("NO-DECREASE CHECK : FAIL")

print(
    "ORIGINAL DATASET MODIFIED : NO"
)

print("=" * 70)