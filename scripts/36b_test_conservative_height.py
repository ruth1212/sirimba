from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 36B
# TEST CONSERVATIVE HEIGHT CORRECTION
#
# Method:
#   Group-age mean + cumulative maximum
#
# Goal:
#   1. Height tidak boleh turun terhadap umur.
#   2. Koreksi seminimal mungkin.
#   3. Tidak memaksakan strict diminishing rate.
#   4. Mempertahankan pola:
#        cepat awal -> melambat -> stagnasi usia tua.
#
# IMPORTANT:
#   Ini hanya PROTOTYPE.
#   Dataset final TIDAK diubah.
# ============================================================


# ------------------------------------------------------------
# PATH
# ------------------------------------------------------------

BASE_DIR = Path(r"C:\tree_growth_fs\tree_growth_fs")

INPUT_FILE = BASE_DIR / "data" / "dataset_corrected.csv"

OUTPUT_DIR = BASE_DIR / "results" / "height_conservative_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "height_conservative_group_age.csv"
SUMMARY_FILE = OUTPUT_DIR / "height_conservative_summary.csv"


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

GROUP_COLS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]

AGE_COL = "age_months"
HEIGHT_COL = "height_m"


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

print("=" * 70)
print("STEP 36B : CONSERVATIVE HEIGHT CORRECTION TEST")
print("=" * 70)

print("\n[1] Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

required = GROUP_COLS + [AGE_COL, HEIGHT_COL]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print("\nRequired columns: PASS")


# ------------------------------------------------------------
# GROUP-AGE AGGREGATION
# ------------------------------------------------------------

print("\n[2] Aggregating to growth-group + age...")

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

print(f"Group-age rows : {len(group_age):,}")

unique_groups = (
    group_age[GROUP_COLS]
    .drop_duplicates()
    .shape[0]
)

unique_ages = group_age[AGE_COL].nunique()

print(f"Growth groups  : {unique_groups:,}")
print(f"Unique ages    : {unique_ages:,}")


# ------------------------------------------------------------
# VERIFY 30 ROWS PER GROUP-AGE
# ------------------------------------------------------------

print("\n[3] Checking rows per group-age...")

counts = group_age["rows_at_age"]

print(
    f"Rows/group-age min    : {counts.min()}"
)

print(
    f"Rows/group-age max    : {counts.max()}"
)

print(
    f"Rows/group-age median : {counts.median()}"
)

if counts.min() != 30 or counts.max() != 30:
    print(
        "\nWARNING: group-age does not always contain exactly 30 rows."
    )
else:
    print(
        "30 rows per group-age: PASS"
    )


# ------------------------------------------------------------
# SORT
# ------------------------------------------------------------

group_age = group_age.sort_values(
    GROUP_COLS + [AGE_COL]
).reset_index(drop=True)


# ------------------------------------------------------------
# CONSERVATIVE CORRECTION
# ------------------------------------------------------------

print("\n[4] Applying cumulative-maximum correction...")

group_age["corrected_height_m"] = (
    group_age
    .groupby(GROUP_COLS)["actual_height_m"]
    .cummax()
)

group_age["correction_m"] = (
    group_age["corrected_height_m"]
    - group_age["actual_height_m"]
)


# ------------------------------------------------------------
# VALIDATE NO DECREASE
# ------------------------------------------------------------

print("\n[5] Validating corrected trajectories...")

group_age["previous_corrected_height_m"] = (
    group_age
    .groupby(GROUP_COLS)["corrected_height_m"]
    .shift(1)
)

group_age["height_change_corrected_m"] = (
    group_age["corrected_height_m"]
    - group_age["previous_corrected_height_m"]
)

valid_transitions = group_age[
    group_age["previous_corrected_height_m"].notna()
].copy()

decreases = (
    valid_transitions["height_change_corrected_m"] < -1e-9
).sum()

increases = (
    valid_transitions["height_change_corrected_m"] > 1e-9
).sum()

stagnations = (
    valid_transitions["height_change_corrected_m"].abs() <= 1e-9
).sum()

print(f"Valid transitions : {len(valid_transitions):,}")
print(f"Increases         : {increases:,}")
print(f"Stagnations       : {stagnations:,}")
print(f"Decreases         : {decreases:,}")


# ------------------------------------------------------------
# GROWTH RATE
# ------------------------------------------------------------

print("\n[6] Calculating growth rates...")

# Previous age for each growth group
valid_transitions["previous_age_months"] = (
    valid_transitions
    .groupby(GROUP_COLS)[AGE_COL]
    .shift(1)
)

# Age interval in months
valid_transitions["age_gap_months"] = (
    valid_transitions[AGE_COL].astype(float)
    - valid_transitions["previous_age_months"].astype(float)
)

# Height growth rate:
# meters -> centimeters
# months -> years
valid_transitions["growth_rate_cm_year"] = (
    valid_transitions["height_change_corrected_m"]
    * 100
    / (
        valid_transitions["age_gap_months"]
        / 12
    )
)

print("Growth-rate calculation: PASS")

# ------------------------------------------------------------
# STAGE DIAGNOSTIC
# ------------------------------------------------------------

print("\n[6] Early / middle / late growth diagnostic...")


def stage_summary(data, min_age, max_age, name):
    x = data[
        (data[AGE_COL] >= min_age)
        & (data[AGE_COL] <= max_age)
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


stage_results = [
    stage_summary(
        valid_transitions,
        12,
        30,
        "EARLY_P6_P30"
    ),
    stage_summary(
        valid_transitions,
        36,
        60,
        "MIDDLE_P36_P60"
    ),
    stage_summary(
        valid_transitions,
        66,
        300,
        "LATE_P66_P300"
    ),
]

stage_df = pd.DataFrame(stage_results)

print(
    stage_df.to_string(index=False)
)


# ------------------------------------------------------------
# CORRECTION MAGNITUDE
# ------------------------------------------------------------

print("\n[7] Correction magnitude...")

correction = group_age["correction_m"]

print(f"Mean correction   : {correction.mean():.6f} m")
print(f"Median correction : {correction.median():.6f} m")
print(f"Q75               : {correction.quantile(.75):.6f} m")
print(f"Q90               : {correction.quantile(.90):.6f} m")
print(f"Q95               : {correction.quantile(.95):.6f} m")
print(f"Q99               : {correction.quantile(.99):.6f} m")
print(f"Maximum           : {correction.max():.6f} m")


# ------------------------------------------------------------
# CORRECTION BINS
# ------------------------------------------------------------

bins = [
    -np.inf,
    0,
    0.5,
    1.0,
    2.0,
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

correction_distribution = (
    group_age["correction_bin"]
    .value_counts()
    .sort_index()
)

print("\nCorrection distribution:")

for label, count in correction_distribution.items():
    percentage = (
        count / len(group_age) * 100
    )

    print(
        f"{str(label):15s} "
        f"{count:7,d} "
        f"({percentage:6.2f}%)"
    )


# ------------------------------------------------------------
# PHI = 66
# ------------------------------------------------------------

print("\n[8] PHI = 66 validation...")

phi = group_age[
    group_age[AGE_COL] == 66
]

print(f"PHI rows : {len(phi):,}")

print(
    f"Actual PHI mean    : "
    f"{phi['actual_height_m'].mean():.6f} m"
)

print(
    f"Corrected PHI mean  : "
    f"{phi['corrected_height_m'].mean():.6f} m"
)

print(
    f"PHI mean correction : "
    f"{phi['correction_m'].mean():.6f} m"
)

print(
    f"PHI max correction  : "
    f"{phi['correction_m'].max():.6f} m"
)


# ------------------------------------------------------------
# OLD-AGE BEHAVIOR
# ------------------------------------------------------------

print("\n[9] Late-age stagnation diagnostic...")

late = valid_transitions[
    valid_transitions[AGE_COL] >= 66
]

late_growth = late["growth_rate_cm_year"]

late_zero_or_near_zero = (
    late_growth.abs() <= 0.05
).sum()

late_count = len(late)

print(
    f"Late transitions       : {late_count:,}"
)

print(
    f"Near-stagnant <=0.05   : "
    f"{late_zero_or_near_zero:,}"
)

if late_count > 0:
    print(
        f"Percentage near-stagnant: "
        f"{late_zero_or_near_zero / late_count * 100:.2f}%"
    )


# ------------------------------------------------------------
# SAVE GROUP-AGE RESULT
# ------------------------------------------------------------

print("\n[10] Saving prototype result...")

group_age.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"Saved: {OUTPUT_FILE}"
)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

summary_rows = [
    {
        "metric": "total_group_age_rows",
        "value": len(group_age),
    },
    {
        "metric": "valid_transitions",
        "value": len(valid_transitions),
    },
    {
        "metric": "increases",
        "value": increases,
    },
    {
        "metric": "stagnations",
        "value": stagnations,
    },
    {
        "metric": "decreases",
        "value": decreases,
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
        "metric": "phi_actual_mean_m",
        "value": phi["actual_height_m"].mean(),
    },
    {
        "metric": "phi_corrected_mean_m",
        "value": phi["corrected_height_m"].mean(),
    },
    {
        "metric": "phi_mean_correction_m",
        "value": phi["correction_m"].mean(),
    },
    {
        "metric": "phi_max_correction_m",
        "value": phi["correction_m"].max(),
    },
]

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)

print(
    f"Saved: {SUMMARY_FILE}"
)


# ------------------------------------------------------------
# FINAL STATUS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 36B FINISHED")
print("=" * 70)

if decreases == 0:
    print("NO-DECREASE CHECK : PASS")
else:
    print("NO-DECREASE CHECK : FAIL")

print("\nIMPORTANT:")
print("This is ONLY a prototype.")
print("The original dataset has NOT been modified.")
print("=" * 70)