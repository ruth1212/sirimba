from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "dataset_corrected.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "height_decline_recovery_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
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


# ============================================================
# HELPER
# ============================================================

def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def decline_class(value):
    """
    value = positive decline magnitude in meters
    """

    if value < 0.5:
        return "SMALL_<0.5M"

    if value < 1.0:
        return "MODERATE_0.5_1M"

    if value < 2.0:
        return "LARGE_1_2M"

    return "EXTREME_>=2M"


def age_band(age):

    if 6 <= age <= 30:
        return "EARLY_P6_P30"

    if 36 <= age <= 60:
        return "MIDDLE_P36_P60"

    if 66 <= age <= 120:
        return "LATE_P66_P120"

    if 126 <= age <= 180:
        return "LATE_P126_P180"

    if 186 <= age <= 240:
        return "LATE_P186_P240"

    if 246 <= age <= 300:
        return "LATE_P246_P300"

    return "OTHER"


# ============================================================
# LOAD
# ============================================================

section("STEP 36I : DECLINE & RECOVERY PATTERN AUDIT")

print("[1] Loading dataset...")

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan:\n{INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# VALIDATION
# ============================================================

required = GROUP_COLS + [
    AGE_COL,
    HEIGHT_COL,
]

missing = [
    col
    for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print("Required columns: PASS")


# ============================================================
# GROUP + AGE AGGREGATION
# ============================================================

section("2. BUILDING GROUP-AGE TRAJECTORY")

group_age = (
    df.groupby(
        GROUP_COLS + [AGE_COL],
        as_index=False
    )
    .agg(
        actual_height_m=(
            HEIGHT_COL,
            "mean"
        ),
        rows_at_age=(
            HEIGHT_COL,
            "size"
        ),
    )
)

group_age = group_age.sort_values(
    GROUP_COLS + [AGE_COL]
).reset_index(drop=True)

group_age["group_key"] = (
    group_age[GROUP_COLS]
    .astype(str)
    .agg("|".join, axis=1)
)

print(
    f"Group-age rows : {len(group_age):,}"
)

print(
    f"Growth groups  : "
    f"{group_age['group_key'].nunique():,}"
)

print(
    f"Unique ages    : "
    f"{group_age[AGE_COL].nunique():,}"
)

print(
    "30 rows per group-age: "
    + (
        "PASS"
        if group_age["rows_at_age"].eq(30).all()
        else "FAIL"
    )
)


# ============================================================
# CALCULATE TRANSITIONS
# ============================================================

section("3. CALCULATING TRANSITIONS")

group_age["previous_age_months"] = (
    group_age.groupby("group_key")[AGE_COL]
    .shift(1)
)

group_age["previous_height_m"] = (
    group_age.groupby("group_key")[
        "actual_height_m"
    ]
    .shift(1)
)

group_age["next_height_m"] = (
    group_age.groupby("group_key")[
        "actual_height_m"
    ]
    .shift(-1)
)

group_age["height_change_m"] = (
    group_age["actual_height_m"]
    - group_age["previous_height_m"]
)

group_age["next_change_m"] = (
    group_age["next_height_m"]
    - group_age["actual_height_m"]
)

valid = group_age[
    group_age["previous_height_m"].notna()
].copy()

valid["age_band"] = (
    valid[AGE_COL]
    .apply(age_band)
)

declines = valid[
    valid["height_change_m"] < 0
].copy()

declines["decline_magnitude_m"] = (
    -declines["height_change_m"]
)

declines["decline_class"] = (
    declines["decline_magnitude_m"]
    .apply(decline_class)
)

print(
    f"Valid transitions : {len(valid):,}"
)

print(
    f"Declines          : {len(declines):,}"
)


# ============================================================
# 4. DECLINE CLASS SUMMARY
# ============================================================

section("4. DECLINE CLASS SUMMARY")

class_summary = (
    declines.groupby(
        "decline_class",
        observed=False
    )
    .agg(
        count=(
            "decline_magnitude_m",
            "size"
        ),
        mean_decline_m=(
            "decline_magnitude_m",
            "mean"
        ),
        median_decline_m=(
            "decline_magnitude_m",
            "median"
        ),
        max_decline_m=(
            "decline_magnitude_m",
            "max"
        ),
    )
    .reset_index()
)

class_summary["percentage"] = (
    class_summary["count"]
    / len(declines)
    * 100
)

print(
    class_summary.to_string(
        index=False
    )
)

class_summary.to_csv(
    OUTPUT_DIR
    / "decline_class_summary.csv",
    index=False
)


# ============================================================
# 5. RECOVERY CLASSIFICATION
# ============================================================

section("5. RECOVERY AFTER DECLINE")

# A decline is considered "direct recovery" when
# the next 6-month transition is positive.

declines["recovery_m"] = (
    declines["next_change_m"]
)

declines["recovered_next"] = (
    declines["next_change_m"] > 0
)

declines["recovery_amount_m"] = np.where(
    declines["recovered_next"],
    declines["next_change_m"],
    0.0
)

recovery_summary = (
    declines.groupby(
        "decline_class",
        observed=False
    )
    .agg(
        decline_count=(
            "decline_magnitude_m",
            "size"
        ),
        recovered_next=(
            "recovered_next",
            "sum"
        ),
        mean_decline_m=(
            "decline_magnitude_m",
            "mean"
        ),
        mean_recovery_m=(
            "recovery_amount_m",
            "mean"
        ),
    )
    .reset_index()
)

recovery_summary["recovery_pct"] = (
    recovery_summary["recovered_next"]
    / recovery_summary["decline_count"]
    * 100
)

print(
    recovery_summary.to_string(
        index=False
    )
)

recovery_summary.to_csv(
    OUTPUT_DIR
    / "recovery_by_decline_class.csv",
    index=False
)


# ============================================================
# 6. RECOVERY RATIO
# ============================================================

section("6. RECOVERY RATIO")

recovered = declines[
    declines["recovered_next"]
].copy()

recovered["recovery_ratio"] = (
    recovered["recovery_amount_m"]
    / recovered["decline_magnitude_m"]
)

ratio_summary = (
    recovered.groupby(
        "decline_class",
        observed=False
    )
    .agg(
        recovered_count=(
            "recovery_ratio",
            "size"
        ),
        mean_ratio=(
            "recovery_ratio",
            "mean"
        ),
        median_ratio=(
            "recovery_ratio",
            "median"
        ),
        q25_ratio=(
            "recovery_ratio",
            lambda x: x.quantile(.25)
        ),
        q75_ratio=(
            "recovery_ratio",
            lambda x: x.quantile(.75)
        ),
        max_ratio=(
            "recovery_ratio",
            "max"
        ),
    )
    .reset_index()
)

print(
    ratio_summary.to_string(
        index=False
    )
)

ratio_summary.to_csv(
    OUTPUT_DIR
    / "recovery_ratio_by_class.csv",
    index=False
)


# ============================================================
# 7. RECOVERY CATEGORIES
# ============================================================

section("7. RECOVERY COMPLETENESS")

recovered["recovery_completeness"] = pd.cut(
    recovered["recovery_ratio"],
    bins=[
        -np.inf,
        0.25,
        0.50,
        0.75,
        1.00,
        np.inf,
    ],
    labels=[
        "<25%",
        "25-50%",
        "50-75%",
        "75-100%",
        ">100%",
    ],
    right=False,
)

completeness = (
    recovered["recovery_completeness"]
    .value_counts(
        sort=False
    )
    .rename("count")
    .reset_index()
)

completeness["percentage"] = (
    completeness["count"]
    / len(recovered)
    * 100
)

print(
    completeness.to_string(
        index=False
    )
)

completeness.to_csv(
    OUTPUT_DIR
    / "recovery_completeness.csv",
    index=False
)


# ============================================================
# 8. DECLINE CLASS BY AGE BAND
# ============================================================

section("8. DECLINE CLASS BY AGE BAND")

class_age = (
    declines.groupby(
        [
            "age_band",
            "decline_class",
        ],
        observed=False
    )
    .agg(
        count=(
            "decline_magnitude_m",
            "size"
        ),
        mean_decline_m=(
            "decline_magnitude_m",
            "mean"
        ),
        max_decline_m=(
            "decline_magnitude_m",
            "max"
        ),
    )
    .reset_index()
)

print(
    class_age.to_string(
        index=False
    )
)

class_age.to_csv(
    OUTPUT_DIR
    / "decline_class_by_age_band.csv",
    index=False
)


# ============================================================
# 9. RECOVERY BY AGE BAND
# ============================================================

section("9. RECOVERY BY AGE BAND")

band_recovery = (
    declines.groupby(
        "age_band",
        observed=False
    )
    .agg(
        decline_count=(
            "decline_magnitude_m",
            "size"
        ),
        recovered_next=(
            "recovered_next",
            "sum"
        ),
        mean_decline_m=(
            "decline_magnitude_m",
            "mean"
        ),
        mean_recovery_m=(
            "recovery_amount_m",
            "mean"
        ),
    )
    .reset_index()
)

band_recovery["recovery_pct"] = (
    band_recovery["recovered_next"]
    / band_recovery["decline_count"]
    * 100
)

print(
    band_recovery.to_string(
        index=False
    )
)

band_recovery.to_csv(
    OUTPUT_DIR
    / "recovery_by_age_band.csv",
    index=False
)


# ============================================================
# 10. DECLINES THAT DO NOT RECOVER IMMEDIATELY
# ============================================================

section("10. NON-IMMEDIATE RECOVERY")

non_recovered = declines[
    ~declines["recovered_next"]
].copy()

print(
    f"Declines not followed by immediate recovery: "
    f"{len(non_recovered):,}"
)

if len(non_recovered) > 0:

    print(
        f"Percentage: "
        f"{len(non_recovered) / len(declines) * 100:.2f}%"
    )

    non_recovered_summary = (
        non_recovered.groupby(
            "decline_class",
            observed=False
        )
        .agg(
            count=(
                "decline_magnitude_m",
                "size"
            ),
            mean_decline_m=(
                "decline_magnitude_m",
                "mean"
            ),
            max_decline_m=(
                "decline_magnitude_m",
                "max"
            ),
        )
        .reset_index()
    )

    print(
        non_recovered_summary.to_string(
            index=False
        )
    )

    non_recovered_summary.to_csv(
        OUTPUT_DIR
        / "non_immediate_recovery_summary.csv",
        index=False
    )


# ============================================================
# 11. EXTREME DECLINES DETAIL
# ============================================================

section("11. EXTREME DECLINE DETAIL")

extreme = declines[
    declines["decline_class"]
    == "EXTREME_>=2M"
].copy()

extreme = extreme.sort_values(
    "decline_magnitude_m",
    ascending=False
)

print(
    f"Extreme declines: "
    f"{len(extreme):,}"
)

print(
    extreme[
        GROUP_COLS
        + [
            "previous_age_months",
            AGE_COL,
            "previous_height_m",
            "actual_height_m",
            "height_change_m",
            "decline_magnitude_m",
            "next_height_m",
            "next_change_m",
            "recovered_next",
            "age_band",
        ]
    ]
    .head(30)
    .to_string(
        index=False
    )
)

extreme[
    GROUP_COLS
    + [
        "previous_age_months",
        AGE_COL,
        "previous_height_m",
        "actual_height_m",
        "height_change_m",
        "decline_magnitude_m",
        "next_height_m",
        "next_change_m",
        "recovered_next",
        "age_band",
    ]
].to_csv(
    OUTPUT_DIR
    / "extreme_declines_detail.csv",
    index=False
)


# ============================================================
# 12. SAVE ALL DECLINE RECORDS
# ============================================================

declines[
    GROUP_COLS
    + [
        "previous_age_months",
        AGE_COL,
        "previous_height_m",
        "actual_height_m",
        "height_change_m",
        "decline_magnitude_m",
        "decline_class",
        "next_height_m",
        "next_change_m",
        "recovered_next",
        "recovery_amount_m",
        "age_band",
    ]
].to_csv(
    OUTPUT_DIR
    / "all_decline_recovery_records.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

section("STEP 36I FINISHED")

print(
    f"Output directory:\n{OUTPUT_DIR}"
)

print(
    "\nOriginal dataset modified: NO"
)
