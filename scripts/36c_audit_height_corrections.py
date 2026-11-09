from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 36C
# AUDIT LARGE HEIGHT CORRECTIONS
#
# Input:
#   results/height_conservative_test/
#       height_conservative_group_age.csv
#
# Purpose:
#   - menemukan koreksi Height terbesar
#   - melihat trajectory sebelum/sesudah titik koreksi
#   - fokus pada PHI = 66
#   - melihat apakah koreksi terkonsentrasi pada group tertentu
#
# IMPORTANT:
#   TIDAK mengubah dataset.
# ============================================================


BASE_DIR = Path(r"C:\tree_growth_fs\tree_growth_fs")

INPUT_FILE = (
    BASE_DIR
    / "results"
    / "height_conservative_test"
    / "height_conservative_group_age.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "height_conservative_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


GROUP_COLS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]

AGE_COL = "age_months"

ACTUAL_COL = "actual_height_m"
CORRECTED_COL = "corrected_height_m"
CORRECTION_COL = "correction_m"


print("=" * 70)
print("STEP 36C : AUDIT LARGE HEIGHT CORRECTIONS")
print("=" * 70)


# ============================================================
# 1. LOAD
# ============================================================

print("\n[1] Loading STEP 36B result...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

required = (
    GROUP_COLS
    + [
        AGE_COL,
        ACTUAL_COL,
        CORRECTED_COL,
        CORRECTION_COL,
    ]
)

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
# 2. SORT
# ============================================================

df = df.sort_values(
    GROUP_COLS + [AGE_COL]
).reset_index(drop=True)


# ============================================================
# 3. PREVIOUS / NEXT INFORMATION
# ============================================================

print("\n[2] Building trajectory context...")

df["previous_age_months"] = (
    df.groupby(GROUP_COLS)[AGE_COL]
    .shift(1)
)

df["previous_actual_height_m"] = (
    df.groupby(GROUP_COLS)[ACTUAL_COL]
    .shift(1)
)

df["previous_corrected_height_m"] = (
    df.groupby(GROUP_COLS)[CORRECTED_COL]
    .shift(1)
)

df["next_age_months"] = (
    df.groupby(GROUP_COLS)[AGE_COL]
    .shift(-1)
)

df["next_actual_height_m"] = (
    df.groupby(GROUP_COLS)[ACTUAL_COL]
    .shift(-1)
)

df["next_corrected_height_m"] = (
    df.groupby(GROUP_COLS)[CORRECTED_COL]
    .shift(-1)
)


# ============================================================
# 4. TRANSITION DELTAS
# ============================================================

df["actual_change_from_previous_m"] = (
    df[ACTUAL_COL]
    - df["previous_actual_height_m"]
)

df["corrected_change_from_previous_m"] = (
    df[CORRECTED_COL]
    - df["previous_corrected_height_m"]
)

df["next_actual_change_m"] = (
    df["next_actual_height_m"]
    - df[ACTUAL_COL]
)

df["next_corrected_change_m"] = (
    df["next_corrected_height_m"]
    - df[CORRECTED_COL]
)


# ============================================================
# 5. LARGE CORRECTIONS
# ============================================================

print("\n[3] Finding large corrections...")

large = df[
    df[CORRECTION_COL] > 2.0
].copy()

large = large.sort_values(
    CORRECTION_COL,
    ascending=False
)

print(
    f"Group-age points with correction > 2 m: "
    f"{len(large):,}"
)

print("\nTOP 30 LARGEST CORRECTIONS:\n")

display_cols = (
    GROUP_COLS
    + [
        AGE_COL,
        ACTUAL_COL,
        CORRECTED_COL,
        CORRECTION_COL,
        "previous_age_months",
        "previous_actual_height_m",
        "previous_corrected_height_m",
        "actual_change_from_previous_m",
        "corrected_change_from_previous_m",
        "next_actual_height_m",
        "next_corrected_height_m",
    ]
)

print(
    large[
        display_cols
    ]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# 6. SAVE TOP 100
# ============================================================

top100_file = (
    OUTPUT_DIR
    / "top100_largest_height_corrections.csv"
)

large[
    display_cols
].head(100).to_csv(
    top100_file,
    index=False
)

print(
    f"\nSaved: {top100_file}"
)


# ============================================================
# 7. CORRECTIONS BY AGE
# ============================================================

print("\n[4] Correction concentration by age...")

age_summary = (
    df.groupby(AGE_COL)
    .agg(
        group_age_count=(CORRECTION_COL, "size"),
        mean_correction_m=(CORRECTION_COL, "mean"),
        median_correction_m=(CORRECTION_COL, "median"),
        max_correction_m=(CORRECTION_COL, "max"),
        count_over_1m=(
            CORRECTION_COL,
            lambda x: (x > 1).sum()
        ),
        count_over_2m=(
            CORRECTION_COL,
            lambda x: (x > 2).sum()
        ),
    )
    .reset_index()
)

age_summary["pct_over_1m"] = (
    age_summary["count_over_1m"]
    / age_summary["group_age_count"]
    * 100
)

age_summary["pct_over_2m"] = (
    age_summary["count_over_2m"]
    / age_summary["group_age_count"]
    * 100
)

print(
    age_summary.to_string(index=False)
)

age_file = (
    OUTPUT_DIR
    / "height_correction_by_age.csv"
)

age_summary.to_csv(
    age_file,
    index=False
)

print(
    f"\nSaved: {age_file}"
)


# ============================================================
# 8. PHI = 66 AUDIT
# ============================================================

print("\n[5] PHI = 66 audit...")

phi = df[
    df[AGE_COL] == 66
].copy()

phi = phi.sort_values(
    CORRECTION_COL,
    ascending=False
)

print(
    f"PHI group-age rows : {len(phi):,}"
)

print("\nTOP 30 PHI CORRECTIONS:\n")

phi_display = (
    GROUP_COLS
    + [
        AGE_COL,
        ACTUAL_COL,
        CORRECTED_COL,
        CORRECTION_COL,
        "previous_age_months",
        "previous_actual_height_m",
        "previous_corrected_height_m",
        "actual_change_from_previous_m",
        "corrected_change_from_previous_m",
        "next_age_months",
        "next_actual_height_m",
        "next_corrected_height_m",
    ]
)

print(
    phi[
        phi_display
    ]
    .head(30)
    .to_string(index=False)
)

phi_file = (
    OUTPUT_DIR
    / "phi66_height_correction_audit.csv"
)

phi[
    phi_display
].to_csv(
    phi_file,
    index=False
)

print(
    f"\nSaved: {phi_file}"
)


# ============================================================
# 9. GROUP CONCENTRATION
# ============================================================

print("\n[6] Finding groups with repeated large corrections...")

large_group_summary = (
    large.groupby(GROUP_COLS)
    .agg(
        large_correction_points=(
            CORRECTION_COL,
            "size"
        ),
        mean_large_correction_m=(
            CORRECTION_COL,
            "mean"
        ),
        max_correction_m=(
            CORRECTION_COL,
            "max"
        ),
        total_correction_m=(
            CORRECTION_COL,
            "sum"
        ),
    )
    .reset_index()
    .sort_values(
        [
            "large_correction_points",
            "max_correction_m",
        ],
        ascending=False
    )
)

print(
    large_group_summary
    .head(30)
    .to_string(index=False)
)

group_file = (
    OUTPUT_DIR
    / "height_large_correction_by_group.csv"
)

large_group_summary.to_csv(
    group_file,
    index=False
)

print(
    f"\nSaved: {group_file}"
)


# ============================================================
# 10. AGE-BAND SUMMARY
# ============================================================

print("\n[7] Age-band audit...")

def age_band(age):
    if age <= 30:
        return "EARLY_P6_P30"
    elif age <= 60:
        return "MIDDLE_P36_P60"
    elif age == 66:
        return "PHI_P66"
    elif age <= 120:
        return "LATE_P72_P120"
    elif age <= 180:
        return "LATE_P126_P180"
    elif age <= 240:
        return "LATE_P186_P240"
    else:
        return "LATE_P246_P300"


df["age_band"] = df[AGE_COL].apply(age_band)

band_summary = (
    df.groupby("age_band", sort=False)
    .agg(
        points=(CORRECTION_COL, "size"),
        mean_correction_m=(CORRECTION_COL, "mean"),
        median_correction_m=(CORRECTION_COL, "median"),
        q95_correction_m=(
            CORRECTION_COL,
            lambda x: x.quantile(.95)
        ),
        max_correction_m=(CORRECTION_COL, "max"),
        count_over_1m=(
            CORRECTION_COL,
            lambda x: (x > 1).sum()
        ),
        count_over_2m=(
            CORRECTION_COL,
            lambda x: (x > 2).sum()
        ),
    )
    .reset_index()
)

band_summary["pct_over_1m"] = (
    band_summary["count_over_1m"]
    / band_summary["points"]
    * 100
)

band_summary["pct_over_2m"] = (
    band_summary["count_over_2m"]
    / band_summary["points"]
    * 100
)

print(
    band_summary.to_string(index=False)
)

band_file = (
    OUTPUT_DIR
    / "height_correction_age_band_summary.csv"
)

band_summary.to_csv(
    band_file,
    index=False
)

print(
    f"\nSaved: {band_file}"
)


# ============================================================
# 11. PHI TRANSITION PATTERN
# ============================================================

print("\n[8] Inspecting P54 -> PHI -> P72 pattern...")

pivot_ages = df[
    df[AGE_COL].isin([54, 66, 72])
][
    GROUP_COLS
    + [
        AGE_COL,
        ACTUAL_COL,
        CORRECTED_COL,
        CORRECTION_COL,
    ]
].copy()

pivot = pivot_ages.pivot(
    index=GROUP_COLS,
    columns=AGE_COL,
    values=[
        ACTUAL_COL,
        CORRECTED_COL,
        CORRECTION_COL,
    ]
)

pivot = pivot.reset_index()

# Flatten columns
pivot.columns = [
    "_".join(
        [str(x) for x in col if str(x) != ""]
    )
    if isinstance(col, tuple)
    else str(col)
    for col in pivot.columns
]

pivot_file = (
    OUTPUT_DIR
    / "height_p54_phi66_p72_patterns.csv"
)

pivot.to_csv(
    pivot_file,
    index=False
)

print(
    f"Saved: {pivot_file}"
)


# ============================================================
# 12. FINAL AUDIT SUMMARY
# ============================================================

print("\n[9] FINAL AUDIT SUMMARY")

print(
    f"Total group-age points       : {len(df):,}"
)

print(
    f"Correction > 1 m             : "
    f"{(df[CORRECTION_COL] > 1).sum():,}"
)

print(
    f"Correction > 2 m             : "
    f"{(df[CORRECTION_COL] > 2).sum():,}"
)

print(
    f"Maximum correction           : "
    f"{df[CORRECTION_COL].max():.6f} m"
)

print(
    f"Mean correction               : "
    f"{df[CORRECTION_COL].mean():.6f} m"
)

print(
    f"Median correction             : "
    f"{df[CORRECTION_COL].median():.6f} m"
)

print(
    f"PHI mean correction           : "
    f"{phi[CORRECTION_COL].mean():.6f} m"
)

print(
    f"PHI max correction            : "
    f"{phi[CORRECTION_COL].max():.6f} m"
)

print("\n" + "=" * 70)
print("STEP 36C FINISHED")
print("=" * 70)

print("NO DATASET MODIFICATION WAS PERFORMED.")
print("=" * 70)