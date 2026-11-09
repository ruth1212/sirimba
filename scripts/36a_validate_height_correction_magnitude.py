from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# STEP 36A
# VALIDATE HEIGHT CORRECTION MAGNITUDE
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    BASE_DIR
    / "results"
    / "height_shape_constraint_test"
    / "height_shape_constrained_test.csv"
)

RESULT_DIR = (
    BASE_DIR
    / "results"
    / "height_correction_magnitude"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)


GROUP_COLS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]

AGE_COL = "age_months"

ACTUAL_COL = "height_actual"
CORRECTED_COL = "height_corrected_test"
DELTA_COL = "height_correction_delta"


print("=" * 80)
print("STEP 36A — VALIDATE HEIGHT CORRECTION MAGNITUDE")
print("=" * 80)


# ============================================================
# LOAD
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input tidak ditemukan:\n{INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)

print(f"\nRows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# ABSOLUTE CORRECTION
# ============================================================

df["abs_correction"] = (
    df[DELTA_COL].abs()
)


# ============================================================
# 1. OVERALL DISTRIBUTION
# ============================================================

print("\n" + "-" * 80)
print("1. OVERALL CORRECTION DISTRIBUTION")
print("-" * 80)

print(
    df["abs_correction"]
    .describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
    .to_string()
)


# ============================================================
# 2. MAGNITUDE BINS
# ============================================================

print("\n" + "-" * 80)
print("2. CORRECTION MAGNITUDE BINS")
print("-" * 80)

bins = [
    -np.inf,
    1e-9,
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

df["correction_category"] = pd.cut(
    df["abs_correction"],
    bins=bins,
    labels=labels,
    right=True,
    include_lowest=True,
)

magnitude_summary = (
    df["correction_category"]
    .value_counts(
        sort=False
    )
    .rename_axis("correction_category")
    .reset_index(
        name="rows"
    )
)

magnitude_summary["percentage"] = (
    magnitude_summary["rows"]
    / len(df)
    * 100
)

print(
    magnitude_summary
    .to_string(index=False)
)


# ============================================================
# 3. BY AGE
# ============================================================

print("\n" + "-" * 80)
print("3. CORRECTION BY AGE")
print("-" * 80)

age_summary = (
    df.groupby(
        AGE_COL,
        observed=True
    )
    .agg(
        rows=("abs_correction", "size"),
        mean_abs_correction=(
            "abs_correction",
            "mean"
        ),
        median_abs_correction=(
            "abs_correction",
            "median"
        ),
        max_abs_correction=(
            "abs_correction",
            "max"
        ),
        changed_rows=(
            "abs_correction",
            lambda x: int(
                (x > 1e-9).sum()
            ),
        ),
        large_correction_rows=(
            "abs_correction",
            lambda x: int(
                (x > 2.0).sum()
            ),
        ),
    )
    .reset_index()
)

age_summary["changed_percentage"] = (
    age_summary["changed_rows"]
    / age_summary["rows"]
    * 100
)

age_summary["large_correction_percentage"] = (
    age_summary["large_correction_rows"]
    / age_summary["rows"]
    * 100
)

print(
    age_summary[
        [
            AGE_COL,
            "mean_abs_correction",
            "median_abs_correction",
            "max_abs_correction",
            "changed_percentage",
            "large_correction_percentage",
        ]
    ]
    .to_string(index=False)
)


# ============================================================
# 4. AGE-BAND SUMMARY
# ============================================================

print("\n" + "-" * 80)
print("4. AGE-BAND SUMMARY")
print("-" * 80)


def age_band(age):
    if age <= 30:
        return "EARLY_P6_P30"
    elif age <= 60:
        return "MIDDLE_P36_P60"
    elif age <= 66:
        return "PHI_P66"
    elif age <= 120:
        return "LATE_72_120"
    elif age <= 180:
        return "LATE_126_180"
    elif age <= 240:
        return "LATE_186_240"
    else:
        return "LATE_246_300"


df["age_band"] = df[AGE_COL].apply(
    age_band
)

age_band_summary = (
    df.groupby(
        "age_band",
        observed=True
    )
    .agg(
        rows=("abs_correction", "size"),
        mean_abs_correction=(
            "abs_correction",
            "mean"
        ),
        median_abs_correction=(
            "abs_correction",
            "median"
        ),
        max_abs_correction=(
            "abs_correction",
            "max"
        ),
        changed_rows=(
            "abs_correction",
            lambda x: int(
                (x > 1e-9).sum()
            ),
        ),
        correction_over_2m=(
            "abs_correction",
            lambda x: int(
                (x > 2.0).sum()
            ),
        ),
    )
    .reset_index()
)

age_band_summary["changed_percentage"] = (
    age_band_summary["changed_rows"]
    / age_band_summary["rows"]
    * 100
)

age_band_summary["over_2m_percentage"] = (
    age_band_summary["correction_over_2m"]
    / age_band_summary["rows"]
    * 100
)

print(
    age_band_summary
    .to_string(index=False)
)


# ============================================================
# 5. GROUPS WITH LARGE CORRECTIONS
# ============================================================

print("\n" + "-" * 80)
print("5. GROUPS WITH LARGE CORRECTIONS")
print("-" * 80)

group_summary = (
    df.groupby(
        GROUP_COLS,
        as_index=False
    )
    .agg(
        mean_abs_correction=(
            "abs_correction",
            "mean"
        ),
        median_abs_correction=(
            "abs_correction",
            "median"
        ),
        max_abs_correction=(
            "abs_correction",
            "max"
        ),
        rows_changed=(
            "abs_correction",
            lambda x: int(
                (x > 1e-9).sum()
            ),
        ),
        rows_over_2m=(
            "abs_correction",
            lambda x: int(
                (x > 2.0).sum()
            ),
        ),
    )
)

group_summary["percentage_over_2m"] = (
    group_summary["rows_over_2m"]
    /
    group_summary["rows_changed"].replace(
        0,
        np.nan
    )
    * 100
)

large_groups = (
    group_summary
    .sort_values(
        "max_abs_correction",
        ascending=False
    )
    .head(20)
)

print(
    large_groups
    .to_string(index=False)
)


# ============================================================
# 6. PHI
# ============================================================

print("\n" + "-" * 80)
print("6. PHI = 66 MONTHS")
print("-" * 80)

phi = df[
    df[AGE_COL] == 66
]

print(
    f"\nPHI rows : {len(phi):,}"
)

print(
    f"PHI mean absolute correction : "
    f"{phi['abs_correction'].mean():.6f} m"
)

print(
    f"PHI median absolute correction : "
    f"{phi['abs_correction'].median():.6f} m"
)

print(
    f"PHI maximum correction : "
    f"{phi['abs_correction'].max():.6f} m"
)


# ============================================================
# 7. EXTREME CORRECTIONS
# ============================================================

print("\n" + "-" * 80)
print("7. TOP 20 EXTREME CORRECTIONS")
print("-" * 80)

extreme = (
    df[
        [
            *GROUP_COLS,
            AGE_COL,
            ACTUAL_COL,
            CORRECTED_COL,
            DELTA_COL,
            "abs_correction",
        ]
    ]
    .sort_values(
        "abs_correction",
        ascending=False
    )
    .head(20)
)

print(
    extreme.to_string(
        index=False
    )
)


# ============================================================
# 8. CORRECTION DIRECTION
# ============================================================

print("\n" + "-" * 80)
print("8. CORRECTION DIRECTION")
print("-" * 80)

positive = (
    df[DELTA_COL] > 1e-9
).sum()

zero = (
    df[DELTA_COL].abs() <= 1e-9
).sum()

negative = (
    df[DELTA_COL] < -1e-9
).sum()

print(
    f"\nPositive correction : {positive:,}"
)

print(
    f"Zero correction     : {zero:,}"
)

print(
    f"Negative correction : {negative:,}"
)


# ============================================================
# SAVE
# ============================================================

magnitude_path = (
    RESULT_DIR
    / "height_correction_magnitude_summary.csv"
)

age_path = (
    RESULT_DIR
    / "height_correction_by_age.csv"
)

age_band_path = (
    RESULT_DIR
    / "height_correction_by_age_band.csv"
)

group_path = (
    RESULT_DIR
    / "height_correction_by_group.csv"
)

extreme_path = (
    RESULT_DIR
    / "height_extreme_corrections.csv"
)


magnitude_summary.to_csv(
    magnitude_path,
    index=False
)

age_summary.to_csv(
    age_path,
    index=False
)

age_band_summary.to_csv(
    age_band_path,
    index=False
)

group_summary.to_csv(
    group_path,
    index=False
)

extreme.to_csv(
    extreme_path,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("STEP 36A RESULT")
print("=" * 80)

print("\nValidation completed.")

print("\nNO DATASET WAS MODIFIED.")

print("\nOutput files:")
print(f"- {magnitude_path}")
print(f"- {age_path}")
print(f"- {age_band_path}")
print(f"- {group_path}")
print(f"- {extreme_path}")

print("\nNEXT:")
print(
    "Gunakan magnitude correction untuk menentukan "
    "apakah shape constraint dapat diterapkan langsung "
    "atau perlu dibuat lebih konservatif."
)

print("=" * 80)