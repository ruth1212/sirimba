from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "results"
    / "dashboard_growth_trajectory.csv"
)


print("=" * 78)
print("STEP 21C — VALIDATE PINNs DASHBOARD OUTPUT")
print("=" * 78)

print()
print("Loading dashboard dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df)}")


# ==========================================================
# 1. PINNs VOLUME
# ==========================================================

print()
print("=" * 78)
print("1. PINNs VOLUME CHECK")
print("=" * 78)

volume = df["pinns_volume_m3_tree"]

negative_volume = df[
    volume < 0
]

print(f"Negative PINNs volume rows : {len(negative_volume):,}")
print(f"Minimum PINNs volume       : {volume.min():.10f}")
print(f"Maximum PINNs volume       : {volume.max():.10f}")


# ==========================================================
# 2. NEGATIVE VOLUME BY AGE
# ==========================================================

print()
print("=" * 78)
print("2. NEGATIVE VOLUME BY AGE")
print("=" * 78)

if len(negative_volume) > 0:

    age_summary = (
        negative_volume
        .groupby("age_months")
        .size()
        .reset_index(name="negative_rows")
        .sort_values("age_months")
    )

    print(age_summary.to_string(index=False))

else:

    print("No negative PINNs volume found.")


# ==========================================================
# 3. NEGATIVE VOLUME BY SPECIES / CLONE
# ==========================================================

print()
print("=" * 78)
print("3. NEGATIVE VOLUME BY SPECIES / CLONE")
print("=" * 78)

if len(negative_volume) > 0:

    combo_summary = (
        negative_volume
        .groupby(
            [
                "species",
                "clone",
            ]
        )
        .size()
        .reset_index(
            name="negative_rows"
        )
        .sort_values(
            "negative_rows",
            ascending=False,
        )
    )

    print(
        combo_summary.to_string(
            index=False
        )
    )

else:

    print("No negative PINNs volume found.")


# ==========================================================
# 4. NEGATIVE VOLUME RANGE
# ==========================================================

print()
print("=" * 78)
print("4. NEGATIVE VOLUME RANGE")
print("=" * 78)

if len(negative_volume) > 0:

    print(
        negative_volume[
            [
                "regional_id",
                "compartment_id",
                "species",
                "clone",
                "age_months",
                "growth_period",
                "pinns_dbh_cm",
                "pinns_height_m",
                "pinns_volume_m3_tree",
                "pinns_mai_m3_tree_year",
                "pinns_volume_m3_ha",
                "pinns_total_volume_m3",
            ]
        ]
        .sort_values(
            "pinns_volume_m3_tree"
        )
        .head(20)
        .to_string(index=False)
    )

else:

    print("No negative rows to inspect.")


# ==========================================================
# 5. OTHER NON-FINITE VALUES
# ==========================================================

print()
print("=" * 78)
print("5. NON-FINITE PINNs OUTPUT CHECK")
print("=" * 78)

pinns_columns = [
    "pinns_dbh_cm",
    "pinns_height_m",
    "pinns_volume_m3_tree",
    "pinns_mai_m3_tree_year",
    "pinns_volume_m3_ha",
    "pinns_total_volume_m3",
    "pinns_mai_m3_ha_year",
]

for column in pinns_columns:

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    nonfinite_count = (
        ~np.isfinite(values)
    ).sum()

    print(
        f"{column:<32} "
        f"non-finite: {nonfinite_count:,}"
    )


# ==========================================================
# 6. NEGATIVE DERIVED METRICS
# ==========================================================

print()
print("=" * 78)
print("6. NEGATIVE DERIVED METRICS")
print("=" * 78)

for column in [
    "pinns_mai_m3_tree_year",
    "pinns_volume_m3_ha",
    "pinns_total_volume_m3",
    "pinns_mai_m3_ha_year",
]:

    count = (
        pd.to_numeric(
            df[column],
            errors="coerce",
        )
        < 0
    ).sum()

    print(
        f"{column:<32} "
        f"negative: {count:,}"
    )


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print()
print("=" * 78)
print("FINAL SUMMARY")
print("=" * 78)

print(
    f"Total rows              : {len(df):,}"
)

print(
    f"Negative volume rows    : "
    f"{len(negative_volume):,}"
)

print(
    f"Minimum PINNs volume    : "
    f"{volume.min():.10f}"
)

if len(negative_volume) == 0:

    print()
    print("PINNs output validation: PASSED")
    print("No negative volume detected.")

else:

    print()
    print(
        "PINNs output validation: "
        "REQUIRES REVIEW"
    )

    print(
        "Negative volume values "
        "were detected."
    )

print()
print(
    "No model training was performed."
)
print("=" * 78)