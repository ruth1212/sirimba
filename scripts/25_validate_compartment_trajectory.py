from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "compartment_trajectory_smoke_test.csv"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print("STEP 18C — VALIDATE COMPARTMENT TRAJECTORY")
print("=" * 75)

df = pd.read_csv(INPUT_PATH)

print("\nLoaded:")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 1. COLUMN CHECK
# ============================================================

print("\n" + "=" * 75)
print("1. COLUMN CHECK")
print("=" * 75)

print("\nColumns:")

for i, column in enumerate(df.columns, start=1):
    print(f"{i:02d}. {column}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "regional_id",
    "compartment_id",
    "growth_period",
    "age_months",
    "area_ha",
    "spacing_m",
    "trees_per_ha",
    "estimated_trees_compartment",

    "rf_dbh_cm",
    "rf_height_m",
    "rf_volume_m3_tree",
    "rf_tha_m",
    "rf_mai_m3_tree_year",
    "rf_volume_m3_ha",
    "rf_total_volume_m3",

    "pinns_dbh_cm",
    "pinns_height_m",
    "pinns_volume_m3_tree",
    "pinns_mai_m3_tree_year",
    "pinns_volume_m3_ha",
    "pinns_total_volume_m3",
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        "Missing columns:\n"
        + "\n".join(missing)
    )

print("\nRequired columns: OK")


# ============================================================
# 2. ROW COUNT
# ============================================================

print("\n" + "=" * 75)
print("2. TRAJECTORY ROW COUNT")
print("=" * 75)

expected_ages = list(range(6, 301, 6))

actual_ages = (
    df["age_months"]
    .astype(int)
    .tolist()
)

print(
    f"Expected ages : {len(expected_ages)}"
)

print(
    f"Actual ages   : {len(actual_ages)}"
)

if actual_ages != expected_ages:
    raise ValueError(
        "Age trajectory does not match "
        "6-month interval from 6 to 300."
    )

print("Age trajectory: OK")


# ============================================================
# 3. GROWTH PERIOD
# ============================================================

print("\n" + "=" * 75)
print("3. GROWTH PERIOD")
print("=" * 75)

print(
    df[
        [
            "growth_period",
            "age_months",
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print("\nLast periods:")

print(
    df[
        [
            "growth_period",
            "age_months",
        ]
    ]
    .tail(10)
    .to_string(index=False)
)


# ============================================================
# 4. PHI CHECK
# ============================================================

print("\n" + "=" * 75)
print("4. PHI CHECK")
print("=" * 75)

phi = df[
    df["age_months"] >= 72
]

pre_phi = df[
    df["age_months"] < 72
]

print(
    f"Pre-PHI rows : {len(pre_phi)}"
)

print(
    f"PHI rows     : {len(phi)}"
)

if not (
    pre_phi["growth_period"]
    .str.startswith("P")
    .all()
):
    raise ValueError(
        "Unexpected pre-PHI period."
    )

if not (
    phi["growth_period"]
    == "PHI"
).all():
    raise ValueError(
        "All ages >=72 must be PHI."
    )

print("PHI rule: OK")


# ============================================================
# 5. NULL CHECK
# ============================================================

print("\n" + "=" * 75)
print("5. NULL CHECK")
print("=" * 75)

null_counts = (
    df[required_columns]
    .isna()
    .sum()
)

null_total = int(
    null_counts.sum()
)

print(
    f"Total required-column NaN: {null_total}"
)

if null_total > 0:

    print(
        null_counts[
            null_counts > 0
        ]
    )

    raise ValueError(
        "NaN detected in required output."
    )

print("NaN check: OK")


# ============================================================
# 6. NUMERIC FINITENESS
# ============================================================

print("\n" + "=" * 75)
print("6. FINITE VALUE CHECK")
print("=" * 75)

numeric_columns = [
    col
    for col in required_columns
    if pd.api.types.is_numeric_dtype(
        df[col]
    )
]

nonfinite_count = 0

for column in numeric_columns:

    values = df[column].to_numpy()

    nonfinite_count += int(
        (~np.isfinite(values)).sum()
    )

print(
    f"Non-finite values: {nonfinite_count}"
)

if nonfinite_count > 0:
    raise ValueError(
        "Non-finite numeric value detected."
    )

print("Finite value check: OK")


# ============================================================
# 7. VOLUME/HA VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("7. VOLUME / HA VALIDATION")
print("=" * 75)

expected_rf_volume_ha = (
    df["rf_volume_m3_tree"]
    * df["trees_per_ha"]
)

expected_pinns_volume_ha = (
    df["pinns_volume_m3_tree"]
    * df["trees_per_ha"]
)

rf_error = np.abs(
    df["rf_volume_m3_ha"]
    - expected_rf_volume_ha
)

pinns_error = np.abs(
    df["pinns_volume_m3_ha"]
    - expected_pinns_volume_ha
)

print(
    f"RF max error    : {rf_error.max():.12f}"
)

print(
    f"PINNs max error : {pinns_error.max():.12f}"
)

if rf_error.max() > 1e-9:
    raise ValueError(
        "RF volume/ha calculation mismatch."
    )

if pinns_error.max() > 1e-9:
    raise ValueError(
        "PINNs volume/ha calculation mismatch."
    )

print("Volume/ha calculation: OK")


# ============================================================
# 8. TOTAL VOLUME VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("8. TOTAL COMPARTMENT VOLUME")
print("=" * 75)

expected_rf_total = (
    df["rf_volume_m3_tree"]
    * df["estimated_trees_compartment"]
)

expected_pinns_total = (
    df["pinns_volume_m3_tree"]
    * df["estimated_trees_compartment"]
)

rf_total_error = np.abs(
    df["rf_total_volume_m3"]
    - expected_rf_total
)

pinns_total_error = np.abs(
    df["pinns_total_volume_m3"]
    - expected_pinns_total
)

print(
    f"RF max error    : "
    f"{rf_total_error.max():.12f}"
)

print(
    f"PINNs max error : "
    f"{pinns_total_error.max():.12f}"
)

if rf_total_error.max() > 1e-9:
    raise ValueError(
        "RF total volume calculation mismatch."
    )

if pinns_total_error.max() > 1e-9:
    raise ValueError(
        "PINNs total volume calculation mismatch."
    )

print("Total volume calculation: OK")


# ============================================================
# 9. MAI VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("9. MAI VALIDATION")
print("=" * 75)

age_years = (
    df["age_months"] / 12.0
)

expected_rf_mai = (
    df["rf_volume_m3_tree"]
    / age_years
)

expected_pinns_mai = (
    df["pinns_volume_m3_tree"]
    / age_years
)

rf_mai_error = np.abs(
    df["rf_mai_m3_tree_year"]
    - expected_rf_mai
)

pinns_mai_error = np.abs(
    df["pinns_mai_m3_tree_year"]
    - expected_pinns_mai
)

print(
    f"RF max error    : "
    f"{rf_mai_error.max():.12f}"
)

print(
    f"PINNs max error : "
    f"{pinns_mai_error.max():.12f}"
)

if rf_mai_error.max() > 1e-9:
    raise ValueError(
        "RF MAI calculation mismatch."
    )

if pinns_mai_error.max() > 1e-9:
    raise ValueError(
        "PINNs MAI calculation mismatch."
    )

print("MAI calculation: OK")


# ============================================================
# 10. MODEL DIFFERENCE
# ============================================================

print("\n" + "=" * 75)
print("10. RF VS PINNs DIFFERENCE")
print("=" * 75)

volume_difference = (
    df["rf_volume_m3_tree"]
    - df["pinns_volume_m3_tree"]
)

print(
    f"Volume difference min : "
    f"{volume_difference.min():.6f}"
)

print(
    f"Volume difference max : "
    f"{volume_difference.max():.6f}"
)

print(
    f"Volume difference mean: "
    f"{volume_difference.mean():.6f}"
)


# ============================================================
# 11. TRAJECTORY SAMPLE
# ============================================================

print("\n" + "=" * 75)
print("11. TRAJECTORY SAMPLE")
print("=" * 75)

sample_columns = [
    "growth_period",
    "age_months",
    "rf_dbh_cm",
    "rf_height_m",
    "rf_volume_m3_tree",
    "rf_tha_m",
    "rf_mai_m3_tree_year",
    "rf_volume_m3_ha",
    "rf_total_volume_m3",
]

print(
    df[
        sample_columns
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 12. FINAL
# ============================================================

print("\n" + "=" * 75)
print("STEP 18C COMPLETED")
print("=" * 75)

print(
    "\nCompartment trajectory validation: PASSED"
)

print(
    "No model training was performed."
)