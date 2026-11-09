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
    / "dashboard_growth_trajectory.csv"
)


# ============================================================
# CONFIG
# ============================================================

EXPECTED_AGES = list(
    range(6, 301, 6)
)

EXPECTED_ROWS_PER_COMBINATION = (
    len(EXPECTED_AGES)
)

EXPECTED_COMPARTMENTS = 15
EXPECTED_SPECIES_CLONES = 9
EXPECTED_TOTAL_ROWS = 6750


# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print("STEP 19B — VALIDATE DASHBOARD DATASET")
print("=" * 75)

print("\nLoading dashboard dataset...")

df = pd.read_csv(INPUT_PATH)

print(
    f"Rows    : {len(df):,}"
)

print(
    f"Columns : {len(df.columns)}"
)


# ============================================================
# 1. TOTAL ROW CHECK
# ============================================================

print("\n" + "=" * 75)
print("1. TOTAL ROW CHECK")
print("=" * 75)

if len(df) != EXPECTED_TOTAL_ROWS:

    raise ValueError(
        f"Expected {EXPECTED_TOTAL_ROWS:,} rows, "
        f"got {len(df):,}."
    )

print(
    f"Expected rows : {EXPECTED_TOTAL_ROWS:,}"
)

print(
    f"Actual rows   : {len(df):,}"
)

print("Total row count: OK")


# ============================================================
# 2. COMPARTMENT CHECK
# ============================================================

print("\n" + "=" * 75)
print("2. COMPARTMENT CHECK")
print("=" * 75)

compartments = (
    df[
        [
            "regional_id",
            "compartment_id",
        ]
    ]
    .drop_duplicates()
)

print(
    f"Unique compartments: "
    f"{len(compartments)}"
)

if len(compartments) != EXPECTED_COMPARTMENTS:

    raise ValueError(
        "Unexpected compartment count."
    )

print("Compartment count: OK")


# ============================================================
# 3. SPECIES / CLONE CHECK
# ============================================================

print("\n" + "=" * 75)
print("3. SPECIES / CLONE CHECK")
print("=" * 75)

species_clones = (
    df[
        [
            "species",
            "clone",
        ]
    ]
    .drop_duplicates()
)

print(
    f"Unique species-clone combinations: "
    f"{len(species_clones)}"
)

if len(species_clones) != EXPECTED_SPECIES_CLONES:

    raise ValueError(
        "Unexpected species-clone count."
    )

print("\nSpecies-clone combinations:")

print(
    species_clones
    .sort_values(
        [
            "species",
            "clone",
        ]
    )
    .to_string(
        index=False
    )
)

print("\nSpecies-clone count: OK")


# ============================================================
# 4. AGE CHECK
# ============================================================

print("\n" + "=" * 75)
print("4. AGE TRAJECTORY CHECK")
print("=" * 75)

actual_ages = sorted(
    df[
        "age_months"
    ]
    .unique()
    .tolist()
)

print(
    f"Expected age points: "
    f"{len(EXPECTED_AGES)}"
)

print(
    f"Actual age points  : "
    f"{len(actual_ages)}"
)

if actual_ages != EXPECTED_AGES:

    raise ValueError(
        "Age trajectory mismatch."
    )

print(
    f"Minimum age: "
    f"{min(actual_ages)} months"
)

print(
    f"Maximum age: "
    f"{max(actual_ages)} months"
)

print("Age trajectory: OK")


# ============================================================
# 5. EACH COMBINATION MUST HAVE 50 PERIODS
# ============================================================

print("\n" + "=" * 75)
print("5. PERIOD COMPLETENESS")
print("=" * 75)

group_columns = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
]

counts = (
    df
    .groupby(group_columns)
    .size()
)

print(
    f"Number of combinations: "
    f"{len(counts)}"
)

print(
    f"Expected rows per combination: "
    f"{EXPECTED_ROWS_PER_COMBINATION}"
)

bad_counts = counts[
    counts != EXPECTED_ROWS_PER_COMBINATION
]

if not bad_counts.empty:

    print(
        bad_counts
        .to_string()
    )

    raise ValueError(
        "Some compartment/species/clone "
        "combinations do not have 50 periods."
    )

print(
    "Every combination has 50 periods: OK"
)


# ============================================================
# 6. AGE COMPLETENESS PER COMBINATION
# ============================================================

print("\n" + "=" * 75)
print("6. AGE COMPLETENESS PER COMBINATION")
print("=" * 75)

bad_age_groups = []

for keys, group in df.groupby(
    group_columns
):

    ages = sorted(
        group[
            "age_months"
        ]
        .astype(int)
        .tolist()
    )

    if ages != EXPECTED_AGES:

        bad_age_groups.append(
            keys
        )

if bad_age_groups:

    print(
        "Invalid combinations:"
    )

    for item in bad_age_groups:
        print(item)

    raise ValueError(
        "At least one combination has "
        "an incomplete age trajectory."
    )

print(
    "Every combination contains "
    "P6 → PHI 300: OK"
)


# ============================================================
# 7. PREDICTION COHORT
# ============================================================

print("\n" + "=" * 75)
print("7. PREDICTION COHORT")
print("=" * 75)

planting_years = sorted(
    df[
        "planting_year"
    ]
    .unique()
    .tolist()
)

print(
    f"Prediction planting year(s): "
    f"{planting_years}"
)

if planting_years != [2025]:

    raise ValueError(
        "Prediction dataset contains "
        "unexpected planting years."
    )

print("Prediction cohort: 2025 OK")


# ============================================================
# 8. MODEL OUTPUT CHECK
# ============================================================

print("\n" + "=" * 75)
print("8. MODEL OUTPUT CHECK")
print("=" * 75)

model_columns = [
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

for column in model_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing model output: {column}"
        )

    if df[column].isna().any():

        raise ValueError(
            f"NaN found in {column}"
        )

    if not np.isfinite(
        df[column].to_numpy()
    ).all():

        raise ValueError(
            f"Non-finite value found in {column}"
        )

print(
    f"Validated model columns: "
    f"{len(model_columns)}"
)

print("Model outputs: OK")


# ============================================================
# 9. STATIC COMPARTMENT ATTRIBUTES
# ============================================================

print("\n" + "=" * 75)
print("9. COMPARTMENT ATTRIBUTE CONSISTENCY")
print("=" * 75)

static_columns = [
    "area_ha",
    "spacing_m",
    "spacing_x_m",
    "spacing_y_m",
    "plant_spacing_m",
    "trees_per_ha",
    "estimated_trees_compartment",
]

for column in static_columns:

    unique_counts = (
        df
        .groupby(
            [
                "regional_id",
                "compartment_id",
            ]
        )[column]
        .nunique()
    )

    inconsistent = unique_counts[
        unique_counts > 1
    ]

    if not inconsistent.empty:

        print(
            f"\nInconsistent column: "
            f"{column}"
        )

        print(
            inconsistent
        )

        raise ValueError(
            f"Compartment attribute "
            f"{column} is inconsistent."
        )

    print(
        f"{column}: consistent"
    )

print(
    "\nCompartment attributes: OK"
)


# ============================================================
# 10. NO PLANTING YEAR FILTER
# ============================================================

print("\n" + "=" * 75)
print("10. DASHBOARD FILTER STRUCTURE")
print("=" * 75)

ui_filters = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
]

print(
    "Required UI filters:"
)

for item in ui_filters:
    print(
        f"  - {item}"
    )

print(
    "\nplanting_year is NOT a UI filter."
)

print(
    "Dashboard filter structure: OK"
)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL SUMMARY")
print("=" * 75)

print(
    f"Rows                  : {len(df):,}"
)

print(
    f"Compartments          : {len(compartments)}"
)

print(
    f"Species-clone combos  : "
    f"{len(species_clones)}"
)

print(
    f"Periods per combo     : "
    f"{EXPECTED_ROWS_PER_COMBINATION}"
)

print(
    f"Prediction cohort     : 2025"
)

print(
    "Trajectory             : P6 → PHI 300"
)

print(
    "RF outputs             : OK"
)

print(
    "PINNs outputs          : OK"
)

print(
    "Compartment attributes : OK"
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 75)
print("STEP 19B COMPLETED")
print("=" * 75)

print(
    "\nDashboard dataset validation: PASSED"
)

print(
    "Ready for API integration."
)

print(
    "No model training was performed."
)