from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "dataset.csv"
)


# ============================================================
# COLUMNS
# ============================================================

COLUMNS = [
    "regional_id",
    "regional_name",
    "compartment_id",
    "compartment_code",
    "area_ha",
    "stand_area_total_ha",
    "spacing_m",
    "spacing_x_m",
    "spacing_y_m",
    "plant_spacing_m",
    "trees_per_ha",
    "estimated_trees_compartment",
    "species_code",
    "species",
    "clone",
    "clone_factor",
    "species_clone",
    "planting_year",
    "planting_cohort",
    "inventory_year",
    "age_months",
    "growth_period",
    "disease_score",
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
    "tha_m",
]


print("=" * 75)
print("STEP 18A — INSPECT GROWTH TRAJECTORY STRUCTURE")
print("=" * 75)


# ============================================================
# LOAD
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    usecols=COLUMNS,
)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df)}")


# ============================================================
# BASIC VALIDATION
# ============================================================

missing_columns = [
    col
    for col in COLUMNS
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing columns: "
        + ", ".join(missing_columns)
    )


print("\nRequired columns: OK")


# ============================================================
# DUPLICATES
# ============================================================

duplicates = df.duplicated().sum()

print(
    f"\nDuplicate rows: {duplicates:,}"
)


# ============================================================
# REGIONS
# ============================================================

print("\n" + "=" * 75)
print("1. REGIONAL STRUCTURE")
print("=" * 75)

print(
    df[
        [
            "regional_id",
            "regional_name",
        ]
    ]
    .drop_duplicates()
    .sort_values("regional_id")
    .to_string(index=False)
)


# ============================================================
# COMPARTMENTS
# ============================================================

print("\n" + "=" * 75)
print("2. COMPARTMENT STRUCTURE")
print("=" * 75)

compartments = (
    df[
        [
            "regional_id",
            "regional_name",
            "compartment_id",
            "compartment_code",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "regional_id",
            "compartment_id",
        ]
    )
)

print(
    f"Unique compartments: "
    f"{len(compartments):,}"
)

print(
    compartments.to_string(index=False)
)


# ============================================================
# SPECIES
# ============================================================

print("\n" + "=" * 75)
print("3. SPECIES")
print("=" * 75)

species = (
    df[
        [
            "species_code",
            "species",
        ]
    ]
    .drop_duplicates()
    .sort_values("species_code")
)

print(
    f"Unique species: {len(species)}"
)

print(
    species.to_string(index=False)
)


# ============================================================
# CLONES
# ============================================================

print("\n" + "=" * 75)
print("4. CLONE")
print("=" * 75)

clones = (
    df[
        [
            "species",
            "clone",
            "clone_factor",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "species",
            "clone",
        ]
    )
)

print(
    f"Unique species-clone combinations: "
    f"{len(clones):,}"
)

print(
    clones.to_string(index=False)
)


# ============================================================
# PLANTING YEARS
# ============================================================

print("\n" + "=" * 75)
print("5. HISTORICAL PLANTING YEARS")
print("=" * 75)

planting_years = sorted(
    df["planting_year"]
    .dropna()
    .astype(int)
    .unique()
)

print(
    "Planting years:"
)

print(
    planting_years
)


if len(planting_years) >= 2:

    intervals = np.diff(
        planting_years
    )

    print(
        "\nPlanting-year intervals:"
    )

    print(
        intervals.tolist()
    )

    print(
        "\nMedian planting interval:"
    )

    print(
        float(np.median(intervals))
    )


# ============================================================
# GROWTH PERIOD
# ============================================================

print("\n" + "=" * 75)
print("6. GROWTH PERIOD STRUCTURE")
print("=" * 75)

period_summary = (
    df[
        [
            "growth_period",
            "age_months",
        ]
    ]
    .drop_duplicates()
    .sort_values("age_months")
)

print(
    period_summary.to_string(index=False)
)


# ============================================================
# PERIOD COUNT
# ============================================================

print("\nUnique growth periods:")

print(
    df["growth_period"]
    .nunique()
)


# ============================================================
# COMPARTMENT ATTRIBUTES
# ============================================================

print("\n" + "=" * 75)
print("7. COMPARTMENT ATTRIBUTES")
print("=" * 75)

compartment_attributes = (
    df.groupby(
        [
            "regional_id",
            "compartment_id",
        ],
        as_index=False,
    )
    .agg(
        area_ha=("area_ha", "first"),
        stand_area_total_ha=(
            "stand_area_total_ha",
            "first",
        ),
        spacing_m=("spacing_m", "first"),
        spacing_x_m=("spacing_x_m", "first"),
        spacing_y_m=("spacing_y_m", "first"),
        plant_spacing_m=(
            "plant_spacing_m",
            "first",
        ),
        trees_per_ha=(
            "trees_per_ha",
            "mean",
        ),
        estimated_trees_compartment=(
            "estimated_trees_compartment",
            "mean",
        ),
    )
)

print(
    f"Compartment attribute rows: "
    f"{len(compartment_attributes):,}"
)

print(
    compartment_attributes.head(
        20
    ).to_string(index=False)
)


# ============================================================
# COMPARTMENT × PLANTING YEAR
# ============================================================

print("\n" + "=" * 75)
print("8. COMPARTMENT × PLANTING YEAR")
print("=" * 75)

cohort_summary = (
    df.groupby(
        [
            "regional_id",
            "compartment_id",
            "planting_year",
        ],
        as_index=False,
    )
    .agg(
        species_count=(
            "species",
            "nunique",
        ),
        clone_count=(
            "clone",
            "nunique",
        ),
        min_age_months=(
            "age_months",
            "min",
        ),
        max_age_months=(
            "age_months",
            "max",
        ),
        period_count=(
            "growth_period",
            "nunique",
        ),
        rows=("age_months", "size"),
    )
)

print(
    f"Compartment-cohort groups: "
    f"{len(cohort_summary):,}"
)

print(
    cohort_summary.head(
        30
    ).to_string(index=False)
)


# ============================================================
# COMPARTMENT × PERIOD AVERAGES
# ============================================================

print("\n" + "=" * 75)
print("9. AVERAGE GROWTH PER COMPARTMENT × PERIOD")
print("=" * 75)

growth_summary = (
    df.groupby(
        [
            "regional_id",
            "compartment_id",
            "growth_period",
            "age_months",
        ],
        as_index=False,
    )
    .agg(
        avg_dbh_cm=(
            "dbh_cm",
            "mean",
        ),
        avg_height_m=(
            "height_m",
            "mean",
        ),
        avg_volume_m3_tree=(
            "volume_m3_tree",
            "mean",
        ),
        avg_disease_score=(
            "disease_score",
            "mean",
        ),
        avg_tha_m=(
            "tha_m",
            "mean",
        ),
        trees_per_ha=(
            "trees_per_ha",
            "mean",
        ),
        area_ha=(
            "area_ha",
            "first",
        ),
        estimated_trees_compartment=(
            "estimated_trees_compartment",
            "first",
        ),
        tree_rows=(
            "dbh_cm",
            "size",
        ),
    )
)

print(
    f"Growth summary rows: "
    f"{len(growth_summary):,}"
)

print(
    growth_summary.head(
        30
    ).to_string(index=False)
)


# ============================================================
# PHI
# ============================================================

print("\n" + "=" * 75)
print("10. PHI STRUCTURE")
print("=" * 75)

phi = df[
    df["growth_period"]
    .astype(str)
    .str.upper()
    .eq("PHI")
]

print(
    f"PHI rows: {len(phi):,}"
)

print(
    f"PHI compartments: "
    f"{phi['compartment_id'].nunique():,}"
)

print(
    f"PHI species: "
    f"{phi['species'].nunique():,}"
)

print(
    f"PHI clones: "
    f"{phi['clone'].nunique():,}"
)


# ============================================================
# PHI SPECIES/CLONE SUMMARY
# ============================================================

phi_summary = (
    phi.groupby(
        [
            "species",
            "clone",
        ],
        as_index=False,
    )
    .agg(
        avg_dbh_cm=(
            "dbh_cm",
            "mean",
        ),
        avg_height_m=(
            "height_m",
            "mean",
        ),
        avg_volume_m3_tree=(
            "volume_m3_tree",
            "mean",
        ),
        avg_tha_m=(
            "tha_m",
            "mean",
        ),
        avg_disease_score=(
            "disease_score",
            "mean",
        ),
        observations=(
            "dbh_cm",
            "size",
        ),
    )
)

print("\nPHI species-clone summary:")

print(
    phi_summary.to_string(
        index=False
    )
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("STEP 18A COMPLETED")
print("=" * 75)

print(
    "\nNO MODEL TRAINING WAS PERFORMED."
)

print(
    "This step only inspects the actual "
    "growth structure needed for the "
    "compartment trajectory engine."
)

print(
    "\nNext:"
)

print(
    "STEP 18B — BUILD COMPARTMENT "
    "GROWTH TRAJECTORY ENGINE"
)

print("=" * 75)