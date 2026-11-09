from pathlib import Path

import pandas as pd


# ============================================================
# STEP 37A
# INSPECT EXACT DUPLICATES
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "dataset_phi66_final.csv"
)

GROUP_COLS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]

AGE_COL = "age_months"

MODEL_COLUMNS = [
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha",
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]


print("=" * 80)
print("STEP 37A - INSPECT EXACT DUPLICATES")
print("=" * 80)


# ============================================================
# 1. LOAD
# ============================================================

print("\n[1/5] Loading final dataset...")

df = pd.read_csv(INPUT_FILE)

print(
    f"Rows    : {len(df):,}"
)

print(
    f"Columns : {len(df.columns):,}"
)


# ============================================================
# 2. FIND DUPLICATES
# ============================================================

print("\n[2/5] Finding exact duplicate rows...")

duplicate_mask = df.duplicated(
    keep=False
)

duplicates = (
    df.loc[
        duplicate_mask
    ]
    .copy()
)

duplicate_count = int(
    df.duplicated().sum()
)

print(
    f"Duplicate rows excluding first occurrence: "
    f"{duplicate_count:,}"
)

print(
    f"All rows participating in duplicate sets: "
    f"{len(duplicates):,}"
)


# ============================================================
# 3. DISPLAY DUPLICATES
# ============================================================

print("\n[3/5] Duplicate rows:")

if len(duplicates) > 0:

    print(
        duplicates[
            MODEL_COLUMNS
        ].to_string(
            index=False
        )
    )

else:

    print(
        "No duplicates found."
    )


# ============================================================
# 4. GROUP / AGE CONTEXT
# ============================================================

print(
    "\n[4/5] Duplicate group-age context..."
)

if len(duplicates) > 0:

    duplicate_context = (
        duplicates
        .groupby(
            GROUP_COLS + [AGE_COL]
        )
        .size()
        .reset_index(
            name="duplicate_rows"
        )
    )

    print(
        duplicate_context.to_string(
            index=False
        )
    )

    print(
        "\nNumber of affected group-age combinations:",
        len(duplicate_context)
    )

    print(
        "Number of affected growth groups:",
        duplicate_context[
            GROUP_COLS
        ]
        .drop_duplicates()
        .shape[0]
    )

else:

    print(
        "No affected groups."
    )


# ============================================================
# 5. FINAL INSPECTION SUMMARY
# ============================================================

print(
    "\n[5/5] Inspection summary..."
)

print(
    f"Exact duplicate count: "
    f"{duplicate_count:,}"
)

print(
    "\nIMPORTANT:"
)

print(
    "Dataset has NOT been modified."
)

print(
    "No duplicate rows have been removed."
)

print(
    "\n" + "=" * 80
)

print(
    "STEP 37A STATUS: INSPECTION COMPLETE"
)

print("=" * 80)