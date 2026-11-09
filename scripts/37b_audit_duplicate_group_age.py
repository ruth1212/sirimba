from pathlib import Path

import pandas as pd


# ============================================================
# STEP 37B
# AUDIT DUPLICATE GROUP-AGE ROW COUNTS
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


print("=" * 80)
print("STEP 37B - AUDIT DUPLICATE GROUP-AGE")
print("=" * 80)


# ============================================================
# 1. LOAD
# ============================================================

print("\n[1/5] Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(
    f"Rows: {len(df):,}"
)


# ============================================================
# 2. FIND DUPLICATE GROUPS
# ============================================================

print(
    "\n[2/5] Finding affected group-age combinations..."
)

duplicate_mask = df.duplicated(
    keep=False
)

duplicate_rows = df.loc[
    duplicate_mask
].copy()

affected = (
    duplicate_rows[
        GROUP_COLS + [AGE_COL]
    ]
    .drop_duplicates()
    .sort_values(
        GROUP_COLS + [AGE_COL]
    )
)

print(
    f"Affected group-age combinations: "
    f"{len(affected):,}"
)


# ============================================================
# 3. COUNT ALL ROWS
# ============================================================

print(
    "\n[3/5] Counting all observations "
    "inside affected group-age..."
)

group_counts = (
    df.groupby(
        GROUP_COLS + [AGE_COL]
    )
    .size()
    .reset_index(
        name="row_count"
    )
)

affected_counts = affected.merge(
    group_counts,
    on=GROUP_COLS + [AGE_COL],
    how="left",
    validate="one_to_one",
)


print(
    affected_counts.to_string(
        index=False
    )
)


# ============================================================
# 4. CHECK DISTINCT ROWS
# ============================================================

print(
    "\n[4/5] Checking distinct rows "
    "inside affected group-age..."
)

for _, row in affected.iterrows():

    mask = pd.Series(
        True,
        index=df.index
    )

    for col in GROUP_COLS:

        mask &= (
            df[col]
            == row[col]
        )

    mask &= (
        df[AGE_COL]
        == row[AGE_COL]
    )

    group = df.loc[
        mask
    ].copy()

    total = len(group)

    distinct = (
        group
        .drop_duplicates()
        .shape[0]
    )

    print(
        f"\nGroup: "
        f"{row['regional_id']} / "
        f"{row['compartment_id']} / "
        f"{row['species']} / "
        f"{row['clone']} / "
        f"{row['planting_year']} / "
        f"P{row[AGE_COL]}"
    )

    print(
        f"  Total rows    : {total}"
    )

    print(
        f"  Distinct rows : {distinct}"
    )

    print(
        f"  Extra duplicate: "
        f"{total - distinct}"
    )


# ============================================================
# 5. FINAL DECISION
# ============================================================

print(
    "\n[5/5] Duplicate audit summary..."
)

all_30 = (
    (
        affected_counts["row_count"]
        == 30
    ).all()
)

all_31 = (
    (
        affected_counts["row_count"]
        == 31
    ).all()
)

print(
    f"All affected group-age = 30 rows: "
    f"{all_30}"
)

print(
    f"All affected group-age = 31 rows: "
    f"{all_31}"
)

print(
    "\nDataset was NOT modified."
)

print("\n" + "=" * 80)
print(
    "STEP 37B STATUS: AUDIT COMPLETE"
)
print("=" * 80)