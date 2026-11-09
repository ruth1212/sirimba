from pathlib import Path
import pandas as pd


# ============================================================
# STEP 36N
# TRIM DATASET TO PHI = 66 MONTHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "dataset_growth_logic_v2.csv"
OUTPUT_FILE = BASE_DIR / "data" / "dataset_phi66.csv"

AGE_COL = "age_months"
PHI_AGE = 66


print("=" * 70)
print("STEP 36N - TRIM DATASET TO PHI = 66 MONTHS")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD
# ------------------------------------------------------------

print("\n[1/6] Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Input rows    : {len(df):,}")
print(f"Input columns : {len(df.columns):,}")


# ------------------------------------------------------------
# 2. CHECK AGE
# ------------------------------------------------------------

print("\n[2/6] Checking age range...")

print(
    f"Original age range: "
    f"P{int(df[AGE_COL].min())} - P{int(df[AGE_COL].max())}"
)

print(
    f"Rows at PHI P66: "
    f"{(df[AGE_COL] == PHI_AGE).sum():,}"
)


# ------------------------------------------------------------
# 3. TRIM
# ------------------------------------------------------------

print("\n[3/6] Keeping only P6-P66...")

df_phi = df[
    (df[AGE_COL] >= 6)
    & (df[AGE_COL] <= PHI_AGE)
].copy()

df_phi = df_phi.sort_values(
    [
        "regional_id",
        "compartment_id",
        "species",
        "clone",
        "planting_year",
        AGE_COL,
    ]
).reset_index(drop=True)


# ------------------------------------------------------------
# 4. VALIDATE AGE SET
# ------------------------------------------------------------

print("\n[4/6] Validating final age structure...")

ages = sorted(
    df_phi[AGE_COL]
    .dropna()
    .unique()
    .tolist()
)

print("Final ages:")

for age in ages:
    label = "PHI" if age == PHI_AGE else f"P{int(age)}"
    count = (df_phi[AGE_COL] == age).sum()

    print(
        f"  {label:>4} : {count:,} rows"
    )


expected_ages = list(range(6, 67, 6))

if ages != expected_ages:
    raise ValueError(
        f"Unexpected age structure.\n"
        f"Expected: {expected_ages}\n"
        f"Found   : {ages}"
    )


# ------------------------------------------------------------
# 5. VALIDATE PHI
# ------------------------------------------------------------

phi_rows = df_phi[
    df_phi[AGE_COL] == PHI_AGE
]

if phi_rows.empty:
    raise ValueError(
        "PHI P66 tidak ditemukan."
    )

rows_above_phi = (
    df_phi[AGE_COL] > PHI_AGE
).sum()

if rows_above_phi != 0:
    raise ValueError(
        f"Masih ada {rows_above_phi:,} rows di atas P66."
    )


# ------------------------------------------------------------
# 6. SAVE
# ------------------------------------------------------------

print("\n[5/6] Saving final PHI dataset...")

df_phi.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(
    f"Output: {OUTPUT_FILE}"
)


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\n[6/6] Final validation...")

print("=" * 70)

print(
    f"Final rows       : {len(df_phi):,}"
)

print(
    f"Final columns    : {len(df_phi.columns):,}"
)

print(
    f"Minimum age      : P{int(df_phi[AGE_COL].min())}"
)

print(
    f"Maximum age      : P{int(df_phi[AGE_COL].max())}"
)

print(
    f"PHI age          : P{PHI_AGE}"
)

print(
    f"Rows above PHI   : "
    f"{(df_phi[AGE_COL] > PHI_AGE).sum():,}"
)


# Group-age count
group_cols = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]

group_age = (
    df_phi.groupby(group_cols + [AGE_COL])
    .size()
)

print(
    f"Growth groups    : "
    f"{df_phi[group_cols].drop_duplicates().shape[0]:,}"
)

print(
    f"Group-age rows   : "
    f"{len(group_age):,}"
)

print(
    f"Observations per "
    f"group-age:\n{group_age.value_counts().sort_index()}"
)

# Expected rows
expected_rows = (
    df_phi[group_cols].drop_duplicates().shape[0]
    * len(expected_ages)
    * 30
)

print(
    f"\nExpected rows    : {expected_rows:,}"
)

print(
    f"Actual rows      : {len(df_phi):,}"
)

print(
    "\nFINAL CHECK:"
)

print(
    "PHI = P66 only   :",
    "PASS"
    if (
        df_phi[AGE_COL].max() == 66
        and rows_above_phi == 0
    )
    else "FAIL"
)

print(
    "Age structure    :",
    "PASS"
    if ages == expected_ages
    else "FAIL"
)

print("=" * 70)
print("Original dataset_growth_logic_v2.csv was NOT modified.")
print("Raw dataset.csv was NOT modified.")
print("=" * 70)