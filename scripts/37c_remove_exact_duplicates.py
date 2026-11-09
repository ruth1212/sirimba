from pathlib import Path

import pandas as pd


# ============================================================
# STEP 37C
# REMOVE VERIFIED EXACT DUPLICATES
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "dataset_phi66_final.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "dataset_phi66_model.csv"
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
print("STEP 37C - REMOVE VERIFIED EXACT DUPLICATES")
print("=" * 80)


# ============================================================
# 1. LOAD
# ============================================================

print("\n[1/7] Loading validated dataset...")

df = pd.read_csv(INPUT_FILE)

print(
    f"Rows before: {len(df):,}"
)

print(
    f"Columns     : {len(df.columns):,}"
)


# ============================================================
# 2. COUNT DUPLICATES
# ============================================================

print("\n[2/7] Counting exact duplicates...")

duplicate_count = int(
    df.duplicated().sum()
)

print(
    f"Exact duplicates: {duplicate_count:,}"
)

if duplicate_count != 4:
    raise ValueError(
        "Expected exactly 4 verified exact "
        "duplicate rows."
    )


# ============================================================
# 3. REMOVE
# ============================================================

print("\n[3/7] Removing exact duplicates...")

df_model = (
    df.drop_duplicates(
        keep="first"
    )
    .reset_index(drop=True)
)

print(
    f"Rows after: {len(df_model):,}"
)


# ============================================================
# 4. VERIFY NO DUPLICATES
# ============================================================

print("\n[4/7] Verifying duplicate removal...")

remaining_duplicates = int(
    df_model.duplicated().sum()
)

print(
    f"Remaining exact duplicates: "
    f"{remaining_duplicates:,}"
)

if remaining_duplicates != 0:
    raise ValueError(
        "Duplicates remain after removal."
    )


# ============================================================
# 5. VERIFY GROUP-AGE STRUCTURE
# ============================================================

print(
    "\n[5/7] Verifying group-age structure..."
)

group_age = (
    df_model.groupby(
        GROUP_COLS + [AGE_COL]
    )
    .size()
)

print(
    f"Group-age combinations: "
    f"{len(group_age):,}"
)

print(
    "Observations/group-age:"
)

print(
    group_age
    .value_counts()
    .sort_index()
)

group_age_pass = (
    group_age.nunique() == 1
    and group_age.iloc[0] == 30
)

print(
    "\n30 observations/group-age:",
    "PASS"
    if group_age_pass
    else "FAIL"
)

if not group_age_pass:
    raise ValueError(
        "Group-age structure changed unexpectedly."
    )


# ============================================================
# 6. SAVE
# ============================================================

print("\n[6/7] Saving model dataset...")

df_model.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"Saved to:\n{OUTPUT_FILE}"
)


# ============================================================
# 7. FINAL CHECK
# ============================================================

print("\n[7/7] Final checks...")

checks = {
    "Rows":
        len(df_model) == 218696,

    "Columns":
        len(df_model.columns) == 31,

    "Duplicates":
        remaining_duplicates == 0,

    "Group-age":
        group_age_pass,
}


for name, result in checks.items():

    print(
        f"{name:<16}: "
        f"{'PASS' if result else 'FAIL'}"
    )


print("\n" + "=" * 80)

if all(checks.values()):

    print(
        "STEP 37C STATUS: PASS"
    )

    print(
        "\nModel dataset is ready for "
        "the next preparation audit."
    )

else:

    print(
        "STEP 37C STATUS: REVIEW REQUIRED"
    )

print("=" * 80)

print(
    "\nOriginal dataset_phi66_final.csv "
    "was NOT modified."
)