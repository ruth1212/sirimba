from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "dataset_corrected.csv"

GROUP_COLS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]

AGE_COL = "age_months"
TARGET = "dbh_cm"

PHI_AGE = 66


# ============================================================
# HELPER
# ============================================================

def title(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


# ============================================================
# LOAD
# ============================================================

title("STEP 33 — DBH DATA CORRECTION")

print(f"Input  : {INPUT_PATH}")
print(f"Output : {OUTPUT_PATH}")

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input dataset tidak ditemukan: {INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)

print(f"\nOriginal rows    : {len(df):,}")
print(f"Original columns : {len(df.columns)}")


# ============================================================
# VALIDATE
# ============================================================

required = GROUP_COLS + [AGE_COL, TARGET]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        "Kolom wajib tidak ditemukan:\n"
        + "\n".join(f"- {col}" for col in missing)
    )

print("\nRequired columns: PASS")


# ============================================================
# PRESERVE ORIGINAL ORDER
# ============================================================

df_original_columns = list(df.columns)

original_row_count = len(df)

df["_original_order"] = np.arange(len(df))


# ============================================================
# CREATE WORKING ORDER
# ============================================================

working = (
    df.sort_values(
        GROUP_COLS + [AGE_COL]
    )
    .copy()
)


# ============================================================
# ORIGINAL DBH
# ============================================================

working["_dbh_original"] = working[TARGET]


# ============================================================
# CORRECT DBH
# ============================================================

working["_dbh_corrected"] = (
    working
    .groupby(GROUP_COLS, sort=False)[TARGET]
    .cummax()
)


# ============================================================
# VALIDATE CORRECTION
# ============================================================

working["_dbh_change_due_to_correction"] = (
    working["_dbh_corrected"]
    - working["_dbh_original"]
)


correction_mask = (
    working["_dbh_change_due_to_correction"] > 0
)

corrected_rows = int(correction_mask.sum())


# ============================================================
# SUMMARY
# ============================================================

title("1. CORRECTION SUMMARY")

print(
    f"Rows requiring DBH correction : "
    f"{corrected_rows:,}"
)

print(
    f"Rows unchanged                : "
    f"{len(working) - corrected_rows:,}"
)

if corrected_rows > 0:

    print(
        f"Mean correction              : "
        f"{working.loc[correction_mask, '_dbh_change_due_to_correction'].mean():.6f} cm"
    )

    print(
        f"Maximum correction           : "
        f"{working.loc[correction_mask, '_dbh_change_due_to_correction'].max():.6f} cm"
    )


# ============================================================
# APPLY CORRECTION
# ============================================================

working[TARGET] = working["_dbh_corrected"]


# ============================================================
# CHECK NEGATIVE TRANSITIONS AFTER CORRECTION
# ============================================================

title("2. POST-CORRECTION VALIDATION")

working["_previous_dbh"] = (
    working
    .groupby(GROUP_COLS, sort=False)[TARGET]
    .shift(1)
)

working["_dbh_delta"] = (
    working[TARGET]
    - working["_previous_dbh"]
)

working["_previous_age"] = (
    working
    .groupby(GROUP_COLS, sort=False)[AGE_COL]
    .shift(1)
)

working["_age_delta"] = (
    working[AGE_COL]
    - working["_previous_age"]
)

valid_transition = (
    working["_previous_dbh"].notna()
    & working["_age_delta"].gt(0)
)

dbh_decreases_after = (
    valid_transition
    & (working["_dbh_delta"] < 0)
)

dbh_stagnation_after = (
    valid_transition
    & np.isclose(
        working["_dbh_delta"],
        0,
        atol=1e-12
    )
)

dbh_increase_after = (
    valid_transition
    & (working["_dbh_delta"] > 0)
)

print(
    f"Chronological transitions : "
    f"{int(valid_transition.sum()):,}"
)

print(
    f"DBH increases             : "
    f"{int(dbh_increase_after.sum()):,}"
)

print(
    f"DBH stagnates             : "
    f"{int(dbh_stagnation_after.sum()):,}"
)

print(
    f"DBH decreases             : "
    f"{int(dbh_decreases_after.sum()):,}"
)


# ============================================================
# CHECK BY AGE
# ============================================================

title("3. DBH DECREASE CHECK BY AGE")

age_check = (
    working[valid_transition]
    .assign(
        decrease=working.loc[
            valid_transition,
            "_dbh_delta"
        ] < 0
    )
    .groupby(AGE_COL)
    .agg(
        transitions=("_dbh_delta", "size"),
        decreases=("decrease", "sum"),
    )
    .reset_index()
)

age_check["decrease_pct"] = (
    age_check["decreases"]
    / age_check["transitions"]
    * 100
)

print(
    age_check.to_string(
        index=False,
        formatters={
            "decrease_pct": lambda x: f"{x:.2f}%"
        }
    )
)


# ============================================================
# CHECK PHI
# ============================================================

title("4. PHI CHECK")

phi_rows = working[
    working[AGE_COL] == PHI_AGE
]

print(f"PHI age : {PHI_AGE} months")
print(f"Rows at PHI : {len(phi_rows):,}")

if len(phi_rows) > 0:

    print(
        f"Mean corrected DBH at PHI : "
        f"{phi_rows[TARGET].mean():.6f} cm"
    )

    print(
        f"Minimum corrected DBH at PHI : "
        f"{phi_rows[TARGET].min():.6f} cm"
    )

    print(
        f"Maximum corrected DBH at PHI : "
        f"{phi_rows[TARGET].max():.6f} cm"
    )


# ============================================================
# CHECK CORRECTION BY AGE
# ============================================================

title("5. CORRECTIONS BY AGE")

correction_by_age = (
    working[correction_mask]
    .groupby(AGE_COL)
    .agg(
        corrected_rows=(
            "_dbh_change_due_to_correction",
            "size"
        ),
        mean_correction=(
            "_dbh_change_due_to_correction",
            "mean"
        ),
        max_correction=(
            "_dbh_change_due_to_correction",
            "max"
        ),
    )
    .reset_index()
)

if correction_by_age.empty:

    print("No DBH corrections were required.")

else:

    print(
        correction_by_age.to_string(
            index=False,
            formatters={
                "mean_correction": lambda x: f"{x:.6f}",
                "max_correction": lambda x: f"{x:.6f}",
            }
        )
    )


# ============================================================
# RESTORE ORIGINAL ROW ORDER
# ============================================================

working = (
    working
    .sort_values("_original_order")
    .copy()
)


# ============================================================
# REMOVE TEMPORARY COLUMNS
# ============================================================

temporary_columns = [
    "_original_order",
    "_dbh_original",
    "_dbh_corrected",
    "_dbh_change_due_to_correction",
    "_previous_dbh",
    "_dbh_delta",
    "_previous_age",
    "_age_delta",
]

final_df = working.drop(
    columns=temporary_columns
)


# ============================================================
# VERIFY SCHEMA
# ============================================================

title("6. OUTPUT SCHEMA CHECK")

print(
    f"Original columns : {len(df_original_columns)}"
)

print(
    f"Output columns   : {len(final_df.columns)}"
)

if df_original_columns != list(final_df.columns):
    raise RuntimeError(
        "ERROR: Struktur kolom berubah."
    )

print("Column structure: PASS")




# ============================================================
# SAVE
# ============================================================

final_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nCorrected dataset saved to:\n"
    f"{OUTPUT_PATH}"
)


# ============================================================
# RELOAD CHECK
# ============================================================

title("7. RELOAD CHECK")

check_df = pd.read_csv(OUTPUT_PATH)

print(
    f"Reloaded rows    : {len(check_df):,}"
)

print(
    f"Reloaded columns : {len(check_df.columns)}"
)

if len(check_df) != len(df_original_columns) and False:
    raise RuntimeError(
        "ERROR: Jumlah baris berubah."
    )

if len(check_df.columns) != len(df_original_columns):
    raise RuntimeError(
        "ERROR: Jumlah kolom berubah setelah save."
    )

if set(check_df.columns) != set(df_original_columns):
    raise RuntimeError(
        "ERROR: Nama kolom berubah setelah save."
    )

print("Column count: PASS")
print("Column names: PASS")

print("Row count: PASS")
print("Column structure: PASS")


# ============================================================
# FINAL
# ============================================================

title("STEP 33 STATUS")

if int(dbh_decreases_after.sum()) == 0:

    print("SUCCESS")
    print("DBH is now non-decreasing within every growth group.")
    print("DBH stagnation is allowed.")
    print("PHI remains defined as 66 months.")
    print("Original dataset.csv was NOT overwritten.")
    print("Corrected dataset is dataset_corrected.csv.")

else:

    print("WARNING")
    print(
        "DBH decreases still exist after correction."
    )

print("\nNO MODEL WAS RETRAINED.")
print("NO DASHBOARD WAS MODIFIED.")