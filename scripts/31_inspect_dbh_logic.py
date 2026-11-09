from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"

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
# LOAD DATA
# ============================================================

title("STEP 32 — DBH GROWTH LOGIC INSPECTION")

print(f"Dataset : {DATA_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# VALIDATION
# ============================================================

required = GROUP_COLS + [AGE_COL, TARGET]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        "Kolom berikut tidak ditemukan:\n"
        + "\n".join(f"- {col}" for col in missing)
    )

print("\nRequired columns: PASS")


# ============================================================
# GROUP + AGE MEAN
# ============================================================

title("1. GROUP-AGE DBH PROFILE")

group_age = (
    df.groupby(
        GROUP_COLS + [AGE_COL],
        dropna=False
    )[TARGET]
    .mean()
    .reset_index()
    .sort_values(GROUP_COLS + [AGE_COL])
)

print(f"Group-age records : {len(group_age):,}")


# ============================================================
# PREVIOUS AGE
# ============================================================

group_age["previous_age"] = (
    group_age.groupby(GROUP_COLS)[AGE_COL]
    .shift(1)
)

group_age["previous_dbh"] = (
    group_age.groupby(GROUP_COLS)[TARGET]
    .shift(1)
)

group_age["age_gap"] = (
    group_age[AGE_COL] -
    group_age["previous_age"]
)

group_age["dbh_delta"] = (
    group_age[TARGET] -
    group_age["previous_dbh"]
)

transitions = group_age[
    group_age["previous_age"].notna()
    & group_age["age_gap"].gt(0)
].copy()

print(f"Chronological transitions : {len(transitions):,}")


# ============================================================
# 2. DBH DECREASE CLASSIFICATION
# ============================================================

title("2. DBH DECREASE CLASSIFICATION")

transitions["dbh_decrease"] = (
    transitions["dbh_delta"] < 0
)

transitions["age_zone"] = np.where(
    transitions[AGE_COL] <= PHI_AGE,
    "P6-PHI",
    "AFTER_PHI"
)

print("\nPHI definition:")
print(f"PHI = {PHI_AGE} months")

print("\nAge zones:")
print("P6-PHI    : current age <= 66 months")
print("AFTER_PHI : current age > 66 months")


# ============================================================
# 3. SUMMARY BY AGE ZONE
# ============================================================

title("3. DBH DECREASE BY AGE ZONE")

summary = (
    transitions
    .groupby("age_zone")
    .agg(
        transitions=("dbh_delta", "size"),
        decreases=("dbh_decrease", "sum"),
        mean_delta=("dbh_delta", "mean"),
        minimum_delta=("dbh_delta", "min"),
        maximum_delta=("dbh_delta", "max"),
    )
    .reset_index()
)

summary["decrease_pct"] = (
    summary["decreases"]
    / summary["transitions"]
    * 100
)

print(
    summary.to_string(
        index=False,
        formatters={
            "mean_delta": lambda x: f"{x:.6f}",
            "minimum_delta": lambda x: f"{x:.6f}",
            "maximum_delta": lambda x: f"{x:.6f}",
            "decrease_pct": lambda x: f"{x:.2f}%",
        }
    )
)


# ============================================================
# 4. DECREASES BEFORE PHI
# ============================================================

title("4. DBH DECREASES BEFORE / AT PHI")

before_phi = transitions[
    transitions[AGE_COL] <= PHI_AGE
].copy()

print(f"Transitions P6-PHI : {len(before_phi):,}")
print(
    f"DBH decreases      : "
    f"{before_phi['dbh_decrease'].sum():,}"
)

if len(before_phi) > 0:
    print(
        f"Decrease percentage: "
        f"{before_phi['dbh_decrease'].mean() * 100:.2f}%"
    )


# ============================================================
# 5. DECREASES AFTER PHI
# ============================================================

title("5. DBH DECREASES AFTER PHI")

after_phi = transitions[
    transitions[AGE_COL] > PHI_AGE
].copy()

print(f"Transitions > PHI : {len(after_phi):,}")
print(
    f"DBH decreases     : "
    f"{after_phi['dbh_decrease'].sum():,}"
)

if len(after_phi) > 0:
    print(
        f"Decrease percentage: "
        f"{after_phi['dbh_decrease'].mean() * 100:.2f}%"
    )


# ============================================================
# 6. DBH DECREASE BY AGE
# ============================================================

title("6. DBH DECREASE RATE BY AGE")

by_age = (
    transitions
    .groupby(AGE_COL)
    .agg(
        transitions=("dbh_delta", "size"),
        decreases=("dbh_decrease", "sum"),
        mean_delta=("dbh_delta", "mean"),
        min_delta=("dbh_delta", "min"),
    )
    .reset_index()
)

by_age["decrease_pct"] = (
    by_age["decreases"]
    / by_age["transitions"]
    * 100
)

print(
    by_age.to_string(
        index=False,
        formatters={
            "mean_delta": lambda x: f"{x:.6f}",
            "min_delta": lambda x: f"{x:.6f}",
            "decrease_pct": lambda x: f"{x:.2f}%",
        }
    )
)


# ============================================================
# 7. LARGEST DBH DECREASES
# ============================================================

title("7. LARGEST DBH DECREASES")

worst = (
    transitions[
        transitions["dbh_decrease"]
    ]
    .sort_values("dbh_delta")
    .head(20)
)

worst_columns = (
    GROUP_COLS
    + [
        "previous_age",
        AGE_COL,
        "previous_dbh",
        TARGET,
        "dbh_delta",
    ]
)

print(
    worst[worst_columns].to_string(
        index=False
    )
)


# ============================================================
# 8. DBH DECREASES BEFORE 66 MONTHS
# ============================================================

title("8. DBH DECREASES BEFORE / AT 66 MONTHS")

early = transitions[
    transitions[AGE_COL] <= PHI_AGE
].copy()

early_decreases = early[
    early["dbh_decrease"]
].copy()

print(
    f"Total early decreases : "
    f"{len(early_decreases):,}"
)

if not early_decreases.empty:

    early_by_age = (
        early_decreases
        .groupby(AGE_COL)
        .size()
        .reset_index(
            name="decrease_count"
        )
        .sort_values(AGE_COL)
    )

    print("\nEarly decrease ages:")

    print(
        early_by_age.to_string(
            index=False
        )
    )


# ============================================================
# 9. DBH DECREASE MAGNITUDE
# ============================================================

title("9. DBH DECREASE MAGNITUDE")

decreases = transitions[
    transitions["dbh_decrease"]
].copy()

if decreases.empty:

    print("No DBH decreases detected.")

else:

    print(
        f"Number of decreases : "
        f"{len(decreases):,}"
    )

    print(
        f"Mean decrease       : "
        f"{decreases['dbh_delta'].mean():.6f} cm"
    )

    print(
        f"Median decrease     : "
        f"{decreases['dbh_delta'].median():.6f} cm"
    )

    print(
        f"Minimum decrease    : "
        f"{decreases['dbh_delta'].min():.6f} cm"
    )

    print(
        f"Maximum decrease    : "
        f"{decreases['dbh_delta'].max():.6f} cm"
    )


# ============================================================
# 10. CHECK WHETHER DECREASE IS ONLY A LATE-AGE PROBLEM
# ============================================================

title("10. EARLY VS LATE DBH PROBLEM")

early_count = int(
    (
        transitions[AGE_COL] <= PHI_AGE
    )
    .sum()
)

late_count = int(
    (
        transitions[AGE_COL] > PHI_AGE
    )
    .sum()
)

early_decrease_count = int(
    (
        (transitions[AGE_COL] <= PHI_AGE)
        & transitions["dbh_decrease"]
    ).sum()
)

late_decrease_count = int(
    (
        (transitions[AGE_COL] > PHI_AGE)
        & transitions["dbh_decrease"]
    ).sum()
)

print(
    f"Before/at PHI transitions : "
    f"{early_count:,}"
)

print(
    f"Before/at PHI decreases   : "
    f"{early_decrease_count:,}"
)

print(
    f"After PHI transitions     : "
    f"{late_count:,}"
)

print(
    f"After PHI decreases       : "
    f"{late_decrease_count:,}"
)


# ============================================================
# 11. PROPOSED RULE — NO DATA MODIFICATION
# ============================================================

title("11. LOGIC CHECK")

print(
    """
Mentor rule:
DBH tidak boleh menurun terhadap umur.
DBH boleh stagnan pada umur tua.

Current dataset:
- decrease transitions masih ditemukan.
- decrease ditemukan sebelum dan sesudah PHI.
- decrease lebih sering setelah PHI.

KESIMPULAN:
Dataset memerlukan aturan koreksi DBH,
tetapi metode koreksi BELUM ditentukan pada tahap ini.

Script ini TIDAK mengubah dataset.
"""
)


# ============================================================
# FINAL STATUS
# ============================================================

title("STEP 32 STATUS")

print("DBH logic inspection completed.")
print("Dataset was NOT modified.")
print("No DBH correction was applied.")
print("No model was retrained.")
print("PHI is fixed at 66 months.")