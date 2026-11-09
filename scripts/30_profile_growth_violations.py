from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
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

TARGETS = [
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]

AUX_COLS = [
    "disease_score",
    "condition_score",
]


# ============================================================
# HELPERS
# ============================================================

def print_title(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def format_pct(value):
    return f"{value:.2f}%"


# ============================================================
# LOAD DATA
# ============================================================

print_title("STEP 30 — GROWTH VIOLATION PROFILING")

print(f"Dataset : {DATA_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = GROUP_COLS + [AGE_COL] + TARGETS + AUX_COLS

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        "Kolom berikut tidak ditemukan:\n"
        + "\n".join(f"- {c}" for c in missing)
    )

print("\nRequired columns: PASS")


# ============================================================
# BASIC STRUCTURE
# ============================================================

print_title("1. DATA STRUCTURE")

print(f"Unique growth groups : {df[GROUP_COLS].drop_duplicates().shape[0]:,}")
print(f"Unique age_months    : {df[AGE_COL].nunique():,}")

print("\nAge distribution:")

age_distribution = (
    df.groupby(AGE_COL)
      .size()
      .reset_index(name="rows")
      .sort_values(AGE_COL)
)

print(age_distribution.to_string(index=False))


# ============================================================
# GROUP + AGE AGGREGATION
# ============================================================

print_title("2. GROUP-AGE AGGREGATION")

print(
    "\nGrowth comparison is performed at:\n"
    "regional_id + compartment_id + species + clone + planting_year\n"
    "and then aggregated by age_months.\n"
)

agg_cols = TARGETS + AUX_COLS

group_age = (
    df.groupby(GROUP_COLS + [AGE_COL], dropna=False)[agg_cols]
      .mean()
      .reset_index()
      .sort_values(GROUP_COLS + [AGE_COL])
)

print(f"Group-age records : {len(group_age):,}")


# ============================================================
# CONSECUTIVE AGE TRANSITIONS
# ============================================================

print_title("3. CONSECUTIVE AGE TRANSITIONS")

group_age["previous_age_months"] = (
    group_age.groupby(GROUP_COLS)[AGE_COL]
             .shift(1)
)

for target in TARGETS:
    group_age[f"previous_{target}"] = (
        group_age.groupby(GROUP_COLS)[target]
                 .shift(1)
    )

for aux in AUX_COLS:
    group_age[f"previous_{aux}"] = (
        group_age.groupby(GROUP_COLS)[aux]
                 .shift(1)
    )


# Only actual chronological transitions.
# This avoids comparing unrelated ages if an age is missing.
group_age["age_gap"] = (
    group_age[AGE_COL] -
    group_age["previous_age_months"]
)

transitions = group_age[
    group_age["previous_age_months"].notna()
    & group_age["age_gap"].gt(0)
].copy()

print(f"Transitions analyzed : {len(transitions):,}")


# ============================================================
# TARGET DELTAS
# ============================================================

for target in TARGETS:
    transitions[f"delta_{target}"] = (
        transitions[target]
        - transitions[f"previous_{target}"]
    )


# ============================================================
# 4. VIOLATION SUMMARY
# ============================================================

print_title("4. GROWTH DECREASE SUMMARY")

for target in TARGETS:

    delta_col = f"delta_{target}"

    decrease = transitions[delta_col] < 0

    total = int(decrease.sum())

    unique_groups = (
        transitions.loc[decrease, GROUP_COLS]
        .drop_duplicates()
        .shape[0]
    )

    print(f"\nTARGET: {target}")
    print(f"  Decrease transitions : {total:,}")
    print(f"  Affected groups      : {unique_groups:,}")

    if len(transitions) > 0:
        pct = total / len(transitions) * 100
        print(f"  Percentage           : {format_pct(pct)}")


# ============================================================
# 5. DECREASES BY AGE
# ============================================================

print_title("5. DECREASES BY CURRENT AGE")

for target in TARGETS:

    delta_col = f"delta_{target}"

    table = (
        transitions.assign(
            decrease=transitions[delta_col] < 0
        )
        .groupby(AGE_COL)
        .agg(
            transitions=(delta_col, "size"),
            decreases=("decrease", "sum"),
        )
        .reset_index()
    )

    table["decrease_pct"] = (
        table["decreases"]
        / table["transitions"]
        * 100
    )

    table = table[table["decreases"] > 0]

    print(f"\n--- {target} ---")

    if table.empty:
        print("No decreases detected.")
    else:
        print(
            table.to_string(
                index=False,
                formatters={
                    "decrease_pct": lambda x: f"{x:.2f}%"
                },
            )
        )


# ============================================================
# 6. BEFORE / AFTER 66 MONTHS
# ============================================================

print_title("6. VIOLATIONS BEFORE VS AFTER 66 MONTHS")

for target in TARGETS:

    delta_col = f"delta_{target}"

    temp = transitions.copy()
    temp["decrease"] = temp[delta_col] < 0

    before_66 = temp[temp[AGE_COL] <= 66]
    after_66 = temp[temp[AGE_COL] > 66]

    print(f"\n--- {target} ---")

    print(
        f"<= 66 months : "
        f"{int(before_66['decrease'].sum()):,} decreases "
        f"/ {len(before_66):,} transitions"
    )

    if len(before_66) > 0:
        print(
            f"               "
            f"{before_66['decrease'].mean() * 100:.2f}%"
        )

    print(
        f"> 66 months  : "
        f"{int(after_66['decrease'].sum()):,} decreases "
        f"/ {len(after_66):,} transitions"
    )

    if len(after_66) > 0:
        print(
            f"               "
            f"{after_66['decrease'].mean() * 100:.2f}%"
        )


# ============================================================
# 7. LARGEST DECREASES
# ============================================================

print_title("7. LARGEST DECREASE TRANSITIONS")

for target in TARGETS:

    delta_col = f"delta_{target}"

    cols = (
        GROUP_COLS
        + [
            "previous_age_months",
            AGE_COL,
            f"previous_{target}",
            target,
            delta_col,
        ]
    )

    worst = (
        transitions[transitions[delta_col] < 0]
        .sort_values(delta_col)
        .head(10)[cols]
    )

    print(f"\n--- {target} ---")

    if worst.empty:
        print("No decreases.")
    else:
        print(worst.to_string(index=False))


# ============================================================
# 8. VOLUME + DISEASE / CONDITION
# ============================================================

print_title("8. VOLUME DECREASE + DISEASE / CONDITION")

volume_decrease = transitions[
    transitions["delta_volume_m3_tree"] < 0
].copy()

print(
    f"Volume decrease transitions : "
    f"{len(volume_decrease):,}"
)

if not volume_decrease.empty:

    volume_decrease["delta_disease_score"] = (
        volume_decrease["disease_score"]
        - volume_decrease["previous_disease_score"]
    )

    volume_decrease["delta_condition_score"] = (
        volume_decrease["condition_score"]
        - volume_decrease["previous_condition_score"]
    )

    print("\nMean values for volume-decrease transitions:")

    summary = volume_decrease[
        [
            "disease_score",
            "previous_disease_score",
            "delta_disease_score",
            "condition_score",
            "previous_condition_score",
            "delta_condition_score",
            "delta_volume_m3_tree",
        ]
    ].mean()

    print(summary.to_string())

    print("\nVolume decreases where disease_score increased:")

    disease_increased = (
        volume_decrease["delta_disease_score"] > 0
    )

    print(
        f"Count : {int(disease_increased.sum()):,}"
    )

    if len(volume_decrease) > 0:
        print(
            f"Percent : "
            f"{disease_increased.mean() * 100:.2f}%"
        )

    print("\nVolume decreases where condition_score decreased:")

    condition_decreased = (
        volume_decrease["delta_condition_score"] < 0
    )

    print(
        f"Count : {int(condition_decreased.sum()):,}"
    )

    if len(volume_decrease) > 0:
        print(
            f"Percent : "
            f"{condition_decreased.mean() * 100:.2f}%"
        )


# ============================================================
# 9. SAME GROUP AVAILABILITY
# ============================================================

print_title("9. GROUP CONTINUITY")

print(
    "\nThis section checks whether the same growth group exists "
    "across ages.\n"
)

age_group_counts = (
    group_age.groupby(AGE_COL)
              .size()
              .reset_index(name="growth_groups")
              .sort_values(AGE_COL)
)

print(age_group_counts.to_string(index=False))


# ============================================================
# 10. IMPORTANT DATA LIMITATION
# ============================================================

print_title("10. INTERPRETATION LIMITATION")

print(
    """
IMPORTANT:

The dataset does not contain an individual tree ID.

Therefore, a DBH/height decrease detected here means:

    the GROUP MEAN at a later age is lower than the
    GROUP MEAN at the previous available age.

It does NOT prove that the exact same individual tree
physically became shorter or smaller.

Possible explanations that require further investigation include:

1. Different tree composition in the available records.
2. Missing observations at certain ages.
3. Inventory / measurement variation.
4. Environmental or health-related growth effects.
5. Dataset construction or aggregation logic.

Therefore this script DOES NOT modify or correct the dataset.
"""
)


# ============================================================
# FINAL STATUS
# ============================================================

print_title("STEP 30 STATUS")

print("Growth violation profiling completed.")
print("Dataset was NOT modified.")
print("No monotonic correction was applied.")
print("No PHI transformation was applied.")
print("No model was retrained.")