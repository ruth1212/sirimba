from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "dataset_corrected.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "height_decline_inspection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GROUP_COLS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]

AGE_COL = "age_months"
HEIGHT_COL = "height_m"


# ============================================================
# HELPER
# ============================================================

def title(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ============================================================
# LOAD
# ============================================================

title("STEP 36H : INSPECT ORIGINAL HEIGHT DECLINES")

print("[1] Loading dataset...")

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan:\n{INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# VALIDATION
# ============================================================

required = GROUP_COLS + [
    AGE_COL,
    HEIGHT_COL,
]

missing = [
    col
    for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print("Required columns: PASS")


# ============================================================
# CREATE GROUP-AGE MEAN
# ============================================================

title("2. CREATING GROUP-AGE TRAJECTORY")

group_age = (
    df.groupby(
        GROUP_COLS + [AGE_COL],
        as_index=False
    )
    .agg(
        actual_height_m=(
            HEIGHT_COL,
            "mean"
        ),
        rows_at_age=(
            HEIGHT_COL,
            "size"
        ),
    )
)

print(
    f"Group-age rows : {len(group_age):,}"
)

print(
    f"Growth groups  : "
    f"{group_age[GROUP_COLS].drop_duplicates().shape[0]:,}"
)

print(
    f"Unique ages    : "
    f"{group_age[AGE_COL].nunique():,}"
)

rows_check = (
    group_age["rows_at_age"]
    == 30
).all()

print(
    "30 rows per group-age: "
    + ("PASS" if rows_check else "FAIL")
)


# ============================================================
# SORT TRAJECTORIES
# ============================================================

group_age = group_age.sort_values(
    GROUP_COLS + [AGE_COL]
).reset_index(drop=True)

group_age["group_key"] = (
    group_age[GROUP_COLS]
    .astype(str)
    .agg("|".join, axis=1)
)


# ============================================================
# CALCULATE ORIGINAL CHANGES
# ============================================================

group_age["previous_age_months"] = (
    group_age.groupby("group_key")[AGE_COL]
    .shift(1)
)

group_age["previous_height_m"] = (
    group_age.groupby("group_key")[
        "actual_height_m"
    ]
    .shift(1)
)

group_age["height_change_m"] = (
    group_age["actual_height_m"]
    - group_age["previous_height_m"]
)

group_age["height_change_cm"] = (
    group_age["height_change_m"] * 100
)


valid = group_age[
    group_age["previous_height_m"].notna()
].copy()

print(
    f"Valid transitions : {len(valid):,}"
)


# ============================================================
# 1. OVERALL DECLINE STATISTICS
# ============================================================

title("3. OVERALL DECLINE STATISTICS")

declines = valid[
    valid["height_change_m"] < 0
].copy()

non_declines = valid[
    valid["height_change_m"] >= 0
].copy()

print(
    f"Total transitions : {len(valid):,}"
)

print(
    f"Declines          : {len(declines):,}"
)

print(
    f"Non-declines      : {len(non_declines):,}"
)

print(
    f"Decline percentage: "
    f"{len(declines) / len(valid) * 100:.2f}%"
)

if len(declines) > 0:

    print(
        f"\nMean decline     : "
        f"{declines['height_change_m'].mean():.6f} m"
    )

    print(
        f"Median decline   : "
        f"{declines['height_change_m'].median():.6f} m"
    )

    print(
        f"Q75 decline      : "
        f"{declines['height_change_m'].quantile(.75):.6f} m"
    )

    print(
        f"Q90 decline      : "
        f"{declines['height_change_m'].quantile(.90):.6f} m"
    )

    print(
        f"Q95 decline      : "
        f"{declines['height_change_m'].quantile(.95):.6f} m"
    )

    print(
        f"Maximum decline  : "
        f"{declines['height_change_m'].min():.6f} m"
    )


# ============================================================
# 2. DECLINE MAGNITUDE BINS
# ============================================================

title("4. DECLINE MAGNITUDE DISTRIBUTION")

declines["decline_magnitude_m"] = (
    -declines["height_change_m"]
)

bins = [
    -np.inf,
    0.5,
    1.0,
    2.0,
    np.inf,
]

labels = [
    "0 to <0.5 m",
    "0.5 to <1 m",
    "1 to <2 m",
    ">=2 m",
]

declines["decline_bin"] = pd.cut(
    declines["decline_magnitude_m"],
    bins=bins,
    labels=labels,
    right=False,
)

distribution = (
    declines["decline_bin"]
    .value_counts(
        sort=False
    )
    .rename("count")
    .reset_index()
)

distribution["percentage"] = (
    distribution["count"]
    / len(declines)
    * 100
)

print(
    distribution.to_string(
        index=False
    )
)

distribution.to_csv(
    OUTPUT_DIR
    / "decline_magnitude_distribution.csv",
    index=False
)


# ============================================================
# 3. DECLINE BY AGE
# ============================================================

title("5. DECLINE BY AGE")

decline_by_age = (
    valid.groupby(AGE_COL)
    .agg(
        transitions=(
            "height_change_m",
            "size"
        ),
        declines=(
            "height_change_m",
            lambda x: (x < 0).sum()
        ),
        mean_change_m=(
            "height_change_m",
            "mean"
        ),
        mean_decline_m=(
            "height_change_m",
            lambda x: (
                -x[x < 0].mean()
                if (x < 0).any()
                else 0
            )
        ),
        max_decline_m=(
            "height_change_m",
            lambda x: (
                -x.min()
                if x.min() < 0
                else 0
            )
        ),
    )
    .reset_index()
)

decline_by_age["decline_pct"] = (
    decline_by_age["declines"]
    / decline_by_age["transitions"]
    * 100
)

print(
    decline_by_age.to_string(
        index=False
    )
)

decline_by_age.to_csv(
    OUTPUT_DIR
    / "decline_by_age.csv",
    index=False
)


# ============================================================
# 4. AGE BAND ANALYSIS
# ============================================================

title("6. DECLINE BY AGE BAND")

def age_band(age):

    if 6 <= age <= 30:
        return "EARLY_P6_P30"

    if 36 <= age <= 60:
        return "MIDDLE_P36_P60"

    if 66 <= age <= 120:
        return "LATE_P66_P120"

    if 126 <= age <= 180:
        return "LATE_P126_P180"

    if 186 <= age <= 240:
        return "LATE_P186_P240"

    if 246 <= age <= 300:
        return "LATE_P246_P300"

    return "OTHER"


valid["age_band"] = (
    valid[AGE_COL]
    .apply(age_band)
)

band_summary = (
    valid[
        valid["age_band"] != "OTHER"
    ]
    .groupby("age_band")
    .agg(
        transitions=(
            "height_change_m",
            "size"
        ),
        declines=(
            "height_change_m",
            lambda x: (x < 0).sum()
        ),
        mean_change_m=(
            "height_change_m",
            "mean"
        ),
        mean_decline_m=(
            "height_change_m",
            lambda x: (
                -x[x < 0].mean()
                if (x < 0).any()
                else 0
            )
        ),
        max_decline_m=(
            "height_change_m",
            lambda x: (
                -x.min()
                if x.min() < 0
                else 0
            )
        ),
    )
    .reset_index()
)

band_summary["decline_pct"] = (
    band_summary["declines"]
    / band_summary["transitions"]
    * 100
)

print(
    band_summary.to_string(
        index=False
    )
)

band_summary.to_csv(
    OUTPUT_DIR
    / "decline_by_age_band.csv",
    index=False
)


# ============================================================
# 5. DECLINE FOLLOWED BY RECOVERY
# ============================================================

title("7. DECLINE FOLLOWED BY RECOVERY")

valid["next_height_m"] = (
    valid.groupby("group_key")[
        "actual_height_m"
    ]
    .shift(-1)
)

valid["next_change_m"] = (
    valid["next_height_m"]
    - valid["actual_height_m"]
)

decline_recovery = valid[
    (valid["height_change_m"] < 0)
    & (valid["next_change_m"] > 0)
].copy()

print(
    f"Decline transitions : "
    f"{len(declines):,}"
)

print(
    f"Decline followed by recovery: "
    f"{len(decline_recovery):,}"
)

if len(declines) > 0:

    print(
        f"Recovery percentage: "
        f"{len(decline_recovery) / len(declines) * 100:.2f}%"
    )


# ============================================================
# 6. CONSECUTIVE DECLINES
# ============================================================

title("8. CONSECUTIVE DECLINE PATTERNS")

valid["is_decline"] = (
    valid["height_change_m"] < 0
)

consecutive_records = []

for group_key, g in valid.groupby(
    "group_key"
):

    g = g.sort_values(
        AGE_COL
    ).reset_index(drop=True)

    current = 0
    start_age = None

    for i in range(len(g)):

        if bool(g.loc[i, "is_decline"]):

            if current == 0:
                start_age = (
                    g.loc[i, AGE_COL]
                )

            current += 1

        else:

            if current > 0:

                consecutive_records.append(
                    {
                        "group_key": group_key,
                        "start_age": start_age,
                        "end_age": g.loc[i - 1, AGE_COL],
                        "consecutive_declines": current,
                        "duration_months": current * 6,
                    }
                )

            current = 0
            start_age = None

    if current > 0:

        consecutive_records.append(
            {
                "group_key": group_key,
                "start_age": start_age,
                "end_age": g.loc[len(g) - 1, AGE_COL],
                "consecutive_declines": current,
                "duration_months": current * 6,
            }
        )

consecutive = pd.DataFrame(
    consecutive_records
)

if len(consecutive) > 0:

    print(
        f"Decline sequences : "
        f"{len(consecutive):,}"
    )

    print(
        f"Maximum consecutive declines : "
        f"{consecutive['consecutive_declines'].max():,}"
    )

    print(
        f"Maximum duration : "
        f"{consecutive['duration_months'].max():,} months"
    )

    print(
        "\nDistribution:"
    )

    print(
        consecutive[
            "consecutive_declines"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    consecutive.to_csv(
        OUTPUT_DIR
        / "consecutive_decline_sequences.csv",
        index=False
    )


# ============================================================
# 7. TOP 30 LARGEST DECLINES
# ============================================================

title("9. TOP 30 LARGEST DECLINES")

top_declines = (
    declines[
        GROUP_COLS
        + [
            "previous_age_months",
            AGE_COL,
            "previous_height_m",
            "actual_height_m",
            "height_change_m",
            "decline_magnitude_m",
        ]
    ]
    .sort_values(
        "decline_magnitude_m",
        ascending=False
    )
    .head(30)
)

print(
    top_declines.to_string(
        index=False
    )
)

top_declines.to_csv(
    OUTPUT_DIR
    / "top_30_largest_declines.csv",
    index=False
)


# ============================================================
# 8. TOP DECLINES BY AGE BAND
# ============================================================

title("10. TOP DECLINES BY AGE BAND")

top_by_band = []

for band, g in declines.groupby(
    declines["age_months"].apply(age_band)
):

    if band == "OTHER":
        continue

    row = (
        g.sort_values(
            "decline_magnitude_m",
            ascending=False
        )
        .head(1)
    )

    if len(row) > 0:

        top_by_band.append(
            row.iloc[0]
        )

if top_by_band:

    top_band_df = pd.DataFrame(
        top_by_band
    )

    print(
        top_band_df[
            GROUP_COLS
            + [
                "previous_age_months",
                AGE_COL,
                "previous_height_m",
                "actual_height_m",
                "decline_magnitude_m",
            ]
        ].to_string(
            index=False
        )
    )

    top_band_df.to_csv(
        OUTPUT_DIR
        / "top_decline_by_age_band.csv",
        index=False
    )


# ============================================================
# 9. SAVE ALL TRANSITIONS
# ============================================================

valid[
    GROUP_COLS
    + [
        "previous_age_months",
        AGE_COL,
        "previous_height_m",
        "actual_height_m",
        "height_change_m",
        "height_change_cm",
        "age_band",
    ]
].to_csv(
    OUTPUT_DIR
    / "all_height_transitions.csv",
    index=False
)


# ============================================================
# FINISH
# ============================================================

title("STEP 36H FINISHED")

print(
    f"Output directory:\n{OUTPUT_DIR}"
)

print(
    "\nOriginal dataset modified: NO"
)