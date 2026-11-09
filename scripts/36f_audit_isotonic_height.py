from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "height_isotonic_test"
    / "height_isotonic_group_age.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "height_isotonic_audit"
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
ACTUAL_COL = "actual_height_m"
CORRECTED_COL = "corrected_height_m"
CORRECTION_COL = "absolute_correction_m"
CHANGE_COL = "corrected_change_m"


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

title("STEP 36F : AUDIT ISOTONIC HEIGHT CORRECTION")

print("[1] Loading isotonic prototype...")

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input tidak ditemukan:\n{INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

required = (
    GROUP_COLS
    + [
        AGE_COL,
        ACTUAL_COL,
        CORRECTED_COL,
        CORRECTION_COL,
        CHANGE_COL,
    ]
)

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print("Required columns: PASS")


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    GROUP_COLS + [AGE_COL]
).reset_index(drop=True)


# ============================================================
# GROUP KEY
# ============================================================

df["group_key"] = (
    df[GROUP_COLS]
    .astype(str)
    .agg("|".join, axis=1)
)


# ============================================================
# 1. BASIC TRAJECTORY CHECK
# ============================================================

title("2. BASIC TRAJECTORY CHECK")

group_count = df["group_key"].nunique()
age_count = df[AGE_COL].nunique()

print(f"Growth groups : {group_count}")
print(f"Unique ages   : {age_count}")

valid = df.groupby("group_key").size()

print(
    f"Groups with 50 ages : "
    f"{(valid == age_count).sum():,}"
)

# corrected trajectory must never decrease
df["corrected_change_check"] = (
    df.groupby("group_key")[CORRECTED_COL]
    .diff()
)

decreases = (
    df["corrected_change_check"] < -1e-10
).sum()

print(f"Corrected decreases : {decreases:,}")

if decreases == 0:
    print("NO-DECREASE CHECK: PASS")
else:
    print("NO-DECREASE CHECK: FAIL")


# ============================================================
# 2. PLATEAU LENGTH ANALYSIS
# ============================================================

title("3. PLATEAU LENGTH ANALYSIS")

df["stagnant"] = (
    df["corrected_change_check"].abs() <= 1e-10
)

plateau_records = []

for group_key, g in df.groupby("group_key"):

    g = g.sort_values(AGE_COL).reset_index(drop=True)

    current_length = 0
    start_age = None

    for i in range(1, len(g)):

        stagnant = bool(
            abs(
                g.loc[i, CORRECTED_COL]
                - g.loc[i - 1, CORRECTED_COL]
            ) <= 1e-10
        )

        if stagnant:

            if current_length == 0:
                start_age = g.loc[i - 1, AGE_COL]

            current_length += 1

        else:

            if current_length > 0:

                end_age = g.loc[i - 1, AGE_COL]

                plateau_records.append(
                    {
                        "group_key": group_key,
                        "start_age_months": start_age,
                        "end_age_months": end_age,
                        "stagnant_transitions": current_length,
                        "duration_months": current_length * 6,
                    }
                )

                current_length = 0
                start_age = None

    if current_length > 0:

        end_age = g.loc[len(g) - 1, AGE_COL]

        plateau_records.append(
            {
                "group_key": group_key,
                "start_age_months": start_age,
                "end_age_months": end_age,
                "stagnant_transitions": current_length,
                "duration_months": current_length * 6,
            }
        )


plateaus = pd.DataFrame(plateau_records)

if len(plateaus) == 0:

    print("No plateaus detected.")

else:

    print(
        f"Total plateau segments : "
        f"{len(plateaus):,}"
    )

    print(
        f"Maximum stagnant transitions : "
        f"{plateaus['stagnant_transitions'].max():,}"
    )

    print(
        f"Maximum plateau duration : "
        f"{plateaus['duration_months'].max():,} months"
    )

    print("\nPlateau distribution:")

    print(
        plateaus[
            "stagnant_transitions"
        ]
        .describe()
        .to_string()
    )

    plateau_thresholds = [
        1,
        2,
        3,
        5,
        10,
        15,
        20,
    ]

    print("\nPlateau counts:")

    for threshold in plateau_thresholds:

        count = (
            plateaus["stagnant_transitions"]
            >= threshold
        ).sum()

        print(
            f">= {threshold:2d} transitions : "
            f"{count:,}"
        )

    plateaus.to_csv(
        OUTPUT_DIR
        / "plateau_segments.csv",
        index=False
    )


# ============================================================
# 3. LARGEST CORRECTIONS
# ============================================================

title("4. LARGEST CORRECTIONS")

top_corrections = (
    df[
        GROUP_COLS
        + [
            AGE_COL,
            ACTUAL_COL,
            CORRECTED_COL,
            CORRECTION_COL,
        ]
    ]
    .sort_values(
        CORRECTION_COL,
        ascending=False
    )
    .head(30)
)

print(
    top_corrections.to_string(
        index=False
    )
)

top_corrections.to_csv(
    OUTPUT_DIR
    / "top_30_corrections.csv",
    index=False
)


# ============================================================
# 4. CORRECTION BY AGE
# ============================================================

title("5. CORRECTION BY AGE")

age_correction = (
    df.groupby(AGE_COL)
    .agg(
        rows=(ACTUAL_COL, "size"),
        mean_correction=(
            CORRECTION_COL,
            "mean"
        ),
        median_correction=(
            CORRECTION_COL,
            "median"
        ),
        q95_correction=(
            CORRECTION_COL,
            lambda x: x.quantile(0.95)
        ),
        max_correction=(
            CORRECTION_COL,
            "max"
        ),
        changed_rows=(
            CORRECTION_COL,
            lambda x: (x > 1e-10).sum()
        ),
    )
    .reset_index()
)

age_correction["changed_pct"] = (
    age_correction["changed_rows"]
    / age_correction["rows"]
    * 100
)

print(
    age_correction.to_string(
        index=False
    )
)

age_correction.to_csv(
    OUTPUT_DIR
    / "correction_by_age.csv",
    index=False
)


# ============================================================
# 5. EARLY / MIDDLE / LATE BY GROUP
# ============================================================

title("6. EARLY / MIDDLE / LATE BY GROUP")

def stage_rate(g, start_age, end_age):

    x = g[
        (g[AGE_COL] >= start_age)
        & (g[AGE_COL] <= end_age)
    ].copy()

    if len(x) < 2:
        return np.nan

    x = x.sort_values(AGE_COL)

    delta_height = (
        x[CORRECTED_COL].iloc[-1]
        - x[CORRECTED_COL].iloc[0]
    )

    delta_years = (
        x[AGE_COL].iloc[-1]
        - x[AGE_COL].iloc[0]
    ) / 12

    if delta_years <= 0:
        return np.nan

    return delta_height / delta_years


stage_records = []

for group_key, g in df.groupby("group_key"):

    early = stage_rate(g, 6, 30)
    middle = stage_rate(g, 36, 60)
    late = stage_rate(g, 66, 300)

    stage_records.append(
        {
            "group_key": group_key,
            "early_cm_year": early,
            "middle_cm_year": middle,
            "late_cm_year": late,
        }
    )

stage_df = pd.DataFrame(stage_records)

stage_df["early_gt_middle"] = (
    stage_df["early_cm_year"]
    > stage_df["middle_cm_year"]
)

stage_df["middle_gt_late"] = (
    stage_df["middle_cm_year"]
    > stage_df["late_cm_year"]
)

stage_df["three_stage_slowdown"] = (
    stage_df["early_gt_middle"]
    & stage_df["middle_gt_late"]
)

print(
    f"Groups : {len(stage_df):,}"
)

print(
    f"Early > Middle : "
    f"{stage_df['early_gt_middle'].sum():,}"
)

print(
    f"Middle > Late : "
    f"{stage_df['middle_gt_late'].sum():,}"
)

print(
    f"Three-stage slowdown : "
    f"{stage_df['three_stage_slowdown'].sum():,}"
)

print(
    f"Three-stage percentage : "
    f"{stage_df['three_stage_slowdown'].mean() * 100:.2f}%"
)

stage_df.to_csv(
    OUTPUT_DIR
    / "stage_growth_by_group.csv",
    index=False
)


# ============================================================
# 6. PHI AUDIT
# ============================================================

title("7. PHI = 66 AUDIT")

phi = df[
    df[AGE_COL] == 66
].copy()

print(f"PHI rows : {len(phi):,}")

print(
    f"Actual PHI mean    : "
    f"{phi[ACTUAL_COL].mean():.6f}"
)

print(
    f"Corrected PHI mean : "
    f"{phi[CORRECTED_COL].mean():.6f}"
)

print(
    f"Mean correction    : "
    f"{phi[CORRECTION_COL].mean():.6f}"
)

print(
    f"Median correction  : "
    f"{phi[CORRECTION_COL].median():.6f}"
)

print(
    f"Q95 correction     : "
    f"{phi[CORRECTION_COL].quantile(.95):.6f}"
)

print(
    f"Maximum correction : "
    f"{phi[CORRECTION_COL].max():.6f}"
)


# ============================================================
# 7. SAVE AUDIT SUMMARY
# ============================================================

summary = {
    "rows": len(df),
    "growth_groups": group_count,
    "unique_ages": age_count,
    "corrected_decreases": int(decreases),
    "mean_correction_m": df[CORRECTION_COL].mean(),
    "median_correction_m": df[CORRECTION_COL].median(),
    "q95_correction_m": df[CORRECTION_COL].quantile(.95),
    "max_correction_m": df[CORRECTION_COL].max(),
    "max_plateau_transitions": (
        int(plateaus["stagnant_transitions"].max())
        if len(plateaus) > 0
        else 0
    ),
    "max_plateau_months": (
        int(plateaus["duration_months"].max())
        if len(plateaus) > 0
        else 0
    ),
    "three_stage_slowdown_pct": (
        stage_df["three_stage_slowdown"].mean()
        * 100
    ),
}

summary_df = pd.DataFrame(
    [summary]
)

summary_df.to_csv(
    OUTPUT_DIR
    / "isotonic_audit_summary.csv",
    index=False
)


# ============================================================
# FINISH
# ============================================================

title("STEP 36F FINISHED")

print(
    f"Audit output:\n{OUTPUT_DIR}"
)

print(
    "\nOriginal dataset modified: NO"
)