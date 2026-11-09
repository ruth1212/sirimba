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
    / "height_conservative_test"
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
# PARAMETERS
# ============================================================

# Maximum correction that can be applied to one transition.
# This is deliberately conservative.
MAX_CORRECTION_M = 1.0

# Small decline threshold.
# Declines below this value are treated as measurement-level
# noise candidates.
SMALL_DECLINE_M = 0.5


# ============================================================
# HELPERS
# ============================================================

def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


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


# ============================================================
# LOAD
# ============================================================

section("STEP 36J : CONSERVATIVE HEIGHT PROTOTYPE")

print("[1] Loading dataset...")

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan:\n{INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# VALIDATE
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
# GROUP + AGE
# ============================================================

section("2. BUILD GROUP-AGE TRAJECTORY")

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

group_age = group_age.sort_values(
    GROUP_COLS + [AGE_COL]
).reset_index(drop=True)

group_age["group_key"] = (
    group_age[GROUP_COLS]
    .astype(str)
    .agg("|".join, axis=1)
)

print(
    f"Group-age rows : {len(group_age):,}"
)

print(
    f"Growth groups  : "
    f"{group_age['group_key'].nunique():,}"
)

print(
    f"Unique ages    : "
    f"{group_age[AGE_COL].nunique():,}"
)


# ============================================================
# PREPARE TRAJECTORIES
# ============================================================

section("3. PREPARING TRAJECTORIES")

group_age["previous_height_m"] = (
    group_age.groupby("group_key")[
        "actual_height_m"
    ].shift(1)
)

group_age["next_height_m"] = (
    group_age.groupby("group_key")[
        "actual_height_m"
    ].shift(-1)
)

group_age["previous_age_months"] = (
    group_age.groupby("group_key")[
        AGE_COL
    ].shift(1)
)

group_age["next_age_months"] = (
    group_age.groupby("group_key")[
        AGE_COL
    ].shift(-1)
)

group_age["original_change_m"] = (
    group_age["actual_height_m"]
    - group_age["previous_height_m"]
)

group_age["age_band"] = (
    group_age[AGE_COL]
    .apply(age_band)
)

group_age["corrected_height_m"] = (
    group_age["actual_height_m"]
)


# ============================================================
# CONSERVATIVE CORRECTION
# ============================================================

section("4. APPLYING CONSERVATIVE LOCAL CORRECTION")

correction_records = []

for group_key, g in group_age.groupby(
    "group_key",
    sort=False
):

    g = g.sort_values(
        AGE_COL
    ).copy()

    indices = list(g.index)

    for position in range(1, len(indices)):

        idx = indices[position]
        prev_idx = indices[position - 1]

        current_actual = float(
            group_age.loc[
                idx,
                "actual_height_m"
            ]
        )

        previous_actual = float(
            group_age.loc[
                prev_idx,
                "actual_height_m"
            ]
        )

        if current_actual >= previous_actual:
            continue

        decline = (
            previous_actual
            - current_actual
        )

        # ----------------------------------------------------
        # SMALL DECLINE
        # ----------------------------------------------------

        if decline < SMALL_DECLINE_M:

            target = previous_actual

            correction = (
                target
                - current_actual
            )

            correction = min(
                correction,
                MAX_CORRECTION_M
            )

            corrected = (
                current_actual
                + correction
            )

        # ----------------------------------------------------
        # LARGER DECLINE
        # ----------------------------------------------------

        else:

            next_idx = (
                indices[position + 1]
                if position + 1 < len(indices)
                else None
            )

            if next_idx is not None:

                next_actual = float(
                    group_age.loc[
                        next_idx,
                        "actual_height_m"
                    ]
                )

                # Use the local neighborhood rather than
                # the historical maximum.

                local_reference = (
                    previous_actual
                    + next_actual
                ) / 2.0

                # Never lower the current value.
                target = max(
                    current_actual,
                    local_reference
                )

            else:

                target = previous_actual

            correction = (
                target
                - current_actual
            )

            correction = max(
                0.0,
                min(
                    correction,
                    MAX_CORRECTION_M
                )
            )

            corrected = (
                current_actual
                + correction
            )

        # Safety: corrected height must not decrease
        # relative to the ORIGINAL previous value.

        corrected = max(
            corrected,
            previous_actual
        )

        group_age.loc[
            idx,
            "corrected_height_m"
        ] = corrected

        correction_records.append(
            {
                "group_key": group_key,
                "age_months": group_age.loc[
                    idx,
                    AGE_COL
                ],
                "previous_age_months": group_age.loc[
                    idx,
                    "previous_age_months"
                ],
                "previous_actual_m": previous_actual,
                "actual_height_m": current_actual,
                "corrected_height_m": corrected,
                "correction_m": (
                    corrected
                    - current_actual
                ),
                "original_decline_m": decline,
                "age_band": group_age.loc[
                    idx,
                    "age_band"
                ],
            }
        )


corrections = pd.DataFrame(
    correction_records
)

print(
    f"Declines corrected : "
    f"{len(corrections):,}"
)


# ============================================================
# VALIDATE TRANSITIONS
# ============================================================

section("5. VALIDATING CORRECTED TRAJECTORIES")

group_age["corrected_previous_m"] = (
    group_age.groupby("group_key")[
        "corrected_height_m"
    ].shift(1)
)

group_age["corrected_change_m"] = (
    group_age["corrected_height_m"]
    - group_age["corrected_previous_m"]
)

valid = group_age[
    group_age["corrected_previous_m"].notna()
].copy()

corrected_decreases = valid[
    valid["corrected_change_m"] < -1e-9
]

print(
    f"Valid transitions : {len(valid):,}"
)

print(
    f"Corrected decreases : "
    f"{len(corrected_decreases):,}"
)

print(
    "NO-DECREASE: "
    + (
        "PASS"
        if len(corrected_decreases) == 0
        else "FAIL"
    )
)


# ============================================================
# CORRECTION MAGNITUDE
# ============================================================

section("6. CORRECTION MAGNITUDE")

correction_values = (
    group_age["corrected_height_m"]
    - group_age["actual_height_m"]
)

print(
    f"Mean correction   : "
    f"{correction_values.mean():.6f} m"
)

print(
    f"Median correction : "
    f"{correction_values.median():.6f} m"
)

print(
    f"Q75               : "
    f"{correction_values.quantile(.75):.6f} m"
)

print(
    f"Q90               : "
    f"{correction_values.quantile(.90):.6f} m"
)

print(
    f"Q95               : "
    f"{correction_values.quantile(.95):.6f} m"
)

print(
    f"Q99               : "
    f"{correction_values.quantile(.99):.6f} m"
)

print(
    f"Maximum           : "
    f"{correction_values.max():.6f} m"
)


# ============================================================
# CORRECTION BINS
# ============================================================

section("7. CORRECTION DISTRIBUTION")

bins = [
    -np.inf,
    0.0,
    0.5,
    1.0,
    2.0,
    np.inf,
]

labels = [
    "0 m",
    ">0 to <0.5 m",
    "0.5 to <1 m",
    "1 to <2 m",
    ">=2 m",
]

correction_bins = pd.cut(
    correction_values,
    bins=bins,
    labels=labels,
    right=False
)

distribution = (
    correction_bins
    .value_counts(
        sort=False
    )
    .rename("count")
    .reset_index()
)

distribution["percentage"] = (
    distribution["count"]
    / len(group_age)
    * 100
)

print(
    distribution.to_string(
        index=False
    )
)


# ============================================================
# PHI
# ============================================================

section("8. PHI = 66 MONTHS")

phi = group_age[
    group_age[AGE_COL] == 66
].copy()

phi_correction = (
    phi["corrected_height_m"]
    - phi["actual_height_m"]
)

print(
    f"PHI rows : {len(phi):,}"
)

print(
    f"Actual mean    : "
    f"{phi['actual_height_m'].mean():.6f} m"
)

print(
    f"Corrected mean : "
    f"{phi['corrected_height_m'].mean():.6f} m"
)

print(
    f"Mean correction: "
    f"{phi_correction.mean():.6f} m"
)

print(
    f"Max correction : "
    f"{phi_correction.max():.6f} m"
)


# ============================================================
# STAGE GROWTH RATE
# ============================================================

section("9. STAGE GROWTH SHAPE")

stage_data = valid.copy()

stage_data["growth_rate_cm_per_year"] = (
    stage_data["corrected_change_m"]
    / (
        (
            stage_data[AGE_COL]
            - stage_data["corrected_previous_m"].notna()
            * 0
        )
        .replace(0, np.nan)
    )
)

# For this audit, calculate directly from the
# 6-month interval.
stage_data["growth_rate_cm_per_year"] = (
    stage_data["corrected_change_m"]
    * 100
    / 0.5
)

def stage(age):

    if 12 <= age <= 30:
        return "EARLY"

    if 36 <= age <= 60:
        return "MIDDLE"

    if 66 <= age <= 300:
        return "LATE"

    return "OTHER"


stage_data["stage"] = (
    stage_data[AGE_COL]
    .apply(stage)
)

stage_summary = (
    stage_data[
        stage_data["stage"] != "OTHER"
    ]
    .groupby("stage")
    .agg(
        count=(
            "growth_rate_cm_per_year",
            "size"
        ),
        mean_growth_cm_year=(
            "growth_rate_cm_per_year",
            "mean"
        ),
        median_growth_cm_year=(
            "growth_rate_cm_per_year",
            "median"
        ),
        min_growth_cm_year=(
            "growth_rate_cm_per_year",
            "min"
        ),
        max_growth_cm_year=(
            "growth_rate_cm_per_year",
            "max"
        ),
    )
    .reset_index()
)

print(
    stage_summary.to_string(
        index=False
    )
)


# ============================================================
# LONG PLATEAUS
# ============================================================

section("10. PLATEAU AUDIT")

valid["is_stagnant"] = (
    valid["corrected_change_m"]
    .abs()
    <= 1e-9
)

plateau_lengths = []

for group_key, g in valid.groupby(
    "group_key"
):

    g = g.sort_values(
        AGE_COL
    )

    current = 0

    for stagnant in g[
        "is_stagnant"
    ]:

        if stagnant:

            current += 1

        else:

            if current > 0:
                plateau_lengths.append(
                    current
                )

            current = 0

    if current > 0:
        plateau_lengths.append(
            current
        )

if plateau_lengths:

    plateau_lengths = np.array(
        plateau_lengths
    )

    print(
        f"Plateau segments : "
        f"{len(plateau_lengths):,}"
    )

    print(
        f"Mean plateau transitions : "
        f"{plateau_lengths.mean():.2f}"
    )

    print(
        f"Median plateau transitions : "
        f"{np.median(plateau_lengths):.2f}"
    )

    print(
        f"Maximum plateau transitions : "
        f"{plateau_lengths.max():,}"
    )

    print(
        f"Maximum plateau duration : "
        f"{plateau_lengths.max() * 6:,} months"
    )


# ============================================================
# EXTREME CORRECTIONS
# ============================================================

section("11. TOP 30 CORRECTIONS")

top = (
    group_age[
        GROUP_COLS
        + [
            AGE_COL,
            "actual_height_m",
            "corrected_height_m",
            "corrected_previous_m",
        ]
    ]
    .assign(
        correction_m=lambda x:
            x["corrected_height_m"]
            - x["actual_height_m"]
    )
    .sort_values(
        "correction_m",
        ascending=False
    )
    .head(30)
)

print(
    top.to_string(
        index=False
    )
)


# ============================================================
# SAVE PROTOTYPE
# ============================================================

section("12. SAVING PROTOTYPE")

prototype = (
    df.merge(
        group_age[
            GROUP_COLS
            + [
                AGE_COL,
                "corrected_height_m",
            ]
        ],
        on=GROUP_COLS + [AGE_COL],
        how="left"
    )
)

prototype["height_correction_m"] = (
    prototype["corrected_height_m"]
    - prototype[HEIGHT_COL]
)

prototype.to_csv(
    OUTPUT_DIR
    / "height_conservative_test.csv",
    index=False
)

group_age.to_csv(
    OUTPUT_DIR
    / "height_conservative_group_age.csv",
    index=False
)

corrections.to_csv(
    OUTPUT_DIR
    / "height_correction_records.csv",
    index=False
)

distribution.to_csv(
    OUTPUT_DIR
    / "correction_distribution.csv",
    index=False
)

stage_summary.to_csv(
    OUTPUT_DIR
    / "stage_growth_summary.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

section("STEP 36J FINISHED")

print(
    f"Output directory:\n{OUTPUT_DIR}"
)

print(
    "\nIMPORTANT:"
)

print(
    "dataset_corrected.csv was NOT modified."
)

print(
    "This is a PROTOTYPE ONLY."
)