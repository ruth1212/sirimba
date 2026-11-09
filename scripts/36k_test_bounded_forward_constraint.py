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
    / "height_bounded_forward_test"
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
# AGE-DEPENDENT MAXIMUM CORRECTION
# ============================================================
#
# The correction becomes more conservative as trees get older.
#
# Early growth:
#   do not aggressively alter the trajectory.
#
# Middle growth:
#   allow moderate correction.
#
# Late growth:
#   correction is increasingly limited because growth
#   should approach stagnation.
#
# These are PROTOTYPE parameters only.
# ============================================================

def maximum_allowed_correction(age_months):

    if age_months <= 30:
        return 0.25

    if age_months <= 60:
        return 0.50

    if age_months <= 120:
        return 0.75

    if age_months <= 180:
        return 0.75

    if age_months <= 240:
        return 0.60

    return 0.50


# ============================================================
# AGE-DEPENDENT GROWTH FLOOR
# ============================================================
#
# This is NOT a biological claim.
# It is only a prototype constraint.
#
# The objective is:
#
#   corrected current >= corrected previous
#
# while allowing stagnation.
#
# Therefore the minimum allowed increment is zero.
# ============================================================

def minimum_growth(age_months):

    return 0.0


# ============================================================
# STAGE
# ============================================================

def age_stage(age):

    if 6 <= age <= 30:
        return "EARLY"

    if 36 <= age <= 60:
        return "MIDDLE"

    if 66 <= age <= 300:
        return "LATE"

    return "OTHER"


# ============================================================
# SECTION HELPER
# ============================================================

def section(title):

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ============================================================
# LOAD
# ============================================================

section("STEP 36K : BOUNDED FORWARD CONSTRAINT")

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
    column
    for column in required
    if column not in df.columns
]

if missing:

    raise ValueError(
        f"Missing required columns: {missing}"
    )

print("Required columns: PASS")


# ============================================================
# GROUP + AGE AGGREGATION
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

print(
    "30 rows per group-age: "
    + (
        "PASS"
        if group_age["rows_at_age"].eq(30).all()
        else "FAIL"
    )
)


# ============================================================
# INITIALIZE CORRECTED HEIGHT
# ============================================================

group_age["corrected_height_m"] = (
    group_age["actual_height_m"]
)


# ============================================================
# BOUNDED FORWARD CORRECTION
# ============================================================

section("3. APPLYING BOUNDED FORWARD CONSTRAINT")

correction_records = []


for group_key, trajectory in group_age.groupby(
    "group_key",
    sort=False
):

    trajectory = trajectory.sort_values(
        AGE_COL
    )

    indices = list(
        trajectory.index
    )

    for position in range(
        1,
        len(indices)
    ):

        current_idx = indices[position]
        previous_idx = indices[position - 1]

        age = float(
            group_age.loc[
                current_idx,
                AGE_COL
            ]
        )

        actual_current = float(
            group_age.loc[
                current_idx,
                "actual_height_m"
            ]
        )

        actual_previous = float(
            group_age.loc[
                previous_idx,
                "actual_height_m"
            ]
        )

        corrected_previous = float(
            group_age.loc[
                previous_idx,
                "corrected_height_m"
            ]
        )

        # ----------------------------------------------------
        # NO DECLINE
        # ----------------------------------------------------

        if actual_current >= corrected_previous:

            group_age.loc[
                current_idx,
                "corrected_height_m"
            ] = actual_current

            continue

        # ----------------------------------------------------
        # DECLINE DETECTED
        # ----------------------------------------------------

        decline_from_corrected_previous = (
            corrected_previous
            - actual_current
        )

        max_correction = (
            maximum_allowed_correction(
                age
            )
        )

        # ----------------------------------------------------
        # LOCAL FORWARD INFORMATION
        # ----------------------------------------------------

        if position + 1 < len(indices):

            next_idx = indices[
                position + 1
            ]

            actual_next = float(
                group_age.loc[
                    next_idx,
                    "actual_height_m"
                ]
            )

            # Expected local direction based on
            # current-to-next observation.
            forward_change = (
                actual_next
                - actual_current
            )

        else:

            actual_next = np.nan
            forward_change = 0.0

        # ----------------------------------------------------
        # BASE TARGET
        # ----------------------------------------------------
        #
        # Instead of jumping all the way to the previous
        # corrected height, move only a bounded fraction
        # toward the previous corrected height.
        #
        # The fraction is intentionally conservative.
        # ----------------------------------------------------

        if decline_from_corrected_previous <= 0:

            target = actual_current

        else:

            if age <= 60:

                fraction = 0.50

            elif age <= 120:

                fraction = 0.35

            elif age <= 180:

                fraction = 0.25

            else:

                fraction = 0.20

            bounded_gap = (
                decline_from_corrected_previous
                * fraction
            )

            bounded_gap = min(
                bounded_gap,
                max_correction
            )

            target = (
                actual_current
                + bounded_gap
            )

        # ----------------------------------------------------
        # ENFORCE NON-DECREASING CONSTRAINT
        # ----------------------------------------------------
        #
        # This is the critical constraint:
        #
        # corrected_current >= corrected_previous
        #
        # If the bounded target is still below the
        # corrected previous value, we must raise it to
        # the previous corrected value.
        #
        # This guarantees no decrease.
        # ----------------------------------------------------

        minimum_allowed = (
            corrected_previous
            + minimum_growth(age)
        )

        corrected_current = max(
            target,
            minimum_allowed
        )

        # Never lower an actual observation.
        corrected_current = max(
            corrected_current,
            actual_current
        )

        correction = (
            corrected_current
            - actual_current
        )

        group_age.loc[
            current_idx,
            "corrected_height_m"
        ] = corrected_current

        correction_records.append(
            {
                "group_key": group_key,
                "age_months": age,
                "actual_previous_m": actual_previous,
                "corrected_previous_m": corrected_previous,
                "actual_current_m": actual_current,
                "corrected_current_m": corrected_current,
                "correction_m": correction,
                "original_decline_m": (
                    actual_previous
                    - actual_current
                ),
                "corrected_decline_m": (
                    corrected_previous
                    - corrected_current
                ),
                "max_allowed_correction_m": (
                    max_correction
                ),
                "forward_change_m": (
                    forward_change
                ),
                "stage": age_stage(age),
            }
        )


corrections = pd.DataFrame(
    correction_records
)

print(
    f"Decline records processed : "
    f"{len(corrections):,}"
)


# ============================================================
# VALIDATION
# ============================================================

section("4. VALIDATING CORRECTED TRAJECTORIES")

group_age["corrected_previous_m"] = (
    group_age.groupby(
        "group_key"
    )[
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
    f"Valid transitions : "
    f"{len(valid):,}"
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

section("5. CORRECTION MAGNITUDE")

group_age["correction_m"] = (
    group_age["corrected_height_m"]
    - group_age["actual_height_m"]
)

correction_values = (
    group_age["correction_m"]
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
# CORRECTION DISTRIBUTION
# ============================================================

section("6. CORRECTION DISTRIBUTION")

bins = [
    -1e-12,
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    np.inf,
]

labels = [
    "0 m",
    ">0 to <0.25 m",
    "0.25 to <0.5 m",
    "0.5 to <1 m",
    "1 to <2 m",
    ">=2 m",
]

positive_bins = pd.cut(
    correction_values,
    bins=bins,
    labels=labels,
    right=False,
    include_lowest=True
)

distribution = (
    positive_bins
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

section("7. PHI = 66 MONTHS")

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
    f"Median correction: "
    f"{phi_correction.median():.6f} m"
)

print(
    f"Max correction : "
    f"{phi_correction.max():.6f} m"
)


# ============================================================
# STAGE GROWTH SHAPE
# ============================================================

section("8. STAGE GROWTH SHAPE")

valid["growth_rate_cm_year"] = (
    valid["corrected_change_m"]
    * 100
    / 0.5
)

valid["stage"] = (
    valid[AGE_COL]
    .apply(age_stage)
)

stage_summary = (
    valid[
        valid["stage"] != "OTHER"
    ]
    .groupby("stage")
    .agg(
        count=(
            "growth_rate_cm_year",
            "size"
        ),
        mean_growth_cm_year=(
            "growth_rate_cm_year",
            "mean"
        ),
        median_growth_cm_year=(
            "growth_rate_cm_year",
            "median"
        ),
        min_growth_cm_year=(
            "growth_rate_cm_year",
            "min"
        ),
        max_growth_cm_year=(
            "growth_rate_cm_year",
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
# STAGE ORDER CHECK
# ============================================================

stage_means = (
    stage_summary.set_index(
        "stage"
    )[
        "mean_growth_cm_year"
    ]
)

early_mean = stage_means.get(
    "EARLY",
    np.nan
)

middle_mean = stage_means.get(
    "MIDDLE",
    np.nan
)

late_mean = stage_means.get(
    "LATE",
    np.nan
)

print(
    "\nThree-stage slowdown:"
)

if (
    early_mean > middle_mean
    and middle_mean > late_mean
):

    print(
        "FAST > MODERATE > SLOW: PASS"
    )

else:

    print(
        "FAST > MODERATE > SLOW: FAIL"
    )


# ============================================================
# PLATEAU AUDIT
# ============================================================

section("9. PLATEAU AUDIT")

valid["is_stagnant"] = (
    valid["corrected_change_m"]
    .abs()
    <= 1e-9
)

plateau_lengths = []

for group_key, trajectory in valid.groupby(
    "group_key"
):

    trajectory = trajectory.sort_values(
        AGE_COL
    )

    current_length = 0

    for stagnant in trajectory[
        "is_stagnant"
    ]:

        if stagnant:

            current_length += 1

        else:

            if current_length > 0:

                plateau_lengths.append(
                    current_length
                )

            current_length = 0

    if current_length > 0:

        plateau_lengths.append(
            current_length
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

    print(
        f"Plateaus >= 10 transitions : "
        f"{(plateau_lengths >= 10).sum():,}"
    )


# ============================================================
# TOP CORRECTIONS
# ============================================================

section("10. TOP 30 CORRECTIONS")

top = (
    group_age[
        GROUP_COLS
        + [
            AGE_COL,
            "actual_height_m",
            "corrected_height_m",
            "correction_m",
            "corrected_previous_m",
        ]
    ]
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

section("11. SAVING PROTOTYPE")

prototype = df.merge(
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

prototype["height_correction_m"] = (
    prototype["corrected_height_m"]
    - prototype[HEIGHT_COL]
)

prototype.to_csv(
    OUTPUT_DIR
    / "height_bounded_forward_test.csv",
    index=False
)

group_age.to_csv(
    OUTPUT_DIR
    / "height_bounded_forward_group_age.csv",
    index=False
)

corrections.to_csv(
    OUTPUT_DIR
    / "height_bounded_forward_corrections.csv",
    index=False
)

distribution.to_csv(
    OUTPUT_DIR
    / "height_bounded_forward_distribution.csv",
    index=False
)

stage_summary.to_csv(
    OUTPUT_DIR
    / "height_bounded_forward_stage_summary.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

section("STEP 36K FINISHED")

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