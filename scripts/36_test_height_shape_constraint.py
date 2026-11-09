from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# STEP 36
# TEST SHAPE-CONSTRAINED HEIGHT TRAJECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "dataset_corrected.csv"

RESULT_DIR = (
    BASE_DIR
    / "results"
    / "height_shape_constraint_test"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)


GROUP_COLS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]

AGE_COL = "age_months"
TARGET = "height_m"

PHI_AGE = 66


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("STEP 36 — TEST SHAPE-CONSTRAINED HEIGHT")
print("=" * 80)

print(f"\nDataset : {DATA_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# GROUP + AGE MEAN
# ============================================================

trajectory = (
    df.groupby(
        GROUP_COLS + [AGE_COL],
        as_index=False
    )
    .agg(
        height_actual=(TARGET, "mean"),
        row_count=(TARGET, "size"),
    )
)

trajectory = trajectory.sort_values(
    GROUP_COLS + [AGE_COL]
).reset_index(drop=True)

print(
    f"\nGroup-age rows : {len(trajectory):,}"
)


# ============================================================
# SHAPE-CONSTRAINED PROJECTION
#
# We construct increments:
#
# delta_1 >= delta_2 >= ... >= delta_n >= 0
#
# Then reconstruct:
#
# H_t = H_6 + cumulative(delta)
#
# This enforces:
#   1. non-decreasing height
#   2. non-increasing growth increments
#   3. possible late-age stagnation
#
# ============================================================


def project_nonincreasing_nonnegative(values):
    """
    Project a sequence onto:
        x1 >= x2 >= ... >= xn >= 0

    Uses a simple Pool Adjacent Violators procedure
    for non-increasing values, followed by clipping at zero.
    """

    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values.copy()

    # Work with blocks.
    blocks = []

    for value in values:
        blocks.append(
            {
                "sum": float(value),
                "count": 1,
                "mean": float(value),
            }
        )

        # For non-increasing sequence:
        # previous mean must be >= current mean.
        while (
            len(blocks) >= 2
            and blocks[-2]["mean"] < blocks[-1]["mean"]
        ):
            right = blocks.pop()
            left = blocks.pop()

            total_sum = left["sum"] + right["sum"]
            total_count = left["count"] + right["count"]

            merged = {
                "sum": total_sum,
                "count": total_count,
                "mean": total_sum / total_count,
            }

            blocks.append(merged)

    projected = []

    for block in blocks:
        projected.extend(
            [block["mean"]] * block["count"]
        )

    projected = np.asarray(
        projected,
        dtype=float
    )

    # Growth cannot be negative.
    projected = np.maximum(
        projected,
        0.0
    )

    # Re-check monotonicity after clipping.
    for i in range(1, len(projected)):
        if projected[i] > projected[i - 1]:
            projected[i] = projected[i - 1]

    return projected


def shape_constrained_height(ages, heights):
    """
    Fit a trajectory where:

        height is non-decreasing

        growth increment is non-increasing

    The first height value is preserved.

    The interval increments are projected onto:

        delta_1 >= delta_2 >= ... >= delta_n >= 0
    """

    ages = np.asarray(
        ages,
        dtype=float
    )

    heights = np.asarray(
        heights,
        dtype=float
    )

    if len(ages) < 2:
        return heights.copy()

    # Actual age intervals.
    age_intervals = np.diff(ages)

    # Actual height change.
    height_deltas = np.diff(heights)

    # Convert to growth rate per month.
    raw_rates = (
        height_deltas
        / age_intervals
    )

    # We need the growth-rate shape:
    #
    # rate_1 >= rate_2 >= ... >= rate_n >= 0
    #
    constrained_rates = (
        project_nonincreasing_nonnegative(
            raw_rates
        )
    )

    # Convert rates back to height increments.
    constrained_deltas = (
        constrained_rates
        * age_intervals
    )

    # Reconstruct height.
    corrected = np.empty_like(
        heights,
        dtype=float
    )

    corrected[0] = heights[0]

    corrected[1:] = (
        heights[0]
        + np.cumsum(
            constrained_deltas
        )
    )

    return corrected


# ============================================================
# APPLY TO EACH GROUP
# ============================================================

results = []

for group_key, group in trajectory.groupby(
    GROUP_COLS,
    sort=False
):

    group = group.sort_values(
        AGE_COL
    ).copy()

    ages = group[AGE_COL].to_numpy()

    actual = group[
        "height_actual"
    ].to_numpy()

    corrected = shape_constrained_height(
        ages,
        actual
    )

    group["height_corrected_test"] = corrected

    group["height_correction_delta"] = (
        corrected - actual
    )

    results.append(group)


test_result = pd.concat(
    results,
    ignore_index=True
)

test_result = test_result.sort_values(
    GROUP_COLS + [AGE_COL]
).reset_index(drop=True)


# ============================================================
# VALIDATION 1
# HEIGHT MUST NOT DECREASE
# ============================================================

print("\n" + "-" * 80)
print("1. MONOTONIC HEIGHT VALIDATION")
print("-" * 80)

test_result["previous_corrected"] = (
    test_result.groupby(GROUP_COLS)[
        "height_corrected_test"
    ].shift(1)
)

test_result["corrected_delta"] = (
    test_result["height_corrected_test"]
    - test_result["previous_corrected"]
)

valid_transitions = test_result[
    test_result["previous_corrected"].notna()
].copy()

valid_transitions = valid_transitions[
    valid_transitions[AGE_COL]
    >
    valid_transitions.groupby(GROUP_COLS)[AGE_COL]
    .shift(1)
]

decreases = (
    valid_transitions["corrected_delta"] < -1e-9
).sum()

stagnations = (
    valid_transitions["corrected_delta"].abs()
    <= 1e-9
).sum()

increases = (
    valid_transitions["corrected_delta"] > 1e-9
).sum()

print(f"\nTransitions : {len(valid_transitions):,}")
print(f"Increases   : {increases:,}")
print(f"Stagnations : {stagnations:,}")
print(f"Decreases   : {decreases:,}")


# ============================================================
# VALIDATION 2
# GROWTH RATE MUST NOT INCREASE
# ============================================================

print("\n" + "-" * 80)
print("2. GROWTH-RATE SHAPE VALIDATION")
print("-" * 80)

test_result["previous_age"] = (
    test_result.groupby(GROUP_COLS)[
        AGE_COL
    ].shift(1)
)

test_result["age_interval"] = (
    test_result[AGE_COL]
    - test_result["previous_age"]
)

test_result["corrected_growth_rate_cm_year"] = (
    test_result["corrected_delta"]
    /
    test_result["age_interval"]
    * 12
)

rates = test_result[
    test_result["corrected_growth_rate_cm_year"].notna()
].copy()

rates["previous_rate"] = (
    rates.groupby(GROUP_COLS)[
        "corrected_growth_rate_cm_year"
    ].shift(1)
)

rate_transitions = rates[
    rates["previous_rate"].notna()
].copy()

rate_increases = (
    rate_transitions[
        "corrected_growth_rate_cm_year"
    ]
    >
    rate_transitions["previous_rate"]
    + 1e-9
).sum()

rate_decreases = (
    rate_transitions[
        "corrected_growth_rate_cm_year"
    ]
    <
    rate_transitions["previous_rate"]
    - 1e-9
).sum()

rate_stable = (
    np.abs(
        rate_transitions[
            "corrected_growth_rate_cm_year"
        ]
        -
        rate_transitions["previous_rate"]
    )
    <= 1e-9
).sum()

print(
    f"\nRate transitions : "
    f"{len(rate_transitions):,}"
)

print(
    f"Rate increases   : "
    f"{rate_increases:,}"
)

print(
    f"Rate decreases   : "
    f"{rate_decreases:,}"
)

print(
    f"Rate stable      : "
    f"{rate_stable:,}"
)


# ============================================================
# VALIDATION 3
# EARLY / MIDDLE / LATE
# ============================================================

print("\n" + "-" * 80)
print("3. EARLY / MIDDLE / LATE GROWTH")
print("-" * 80)

periods = {
    "EARLY": (12, 30),
    "MIDDLE": (36, 60),
    "LATE": (66, 300),
}

period_rows = []

for name, (min_age, max_age) in periods.items():

    subset = rates[
        (rates[AGE_COL] >= min_age)
        &
        (rates[AGE_COL] <= max_age)
    ]

    period_rows.append(
        {
            "period": name,
            "mean_growth_cm_year":
                subset[
                    "corrected_growth_rate_cm_year"
                ].mean(),
            "median_growth_cm_year":
                subset[
                    "corrected_growth_rate_cm_year"
                ].median(),
            "min_growth_cm_year":
                subset[
                    "corrected_growth_rate_cm_year"
                ].min(),
            "max_growth_cm_year":
                subset[
                    "corrected_growth_rate_cm_year"
                ].max(),
        }
    )

period_summary = pd.DataFrame(
    period_rows
)

print(
    period_summary.to_string(
        index=False
    )
)


# ============================================================
# VALIDATION 4
# CORRECTION MAGNITUDE
# ============================================================

print("\n" + "-" * 80)
print("4. CORRECTION MAGNITUDE")
print("-" * 80)

correction_abs = (
    test_result[
        "height_correction_delta"
    ].abs()
)

print(
    f"\nMean absolute correction : "
    f"{correction_abs.mean():.6f} m"
)

print(
    f"Median absolute correction : "
    f"{correction_abs.median():.6f} m"
)

print(
    f"Maximum absolute correction : "
    f"{correction_abs.max():.6f} m"
)

print(
    f"Rows changed : "
    f"{(
        correction_abs > 1e-9
    ).sum():,}"
)


# ============================================================
# VALIDATION 5
# SAMPLE TRAJECTORIES
# ============================================================

print("\n" + "-" * 80)
print("5. SAMPLE TRAJECTORIES")
print("-" * 80)

sample_group = (
    test_result.groupby(GROUP_COLS)
    .size()
    .index
)

sample_groups = list(
    sample_group[:3]
)

for i, group_key in enumerate(
    sample_groups,
    start=1
):

    sample = test_result[
        (
            test_result[GROUP_COLS]
            ==
            pd.Series(
                group_key,
                index=GROUP_COLS
            )
        ).all(axis=1)
    ].copy()

    print(
        f"\n--- SAMPLE GROUP {i} ---"
    )

    print(
        sample[
            [
                *GROUP_COLS,
                AGE_COL,
                "height_actual",
                "height_corrected_test",
                "height_correction_delta",
            ]
        ]
        .to_string(index=False)
    )


# ============================================================
# SAVE
# ============================================================

trajectory_path = (
    RESULT_DIR
    / "height_shape_constrained_test.csv"
)

period_path = (
    RESULT_DIR
    / "height_shape_period_summary.csv"
)

test_result.to_csv(
    trajectory_path,
    index=False
)

period_summary.to_csv(
    period_path,
    index=False
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 80)
print("STEP 36 RESULT")
print("=" * 80)

if decreases == 0:
    print(
        "\nMonotonic Height : PASS"
    )
else:
    print(
        f"\nMonotonic Height : FAIL "
        f"({decreases} decreases)"
    )

if rate_increases == 0:
    print(
        "Diminishing growth : PASS"
    )
else:
    print(
        f"Diminishing growth : FAIL "
        f"({rate_increases} rate increases)"
    )

print(
    "\nIMPORTANT:"
)

print(
    "This is ONLY a prototype test."
)

print(
    "dataset_corrected.csv was NOT modified."
)

print(
    "\nOutput:"
)

print(
    trajectory_path
)

print(
    period_path
)

print("=" * 80)