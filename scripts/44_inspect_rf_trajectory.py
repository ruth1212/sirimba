from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# STEP 44
# RANDOM FOREST TRAJECTORY INSPECTION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAJECTORY_PATH = (
    ROOT
    / "results"
    / "random_forest"
    / "actual_vs_predicted_by_age.csv"
)

OUTPUT_PATH = (
    ROOT
    / "results"
    / "random_forest"
    / "trajectory_inspection.csv"
)


EXPECTED_MODELS = [
    "RF_DBH",
    "RF_HEIGHT",
    "RF_VOLUME",
]


EXPECTED_AGES = {
    "RF_DBH": [
        6, 12, 18, 24, 30,
        36, 42, 48, 54, 60, 66
    ],

    "RF_HEIGHT": [
        6, 12, 18, 24, 30,
        36, 42, 48, 54, 60, 66
    ],

    "RF_VOLUME": [
        18, 24, 30, 36, 42,
        48, 54, 60, 66
    ],
}


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    print()
    print("=" * 80)
    print("STEP 44 STATUS: FAIL")
    print("=" * 80)
    print(message)
    sys.exit(1)


def calculate_growth_rates(
    df,
    value_column
):
    """
    Calculate change per year between consecutive
    six-month observations.

    age_months difference = 6 months = 0.5 year.
    """

    df = df.sort_values(
        "age_months"
    ).copy()

    df["delta_value"] = (
        df[value_column]
        .diff()
    )

    df["delta_years"] = (
        df["age_months"]
        .diff()
        / 12.0
    )

    df["growth_per_year"] = (
        df["delta_value"]
        / df["delta_years"]
    )

    return df


# ============================================================
# START
# ============================================================

print("=" * 80)
print("STEP 44 - RANDOM FOREST TRAJECTORY INSPECTION")
print("=" * 80)

print()
print("This step:")
print("  - Reads RF test trajectory results")
print("  - Does NOT retrain models")
print("  - Does NOT modify the dataset")
print("  - Does NOT modify RF models")
print()


# ============================================================
# [1/8] LOAD TRAJECTORY
# ============================================================

print("[1/8] Loading RF trajectory...")

if not TRAJECTORY_PATH.exists():
    fail(
        f"Trajectory file not found:\n"
        f"{TRAJECTORY_PATH}"
    )

df = pd.read_csv(
    TRAJECTORY_PATH
)

print(
    f"Rows loaded: {len(df):,}"
)

required_columns = [
    "model",
    "target",
    "age_months",
    "age_years",
    "actual_mean",
    "predicted_mean",
    "absolute_error",
    "n",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    fail(
        "Trajectory file missing columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing
        )
    )

print("Trajectory structure: PASS")


# ============================================================
# [2/8] VERIFY MODELS
# ============================================================

print("\n[2/8] Checking RF models...")

models = sorted(
    df["model"].unique()
)

print(
    f"Models found: {models}"
)

if models != sorted(EXPECTED_MODELS):
    fail(
        "Unexpected RF model set."
    )

print("Model set: PASS")


# ============================================================
# [3/8] PRINT AGE-BY-AGE RESULTS
# ============================================================

print("\n[3/8] Actual vs predicted by age...")

inspection_rows = []

for model_name in EXPECTED_MODELS:

    model_df = df[
        df["model"] == model_name
    ].sort_values(
        "age_months"
    ).copy()

    expected_ages = EXPECTED_AGES[
        model_name
    ]

    actual_ages = (
        model_df["age_months"]
        .astype(int)
        .tolist()
    )

    if actual_ages != expected_ages:
        fail(
            f"{model_name} has unexpected age structure.\n"
            f"Expected: {expected_ages}\n"
            f"Actual  : {actual_ages}"
        )

    print()
    print("=" * 80)
    print(model_name)
    print("=" * 80)

    print(
        model_df[
            [
                "age_months",
                "actual_mean",
                "predicted_mean",
                "absolute_error",
                "n",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    for _, row in model_df.iterrows():

        inspection_rows.append({
            "model": model_name,
            "target": row["target"],
            "age_months": int(
                row["age_months"]
            ),
            "actual_mean": row[
                "actual_mean"
            ],
            "predicted_mean": row[
                "predicted_mean"
            ],
            "absolute_error": row[
                "absolute_error"
            ],
            "n": int(row["n"]),
        })


# ============================================================
# [4/8] DBH / HEIGHT MONOTONICITY
# ============================================================

print("\n[4/8] Checking DBH and Height trajectory...")

monotonic_results = []

for model_name in [
    "RF_DBH",
    "RF_HEIGHT",
]:

    model_df = df[
        df["model"] == model_name
    ].sort_values(
        "age_months"
    )

    predicted = (
        model_df["predicted_mean"]
        .to_numpy()
    )

    actual = (
        model_df["actual_mean"]
        .to_numpy()
    )

    predicted_delta = np.diff(
        predicted
    )

    actual_delta = np.diff(
        actual
    )

    predicted_decreases = int(
        np.sum(
            predicted_delta < 0
        )
    )

    actual_decreases = int(
        np.sum(
            actual_delta < 0
        )
    )

    print()
    print(model_name)

    print(
        f"Actual decreases    : "
        f"{actual_decreases}"
    )

    print(
        f"Predicted decreases : "
        f"{predicted_decreases}"
    )

    if predicted_decreases > 0:

        print(
            "WARNING: predicted mean trajectory "
            "contains a decrease."
        )

    else:

        print(
            "Predicted mean trajectory: "
            "NON-DECREASING"
        )

    monotonic_results.append({
        "model": model_name,
        "actual_decreases": actual_decreases,
        "predicted_decreases": predicted_decreases,
        "prediction_non_decreasing": (
            predicted_decreases == 0
        ),
    })


# ============================================================
# [5/8] EARLY / MIDDLE / LATE GROWTH
# ============================================================

print("\n[5/8] Checking growth-rate pattern...")

growth_summary = []

# Defined on the final P6-P66 scope:
#
# Early  : P12-P30
# Middle : P36-P48
# Late   : P54-P66
#
# This follows the already established project
# trajectory audit structure.

stages = {
    "EARLY": (12, 30),
    "MIDDLE": (36, 48),
    "LATE": (54, 66),
}


for model_name in [
    "RF_DBH",
    "RF_HEIGHT",
]:

    model_df = df[
        df["model"] == model_name
    ].sort_values(
        "age_months"
    ).copy()

    model_df["delta"] = (
        model_df["predicted_mean"]
        .diff()
    )

    model_df["delta_years"] = (
        model_df["age_months"]
        .diff()
        / 12.0
    )

    model_df["growth_per_year"] = (
        model_df["delta"]
        / model_df["delta_years"]
    )

    print()
    print(model_name)

    stage_values = {}

    for stage_name, (
        min_age,
        max_age
    ) in stages.items():

        stage = model_df[
            (
                model_df["age_months"]
                >= min_age
            )
            &
            (
                model_df["age_months"]
                <= max_age
            )
        ]

        value = stage[
            "growth_per_year"
        ].mean()

        stage_values[
            stage_name
        ] = value

        print(
            f"{stage_name:<8} "
            f"{value:.6f}"
        )

        growth_summary.append({
            "model": model_name,
            "stage": stage_name,
            "mean_predicted_growth_per_year": value,
        })

    early = stage_values["EARLY"]
    middle = stage_values["MIDDLE"]
    late = stage_values["LATE"]

    slowdown_pass = (
        early > middle > late
    )

    print(
        f"Early > Middle > Late: "
        f"{'PASS' if slowdown_pass else 'FAIL'}"
    )


# ============================================================
# [6/8] VOLUME TRAJECTORY
# ============================================================

print("\n[6/8] Checking Volume trajectory...")

volume_df = df[
    df["model"] == "RF_VOLUME"
].sort_values(
    "age_months"
).copy()

volume_actual = (
    volume_df["actual_mean"]
    .to_numpy()
)

volume_predicted = (
    volume_df["predicted_mean"]
    .to_numpy()
)

actual_volume_decreases = int(
    np.sum(
        np.diff(volume_actual) < 0
    )
)

predicted_volume_decreases = int(
    np.sum(
        np.diff(volume_predicted) < 0
    )
)

print(
    f"Actual volume decreases    : "
    f"{actual_volume_decreases}"
)

print(
    f"Predicted volume decreases : "
    f"{predicted_volume_decreases}"
)

print()
print(
    "Volume decreases are ALLOWED "
    "under the project growth logic."
)


# ============================================================
# [7/8] PHI/P66 INSPECTION
# ============================================================

print("\n[7/8] Checking P66 / PHI trajectory endpoint...")

phi_df = df[
    df["age_months"] == 66
].copy()

if len(phi_df) != 3:
    fail(
        "Expected exactly three P66 model rows."
    )

print()

print(
    phi_df[
        [
            "model",
            "target",
            "actual_mean",
            "predicted_mean",
            "absolute_error",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("P66 / PHI endpoint: PASS")


# ============================================================
# [8/8] SAVE INSPECTION
# ============================================================

print("\n[8/8] Saving trajectory inspection...")

inspection_df = pd.DataFrame(
    inspection_rows
)

inspection_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"Saved: {OUTPUT_PATH}"
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("STEP 44 STATUS: PASS")
print("=" * 80)

print()
print("RF TRAJECTORY INSPECTION COMPLETED.")

print()
print("Checked:")
print("  - Actual vs predicted by age")
print("  - DBH monotonicity")
print("  - Height monotonicity")
print("  - Early/Middle/Late growth")
print("  - Volume trajectory")
print("  - P66 / PHI endpoint")

print()
print(
    "No dataset was modified."
)
print(
    "No RF model was modified."
)
print(
    "No RF model was retrained."
)

print()
print(
    "Saved inspection:"
)
print(
    OUTPUT_PATH
)

print("=" * 80)