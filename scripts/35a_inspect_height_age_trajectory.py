from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 35A
# INSPECT HEIGHT TRAJECTORY AT GROUP-AGE LEVEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "dataset_corrected.csv"
RESULT_DIR = BASE_DIR / "results" / "height_trajectory_inspection"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

GROUP_COLS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]

AGE_COL = "age_months"
TARGET = "height_m"

REQUIRED_COLUMNS = GROUP_COLS + [
    AGE_COL,
    TARGET,
]

EXPECTED_AGES = list(range(6, 301, 6))

PHI_AGE = 66


# ------------------------------------------------------------
# HELPER
# ------------------------------------------------------------

def growth_period(age):
    if age == 66:
        return "PHI"

    return f"P{int(age)}"


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("=" * 80)
print("STEP 35A — HEIGHT AGE-LEVEL TRAJECTORY INSPECTION")
print("=" * 80)

print(f"\nDataset : {DATA_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ------------------------------------------------------------
# VALIDATE SCHEMA
# ------------------------------------------------------------

missing = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing:
    raise ValueError(
        "Kolom yang dibutuhkan tidak ditemukan:\n"
        + "\n".join(f"- {c}" for c in missing)
    )

print("\nSchema validation : PASS")


# ------------------------------------------------------------
# BASIC AGE CHECK
# ------------------------------------------------------------

ages = sorted(df[AGE_COL].dropna().unique())

print("\nUnique ages:")
print(ages)

unexpected_ages = [
    age for age in ages
    if age not in EXPECTED_AGES
]

if unexpected_ages:
    print("\nWARNING — unexpected ages:")
    print(unexpected_ages)
else:
    print("\nAge structure : PASS")


# ------------------------------------------------------------
# CHECK DUPLICATE ROWS PER GROUP + AGE
# ------------------------------------------------------------

print("\n" + "-" * 80)
print("1. CHECK ROWS PER GROWTH GROUP + AGE")
print("-" * 80)

group_age_counts = (
    df.groupby(GROUP_COLS + [AGE_COL])
      .size()
      .reset_index(name="row_count")
)

print("\nRows per group-age:")
print(group_age_counts["row_count"].describe())


duplicate_group_age = group_age_counts[
    group_age_counts["row_count"] > 1
].copy()

print(
    f"\nGroup-age combinations with >1 row : "
    f"{len(duplicate_group_age):,}"
)

print(
    f"Maximum rows within one group-age   : "
    f"{group_age_counts['row_count'].max():,}"
)


duplicate_group_age.to_csv(
    RESULT_DIR / "duplicate_group_age_summary.csv",
    index=False
)


# ------------------------------------------------------------
# CREATE AGE-LEVEL TRAJECTORY
# ------------------------------------------------------------

print("\n" + "-" * 80)
print("2. CREATE GROUP-AGE TRAJECTORY")
print("-" * 80)

trajectory = (
    df.groupby(GROUP_COLS + [AGE_COL], as_index=False)
      .agg(
          height_mean=(TARGET, "mean"),
          height_median=(TARGET, "median"),
          height_min=(TARGET, "min"),
          height_max=(TARGET, "max"),
          row_count=(TARGET, "size"),
      )
)

trajectory["growth_period"] = trajectory[AGE_COL].apply(
    growth_period
)

trajectory = trajectory.sort_values(
    GROUP_COLS + [AGE_COL]
).reset_index(drop=True)


print(
    f"\nGroup-age trajectory rows : "
    f"{len(trajectory):,}"
)

print(
    "\nThis is the trajectory that will be used "
    "for biological-growth inspection."
)


# ------------------------------------------------------------
# CALCULATE WITHIN-GROUP AGE TRANSITIONS
# ------------------------------------------------------------

trajectory["previous_age"] = (
    trajectory.groupby(GROUP_COLS)[AGE_COL]
              .shift(1)
)

trajectory["previous_height_mean"] = (
    trajectory.groupby(GROUP_COLS)["height_mean"]
              .shift(1)
)

trajectory["height_delta"] = (
    trajectory["height_mean"]
    - trajectory["previous_height_mean"]
)

trajectory["growth_rate_cm_per_month"] = (
    trajectory["height_delta"]
    /
    (
        trajectory[AGE_COL]
        - trajectory["previous_age"]
    )
)

trajectory["growth_rate_cm_per_year"] = (
    trajectory["growth_rate_cm_per_month"] * 12
)


# ------------------------------------------------------------
# REMOVE FIRST OBSERVATION OF EACH GROUP
# ------------------------------------------------------------

transitions = trajectory[
    trajectory["previous_age"].notna()
].copy()

# Make sure transitions are actually age-to-age
transitions = transitions[
    transitions[AGE_COL] > transitions["previous_age"]
].copy()


# ------------------------------------------------------------
# CLASSIFY TRANSITIONS
# ------------------------------------------------------------

EPS = 1e-9

transitions["transition_type"] = np.select(
    [
        transitions["height_delta"] > EPS,
        transitions["height_delta"].abs() <= EPS,
        transitions["height_delta"] < -EPS,
    ],
    [
        "increase",
        "stagnation",
        "decrease",
    ],
    default="unknown",
)


# ------------------------------------------------------------
# TRANSITION SUMMARY
# ------------------------------------------------------------

print("\n" + "-" * 80)
print("3. VALID AGE-TO-AGE TRANSITION SUMMARY")
print("-" * 80)

total_transitions = len(transitions)

increase_count = (
    transitions["transition_type"] == "increase"
).sum()

stagnation_count = (
    transitions["transition_type"] == "stagnation"
).sum()

decrease_count = (
    transitions["transition_type"] == "decrease"
).sum()

print(f"\nTotal transitions : {total_transitions:,}")
print(f"Increases         : {increase_count:,}")
print(f"Stagnations       : {stagnation_count:,}")
print(f"Decreases         : {decrease_count:,}")

if total_transitions > 0:
    print(
        f"Increase %        : "
        f"{increase_count / total_transitions * 100:.2f}%"
    )

    print(
        f"Stagnation %      : "
        f"{stagnation_count / total_transitions * 100:.2f}%"
    )

    print(
        f"Decrease %        : "
        f"{decrease_count / total_transitions * 100:.2f}%"
    )


# ------------------------------------------------------------
# AGE-LEVEL GLOBAL PROFILE
# ------------------------------------------------------------

print("\n" + "-" * 80)
print("4. GLOBAL HEIGHT PROFILE BY AGE")
print("-" * 80)

age_profile = (
    df.groupby(AGE_COL)[TARGET]
      .agg(
          mean="mean",
          median="median",
          min="min",
          max="max",
          std="std",
          count="count",
      )
      .reset_index()
      .sort_values(AGE_COL)
)

age_profile["growth_period"] = age_profile[AGE_COL].apply(
    growth_period
)

age_profile["delta_from_previous_age"] = (
    age_profile["mean"].diff()
)

age_profile["growth_rate_cm_per_month"] = (
    age_profile["delta_from_previous_age"]
    /
    age_profile[AGE_COL].diff()
)

age_profile["growth_rate_cm_per_year"] = (
    age_profile["growth_rate_cm_per_month"] * 12
)

print(
    age_profile[
        [
            AGE_COL,
            "growth_period",
            "mean",
            "median",
            "delta_from_previous_age",
            "growth_rate_cm_per_year",
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# EARLY / MIDDLE / LATE GROWTH
# ------------------------------------------------------------

print("\n" + "-" * 80)
print("5. EARLY / MIDDLE / LATE GROWTH DIAGNOSTIC")
print("-" * 80)


def summarize_period(name, min_age, max_age):
    subset = age_profile[
        (age_profile[AGE_COL] >= min_age)
        &
        (age_profile[AGE_COL] <= max_age)
        &
        (age_profile["growth_rate_cm_per_year"].notna())
    ].copy()

    if subset.empty:
        return {
            "period": name,
            "min_age": min_age,
            "max_age": max_age,
            "mean_growth_cm_year": np.nan,
            "median_growth_cm_year": np.nan,
            "min_growth_cm_year": np.nan,
            "max_growth_cm_year": np.nan,
        }

    return {
        "period": name,
        "min_age": min_age,
        "max_age": max_age,
        "mean_growth_cm_year":
            subset["growth_rate_cm_per_year"].mean(),
        "median_growth_cm_year":
            subset["growth_rate_cm_per_year"].median(),
        "min_growth_cm_year":
            subset["growth_rate_cm_per_year"].min(),
        "max_growth_cm_year":
            subset["growth_rate_cm_per_year"].max(),
    }


period_summary = pd.DataFrame([
    summarize_period("EARLY", 6, 30),
    summarize_period("MIDDLE", 36, 60),
    summarize_period("LATE", 66, 300),
])

print(
    period_summary.to_string(index=False)
)


# ------------------------------------------------------------
# CHECK DIMINISHING GROWTH
# ------------------------------------------------------------

print("\n" + "-" * 80)
print("6. DIMINISHING-GROWTH CHECK")
print("-" * 80)

valid_rates = age_profile[
    age_profile["growth_rate_cm_per_year"].notna()
].copy()

if len(valid_rates) >= 2:

    rate_values = (
        valid_rates["growth_rate_cm_per_year"]
        .to_numpy()
    )

    rate_diffs = np.diff(rate_values)

    rate_increase_count = np.sum(rate_diffs > EPS)
    rate_decrease_count = np.sum(rate_diffs < -EPS)
    rate_equal_count = np.sum(np.abs(rate_diffs) <= EPS)

    print(
        f"\nGrowth-rate increases : {rate_increase_count:,}"
    )

    print(
        f"Growth-rate decreases : {rate_decrease_count:,}"
    )

    print(
        f"Growth-rate stable    : {rate_equal_count:,}"
    )

    print(
        "\nInterpretation:"
    )

    print(
        "- Kita TIDAK mengharuskan setiap interval "
        "harus turun."
    )

    print(
        "- Yang dicari adalah pola umum: "
        "pertumbuhan tinggi di awal, kemudian melambat."
    )

else:
    print("\nNot enough data for diminishing-growth check.")


# ------------------------------------------------------------
# PHI CHECK
# ------------------------------------------------------------

print("\n" + "-" * 80)
print("7. PHI CHECK")
print("-" * 80)

phi_rows = trajectory[
    trajectory[AGE_COL] == PHI_AGE
]

print(f"\nPHI age : {PHI_AGE}")
print(f"PHI group-age rows : {len(phi_rows):,}")

if len(phi_rows) > 0:

    print(
        "\nPHI height statistics:"
    )

    print(
        phi_rows["height_mean"]
        .describe()
        .to_string()
    )

else:
    print("\nWARNING: PHI age 66 not found.")


# ------------------------------------------------------------
# SAVE OUTPUTS
# ------------------------------------------------------------

trajectory_path = (
    RESULT_DIR / "height_group_age_trajectory.csv"
)

transitions_path = (
    RESULT_DIR / "height_age_transitions.csv"
)

profile_path = (
    RESULT_DIR / "height_age_profile.csv"
)

period_path = (
    RESULT_DIR / "height_growth_period_summary.csv"
)


trajectory.to_csv(
    trajectory_path,
    index=False
)

transitions.to_csv(
    transitions_path,
    index=False
)

age_profile.to_csv(
    profile_path,
    index=False
)

period_summary.to_csv(
    period_path,
    index=False
)


# ------------------------------------------------------------
# FINAL RESULT
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 35A RESULT")
print("=" * 80)

print("\nInspection completed.")
print("\nOutput files:")

print(f"- {trajectory_path}")
print(f"- {transitions_path}")
print(f"- {profile_path}")
print(f"- {period_path}")
print(
    f"- {RESULT_DIR / 'duplicate_group_age_summary.csv'}"
)

print("\nIMPORTANT:")
print("NO DATA WAS MODIFIED IN THIS STEP.")

print(
    "\nNext decision will be based on the actual "
    "age-level trajectory and growth-rate pattern."
)

print("\n" + "=" * 80)