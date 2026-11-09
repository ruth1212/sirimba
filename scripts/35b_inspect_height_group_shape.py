from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 35B
# INSPECT HEIGHT GROWTH SHAPE PER GROWTH GROUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "dataset_corrected.csv"
RESULT_DIR = BASE_DIR / "results" / "height_group_shape_inspection"

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

EARLY_END = 30
MIDDLE_END = 60


print("=" * 80)
print("STEP 35B — HEIGHT GROWTH SHAPE PER GROUP")
print("=" * 80)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"\nRows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ------------------------------------------------------------
# GROUP-AGE AGGREGATION
# ------------------------------------------------------------

trajectory = (
    df.groupby(GROUP_COLS + [AGE_COL], as_index=False)
      .agg(
          height_mean=(TARGET, "mean"),
          height_median=(TARGET, "median"),
          row_count=(TARGET, "size"),
      )
)

trajectory = trajectory.sort_values(
    GROUP_COLS + [AGE_COL]
).reset_index(drop=True)


print(
    f"Group-age trajectory rows : "
    f"{len(trajectory):,}"
)


# ------------------------------------------------------------
# AGE-TO-AGE TRANSITIONS
# ------------------------------------------------------------

trajectory["previous_age"] = (
    trajectory.groupby(GROUP_COLS)[AGE_COL]
    .shift(1)
)

trajectory["previous_height"] = (
    trajectory.groupby(GROUP_COLS)["height_mean"]
    .shift(1)
)

trajectory["delta_height"] = (
    trajectory["height_mean"]
    - trajectory["previous_height"]
)

trajectory["growth_rate_cm_year"] = (
    trajectory["delta_height"]
    /
    (trajectory[AGE_COL] - trajectory["previous_age"])
    * 12
)

transitions = trajectory[
    trajectory["previous_age"].notna()
    &
    (trajectory[AGE_COL] > trajectory["previous_age"])
].copy()


# ------------------------------------------------------------
# CLASSIFY
# ------------------------------------------------------------

EPS = 1e-9

transitions["direction"] = np.select(
    [
        transitions["delta_height"] > EPS,
        transitions["delta_height"].abs() <= EPS,
        transitions["delta_height"] < -EPS,
    ],
    [
        "increase",
        "stagnation",
        "decrease",
    ],
    default="unknown",
)


# ------------------------------------------------------------
# PERIOD SUMMARY PER GROUP
# ------------------------------------------------------------

group_summaries = []

for group_key, g in trajectory.groupby(GROUP_COLS):

    g = g.sort_values(AGE_COL)

    t = transitions.merge(
        pd.DataFrame(
            [dict(zip(GROUP_COLS, group_key))]
        ),
        on=GROUP_COLS,
        how="inner",
    )

    early = t[
        (t[AGE_COL] > 6)
        & (t[AGE_COL] <= EARLY_END)
    ]

    middle = t[
        (t[AGE_COL] > EARLY_END)
        & (t[AGE_COL] <= MIDDLE_END)
    ]

    late = t[
        t[AGE_COL] > MIDDLE_END
    ]

    def mean_rate(x):
        if len(x) == 0:
            return np.nan
        return x["growth_rate_cm_year"].mean()

    def median_rate(x):
        if len(x) == 0:
            return np.nan
        return x["growth_rate_cm_year"].median()

    def count_direction(x, direction):
        return int(
            (x["direction"] == direction).sum()
        )

    early_rate = mean_rate(early)
    middle_rate = mean_rate(middle)
    late_rate = mean_rate(late)

    group_summaries.append(
        {
            **dict(zip(GROUP_COLS, group_key)),

            "height_p6":
                g.loc[g[AGE_COL] == 6, "height_mean"].iloc[0]
                if (g[AGE_COL] == 6).any()
                else np.nan,

            "height_p30":
                g.loc[g[AGE_COL] == 30, "height_mean"].iloc[0]
                if (g[AGE_COL] == 30).any()
                else np.nan,

            "height_p60":
                g.loc[g[AGE_COL] == 60, "height_mean"].iloc[0]
                if (g[AGE_COL] == 60).any()
                else np.nan,

            "height_phi66":
                g.loc[g[AGE_COL] == PHI_AGE, "height_mean"].iloc[0]
                if (g[AGE_COL] == PHI_AGE).any()
                else np.nan,

            "early_mean_growth_cm_year":
                early_rate,

            "middle_mean_growth_cm_year":
                middle_rate,

            "late_mean_growth_cm_year":
                late_rate,

            "early_median_growth_cm_year":
                median_rate(early),

            "middle_median_growth_cm_year":
                median_rate(middle),

            "late_median_growth_cm_year":
                median_rate(late),

            "early_decreases":
                count_direction(early, "decrease"),

            "middle_decreases":
                count_direction(middle, "decrease"),

            "late_decreases":
                count_direction(late, "decrease"),

            "early_stagnations":
                count_direction(early, "stagnation"),

            "middle_stagnations":
                count_direction(middle, "stagnation"),

            "late_stagnations":
                count_direction(late, "stagnation"),
        }
    )


summary = pd.DataFrame(group_summaries)


# ------------------------------------------------------------
# GROWTH-SHAPE FLAGS
# ------------------------------------------------------------

summary["early_gt_middle"] = (
    summary["early_mean_growth_cm_year"]
    >
    summary["middle_mean_growth_cm_year"]
)

summary["middle_gt_late"] = (
    summary["middle_mean_growth_cm_year"]
    >
    summary["late_mean_growth_cm_year"]
)

summary["three_stage_slowdown"] = (
    summary["early_gt_middle"]
    &
    summary["middle_gt_late"]
)

summary["has_late_decrease"] = (
    summary["late_decreases"] > 0
)

summary["has_any_decrease"] = (
    (
        summary["early_decreases"]
        + summary["middle_decreases"]
        + summary["late_decreases"]
    ) > 0
)


# ------------------------------------------------------------
# SUMMARY STATISTICS
# ------------------------------------------------------------

print("\n" + "-" * 80)
print("1. GROUP-LEVEL SHAPE SUMMARY")
print("-" * 80)

total_groups = len(summary)

three_stage_count = int(
    summary["three_stage_slowdown"].sum()
)

late_decrease_count = int(
    summary["has_late_decrease"].sum()
)

any_decrease_count = int(
    summary["has_any_decrease"].sum()
)

print(f"\nTotal growth groups       : {total_groups:,}")

print(
    f"Three-stage slowdown      : "
    f"{three_stage_count:,} "
    f"({three_stage_count / total_groups * 100:.2f}%)"
)

print(
    f"Groups with late decrease : "
    f"{late_decrease_count:,} "
    f"({late_decrease_count / total_groups * 100:.2f}%)"
)

print(
    f"Groups with any decrease  : "
    f"{any_decrease_count:,} "
    f"({any_decrease_count / total_groups * 100:.2f}%)"
)


# ------------------------------------------------------------
# GROWTH RATE DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "-" * 80)
print("2. GROWTH RATE DISTRIBUTION")
print("-" * 80)

rate_columns = [
    "early_mean_growth_cm_year",
    "middle_mean_growth_cm_year",
    "late_mean_growth_cm_year",
]

print(
    summary[rate_columns]
    .describe()
    .to_string()
)


# ------------------------------------------------------------
# EXAMPLES OF GOOD SHAPES
# ------------------------------------------------------------

print("\n" + "-" * 80)
print("3. EXAMPLES — THREE-STAGE SLOWDOWN")
print("-" * 80)

good = summary[
    summary["three_stage_slowdown"]
].copy()

if len(good) > 0:
    print(
        good[
            GROUP_COLS
            + rate_columns
            + [
                "early_decreases",
                "middle_decreases",
                "late_decreases",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
else:
    print("No groups found.")


# ------------------------------------------------------------
# EXAMPLES OF PROBLEMATIC SHAPES
# ------------------------------------------------------------

print("\n" + "-" * 80)
print("4. EXAMPLES — LATE-AGE DECREASE")
print("-" * 80)

problem = summary[
    summary["has_late_decrease"]
].copy()

problem = problem.sort_values(
    "late_decreases",
    ascending=False
)

if len(problem) > 0:
    print(
        problem[
            GROUP_COLS
            + rate_columns
            + [
                "early_decreases",
                "middle_decreases",
                "late_decreases",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
else:
    print("No late-age decrease groups found.")


# ------------------------------------------------------------
# GLOBAL PERIOD MEDIANS
# ------------------------------------------------------------

print("\n" + "-" * 80)
print("5. MEDIAN GROWTH RATE BY GROUP")
print("-" * 80)

print(
    f"\nEarly  median : "
    f"{summary['early_mean_growth_cm_year'].median():.4f} cm/year"
)

print(
    f"Middle median: "
    f"{summary['middle_mean_growth_cm_year'].median():.4f} cm/year"
)

print(
    f"Late   median: "
    f"{summary['late_mean_growth_cm_year'].median():.4f} cm/year"
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

summary_path = (
    RESULT_DIR / "height_group_shape_summary.csv"
)

transitions_path = (
    RESULT_DIR / "height_group_transitions.csv"
)

trajectory_path = (
    RESULT_DIR / "height_group_age_trajectory.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

transitions.to_csv(
    transitions_path,
    index=False
)

trajectory.to_csv(
    trajectory_path,
    index=False
)


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 35B RESULT")
print("=" * 80)

print("\nInspection completed.")
print("NO DATA WAS MODIFIED.")

print("\nOutput:")
print(f"- {summary_path}")
print(f"- {transitions_path}")
print(f"- {trajectory_path}")

print(
    "\nNEXT DECISION:"
)

print(
    "Gunakan hasil ini untuk menentukan apakah Height "
    "memerlukan shape-constrained correction."
)

print("=" * 80)