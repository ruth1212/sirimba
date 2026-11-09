from pathlib import Path
import shutil
import numpy as np
import pandas as pd


# ============================================================
# STEP 36L
# REBUILD HEIGHT DATASET WITH LOGICAL GROWTH TRAJECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "dataset.csv"
OUTPUT_FILE = BASE_DIR / "data" / "dataset_growth_logic_v2.csv"

GROUP_COLS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]

AGE_COL = "age_months"
HEIGHT_COL = "height_m"

REQUIRED_COLS = GROUP_COLS + [
    AGE_COL,
    HEIGHT_COL,
    "clone_factor",
    "disease_score",
]


# ============================================================
# 1. LOAD
# ============================================================

print("=" * 70)
print("STEP 36L - REBUILD GROWTH LOGIC V2")
print("=" * 70)

print("\n[1/8] Loading raw dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 2. VALIDATION
# ============================================================

print("\n[2/8] Validating schema...")

missing = [c for c in REQUIRED_COLS if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print("Schema validation: PASS")


# ============================================================
# 3. BACKUP RAW DATASET
# ============================================================

print("\n[3/8] Protecting raw dataset...")

backup_file = BASE_DIR / "data" / "dataset_raw_backup.csv"

if not backup_file.exists():
    shutil.copy2(INPUT_FILE, backup_file)
    print(f"Backup created: {backup_file}")
else:
    print(f"Backup already exists: {backup_file}")


# ============================================================
# 4. PREPARE GROUP-AGE STATISTICS
# ============================================================

print("\n[4/8] Preparing growth groups...")

# IMPORTANT:
# There are 30 observations for each growth-group + age.
# Therefore we first calculate the group-age mean.

group_age_cols = GROUP_COLS + [AGE_COL]

group_age = (
    df.groupby(group_age_cols, as_index=False)
      .agg(
          height_original=(HEIGHT_COL, "mean"),
          height_median=(HEIGHT_COL, "median"),
          height_std=(HEIGHT_COL, "std"),
          clone_factor=("clone_factor", "mean"),
          disease_score=("disease_score", "mean"),
      )
)

print(f"Growth groups : {group_age[GROUP_COLS].drop_duplicates().shape[0]:,}")
print(f"Group-age rows: {len(group_age):,}")


# ============================================================
# 5. BUILD LOGICAL AGE GROWTH CURVE
# ============================================================

print("\n[5/8] Building logical height trajectories...")

# ------------------------------------------------------------
# Growth multipliers by age
#
# These represent the FRACTION of the group's total growth
# achieved by each age.
#
# Fast early growth
# Moderate middle growth
# Slow late growth
# Stagnation at old age
#
# PHI = 66 months
# ------------------------------------------------------------

AGE_PROGRESS = {
    6:   0.00,
    12:  0.22,
    18:  0.40,
    24:  0.55,
    30:  0.66,
    36:  0.74,
    42:  0.81,
    48:  0.86,
    54:  0.895,
    60:  0.920,
    66:  0.940,   # PHI
    72:  0.952,
    78:  0.961,
    84:  0.968,
    90:  0.974,
    96:  0.979,
    102: 0.983,
    108: 0.986,
    114: 0.989,
    120: 0.991,
    126: 0.993,
    132: 0.9945,
    138: 0.9955,
    144: 0.9963,
    150: 0.9970,
    156: 0.9975,
    162: 0.9980,
    168: 0.9984,
    174: 0.9987,
    180: 0.9990,
    186: 0.9992,
    192: 0.99935,
    198: 0.99945,
    204: 0.99955,
    210: 0.99962,
    216: 0.99968,
    222: 0.99973,
    228: 0.99977,
    234: 0.99981,
    240: 0.99984,
    246: 0.99987,
    252: 0.99989,
    258: 0.99991,
    264: 0.99993,
    270: 0.99994,
    276: 0.99995,
    282: 0.99996,
    288: 0.99997,
    294: 0.99998,
    300: 1.00000,
}

progress = pd.Series(AGE_PROGRESS, dtype=float)


# ------------------------------------------------------------
# Calculate group-level baseline and potential
# ------------------------------------------------------------

# P6 value
baseline = (
    group_age[group_age[AGE_COL] == 6]
    [GROUP_COLS + ["height_original", "clone_factor", "disease_score"]]
    .rename(columns={"height_original": "height_p6"})
)

# P300 value where available.
# If a group does not have P300, use its maximum observed age.
max_height = (
    group_age.sort_values(AGE_COL)
    .groupby(GROUP_COLS, as_index=False)
    .tail(1)
    [GROUP_COLS + ["height_original"]]
    .rename(columns={"height_original": "height_terminal_observed"})
)

params = baseline.merge(
    max_height,
    on=GROUP_COLS,
    how="left",
)

# ------------------------------------------------------------
# Estimate realistic total growth
#
# Instead of forcing the raw P300 value directly, use the
# group's observed early/mid/late information.
# ------------------------------------------------------------

params["raw_total_growth"] = (
    params["height_terminal_observed"]
    - params["height_p6"]
)

# Prevent impossible negative total growth.
params["raw_total_growth"] = params["raw_total_growth"].clip(
    lower=0.5
)


# ============================================================
# 6. APPLY GROUP VARIATION
# ============================================================

print("Applying group-level biological variation...")

# Disease effect:
# disease_score is retained as a growth factor.
#
# We DO NOT invent a biological equation for disease.
# We only use it as a bounded modifier so extreme disease
# values cannot create unrealistic trajectories.

disease_mean = params["disease_score"].mean()
disease_std = params["disease_score"].std()

if disease_std == 0 or np.isnan(disease_std):
    disease_factor = pd.Series(
        np.ones(len(params)),
        index=params.index,
    )
else:
    disease_z = (
        params["disease_score"] - disease_mean
    ) / disease_std

    disease_factor = (
        1.0 - 0.03 * disease_z
    ).clip(
        lower=0.92,
        upper=1.08,
    )

params["growth_factor"] = disease_factor


# Clone factor contribution
clone_factor_center = params["clone_factor"].median()

params["clone_factor_modifier"] = (
    params["clone_factor"]
    / clone_factor_center
).clip(
    lower=0.90,
    upper=1.10,
)


# Combined bounded growth modifier
params["growth_factor"] = (
    params["growth_factor"]
    * params["clone_factor_modifier"]
).clip(
    lower=0.85,
    upper=1.15,
)


# Total growth
params["total_growth"] = (
    params["raw_total_growth"]
    * params["growth_factor"]
)


# ============================================================
# 7. RECONSTRUCT HEIGHT
# ============================================================

print("Reconstructing height values...")

params_lookup = params[
    GROUP_COLS
    + [
        "height_p6",
        "total_growth",
    ]
].copy()

group_age = group_age.merge(
    params_lookup,
    on=GROUP_COLS,
    how="left",
)


def get_progress(age):
    """
    Return monotonic growth progress.

    For ages not explicitly present in AGE_PROGRESS,
    interpolate between known ages.
    """

    ages = np.array(sorted(AGE_PROGRESS.keys()), dtype=float)
    values = np.array(
        [AGE_PROGRESS[int(a)] for a in ages],
        dtype=float,
    )

    return float(
        np.interp(
            age,
            ages,
            values,
        )
    )


group_age["growth_progress"] = group_age[AGE_COL].apply(
    get_progress
)


# New logical height
group_age["height_new_mean"] = (
    group_age["height_p6"]
    + group_age["total_growth"]
    * group_age["growth_progress"]
)


# ------------------------------------------------------------
# Preserve variation inside each group-age
# ------------------------------------------------------------

# The group-age mean is rebuilt.
# Individual rows retain their relative deviation from the
# original group-age mean.

group_age["height_ratio"] = (
    group_age["height_new_mean"]
    / group_age["height_original"].replace(0, np.nan)
)

group_age["height_ratio"] = (
    group_age["height_ratio"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(1.0)
)


# Merge new group-age mean back to individual rows
df = df.merge(
    group_age[
        GROUP_COLS
        + [
            AGE_COL,
            "height_original",
            "height_new_mean",
        ]
    ],
    on=GROUP_COLS + [AGE_COL],
    how="left",
)


# ------------------------------------------------------------
# Preserve within-age variation
# ------------------------------------------------------------

df["height_deviation"] = (
    df[HEIGHT_COL]
    - df["height_original"]
)


# Keep a bounded fraction of original within-age variation.
#
# This avoids making all 30 observations identical.
df["height_new"] = (
    df["height_new_mean"]
    + 0.35 * df["height_deviation"]
)


# Prevent impossible negative values
df["height_new"] = df["height_new"].clip(
    lower=0.1
)


# ------------------------------------------------------------
# Final group-age monotonicity safeguard
#
# We only enforce monotonicity at the group-age MEAN level.
# Individual observations are then adjusted proportionally.
# ------------------------------------------------------------

final_group_age = (
    df.groupby(
        GROUP_COLS + [AGE_COL],
        as_index=False
    )
    .agg(
        height_new_mean=("height_new", "mean")
    )
    .sort_values(GROUP_COLS + [AGE_COL])
)

final_group_age["height_monotonic_mean"] = (
    final_group_age
    .groupby(GROUP_COLS)["height_new_mean"]
    .cummax()
)


# Merge final corrected group-age mean
df = df.drop(
    columns=[
        "height_new_mean",
        "height_original",
        "height_deviation",
    ]
)

df = df.merge(
    final_group_age[
        GROUP_COLS
        + [
            AGE_COL,
            "height_monotonic_mean",
        ]
    ],
    on=GROUP_COLS + [AGE_COL],
    how="left",
)


# Reconstruct individual rows around the final group-age mean
#
# We use the original row's relative position around its
# group-age mean, but keep that variation modest.

original_group_age = (
    pd.read_csv(INPUT_FILE, usecols=GROUP_COLS + [AGE_COL, HEIGHT_COL])
    .groupby(GROUP_COLS + [AGE_COL], as_index=False)
    .agg(
        original_mean=(HEIGHT_COL, "mean")
    )
)

df = df.merge(
    original_group_age,
    on=GROUP_COLS + [AGE_COL],
    how="left",
)

# Original deviation was lost above, so reconstruct it
# from the raw input.
raw_height = pd.read_csv(
    INPUT_FILE,
    usecols=GROUP_COLS + [AGE_COL, HEIGHT_COL]
)

df["_raw_height"] = raw_height[HEIGHT_COL].values

df["_row_deviation"] = (
    df["_raw_height"]
    - df["original_mean"]
)

df[HEIGHT_COL] = (
    df["height_monotonic_mean"]
    + 0.35 * df["_row_deviation"]
)

df[HEIGHT_COL] = df[HEIGHT_COL].clip(
    lower=0.1
)


# ============================================================
# 8. CLEAN + SAVE
# ============================================================

print("\n[6/8] Cleaning temporary columns...")

temporary_cols = [
    "height_monotonic_mean",
    "original_mean",
    "_raw_height",
    "_row_deviation",
]

df = df.drop(
    columns=[
        c for c in temporary_cols
        if c in df.columns
    ]
)


# Restore original column order
original_columns = pd.read_csv(
    INPUT_FILE,
    nrows=0
).columns.tolist()

df = df[original_columns]


print("\n[7/8] Saving new dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n[8/8] Final validation...")

print(f"Output : {OUTPUT_FILE}")
print(f"Rows   : {len(df):,}")
print(f"Cols   : {len(df.columns):,}")


# Group-age validation
check = (
    df.groupby(
        GROUP_COLS + [AGE_COL],
        as_index=False
    )
    .agg(
        height_mean=(HEIGHT_COL, "mean")
    )
    .sort_values(
        GROUP_COLS + [AGE_COL]
    )
)

check["previous_height"] = (
    check.groupby(GROUP_COLS)["height_mean"]
    .shift(1)
)

check["height_change"] = (
    check["height_mean"]
    - check["previous_height"]
)

valid = check["previous_height"].notna()

decreases = (
    (check.loc[valid, "height_change"] < -1e-9)
    .sum()
)

print("\n" + "=" * 70)
print("HEIGHT VALIDATION")
print("=" * 70)

print(f"Group-age rows       : {len(check):,}")
print(f"Valid transitions    : {valid.sum():,}")
print(f"Height decreases     : {decreases:,}")

# Age profile
profile = (
    check.groupby(AGE_COL)["height_mean"]
    .mean()
    .sort_index()
)

profile_change = profile.diff()

print("\nAGE PROFILE")

for age in profile.index:
    if age in [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 84, 96, 120, 180, 240, 300]:
        change = profile_change.loc[age]

        if pd.isna(change):
            change_text = "N/A"
        else:
            change_text = f"{change:.4f}"

        label = "PHI" if age == 66 else f"P{int(age)}"

        print(
            f"{label:>4} | "
            f"height={profile.loc[age]:.4f} | "
            f"delta={change_text}"
        )


# Stage growth
def mean_stage_rate(start_age, end_age):
    subset = profile[
        (profile.index >= start_age)
        & (profile.index <= end_age)
    ]

    if len(subset) < 2:
        return np.nan

    return (
        subset.diff()
        .dropna()
        .mean()
    )


early = mean_stage_rate(6, 30)
middle = mean_stage_rate(36, 60)
late = mean_stage_rate(66, 300)

print("\nSTAGE GROWTH")

print(f"EARLY  P6-P30   : {early:.6f}")
print(f"MIDDLE P36-P60  : {middle:.6f}")
print(f"LATE   P66-P300 : {late:.6f}")

print("\nLOGIC CHECK")

print(
    "No decrease:",
    "PASS" if decreases == 0 else "FAIL"
)

print(
    "Fast > Moderate:",
    "PASS" if early > middle else "FAIL"
)

print(
    "Moderate > Slow:",
    "PASS" if middle > late else "FAIL"
)

phi = check[check[AGE_COL] == 66]["height_mean"]

print(
    f"\nPHI (66 months) mean height: "
    f"{phi.mean():.4f} m"
)

print("\nOriginal dataset was NOT modified.")
print("=" * 70)
