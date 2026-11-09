from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# STEP 47 - CALIBRATE PHYSICS COEFFICIENT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    ROOT
    / "data"
    / "model_ready"
    / "train.csv"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "pinns"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PROJECT RULES
# ============================================================

MIN_AGE = 18
MAX_AGE = 66

REQUIRED_COLUMNS = [
    "age_months",
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]


# ============================================================
# START
# ============================================================

print("=" * 80)
print("STEP 47 - PHYSICS COEFFICIENT CALIBRATION")
print("=" * 80)

print("""
Physics relation:

    V = C * DBH^2 * Height

Coefficient:

    C = V / (DBH^2 * Height)

Calibration data:
    - Official TRAIN dataset only
    - P18-P66 only
    - No test data
    - No RF model
    - No PINNs training
    - Dataset will NOT be modified
""")


# ============================================================
# 1. LOAD TRAIN DATA
# ============================================================

print("\n[1/7] Loading official training data...")

if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Training dataset not found: {TRAIN_PATH}"
    )

df = pd.read_csv(TRAIN_PATH)

print(f"Train rows loaded: {len(df):,}")


missing = [
    col
    for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print("Required columns: PASS")


# ============================================================
# 2. APPLY PHYSICS WINDOW
# ============================================================

print("\n[2/7] Applying P18-P66 physics window...")

physics_df = df[
    df["age_months"].between(
        MIN_AGE,
        MAX_AGE
    )
].copy()

print(
    f"Physics rows: {len(physics_df):,}"
)

ages = sorted(
    physics_df["age_months"]
    .unique()
    .tolist()
)

expected_ages = list(
    range(MIN_AGE, MAX_AGE + 1, 6)
)

print(f"Physics ages: {ages}")

if ages != expected_ages:
    raise ValueError(
        f"Unexpected physics ages.\n"
        f"Expected: {expected_ages}\n"
        f"Found: {ages}"
    )

print("Physics window P18-P66: PASS")


# ============================================================
# 3. VALIDATE PHYSICAL VALUES
# ============================================================

print("\n[3/7] Validating physical values...")

for col in [
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]:
    if physics_df[col].isna().any():
        raise ValueError(
            f"{col} contains missing values."
        )

    if not np.isfinite(
        physics_df[col].to_numpy()
    ).all():
        raise ValueError(
            f"{col} contains NaN or Inf."
        )

if (physics_df["dbh_cm"] <= 0).any():
    raise ValueError(
        "DBH contains zero or negative values."
    )

if (physics_df["height_m"] <= 0).any():
    raise ValueError(
        "Height contains zero or negative values."
    )

if (physics_df["volume_m3_tree"] < 0).any():
    raise ValueError(
        "Volume contains negative values."
    )

print("DBH positive: PASS")
print("Height positive: PASS")
print("Volume non-negative: PASS")


# ============================================================
# 4. CALCULATE C
# ============================================================

print("\n[4/7] Calculating physics coefficient C...")

denominator = (
    physics_df["dbh_cm"] ** 2
    * physics_df["height_m"]
)

if (denominator <= 0).any():
    raise ValueError(
        "Invalid DBH² × Height denominator."
    )

physics_df["physics_coefficient"] = (
    physics_df["volume_m3_tree"]
    / denominator
)

if not np.isfinite(
    physics_df["physics_coefficient"].to_numpy()
).all():
    raise ValueError(
        "Calculated physics coefficient contains "
        "NaN or Inf."
    )

if (
    physics_df["physics_coefficient"] <= 0
).any():
    raise ValueError(
        "Physics coefficient contains "
        "zero or negative values."
    )

print("Coefficient calculation: PASS")


# ============================================================
# 5. DISTRIBUTION ANALYSIS
# ============================================================

print("\n[5/7] Analyzing coefficient distribution...")

c = physics_df["physics_coefficient"]

summary = {
    "count": int(c.count()),
    "mean": float(c.mean()),
    "median": float(c.median()),
    "std": float(c.std()),
    "min": float(c.min()),
    "q01": float(c.quantile(0.01)),
    "q05": float(c.quantile(0.05)),
    "q25": float(c.quantile(0.25)),
    "q75": float(c.quantile(0.75)),
    "q95": float(c.quantile(0.95)),
    "q99": float(c.quantile(0.99)),
    "max": float(c.max()),
}

print("\nPhysics coefficient distribution:")

for key, value in summary.items():
    if key == "count":
        print(f"{key:>8}: {value:,}")
    else:
        print(f"{key:>8}: {value:.12f}")


# ============================================================
# 6. AGE-BY-AGE CALIBRATION CHECK
# ============================================================

print("\n[6/7] Checking coefficient by age...")

age_summary = (
    physics_df
    .groupby("age_months")["physics_coefficient"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
    )
    .reset_index()
)

print("\nCoefficient by age:")

print(
    age_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.10f}"
    )
)


# ============================================================
# FINAL COEFFICIENT
# ============================================================

final_coefficient = float(
    c.median()
)

print("\nSelected physics coefficient:")
print(
    f"C = {final_coefficient:.12f}"
)


# ============================================================
# RESIDUAL CHECK
# ============================================================

physics_df["volume_physics"] = (
    final_coefficient
    * physics_df["dbh_cm"] ** 2
    * physics_df["height_m"]
)

physics_df["physics_residual"] = (
    physics_df["volume_m3_tree"]
    - physics_df["volume_physics"]
)

physics_df["absolute_physics_error"] = (
    physics_df["physics_residual"].abs()
)

residual_mae = float(
    physics_df["absolute_physics_error"].mean()
)

residual_rmse = float(
    np.sqrt(
        np.mean(
            physics_df["physics_residual"] ** 2
        )
    )
)

print("\nPhysics residual check:")
print(
    f"Physics MAE : {residual_mae:.8f}"
)
print(
    f"Physics RMSE: {residual_rmse:.8f}"
)


# ============================================================
# SAVE DETAILED CALIBRATION DATA
# ============================================================

print("\nSaving calibration results...")

coefficient_path = (
    RESULT_DIR
    / "physics_coefficient_by_row.csv"
)

physics_df[
    [
        "age_months",
        "dbh_cm",
        "height_m",
        "volume_m3_tree",
        "physics_coefficient",
        "volume_physics",
        "physics_residual",
        "absolute_physics_error",
    ]
].to_csv(
    coefficient_path,
    index=False,
)

age_path = (
    RESULT_DIR
    / "physics_coefficient_by_age.csv"
)

age_summary.to_csv(
    age_path,
    index=False,
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

physics_config = {
    "formula": "V = C * DBH^2 * Height",

    "coefficient_method": "median",

    "coefficient": final_coefficient,

    "calibration_source": (
        "data/model_ready/train.csv"
    ),

    "calibration_age_min_months": MIN_AGE,
    "calibration_age_max_months": MAX_AGE,

    "calibration_rows": int(len(physics_df)),

    "residual": {
        "mae": residual_mae,
        "rmse": residual_rmse,
    },

    "test_data_used": False,

    "dataset_modified": False,

    "rf_model_used": False,

    "pinns_training_performed": False,
}

config_path = (
    RESULT_DIR
    / "physics_coefficient_config.json"
)

with open(
    config_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        physics_config,
        f,
        indent=2,
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("STEP 47 STATUS: PASS")
print("=" * 80)

print("""
Physics coefficient calibration completed.

Verified:
  - Official TRAIN data only
  - P18-P66 only
  - DBH positive
  - Height positive
  - Volume non-negative
  - C calculated from actual physical values
  - Median selected as robust coefficient
  - Physics residual calculated
  - Calibration configuration saved

NO DATASET WAS MODIFIED.
NO RF MODEL WAS MODIFIED.
NO PINNs MODEL WAS TRAINED.
NO TEST DATA WAS USED.
""")

print(f"Coefficient config:")
print(config_path)

print("=" * 80)