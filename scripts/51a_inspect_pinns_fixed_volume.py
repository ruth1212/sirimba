from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib


# ============================================================
# STEP 51A
# INSPECT PINNs FIXED VOLUME TRAJECTORY
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PIPELINE_DIR = (
    ROOT
    / "data"
    / "model_ready"
    / "pipeline"
)

PREPROCESSING_DIR = (
    PIPELINE_DIR
    / "preprocessing"
)

TRAIN_DBH_HEIGHT_PATH = (
    PIPELINE_DIR
    / "train_dbh_height.csv"
)

TRAIN_VOLUME_PATH = (
    PIPELINE_DIR
    / "train_volume_p18.csv"
)

TARGET_SCALER_PATH = (
    PREPROCESSING_DIR
    / "target_scaler.joblib"
)

MODEL_PATH = (
    ROOT
    / "results"
    / "pinns"
    / "models"
    / "pinns_growth_model_fixed_volume.pt"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "pinns"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "fixed_volume_trajectory_inspection.csv"
)

OUTPUT_REPORT = (
    OUTPUT_DIR
    / "fixed_volume_trajectory_inspection.json"
)

DEVICE = torch.device("cpu")


# ============================================================
# MODEL
# ============================================================

class PINNsGrowthModel(nn.Module):

    def __init__(
        self,
        input_size=6,
        hidden_size=64
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(
                input_size,
                hidden_size
            ),
            nn.Tanh(),

            nn.Linear(
                hidden_size,
                hidden_size
            ),
            nn.Tanh(),

            nn.Linear(
                hidden_size,
                hidden_size
            ),
            nn.Tanh(),
        )

        self.dbh_head = nn.Linear(
            hidden_size,
            1
        )

        self.height_head = nn.Linear(
            hidden_size,
            1
        )

        self.volume_head = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        hidden = self.shared(x)

        dbh_scaled = self.dbh_head(hidden)

        height_scaled = self.height_head(hidden)

        volume_latent = self.volume_head(hidden)

        return (
            dbh_scaled,
            height_scaled,
            volume_latent
        )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 51A — PINNs FIXED VOLUME TRAJECTORY INSPECTION")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/7] Loading pipeline data...")

dbh_height = pd.read_csv(
    TRAIN_DBH_HEIGHT_PATH
)

volume = pd.read_csv(
    TRAIN_VOLUME_PATH
)

print(
    f"DBH/Height rows : {len(dbh_height):,}"
)

print(
    f"Volume rows     : {len(volume):,}"
)


# ============================================================
# 2. LOAD SCALERS
# ============================================================

print("\n[2/7] Loading target scalers...")

scaler_data = joblib.load(
    TARGET_SCALER_PATH
)

dbh_scaler = scaler_data["dbh_cm"]
height_scaler = scaler_data["height_m"]
volume_scaler = scaler_data["volume_m3_tree"]

dbh_mean = float(
    dbh_scaler.mean_[0]
)

dbh_scale = float(
    dbh_scaler.scale_[0]
)

height_mean = float(
    height_scaler.mean_[0]
)

height_scale = float(
    height_scaler.scale_[0]
)

volume_mean = float(
    volume_scaler.mean_[0]
)

volume_scale = float(
    volume_scaler.scale_[0]
)

print(
    f"DBH    mean={dbh_mean:.6f}, "
    f"scale={dbh_scale:.6f}"
)

print(
    f"Height mean={height_mean:.6f}, "
    f"scale={height_scale:.6f}"
)

print(
    f"Volume mean={volume_mean:.6f}, "
    f"scale={volume_scale:.6f}"
)


# ============================================================
# 3. BUILD FEATURES
# ============================================================

print("\n[3/7] Preparing features...")

FEATURE_COLUMNS = [
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha",
]

X_full_df = dbh_height[
    FEATURE_COLUMNS
].copy()

X_volume_df = volume[
    FEATURE_COLUMNS
].copy()


X_full = torch.tensor(
    X_full_df.values,
    dtype=torch.float32,
    device=DEVICE
)

X_volume = torch.tensor(
    X_volume_df.values,
    dtype=torch.float32,
    device=DEVICE
)


# ============================================================
# 4. LOAD MODEL
# ============================================================

print("\n[4/7] Loading fixed-volume PINNs checkpoint...")

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model = PINNsGrowthModel(
    input_size=6,
    hidden_size=64
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Checkpoint loaded : PASS"
)


# ============================================================
# 5. PREDICTION
# ============================================================

print("\n[5/7] Generating predictions...")

with torch.no_grad():

    # Full P6-P66
    dbh_scaled_pred, height_scaled_pred, _ = model(
        X_full
    )

    # Volume P18-P66
    _, _, volume_latent = model(
        X_volume
    )


# ------------------------------------------------------------
# DBH / HEIGHT physical units
# ------------------------------------------------------------

dbh_pred = (
    dbh_scaled_pred.cpu().numpy().reshape(-1)
    * dbh_scale
    + dbh_mean
)

height_pred = (
    height_scaled_pred.cpu().numpy().reshape(-1)
    * height_scale
    + height_mean
)


# ------------------------------------------------------------
# VOLUME physical units
# ------------------------------------------------------------
#
# IMPORTANT:
# Same transformation used during Step 50:
#
# latent
#   -> physical raw volume
#   -> Softplus
#

volume_raw_physical = (
    volume_latent
    * volume_scale
    + volume_mean
)

volume_pred = F.softplus(
    volume_raw_physical
).cpu().numpy().reshape(-1)


# ============================================================
# 6. BUILD AGE TRAJECTORY
# ============================================================

print("\n[6/7] Building age-level trajectory...")

dbh_actual = dbh_height[
    "dbh_cm"
].values

height_actual = dbh_height[
    "height_m"
].values


dbh_height_age = (
    dbh_height
    .assign(
        dbh_pred=dbhh_pred
        if False
        else dbh_pred,
        height_pred=height_pred
    )
    .groupby(
        "age_months",
        as_index=False
    )
    .agg(
        dbh_actual=(
            "dbh_cm",
            "mean"
        ),
        dbh_pred=(
            "dbh_pred",
            "mean"
        ),
        height_actual=(
            "height_m",
            "mean"
        ),
        height_pred=(
            "height_pred",
            "mean"
        ),
        n_dbh=(
            "dbh_cm",
            "size"
        )
    )
)


volume_age = (
    volume
    .assign(
        volume_pred=volume_pred
    )
    .groupby(
        "age_months",
        as_index=False
    )
    .agg(
        volume_actual=(
            "volume_m3_tree",
            "mean"
        ),
        volume_pred=(
            "volume_pred",
            "mean"
        ),
        n_volume=(
            "volume_m3_tree",
            "size"
        )
    )
)


trajectory = dbh_height_age.merge(
    volume_age,
    on="age_months",
    how="left"
)


trajectory["dbh_abs_error"] = (
    trajectory["dbh_pred"]
    - trajectory["dbh_actual"]
).abs()

trajectory["height_abs_error"] = (
    trajectory["height_pred"]
    - trajectory["height_actual"]
).abs()

trajectory["volume_abs_error"] = (
    trajectory["volume_pred"]
    - trajectory["volume_actual"]
).abs()


trajectory["volume_relative_error_pct"] = (
    trajectory["volume_abs_error"]
    / trajectory["volume_actual"].replace(
        0,
        np.nan
    )
    * 100
)


# ============================================================
# GROWTH RATES
# ============================================================

trajectory["dbh_growth"] = (
    trajectory["dbh_pred"].diff()
)

trajectory["height_growth"] = (
    trajectory["height_pred"].diff()
)

trajectory["volume_growth"] = (
    trajectory["volume_pred"].diff()
)


# ============================================================
# SAVE CSV
# ============================================================

trajectory.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# COLLAPSE DIAGNOSTICS
# ============================================================

volume_prediction_min = float(
    volume_pred.min()
)

volume_prediction_max = float(
    volume_pred.max()
)

volume_prediction_mean = float(
    volume_pred.mean()
)

volume_prediction_std = float(
    volume_pred.std()
)

volume_prediction_range = (
    volume_prediction_max
    - volume_prediction_min
)


# Age-level variation

age_volume_std = float(
    trajectory["volume_pred"].std()
)

age_volume_range = float(
    trajectory["volume_pred"].max()
    - trajectory["volume_pred"].min()
)


# Count predicted decreases

volume_decreases = int(
    (
        trajectory["volume_growth"]
        < 0
    )
    .sum()
)

dbh_decreases = int(
    (
        trajectory["dbh_growth"]
        < 0
    )
    .sum()
)

height_decreases = int(
    (
        trajectory["height_growth"]
        < 0
    )
    .sum()
)


# ============================================================
# REPORT
# ============================================================

report = {

    "status": "PASS",

    "model": str(
        MODEL_PATH
    ),

    "rows": {
        "dbh_height":
            int(len(dbh_height)),
        "volume":
            int(len(volume)),
        "trajectory_ages":
            int(len(trajectory))
    },

    "volume_prediction": {

        "min":
            volume_prediction_min,

        "max":
            volume_prediction_max,

        "mean":
            volume_prediction_mean,

        "std":
            volume_prediction_std,

        "range":
            volume_prediction_range,

        "age_level_std":
            age_volume_std,

        "age_level_range":
            age_volume_range
    },

    "trajectory": {

        "volume_decreases":
            volume_decreases,

        "dbh_decreases":
            dbh_decreases,

        "height_decreases":
            height_decreases
    },

    "collapse_diagnostic": {

        "near_constant_threshold":
            0.01,

        "is_near_constant":
            volume_prediction_range < 0.01,

        "interpretation":
            (
                "Volume prediction is flagged as "
                "near-constant when the full training "
                "prediction range is below 0.01 m3/tree."
            )
    }
}


with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PINNs AGE TRAJECTORY")
print("=" * 70)

print(
    trajectory[
        [
            "age_months",
            "dbh_actual",
            "dbh_pred",
            "height_actual",
            "height_pred",
            "volume_actual",
            "volume_pred",
            "volume_abs_error"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


print("\n" + "=" * 70)
print("VOLUME COLLAPSE DIAGNOSTIC")
print("=" * 70)

print(
    f"Prediction minimum : "
    f"{volume_prediction_min:.8f}"
)

print(
    f"Prediction maximum : "
    f"{volume_prediction_max:.8f}"
)

print(
    f"Prediction mean    : "
    f"{volume_prediction_mean:.8f}"
)

print(
    f"Prediction std     : "
    f"{volume_prediction_std:.8f}"
)

print(
    f"Prediction range   : "
    f"{volume_prediction_range:.8f}"
)

print(
    f"Age-level range    : "
    f"{age_volume_range:.8f}"
)

print()
print(
    f"Volume decreases   : "
    f"{volume_decreases}"
)

print(
    f"DBH decreases      : "
    f"{dbh_decreases}"
)

print(
    f"Height decreases   : "
    f"{height_decreases}"
)


if volume_prediction_range < 0.01:

    print()
    print(
        "WARNING: VOLUME PREDICTION "
        "IS NEAR-CONSTANT."
    )

else:

    print()
    print(
        "Volume prediction range "
        "is not classified as near-constant."
    )


print()
print("Saved:")
print(
    f"- {OUTPUT_CSV}"
)

print(
    f"- {OUTPUT_REPORT}"
)

print("=" * 70)
print("STEP 51A COMPLETE")
print("=" * 70)