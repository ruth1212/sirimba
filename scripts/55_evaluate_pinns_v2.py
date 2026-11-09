from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# STEP 55 — PINNs V2 TEST EVALUATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = (
    ROOT
    / "data"
    / "model_ready"
    / "test.csv"
)

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

FEATURE_SCALER_PATH = (
    PREPROCESSING_DIR
    / "feature_scaler.joblib"
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
    / "pinns_v2_growth_model.pt"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "pinns"
)

METRICS_PATH = (
    OUTPUT_DIR
    / "pinns_v2_test_metrics.csv"
)

AGE_PATH = (
    OUTPUT_DIR
    / "pinns_v2_actual_vs_predicted_by_age.csv"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "pinns_v2_test_predictions.csv"
)

PHI_PATH = (
    OUTPUT_DIR
    / "pinns_v2_phi_metrics.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "pinns_v2_evaluation_summary.json"
)

DEVICE = torch.device("cpu")


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_COLUMNS = [
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha",
]

TARGET_COLUMNS = [
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]


# ============================================================
# MODEL
# ============================================================

class PINNsGrowthModelV2(nn.Module):

    def __init__(
        self,
        input_size=6,
        hidden_size=64,
    ):

        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(
                input_size,
                hidden_size,
            ),
            nn.Tanh(),

            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.Tanh(),

            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.Tanh(),
        )

        self.dbh_head = nn.Linear(
            hidden_size,
            1,
        )

        self.height_head = nn.Linear(
            hidden_size,
            1,
        )

        self.volume_head = nn.Linear(
            hidden_size,
            1,
        )

    def forward(self, x):

        hidden = self.shared(x)

        dbh_scaled = self.dbh_head(
            hidden
        )

        height_scaled = self.height_head(
            hidden
        )

        volume_scaled = self.volume_head(
            hidden
        )

        return (
            dbh_scaled,
            height_scaled,
            volume_scaled,
        )


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_mape(
    actual,
    predicted,
):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mask = actual != 0

    if not np.any(mask):
        return np.nan

    return (
        np.mean(
            np.abs(
                (
                    actual[mask]
                    - predicted[mask]
                )
                / actual[mask]
            )
        )
        * 100
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 55 — PINNs V2 TEST EVALUATION")
print("=" * 70)


# ============================================================
# [1/10] LOAD TEST DATA
# ============================================================

print("\n[1/10] Loading official test dataset...")

test = pd.read_csv(
    TEST_PATH
)

print(
    f"Test rows : {len(test):,}"
)

if len(test) != 43740:
    raise RuntimeError(
        f"Unexpected test rows: {len(test):,}"
    )

print(
    f"Age range : "
    f"P{test['age_months'].min()} "
    f"to P{test['age_months'].max()}"
)

print(
    f"Compartments: "
    f"{test['compartment_id'].nunique()}"
)


# ============================================================
# [2/10] LOAD SCALERS
# ============================================================

print("\n[2/10] Loading preprocessing scalers...")

feature_scaler = joblib.load(
    FEATURE_SCALER_PATH
)

target_scalers = joblib.load(
    TARGET_SCALER_PATH
)

dbh_scaler = target_scalers[
    "dbh_cm"
]

height_scaler = target_scalers[
    "height_m"
]

volume_scaler = target_scalers[
    "volume_m3_tree"
]

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


# ============================================================
# [3/10] LOAD MODEL
# ============================================================

print("\n[3/10] Loading PINNs V2 checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

model = PINNsGrowthModelV2(
    input_size=6,
    hidden_size=64,
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.to(DEVICE)
model.eval()

print(
    f"Checkpoint epoch : "
    f"{checkpoint['epoch']}"
)

print(
    f"Checkpoint loss  : "
    f"{checkpoint['loss_total']:.8f}"
)


# ============================================================
# [4/10] PREPARE TEST FEATURES
# ============================================================

print("\n[4/10] Preparing test features...")

# ------------------------------------------------------------
# DBH + HEIGHT
# ------------------------------------------------------------

test_dbh_height = test[
    test["age_months"] >= 6
].copy()

# ------------------------------------------------------------
# VOLUME
# ------------------------------------------------------------

test_volume = test[
    test["age_months"] >= 18
].copy()

print(
    f"DBH/Height test rows : "
    f"{len(test_dbh_height):,}"
)

print(
    f"Volume test rows     : "
    f"{len(test_volume):,}"
)


# ============================================================
# [5/10] PREDICT DBH + HEIGHT
# ============================================================

print("\n[5/10] Predicting DBH and Height...")

X_dbh_height_raw = (
    test_dbh_height[
        FEATURE_COLUMNS
    ]
    .copy()
)

X_dbh_height_scaled = (
    feature_scaler.transform(
        X_dbh_height_raw
    )
)

X_dbh_height = torch.tensor(
    X_dbh_height_scaled,
    dtype=torch.float32,
    device=DEVICE,
)

with torch.no_grad():

    dbh_scaled_pred, height_scaled_pred, _ = (
        model(X_dbh_height)
    )

dbh_pred = (
    dbh_scaled_pred
    .cpu()
    .numpy()
    .reshape(-1)
)

height_pred = (
    height_scaled_pred
    .cpu()
    .numpy()
    .reshape(-1)
)

dbh_pred = (
    dbh_pred
    * dbh_scale
    + dbh_mean
)

height_pred = (
    height_pred
    * height_scale
    + height_mean
)

test_dbh_height[
    "dbh_pred"
] = dbh_pred

test_dbh_height[
    "height_pred"
] = height_pred


# ============================================================
# [6/10] PREDICT VOLUME
# ============================================================

print("\n[6/10] Predicting Volume...")

X_volume_raw = (
    test_volume[
        FEATURE_COLUMNS
    ]
    .copy()
)

X_volume_scaled = (
    feature_scaler.transform(
        X_volume_raw
    )
)

X_volume = torch.tensor(
    X_volume_scaled,
    dtype=torch.float32,
    device=DEVICE,
)

with torch.no_grad():

    _, _, volume_scaled_pred = (
        model(X_volume)
    )

volume_pred = (
    volume_scaled_pred
    .cpu()
    .numpy()
    .reshape(-1)
)

volume_pred = (
    volume_pred
    * volume_scale
    + volume_mean
)

test_volume[
    "volume_pred"
] = volume_pred


# ============================================================
# [7/10] CALCULATE METRICS
# ============================================================

print("\n[7/10] Calculating test metrics...")

evaluation_data = {

    "dbh_cm": (
        test_dbh_height[
            "dbh_cm"
        ].to_numpy(),
        test_dbh_height[
            "dbh_pred"
        ].to_numpy(),
    ),

    "height_m": (
        test_dbh_height[
            "height_m"
        ].to_numpy(),
        test_dbh_height[
            "height_pred"
        ].to_numpy(),
    ),

    "volume_m3_tree": (
        test_volume[
            "volume_m3_tree"
        ].to_numpy(),
        test_volume[
            "volume_pred"
        ].to_numpy(),
    ),
}

metrics = []

for target, (
    actual,
    predicted,
) in evaluation_data.items():

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    r2 = r2_score(
        actual,
        predicted,
    )

    mape = calculate_mape(
        actual,
        predicted,
    )

    negative_count = int(
        np.sum(predicted < 0)
    )

    metrics.append(
        {
            "model": "PINNs V2",
            "target": target,
            "n_test": len(actual),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE_percent": mape,
            "negative_prediction_count":
                negative_count,
        }
    )


metrics_df = pd.DataFrame(
    metrics
)

print(
    metrics_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# [8/10] AGE TRAJECTORY
# ============================================================

print("\n[8/10] Building actual vs predicted trajectory...")

dbh_age = (
    test_dbh_height
    .assign(
        abs_error=lambda d:
            np.abs(
                d["dbh_cm"]
                - d["dbh_pred"]
            )
    )
    .groupby("age_months")
    .agg(
        dbh_actual=(
            "dbh_cm",
            "mean",
        ),
        dbh_pred=(
            "dbh_pred",
            "mean",
        ),
        dbh_mae=(
            "abs_error",
            "mean",
        ),
        height_actual=(
            "height_m",
            "mean",
        ),
        height_pred=(
            "height_pred",
            "mean",
        ),
    )
)

height_age_mae = (
    test_dbh_height
    .assign(
        abs_error=lambda d:
            np.abs(
                d["height_m"]
                - d["height_pred"]
            )
    )
    .groupby("age_months")[
        "abs_error"
    ]
    .mean()
)

volume_age = (
    test_volume
    .assign(
        abs_error=lambda d:
            np.abs(
                d["volume_m3_tree"]
                - d["volume_pred"]
            )
    )
    .groupby("age_months")
    .agg(
        volume_actual=(
            "volume_m3_tree",
            "mean",
        ),
        volume_pred=(
            "volume_pred",
            "mean",
        ),
        volume_mae=(
            "abs_error",
            "mean",
        ),
        negative_volume_count=(
            "volume_pred",
            lambda x:
                int(
                    np.sum(
                        x < 0
                    )
                ),
        ),
    )
)

trajectory = (
    dbh_age
    .join(
        height_age_mae.rename(
            "height_mae"
        )
    )
    .join(
        volume_age,
        how="left",
    )
    .reset_index()
)

trajectory[
    "dbh_error"
] = (
    trajectory["dbh_pred"]
    - trajectory["dbh_actual"]
)

trajectory[
    "height_error"
] = (
    trajectory["height_pred"]
    - trajectory["height_actual"]
)

trajectory[
    "volume_error"
] = (
    trajectory["volume_pred"]
    - trajectory["volume_actual"]
)

print(
    trajectory.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# TRAJECTORY LOGIC
# ============================================================

dbh_changes = np.diff(
    trajectory[
        "dbh_pred"
    ].to_numpy()
)

height_changes = np.diff(
    trajectory[
        "height_pred"
    ].to_numpy()
)

volume_changes = np.diff(
    trajectory[
        "volume_pred"
    ]
    .dropna()
    .to_numpy()
)

dbh_decreases = int(
    np.sum(
        dbh_changes < 0
    )
)

height_decreases = int(
    np.sum(
        height_changes < 0
    )
)

volume_decreases = int(
    np.sum(
        volume_changes < 0
    )
)

negative_volume_total = int(
    np.sum(
        test_volume[
            "volume_pred"
        ] < 0
    )
)

negative_volume_rate = (
    negative_volume_total
    / len(test_volume)
)


# ============================================================
# [9/10] P66 / PHI
# ============================================================

print("\n[9/10] Checking P66 / PHI...")

phi_dbh_height = test_dbh_height[
    test_dbh_height[
        "age_months"
    ] == 66
]

phi_volume = test_volume[
    test_volume[
        "age_months"
    ] == 66
]

phi = {

    "dbh_actual_mean":
        float(
            phi_dbh_height[
                "dbh_cm"
            ].mean()
        ),

    "dbh_pred_mean":
        float(
            phi_dbh_height[
                "dbh_pred"
            ].mean()
        ),

    "dbh_mae":
        float(
            np.mean(
                np.abs(
                    phi_dbh_height[
                        "dbh_cm"
                    ]
                    -
                    phi_dbh_height[
                        "dbh_pred"
                    ]
                )
            )
        ),

    "height_actual_mean":
        float(
            phi_dbh_height[
                "height_m"
            ].mean()
        ),

    "height_pred_mean":
        float(
            phi_dbh_height[
                "height_pred"
            ].mean()
        ),

    "height_mae":
        float(
            np.mean(
                np.abs(
                    phi_dbh_height[
                        "height_m"
                    ]
                    -
                    phi_dbh_height[
                        "height_pred"
                    ]
                )
            )
        ),

    "volume_actual_mean":
        float(
            phi_volume[
                "volume_m3_tree"
            ].mean()
        ),

    "volume_pred_mean":
        float(
            phi_volume[
                "volume_pred"
            ].mean()
        ),

    "volume_mae":
        float(
            np.mean(
                np.abs(
                    phi_volume[
                        "volume_m3_tree"
                    ]
                    -
                    phi_volume[
                        "volume_pred"
                    ]
                )
            )
        ),

    "negative_volume_count":
        int(
            np.sum(
                phi_volume[
                    "volume_pred"
                ] < 0
            )
        ),
}

print(
    f"DBH    actual={phi['dbh_actual_mean']:.6f} "
    f"pred={phi['dbh_pred_mean']:.6f} "
    f"MAE={phi['dbh_mae']:.6f}"
)

print(
    f"Height actual={phi['height_actual_mean']:.6f} "
    f"pred={phi['height_pred_mean']:.6f} "
    f"MAE={phi['height_mae']:.6f}"
)

print(
    f"Volume actual={phi['volume_actual_mean']:.6f} "
    f"pred={phi['volume_pred_mean']:.6f} "
    f"MAE={phi['volume_mae']:.6f}"
)

print(
    "Negative P66 volume count="
    f"{phi['negative_volume_count']}"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

# ============================================================
# SAVE COMPACT PREDICTIONS
# ============================================================

print("\nSaving compact prediction results...")

# Simpan DBH + Height
test_dbh_height[
    [
        "compartment_id",
        "age_months",
        "dbh_cm",
        "dbh_pred",
        "height_m",
        "height_pred",
    ]
].to_csv(
    OUTPUT_DIR / "pinns_v2_test_dbh_height_predictions.csv",
    index=False,
)

# Simpan Volume terpisah
test_volume[
    [
        "compartment_id",
        "age_months",
        "volume_m3_tree",
        "volume_pred",
    ]
].to_csv(
    OUTPUT_DIR / "pinns_v2_test_volume_predictions.csv",
    index=False,
)

print("Compact prediction files saved.")

trajectory.to_csv(
    AGE_PATH,
    index=False,
)

metrics_df.to_csv(
    METRICS_PATH,
    index=False,
)

pd.DataFrame(
    [phi]
).to_csv(
    PHI_PATH,
    index=False,
)


# ============================================================
# [10/10] SAVE SUMMARY
# ============================================================

print("\n[10/10] Saving evaluation summary...")

summary = {

    "status": "COMPLETE",

    "model": "PINNs V2",

    "checkpoint": str(
        MODEL_PATH
    ),

    "checkpoint_epoch":
        int(checkpoint["epoch"]),

    "checkpoint_loss":
        float(checkpoint["loss_total"]),

    "test_rows":
        int(len(test)),

    "metrics":
        metrics,

    "trajectory_logic": {

        "dbh_predicted_decreases":
            dbh_decreases,

        "height_predicted_decreases":
            height_decreases,

        "volume_predicted_decreases":
            volume_decreases,

        "negative_volume_count":
            negative_volume_total,

        "negative_volume_rate":
            float(
                negative_volume_rate
            ),
    },

    "phi66": phi,

    "note": (
        "Evaluation performed on the official "
        "held-out test dataset. No retraining "
        "or dataset modification was performed."
    ),
}

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STEP 55 PINNs V2 TEST EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Metrics     : {METRICS_PATH}"
)

print(
    f"Trajectory  : {AGE_PATH}"
)

print(
    f"Predictions : {PREDICTIONS_PATH}"
)

print(
    f"PHI metrics : {PHI_PATH}"
)

print(
    f"Summary     : {REPORT_PATH}"
)

print("=" * 70)