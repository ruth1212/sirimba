from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib


# ============================================================
# STEP 54 — PINNs V2 TRAJECTORY INSPECTION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PIPELINE_DIR = ROOT / "data" / "model_ready" / "pipeline"
PREPROCESSING_DIR = PIPELINE_DIR / "preprocessing"

TRAIN_DBH_HEIGHT_PATH = PIPELINE_DIR / "train_dbh_height.csv"
TRAIN_VOLUME_PATH = PIPELINE_DIR / "train_volume_p18.csv"

FEATURE_SCALER_PATH = PREPROCESSING_DIR / "feature_scaler.joblib"
TARGET_SCALER_PATH = PREPROCESSING_DIR / "target_scaler.joblib"

MODEL_PATH = (
    ROOT
    / "results"
    / "pinns"
    / "models"
    / "pinns_v2_growth_model.pt"
)

OUTPUT_DIR = ROOT / "results" / "pinns"

TRAJECTORY_PATH = (
    OUTPUT_DIR / "pinns_v2_trajectory_inspection.csv"
)

REPORT_PATH = (
    OUTPUT_DIR / "pinns_v2_trajectory_inspection.json"
)

DEVICE = torch.device("cpu")


# ============================================================
# CONFIG
# ============================================================

FEATURE_COLUMNS = [
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha",
]

AGE_LABELS = {
    6: "P6",
    12: "P12",
    18: "P18",
    24: "P24",
    30: "P30",
    36: "P36",
    42: "P42",
    48: "P48",
    54: "P54",
    60: "P60",
    66: "P66",
}


# ============================================================
# MODEL
# ============================================================

class PINNsGrowthModelV2(nn.Module):

    def __init__(self, input_size=6, hidden_size=64):

        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),

            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),

            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.dbh_head = nn.Linear(hidden_size, 1)
        self.height_head = nn.Linear(hidden_size, 1)
        self.volume_head = nn.Linear(hidden_size, 1)

    def forward(self, x):

        hidden = self.shared(x)

        dbh_scaled = self.dbh_head(hidden)
        height_scaled = self.height_head(hidden)
        volume_scaled = self.volume_head(hidden)

        return (
            dbh_scaled,
            height_scaled,
            volume_scaled,
        )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 54 — PINNs V2 TRAJECTORY INSPECTION")
print("=" * 70)


# ============================================================
# [1/9] LOAD DATA
# ============================================================

print("\n[1/9] Loading training data...")

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
# [2/9] LOAD SCALERS
# ============================================================

print("\n[2/9] Loading scalers...")

feature_scaler = joblib.load(
    FEATURE_SCALER_PATH
)

target_scalers = joblib.load(
    TARGET_SCALER_PATH
)

dbh_scaler = target_scalers["dbh_cm"]
height_scaler = target_scalers["height_m"]
volume_scaler = target_scalers["volume_m3_tree"]

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
# [3/9] LOAD MODEL
# ============================================================

print("\n[3/9] Loading PINNs V2 checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

model = PINNsGrowthModelV2(
    input_size=6,
    hidden_size=64,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print(
    f"Checkpoint epoch : {checkpoint['epoch']}"
)

print(
    f"Checkpoint loss  : "
    f"{checkpoint['loss_total']:.8f}"
)


# ============================================================
# [4/9] PREDICT DBH + HEIGHT
# ============================================================

print("\n[4/9] Predicting DBH and Height...")

X_full_raw = dbh_height[
    FEATURE_COLUMNS
].copy()

X_full_scaled = feature_scaler.transform(
    X_full_raw
)

X_full = torch.tensor(
    X_full_scaled,
    dtype=torch.float32,
    device=DEVICE,
)

with torch.no_grad():

    dbh_scaled_pred, height_scaled_pred, _ = model(
        X_full
    )

dbh_pred = (
    dbh_scaled_pred.numpy().reshape(-1)
    * dbh_scale
    + dbh_mean
)

height_pred = (
    height_scaled_pred.numpy().reshape(-1)
    * height_scale
    + height_mean
)

dbh_height_result = dbh_height.copy()

dbh_height_result["dbh_pred"] = dbh_pred
dbh_height_result["height_pred"] = height_pred

print("DBH prediction complete.")
print("Height prediction complete.")


# ============================================================
# [5/9] PREDICT VOLUME
# ============================================================

print("\n[5/9] Predicting Volume...")

X_volume_raw = volume[
    FEATURE_COLUMNS
].copy()

X_volume_scaled = feature_scaler.transform(
    X_volume_raw
)

X_volume = torch.tensor(
    X_volume_scaled,
    dtype=torch.float32,
    device=DEVICE,
)

with torch.no_grad():

    _, _, volume_scaled_pred = model(
        X_volume
    )

volume_pred = (
    volume_scaled_pred.numpy().reshape(-1)
    * volume_scale
    + volume_mean
)

volume_result = volume.copy()

volume_result["volume_pred"] = volume_pred

print("Volume prediction complete.")


# ============================================================
# [6/9] AGE-LEVEL TRAJECTORY
# ============================================================

print("\n[6/9] Building age-level trajectory...")

dbh_age = (
    dbh_height_result
    .groupby("age_months")
    .agg(
        dbh_actual=("dbh_cm", "mean"),
        dbh_pred=("dbh_pred", "mean"),
        height_actual=("height_m", "mean"),
        height_pred=("height_pred", "mean"),
        dbh_mae=(
            "dbh_pred",
            lambda x: np.nan,
        ),
    )
)

# Calculate MAE separately.
dbh_mae = (
    dbh_height_result
    .assign(
        dbh_abs_error=lambda d:
            np.abs(d["dbh_cm"] - d["dbh_pred"])
    )
    .groupby("age_months")["dbh_abs_error"]
    .mean()
)

height_mae = (
    dbh_height_result
    .assign(
        height_abs_error=lambda d:
            np.abs(d["height_m"] - d["height_pred"])
    )
    .groupby("age_months")["height_abs_error"]
    .mean()
)

volume_age = (
    volume_result
    .assign(
        volume_abs_error=lambda d:
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
            "volume_abs_error",
            "mean",
        ),
        negative_volume_count=(
            "volume_pred",
            lambda x:
                int((x < 0).sum()),
        ),
    )
)

trajectory = (
    dbh_age[
        [
            "dbh_actual",
            "dbh_pred",
            "height_actual",
            "height_pred",
        ]
    ]
    .join(
        volume_age[
            [
                "volume_actual",
                "volume_pred",
                "volume_mae",
                "negative_volume_count",
            ]
        ],
        how="outer",
    )
    .reset_index()
)

trajectory["period"] = (
    trajectory["age_months"]
    .map(AGE_LABELS)
)

trajectory["dbh_mae"] = (
    trajectory["age_months"]
    .map(dbh_mae)
)

trajectory["height_mae"] = (
    trajectory["age_months"]
    .map(height_mae)
)

trajectory["dbh_error"] = (
    trajectory["dbh_pred"]
    - trajectory["dbh_actual"]
)

trajectory["height_error"] = (
    trajectory["height_pred"]
    - trajectory["height_actual"]
)

trajectory["volume_error"] = (
    trajectory["volume_pred"]
    - trajectory["volume_actual"]
)

trajectory = trajectory[
    [
        "age_months",
        "period",
        "dbh_actual",
        "dbh_pred",
        "dbh_mae",
        "height_actual",
        "height_pred",
        "height_mae",
        "volume_actual",
        "volume_pred",
        "volume_mae",
        "dbh_error",
        "height_error",
        "volume_error",
        "negative_volume_count",
    ]
].sort_values(
    "age_months"
)

print(
    trajectory.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# [7/9] TRAJECTORY LOGIC CHECKS
# ============================================================

print("\n[7/9] Checking growth trajectory logic...")

dbh_pred_values = (
    trajectory[
        "dbh_pred"
    ]
    .dropna()
    .to_numpy()
)

height_pred_values = (
    trajectory[
        "height_pred"
    ]
    .dropna()
    .to_numpy()
)

volume_pred_values = (
    trajectory[
        "volume_pred"
    ]
    .dropna()
    .to_numpy()
)

dbh_pred_changes = np.diff(
    dbh_pred_values
)

height_pred_changes = np.diff(
    height_pred_values
)

volume_pred_changes = np.diff(
    volume_pred_values
)

dbh_decreases = int(
    np.sum(dbh_pred_changes < 0)
)

height_decreases = int(
    np.sum(height_pred_changes < 0)
)

volume_decreases = int(
    np.sum(volume_pred_changes < 0)
)

dbh_stagnations = int(
    np.sum(np.isclose(dbh_pred_changes, 0))
)

height_stagnations = int(
    np.sum(np.isclose(height_pred_changes, 0))
)

negative_volume_total = int(
    (volume_result["volume_pred"] < 0).sum()
)

negative_volume_rate = (
    negative_volume_total
    / len(volume_result)
)

print(
    f"DBH predicted decreases     : "
    f"{dbh_decreases}"
)

print(
    f"Height predicted decreases  : "
    f"{height_decreases}"
)

print(
    f"Volume predicted decreases  : "
    f"{volume_decreases}"
)

print(
    f"DBH predicted stagnations   : "
    f"{dbh_stagnations}"
)

print(
    f"Height predicted stagnations: "
    f"{height_stagnations}"
)

print(
    f"Negative volume predictions : "
    f"{negative_volume_total:,}"
)

print(
    f"Negative volume rate        : "
    f"{negative_volume_rate * 100:.4f}%"
)


# ============================================================
# GROWTH RATE CHECK
# ============================================================

def mean_annualized_growth(
    values,
    ages,
    start_age,
    end_age,
):

    start_value = float(
        values.loc[
            ages == start_age
        ].iloc[0]
    )

    end_value = float(
        values.loc[
            ages == end_age
        ].iloc[0]
    )

    years = (
        end_age - start_age
    ) / 12.0

    return (
        end_value - start_value
    ) / years


age_series = trajectory[
    "age_months"
]

dbh_series = trajectory[
    "dbh_pred"
]

height_series = trajectory[
    "height_pred"
]

# Early: P6 -> P30
dbh_early = mean_annualized_growth(
    dbh_series,
    age_series,
    6,
    30,
)

height_early = mean_annualized_growth(
    height_series,
    age_series,
    6,
    30,
)

# Middle: P36 -> P48
dbh_middle = mean_annualized_growth(
    dbh_series,
    age_series,
    36,
    48,
)

height_middle = mean_annualized_growth(
    height_series,
    age_series,
    36,
    48,
)

# Late: P54 -> P66
dbh_late = mean_annualized_growth(
    dbh_series,
    age_series,
    54,
    66,
)

height_late = mean_annualized_growth(
    height_series,
    age_series,
    54,
    66,
)

print("\nPredicted growth rates:")

print(
    f"DBH    Early : {dbh_early:.6f} cm/year"
)

print(
    f"DBH    Middle: {dbh_middle:.6f} cm/year"
)

print(
    f"DBH    Late  : {dbh_late:.6f} cm/year"
)

print(
    f"Height Early : {height_early:.6f} m/year"
)

print(
    f"Height Middle: {height_middle:.6f} m/year"
)

print(
    f"Height Late  : {height_late:.6f} m/year"
)

dbh_shape_pass = (
    dbh_early > dbh_middle
    and dbh_middle > dbh_late
)

height_shape_pass = (
    height_early > height_middle
    and height_middle > height_late
)

dbh_monotonic_pass = (
    dbh_decreases == 0
)

height_monotonic_pass = (
    height_decreases == 0
)

print(
    f"\nDBH early > middle > late : "
    f"{'PASS' if dbh_shape_pass else 'FAIL'}"
)

print(
    f"Height early > middle > late: "
    f"{'PASS' if height_shape_pass else 'FAIL'}"
)

print(
    f"DBH no predicted decrease  : "
    f"{'PASS' if dbh_monotonic_pass else 'FAIL'}"
)

print(
    f"Height no predicted decrease: "
    f"{'PASS' if height_monotonic_pass else 'FAIL'}"
)


# ============================================================
# [8/9] PHI CHECK
# ============================================================

print("\n[8/9] Checking P66 / PHI...")

phi = trajectory[
    trajectory["age_months"] == 66
]

if phi.empty:

    raise RuntimeError(
        "P66 / PHI trajectory is missing."
    )

phi_row = phi.iloc[0]

print(
    f"DBH    actual={phi_row['dbh_actual']:.6f} "
    f"pred={phi_row['dbh_pred']:.6f} "
    f"MAE={phi_row['dbh_mae']:.6f}"
)

print(
    f"Height actual={phi_row['height_actual']:.6f} "
    f"pred={phi_row['height_pred']:.6f} "
    f"MAE={phi_row['height_mae']:.6f}"
)

print(
    f"Volume actual={phi_row['volume_actual']:.6f} "
    f"pred={phi_row['volume_pred']:.6f} "
    f"MAE={phi_row['volume_mae']:.6f}"
)

print(
    f"Negative P66 volume count="
    f"{int(phi_row['negative_volume_count'])}"
)


# ============================================================
# [9/9] SAVE RESULTS
# ============================================================

print("\n[9/9] Saving inspection results...")

trajectory.to_csv(
    TRAJECTORY_PATH,
    index=False,
)

report = {

    "status": "INSPECTED",

    "checkpoint": str(
        MODEL_PATH
    ),

    "checkpoint_epoch": int(
        checkpoint["epoch"]
    ),

    "checkpoint_loss": float(
        checkpoint["loss_total"]
    ),

    "data": {
        "dbh_height_rows": int(
            len(dbh_height)
        ),
        "volume_rows": int(
            len(volume)
        ),
        "dbh_height_age_range": [
            int(dbh_height["age_months"].min()),
            int(dbh_height["age_months"].max()),
        ],
        "volume_age_range": [
            int(volume["age_months"].min()),
            int(volume["age_months"].max()),
        ],
    },

    "trajectory_logic": {

        "dbh_predicted_decreases": dbh_decreases,

        "height_predicted_decreases":
            height_decreases,

        "volume_predicted_decreases":
            volume_decreases,

        "dbh_predicted_stagnations":
            dbh_stagnations,

        "height_predicted_stagnations":
            height_stagnations,

        "dbh_early_growth_cm_per_year":
            float(dbh_early),

        "dbh_middle_growth_cm_per_year":
            float(dbh_middle),

        "dbh_late_growth_cm_per_year":
            float(dbh_late),

        "height_early_growth_m_per_year":
            float(height_early),

        "height_middle_growth_m_per_year":
            float(height_middle),

        "height_late_growth_m_per_year":
            float(height_late),

        "dbh_shape_pass":
            bool(dbh_shape_pass),

        "height_shape_pass":
            bool(height_shape_pass),

        "dbh_monotonic_pass":
            bool(dbh_monotonic_pass),

        "height_monotonic_pass":
            bool(height_monotonic_pass),
    },

    "volume_positivity": {

        "negative_count":
            negative_volume_total,

        "total_predictions":
            int(len(volume_result)),

        "negative_rate":
            float(negative_volume_rate),
    },

    "phi66": {

        "dbh_actual":
            float(phi_row["dbh_actual"]),

        "dbh_pred":
            float(phi_row["dbh_pred"]),

        "dbh_mae":
            float(phi_row["dbh_mae"]),

        "height_actual":
            float(phi_row["height_actual"]),

        "height_pred":
            float(phi_row["height_pred"]),

        "height_mae":
            float(phi_row["height_mae"]),

        "volume_actual":
            float(phi_row["volume_actual"]),

        "volume_pred":
            float(phi_row["volume_pred"]),

        "volume_mae":
            float(phi_row["volume_mae"]),

        "negative_volume_count":
            int(phi_row["negative_volume_count"]),
    },

    "note": (
        "Trajectory inspection only. No training, "
        "dataset modification, or checkpoint modification "
        "was performed."
    ),
}

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STEP 54 TRAJECTORY INSPECTION COMPLETE")
print("=" * 70)

print(
    f"Trajectory file: "
    f"{TRAJECTORY_PATH}"
)

print(
    f"Report file    : "
    f"{REPORT_PATH}"
)

print("=" * 70)