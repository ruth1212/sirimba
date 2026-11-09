from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# STEP 49 - EVALUATE PINNs ON OFFICIAL TEST SET
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = (
    ROOT
    / "data"
    / "model_ready"
    / "test.csv"
)

PREPROCESSING_DIR = (
    ROOT
    / "data"
    / "model_ready"
    / "pipeline"
    / "preprocessing"
)

MODEL_PATH = (
    ROOT
    / "results"
    / "pinns"
    / "models"
    / "pinns_growth_model.pt"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "pinns"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PROJECT DEFINITION
# ============================================================

FEATURES = [
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha",
]

TARGETS = [
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]

PHI_AGE = 66
VOLUME_MIN_AGE = 18
VOLUME_MAX_AGE = 66


# ============================================================
# MODEL
# ============================================================


class TreeGrowthPINNs(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden = nn.Sequential(
            nn.Linear(6, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),
        )

        self.dbh_head = nn.Linear(64, 1)

        self.height_head = nn.Linear(
            64,
            1,
        )

        self.volume_head = nn.Linear(
            64,
            1,
        )

        self.volume_activation = nn.Softplus()

    def forward(self, x):

        hidden = self.hidden(x)

        dbh = self.dbh_head(hidden)

        height = self.height_head(
            hidden
        )

        volume_raw = self.volume_head(
            hidden
        )

        volume = self.volume_activation(
            volume_raw
        )

        return torch.cat(
            [
                dbh,
                height,
                volume,
            ],
            dim=1,
        )


# ============================================================
# METRIC FUNCTIONS
# ============================================================


def calculate_metrics(
    actual,
    predicted,
):
    actual = np.asarray(
        actual,
        dtype=np.float64,
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    error = predicted - actual

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    ss_res = np.sum(
        (actual - predicted) ** 2
    )

    ss_tot = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1.0 - (
            ss_res / ss_tot
        )

    nonzero = actual != 0

    if nonzero.sum() == 0:
        mape = np.nan
    else:
        mape = np.mean(
            np.abs(
                (
                    actual[nonzero]
                    - predicted[nonzero]
                )
                / actual[nonzero]
            )
        ) * 100.0

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE_percent": float(mape),
    }


# ============================================================
# START
# ============================================================

print("=" * 80)
print("STEP 49 - PINNs TEST SET EVALUATION")
print("=" * 80)

print("""
Evaluation only.

NO RETRAINING
NO DATASET MODIFICATION
NO RF MODEL MODIFICATION

Evaluation windows:
    DBH     : P6-P66
    Height  : P6-P66
    Volume  : P18-P66
    PHI     : P66
""")


# ============================================================
# 1. CHECK FILES
# ============================================================

print("\n[1/8] Checking required files...")

for path in [
    TEST_PATH,
    MODEL_PATH,
    PREPROCESSING_DIR
    / "feature_scaler.joblib",
    PREPROCESSING_DIR
    / "target_scaler.joblib",
]:

    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    print(
        f"FOUND: {path.name}"
    )

print("Required files: PASS")


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("\n[2/8] Loading official test set...")

test = pd.read_csv(
    TEST_PATH
)

print(
    f"Test rows: {len(test):,}"
)

missing = [
    col
    for col in FEATURES + TARGETS
    if col not in test.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

expected_ages = list(
    range(6, 67, 6)
)

actual_ages = sorted(
    test["age_months"]
    .unique()
    .tolist()
)

print(
    f"Test ages: {actual_ages}"
)

if actual_ages != expected_ages:
    raise ValueError(
        "Unexpected test age structure."
    )

print("Test dataset: PASS")


# ============================================================
# 3. LOAD SCALERS
# ============================================================

print("\n[3/8] Loading preprocessing artifacts...")

feature_scaler = joblib.load(
    PREPROCESSING_DIR
    / "feature_scaler.joblib"
)

target_scalers = joblib.load(
    PREPROCESSING_DIR
    / "target_scaler.joblib"
)

required_scalers = [
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]

for target in required_scalers:

    if target not in target_scalers:
        raise ValueError(
            f"Missing target scaler: {target}"
        )

print("Feature scaler: PASS")
print("Target scalers: PASS")


# ============================================================
# 4. LOAD MODEL
# ============================================================

print("\n[4/8] Loading trained PINNs model...")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Device: {device}"
)

model = TreeGrowthPINNs().to(
    device
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)

if "model_state_dict" not in checkpoint:
    raise ValueError(
        "Checkpoint does not contain "
        "'model_state_dict'."
    )

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Checkpoint loaded successfully."
)

print(
    f"Checkpoint architecture: "
    f"{checkpoint.get('architecture', {})}"
)

print("Model loading: PASS")


# ============================================================
# 5. RUN PREDICTION
# ============================================================

print("\n[5/8] Running PINNs prediction...")

X_scaled = feature_scaler.transform(
    test[FEATURES]
).astype(
    np.float32
)

X_tensor = torch.tensor(
    X_scaled,
    dtype=torch.float32,
    device=device,
)

with torch.no_grad():

    prediction_scaled = model(
        X_tensor
    )

prediction_scaled = (
    prediction_scaled
    .cpu()
    .numpy()
)


# ------------------------------------------------------------
# INVERSE TRANSFORM EACH TARGET
# ------------------------------------------------------------

pred_dbh = (
    prediction_scaled[:, 0]
    * float(
        target_scalers[
            "dbh_cm"
        ].scale_[0]
    )
    + float(
        target_scalers[
            "dbh_cm"
        ].mean_[0]
    )
)

pred_height = (
    prediction_scaled[:, 1]
    * float(
        target_scalers[
            "height_m"
        ].scale_[0]
    )
    + float(
        target_scalers[
            "height_m"
        ].mean_[0]
    )
)

pred_volume = (
    prediction_scaled[:, 2]
    * float(
        target_scalers[
            "volume_m3_tree"
        ].scale_[0]
    )
    + float(
        target_scalers[
            "volume_m3_tree"
        ].mean_[0]
    )
)

print(
    f"Predictions generated: "
    f"{len(pred_dbh):,}"
)

if not np.isfinite(
    pred_dbh
).all():
    raise ValueError(
        "DBH predictions contain NaN/Inf."
    )

if not np.isfinite(
    pred_height
).all():
    raise ValueError(
        "Height predictions contain NaN/Inf."
    )

if not np.isfinite(
    pred_volume
).all():
    raise ValueError(
        "Volume predictions contain NaN/Inf."
    )

print("Prediction generation: PASS")


# ============================================================
# 6. EVALUATE TARGETS
# ============================================================

print("\n[6/8] Calculating test metrics...")

actual_dbh = test[
    "dbh_cm"
].to_numpy(
    dtype=np.float64
)

actual_height = test[
    "height_m"
].to_numpy(
    dtype=np.float64
)

volume_mask = (
    test["age_months"]
    .between(
        VOLUME_MIN_AGE,
        VOLUME_MAX_AGE,
    )
    .to_numpy()
)

actual_volume = test.loc[
    volume_mask,
    "volume_m3_tree"
].to_numpy(
    dtype=np.float64
)

pred_volume_eval = pred_volume[
    volume_mask
]


metrics_dbh = calculate_metrics(
    actual_dbh,
    pred_dbh,
)

metrics_height = calculate_metrics(
    actual_height,
    pred_height,
)

metrics_volume = calculate_metrics(
    actual_volume,
    pred_volume_eval,
)


metrics_rows = [
    {
        "model": "PINNs",
        "target": "dbh_cm",
        "evaluation_window": "P6-P66",
        **metrics_dbh,
    },
    {
        "model": "PINNs",
        "target": "height_m",
        "evaluation_window": "P6-P66",
        **metrics_height,
    },
    {
        "model": "PINNs",
        "target": "volume_m3_tree",
        "evaluation_window": "P18-P66",
        **metrics_volume,
    },
]

metrics_df = pd.DataFrame(
    metrics_rows
)

print("\n" + "-" * 80)
print("PINNs TEST METRICS")
print("-" * 80)

print(
    metrics_df.to_string(
        index=False
    )
)


# ============================================================
# POSITIVITY CHECK
# ============================================================

negative_volume_count = int(
    np.sum(
        pred_volume_eval < 0
    )
)

zero_volume_count = int(
    np.sum(
        pred_volume_eval == 0
    )
)

print("\nVolume positivity:")
print(
    f"Negative predictions: "
    f"{negative_volume_count:,}"
)

print(
    f"Zero predictions    : "
    f"{zero_volume_count:,}"
)

if negative_volume_count > 0:
    raise ValueError(
        "PINNs produced negative volume predictions."
    )

print("Positive volume: PASS")


# ============================================================
# P66 / PHI
# ============================================================

print("\n[7/8] Checking P66 / PHI endpoint...")

phi_mask = (
    test["age_months"]
    == PHI_AGE
).to_numpy()

if phi_mask.sum() == 0:
    raise ValueError(
        "No P66 observations found."
    )

phi_rows = [
    {
        "model": "PINNs",
        "target": "dbh_cm",
        "actual_mean": float(
            actual_dbh[phi_mask].mean()
        ),
        "predicted_mean": float(
            pred_dbh[phi_mask].mean()
        ),
        "absolute_error": float(
            np.abs(
                actual_dbh[phi_mask]
                - pred_dbh[phi_mask]
            ).mean()
        ),
        "n": int(
            phi_mask.sum()
        ),
    },
    {
        "model": "PINNs",
        "target": "height_m",
        "actual_mean": float(
            actual_height[phi_mask].mean()
        ),
        "predicted_mean": float(
            pred_height[phi_mask].mean()
        ),
        "absolute_error": float(
            np.abs(
                actual_height[phi_mask]
                - pred_height[phi_mask]
            ).mean()
        ),
        "n": int(
            phi_mask.sum()
        ),
    },
    {
        "model": "PINNs",
        "target": "volume_m3_tree",
        "actual_mean": float(
            actual_volume[
                test.loc[
                    volume_mask
                ]["age_months"].to_numpy()
                == PHI_AGE
            ].mean()
        ),
        "predicted_mean": float(
            pred_volume[
                phi_mask
            ].mean()
        ),
        "absolute_error": float(
            np.abs(
                actual_volume[
                    test.loc[
                        volume_mask
                    ]["age_months"].to_numpy()
                    == PHI_AGE
                ]
                -
                pred_volume[
                    phi_mask
                ]
            ).mean()
        ),
        "n": int(
            phi_mask.sum()
        ),
    },
]

phi_df = pd.DataFrame(
    phi_rows
)

print(
    phi_df.to_string(
        index=False
    )
)

print("P66 / PHI: PASS")


# ============================================================
# 8. SAVE RESULTS
# ============================================================

print("\n[8/8] Saving evaluation results...")

metrics_path = (
    RESULT_DIR
    / "test_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False,
)

phi_path = (
    RESULT_DIR
    / "phi_metrics.csv"
)

phi_df.to_csv(
    phi_path,
    index=False,
)


# ------------------------------------------------------------
# SAVE ROW-LEVEL PREDICTIONS
# ------------------------------------------------------------

prediction_df = test[
    [
        "regional_id",
        "regional_name",
        "compartment_id",
        "compartment_code",
        "species",
        "clone",
        "planting_year",
        "age_months",
        "growth_period",
    ]
].copy()

prediction_df[
    "actual_dbh_cm"
] = actual_dbh

prediction_df[
    "predicted_dbh_cm"
] = pred_dbh

prediction_df[
    "actual_height_m"
] = actual_height

prediction_df[
    "predicted_height_m"
] = pred_height

prediction_df[
    "actual_volume_m3_tree"
] = test[
    "volume_m3_tree"
].to_numpy()

prediction_df[
    "predicted_volume_m3_tree"
] = pred_volume

prediction_path = (
    RESULT_DIR
    / "test_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False,
)


# ------------------------------------------------------------
# SUMMARY JSON
# ------------------------------------------------------------

summary = {
    "status": "PASS",

    "test_rows": int(
        len(test)
    ),

    "volume_test_rows": int(
        volume_mask.sum()
    ),

    "evaluation_windows": {
        "dbh_cm": "P6-P66",
        "height_m": "P6-P66",
        "volume_m3_tree": "P18-P66",
    },

    "phi_age_months": PHI_AGE,

    "metrics": metrics_rows,

    "volume_positivity": {
        "negative": negative_volume_count,
        "zero": zero_volume_count,
    },

    "training_performed": False,

    "dataset_modified": False,

    "rf_model_modified": False,
}

summary_path = (
    RESULT_DIR
    / "evaluation_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )

print(
    f"Metrics saved: {metrics_path}"
)

print(
    f"PHI metrics saved: {phi_path}"
)

print(
    f"Predictions saved: {prediction_path}"
)

print(
    f"Summary saved: {summary_path}"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("STEP 49 STATUS: PASS")
print("=" * 80)

print("""
PINNs test evaluation completed.

Verified:
  - Official test set used
  - No retraining
  - DBH evaluated P6-P66
  - Height evaluated P6-P66
  - Volume evaluated P18-P66
  - MAE calculated
  - RMSE calculated
  - R2 calculated
  - MAPE calculated
  - Volume positivity checked
  - P66 / PHI checked

NO DATASET WAS MODIFIED.
NO RF MODEL WAS MODIFIED.
NO PINNs MODEL WAS RETRAINED.
""")

print("=" * 80)