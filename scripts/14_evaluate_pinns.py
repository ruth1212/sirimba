from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# TREE GROWTH ANALYTICS
# STEP 8 - EVALUATE PINNs
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "model_test.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "pinns_models"
    / "best_pinns_model.pt"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

METRICS_PATH = (
    RESULTS_DIR
    / "pinns_metrics.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIR
    / "pinns_predictions.csv"
)


# ============================================================
# FEATURES AND TARGETS
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


# ============================================================
# MODEL
# ============================================================

class TreeGrowthPINN(nn.Module):

    def __init__(
        self,
        input_size=6,
        hidden_size=64,
        hidden_layers=3,
        output_size=3,
    ):
        super().__init__()

        layers = []

        layers.append(
            nn.Linear(
                input_size,
                hidden_size,
            )
        )

        layers.append(
            nn.Tanh()
        )

        for _ in range(hidden_layers - 1):

            layers.append(
                nn.Linear(
                    hidden_size,
                    hidden_size,
                )
            )

            layers.append(
                nn.Tanh()
            )

        layers.append(
            nn.Linear(
                hidden_size,
                output_size,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# SAFE MAPE
# ============================================================

def safe_mape(
    y_true,
    y_pred,
):
    """
    MAPE dengan mengabaikan actual value = 0.

    Jika actual = 0, baris tersebut tidak digunakan
    dalam perhitungan MAPE sehingga tidak menghasilkan
    infinite/undefined value.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0

    if not np.any(mask):
        return np.nan

    return (
        np.mean(
            np.abs(
                (
                    y_true[mask]
                    - y_pred[mask]
                )
                / y_true[mask]
            )
        )
        * 100
    )


# ============================================================
# START
# ============================================================

print("=" * 75)
print("TREE GROWTH ANALYTICS")
print("STEP 8 - EVALUATE PINNs")
print("=" * 75)


# ============================================================
# 1. DEVICE
# ============================================================

print("\n[1/7] Selecting device...")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Device : {device}"
)


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("\n[2/7] Loading test data...")

test_df = pd.read_csv(
    TEST_PATH
)

print(
    f"Test rows : "
    f"{len(test_df):,}"
)

print(
    f"Test groups : "
    f"{test_df['compartment_id'].nunique():,}"
)


# ============================================================
# 3. LOAD MODEL CHECKPOINT
# ============================================================

print("\n[3/7] Loading PINNs model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)

print(
    f"Best epoch : "
    f"{checkpoint['epoch']}"
)

print(
    f"Training best loss : "
    f"{checkpoint['best_loss']:.8f}"
)

print(
    f"Physics C : "
    f"{checkpoint['C']:.12f}"
)

print(
    f"Physics loss mode : "
    f"{checkpoint.get('physics_loss_mode', 'not recorded')}"
)


# ============================================================
# 4. PREPARE TEST TENSORS
# ============================================================

print("\n[4/7] Preparing test tensors...")

X_test_df = test_df[
    FEATURES
].astype(
    np.float32
)

Y_test_df = test_df[
    TARGETS
].astype(
    np.float32
)


# ------------------------------------------------------------
# Retrieve training normalization statistics
# from the saved PINNs checkpoint
# ------------------------------------------------------------

X_mean = np.asarray(
    checkpoint["X_mean"],
    dtype=np.float32,
)

X_std = np.asarray(
    checkpoint["X_std"],
    dtype=np.float32,
)

Y_mean = np.asarray(
    checkpoint["Y_mean"],
    dtype=np.float32,
)

Y_std = np.asarray(
    checkpoint["Y_std"],
    dtype=np.float32,
)


# ------------------------------------------------------------
# Normalize test features using TRAINING statistics
# ------------------------------------------------------------

X_test_normalized = (
    X_test_df.values
    - X_mean
) / X_std


X_tensor = torch.tensor(
    X_test_normalized,
    dtype=torch.float32,
).to(device)


# ============================================================
# 5. BUILD MODEL AND PREDICT
# ============================================================

print("\n[5/7] Running PINNs prediction...")

model = TreeGrowthPINN(
    input_size=len(FEATURES),
    hidden_size=64,
    hidden_layers=3,
    output_size=len(TARGETS),
).to(device)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


with torch.no_grad():

    predictions_normalized = model(
        X_tensor
    )


# ------------------------------------------------------------
# Denormalize predictions
# ------------------------------------------------------------

predictions_normalized = (
    predictions_normalized
    .cpu()
    .numpy()
)


predictions_raw = (
    predictions_normalized
    * Y_std
    + Y_mean
)


print(
    f"Prediction rows : "
    f"{len(predictions_raw):,}"
)


# ============================================================
# 6. CALCULATE METRICS
# ============================================================

print("\n[6/7] Calculating evaluation metrics...")

metrics = []

prediction_output = test_df[
    [
        "compartment_id",
        "age_months",
        "inventory_year",
        "clone_factor",
        "planting_year",
        "disease_score",
        "trees_per_ha",
    ]
].copy()


for index, target in enumerate(TARGETS):

    y_true = Y_test_df[
        target
    ].values

    y_pred = predictions_raw[
        :, index
    ]

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    mape = safe_mape(
        y_true,
        y_pred,
    )

    metrics.append(
        {
            "model": "PINNs",
            "target": target,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE_percent": mape,
            "test_rows": len(y_true),
        }
    )

    prediction_output[
        f"actual_{target}"
    ] = y_true

    prediction_output[
        f"predicted_pinns_{target}"
    ] = y_pred


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(
    metrics
)

metrics_df.to_csv(
    METRICS_PATH,
    index=False,
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_output.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 75)
print("PINNs TEST EVALUATION RESULTS")
print("=" * 75)

for row in metrics:

    print(
        f"\nTarget : {row['target']}"
    )

    print(
        f"  MAE  : "
        f"{row['MAE']:.6f}"
    )

    print(
        f"  RMSE : "
        f"{row['RMSE']:.6f}"
    )

    print(
        f"  R²   : "
        f"{row['R2']:.6f}"
    )

    print(
        f"  MAPE : "
        f"{row['MAPE_percent']:.4f}%"
    )


print("\n" + "=" * 75)
print("PINNs EVALUATION COMPLETED")
print("=" * 75)

print(
    f"Test rows : "
    f"{len(test_df):,}"
)

print("\nSaved:")

print(
    "  - results/pinns_metrics.csv"
)

print(
    "  - results/pinns_predictions.csv"
)

print("=" * 75)