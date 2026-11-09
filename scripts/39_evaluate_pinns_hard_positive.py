import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# TREE GROWTH ANALYTICS
# STEP 21O - EVALUATE HARD-POSITIVE PINNs
# ============================================================

print("=" * 75)
print("TREE GROWTH ANALYTICS")
print("STEP 21O - EVALUATE HARD-POSITIVE PINNs")
print("=" * 75)


# ============================================================
# CONFIGURATION
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

TEST_FILE = "results/model_test.csv"

CHECKPOINT_FILE = (
    "results/pinns_models/"
    "best_pinns_hard_positive_model.pt"
)

OUTPUT_METRICS = (
    "results/pinns_hard_positive_metrics.csv"
)

OUTPUT_PREDICTIONS = (
    "results/pinns_hard_positive_predictions.csv"
)

OUTPUT_CONFIG = (
    "results/pinns_hard_positive_evaluation_config.json"
)


# ============================================================
# DEVICE
# ============================================================

print("\n[1/8] Checking device...")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device :", DEVICE)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n[2/8] Loading test data...")

df = pd.read_csv(TEST_FILE)

print("Test rows :", len(df))

missing_features = [
    col for col in FEATURES
    if col not in df.columns
]

missing_targets = [
    col for col in TARGETS
    if col not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing features: {missing_features}"
    )

if missing_targets:
    raise ValueError(
        f"Missing targets: {missing_targets}"
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\n[3/8] Loading hard-positive checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=DEVICE,
    weights_only=False
)

print("Checkpoint loaded.")
print(
    "Best epoch :",
    checkpoint["best_epoch"]
)

print(
    "Best loss  :",
    checkpoint["best_loss"]
)


# ============================================================
# NORMALIZATION
# ============================================================

X_mean = np.asarray(
    checkpoint["X_mean"],
    dtype=np.float32
)

X_std = np.asarray(
    checkpoint["X_std"],
    dtype=np.float32
)

Y_mean = np.asarray(
    checkpoint["Y_mean"],
    dtype=np.float32
)

Y_std = np.asarray(
    checkpoint["Y_std"],
    dtype=np.float32
)

C = float(
    checkpoint["C"]
)


# ============================================================
# MODEL
# ============================================================

class TreeGrowthPINNHardPositive(nn.Module):

    def __init__(
        self,
        input_dim,
        volume_mean,
        volume_std
    ):
        super().__init__()

        self.volume_mean = float(volume_mean)
        self.volume_std = float(volume_std)

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 3)
        )

        self.softplus = nn.Softplus()

    def forward(self, x):

        raw_output = self.network(x)

        dbh_normalized = (
            raw_output[:, 0:1]
        )

        height_normalized = (
            raw_output[:, 1:2]
        )

        volume_raw = self.softplus(
            raw_output[:, 2:3]
        )

        volume_normalized = (
            volume_raw - self.volume_mean
        ) / self.volume_std

        return torch.cat(
            [
                dbh_normalized,
                height_normalized,
                volume_normalized
            ],
            dim=1
        )


model = TreeGrowthPINNHardPositive(
    input_dim=len(FEATURES),
    volume_mean=Y_mean[2],
    volume_std=Y_std[2]
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# PREPARE TEST DATA
# ============================================================

print("\n[4/8] Preparing test data...")

X = df[FEATURES].astype(
    np.float32
).values

Y_true = df[TARGETS].astype(
    np.float32
).values

X_norm = (
    X - X_mean
) / X_std

X_tensor = torch.tensor(
    X_norm,
    dtype=torch.float32,
    device=DEVICE
)


# ============================================================
# PREDICTION
# ============================================================

print("\n[5/8] Running predictions...")

with torch.no_grad():

    prediction_norm = model(
        X_tensor
    )

    prediction_raw = (
        prediction_norm.cpu().numpy()
        * Y_std
        + Y_mean
    )

pred_dbh = prediction_raw[:, 0]
pred_height = prediction_raw[:, 1]
pred_volume = prediction_raw[:, 2]

print(
    "Prediction rows :",
    len(prediction_raw)
)


# ============================================================
# POSITIVITY CHECK
# ============================================================

print("\n[6/8] Checking physical volume positivity...")

negative_mask = pred_volume < 0

negative_count = int(
    negative_mask.sum()
)

zero_count = int(
    (pred_volume == 0).sum()
)

positive_count = int(
    (pred_volume > 0).sum()
)

print(
    "Positive volume :",
    positive_count
)

print(
    "Zero volume     :",
    zero_count
)

print(
    "Negative volume :",
    negative_count
)

print(
    "Minimum volume  :",
    f"{pred_volume.min():.12f}"
)

print(
    "Maximum volume  :",
    f"{pred_volume.max():.12f}"
)

if negative_count == 0:
    print(
        "HARD PHYSICAL POSITIVITY CHECK : PASSED"
    )
else:
    print(
        "HARD PHYSICAL POSITIVITY CHECK : FAILED"
    )


# ============================================================
# METRICS FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    nonzero = (
        np.abs(y_true) > 1e-8
    )

    if nonzero.any():

        mape = np.mean(
            np.abs(
                (
                    y_true[nonzero]
                    - y_pred[nonzero]
                )
                / y_true[nonzero]
            )
        ) * 100

    else:

        mape = np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape_percent": mape
    }


# ============================================================
# GLOBAL METRICS
# ============================================================

metrics_rows = []

for i, target in enumerate(TARGETS):

    result = calculate_metrics(
        Y_true[:, i],
        prediction_raw[:, i]
    )

    metrics_rows.append({
        "model": "PINNs_Hard_Positive",
        "target": target,
        "n_test": len(Y_true),
        "mae": result["mae"],
        "rmse": result["rmse"],
        "r2": result["r2"],
        "mape_percent": result["mape_percent"],
        "negative_volume_count": (
            negative_count
            if target == "volume_m3_tree"
            else np.nan
        )
    })


# ============================================================
# P6 METRICS
# ============================================================

if "age_months" in df.columns:

    p6_mask = (
        df["age_months"].values
        == 6
    )

    p6_count = int(
        p6_mask.sum()
    )

    print("\nP6 test rows :", p6_count)

    for i, target in enumerate(TARGETS):

        p6_true = Y_true[
            p6_mask,
            i
        ]

        p6_pred = prediction_raw[
            p6_mask,
            i
        ]

        result = calculate_metrics(
            p6_true,
            p6_pred
        )

        metrics_rows.append({
            "model": "PINNs_Hard_Positive",
            "target": target,
            "n_test": p6_count,
            "mae": result["mae"],
            "rmse": result["rmse"],
            "r2": result["r2"],
            "mape_percent": result["mape_percent"],
            "negative_volume_count": (
                int(
                    (
                        p6_pred < 0
                    ).sum()
                )
                if target == "volume_m3_tree"
                else np.nan
            )
        })


# ============================================================
# SAVE PREDICTIONS
# ============================================================

print("\n[7/8] Saving predictions...")

predictions = df.copy()

predictions[
    "predicted_pinns_hard_positive_dbh_cm"
] = pred_dbh

predictions[
    "predicted_pinns_hard_positive_height_m"
] = pred_height

predictions[
    "predicted_pinns_hard_positive_volume_m3_tree"
] = pred_volume

predictions[
    "predicted_pinns_hard_positive_volume_positive"
] = (
    pred_volume > 0
)

predictions.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)

print(
    "Saved:",
    OUTPUT_PREDICTIONS
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(
    metrics_rows
)

metrics_df.to_csv(
    OUTPUT_METRICS,
    index=False
)

print(
    "Saved:",
    OUTPUT_METRICS
)


# ============================================================
# SAVE EVALUATION CONFIG
# ============================================================

evaluation_config = {

    "model":
        "PINNs_Hard_Positive",

    "checkpoint":
        CHECKPOINT_FILE,

    "test_file":
        TEST_FILE,

    "test_rows":
        len(df),

    "features":
        FEATURES,

    "targets":
        TARGETS,

    "best_epoch":
        int(
            checkpoint["best_epoch"]
        ),

    "best_loss":
        float(
            checkpoint["best_loss"]
        ),

    "physics_constant_C":
        C,

    "negative_volume_count":
        negative_count,

    "zero_volume_count":
        zero_count,

    "positive_volume_count":
        positive_count,

    "minimum_volume":
        float(pred_volume.min()),

    "maximum_volume":
        float(pred_volume.max()),

    "physical_positivity_passed":
        bool(
            negative_count == 0
        )
}

with open(
    OUTPUT_CONFIG,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        evaluation_config,
        f,
        indent=2
    )

print(
    "Saved:",
    OUTPUT_CONFIG
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 75)
print("GLOBAL EVALUATION RESULTS")
print("=" * 75)

global_metrics = metrics_df[
    metrics_df["n_test"] == len(df)
]

for _, row in global_metrics.iterrows():

    print(
        f"\n{row['target']}"
    )

    print(
        f"  MAE  : {row['mae']:.6f}"
    )

    print(
        f"  RMSE : {row['rmse']:.6f}"
    )

    print(
        f"  R2   : {row['r2']:.6f}"
    )

    print(
        f"  MAPE : {row['mape_percent']:.6f}%"
    )


print("\n" + "=" * 75)
print("P6 EVALUATION RESULTS")
print("=" * 75)

p6_metrics = metrics_df[
    metrics_df["n_test"] != len(df)
]

for _, row in p6_metrics.iterrows():

    print(
        f"\n{row['target']}"
    )

    print(
        f"  MAE  : {row['mae']:.6f}"
    )

    print(
        f"  RMSE : {row['rmse']:.6f}"
    )

    print(
        f"  R2   : {row['r2']:.6f}"
    )

    print(
        f"  MAPE : {row['mape_percent']:.6f}%"
    )


print("\n" + "=" * 75)
print("STEP 21O COMPLETED")
print("=" * 75)

print(
    "Test rows        :",
    len(df)
)

print(
    "Negative volume  :",
    negative_count
)

print(
    "Minimum volume   :",
    f"{pred_volume.min():.12f}"
)

print(
    "Maximum volume   :",
    f"{pred_volume.max():.12f}"
)

print("=" * 75)