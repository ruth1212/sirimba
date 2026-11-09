from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib


# ============================================================
# STEP 53 — PINNs V2 TRAINING
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PIPELINE_DIR = ROOT / "data" / "model_ready" / "pipeline"
PREPROCESSING_DIR = PIPELINE_DIR / "preprocessing"

TRAIN_DBH_HEIGHT_PATH = PIPELINE_DIR / "train_dbh_height.csv"
TRAIN_VOLUME_PATH = PIPELINE_DIR / "train_volume_p18.csv"

FEATURE_SCALER_PATH = PREPROCESSING_DIR / "feature_scaler.joblib"
TARGET_SCALER_PATH = PREPROCESSING_DIR / "target_scaler.joblib"

PHYSICS_CONFIG_PATH = (
    ROOT / "results" / "pinns" / "physics_coefficient_config.json"
)

OUTPUT_DIR = ROOT / "results" / "pinns" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "pinns_v2_growth_model.pt"
HISTORY_PATH = ROOT / "results" / "pinns" / "training_history_v2.csv"
REPORT_PATH = ROOT / "results" / "pinns" / "training_report_v2.json"

DEVICE = torch.device("cpu")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

FEATURE_COLUMNS = [
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha",
]

EPOCHS = 100
LEARNING_RATE = 0.001
PHYSICS_WEIGHT = 0.1

HIDDEN_SIZE = 64


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

        return dbh_scaled, height_scaled, volume_scaled


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 53 — PINNs V2 TRAINING")
print("=" * 70)

start_time = time.time()


# ============================================================
# [1/10] LOAD DATA
# ============================================================

print("\n[1/10] Loading training data...")

dbh_height = pd.read_csv(TRAIN_DBH_HEIGHT_PATH)
volume = pd.read_csv(TRAIN_VOLUME_PATH)

print(f"DBH/Height rows : {len(dbh_height):,}")
print(f"Volume rows     : {len(volume):,}")

if len(dbh_height) != 174960:
    raise RuntimeError(
        f"Unexpected DBH/Height rows: {len(dbh_height):,}"
    )

if len(volume) != 142560:
    raise RuntimeError(
        f"Unexpected volume rows: {len(volume):,}"
    )


# ============================================================
# [2/10] LOAD SCALERS
# ============================================================

print("\n[2/10] Loading preprocessing scalers...")

feature_scaler = joblib.load(FEATURE_SCALER_PATH)
target_scalers = joblib.load(TARGET_SCALER_PATH)

dbh_scaler = target_scalers["dbh_cm"]
height_scaler = target_scalers["height_m"]
volume_scaler = target_scalers["volume_m3_tree"]

dbh_mean = float(dbh_scaler.mean_[0])
dbh_scale = float(dbh_scaler.scale_[0])

height_mean = float(height_scaler.mean_[0])
height_scale = float(height_scaler.scale_[0])

volume_mean = float(volume_scaler.mean_[0])
volume_scale = float(volume_scaler.scale_[0])

print(
    f"DBH    mean={dbh_mean:.6f}, scale={dbh_scale:.6f}"
)
print(
    f"Height mean={height_mean:.6f}, scale={height_scale:.6f}"
)
print(
    f"Volume mean={volume_mean:.6f}, scale={volume_scale:.6f}"
)


# ============================================================
# [3/10] LOAD PHYSICS
# ============================================================

print("\n[3/10] Loading physics coefficient...")

with open(PHYSICS_CONFIG_PATH, "r", encoding="utf-8") as f:
    physics_config = json.load(f)

C_PHYSICS = float(physics_config["coefficient"])

print(f"Physics coefficient C = {C_PHYSICS:.12f}")

if physics_config["calibration_age_min_months"] != 18:
    raise RuntimeError("Physics calibration minimum age is not P18.")

if physics_config["calibration_age_max_months"] != 66:
    raise RuntimeError("Physics calibration maximum age is not P66.")


# ============================================================
# [4/10] PREPARE INPUTS
# ============================================================

print("\n[4/10] Preparing tensors...")

# Feature scaler was fitted on the official train data.
X_dbh_height_raw = dbh_height[FEATURE_COLUMNS].copy()
X_volume_raw = volume[FEATURE_COLUMNS].copy()

X_dbh_height_scaled = feature_scaler.transform(
    X_dbh_height_raw
)

X_volume_scaled = feature_scaler.transform(
    X_volume_raw
)

X_full = torch.tensor(
    X_dbh_height_scaled,
    dtype=torch.float32,
    device=DEVICE,
)

X_volume = torch.tensor(
    X_volume_scaled,
    dtype=torch.float32,
    device=DEVICE,
)

y_dbh = torch.tensor(
    dbh_scaler.transform(
        dbh_height[["dbh_cm"]]
    ),
    dtype=torch.float32,
    device=DEVICE,
)

y_height = torch.tensor(
    height_scaler.transform(
        dbh_height[["height_m"]]
    ),
    dtype=torch.float32,
    device=DEVICE,
)

y_volume = torch.tensor(
    volume_scaler.transform(
        volume[["volume_m3_tree"]]
    ),
    dtype=torch.float32,
    device=DEVICE,
)

print(f"X DBH/Height : {tuple(X_full.shape)}")
print(f"X Volume     : {tuple(X_volume.shape)}")


# ============================================================
# [5/10] BUILD MODEL
# ============================================================

print("\n[5/10] Building PINNs V2 model...")

model = PINNsGrowthModelV2(
    input_size=6,
    hidden_size=HIDDEN_SIZE,
).to(DEVICE)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Architecture : 6 -> 64 -> 64 -> 64 -> 3")
print(f"Parameters   : {parameter_count:,}")
print(f"Device       : {DEVICE}")


# ============================================================
# [6/10] OPTIMIZER
# ============================================================

print("\n[6/10] Creating optimizer...")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

mse_loss = nn.MSELoss()


# ============================================================
# [7/10] TRAINING
# ============================================================

print("\n[7/10] Starting training...")
print("-" * 70)

history = []

best_loss = float("inf")
best_epoch = 0

for epoch in range(1, EPOCHS + 1):

    model.train()
    optimizer.zero_grad()

    # ========================================================
    # 1. DBH + HEIGHT SUPERVISED LOSS
    #    P6-P66
    # ========================================================

    dbh_pred_scaled, height_pred_scaled, _ = model(
        X_full
    )

    loss_dbh = mse_loss(
        dbh_pred_scaled,
        y_dbh,
    )

    loss_height = mse_loss(
        height_pred_scaled,
        y_height,
    )

    # ========================================================
    # 2. VOLUME SUPERVISED LOSS
    #    P18-P66
    # ========================================================

    dbh_volume_scaled, height_volume_scaled, volume_pred_scaled = model(
        X_volume
    )

    loss_volume = mse_loss(
        volume_pred_scaled,
        y_volume,
    )

    # ========================================================
    # 3. PHYSICS LOSS
    #    ONLY P18-P66
    #
    #    V = C * DBH^2 * Height
    # ========================================================

    dbh_physical = (
        dbh_volume_scaled * dbh_scale
        + dbh_mean
    )

    height_physical = (
        height_volume_scaled * height_scale
        + height_mean
    )

    volume_physical = (
        volume_pred_scaled * volume_scale
        + volume_mean
    )

    physics_expected_volume = (
        C_PHYSICS
        * torch.square(dbh_physical)
        * height_physical
    )

    physics_residual_scaled = (
        volume_physical
        - physics_expected_volume
    ) / volume_scale

    loss_physics = torch.mean(
        torch.square(physics_residual_scaled)
    )

    # ========================================================
    # 4. TOTAL LOSS
    # ========================================================

    total_loss = (
        loss_dbh
        + loss_height
        + loss_volume
        + PHYSICS_WEIGHT * loss_physics
    )

    # ========================================================
    # 5. BACKPROPAGATION
    # ========================================================

    total_loss.backward()

    optimizer.step()

    # ========================================================
    # 6. RECORD HISTORY
    # ========================================================

    total_value = float(total_loss.item())

    row = {
        "epoch": epoch,
        "loss_total": total_value,
        "loss_dbh": float(loss_dbh.item()),
        "loss_height": float(loss_height.item()),
        "loss_volume": float(loss_volume.item()),
        "loss_physics": float(loss_physics.item()),
    }

    history.append(row)

    # ========================================================
    # 7. SAVE BEST CHECKPOINT
    # ========================================================

    if total_value < best_loss:

        best_loss = total_value
        best_epoch = epoch

        torch.save(
            {
                "model_state_dict": model.state_dict(),

                "architecture": {
                    "input_size": 6,
                    "hidden_size": 64,
                    "hidden_layers": 3,
                    "parameter_count": parameter_count,
                },

                "features": FEATURE_COLUMNS,

                "physics_coefficient": C_PHYSICS,

                "physics_weight": PHYSICS_WEIGHT,

                "learning_rate": LEARNING_RATE,

                "epoch": epoch,

                "loss_total": total_value,
            },
            MODEL_PATH,
        )

    # ========================================================
    # 8. TERMINAL PROGRESS
    # ========================================================

    if (
        epoch == 1
        or epoch % 10 == 0
        or epoch == EPOCHS
    ):

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Total={total_loss.item():.8f} | "
            f"DBH={loss_dbh.item():.8f} | "
            f"Height={loss_height.item():.8f} | "
            f"Volume={loss_volume.item():.8f} | "
            f"Physics={loss_physics.item():.8f}"
        )

# ============================================================
# [8/10] SAVE HISTORY
# ============================================================

print("\n[8/10] Saving training history...")

history_df = pd.DataFrame(history)

history_df.to_csv(
    HISTORY_PATH,
    index=False,
)


# ============================================================
# [9/10] FINAL TRAINING SANITY CHECK
# ============================================================

print("\n[9/10] Running training sanity check...")

model.eval()

with torch.no_grad():

    dbh_scaled_pred, height_scaled_pred, _ = model(
        X_full
    )

    _, _, volume_scaled_pred = model(
        X_volume
    )

    dbh_pred = (
        dbh_scaled_pred * dbh_scale
        + dbh_mean
    )

    height_pred = (
        height_scaled_pred * height_scale
        + height_mean
    )

    volume_pred = (
        volume_scaled_pred * volume_scale
        + volume_mean
    )

    volume_min = float(volume_pred.min().item())
    volume_max = float(volume_pred.max().item())
    volume_mean_pred = float(volume_pred.mean().item())

    negative_count = int(
        (volume_pred < 0).sum().item()
    )

print(
    f"Predicted DBH range    : "
    f"{dbh_pred.min().item():.6f} - "
    f"{dbh_pred.max().item():.6f}"
)

print(
    f"Predicted Height range : "
    f"{height_pred.min().item():.6f} - "
    f"{height_pred.max().item():.6f}"
)

print(
    f"Predicted Volume range : "
    f"{volume_min:.6f} - {volume_max:.6f}"
)

print(
    f"Predicted Volume mean  : "
    f"{volume_mean_pred:.6f}"
)

print(
    f"Negative volume count  : "
    f"{negative_count:,}"
)


# ============================================================
# [10/10] REPORT
# ============================================================

elapsed = time.time() - start_time

report = {
    "status": "PASS",
    "step": 53,
    "model": "PINNs V2",

    "architecture": {
        "input_size": 6,
        "hidden_size": 64,
        "hidden_layers": 3,
        "parameter_count": parameter_count,
    },

    "training": {
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "physics_weight": PHYSICS_WEIGHT,
        "optimizer": "Adam",
        "device": str(DEVICE),
        "seed": SEED,
        "training_time_seconds": elapsed,
    },

    "data": {
        "train_dbh_height_rows": len(dbh_height),
        "train_volume_rows": len(volume),
        "dbh_height_age_min": int(dbh_height["age_months"].min()),
        "dbh_height_age_max": int(dbh_height["age_months"].max()),
        "volume_age_min": int(volume["age_months"].min()),
        "volume_age_max": int(volume["age_months"].max()),
    },

    "physics": {
        "formula": "V = C * DBH^2 * Height",
        "coefficient": C_PHYSICS,
        "age_min_months": 18,
        "age_max_months": 66,
    },

    "best_model": {
        "epoch": best_epoch,
        "loss_total": best_loss,
        "checkpoint": str(MODEL_PATH),
    },

    "volume_sanity": {
        "min": volume_min,
        "max": volume_max,
        "mean": volume_mean_pred,
        "negative_count": negative_count,
    },

    "note": (
        "PINNs V2 uses direct scaled outputs for DBH, Height, "
        "and Volume. No Softplus is applied to the supervised "
        "volume output. Physics is calculated after converting "
        "predictions back to physical units."
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
print("STEP 53 PINNs V2 TRAINING COMPLETE")
print("=" * 70)

print(f"Best epoch       : {best_epoch}")
print(f"Best total loss  : {best_loss:.8f}")
print(f"Training time    : {elapsed:.2f} seconds")

print(f"\nModel:")
print(MODEL_PATH)

print(f"\nHistory:")
print(HISTORY_PATH)

print(f"\nReport:")
print(REPORT_PATH)

print("=" * 70)