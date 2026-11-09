import os
import json
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# TREE GROWTH ANALYTICS
# STEP 21N - TRAIN HARD-POSITIVE PINNs
# ============================================================

print("=" * 75)
print("TREE GROWTH ANALYTICS")
print("STEP 21N - TRAIN HARD-POSITIVE PINNs")
print("=" * 75)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

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

EPOCHS = 30
BATCH_SIZE = 2048
LEARNING_RATE = 0.001

LAMBDA_PHYSICS = 0.10

TRAIN_FILE = "results/model_train.csv"

OUTPUT_DIR = "results/pinns_models"

CHECKPOINT_FILE = (
    "results/pinns_models/"
    "best_pinns_hard_positive_model.pt"
)

HISTORY_FILE = (
    "results/"
    "pinns_hard_positive_training_history.csv"
)

CONFIG_FILE = (
    "results/"
    "pinns_hard_positive_training_config.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

print("\n[1/8] Checking device...")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device :", DEVICE)


# ============================================================
# LOAD DATA
# ============================================================

print("\n[2/8] Loading training data...")

df = pd.read_csv(TRAIN_FILE)

print("Training rows :", len(df))

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
# EXTRACT DATA
# ============================================================

X = df[FEATURES].astype(np.float32).values
Y = df[TARGETS].astype(np.float32).values


# ============================================================
# NORMALIZATION
# ============================================================

print("\n[3/8] Calculating normalization...")

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

Y_mean = Y.mean(axis=0)
Y_std = Y.std(axis=0)

X_std[X_std == 0] = 1.0
Y_std[Y_std == 0] = 1.0

X_norm = (X - X_mean) / X_std
Y_norm = (Y - Y_mean) / Y_std

print("\nInput normalization:")

for i, feature in enumerate(FEATURES):
    print(
        f"  {feature}: "
        f"mean={X_mean[i]:.6f}, "
        f"std={X_std[i]:.6f}"
    )

print("\nTarget normalization:")

for i, target in enumerate(TARGETS):
    print(
        f"  {target}: "
        f"mean={Y_mean[i]:.6f}, "
        f"std={Y_std[i]:.6f}"
    )


# ============================================================
# PHYSICS CONSTANT
# ============================================================

print("\n[4/8] Calculating physics constant...")

dbh = Y[:, 0]
height = Y[:, 1]
volume = Y[:, 2]

physics_base = (dbh ** 2) * height

valid = (
    np.isfinite(physics_base)
    & np.isfinite(volume)
    & (physics_base > 0)
    & (volume > 0)
)

C = float(
    np.mean(
        volume[valid] / physics_base[valid]
    )
)

print("Physics constant C :", f"{C:.12f}")

print(
    "Valid physics rows :",
    int(valid.sum())
)


# ============================================================
# DATASET
# ============================================================

X_tensor = torch.tensor(
    X_norm,
    dtype=torch.float32
)

Y_tensor = torch.tensor(
    Y_norm,
    dtype=torch.float32
)

dataset = TensorDataset(
    X_tensor,
    Y_tensor
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
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

print("\nModel:")
print(model)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\n[5/8] Starting training...")
print()
print("Epochs     :", EPOCHS)
print("Batch size :", BATCH_SIZE)
print("LR         :", LEARNING_RATE)
print("Physics λ  :", LAMBDA_PHYSICS)
print()

best_loss = float("inf")
best_epoch = 0

history = []

training_start = time.time()


for epoch in range(1, EPOCHS + 1):

    epoch_start = time.time()

    model.train()

    total_loss_sum = 0.0
    data_loss_sum = 0.0
    physics_loss_sum = 0.0

    sample_count = 0

    for batch_x, batch_y in loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        prediction_norm = model(batch_x)

        # ----------------------------------------------------
        # DATA LOSS
        # ----------------------------------------------------

        data_loss = torch.mean(
            (prediction_norm - batch_y) ** 2
        )

        # ----------------------------------------------------
        # DENORMALIZE PREDICTIONS
        # ----------------------------------------------------

        prediction_raw = (
            prediction_norm
            * torch.tensor(
                Y_std,
                dtype=torch.float32,
                device=DEVICE
            )
            + torch.tensor(
                Y_mean,
                dtype=torch.float32,
                device=DEVICE
            )
        )

        pred_dbh = prediction_raw[:, 0:1]
        pred_height = prediction_raw[:, 1:2]
        pred_volume = prediction_raw[:, 2:3]

        # ----------------------------------------------------
        # PHYSICS RELATION
        #
        # V = C * DBH² * Height
        # ----------------------------------------------------

        physics_volume = (
            C
            * (pred_dbh ** 2)
            * pred_height
        )

        volume_std_tensor = torch.tensor(
            Y_std[2],
            dtype=torch.float32,
            device=DEVICE
        )

        physics_residual = (
            pred_volume - physics_volume
        ) / volume_std_tensor

        physics_loss = torch.mean(
            physics_residual ** 2
        )

        # ----------------------------------------------------
        # TOTAL LOSS
        # ----------------------------------------------------

        loss = (
            data_loss
            + LAMBDA_PHYSICS * physics_loss
        )

        loss.backward()

        optimizer.step()

        batch_size_actual = batch_x.shape[0]

        total_loss_sum += (
            float(loss.item())
            * batch_size_actual
        )

        data_loss_sum += (
            float(data_loss.item())
            * batch_size_actual
        )

        physics_loss_sum += (
            float(physics_loss.item())
            * batch_size_actual
        )

        sample_count += batch_size_actual

    epoch_loss = (
        total_loss_sum / sample_count
    )

    epoch_data_loss = (
        data_loss_sum / sample_count
    )

    epoch_physics_loss = (
        physics_loss_sum / sample_count
    )

    epoch_time = time.time() - epoch_start

    history.append({
        "epoch": epoch,
        "total_loss": epoch_loss,
        "data_loss": epoch_data_loss,
        "physics_loss": epoch_physics_loss,
        "epoch_time_sec": epoch_time
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Total: {epoch_loss:.8f} | "
        f"Data: {epoch_data_loss:.8f} | "
        f"Physics: {epoch_physics_loss:.8f} | "
        f"Time: {epoch_time:.2f}s"
    )

    if epoch_loss < best_loss:

        best_loss = epoch_loss
        best_epoch = epoch

        os.makedirs(
            OUTPUT_DIR,
            exist_ok=True
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "best_loss":
                    best_loss,

                "best_epoch":
                    best_epoch,

                "features":
                    FEATURES,

                "targets":
                    TARGETS,

                "X_mean":
                    X_mean,

                "X_std":
                    X_std,

                "Y_mean":
                    Y_mean,

                "Y_std":
                    Y_std,

                "C":
                    C,

                "lambda_physics":
                    LAMBDA_PHYSICS,

                "physics_loss_mode":
                    "volume_relation_raw_space",

                "volume_activation":
                    "softplus_raw_space",

                "architecture":
                    "6-64-64-64-3"
            },
            CHECKPOINT_FILE
        )


# ============================================================
# TRAINING COMPLETE
# ============================================================

training_time = (
    time.time() - training_start
)

print("\n[6/8] Training completed.")

print(
    "Best epoch :",
    best_epoch
)

print(
    "Best loss  :",
    f"{best_loss:.8f}"
)

print(
    "Training time :",
    f"{training_time:.2f} sec"
)


# ============================================================
# SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    HISTORY_FILE,
    index=False
)

print("\n[7/8] Training history saved:")
print(
    " ",
    HISTORY_FILE
)


# ============================================================
# SAVE CONFIG
# ============================================================

config = {
    "step": "21N",
    "model": "PINNs Hard Positive",
    "features": FEATURES,
    "targets": TARGETS,
    "architecture": "6-64-64-64-3",
    "activation": "Tanh",
    "volume_activation": "Softplus raw physical space",
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "lambda_physics": LAMBDA_PHYSICS,
    "device": str(DEVICE),
    "training_rows": len(df),
    "best_epoch": best_epoch,
    "best_loss": best_loss,
    "training_time_sec": training_time,
    "physics_constant_C": C,
    "normalization": {
        "X_mean": X_mean.tolist(),
        "X_std": X_std.tolist(),
        "Y_mean": Y_mean.tolist(),
        "Y_std": Y_std.tolist()
    }
}

with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        config,
        f,
        indent=2
    )

print("\n[8/8] Configuration saved:")
print(
    " ",
    CONFIG_FILE
)


print("\n" + "=" * 75)
print("STEP 21N COMPLETED")
print("=" * 75)

print(
    "Hard-positive PINNs training completed successfully."
)

print(
    "Checkpoint:",
    CHECKPOINT_FILE
)

print("=" * 75)