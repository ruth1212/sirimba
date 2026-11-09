from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# STEP 7D-2C
# TRAIN PINNs
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "results" / "model_train.csv"
PHYSICS_CONFIG_PATH = (
    PROJECT_ROOT / "results" / "pinns_physics_config.json"
)
MODEL_CONFIG_PATH = (
    PROJECT_ROOT / "results" / "pinns_model_config.json"
)

RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = RESULTS_DIR / "pinns_models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

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

# CPU-friendly pilot configuration
EPOCHS = 30
BATCH_SIZE = 2048
LEARNING_RATE = 0.001

LAMBDA_PHYSICS = 0.10

PRINT_EVERY = 1

EARLY_STOPPING_PATIENCE = 7

MIN_DELTA = 1e-5


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


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
# PHYSICS LOSS
# ============================================================

def volume_physics_loss(
    predictions,
    C,
):

    dbh = predictions[:, 0]
    height = predictions[:, 1]
    volume = predictions[:, 2]

    volume_physics = (
        C
        * dbh.pow(2)
        * height
    )

    return torch.mean(
        (
            volume
            - volume_physics
        ).pow(2)
    )


# ============================================================
# TOTAL LOSS
# ============================================================

def calculate_losses(
    predictions,
    targets,
    C,
    lambda_physics,
):

    data_loss = torch.mean(
        (
            predictions
            - targets
        ).pow(2)
    )

    physics_loss = volume_physics_loss(
        predictions,
        C,
    )

    total_loss = (
        data_loss
        + lambda_physics
        * physics_loss
    )

    return (
        total_loss,
        data_loss,
        physics_loss,
    )


# ============================================================
# START
# ============================================================

print("=" * 75)
print("TREE GROWTH ANALYTICS")
print("STEP 7D-2C - TRAIN PINNs")
print("=" * 75)


# ============================================================
# 1. DEVICE
# ============================================================

print("\n[1/8] Selecting device...")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Device : {device}")


# ============================================================
# 2. LOAD CONFIG
# ============================================================

print("\n[2/8] Loading PINNs configuration...")

with open(
    PHYSICS_CONFIG_PATH,
    "r",
    encoding="utf-8",
) as file:

    physics_config = json.load(file)


with open(
    MODEL_CONFIG_PATH,
    "r",
    encoding="utf-8",
) as file:

    model_config = json.load(file)


C_VALUE = float(
    physics_config["C"]
)

print(
    f"Physics C : "
    f"{C_VALUE:.12f}"
)

print(
    f"Lambda physics : "
    f"{LAMBDA_PHYSICS}"
)


# ============================================================
# 3. LOAD TRAINING DATA
# ============================================================

print("\n[3/8] Loading training data...")

df = pd.read_csv(
    TRAIN_PATH
)

print(
    f"Training rows : "
    f"{len(df):,}"
)


# ============================================================
# 4. NORMALIZATION
# ============================================================

print("\n[4/8] Preparing normalized tensors...")

X_df = df[
    FEATURES
].astype(
    np.float32
)

Y_df = df[
    TARGETS
].astype(
    np.float32
)


X_mean = X_df.mean().values.astype(
    np.float32
)

X_std = X_df.std().values.astype(
    np.float32
)

Y_mean = Y_df.mean().values.astype(
    np.float32
)

Y_std = Y_df.std().values.astype(
    np.float32
)


X_std[
    X_std == 0
] = 1.0

Y_std[
    Y_std == 0
] = 1.0


X_normalized = (
    X_df.values
    - X_mean
) / X_std

Y_normalized = (
    Y_df.values
    - Y_mean
) / Y_std


X_tensor = torch.tensor(
    X_normalized,
    dtype=torch.float32,
)

Y_tensor = torch.tensor(
    Y_normalized,
    dtype=torch.float32,
)


# ============================================================
# 5. DATALOADER
# ============================================================

print("\n[5/8] Creating mini-batch DataLoader...")

dataset = TensorDataset(
    X_tensor,
    Y_tensor,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)

print(
    f"Batch size : "
    f"{BATCH_SIZE:,}"
)

print(
    f"Batches per epoch : "
    f"{len(loader):,}"
)


# ============================================================
# 6. BUILD MODEL
# ============================================================

print("\n[6/8] Building PINNs model...")

model = TreeGrowthPINN(
    input_size=len(FEATURES),
    hidden_size=64,
    hidden_layers=3,
    output_size=len(TARGETS),
).to(device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


C_tensor = torch.tensor(
    C_VALUE,
    dtype=torch.float32,
    device=device,
)


print(
    f"Learning rate : "
    f"{LEARNING_RATE}"
)

print(
    f"Epochs        : "
    f"{EPOCHS}"
)


# ============================================================
# 7. TRAINING
# ============================================================

print("\n[7/8] Starting PINNs training...")
print("=" * 75)

history = []

best_loss = float("inf")
best_epoch = 0
patience_counter = 0

start_training = time.time()


for epoch in range(
    1,
    EPOCHS + 1,
):

    epoch_start = time.time()

    model.train()

    total_epoch_loss = 0.0
    total_data_loss = 0.0
    total_physics_loss = 0.0

    sample_count = 0

    for batch_X, batch_Y in loader:

        batch_X = batch_X.to(
            device,
            non_blocking=False,
        )

        batch_Y = batch_Y.to(
            device,
            non_blocking=False,
        )

        optimizer.zero_grad()

        predictions = model(
            batch_X
        )

        (
            total_loss,
            data_loss,
            physics_loss,
        ) = calculate_losses(
            predictions,
            batch_Y,
            C_tensor,
            LAMBDA_PHYSICS,
        )

        if not torch.isfinite(
            total_loss
        ):

            raise RuntimeError(
                f"Non-finite loss detected "
                f"at epoch {epoch}."
            )

        total_loss.backward()

        optimizer.step()

        batch_size_actual = (
            batch_X.shape[0]
        )

        total_epoch_loss += (
            total_loss.item()
            * batch_size_actual
        )

        total_data_loss += (
            data_loss.item()
            * batch_size_actual
        )

        total_physics_loss += (
            physics_loss.item()
            * batch_size_actual
        )

        sample_count += (
            batch_size_actual
        )


    epoch_loss = (
        total_epoch_loss
        / sample_count
    )

    epoch_data_loss = (
        total_data_loss
        / sample_count
    )

    epoch_physics_loss = (
        total_physics_loss
        / sample_count
    )

    epoch_time = (
        time.time()
        - epoch_start
    )

    history.append(
        {
            "epoch": epoch,
            "total_loss": epoch_loss,
            "data_loss": epoch_data_loss,
            "physics_loss": epoch_physics_loss,
            "epoch_time_seconds": epoch_time,
        }
    )


    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Total: {epoch_loss:.6f} | "
        f"Data: {epoch_data_loss:.6f} | "
        f"Physics: {epoch_physics_loss:.6f} | "
        f"Time: {epoch_time:.2f}s"
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    if (
        best_loss
        - epoch_loss
        > MIN_DELTA
    ):

        best_loss = epoch_loss
        best_epoch = epoch
        patience_counter = 0

        checkpoint = {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "best_loss": best_loss,
            "features": FEATURES,
            "targets": TARGETS,
            "C": C_VALUE,
            "lambda_physics":
                LAMBDA_PHYSICS,
            "X_mean": X_mean,
            "X_std": X_std,
            "Y_mean": Y_mean,
            "Y_std": Y_std,
        }

        torch.save(
            checkpoint,
            MODEL_DIR
            / "best_pinns_model.pt",
        )

    else:

        patience_counter += 1


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        patience_counter
        >= EARLY_STOPPING_PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        break


total_training_time = (
    time.time()
    - start_training
)


# ============================================================
# 8. SAVE HISTORY
# ============================================================

print("\n[8/8] Saving training results...")

history_df = pd.DataFrame(
    history
)

history_path = (
    RESULTS_DIR
    / "pinns_training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False,
)


training_config = {

    "model": "TreeGrowthPINN",

    "random_state": RANDOM_STATE,

    "features": FEATURES,

    "targets": TARGETS,

    "epochs_requested": EPOCHS,

    "epochs_completed":
        len(history),

    "batch_size": BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "lambda_physics":
        LAMBDA_PHYSICS,

    "physics_C":
        C_VALUE,

    "best_epoch":
        best_epoch,

    "best_loss":
        best_loss,

    "total_training_time_seconds":
        total_training_time,

    "device":
        str(device),
}


config_path = (
    RESULTS_DIR
    / "pinns_training_config.json"
)

with open(
    config_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        training_config,
        file,
        indent=4,
    )


print("\n" + "=" * 75)
print("PINNs TRAINING COMPLETED")
print("=" * 75)

print(
    f"Epochs completed : "
    f"{len(history)}"
)

print(
    f"Best epoch       : "
    f"{best_epoch}"
)

print(
    f"Best loss        : "
    f"{best_loss:.8f}"
)

print(
    f"Training time    : "
    f"{total_training_time:.2f} seconds"
)

print("\nSaved:")

print(
    "  - results/pinns_models/best_pinns_model.pt"
)

print(
    "  - results/pinns_training_history.csv"
)

print(
    "  - results/pinns_training_config.json"
)

print("=" * 75)