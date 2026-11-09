from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# STEP 7D-2B
# BUILD PINNs ARCHITECTURE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "results" / "model_train.csv"
PHYSICS_CONFIG_PATH = (
    PROJECT_ROOT / "results" / "pinns_physics_config.json"
)

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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

INPUT_SIZE = len(FEATURES)
OUTPUT_SIZE = len(TARGETS)

HIDDEN_SIZE = 64
HIDDEN_LAYERS = 3

LAMBDA_PHYSICS = 0.10


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


# ============================================================
# PINNs MODEL
# ============================================================

class TreeGrowthPINN(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        hidden_layers,
        output_size,
    ):
        super().__init__()

        layers = []

        # Input layer
        layers.append(
            nn.Linear(
                input_size,
                hidden_size,
            )
        )

        layers.append(
            nn.Tanh()
        )

        # Hidden layers
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

        # Output layer
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

    loss = torch.mean(
        (
            volume
            - volume_physics
        ).pow(2)
    )

    return loss


# ============================================================
# TOTAL LOSS
# ============================================================

def pinn_loss(
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
print("STEP 7D-2B - BUILD PINNs ARCHITECTURE")
print("=" * 75)


# ============================================================
# 1. LOAD PHYSICS CONFIG
# ============================================================

print("\n[1/6] Loading physics configuration...")

with open(
    PHYSICS_CONFIG_PATH,
    "r",
    encoding="utf-8",
) as file:

    physics_config = json.load(file)


C_VALUE = float(
    physics_config["C"]
)

print(
    f"Physics constant C : "
    f"{C_VALUE:.12f}"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n[2/6] Loading training data...")

train_df = pd.read_csv(
    TRAIN_PATH
)

print(
    f"Training rows : "
    f"{len(train_df):,}"
)


# ============================================================
# 3. CALCULATE NORMALIZATION
# ============================================================

print("\n[3/6] Calculating normalization parameters...")

X = train_df[
    FEATURES
].astype(
    np.float32
)

Y = train_df[
    TARGETS
].astype(
    np.float32
)


X_mean = X.mean()
X_std = X.std()

Y_mean = Y.mean()
Y_std = Y.std()


# Prevent division by zero
X_std = X_std.replace(
    0,
    1.0,
)

Y_std = Y_std.replace(
    0,
    1.0,
)


print("\nInput normalization:")

for feature in FEATURES:

    print(
        f"  {feature}: "
        f"mean={X_mean[feature]:.6f}, "
        f"std={X_std[feature]:.6f}"
    )


print("\nTarget normalization:")

for target in TARGETS:

    print(
        f"  {target}: "
        f"mean={Y_mean[target]:.6f}, "
        f"std={Y_std[target]:.6f}"
    )


# ============================================================
# 4. BUILD MODEL
# ============================================================

print("\n[4/6] Building PINNs model...")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = TreeGrowthPINN(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    hidden_layers=HIDDEN_LAYERS,
    output_size=OUTPUT_SIZE,
).to(device)


print("\nArchitecture:")

print(model)

print(
    f"\nDevice : {device}"
)

print(
    f"Input size  : {INPUT_SIZE}"
)

print(
    f"Output size : {OUTPUT_SIZE}"
)


# ============================================================
# 5. FORWARD PASS TEST
# ============================================================

print("\n[5/6] Running forward-pass test...")

# Use first 8 training rows only.
sample_X = X.iloc[
    :8
].values

sample_Y = Y.iloc[
    :8
].values


# Normalize
sample_X_norm = (
    sample_X
    - X_mean.values
) / X_std.values

sample_Y_norm = (
    sample_Y
    - Y_mean.values
) / Y_std.values


X_tensor = torch.tensor(
    sample_X_norm,
    dtype=torch.float32,
    device=device,
)

Y_tensor = torch.tensor(
    sample_Y_norm,
    dtype=torch.float32,
    device=device,
)


model.eval()

with torch.no_grad():

    predictions = model(
        X_tensor
    )


print(
    f"Input tensor shape  : "
    f"{tuple(X_tensor.shape)}"
)

print(
    f"Output tensor shape : "
    f"{tuple(predictions.shape)}"
)


expected_shape = (
    8,
    OUTPUT_SIZE,
)

if tuple(predictions.shape) != expected_shape:

    raise RuntimeError(
        "PINNs forward-pass shape mismatch. "
        f"Expected {expected_shape}, "
        f"got {tuple(predictions.shape)}"
    )


# ============================================================
# 6. LOSS TEST
# ============================================================

print("\n[6/6] Testing data loss + physics loss...")

total_loss, data_loss, physics_loss = pinn_loss(
    predictions,
    Y_tensor,
    torch.tensor(
        C_VALUE,
        dtype=torch.float32,
        device=device,
    ),
    LAMBDA_PHYSICS,
)


print(
    f"Data loss    : "
    f"{data_loss.item():.8f}"
)

print(
    f"Physics loss : "
    f"{physics_loss.item():.8f}"
)

print(
    f"Total loss   : "
    f"{total_loss.item():.8f}"
)


if not torch.isfinite(
    total_loss
):

    raise RuntimeError(
        "Loss contains NaN or Inf."
    )


# ============================================================
# SAVE MODEL CONFIG
# ============================================================

config = {

    "model": "TreeGrowthPINN",

    "random_state": RANDOM_STATE,

    "features": FEATURES,

    "targets": TARGETS,

    "architecture": {

        "input_size": INPUT_SIZE,

        "hidden_size": HIDDEN_SIZE,

        "hidden_layers": HIDDEN_LAYERS,

        "activation": "Tanh",

        "output_size": OUTPUT_SIZE,
    },

    "physics": {

        "relationship":
            "volume ~= C * dbh^2 * height",

        "C": C_VALUE,

        "lambda_physics":
            LAMBDA_PHYSICS,
    },

    "device": str(device),
}


config_path = (
    RESULTS_DIR
    / "pinns_model_config.json"
)

with open(
    config_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        config,
        file,
        indent=4,
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("STEP 7D-2B COMPLETED")
print("=" * 75)

print(
    "\nPINNs architecture successfully built."
)

print(
    "\nForward pass successfully verified."
)

print(
    "Data loss successfully verified."
)

print(
    "Physics loss successfully verified."
)

print(
    f"\nSaved configuration:"
)

print(
    f"  - {config_path}"
)

print(
    "\nNO TRAINING WAS PERFORMED YET."
)

print(
    "The model is ready for the training step."
)

print("=" * 75)