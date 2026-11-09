import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# TREE GROWTH ANALYTICS
# STEP 21I - TRAIN PINNs WITH POSITIVE VOLUME CONSTRAINT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_ROOT
    / "results"
    / "model_train.csv"
)

PHYSICS_CONFIG_PATH = (
    PROJECT_ROOT
    / "results"
    / "pinns_physics_config.json"
)

MODEL_CONFIG_PATH = (
    PROJECT_ROOT
    / "results"
    / "pinns_model_config.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

MODEL_DIR = (
    RESULTS_DIR
    / "pinns_models"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

EPOCHS = 30

BATCH_SIZE = 2048

LEARNING_RATE = 0.001

LAMBDA_PHYSICS = 0.10

# New positivity constraint
LAMBDA_POSITIVE = 0.50

PRINT_EVERY = 1

EARLY_STOPPING_PATIENCE = 7

MIN_DELTA = 1e-5


# ============================================================
# OUTPUT FILES
# ============================================================

MODEL_OUTPUT_PATH = (
    MODEL_DIR
    / "best_pinns_positive_model.pt"
)

HISTORY_OUTPUT_PATH = (
    RESULTS_DIR
    / "pinns_positive_training_history.csv"
)

CONFIG_OUTPUT_PATH = (
    RESULTS_DIR
    / "pinns_positive_training_config.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)

np.random.seed(
    RANDOM_STATE
)

torch.manual_seed(
    RANDOM_STATE
)


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

        # ----------------------------------------------------
        # Input layer
        # ----------------------------------------------------

        layers.append(
            nn.Linear(
                input_size,
                hidden_size,
            )
        )

        layers.append(
            nn.Tanh()
        )

        # ----------------------------------------------------
        # Hidden layers
        # ----------------------------------------------------

        for _ in range(
            hidden_layers - 1
        ):

            layers.append(
                nn.Linear(
                    hidden_size,
                    hidden_size,
                )
            )

            layers.append(
                nn.Tanh()
            )

        # ----------------------------------------------------
        # Output layer
        # ----------------------------------------------------

        layers.append(
            nn.Linear(
                hidden_size,
                output_size,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x,
    ):

        return self.network(x)


# ============================================================
# PHYSICS LOSS
# ============================================================

def volume_physics_loss(
    predictions_normalized,
    C,
    Y_mean,
    Y_std,
):
    """
    Physics relationship:

        Volume = C * DBH^2 * Height

    The model predicts normalized targets.

    Therefore:

        normalized -> raw units

    before calculating the physics relationship.
    """

    # --------------------------------------------------------
    # Convert predictions to raw units
    # --------------------------------------------------------

    predictions_raw = (
        predictions_normalized
        * Y_std
        + Y_mean
    )

    # --------------------------------------------------------
    # Extract targets
    # --------------------------------------------------------

    dbh = predictions_raw[:, 0]

    height = predictions_raw[:, 1]

    volume = predictions_raw[:, 2]

    # --------------------------------------------------------
    # Physics-based volume
    # --------------------------------------------------------

    volume_physics = (
        C
        * dbh.pow(2)
        * height
    )

    # --------------------------------------------------------
    # Physics residual
    # --------------------------------------------------------

    physics_residual_raw = (
        volume
        - volume_physics
    )

    # --------------------------------------------------------
    # Normalize residual
    # --------------------------------------------------------

    volume_std = Y_std[2]

    physics_residual_normalized = (
        physics_residual_raw
        / volume_std
    )

    # --------------------------------------------------------
    # Mean squared physics loss
    # --------------------------------------------------------

    return torch.mean(
        physics_residual_normalized.pow(2)
    )


# ============================================================
# POSITIVITY LOSS
# ============================================================

def volume_positivity_loss(
    predictions_normalized,
    Y_mean,
    Y_std,
):
    """
    Enforce:

        Volume >= 0

    Only negative predicted volume receives a penalty.

    The calculation is performed in raw physical units and
    then scaled by the training volume standard deviation.
    """

    # --------------------------------------------------------
    # Convert normalized prediction to raw units
    # --------------------------------------------------------

    predictions_raw = (
        predictions_normalized
        * Y_std
        + Y_mean
    )

    # --------------------------------------------------------
    # Extract volume
    # --------------------------------------------------------

    volume = predictions_raw[:, 2]

    # --------------------------------------------------------
    # Negative part only
    #
    # Positive volume:
    #     penalty = 0
    #
    # Negative volume:
    #     penalty = (-volume)^2
    # --------------------------------------------------------

    negative_volume = torch.relu(
        -volume
    )

    # --------------------------------------------------------
    # Normalize by volume standard deviation
    # --------------------------------------------------------

    volume_std = Y_std[2]

    normalized_negative_volume = (
        negative_volume
        / volume_std
    )

    # --------------------------------------------------------
    # Mean squared positivity penalty
    # --------------------------------------------------------

    return torch.mean(
        normalized_negative_volume.pow(2)
    )


# ============================================================
# TOTAL LOSS
# ============================================================

def calculate_losses(
    predictions,
    targets,
    C,
    Y_mean,
    Y_std,
    lambda_physics,
    lambda_positive,
):
    # --------------------------------------------------------
    # Data loss
    # --------------------------------------------------------

    data_loss = torch.mean(
        (
            predictions
            - targets
        ).pow(2)
    )

    # --------------------------------------------------------
    # Physics loss
    # --------------------------------------------------------

    physics_loss = volume_physics_loss(
        predictions_normalized=predictions,
        C=C,
        Y_mean=Y_mean,
        Y_std=Y_std,
    )

    # --------------------------------------------------------
    # Positivity loss
    # --------------------------------------------------------

    positive_loss = volume_positivity_loss(
        predictions_normalized=predictions,
        Y_mean=Y_mean,
        Y_std=Y_std,
    )

    # --------------------------------------------------------
    # Total loss
    # --------------------------------------------------------

    total_loss = (
        data_loss
        + (
            lambda_physics
            * physics_loss
        )
        + (
            lambda_positive
            * positive_loss
        )
    )

    return (
        total_loss,
        data_loss,
        physics_loss,
        positive_loss,
    )


# ============================================================
# START
# ============================================================

print("=" * 75)
print("TREE GROWTH ANALYTICS")
print("STEP 21I - TRAIN PINNs WITH POSITIVE VOLUME CONSTRAINT")
print("=" * 75)


# ============================================================
# 1. DEVICE
# ============================================================

print("\n[1/10] Selecting device...")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Device : {device}"
)


# ============================================================
# 2. LOAD CONFIGURATION
# ============================================================

print(
    "\n[2/10] Loading PINNs configuration..."
)


with open(
    PHYSICS_CONFIG_PATH,
    "r",
    encoding="utf-8",
) as file:

    physics_config = json.load(
        file
    )


with open(
    MODEL_CONFIG_PATH,
    "r",
    encoding="utf-8",
) as file:

    model_config = json.load(
        file
    )


C_VALUE = float(
    physics_config["C"]
)


print(
    f"Physics C       : "
    f"{C_VALUE:.12f}"
)

print(
    f"Lambda physics  : "
    f"{LAMBDA_PHYSICS}"
)

print(
    f"Lambda positive : "
    f"{LAMBDA_POSITIVE}"
)


# ============================================================
# 3. LOAD TRAINING DATA
# ============================================================

print(
    "\n[3/10] Loading training data..."
)

df = pd.read_csv(
    TRAIN_PATH
)

print(
    f"Training rows : "
    f"{len(df):,}"
)


# ============================================================
# 4. PREPARE DATA + NORMALIZATION
# ============================================================

print(
    "\n[4/10] Preparing training data..."
)


# ------------------------------------------------------------
# Verify required columns
# ------------------------------------------------------------

required_columns = (
    FEATURES
    + TARGETS
    + ["compartment_id"]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise RuntimeError(
        "Missing required columns: "
        + ", ".join(
            missing_columns
        )
    )


# ------------------------------------------------------------
# Feature dataframe
# ------------------------------------------------------------

X_df = df[
    FEATURES
].copy()


# ------------------------------------------------------------
# Target dataframe
# ------------------------------------------------------------

Y_df = df[
    TARGETS
].copy()


# ------------------------------------------------------------
# Convert to float32
# ------------------------------------------------------------

X_df = X_df.astype(
    np.float32
)

Y_df = Y_df.astype(
    np.float32
)


# ------------------------------------------------------------
# Calculate normalization statistics
# ------------------------------------------------------------

X_mean = (
    X_df.mean()
    .values
    .astype(
        np.float32
    )
)

X_std = (
    X_df.std()
    .values
    .astype(
        np.float32
    )
)

Y_mean = (
    Y_df.mean()
    .values
    .astype(
        np.float32
    )
)

Y_std = (
    Y_df.std()
    .values
    .astype(
        np.float32
    )
)


# ------------------------------------------------------------
# Prevent division by zero
# ------------------------------------------------------------

X_std[
    X_std == 0
] = 1.0

Y_std[
    Y_std == 0
] = 1.0


# ------------------------------------------------------------
# Display input normalization
# ------------------------------------------------------------

print(
    "\nInput normalization:"
)

for i, feature in enumerate(
    FEATURES
):

    print(
        f"  {feature}: "
        f"mean={X_mean[i]:.6f}, "
        f"std={X_std[i]:.6f}"
    )


# ------------------------------------------------------------
# Display target normalization
# ------------------------------------------------------------

print(
    "\nTarget normalization:"
)

for i, target in enumerate(
    TARGETS
):

    print(
        f"  {target}: "
        f"mean={Y_mean[i]:.6f}, "
        f"std={Y_std[i]:.6f}"
    )


# ------------------------------------------------------------
# Normalize inputs
# ------------------------------------------------------------

X_normalized = (
    X_df.values
    - X_mean
) / X_std


# ------------------------------------------------------------
# Normalize targets
# ------------------------------------------------------------

Y_normalized = (
    Y_df.values
    - Y_mean
) / Y_std


# ------------------------------------------------------------
# Convert to tensors
# ------------------------------------------------------------

X_tensor = torch.tensor(
    X_normalized,
    dtype=torch.float32,
)

Y_tensor = torch.tensor(
    Y_normalized,
    dtype=torch.float32,
)


# ------------------------------------------------------------
# Move normalization statistics to device
# ------------------------------------------------------------

Y_mean_tensor = torch.tensor(
    Y_mean,
    dtype=torch.float32,
    device=device,
)

Y_std_tensor = torch.tensor(
    Y_std,
    dtype=torch.float32,
    device=device,
)


# ============================================================
# 5. CREATE DATALOADER
# ============================================================

print(
    "\n[5/10] Creating DataLoader..."
)


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
    f"Batch size       : "
    f"{BATCH_SIZE:,}"
)

print(
    f"Batches per epoch: "
    f"{len(loader):,}"
)


# ============================================================
# 6. BUILD MODEL
# ============================================================

print(
    "\n[6/10] Building PINNs model..."
)


model = TreeGrowthPINN(
    input_size=len(FEATURES),
    hidden_size=64,
    hidden_layers=3,
    output_size=len(TARGETS),
).to(device)


print(
    "\nArchitecture:"
)

print(model)


print(
    f"\nInput size  : "
    f"{len(FEATURES)}"
)

print(
    f"Output size : "
    f"{len(TARGETS)}"
)


# ============================================================
# 7. OPTIMIZER
# ============================================================

print(
    "\n[7/10] Creating optimizer..."
)


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
    f"Learning rate   : "
    f"{LEARNING_RATE}"
)

print(
    f"Epochs          : "
    f"{EPOCHS}"
)

print(
    f"Physics lambda  : "
    f"{LAMBDA_PHYSICS}"
)

print(
    f"Positive lambda : "
    f"{LAMBDA_POSITIVE}"
)


# ============================================================
# 8. TRAINING
# ============================================================

print(
    "\n[8/10] Starting PINNs training..."
)

history = []

best_loss = float(
    "inf"
)

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

    total_positive_loss = 0.0

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
            positive_loss,
        ) = calculate_losses(
            predictions,
            batch_Y,
            C_tensor,
            Y_mean_tensor,
            Y_std_tensor,
            LAMBDA_PHYSICS,
            LAMBDA_POSITIVE,
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

        total_positive_loss += (
            positive_loss.item()
            * batch_size_actual
        )

        sample_count += (
            batch_size_actual
        )


    # ========================================================
    # Epoch averages
    # ========================================================

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

    epoch_positive_loss = (
        total_positive_loss
        / sample_count
    )

    epoch_time = (
        time.time()
        - epoch_start
    )


    # ========================================================
    # Save history
    # ========================================================

    history.append(
        {
            "epoch": epoch,
            "total_loss": epoch_loss,
            "data_loss": epoch_data_loss,
            "physics_loss": epoch_physics_loss,
            "positive_loss": epoch_positive_loss,
            "epoch_time_seconds": epoch_time,
        }
    )


    # ========================================================
    # Progress
    # ========================================================

    if (
        epoch % PRINT_EVERY == 0
    ):

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Total: {epoch_loss:.6f} | "
            f"Data: {epoch_data_loss:.6f} | "
            f"Physics: {epoch_physics_loss:.6f} | "
            f"Positive: {epoch_positive_loss:.6f} | "
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

            "best_loss":
                best_loss,

            "features":
                FEATURES,

            "targets":
                TARGETS,

            "C":
                C_VALUE,

            "lambda_physics":
                LAMBDA_PHYSICS,

            "lambda_positive":
                LAMBDA_POSITIVE,

            "X_mean":
                X_mean,

            "X_std":
                X_std,

            "Y_mean":
                Y_mean,

            "Y_std":
                Y_std,

            "physics_loss_mode":
                "denormalized_raw_units",

            "positive_volume_constraint":
                True,

            "positive_loss_mode":
                "normalized_raw_volume",
        }


        torch.save(
            checkpoint,
            MODEL_OUTPUT_PATH,
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


# ============================================================
# 9. SAVE TRAINING HISTORY
# ============================================================

print(
    "\n[9/10] Saving training results..."
)


total_training_time = (
    time.time()
    - start_training
)


history_df = pd.DataFrame(
    history
)


history_df.to_csv(
    HISTORY_OUTPUT_PATH,
    index=False,
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

training_config = {

    "model":
        "TreeGrowthPINN",

    "variant":
        "positive_volume_constraint",

    "random_state":
        RANDOM_STATE,

    "features":
        FEATURES,

    "targets":
        TARGETS,

    "epochs_requested":
        EPOCHS,

    "epochs_completed":
        len(history),

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "lambda_physics":
        LAMBDA_PHYSICS,

    "lambda_positive":
        LAMBDA_POSITIVE,

    "physics_C":
        C_VALUE,

    "physics_loss_mode":
        "denormalized_raw_units",

    "positive_volume_constraint":
        True,

    "positive_loss_mode":
        "normalized_raw_volume",

    "physics_residual_scaling":
        "volume_std",

    "best_epoch":
        best_epoch,

    "best_loss":
        best_loss,

    "total_training_time_seconds":
        total_training_time,

    "device":
        str(device),

    "model_output":
        str(MODEL_OUTPUT_PATH),
}


with open(
    CONFIG_OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        training_config,
        file,
        indent=4,
    )


# ============================================================
# 10. FINAL OUTPUT
# ============================================================

print(
    "\n[10/10] Final output"
)

print("\n" + "=" * 75)

print(
    "PINNs POSITIVE-CONSTRAINT TRAINING COMPLETED"
)

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

print(
    f"\nLambda physics   : "
    f"{LAMBDA_PHYSICS}"
)

print(
    f"Lambda positive  : "
    f"{LAMBDA_POSITIVE}"
)

print("\nSaved:")

print(
    f"  - {MODEL_OUTPUT_PATH}"
)

print(
    f"  - {HISTORY_OUTPUT_PATH}"
)

print(
    f"  - {CONFIG_OUTPUT_PATH}"
)

print("=" * 75)