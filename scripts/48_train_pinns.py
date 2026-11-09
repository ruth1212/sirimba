from pathlib import Path
import json
import random
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# STEP 48 - TRAIN PINNs
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "model_ready" / "train.csv"

PREPROCESSING_DIR = (
    ROOT
    / "data"
    / "model_ready"
    / "pipeline"
    / "preprocessing"
)

PINNS_DIR = ROOT / "results" / "pinns"
MODEL_DIR = PINNS_DIR / "models"

PINNS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = PINNS_DIR / "pinns_config.json"
PHYSICS_CONFIG_PATH = (
    PINNS_DIR
    / "physics_coefficient_config.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

EPOCHS = 100
LEARNING_RATE = 0.001

PHYSICS_WEIGHT = 0.10

VOLUME_MIN_AGE = 18
VOLUME_MAX_AGE = 66

PHI_AGE = 66


# ============================================================
# START
# ============================================================

print("=" * 80)
print("STEP 48 - TRAIN PINNs")
print("=" * 80)

print("""
NEW PINNs TRAINING
------------------
Model is trained from scratch.

DBH:
    P6-P66

Height:
    P6-P66

Volume:
    P18-P66

Physics:
    P18-P66

PHI:
    P66

Old PINNs checkpoint:
    NOT USED
""")


# ============================================================
# 1. LOAD CONFIGURATION
# ============================================================

print("\n[1/10] Loading configurations...")

if not CONFIG_PATH.exists():
    raise FileNotFoundError(CONFIG_PATH)

if not PHYSICS_CONFIG_PATH.exists():
    raise FileNotFoundError(PHYSICS_CONFIG_PATH)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    pinns_config = json.load(f)

with open(PHYSICS_CONFIG_PATH, "r", encoding="utf-8") as f:
    physics_config = json.load(f)

C_VALUE = float(
    physics_config["coefficient"]
)

print(
    f"Physics coefficient C: "
    f"{C_VALUE:.12f}"
)

if C_VALUE <= 0:
    raise ValueError(
        "Physics coefficient must be positive."
    )

print("Configuration: PASS")


# ============================================================
# 2. DEVICE
# ============================================================

print("\n[2/10] Checking PyTorch device...")

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")

if device.type == "cuda":
    print(
        f"CUDA device: "
        f"{torch.cuda.get_device_name(0)}"
    )
else:
    print("CUDA unavailable - CPU training.")

print("Device: PASS")


# ============================================================
# 3. LOAD TRAIN DATA
# ============================================================

print("\n[3/10] Loading official training dataset...")

if not TRAIN_PATH.exists():
    raise FileNotFoundError(TRAIN_PATH)

train = pd.read_csv(TRAIN_PATH)

print(
    f"Training rows: {len(train):,}"
)

missing = [
    col
    for col in FEATURES + TARGETS
    if col not in train.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

print("Training schema: PASS")


# ============================================================
# 4. PREPARE TRAINING DATA
# ============================================================

print("\n[4/10] Preparing training tensors...")

# ------------------------------------------------------------
# DBH / HEIGHT
# ------------------------------------------------------------

feature_values = train[FEATURES].to_numpy(
    dtype=np.float32
)

dbh_height_targets = train[
    ["dbh_cm", "height_m"]
].to_numpy(
    dtype=np.float32
)

# ------------------------------------------------------------
# VOLUME
# ------------------------------------------------------------

volume_mask_np = (
    train["age_months"].to_numpy()
    >= VOLUME_MIN_AGE
)

volume_mask_np &= (
    train["age_months"].to_numpy()
    <= VOLUME_MAX_AGE
)

print(
    f"DBH/Height rows: "
    f"{len(feature_values):,}"
)

print(
    f"Volume rows: "
    f"{volume_mask_np.sum():,}"
)

if volume_mask_np.sum() == 0:
    raise ValueError(
        "No P18-P66 volume rows found."
    )

# ------------------------------------------------------------
# SCALE FEATURES
# ------------------------------------------------------------

feature_scaler = joblib.load(
    PREPROCESSING_DIR
    / "feature_scaler.joblib"
)

target_scalers = joblib.load(
    PREPROCESSING_DIR
    / "target_scaler.joblib"
)

feature_scaled = feature_scaler.transform(
    train[FEATURES]
).astype(np.float32)


# ------------------------------------------------------------
# SCALE TARGETS
# ------------------------------------------------------------

dbh_scaled = target_scalers[
    "dbh_cm"
].transform(
    train[["dbh_cm"]]
).astype(np.float32)

height_scaled = target_scalers[
    "height_m"
].transform(
    train[["height_m"]]
).astype(np.float32)

volume_scaled = target_scalers[
    "volume_m3_tree"
].transform(
    train.loc[
        volume_mask_np,
        ["volume_m3_tree"]
    ]
).astype(np.float32)


# ------------------------------------------------------------
# TENSORS
# ------------------------------------------------------------

X = torch.tensor(
    feature_scaled,
    dtype=torch.float32,
    device=device,
)

Y_dbh = torch.tensor(
    dbh_scaled,
    dtype=torch.float32,
    device=device,
)

Y_height = torch.tensor(
    height_scaled,
    dtype=torch.float32,
    device=device,
)

volume_indices = np.where(
    volume_mask_np
)[0]

X_volume = torch.tensor(
    feature_scaled[volume_indices],
    dtype=torch.float32,
    device=device,
)

Y_volume = torch.tensor(
    volume_scaled,
    dtype=torch.float32,
    device=device,
)

ages_volume = torch.tensor(
    train.iloc[
        volume_indices
    ]["age_months"].to_numpy(),
    dtype=torch.int64,
    device=device,
)

print(
    f"X shape: {tuple(X.shape)}"
)

print(
    f"X_volume shape: "
    f"{tuple(X_volume.shape)}"
)

print("Training tensors: PASS")


# ============================================================
# 5. MODEL
# ============================================================

print("\n[5/10] Building new PINNs model...")


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
            1
        )

        self.volume_head = nn.Linear(
            64,
            1
        )

        self.volume_activation = nn.Softplus()

    def forward(self, x):

        hidden = self.hidden(x)

        dbh = self.dbh_head(hidden)

        height = self.height_head(hidden)

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


model = TreeGrowthPINNs().to(device)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Trainable parameters: "
    f"{parameter_count:,}"
)

print("New model created: PASS")


# ============================================================
# 6. LOSS FUNCTIONS
# ============================================================

print("\n[6/10] Preparing losses...")


def inverse_targets(
    y_scaled
):
    """
    Convert scaled predictions into physical units.
    """

    dbh = (
        y_scaled[:, 0:1]
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

    height = (
        y_scaled[:, 1:2]
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

    volume = (
        y_scaled[:, 2:3]
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

    return torch.cat(
        [dbh, height, volume],
        dim=1,
    )


def physics_volume(
    dbh_cm,
    height_m,
):
    return (
        C_VALUE
        * dbh_cm.pow(2)
        * height_m
    )


criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


print(
    f"Physics weight: "
    f"{PHYSICS_WEIGHT}"
)

print(
    f"Learning rate: "
    f"{LEARNING_RATE}"
)

print("Loss functions: PASS")


# ============================================================
# 7. TRAINING LOOP
# ============================================================

print("\n[7/10] Starting PINNs training...")
print("=" * 80)

start_time = time.time()

history = []

best_loss = float("inf")
best_epoch = None

best_state = None


for epoch in range(
    1,
    EPOCHS + 1
):

    optimizer.zero_grad()

    # --------------------------------------------------------
    # FULL FORWARD PASS
    # --------------------------------------------------------

    predictions = model(X)

    # --------------------------------------------------------
    # SUPERVISED DBH
    # --------------------------------------------------------

    dbh_loss = criterion(
        predictions[:, 0:1],
        Y_dbh,
    )

    # --------------------------------------------------------
    # SUPERVISED HEIGHT
    # --------------------------------------------------------

    height_loss = criterion(
        predictions[:, 1:2],
        Y_height,
    )

    # --------------------------------------------------------
    # VOLUME P18-P66 ONLY
    # --------------------------------------------------------

    volume_predictions = model(
        X_volume
    )[:, 2:3]

    volume_loss = criterion(
        volume_predictions,
        Y_volume,
    )

    # --------------------------------------------------------
    # PHYSICS LOSS
    # --------------------------------------------------------

    physical_volume_predictions = (
        inverse_targets(
            torch.cat(
                [
                    model(X_volume)[:, 0:1],
                    model(X_volume)[:, 1:2],
                    volume_predictions,
                ],
                dim=1,
            )
        )[:, 2:3]
    )

    physical_dbh = (
        inverse_targets(
            torch.cat(
                [
                    model(X_volume)[:, 0:1],
                    model(X_volume)[:, 1:2],
                    volume_predictions,
                ],
                dim=1,
            )
        )[:, 0:1]
    )

    physical_height = (
        inverse_targets(
            torch.cat(
                [
                    model(X_volume)[:, 0:1],
                    model(X_volume)[:, 1:2],
                    volume_predictions,
                ],
                dim=1,
            )
        )[:, 1:2]
    )

    expected_physics_volume = physics_volume(
        physical_dbh,
        physical_height,
    )

    physics_loss = criterion(
        physical_volume_predictions,
        expected_physics_volume,
    )

    # --------------------------------------------------------
    # TOTAL LOSS
    # --------------------------------------------------------

    total_loss = (
        dbh_loss
        + height_loss
        + volume_loss
        + PHYSICS_WEIGHT * physics_loss
    )

    if not torch.isfinite(total_loss):
        raise RuntimeError(
            f"Non-finite loss at epoch {epoch}"
        )

    total_loss.backward()

    optimizer.step()

    # --------------------------------------------------------
    # RECORD
    # --------------------------------------------------------

    total_value = float(
        total_loss.detach().cpu()
    )

    dbh_value = float(
        dbh_loss.detach().cpu()
    )

    height_value = float(
        height_loss.detach().cpu()
    )

    volume_value = float(
        volume_loss.detach().cpu()
    )

    physics_value = float(
        physics_loss.detach().cpu()
    )

    history.append(
        {
            "epoch": epoch,
            "total_loss": total_value,
            "dbh_loss": dbh_value,
            "height_loss": height_value,
            "volume_loss": volume_value,
            "physics_loss": physics_value,
        }
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if total_value < best_loss:

        best_loss = total_value
        best_epoch = epoch

        best_state = {
            key: value.detach().cpu().clone()
            for key, value
            in model.state_dict().items()
        }

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        epoch == 1
        or epoch % 10 == 0
        or epoch == EPOCHS
    ):

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Total={total_value:.8f} | "
            f"DBH={dbh_value:.8f} | "
            f"Height={height_value:.8f} | "
            f"Volume={volume_value:.8f} | "
            f"Physics={physics_value:.8f} | "
            f"Time={elapsed:.1f}s"
        )


# ============================================================
# 8. RESTORE BEST MODEL
# ============================================================

print("\n[8/10] Restoring best model...")

if best_state is None:
    raise RuntimeError(
        "Best model state was never created."
    )

model.load_state_dict(
    best_state
)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best total loss: "
    f"{best_loss:.10f}"
)

print("Best model restored: PASS")


# ============================================================
# 9. SAVE MODEL + HISTORY
# ============================================================

print("\n[9/10] Saving PINNs model...")

MODEL_PATH = (
    MODEL_DIR
    / "pinns_growth_model.pt"
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "architecture": {
            "input_size": 6,
            "hidden_layers": [
                64,
                64,
                64,
            ],
            "activation": "Tanh",
            "output_size": 3,
            "volume_activation":
                "Softplus",
        },

        "features": FEATURES,

        "targets": TARGETS,

        "physics": {
            "coefficient": C_VALUE,
            "formula":
                "V = C * DBH^2 * Height",
            "min_age_months": 18,
            "max_age_months": 66,
        },

        "training": {
            "epochs": EPOCHS,
            "best_epoch": best_epoch,
            "learning_rate":
                LEARNING_RATE,
            "physics_weight":
                PHYSICS_WEIGHT,
            "random_seed": SEED,
        },
    },
    MODEL_PATH,
)


history_path = (
    PINNS_DIR
    / "training_history.csv"
)

pd.DataFrame(history).to_csv(
    history_path,
    index=False,
)

training_report = {
    "status": "PASS",

    "epochs_requested": EPOCHS,

    "epochs_completed": EPOCHS,

    "best_epoch": best_epoch,

    "best_total_loss": best_loss,

    "learning_rate": LEARNING_RATE,

    "physics_weight": PHYSICS_WEIGHT,

    "physics_coefficient": C_VALUE,

    "train_rows": int(len(train)),

    "volume_train_rows": int(
        len(volume_indices)
    ),

    "parameter_count":
        parameter_count,

    "device": str(device),

    "old_checkpoint_used": False,

    "test_data_used": False,

    "dataset_modified": False,
}

report_path = (
    PINNS_DIR
    / "training_report.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        training_report,
        f,
        indent=2,
    )

print(
    f"Model saved: {MODEL_PATH}"
)

print(
    f"History saved: {history_path}"
)

print(
    f"Report saved: {report_path}"
)


# ============================================================
# 10. FINAL
# ============================================================

elapsed_total = (
    time.time()
    - start_time
)

print("\n" + "=" * 80)
print("STEP 48 STATUS: PASS")
print("=" * 80)

print(
    f"""
PINNs training completed.

Epochs:
    {EPOCHS}

Best epoch:
    {best_epoch}

Best total loss:
    {best_loss:.10f}

Training time:
    {elapsed_total:.1f} seconds

Physics coefficient:
    {C_VALUE:.12f}

Physics weight:
    {PHYSICS_WEIGHT}

Volume training:
    P18-P66

PHI:
    P66

Old PINNs checkpoint:
    NOT USED

Test data:
    NOT USED

Dataset:
    NOT MODIFIED
"""
)

print("=" * 80)