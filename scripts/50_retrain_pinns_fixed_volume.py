from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib


# ============================================================
# STEP 50
# PINNs RETRAIN — FIXED PHYSICAL VOLUME OUTPUT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PIPELINE_DIR = ROOT / "data" / "model_ready" / "pipeline"
PREPROCESSING_DIR = PIPELINE_DIR / "preprocessing"

TRAIN_DBH_HEIGHT_PATH = (
    PIPELINE_DIR / "train_dbh_height.csv"
)

TRAIN_VOLUME_PATH = (
    PIPELINE_DIR / "train_volume_p18.csv"
)

TARGET_SCALER_PATH = (
    PREPROCESSING_DIR / "target_scaler.joblib"
)

PHYSICS_CONFIG_PATH = (
    ROOT
    / "results"
    / "pinns"
    / "physics_coefficient_config.json"
)

OUTPUT_DIR = ROOT / "results" / "pinns"
MODEL_DIR = OUTPUT_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_PATH = (
    MODEL_DIR
    / "pinns_growth_model_fixed_volume.pt"
)

HISTORY_PATH = (
    OUTPUT_DIR
    / "training_history_fixed_volume.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "training_report_fixed_volume.json"
)


# ============================================================
# CONFIG
# ============================================================

RANDOM_SEED = 42

EPOCHS = 100
LEARNING_RATE = 0.001
PHYSICS_WEIGHT = 0.10

HIDDEN_SIZE = 64

DEVICE = torch.device("cpu")


# ============================================================
# SCHEMA
# ============================================================

FEATURE_COLUMNS = [
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha",
]

DBH_HEIGHT_COLUMNS = [
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha",
    "dbh_cm",
    "height_m",
    "compartment_id",
]

VOLUME_COLUMNS = [
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha",
    "volume_m3_tree",
    "compartment_id",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# MODEL
# ============================================================

class PINNsGrowthModel(nn.Module):

    def __init__(
        self,
        input_size=6,
        hidden_size=64
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(
                input_size,
                hidden_size
            ),
            nn.Tanh(),

            nn.Linear(
                hidden_size,
                hidden_size
            ),
            nn.Tanh(),

            nn.Linear(
                hidden_size,
                hidden_size
            ),
            nn.Tanh(),
        )

        self.dbh_head = nn.Linear(
            hidden_size,
            1
        )

        self.height_head = nn.Linear(
            hidden_size,
            1
        )

        # IMPORTANT:
        # This head produces an unconstrained latent value.
        # Softplus is applied AFTER converting this latent
        # into the physical-volume representation.
        self.volume_head = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        hidden = self.shared(x)

        dbh_scaled = self.dbh_head(hidden)

        height_scaled = self.height_head(hidden)

        volume_latent = self.volume_head(hidden)

        return (
            dbh_scaled,
            height_scaled,
            volume_latent
        )


# ============================================================
# LOAD TARGET SCALERS
# ============================================================

def load_target_scaler():

    scaler_data = joblib.load(
        TARGET_SCALER_PATH
    )

    required = [
        "dbh_cm",
        "height_m",
        "volume_m3_tree",
    ]

    for target in required:

        if target not in scaler_data:
            raise RuntimeError(
                f"Missing target scaler: {target}"
            )

        scaler = scaler_data[target]

        if not hasattr(scaler, "mean_"):
            raise RuntimeError(
                f"Target scaler for {target} "
                f"does not contain mean_."
            )

        if not hasattr(scaler, "scale_"):
            raise RuntimeError(
                f"Target scaler for {target} "
                f"does not contain scale_."
            )

        if len(scaler.mean_) != 1:
            raise RuntimeError(
                f"Unexpected scaler dimension "
                f"for {target}: {len(scaler.mean_)}"
            )

    return scaler_data


# ============================================================
# LOAD PHYSICS COEFFICIENT
# ============================================================

def load_physics_coefficient():

    with open(
        PHYSICS_CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        config = json.load(f)

    if "coefficient" not in config:
        raise RuntimeError(
            "Physics coefficient key 'coefficient' "
            "was not found in config."
        )

    coefficient = float(
        config["coefficient"]
    )

    if not np.isfinite(coefficient):
        raise RuntimeError(
            "Physics coefficient is not finite."
        )

    if coefficient <= 0:
        raise RuntimeError(
            "Physics coefficient must be > 0."
        )

    return coefficient


# ============================================================
# START
# ============================================================

print("=" * 70)
print(
    "STEP 50 — PINNs RETRAIN "
    "WITH FIXED PHYSICAL VOLUME"
)
print("=" * 70)


# ============================================================
# 1. LOAD ACTUAL PIPELINE FILES
# ============================================================

print("\n[1/10] Loading pipeline files...")

dbh_height = pd.read_csv(
    TRAIN_DBH_HEIGHT_PATH
)

volume = pd.read_csv(
    TRAIN_VOLUME_PATH
)

print(
    f"DBH + Height : {dbh_height.shape}"
)

print(
    f"Volume P18-P66: {volume.shape}"
)


# ============================================================
# 2. VERIFY SCHEMAS
# ============================================================

print("\n[2/10] Verifying schemas...")

if list(dbh_height.columns) != DBH_HEIGHT_COLUMNS:

    raise RuntimeError(
        "\nDBH/Height pipeline schema mismatch.\n"
        f"Expected: {DBH_HEIGHT_COLUMNS}\n"
        f"Found   : {list(dbh_height.columns)}"
    )


if list(volume.columns) != VOLUME_COLUMNS:

    raise RuntimeError(
        "\nVolume pipeline schema mismatch.\n"
        f"Expected: {VOLUME_COLUMNS}\n"
        f"Found   : {list(volume.columns)}"
    )


print(
    "DBH/Height pipeline schema : PASS"
)

print(
    "Volume pipeline schema     : PASS"
)


# ============================================================
# 3. VERIFY AGE RANGES
# ============================================================

print("\n[3/10] Verifying age ranges...")

dbh_height_ages = sorted(
    dbh_height["age_months"]
    .unique()
    .tolist()
)

volume_ages = sorted(
    volume["age_months"]
    .unique()
    .tolist()
)

print(
    f"DBH/Height ages : {dbh_height_ages}"
)

print(
    f"Volume ages     : {volume_ages}"
)


EXPECTED_FULL_AGES = [
    6,
    12,
    18,
    24,
    30,
    36,
    42,
    48,
    54,
    60,
    66,
]

EXPECTED_VOLUME_AGES = [
    18,
    24,
    30,
    36,
    42,
    48,
    54,
    60,
    66,
]


if dbh_height_ages != EXPECTED_FULL_AGES:

    raise RuntimeError(
        "DBH/Height age range is not P6-P66."
    )


if volume_ages != EXPECTED_VOLUME_AGES:

    raise RuntimeError(
        "Volume age range is not P18-P66."
    )


print(
    "DBH/Height age range P6-P66 : PASS"
)

print(
    "Volume age range P18-P66    : PASS"
)


# ============================================================
# 4. VERIFY ROW COUNTS
# ============================================================

print("\n[4/10] Verifying row counts...")

if len(dbh_height) != 174960:

    raise RuntimeError(
        "Unexpected DBH/Height training row count."
    )


if len(volume) != 142560:

    raise RuntimeError(
        "Unexpected volume training row count."
    )


print(
    "DBH/Height rows : 174,960 PASS"
)

print(
    "Volume rows     : 142,560 PASS"
)


# ============================================================
# 5. VERIFY FEATURE / TARGET DATA
# ============================================================

print("\n[5/10] Preparing model matrices...")

X_full_df = dbh_height[
    FEATURE_COLUMNS
].copy()

X_volume_df = volume[
    FEATURE_COLUMNS
].copy()

Y_dbh_df = dbh_height[
    ["dbh_cm"]
].copy()

Y_height_df = dbh_height[
    ["height_m"]
].copy()

Y_volume_df = volume[
    ["volume_m3_tree"]
].copy()


# Verify no missing values

if X_full_df.isna().any().any():

    raise RuntimeError(
        "Missing values found in full features."
    )


if X_volume_df.isna().any().any():

    raise RuntimeError(
        "Missing values found in volume features."
    )


if Y_dbh_df.isna().any().any():

    raise RuntimeError(
        "Missing DBH target values."
    )


if Y_height_df.isna().any().any():

    raise RuntimeError(
        "Missing height target values."
    )


if Y_volume_df.isna().any().any():

    raise RuntimeError(
        "Missing volume target values."
    )


print(
    "Six model features : PASS"
)

print(
    "DBH target          : PASS"
)

print(
    "Height target       : PASS"
)

print(
    "Volume target       : PASS"
)


# ============================================================
# 6. VERIFY COMPARTMENT SPLIT / ALIGNMENT
# ============================================================

print("\n[6/10] Verifying compartment metadata...")

train_compartments_dbh = sorted(
    dbh_height[
        "compartment_id"
    ].unique()
    .tolist()
)

train_compartments_volume = sorted(
    volume[
        "compartment_id"
    ].unique()
    .tolist()
)


if train_compartments_dbh != train_compartments_volume:

    raise RuntimeError(
        "Volume and DBH/Height compartment sets differ."
    )


print(
    f"Training compartments: "
    f"{len(train_compartments_dbh)}"
)

print(
    "Compartment consistency : PASS"
)


# ============================================================
# 7. LOAD SCALERS + PHYSICS
# ============================================================

print("\n[7/10] Loading scalers and physics...")

scaler_data = load_target_scaler()

dbh_scaler = scaler_data["dbh_cm"]
height_scaler = scaler_data["height_m"]
volume_scaler = scaler_data["volume_m3_tree"]

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

C_PHYSICS = load_physics_coefficient()


print(
    f"DBH    mean={dbh_mean:.6f}, "
    f"scale={dbh_scale:.6f}"
)

print(
    f"Height mean={height_mean:.6f}, "
    f"scale={height_scale:.6f}"
)

print(
    f"Volume mean={volume_mean:.6f}, "
    f"scale={volume_scale:.6f}"
)

print(
    f"Physics C={C_PHYSICS:.12f}"
)


# ============================================================
# 8. BUILD TENSORS
# ============================================================

print("\n[8/10] Building tensors...")

X_full = torch.tensor(
    X_full_df.values,
    dtype=torch.float32,
    device=DEVICE
)

X_volume = torch.tensor(
    X_volume_df.values,
    dtype=torch.float32,
    device=DEVICE
)

Y_dbh = torch.tensor(
    Y_dbh_df.values,
    dtype=torch.float32,
    device=DEVICE
)

Y_height = torch.tensor(
    Y_height_df.values,
    dtype=torch.float32,
    device=DEVICE
)

Y_volume = torch.tensor(
    Y_volume_df.values,
    dtype=torch.float32,
    device=DEVICE
)


print(
    f"X full   : {tuple(X_full.shape)}"
)

print(
    f"X volume : {tuple(X_volume.shape)}"
)

print(
    f"Y DBH    : {tuple(Y_dbh.shape)}"
)

print(
    f"Y Height : {tuple(Y_height.shape)}"
)

print(
    f"Y Volume : {tuple(Y_volume.shape)}"
)


# ============================================================
# 9. MODEL + TRAINING
# ============================================================

print("\n[9/10] Building and training PINNs...")

model = PINNsGrowthModel(
    input_size=6,
    hidden_size=HIDDEN_SIZE
).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.MSELoss()

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "Architecture : 6 -> 64 -> 64 -> 64 -> 3"
)

print(
    f"Parameters   : {parameter_count:,}"
)

print(
    f"Device       : {DEVICE}"
)

print(
    f"Epochs       : {EPOCHS}"
)

print(
    f"Learning rate: {LEARNING_RATE}"
)

print(
    f"Physics λ    : {PHYSICS_WEIGHT}"
)

print()
print(
    "Volume transformation:"
)

print(
    "latent -> physical volume -> Softplus"
)

print(
    "Physics: physical units, P18-P66"
)

print("-" * 70)


start_time = time.time()

history = []

best_loss = float("inf")
best_epoch = None


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    optimizer.zero_grad()


    # --------------------------------------------------------
    # DBH + HEIGHT
    # --------------------------------------------------------

    dbh_scaled_pred, height_scaled_pred, _ = model(
        X_full
    )

    dbh_target_scaled = (
        Y_dbh - dbh_mean
    ) / dbh_scale

    height_target_scaled = (
        Y_height - height_mean
    ) / height_scale


    loss_dbh = criterion(
        dbh_scaled_pred,
        dbh_target_scaled
    )

    loss_height = criterion(
        height_scaled_pred,
        height_target_scaled
    )


    # --------------------------------------------------------
    # VOLUME P18-P66
    # --------------------------------------------------------

    # Volume head receives the SAME six features,
    # but only P18-P66 rows are supplied.

    _, _, volume_latent = model(
        X_volume
    )


    # IMPORTANT:
    # Softplus is applied to physical volume.
    #
    # We first map the latent output to physical scale,
    # then impose positivity.

    volume_raw_physical = (
        volume_latent
        * volume_scale
        + volume_mean
    )


    volume_physical_pred = F.softplus(
        volume_raw_physical
    )


    volume_target_scaled = (
        Y_volume - volume_mean
    ) / volume_scale


    volume_pred_scaled = (
        volume_physical_pred
        - volume_mean
    ) / volume_scale


    loss_volume = criterion(
        volume_pred_scaled,
        volume_target_scaled
    )


    # --------------------------------------------------------
    # PHYSICS
    # --------------------------------------------------------

    dbh_physical_pred = (
        dbh_scaled_pred
        * dbh_scale
        + dbh_mean
    )

    height_physical_pred = (
        height_scaled_pred
        * height_scale
        + height_mean
    )


    age_array = dbh_height[
    "age_months"
    ].to_numpy(copy=True)

    physics_mask = (
        (age_array >= 18)
        & (age_array <= 66)
    )


    physics_mask_tensor = torch.tensor(
        physics_mask,
        dtype=torch.bool,
        device=DEVICE
    )


    dbh_physics = (
        dbh_physical_pred[
            physics_mask_tensor
        ]
    )

    height_physics = (
        height_physical_pred[
            physics_mask_tensor
        ]
    )


    expected_volume_physical = (
        C_PHYSICS
        * torch.square(
            dbh_physics
        )
        * height_physics
    )


    physics_residual = (
        volume_physical_pred
        - expected_volume_physical
    )


    physics_residual_scaled = (
        physics_residual
        / volume_scale
    )


    loss_physics = torch.mean(
        torch.square(
            physics_residual_scaled
        )
    )


    # --------------------------------------------------------
    # TOTAL
    # --------------------------------------------------------

    total_loss = (
        loss_dbh
        + loss_height
        + loss_volume
        + (
            PHYSICS_WEIGHT
            * loss_physics
        )
    )


    total_loss.backward()

    optimizer.step()


    total_value = float(
        total_loss.item()
    )


    history.append(
        {
            "epoch": epoch,
            "total_loss": total_value,
            "loss_dbh": float(
                loss_dbh.item()
            ),
            "loss_height": float(
                loss_height.item()
            ),
            "loss_volume": float(
                loss_volume.item()
            ),
            "loss_physics": float(
                loss_physics.item()
            ),
        }
    )


    if total_value < best_loss:

        best_loss = total_value
        best_epoch = epoch


    if (
        epoch == 1
        or epoch % 10 == 0
        or epoch == EPOCHS
    ):

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Total {total_value:.8f} | "
            f"DBH {loss_dbh.item():.8f} | "
            f"Height {loss_height.item():.8f} | "
            f"Volume {loss_volume.item():.8f} | "
            f"Physics {loss_physics.item():.8f}"
        )


training_time = (
    time.time()
    - start_time
)


# ============================================================
# 10. SAVE + SANITY CHECK
# ============================================================

print("\n[10/10] Saving and validating...")

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "architecture": {
            "input_size": 6,
            "hidden_size": 64,
            "hidden_layers": 3,
        },

        "feature_columns":
            FEATURE_COLUMNS,

        "target_columns": [
            "dbh_cm",
            "height_m",
            "volume_m3_tree",
        ],

        "volume_output":
            "physical_positive_volume",

        "volume_activation":
            "Softplus",

        "physics_coefficient":
            C_PHYSICS,

        "physics_start_age":
            18,

        "physics_end_age":
            66,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "physics_weight":
            PHYSICS_WEIGHT,

        "random_seed":
            RANDOM_SEED,
    },
    CHECKPOINT_PATH
)


history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)


# Final forward pass

model.eval()

with torch.no_grad():

    dbh_scaled_pred, height_scaled_pred, _ = model(
        X_full
    )

    _, _, volume_latent = model(
        X_volume
    )


    volume_raw_physical = (
        volume_latent
        * volume_scale
        + volume_mean
    )

    volume_physical_pred = F.softplus(
        volume_raw_physical
    )


dbh_physical_pred = (
    dbh_scaled_pred
    * dbh_scale
    + dbh_mean
)

height_physical_pred = (
    height_scaled_pred
    * height_scale
    + height_mean
)


volume_np = (
    volume_physical_pred
    .cpu()
    .numpy()
    .reshape(-1)
)

dbh_np = (
    dbh_physical_pred
    .cpu()
    .numpy()
    .reshape(-1)
)

height_np = (
    height_physical_pred
    .cpu()
    .numpy()
    .reshape(-1)
)


negative_volume_count = int(
    np.sum(volume_np < 0)
)

zero_volume_count = int(
    np.sum(volume_np == 0)
)


if negative_volume_count != 0:

    raise RuntimeError(
        "Negative physical volume predictions detected."
    )


if zero_volume_count != 0:

    raise RuntimeError(
        "Zero physical volume predictions detected."
    )


report = {

    "status": "PASS",

    "description":
        "PINNs retraining with physical-space "
        "positive volume output.",

    "input_files": {
        "dbh_height":
            str(TRAIN_DBH_HEIGHT_PATH),

        "volume":
            str(TRAIN_VOLUME_PATH),
    },

    "rows": {
        "dbh_height":
            int(len(dbh_height)),

        "volume_p18_p66":
            int(len(volume)),
    },

    "ages": {
        "dbh_height":
            dbh_height_ages,

        "volume":
            volume_ages,
    },

    "architecture": {
        "features":
            FEATURE_COLUMNS,

        "targets": [
            "dbh_cm",
            "height_m",
            "volume_m3_tree",
        ],

        "hidden_size":
            HIDDEN_SIZE,

        "hidden_layers":
            3,

        "parameter_count":
            parameter_count,
    },

    "training": {
        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "physics_weight":
            PHYSICS_WEIGHT,

        "best_epoch":
            best_epoch,

        "best_total_loss":
            best_loss,

        "training_time_seconds":
            training_time,
    },

    "physics": {
        "coefficient":
            C_PHYSICS,

        "start_age_months":
            18,

        "end_age_months":
            66,
    },

    "volume_output": {
        "space":
            "physical",

        "activation":
            "Softplus",

        "negative_predictions":
            negative_volume_count,

        "zero_predictions":
            zero_volume_count,

        "minimum":
            float(volume_np.min()),

        "maximum":
            float(volume_np.max()),
    },

    "sanity": {
        "dbh_min":
            float(dbh_np.min()),

        "dbh_max":
            float(dbh_np.max()),

        "height_min":
            float(height_np.min()),

        "height_max":
            float(height_np.max()),
    },

    "artifacts": {
        "checkpoint":
            str(CHECKPOINT_PATH),

        "history":
            str(HISTORY_PATH),

        "report":
            str(REPORT_PATH),
    },
}


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


print()
print("=" * 70)
print("STEP 50 COMPLETE")
print("=" * 70)

print(
    "Status          : PASS"
)

print(
    f"Best epoch      : {best_epoch}"
)

print(
    f"Best loss       : {best_loss:.8f}"
)

print(
    f"Training time   : {training_time:.2f} sec"
)

print()
print("Volume sanity:")

print(
    f"Negative        : {negative_volume_count}"
)

print(
    f"Zero            : {zero_volume_count}"
)

print(
    f"Minimum         : {volume_np.min():.8f}"
)

print(
    f"Maximum         : {volume_np.max():.8f}"
)

print()
print("Saved:")

print(
    f"- {CHECKPOINT_PATH}"
)

print(
    f"- {HISTORY_PATH}"
)

print(
    f"- {REPORT_PATH}"
)

print()
print(
    "TEST EVALUATION HAS NOT BEEN RUN."
)

print(
    "Do not compare MAPE yet."
)

print("=" * 70)