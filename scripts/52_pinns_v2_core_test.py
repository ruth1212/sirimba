from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# STEP 52
# PINNs V2 CORE TEST
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PIPELINE_DIR = (
    ROOT
    / "data"
    / "model_ready"
    / "pipeline"
)

PREPROCESSING_DIR = (
    PIPELINE_DIR
    / "preprocessing"
)

TRAIN_DBH_HEIGHT_PATH = (
    PIPELINE_DIR
    / "train_dbh_height.csv"
)

TRAIN_VOLUME_PATH = (
    PIPELINE_DIR
    / "train_volume_p18.csv"
)

TARGET_SCALER_PATH = (
    PREPROCESSING_DIR
    / "target_scaler.joblib"
)

PHYSICS_CONFIG_PATH = (
    ROOT
    / "results"
    / "pinns"
    / "physics_coefficient_config.json"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "pinns"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "pinns_v2_core_test_report.json"
)

DEVICE = torch.device("cpu")

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


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


# ============================================================
# MODEL V2
# ============================================================

class PINNsGrowthModelV2(nn.Module):

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

        self.volume_head = nn.Linear(
            hidden_size,
            1
        )


    def forward(self, x):

        hidden = self.shared(x)

        dbh_scaled = self.dbh_head(
            hidden
        )

        height_scaled = self.height_head(
            hidden
        )

        volume_scaled = self.volume_head(
            hidden
        )

        return (
            dbh_scaled,
            height_scaled,
            volume_scaled
        )


# ============================================================
# LOAD SCALERS
# ============================================================

print("=" * 70)
print("STEP 52 — PINNs V2 CORE TEST")
print("=" * 70)

print("\n[1/8] Loading scalers...")

import joblib

scaler_data = joblib.load(
    TARGET_SCALER_PATH
)

dbh_scaler = scaler_data[
    "dbh_cm"
]

height_scaler = scaler_data[
    "height_m"
]

volume_scaler = scaler_data[
    "volume_m3_tree"
]

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


# ============================================================
# PHYSICS
# ============================================================

print("\n[2/8] Loading physics coefficient...")

with open(
    PHYSICS_CONFIG_PATH,
    "r",
    encoding="utf-8"
) as f:

    physics_config = json.load(f)

C_PHYSICS = float(
    physics_config["coefficient"]
)

print(
    f"C = {C_PHYSICS:.12f}"
)


# ============================================================
# DATA
# ============================================================

print("\n[3/8] Loading model pipeline...")

dbh_height = pd.read_csv(
    TRAIN_DBH_HEIGHT_PATH
)

volume = pd.read_csv(
    TRAIN_VOLUME_PATH
)

print(
    f"DBH/Height rows : {len(dbh_height):,}"
)

print(
    f"Volume rows     : {len(volume):,}"
)


# ============================================================
# FEATURES
# ============================================================

X_full = torch.tensor(
    dbh_height[
        FEATURE_COLUMNS
    ].values,
    dtype=torch.float32
)

X_volume = torch.tensor(
    volume[
        FEATURE_COLUMNS
    ].values,
    dtype=torch.float32
)


# ============================================================
# MODEL
# ============================================================

print("\n[4/8] Building V2 model...")

model = PINNsGrowthModelV2(
    input_size=6,
    hidden_size=64
)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "Architecture : "
    "6 -> 64 -> 64 -> 64 -> 3"
)

print(
    f"Parameters   : {parameter_count:,}"
)


# ============================================================
# FORWARD TEST
# ============================================================

print("\n[5/8] Testing forward pass...")

model.eval()

with torch.no_grad():

    dbh_scaled, height_scaled, volume_scaled = (
        model(X_volume)
    )

print(
    f"DBH output    : {tuple(dbh_scaled.shape)}"
)

print(
    f"Height output : {tuple(height_scaled.shape)}"
)

print(
    f"Volume output : {tuple(volume_scaled.shape)}"
)


if dbh_scaled.shape != (
    len(volume),
    1
):

    raise RuntimeError(
        "DBH output shape failed."
    )


if height_scaled.shape != (
    len(volume),
    1
):

    raise RuntimeError(
        "Height output shape failed."
    )


if volume_scaled.shape != (
    len(volume),
    1
):

    raise RuntimeError(
        "Volume output shape failed."
    )


print(
    "Forward shapes : PASS"
)


# ============================================================
# PHYSICAL RECONSTRUCTION
# ============================================================

print("\n[6/8] Testing physical reconstruction...")

dbh_physical = (
    dbh_scaled
    * dbh_scale
    + dbh_mean
)

height_physical = (
    height_scaled
    * height_scale
    + height_mean
)

volume_physical = (
    volume_scaled
    * volume_scale
    + volume_mean
)


print(
    f"Initial DBH physical range: "
    f"{dbh_physical.min().item():.6f} - "
    f"{dbh_physical.max().item():.6f}"
)

print(
    f"Initial Height physical range: "
    f"{height_physical.min().item():.6f} - "
    f"{height_physical.max().item():.6f}"
)

print(
    f"Initial Volume physical range: "
    f"{volume_physical.min().item():.6f} - "
    f"{volume_physical.max().item():.6f}"
)


# ============================================================
# PHYSICS TEST
# ============================================================

print("\n[7/8] Testing physics computation...")

physics_expected_volume = (
    C_PHYSICS
    * torch.square(
        dbh_physical
    )
    * height_physical
)

physics_residual = (
    volume_physical
    - physics_expected_volume
)

physics_loss = torch.mean(
    torch.square(
        physics_residual
        / volume_scale
    )
)


print(
    f"Physics expected range: "
    f"{physics_expected_volume.min().item():.6f} - "
    f"{physics_expected_volume.max().item():.6f}"
)

print(
    f"Physics loss: "
    f"{physics_loss.item():.8f}"
)


if not torch.isfinite(
    physics_loss
):

    raise RuntimeError(
        "Physics loss is not finite."
    )


print(
    "Physics computation : PASS"
)


# ============================================================
# BACKWARD TEST
# ============================================================

print("\n[8/8] Testing backward pass...")

model.train()

dbh_scaled, height_scaled, volume_scaled = (
    model(X_volume[:512])
)

dbh_target = torch.zeros_like(
    dbh_scaled
)

height_target = torch.zeros_like(
    height_scaled
)

volume_target = torch.zeros_like(
    volume_scaled
)


loss_dbh = torch.mean(
    torch.square(
        dbh_scaled
        - dbh_target
    )
)

loss_height = torch.mean(
    torch.square(
        height_scaled
        - height_target
    )
)

loss_volume = torch.mean(
    torch.square(
        volume_scaled
        - volume_target
    )
)


dbh_phys = (
    dbh_scaled
    * dbh_scale
    + dbh_mean
)

height_phys = (
    height_scaled
    * height_scale
    + height_mean
)

volume_phys = (
    volume_scaled
    * volume_scale
    + volume_mean
)


physics_expected = (
    C_PHYSICS
    * torch.square(
        dbh_phys
    )
    * height_phys
)


physics_loss = torch.mean(
    torch.square(
        (
            volume_phys
            - physics_expected
        )
        / volume_scale
    )
)


total_loss = (
    loss_dbh
    + loss_height
    + loss_volume
    + 0.1 * physics_loss
)


total_loss.backward()


gradient_count = 0

nonfinite_gradient_count = 0

for parameter in model.parameters():

    if parameter.grad is not None:

        gradient_count += 1

        if not torch.isfinite(
            parameter.grad
        ).all():

            nonfinite_gradient_count += 1


print(
    f"Parameters with gradients: "
    f"{gradient_count}"
)

print(
    f"Non-finite gradients: "
    f"{nonfinite_gradient_count}"
)


if gradient_count == 0:

    raise RuntimeError(
        "No gradients detected."
    )


if nonfinite_gradient_count != 0:

    raise RuntimeError(
        "Non-finite gradients detected."
    )


print(
    "Backward pass : PASS"
)


# ============================================================
# SAVE REPORT
# ============================================================

report = {

    "status": "PASS",

    "architecture": {
        "input_size": 6,
        "hidden_size": 64,
        "hidden_layers": 3,
        "parameter_count":
            parameter_count
    },

    "volume_output": {
        "space":
            "scaled_direct_output",
        "physical_reconstruction":
            "volume_scaled * scale + mean",
        "positivity_activation":
            "none_in_supervised_head"
    },

    "physics": {
        "coefficient":
            C_PHYSICS,
        "age_min":
            18,
        "age_max":
            66,
        "loss_weight":
            0.1
    },

    "tests": {
        "forward":
            True,
        "physical_reconstruction":
            True,
        "physics":
            True,
        "backward":
            True,
        "finite_gradients":
            True
    },

    "note": (
        "This is a core architecture test only. "
        "No model training was performed."
    )
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


print("\n" + "=" * 70)
print("STEP 52 CORE TEST COMPLETE")
print("=" * 70)

print(
    "STATUS: PASS"
)

print(
    "No training performed."
)

print(
    f"Report: {REPORT_PATH}"
)

print("=" * 70)