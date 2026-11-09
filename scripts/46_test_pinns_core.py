from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib


# ============================================================
# STEP 46 - TEST PINNs CORE + PHYSICS LOSS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "model_ready" / "train.csv"
TEST_PATH = ROOT / "data" / "model_ready" / "test.csv"

PREPROCESSING_DIR = (
    ROOT
    / "data"
    / "model_ready"
    / "pipeline"
    / "preprocessing"
)

CONFIG_PATH = (
    ROOT
    / "results"
    / "pinns"
    / "pinns_config.json"
)

RESULT_DIR = ROOT / "results" / "pinns"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# OFFICIAL PROJECT DEFINITION
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
# LOAD CONFIG
# ============================================================

print("=" * 80)
print("STEP 46 - PINNs CORE + PHYSICS LOSS TEST")
print("=" * 80)

print("\n[1/9] Loading PINNs configuration...")

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Missing PINNs configuration: {CONFIG_PATH}"
    )

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

print("Configuration loaded: PASS")


# ============================================================
# CHECK PYTORCH
# ============================================================

print("\n[2/9] Checking PyTorch...")

print(f"PyTorch version: {torch.__version__}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")

if device.type == "cuda":
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available - using CPU.")

print("PyTorch: PASS")


# ============================================================
# LOAD DATA
# ============================================================

print("\n[3/9] Loading official train/test data...")

if not TRAIN_PATH.exists():
    raise FileNotFoundError(TRAIN_PATH)

if not TEST_PATH.exists():
    raise FileNotFoundError(TEST_PATH)

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(f"Train rows: {len(train):,}")
print(f"Test rows : {len(test):,}")

for col in FEATURES + TARGETS:
    if col not in train.columns:
        raise ValueError(f"Missing train column: {col}")

    if col not in test.columns:
        raise ValueError(f"Missing test column: {col}")

print("Data schema: PASS")


# ============================================================
# LOAD SCALERS
# ============================================================

print("\n[4/9] Loading preprocessing artifacts...")

FEATURE_SCALER_PATH = PREPROCESSING_DIR / "feature_scaler.joblib"
TARGET_SCALER_PATH = PREPROCESSING_DIR / "target_scaler.joblib"

if not FEATURE_SCALER_PATH.exists():
    raise FileNotFoundError(FEATURE_SCALER_PATH)

if not TARGET_SCALER_PATH.exists():
    raise FileNotFoundError(TARGET_SCALER_PATH)

feature_scaler = joblib.load(FEATURE_SCALER_PATH)
target_scaler = joblib.load(TARGET_SCALER_PATH)

print("Feature scaler: loaded")
print("Target scaler : loaded")

# ------------------------------------------------------------
# Feature scaler
# ------------------------------------------------------------

if not hasattr(feature_scaler, "mean_"):
    raise TypeError(
        "Feature scaler does not contain mean_. "
        f"Found type: {type(feature_scaler)}"
    )

if not hasattr(feature_scaler, "scale_"):
    raise TypeError(
        "Feature scaler does not contain scale_. "
        f"Found type: {type(feature_scaler)}"
    )

if len(feature_scaler.mean_) != 6:
    raise ValueError(
        f"Expected 6 feature scaler values, "
        f"found {len(feature_scaler.mean_)}"
    )

# ------------------------------------------------------------
# Target scaler
#
# preprocessing step stored the target scalers in a dictionary.
# ------------------------------------------------------------

if not isinstance(target_scaler, dict):
    raise TypeError(
        "Expected target_scaler.joblib to contain a dictionary. "
        f"Found type: {type(target_scaler)}"
    )

print(
    "Target scaler keys:",
    list(target_scaler.keys())
)

# The preprocessing pipeline should contain
# one scaler for each target.
required_target_keys = [
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]

missing_target_keys = [
    key
    for key in required_target_keys
    if key not in target_scaler
]

if missing_target_keys:
    raise ValueError(
        "Missing target scaler keys: "
        f"{missing_target_keys}"
    )

for target_name in required_target_keys:

    scaler = target_scaler[target_name]

    if not hasattr(scaler, "mean_"):
        raise TypeError(
            f"Target scaler '{target_name}' "
            "does not contain mean_."
        )

    if not hasattr(scaler, "scale_"):
        raise TypeError(
            f"Target scaler '{target_name}' "
            "does not contain scale_."
        )

    print(
        f"{target_name}: "
        f"mean={scaler.mean_[0]:.6f}, "
        f"scale={scaler.scale_[0]:.6f}"
    )

print("Scaler dimensions: PASS")

print("Scaler dimensions: PASS")


# ============================================================
# PINNs MODEL
# ============================================================

print("\n[5/9] Building PINNs architecture...")


class TreeGrowthPINNs(nn.Module):
    """
    Tree Growth PINNs

    Input:
        6 features

    Hidden:
        64 -> 64 -> 64

    Activation:
        Tanh

    Output:
        DBH
        Height
        Volume

    Volume:
        Softplus to guarantee positive prediction.
    """

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
        self.height_head = nn.Linear(64, 1)

        self.volume_head = nn.Linear(64, 1)
        self.volume_activation = nn.Softplus()

    def forward(self, x):
        h = self.hidden(x)

        dbh = self.dbh_head(h)
        height = self.height_head(h)

        volume_raw = self.volume_head(h)

        volume = self.volume_activation(volume_raw)

        return torch.cat(
            [dbh, height, volume],
            dim=1
        )


model = TreeGrowthPINNs().to(device)

print(model)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(f"\nTrainable parameters: {parameter_count:,}")

print("Architecture: PASS")


# ============================================================
# PHYSICS FUNCTION
# ============================================================

print("\n[6/9] Testing physical-unit conversion + physics relation...")


def inverse_transform_targets(y_scaled):
    """
    Convert scaled PINNs outputs back to physical units.

    Target order:
        column 0 = DBH
        column 1 = Height
        column 2 = Volume

    Each target uses its own train-fitted scaler.
    """

    dbh_mean = float(
        target_scaler["dbh_cm"].mean_[0]
    )

    dbh_scale = float(
        target_scaler["dbh_cm"].scale_[0]
    )

    height_mean = float(
        target_scaler["height_m"].mean_[0]
    )

    height_scale = float(
        target_scaler["height_m"].scale_[0]
    )

    volume_mean = float(
        target_scaler["volume_m3_tree"].mean_[0]
    )

    volume_scale = float(
        target_scaler["volume_m3_tree"].scale_[0]
    )

    dbh = (
        y_scaled[:, 0] * dbh_scale
        + dbh_mean
    )

    height = (
        y_scaled[:, 1] * height_scale
        + height_mean
    )

    volume = (
        y_scaled[:, 2] * volume_scale
        + volume_mean
    )

    return torch.stack(
        [dbh, height, volume],
        dim=1,
    )


def physics_volume(dbh_cm, height_m, coefficient):
    """
    Physics-inspired volume relation:

        V = C * DBH^2 * Height

    Important:
        DBH and Height must be in physical units.

    The coefficient C is supplied externally.
    """

    return coefficient * (
        dbh_cm.pow(2) * height_m
    )


# ------------------------------------------------------------
# Test inverse transformation
# ------------------------------------------------------------

dummy_scaled = torch.zeros(
    (4, 3),
    dtype=torch.float32,
    device=device,
)

dummy_physical = inverse_transform_targets(
    dummy_scaled
)

print("Inverse transformation shape:", tuple(dummy_physical.shape))

if tuple(dummy_physical.shape) != (4, 3):
    raise ValueError(
        "Inverse target transformation returned wrong shape."
    )


# ------------------------------------------------------------
# Test physics relation
# ------------------------------------------------------------

dummy_dbh = torch.tensor(
    [10.0, 15.0, 20.0, 25.0],
    dtype=torch.float32,
    device=device,
)

dummy_height = torch.tensor(
    [10.0, 15.0, 20.0, 25.0],
    dtype=torch.float32,
    device=device,
)

# Temporary coefficient for CORE TEST ONLY.
#
# This is NOT the final trained physics coefficient.
# Final coefficient calibration will be handled during
# the training step.
dummy_coefficient = torch.tensor(
    0.0001,
    dtype=torch.float32,
    device=device,
    requires_grad=True,
)

dummy_volume = physics_volume(
    dummy_dbh,
    dummy_height,
    dummy_coefficient,
)

print(
    "Physics output:",
    dummy_volume.detach().cpu().numpy()
)

if not torch.isfinite(dummy_volume).all():
    raise ValueError(
        "Physics calculation produced NaN or Inf."
    )

print("Physics relation: PASS")


# ============================================================
# PHYSICS MASK
# ============================================================

print("\n[7/9] Testing P18-P66 physics mask...")


def volume_physics_mask(age_months):
    """
    Physics applies ONLY to P18-P66.

    P6 and P12 are excluded.
    """

    return (
        (age_months >= 18)
        & (age_months <= 66)
    )


age_test = torch.tensor(
    [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66],
    dtype=torch.int64,
    device=device,
)

mask = volume_physics_mask(age_test)

print("Ages :", age_test.cpu().tolist())
print("Mask :", mask.cpu().tolist())

expected_mask = [
    False,
    False,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
]

if mask.cpu().tolist() != expected_mask:
    raise ValueError(
        "P18-P66 physics mask is incorrect."
    )

print("P18-P66 physics mask: PASS")


# ============================================================
# FORWARD PASS
# ============================================================

print("\n[8/9] Testing forward pass + backward pass...")

# Use a small real batch from the official training dataset.
#
# The batch must contain P18-P66 observations because
# the physics loss is intentionally evaluated only on
# P18-P66.

sample = (
    train[
        train["age_months"].between(18, 66)
    ][FEATURES]
    .head(32)
    .copy()
)

if len(sample) < 32:
    raise ValueError(
        "Unable to create a 32-row P18-P66 test batch."
    )

sample_scaled = feature_scaler.transform(sample)

x = torch.tensor(
    sample_scaled,
    dtype=torch.float32,
    device=device,
    requires_grad=True,
)

prediction = model(x)

print("Input shape :", tuple(x.shape))
print("Output shape:", tuple(prediction.shape))

if tuple(prediction.shape) != (32, 3):
    raise ValueError(
        f"Expected output shape (32, 3), "
        f"got {tuple(prediction.shape)}"
    )

if not torch.isfinite(prediction).all():
    raise ValueError(
        "Model produced NaN or Inf."
    )

pred_physical = inverse_transform_targets(
    prediction
)

dbh_pred = pred_physical[:, 0]
height_pred = pred_physical[:, 1]
volume_pred = pred_physical[:, 2]

print(
    f"DBH physical range: "
    f"{dbh_pred.min().item():.6f} "
    f"to "
    f"{dbh_pred.max().item():.6f}"
)

print(
    f"Height physical range: "
    f"{height_pred.min().item():.6f} "
    f"to "
    f"{height_pred.max().item():.6f}"
)

print(
    f"Volume physical range: "
    f"{volume_pred.min().item():.6f} "
    f"to "
    f"{volume_pred.max().item():.6f}"
)

# Simple supervised loss only for computational test.
dummy_target = torch.zeros_like(prediction)

supervised_loss = torch.mean(
    (prediction - dummy_target) ** 2
)

# Use the P18-P66 mask for the physics calculation.
sample_age = torch.tensor(
    train[
        train["age_months"].between(18, 66)
    ]["age_months"]
    .head(32)
    .to_numpy(),
    dtype=torch.int64,
    device=device,
)

physics_mask = volume_physics_mask(sample_age)

if physics_mask.sum().item() == 0:
    raise ValueError(
        "Test batch contains no P18-P66 observations."
    )

physics_dbh = dbh_pred[physics_mask]
physics_height = height_pred[physics_mask]
physics_volume_prediction = volume_pred[physics_mask]

physics_volume_expected = physics_volume(
    physics_dbh,
    physics_height,
    dummy_coefficient,
)

physics_loss = torch.mean(
    (
        physics_volume_prediction
        - physics_volume_expected
    ) ** 2
)

total_loss = (
    supervised_loss
    + 0.1 * physics_loss
)

total_loss.backward()

print(
    f"Supervised test loss: "
    f"{supervised_loss.item():.8f}"
)

print(
    f"Physics test loss    : "
    f"{physics_loss.item():.8f}"
)

print(
    f"Total test loss      : "
    f"{total_loss.item():.8f}"
)

if not torch.isfinite(total_loss):
    raise ValueError(
        "Total loss is NaN or Inf."
    )

gradient_found = any(
    parameter.grad is not None
    for parameter in model.parameters()
)

if not gradient_found:
    raise ValueError(
        "No model gradients were produced."
    )

print("Forward pass: PASS")
print("Backward pass: PASS")


# ============================================================
# VOLUME POSITIVITY
# ============================================================

print("\n[9/9] Checking positive volume output...")

if torch.any(volume_pred <= 0):
    raise ValueError(
        "Softplus volume output contains "
        "zero or negative values."
    )

print(
    f"Minimum predicted volume in test batch: "
    f"{volume_pred.min().item():.8f}"
)

print("Positive volume output: PASS")


# ============================================================
# SAVE CORE TEST REPORT
# ============================================================

report = {
    "step": 46,
    "status": "PASS",
    "device": str(device),
    "architecture": {
        "input": 6,
        "hidden": [64, 64, 64],
        "activation": "Tanh",
        "outputs": 3,
        "volume_activation": "Softplus",
    },
    "target_windows": {
        "dbh_cm": "P6-P66",
        "height_m": "P6-P66",
        "volume_m3_tree": "P18-P66",
    },
    "physics_window": "P18-P66",
    "phi": "P66 / 66 months",
    "old_checkpoint_reused": False,
    "parameter_count": parameter_count,
}

report_path = RESULT_DIR / "pinns_core_test_report.json"

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(f"\nSaved: {report_path}")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("STEP 46 STATUS: PASS")
print("=" * 80)

print("""
PINNs core implementation successfully tested.

Verified:
  - 6-input architecture
  - 64-64-64 hidden layers
  - Tanh activation
  - 3 outputs
  - Softplus volume output
  - Physical-unit inverse transformation
  - V = C * DBH^2 * Height relation
  - Physics mask P18-P66
  - Forward pass
  - Backward pass
  - Positive volume output

NO MODEL TRAINING WAS PERFORMED.
NO OLD PINNs CHECKPOINT WAS USED.
NO DATASET WAS MODIFIED.
""")

print("=" * 80)