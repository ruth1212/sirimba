import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================
# TREE GROWTH ANALYTICS
# STEP 21F — ANALYZE PINNs POSITIVITY PENALTY
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DASHBOARD_PATH = (
    PROJECT_ROOT
    / "results"
    / "dashboard_growth_trajectory.csv"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "pinns_models"
    / "best_pinns_model.pt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "pinns_positive_penalty_analysis.csv"
)


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


print("=" * 75)
print("STEP 21F — ANALYZE PINNs POSITIVITY PENALTY")
print("=" * 75)

print(f"\nDashboard : {DASHBOARD_PATH}")
print(f"Checkpoint: {CHECKPOINT_PATH}")


# ============================================================
# 1. LOAD DASHBOARD DATA
# ============================================================

print("\n" + "=" * 75)
print("1. LOAD DASHBOARD DATA")
print("=" * 75)

df = pd.read_csv(DASHBOARD_PATH)

negative_df = df[
    df["pinns_volume_m3_tree"] < 0
].copy()

print(f"Total dashboard rows   : {len(df):,}")
print(f"Negative volume rows   : {len(negative_df):,}")

if len(negative_df) == 0:
    print("\nNo negative PINNs volume found.")
    raise SystemExit(0)


# ============================================================
# 2. LOAD CHECKPOINT
# ============================================================

print("\n" + "=" * 75)
print("2. LOAD PINNs CHECKPOINT")
print("=" * 75)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False,
)

print("Checkpoint loaded: OK")

X_mean = np.asarray(
    checkpoint["X_mean"],
    dtype=np.float32,
)

X_std = np.asarray(
    checkpoint["X_std"],
    dtype=np.float32,
)

Y_mean = np.asarray(
    checkpoint["Y_mean"],
    dtype=np.float32,
)

Y_std = np.asarray(
    checkpoint["Y_std"],
    dtype=np.float32,
)

print(f"X mean shape : {X_mean.shape}")
print(f"X std shape  : {X_std.shape}")
print(f"Y mean shape : {Y_mean.shape}")
print(f"Y std shape  : {Y_std.shape}")


# ============================================================
# 3. LOAD MODEL CLASS
# ============================================================

print("\n" + "=" * 75)
print("3. LOAD PINNs MODEL")
print("=" * 75)

model_module = importlib.import_module(
    "12_build_pinns_model"
)

model_class = getattr(
    model_module,
    "TreeGrowthPINN",
)

model = model_class(
    input_size=len(FEATURES),
    hidden_size=64,
    hidden_layers=3,
    output_size=len(TARGETS),
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("PINNs model loaded: OK")


# ============================================================
# 4. PREPARE INPUT
# ============================================================

print("\n" + "=" * 75)
print("4. PREPARE NEGATIVE ROW INPUTS")
print("=" * 75)

X_raw = negative_df[
    FEATURES
].values.astype(np.float32)

X_normalized = (
    X_raw - X_mean
) / X_std

X_tensor = torch.tensor(
    X_normalized,
    dtype=torch.float32,
)

print(
    f"Input tensor shape : {tuple(X_tensor.shape)}"
)


# ============================================================
# 5. RAW PINNs INFERENCE
# ============================================================

print("\n" + "=" * 75)
print("5. RUN RAW PINNs INFERENCE")
print("=" * 75)

with torch.no_grad():
    predictions_normalized = model(
        X_tensor
    )

predictions_raw = (
    predictions_normalized.numpy()
    * Y_std
    + Y_mean
)

pred_dbh = predictions_raw[:, 0]
pred_height = predictions_raw[:, 1]
pred_volume = predictions_raw[:, 2]


# ============================================================
# 6. CALCULATE PHYSICS VOLUME
# ============================================================

print("\n" + "=" * 75)
print("6. CALCULATE PHYSICS-BASED VOLUME")
print("=" * 75)

C = float(
    checkpoint["C"]
)

physics_volume = (
    C
    * np.square(pred_dbh)
    * pred_height
)

volume_std = float(
    Y_std[2]
)


# ============================================================
# 7. CALCULATE POSITIVITY PENALTY
# ============================================================

print("\n" + "=" * 75)
print("7. CALCULATE POSITIVITY PENALTY")
print("=" * 75)

negative_part = np.maximum(
    0.0,
    -pred_volume,
)

positivity_penalty = np.square(
    negative_part
)

positivity_penalty_normalized = (
    positivity_penalty
    / (volume_std ** 2)
)

physics_residual = (
    pred_volume
    - physics_volume
)

physics_loss_per_row = (
    np.square(
        physics_residual
        / volume_std
    )
)

result = negative_df[
    [
        "regional_id",
        "compartment_id",
        "species",
        "clone",
        "age_months",
        "inventory_year",
        "planting_year",
        "disease_score",
        "trees_per_ha",
    ]
].copy()

result["pred_dbh_cm"] = pred_dbh
result["pred_height_m"] = pred_height
result["pred_volume_m3_tree"] = pred_volume
result["physics_volume_m3_tree"] = physics_volume

result["physics_residual"] = physics_residual
result["physics_loss_per_row"] = physics_loss_per_row

result["negative_part"] = negative_part
result["positivity_penalty"] = positivity_penalty
result[
    "positivity_penalty_normalized"
] = positivity_penalty_normalized

result.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# 8. SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("8. PENALTY SUMMARY")
print("=" * 75)

print(
    f"Negative rows                 : "
    f"{len(result):,}"
)

print(
    f"Minimum predicted volume      : "
    f"{pred_volume.min():.12f}"
)

print(
    f"Maximum predicted volume      : "
    f"{pred_volume.max():.12f}"
)

print(
    f"Mean negative magnitude       : "
    f"{negative_part.mean():.12f}"
)

print(
    f"Maximum negative magnitude    : "
    f"{negative_part.max():.12f}"
)

print(
    f"Mean positivity penalty       : "
    f"{positivity_penalty.mean():.12f}"
)

print(
    f"Mean normalized penalty       : "
    f"{positivity_penalty_normalized.mean():.12f}"
)

print(
    f"Maximum normalized penalty    : "
    f"{positivity_penalty_normalized.max():.12f}"
)

print(
    f"Mean physics loss/row         : "
    f"{physics_loss_per_row.mean():.12f}"
)

print(
    f"Mean physics volume           : "
    f"{physics_volume.mean():.12f}"
)


# ============================================================
# 9. AGE DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("9. NEGATIVE VOLUME BY AGE")
print("=" * 75)

age_summary = (
    result
    .groupby("age_months")
    .agg(
        rows=("pred_volume_m3_tree", "size"),
        min_volume=(
            "pred_volume_m3_tree",
            "min",
        ),
        mean_volume=(
            "pred_volume_m3_tree",
            "mean",
        ),
        mean_penalty=(
            "positivity_penalty_normalized",
            "mean",
        ),
    )
    .reset_index()
)

print(age_summary.to_string(index=False))


# ============================================================
# 10. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("STEP 21F COMPLETED")
print("=" * 75)

print(
    f"Analysis saved to:\n"
    f"{OUTPUT_PATH}"
)

print("\nNO TRAINING WAS PERFORMED.")
print("NO MODEL FILE WAS MODIFIED.")
print("NO DASHBOARD DATA WAS MODIFIED.")

print("=" * 75)