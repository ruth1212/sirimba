from pathlib import Path
import importlib
import sys

import numpy as np
import pandas as pd
import torch


# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "results"
    / "dashboard_growth_trajectory.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "pinns_models"
    / "best_pinns_model.pt"
)


print("=" * 78)
print("STEP 21D — TRACE RAW PINNs PREDICTION")
print("=" * 78)

print()
print(f"Dashboard dataset : {DATA_PATH}")
print(f"PINNs checkpoint  : {MODEL_PATH}")


# ==========================================================
# 1. LOAD DASHBOARD DATA
# ==========================================================

print()
print("=" * 78)
print("1. LOAD DASHBOARD DATA")
print("=" * 78)

df = pd.read_csv(DATA_PATH)

negative_df = df[
    df["pinns_volume_m3_tree"] < 0
].copy()

print(f"Total dashboard rows      : {len(df):,}")
print(f"Negative volume rows      : {len(negative_df):,}")

if negative_df.empty:
    print()
    print("No negative rows found.")
    print("Nothing to trace.")
    raise SystemExit(0)


# ==========================================================
# 2. CHECK MODEL CHECKPOINT
# ==========================================================

print()
print("=" * 78)
print("2. CHECK PINNs CHECKPOINT")
print("=" * 78)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"PINNs checkpoint not found: {MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False,
)

print("Checkpoint loaded: OK")

print()
print("Checkpoint keys:")

for key in checkpoint.keys():
    print(f"  - {key}")


# ==========================================================
# 3. READ MODEL CONFIG
# ==========================================================

print()
print("=" * 78)
print("3. READ MODEL CONFIGURATION")
print("=" * 78)

features = checkpoint["features"]
targets = checkpoint["targets"]

print("Features:")
for feature in features:
    print(f"  - {feature}")

print()
print("Targets:")
for target in targets:
    print(f"  - {target}")

print()
print(
    f"Physics constant C : "
    f"{checkpoint.get('C')}"
)

print(
    f"Lambda physics     : "
    f"{checkpoint.get('lambda_physics')}"
)


# ==========================================================
# 4. IMPORT PINNs MODEL CLASS
# ==========================================================

print()
print("=" * 78)
print("4. LOAD PINNs MODEL CLASS")
print("=" * 78)

SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPTS_DIR),
    )


# The model architecture was defined
# in the PINNs training script.
#
# We locate the class dynamically so this
# trace does not create a second architecture.

# ==========================================================
# LOAD PINNs MODEL CLASS
# ==========================================================

model_module = importlib.import_module(
    "12_build_pinns_model"
)

print(
    "PINNs module loaded from: "
    "12_build_pinns_model.py"
)

model_class = None

for name in dir(model_module):

    obj = getattr(
        model_module,
        name,
    )

    if isinstance(obj, type):

        print(
            f"Candidate class found: {name}"
        )

        if name == "TreeGrowthPINN":

            model_class = obj
            break


if model_class is None:

    raise RuntimeError(
        "No PINNs model class was found "
        "inside 12_build_pinns_model.py."
    )

print(
    f"PINNs model class: "
    f"{model_class.__name__}"
)


# ==========================================================
# 5. RECONSTRUCT MODEL
# ==========================================================

print()
print("=" * 78)
print("5. RECONSTRUCT PINNs MODEL")
print("=" * 78)

model = model_class(
    input_size=len(features),
    hidden_size=64,
    hidden_layers=3,
    output_size=len(targets),
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("PINNs model loaded: OK")


# ==========================================================
# 6. NORMALIZATION STATISTICS
# ==========================================================

print()
print("=" * 78)
print("6. LOAD NORMALIZATION STATISTICS")
print("=" * 78)

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

print(
    f"X mean shape : {X_mean.shape}"
)

print(
    f"X std shape  : {X_std.shape}"
)

print(
    f"Y mean shape : {Y_mean.shape}"
)

print(
    f"Y std shape  : {Y_std.shape}"
)


# ==========================================================
# 7. PREPARE EXACT NEGATIVE ROW INPUTS
# ==========================================================

print()
print("=" * 78)
print("7. PREPARE NEGATIVE ROW INPUTS")
print("=" * 78)

missing_features = [
    feature
    for feature in features
    if feature not in negative_df.columns
]

if missing_features:

    raise RuntimeError(
        "Missing model features: "
        + ", ".join(missing_features)
    )


X_raw = (
    negative_df[features]
    .astype(np.float32)
    .to_numpy()
)

X_norm = (
    X_raw - X_mean
) / X_std


# ==========================================================
# 8. RAW PINNs INFERENCE
# ==========================================================

print()
print("=" * 78)
print("8. RUN RAW PINNs INFERENCE")
print("=" * 78)

with torch.no_grad():

    X_tensor = torch.tensor(
        X_norm,
        dtype=torch.float32,
    )

    prediction_norm = model(
        X_tensor
    ).cpu().numpy()


# Denormalize predictions

prediction_raw = (
    prediction_norm * Y_std
) + Y_mean


raw_prediction_df = pd.DataFrame(
    prediction_raw,
    columns=targets,
)


# ==========================================================
# 9. COMPARE RAW MODEL VS DASHBOARD
# ==========================================================

print()
print("=" * 78)
print("9. RAW MODEL VS DASHBOARD")
print("=" * 78)

dashboard_volume = (
    negative_df[
        "pinns_volume_m3_tree"
    ]
    .to_numpy()
)

raw_volume = (
    raw_prediction_df[
        "volume_m3_tree"
    ]
    .to_numpy()
)

difference = (
    raw_volume
    - dashboard_volume
)

comparison = pd.DataFrame(
    {
        "dashboard_volume":
            dashboard_volume,

        "raw_model_volume":
            raw_volume,

        "difference":
            difference,
    }
)

print(
    comparison.head(20)
    .to_string(index=False)
)


# ==========================================================
# 10. NUMERICAL COMPARISON
# ==========================================================

print()
print("=" * 78)
print("10. NUMERICAL COMPARISON")
print("=" * 78)

print(
    "Maximum absolute difference : "
    f"{np.abs(difference).max():.12f}"
)

print(
    "Mean absolute difference    : "
    f"{np.abs(difference).mean():.12f}"
)

print(
    "Raw model minimum volume    : "
    f"{raw_volume.min():.12f}"
)

print(
    "Dashboard minimum volume    : "
    f"{dashboard_volume.min():.12f}"
)


# ==========================================================
# 11. TRACE FIRST PROBLEMATIC ROW
# ==========================================================

print()
print("=" * 78)
print("11. FIRST PROBLEMATIC ROW")
print("=" * 78)

first_row = negative_df.iloc[0]

first_raw_prediction = raw_prediction_df.iloc[0]

print()
print("INPUT FEATURES:")

for feature in features:

    print(
        f"  {feature:<20}: "
        f"{first_row[feature]}"
    )

print()
print("DASHBOARD OUTPUT:")

print(
    f"  DBH       : "
    f"{first_row['pinns_dbh_cm']}"
)

print(
    f"  Height    : "
    f"{first_row['pinns_height_m']}"
)

print(
    f"  Volume    : "
    f"{first_row['pinns_volume_m3_tree']}"
)

print()
print("RAW MODEL OUTPUT:")

print(
    f"  DBH       : "
    f"{first_raw_prediction['dbh_cm']}"
)

print(
    f"  Height    : "
    f"{first_raw_prediction['height_m']}"
)

print(
    f"  Volume    : "
    f"{first_raw_prediction['volume_m3_tree']}"
)


# ==========================================================
# 12. SAVE TRACE RESULT
# ==========================================================

output_path = (
    PROJECT_ROOT
    / "results"
    / "pinns_negative_volume_trace.csv"
)

trace_output = negative_df[
    [
        "regional_id",
        "compartment_id",
        "species",
        "clone",
        "age_months",
        "growth_period",
        "pinns_dbh_cm",
        "pinns_height_m",
        "pinns_volume_m3_tree",
    ]
].copy()

trace_output[
    "raw_model_volume_m3_tree"
] = raw_volume

trace_output[
    "raw_minus_dashboard"
] = difference

trace_output.to_csv(
    output_path,
    index=False,
)

print()
print(
    f"Trace file saved: {output_path}"
)


# ==========================================================
# FINAL
# ==========================================================

print()
print("=" * 78)
print("STEP 21D COMPLETED")
print("=" * 78)

if np.allclose(
    raw_volume,
    dashboard_volume,
    rtol=1e-5,
    atol=1e-7,
):

    print(
        "TRACE RESULT: RAW MODEL MATCHES DASHBOARD"
    )

    print(
        "Negative volume originates from the "
        "PINNs model output."
    )

else:

    print(
        "TRACE RESULT: RAW MODEL DOES NOT MATCH DASHBOARD"
    )

    print(
        "The negative value is being introduced "
        "during dashboard dataset generation."
    )

print()
print("No model training was performed.")
