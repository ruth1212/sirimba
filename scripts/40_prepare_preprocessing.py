from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# STEP 40
# PREPROCESSING & SCALING
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PIPELINE_DIR = ROOT / "data" / "model_ready" / "pipeline"
PREPROCESSING_DIR = PIPELINE_DIR / "preprocessing"

TRAIN_PATH = PIPELINE_DIR / "train_dbh_height.csv"
TEST_PATH = PIPELINE_DIR / "test_dbh_height.csv"

TRAIN_VOLUME_PATH = PIPELINE_DIR / "train_volume_p18.csv"
TEST_VOLUME_PATH = PIPELINE_DIR / "test_volume_p18.csv"

PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OFFICIAL FEATURES / TARGETS
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

DBH_HEIGHT_TARGETS = [
    "dbh_cm",
    "height_m",
]

VOLUME_TARGET = "volume_m3_tree"


# ============================================================
# OUTPUTS
# ============================================================

FEATURE_SCALER_PATH = (
    PREPROCESSING_DIR / "feature_scaler.joblib"
)

TARGET_SCALER_PATH = (
    PREPROCESSING_DIR / "target_scaler.joblib"
)

PREPROCESSING_CONFIG_PATH = (
    PREPROCESSING_DIR / "preprocessing_config.json"
)

TRAIN_FEATURES_PATH = (
    PREPROCESSING_DIR / "train_features_scaled.csv"
)

TEST_FEATURES_PATH = (
    PREPROCESSING_DIR / "test_features_scaled.csv"
)

TRAIN_TARGETS_PATH = (
    PREPROCESSING_DIR / "train_targets_scaled.csv"
)

TEST_TARGETS_PATH = (
    PREPROCESSING_DIR / "test_targets_scaled.csv"
)


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    print()
    print("=" * 80)
    print("STEP 40 STATUS: FAIL")
    print("=" * 80)
    print(message)
    sys.exit(1)


def check_finite(df, columns, name):
    values = df[columns].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        fail(
            f"{name} contains NaN or infinite values."
        )


def print_scaler_statistics(name, scaler, columns):
    print(f"\n{name} scaler statistics:")

    for i, column in enumerate(columns):
        print(
            f"  {column:<20} "
            f"mean={scaler.mean_[i]:.6f}  "
            f"scale={scaler.scale_[i]:.6f}"
        )


print("=" * 80)
print("STEP 40 - PREPROCESSING & SCALING")
print("=" * 80)


# ============================================================
# [1/10] LOAD DATA
# ============================================================

print("\n[1/10] Loading model pipeline datasets...")

required_files = [
    TRAIN_PATH,
    TEST_PATH,
    TRAIN_VOLUME_PATH,
    TEST_VOLUME_PATH,
]

for path in required_files:
    if not path.exists():
        fail(
            f"Required pipeline file not found:\n{path}"
        )

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

train_volume = pd.read_csv(TRAIN_VOLUME_PATH)
test_volume = pd.read_csv(TEST_VOLUME_PATH)

print(f"Train DBH/Height rows : {len(train):,}")
print(f"Test DBH/Height rows  : {len(test):,}")
print(f"Train Volume rows     : {len(train_volume):,}")
print(f"Test Volume rows      : {len(test_volume):,}")

print("Pipeline datasets loaded: PASS")


# ============================================================
# [2/10] CHECK FEATURES
# ============================================================

print("\n[2/10] Checking official features...")

for name, df in [
    ("Train", train),
    ("Test", test),
    ("Train Volume", train_volume),
    ("Test Volume", test_volume),
]:

    missing = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        fail(
            f"{name} is missing features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing
            )
        )

print("Six official features: PASS")


# ============================================================
# [3/10] CHECK TARGETS
# ============================================================

print("\n[3/10] Checking official targets...")

for target in DBH_HEIGHT_TARGETS:

    if target not in train.columns:
        fail(
            f"Missing DBH/Height target in train: {target}"
        )

    if target not in test.columns:
        fail(
            f"Missing DBH/Height target in test: {target}"
        )

if VOLUME_TARGET not in train_volume.columns:
    fail("Volume target missing from train volume data.")

if VOLUME_TARGET not in test_volume.columns:
    fail("Volume target missing from test volume data.")

print("DBH target: PASS")
print("Height target: PASS")
print("Volume target: PASS")


# ============================================================
# [4/10] CHECK TRAIN/TEST VALUES
# ============================================================

print("\n[4/10] Checking numerical values...")

check_finite(
    train,
    FEATURES + DBH_HEIGHT_TARGETS,
    "Train DBH/Height"
)

check_finite(
    test,
    FEATURES + DBH_HEIGHT_TARGETS,
    "Test DBH/Height"
)

check_finite(
    train_volume,
    FEATURES + [VOLUME_TARGET],
    "Train Volume"
)

check_finite(
    test_volume,
    FEATURES + [VOLUME_TARGET],
    "Test Volume"
)

print("No NaN/Inf values: PASS")


# ============================================================
# [5/10] FIT FEATURE SCALER — TRAIN ONLY
# ============================================================

print("\n[5/10] Fitting feature scaler using TRAIN ONLY...")

feature_scaler = StandardScaler()

X_train = train[FEATURES].to_numpy(dtype=float)
X_test = test[FEATURES].to_numpy(dtype=float)

feature_scaler.fit(X_train)

X_train_scaled = feature_scaler.transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

print_scaler_statistics(
    "Feature",
    feature_scaler,
    FEATURES
)

print("\nFeature scaler fit source: TRAIN ONLY")
print("Feature scaling: PASS")


# ============================================================
# [6/10] FIT TARGET SCALER — TRAIN ONLY
# ============================================================

print("\n[6/10] Fitting target scaler using TRAIN ONLY...")

# IMPORTANT:
# The target scaler is fitted on the three official targets
# using the complete DBH/Height train data plus the P18-P66
# volume train data.
#
# This creates one common physical target space for PINNs.
#
# Volume is only allowed from P18 onward.

train_targets_for_scaling = pd.DataFrame({
    "dbh_cm": train["dbh_cm"],
    "height_m": train["height_m"],
})

# Volume is not aligned row-for-row with the DBH/Height dataset
# because volume starts at P18.
#
# Therefore each target receives its own scaler statistics.

dbh_scaler = StandardScaler()
height_scaler = StandardScaler()
volume_scaler = StandardScaler()

dbh_scaler.fit(
    train[["dbh_cm"]].to_numpy(dtype=float)
)

height_scaler.fit(
    train[["height_m"]].to_numpy(dtype=float)
)

volume_scaler.fit(
    train_volume[["volume_m3_tree"]].to_numpy(dtype=float)
)

print(
    f"DBH scaler    mean={dbh_scaler.mean_[0]:.6f} "
    f"scale={dbh_scaler.scale_[0]:.6f}"
)

print(
    f"Height scaler mean={height_scaler.mean_[0]:.6f} "
    f"scale={height_scaler.scale_[0]:.6f}"
)

print(
    f"Volume scaler mean={volume_scaler.mean_[0]:.6f} "
    f"scale={volume_scaler.scale_[0]:.6f}"
)

print("Target scalers fit source: TRAIN ONLY")
print("Target scaling: PASS")


# ============================================================
# [7/10] VERIFY TEST WAS NOT USED TO FIT SCALERS
# ============================================================

print("\n[7/10] Verifying train-only preprocessing...")

# Recreate independent scalers from train to verify
# the saved statistics are exactly train-derived.

check_feature_scaler = StandardScaler()
check_feature_scaler.fit(X_train)

if not np.allclose(
    feature_scaler.mean_,
    check_feature_scaler.mean_
):
    fail(
        "Feature scaler verification failed."
    )

check_dbh_scaler = StandardScaler()
check_dbh_scaler.fit(
    train[["dbh_cm"]].to_numpy(dtype=float)
)

if not np.allclose(
    dbh_scaler.mean_,
    check_dbh_scaler.mean_
):
    fail(
        "DBH scaler verification failed."
    )

check_height_scaler = StandardScaler()
check_height_scaler.fit(
    train[["height_m"]].to_numpy(dtype=float)
)

if not np.allclose(
    height_scaler.mean_,
    check_height_scaler.mean_
):
    fail(
        "Height scaler verification failed."
    )

check_volume_scaler = StandardScaler()
check_volume_scaler.fit(
    train_volume[["volume_m3_tree"]]
    .to_numpy(dtype=float)
)

if not np.allclose(
    volume_scaler.mean_,
    check_volume_scaler.mean_
):
    fail(
        "Volume scaler verification failed."
    )

print("Scalers confirmed train-derived: PASS")


# ============================================================
# [8/10] VERIFY SCALED FEATURES
# ============================================================

print("\n[8/10] Checking scaled feature matrices...")

if not np.isfinite(X_train_scaled).all():
    fail(
        "Scaled train features contain NaN/Inf."
    )

if not np.isfinite(X_test_scaled).all():
    fail(
        "Scaled test features contain NaN/Inf."
    )

print(
    f"Scaled train shape: {X_train_scaled.shape}"
)

print(
    f"Scaled test shape : {X_test_scaled.shape}"
)

print("Scaled feature matrices: PASS")


# ============================================================
# [9/10] SAVE SCALERS AND SCALED DATA
# ============================================================

print("\n[9/10] Saving preprocessing artifacts...")

# Save feature scaler
joblib.dump(
    feature_scaler,
    FEATURE_SCALER_PATH
)

# Save individual target scalers together
target_scalers = {
    "dbh_cm": dbh_scaler,
    "height_m": height_scaler,
    "volume_m3_tree": volume_scaler,
}

joblib.dump(
    target_scalers,
    TARGET_SCALER_PATH
)


# Save scaled feature matrices
train_features_scaled = pd.DataFrame(
    X_train_scaled,
    columns=FEATURES
)

test_features_scaled = pd.DataFrame(
    X_test_scaled,
    columns=FEATURES
)

train_features_scaled.to_csv(
    TRAIN_FEATURES_PATH,
    index=False
)

test_features_scaled.to_csv(
    TEST_FEATURES_PATH,
    index=False
)


# Save scaled DBH/Height targets
train_targets_scaled = pd.DataFrame({
    "dbh_cm": dbh_scaler.transform(
        train[["dbh_cm"]]
    ).ravel(),

    "height_m": height_scaler.transform(
        train[["height_m"]]
    ).ravel(),
})

test_targets_scaled = pd.DataFrame({
    "dbh_cm": dbh_scaler.transform(
        test[["dbh_cm"]]
    ).ravel(),

    "height_m": height_scaler.transform(
        test[["height_m"]]
    ).ravel(),
})

train_targets_scaled.to_csv(
    TRAIN_TARGETS_PATH,
    index=False
)

test_targets_scaled.to_csv(
    TEST_TARGETS_PATH,
    index=False
)


print(f"Saved: {FEATURE_SCALER_PATH}")
print(f"Saved: {TARGET_SCALER_PATH}")
print(f"Saved: {TRAIN_FEATURES_PATH}")
print(f"Saved: {TEST_FEATURES_PATH}")
print(f"Saved: {TRAIN_TARGETS_PATH}")
print(f"Saved: {TEST_TARGETS_PATH}")

print("Preprocessing artifacts: PASS")


# ============================================================
# [10/10] SAVE CONFIGURATION
# ============================================================

print("\n[10/10] Saving preprocessing configuration...")

config = {
    "step": "40",
    "method": "StandardScaler",
    "feature_scaler": {
        "fit_source": "train_only",
        "features": FEATURES,
    },
    "target_scalers": {
        "dbh_cm": {
            "fit_source": "train_only",
            "target": "dbh_cm",
            "valid_age_range": "P6-P66",
        },
        "height_m": {
            "fit_source": "train_only",
            "target": "height_m",
            "valid_age_range": "P6-P66",
        },
        "volume_m3_tree": {
            "fit_source": "train_volume_p18_only",
            "target": "volume_m3_tree",
            "valid_age_range": "P18-P66",
        },
    },
    "random_forest": {
        "feature_scaling_required": False,
        "target_scaling_required": False,
        "uses_raw_physical_units": True,
    },
    "pinns": {
        "feature_scaling": True,
        "target_scaling": True,
        "physics_loss_units": "physical_units",
        "physics_relation": "V ~ DBH^2 * Height",
        "volume_physics_start_age_months": 18,
    },
    "prediction_rules": {
        "dbh": "P6-P66",
        "height": "P6-P66",
        "volume": "P18-P66",
        "phi": "P66 = 66 months",
    },
}

with open(
    PREPROCESSING_CONFIG_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        config,
        f,
        indent=2,
        ensure_ascii=False
    )

print(
    f"Saved: {PREPROCESSING_CONFIG_PATH}"
)

print("Configuration: PASS")


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("STEP 40 STATUS: PASS")
print("=" * 80)

print()
print("PREPROCESSING IS READY.")

print()
print("Random Forest:")
print("  - Uses raw feature values")
print("  - Uses raw target values")
print("  - No scaling required")

print()
print("PINNs:")
print("  - Uses scaled features")
print("  - Uses scaled targets for neural-network optimization")
print("  - Physics calculations must return to physical units")
print("  - Volume physics starts at P18")

print()
print("Scaler fitting:")
print("  - Feature scaler : TRAIN ONLY")
print("  - DBH scaler     : TRAIN ONLY")
print("  - Height scaler  : TRAIN ONLY")
print("  - Volume scaler  : TRAIN P18-P66 ONLY")

print()
print("NO MODEL TRAINING WAS PERFORMED.")

print("=" * 80)