from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# TREE GROWTH ANALYTICS
# STEP 11 - PREDICTION ENGINE
# RANDOM FOREST + PINNs
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
RF_MODEL_DIR = RESULTS_DIR / "random_forest_models"
PINNS_MODEL_PATH = (
    RESULTS_DIR
    / "pinns_models"
    / "best_pinns_model.pt"
)


# ============================================================
# CONFIGURATION
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


RF_MODEL_PATHS = {
    "dbh_cm":
        RF_MODEL_DIR
        / "random_forest_dbh_cm.joblib",

    "height_m":
        RF_MODEL_DIR
        / "random_forest_height_m.joblib",

    "volume_m3_tree":
        RF_MODEL_DIR
        / "random_forest_volume_m3_tree.joblib",
}


# ============================================================
# PINNs ARCHITECTURE
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
# LOAD RANDOM FOREST
# ============================================================

def load_random_forest_models():

    print("\nLoading Random Forest models...")

    models = {}

    for target, path in (
        RF_MODEL_PATHS.items()
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"RF model tidak ditemukan:\n"
                f"{path}"
            )

        models[target] = joblib.load(
            path
        )

        print(
            f"  {target:<20} : "
            f"LOADED"
        )

    return models


# ============================================================
# LOAD PINNs
# ============================================================

def load_pinns_model():

    print("\nLoading PINNs model...")

    if not PINNS_MODEL_PATH.exists():

        raise FileNotFoundError(
            "PINNs model tidak ditemukan:\n"
            f"{PINNS_MODEL_PATH}"
        )

    checkpoint = torch.load(
        PINNS_MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = TreeGrowthPINN(
        input_size=len(FEATURES),
        hidden_size=64,
        hidden_layers=3,
        output_size=len(TARGETS),
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        "  Architecture : 6 -> 64 -> 64 -> 64 -> 3"
    )

    print(
        f"  Best epoch   : "
        f"{checkpoint['epoch']}"
    )

    print(
        f"  Best loss    : "
        f"{checkpoint['best_loss']:.8f}"
    )

    return model, checkpoint


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input(input_data):

    print("\nValidating input...")

    if not isinstance(
        input_data,
        dict,
    ):

        raise TypeError(
            "input_data harus berupa dictionary."
        )

    missing = [
        feature
        for feature in FEATURES
        if feature not in input_data
    ]

    if missing:

        raise ValueError(
            "Feature input yang missing: "
            f"{missing}"
        )

    values = []

    for feature in FEATURES:

        value = input_data[
            feature
        ]

        if not np.isfinite(
            float(value)
        ):

            raise ValueError(
                f"Input {feature} "
                "harus berupa angka finite."
            )

        values.append(
            float(value)
        )

    X = np.array(
        [values],
        dtype=np.float32,
    )

    print("  Features : OK")

    return X


# ============================================================
# RANDOM FOREST PREDICTION
# ============================================================

def predict_random_forest(
    models,
    X,
):

    # Random Forest dilatih menggunakan
    # pandas DataFrame dengan nama feature.
    # Gunakan nama feature yang sama saat prediction
    # agar sklearn tidak menghasilkan warning.

    X_df = pd.DataFrame(
    X,
    columns=FEATURES,
)

    predictions = {}

    for target in TARGETS:

        model = models[target]

        prediction = model.predict(
            X_df
        )

        predictions[target] = float(
            prediction[0]
        )

    return predictions


# ============================================================
# PINNs PREDICTION
# ============================================================

def predict_pinns(
    model,
    checkpoint,
    X,
):

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

    X_std = np.where(
        X_std == 0,
        1.0,
        X_std,
    )

    Y_std = np.where(
        Y_std == 0,
        1.0,
        Y_std,
    )

    X_normalized = (
        X - X_mean
    ) / X_std

    X_tensor = torch.tensor(
        X_normalized,
        dtype=torch.float32,
    )

    with torch.no_grad():

        Y_normalized = model(
            X_tensor
        ).numpy()

    Y_raw = (
        Y_normalized
        * Y_std
        + Y_mean
    )

    predictions = {}

    for index, target in enumerate(
        TARGETS
    ):

        predictions[target] = float(
            Y_raw[0, index]
        )

    return predictions


# ============================================================
# COMBINED PREDICTION
# ============================================================

def predict(
    input_data,
    rf_models,
    pinns_model,
    pinns_checkpoint,
):

    X = validate_input(
        input_data
    )

    rf_result = predict_random_forest(
        rf_models,
        X,
    )

    pinns_result = predict_pinns(
        pinns_model,
        pinns_checkpoint,
        X,
    )

    result = {

        "inputs": {
            feature:
                float(
                    input_data[feature]
                )
            for feature in FEATURES
        },

        "random_forest": rf_result,

        "pinns": pinns_result,
    }

    return result


# ============================================================
# TEST PREDICTION
# ============================================================

if __name__ == "__main__":

    print("=" * 75)
    print("TREE GROWTH ANALYTICS")
    print("STEP 11 - PREDICTION ENGINE")
    print("=" * 75)

    print("\n[1/4] Loading Random Forest...")
    rf_models = load_random_forest_models()

    print("\n[2/4] Loading PINNs...")
    pinns_model, pinns_checkpoint = (
        load_pinns_model()
    )

    print("\n[3/4] Running test prediction...")

    test_input = {

        "age_months":
            60,

        "inventory_year":
            2020,

        "clone_factor":
            1.0,

        "planting_year":
            2015,

        "disease_score":
            0.10,

        "trees_per_ha":
            1300,
    }

    result = predict(
        test_input,
        rf_models,
        pinns_model,
        pinns_checkpoint,
    )

    print("\n[4/4] Prediction result")

    print("=" * 75)

    print("\nINPUT")

    for feature, value in (
        result["inputs"].items()
    ):

        print(
            f"  {feature:<20}: "
            f"{value}"
        )

    print("\nRANDOM FOREST")

    for target, value in (
        result["random_forest"].items()
    ):

        print(
            f"  {target:<20}: "
            f"{value:.6f}"
        )

    print("\nPINNs")

    for target, value in (
        result["pinns"].items()
    ):

        print(
            f"  {target:<20}: "
            f"{value:.6f}"
        )

    print("\n" + "=" * 75)
    print("STEP 11 TEST COMPLETED")
    print("=" * 75)