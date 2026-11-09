from pathlib import Path

import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"

RF_MODEL_DIR = (
    RESULTS_DIR / "random_forest_models"
)

THA_RF_MODEL = (
    RESULTS_DIR
    / "tha_random_forest_models"
    / "random_forest_tha_m.joblib"
)

PINNS_MODEL_PATH = (
    RESULTS_DIR
    / "pinns_models"
    / "best_pinns_hard_positive_model.pt"
)


# ============================================================
# MODEL FEATURES
# ============================================================

MODEL_FEATURES = [
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
# RANDOM FOREST MODEL
# ============================================================

RF_MODEL_FILES = {
    "dbh_cm": RF_MODEL_DIR / "random_forest_dbh_cm.joblib",
    "height_m": RF_MODEL_DIR / "random_forest_height_m.joblib",
    "volume_m3_tree": RF_MODEL_DIR / "random_forest_volume_m3_tree.joblib",
}


# ============================================================
# HARD-POSITIVE PINNs
# ============================================================

class TreeGrowthPINNHardPositive(nn.Module):
    """
    Hard-Positive PINNs.

    DBH:
        direct normalized output

    Height:
        direct normalized output

    Volume:
        raw output -> Softplus -> positive physical volume
        -> normalized using training Y_mean/Y_std

    This guarantees:
        volume_m3_tree > 0
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        hidden_layers,
        output_size,
        volume_mean,
        volume_std,
    ):
        super().__init__()

        self.volume_mean = float(
            volume_mean
        )

        self.volume_std = float(
            volume_std
        )

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

        self.softplus = nn.Softplus()

    def forward(self, x):

        raw_output = self.network(x)

        # ----------------------------------------
        # DBH
        # ----------------------------------------

        dbh_normalized = (
            raw_output[:, 0:1]
        )

        # ----------------------------------------
        # HEIGHT
        # ----------------------------------------

        height_normalized = (
            raw_output[:, 1:2]
        )

        # ----------------------------------------
        # VOLUME
        # ----------------------------------------

        volume_raw = self.softplus(
            raw_output[:, 2:3]
        )

        volume_normalized = (
            volume_raw
            - self.volume_mean
        ) / self.volume_std

        return torch.cat(
            [
                dbh_normalized,
                height_normalized,
                volume_normalized,
            ],
            dim=1,
        )


# ============================================================
# PREDICTION ENGINE
# ============================================================

class TreeGrowthPredictionEngine:

    def __init__(
        self,
        device=None,
    ):

        print("=" * 70)
        print(
            "TREE GROWTH ANALYTICS"
        )
        print(
            "PREDICTION ENGINE"
        )
        print("=" * 70)

        # ----------------------------------------
        # DEVICE
        # ----------------------------------------

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = torch.device(
                device
            )

        print(
            f"\nDevice: {self.device}"
        )

        # ----------------------------------------
        # LOAD MODELS
        # ----------------------------------------

        self._load_random_forest()

        self._load_tha_random_forest()

        self._load_pinns()

        print(
            "\nAll prediction models loaded."
        )


    # ========================================================
    # LOAD RANDOM FOREST
    # ========================================================

    def _load_random_forest(
        self,
    ):

        print(
            "\nLoading Random Forest models..."
        )

        self.rf_models = {}

        for target, path in RF_MODEL_FILES.items():

            print(
                f"  Loading RF {target}..."
            )

            if not path.exists():

                raise FileNotFoundError(
                    f"Random Forest model not found:\n{path}"
                )

            self.rf_models[target] = (
                joblib.load(path)
            )

        print(
            "  Random Forest: OK"
        )


    # ========================================================
    # LOAD THA RANDOM FOREST
    # ========================================================

    def _load_tha_random_forest(
        self,
    ):

        print(
            "\nLoading THA Random Forest model..."
        )

        if not THA_RF_MODEL.exists():

            raise FileNotFoundError(
                f"THA RF model not found:\n"
                f"{THA_RF_MODEL}"
            )

        self.tha_model = joblib.load(
            THA_RF_MODEL
        )

        print(
            "  RF THA: OK"
        )


    # ========================================================
    # LOAD HARD-POSITIVE PINNs
    # ========================================================

    def _load_pinns(
        self,
    ):

        print(
            "\nLoading Hard-Positive PINNs model..."
        )

        if not PINNS_MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Hard-Positive PINNs model not found:\n"
                f"{PINNS_MODEL_PATH}"
            )

        checkpoint = torch.load(
            PINNS_MODEL_PATH,
            map_location=self.device,
            weights_only=False,
        )

        # ----------------------------------------
        # CHECKPOINT METADATA
        # ----------------------------------------

        self.pinns_features = checkpoint[
            "features"
        ]

        self.pinns_targets = checkpoint[
            "targets"
        ]

        self.C = float(
            checkpoint["C"]
        )

        self.lambda_physics = float(
            checkpoint["lambda_physics"]
        )

        # ----------------------------------------
        # NORMALIZATION
        # ----------------------------------------

        self.X_mean = np.asarray(
            checkpoint["X_mean"],
            dtype=np.float32,
        )

        self.X_std = np.asarray(
            checkpoint["X_std"],
            dtype=np.float32,
        )

        self.Y_mean = np.asarray(
            checkpoint["Y_mean"],
            dtype=np.float32,
        )

        self.Y_std = np.asarray(
            checkpoint["Y_std"],
            dtype=np.float32,
        )

        # ----------------------------------------
        # VALIDATE FEATURE ORDER
        # ----------------------------------------

        if list(
            self.pinns_features
        ) != list(
            MODEL_FEATURES
        ):

            raise ValueError(
                "\nPINNs feature order mismatch.\n\n"
                f"Checkpoint features:\n"
                f"{self.pinns_features}\n\n"
                f"Expected features:\n"
                f"{MODEL_FEATURES}"
            )

        # ----------------------------------------
        # VALIDATE TARGET ORDER
        # ----------------------------------------

        if list(
            self.pinns_targets
        ) != list(
            TARGETS
        ):

            raise ValueError(
                "\nPINNs target order mismatch.\n\n"
                f"Checkpoint targets:\n"
                f"{self.pinns_targets}\n\n"
                f"Expected targets:\n"
                f"{TARGETS}"
            )

        # ----------------------------------------
        # BUILD HARD-POSITIVE MODEL
        # ----------------------------------------

        self.pinns_model = (
            TreeGrowthPINNHardPositive(
                input_size=len(
                    self.pinns_features
                ),
                hidden_size=64,
                hidden_layers=3,
                output_size=len(
                    self.pinns_targets
                ),
                volume_mean=float(
                    self.Y_mean[2]
                ),
                volume_std=float(
                    self.Y_std[2]
                ),
            )
        )

        # ----------------------------------------
        # LOAD CHECKPOINT
        # ----------------------------------------

        self.pinns_model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.pinns_model.to(
            self.device
        )

        self.pinns_model.eval()

        print(
            "  Hard-Positive PINNs: OK"
        )

        print(
            f"  Features: {self.pinns_features}"
        )

        print(
            f"  Targets: {self.pinns_targets}"
        )

        print(
            f"  Volume positivity: HARD"
        )


    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    def _validate_input(
        self,
        values,
    ):

        missing = []

        for feature in MODEL_FEATURES:

            if feature not in values:

                missing.append(
                    feature
                )

        if missing:

            raise ValueError(
                "Missing required features: "
                + ", ".join(missing)
            )

        for feature in MODEL_FEATURES:

            value = values[feature]

            if value is None:

                raise ValueError(
                    f"Feature '{feature}' cannot be None."
                )

            if not np.isfinite(
                float(value)
            ):

                raise ValueError(
                    f"Feature '{feature}' must be finite."
                )


    # ========================================================
    # PREPARE FEATURE ARRAY
    # ========================================================

    def _prepare_features(
        self,
        values,
    ):

        self._validate_input(
            values
        )

        X = np.array(
            [
                [
                    float(
                        values[feature]
                    )
                    for feature in MODEL_FEATURES
                ]
            ],
            dtype=np.float32,
        )

        return X


    # ========================================================
    # RANDOM FOREST PREDICTION
    # ========================================================

    def _predict_random_forest(
        self,
        X,
    ):

        predictions = {}

        for target in TARGETS:

            model = self.rf_models[
                target
            ]

            prediction = model.predict(
                X
            )[0]

            predictions[target] = float(
                prediction
            )

        return predictions


    # ========================================================
    # THA PREDICTION
    # ========================================================

    def _predict_tha(
        self,
        X,
    ):

        prediction = (
            self.tha_model.predict(
                X
            )[0]
        )

        return float(
            prediction
        )


    # ========================================================
    # PINNs PREDICTION
    # ========================================================

    def _predict_pinns(
        self,
        X,
    ):

        # ----------------------------------------
        # NORMALIZE INPUT
        # ----------------------------------------

        X_normalized = (
            X - self.X_mean
        ) / self.X_std

        X_tensor = torch.tensor(
            X_normalized,
            dtype=torch.float32,
            device=self.device,
        )

        # ----------------------------------------
        # PREDICTION
        # ----------------------------------------

        with torch.no_grad():

            Y_normalized = (
                self.pinns_model(
                    X_tensor
                )
            )

        Y_normalized = (
            Y_normalized
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------
        # DENORMALIZE
        # ----------------------------------------

        Y_prediction = (
            Y_normalized
            * self.Y_std
        ) + self.Y_mean

        dbh_prediction = float(
            Y_prediction[0, 0]
        )

        height_prediction = float(
            Y_prediction[0, 1]
        )

        volume_prediction = float(
            Y_prediction[0, 2]
        )

        # ----------------------------------------
        # SAFETY CHECK
        # ----------------------------------------

        if volume_prediction <= 0:

            raise RuntimeError(
                "Hard-Positive PINNs produced "
                f"non-positive volume: "
                f"{volume_prediction}"
            )

        return {
            "dbh_cm": dbh_prediction,
            "height_m": height_prediction,
            "volume_m3_tree": volume_prediction,
        }


    # ========================================================
    # MAI
    # ========================================================

    @staticmethod
    def _calculate_mai(
        volume_m3_tree,
        age_months,
    ):

        age_years = (
            float(age_months)
            / 12.0
        )

        if age_years <= 0:

            return None

        return float(
            volume_m3_tree
            / age_years
        )


    # ========================================================
    # MAIN PREDICTION
    # ========================================================

    def predict(
        self,
        values,
    ):

        X = self._prepare_features(
            values
        )

        # ----------------------------------------
        # RANDOM FOREST
        # ----------------------------------------

        rf_prediction = (
            self._predict_random_forest(
                X
            )
        )

        # ----------------------------------------
        # THA
        # ----------------------------------------

        tha_prediction = (
            self._predict_tha(
                X
            )
        )

        # ----------------------------------------
        # PINNs
        # ----------------------------------------

        pinns_prediction = (
            self._predict_pinns(
                X
            )
        )

        # ----------------------------------------
        # MAI
        # ----------------------------------------

        age_months = float(
            values["age_months"]
        )

        rf_mai = (
            self._calculate_mai(
                rf_prediction[
                    "volume_m3_tree"
                ],
                age_months,
            )
        )

        pinns_mai = (
            self._calculate_mai(
                pinns_prediction[
                    "volume_m3_tree"
                ],
                age_months,
            )
        )

        # ----------------------------------------
        # MODEL DIFFERENCE
        # ----------------------------------------

        difference = {

            "dbh_cm":
                rf_prediction[
                    "dbh_cm"
                ]
                - pinns_prediction[
                    "dbh_cm"
                ],

            "height_m":
                rf_prediction[
                    "height_m"
                ]
                - pinns_prediction[
                    "height_m"
                ],

            "volume_m3_tree":
                rf_prediction[
                    "volume_m3_tree"
                ]
                - pinns_prediction[
                    "volume_m3_tree"
                ],

            "mai_m3_tree_year":
                rf_mai
                - pinns_mai,
        }

        # ----------------------------------------
        # RESULT
        # ----------------------------------------

        result = {

            "input": {
                feature: float(
                    values[feature]
                )
                for feature in MODEL_FEATURES
            },

            "random_forest": {
                "dbh_cm":
                    rf_prediction[
                        "dbh_cm"
                    ],

                "height_m":
                    rf_prediction[
                        "height_m"
                    ],

                "volume_m3_tree":
                    rf_prediction[
                        "volume_m3_tree"
                    ],

                "tha_m":
                    tha_prediction,

                "mai_m3_tree_year":
                    rf_mai,
            },

            "pinns": {
                "dbh_cm":
                    pinns_prediction[
                        "dbh_cm"
                    ],

                "height_m":
                    pinns_prediction[
                        "height_m"
                    ],

                "volume_m3_tree":
                    pinns_prediction[
                        "volume_m3_tree"
                    ],

                "mai_m3_tree_year":
                    pinns_mai,
            },

            "difference_rf_minus_pinns":
                difference,

            "metadata": {

                "pinns_model":
                    "Hard-Positive PINNs",

                "pinns_checkpoint":
                    str(
                        PINNS_MODEL_PATH
                    ),

                "tha_definition":
                    (
                        "THA (Top Height) = "
                        "tinggi dominan tegakan, "
                        "yaitu ukuran tinggi "
                        "kelompok pohon-pohon "
                        "dominan/tertinggi."
                    ),

                "tha_source_method":
                    "Auxiliary Random Forest THA model",

                "pinns_directly_predicts_tha":
                    False,

                "mai_definition":
                    (
                        "MAI = predicted volume "
                        "per tree / age in years"
                    ),

                "mai_unit":
                    "m3/tree/year",
            },

            "units": {

                "dbh_cm":
                    "cm",

                "height_m":
                    "m",

                "volume_m3_tree":
                    "m3/tree",

                "tha_m":
                    "m",

                "mai_m3_tree_year":
                    "m3/tree/year",
            },
        }

        return result


# ============================================================
# JSON SERIALIZATION
# ============================================================

def _convert_numpy(
    obj,
):

    if isinstance(
        obj,
        np.integer,
    ):

        return int(obj)

    if isinstance(
        obj,
        np.floating,
    ):

        return float(obj)

    if isinstance(
        obj,
        np.ndarray,
    ):

        return obj.tolist()

    if isinstance(
        obj,
        dict,
    ):

        return {
            key: _convert_numpy(
                value
            )
            for key, value in obj.items()
        }

    if isinstance(
        obj,
        list,
    ):

        return [
            _convert_numpy(
                value
            )
            for value in obj
        ]

    return obj


# ============================================================
# SMOKE TEST
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(
        "STEP 22 — PREDICTION ENGINE SMOKE TEST"
    )
    print("=" * 70)

    # ----------------------------------------
    # SAMPLE INPUT
    # ----------------------------------------

    sample_input = {

        "age_months":
            60,

        "inventory_year":
            2025,

        "clone_factor":
            1.0,

        "planting_year":
            2025,

        "disease_score":
            0.0,

        "trees_per_ha":
            1100.0,
    }

    print(
        "\nSample input:"
    )

    for key, value in sample_input.items():

        print(
            f"  {key}: {value}"
        )

    # ----------------------------------------
    # LOAD ENGINE
    # ----------------------------------------

    engine = (
        TreeGrowthPredictionEngine()
    )

    # ----------------------------------------
    # PREDICT
    # ----------------------------------------

    result = engine.predict(
        sample_input
    )

    # ----------------------------------------
    # DISPLAY RF
    # ----------------------------------------

    print("\n")
    print(
        "-" * 70
    )
    print(
        "RANDOM FOREST"
    )
    print(
        "-" * 70
    )

    rf = result[
        "random_forest"
    ]

    print(
        f"  DBH    : {rf['dbh_cm']:.6f} cm"
    )

    print(
        f"  Height : {rf['height_m']:.6f} m"
    )

    print(
        f"  Volume : {rf['volume_m3_tree']:.6f} m3/tree"
    )

    print(
        f"  THA    : {rf['tha_m']:.6f} m"
    )

    print(
        f"  MAI    : {rf['mai_m3_tree_year']:.6f} "
        "m3/tree/year"
    )

    # ----------------------------------------
    # DISPLAY PINNs
    # ----------------------------------------

    print("\n")
    print(
        "-" * 70
    )
    print(
        "HARD-POSITIVE PINNs"
    )
    print(
        "-" * 70
    )

    pinns = result[
        "pinns"
    ]

    print(
        f"  DBH    : {pinns['dbh_cm']:.6f} cm"
    )

    print(
        f"  Height : {pinns['height_m']:.6f} m"
    )

    print(
        f"  Volume : {pinns['volume_m3_tree']:.6f} m3/tree"
    )

    print(
        f"  MAI    : {pinns['mai_m3_tree_year']:.6f} "
        "m3/tree/year"
    )

    # ----------------------------------------
    # POSITIVITY CHECK
    # ----------------------------------------

    volume = pinns[
        "volume_m3_tree"
    ]

    print("\n")
    print(
        "-" * 70
    )
    print(
        "HARD POSITIVITY CHECK"
    )
    print(
        "-" * 70
    )

    if volume > 0:

        print(
            f"  PASS — PINNs volume = "
            f"{volume:.9f} > 0"
        )

    else:

        raise RuntimeError(
            "FAIL — PINNs volume is not positive."
        )

    # ----------------------------------------
    # MAI CHECK
    # ----------------------------------------

    mai = pinns[
        "mai_m3_tree_year"
    ]

    if mai > 0:

        print(
            f"  PASS — PINNs MAI = "
            f"{mai:.9f} > 0"
        )

    else:

        raise RuntimeError(
            "FAIL — PINNs MAI is not positive."
        )

    # ----------------------------------------
    # DIFFERENCE
    # ----------------------------------------

    print("\n")
    print(
        "-" * 70
    )
    print(
        "RF - PINNs DIFFERENCE"
    )
    print(
        "-" * 70
    )

    difference = result[
        "difference_rf_minus_pinns"
    ]

    print(
        f"  DBH    : {difference['dbh_cm']:.6f}"
    )

    print(
        f"  Height : {difference['height_m']:.6f}"
    )

    print(
        f"  Volume : {difference['volume_m3_tree']:.6f}"
    )

    print(
        f"  MAI    : {difference['mai_m3_tree_year']:.6f}"
    )

    # ----------------------------------------
    # SAVE SMOKE TEST
    # ----------------------------------------

    output_path = (
        RESULTS_DIR
        / "prediction_engine_smoke_test.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            _convert_numpy(result),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n")
    print(
        f"Smoke test result saved:"
    )

    print(
        f"  {output_path}"
    )

    print("\n")
    print(
        "=" * 70
    )
    print(
        "STEP 22 SUCCESS"
    )
    print(
        "Prediction Engine is ready."
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()