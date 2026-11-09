from pathlib import Path
import importlib.util
import json
import time

import numpy as np
import pandas as pd
import torch


# ============================================================
# PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results"
MODEL_PATH = RESULTS_DIR / "pinns_models" / "best_pinns_positive_model.pt"
TEST_PATH = RESULTS_DIR / "model_test.csv"

BASELINE_METRICS_PATH = RESULTS_DIR / "model_comparison_metrics.csv"

OUTPUT_METRICS_PATH = RESULTS_DIR / "pinns_positive_test_metrics.csv"
OUTPUT_PREDICTIONS_PATH = RESULTS_DIR / "pinns_positive_test_predictions.csv"
OUTPUT_CONFIG_PATH = RESULTS_DIR / "pinns_positive_evaluation_config.json"


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

EXPECTED_TEST_ROWS = 121_498


# ============================================================
# LOAD MODEL CLASS
# ============================================================

def load_pinns_class():
    model_file = ROOT / "scripts" / "12_build_pinns_model.py"

    if not model_file.exists():
        raise FileNotFoundError(
            f"PINNs model definition not found:\n{model_file}"
        )

    spec = importlib.util.spec_from_file_location(
        "pinns_model_definition",
        model_file,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "TreeGrowthPINN"):
        raise AttributeError(
            "TreeGrowthPINN class was not found in "
            "scripts/12_build_pinns_model.py"
        )

    return module.TreeGrowthPINN


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    error = y_true - y_pred

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1.0 - (ss_res / ss_tot)

    nonzero_mask = np.abs(y_true) > 1e-12

    if np.any(nonzero_mask):
        mape = (
            np.mean(
                np.abs(
                    error[nonzero_mask]
                    / y_true[nonzero_mask]
                )
            )
            * 100.0
        )
    else:
        mape = np.nan

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE_percent": float(mape),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 75)
    print("TREE GROWTH ANALYTICS")
    print("STEP 21J - EVALUATE PINNs POSITIVE-CONSTRAINT")
    print("=" * 75)

    # --------------------------------------------------------
    # 1/9 DEVICE
    # --------------------------------------------------------

    print("\n[1/9] Selecting device...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device : {device}")

    # --------------------------------------------------------
    # 2/9 CHECK FILES
    # --------------------------------------------------------

    print("\n[2/9] Checking required files...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Positive PINNs checkpoint not found:\n{MODEL_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_PATH}"
        )

    print(f"Positive model : {MODEL_PATH}")
    print(f"Test dataset  : {TEST_PATH}")

    # --------------------------------------------------------
    # 3/9 LOAD TEST DATA
    # --------------------------------------------------------

    print("\n[3/9] Loading test data...")

    test_df = pd.read_csv(TEST_PATH)

    print(f"Test rows : {len(test_df):,}")
    print(f"Test cols : {len(test_df.columns)}")

    if len(test_df) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Unexpected test row count.\n"
            f"Expected : {EXPECTED_TEST_ROWS:,}\n"
            f"Actual   : {len(test_df):,}"
        )

    missing_features = [
        col for col in FEATURES
        if col not in test_df.columns
    ]

    missing_targets = [
        col for col in TARGETS
        if col not in test_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns: {missing_features}"
        )

    if missing_targets:
        raise ValueError(
            f"Missing target columns: {missing_targets}"
        )

    print("Feature columns : OK")
    print("Target columns  : OK")

    # --------------------------------------------------------
    # 4/9 LOAD CHECKPOINT
    # --------------------------------------------------------

    print("\n[4/9] Loading positive PINNs checkpoint...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    required_checkpoint_keys = [
        "model_state_dict",
        "features",
        "targets",
        "X_mean",
        "X_std",
        "Y_mean",
        "Y_std",
    ]

    missing_keys = [
        key
        for key in required_checkpoint_keys
        if key not in checkpoint
    ]

    if missing_keys:
        raise ValueError(
            f"Checkpoint missing required keys: {missing_keys}"
        )

    print("Checkpoint loaded successfully.")

    print(
        f"Best epoch : "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Best loss  : "
        f"{checkpoint.get('best_loss', 'unknown')}"
    )

    print(
        f"Physics lambda : "
        f"{checkpoint.get('lambda_physics', 'unknown')}"
    )

    print(
        f"Positive lambda : "
        f"{checkpoint.get('lambda_positive', 'unknown')}"
    )

    # --------------------------------------------------------
    # 5/9 BUILD MODEL
    # --------------------------------------------------------

    print("\n[5/9] Building PINNs model...")

    TreeGrowthPINN = load_pinns_class()

    model = TreeGrowthPINN(
        input_size=len(FEATURES),
        hidden_size=64,
        hidden_layers=3,
        output_size=len(TARGETS),
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    print(model)

    # --------------------------------------------------------
    # 6/9 NORMALIZATION
    # --------------------------------------------------------

    print("\n[6/9] Preparing normalized test data...")

    X = test_df[FEATURES].to_numpy(
        dtype=np.float32
    )

    Y = test_df[TARGETS].to_numpy(
        dtype=np.float32
    )

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

    # Safety against zero standard deviation.
    X_std_safe = np.where(
        X_std == 0,
        1.0,
        X_std,
    )

    Y_std_safe = np.where(
        Y_std == 0,
        1.0,
        Y_std,
    )

    X_normalized = (
        X - X_mean
    ) / X_std_safe

    print("Normalization statistics loaded from checkpoint.")

    # --------------------------------------------------------
    # 7/9 PREDICTION
    # --------------------------------------------------------

    print("\n[7/9] Running positive PINNs predictions...")

    X_tensor = torch.tensor(
        X_normalized,
        dtype=torch.float32,
        device=device,
    )

    prediction_batches = []

    batch_size = 8192

    with torch.no_grad():

        for start in range(
            0,
            len(X_tensor),
            batch_size,
        ):

            end = min(
                start + batch_size,
                len(X_tensor),
            )

            batch = X_tensor[start:end]

            output = model(batch)

            output_np = (
                output
                .detach()
                .cpu()
                .numpy()
            )

            prediction_batches.append(output_np)

    predictions_normalized = np.vstack(
        prediction_batches
    )

    predictions = (
        predictions_normalized
        * Y_std_safe
    ) + Y_mean

    print(
        f"Predictions generated : "
        f"{len(predictions):,}"
    )

    if len(predictions) != len(test_df):
        raise ValueError(
            "Prediction row count does not match test data."
        )

    # --------------------------------------------------------
    # 8/9 METRICS + NEGATIVE VOLUME ANALYSIS
    # --------------------------------------------------------

    print("\n[8/9] Calculating metrics and volume constraint...")

    metric_rows = []

    for i, target in enumerate(TARGETS):

        metrics = calculate_metrics(
            Y[:, i],
            predictions[:, i],
        )

        metric_rows.append(
            {
                "model": "PINNs Positive Constraint",
                "target": target,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "MAPE_percent": metrics["MAPE_percent"],
            }
        )

    metrics_df = pd.DataFrame(metric_rows)

    volume_index = TARGETS.index(
        "volume_m3_tree"
    )

    predicted_volume = predictions[
        :, volume_index
    ]

    negative_mask = predicted_volume < 0

    negative_count = int(
        negative_mask.sum()
    )

    negative_percentage = (
        negative_count
        / len(predicted_volume)
        * 100.0
    )

    minimum_volume = float(
        np.min(predicted_volume)
    )

    maximum_volume = float(
        np.max(predicted_volume)
    )

    mean_volume = float(
        np.mean(predicted_volume)
    )

    print("\nMetric results:")
    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nPositive-volume validation:")
    print(
        f"Negative volume predictions : "
        f"{negative_count:,}"
    )

    print(
        f"Negative percentage          : "
        f"{negative_percentage:.6f}%"
    )

    print(
        f"Minimum predicted volume     : "
        f"{minimum_volume:.9f}"
    )

    print(
        f"Maximum predicted volume     : "
        f"{maximum_volume:.9f}"
    )

    print(
        f"Mean predicted volume        : "
        f"{mean_volume:.9f}"
    )

    # --------------------------------------------------------
    # DETAILED PREDICTION DATASET
    # --------------------------------------------------------

    prediction_df = test_df.copy()

    for i, target in enumerate(TARGETS):

        prediction_df[
            f"pinns_positive_{target}"
        ] = predictions[:, i]

        prediction_df[
            f"error_{target}"
        ] = (
            predictions[:, i]
            - Y[:, i]
        )

    prediction_df[
        "pinns_positive_volume_negative"
    ] = negative_mask

    # --------------------------------------------------------
    # P6 SPECIFIC CHECK
    # --------------------------------------------------------

    if "age_months" in prediction_df.columns:

        p6_mask = (
            prediction_df["age_months"]
            == 6
        )

        p6_count = int(
            p6_mask.sum()
        )

        p6_negative_count = int(
            (
                p6_mask
                & negative_mask
            ).sum()
        )

        print("\nP6 validation:")
        print(
            f"P6 test rows               : "
            f"{p6_count:,}"
        )

        print(
            f"P6 negative predictions    : "
            f"{p6_negative_count:,}"
        )

    else:

        p6_count = None
        p6_negative_count = None

    # --------------------------------------------------------
    # 9/9 SAVE
    # --------------------------------------------------------

    print("\n[9/9] Saving evaluation results...")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.to_csv(
        OUTPUT_METRICS_PATH,
        index=False,
    )

    prediction_df.to_csv(
        OUTPUT_PREDICTIONS_PATH,
        index=False,
    )

    config = {
        "evaluation_type": (
            "PINNs positive-volume constraint"
        ),
        "model_path": str(
            MODEL_PATH
        ),
        "test_path": str(
            TEST_PATH
        ),
        "device": str(device),
        "features": FEATURES,
        "targets": TARGETS,
        "test_rows": int(len(test_df)),
        "expected_test_rows": (
            EXPECTED_TEST_ROWS
        ),
        "best_epoch": checkpoint.get(
            "epoch"
        ),
        "best_loss": checkpoint.get(
            "best_loss"
        ),
        "lambda_physics": checkpoint.get(
            "lambda_physics"
        ),
        "lambda_positive": checkpoint.get(
            "lambda_positive"
        ),
        "negative_volume_count": (
            negative_count
        ),
        "negative_volume_percentage": (
            negative_percentage
        ),
        "minimum_predicted_volume": (
            minimum_volume
        ),
        "maximum_predicted_volume": (
            maximum_volume
        ),
        "mean_predicted_volume": (
            mean_volume
        ),
        "p6_test_rows": p6_count,
        "p6_negative_volume_count": (
            p6_negative_count
        ),
        "training_baseline_model_replaced": False,
    }

    with open(
        OUTPUT_CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    print("\n")
    print("=" * 75)
    print("PINNs POSITIVE-CONSTRAINT EVALUATION COMPLETED")
    print("=" * 75)

    print(
        f"Test rows             : "
        f"{len(test_df):,}"
    )

    print(
        f"Negative volumes      : "
        f"{negative_count:,}"
    )

    print(
        f"Negative percentage   : "
        f"{negative_percentage:.6f}%"
    )

    print(
        f"Minimum volume        : "
        f"{minimum_volume:.9f}"
    )

    print(
        f"Evaluation time       : "
        f"{elapsed:.2f} seconds"
    )

    print("\nSaved:")
    print(
        f"  - {OUTPUT_METRICS_PATH}"
    )

    print(
        f"  - {OUTPUT_PREDICTIONS_PATH}"
    )

    print(
        f"  - {OUTPUT_CONFIG_PATH}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()