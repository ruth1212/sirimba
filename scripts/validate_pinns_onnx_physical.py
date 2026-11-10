from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort

from export_pinns_onnx import TreeGrowthPINN


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "results" / "pinns" / "models"
PREP_DIR = ROOT / "data" / "model_ready" / "pipeline" / "preprocessing"

CHECKPOINT = MODEL_DIR / "pinns_v2_growth_model.pt"
ONNX_MODEL = MODEL_DIR / "pinns_v2_growth_model.onnx"
TEST_FEATURES = PREP_DIR / "test_features_scaled.csv"

TARGETS = ["dbh_cm", "height_m", "volume_m3_tree"]


def load_model():
    model = TreeGrowthPINN(6, 64)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)
    model.eval()
    return model


def inverse_scale(predictions, target_scalers):
    result = np.zeros_like(predictions, dtype=np.float64)

    for index, target in enumerate(TARGETS):
        result[:, index] = target_scalers[target].inverse_transform(
            predictions[:, index].reshape(-1, 1)
        ).ravel()

    return result


def main():
    X = pd.read_csv(TEST_FEATURES).to_numpy(dtype=np.float32)

    feature_scaler = joblib.load(PREP_DIR / "feature_scaler.joblib")
    target_scalers = joblib.load(PREP_DIR / "target_scaler.joblib")

    if X.shape[1] != feature_scaler.n_features_in_:
        raise ValueError("Jumlah fitur tidak sesuai dengan feature scaler.")

    model = load_model()

    with torch.no_grad():
        torch_scaled = model(torch.from_numpy(X)).numpy()

    session = ort.InferenceSession(
        str(ONNX_MODEL),
        providers=["CPUExecutionProvider"],
    )

    onnx_scaled = session.run(
        ["predictions_scaled"],
        {"features_scaled": X},
    )[0]

    torch_raw = inverse_scale(torch_scaled, target_scalers)
    onnx_raw = inverse_scale(onnx_scaled, target_scalers)

    print("=" * 65)
    print("SIRIMBA - PINNs PHYSICAL UNIT VALIDATION")
    print("=" * 65)

    all_passed = True

    for index, target in enumerate(TARGETS):
        diff = np.abs(torch_raw[:, index] - onnx_raw[:, index])

        passed = np.allclose(
            torch_raw[:, index],
            onnx_raw[:, index],
            rtol=1e-4,
            atol=1e-5,
        )

        all_passed = all_passed and passed

        print(f"\n{target}")
        print("PyTorch mean :", float(np.mean(torch_raw[:, index])))
        print("ONNX mean    :", float(np.mean(onnx_raw[:, index])))
        print("MAE diff     :", float(np.mean(diff)))
        print("Max diff     :", float(np.max(diff)))
        print("Agreement    :", "PASSED" if passed else "FAILED")

    print("\nOVERALL:", "PASSED" if all_passed else "FAILED")
    print("=" * 65)

    if not all_passed:
        raise RuntimeError("Physical-unit comparison failed.")


if __name__ == "__main__":
    main()
