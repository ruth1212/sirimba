from pathlib import Path

import numpy as np
import pandas as pd
import torch
import onnxruntime as ort

from export_pinns_onnx import TreeGrowthPINN


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "results" / "pinns" / "models"
PREPROCESSING_DIR = (
    ROOT / "data" / "model_ready" / "pipeline" / "preprocessing"
)

CHECKPOINT = MODEL_DIR / "pinns_v2_growth_model.pt"
ONNX_MODEL = MODEL_DIR / "pinns_v2_growth_model.onnx"
TEST_FEATURES = PREPROCESSING_DIR / "test_features_scaled.csv"


def load_pytorch_model():

    model = TreeGrowthPINN(
        input_size=6,
        hidden_size=64,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
    )

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def main():

    df = pd.read_csv(TEST_FEATURES)

    X = df.to_numpy(dtype=np.float32)

    model = load_pytorch_model()

    with torch.no_grad():
        pytorch_output = model(
            torch.from_numpy(X)
        ).numpy()

    session = ort.InferenceSession(
        str(ONNX_MODEL),
        providers=["CPUExecutionProvider"],
    )

    onnx_output = session.run(
        ["predictions_scaled"],
        {"features_scaled": X},
    )[0]

    difference = np.abs(
        pytorch_output - onnx_output
    )

    print("=" * 65)
    print("SIRIMBA - PINNs ONNX REAL TEST VALIDATION")
    print("=" * 65)

    print("Test rows:", len(X))
    print("Input shape:", X.shape)
    print("PyTorch shape:", pytorch_output.shape)
    print("ONNX shape:", onnx_output.shape)

    print("\nOUTPUT COMPARISON")

    for index, name in enumerate(
        ["dbh_cm", "height_m", "volume_m3_tree"]
    ):

        errors = difference[:, index]

        mae = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        max_error = float(np.max(errors))

        print(f"\n{name}")
        print("MAE difference :", mae)
        print("RMSE difference:", rmse)
        print("Maximum error  :", max_error)

    passed = np.allclose(
        pytorch_output,
        onnx_output,
        rtol=1e-4,
        atol=1e-5,
    )

    print("\nNUMERICAL AGREEMENT:", "PASSED" if passed else "FAILED")
    print("=" * 65)

    if not passed:
        raise RuntimeError(
            "ONNX and PyTorch outputs are not sufficiently close."
        )


if __name__ == "__main__":
    main()
