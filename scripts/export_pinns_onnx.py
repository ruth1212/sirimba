from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "pinns"
    / "models"
    / "pinns_v2_growth_model.pt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "pinns"
    / "models"
    / "pinns_v2_growth_model.onnx"
)


class TreeGrowthPINN(nn.Module):

    def __init__(self, input_size=6, hidden_size=64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),

            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),

            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.dbh_head = nn.Linear(hidden_size, 1)
        self.height_head = nn.Linear(hidden_size, 1)
        self.volume_head = nn.Linear(hidden_size, 1)

    def forward(self, x):

        shared_features = self.shared(x)

        dbh = self.dbh_head(shared_features)
        height = self.height_head(shared_features)
        volume = self.volume_head(shared_features)

        return torch.cat(
            [dbh, height, volume],
            dim=1,
        )


def main():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    print("=" * 60)
    print("SIRIMBA - PINNs V2 ONNX EXPORT")
    print("=" * 60)

    model = TreeGrowthPINN(
        input_size=6,
        hidden_size=64,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print("PyTorch checkpoint : LOADED")

    dummy_input = torch.randn(4, 6, dtype=torch.float32)

    with torch.no_grad():
        pytorch_output = model(dummy_input).numpy()

    torch.onnx.export(
        model,
        dummy_input,
        str(OUTPUT_PATH),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["features_scaled"],
        output_names=["predictions_scaled"],
        dynamic_axes={
            "features_scaled": {0: "batch_size"},
            "predictions_scaled": {0: "batch_size"},
        },
        dynamo=False,
    )

    print("ONNX export        : COMPLETED")

    onnx_model = onnx.load(str(OUTPUT_PATH))
    onnx.checker.check_model(onnx_model)

    print("ONNX checker       : PASSED")

    session = ort.InferenceSession(
        str(OUTPUT_PATH),
        providers=["CPUExecutionProvider"],
    )

    onnx_output = session.run(
        ["predictions_scaled"],
        {"features_scaled": dummy_input.numpy()},
    )[0]

    max_difference = float(
        np.max(np.abs(pytorch_output - onnx_output))
    )

    is_close = np.allclose(
        pytorch_output,
        onnx_output,
        rtol=1e-4,
        atol=1e-5,
    )

    print("PyTorch output shape:", pytorch_output.shape)
    print("ONNX output shape   :", onnx_output.shape)
    print("Maximum difference  :", max_difference)
    print("Numerical agreement :", "PASSED" if is_close else "FAILED")

    print(
        "ONNX file size      :",
        round(OUTPUT_PATH.stat().st_size / 1024, 2),
        "KB",
    )

    if not is_close:
        raise RuntimeError(
            "ONNX output differs from PyTorch. "
            "Original model remains unchanged."
        )

    print("=" * 60)
    print("EXPORT AND VALIDATION SUCCESS")
    print("=" * 60)


if __name__ == "__main__":
    main()
