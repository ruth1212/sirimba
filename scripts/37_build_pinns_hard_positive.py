from pathlib import Path

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

TRAIN_PATH = RESULTS / "model_train.csv"
CONFIG_PATH = RESULTS / "pinns_hard_positive_model_config.json"


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


class TreeGrowthPINNHardPositive(nn.Module):
    def __init__(self, input_dim, volume_mean, volume_std):
        super().__init__()

        self.volume_mean = float(volume_mean)
        self.volume_std = float(volume_std)

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 3)
        )

        self.softplus = nn.Softplus()

    def forward(self, x):
        raw_output = self.network(x)

        # DBH dan Height tetap berada pada normalized target space
        dbh_normalized = raw_output[:, 0:1]
        height_normalized = raw_output[:, 1:2]

        # Volume:
        # 1. network menghasilkan latent raw-volume
        # 2. Softplus diterapkan pada RAW physical space
        # 3. hasil positif dikembalikan ke normalized target space
        volume_raw = self.softplus(raw_output[:, 2:3])

        volume_normalized = (
            volume_raw - self.volume_mean
        ) / self.volume_std

        return torch.cat(
            [
                dbh_normalized,
                height_normalized,
                volume_normalized
            ],
            dim=1
        )


def main():

    print("=" * 75)
    print("TREE GROWTH ANALYTICS")
    print("STEP 21M - BUILD HARD-POSITIVE PINNs")
    print("=" * 75)

    print("\n[1/6] Checking device...")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device : {device}"
    )

    print("\n[2/6] Loading training data...")

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found:\n"
            f"{TRAIN_PATH}"
        )

    df = pd.read_csv(
        TRAIN_PATH
    )

    print(
        f"Training rows : {len(df):,}"
    )

    missing = [
        col
        for col in FEATURES + TARGETS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    print(
        "\n[3/6] Calculating normalization..."
    )

    X = df[
        FEATURES
    ].to_numpy(
        dtype=np.float32
    )

    Y = df[
        TARGETS
    ].to_numpy(
        dtype=np.float32
    )

    X_mean = X.mean(
        axis=0
    )

    X_std = X.std(
        axis=0
    )

    Y_mean = Y.mean(
        axis=0
    )

    Y_std = Y.std(
        axis=0
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

    print(
        "\nInput normalization:"
    )

    for i, feature in enumerate(FEATURES):

        print(
            f"  {feature}: "
            f"mean={X_mean[i]:.6f}, "
            f"std={X_std[i]:.6f}"
        )

    print(
        "\nTarget normalization:"
    )

    for i, target in enumerate(TARGETS):

        print(
            f"  {target}: "
            f"mean={Y_mean[i]:.6f}, "
            f"std={Y_std[i]:.6f}"
        )

    print(
        "\n[4/6] Building hard-positive PINNs..."
    )

    model = TreeGrowthPINNHardPositive(
    input_dim=len(FEATURES),
    volume_mean=Y_mean[2],
    volume_std=Y_std[2]
)

    model.to(device)

    print(model)

    print(
        "\n[5/6] Testing forward pass..."
    )

    sample = torch.tensor(
        (
            (
                X[:8]
                - X_mean
            )
            / X_std
        ),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():

        output = model(
            sample
        )

    output_np = (
        output
        .cpu()
        .numpy()
    )

    print(
        f"Input shape  : "
        f"{tuple(sample.shape)}"
    )

    print(
        f"Output shape : "
        f"{tuple(output.shape)}"
    )

    volume_output = output_np[
        :, 2
    ]

    print(
        "\nRaw normalized volume output:"
    )

    print(
        f"Minimum : "
        f"{volume_output.min():.9f}"
    )

    print(
        f"Maximum : "
        f"{volume_output.max():.9f}"
    )

    if np.any(
        volume_output <= 0
    ):

        raise RuntimeError(
            "Hard-positive constraint failed: "
            "volume output is not strictly positive."
        )

    print(
        "\nHARD POSITIVITY CHECK : PASSED"
    )

    print(
        "All normalized volume outputs "
        "are strictly positive."
    )

    print(
        "\n[6/6] Saving configuration..."
    )

    config = {
        "model_type": (
            "PINNs Hard Positive"
        ),
        "architecture": {
            "input_size": 6,
            "hidden_size": 64,
            "hidden_layers": 3,
            "output_size": 3,
        },
        "features": FEATURES,
        "targets": TARGETS,
        "volume_constraint": (
            "Softplus(raw_volume)"
        ),
        "volume_constraint_type": (
            "hard_positive"
        ),
        "device": str(device),
        "training_performed": False,
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    print(
        f"Saved configuration:\n"
        f"  {CONFIG_PATH}"
    )

    print("\n")
    print("=" * 75)
    print("STEP 21M COMPLETED")
    print("=" * 75)

    print(
        "Hard-positive PINNs architecture "
        "successfully built."
    )

    print(
        "NO TRAINING WAS PERFORMED."
    )

    print("=" * 75)


if __name__ == "__main__":
    main()