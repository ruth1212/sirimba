from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "pinns"
    / "pinns_v2_test_volume_predictions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "pinns"
    / "pinns_v2_test_volume_predictions_constrained.csv"
)

AGE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "pinns"
    / "pinns_v2_constrained_volume_by_age.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "results"
    / "pinns"
    / "step58_volume_constraint_report.json"
)


def main():

    print("=" * 80)
    print("STEP 58 — PINNs V2 POSITIVE VOLUME OUTPUT CONSTRAINT")
    print("=" * 80)

    # ---------------------------------------------------------------
    # 1. Load original PINNs V2 predictions
    # ---------------------------------------------------------------
    print("\n[1/6] Loading PINNs V2 predictions...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required = [
        "compartment_id",
        "age_months",
        "volume_m3_tree",
        "volume_pred",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    print(f"Rows: {len(df):,}")

    # ---------------------------------------------------------------
    # 2. Preserve original prediction
    # ---------------------------------------------------------------
    print("\n[2/6] Preserving raw PINNs V2 output...")

    df["volume_pred_raw"] = pd.to_numeric(
        df["volume_pred"],
        errors="coerce"
    )

    if df["volume_pred_raw"].isna().any():
        raise ValueError(
            "Found NaN in PINNs V2 predictions."
        )

    # ---------------------------------------------------------------
    # 3. Apply physical constraint
    # ---------------------------------------------------------------
    print("\n[3/6] Applying volume >= 0 constraint...")

    negative_mask = (
        df["volume_pred_raw"] < 0
    )

    negative_before = int(
        negative_mask.sum()
    )

    print(
        f"Negative predictions before: "
        f"{negative_before:,}"
    )

    df["volume_pred"] = np.maximum(
        df["volume_pred_raw"].to_numpy(),
        0.0
    )

    negative_after = int(
        (df["volume_pred"] < 0).sum()
    )

    print(
        f"Negative predictions after : "
        f"{negative_after:,}"
    )

    # ---------------------------------------------------------------
    # 4. Diagnostic
    # ---------------------------------------------------------------
    print("\n[4/6] Building diagnostic...")

    df["absolute_error_raw"] = (
        df["volume_pred_raw"]
        - df["volume_m3_tree"]
    ).abs()

    df["absolute_error_constrained"] = (
        df["volume_pred"]
        - df["volume_m3_tree"]
    ).abs()

    age_summary = (
        df
        .groupby("age_months")
        .agg(
            n=("volume_pred", "size"),

            actual_mean=(
                "volume_m3_tree",
                "mean"
            ),

            raw_pred_mean=(
                "volume_pred_raw",
                "mean"
            ),

            constrained_pred_mean=(
                "volume_pred",
                "mean"
            ),

            raw_pred_min=(
                "volume_pred_raw",
                "min"
            ),

            constrained_pred_min=(
                "volume_pred",
                "min"
            ),

            raw_negative_count=(
                "volume_pred_raw",
                lambda x: int((x < 0).sum())
            ),

            constrained_negative_count=(
                "volume_pred",
                lambda x: int((x < 0).sum())
            ),

            raw_mae=(
                "absolute_error_raw",
                "mean"
            ),

            constrained_mae=(
                "absolute_error_constrained",
                "mean"
            ),
        )
        .reset_index()
    )

    print("\nAge trajectory:")

    print(
        age_summary.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # 5. Save
    # ---------------------------------------------------------------
    print("\n[5/6] Saving constrained predictions...")

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    age_summary.to_csv(
        AGE_OUTPUT_PATH,
        index=False
    )

    # ---------------------------------------------------------------
    # 6. Final report
    # ---------------------------------------------------------------
    print("\n[6/6] Saving report...")

    total = len(df)

    changed_count = int(
        (
            df["volume_pred_raw"]
            != df["volume_pred"]
        ).sum()
    )

    raw_mae = float(
        df["absolute_error_raw"].mean()
    )

    constrained_mae = float(
        df["absolute_error_constrained"].mean()
    )

    report = {
        "step": 58,

        "method": (
            "Physical output constraint: "
            "volume_prediction = max(raw_prediction, 0)"
        ),

        "total_predictions": int(total),

        "negative_before": int(
            negative_before
        ),

        "negative_after": int(
            negative_after
        ),

        "changed_predictions": int(
            changed_count
        ),

        "changed_percentage": float(
            changed_count / total * 100
        ),

        "raw_min_volume": float(
            df["volume_pred_raw"].min()
        ),

        "constrained_min_volume": float(
            df["volume_pred"].min()
        ),

        "raw_mae": raw_mae,

        "constrained_mae": constrained_mae,

        "mae_change": (
            constrained_mae - raw_mae
        ),

        "dataset_modified": False,

        "checkpoint_modified": False,

        "retraining_performed": False,

        "constraint_only": True,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print("STEP 58 COMPLETE")
    print("=" * 80)

    print(
        f"\nNegative before : {negative_before:,}"
    )

    print(
        f"Negative after  : {negative_after:,}"
    )

    print(
        f"Changed rows    : {changed_count:,}"
    )

    print(
        f"Raw MAE         : {raw_mae:.6f}"
    )

    print(
        f"Constrained MAE : {constrained_mae:.6f}"
    )

    print("\nDataset modified : NO")
    print("Model checkpoint : NO")
    print("Retraining       : NO")

    print("\nOutputs:")
    print(OUTPUT_PATH)
    print(AGE_OUTPUT_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()