from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_VOLUME_PATH = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "pipeline"
    / "test_volume_p18.csv"
)

PREDICTION_PATH = (
    PROJECT_ROOT
    / "results"
    / "pinns"
    / "pinns_v2_test_volume_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "pinns"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_NEGATIVE = (
    OUTPUT_DIR / "step56_negative_volume_predictions.csv"
)

OUTPUT_AGE_SUMMARY = (
    OUTPUT_DIR / "step56_volume_prediction_diagnostic_by_age.csv"
)

OUTPUT_REPORT = (
    OUTPUT_DIR / "step56_negative_volume_inspection.json"
)


def main():

    print("=" * 80)
    print("STEP 56 — PINNs V2 NEGATIVE VOLUME INSPECTION")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load official test volume data
    # ------------------------------------------------------------------
    print("\n[1/7] Loading official test volume data...")

    if not TEST_VOLUME_PATH.exists():
        raise FileNotFoundError(
            f"Test volume file not found:\n{TEST_VOLUME_PATH}"
        )

    test_df = pd.read_csv(TEST_VOLUME_PATH)

    print(f"Test rows: {len(test_df):,}")
    print(f"Columns : {list(test_df.columns)}")

    required_test_cols = [
        "age_months",
        "inventory_year",
        "clone_factor",
        "planting_year",
        "disease_score",
        "trees_per_ha",
        "volume_m3_tree",
        "compartment_id",
    ]

    missing = [
        c for c in required_test_cols
        if c not in test_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required test columns: {missing}"
        )

    # ------------------------------------------------------------------
    # 2. Load PINNs V2 predictions
    # ------------------------------------------------------------------
    print("\n[2/7] Loading PINNs V2 predictions...")

    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTION_PATH}"
        )

    pred_df = pd.read_csv(PREDICTION_PATH)

    print(f"Prediction rows: {len(pred_df):,}")
    print(f"Prediction columns: {list(pred_df.columns)}")

    # Find prediction column robustly.
    possible_prediction_cols = [
    "volume_pred",
    "predicted_volume_m3_tree",
    "volume_predicted",
    "pred_volume_m3_tree",
    "prediction",
    "predicted",
    ]

    prediction_col = None

    for col in possible_prediction_cols:
        if col in pred_df.columns:
            prediction_col = col
            break

    if prediction_col is None:
        raise ValueError(
            "Could not find volume prediction column.\n"
            f"Available columns: {list(pred_df.columns)}"
        )

    print(f"Prediction column: {prediction_col}")

    # ------------------------------------------------------------------
    # 3. Align prediction and test rows
    # ------------------------------------------------------------------
    print("\n[3/7] Aligning prediction rows with official test rows...")

    if len(test_df) != len(pred_df):
        raise ValueError(
            "Row count mismatch between test data and predictions:\n"
            f"test={len(test_df):,}\n"
            f"predictions={len(pred_df):,}"
        )

    df = test_df.copy()

    df["predicted_volume_m3_tree"] = pd.to_numeric(
        pred_df[prediction_col],
        errors="coerce"
    ).to_numpy()

    df["actual_volume_m3_tree"] = pd.to_numeric(
        df["volume_m3_tree"],
        errors="coerce"
    )

    if df["predicted_volume_m3_tree"].isna().any():
        raise ValueError(
            "Prediction file contains NaN/non-numeric values."
        )

    # ------------------------------------------------------------------
    # 4. Basic negative-volume inspection
    # ------------------------------------------------------------------
    print("\n[4/7] Inspecting negative predictions...")

    negative_mask = (
        df["predicted_volume_m3_tree"] < 0
    )

    negative_df = df.loc[negative_mask].copy()

    negative_count = len(negative_df)
    total_count = len(df)

    negative_percentage = (
        negative_count / total_count * 100
        if total_count > 0
        else 0
    )

    print(f"Total predictions : {total_count:,}")
    print(f"Negative volume   : {negative_count:,}")
    print(f"Negative percent  : {negative_percentage:.6f}%")

    if negative_count > 0:

        print("\nNegative prediction statistics:")

        print(
            negative_df[
                [
                    "age_months",
                    "actual_volume_m3_tree",
                    "predicted_volume_m3_tree",
                    "disease_score",
                    "clone_factor",
                    "trees_per_ha",
                    "inventory_year",
                    "planting_year",
                    "compartment_id",
                ]
            ].describe()
        )

        print("\nNegative predictions by age:")

        negative_by_age = (
            negative_df
            .groupby("age_months")
            .agg(
                negative_count=(
                    "predicted_volume_m3_tree",
                    "size",
                ),
                actual_mean=(
                    "actual_volume_m3_tree",
                    "mean",
                ),
                predicted_mean=(
                    "predicted_volume_m3_tree",
                    "mean",
                ),
                predicted_min=(
                    "predicted_volume_m3_tree",
                    "min",
                ),
                predicted_max=(
                    "predicted_volume_m3_tree",
                    "max",
                ),
                disease_mean=(
                    "disease_score",
                    "mean",
                ),
            )
            .reset_index()
        )

        print(negative_by_age.to_string(index=False))

    else:
        negative_by_age = pd.DataFrame()

    # ------------------------------------------------------------------
    # 5. Compare all ages
    # ------------------------------------------------------------------
    print("\n[5/7] Building complete age-level diagnostic...")

    age_summary = (
        df
        .groupby("age_months")
        .agg(
            n=("predicted_volume_m3_tree", "size"),
            actual_mean=("actual_volume_m3_tree", "mean"),
            predicted_mean=(
                "predicted_volume_m3_tree",
                "mean",
            ),
            predicted_min=(
                "predicted_volume_m3_tree",
                "min",
            ),
            predicted_max=(
                "predicted_volume_m3_tree",
                "max",
            ),
            predicted_std=(
                "predicted_volume_m3_tree",
                "std",
            ),
            negative_count=(
                "predicted_volume_m3_tree",
                lambda x: int((x < 0).sum()),
            ),
            negative_pct=(
                "predicted_volume_m3_tree",
                lambda x: float((x < 0).mean() * 100),
            ),
        )
        .reset_index()
    )

    age_summary["absolute_error_mean"] = (
        age_summary["predicted_mean"]
        - age_summary["actual_mean"]
    ).abs()

    print(age_summary.to_string(index=False))

    # ------------------------------------------------------------------
    # 6. Check whether negatives occur outside P18
    # ------------------------------------------------------------------
    print("\n[6/7] Checking age localization...")

    negative_ages = sorted(
        negative_df["age_months"].unique().tolist()
    )

    print(f"Negative prediction ages: {negative_ages}")

    if negative_count == 0:
        localization = "NO_NEGATIVE_PREDICTIONS"
    elif negative_ages == [18]:
        localization = "NEGATIVES_ONLY_AT_P18"
    else:
        localization = "NEGATIVES_AT_MULTIPLE_AGES"

    print(f"Localization result: {localization}")

    # ------------------------------------------------------------------
    # 7. Save diagnostic outputs
    # ------------------------------------------------------------------
    print("\n[7/7] Saving inspection outputs...")

    negative_df.to_csv(
        OUTPUT_NEGATIVE,
        index=False,
    )

    age_summary.to_csv(
        OUTPUT_AGE_SUMMARY,
        index=False,
    )

    report = {
        "step": 56,
        "purpose": "Inspect negative PINNs V2 volume predictions before any model modification",
        "test_rows": int(total_count),
        "negative_volume_count": int(negative_count),
        "negative_volume_percentage": float(negative_percentage),
        "negative_prediction_ages": negative_ages,
        "localization": localization,
        "min_predicted_volume": float(
            df["predicted_volume_m3_tree"].min()
        ),
        "max_predicted_volume": float(
            df["predicted_volume_m3_tree"].max()
        ),
        "mean_predicted_volume": float(
            df["predicted_volume_m3_tree"].mean()
        ),
        "dataset_modified": False,
        "model_modified": False,
        "retraining_performed": False,
    }

    OUTPUT_REPORT.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("STEP 56 COMPLETE")
    print("=" * 80)

    print("\nOutputs:")
    print(OUTPUT_NEGATIVE)
    print(OUTPUT_AGE_SUMMARY)
    print(OUTPUT_REPORT)

    print("\nStatus:")
    print("Dataset modified     : NO")
    print("Model modified       : NO")
    print("Retraining performed : NO")

    print("\nNext decision will be based ONLY on this inspection.")


if __name__ == "__main__":
    main()