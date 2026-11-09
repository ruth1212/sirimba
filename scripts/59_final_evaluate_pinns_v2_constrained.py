from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DBH_HEIGHT = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "pipeline"
    / "test_dbh_height.csv"
)

TEST_VOLUME = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "pipeline"
    / "test_volume_p18.csv"
)

DBH_HEIGHT_PRED = (
    PROJECT_ROOT
    / "results"
    / "pinns"
    / "pinns_v2_test_dbh_height_predictions.csv"
)

VOLUME_PRED = (
    PROJECT_ROOT
    / "results"
    / "pinns"
    / "pinns_v2_test_volume_predictions_constrained.csv"
)

OUTPUT_METRICS = (
    PROJECT_ROOT
    / "results"
    / "pinns"
    / "pinns_v2_final_test_metrics.csv"
)

OUTPUT_AGE = (
    PROJECT_ROOT
    / "results"
    / "pinns"
    / "pinns_v2_final_actual_vs_predicted_by_age.csv"
)

OUTPUT_PHI = (
    PROJECT_ROOT
    / "results"
    / "pinns"
    / "pinns_v2_final_phi_metrics.csv"
)

OUTPUT_REPORT = (
    PROJECT_ROOT
    / "results"
    / "pinns"
    / "step59_final_evaluation_report.json"
)


def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    nonzero_mask = actual != 0

    if nonzero_mask.any():
        mape = (
            np.mean(
                np.abs(
                    (
                        actual[nonzero_mask]
                        - predicted[nonzero_mask]
                    )
                    / actual[nonzero_mask]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE_percent": float(mape),
    }


def find_column(df, candidates, label):

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find {label} column.\n"
        f"Available columns: {list(df.columns)}"
    )


def main():

    print("=" * 80)
    print("STEP 59 — FINAL EVALUATION PINNs V2 CONSTRAINED")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load DBH + Height actual
    # ------------------------------------------------------------------
    print("\n[1/8] Loading DBH + Height test data...")

    dbh_height_actual = pd.read_csv(
        TEST_DBH_HEIGHT
    )

    dbh_height_pred = pd.read_csv(
        DBH_HEIGHT_PRED
    )

    print(
        f"Actual DBH/Height rows : "
        f"{len(dbh_height_actual):,}"
    )

    print(
        f"Prediction rows        : "
        f"{len(dbh_height_pred):,}"
    )

    if len(dbh_height_actual) != len(dbh_height_pred):
        raise ValueError(
            "DBH/Height row count mismatch."
        )

    dbh_pred_col = find_column(
        dbh_height_pred,
        [
            "dbh_pred",
            "predicted_dbh_cm",
            "dbh_cm_pred",
            "prediction_dbh",
        ],
        "DBH prediction",
    )

    height_pred_col = find_column(
        dbh_height_pred,
        [
            "height_pred",
            "predicted_height_m",
            "height_m_pred",
            "prediction_height",
        ],
        "Height prediction",
    )

    # ------------------------------------------------------------------
    # 2. Load constrained volume
    # ------------------------------------------------------------------
    print("\n[2/8] Loading constrained volume predictions...")

    volume_actual = pd.read_csv(
        TEST_VOLUME
    )

    volume_pred = pd.read_csv(
        VOLUME_PRED
    )

    print(
        f"Actual volume rows : "
        f"{len(volume_actual):,}"
    )

    print(
        f"Prediction rows    : "
        f"{len(volume_pred):,}"
    )

    if len(volume_actual) != len(volume_pred):
        raise ValueError(
            "Volume row count mismatch."
        )

    volume_pred_col = find_column(
        volume_pred,
        [
            "volume_pred",
            "predicted_volume_m3_tree",
            "volume_predicted",
        ],
        "volume prediction",
    )

    # ------------------------------------------------------------------
    # 3. Build actual/prediction tables
    # ------------------------------------------------------------------
    print("\n[3/8] Building evaluation tables...")

    dbh_df = pd.DataFrame(
        {
            "compartment_id": dbh_height_actual[
                "compartment_id"
            ].to_numpy(),

            "age_months": dbh_height_actual[
                "age_months"
            ].to_numpy(),

            "actual": dbh_height_actual[
                "dbh_cm"
            ].to_numpy(),

            "predicted": dbh_height_pred[
                dbh_pred_col
            ].to_numpy(),
        }
    )

    height_df = pd.DataFrame(
        {
            "compartment_id": dbh_height_actual[
                "compartment_id"
            ].to_numpy(),

            "age_months": dbh_height_actual[
                "age_months"
            ].to_numpy(),

            "actual": dbh_height_actual[
                "height_m"
            ].to_numpy(),

            "predicted": dbh_height_pred[
                height_pred_col
            ].to_numpy(),
        }
    )

    volume_df = pd.DataFrame(
        {
            "compartment_id": volume_actual[
                "compartment_id"
            ].to_numpy(),

            "age_months": volume_actual[
                "age_months"
            ].to_numpy(),

            "actual": volume_actual[
                "volume_m3_tree"
            ].to_numpy(),

            "predicted": volume_pred[
                volume_pred_col
            ].to_numpy(),
        }
    )

    # ------------------------------------------------------------------
    # 4. Final metrics
    # ------------------------------------------------------------------
    print("\n[4/8] Calculating final test metrics...")

    dbh_metrics = calculate_metrics(
        dbh_df["actual"],
        dbh_df["predicted"],
    )

    height_metrics = calculate_metrics(
        height_df["actual"],
        height_df["predicted"],
    )

    volume_metrics = calculate_metrics(
        volume_df["actual"],
        volume_df["predicted"],
    )

    metrics_rows = [
        {
            "model": "PINNs V2 Constrained",
            "target": "dbh_cm",
            "test_rows": len(dbh_df),
            **dbh_metrics,
        },
        {
            "model": "PINNs V2 Constrained",
            "target": "height_m",
            "test_rows": len(height_df),
            **height_metrics,
        },
        {
            "model": "PINNs V2 Constrained",
            "target": "volume_m3_tree",
            "test_rows": len(volume_df),
            **volume_metrics,
        },
    ]

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    print(
        metrics_df.to_string(index=False)
    )

    # ------------------------------------------------------------------
    # 5. Negative prediction validation
    # ------------------------------------------------------------------
    print("\n[5/8] Validating physical output constraints...")

    negative_dbh = int(
        (dbh_df["predicted"] < 0).sum()
    )

    negative_height = int(
        (height_df["predicted"] < 0).sum()
    )

    negative_volume = int(
        (volume_df["predicted"] < 0).sum()
    )

    zero_volume = int(
        (volume_df["predicted"] == 0).sum()
    )

    print(
        f"Negative DBH    : {negative_dbh}"
    )

    print(
        f"Negative Height : {negative_height}"
    )

    print(
        f"Negative Volume : {negative_volume}"
    )

    print(
        f"Zero Volume     : {zero_volume}"
    )

    # ------------------------------------------------------------------
    # 6. Age trajectory
    # ------------------------------------------------------------------
    print("\n[6/8] Building actual-vs-predicted trajectory...")

    trajectory_rows = []

    target_tables = {
        "dbh_cm": dbh_df,
        "height_m": height_df,
        "volume_m3_tree": volume_df,
    }

    for target, data in target_tables.items():

        for age, group in data.groupby(
            "age_months"
        ):

            trajectory_rows.append(
                {
                    "target": target,
                    "age_months": int(age),
                    "n": int(len(group)),
                    "actual_mean": float(
                        group["actual"].mean()
                    ),
                    "predicted_mean": float(
                        group["predicted"].mean()
                    ),
                    "MAE": float(
                        np.mean(
                            np.abs(
                                group["actual"]
                                - group["predicted"]
                            )
                        )
                    ),
                    "actual_min": float(
                        group["actual"].min()
                    ),
                    "actual_max": float(
                        group["actual"].max()
                    ),
                    "predicted_min": float(
                        group["predicted"].min()
                    ),
                    "predicted_max": float(
                        group["predicted"].max()
                    ),
                    "negative_predictions": int(
                        (
                            group["predicted"]
                            < 0
                        ).sum()
                    ),
                }
            )

    trajectory_df = pd.DataFrame(
        trajectory_rows
    ).sort_values(
        ["target", "age_months"]
    )

    print(
        trajectory_df.to_string(index=False)
    )

    # ------------------------------------------------------------------
    # 7. PHI P66
    # ------------------------------------------------------------------
    print("\n[7/8] Calculating PHI / P66 metrics...")

    phi_rows = []

    for target, data in target_tables.items():

        phi = data[
            data["age_months"] == 66
        ].copy()

        phi_metrics = calculate_metrics(
            phi["actual"],
            phi["predicted"],
        )

        phi_rows.append(
            {
                "target": target,
                "age_months": 66,
                "n": len(phi),
                **phi_metrics,
                "actual_mean": float(
                    phi["actual"].mean()
                ),
                "predicted_mean": float(
                    phi["predicted"].mean()
                ),
            }
        )

    phi_df = pd.DataFrame(
        phi_rows
    )

    print(
        phi_df.to_string(index=False)
    )

    # ------------------------------------------------------------------
    # 8. Save final outputs
    # ------------------------------------------------------------------
    print("\n[8/8] Saving final evaluation outputs...")

    metrics_df.to_csv(
        OUTPUT_METRICS,
        index=False,
    )

    trajectory_df.to_csv(
        OUTPUT_AGE,
        index=False,
    )

    phi_df.to_csv(
        OUTPUT_PHI,
        index=False,
    )

    report = {
        "step": 59,

        "model": "PINNs V2",

        "evaluation_type": (
            "Final held-out test evaluation "
            "using constrained volume output"
        ),

        "test_rows_dbh": int(len(dbh_df)),
        "test_rows_height": int(len(height_df)),
        "test_rows_volume": int(len(volume_df)),

        "volume_constraint": (
            "volume_prediction = max(raw_prediction, 0)"
        ),

        "negative_dbh_predictions": negative_dbh,
        "negative_height_predictions": negative_height,
        "negative_volume_predictions": negative_volume,
        "zero_volume_predictions": zero_volume,

        "dataset_modified": False,
        "model_checkpoint_modified": False,
        "retraining_performed": False,

        "metrics": {
            "dbh_cm": dbh_metrics,
            "height_m": height_metrics,
            "volume_m3_tree": volume_metrics,
        },

        "phi_p66": phi_df.to_dict(
            orient="records"
        ),
    }

    OUTPUT_REPORT.write_text(
        json.dumps(
            report,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print("STEP 59 COMPLETE")
    print("=" * 80)

    print("\nFINAL PINNs V2 CONSTRAINED METRICS:")
    print(
        metrics_df.to_string(index=False)
    )

    print("\nPhysical constraint:")
    print(
        f"Negative volume = {negative_volume}"
    )

    print("\nOutputs:")
    print(OUTPUT_METRICS)
    print(OUTPUT_AGE)
    print(OUTPUT_PHI)
    print(OUTPUT_REPORT)


if __name__ == "__main__":
    main()