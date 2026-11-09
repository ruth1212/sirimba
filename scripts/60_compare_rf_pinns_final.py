from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RF_METRICS = (
    PROJECT_ROOT
    / "results"
    / "random_forest"
    / "test_metrics.csv"
)

PINNS_METRICS = (
    PROJECT_ROOT
    / "results"
    / "pinns"
    / "pinns_v2_final_test_metrics.csv"
)

RF_TRAJECTORY = (
    PROJECT_ROOT
    / "results"
    / "random_forest"
    / "actual_vs_predicted_by_age.csv"
)

PINNS_TRAJECTORY = (
    PROJECT_ROOT
    / "results"
    / "pinns"
    / "pinns_v2_final_actual_vs_predicted_by_age.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "model_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_METRICS = (
    OUTPUT_DIR
    / "rf_vs_pinns_final_metrics.csv"
)

OUTPUT_TRAJECTORY = (
    OUTPUT_DIR
    / "rf_vs_pinns_trajectory.csv"
)

OUTPUT_PHI = (
    OUTPUT_DIR
    / "rf_vs_pinns_phi_p66.csv"
)

OUTPUT_REPORT = (
    OUTPUT_DIR
    / "step60_comparison_report.json"
)


TARGETS = [
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]


def normalize_target(value):

    value = str(value).strip().lower()

    mapping = {
        "dbh": "dbh_cm",
        "dbh_cm": "dbh_cm",

        "height": "height_m",
        "height_m": "height_m",

        "volume": "volume_m3_tree",
        "volume_m3_tree": "volume_m3_tree",
    }

    return mapping.get(value, value)


def normalize_trajectory(df, model_name):

    df = df.copy()

    if "target" not in df.columns:
        raise ValueError(
            "Trajectory file tidak memiliki kolom 'target'."
        )

    if "age_months" not in df.columns:
        raise ValueError(
            "Trajectory file tidak memiliki kolom 'age_months'."
        )

    # ---------------------------------------------------------------
    # Detect actual mean
    # ---------------------------------------------------------------
    actual_col = None

    for col in [
        "actual_mean",
        "actual",
        "mean_actual",
    ]:
        if col in df.columns:
            actual_col = col
            break

    if actual_col is None:
        raise ValueError(
            "Kolom actual tidak ditemukan.\n"
            f"Columns: {list(df.columns)}"
        )

    # ---------------------------------------------------------------
    # Detect predicted mean
    # ---------------------------------------------------------------
    predicted_col = None

    for col in [
        "predicted_mean",
        "predicted",
        "prediction_mean",
        "mean_predicted",
    ]:
        if col in df.columns:
            predicted_col = col
            break

    if predicted_col is None:
        raise ValueError(
            "Kolom prediction tidak ditemukan.\n"
            f"Columns: {list(df.columns)}"
        )

    result = pd.DataFrame(
        {
            "model": model_name,

            "target": df["target"].map(
                normalize_target
            ),

            "age_months": pd.to_numeric(
                df["age_months"],
                errors="coerce"
            ),

            "actual_mean": pd.to_numeric(
                df[actual_col],
                errors="coerce"
            ),

            "predicted_mean": pd.to_numeric(
                df[predicted_col],
                errors="coerce"
            ),
        }
    )

    # ---------------------------------------------------------------
    # Recalculate MAE from actual and predicted means
    #
    # IMPORTANT:
    # This is trajectory-level MAE between age-level means.
    # It is NOT the overall row-level test MAE.
    # ---------------------------------------------------------------
    result["trajectory_abs_error"] = (
        result["actual_mean"]
        - result["predicted_mean"]
    ).abs()

    result["MAE"] = result[
        "trajectory_abs_error"
    ]

    return result


def load_overall_metrics(path, model_name):

    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file not found:\n{path}"
        )

    df = pd.read_csv(path)

    if "target" not in df.columns:
        raise ValueError(
            f"'target' column missing in:\n{path}"
        )

    df = df.copy()

    df["target"] = df[
        "target"
    ].map(normalize_target)

    df["model"] = model_name

    return df


def main():

    print("=" * 80)
    print("STEP 60 — FINAL RF vs PINNs V2 COMPARISON")
    print("=" * 80)

    # ===============================================================
    # 1. OVERALL TEST METRICS
    # ===============================================================
    print("\n[1/5] Loading final test metrics...")

    rf_metrics = load_overall_metrics(
        RF_METRICS,
        "Random Forest",
    )

    pinns_metrics = load_overall_metrics(
        PINNS_METRICS,
        "PINNs V2 Constrained",
    )

    rf_metrics = rf_metrics[
        rf_metrics["target"].isin(TARGETS)
    ].copy()

    pinns_metrics = pinns_metrics[
        pinns_metrics["target"].isin(TARGETS)
    ].copy()

    metric_columns = [
        "model",
        "target",
        "test_rows",
        "MAE",
        "RMSE",
        "R2",
        "MAPE_percent",
    ]

    for col in metric_columns:
        if col not in rf_metrics.columns:
            raise ValueError(
                f"RF metrics missing column: {col}"
            )

        if col not in pinns_metrics.columns:
            raise ValueError(
                f"PINNs metrics missing column: {col}"
            )

    metrics = pd.concat(
        [
            rf_metrics[metric_columns],
            pinns_metrics[metric_columns],
        ],
        ignore_index=True,
    )

    print(
        metrics.to_string(index=False)
    )

    # ===============================================================
    # 2. TRAJECTORIES
    # ===============================================================
    print("\n[2/5] Loading age trajectories...")

    if not RF_TRAJECTORY.exists():
        raise FileNotFoundError(
            f"RF trajectory not found:\n{RF_TRAJECTORY}"
        )

    if not PINNS_TRAJECTORY.exists():
        raise FileNotFoundError(
            f"PINNs trajectory not found:\n{PINNS_TRAJECTORY}"
        )

    rf_raw = pd.read_csv(
        RF_TRAJECTORY
    )

    pinns_raw = pd.read_csv(
        PINNS_TRAJECTORY
    )

    print(
        "RF trajectory columns:",
        list(rf_raw.columns)
    )

    print(
        "PINNs trajectory columns:",
        list(pinns_raw.columns)
    )

    rf_traj = normalize_trajectory(
        rf_raw,
        "Random Forest",
    )

    pinns_traj = normalize_trajectory(
        pinns_raw,
        "PINNs V2 Constrained",
    )

    trajectory = pd.concat(
        [
            rf_traj,
            pinns_traj,
        ],
        ignore_index=True,
    )

    trajectory = trajectory[
        trajectory["target"].isin(TARGETS)
    ].copy()

    trajectory = trajectory.sort_values(
        [
            "target",
            "age_months",
            "model",
        ]
    ).reset_index(drop=True)

    print("\nRecalculated trajectory errors:")

    print(
        trajectory[
            [
                "model",
                "target",
                "age_months",
                "actual_mean",
                "predicted_mean",
                "MAE",
            ]
        ].to_string(index=False)
    )

    # ===============================================================
    # 3. P66 / PHI
    # ===============================================================
    print("\n[3/5] Calculating P66 / PHI directly from trajectory...")

    phi = trajectory[
        trajectory["age_months"] == 66
    ].copy()

    if phi.empty:
        raise ValueError(
            "P66 trajectory tidak ditemukan."
        )

    phi["age_months"] = phi[
        "age_months"
    ].astype(int)

    phi = phi[
        [
            "model",
            "target",
            "age_months",
            "actual_mean",
            "predicted_mean",
            "MAE",
        ]
    ].copy()

    phi = phi.rename(
        columns={
            "MAE": "trajectory_MAE"
        }
    )

    print(
        phi.to_string(index=False)
    )

    # ===============================================================
    # 4. INTEGRITY CHECKS
    # ===============================================================
    print("\n[4/5] Running integrity checks...")

    checks = {}

    checks["models"] = (
        set(metrics["model"].unique())
        == {
            "Random Forest",
            "PINNs V2 Constrained",
        }
    )

    checks["targets"] = (
        set(metrics["target"].unique())
        == set(TARGETS)
    )

    checks["trajectory_models"] = (
        set(trajectory["model"].unique())
        == {
            "Random Forest",
            "PINNs V2 Constrained",
        }
    )

    checks["trajectory_targets"] = (
        set(trajectory["target"].unique())
        == set(TARGETS)
    )

    checks["p66_present"] = (
        66 in trajectory["age_months"].unique()
    )

    checks["no_nan_trajectory"] = (
        not trajectory[
            [
                "actual_mean",
                "predicted_mean",
                "MAE",
            ]
        ].isna().any().any()
    )

    # Check same number of trajectory ages
    trajectory_counts = (
        trajectory
        .groupby(
            ["model", "target"]
        )["age_months"]
        .nunique()
    )

    checks["trajectory_age_counts"] = (
        len(trajectory_counts) == 6
        and trajectory_counts.min() > 0
    )

    # Same test rows per target
    checks["same_test_rows"] = True

    for target in TARGETS:

        rows = metrics[
            metrics["target"] == target
        ]

        if len(rows) != 2:
            checks["same_test_rows"] = False
            continue

        unique_counts = (
            rows["test_rows"]
            .dropna()
            .unique()
        )

        if len(unique_counts) != 1:
            checks["same_test_rows"] = False

    for name, value in checks.items():

        print(
            f"{name}: "
            f"{'PASS' if value else 'FAIL'}"
        )

    if not all(checks.values()):
        raise RuntimeError(
            "Comparison integrity check FAILED."
        )

    # ===============================================================
    # 5. SAVE
    # ===============================================================
    print("\n[5/5] Saving final comparison outputs...")

    metrics.to_csv(
        OUTPUT_METRICS,
        index=False,
    )

    trajectory.to_csv(
        OUTPUT_TRAJECTORY,
        index=False,
    )

    phi.to_csv(
        OUTPUT_PHI,
        index=False,
    )

    report = {
        "step": 60,

        "purpose": (
            "Final factual comparison of "
            "Random Forest and PINNs V2 "
            "using the same official held-out test split."
        ),

        "models": [
            "Random Forest",
            "PINNs V2 Constrained",
        ],

        "targets": TARGETS,

        "phi": {
            "age_months": 66,
            "label": "PHI",
        },

        "volume_prediction_age": {
            "min": 18,
            "max": 66,
        },

        "dataset_modified": False,

        "retraining_performed": False,

        "integrity_checks": {
            str(k): bool(v)
            for k, v in checks.items()
        },

        "outputs": {
            "metrics": str(
                OUTPUT_METRICS
            ),
            "trajectory": str(
                OUTPUT_TRAJECTORY
            ),
            "phi": str(
                OUTPUT_PHI
            ),
        },
    }

    OUTPUT_REPORT.write_text(
        json.dumps(
            report,
            indent=2
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("STEP 60 COMPLETE")
    print("=" * 80)

    print("\nFINAL TEST METRICS:")
    print(
        metrics.to_string(index=False)
    )

    print("\nP66 / PHI:")
    print(
        phi.to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT_METRICS)
    print(OUTPUT_TRAJECTORY)
    print(OUTPUT_PHI)
    print(OUTPUT_REPORT)


if __name__ == "__main__":
    main()