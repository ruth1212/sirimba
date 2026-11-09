from pathlib import Path
import json
import shutil

import pandas as pd


# ============================================================
# STEP 62 — FINALIZE MODEL ARTIFACTS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results"
MODEL_COMPARISON = RESULTS / "model_comparison"

FINAL_DIR = RESULTS / "dashboard_ready"

FINAL_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("STEP 62 — FINALIZE MODEL ARTIFACTS FOR DASHBOARD")
print("=" * 80)


# ============================================================
# SOURCE FILES
# ============================================================

sources = {
    "rf_metrics.csv":
        MODEL_COMPARISON / "rf_vs_pinns_final_metrics.csv",

    "rf_vs_pinns_trajectory.csv":
        MODEL_COMPARISON / "rf_vs_pinns_trajectory.csv",

    "rf_vs_pinns_phi_p66.csv":
        MODEL_COMPARISON / "rf_vs_pinns_phi_p66.csv",

    "comparison_report.json":
        MODEL_COMPARISON / "step60_comparison_report.json",

    "trajectory_plot_data.csv":
        MODEL_COMPARISON / "plots" / "trajectory_plot_data.csv",
}


# ============================================================
# 1. CHECK SOURCE FILES
# ============================================================

print("\n[1/5] Checking source artifacts...")

missing = []

for name, path in sources.items():

    if not path.exists():
        missing.append(str(path))
        print(f"  MISSING: {path}")
    else:
        print(f"  FOUND  : {path.name}")


if missing:
    raise FileNotFoundError(
        "Required artifact(s) missing:\n"
        + "\n".join(missing)
    )

print("Source artifact check: PASS")


# ============================================================
# 2. COPY FINAL CSV / JSON FILES
# ============================================================

print("\n[2/5] Copying final artifacts...")

for output_name, source_path in sources.items():

    destination = FINAL_DIR / output_name

    shutil.copy2(
        source_path,
        destination
    )

    print(
        f"  Saved: {destination.name}"
    )


# ============================================================
# 3. VALIDATE METRICS
# ============================================================

print("\n[3/5] Validating metrics artifact...")

metrics_path = FINAL_DIR / "rf_metrics.csv"

metrics = pd.read_csv(metrics_path)

required_metric_columns = {
    "model",
    "target",
    "test_rows",
    "MAE",
    "RMSE",
    "R2",
    "MAPE_percent",
}

missing_metric_columns = (
    required_metric_columns
    - set(metrics.columns)
)

if missing_metric_columns:
    raise ValueError(
        "Missing metric columns: "
        + str(sorted(missing_metric_columns))
    )


expected_models = {
    "Random Forest",
    "PINNs V2 Constrained",
}

expected_targets = {
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
}


if set(metrics["model"].unique()) != expected_models:
    raise ValueError(
        "Unexpected model names."
    )


if set(metrics["target"].unique()) != expected_targets:
    raise ValueError(
        "Unexpected target names."
    )


if metrics[
    ["MAE", "RMSE", "R2", "MAPE_percent"]
].isna().any().any():
    raise ValueError(
        "NaN detected in metrics."
    )


print(
    f"Metric rows: {len(metrics)}"
)

print("Metrics validation: PASS")


# ============================================================
# 4. VALIDATE TRAJECTORY
# ============================================================

print("\n[4/5] Validating trajectory artifact...")

trajectory_path = (
    FINAL_DIR
    / "rf_vs_pinns_trajectory.csv"
)

trajectory = pd.read_csv(
    trajectory_path
)

required_trajectory_columns = {
    "model",
    "target",
    "age_months",
    "actual_mean",
    "predicted_mean",
}

missing_trajectory_columns = (
    required_trajectory_columns
    - set(trajectory.columns)
)

if missing_trajectory_columns:
    raise ValueError(
        "Missing trajectory columns: "
        + str(sorted(missing_trajectory_columns))
    )


if trajectory[
    [
        "age_months",
        "actual_mean",
        "predicted_mean",
    ]
].isna().any().any():
    raise ValueError(
        "NaN detected in trajectory."
    )


dbh_height_ages = {
    6, 12, 18, 24, 30,
    36, 42, 48, 54,
    60, 66
}

volume_ages = {
    18, 24, 30, 36,
    42, 48, 54, 60,
    66
}


for target in [
    "dbh_cm",
    "height_m",
]:

    ages = set(
        trajectory.loc[
            trajectory["target"] == target,
            "age_months"
        ].unique()
    )

    if ages != dbh_height_ages:
        raise ValueError(
            f"Unexpected ages for {target}: "
            f"{sorted(ages)}"
        )


volume_actual_ages = set(
    trajectory.loc[
        trajectory["target"] == "volume_m3_tree",
        "age_months"
    ].unique()
)

if volume_actual_ages != volume_ages:
    raise ValueError(
        "Unexpected volume age structure."
    )


print(
    f"Trajectory rows: {len(trajectory)}"
)

print("Trajectory validation: PASS")


# ============================================================
# 5. CREATE MANIFEST
# ============================================================

print("\n[5/5] Creating dashboard manifest...")

manifest = {
    "project": "TREE GROWTH ANALYTICS",
    "status": "FINAL_MODEL_EVALUATION",
    "dataset_scope": {
        "age_min_months": 6,
        "age_max_months": 66,
        "phi_months": 66,
        "phi_label": "P66",
    },
    "models": [
        "Random Forest",
        "PINNs V2 Constrained",
    ],
    "targets": [
        "dbh_cm",
        "height_m",
        "volume_m3_tree",
    ],
    "evaluation": {
        "same_test_split": True,
        "group_aware_split": True,
        "test_compartments": 3,
        "dbh_height_age_range": "P6-P66",
        "volume_age_range": "P18-P66",
    },
    "artifacts": [
        "rf_metrics.csv",
        "rf_vs_pinns_trajectory.csv",
        "rf_vs_pinns_phi_p66.csv",
        "comparison_report.json",
        "trajectory_plot_data.csv",
    ],
}


manifest_path = (
    FINAL_DIR
    / "dashboard_manifest.json"
)

manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    f"Saved: {manifest_path}"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("STEP 62 COMPLETE")
print("=" * 80)

print("\nDashboard-ready directory:")

print(FINAL_DIR)

print("\nFiles:")

for path in sorted(FINAL_DIR.iterdir()):
    print(
        f"  - {path.name}"
    )

print(
    "\nAll final model artifacts are ready for dashboard integration."
)