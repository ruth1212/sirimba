from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STEP 61 — RF vs PINNs V2 TRAJECTORY VISUALIZATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "model_comparison"
    / "rf_vs_pinns_trajectory.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "model_comparison"
    / "plots"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("STEP 61 — RF vs PINNs V2 TRAJECTORY VISUALIZATION")
print("=" * 80)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/4] Loading trajectory data...")

if not INPUT.exists():
    raise FileNotFoundError(
        f"Trajectory file not found:\n{INPUT}"
    )

df = pd.read_csv(INPUT)

required_columns = {
    "model",
    "target",
    "age_months",
    "actual_mean",
    "predicted_mean",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

print(f"Rows loaded : {len(df):,}")
print(f"Models      : {sorted(df['model'].unique())}")
print(f"Targets     : {sorted(df['target'].unique())}")
print(
    f"Ages        : "
    f"{sorted(df['age_months'].unique())}"
)


# ============================================================
# 2. VALIDATION
# ============================================================

print("\n[2/4] Validating trajectory structure...")

expected_targets = {
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
}

actual_targets = set(df["target"].unique())

if actual_targets != expected_targets:
    raise ValueError(
        f"Unexpected targets.\n"
        f"Expected: {expected_targets}\n"
        f"Found: {actual_targets}"
    )


expected_ages_dbh_height = {
    6, 12, 18, 24, 30, 36,
    42, 48, 54, 60, 66
}

expected_ages_volume = {
    18, 24, 30, 36,
    42, 48, 54, 60, 66
}


# Check NaN
if df[
    ["age_months", "actual_mean", "predicted_mean"]
].isna().any().any():
    raise ValueError(
        "NaN detected in trajectory data."
    )


# Check RF and PINNs
expected_models = {
    "Random Forest",
    "PINNs V2 Constrained",
}

actual_models = set(df["model"].unique())

if actual_models != expected_models:
    raise ValueError(
        f"Unexpected models.\n"
        f"Expected: {expected_models}\n"
        f"Found: {actual_models}"
    )


# Check target-specific ages
for target in ["dbh_cm", "height_m"]:
    ages = set(
        df.loc[
            df["target"] == target,
            "age_months"
        ].unique()
    )

    if ages != expected_ages_dbh_height:
        raise ValueError(
            f"Unexpected ages for {target}: {sorted(ages)}"
        )


ages_volume = set(
    df.loc[
        df["target"] == "volume_m3_tree",
        "age_months"
    ].unique()
)

if ages_volume != expected_ages_volume:
    raise ValueError(
        f"Unexpected volume ages: {sorted(ages_volume)}"
    )


print("Targets structure : PASS")
print("Age structure     : PASS")
print("Model structure   : PASS")
print("NaN check         : PASS")


# ============================================================
# 3. CREATE PLOTS
# ============================================================

print("\n[3/4] Creating trajectory plots...")


plot_config = {
    "dbh_cm": {
        "title": "DBH Growth Trajectory — Actual vs Predicted",
        "ylabel": "DBH (cm)",
        "filename": "trajectory_dbh.png",
        "unit": "cm",
    },
    "height_m": {
        "title": "Height Growth Trajectory — Actual vs Predicted",
        "ylabel": "Height (m)",
        "filename": "trajectory_height.png",
        "unit": "m",
    },
    "volume_m3_tree": {
        "title": "Volume Growth Trajectory — Actual vs Predicted",
        "ylabel": "Volume (m³/tree)",
        "filename": "trajectory_volume.png",
        "unit": "m³/tree",
    },
}


for target, config in plot_config.items():

    print(f"  Creating {target} plot...")

    target_df = df[
        df["target"] == target
    ].copy()

    target_df = target_df.sort_values(
        ["model", "age_months"]
    )

    actual = (
        target_df[
            ["age_months", "actual_mean"]
        ]
        .drop_duplicates()
        .sort_values("age_months")
    )

    rf = (
        target_df[
            target_df["model"] == "Random Forest"
        ]
        .sort_values("age_months")
    )

    pinns = (
        target_df[
            target_df["model"] == "PINNs V2 Constrained"
        ]
        .sort_values("age_months")
    )

    plt.figure(figsize=(11, 6))

    # Actual
    plt.plot(
        actual["age_months"],
        actual["actual_mean"],
        marker="o",
        linewidth=2.5,
        label="Actual",
    )

    # Random Forest
    plt.plot(
        rf["age_months"],
        rf["predicted_mean"],
        marker="s",
        linewidth=2,
        label="Random Forest",
    )

    # PINNs
    plt.plot(
        pinns["age_months"],
        pinns["predicted_mean"],
        marker="^",
        linewidth=2,
        label="PINNs V2",
    )

    # PHI marker
    plt.axvline(
        x=66,
        linestyle="--",
        linewidth=1.5,
        label="PHI = P66",
    )

    plt.title(
        config["title"],
        fontsize=14,
        fontweight="bold",
    )

    plt.xlabel(
        "Tree Age (months)"
    )

    plt.ylabel(
        config["ylabel"]
    )

    plt.xticks(
        sorted(
            target_df["age_months"].unique()
        )
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / config["filename"]
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"    Saved: {output_path}"
    )


# ============================================================
# 4. CREATE COMBINED CSV FOR DOCUMENTATION
# ============================================================

print("\n[4/4] Saving plotting summary...")

summary_rows = []

for target, config in plot_config.items():

    target_df = df[
        df["target"] == target
    ].copy()

    for age in sorted(
        target_df["age_months"].unique()
    ):

        age_df = target_df[
            target_df["age_months"] == age
        ]

        actual_value = (
            age_df["actual_mean"]
            .iloc[0]
        )

        rf_rows = age_df[
            age_df["model"] == "Random Forest"
        ]

        pinns_rows = age_df[
            age_df["model"] == "PINNs V2 Constrained"
        ]

        rf_value = (
            rf_rows["predicted_mean"].iloc[0]
            if len(rf_rows) > 0
            else None
        )

        pinns_value = (
            pinns_rows["predicted_mean"].iloc[0]
            if len(pinns_rows) > 0
            else None
        )

        summary_rows.append(
            {
                "target": target,
                "age_months": age,
                "actual_mean": actual_value,
                "random_forest_predicted": rf_value,
                "pinns_v2_predicted": pinns_value,
            }
        )


summary = pd.DataFrame(summary_rows)

summary_path = (
    OUTPUT_DIR
    / "trajectory_plot_data.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)


print(
    f"Saved: {summary_path}"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("STEP 61 COMPLETE")
print("=" * 80)

print("\nGenerated files:")

for path in sorted(OUTPUT_DIR.glob("*")):
    print(f"  - {path.name}")

print("\nTrajectory visualization completed successfully.")