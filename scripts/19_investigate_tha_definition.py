"""
TREE GROWTH ANALYTICS
STEP 12B - INVESTIGATE THA DEFINITION

Purpose:
- Investigate how dataset field tha_m relates to
  common Top Height / Dominant Height definitions.
- Do NOT modify the dataset.
- Do NOT create a production THA formula yet.

Candidate methods:
1. Mean height of top 10% tallest trees
2. Mean height of top 20% tallest trees
3. Mean height of top 50% tallest trees
4. Mean height of top 100 tallest trees
5. Mean height of trees with the largest DBH
   (top 100 by DBH)

The goal is to determine whether one candidate
reproduces the existing dataset tha_m.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "dataset.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

REQUIRED_COLUMNS = [
    "compartment_id",
    "age_months",
    "height_m",
    "dbh_cm",
    "tha_m",
]


# ============================================================
# TOP HEIGHT FUNCTIONS
# ============================================================

def mean_top_percentage(
    group,
    percentage,
):
    """
    Mean height of the tallest percentage of trees.
    """

    n = len(group)

    k = max(
        1,
        int(np.ceil(
            n * percentage
        ))
    )

    selected = (
        group
        .nlargest(
            k,
            "height_m"
        )
    )

    return float(
        selected["height_m"].mean()
    )


def mean_top_n_by_height(
    group,
    n_trees,
):
    """
    Mean height of the n tallest trees.
    """

    k = min(
        n_trees,
        len(group)
    )

    selected = (
        group
        .nlargest(
            k,
            "height_m"
        )
    )

    return float(
        selected["height_m"].mean()
    )


def mean_top_n_by_dbh(
    group,
    n_trees,
):
    """
    Mean height of trees with the largest DBH.
    """

    k = min(
        n_trees,
        len(group)
    )

    selected = (
        group
        .nlargest(
            k,
            "dbh_cm"
        )
    )

    return float(
        selected["height_m"].mean()
    )


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    actual,
    predicted,
):
    """
    Compare candidate THA against dataset tha_m.
    """

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    error = predicted - actual

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    bias = np.mean(
        error
    )

    ss_res = np.sum(
        error ** 2
    )

    ss_tot = np.sum(
        (actual - actual.mean()) ** 2
    )

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1 - (
            ss_res / ss_tot
        )

    correlation = np.corrcoef(
        actual,
        predicted,
    )[0, 1]

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias,
        "R2": r2,
        "Correlation": correlation,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("TREE GROWTH ANALYTICS")
    print("STEP 12B - INVESTIGATE THA DEFINITION")
    print("=" * 75)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    print("\n[1/5] Loading dataset...")

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH,
        usecols=REQUIRED_COLUMNS,
    )

    print(
        f"  Rows    : {len(df):,}"
    )

    print(
        f"  Columns : {len(df.columns)}"
    )

    # --------------------------------------------------------
    # 2. Validate data
    # --------------------------------------------------------

    print("\n[2/5] Validating data...")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns: "
            f"{missing_columns}"
        )

    missing_values = (
        df[REQUIRED_COLUMNS]
        .isna()
        .sum()
    )

    if missing_values.sum() > 0:

        print(
            "\nMissing values:"
        )

        print(
            missing_values
        )

        raise ValueError(
            "Required THA data contains "
            "missing values."
        )

    print(
        "  Required columns : OK"
    )

    print(
        "  Missing values   : NONE"
    )

    # --------------------------------------------------------
    # 3. Calculate candidate THA
    # --------------------------------------------------------

    print(
        "\n[3/5] Calculating candidate "
        "Top Height definitions..."
    )

    group_columns = [
        "compartment_id",
        "age_months",
    ]

    records = []

    grouped = df.groupby(
        group_columns,
        sort=False,
    )

    total_groups = len(grouped)

    print(
        f"  Groups: {total_groups:,}"
    )

    for (
        compartment_id,
        age_months,
    ), group in grouped:

        actual_tha = float(
            group["tha_m"].iloc[0]
        )

        top_10 = (
            mean_top_percentage(
                group,
                0.10,
            )
        )

        top_20 = (
            mean_top_percentage(
                group,
                0.20,
            )
        )

        top_50 = (
            mean_top_percentage(
                group,
                0.50,
            )
        )

        top_100_height = (
            mean_top_n_by_height(
                group,
                100,
            )
        )

        top_100_dbh = (
            mean_top_n_by_dbh(
                group,
                100,
            )
        )

        records.append({

            "compartment_id":
                compartment_id,

            "age_months":
                age_months,

            "tha_m":
                actual_tha,

            "top_10_height":
                top_10,

            "top_20_height":
                top_20,

            "top_50_height":
                top_50,

            "top_100_height":
                top_100_height,

            "top_100_dbh_height":
                top_100_dbh,
        })

    comparison = pd.DataFrame(
        records
    )

    print(
        "  Candidate calculation : OK"
    )

    # --------------------------------------------------------
    # 4. Evaluate candidates
    # --------------------------------------------------------

    print(
        "\n[4/5] Comparing candidates "
        "against dataset tha_m..."
    )

    candidates = {

        "top_10_height":
            "Mean height of top 10% tallest trees",

        "top_20_height":
            "Mean height of top 20% tallest trees",

        "top_50_height":
            "Mean height of top 50% tallest trees",

        "top_100_height":
            "Mean height of 100 tallest trees",

        "top_100_dbh_height":
            "Mean height of 100 largest-DBH trees",
    }

    results = []

    for (
        column,
        description,
    ) in candidates.items():

        metrics = calculate_metrics(
            comparison["tha_m"],
            comparison[column],
        )

        results.append({

            "method":
                column,

            "description":
                description,

            "MAE":
                metrics["MAE"],

            "RMSE":
                metrics["RMSE"],

            "Bias":
                metrics["Bias"],

            "R2":
                metrics["R2"],

            "Correlation":
                metrics["Correlation"],
        })

    results_df = pd.DataFrame(
        results
    )

    # Sort only for inspection.
    # This is NOT declaring a production definition.
    results_df = results_df.sort_values(
        "MAE"
    )

    # --------------------------------------------------------
    # 5. Save results
    # --------------------------------------------------------

    print(
        "\n[5/5] Saving investigation results..."
    )

    results_dir = (
        PROJECT_ROOT
        / "results"
    )

    results_dir.mkdir(
        exist_ok=True
    )

    comparison_path = (
        results_dir
        / "tha_definition_comparison.csv"
    )

    metrics_path = (
        results_dir
        / "tha_definition_metrics.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    results_df.to_csv(
        metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 75
    )

    print(
        "THA DEFINITION INVESTIGATION"
    )

    print(
        "=" * 75
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nSample comparison:"
    )

    print(
        comparison.head(15).to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {comparison_path}"
    )

    print(
        f"  {metrics_path}"
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "STEP 12B INVESTIGATION COMPLETED"
    )

    print(
        "=" * 75
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()