from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

BASELINE_PATH = RESULTS / "model_comparison_predictions.csv"
POSITIVE_PATH = RESULTS / "pinns_positive_test_predictions.csv"

OUTPUT_PATH = RESULTS / "pinns_positive_diagnostic_comparison.csv"

TARGETS = [
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]


def metric_block(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    error = y_pred - y_true

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    ss_res = np.sum(error ** 2)
    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot != 0
        else np.nan
    )

    mask = np.abs(y_true) > 1e-12

    mape = (
        np.mean(
            np.abs(
                error[mask] / y_true[mask]
            )
        ) * 100.0
        if mask.any()
        else np.nan
    )

    return mae, rmse, r2, mape


def main():

    print("=" * 75)
    print("TREE GROWTH ANALYTICS")
    print("STEP 21K - PINNs POSITIVE DIAGNOSTIC COMPARISON")
    print("=" * 75)

    # ========================================================
    # 1. LOAD FILES
    # ========================================================

    print("\n[1/7] Loading prediction files...")

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Baseline predictions not found:\n"
            f"{BASELINE_PATH}"
        )

    if not POSITIVE_PATH.exists():
        raise FileNotFoundError(
            f"Positive predictions not found:\n"
            f"{POSITIVE_PATH}"
        )

    baseline = pd.read_csv(BASELINE_PATH)
    positive = pd.read_csv(POSITIVE_PATH)

    print(
        f"Baseline rows : {len(baseline):,}"
    )

    print(
        f"Positive rows : {len(positive):,}"
    )

    # ========================================================
    # 2. VERIFY SAME TEST SET
    # ========================================================

    print("\n[2/7] Verifying identical test rows...")

    if len(baseline) != len(positive):
        raise ValueError(
            "Baseline and positive prediction files "
            "have different row counts."
        )

    if len(baseline) != 121_498:
        raise ValueError(
            f"Expected 121,498 rows, got "
            f"{len(baseline):,}"
        )

    comparison_keys = [
        "age_months",
        "inventory_year",
        "clone_factor",
        "planting_year",
        "disease_score",
        "trees_per_ha",
        "compartment_id",
    ]

    common_keys = [
        c
        for c in comparison_keys
        if c in baseline.columns
        and c in positive.columns
    ]

    if not common_keys:
        raise ValueError(
            "No common identifying columns found."
        )

    print(
        "Comparison keys:",
        ", ".join(common_keys)
    )

    for key in common_keys:

        baseline_values = baseline[key].to_numpy()
        positive_values = positive[key].to_numpy()

        if not np.array_equal(
            baseline_values,
            positive_values,
        ):
            raise ValueError(
                f"Test ordering/content mismatch "
                f"in column: {key}"
            )

    print("Same test rows/order : YES")

    # ========================================================
    # 3. FIND PREDICTION COLUMNS
    # ========================================================

    print("\n[3/7] Locating prediction columns...")

    print(
        "Baseline columns:",
        ", ".join(baseline.columns)
    )

    print(
        "Positive columns:",
        ", ".join(positive.columns)
    )

    # IMPORTANT:
    # These names come directly from the actual
    # model_comparison_predictions.csv file.
    baseline_pred_candidates = {
        "dbh_cm": "predicted_pinns_dbh_cm",
        "height_m": "predicted_pinns_height_m",
        "volume_m3_tree": "predicted_pinns_volume_m3_tree",
    }

    positive_pred_candidates = {
        "dbh_cm": "pinns_positive_dbh_cm",
        "height_m": "pinns_positive_height_m",
        "volume_m3_tree": "pinns_positive_volume_m3_tree",
    }

    for target, column in baseline_pred_candidates.items():

        if column not in baseline.columns:
            raise ValueError(
                f"Missing baseline PINNs prediction "
                f"column for {target}: {column}"
            )

    for target, column in positive_pred_candidates.items():

        if column not in positive.columns:
            raise ValueError(
                f"Missing positive PINNs prediction "
                f"column for {target}: {column}"
            )

    print("\nBaseline prediction mapping:")

    for target, column in baseline_pred_candidates.items():

        print(
            f"  {target:<20} <- {column}"
        )

    print("\nPositive prediction mapping:")

    for target, column in positive_pred_candidates.items():

        print(
            f"  {target:<20} <- {column}"
        )

    # ========================================================
    # 4. FIND ACTUAL TARGET COLUMNS
    # ========================================================

    print("\n[4/7] Locating actual target columns...")

    actual_target_columns = {}

    for target in TARGETS:

        if target in positive.columns:

            actual_target_columns[target] = target

        elif f"actual_{target}" in positive.columns:

            actual_target_columns[target] = (
                f"actual_{target}"
            )

        elif f"true_{target}" in positive.columns:

            actual_target_columns[target] = (
                f"true_{target}"
            )

        else:

            raise ValueError(
                f"Actual target not found: {target}"
            )

    for target, column in actual_target_columns.items():

        print(
            f"  {target:<20} <- {column}"
        )

    # ========================================================
    # 5. GLOBAL COMPARISON
    # ========================================================

    print("\n[5/7] Calculating global comparison...")

    comparison_rows = []

    for target in TARGETS:

        y_true = positive[
            actual_target_columns[target]
        ].to_numpy(dtype=np.float64)

        baseline_pred = baseline[
            baseline_pred_candidates[target]
        ].to_numpy(dtype=np.float64)

        positive_pred = positive[
            positive_pred_candidates[target]
        ].to_numpy(dtype=np.float64)

        baseline_metrics = metric_block(
            y_true,
            baseline_pred,
        )

        positive_metrics = metric_block(
            y_true,
            positive_pred,
        )

        comparison_rows.append(
            {
                "scope": "ALL_TEST",
                "target": target,
                "model": "PINNs Baseline",
                "MAE": baseline_metrics[0],
                "RMSE": baseline_metrics[1],
                "R2": baseline_metrics[2],
                "MAPE_percent": baseline_metrics[3],
            }
        )

        comparison_rows.append(
            {
                "scope": "ALL_TEST",
                "target": target,
                "model": "PINNs Positive",
                "MAE": positive_metrics[0],
                "RMSE": positive_metrics[1],
                "R2": positive_metrics[2],
                "MAPE_percent": positive_metrics[3],
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    print("\nGlobal comparison:")

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ========================================================
    # 6. P6 + VOLUME DIAGNOSTIC
    # ========================================================

    print(
        "\n[6/7] Performing P6 and volume diagnostics..."
    )

    if "age_months" not in positive.columns:
        raise ValueError(
            "age_months is required for P6 analysis."
        )

    p6_mask = (
        positive["age_months"].to_numpy()
        == 6
    )

    print(
        f"P6 rows : {int(p6_mask.sum()):,}"
    )

    baseline_volume = baseline[
        baseline_pred_candidates[
            "volume_m3_tree"
        ]
    ].to_numpy(dtype=np.float64)

    positive_volume = positive[
        positive_pred_candidates[
            "volume_m3_tree"
        ]
    ].to_numpy(dtype=np.float64)

    true_volume = positive[
        actual_target_columns[
            "volume_m3_tree"
        ]
    ].to_numpy(dtype=np.float64)

    # --------------------------------------------------------
    # Negative predictions
    # --------------------------------------------------------

    baseline_negative = (
        baseline_volume < 0
    )

    positive_negative = (
        positive_volume < 0
    )

    print("\nNegative volume counts:")

    print(
        f"Baseline : "
        f"{int(baseline_negative.sum()):,}"
    )

    print(
        f"Positive : "
        f"{int(positive_negative.sum()):,}"
    )

    # --------------------------------------------------------
    # P6 negatives
    # --------------------------------------------------------

    baseline_p6_negative = (
        p6_mask
        & baseline_negative
    )

    positive_p6_negative = (
        p6_mask
        & positive_negative
    )

    print("\nP6 negative volume counts:")

    print(
        f"Baseline : "
        f"{int(baseline_p6_negative.sum()):,}"
    )

    print(
        f"Positive : "
        f"{int(positive_p6_negative.sum()):,}"
    )

    # --------------------------------------------------------
    # P6 metrics
    # --------------------------------------------------------

    baseline_p6_metrics = metric_block(
        true_volume[p6_mask],
        baseline_volume[p6_mask],
    )

    positive_p6_metrics = metric_block(
        true_volume[p6_mask],
        positive_volume[p6_mask],
    )

    print("\nP6 volume metrics:")

    print(
        f"Baseline MAE  : "
        f"{baseline_p6_metrics[0]:.9f}"
    )

    print(
        f"Positive MAE  : "
        f"{positive_p6_metrics[0]:.9f}"
    )

    print(
        f"Baseline RMSE : "
        f"{baseline_p6_metrics[1]:.9f}"
    )

    print(
        f"Positive RMSE : "
        f"{positive_p6_metrics[1]:.9f}"
    )

    # --------------------------------------------------------
    # Prediction change
    # --------------------------------------------------------

    volume_change = (
        positive_volume
        - baseline_volume
    )

    abs_volume_change = np.abs(
        volume_change
    )

    print("\nVolume prediction change:")

    print(
        f"Mean change         : "
        f"{np.mean(volume_change):.9f}"
    )

    print(
        f"Mean absolute change: "
        f"{np.mean(abs_volume_change):.9f}"
    )

    print(
        f"Maximum abs change  : "
        f"{np.max(abs_volume_change):.9f}"
    )

    # --------------------------------------------------------
    # Status transitions
    # --------------------------------------------------------

    baseline_negative_to_positive = (
        baseline_negative
        & (~positive_negative)
    )

    baseline_positive_to_negative = (
        (~baseline_negative)
        & positive_negative
    )

    both_negative = (
        baseline_negative
        & positive_negative
    )

    both_positive = (
        (~baseline_negative)
        & (~positive_negative)
    )

    print("\nNegative-status transition:")

    print(
        f"Baseline negative -> Positive : "
        f"{int(baseline_negative_to_positive.sum()):,}"
    )

    print(
        f"Baseline positive -> Negative : "
        f"{int(baseline_positive_to_negative.sum()):,}"
    )

    print(
        f"Negative in both               : "
        f"{int(both_negative.sum()):,}"
    )

    print(
        f"Positive in both               : "
        f"{int(both_positive.sum()):,}"
    )

    # --------------------------------------------------------
    # Detailed diagnostic dataframe
    # --------------------------------------------------------

    diagnostic = positive.copy()

    diagnostic[
        "baseline_pinns_volume"
    ] = baseline_volume

    diagnostic[
        "positive_pinns_volume"
    ] = positive_volume

    diagnostic[
        "volume_change_positive_minus_baseline"
    ] = volume_change

    diagnostic[
        "baseline_volume_negative"
    ] = baseline_negative

    diagnostic[
        "positive_volume_negative"
    ] = positive_negative

    diagnostic[
        "true_volume"
    ] = true_volume

    diagnostic[
        "absolute_error_baseline"
    ] = np.abs(
        baseline_volume - true_volume
    )

    diagnostic[
        "absolute_error_positive"
    ] = np.abs(
        positive_volume - true_volume
    )

    # --------------------------------------------------------
    # P6 worst cases
    # --------------------------------------------------------

    p6_diagnostic = diagnostic[
        diagnostic["age_months"] == 6
    ].copy()

    p6_diagnostic = p6_diagnostic.sort_values(
        "positive_pinns_volume"
    )

    print(
        "\nWorst 10 positive-model P6 predictions:"
    )

    columns_to_show = [
        c
        for c in [
            "age_months",
            "inventory_year",
            "clone_factor",
            "planting_year",
            "disease_score",
            "trees_per_ha",
            "true_volume",
            "baseline_pinns_volume",
            "positive_pinns_volume",
            "volume_change_positive_minus_baseline",
        ]
        if c in p6_diagnostic.columns
    ]

    print(
        p6_diagnostic[
            columns_to_show
        ]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.9f}",
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    print(
        "\n[7/7] Saving diagnostic dataset..."
    )

    diagnostic.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved:\n"
        f"  {OUTPUT_PATH}"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n")
    print("=" * 75)
    print("STEP 21K COMPLETED")
    print("=" * 75)

    print(
        f"Baseline negative volumes : "
        f"{int(baseline_negative.sum()):,}"
    )

    print(
        f"Positive negative volumes : "
        f"{int(positive_negative.sum()):,}"
    )

    print(
        f"Baseline P6 negatives     : "
        f"{int(baseline_p6_negative.sum()):,}"
    )

    print(
        f"Positive P6 negatives     : "
        f"{int(positive_p6_negative.sum()):,}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()