from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

INPUT_PATH = (
    RESULTS / "pinns_positive_diagnostic_comparison.csv"
)

OUTPUT_AGE = (
    RESULTS / "pinns_negative_by_age.csv"
)

OUTPUT_CLONE = (
    RESULTS / "pinns_negative_by_clone_factor.csv"
)

OUTPUT_DISEASE = (
    RESULTS / "pinns_negative_by_disease.csv"
)

OUTPUT_TREES = (
    RESULTS / "pinns_negative_by_trees_per_ha.csv"
)

OUTPUT_COMBINED = (
    RESULTS / "pinns_negative_pattern_summary.csv"
)


def summarize_group(
    df,
    group_col,
    baseline_col="baseline_pinns_volume",
    positive_col="positive_pinns_volume",
    true_col="true_volume",
):
    rows = []

    for value, group in df.groupby(
        group_col,
        dropna=False,
    ):

        baseline_negative = (
            group[baseline_col] < 0
        )

        positive_negative = (
            group[positive_col] < 0
        )

        true = group[true_col].to_numpy(
            dtype=float
        )

        baseline = group[baseline_col].to_numpy(
            dtype=float
        )

        positive = group[positive_col].to_numpy(
            dtype=float
        )

        rows.append(
            {
                group_col: value,
                "rows": len(group),

                "baseline_negative": int(
                    baseline_negative.sum()
                ),

                "positive_negative": int(
                    positive_negative.sum()
                ),

                "baseline_negative_pct": (
                    baseline_negative.mean() * 100
                ),

                "positive_negative_pct": (
                    positive_negative.mean() * 100
                ),

                "true_volume_mean": np.mean(true),

                "baseline_volume_mean": np.mean(
                    baseline
                ),

                "positive_volume_mean": np.mean(
                    positive
                ),

                "baseline_mae": np.mean(
                    np.abs(baseline - true)
                ),

                "positive_mae": np.mean(
                    np.abs(positive - true)
                ),

                "positive_min": np.min(
                    positive
                ),

                "positive_max": np.max(
                    positive
                ),
            }
        )

    return pd.DataFrame(rows)


def main():

    print("=" * 75)
    print("TREE GROWTH ANALYTICS")
    print("STEP 21L - ANALYZE PINNs NEGATIVE PATTERNS")
    print("=" * 75)

    # ========================================================
    # 1. LOAD
    # ========================================================

    print("\n[1/8] Loading diagnostic dataset...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(
        f"Rows : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    required = [
        "age_months",
        "clone_factor",
        "disease_score",
        "trees_per_ha",
        "baseline_pinns_volume",
        "positive_pinns_volume",
        "true_volume",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # ========================================================
    # 2. GLOBAL
    # ========================================================

    print("\n[2/8] Global negative-volume analysis...")

    baseline_negative = (
        df["baseline_pinns_volume"] < 0
    )

    positive_negative = (
        df["positive_pinns_volume"] < 0
    )

    print(
        f"Baseline negative : "
        f"{int(baseline_negative.sum()):,}"
    )

    print(
        f"Positive negative : "
        f"{int(positive_negative.sum()):,}"
    )

    print(
        f"Positive negative % : "
        f"{positive_negative.mean() * 100:.6f}%"
    )

    # ========================================================
    # 3. AGE
    # ========================================================

    print("\n[3/8] Analyzing by age...")

    age_summary = summarize_group(
        df,
        "age_months",
    )

    age_summary.to_csv(
        OUTPUT_AGE,
        index=False,
    )

    print("\nAge groups with positive negative predictions:")

    print(
        age_summary[
            age_summary["positive_negative"] > 0
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ========================================================
    # 4. CLONE FACTOR
    # ========================================================

    print("\n[4/8] Analyzing by clone_factor...")

    clone_summary = summarize_group(
        df,
        "clone_factor",
    )

    clone_summary.to_csv(
        OUTPUT_CLONE,
        index=False,
    )

    print("\nClone-factor groups:")

    print(
        clone_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ========================================================
    # 5. DISEASE
    # ========================================================

    print("\n[5/8] Analyzing by disease_score...")

    disease_summary = summarize_group(
        df,
        "disease_score",
    )

    disease_summary.to_csv(
        OUTPUT_DISEASE,
        index=False,
    )

    print("\nDisease groups with positive negative predictions:")

    print(
        disease_summary[
            disease_summary["positive_negative"] > 0
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ========================================================
    # 6. TREES PER HA
    # ========================================================

    print("\n[6/8] Analyzing by trees_per_ha...")

    trees_summary = summarize_group(
        df,
        "trees_per_ha",
    )

    trees_summary.to_csv(
        OUTPUT_TREES,
        index=False,
    )

    print("\nTrees/ha groups with positive negative predictions:")

    print(
        trees_summary[
            trees_summary["positive_negative"] > 0
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ========================================================
    # 7. NEGATIVE ROW CHARACTERISTICS
    # ========================================================

    print("\n[7/8] Analyzing negative-row characteristics...")

    negative_rows = df[
        positive_negative
    ].copy()

    positive_rows = df[
        ~positive_negative
    ].copy()

    print(
        f"\nPositive-model negative rows : "
        f"{len(negative_rows):,}"
    )

    print(
        f"Positive-model non-negative rows : "
        f"{len(positive_rows):,}"
    )

    numeric_columns = [
        "age_months",
        "clone_factor",
        "disease_score",
        "trees_per_ha",
        "true_volume",
        "baseline_pinns_volume",
        "positive_pinns_volume",
    ]

    print("\nNegative-row means:")

    print(
        negative_rows[numeric_columns]
        .mean()
        .to_string(
            float_format=lambda x: f"{x:.9f}"
        )
    )

    print("\nNon-negative-row means:")

    print(
        positive_rows[numeric_columns]
        .mean()
        .to_string(
            float_format=lambda x: f"{x:.9f}"
        )
    )

    # --------------------------------------------------------
    # Age distribution
    # --------------------------------------------------------

    print("\nNegative counts by age:")

    negative_by_age = (
        negative_rows
        .groupby("age_months")
        .size()
        .reset_index(
            name="negative_count"
        )
    )

    print(
        negative_by_age.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Species/clone information if available
    # --------------------------------------------------------

    if "species" in df.columns:

        print(
            "\nNegative counts by species:"
        )

        print(
            negative_rows
            .groupby("species")
            .size()
            .reset_index(
                name="negative_count"
            )
            .to_string(
                index=False
            )
        )

    if "clone" in df.columns:

        print(
            "\nNegative counts by clone:"
        )

        print(
            negative_rows
            .groupby("clone")
            .size()
            .reset_index(
                name="negative_count"
            )
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Most negative predictions
    # --------------------------------------------------------

    print(
        "\n10 most negative positive-model predictions:"
    )

    worst_columns = [
        "age_months",
        "inventory_year",
        "clone_factor",
        "planting_year",
        "disease_score",
        "trees_per_ha",
        "true_volume",
        "baseline_pinns_volume",
        "positive_pinns_volume",
    ]

    worst_columns = [
        col for col in worst_columns
        if col in negative_rows.columns
    ]

    print(
        negative_rows.sort_values(
            "positive_pinns_volume"
        )[worst_columns]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.9f}",
        )
    )

    # ========================================================
    # 8. COMBINED SUMMARY
    # ========================================================

    print("\n[8/8] Saving summary...")

    combined_rows = [
        {
            "analysis": "global",
            "group": "ALL",
            "rows": len(df),
            "baseline_negative": int(
                baseline_negative.sum()
            ),
            "positive_negative": int(
                positive_negative.sum()
            ),
            "positive_negative_pct": (
                positive_negative.mean() * 100
            ),
        }
    ]

    combined_rows.append(
        {
            "analysis": "age",
            "group": "P6",
            "rows": int(
                (df["age_months"] == 6).sum()
            ),
            "baseline_negative": int(
                (
                    (df["age_months"] == 6)
                    & baseline_negative
                ).sum()
            ),
            "positive_negative": int(
                (
                    (df["age_months"] == 6)
                    & positive_negative
                ).sum()
            ),
            "positive_negative_pct": (
                (
                    (df["age_months"] == 6)
                    & positive_negative
                ).sum()
                / max(
                    (df["age_months"] == 6).sum(),
                    1,
                )
                * 100
            ),
        }
    )

    combined_rows.append(
        {
            "analysis": "age",
            "group": "AGE_GT_6",
            "rows": int(
                (df["age_months"] > 6).sum()
            ),
            "baseline_negative": int(
                (
                    (df["age_months"] > 6)
                    & baseline_negative
                ).sum()
            ),
            "positive_negative": int(
                (
                    (df["age_months"] > 6)
                    & positive_negative
                ).sum()
            ),
            "positive_negative_pct": (
                (
                    (df["age_months"] > 6)
                    & positive_negative
                ).sum()
                / max(
                    (df["age_months"] > 6).sum(),
                    1,
                )
                * 100
            ),
        }
    )

    combined_df = pd.DataFrame(
        combined_rows
    )

    combined_df.to_csv(
        OUTPUT_COMBINED,
        index=False,
    )

    print(
        f"\nSaved:"
    )

    print(
        f"  - {OUTPUT_AGE}"
    )

    print(
        f"  - {OUTPUT_CLONE}"
    )

    print(
        f"  - {OUTPUT_DISEASE}"
    )

    print(
        f"  - {OUTPUT_TREES}"
    )

    print(
        f"  - {OUTPUT_COMBINED}"
    )

    print("\n")
    print("=" * 75)
    print("STEP 21L COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()