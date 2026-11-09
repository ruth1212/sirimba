from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "dataset_phi66_final.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "dataset_phi66_final_volume_fixed.csv"
)

BACKUP_PATH = (
    PROJECT_ROOT
    / "data"
    / "dataset_phi66_final_before_volume_fix.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "results"
    / "pinns"
    / "step57_actual_volume_audit.json"
)

AGE_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "pinns"
    / "step57_actual_volume_by_age.csv"
)


GROUP_COLS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]

REQUIRED_COLS = [
    "age_months",
    "growth_period",
    "volume_m3_tree",
]


def main():

    print("=" * 80)
    print("STEP 57 — AUDIT & REPAIR ACTUAL VOLUME")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load final dataset
    # ------------------------------------------------------------------
    print("\n[1/8] Loading final dataset...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    missing = [
        col
        for col in REQUIRED_COLS + GROUP_COLS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {missing}"
        )

    # ------------------------------------------------------------------
    # 2. Basic actual volume audit
    # ------------------------------------------------------------------
    print("\n[2/8] Auditing actual volume...")

    volume = pd.to_numeric(
        df["volume_m3_tree"],
        errors="coerce"
    )

    nan_count = int(volume.isna().sum())
    inf_count = int(
        np.isinf(volume.to_numpy()).sum()
    )
    negative_count = int((volume < 0).sum())
    zero_count = int((volume == 0).sum())
    positive_count = int((volume > 0).sum())

    print(f"NaN volume      : {nan_count:,}")
    print(f"Inf volume      : {inf_count:,}")
    print(f"Negative volume : {negative_count:,}")
    print(f"Zero volume     : {zero_count:,}")
    print(f"Positive volume : {positive_count:,}")

    print("\nVolume statistics:")

    print(
        volume.describe().to_string()
    )

    # ------------------------------------------------------------------
    # 3. Age profile
    # ------------------------------------------------------------------
    print("\n[3/8] Building actual volume age profile...")

    age_summary = (
        df
        .groupby(
            ["age_months", "growth_period"],
            as_index=False
        )
        .agg(
            rows=("volume_m3_tree", "size"),
            mean_volume=("volume_m3_tree", "mean"),
            median_volume=("volume_m3_tree", "median"),
            min_volume=("volume_m3_tree", "min"),
            max_volume=("volume_m3_tree", "max"),
            zero_count=(
                "volume_m3_tree",
                lambda x: int((x == 0).sum())
            ),
            negative_count=(
                "volume_m3_tree",
                lambda x: int((x < 0).sum())
            ),
        )
        .sort_values("age_months")
    )

    print(
        age_summary.to_string(index=False)
    )

    # ------------------------------------------------------------------
    # 4. Decide whether repair is necessary
    # ------------------------------------------------------------------
    print("\n[4/8] Determining whether dataset repair is necessary...")

    invalid_mask = (
        volume.isna()
        | np.isinf(volume.to_numpy())
        | (volume < 0)
    )

    invalid_count = int(invalid_mask.sum())

    if invalid_count == 0:

        print("\nRESULT:")
        print("No invalid actual volume found.")
        print("Dataset will NOT be modified.")

        repair_performed = False
        repaired_rows = 0

        # Preserve exact dataset
        output_path = INPUT_PATH

    else:

        print(
            f"\nFound {invalid_count:,} invalid volume rows."
        )

        print(
            "Repair will be performed using "
            "group-age neighboring values."
        )

        # --------------------------------------------------------------
        # 5. Backup before repair
        # --------------------------------------------------------------
        print("\n[5/8] Creating backup...")

        shutil.copy2(
            INPUT_PATH,
            BACKUP_PATH
        )

        print(f"Backup saved:\n{BACKUP_PATH}")

        # --------------------------------------------------------------
        # 6. Repair invalid volume
        # --------------------------------------------------------------
        print("\n[6/8] Repairing invalid actual volume...")

        repaired = df.copy()

        repaired["volume_m3_tree"] = pd.to_numeric(
            repaired["volume_m3_tree"],
            errors="coerce"
        )

        # Process each growth group separately.
        #
        # We do NOT use the PINNs predictions.
        # We do NOT use RF predictions.
        #
        # For invalid values, use interpolation along the same
        # biological growth group. If interpolation cannot be
        # performed at an edge, use the nearest valid value.
        #
        # This preserves the observed group trajectory as much
        # as possible without introducing model predictions.

        repair_log = []

        repaired = repaired.sort_values(
            GROUP_COLS + ["age_months"]
        ).copy()

        for group_key, group in repaired.groupby(
            GROUP_COLS,
            sort=False
        ):

            idx = group.index

            values = group["volume_m3_tree"].copy()

            valid = (
                values.notna()
                & np.isfinite(values)
                & (values >= 0)
            )

            invalid_indices = values.index[~valid]

            if len(invalid_indices) == 0:
                continue

            valid_values = values.where(valid)

            interpolated = (
                valid_values
                .interpolate(
                    method="linear",
                    limit_direction="both"
                )
            )

            for row_idx in invalid_indices:

                old_value = values.loc[row_idx]
                new_value = interpolated.loc[row_idx]

                if pd.isna(new_value):
                    raise ValueError(
                        "Tidak ada nilai volume valid "
                        "untuk memperbaiki group."
                    )

                new_value = max(
                    float(new_value),
                    0.0
                )

                repaired.loc[
                    row_idx,
                    "volume_m3_tree"
                ] = new_value

                repair_log.append(
                    {
                        "row_index": int(row_idx),
                        "old_volume": (
                            None
                            if pd.isna(old_value)
                            else float(old_value)
                        ),
                        "new_volume": float(new_value),
                        "age_months": int(
                            repaired.loc[
                                row_idx,
                                "age_months"
                            ]
                        ),
                    }
                )

        repaired_rows = len(repair_log)

        if repaired_rows != invalid_count:
            raise RuntimeError(
                "Jumlah repaired rows tidak sama "
                "dengan jumlah invalid rows."
            )

        df = repaired.sort_index()

        output_path = OUTPUT_PATH

        df.to_csv(
            output_path,
            index=False
        )

        repair_performed = True

        print(
            f"Repaired rows: {repaired_rows:,}"
        )

        print(
            f"Saved repaired dataset:\n{output_path}"
        )

    # ------------------------------------------------------------------
    # 7. Final validation
    # ------------------------------------------------------------------
    print("\n[7/8] Final volume validation...")

    final_df = pd.read_csv(output_path)

    final_volume = pd.to_numeric(
        final_df["volume_m3_tree"],
        errors="coerce"
    )

    final_nan = int(final_volume.isna().sum())
    final_inf = int(
        np.isinf(final_volume.to_numpy()).sum()
    )
    final_negative = int(
        (final_volume < 0).sum()
    )

    print(f"Final NaN      : {final_nan:,}")
    print(f"Final Inf      : {final_inf:,}")
    print(f"Final negative : {final_negative:,}")

    if (
        final_nan > 0
        or final_inf > 0
        or final_negative > 0
    ):
        raise RuntimeError(
            "FINAL VALIDATION FAILED: "
            "actual volume masih memiliki nilai invalid."
        )

    # Age restriction
    ages = sorted(
        final_df["age_months"]
        .dropna()
        .unique()
        .tolist()
    )

    print(f"Final ages: {ages}")

    if min(ages) != 6 or max(ages) != 66:
        raise RuntimeError(
            "Age range bukan P6-P66."
        )

    # Volume prediction range rule
    volume_prediction_ages = sorted(
        [
            age
            for age in ages
            if age >= 18
        ]
    )

    print(
        "Volume modeling ages:",
        volume_prediction_ages
    )

    # Rebuild age summary after repair
    final_age_summary = (
        final_df
        .groupby(
            ["age_months", "growth_period"],
            as_index=False
        )
        .agg(
            rows=("volume_m3_tree", "size"),
            mean_volume=("volume_m3_tree", "mean"),
            median_volume=("volume_m3_tree", "median"),
            min_volume=("volume_m3_tree", "min"),
            max_volume=("volume_m3_tree", "max"),
            zero_count=(
                "volume_m3_tree",
                lambda x: int((x == 0).sum())
            ),
            negative_count=(
                "volume_m3_tree",
                lambda x: int((x < 0).sum())
            ),
        )
        .sort_values("age_months")
    )

    final_age_summary.to_csv(
        AGE_SUMMARY_PATH,
        index=False
    )

    # ------------------------------------------------------------------
    # 8. Report
    # ------------------------------------------------------------------
    print("\n[8/8] Saving audit report...")

    report = {
        "step": 57,
        "input_dataset": str(INPUT_PATH),
        "output_dataset": str(output_path),
        "rows": int(len(final_df)),
        "columns": int(len(final_df.columns)),
        "initial_invalid_volume_rows": int(invalid_count),
        "initial_negative_volume_rows": int(negative_count),
        "initial_zero_volume_rows": int(zero_count),
        "initial_nan_volume_rows": int(nan_count),
        "initial_inf_volume_rows": int(inf_count),
        "repair_performed": bool(repair_performed),
        "repaired_rows": int(repaired_rows),
        "final_negative_volume_rows": int(final_negative),
        "final_zero_volume_rows": int(
            (final_volume == 0).sum()
        ),
        "final_nan_volume_rows": int(final_nan),
        "final_inf_volume_rows": int(final_inf),
        "age_min": int(min(ages)),
        "age_max": int(max(ages)),
        "volume_prediction_age_min": 18,
        "volume_prediction_age_max": 66,
        "dataset_used_for_repair": (
            "same growth group neighboring actual values"
            if repair_performed
            else "none"
        ),
        "rf_prediction_used": False,
        "pinns_prediction_used": False,
        "dataset_modified": bool(repair_performed),
        "model_modified": False,
        "retraining_performed": False,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print("STEP 57 COMPLETE")
    print("=" * 80)

    print(f"\nDataset output:")
    print(output_path)

    print(f"\nAudit report:")
    print(REPORT_PATH)

    print(f"\nAge summary:")
    print(AGE_SUMMARY_PATH)

    print("\nFinal status:")
    print(f"Actual volume negative : {final_negative:,}")
    print(f"Actual volume NaN      : {final_nan:,}")
    print(f"Actual volume Inf      : {final_inf:,}")

    if repair_performed:
        print(
            f"Actual volume repaired : {repaired_rows:,} rows"
        )
    else:
        print(
            "Actual volume repair   : NOT NEEDED"
        )

    print("\nPINNs retraining: NOT PERFORMED")


if __name__ == "__main__":
    main()