from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "actual_growth_trajectory.csv"


DISPLAY_PERIODS = [
    "P6",
    "P12",
    "P18",
    "P24",
    "P30",
    "P36",
    "P42",
    "P48",
    "P54",
    "P60",
    "PHI",
]


FILTER_COLUMNS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
]


TARGET_COLUMNS = [
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]


def load_dataset() -> pd.DataFrame:
    print("=" * 70)
    print("ACTUAL GROWTH TRAJECTORY ENGINE")
    print("=" * 70)

    print(f"\nLoading dataset:")
    print(DATASET_PATH)

    df = pd.read_csv(DATASET_PATH)

    print(f"Dataset rows : {len(df):,}")
    print(f"Dataset cols : {len(df.columns)}")

    return df


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = (
        FILTER_COLUMNS
        + [
            "growth_period",
            "age_months",
        ]
        + TARGET_COLUMNS
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def build_actual_trajectory(
    df: pd.DataFrame,
    regional_id: str,
    compartment_id: int,
    species: str | None = None,
    clone: str | None = None,
) -> pd.DataFrame:

    print("\n" + "-" * 70)
    print("FILTER")
    print("-" * 70)

    print(f"Regional    : {regional_id}")
    print(f"Compartment : {compartment_id}")
    print(f"Species     : {species or 'All'}")
    print(f"Clone       : {clone or 'All'}")

    filtered = df[
        (df["regional_id"] == regional_id)
        & (df["compartment_id"] == compartment_id)
    ].copy()

    if species is not None:
        filtered = filtered[
            filtered["species"] == species
        ]

    if clone is not None:
        filtered = filtered[
            filtered["clone"] == clone
        ]

    if filtered.empty:
        raise ValueError(
            "No actual data found for the selected filters."
        )

    print(f"\nFiltered rows: {len(filtered):,}")

    # --------------------------------------------------------------
    # Keep only periods used by the dashboard.
    # P66 is intentionally excluded.
    # --------------------------------------------------------------
    filtered = filtered[
        filtered["growth_period"].isin(DISPLAY_PERIODS)
    ].copy()

    # --------------------------------------------------------------
    # For P6-P60:
    # one growth_period corresponds to one exact age.
    #
    # For PHI:
    # age_months ranges from 72 to 300.
    # PHI is therefore represented as the aggregate of all
    # observations classified as PHI.
    # --------------------------------------------------------------

    grouped = (
        filtered
        .groupby("growth_period", as_index=False)
        .agg(
            age_months_mean=("age_months", "mean"),
            dbh_cm=("dbh_cm", "mean"),
            height_m=("height_m", "mean"),
            volume_m3_tree=("volume_m3_tree", "mean"),
            actual_rows=("age_months", "size"),
        )
    )

    # --------------------------------------------------------------
    # Sort using dashboard display order rather than alphabetically.
    # --------------------------------------------------------------
    period_order = {
        period: index
        for index, period in enumerate(DISPLAY_PERIODS)
    }

    grouped["period_order"] = grouped[
        "growth_period"
    ].map(period_order)

    grouped = grouped.sort_values(
        "period_order"
    ).drop(
        columns=["period_order"]
    )

    # --------------------------------------------------------------
    # Make the source clear.
    # --------------------------------------------------------------
    grouped.insert(
        0,
        "model",
        "actual"
    )

    grouped.insert(
        1,
        "regional_id",
        regional_id
    )

    grouped.insert(
        2,
        "compartment_id",
        compartment_id
    )

    grouped.insert(
        3,
        "species",
        species if species is not None else "All"
    )

    grouped.insert(
        4,
        "clone",
        clone if clone is not None else "All"
    )

    return grouped


def main() -> None:

    df = load_dataset()

    validate_columns(df)

    result = build_actual_trajectory(
        df=df,
        regional_id="R01",
        compartment_id=101,
        species="Eucalyptus grandis",
        clone="EG-01",
    )

    print("\n" + "-" * 70)
    print("ACTUAL TRAJECTORY")
    print("-" * 70)

    print(
        result[
            [
                "growth_period",
                "age_months_mean",
                "dbh_cm",
                "height_m",
                "volume_m3_tree",
                "actual_rows",
            ]
        ].round(6).to_string(index=False)
    )

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    expected_periods = set(DISPLAY_PERIODS)
    actual_periods = set(
        result["growth_period"].tolist()
    )

    missing_periods = expected_periods - actual_periods

    if missing_periods:
        raise ValueError(
            f"Missing dashboard periods: {sorted(missing_periods)}"
        )

    if len(result) != len(DISPLAY_PERIODS):
        raise ValueError(
            f"Expected {len(DISPLAY_PERIODS)} periods, "
            f"got {len(result)}"
        )

    numeric_columns = [
        "dbh_cm",
        "height_m",
        "volume_m3_tree",
    ]

    for column in numeric_columns:
        if not result[column].notna().all():
            raise ValueError(
                f"Null values detected in {column}"
            )

    print("\nValidation: PASSED")
    print(f"Periods    : {len(result)}")
    print("P66 shown  : NO")
    print("PHI shown  : YES")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("ACTUAL TRAJECTORY ENGINE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()