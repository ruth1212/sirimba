from pathlib import Path
import importlib
import json

import pandas as pd


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "dataset.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "dashboard_growth_trajectory.csv"
)

CONFIG_PATH = (
    RESULTS_DIR
    / "dashboard_growth_trajectory_config.json"
)


# ============================================================
# IMPORT EXISTING TRAJECTORY ENGINE
# ============================================================

import sys

SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

trajectory_module = importlib.import_module(
    "24_compartment_trajectory_engine"
)

CompartmentTrajectoryEngine = (
    trajectory_module.CompartmentTrajectoryEngine
)


# ============================================================
# CONFIG
# ============================================================

MODEL_FEATURES = [
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha",
]

TRAJECTORY_AGES = list(
    range(6, 301, 6)
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("STEP 19A — BUILD DASHBOARD GROWTH DATASET")
    print("=" * 75)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load existing engine
    # --------------------------------------------------------

    print("\nInitializing trajectory engine...")

    engine = CompartmentTrajectoryEngine(
        DATA_PATH
    )

    # --------------------------------------------------------
    # Get all compartments
    # --------------------------------------------------------

    compartments = (
        engine.df[
            [
                "regional_id",
                "compartment_id",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "regional_id",
                "compartment_id",
            ]
        )
    )

    print(
        f"\nCompartments found: "
        f"{len(compartments)}"
    )

    # --------------------------------------------------------
    # Get all species-clone combinations
    # --------------------------------------------------------

    combinations = (
        engine.df[
            [
                "species",
                "clone",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "species",
                "clone",
            ]
        )
    )

    print(
        f"Species-clone combinations: "
        f"{len(combinations)}"
    )

    # --------------------------------------------------------
    # Expected size
    # --------------------------------------------------------

    expected_rows = (
        len(compartments)
        * len(combinations)
        * len(TRAJECTORY_AGES)
    )

    print(
        f"Expected trajectory rows: "
        f"{expected_rows:,}"
    )

    # --------------------------------------------------------
    # Build all trajectories
    # --------------------------------------------------------

    all_results = []

    total_compartments = len(
        compartments
    )

    for index, compartment in enumerate(
        compartments.itertuples(
            index=False
        ),
        start=1,
    ):

        regional_id = (
            compartment.regional_id
        )

        compartment_id = (
            compartment.compartment_id
        )

        print(
            f"\n[{index}/{total_compartments}] "
            f"{regional_id} / "
            f"COMP-{compartment_id}"
        )

        for combo in combinations.itertuples(
            index=False
        ):

            species = combo.species
            clone = combo.clone

            print(
                f"  → {species} / {clone}"
            )

            try:

                result = (
                    engine.predict_combination(
                        regional_id=regional_id,
                        compartment_id=compartment_id,
                        species=species,
                        clone=clone,
                    )
                )

                all_results.append(
                    result
                )

            except Exception as exc:

                raise RuntimeError(
                    f"\nFailed trajectory:\n"
                    f"Regional    : {regional_id}\n"
                    f"Compartment : {compartment_id}\n"
                    f"Species     : {species}\n"
                    f"Clone       : {clone}\n"
                    f"Error       : {exc}"
                ) from exc

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    print("\nCombining trajectories...")

    dashboard_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Remove accidental duplicates
    # --------------------------------------------------------

    key_columns = [
        "regional_id",
        "compartment_id",
        "species",
        "clone",
        "age_months",
    ]

    duplicates = dashboard_df.duplicated(
        subset=key_columns
    ).sum()

    print(
        f"Duplicate trajectory keys: "
        f"{duplicates}"
    )

    if duplicates > 0:

        raise ValueError(
            "Duplicate trajectory keys detected."
        )

    # --------------------------------------------------------
    # Validate expected rows
    # --------------------------------------------------------

    actual_rows = len(
        dashboard_df
    )

    print(
        f"\nActual rows: "
        f"{actual_rows:,}"
    )

    if actual_rows != expected_rows:

        raise ValueError(
            f"Unexpected row count. "
            f"Expected {expected_rows:,}, "
            f"got {actual_rows:,}."
        )

    # --------------------------------------------------------
    # Validate all compartments
    # --------------------------------------------------------

    actual_compartments = (
        dashboard_df[
            [
                "regional_id",
                "compartment_id",
            ]
        ]
        .drop_duplicates()
    )

    if len(actual_compartments) != len(
        compartments
    ):

        raise ValueError(
            "Not all compartments were generated."
        )

    # --------------------------------------------------------
    # Validate all species-clones
    # --------------------------------------------------------

    actual_combinations = (
        dashboard_df[
            [
                "species",
                "clone",
            ]
        ]
        .drop_duplicates()
    )

    if len(actual_combinations) != len(
        combinations
    ):

        raise ValueError(
            "Not all species-clone "
            "combinations were generated."
        )

    # --------------------------------------------------------
    # Validate age trajectory
    # --------------------------------------------------------

    actual_ages = sorted(
        dashboard_df[
            "age_months"
        ]
        .unique()
        .tolist()
    )

    if actual_ages != TRAJECTORY_AGES:

        raise ValueError(
            "Age trajectory does not match "
            "expected 6-month intervals."
        )

    # --------------------------------------------------------
    # Validate next planting year
    # --------------------------------------------------------

    planting_years = sorted(
        dashboard_df[
            "planting_year"
        ]
        .unique()
        .tolist()
    )

    if planting_years != [
        engine.next_planting_year
    ]:

        raise ValueError(
            "Unexpected prediction planting year."
        )

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "regional_id",
        "compartment_id",
        "compartment_code",

        "species",
        "clone",
        "clone_factor",

        "planting_year",
        "inventory_year",

        "growth_period",
        "age_months",

        "disease_score",

        "area_ha",

        "spacing_m",
        "spacing_x_m",
        "spacing_y_m",
        "plant_spacing_m",

        "trees_per_ha",
        "estimated_trees_compartment",

        "rf_dbh_cm",
        "rf_height_m",
        "rf_volume_m3_tree",
        "rf_tha_m",
        "rf_mai_m3_tree_year",
        "rf_volume_m3_ha",
        "rf_total_volume_m3",
        "rf_mai_m3_ha_year",

        "pinns_dbh_cm",
        "pinns_height_m",
        "pinns_volume_m3_tree",
        "pinns_mai_m3_tree_year",
        "pinns_volume_m3_ha",
        "pinns_total_volume_m3",
        "pinns_mai_m3_ha_year",
    ]

    missing = [
        col
        for col in required_columns
        if col not in dashboard_df.columns
    ]

    if missing:

        raise ValueError(
            "Missing dashboard columns:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # Validate NaN
    # --------------------------------------------------------

    required_nan = (
        dashboard_df[
            required_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Required-column NaN: "
        f"{required_nan}"
    )

    if required_nan > 0:

        raise ValueError(
            "NaN detected in dashboard dataset."
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    dashboard_df = (
        dashboard_df
        .sort_values(
            [
                "regional_id",
                "compartment_id",
                "species",
                "clone",
                "age_months",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    print("\nSaving dashboard dataset...")

    dashboard_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    config = {
        "source_dataset":
            "data/dataset.csv",

        "historical_planting_years":
            engine.planting_years,

        "prediction_planting_year":
            engine.next_planting_year,

        "planting_interval_years":
            engine.median_planting_interval,

        "compartment_count":
            int(len(compartments)),

        "species_clone_count":
            int(len(combinations)),

        "trajectory_period_count":
            int(len(TRAJECTORY_AGES)),

        "expected_rows":
            int(expected_rows),

        "actual_rows":
            int(actual_rows),

        "model_features":
            MODEL_FEATURES,

        "trajectory_age_months":
            TRAJECTORY_AGES,

        "ui_filters": [
            "regional_id",
            "compartment_id",
            "species",
            "clone",
        ],

        "planting_year_ui_filter":
            False,

        "model_selector": [
            "random_forest",
            "pinns",
        ],

        "mai_tree_unit":
            "m3/tree/year",

        "mai_stand_unit":
            "m3/ha/year",

        "volume_tree_unit":
            "m3/tree",

        "volume_ha_unit":
            "m3/ha",

        "total_volume_unit":
            "m3",

        "tha_unit":
            "m",

        "note":
            "Dashboard trajectory dataset generated "
            "from the existing RF, PINNs, and THA models. "
            "No new model training was performed.",
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("BUILD SUMMARY")
    print("=" * 75)

    print(
        f"Compartments          : "
        f"{len(compartments)}"
    )

    print(
        f"Species-clone combos  : "
        f"{len(combinations)}"
    )

    print(
        f"Periods per combo     : "
        f"{len(TRAJECTORY_AGES)}"
    )

    print(
        f"Expected rows         : "
        f"{expected_rows:,}"
    )

    print(
        f"Actual rows           : "
        f"{actual_rows:,}"
    )

    print(
        f"Prediction cohort     : "
        f"{engine.next_planting_year}"
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        CONFIG_PATH
    )

    print("\n" + "=" * 75)
    print("STEP 19A COMPLETED")
    print("=" * 75)

    print(
        "\nDashboard trajectory dataset "
        "is ready for API integration."
    )

    print(
        "No model training was performed."
    )


if __name__ == "__main__":
    main()