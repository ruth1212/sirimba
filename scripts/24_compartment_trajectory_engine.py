from pathlib import Path
import sys
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

ENGINE_OUTPUT = RESULTS_DIR / "compartment_trajectory_smoke_test.csv"
ENGINE_CONFIG = RESULTS_DIR / "compartment_trajectory_config.json"

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT EXISTING PREDICTION ENGINE
# ============================================================

import importlib

prediction_engine_module = importlib.import_module(
    "scripts.22_prediction_engine_final"
)

TreeGrowthPredictionEngine = (
    prediction_engine_module.TreeGrowthPredictionEngine
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FEATURES = [
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha",
]

TARGETS = [
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]

TRAJECTORY_AGES = list(range(6, 301, 6))


# ============================================================
# DATASET COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "regional_id",
    "regional_name",
    "compartment_id",
    "compartment_code",
    "area_ha",
    "spacing_m",
    "spacing_x_m",
    "spacing_y_m",
    "plant_spacing_m",
    "trees_per_ha",
    "estimated_trees_compartment",
    "species_code",
    "species",
    "clone",
    "clone_factor",
    "planting_year",
    "inventory_year",
    "age_months",
    "growth_period",
    "disease_score",
    "tha_m",
]


# ============================================================
# HELPERS
# ============================================================

def period_label(age_months: int) -> str:
    """
    Convert age in months to dashboard period label.

    P6 ... P66
    >=72 months -> PHI
    """

    if age_months <= 66:
        return f"P{age_months}"

    return "PHI"


def calculate_mai(volume_per_tree: float, age_months: int) -> float:
    """
    MAI per tree:

        MAI = volume / age_years

    Unit:
        m3/tree/year
    """

    if age_months <= 0:
        return np.nan

    age_years = age_months / 12.0

    return volume_per_tree / age_years


def calculate_stand_mai(
    volume_per_ha: float,
    age_months: int,
) -> float:
    """
    Stand-level MAI:

        MAI = volume/ha / age_years

    Unit:
        m3/ha/year
    """

    if age_months <= 0:
        return np.nan

    age_years = age_months / 12.0

    return volume_per_ha / age_years


# ============================================================
# ENGINE
# ============================================================

class CompartmentTrajectoryEngine:

    def __init__(self, dataset_path: Path):

        print("=" * 75)
        print("STEP 18B — COMPARTMENT GROWTH TRAJECTORY ENGINE")
        print("=" * 75)

        self.dataset_path = dataset_path

        print("\nLoading dataset...")
        self.df = pd.read_csv(dataset_path)

        print(f"Rows    : {len(self.df):,}")
        print(f"Columns : {len(self.df.columns)}")

        self._validate_dataset()

        # ----------------------------------------------------
        # Remove exact duplicates only
        # ----------------------------------------------------

        before = len(self.df)

        self.df = self.df.drop_duplicates().copy()

        removed = before - len(self.df)

        print(f"Duplicate rows removed: {removed}")

        # ----------------------------------------------------
        # Historical planting years
        # ----------------------------------------------------

        self.planting_years = sorted(
            self.df["planting_year"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        if len(self.planting_years) < 2:
            raise ValueError(
                "At least two historical planting years are required."
            )

        intervals = np.diff(self.planting_years)

        self.median_planting_interval = int(
            np.median(intervals)
        )

        self.next_planting_year = (
            max(self.planting_years)
            + self.median_planting_interval
        )

        print(
            f"Historical planting years : {self.planting_years}"
        )

        print(
            f"Planting interval          : "
            f"{self.median_planting_interval} years"
        )

        print(
            f"Next planting year         : "
            f"{self.next_planting_year}"
        )

        # ----------------------------------------------------
        # Compartment attributes
        # ----------------------------------------------------

        self.compartment_attributes = (
            self.df[
                [
                    "regional_id",
                    "regional_name",
                    "compartment_id",
                    "compartment_code",
                    "area_ha",
                    "spacing_m",
                    "spacing_x_m",
                    "spacing_y_m",
                    "plant_spacing_m",
                    "trees_per_ha",
                    "estimated_trees_compartment",
                ]
            ]
            .drop_duplicates(
                subset=[
                    "regional_id",
                    "compartment_id",
                ]
            )
            .copy()
        )

        # ----------------------------------------------------
        # Disease history
        # ----------------------------------------------------

        self._prepare_disease_history()

        # ----------------------------------------------------
        # Prediction model
        # ----------------------------------------------------

        print("\nLoading existing prediction models...")

        self.model_engine = TreeGrowthPredictionEngine()

        print("Prediction models loaded successfully.")

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_dataset(self):

        missing = [
            col
            for col in REQUIRED_COLUMNS
            if col not in self.df.columns
        ]

        if missing:
            raise ValueError(
                "Missing required columns:\n"
                + "\n".join(missing)
            )

        print("Required dataset columns: OK")

    # ========================================================
    # DISEASE HISTORY
    # ========================================================

    def _prepare_disease_history(self):

        print("\nPreparing historical disease pattern...")

        disease_columns = [
            "compartment_id",
            "species",
            "clone",
            "age_months",
            "disease_score",
        ]

        disease_df = self.df[disease_columns].copy()

        disease_df = (
            disease_df
            .groupby(
                [
                    "compartment_id",
                    "species",
                    "clone",
                    "age_months",
                ],
                as_index=False,
            )["disease_score"]
            .mean()
        )

        self.disease_history = disease_df

        # ----------------------------------------------------
        # Global fallback by species + clone + age
        # ----------------------------------------------------

        self.disease_species_clone = (
            self.df[
                [
                    "species",
                    "clone",
                    "age_months",
                    "disease_score",
                ]
            ]
            .groupby(
                [
                    "species",
                    "clone",
                    "age_months",
                ],
                as_index=False,
            )["disease_score"]
            .mean()
        )

        # ----------------------------------------------------
        # Global fallback by age
        # ----------------------------------------------------

        self.disease_age = (
            self.df[
                [
                    "age_months",
                    "disease_score",
                ]
            ]
            .groupby(
                ["age_months"],
                as_index=False,
            )["disease_score"]
            .mean()
        )

        print(
            f"Disease history rows: "
            f"{len(self.disease_history):,}"
        )

    # ========================================================
    # DISEASE LOOKUP
    # ========================================================

    def _get_disease_score(
        self,
        compartment_id,
        species,
        clone,
        age_months,
    ):

        # ----------------------------------------------------
        # 1. Exact compartment + species + clone + age
        # ----------------------------------------------------

        result = self.disease_history[
            (
                self.disease_history["compartment_id"]
                == compartment_id
            )
            &
            (
                self.disease_history["species"]
                == species
            )
            &
            (
                self.disease_history["clone"]
                == clone
            )
            &
            (
                self.disease_history["age_months"]
                == age_months
            )
        ]

        if not result.empty:
            return float(
                result["disease_score"].iloc[0]
            )

        # ----------------------------------------------------
        # 2. Species + clone + age
        # ----------------------------------------------------

        result = self.disease_species_clone[
            (
                self.disease_species_clone["species"]
                == species
            )
            &
            (
                self.disease_species_clone["clone"]
                == clone
            )
            &
            (
                self.disease_species_clone["age_months"]
                == age_months
            )
        ]

        if not result.empty:
            return float(
                result["disease_score"].iloc[0]
            )

        # ----------------------------------------------------
        # 3. Global age fallback
        # ----------------------------------------------------

        result = self.disease_age[
            self.disease_age["age_months"]
            == age_months
        ]

        if not result.empty:
            return float(
                result["disease_score"].iloc[0]
            )

        # ----------------------------------------------------
        # 4. Final global fallback
        # ----------------------------------------------------

        return float(
            self.df["disease_score"].mean()
        )

    # ========================================================
    # COMPARTMENT + SPECIES + CLONE TRAJECTORY
    # ========================================================

    def predict_combination(
        self,
        regional_id,
        compartment_id,
        species,
        clone,
    ):

        # ----------------------------------------------------
        # Find compartment
        # ----------------------------------------------------

        compartment_rows = self.df[
            (
                self.df["regional_id"]
                == regional_id
            )
            &
            (
                self.df["compartment_id"]
                == compartment_id
            )
        ]

        if compartment_rows.empty:
            raise ValueError(
                "Compartment not found."
            )

        # ----------------------------------------------------
        # Find species + clone
        # ----------------------------------------------------

        combination = compartment_rows[
            (
                compartment_rows["species"]
                == species
            )
            &
            (
                compartment_rows["clone"]
                == clone
            )
        ]

        if combination.empty:
            raise ValueError(
                "Species + clone combination "
                "not found in compartment."
            )

        # ----------------------------------------------------
        # Static compartment information
        # ----------------------------------------------------

        attributes = (
            compartment_rows.iloc[0]
        )

        area_ha = float(
            attributes["area_ha"]
        )

        spacing_m = attributes["spacing_m"]

        spacing_x_m = float(
            attributes["spacing_x_m"]
        )

        spacing_y_m = float(
            attributes["spacing_y_m"]
        )

        plant_spacing_m = (
            attributes["plant_spacing_m"]
        )

        trees_per_ha = float(
            attributes["trees_per_ha"]
        )

        estimated_trees = float(
            attributes[
                "estimated_trees_compartment"
            ]
        )

        clone_factor = float(
            combination["clone_factor"].iloc[0]
        )

        # ----------------------------------------------------
        # Future planting year
        # ----------------------------------------------------

        planting_year = (
            self.next_planting_year
        )

        results = []

        # ----------------------------------------------------
        # Predict every age
        # ----------------------------------------------------

        for age_months in TRAJECTORY_AGES:

            inventory_year = (
                planting_year
                + age_months / 12.0
            )

            disease_score = (
                self._get_disease_score(
                    compartment_id=compartment_id,
                    species=species,
                    clone=clone,
                    age_months=age_months,
                )
            )

            model_input = {
                "age_months": float(age_months),
                "inventory_year": float(
                    inventory_year
                ),
                "clone_factor": clone_factor,
                "planting_year": float(
                    planting_year
                ),
                "disease_score": disease_score,
                "trees_per_ha": trees_per_ha,
            }

            prediction = (
                self.model_engine.predict(
                    model_input
                )
            )

            rf = prediction["random_forest"]

            pinns = prediction["pinns"]

            # ------------------------------------------------
            # Random Forest
            # ------------------------------------------------

            rf_volume = float(
                rf["volume_m3_tree"]
            )

            rf_mai_tree = calculate_mai(
                rf_volume,
                age_months,
            )

            rf_volume_ha = (
                rf_volume
                * trees_per_ha
            )

            rf_total_volume = (
                rf_volume
                * estimated_trees
            )

            rf_mai_ha = calculate_stand_mai(
                rf_volume_ha,
                age_months,
            )

            # ------------------------------------------------
            # PINNs
            # ------------------------------------------------

            pinns_volume = float(
                pinns["volume_m3_tree"]
            )

            pinns_mai_tree = calculate_mai(
                pinns_volume,
                age_months,
            )

            pinns_volume_ha = (
                pinns_volume
                * trees_per_ha
            )

            pinns_total_volume = (
                pinns_volume
                * estimated_trees
            )

            pinns_mai_ha = calculate_stand_mai(
                pinns_volume_ha,
                age_months,
            )

            # ------------------------------------------------
            # Row
            # ------------------------------------------------

            row = {
                "regional_id": regional_id,
                "compartment_id": compartment_id,
                "compartment_code": attributes[
                    "compartment_code"
                ],

                "species": species,
                "clone": clone,
                "clone_factor": clone_factor,

                "planting_year": planting_year,
                "inventory_year": inventory_year,

                "growth_period": period_label(
                    age_months
                ),

                "age_months": age_months,

                "disease_score": disease_score,

                "area_ha": area_ha,

                "spacing_m": spacing_m,
                "spacing_x_m": spacing_x_m,
                "spacing_y_m": spacing_y_m,
                "plant_spacing_m": plant_spacing_m,

                "trees_per_ha": trees_per_ha,

                "estimated_trees_compartment":
                    estimated_trees,

                # --------------------------------------------
                # RF
                # --------------------------------------------

                "rf_dbh_cm": float(
                    rf["dbh_cm"]
                ),

                "rf_height_m": float(
                    rf["height_m"]
                ),

                "rf_volume_m3_tree":
                    rf_volume,

                "rf_tha_m": float(
                    rf["tha_m"]
                ),

                "rf_mai_m3_tree_year":
                    rf_mai_tree,

                "rf_volume_m3_ha":
                    rf_volume_ha,

                "rf_total_volume_m3":
                    rf_total_volume,

                "rf_mai_m3_ha_year":
                    rf_mai_ha,

                # --------------------------------------------
                # PINNs
                # --------------------------------------------

                "pinns_dbh_cm": float(
                    pinns["dbh_cm"]
                ),

                "pinns_height_m": float(
                    pinns["height_m"]
                ),

                "pinns_volume_m3_tree":
                    pinns_volume,

                "pinns_mai_m3_tree_year":
                    pinns_mai_tree,

                "pinns_volume_m3_ha":
                    pinns_volume_ha,

                "pinns_total_volume_m3":
                    pinns_total_volume,

                "pinns_mai_m3_ha_year":
                    pinns_mai_ha,

                # --------------------------------------------
                # Comparison
                # --------------------------------------------

                "difference_volume_rf_minus_pinns":
                    rf_volume - pinns_volume,

                "difference_dbh_rf_minus_pinns":
                    float(rf["dbh_cm"])
                    - float(pinns["dbh_cm"]),

                "difference_height_rf_minus_pinns":
                    float(rf["height_m"])
                    - float(pinns["height_m"]),
            }

            results.append(row)

        return pd.DataFrame(results)

    # ========================================================
    # COMPARTMENT TRAJECTORY
    # ========================================================

    def predict_compartment(
        self,
        regional_id,
        compartment_id,
        species=None,
        clone=None,
    ):

        compartment_rows = self.df[
            (
                self.df["regional_id"]
                == regional_id
            )
            &
            (
                self.df["compartment_id"]
                == compartment_id
            )
        ].copy()

        if compartment_rows.empty:
            raise ValueError(
                "Compartment not found."
            )

        # ----------------------------------------------------
        # Determine combinations
        # ----------------------------------------------------

        combinations = (
            compartment_rows[
                ["species", "clone"]
            ]
            .drop_duplicates()
            .sort_values(
                ["species", "clone"]
            )
        )

        if species is not None:
            combinations = combinations[
                combinations["species"]
                == species
            ]

        if clone is not None:
            combinations = combinations[
                combinations["clone"]
                == clone
            ]

        if combinations.empty:
            raise ValueError(
                "No species/clone combination "
                "matches the requested filters."
            )

        # ----------------------------------------------------
        # Predict each combination
        # ----------------------------------------------------

        combination_results = []

        for _, combo in combinations.iterrows():

            result = self.predict_combination(
                regional_id=regional_id,
                compartment_id=compartment_id,
                species=combo["species"],
                clone=combo["clone"],
            )

            combination_results.append(result)

        detailed = pd.concat(
            combination_results,
            ignore_index=True,
        )

        # ----------------------------------------------------
        # Aggregate compartment average
        # ----------------------------------------------------

        numeric_prediction_columns = [
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

        group_columns = [
            "regional_id",
            "compartment_id",
            "compartment_code",
            "planting_year",
            "inventory_year",
            "growth_period",
            "age_months",
            "area_ha",
            "spacing_m",
            "spacing_x_m",
            "spacing_y_m",
            "plant_spacing_m",
            "trees_per_ha",
            "estimated_trees_compartment",
        ]

        trajectory = (
            detailed
            .groupby(
                group_columns,
                as_index=False,
            )[numeric_prediction_columns]
            .mean()
        )

        # ----------------------------------------------------
        # Add metadata
        # ----------------------------------------------------

        trajectory["species_filter"] = (
            species if species is not None
            else "All"
        )

        trajectory["clone_filter"] = (
            clone if clone is not None
            else "All"
        )

        trajectory["species_clone_count"] = (
            detailed.groupby(
                [
                    "growth_period",
                    "age_months",
                ]
            )["clone"].nunique()
            .values
        )

        return detailed, trajectory


# ============================================================
# SMOKE TEST
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    engine = CompartmentTrajectoryEngine(
        DATA_PATH
    )

    # --------------------------------------------------------
    # Smoke test
    #
    # First compartment:
    # R01 / 101
    #
    # No species/clone filter:
    # therefore average across all
    # available species-clone combinations.
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("SMOKE TEST")
    print("=" * 75)

    detailed, trajectory = (
        engine.predict_compartment(
            regional_id="R01",
            compartment_id=101,
            species=None,
            clone=None,
        )
    )

    print(
        f"\nDetailed rows : {len(detailed):,}"
    )

    print(
        f"Trajectory rows : {len(trajectory):,}"
    )

    print("\nTrajectory preview:")

    preview_columns = [
        "regional_id",
        "compartment_id",
        "growth_period",
        "age_months",
        "rf_dbh_cm",
        "rf_height_m",
        "rf_volume_m3_tree",
        "rf_tha_m",
        "rf_mai_m3_tree_year",
        "rf_volume_m3_ha",
        "rf_total_volume_m3",
        "pinns_dbh_cm",
        "pinns_height_m",
        "pinns_volume_m3_tree",
        "pinns_mai_m3_tree_year",
        "pinns_volume_m3_ha",
        "pinns_total_volume_m3",
    ]

    print(
        trajectory[
            preview_columns
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    trajectory.to_csv(
        ENGINE_OUTPUT,
        index=False,
    )

    config = {
        "historical_planting_years":
            engine.planting_years,

        "median_planting_interval_years":
            engine.median_planting_interval,

        "next_planting_year":
            engine.next_planting_year,

        "model_features":
            MODEL_FEATURES,

        "targets":
            TARGETS,

        "trajectory_age_months":
            TRAJECTORY_AGES,

        "growth_period_rule": {
            "P6_to_P66":
                "6-month intervals",

            "PHI":
                "age >= 72 months",
        },

        "mai_tree_definition":
            "predicted_volume_m3_tree / age_years",

        "mai_tree_unit":
            "m3/tree/year",

        "mai_stand_definition":
            "predicted_volume_m3_ha / age_years",

        "mai_stand_unit":
            "m3/ha/year",

        "next_cohort_method":
            "last historical planting year + median historical planting interval",

        "disease_method":
            "historical disease pattern with hierarchical fallback",

        "tha_method":
            "Random Forest THA model",

        "note":
            "Trajectory uses the existing RF and PINNs models. "
            "No new growth model is trained in this step.",
    }

    with open(
        ENGINE_CONFIG,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    print("\nSaved:")
    print(
        ENGINE_OUTPUT
    )

    print(
        ENGINE_CONFIG
    )

    print("\n" + "=" * 75)
    print("STEP 18B COMPLETED")
    print("=" * 75)

    print(
        "\nNo model training was performed."
    )


if __name__ == "__main__":
    main()