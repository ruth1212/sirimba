from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DASHBOARD_DATASET = (
    PROJECT_ROOT
    / "results"
    / "dashboard_growth_trajectory.csv"
)

ACTUAL_DATASET = (
    PROJECT_ROOT
    / "data"
    / "dataset.csv"
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Tree Growth Analytics API",
    description=(
        "Dashboard API for Random Forest and PINNs "
        "tree growth trajectory"
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL DATA
# ============================================================

dashboard_df: Optional[pd.DataFrame] = None
actual_df: Optional[pd.DataFrame] = None


# ============================================================
# LOAD DATASET
# ============================================================

def load_dashboard_dataset() -> pd.DataFrame:
    global dashboard_df

    if dashboard_df is None:

        if not DASHBOARD_DATASET.exists():
            raise FileNotFoundError(
                f"Dashboard dataset not found: {DASHBOARD_DATASET}"
            )

        dashboard_df = pd.read_csv(
            DASHBOARD_DATASET
        )

    return dashboard_df


def load_actual_dataset() -> pd.DataFrame:
    global actual_df

    if actual_df is None:

        if not ACTUAL_DATASET.exists():
            raise FileNotFoundError(
                f"Actual dataset not found: {ACTUAL_DATASET}"
            )

        actual_df = pd.read_csv(
            ACTUAL_DATASET
        )

    return actual_df

# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    df = load_dashboard_dataset()

    print("=" * 70)
    print("TREE GROWTH ANALYTICS API")
    print("=" * 70)

    print(
        f"Dashboard dataset : {DASHBOARD_DATASET}"
    )

    print(
        f"Rows              : {len(df):,}"
    )

    print(
        f"Columns           : {len(df.columns)}"
    )

    print(
        f"Compartments      : {df['compartment_id'].nunique()}"
    )

    print(
        f"Species           : {df['species'].nunique()}"
    )

    print(
        f"Clones            : {df['clone'].nunique()}"
    )

    print("=" * 70)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "success": True,
        "service": "Tree Growth Analytics API",
        "version": "1.0.0",
        "dashboard_dataset": (
            "dashboard_growth_trajectory.csv"
        ),
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    df = load_dashboard_dataset()

    return {
        "success": True,
        "status": "healthy",
        "rows": len(df),
        "compartments": int(
            df["compartment_id"].nunique()
        ),
        "species": int(
            df["species"].nunique()
        ),
        "clones": int(
            df["clone"].nunique()
        ),
    }


# ============================================================
# DASHBOARD FILTERS
# ============================================================

@app.get("/dashboard/filters")
def dashboard_filters():

    df = load_dashboard_dataset()

    regional = (
        df[["regional_id"]]
        .drop_duplicates()
        .sort_values("regional_id")
    )

    compartments = (
        df[
            [
                "regional_id",
                "compartment_id",
                "compartment_code",
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

    species = sorted(
        df["species"]
        .dropna()
        .unique()
        .tolist()
    )

    clones = sorted(
        df["clone"]
        .dropna()
        .unique()
        .tolist()
    )

    return {

        "success": True,

        "filters": {

            "regional": [
                {
                    "regional_id": row[
                        "regional_id"
                    ]
                }
                for _, row in regional.iterrows()
            ],

            "compartment": [
                {
                    "regional_id": row[
                        "regional_id"
                    ],

                    "compartment_id": int(
                        row["compartment_id"]
                    ),

                    "compartment_code": row[
                        "compartment_code"
                    ],
                }

                for _, row in compartments.iterrows()
            ],

            "species": species,

            "clone": clones,
        },

        "ui_filter_rules": {

            "regional": True,

            "compartment": True,

            "species": True,

            "clone": True,

            "planting_year": False,
        },
    }


# ============================================================
# DASHBOARD TRAJECTORY
# ============================================================

@app.get("/dashboard/trajectory")
def dashboard_trajectory(

    regional_id: Optional[str] = Query(
        default=None
    ),

    compartment_id: Optional[int] = Query(
        default=None
    ),

    species: Optional[str] = Query(
        default=None
    ),

    clone: Optional[str] = Query(
        default=None
    ),

    model: str = Query(
        default="random_forest"
    ),
):

    df = load_dashboard_dataset().copy()

    model = model.lower().strip()

    # --------------------------------------------------------
    # VALID MODEL
    # --------------------------------------------------------

    if model not in [
        "random_forest",
        "pinns",
    ]:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid model. "
                "Use 'random_forest' or 'pinns'."
            ),
        )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    if (
        regional_id is not None
        and regional_id != "All"
    ):

        df = df[
            df["regional_id"]
            == regional_id
        ]

    if compartment_id is not None:

        df = df[
            df["compartment_id"]
            == compartment_id
        ]

    if (
        species is not None
        and species != "All"
    ):

        df = df[
            df["species"]
            == species
        ]

    if (
        clone is not None
        and clone != "All"
    ):

        df = df[
            df["clone"]
            == clone
        ]

    if df.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                "No trajectory data found "
                "for the selected filters."
            ),
        )

    # --------------------------------------------------------
    # COMMON COLUMNS
    # --------------------------------------------------------

    common_columns = [
        "age_months",
        "growth_period",
        "regional_id",
        "compartment_id",
        "compartment_code",
        "species",
        "clone",
    ]

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    if model == "random_forest":

        model_columns = [

            "rf_dbh_cm",

            "rf_height_m",

            "rf_volume_m3_tree",

            "rf_tha_m",

            "rf_mai_m3_tree_year",

            "rf_volume_m3_ha",

            "rf_total_volume_m3",

            "rf_mai_m3_ha_year",
        ]

        rename_map = {

            "rf_dbh_cm":
                "dbh_cm",

            "rf_height_m":
                "height_m",

            "rf_volume_m3_tree":
                "volume_m3_tree",

            "rf_tha_m":
                "tha_m",

            "rf_mai_m3_tree_year":
                "mai_m3_tree_year",

            "rf_volume_m3_ha":
                "volume_m3_ha",

            "rf_total_volume_m3":
                "total_volume_m3",

            "rf_mai_m3_ha_year":
                "stand_mai_m3_ha_year",
        }

    # --------------------------------------------------------
    # PINNs
    # --------------------------------------------------------

    else:

        model_columns = [

            "pinns_dbh_cm",

            "pinns_height_m",

            "pinns_volume_m3_tree",

            "pinns_mai_m3_tree_year",

            "pinns_volume_m3_ha",

            "pinns_total_volume_m3",

            "pinns_mai_m3_ha_year",
        ]

        rename_map = {

            "pinns_dbh_cm":
                "dbh_cm",

            "pinns_height_m":
                "height_m",

            "pinns_volume_m3_tree":
                "volume_m3_tree",

            "pinns_mai_m3_tree_year":
                "mai_m3_tree_year",

            "pinns_volume_m3_ha":
                "volume_m3_ha",

            "pinns_total_volume_m3":
                "total_volume_m3",

            "pinns_mai_m3_ha_year":
                "stand_mai_m3_ha_year",
        }

    # --------------------------------------------------------
    # COLUMN CHECK
    # --------------------------------------------------------

    required_columns = (
        common_columns
        + model_columns
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Required dashboard "
                    "columns are missing."
                ),
                "missing_columns": missing,
            },
        )

    # --------------------------------------------------------
    # BUILD RESULT
    # --------------------------------------------------------

    result = (
        df[required_columns]
        .rename(columns=rename_map)
        .sort_values(
            [
                "age_months",
                "species",
                "clone",
            ]
        )
    )

    return {

        "success": True,

        "model": model,

        "prediction_cohort": 2025,

        "filters": {

            "regional_id":
                regional_id or "All",

            "compartment_id":
                compartment_id or "All",

            "species":
                species or "All",

            "clone":
                clone or "All",
        },

        "trajectory":
            result.to_dict(
                orient="records"
            ),

        "row_count":
            len(result),
    }

# =======================================================
# DASHBOARD ACTUAL TRAJECTORY
# =======================================================

@app.get("/dashboard/actual-trajectory")
def dashboard_actual_trajectory(
    regional_id: Optional[str] = Query(
        default=None
    ),

    compartment_id: Optional[int] = Query(
        default=None
    ),

    species: Optional[str] = Query(
        default=None
    ),

    clone: Optional[str] = Query(
        default=None
    ),
):
    df = load_actual_dataset().copy()

    # ---------------------------------------------------
    # VALIDATE REQUIRED COLUMNS
    # ---------------------------------------------------

    required_columns = [
        "regional_id",
        "compartment_id",
        "species",
        "clone",
        "age_months",
        "growth_period",
        "dbh_cm",
        "height_m",
        "volume_m3_tree",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Required actual dataset "
                    "columns are missing."
                ),
                "missing_columns": missing_columns,
            },
        )

    # ---------------------------------------------------
    # FILTER
    # ---------------------------------------------------

    if regional_id is not None:
        df = df[
            df["regional_id"] == regional_id
        ]

    if compartment_id is not None:
        df = df[
            df["compartment_id"] == compartment_id
        ]

    if species is not None:
        df = df[
            df["species"] == species
        ]

    if clone is not None:
        df = df[
            df["clone"] == clone
        ]

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "No actual data found "
                "for the selected filters."
            ),
        )

    # ---------------------------------------------------
    # DASHBOARD DISPLAY PERIODS
    #
    # P66 is intentionally excluded.
    # PHI represents observations classified as PHI.
    # ---------------------------------------------------

    display_periods = [
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
        "P66",
    ]

    df = df[
        df["growth_period"].isin(
            display_periods
        )
    ].copy()

    # ---------------------------------------------------
    # AGGREGATE ACTUAL DATA
    # ---------------------------------------------------

    result = (
        df
        .groupby(
            "growth_period",
            as_index=False
        )
        .agg(
            age_months_mean=(
                "age_months",
                "mean"
            ),

            dbh_cm=(
                "dbh_cm",
                "mean"
            ),

            height_m=(
                "height_m",
                "mean"
            ),

            volume_m3_tree=(
                "volume_m3_tree",
                "mean"
            ),

            actual_rows=(
                "age_months",
                "size"
            ),
        )
    )

    # ---------------------------------------------------
    # SORT IN DASHBOARD ORDER
    # ---------------------------------------------------

    period_order = {
        period: index
        for index, period
        in enumerate(display_periods)
    }

    result["period_order"] = (
        result["growth_period"]
        .map(period_order)
    )

    result = (
        result
        .sort_values("period_order")
        .drop(columns=["period_order"])
    )

    # ---------------------------------------------------
    # RETURN
    # ---------------------------------------------------

    return {
        "success": True,
        "source": "historical_actual",
        "filters": {
            "regional_id": regional_id,
            "compartment_id": compartment_id,
            "species": (
                species
                if species is not None
                else "All"
            ),
            "clone": (
                clone
                if clone is not None
                else "All"
            ),
        },
        "periods": result.to_dict(
            orient="records"
        ),
    }


# ============================================================
# DASHBOARD SUMMARY
# ============================================================

@app.get("/dashboard/summary")
def dashboard_summary(

    regional_id: Optional[str] = Query(
        default=None
    ),

    compartment_id: Optional[int] = Query(
        default=None
    ),

    species: Optional[str] = Query(
        default=None
    ),

    clone: Optional[str] = Query(
        default=None
    ),

    model: str = Query(
        default="random_forest"
    ),
):

    df = load_dashboard_dataset().copy()

    model = model.lower().strip()

    if model not in [
        "random_forest",
        "pinns",
    ]:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid model. "
                "Use 'random_forest' or 'pinns'."
            ),
        )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    if (
        regional_id is not None
        and regional_id != "All"
    ):

        df = df[
            df["regional_id"]
            == regional_id
        ]

    if compartment_id is not None:

        df = df[
            df["compartment_id"]
            == compartment_id
        ]

    if (
        species is not None
        and species != "All"
    ):

        df = df[
            df["species"]
            == species
        ]

    if (
        clone is not None
        and clone != "All"
    ):

        df = df[
            df["clone"]
            == clone
        ]

    if df.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                "No summary data found "
                "for the selected filters."
            ),
        )

    # --------------------------------------------------------
    # PHI = LATEST AGE AVAILABLE
    # --------------------------------------------------------

    phi = (
        df
        .sort_values("age_months")
        .groupby(
            [
                "regional_id",
                "compartment_id",
                "species",
                "clone",
            ],
            as_index=False,
        )
        .tail(1)
    )

    # --------------------------------------------------------
    # RANDOM FOREST SUMMARY
    # --------------------------------------------------------

    if model == "random_forest":

        summary = {

            "average_dbh_cm":
                float(
                    phi[
                        "rf_dbh_cm"
                    ].mean()
                ),

            "average_height_m":
                float(
                    phi[
                        "rf_height_m"
                    ].mean()
                ),

            "average_volume_m3_tree":
                float(
                    phi[
                        "rf_volume_m3_tree"
                    ].mean()
                ),

            "average_tha_m":
                float(
                    phi[
                        "rf_tha_m"
                    ].mean()
                ),

            "average_mai_m3_tree_year":
                float(
                    phi[
                        "rf_mai_m3_tree_year"
                    ].mean()
                ),

            "average_volume_m3_ha":
                float(
                    phi[
                        "rf_volume_m3_ha"
                    ].mean()
                ),

            "average_total_volume_m3":
                float(
                    phi[
                        "rf_total_volume_m3"
                    ].mean()
                ),

            "average_stand_mai_m3_ha_year":
                float(
                    phi[
                        "rf_mai_m3_ha_year"
                    ].mean()
                ),
        }

    # --------------------------------------------------------
    # PINNs SUMMARY
    # --------------------------------------------------------

    else:

        summary = {

            "average_dbh_cm":
                float(
                    phi[
                        "pinns_dbh_cm"
                    ].mean()
                ),

            "average_height_m":
                float(
                    phi[
                        "pinns_height_m"
                    ].mean()
                ),

            "average_volume_m3_tree":
                float(
                    phi[
                        "pinns_volume_m3_tree"
                    ].mean()
                ),

            "average_mai_m3_tree_year":
                float(
                    phi[
                        "pinns_mai_m3_tree_year"
                    ].mean()
                ),

            "average_volume_m3_ha":
                float(
                    phi[
                        "pinns_volume_m3_ha"
                    ].mean()
                ),

            "average_total_volume_m3":
                float(
                    phi[
                        "pinns_total_volume_m3"
                    ].mean()
                ),

            "average_stand_mai_m3_ha_year":
                float(
                    phi[
                        "pinns_mai_m3_ha_year"
                    ].mean()
                ),
        }

        summary["tha_note"] = (
            "THA is available from the "
            "auxiliary Random Forest THA model. "
            "PINNs does not directly predict THA."
        )

    # --------------------------------------------------------
    # COMPARTMENT ATTRIBUTES
    # --------------------------------------------------------

    attributes = {

    "area_ha":
        float(
            df["area_ha"].iloc[0]
        ),

    "spacing_m":
        str(
            df["spacing_m"].iloc[0]
        ),

        "spacing_x_m":
            str(
                df["spacing_x_m"].iloc[0]
            ),

        "spacing_y_m":
            str(
                df["spacing_y_m"].iloc[0]
            ),

        "plant_spacing_m":
            str(
                df["plant_spacing_m"].iloc[0]
            ),

        "trees_per_ha":
            float(
                df["trees_per_ha"].iloc[0]
            ),

        "estimated_trees_compartment":
            int(
                df[
                    "estimated_trees_compartment"
                ].iloc[0]
            ),
    }

    return {

        "success": True,

        "model": model,

        "prediction_cohort": 2025,

        "filters": {

            "regional_id":
                regional_id or "All",

            "compartment_id":
                compartment_id or "All",

            "species":
                species or "All",

            "clone":
                clone or "All",
        },

        "attributes":
            attributes,

        "summary":
            summary,
    }


# ============================================================
# PHI SPECIES / CLONE COMPARISON
# ============================================================

@app.get("/dashboard/phi-comparison")
def phi_comparison(

    regional_id: Optional[str] = Query(
        default=None
    ),

    compartment_id: Optional[int] = Query(
        default=None
    ),

    model: str = Query(
        default="random_forest"
    ),
):

    df = load_dashboard_dataset().copy()

    model = model.lower().strip()

    if model not in [
        "random_forest",
        "pinns",
    ]:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid model. "
                "Use 'random_forest' or 'pinns'."
            ),
        )

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    if (
        regional_id is not None
        and regional_id != "All"
    ):

        df = df[
            df["regional_id"]
            == regional_id
        ]

    if compartment_id is not None:

        df = df[
            df["compartment_id"]
            == compartment_id
        ]

    if df.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                "No PHI data found "
                "for the selected filters."
            ),
        )

    # --------------------------------------------------------
    # LATEST PHI RECORD
    # --------------------------------------------------------

    phi = (
        df
        .sort_values("age_months")
        .groupby(
            [
                "species",
                "clone",
            ],
            as_index=False,
        )
        .tail(1)
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    if model == "random_forest":

        result = phi[
            [
                "species",
                "clone",
                "age_months",
                "rf_dbh_cm",
                "rf_height_m",
                "rf_volume_m3_tree",
                "rf_tha_m",
                "rf_mai_m3_tree_year",
                "rf_volume_m3_ha",
                "rf_total_volume_m3",
                "rf_mai_m3_ha_year",
            ]
        ].rename(
            columns={

                "rf_dbh_cm":
                    "dbh_cm",

                "rf_height_m":
                    "height_m",

                "rf_volume_m3_tree":
                    "volume_m3_tree",

                "rf_tha_m":
                    "tha_m",

                "rf_mai_m3_tree_year":
                    "mai_m3_tree_year",

                "rf_volume_m3_ha":
                    "volume_m3_ha",

                "rf_total_volume_m3":
                    "total_volume_m3",

                "rf_mai_m3_ha_year":
                    "stand_mai_m3_ha_year",
            }
        )

    # --------------------------------------------------------
    # PINNs
    # --------------------------------------------------------

    else:

        result = phi[
            [
                "species",
                "clone",
                "age_months",
                "pinns_dbh_cm",
                "pinns_height_m",
                "pinns_volume_m3_tree",
                "pinns_mai_m3_tree_year",
                "pinns_volume_m3_ha",
                "pinns_total_volume_m3",
                "pinns_mai_m3_ha_year",
            ]
        ].rename(
            columns={

                "pinns_dbh_cm":
                    "dbh_cm",

                "pinns_height_m":
                    "height_m",

                "pinns_volume_m3_tree":
                    "volume_m3_tree",

                "pinns_mai_m3_tree_year":
                    "mai_m3_tree_year",

                "pinns_volume_m3_ha":
                    "volume_m3_ha",

                "pinns_total_volume_m3":
                    "total_volume_m3",

                "pinns_mai_m3_ha_year":
                    "stand_mai_m3_ha_year",
            }
        )

    result = result.sort_values(
        [
            "species",
            "clone",
        ]
    )

    return {

        "success": True,

        "model": model,

        "prediction_cohort": 2025,

        "comparison_basis": (
            "Latest available age "
            "for each species-clone "
            "combination."
        ),

        "rows":
            result.to_dict(
                orient="records"
            ),

        "row_count":
            len(result),
    }