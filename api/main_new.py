from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import onnxruntime as ort
import json
import uuid
import threading
import asyncio
import functools
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from api.report_generator import (
        generate_pdf_report,
        generate_pptx_report,
        generate_mai_pdf_report,
    )
except ModuleNotFoundError:
    from report_generator import (
        generate_pdf_report,
        generate_pptx_report,
        generate_mai_pdf_report,
    )



# ============================================================
try:
    from api.ai_graph import ai_graph
except ModuleNotFoundError:
    from ai_graph import ai_graph

try:
    from api.dataset_standardizer import standardize_dataset
except ModuleNotFoundError:
    from dataset_standardizer import standardize_dataset
# TREE GROWTH ANALYTICS
# FINAL BACKEND API
#
# Dataset:
#   data/dataset_phi66_final.csv
#
# Scope:
#   P6 - P66
#   P66 = PHI
#
# Models:
#   Random Forest
#   PINNs V2
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(Path(__file__).resolve().parent / ".env")

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "dataset_phi66_final.csv"
)

RF_MODEL_DIR = (
    PROJECT_ROOT
    / "results"
    / "random_forest"
    / "models"
)

PINNS_ONNX_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "pinns"
    / "models"
    / "pinns_v2_growth_model.onnx"
)

PREPROCESSING_DIR = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "pipeline"
    / "preprocessing"
)

FEATURE_SCALER_PATH = (
    PREPROCESSING_DIR
    / "feature_scaler.joblib"
)

TARGET_SCALER_PATH = (
    PREPROCESSING_DIR
    / "target_scaler.joblib"
)


# ============================================================
# FINAL MODEL FEATURES
# ============================================================

FEATURES = [
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


# ============================================================
# AGE SCOPE
# ============================================================

DISPLAY_AGES = [
    6,
    12,
    18,
    24,
    30,
    36,
    42,
    48,
    54,
    60,
    66,
]

DISPLAY_PERIODS = {
    6: "P6",
    12: "P12",
    18: "P18",
    24: "P24",
    30: "P30",
    36: "P36",
    42: "P42",
    48: "P48",
    54: "P54",
    60: "P60",
    66: "P66",
}


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Tree Growth Analytics API",
    version="FINAL-PHI66",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ============================================================
# AI ASSISTANT ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Pertanyaan tentang data pertumbuhan pohon",
    )
    history: list[dict[str, str]] = Field(
        default_factory=list,
        max_length=12,
        description="Riwayat percakapan sebelumnya",
    )


# ============================================================
# AI ASSISTANT ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â CSV QUERY ENGINE
# ============================================================


def query_growth_data(question: str) -> dict:
    """
    Mengambil ringkasan faktual dari dataset lokal.
    Tidak menggunakan API AI eksternal.
    """

    if df is None:
        raise HTTPException(
            status_code=503,
            detail="Dataset belum dimuat oleh backend.",
        )

    import re

    q = question.lower().strip()
    data = df
    matched_filters = {
        "regional_name": None,
        "regional_id": None,
        "compartment_code": None,
        "compartment_id": None,
        "species": [],
        "clone": None,
        "period": None,
    }

    def find_match(column: str):
        """Cari nilai kolom yang benar-benar disebutkan dalam pertanyaan."""
        if column not in data.columns:
            return None

        values = (
            df[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        # Nilai terpanjang diperiksa dahulu agar nama yang mirip
        # tidak mendahului nama yang lebih spesifik.
        values.sort(key=len, reverse=True)

        for value in values:
            pattern = (
                r"(?<!\w)"
                + re.escape(value.lower())
                + r"(?!\w)"
            )
            if re.search(pattern, q):
                return value

        return None

    # Filter wilayah berdasarkan nama atau ID.
    # Nama wilayah diperiksa terlebih dahulu.
    region_name = find_match("regional_name")
    if region_name is not None:
        data = data[
            data["regional_name"].astype(str).str.lower()
            == region_name.lower()
        ]
        matched_filters["regional_name"] = region_name

    else:
        region_id = find_match("regional_id")
        if region_id is not None:
            data = data[
                data["regional_id"].astype(str).str.lower()
                == region_id.lower()
            ]
            matched_filters["regional_id"] = region_id


    # --------------------------------------------------------
    # FILTER DAN VALIDASI KOMPARTEMEN
    # Kompartemen yang disebut tetapi tidak tersedia harus ditolak.
    # --------------------------------------------------------

    compartment_code = find_match("compartment_code")
    compartment_id = None

    # Deteksi kode kompartemen yang disebut secara eksplisit,
    # termasuk kode yang tidak terdapat dalam dataset.
    requested_code_match = re.search(
        r"(?<!\w)comp-\d+(?!\w)",
        q,
        re.IGNORECASE,
    )

    if requested_code_match:
        requested_code = requested_code_match.group(0).upper()

        available_codes = (
            df["compartment_code"]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
            .tolist()
        )

        if requested_code not in available_codes:
            matched_filters["compartment_code"] = requested_code

            return {
                "found": False,
                "message": (
                    f"Kompartemen {requested_code} tidak ditemukan "
                    "dalam dataset. Periksa kembali kode kompartemen."
                ),
                "filters": matched_filters,
            }

        compartment_code = next(
            code
            for code in df["compartment_code"].dropna().astype(str).unique()
            if code.upper() == requested_code
        )

    if compartment_code is not None:
        data = data[
            data["compartment_code"].astype(str).str.lower()
            == compartment_code.lower()
        ]
        matched_filters["compartment_code"] = compartment_code

    else:
        compartment_id = find_match("compartment_id")

        if compartment_id is not None:
            data = data[
                data["compartment_id"].astype(str).str.lower()
                == compartment_id.lower()
            ]
            matched_filters["compartment_id"] = compartment_id

    # --------------------------------------------------------
    # FILTER DAN VALIDASI SPESIES
    # Nama spesies yang disebut tetapi tidak tersedia
    # harus ditolak, bukan diabaikan.
    # --------------------------------------------------------

    if "species" in data.columns:
        available_species = (
            df["species"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        # Periksa nama spesies terpanjang terlebih dahulu.
        available_species.sort(key=len, reverse=True)

        for species in available_species:
            pattern = (
                r"(?<!\w)"
                + re.escape(species.lower())
                + r"(?!\w)"
            )

            if re.search(pattern, q):
                matched_filters["species"].append(species)

    
        # --------------------------------------------------------
        # VALIDASI NAMA SPESIES YANG DISEBUTKAN
        # --------------------------------------------------------

        species_keyword_match = re.search(
            r"\bspesies\b",
            q,
            re.IGNORECASE,
        )

        # Jika tidak cocok dengan spesies dataset, periksa apakah
        # pertanyaan menyebut nama yang diawali "Eucalyptus".
        eucalyptus_mention = re.search(
            r"(?<!\w)eucalyptus\b",
            q,
            re.IGNORECASE,
        )

        if (
            (species_keyword_match or eucalyptus_mention)
            and not matched_filters["species"]
        ):
            return {
                "found": False,
                "message": (
                    "Spesies yang disebutkan tidak ditemukan "
                    "dalam dataset. Periksa kembali nama spesies."
                ),
                "filters": matched_filters,
            }

        # Terapkan filter jika spesies valid ditemukan.
        if matched_filters["species"]:
            species_lower = [
                s.lower()
                for s in matched_filters["species"]
            ]

            data = data[
                data["species"]
                .astype(str)
                .str.lower()
                .isin(species_lower)
            ]

    # --------------------------------------------------------
    # FILTER DAN VALIDASI KLON
    # --------------------------------------------------------

    clone = find_match("clone")

    # Deteksi kode klon yang disebutkan secara eksplisit.
    requested_clone_match = re.search(
        r"(?<!\w)(?:clone-\d+|[a-z]{2,5}-\d+)(?!\w)",
        q,
        re.IGNORECASE,
    )

    if requested_clone_match:
        requested_clone = requested_clone_match.group(0).upper()

        available_clones = (
            df["clone"]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
            .tolist()
        )

        # Tolak kode klon yang tidak ada dalam dataset.
        if requested_clone not in available_clones:
            matched_filters["clone"] = requested_clone

            return {
                "found": False,
                "message": (
                    f"Klon {requested_clone} tidak ditemukan "
                    "dalam dataset. Periksa kembali kode klon."
                ),
                "filters": matched_filters,
            }

        # Gunakan ejaan asli yang terdapat dalam dataset.
        clone = next(
            value
            for value in df["clone"].dropna().astype(str).unique()
            if value.upper() == requested_clone
        )

    # Terapkan filter hanya jika klon valid ditemukan.
    if clone is not None:
        data = data[
            data["clone"].astype(str).str.lower()
            == clone.lower()
        ]
        matched_filters["clone"] = clone

    
    # --------------------------------------------------------
    # VALIDASI DAN FILTER PERIODE
    # P6-P66 didukung; PHI merupakan alias untuk P66.
    # Periode tidak valid tidak boleh diabaikan.
    # --------------------------------------------------------

    period_match = re.search(
        r"(?<![a-z0-9])p\d+(?![a-z0-9])",
        q,
    )

    phi_match = re.search(
        r"(?<!\w)phi(?!\w)",
        q,
    )

    requested_period = None

    if period_match:
        requested_period = period_match.group(0).upper()
    elif phi_match:
        requested_period = "P66"

    if requested_period is not None:
        period_to_age = {
            period: age
            for age, period in DISPLAY_PERIODS.items()
        }

        matched_filters["period"] = requested_period

        if requested_period not in period_to_age:
            return {
                "found": False,
                "message": (
                    f"Periode {requested_period} tidak valid "
                    "atau tidak didukung. "
                    "Gunakan periode P6 sampai P66."
                ),
                "filters": matched_filters,
            }

        age = period_to_age[requested_period]

        data = data[
            data["age_months"] == age
        ]

    
    # --------------------------------------------------------
    # VALIDASI DAN FILTER TAHUN
    # Tahun pengukuran menggunakan inventory_year.
    # Tahun tanam menggunakan planting_year.
    # --------------------------------------------------------

    year_match = re.search(
        r"(?<!\d)(?:19|20|21)\d{2}(?!\d)",
        q,
    )

    if year_match:
        requested_year = int(year_match.group(0))

        # Bedakan tahun tanam dari tahun pengukuran.
        planting_year_question = bool(
            re.search(
                r"\btahun\s+tanam\b"
                r"|\btahun\s+penanaman\b"
                r"|\btahun\s+ditanam\b"
                r"|\bditanam\s+(?:pada\s+)?tahun\b"
                r"|\bplanting\s+year\b",
                q,
            )
        )

        year_column = (
            "planting_year"
            if planting_year_question
            else "inventory_year"
        )

        matched_filters[year_column] = requested_year

        if year_column not in df.columns:
            return {
                "found": False,
                "message": (
                    f"Kolom {year_column} tidak tersedia "
                    "dalam dataset."
                ),
                "filters": matched_filters,
            }

        available_years = pd.to_numeric(
            df[year_column],
            errors="coerce",
        )

        if not (available_years == requested_year).any():
            if year_column == "inventory_year":
                message = (
                    f"Data historis tahun {requested_year} "
                    "tidak tersedia. Dataset hanya mencakup "
                    "tahun pengukuran 2000ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“2024."
                )
            else:
                message = (
                    f"Data tahun tanam {requested_year} "
                    "tidak tersedia. Dataset hanya memiliki "
                    "tahun tanam 2000, 2005, 2010, 2015, dan 2020."
                )

            return {
                "found": False,
                "message": message,
                "filters": matched_filters,
            }

        data_years = pd.to_numeric(
            data[year_column],
            errors="coerce",
        )

        data = data[data_years == requested_year]


    numeric_columns = [
        col
        for col in [
            "disease_score",
            "condition_score",
            "dbh_cm",
            "height_m",
            "volume_m3_tree",
            "trees_per_ha",
        ]
        if col in data.columns
    ]

  
    summary = {}
    for col in numeric_columns:
        values = data[col].dropna()

        if not values.empty:
            summary[col] = {
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
                "count": int(values.count()),
            }

    return {
        "found": True,
        "row_count": int(len(data)),
        "filters": matched_filters,
        "statistics": summary,
    }


# ============================================================
# GLOBAL OBJECTS
# ============================================================

df = None

rf_dbh = None
rf_height = None
rf_volume = None

pinns_model = None
feature_scaler = None
target_scaler = None


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    global df

    dataset_path = DATASET_PATH

    # --------------------------------------------------------
    # LOAD ACTIVE DATASET FROM REGISTRY
    # --------------------------------------------------------

    if DATASET_REGISTRY_PATH.exists():

        with DATASET_REGISTRY_PATH.open(
            "r",
            encoding="utf-8"
        ) as registry_file:

            registry = json.load(registry_file)

        active_dataset = next(
            (
                item for item in registry
                if isinstance(item, dict)
                and item.get("dataset_activated") is True
                and item.get("status") == "active"
            ),
            None
        )

        if active_dataset:

            relative_path = active_dataset.get("dataset_path")

            if not relative_path:
                raise RuntimeError(
                    "Dataset aktif Registry tidak memiliki dataset_path."
                )

            dataset_path = (
                PROJECT_ROOT / relative_path
            ).resolve()

            if not dataset_path.is_file():
                raise FileNotFoundError(
                    f"Dataset aktif Registry tidak ditemukan: {dataset_path}"
                )

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {dataset_path}"
        )

    # --------------------------------------------------------
    # READ DATASET BASED ON FILE EXTENSION
    # --------------------------------------------------------

    extension = dataset_path.suffix.lower()

    try:
        if extension == ".csv":
            try:
                uploaded_df = pd.read_csv(
                    dataset_path,
                    encoding="utf-8-sig",
                )
            except UnicodeDecodeError:
                uploaded_df = pd.read_csv(
                    dataset_path,
                    encoding="latin1",
                )

        elif extension in {".xlsx", ".xls"}:
            uploaded_df = pd.read_excel(dataset_path)

        else:
            raise RuntimeError(
                f"Format dataset tidak didukung: {extension}"
            )

    except Exception as exc:
        raise RuntimeError(
            f"Gagal membaca dataset aktif: {exc}"
        ) from exc

    if uploaded_df.empty:
        raise RuntimeError("Dataset aktif tidak memiliki data.")

    # --------------------------------------------------------
    # STANDARDIZE AND VALIDATE ACTIVE DATASET
    # --------------------------------------------------------

    df, startup_validation_report = standardize_dataset(
        uploaded_df,
        required_columns=UPLOAD_REQUIRED_COLUMNS,
    )

    if not startup_validation_report["valid"]:
        raise RuntimeError(
            "Dataset aktif gagal validasi standardisasi SIRIMBA: "
            + str(startup_validation_report)
        )

    required_columns = [
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
        "condition_score",
        "dbh_cm",
        "height_m",
        "volume_m3_tree",
        "disease_score",
        "disease_status",
        "tha_m",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Dataset is missing columns: "
            + ", ".join(missing)
        )

    df = df.sort_values(
        [
            "regional_id",
            "compartment_id",
            "species",
            "clone",
            "planting_year",
            "age_months",
        ]
    ).reset_index(drop=True)

    print("=" * 70)
    print("TREE GROWTH ANALYTICS - FINAL BACKEND")
    print("=" * 70)

    print(f"Dataset : {dataset_path}")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")
    print(
        "Dashboard compatible : "
        f"{startup_validation_report['dashboard_compatible']}"
    )
    print(
        "Unsupported ages preserved : "
        f"{not startup_validation_report['dashboard_compatible']}"
    )

    print(
        "Age     : "
        f"{df['age_months'].min()} "
        "to "
        f"{df['age_months'].max()} months"
    )

    print("PHI     : P66 / 66 months")
    print("=" * 70)


# LOAD RANDOM FOREST
# ============================================================

def load_random_forest():

    global rf_dbh
    global rf_height
    global rf_volume

    dbh_path = (
        RF_MODEL_DIR
        / "RF_DBH.joblib"
    )

    height_path = (
        RF_MODEL_DIR
        / "RF_HEIGHT.joblib"
    )

    volume_path = (
        RF_MODEL_DIR
        / "RF_VOLUME.joblib"
    )

    for path in [
        dbh_path,
        height_path,
        volume_path,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Random Forest model not found: {path}"
            )

    rf_dbh = joblib.load(
        dbh_path
    )

    rf_height = joblib.load(
        height_path
    )

    rf_volume = joblib.load(
        volume_path
    )

    print(
        "Random Forest models : OK"
    )


# ============================================================
# LOAD PINNS
# ============================================================

def load_pinns():

    global pinns_model
    global feature_scaler
    global target_scaler

    if not PINNS_ONNX_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"PINNs ONNX model not found: "
            f"{PINNS_ONNX_MODEL_PATH}"
        )

    if not FEATURE_SCALER_PATH.exists():

        raise FileNotFoundError(
            f"Feature scaler not found: "
            f"{FEATURE_SCALER_PATH}"
        )

    if not TARGET_SCALER_PATH.exists():

        raise FileNotFoundError(
            f"Target scaler not found: "
            f"{TARGET_SCALER_PATH}"
        )

    feature_scaler = joblib.load(
        FEATURE_SCALER_PATH
    )

    target_scaler = joblib.load(
        TARGET_SCALER_PATH
    )

    pinns_model = ort.InferenceSession(
        str(PINNS_ONNX_MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    print(
        "PINNs ONNX model : OK"
    )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup():

    load_dataset()

    load_random_forest()

    load_pinns()

    print(
        "FINAL BACKEND STARTUP : SUCCESS"
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "success": True,
        "project": "TREE GROWTH ANALYTICS",
        "status": "FINAL_MODEL_EVALUATION",
        "dataset": "dataset_phi66_final.csv",
        "age_min_months": 6,
        "age_max_months": 66,
        "phi_age_months": 66,
        "volume_prediction_start_months": 18,
        "models": [
            "random_forest",
            "pinns",
        ],
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "success": True,
        "status": "healthy",
        "dataset_loaded": df is not None,
        "random_forest_loaded": (
            rf_dbh is not None
            and rf_height is not None
            and rf_volume is not None
        ),
        "pinns_loaded": (
            pinns_model is not None
        ),
        "dataset_rows": (
            len(df)
            if df is not None
            else 0
        ),
        "phi": "P66",
    }


# ============================================================
# FILTERS
# ============================================================

@app.get("/dashboard/filters")
def dashboard_filters():

    if df is None:

        raise HTTPException(
            status_code=500,
            detail="Dataset not loaded.",
        )

    regional = (
        df[
            [
                "regional_id"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "regional_id"
        )
        .to_dict(
            orient="records"
        )
    )

    compartment = (
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
        .to_dict(
            orient="records"
        )
    )

    species = sorted(
        df[
            "species"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    clone = sorted(
        df[
            "clone"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return {
        "success": True,
        "filters": {
            "regional": regional,
            "compartment": compartment,
            "species": species,
            "clone": clone,
        },
        "ui_filter_rules": {
            "planting_year_filter": False,
            "age_scope": "P6-P66",
            "phi": "P66",
        },
    }


# ============================================================
# APPLY FILTERS
# ============================================================

def apply_filters(
    source_df,
    regional_id: Optional[str] = None,
    compartment_id: Optional[int] = None,
    species: Optional[str] = None,
    clone: Optional[str] = None,
):

    result = source_df

    if regional_id is not None:

        result = result[
            result[
                "regional_id"
            ].astype(str)
            == str(regional_id)
        ]

    if compartment_id is not None:

        result = result[
            result[
                "compartment_id"
            ].astype(int)
            == int(compartment_id)
        ]

    if species is not None:

        result = result[
            result[
                "species"
            ].astype(str)
            == str(species)
        ]

    if clone is not None:

        result = result[
            result[
                "clone"
            ].astype(str)
            == str(clone)
        ]

    return result.copy()


# ============================================================
# PREDICT RANDOM FOREST
# ============================================================

def predict_rf(
    rows: pd.DataFrame,
):

    X = rows[
        FEATURES
    ]

    dbh_prediction = (
        rf_dbh.predict(X)
    )

    height_prediction = (
        rf_height.predict(X)
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # RF volume was trained only on P18-P66.
    # --------------------------------------------------------

    volume_prediction = (
        np.full(
            len(rows),
            np.nan,
            dtype=float,
        )
    )

    volume_mask = (
        rows[
            "age_months"
        ]
        >= 18
    )

    if volume_mask.any():

        volume_prediction[
            volume_mask.to_numpy()
        ] = rf_volume.predict(
            rows.loc[
                volume_mask,
                FEATURES,
            ]
        )

    return (
        dbh_prediction,
        height_prediction,
        volume_prediction,
    )


# ============================================================
# PREDICT PINNS
# ============================================================

def predict_pinns(
    rows: pd.DataFrame,
):
    X = rows[FEATURES].astype(np.float32)

    X_scaled = feature_scaler.transform(X)
    X_scaled = np.asarray(X_scaled, dtype=np.float32)

    predictions_scaled = pinns_model.run(
        ["predictions_scaled"],
        {
            "features_scaled": X_scaled,
        },
    )[0]

    predictions_raw = np.zeros_like(
        predictions_scaled,
        dtype=float,
    )

    for index, target in enumerate(TARGETS):
        scaler = target_scaler[target]
        predictions_raw[:, index] = (
            scaler.inverse_transform(
                predictions_scaled[:, index].reshape(-1, 1)
            ).reshape(-1)
        )

    dbh_prediction = predictions_raw[:, 0]
    height_prediction = predictions_raw[:, 1]

    volume_prediction = np.full(
        len(rows),
        np.nan,
        dtype=float,
    )

    volume_mask = rows["age_months"] >= 18

    if volume_mask.any():
        volume_raw = predictions_raw[
            volume_mask.to_numpy(),
            2,
        ]
        volume_prediction[
            volume_mask.to_numpy()
        ] = np.maximum(volume_raw, 0.0)

    return (
        dbh_prediction,
        height_prediction,
        volume_prediction,
    )


# ============================================================
# BUILD TRAJECTORY
# ============================================================


def build_trajectory(
    rows: pd.DataFrame,
    model: str,
):
    if rows.empty:
        return []

    rows = rows.sort_values(
        ["age_months"]
    ).copy()

    if model == "random_forest":
        (
            dbh_prediction,
            height_prediction,
            volume_prediction,
        ) = predict_rf(rows)

    elif model == "pinns":
        (
            dbh_prediction,
            height_prediction,
            volume_prediction,
        ) = predict_pinns(rows)

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "model must be "
                "'random_forest' or 'pinns'."
            ),
        )

    rows["pred_dbh"] = dbh_prediction
    rows["pred_height"] = height_prediction
    rows["pred_volume"] = volume_prediction

    # Identitas menggambarkan seluruh cakupan data,
    # bukan hanya baris pertama pada setiap usia.
    def scope_label(column, all_label):
        values = (
            rows[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if len(values) == 1:
            return values[0]

        return all_label

    regional_label = scope_label(
        "regional_id", "ALL_REGIONS"
    )
    compartment_label = scope_label(
        "compartment_code", "ALL_COMPARTMENTS"
    )
    species_label = scope_label(
        "species", "ALL_SPECIES"
    )
    clone_label = scope_label(
        "clone", "ALL_CLONES"
    )

    # ID numerik hanya diberikan jika cakupan
    # benar-benar memiliki satu compartment_id.
    compartment_ids = (
        rows["compartment_id"]
        .dropna()
        .unique()
        .tolist()
    )

    compartment_id_label = (
        int(compartment_ids[0])
        if len(compartment_ids) == 1
        else None
    )

    trajectory = []

    for age in DISPLAY_AGES:
        age_rows = rows[
            rows["age_months"] == age
        ]

        if age_rows.empty:
            continue

        dbh_value = float(
            age_rows["pred_dbh"].mean()
        )

        height_value = float(
            age_rows["pred_height"].mean()
        )

        volume_series = age_rows["pred_volume"]

        if volume_series.notna().any():
            volume_value = float(
                volume_series.mean()
            )
        else:
            volume_value = None

        years = age / 12.0

        mai_value = (
            volume_value / years
            if volume_value is not None
            else None
        )

        # Hitung volume per hektare per baris,
        # kemudian rata-ratakan hasilnya.
        valid_volume = age_rows[
            age_rows["pred_volume"].notna()
        ].copy()

        if not valid_volume.empty:
            volume_per_ha = float(
                (
                    valid_volume["pred_volume"]
                    * valid_volume["trees_per_ha"]
                ).mean()
            )

            total_volume = float(
                (
                    valid_volume["pred_volume"]
                    * valid_volume[
                        "estimated_trees_compartment"
                    ]
                ).mean()
            )
        else:
            volume_per_ha = None
            total_volume = None

        stand_mai = (
            volume_per_ha / years
            if volume_per_ha is not None
            else None
        )

        trajectory.append({
            "age_months": age,
            "growth_period": DISPLAY_PERIODS[age],

            "regional_id": regional_label,
            "compartment_id": compartment_id_label,
            "compartment_code": compartment_label,
            "species": species_label,
            "clone": clone_label,

            "dbh_cm": dbh_value,
            "height_m": height_value,
            "volume_m3_tree": volume_value,
            "mai_m3_tree_year": mai_value,
            "volume_m3_ha": volume_per_ha,
            "total_volume_m3": total_volume,
            "stand_mai_m3_ha_year": stand_mai,

            "tha_m": float(
                age_rows["tha_m"].mean()
            ),
            "actual_rows": int(len(age_rows)),
        })

    return trajectory



# ============================================================
# TRAJECTORY ENDPOINT
# ============================================================

@app.get(
    "/dashboard/trajectory"
)
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

    if model not in [
        "random_forest",
        "pinns",
    ]:

        raise HTTPException(
            status_code=400,
            detail=(
                "model must be "
                "'random_forest' or "
                "'pinns'."
            ),
        )

    rows = apply_filters(
        df,
        regional_id,
        compartment_id,
        species,
        clone,
    )

    if rows.empty:

        return {
            "success": True,
            "model": model,
            "prediction_cohort": None,
            "filters": {
                "regional_id":
                    regional_id,
                "compartment_id":
                    compartment_id,
                "species":
                    species,
                "clone":
                    clone,
            },
            "trajectory": [],
            "row_count": 0,
        }

    trajectory = build_trajectory(
        rows,
        model,
    )

    planting_years = (
        pd.to_numeric(
            rows["planting_year"],
            errors="coerce",
        )
        .dropna()
    )

    if planting_years.empty:
        planting_year_min = None
        planting_year_max = None
    else:
        planting_year_min = int(planting_years.min())
        planting_year_max = int(planting_years.max())

    return {
        "success": True,

        "model": model,

        # Kept temporarily for frontend compatibility.
        # This is the latest planting year in the
        # selected dataset, NOT a future prediction year.
        # Dipertahankan agar frontend lama tetap kompatibel.
        "prediction_cohort": planting_year_max,

        # Metadata eksplisit untuk trajectory agregat.
        "trajectory_scope": "all_selected_cohorts",
        "planting_year_range": {
            "min": planting_year_min,
            "max": planting_year_max,
        },

        "filters": {
            "regional_id":
                regional_id,
            "compartment_id":
                compartment_id,
            "species":
                species,
            "clone":
                clone,
        },

        "trajectory":
            trajectory,

        "row_count":
            len(trajectory),
    }


# ============================================================
# ACTUAL TRAJECTORY
# ============================================================

@app.get(
    "/dashboard/actual-trajectory"
)
def actual_trajectory(
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

    rows = apply_filters(
        df,
        regional_id,
        compartment_id,
        species,
        clone,
    )

    if rows.empty:

        return {
            "success": True,
            "source":
                "dataset_phi66_final.csv",
            "filters": {
                "regional_id":
                    regional_id,
                "compartment_id":
                    compartment_id,
                "species":
                    species,
                "clone":
                    clone,
            },
            "periods": [],
        }

    periods = []

    for age in DISPLAY_AGES:

        age_rows = rows[
            rows[
                "age_months"
            ] == age
        ]

        if age_rows.empty:

            continue

        periods.append(
            {
                "growth_period":
                    DISPLAY_PERIODS[
                        age
                    ],

                "age_months_mean":
                    float(
                        age_rows[
                            "age_months"
                        ].mean()
                    ),

                "dbh_cm":
                    float(
                        age_rows[
                            "dbh_cm"
                        ].mean()
                    ),

                "height_m":
                    float(
                        age_rows[
                            "height_m"
                        ].mean()
                    ),

                "volume_m3_tree":
                    float(
                        age_rows[
                            "volume_m3_tree"
                        ].mean()
                    ),

                "actual_rows":
                    int(
                        len(age_rows)
                    ),
            }
        )

    return {
        "success": True,

        "source":
            "dataset_phi66_final.csv",

        "filters": {
            "regional_id":
                regional_id,
            "compartment_id":
                compartment_id,
            "species":
                species,
            "clone":
                clone,
        },

        "periods":
            periods,
    }


# ============================================================
# SUMMARY
# ============================================================

@app.get(
    "/dashboard/summary"
)
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

    rows = apply_filters(
        df,
        regional_id,
        compartment_id,
        species,
        clone,
    )

    if rows.empty:

        return {
            "success": True,
            "model": model,
            "prediction_cohort": None,
            "filters": {},
            "attributes": {},
            "summary": {},
        }

    trajectory = build_trajectory(
        rows,
        model,
    )

    phi_row = None

    for row in trajectory:

        if (
            row[
                "age_months"
            ] == 66
        ):

            phi_row = row
            break

    if phi_row is None:

        raise HTTPException(
            status_code=500,
            detail="P66/PHI is missing.",
        )

    first = rows.iloc[0]

    return {
        "success": True,

        "model": model,

        "prediction_cohort":
            int(
                rows[
                    "planting_year"
                ].max()
            ),

        "filters": {
            "regional_id":
                regional_id,
            "compartment_id":
                compartment_id,
            "species":
                species,
            "clone":
                clone,
        },

        "attributes": {
            "area_ha":
                float(
                    rows[
                        "area_ha"
                    ].mean()
                ),

            "spacing_m":
                str(
                    first[
                        "spacing_m"
                    ]
                ),

            "spacing_x_m":
                str(
                    first[
                        "spacing_x_m"
                    ]
                ),

            "spacing_y_m":
                str(
                    first[
                        "spacing_y_m"
                    ]
                ),

            "plant_spacing_m":
                str(
                    first[
                        "plant_spacing_m"
                    ]
                ),

            "trees_per_ha":
                float(
                    rows[
                        "trees_per_ha"
                    ].mean()
                ),

            "estimated_trees_compartment":
                float(
                    rows[
                        "estimated_trees_compartment"
                    ].mean()
                ),
        },

        "summary": {
            "average_dbh_cm":
                phi_row[
                    "dbh_cm"
                ],

            "average_height_m":
                phi_row[
                    "height_m"
                ],

            "average_volume_m3_tree":
                phi_row[
                    "volume_m3_tree"
                ],

            "average_mai_m3_tree_year":
                phi_row[
                    "mai_m3_tree_year"
                ],

            "average_volume_m3_ha":
                phi_row[
                    "volume_m3_ha"
                ],

            "average_total_volume_m3":
                phi_row[
                    "total_volume_m3"
                ],

            "average_stand_mai_m3_ha_year":
                phi_row[
                    "stand_mai_m3_ha_year"
                ],

            "tha_note":
                (
                    "THA is retained from the "
                    "dataset as contextual data. "
                    "It is not a direct PINNs output."
                ),
        },
    }


# ============================================================

# ============================================================
# MAI REPORT â€” ACTUAL DATA, NOT MODEL PREDICTIONS
# ============================================================

@app.get("/dashboard/mai-report")
def mai_report(
    planting_year_start: Optional[int] = Query(default=None),
    planting_year_end: Optional[int] = Query(default=None),
    regional_id: Optional[str] = Query(default=None),
    compartment_id: Optional[str] = Query(default=None),
    species: Optional[str] = Query(default=None),
    clone: Optional[str] = Query(default=None),
):
    import json

    required = [
        "regional_id", "regional_name",
        "compartment_id", "compartment_code",
        "area_ha", "trees_per_ha",
        "species", "clone", "planting_year",
        "age_months", "growth_period",
        "volume_m3_tree",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Kolom dataset untuk laporan MAI tidak tersedia: {missing}",
        )

    data = df.copy()

    data["planting_year"] = pd.to_numeric(
        data["planting_year"], errors="coerce"
    )
    data["age_months"] = pd.to_numeric(
        data["age_months"], errors="coerce"
    )
    data["volume_m3_tree"] = pd.to_numeric(
        data["volume_m3_tree"], errors="coerce"
    )
    data["trees_per_ha"] = pd.to_numeric(
        data["trees_per_ha"], errors="coerce"
    )
    data["area_ha"] = pd.to_numeric(
        data["area_ha"], errors="coerce"
    )

    if planting_year_start is not None:
        data = data[
            data["planting_year"] >= planting_year_start
        ]
    if planting_year_end is not None:
        data = data[
            data["planting_year"] <= planting_year_end
        ]
    if (
        planting_year_start is not None
        and planting_year_end is not None
        and planting_year_start > planting_year_end
    ):
        raise HTTPException(
            status_code=400,
            detail="Tahun awal tidak boleh melebihi tahun akhir.",
        )

    for column, value in [
        ("regional_id", regional_id),
        ("compartment_id", compartment_id),
        ("species", species),
        ("clone", clone),
    ]:
        if value is not None:
            data = data[
                data[column].astype(str) == str(value)
            ]

    if data.empty:
        return {
            "status": "no_data",
            "message": "Tidak ada data yang sesuai dengan filter.",
            "filters": {
                "planting_year_start": planting_year_start,
                "planting_year_end": planting_year_end,
                "regional_id": regional_id,
                "compartment_id": compartment_id,
                "species": species,
                "clone": clone,
            },
            "summary": {},
            "by_planting_year": [],
            "details": [],
            "methodology": {
                "threshold_m3_ha_year": 25,
            },
        }

    # Pilih umur pengamatan terakhir untuk setiap
    # kombinasi kompartemen dan tahun tanam.
    data = data.dropna(
        subset=[
            "compartment_id", "planting_year",
            "age_months", "volume_m3_tree",
            "trees_per_ha", "area_ha",
        ]
    )

    if data.empty:
        raise HTTPException(
            status_code=422,
            detail="Tidak ada baris dengan data numerik yang memadai.",
        )

    keys = ["compartment_id", "planting_year"]
    max_age = data.groupby(keys)["age_months"].transform("max")
    latest = data[data["age_months"] == max_age].copy()

    # Satu detail per kompartemen dan tahun tanam.
    group_keys = [
        "compartment_id", "compartment_code", "planting_year"
    ]
    details = latest.groupby(
        group_keys, dropna=False, as_index=False
    ).agg(
        regional_id=("regional_id", "first"),
        regional_name=("regional_name", "first"),
        area_ha=("area_ha", "first"),
        age_months=("age_months", "first"),
        growth_period=("growth_period", "last"),
        trees_per_ha=("trees_per_ha", "first"),
        mean_volume_m3_tree=("volume_m3_tree", "mean"),
        measurement_rows=("volume_m3_tree", "count"),
        species=("species", lambda s: ", ".join(sorted(set(s.dropna().astype(str))))),
        clone=("clone", lambda s: ", ".join(sorted(set(s.dropna().astype(str))))),
    )

    details = details[
        (details["age_months"] > 0)
        & (details["area_ha"] > 0)
        & (details["trees_per_ha"] > 0)
    ].copy()

    if details.empty:
        raise HTTPException(
            status_code=422,
            detail="Data tidak memiliki luas, umur, atau kepadatan positif untuk menghitung MAI.",
        )

    details["estimated_volume_m3_ha"] = (
        details["mean_volume_m3_tree"]
        * details["trees_per_ha"]
    )
    details["mai_m3_ha_year"] = (
        details["estimated_volume_m3_ha"]
        / (details["age_months"] / 12.0)
    )
    details["mai_x_area"] = (
        details["mai_m3_ha_year"] * details["area_ha"]
    )
    details["status"] = details["mai_m3_ha_year"].apply(
        lambda value: "Baik" if value >= 25 else "Di bawah ambang batas"
    )

    # Rata-rata MAI berbobot luas.
    total_area = float(details["area_ha"].sum())
    weighted_mai = (
        float(details["mai_x_area"].sum()) / total_area
        if total_area > 0 else None
    )

    # Agregasi per tahun tanam.
    yearly_records = []
    for year, part in details.groupby("planting_year"):
        area = float(part["area_ha"].sum())
        weighted = (
            float(part["mai_x_area"].sum()) / area
            if area > 0 else None
        )
        yearly_records.append({
            "planting_year": int(year),
            "cohort_area_ha": area,
            "compartment_count": int(len(part)),
            "mai_weighted_m3_ha_year": weighted,
            "mai_min_m3_ha_year": float(part["mai_m3_ha_year"].min()),
            "mai_max_m3_ha_year": float(part["mai_m3_ha_year"].max()),
            "good_count": int((part["status"] == "Baik").sum()),
            "below_threshold_count": int(
                (part["status"] == "Di bawah ambang batas").sum()
            ),
        })

    details = details.sort_values(
        ["planting_year", "compartment_id"]
    )

    def records_json(frame):
        return json.loads(
            frame.replace({np.nan: None}).to_json(
                orient="records"
            )
        )

    detail_columns = [
        "regional_id", "regional_name",
        "compartment_id", "compartment_code",
        "planting_year", "area_ha",
        "age_months", "growth_period",
        "species", "clone",
        "measurement_rows", "mean_volume_m3_tree",
        "trees_per_ha", "estimated_volume_m3_ha",
        "mai_m3_ha_year", "status",
    ]

    return {
        "status": "ok",
        "source": "data/dataset_phi66_final.csv",
        "filters": {
            "planting_year_start": planting_year_start,
            "planting_year_end": planting_year_end,
            "regional_id": regional_id,
            "compartment_id": compartment_id,
            "species": species,
            "clone": clone,
        },
        "summary": {
            "cohort_compartment_count": int(len(details)),
            "cohort_area_ha": total_area,
            "mai_weighted_m3_ha_year": weighted_mai,
            "threshold_m3_ha_year": 25.0,
            "good_count": int((details["status"] == "Baik").sum()),
            "below_threshold_count": int(
                (details["status"] == "Di bawah ambang batas").sum()
            ),
            "mai_min_m3_ha_year": float(details["mai_m3_ha_year"].min()),
            "mai_max_m3_ha_year": float(details["mai_m3_ha_year"].max()),
        },
        "by_planting_year": yearly_records,
        "details": records_json(details[detail_columns]),
        "methodology": {
            "source_metric": "volume_m3_tree aktual pada umur pengamatan terakhir",
            "estimated_volume_formula": "mean(volume_m3_tree) Ã— trees_per_ha",
            "mai_formula": "estimated_volume_m3_ha Ã· (age_months / 12)",
            "weighted_formula": "sum(MAI Ã— area_ha) Ã· sum(area_ha)",
            "threshold_rule": "MAI >= 25 mÂ³/ha/tahun = Baik",
            "density_note": (
                "Estimasi menggunakan trees_per_ha yang tercatat; "
                "belum tentu sama dengan kepadatan pohon hidup aktual."
            ),
            "area_note": (
                "Luas dijumlahkan per kombinasi kompartemen dan tahun tanam. "
                "Jika beberapa kohort dipilih, luas kohort bukan luas fisik unik."
            ),
            "prediction_used": False,
        },
    }


# PHI COMPARISON
# ============================================================

@app.get(
    "/dashboard/phi-comparison"
)
def phi_comparison(
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

    if model not in [
        "random_forest",
        "pinns",
    ]:

        raise HTTPException(
            status_code=400,
            detail="Invalid model.",
        )

    rows = apply_filters(
        df,
        regional_id,
        compartment_id,
        species,
        clone,
    )

    if rows.empty:

        return {
            "success": True,
            "model": model,
            "phi_age_months": 66,
            "growth_period": "PHI",
            "data": None,
        }

    trajectory = build_trajectory(
        rows,
        model,
    )

    phi = next(
        (
            row
            for row in trajectory
            if row[
                "age_months"
            ] == 66
        ),
        None,
    )

    if phi is None:

        raise HTTPException(
            status_code=500,
            detail="P66/PHI not available.",
        )

    return {
        "success": True,

        "model": model,

        "phi_age_months": 66,

        "growth_period": "PHI",

        "prediction_cohort":
            int(
                rows[
                    "planting_year"
                ].max()
            ),

        "comparison_basis":
            "P66 is explicitly defined as PHI.",

        "data":
            phi,
    }

def compare_growth_groups(question: str):
    """
    Membandingkan statistik pertumbuhan pohon berdasarkan
    wilayah, spesies, atau klon menggunakan dataset lokal.

    Mengembalikan hasil analisis beserta data grafik terstruktur.
    Tidak menggunakan API AI eksternal.
    """
    import re
    import pandas as pd

    if df is None or df.empty:
        return None

    q = question.lower().strip()

    
    comparison_words = [
        "bandingkan",
        "perbandingan",
        "dibandingkan",
        "dibanding",
        "mana yang tertinggi",
        "mana tertinggi",
        "mana yang terbesar",
        "mana terbesar",
        "mana yang terendah",
        "mana terendah",
        "paling tinggi",
        "paling besar",
        "paling rendah",
        "tertinggi",
        "terbesar",
        "terendah",
        "terbaik",
        "terbagus",
        "mana yang paling bagus",
        "mana paling bagus",
        "mana yang paling baik",
        "mana paling baik",
        "mana yang menghasilkan",
        "menghasilkan pertumbuhan",
        "pertumbuhan terbaik",
        "pertumbuhan paling baik",
        "compartment mana",
        "kompartemen mana",
        "clone mana",
        "klon mana",
        "regional mana",
        "wilayah mana",
    ]


    if not any(word in q for word in comparison_words):
        return None

    metric_aliases = {
        "height_m": [
            "tinggi pohon",
            "rata-rata tinggi",
            "height_m",
            "height",
            "tinggi",
        ],
        "dbh_cm": [
            "diameter pohon",
            "diameter batang",
            "dbh_cm",
            "dbh cm",
            "dbh",
            "diameter",
        ],
        "volume_m3_tree": [
            "volume per pohon",
            "volume pohon",
            "volume_m3_tree",
            "volume",
        ],
        "disease_score": [
            "disease score", "skor penyakit",
        ],
        "condition_score": [
            "condition score", "skor kondisi",
        ],
        "trees_per_ha": [
            "trees per ha", "kerapatan pohon",
            "pohon per hektare", "pohon per hektar",
        ],
    }

    metric_labels = {
        "height_m": "Tinggi pohon (m)",
        "dbh_cm": "Diameter pohon (cm)",
        "volume_m3_tree": "Volume per pohon (m³)",
        "disease_score": "Disease score",
        "condition_score": "Condition score",
        "trees_per_ha": "Jumlah pohon per hektare",
    }

    metric = None
    aliases = sorted(
        [
            (alias, column)
            for column, items in metric_aliases.items()
            for alias in items
        ],
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for alias, column in aliases:
        if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", q):
            metric = column
            break

    if metric is None or metric not in df.columns:
        return None



    # Tentukan kelompok yang ingin dibandingkan.
    # Dahulukan frasa "berdasarkan ..." untuk menentukan
    # kelompok perbandingan yang diminta secara eksplisit.
    group_column = None
    group_label = None

    grouping_patterns = [
        (
            r"\bberdasarkan\s+(?:jenis\s+)?(?:klon|clone)\b",
            "clone",
            "Klon",
        ),
        (
            r"\bberdasarkan\s+(?:compartment|kompartemen|comp)\b",
            "compartment_code",
            "Compartment",
        ),
        (
            r"\bberdasarkan\s+(?:spesies|species)\b",
            "species",
            "Spesies",
        ),
        (
            r"\bberdasarkan\s+(?:wilayah|regional)\b",
            "regional_name",
            "Wilayah",
        ),
    ]

    for pattern, column, label in grouping_patterns:
        if re.search(pattern, q):
            group_column = column
            group_label = label
            break

    # Fallback untuk pertanyaan yang tidak memakai
    # frasa "berdasarkan", dengan prioritas sebelumnya.
    if group_column is None:
        
        if re.search(r"\b(klon|clone)\b", q):
            group_column = "clone"
            group_label = "Klon"

        elif re.search(r"\b(compartment|kompartemen|kompartemen|comp)\b", q):
            group_column = "compartment_code"
            group_label = "Compartment"

        elif re.search(r"\b(spesies|species)\b", q):
            group_column = "species"
            group_label = "Spesies"

        elif re.search(r"\b(wilayah|regional)\b", q):
            group_column = "regional_name"
            group_label = "Wilayah"

        else:
            return None

    if group_column not in df.columns:
        return None

    # Salin dataset agar data asli tidak berubah.
    data = df.copy()

    # ========================================================
    # FILTER SPESIFIK DARI PERTANYAAN
    # ========================================================

    
    # 0. Filter berdasarkan wilayah jika disebutkan.
    if "regional_name" in data.columns:
        available_regions = (
            data["regional_name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        available_regions.sort(key=len, reverse=True)

        matched_region = None

        for region_name in available_regions:
            pattern = (
                r"(?<!\w)"
                + re.escape(region_name.lower())
                + r"(?!\w)"
            )

            if re.search(pattern, q):
                matched_region = region_name
                break

        if matched_region is not None:
            data = data[
                data["regional_name"]
                .astype(str)
                .str.lower()
                == matched_region.lower()
            ]

    # 1. Filter berdasarkan spesies yang disebutkan.
    if "species" in data.columns:
        available_species = (
            data["species"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        available_species.sort(key=len, reverse=True)

        for species_name in available_species:
            pattern = (
                r"(?<!\w)"
                + re.escape(species_name.lower())
                + r"(?!\w)"
            )

            if re.search(pattern, q):
                data = data[
                    data["species"]
                    .astype(str)
                    .str.lower()
                    == species_name.lower()
                ]
                break
    # 1b. Filter klon jika nama klon tertentu disebutkan.
    if "clone" in data.columns:
        available_clones = (
            data["clone"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        available_clones.sort(key=len, reverse=True)

        matched_clone = None

        for clone_name in available_clones:
            pattern = (
                r"(?<!\w)"
                + re.escape(clone_name.lower())
                + r"(?!\w)"
            )

            if re.search(pattern, q):
                matched_clone = clone_name
                break

        if matched_clone is not None:
            # Jangan memfilter klon jika klon adalah
            # kelompok yang sedang dibandingkan.
            if group_column != "clone":
                data = data[
                    data["clone"]
                    .astype(str)
                    .str.lower()
                    == matched_clone.lower()
                ]
    # 2. Filter berdasarkan kode atau ID compartment.
    # Satu atau lebih compartment yang disebutkan harus diterapkan.
    
    if re.search(r"\b(compartment|kompartemen|comp)\b", q):
        requested_compartments = []

        # Pertanyaan terbuka boleh membandingkan semua compartment.
        # Kode atau ID eksplisit tetap harus divalidasi.
        explicit_compartment_reference = bool(
            re.search(
                r"(?<!\w)comp-\d+(?!\w)",
                q,
                re.IGNORECASE,
            )
            or re.search(
                r"\b(?:compartment|kompartemen|comp)\s*[-:]?\s*\d+\b",
                q,
                re.IGNORECASE,
            )
        )

        open_compartment_ranking = (
            group_column == "compartment_code"
            and not explicit_compartment_reference
        )


        # Cari kode eksplisit, misalnya COMP-101.
        requested_code_matches = re.findall(
            r"(?<!\w)comp-\d+(?!\w)",
            q,
            re.IGNORECASE,
        )

        available_codes = (
            data["compartment_code"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
            if "compartment_code" in data.columns
            else []
        )

        code_lookup = {
            code.upper(): code
            for code in available_codes
        }

        for requested_code in dict.fromkeys(
            code.upper() for code in requested_code_matches
        ):
            if requested_code not in code_lookup:
                return {
                    "success": True,
                    "answer": (
                        f"Kompartemen {requested_code} tidak ditemukan "
                        "dalam data yang sesuai dengan filter sebelumnya."
                    ),
                    "mode": "python_local",
                    "data": {"found": False},
                    "chart": None,
                    "chart_offer": None,
                }

            requested_compartments.append(
                ("compartment_code", code_lookup[requested_code])
            )

        # Jika tidak ada kode COMP-xxx, cari ID seperti compartment 101.
        if not requested_compartments:
            numbers = re.findall(
                r"\b(?:compartment|kompartemen|comp)\s*[-:]?\s*(\d+)\b",
                q,
                re.IGNORECASE,
            )

            if numbers and "compartment_id" in data.columns:
                available_ids = (
                    data["compartment_id"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                for number in dict.fromkeys(numbers):
                    if number not in available_ids:
                        return {
                            "success": True,
                            "answer": (
                                f"ID kompartemen {number} tidak ditemukan "
                                "dalam data yang sesuai dengan filter sebelumnya."
                            ),
                            "mode": "python_local",
                            "data": {"found": False},
                            "chart": None,
                            "chart_offer": None,
                        }

                    requested_compartments.append(
                        ("compartment_id", number)
                    )

        if not requested_compartments and not open_compartment_ranking:
            return {
                "success": True,
                "answer": (
                    "Kode atau ID kompartemen tidak dapat dikenali. "
                    "Periksa kembali kompartemen yang disebutkan."
                ),
                "mode": "python_local",
                "data": {"found": False},
                "chart": None,
                "chart_offer": None,
            }

        
        # Terapkan filter hanya jika kode atau ID disebutkan.
        # Jika pertanyaan terbuka, pertahankan seluruh compartment.
        if requested_compartments:
            columns = {
                column
                for column, _ in requested_compartments
            }

            if len(columns) != 1:
                return {
                    "success": True,
                    "answer": (
                        "Gunakan kode kompartemen atau ID kompartemen "
                        "secara konsisten dalam satu pertanyaan."
                    ),
                    "mode": "python_local",
                    "data": {"found": False},
                    "chart": None,
                    "chart_offer": None,
                }

            column = requested_compartments[0][0]
            values = list(dict.fromkeys(
                value for _, value in requested_compartments
            ))

            data = data[
                data[column].astype(str).isin(values)
            ]

    # Hentikan perbandingan jika filter tidak menghasilkan data.
    if data.empty:
        return {
            "success": True,
            "answer": (
                "Tidak ditemukan data yang sesuai dengan filter "
                "spesifik pada pertanyaan."
            ),
            "mode": "python_local",
            "data": {
                "found": False,
            },
        }


    # --------------------------------------------------------
    # VALIDASI DAN FILTER PERIODE UNTUK PERBANDINGAN
    # P6-P66 didukung; PHI merupakan alias P66.
    # Periode tidak valid tidak boleh diabaikan.
    # --------------------------------------------------------

    period_match = re.search(
        r"(?<![a-z0-9])p\d+(?![a-z0-9])",
        q,
    )

    phi_match = re.search(
        r"(?<!\w)phi(?!\w)",
        q,
    )

    requested_period = None

    if period_match:
        requested_period = period_match.group(0).upper()
    elif phi_match:
        requested_period = "P66"

    period_found = None

    if requested_period is not None:
        period_to_age = {
            period: age
            for age, period in DISPLAY_PERIODS.items()
        }

        if requested_period not in period_to_age:
            return {
                "success": True,
                "answer": (
                    f"Periode {requested_period} tidak valid "
                    "atau tidak didukung. "
                    "Gunakan periode P6 sampai P66."
                ),
                "mode": "python_local",
                "data": {
                    "found": False,
                    "filters": {
                        "period": requested_period,
                    },
                },
                "chart": None,
                "chart_offer": None,
            }

        period_found = requested_period
        age = period_to_age[requested_period]

        data = data[data["age_months"] == age]

    # Hentikan proses jika periode valid tetapi tidak menghasilkan data.
    if data.empty:
        return {
            "success": True,
            "answer": (
                "Tidak ditemukan data yang sesuai dengan "
                "filter periode pada perbandingan."
            ),
            "mode": "python_local",
            "data": {
                "found": False,
                "filters": {
                    "period": period_found,
                },
            },
            "chart": None,
            "chart_offer": None,
        }


    data[metric] = pd.to_numeric(data[metric], errors="coerce")

    grouped = (
        data.groupby(group_column, dropna=True)[metric]
        .agg(["mean", "count"])
        .dropna(subset=["mean"])
    )

    if grouped.empty:
        return {
            "success": True,
            "answer": (
                "Tidak ada data yang dapat dibandingkan "
                "untuk pertanyaan ini."
            ),
            "mode": "python_local",
            "chart": None,
            "chart_offer": None,
        }

    lowest_requested = bool(
        re.search(r"\b(terendah|paling rendah)\b", q)
    )

    grouped = grouped.sort_values(
        "mean",
        ascending=lowest_requested,
    )

    
    # Data kategori, nilai rata-rata, dan jumlah observasi valid.
    chart_categories = [str(name) for name in grouped.index]

    chart_values = [
        round(float(value), 6)
        for value in grouped["mean"].tolist()
    ]

    chart_counts = [
        int(value)
        for value in grouped["count"].tolist()
    ]


    chart_title = (
        "Perbandingan "
        + metric_labels[metric]
        + " berdasarkan "
        + group_label.lower()
        + (f" pada {period_found}" if period_found else "")
    )

    
    # ========================================================
    # NARASI JAWABAN: JAWAB LANGSUNG, LALU JELASKAN
    # ========================================================

    metric_names = {
        "height_m": "tinggi pohon",
        "dbh_cm": "diameter pohon (DBH)",
        "volume_m3_tree": "volume pohon",
        "disease_score": "disease score",
        "condition_score": "condition score",
        "trees_per_ha": "jumlah pohon per hektare",
    }

    metric_units = {
        "height_m": "m",
        "dbh_cm": "cm",
        "volume_m3_tree": "m³ per pohon",
        "trees_per_ha": "pohon/ha",
    }

    group_names = {
        "Wilayah": "Regional",
        "Klon": "Clone",
        "Compartment": "Compartment",
        "Spesies": "Spesies",
    }

    metric_name = metric_names.get(metric, metric_labels[metric])
    group_name = group_names.get(group_label, group_label)
    unit = metric_units.get(metric, "")

    winner = str(grouped.index[0])
    winner_value = float(grouped.iloc[0]["mean"])

    
    # Hindari pengulangan nama kelompok.
    def display_group_name(value):
        value = str(value).strip()
        prefix = group_name.lower() + " "

        if value.lower().startswith(prefix):
            return value

        return f"{group_name} {value}"
    winner_display = display_group_name(winner)


    def format_value(value):
        return f"{value:,.4f}".replace(",", "_").replace(".", ",").replace("_", ".")

    winner_formatted = format_value(winner_value)
    unit_text = f" {unit}" if unit else ""

    if lowest_requested:
        answer_intro = (
            f"{winner_display} memiliki rata-rata {metric_name} "
            f"terendah, yaitu {winner_formatted}{unit_text}."
        )
        comparison_heading = "Urutan dari nilai rata-rata terendah:"
        conclusion = (
            f"Berdasarkan data yang dibandingkan, {winner_display} "
            f"menempati peringkat pertama berdasarkan rata-rata {metric_name}."
        )
    else:
        answer_intro = (
            f"{winner_display} memiliki rata-rata {metric_name} "
            f"tertinggi, yaitu {winner_formatted}{unit_text}."
        )
        comparison_heading = "Urutan dari nilai rata-rata tertinggi:"
        
        conclusion = (
            f"Berdasarkan data yang dibandingkan, {winner_display} "
            f"menempati peringkat pertama berdasarkan rata-rata {metric_name}."
        )


    lines = [
        answer_intro,
        "",
        f"Berikut perbandingan {metric_name} berdasarkan {group_name.lower()}:",
    ]

    if period_found:
        lines.append(f"Periode pengamatan: {period_found}")

    lines.extend([
        "",
        comparison_heading,
    ])

    for rank, (name, row) in enumerate(grouped.iterrows(), start=1):
        value_text = format_value(float(row["mean"]))
        count_text = f"{int(row['count']):,}".replace(",", ".")

        lines.append(
            f"{rank}. {display_group_name(name)}: "
            f"{value_text}{unit_text} "
            f"(jumlah data: {count_text})"
        )

    lines.extend([
        "",
        conclusion,
    ])


    return {
        "success": True,
        "answer": "\n".join(lines),
        "mode": "python_local",
        "data": {
            "metric": metric,
            "metric_label": metric_labels[metric],
            "group_by": group_column,
            "group_label": group_label,
            "period": period_found,
            "groups": int(len(grouped)),
            "filters": {
                "regional": (
                    str(data["regional_name"].iloc[0])
                    if "regional_name" in data.columns
                    and data["regional_name"].nunique() == 1
                    and re.search(
                        r"(?<!\w)"
                        + re.escape(
                            str(data["regional_name"].iloc[0]).lower()
                        )
                        + r"(?!\w)",
                        q,
                    )
                    else None
                ),
                "species": (
                    str(data["species"].iloc[0])
                    if "species" in data.columns
                    and data["species"].nunique() == 1
                    and re.search(
                        r"(?<!\w)"
                        + re.escape(str(data["species"].iloc[0]).lower())
                        + r"(?!\w)",
                        q,
                    )
                    else None
                ),
                "compartments": (
                    sorted(
                        data["compartment_code"]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                    if re.search(
                        r"\b(compartment|kompartemen|comp)\b",
                        q,
                    )
                    and "compartment_code" in data.columns
                    else []
                ),
                "clone": matched_clone,
            },

        },
        "chart_offer": {
            "available": True,
            "message": "Apakah kamu ingin melihat grafik perbandingannya?",
            "title": chart_title,
            "chart_type": "bar",
        },
        "chart": {
            "type": "bar",
            "title": chart_title,
            "x_key": "category",
            "y_key": "value",
            "x_label": group_label,
            "y_label": metric_labels[metric],
            "period": period_found,
            
            "data": [
                {
                    "category": category,
                    "value": value,
                    "count": count,
                }
                for category, value, count in zip(
                    chart_categories,
                    chart_values,
                    chart_counts,
                )
            ],
        },
    }


def answer_prediction_question(question: str):
    """
    Menjawab permintaan prediksi menggunakan trajectory model
    yang sudah tersedia di backend.
    Mengembalikan None jika bukan pertanyaan prediksi.
    """
    import re

    q = question.lower().strip()

    prediction_words = [
        "prediksi",
        "prediksikan",
        "perkirakan",
        "perkiraan pertumbuhan",
        "forecast",
        "ramalkan",
    ]

    if not any(word in q for word in prediction_words):
        return None

    if df is None or df.empty:
        return {
            "success": False,
            "answer": "Dataset belum tersedia untuk prediksi.",
            "mode": "python_local",
        }

    # Pilih model berdasarkan permintaan pengguna.
    # Random Forest menjadi default jika PINNs tidak disebutkan.
    model = "pinns" if re.search(r"\b(pinns|physics-informed)\b", q) else "random_forest"

    # Ambil hanya nilai filter yang benar-benar ada di dataset.
    def find_value(column: str):
        if column not in df.columns:
            return None

        values = df[column].dropna().astype(str).unique().tolist()
        values.sort(key=len, reverse=True)

        for value in values:
            pattern = r"(?<!\w)" + re.escape(value.lower()) + r"(?!\w)"

            if re.search(pattern, q):
                return value

        # Penting: return None berada di luar loop.
        return None

    regional_id = find_value("regional_id")

    # Validasi jika pengguna menyebut regional
    # yang tidak tersedia dalam dataset.
    regional_keyword_match = re.search(
        r"\b(regional|region|wilayah)\b",
        q,
        re.IGNORECASE,
    )

    regional_code_match = re.search(
        r"\bR\d+\b",
        q,
        re.IGNORECASE,
    )

    if (
        regional_keyword_match
        and regional_code_match
        and regional_id is None
    ):
        available_regionals = sorted(
            df["regional_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        return {
            "success": False,
            "answer": (
                "Regional yang disebutkan tidak ditemukan "
                "dalam dataset. Regional yang tersedia: "
                + ", ".join(available_regionals)
            ),
            "mode": "python_local",
        }

    species = find_value("species")
    clone = find_value("clone")

    # Validasi jika pengguna menyebut spesies
    # tetapi namanya tidak ditemukan dalam dataset.
    species_keyword_match = re.search(
        r"\bspesies\b",
        q,
        re.IGNORECASE,
    )

    eucalyptus_mention = re.search(
        r"(?<!\w)eucalyptus\b",
        q,
        re.IGNORECASE,
    )

    if (
        (species_keyword_match or eucalyptus_mention)
        and species is None
    ):
        return {
            "success": False,
            "answer": (
                "Spesies yang disebutkan tidak ditemukan "
                "dalam dataset. Periksa kembali nama spesies."
            ),
            "mode": "python_local",
        }

    compartment_id = None
    if "compartment_id" in df.columns:
        value = find_value("compartment_id")
        if value is not None:
            try:
                compartment_id = int(float(value))
            except (TypeError, ValueError):
                compartment_id = None

        # Filter menggunakan fungsi backend yang sudah dipakai endpoint.
    rows = apply_filters(
        df,
        regional_id,
        compartment_id,
        species,
        clone,
    )

    if rows.empty:
        return {
            "success": True,
            "answer": (
                "Tidak ditemukan data untuk kombinasi filter prediksi "
                "yang diminta. Periksa kembali wilayah, kompartemen, "
                "spesies, atau klon."
            ),
            "mode": "python_local",
        }

    # Jalankan model yang sudah ada, bukan membuat prediksi sendiri.
    trajectory = build_trajectory(rows, model)

    if not trajectory:
        return {
            "success": True,
            "answer": "Model tidak menghasilkan trajectory untuk filter ini.",
            "mode": "python_local",
        }

    
    # Gunakan periode yang disebutkan pengguna.
    requested_age = None

    # Deteksi periode P6, P12, ..., P66.
    for age, period in sorted(
        DISPLAY_PERIODS.items(),
        key=lambda item: item[0],
        reverse=True,
    ):
        if re.search(
            r"(?<![a-z0-9])" + re.escape(period.lower()) + r"(?![0-9])",
            q,
        ):
            requested_age = age
            break

    # Deteksi umur numerik, termasuk angka 3 digit atau lebih.
    if requested_age is None:
        age_match = re.search(
            r"\b(?:umur|usia)\s+(\d+)\s+bulan\b",
            q,
        )

        if age_match:
            requested_age = int(age_match.group(1))

    # Jangan menggunakan default P66 jika umur tidak disebutkan.
    if requested_age is None:
        return {
            "success": False,
            "answer": (
                "Umur pohon belum disebutkan. "
                "Silakan masukkan umur dalam bulan, misalnya 36 bulan."
            ),
            "mode": "python_local",
        }

    
    result_row = next(
        (
            item for item in trajectory
            if item["age_months"] == requested_age
        ),
        None,
    )

    if result_row is None:
        available_periods = ", ".join(
            item["growth_period"] for item in trajectory
        )
        return {
            "success": True,
            "answer": (
                f"Periode umur {requested_age} bulan tidak tersedia "
                f"pada trajectory ini. Periode tersedia: {available_periods}."
            ),
            "mode": "python_local",
        }

    metric_aliases = {
        "dbh_cm": [
            "diameter", "dbh",
        ],
        "height_m": [
            "tinggi",
        ],
        "volume_m3_tree": [
            "volume",
        ],
    }

    metric_labels = {
        "dbh_cm": "Diameter pohon",
        "height_m": "Tinggi pohon",
        "volume_m3_tree": "Volume per pohon",
    }

    metric_units = {
        "dbh_cm": "cm",
        "height_m": "m",
        "volume_m3_tree": "m³",
    }

    selected_metric = None
    aliases = sorted(
        [
            (alias, metric)
            for metric, items in metric_aliases.items()
            for alias in items
        ],
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for alias, metric in aliases:
        if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", q):
            selected_metric = metric
            break

    # Jika metrik tidak disebutkan, tampilkan ketiganya.
    metrics_to_show = (
        [selected_metric]
        if selected_metric
        else ["height_m", "dbh_cm", "volume_m3_tree"]
    )

    lines = [
        f"Hasil prediksi {model.replace('_', ' ').upper()}",
        f"Periode: {result_row['growth_period']} "
        f"({result_row['age_months']} bulan)",
    ]

    if regional_id:
        lines.append(f"Wilayah: {regional_id}")
    if compartment_id is not None:
        lines.append(f"ID kompartemen: {compartment_id}")
    if species:
        lines.append(f"Spesies: {species}")
    if clone:
        lines.append(f"Klon: {clone}")

    lines.append("")

    for metric in metrics_to_show:
        value = result_row.get(metric)

        if value is None:
            lines.append(
                f"{metric_labels[metric]}: tidak tersedia dari model."
            )
        else:
            lines.append(
                f"{metric_labels[metric]}: "
                f"{float(value):.4f} {metric_units[metric]}"
            )

    lines.extend([
        "",
        "Angka di atas merupakan hasil model untuk cakupan data terpilih, "
        "bukan jaminan pertumbuhan aktual.",
    ])

    return {
        "success": True,
        "answer": "\n".join(lines),
        "mode": "python_local",
        "data": {
            "model": model,
            "period": result_row["growth_period"],
            "age_months": result_row["age_months"],
            "filters": {
                "regional_id": regional_id,
                "compartment_id": compartment_id,
                "species": species,
                "clone": clone,
            },
            "prediction": {
                metric: result_row.get(metric)
                for metric in metrics_to_show
            },
        },
    }


# ============================================================
# AI ASSISTANT ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â CHAT ENDPOINT
# ============================================================




@app.post("/ai/chat")
def ai_chat(request: ChatRequest):
    """Chat analisis dataset lokal menggunakan Python, tanpa API eksternal."""

    import re

    
    question = request.question.strip()

    # Jalur khusus laporan MAI untuk Chat Assistant.
    if re.search(
        r"\bMAI\b|mean annual increment|increment tahunan",
        question,
        re.I,
    ):
        years = sorted({
            int(value)
            for value in re.findall(
                r"\b(?:19|20)\d{2}\b",
                question,
            )
        })
        year_start = years[0] if years else None
        year_end = years[-1] if years else None

        report = mai_report(
            planting_year_start=year_start,
            planting_year_end=year_end,
            regional_id=None,
            compartment_id=None,
            species=None,
            clone=None,
        )

        if report.get("status") != "ok":
            return {
                "success": True,
                "answer": report.get(
                    "message",
                    "Data MAI tidak tersedia untuk filter tersebut.",
                ),
                "mode": "mai_report",
                "data": report,
            }

        summary = report["summary"]
        answer = (
            "Berikut ringkasan laporan MAI berdasarkan data pengukuran aktual. "
            f"Jumlah kombinasi kompartemen-kohort: "
            f"{summary['cohort_compartment_count']}. "
            f"Luas kohort: {summary['cohort_area_ha']:.2f} ha. "
            f"MAI tertimbang: "
            f"{summary['mai_weighted_m3_ha_year']:.3f} m3/ha/tahun. "
            f"Status Baik: {summary['good_count']}; "
            f"di bawah ambang batas: {summary['below_threshold_count']}. "
            "Laporan PDF MAI khusus tersedia melalui tombol unduhan."
        )

        return {
            "success": True,
            "answer": answer,
            "mode": "mai_report",
            "data": report,
            "chart": None,
        }

    # Resolusi konteks percakapan dengan prioritas pertanyaan terbaru.
    question = request.question.strip()
    current_lower = question.lower()

    period_pattern = (
        r"(?<![a-z0-9])p(6|12|18|24|30|36|42|48|54|60|66)"
        r"(?![a-z0-9])"
    )
    metric_patterns = (
        (r"\b(?:volume_m3_tree|volume per pohon|volume pohon|volume)\b",
         "volume pohon"),
        (r"\b(?:dbh_cm|dbh|diameter batang|diameter pohon|diameter)\b",
         "diameter pohon"),
        (r"\b(?:height_m|height|tinggi pohon|tinggi)\b",
         "tinggi pohon"),
        (r"\b(?:disease_score|disease score|skor penyakit)\b",
         "skor penyakit"),
        (r"\b(?:condition_score|condition score|skor kondisi)\b",
         "skor kondisi"),
        (r"\b(?:trees_per_ha|trees per ha|kerapatan pohon)\b",
         "kerapatan pohon"),
    )
    group_patterns = (
        (r"\bberdasarkan\s+(?:jenis\s+)?(?:klon|clone)\b", "klon"),
        (r"\bberdasarkan\s+(?:spesies|species)\b", "spesies"),
        (r"\bberdasarkan\s+(?:compartment|kompartemen|comp)\b",
         "kompartemen"),
        (r"\bberdasarkan\s+(?:wilayah|regional)\b", "wilayah"),
    )

    explicit_period = re.search(period_pattern, question, re.I)
    explicit_phi = re.search(r"(?<!\w)phi(?!\w)", question, re.I)
    current_metric = next(
        (label for pattern, label in metric_patterns
         if re.search(pattern, current_lower, re.I)),
        None,
    )
    current_group = next(
        (label for pattern, label in group_patterns
         if re.search(pattern, current_lower, re.I)),
        None,
    )

    unrelated_question = re.search(
        r"\b(?:apa itu|jelaskan|definisi)\s+"
        r"(?:pinns|random forest|machine learning)\b",
        current_lower,
    )
    followup = re.search(
        r"\b(?:ganti|ubah|sekarang|lalu|kalau|lanjut|"
        r"berdasarkan|pengelompokan|kelompokkan|yang sama)\b",
        current_lower,
    )

    if not unrelated_question and followup:
        previous_questions = [
            str(item.get("content", "")).strip()
            for item in reversed(request.history)
            if item.get("role") == "user"
            and str(item.get("content", "")).strip()
        ]

        previous_questions = [
            previous for previous in previous_questions
            if re.search(
                r"\b(?:bandingkan|perbandingan|pertumbuhan|growth|"
                r"volume|diameter|dbh|tinggi|height|spesies|species|"
                r"klon|clone|kompartemen|compartment|regional|wilayah)\b",
                previous,
                re.I,
            )
        ]

        additions = []

        if not explicit_period and not explicit_phi:
            for previous in previous_questions:
                old_period = re.search(period_pattern, previous, re.I)
                old_phi = re.search(r"(?<!\w)phi(?!\w)", previous, re.I)
                if old_period:
                    additions.append(f"periode P{old_period.group(1)}")
                    break
                if old_phi:
                    additions.append("periode PHI")
                    break

        if current_metric is None:
            for previous in previous_questions:
                old_metric = next(
                    (label for pattern, label in metric_patterns
                     if re.search(pattern, previous, re.I)),
                    None,
                )
                if old_metric:
                    additions.append(f"metrik {old_metric}")
                    break

        if current_group is None:
            for previous in previous_questions:
                old_group = next(
                    (label for pattern, label in group_patterns
                     if re.search(pattern, previous, re.I)),
                    None,
                )
                if old_group:
                    additions.append(f"berdasarkan {old_group}")
                    break

        if additions:
            if not re.search(
                r"\b(?:bandingkan|perbandingan|dibandingkan|dibanding)\b",
                current_lower,
            ):
                question = "Bandingkan " + question
            question += " " + " ".join(additions)

    q = question.lower()

    # Tangani pertanyaan konsep agar tidak masuk ke analisis dataset.
    if re.search(
        r"\b(?:apa itu|jelaskan|definisi)\s+pinns\b",
        question,
        re.IGNORECASE,
    ):
        return {
            "success": True,
            "answer": (
                "PINNs (Physics-Informed Neural Networks) adalah jaringan "
                "saraf yang menggabungkan data dengan aturan atau persamaan "
                "fisika sebagai bagian dari proses pembelajaran. "
                "Untuk analisis pertumbuhan pohon, pendekatan ini dapat "
                "digunakan untuk mempelajari pola pertumbuhan dari data "
                "sekaligus mempertimbangkan hubungan matematis yang relevan. "
                "Kualitas hasilnya tetap bergantung pada data, persamaan "
                "yang digunakan, dan proses validasi model."
            ),
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }

    # Jawaban pengetahuan umum tentang Random Forest.
    if re.search(
        r"\b(?:apa itu|jelaskan|definisi)\s+random\s+forest\b",
        question,
        re.IGNORECASE,
    ):
        return {
            "success": True,
            "answer": (
                "Random Forest adalah metode machine learning yang "
                "menggabungkan banyak pohon keputusan untuk menghasilkan "
                "prediksi. Untuk tugas regresi, prediksi akhir umumnya "
                "merupakan rata-rata prediksi dari pohon-pohon tersebut. "
                "Dalam analisis pertumbuhan pohon, metode ini dapat "
                "digunakan untuk memprediksi target seperti DBH, tinggi, "
                "atau volume berdasarkan fitur yang tersedia. "
                "Akurasi model tetap perlu dievaluasi menggunakan data "
                "pengujian yang sesuai."
            ),
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }

    # Normalisasi ringan untuk FAQ perbandingan metode.
    comparison_faq_text = question.lower()
    comparison_faq_text = re.sub(
        r"\brndom\b", "random", comparison_faq_text
    )
    comparison_faq_text = re.sub(
        r"\bperbeda+a+n\b", "perbedaan", comparison_faq_text
    )

    comparison_pattern = (
        r"\b(?:apa\s+(?:itu\s+)?(?:perbedaan|bedanya)|"
        r"bedanya|perbedaan)\s+"
        r"(?:antara\s+)?"
        r"(?:pinns|physics-informed neural networks)"
        r"\s+(?:dan|dengan|vs\.?)\s+random\s+forest\b"
        r"|\b(?:apa\s+(?:itu\s+)?(?:perbedaan|bedanya)|"
        r"bedanya|perbedaan)\s+"
        r"random\s+forest\s+(?:dan|dengan|vs\.?)\s+"
        r"(?:pinns|physics-informed neural networks)\b"
    )

    if re.search(comparison_pattern, comparison_faq_text, re.IGNORECASE):
        return {
            "success": True,
            "answer": (
                "PINNs dan Random Forest memiliki pendekatan berbeda. "
                "Random Forest menggabungkan prediksi dari banyak pohon "
                "keputusan. PINNs menggunakan jaringan saraf dan dapat "
                "memasukkan persamaan atau batasan fisika ke dalam proses "
                "pelatihan melalui komponen physics loss. "
                "Random Forest tidak secara inheren menerapkan batasan "
                "fisika tersebut. Pada dashboard pertumbuhan pohon, "
                "perbandingan kinerja kedua metode harus didasarkan pada "
                "data pengujian, target prediksi, dan metrik evaluasi "
                "yang sama. Tidak ada metode yang otomatis selalu lebih baik."
            ),
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }



    # FAQ_INTERPRETASI_R2_V1
    r2_question = question.lower()

    r2_high_question = (
        re.search(r"\b(?:r\s*2|R2)\b", r2_question)
        and re.search(r"\b(?:tinggi|besar)\b", r2_question)
        and re.search(r"\b(?:bagus|baik|akurat|selalu)\b", r2_question)
    )

    if r2_high_question:
        return {
            "success": True,
            "answer": (
                "Tidak selalu. Nilai R2 yang tinggi menunjukkan bahwa model "
                "dapat menjelaskan sebagian besar variasi target pada data "
                "yang dievaluasi, tetapi tidak otomatis berarti prediksinya "
                "akurat untuk semua kebutuhan. Periksa juga MAE dan RMSE "
                "untuk mengetahui besar kesalahan prediksi, pastikan evaluasi "
                "dilakukan pada data uji yang sesuai, dan waspadai kebocoran "
                "data atau pembagian data yang tidak mewakili penggunaan nyata. "
                "Untuk pertumbuhan pohon, periksa metrik pada target dan "
                "satuan masing-masing, misalnya DBH, tinggi, atau volume. "
                "Kesimpulan tentang kualitas model harus didasarkan pada "
                "hasil evaluasi aktual, bukan R2 saja."
            ),
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }

    # FAQ_METRIK_EVALUASI_V1
    # Penjelasan umum metrik evaluasi, bukan klaim hasil model aktual.
    metric_question = question.lower()

    metric_compare = re.search(
        r"\b(?:apa\s+perbedaan|perbedaan|bedanya|"
        r"bandingkan|beda)\b",
        metric_question,
    ) and re.search(
        r"\b(?:mae|rmse|r2|R2|r\s*2)\b",
        metric_question,
    )

    metric_match = re.search(
        r"\b(mae|rmse|r2|R2|r\s*2)\b",
        metric_question,
    )

    metric_explainer = re.search(
        r"\b(?:apa\s+itu|jelaskan|definisi|arti|"
        r"maksud|bagaimana\s+(?:cara\s+)?membaca|"
        r"cara\s+menginterpretasikan)\b",
        metric_question,
    )

    if metric_compare:
        metric_answer = (
            "MAE, RMSE, dan R2 mengukur aspek kinerja model regresi "
            "yang berbeda. MAE adalah rata-rata nilai absolut selisih "
            "prediksi dan nilai aktual; satuannya sama dengan target. "
            "RMSE adalah akar dari rata-rata kuadrat selisih, sehingga "
            "lebih sensitif terhadap kesalahan besar; satuannya juga "
            "sama dengan target. R2 menunjukkan seberapa baik model "
            "menjelaskan variasi target dibandingkan baseline rata-rata. "
            "R2 yang mendekati 1 umumnya menunjukkan kecocokan yang "
            "lebih baik, tetapi R2 dapat bernilai negatif pada data uji. "
            "MAE dan RMSE yang lebih rendah umumnya lebih baik. "
            "Bandingkan metrik pada data uji, target, dan pembagian data "
            "yang sama. Tidak ada ambang universal untuk menyatakan model "
            "bagus; pertimbangkan juga satuan target dan kebutuhan penggunaan."
        )
    elif metric_match and metric_explainer:
        metric_name = metric_match.group(1).replace(" ", "").upper()
        if metric_name == "R2":
            metric_name = "R2"
        elif metric_name in ("R2", "R 2"):
            metric_name = "R2"

        metric_definitions = {
            "MAE": (
                "MAE (Mean Absolute Error) adalah rata-rata nilai absolut "
                "selisih antara prediksi dan nilai aktual. Nilai MAE yang "
                "lebih rendah umumnya lebih baik, dan satuannya sama "
                "dengan target prediksi."
            ),
            "RMSE": (
                "RMSE (Root Mean Squared Error) adalah akar kuadrat dari "
                "rata-rata kuadrat selisih prediksi dan nilai aktual. "
                "RMSE memberi penalti lebih besar pada kesalahan besar. "
                "Nilai lebih rendah umumnya lebih baik, dan satuannya "
                "sama dengan target prediksi."
            ),
            "R2": (
                "R2 (koefisien determinasi) menunjukkan seberapa baik model "
                "menjelaskan variasi nilai target dibandingkan baseline "
                "yang menggunakan rata-rata target. Nilai mendekati 1 "
                "umumnya menunjukkan kecocokan lebih baik, tetapi R2 dapat "
                "bernilai negatif pada data uji. R2 perlu dinilai bersama "
                "MAE atau RMSE dan konteks data."
            ),
        }
        metric_answer = metric_definitions[metric_name]
    else:
        metric_answer = None

    asks_rf_evaluation_early = (
        any(term in metric_question for term in (
            "random forest", "random_forest", "randomforest",
        ))
        and any(term in metric_question for term in (
            "evaluasi", "kualitas", "akurasi", "akurat",
            "mae", "rmse", "mape", "r2", "R2", "r^2",
            "hasil model", "hasil test",
        ))
        and not any(term in metric_question for term in (
            "pinns", "physics-informed",
        ))
    )

    if metric_answer is not None and not asks_rf_evaluation_early and not (
            "pinns" in metric_question
            and re.search(
                r"\b(?:r\s*2|R2|r\^2|r-squared)\b",
                metric_question,
                re.IGNORECASE,
            )
            and re.search(
                r"\b(?:evaluasi|test|hasil)\b",
                metric_question,
                re.IGNORECASE,
            )
        ):
        return {
            "success": True,
            "answer": metric_answer,
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }

    # Prioritaskan permintaan prediksi.

    # Prioritaskan permintaan prediksi.

    
    
    # FAQ_PERBANDINGAN_RF_PINNS_V1
    comparison_question = question.lower().strip()

    asks_rf_and_pinns = (
        any(term in comparison_question for term in (
            "random forest", "random_forest", "randomforest",
        ))
        and any(term in comparison_question for term in (
            "pinns", "physics-informed",
        ))
    )

    asks_model_comparison = any(term in comparison_question for term in (
        "bandingkan", "perbandingan", "dibandingkan",
        "bandingkanlah", "mana lebih baik", "lebih unggul",
        "komparasi", "compare", "comparison",
    ))

    if asks_rf_and_pinns and asks_model_comparison:
        comparison_answer = (
            "PERBANDINGAN RANDOM FOREST DAN PINNs V2\n\n"
            "Berdasarkan metrik evaluasi pada data test yang tersedia, "
            "Random Forest memperoleh hasil lebih baik pada ketiga target "
            "yang dibandingkan. Kesimpulan ini terbatas pada evaluasi "
            "tersebut, bukan jaminan bahwa model akan selalu lebih baik "
            "pada semua dataset.\n\n"
            "1. DBH (cm)\n"
            "Random Forest: MAE 0,4234 cm; RMSE 0,5338 cm; "
            "R2 0,9881; MAPE 3,3188%.\n"
            "PINNs V2: MAE 1,1031 cm; RMSE 1,3598 cm; "
            "R2 0,9230; MAPE 12,6193%.\n\n"
            "2. Tinggi pohon (m)\n"
            "Random Forest: MAE 0,5523 m; RMSE 0,6970 m; "
            "R2 0,9884; MAPE 3,1253%.\n"
            "PINNs V2: MAE 1,5767 m; RMSE 1,9791 m; "
            "R2 0,9066; MAPE 13,8617%.\n\n"
            "3. Volume (m^3/pohon)\n"
            "Random Forest: MAE 0,02648 m^3/pohon; "
            "RMSE 0,03787 m^3/pohon; R2 0,8689; MAPE 15,5522%.\n"
            "PINNs V2: MAE 0,03323 m^3/pohon; "
            "RMSE 0,04501 m^3/pohon; R2 0,8148; MAPE 23,7649%.\n\n"
            "INTERPRETASI\n"
            "MAE, RMSE, dan MAPE yang lebih rendah menunjukkan "
            "kesalahan prediksi yang lebih kecil, sedangkan R2 yang "
            "lebih tinggi menunjukkan kecocokan terhadap variasi target "
            "yang lebih baik pada evaluasi ini. Pada ketiga target, "
            "Random Forest memiliki MAE, RMSE, dan MAPE lebih rendah "
            "serta R2 lebih tinggi daripada PINNs V2.\n\n"
            "Catatan: jumlah baris test untuk DBH dan tinggi adalah "
            "43.740 per target, sedangkan untuk volume adalah 35.640. "
            "Perbandingan ini menggunakan hasil evaluasi yang tersedia; "
            "pastikan pembagian data test dan prosedur evaluasi kedua "
            "model setara sebelum menarik kesimpulan yang lebih luas."
        )

        return {
            "success": True,
            "answer": comparison_answer,
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }
    # FAQ_KELEMAHAN_RANDOM_FOREST_V1
    rf_weakness_question = question.lower().strip()

    asks_rf_weakness = (
        any(term in rf_weakness_question for term in (
            "random forest",
            "random_forest",
            "randomforest",
        ))
        and any(term in rf_weakness_question for term in (
            "kelemahan",
            "kekurangan",
            "keterbatasan",
            "risiko model",
            "risiko prediksi",
            "apa yang perlu diperbaiki",
        ))
        and not any(term in rf_weakness_question for term in (
            "pinns",
            "physics-informed",
        ))
    )

    if asks_rf_weakness:
        return {
            "success": True,
            "answer": (
                "ANALISIS KELEMAHAN RANDOM FOREST\n\n"
                "Berdasarkan hasil evaluasi pada data test, "
                "target volume merupakan bagian yang paling "
                "perlu mendapat perhatian dibandingkan DBH "
                "dan tinggi pohon.\n\n"
                "1. Prediksi volume pohon\n"
                "R2: 0,8689\n"
                "MAPE: 15,5522%\n"
                "MAE: 0,02648 m^3/pohon\n"
                "RMSE: 0,03787 m^3/pohon\n\n"
                "2. Perbandingan dengan target lain\n"
                "DBH memiliki R2 0,9881 dan MAPE 3,3188%. "
                "Tinggi pohon memiliki R2 0,9884 dan "
                "MAPE 3,1253%. Dengan demikian, metrik "
                "volume menunjukkan kecocokan yang lebih "
                "rendah dan kesalahan persentase yang lebih "
                "tinggi pada evaluasi ini.\n\n"
                "3. Batas interpretasi\n"
                "Hasil ini belum menjelaskan penyebab "
                "kesalahan volume. Untuk mengetahui penyebab "
                "dan kelompok pohon yang paling sulit "
                "diprediksi, diperlukan analisis residual "
                "berdasarkan umur, wilayah, spesies, atau "
                "kelompok data lain yang tersedia.\n\n"
                "Kesimpulan: kelemahan yang terlihat dari "
                "metrik evaluasi saat ini terutama berada "
                "pada prediksi volume. Ini merupakan temuan "
                "dari data test yang tersedia, bukan bukti "
                "bahwa Random Forest selalu lemah untuk "
                "prediksi volume."
            ),
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }


    # FAQ_EVALUASI_RANDOM_FOREST_V1
    # Menjawab evaluasi semua target atau target tertentu.
    rf_evaluation_question = question.lower().strip()

    asks_rf_evaluation = (
        any(term in rf_evaluation_question for term in (
            "random forest",
            "random_forest",
            "randomforest",
        ))
        and any(term in rf_evaluation_question for term in (
            "evaluasi",
            "kualitas",
            "akurasi",
            "akurat",
            "mae",
            "rmse",
            "mape",
            "r2",
            "R2",
            "r^2",
            "hasil model",
            "hasil test",
        ))
        and not any(term in rf_evaluation_question for term in (
            "pinns",
            "physics-informed",
        ))
    )

    if asks_rf_evaluation:
        q = rf_evaluation_question

        asks_dbh = bool(re.search(
            r"\b(?:dbh|diameter)\b",
            q,
        ))
        asks_height = any(term in q for term in (
            "tinggi",
            "height",
        ))
        asks_volume = "volume" in q

        target_count = sum((
            asks_dbh,
            asks_height,
            asks_volume,
        ))

        if target_count == 1:
            if asks_dbh:
                target_answer = (
                    "EVALUASI RANDOM FOREST - DBH\n\n"
                    "MAE: 0,4234 cm\n"
                    "RMSE: 0,5338 cm\n"
                    "R2: 0,9881\n"
                    "MAPE: 3,3188%\n"
                    " Jumlah data test: 43.740 observasi.\n\n"
                    "Interpretasi: R2 menunjukkan kecocokan "
                    "yang tinggi pada data test. MAE dan RMSE "
                    "menunjukkan besarnya kesalahan prediksi "
                    "dalam satuan cm."
                )
            elif asks_height:
                target_answer = (
                    "EVALUASI RANDOM FOREST - TINGGI POHON\n\n"
                    "MAE: 0,5523 m\n"
                    "RMSE: 0,6970 m\n"
                    "R2: 0,9884\n"
                    "MAPE: 3,1253%\n"
                    " Jumlah data test: 43.740 observasi.\n\n"
                    "Interpretasi: R2 menunjukkan kecocokan "
                    "yang tinggi pada data test. MAE dan RMSE "
                    "menunjukkan besarnya kesalahan prediksi "
                    "dalam satuan meter."
                )
            else:
                target_answer = (
                    "EVALUASI RANDOM FOREST - VOLUME POHON\n\n"
                    "MAE: 0,02648 m^3/pohon\n"
                    "RMSE: 0,03787 m^3/pohon\n"
                    "R2: 0,8689\n"
                    "MAPE: 15,5522%\n"
                    "Jumlah data test: 35.640 observasi.\n"
                    "Rentang umur evaluasi: P18-P66.\n\n"
                    "Interpretasi: dibandingkan target DBH "
                    "dan tinggi, volume memiliki R2 lebih "
                    "rendah dan MAPE lebih tinggi. Karena itu, "
                    "hasil prediksi volume perlu diperiksa "
                    "lebih lanjut sesuai kebutuhan penggunaan."
                )
        else:
            target_answer = (
                "RINGKASAN EVALUASI RANDOM FOREST\n\n"
                "Berdasarkan "
                "results/random_forest/test_metrics.csv, "
                "hasil evaluasi pada data test adalah:\n\n"
                "1. DBH (diameter batang)\n"
                "MAE: 0,4234 cm; RMSE: 0,5338 cm; "
                "R2: 0,9881; MAPE: 3,3188%. "
                " Jumlah data test: 43.740 observasi.\n\n"
                "2. Tinggi pohon\n"
                "MAE: 0,5523 m; RMSE: 0,6970 m; "
                "R2: 0,9884; MAPE: 3,1253%. "
                " Jumlah data test: 43.740 observasi.\n\n"
                "3. Volume pohon\n"
                "MAE: 0,02648 m^3/pohon; "
                "RMSE: 0,03787 m^3/pohon; "
                "R2: 0,8689; MAPE: 15,5522%. "
                "Jumlah data test: 35.640 observasi, "
                "dengan rentang umur P18-P66.\n\n"
                "DBH dan tinggi memiliki R2 sekitar 0,988, "
                "sedangkan volume memiliki R2 sekitar 0,869 "
                "dan MAPE lebih tinggi. MAE dan RMSE tidak "
                "boleh dibandingkan langsung antar-target "
                "tanpa memperhatikan satuannya."
            )

        return {
            "success": True,
            "answer": target_answer,
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }
    # FAQ_KUALITAS_PINNS_V1
    pinns_quality_question = question.lower()

    
    asks_pinns_quality = (
        "pinns" in pinns_quality_question
        and any(word in pinns_quality_question for word in (
            "kualitas model",
            "evaluasi",
            "akurasi",
            "akurat",
            "mae",
            "rmse",
            "r2",
            "r^2",
            "generalisasi",
            "ekstrapolasi",
        ))
        and any(word in pinns_quality_question for word in (
            "bagaimana",
            "jelaskan",
            "nilai",
            "hasil",
            "kualitas",
            "berdasarkan",
            "interpretasi",
        ))
    )


    
    # FAQ_AKURASI_PINNS_V1
        
    asks_pinns_accuracy = (
        asks_pinns_quality
        and any(
            word in pinns_quality_question
            for word in (
                "akurat",
                "akurasi",
                "tingkat akurasi",
                "seberapa baik",
            )
        )
        and not any(
            word in pinns_quality_question
            for word in (
                "perbaiki",
                "diperbaiki",
                "perbaikan",
                "ditingkatkan",
                "meningkatkan",
                "lebih akurat",
                "terlebih dahulu",
                "langkah pertama",
                "apa yang sebaiknya",
                "rekomendasi",
            )
        )
    )

    
    # FAQ_EVALUASI_PINNS_SEMUA_TARGET_V1
    # Jawaban ringkasan lengkap hanya untuk evaluasi seluruh target.
    asks_all_target_evaluation = (
        "pinns" in pinns_quality_question
        and any(term in pinns_quality_question for term in (
            "evaluasi",
            "kualitas",
            "akurasi",
            "akurat",
            "hasil model",
        ))
        and not any(term in pinns_quality_question for term in (
            "tinggi pohon",
            "tinggi saja",
            "dbh saja",
            "diameter saja",
            "volume saja",
        ))
        and not re.search(
            r"\b(?:dbh|diameter|volume|tinggi)\b",
            pinns_quality_question,
        )
        and not any(term in pinns_quality_question for term in (
            "kelemahan",
            "kekurangan",
            "perbaiki",
            "diperbaiki",
            "perbaikan",
            "rekomendasi",
        ))
    )

    if asks_all_target_evaluation:
        return {
            "success": True,
            "answer": (
                "RINGKASAN EVALUASI PINNs V2\n\n"
                "Berdasarkan metrik evaluasi data test yang tersedia:\n\n"
                "1. DBH (diameter batang)\n"
                "MAE: 1,1031 cm; RMSE: 1,3598 cm; "
                "R2: 0,9230; MAPE: 12,62%.\n\n"
                "2. Tinggi pohon\n"
                "MAE: 1,5767 m; RMSE: 1,9791 m; "
                "R2: 0,9066; MAPE: 13,86%.\n\n"
                "3. Volume pohon\n"
                "MAE: 0,0332333 m^3/pohon; RMSE: 0,0450056 m^3/pohon; "
                "R2: 0,8148236; MAPE: 23,7649%.\n\n"
                "INTERPRETASI\n"
                "Di antara ketiga target, DBH memiliki R2 tertinggi "
                "dan MAPE terendah. Tinggi juga menunjukkan kecocokan "
                "yang relatif tinggi. Volume memiliki R2 terendah "
                "dan MAPE tertinggi, sehingga perlu mendapat perhatian "
                "lebih lanjut. Namun, perbandingan MAE langsung "
                "antar-target tidak tepat karena satuan ketiganya "
                "berbeda.\n\n"
                "Jumlah data test yang tercatat adalah 43.740 "
                "observasi untuk DBH dan tinggi, serta 35.640 "
                "observasi untuk volume. Kualitas operasional tetap "
                "perlu dinilai berdasarkan kebutuhan penggunaan "
                "dan pemeriksaan kesalahan per kelompok."
            ),
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }
    
    if asks_pinns_accuracy:
        return {
            "success": True,
            "answer": (
                "PENILAIAN AKURASI MODEL PINNs V2 - TINGGI POHON\n\n"
                "Berdasarkan hasil evaluasi pada 43.740 observasi "
                "data uji, model PINNs V2 menunjukkan kinerja yang "
                "baik pada skenario pengujian tersebut.\n\n"
                "MAE: 1,5767 m. Rata-rata selisih absolut antara "
                "tinggi prediksi dan tinggi aktual sekitar 1,58 m.\n\n"
                "RMSE: 1,9791 m. Nilai ini lebih besar daripada "
                "MAE, yang menunjukkan bahwa kesalahan besar "
                "memiliki pengaruh lebih kuat terhadap metrik ini.\n\n"
                "R2: 0,9066. Model menjelaskan sekitar 90,66 persen "
                "variasi tinggi pohon pada data uji menurut "
                "definisi koefisien determinasi.\n\n"
                "KESIMPULAN\n"
                "Hasil ini mendukung bahwa model memiliki kecocokan "
                "yang baik pada data uji. Namun, apakah tingkat "
                "kesalahannya dapat diterima untuk penggunaan "
                "operasional bergantung pada kebutuhan dan toleransi "
                "kesalahan. Nilai R2 yang baik juga tidak otomatis "
                "membuktikan kemampuan prediksi pada umur pohon "
                "di luar rentang pelatihan.\n\n"
                "Untuk menilai kemampuan tersebut, evaluasi perlu "
                "dilengkapi dengan pemeriksaan kesalahan per periode "
                "pertumbuhan dan pengujian pada rentang umur "
                "yang belum dilihat model."
            ),
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }

    
    # FAQ_KELEMAHAN_PINNS_V1
    asks_pinns_weakness = (
        asks_pinns_quality
        and any(word in pinns_quality_question for word in (
            "kelemahan",
            "kekurangan",
            "keterbatasan",
            "risiko model",
            "risiko prediksi",
            "apa yang perlu diperbaiki",
            "apa yang harus diperbaiki",
        ))
    )

    if asks_pinns_weakness:
        return {
            "success": True,
            "answer": (
                "ANALISIS KELEMAHAN MODEL PINNs V2 - TINGGI POHON\n\n"
                "Berdasarkan hasil evaluasi yang tersedia, ada "
                "beberapa aspek yang perlu diperhatikan.\n\n"
                "1. Kesalahan prediksi masih perlu diperhatikan\n"
                "MAE sebesar 1,5767 m menunjukkan bahwa rata-rata "
                "selisih absolut prediksi dan tinggi aktual sekitar "
                "1,58 m. RMSE sebesar 1,9791 m menunjukkan bahwa "
                "kesalahan besar turut memengaruhi hasil evaluasi. "
                "Apakah kesalahan ini dapat diterima bergantung "
                "pada kebutuhan penggunaan model.\n\n"
                "2. Kinerja belum tentu merata pada setiap periode\n"
                "Metrik keseluruhan belum menunjukkan periode umur, "
                "spesies, regional, atau kelompok pohon yang memiliki "
                "kesalahan terbesar. Evaluasi per kelompok diperlukan "
                "untuk mengetahui bagian mana yang perlu diperbaiki.\n\n"
                "3. Konsistensi biologis belum dapat disimpulkan "
                "dari metrik saja\n"
                "MAE, RMSE, dan R2 tidak cukup untuk membuktikan "
                "bahwa bentuk kurva pertumbuhan selalu realistis. "
                "Kurva prediksi perlu dibandingkan dengan data "
                "aktual berdasarkan umur atau periode pertumbuhan.\n\n"
                "4. Kemampuan ekstrapolasi belum terbukti\n"
                "R2 sebesar 0,9066 menunjukkan kecocokan yang baik "
                "pada data uji yang dievaluasi. Namun, hasil ini "
                "tidak otomatis membuktikan bahwa model akurat "
                "untuk umur pohon atau kondisi yang berada di luar "
                "rentang data pelatihan.\n\n"
                "5. Pengaruh physics loss perlu dievaluasi\n"
                "Penggunaan physics loss tidak otomatis menjamin "
                "prediksi biologis yang benar. Perlu diperiksa "
                "apakah batasan yang digunakan sesuai dengan "
                "pertumbuhan pohon dan bagaimana perubahan bobot "
                "physics loss memengaruhi hasil prediksi.\n\n"
                "REKOMENDASI PERBAIKAN\n"
                "Prioritaskan evaluasi residual per periode dan "
                "kelompok pohon, pemeriksaan kurva terhadap data "
                "aktual, serta pengujian pada rentang umur yang "
                "belum digunakan saat pelatihan. Hasilnya dapat "
                "menunjukkan apakah masalah utama terletak pada "
                "kesalahan prediksi, kelompok data tertentu, atau "
                "kemampuan ekstrapolasi.\n\n"
                "Catatan: hasil evaluasi yang tersedia belum "
                "menentukan penyebab pasti kesalahan tersebut. "
                "Penyebabnya perlu dikonfirmasi melalui pengujian "
                "tambahan."
            ),
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }

    
    # FAQ_REKOMENDASI_PINNS_V1
    asks_pinns_improvement = (
        asks_pinns_quality
        and any(word in pinns_quality_question for word in (
            "perbaiki",
            "diperbaiki",
            "perbaikan",
            "ditingkatkan",
            "meningkatkan",
            "lebih akurat",
            "lebih baik",
            "terlebih dahulu",
            "langkah pertama",
            "apa yang sebaiknya",
            "rekomendasi",
        ))
    )

    if asks_pinns_improvement:
        return {
            "success": True,
            "answer": (
                "REKOMENDASI PERBAIKAN MODEL PINNs V2 - TINGGI POHON\n\n"
                "Prioritas pertama bukan langsung melatih ulang model, "
                "melainkan mencari tahu di mana dan bagaimana kesalahan "
                "prediksi terjadi.\n\n"
                "1. Periksa residual dan kesalahan per kelompok\n"
                "Gunakan selisih antara tinggi prediksi dan tinggi "
                "aktual. Hitung MAE dan RMSE berdasarkan periode "
                "pertumbuhan, umur pohon, spesies, clone, dan regional "
                "jika kolom tersebut tersedia. Langkah ini membantu "
                "mengidentifikasi kelompok yang paling membutuhkan "
                "perbaikan.\n\n"
                "2. Periksa pola kesalahan\n"
                "Buat grafik tinggi aktual versus prediksi dan grafik "
                "residual terhadap umur atau periode. Periksa apakah "
                "model cenderung terlalu tinggi, terlalu rendah, atau "
                "memiliki kesalahan besar pada kelompok tertentu.\n\n"
                "3. Evaluasi fungsi physics loss\n"
                "Setelah pola kesalahan diketahui, periksa rumus "
                "batasan fisika dan skala setiap komponen loss. Jika "
                "ada alasan yang didukung hasil evaluasi, uji beberapa "
                "bobot physics loss secara terkontrol. Jangan langsung "
                "mengubah bobot tanpa pembanding.\n\n"
                "4. Bandingkan eksperimen secara adil\n"
                "Gunakan pembagian data, preprocessing, dan metrik "
                "evaluasi yang sama untuk model awal dan model hasil "
                "perbaikan. Catat MAE, RMSE, R2, serta hasil per "
                "kelompok. Pilih perubahan berdasarkan hasil data "
                "validasi, lalu evaluasi final pada data uji yang "
                "tidak digunakan untuk memilih model.\n\n"
                "PRIORITAS UTAMA\n"
                "Mulai dengan analisis residual per periode dan "
                "kelompok pohon. Hasilnya akan memberi dasar untuk "
                "menentukan apakah perlu memperbaiki data, arsitektur, "
                "proses pelatihan, atau bobot physics loss.\n\n"
                "Catatan: MAE 1,5767 m merupakan hasil evaluasi saat "
                "ini. Belum ada bukti bahwa perubahan tertentu pasti "
                "menurunkan MAE sebelum eksperimen dilakukan."
            ),
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }



    if asks_pinns_quality:
        return {
            "success": True,
            "answer": (
                "EVALUASI KUALITAS MODEL PINNs V2 - TINGGI POHON\n\n"
                "Pada data uji sebanyak 43.740 observasi, model memperoleh "
                "MAE 1,5767 m, RMSE 1,9791 m, dan R2 0,9066.\n\n"
                "1. Ketepatan prediksi\n"
                "MAE 1,5767 m berarti selisih absolut antara prediksi dan "
                "tinggi aktual rata-rata sekitar 1,58 m atau 157,67 cm. "
                "RMSE 1,9791 m lebih tinggi daripada MAE, sehingga kesalahan "
                "besar memberi pengaruh lebih kuat pada nilai evaluasi. "
                "Residual perlu diperiksa untuk menemukan periode atau "
                "kelompok pohon dengan kesalahan terbesar.\n\n"
                "2. Konsistensi pertumbuhan biologis\n"
                "Selisih MAE dan RMSE saja tidak membuktikan bahwa kurva "
                "pertumbuhan konsisten secara biologis. Physics loss membantu "
                "model mengikuti batasan fisika yang dirumuskan, tetapi kurva "
                "prediksi tetap perlu dibandingkan dengan data aktual "
                "berdasarkan umur atau periode.\n\n"
                "3. Generalisasi dan ekstrapolasi\n"
                "R2 0,9066 menunjukkan bahwa model menjelaskan sekitar 90,66% "
                "variasi tinggi pohon pada data uji menurut definisi R2. "
                "Hasil ini mendukung kecocokan pada skenario pengujian tersebut, "
                "tetapi tidak otomatis membuktikan kemampuan ekstrapolasi. "
                "Generalisasi perlu dinilai menggunakan data uji yang sesuai, "
                "sedangkan ekstrapolasi perlu diuji pada rentang umur atau "
                "kondisi di luar data pelatihan.\n\n"
                "KESIMPULAN\n"
                "PINNs V2 menunjukkan kecocokan yang baik pada data uji. "
                "Namun, akurasi per periode, konsistensi biologis kurva, "
                "dan kemampuan ekstrapolasi masih perlu diverifikasi "
                "secara terpisah."
            ),

            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }

    # FAQ_KUALITAS_MODEL_V1
    # Pertanyaan evaluasi umum harus dijawab sebelum handler prediksi.
    model_quality_question = question.lower()

    asks_model_quality = (
        ("bagaimana" in model_quality_question
         and any(word in model_quality_question for word in
                 ("bagus", "baik", "akurat", "kualitas")))
        or any(phrase in model_quality_question for phrase in (
            "menilai model",
            "evaluasi model",
            "kualitas model",
            "model yang bagus",
            "model yang baik",
            "prediksi bagus atau tidak",
            "prediksi baik atau tidak",
            "mengukur akurasi model",
        ))
    )

    asks_for_actual_prediction = any(phrase in model_quality_question for phrase in (
        "prediksi tinggi",
        "prediksi diameter",
        "prediksi dbh",
        "prediksi volume",
        "berapa tinggi",
        "berapa diameter",
        "berapa volume",
        "hitung prediksi",
        "buat prediksi",
    ))

    if asks_model_quality and not asks_for_actual_prediction:
        return {
            "success": True,
            "answer": (
                "Kualitas model prediksi dinilai menggunakan data uji "
                "yang tidak digunakan saat pelatihan. Untuk regresi, "
                "periksa MAE dan RMSE-semakin rendah umumnya semakin baik-"
                "serta R2, yang umumnya semakin baik jika mendekati 1. "
                "Nilai tersebut harus dibaca bersama satuan dan rentang "
                "target, misalnya DBH (cm), tinggi (m), dan volume (m^3 "
                "per pohon). Pastikan pembagian data mewakili kondisi "
                "penggunaan nyata dan tidak mengalami kebocoran data. "
                "Bandingkan model pada data uji, target, dan pembagian "
                "data yang sama. Saya belum menyimpulkan kualitas model "
                "aktual karena pertanyaan ini belum menyertakan hasil "
                "evaluasi MAE, RMSE, dan R2 yang relevan."
            ),
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }

    # FAQ_ARSITEKTUR_DASHBOARD_V1
    architecture_question = question.lower()

    architecture_answers = [
        (
            r"\b(?:apa itu|jelaskan|fungsi)\s+frontend\b",
            "Frontend adalah bagian aplikasi yang berinteraksi langsung "
            "dengan pengguna, seperti halaman dashboard, formulir filter, "
            "tabel, dan grafik. Frontend mengirim permintaan ke backend "
            "ketika membutuhkan data atau hasil pemrosesan."
        ),
        (
            r"\b(?:apa itu|jelaskan|fungsi)\s+backend\b",
            "Backend adalah bagian aplikasi yang menangani logika "
            "pemrosesan, validasi permintaan, akses data, dan layanan "
            "yang dibutuhkan frontend. Pada proyek ini, backend "
            "menggunakan FastAPI berdasarkan kode api/main_new.py."
        ),
        (
            r"\b(?:apa itu|jelaskan|fungsi)\s+api\b",
            "API adalah antarmuka yang memungkinkan bagian aplikasi "
            "berkomunikasi melalui aturan permintaan dan respons. Dalam "
            "arsitektur frontend-backend, frontend dapat mengirim "
            "permintaan melalui API dan menerima data atau hasil "
            "pemrosesan dari backend."
        ),
        (
            r"\b(?:apa itu|jelaskan|fungsi)\s+database\b",
            "Database adalah sistem untuk menyimpan dan mengelola data "
            "agar dapat dicari, dibaca, ditambah, atau diperbarui. "
            "Dalam dashboard analitik, database dapat menjadi sumber "
            "data bagi backend. Jenis database dan cara koneksi yang "
            "benar-benar digunakan proyek perlu dikonfirmasi dari kode "
            "konfigurasi dan akses datanya."
        ),
    ]

    architecture_answer = None

    for architecture_pattern, architecture_text in architecture_answers:
        if re.search(architecture_pattern, architecture_question):
            architecture_answer = architecture_text
            break

    if architecture_answer is not None:
        return {
            "success": True,
            "answer": architecture_answer,
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }

    # FAQ_ALUR_PREDIKSI_V1
    prediction_flow_question = question.lower()

    prediction_flow_patterns = [
        (
            r"\b(?:bagaimana|jelaskan)\b.*"
            r"\b(?:alur|proses)\b.*\bprediksi\b",
            "Secara umum, alur prediksi dimulai ketika pengguna memilih "
            "parameter yang diperlukan. Frontend mengirim permintaan ke "
            "backend melalui API. Backend memvalidasi permintaan, "
            "menyiapkan input, memilih model yang sesuai, menjalankan "
            "prediksi, lalu mengirimkan hasil untuk ditampilkan frontend. "
            "Pada backend proyek ini tersedia fungsi predict_rf() untuk "
            "Random Forest dan predict_pinns() untuk PINNs. Detail "
            "permintaan dan respons setiap halaman bergantung pada kode "
            "frontend dan endpoint yang digunakan."
        ),
        (
            r"\b(?:apa fungsi|fungsi|jelaskan)\b.*"
            r"\b(?:predict_rf|random forest)\b",
            "Fungsi predict_rf() merupakan bagian backend yang menjalankan "
            "prediksi menggunakan model Random Forest. Berdasarkan kode "
            "yang tersedia, fungsi ini dipanggil ketika pemrosesan memilih "
            "model random_forest. Detail fitur input dan bentuk hasil "
            "prediksi mengikuti implementasi fungsi tersebut."
        ),
        (
            r"\b(?:apa fungsi|fungsi|jelaskan)\b.*"
            r"\b(?:predict_pinns|pinns)\b",
            "Fungsi predict_pinns() merupakan bagian backend yang "
            "menjalankan prediksi menggunakan model PINNs. Berdasarkan "
            "kode yang tersedia, fungsi ini dipanggil ketika pemrosesan "
            "memilih model pinns. Detail input, keluaran, dan komponen "
            "fisika perlu merujuk implementasi model yang sebenarnya."
        ),
        (
            r"\b(?:apa fungsi|fungsi|jelaskan)\b.*"
            r"\b(?:ai\s*/\s*chat|endpoint\s+(?:ai\s+)?chat|chat\s+ai)\b",
            "Endpoint POST /ai/chat menerima permintaan AI Growth "
            "Assistant melalui backend FastAPI. Handler ai_chat() "
            "memproses pertanyaan dan menentukan jenis jawaban yang "
            "sesuai, misalnya FAQ, analisis data, atau pertanyaan prediksi. "
            "Endpoint ini berbeda dari endpoint laporan PDF dan PPTX."
        ),
    ]

    prediction_flow_answer = None

    for flow_pattern, flow_answer in prediction_flow_patterns:
        if re.search(flow_pattern, prediction_flow_question):
            prediction_flow_answer = flow_answer
            break

    if prediction_flow_answer is not None:
        return {
            "success": True,
            "answer": prediction_flow_answer,
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }


    # FAQ_TRAINING_PINNS_V1
    training_question = question.lower().strip()

    training_faqs = [
        (
            r"\b(?:apa itu|jelaskan|fungsi)\b.*\bdata loss\b",
            "Data loss mengukur perbedaan antara prediksi model dan "
            "target data. Pada training PINNs V2, MSE dihitung terpisah "
            "untuk DBH, tinggi, dan volume."
        ),
        (
            r"\b(?:apa itu|jelaskan|fungsi)\b.*\bphysics loss\b",
            "Physics loss mengukur ketidaksesuaian prediksi volume dengan "
            "hubungan V = C * DBH^2 * Height. Pada PINNs V2, komponen ini "
            "diberi bobot 0.1 dalam total loss dan dihitung untuk data "
            "periode P18-P66."
        ),
        (
            r"\b(?:apa itu|fungsi|mengapa menggunakan)\b.*\b(?:adam|optimizer)\b",
            "Optimizer Adam memperbarui parameter model berdasarkan "
            "gradien total loss. Skrip training PINNs V2 menggunakan "
            "learning rate 0.001."
        ),
        (
            r"\b(?:apa itu|jelaskan|fungsi)\b.*\bepoch\b",
            "Epoch adalah satu putaran proses training atas data yang "
            "digunakan. Skrip PINNs V2 dikonfigurasi untuk 100 epoch "
            "dan mencatat riwayat loss setiap epoch."
        ),
        (
            r"\b(?:apa perbedaan|bedanya|jelaskan perbedaan)\b.*"
            r"\btraining\b.*\binference\b",
            "Training memperbarui bobot model melalui loss, backpropagation, "
            "dan optimizer. Inference menggunakan bobot model yang sudah "
            "dilatih untuk menghasilkan prediksi tanpa memperbarui bobot. "
            "Pada backend, predict_pinns() menjalankan inference."
        ),
        (
            r"\b(?:bagaimana|jelaskan)\b.*\b(?:alur|proses)\b.*"
            r"\btraining\b.*\bpinns\b",
            "Alur training PINNs V2 adalah memuat data dan scaler, "
            "menyiapkan input dan target, membangun model, menghitung "
            "data loss serta physics loss, menjumlahkan total loss, "
            "menjalankan backpropagation dan Adam, lalu menyimpan "
            "checkpoint dengan total loss terbaik."
        ),
    ]

    for pattern, answer in training_faqs:
        if re.search(pattern, training_question, re.IGNORECASE):
            return {
                "success": True,
                "answer": answer,
                "source": "pinns_training_faq",
            }



    # FAQ_EVALUASI_PINNS_V2
    evaluation_question = question.lower().strip()

    evaluation_faqs = [
        (
            r"\b(?:bagaimana|jelaskan|apa)\b.*"
            r"\b(?:hasil|ringkasan)\b.*\bevaluasi\b.*\bpinns\b",
            (
                "Evaluasi PINNs V2 pada data test menghasilkan "
                "DBH: MAE 1.1031 cm, RMSE 1.3598 cm, R? 0.9230, "
                "dan MAPE 12.62%; "
                "tinggi: MAE 1.5767 m, RMSE 1.9791 m, "
                "R? 0.9066, dan MAPE 13.86%; "
                "volume: MAE 0.03325 m?/pohon, "
                "RMSE 0.04501 m?/pohon, R? 0.8148, "
                "dan MAPE 23.86%. "
                "Jumlah data test adalah 43.740 untuk DBH dan tinggi, "
                "serta 35.640 untuk volume."
            ),
        ),
        (
            r"\b(?:target|variabel)\b.*\b(?:terbaik|tertinggi)\b.*"
            r"\br\s*(?:2|Ã‚Â²|\^2)\b|"
            r"\br\s*(?:2|Ã‚Â²|\^2)\b.*\b(?:terbaik|tertinggi)\b",
            (
                "Di antara tiga target PINNs V2, DBH memiliki R? "
                "tertinggi sebesar 0.9230, diikuti tinggi sebesar "
                "0.9066 dan volume sebesar 0.8148. "
                "R? yang lebih tinggi menunjukkan kecocokan yang "
                "lebih baik terhadap variasi target pada data test, "
                "tetapi bukan satu-satunya ukuran kualitas model."
            ),
        ),
        (
            r"\b(?:apa arti|jelaskan|fungsi)\b.*\bmae\b|"
            r"\bmae\b.*\b(?:artinya|itu apa)\b",
            (
                "MAE (Mean Absolute Error) adalah rata-rata nilai "
                "absolut selisih antara prediksi dan nilai aktual. "
                "Semakin kecil MAE, semakin kecil rata-rata kesalahan "
                "dalam satuan target. Pada PINNs V2, MAE test adalah "
                "1.1031 cm untuk DBH, 1.5767 m untuk tinggi, dan "
                "0.03325 m?/pohon untuk volume."
            ),
        ),
        (
            r"\b(?:apa arti|jelaskan|fungsi)\b.*\brmse\b|"
            r"\brmse\b.*\b(?:artinya|itu apa)\b",
            (
                "RMSE (Root Mean Squared Error) adalah akar dari "
                "rata-rata kuadrat kesalahan prediksi. RMSE memberi "
                "penekanan lebih besar pada kesalahan yang besar "
                "dibandingkan MAE. Pada PINNs V2, RMSE test adalah "
                "1.3598 cm untuk DBH, 1.9791 m untuk tinggi, dan "
                "0.04501 m?/pohon untuk volume."
            ),
        ),
        (
            r"\b(?:apa arti|jelaskan|fungsi)\b.*"
            r"(?:\br\s*2\b|\bR2\b|\br\^2\b|\br-squared\b)|"
            r"(?:\br\s*2\b|\bR2\b|\br\^2\b|\br-squared\b).*"
            r"\b(?:artinya|itu apa)\b",
            (
                "R2 mengukur seberapa baik prediksi menjelaskan "
                "variasi nilai aktual pada data evaluasi. Pada test "
                "PINNs V2, R2 DBH adalah 0.9230, tinggi 0.9066, "
                "dan volume 0.8148. R2 bukan persentase prediksi "
                "yang pasti benar dan tidak boleh digunakan sendirian "
                "untuk menilai kualitas model."
            ),
        ),
        (
            r"\b(?:apa arti|jelaskan|fungsi)\b.*\bmape\b|"
            r"\bmape\b.*\b(?:artinya|itu apa)\b",
            (
                "MAPE (Mean Absolute Percentage Error) menyatakan "
                "rata-rata kesalahan absolut dalam bentuk persentase "
                "terhadap nilai aktual. MAPE test PINNs V2 adalah "
                "12.62% untuk DBH, 13.86% untuk tinggi, dan 23.86% "
                "untuk volume. MAPE dapat sensitif terhadap nilai "
                "aktual yang sangat kecil atau mendekati nol."
            ),
        ),
        (
            r"\b(?:apakah|apa)\b.*\bpinns\b.*"
            r"\b(?:sudah|cukup|tergolong)?\s*akurat\b|"
            r"\bseberapa akurat\b.*\bpinns\b",
            (
                "Pada data test, PINNs V2 memiliki R? 0.9230 untuk "
                "DBH, 0.9066 untuk tinggi, dan 0.8148 untuk volume. "
                "Hasil ini menunjukkan kecocokan relatif lebih tinggi "
                "untuk DBH dan tinggi dibandingkan volume. Namun, "
                "apakah model cukup akurat bergantung pada kebutuhan "
                "penggunaan, toleransi kesalahan, dan pemeriksaan "
                "MAE, RMSE, serta metrik lainnya."
            ),
        ),
        (
            r"\b(?:bandingkan|perbandingan)\b.*"
            r"\b(?:standar|constrained|constraint)\b|"
            r"\bpinns\b.*\bconstrained\b.*\b(?:standar|biasa)\b",
            (
                "Pada laporan test standar dan final constrained, "
                "metrik DBH dan tinggi nyaris sama. Untuk volume, "
                "laporan standar mencatat MAE 0.0332518, RMSE "
                "0.0450144, R? 0.8147513, dan MAPE 23.8630%. "
                "Laporan final constrained mencatat MAE 0.0332333, "
                "RMSE 0.0450056, R? 0.8148236, dan MAPE 23.7649%. "
                "Perbedaannya kecil; angka tersebut tidak dengan "
                "sendirinya membuktikan peningkatan yang signifikan."
            ),
        ),
    ]

    for pattern, answer in evaluation_faqs:
        if re.search(pattern, evaluation_question, re.IGNORECASE):
            return {
                "success": True,
                "answer": answer,
                "data": {"found": False},
                "chart": None,
                "chart_offer": None,
                "source": "pinns_evaluation_faq",
            }


    prediction_result = answer_prediction_question(question)

    if prediction_result is not None:
        return prediction_result

    comparison_result = compare_growth_groups(question)

    if comparison_result is not None:
        return comparison_result

    # LangGraph: klasifikasi intent sebagai fallback.
    try:
        print("[LANGGRAPH] INVOKED:", question)
        graph_result = ai_graph.invoke({
            "question": question,
            "intent": "",
            "answer": "",
        })

        intent = graph_result.get("intent", "general")
        print("[LANGGRAPH] INTENT:", intent)
        print("[LANGGRAPH] TOOL ANSWER:", 
        str(graph_result.get("answer", ""))[:1000])

    except Exception:
        intent = "general"

    # Routing LangGraph ke handler Python yang sudah tersedia.
    if intent == "prediction":
        routed_question = "Prediksi " + question

        prediction_result = answer_prediction_question(
            routed_question
        )

        if prediction_result is not None:
            return prediction_result

    elif intent == "comparison":
        routed_question = "Bandingkan " + question

        comparison_result = compare_growth_groups(
            routed_question
        )

        if comparison_result is not None:
            return comparison_result


    
    try:
        facts = json.loads(graph_result["answer"])
    except (json.JSONDecodeError, TypeError, KeyError):
        facts = query_growth_data(question)

    if not facts.get("found"):
        return {
            "success": True,
            "answer": facts.get(
                "message",
                "Maaf, tidak ditemukan data yang cocok dengan pertanyaanmu. "
                "Coba periksa nama wilayah, kompartemen, spesies, klon, "
                "atau periode yang disebutkan.",
            ),
            "data": facts,
            "mode": "python_local",
        }

    statistics = facts.get("statistics", {})
    filters = facts.get("filters", {})

        # Handler gambaran umum dataset
    question_lower = question.lower()

    general_keywords = [
        "gambaran umum",
        "gambaran dataset",
        "overview dataset",
        "ringkasan dataset",
    ]

    if any(keyword in question_lower for keyword in general_keywords):
        row_count = facts.get("row_count", 0)

        metric_labels = {
            "disease_score": "Skor penyakit",
            "condition_score": "Skor kondisi",
            "dbh_cm": "Diameter pohon (cm)",
            "height_m": "Tinggi pohon (m)",
            "volume_m3_tree": "Volume pohon (mÂ³)",
            "trees_per_ha": "Jumlah pohon per hektare",
        }

        summary_lines = []

        for metric, values in statistics.items():
            label = metric_labels.get(metric, metric)

            mean_value = values.get("mean")
            min_value = values.get("min")
            max_value = values.get("max")

            if mean_value is not None:
                summary_lines.append(
                    f"- {label}: rata-rata {mean_value:.4f}, "
                    f"minimum {min_value}, maksimum {max_value}"
                )

        answer = (
            f"Dataset SIRIMBA memiliki {row_count:,} baris data.\n\n"
            "Berikut gambaran statistik dataset:\n"
            + "\n".join(summary_lines)
        )

        return {
            "success": True,
            "answer": answer,
            "data": facts,
            "mode": "langgraph",
        }

    # Tentukan satu metrik berdasarkan pertanyaan.
    metric_aliases = {
        "disease_score": [
            "disease score",
            "disease_score",
            "skor penyakit",
            "tingkat penyakit",
        ],
        "condition_score": [
            "condition score",
            "condition_score",
            "skor kondisi",
            "kondisi pohon",
        ],
        "dbh_cm": [
            "dbh",
            "dbh_cm",
            "diameter",
            "diameter pohon",
            "diameter batang",
        ],
        "height_m": [
            "height",
            "height_m",
            "tinggi",
            "tinggi pohon",
        ],
        "volume_m3_tree": [
            "volume",
            "volume_m3_tree",
            "volume pohon",
        ],
        "trees_per_ha": [
            "trees per ha",
            "trees_per_ha",
            "pohon per hektare",
            "pohon per hektar",
            "jumlah pohon per hektare",
            "jumlah pohon per hektar",
            "kerapatan pohon",
        ],
    }

    metric_labels = {
        "disease_score": "Disease score",
        "condition_score": "Condition score",
        "dbh_cm": "Diameter pohon (cm)",
        "height_m": "Tinggi pohon (m)",
        "volume_m3_tree": "Volume per pohon (m³)",
        "trees_per_ha": "Jumlah pohon per hektare",
    }

    selected_metric = None

    # Cocokkan frasa terpanjang terlebih dahulu.
    aliases = sorted(
        [
            (alias, metric)
            for metric, metric_alias_list in metric_aliases.items()
            for alias in metric_alias_list
        ],
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for alias, metric in aliases:
        pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
        if re.search(pattern, q):
            selected_metric = metric
            break

    if selected_metric is None:
        return {
            "success": True,
            "answer": (
                "Metrik yang ingin dianalisis belum jelas. "
                "Sebutkan salah satu: disease score, condition score, "
                "diameter pohon, tinggi pohon, volume per pohon, "
                "atau jumlah pohon per hektare."
            ),
            "data": facts,
            "mode": "python_local",
        }

    
    if selected_metric not in statistics:
        requested_year_match = re.search(
            r"(?<!\d)(?:19|20|21)\d{2}(?!\d)",
            q,
        )

        if requested_year_match:
            requested_year = int(requested_year_match.group(0))
            filters = facts.get("filters", {})
            period = filters.get("period")
            region = (
                filters.get("regional_name")
                or filters.get("regional_id")
            )

            if facts.get("row_count", 0) == 0:
                context = []

                if region:
                    context.append(str(region))

                if period:
                    context.append("periode " + str(period))

                context.append("tahun " + str(requested_year))

                return {
                    "success": True,
                    "answer": (
                        "Data historis untuk "
                        + " ".join(context)
                        + " tidak tersedia. "
                        "Dataset hanya mencakup tahun pengukuran 2000ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“2024."
                    ),
                    "data": facts,
                    "mode": "python_local",
                }

        return {
            "success": True,
            "answer": (
                "Metrik "
                + metric_labels[selected_metric]
                + " tidak memiliki nilai yang dapat dihitung "
                "pada data hasil filter ini."
            ),
            "data": facts,
            "mode": "python_local",
        }


    # Tentukan statistik yang diminta.
    if re.search(
        r"\b(minimum|minimal|terendah|paling rendah)\b",
        q,
    ):
        selected_stat = "min"
        stat_label = "Nilai minimum"

    elif re.search(
        r"\b(maksimum|maksimal|tertinggi|paling tinggi|terbesar)\b",
        q,
    ):
        selected_stat = "max"
        stat_label = "Nilai maksimum"

    else:
        # Default: rata-rata, termasuk jika pertanyaan memakai
        # kata "berapa" tanpa menyebut statistik lain.
        selected_stat = "mean"
        stat_label = "Rata-rata"

    value = statistics[selected_metric].get(selected_stat)

    if value is None:
        return {
            "success": True,
            "answer": (
                "Statistik yang diminta tidak tersedia untuk "
                + metric_labels[selected_metric]
                + "."
            ),
            "data": facts,
            "mode": "python_local",
        }

    # Susun konteks filter yang benar-benar digunakan.
    context_parts = []

    if filters.get("regional_name"):
        context_parts.append(
            "Wilayah: " + str(filters["regional_name"])
        )
    elif filters.get("regional_id"):
        context_parts.append(
            "ID wilayah: " + str(filters["regional_id"])
        )

    if filters.get("compartment_code"):
        context_parts.append(
            "Kompartemen: " + str(filters["compartment_code"])
        )
    elif filters.get("compartment_id"):
        context_parts.append(
            "ID kompartemen: " + str(filters["compartment_id"])
        )

    if filters.get("species"):
        context_parts.append(
            "Spesies: " + ", ".join(filters["species"])
        )

    if filters.get("clone"):
        context_parts.append("Klon: " + str(filters["clone"]))

    if filters.get("period"):
        context_parts.append("Periode: " + str(filters["period"]))

    # Tampilkan hanya satu metrik dan satu statistik yang diminta.
    answer_lines = [
        stat_label
        + " "
        + metric_labels[selected_metric]
        + ": "
        + format(float(value), ".4f"),
        "Jumlah baris data: "
        + format(facts["row_count"], ","),
    ]

    if context_parts:
        answer_lines.append("Filter: " + " | ".join(context_parts))

    return {
        "success": True,
        "answer": "\n".join(answer_lines),
        "data": facts,
        "mode": "python_local",
    }



# ============================================================
# REPORT DOWNLOAD ENDPOINTS
# ============================================================

from datetime import datetime
from io import BytesIO
from fastapi.responses import StreamingResponse



def _build_comparison_report(question: str) -> dict:
    """Membentuk laporan perbandingan atau analisis satu periode."""

    import re

    question = question.strip()

    if len(question) < 3:
        raise HTTPException(
            status_code=422,
            detail="Pertanyaan minimal 3 karakter.",
        )

    
    # Jalur pertama: jalankan analisis perbandingan hanya jika
    # pertanyaan memang secara eksplisit meminta perbandingan.
    comparison_keywords = [
        "bandingkan",
        "perbandingan",
        "dibandingkan",
        "dibanding",
        "mana yang tertinggi",
        "mana tertinggi",
        "mana yang terbesar",
        "mana terbesar",
        "mana yang terendah",
        "mana terendah",
        "paling tinggi",
        "paling rendah",
        "tertinggi di antara",
        "terendah di antara",
    ]

    is_comparison_question = any(
        keyword in question.lower()
        for keyword in comparison_keywords
    )

    if is_comparison_question:
        result = compare_growth_groups(question)

        if (
            isinstance(result, dict)
            and result.get("success") is True
            and isinstance(result.get("chart"), dict)
            and isinstance(result["chart"].get("data"), list)
            and result["chart"]["data"]
        ):
            return result


    # Jalur kedua: analisis statistik satu periode.
    facts = query_growth_data(question)

    facts = query_growth_data(question)

    if not facts.get("found"):
        raise HTTPException(
            status_code=422,
            detail=facts.get(
                "message",
                "Tidak ditemukan data yang sesuai dengan pertanyaan.",
            ),
        )

    statistics = facts.get("statistics", {})
    filters = facts.get("filters", {})
    q = question.lower()

    metric_aliases = {
        "disease_score": [
            "disease score",
            "disease_score",
            "skor penyakit",
            "tingkat penyakit",
        ],
        "condition_score": [
            "condition score",
            "condition_score",
            "skor kondisi",
            "kondisi pohon",
        ],
        "dbh_cm": [
            "dbh_cm",
            "dbh",
            "diameter batang",
            "diameter pohon",
            "diameter",
        ],
        "height_m": [
            "height_m",
            "height",
            "tinggi pohon",
            "tinggi",
        ],
        "volume_m3_tree": [
            "volume_m3_tree",
            "volume per pohon",
            "volume pohon",
            "volume",
        ],
        "trees_per_ha": [
            "trees_per_ha",
            "trees per ha",
            "pohon per hektare",
            "pohon per hektar",
            "kerapatan pohon",
        ],
    }

    metric_labels = {
        "disease_score": "Disease score",
        "condition_score": "Condition score",
        "dbh_cm": "Diameter pohon (cm)",
        "height_m": "Tinggi pohon (m)",
        "volume_m3_tree": "Volume per pohon (m³)",
        "trees_per_ha": "Jumlah pohon per hektare",
    }

    aliases = sorted(
        [
            (alias, metric)
            for metric, alias_list in metric_aliases.items()
            for alias in alias_list
        ],
        key=lambda item: len(item[0]),
        reverse=True,
    )

    selected_metric = None

    for alias, metric in aliases:
        if re.search(
            r"(?<!\w)" + re.escape(alias) + r"(?!\w)",
            q,
        ):
            selected_metric = metric
            break

    if selected_metric is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Metrik laporan belum jelas. Sebutkan volume pohon, "
                "tinggi pohon, diameter pohon, disease score, "
                "condition score, atau jumlah pohon per hektare."
            ),
        )

    metric_statistics = statistics.get(selected_metric)

    if not isinstance(metric_statistics, dict):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Metrik {metric_labels[selected_metric]} "
                "tidak tersedia pada data hasil filter."
            ),
        )

    if re.search(
        r"\b(minimum|minimal|terendah|paling rendah)\b",
        q,
    ):
        selected_stat = "min"
        stat_label = "Nilai minimum"
    elif re.search(
        r"\b(maksimum|maksimal|tertinggi|paling tinggi|terbesar)\b",
        q,
    ):
        selected_stat = "max"
        stat_label = "Nilai maksimum"
    else:
        selected_stat = "mean"
        stat_label = "Rata-rata"

    value = metric_statistics.get(selected_stat)

    if value is None:
        raise HTTPException(
            status_code=422,
            detail="Statistik yang diminta tidak tersedia.",
        )

    # Sesuaikan filter dengan format yang dibaca generator laporan.
    report_filters = {
        "regional": (
            filters.get("regional_name")
            or filters.get("regional_id")
        ),
        "species": filters.get("species", []),
        "compartments": (
            [filters["compartment_code"]]
            if filters.get("compartment_code")
            else []
        ),
        "clone": filters.get("clone"),
    }

    period = filters.get("period")

    answer = (
        f"{stat_label} {metric_labels[selected_metric]}: "
        f"{float(value):.4f}\n"
        f"Jumlah baris data: {facts['row_count']:,}"
    )

    return {
    "success": True,
    "report_type": "single_period",
    "answer": answer,
    "mode": "python_local",
    "data": {
        "metric": selected_metric,
        "metric_label": metric_labels[selected_metric],
        "group_label": "Statistik",
        "period": period,
        "filters": report_filters,
        "statistics": metric_statistics,
        "row_count": int(facts["row_count"]),
        "selected_stat": selected_stat,
        "stat_label": stat_label,
    },
    "chart": {
            "type": "bar",
            "title": (
                f"{stat_label} {metric_labels[selected_metric]}"
                + (f" pada {period}" if period else "")
            ),
            "x_key": "category",
            "y_key": "value",
            "x_label": "Statistik",
            "y_label": metric_labels[selected_metric],
            "period": period,
            "data": [
                {
                    "category": stat_label,
                    "value": float(value),
                    "count": int(
                        metric_statistics.get("count", 0)
                    ),
                }
            ],
        },
    }



@app.post("/ai/report/pdf")
def download_comparison_pdf(request: ChatRequest):
    """Menghasilkan dan mengunduh laporan PDF dari analisis aktual."""

    result = _build_comparison_report(request.question)

    try:
        pdf_buffer = generate_pdf_report(
            result,
            request.question,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Laporan PDF tidak dapat dibuat: {exc}",
        ) from exc

    filename = (
        "tree_growth_report_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".pdf"
    )

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )


@app.post("/ai/report/mai/pdf")
def download_mai_pdf(request: ChatRequest):
    """Menghasilkan PDF khusus laporan MAI berdasarkan data aktual."""

    import re

    years = sorted({
        int(year)
        for year in re.findall(r"\b(?:19|20)\d{2}\b", request.question)
    })

    year_start = years[0] if years else None
    year_end = years[-1] if years else None

    # Gunakan perhitungan MAI aktual yang sudah ada.
    result = mai_report(
        planting_year_start=year_start,
        planting_year_end=year_end,
        regional_id=None,
        compartment_id=None,
        species=None,
        clone=None,
    )

    if result.get("status") != "ok":
        raise HTTPException(
            status_code=422,
            detail=result.get("message", "Data MAI tidak tersedia untuk laporan."),
        )

    try:
        pdf_buffer = generate_mai_pdf_report(
            result,
            request.question,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Laporan PDF MAI tidak dapat dibuat: {exc}",
        ) from exc

    filename = (
        "laporan_mai_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".pdf"
    )

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )


@app.post("/ai/report/pptx")
def download_comparison_pptx(request: ChatRequest):
    """Menghasilkan dan mengunduh presentasi PowerPoint aktual."""

    result = _build_comparison_report(request.question)

    try:
        pptx_buffer = generate_pptx_report(
            result,
            request.question,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Laporan PowerPoint tidak dapat dibuat: {exc}",
        ) from exc

    filename = (
        "tree_growth_report_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".pptx"
    )

    return StreamingResponse(
        pptx_buffer,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "presentationml.presentation"
        ),
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )


# ============================================================
# DATASET UPLOAD - STAGING & VALIDATION
# ============================================================

UPLOAD_STAGING_DIR = PROJECT_ROOT / "data" / "dataset_staging"
DATASET_REGISTRY_PATH = PROJECT_ROOT / "data" / "dataset_registry.json"
DATASET_REGISTRY_LOCK = threading.Lock()

UPLOAD_REQUIRED_COLUMNS = [
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
    "condition_score",
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
    "disease_score",
    "disease_status",
    "tha_m",
]


@app.get("/dataset/registry")
def get_dataset_registry():
    """
    Menampilkan seluruh dataset yang tercatat di Registry.
    """

    if not DATASET_REGISTRY_PATH.exists():
        return {
            "success": True,
            "total": 0,
            "datasets": [],
        }

    try:
        with DATASET_REGISTRY_PATH.open(
            "r",
            encoding="utf-8",
        ) as registry_file:
            registry = json.load(registry_file)

        if not isinstance(registry, list):
            raise ValueError("Registry harus berupa JSON list.")

    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Dataset Registry rusak.",
        ) from exc

    return {
        "success": True,
        "total": len(registry),
        "datasets": registry,
    }




@app.get("/dataset/{upload_id}/download")
def download_dataset(upload_id: str):
    """
    Download dataset yang sudah tercatat di Registry.
    Hanya file staging yang diizinkan.
    """

    if not DATASET_REGISTRY_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Dataset Registry belum tersedia."
        )

    try:
        with DATASET_REGISTRY_PATH.open("r", encoding="utf-8") as registry_file:
            registry = json.load(registry_file)

        if not isinstance(registry, list):
            raise ValueError("Registry harus berupa JSON list.")

    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Dataset Registry rusak."
        ) from exc

    dataset_entry = next(
        (
            item for item in registry
            if item.get("upload_id") == upload_id
        ),
        None
    )

    if dataset_entry is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset tidak ditemukan di Registry."
        )

    stored_filename = dataset_entry.get("stored_filename")

    if not stored_filename:
        raise HTTPException(
            status_code=400,
            detail="Nama file dataset tidak valid."
        )

    # Tolak nama file/path yang mencurigakan
    if Path(stored_filename).name != stored_filename:
        raise HTTPException(
            status_code=400,
            detail="Nama file dataset tidak valid."
        )

    staging_root = UPLOAD_STAGING_DIR.resolve()
    dataset_path = (staging_root / stored_filename).resolve()

    # Pastikan file tetap berada di folder staging
    if dataset_path.parent != staging_root:
        raise HTTPException(
            status_code=403,
            detail="Akses file dataset ditolak."
        )

    if not dataset_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="File dataset tidak ditemukan."
        )

    original_filename = dataset_entry.get(
        "original_filename",
        stored_filename
    )

    return FileResponse(
        path=dataset_path,
        filename=original_filename,
        media_type="application/octet-stream"
    )

@app.get("/dataset/{upload_id}/detail")
def get_dataset_detail(upload_id: str):
    """
    Mengambil detail satu dataset berdasarkan upload_id.
    Tidak mengubah dataset aktif maupun model.
    """

    if not DATASET_REGISTRY_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Dataset Registry belum tersedia."
        )

    try:
        with DATASET_REGISTRY_PATH.open(
            "r",
            encoding="utf-8",
        ) as registry_file:
            registry = json.load(registry_file)

        if not isinstance(registry, list):
            raise ValueError("Registry harus berupa JSON list.")

    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Dataset Registry rusak."
        ) from exc

    dataset_entry = next(
        (
            item for item in registry
            if isinstance(item, dict)
            and item.get("upload_id") == upload_id
        ),
        None
    )

    if dataset_entry is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset tidak ditemukan di Registry."
        )

    return {
        "success": True,
        "dataset": dataset_entry,
    }

# ============================================================
def registry_locked(func):
    """
    Mengunci endpoint upload dan aktivasi selama proses berjalan.
    Mendukung endpoint async dan sync.
    """

    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):

            await asyncio.to_thread(
                DATASET_REGISTRY_LOCK.acquire
            )

            try:
                return await func(*args, **kwargs)

            finally:
                DATASET_REGISTRY_LOCK.release()

        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):

        with DATASET_REGISTRY_LOCK:
            return func(*args, **kwargs)

    return sync_wrapper

# DATASET ACTIVATION HELPERS
# ============================================================

def load_dataset_registry():
    """
    Membaca Dataset Registry dan memvalidasi formatnya.
    """

    if not DATASET_REGISTRY_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Dataset Registry belum tersedia.",
        )

    try:
        with DATASET_REGISTRY_PATH.open(
            "r",
            encoding="utf-8",
        ) as registry_file:
            registry = json.load(registry_file)

        if not isinstance(registry, list):
            raise ValueError("Registry harus berupa JSON list.")

    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Dataset Registry rusak.",
        ) from exc

    return registry


def save_dataset_registry(registry):
    """
    Menyimpan Registry menggunakan file sementara
    dan atomic replace.
    """

    registry_temp_path = DATASET_REGISTRY_PATH.with_name(
        f"dataset_registry_activation_{uuid.uuid4().hex}.tmp"
    )

    try:
        with registry_temp_path.open(
            "w",
            encoding="utf-8",
        ) as registry_file:
            json.dump(
                registry,
                registry_file,
                indent=2,
                ensure_ascii=False,
            )

        registry_temp_path.replace(
            DATASET_REGISTRY_PATH
        )

    finally:
        if registry_temp_path.exists():
            registry_temp_path.unlink()

# ============================================================
# DATASET ACTIVATION
# ============================================================

@app.post("/dataset/{upload_id}/activate")
@registry_locked
def activate_dataset(upload_id: str):
    """
    Mengaktifkan dataset staged melalui Registry.

    Aktivasi hanya memperbarui status Registry.
    Dataset baseline dan model RF/PINNs tidak diubah.
    """

    registry = load_dataset_registry()

    # --------------------------------------------------------
    # FIND DATASET
    # --------------------------------------------------------

    dataset_entry = next(
        (
            item for item in registry
            if isinstance(item, dict)
            and item.get("upload_id") == upload_id
        ),
        None,
    )

    if dataset_entry is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset tidak ditemukan di Registry.",
        )

    # --------------------------------------------------------
    # VALIDATE STAGING FILE
    # --------------------------------------------------------

    stored_filename = dataset_entry.get("stored_filename")

    if not stored_filename:
        raise HTTPException(
            status_code=400,
            detail="Nama file dataset tidak valid.",
        )

    if Path(stored_filename).name != stored_filename:
        raise HTTPException(
            status_code=400,
            detail="Nama file dataset tidak valid.",
        )

    staging_root = UPLOAD_STAGING_DIR.resolve()

    dataset_path = (
        staging_root / stored_filename
    ).resolve()

    if dataset_path.parent != staging_root:
        raise HTTPException(
            status_code=403,
            detail="Akses file dataset ditolak.",
        )

    if not dataset_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="File dataset staging tidak ditemukan.",
        )

    # --------------------------------------------------------
    # VALIDATE DATASET CONTENT
    # --------------------------------------------------------

    extension = dataset_path.suffix.lower()

    try:
        if extension == ".csv":
            try:
                activation_df = pd.read_csv(
                    dataset_path,
                    encoding="utf-8-sig",
                )
            except UnicodeDecodeError:
                activation_df = pd.read_csv(
                    dataset_path,
                    encoding="latin1",
                )

        elif extension in {".xlsx", ".xls"}:
            activation_df = pd.read_excel(dataset_path)

        else:
            raise HTTPException(
                status_code=400,
                detail="Format dataset tidak didukung.",
            )

        if activation_df.empty:
            raise HTTPException(
                status_code=400,
                detail="Dataset tidak memiliki data.",
            )

        _, activation_report = standardize_dataset(
            activation_df,
            required_columns=UPLOAD_REQUIRED_COLUMNS,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset tidak dapat dibaca atau divalidasi: {str(exc)}",
        ) from exc

    if not activation_report["valid"]:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Dataset gagal melewati validasi aktivasi SIRIMBA.",
                "validation_report": activation_report,
            },
        )

    # --------------------------------------------------------
    # UPDATE REGISTRY
    # --------------------------------------------------------

    activation_time = datetime.now().isoformat(
        timespec="seconds"
    )

    for item in registry:

        if not isinstance(item, dict):
            continue

        item["dataset_activated"] = False

        if item.get("status") == "active":
            item["status"] = "validated_staged"

    dataset_entry["dataset_activated"] = True
    dataset_entry["status"] = "active"
    dataset_entry["activated_at"] = activation_time
    dataset_entry["dashboard_compatible"] = bool(
        activation_report["dashboard_compatible"]
    )
    dataset_entry["validation_report"] = activation_report

    save_dataset_registry(registry)

    return {
        "success": True,
        "message": "Dataset berhasil diaktifkan di Registry.",
        "upload_id": upload_id,
        "dataset_activated": True,
        "activated_at": activation_time,
        "dataset_path": dataset_entry.get("dataset_path"),
        "note": (
            "Aktivasi Registry tidak mengubah dataset baseline "
            "atau model RF/PINNs."
        ),
    }

@app.post("/dataset/upload")
@registry_locked
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload dataset baru untuk validasi awal.
    Dataset aktif tidak diubah.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Nama file tidak ditemukan.",
        )

    filename = Path(file.filename).name
    extension = Path(filename).suffix.lower()

    if extension not in [".csv", ".xlsx", ".xls"]:
        raise HTTPException(
            status_code=400,
            detail="Format file harus CSV, XLSX, atau XLS.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="File yang diupload kosong.",
        )

    max_size = 100 * 1024 * 1024

    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail="Ukuran file maksimal 100 MB.",
        )

    try:
        if extension == ".csv":
            try:
                uploaded_df = pd.read_csv(
                    BytesIO(content),
                    encoding="utf-8-sig",
                )
            except UnicodeDecodeError:
                uploaded_df = pd.read_csv(
                    BytesIO(content),
                    encoding="latin1",
                )

        else:
            uploaded_df = pd.read_excel(
                BytesIO(content),
            )

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"File gagal dibaca: {str(exc)}",
        ) from exc

    if uploaded_df.empty:
        raise HTTPException(
            status_code=422,
            detail="Dataset tidak memiliki data.",
        )

    standardized_df, validation_report = standardize_dataset(
        uploaded_df,
        required_columns=UPLOAD_REQUIRED_COLUMNS,
    )

    if not validation_report["valid"]:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Dataset gagal melewati validasi standardisasi SIRIMBA.",
                "validation_report": validation_report,
            },
        )

    UPLOAD_STAGING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    upload_id = str(uuid.uuid4())

    saved_filename = (
        upload_id + extension
    )

    saved_path = (
        UPLOAD_STAGING_DIR / saved_filename
    )

    saved_path.write_bytes(content)

    preview_df = standardized_df.head(5).astype(object)
    preview_df = preview_df.where(
        pd.notna(preview_df),
        None,
    )
    preview = preview_df.to_dict(orient="records")

    # --------------------------------------------------------
    # DATASET REGISTRY
    # --------------------------------------------------------

    DATASET_REGISTRY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry = []

    if DATASET_REGISTRY_PATH.exists():
        try:
            with DATASET_REGISTRY_PATH.open(
                "r",
                encoding="utf-8",
            ) as registry_file:
                registry = json.load(registry_file)

            if not isinstance(registry, list):
                raise ValueError("Registry harus berupa JSON list.")

        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(
                status_code=500,
                detail="Dataset Registry rusak. Upload dihentikan.",
            ) from exc

    registry_entry = {
        "upload_id": upload_id,
        "original_filename": filename,
        "stored_filename": saved_filename,
        "rows": int(len(standardized_df)),
        "columns": int(len(standardized_df.columns)),
        "status": "validated_staged",
        "dataset_activated": False,
        "dashboard_compatible": bool(
            validation_report["dashboard_compatible"]
        ),
        "validation_report": validation_report,
        "dataset_path": str(saved_path.relative_to(PROJECT_ROOT)),
        "uploaded_at": datetime.now().isoformat(timespec="seconds"),
    }

    registry.append(registry_entry)

    try:
        save_dataset_registry(registry)
    except Exception:
        try:
            saved_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    return {
        "status": "validated",
        "message": "Dataset berhasil diupload, divalidasi, dan dicatat.",
        "upload_id": upload_id,
        "original_filename": filename,
        "rows": int(len(standardized_df)),
        "columns": int(len(standardized_df.columns)),
        "missing_columns": [],
        "preview": preview,
        "validation_report": validation_report,
        "dashboard_compatible": bool(
            validation_report["dashboard_compatible"]
        ),
        "dataset_activated": False,
        "registry_recorded": True,
        "dataset_path": str(saved_path.relative_to(PROJECT_ROOT)),
    }














