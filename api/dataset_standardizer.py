"""
Dataset Standardizer
Modul standardisasi dataset SIRIMBA.
"""

from __future__ import annotations

import re
from typing import Mapping

import pandas as pd


# Kolom standar berdasarkan backend saat ini.
CANONICAL_COLUMNS = {
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
}


# Kolom standar opsional berdasarkan dataset baseline.
OPTIONAL_CANONICAL_COLUMNS = {
    "stand_area_total_ha",
    "species_clone",
    "planting_cohort",
    "planting_date",
    "inventory_date",
}


# Alias masih kosong sampai mapping aktual diverifikasi.
DEFAULT_COLUMN_ALIASES: dict[str, str] = {}


def normalize_column_name(column_name: object) -> str:
    """Normalisasi nama kolom menjadi lowercase snake_case."""

    name = str(column_name).strip().lower()
    name = re.sub(r"[\s\-\/]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    name = re.sub(r"_+", "_", name)

    return name.strip("_")


def standardize_column_names(
    df: pd.DataFrame,
    aliases: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """
    Standardisasi nama kolom tanpa menghapus kolom tambahan.

    Alias menggunakan format:
        {"nama_alias": "nama_kolom_standar"}
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input harus berupa pandas DataFrame.")

    aliases = (
        DEFAULT_COLUMN_ALIASES
        if aliases is None
        else aliases
    )

    normalized_aliases = {
        normalize_column_name(alias): normalize_column_name(canonical)
        for alias, canonical in aliases.items()
    }

    standardized_names = []

    for original_name in df.columns:
        normalized_name = normalize_column_name(original_name)

        canonical_name = normalized_aliases.get(
            normalized_name,
            normalized_name,
        )

        standardized_names.append(canonical_name)

    result = df.copy()
    result.columns = standardized_names

    duplicated_columns = result.columns[
        result.columns.duplicated()
    ].tolist()

    if duplicated_columns:
        raise ValueError(
            "Standardisasi menghasilkan nama kolom duplikat: "
            f"{duplicated_columns}"
        )

    return result

# ============================================================
# REQUIRED COLUMN VALIDATION
# ============================================================

def validate_required_columns(
    df: pd.DataFrame,
    required_columns: set[str] | None = None,
) -> dict:
    """
    Memeriksa kolom wajib, opsional, dan tambahan
    tanpa menghapus kolom dataset.
    """

    required = (
        CANONICAL_COLUMNS
        if required_columns is None
        else set(required_columns)
    )

    available_columns = set(df.columns)

    missing_columns = sorted(
        required - available_columns
    )

    optional_columns = sorted(
        OPTIONAL_CANONICAL_COLUMNS & available_columns
    )

    extra_columns = sorted(
        available_columns
        - required
        - OPTIONAL_CANONICAL_COLUMNS
    )

    return {
        "valid": len(missing_columns) == 0,
        "missing_columns": missing_columns,
        "optional_columns": optional_columns,
        "extra_columns": extra_columns,
        "total_columns": len(df.columns),
    }

# ============================================================
# AGE VALIDATION
# ============================================================

DISPLAY_AGES = {
    6, 12, 18, 24, 30, 36,
    42, 48, 54, 60, 66,
}


def validate_age_months(
    df: pd.DataFrame,
    age_column: str = "age_months",
) -> dict:
    """
    Memeriksa usia tanaman tanpa menghapus baris dataset.
    """

    if age_column not in df.columns:
        return {
            "valid": False,
            "error": f"Kolom {age_column} tidak ditemukan.",
            "missing_count": None,
            "invalid_count": None,
            "supported_count": None,
            "unsupported_ages": [],
        }

    ages = pd.to_numeric(
        df[age_column],
        errors="coerce",
    )

    missing_count = int(ages.isna().sum())

    invalid_mask = (
        ages.notna()
        & (
            (ages % 1 != 0)
            | (ages <= 0)
        )
    )

    invalid_count = int(invalid_mask.sum())

    valid_ages = ages.dropna()
    supported_mask = valid_ages.isin(DISPLAY_AGES)

    unsupported_ages = sorted(
        valid_ages[~supported_mask]
        .unique()
        .tolist()
    )

    return {
        "valid": missing_count == 0 and invalid_count == 0,
        "missing_count": missing_count,
        "invalid_count": invalid_count,
        "supported_count": int(supported_mask.sum()),
        "unsupported_ages": unsupported_ages,
        "dashboard_compatible": (
            missing_count == 0
            and invalid_count == 0
            and len(unsupported_ages) == 0
        ),
        "total_rows": int(len(df)),
    }






# ============================================================
# DATA TYPE DEFINITIONS
# ============================================================

INTEGER_COLUMNS = {
    "compartment_id",
    "trees_per_ha",
    "estimated_trees_compartment",
    "planting_year",
    "planting_cohort",
    "inventory_year",
    "age_months",
}

FLOAT_COLUMNS = {
    "area_ha",
    "stand_area_total_ha",
    "spacing_x_m",
    "spacing_y_m",
    "clone_factor",
    "condition_score",
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
    "disease_score",
    "tha_m",
}

STRING_COLUMNS = {
    "regional_id",
    "regional_name",
    "compartment_code",
    "spacing_m",
    "plant_spacing_m",
    "species_code",
    "species",
    "clone",
    "species_clone",
    "planting_date",
    "inventory_date",
    "growth_period",
    "disease_status",
}


# ============================================================
# DATA TYPE VALIDATION
# ============================================================

def validate_data_types(df: pd.DataFrame) -> dict:
    """
    Memvalidasi tipe nilai kolom tanpa mengubah dataset asli.
    """

    results = {}
    invalid_columns = {}

    for column in INTEGER_COLUMNS | FLOAT_COLUMNS:
        if column not in df.columns:
            continue

        original_values = df[column]
        converted_values = pd.to_numeric(
            original_values,
            errors="coerce",
        )

        missing_mask = original_values.isna()
        invalid_mask = (
            original_values.notna()
            & converted_values.isna()
        )

        if column in INTEGER_COLUMNS:
            invalid_mask = invalid_mask | (
                converted_values.notna()
                & (converted_values % 1 != 0)
            )

        missing_count = int(missing_mask.sum())
        invalid_count = int(invalid_mask.sum())

        results[column] = {
            "expected_type": (
                "integer"
                if column in INTEGER_COLUMNS
                else "float"
            ),
            "missing_count": missing_count,
            "invalid_count": invalid_count,
            "valid": invalid_count == 0,
        }

        if invalid_count > 0:
            invalid_columns[column] = invalid_count

    return {
        "valid": len(invalid_columns) == 0,
        "columns_checked": len(results),
        "invalid_columns": invalid_columns,
        "details": results,
    }


# ============================================================
# MISSING VALUE VALIDATION
# ============================================================

MISSING_VALUE_MARKERS = {
    "",
    "n/a",
    "na",
    "null",
    "none",
    "-",
}


def validate_missing_values(
    df: pd.DataFrame,
    required_columns: set[str] | None = None,
) -> dict:
    """
    Memeriksa missing value, string kosong, dan placeholder
    tanpa mengubah dataset asli.

    Kolom wajib yang memiliki nilai kosong memengaruhi status
    validasi. Kolom opsional dan tambahan hanya dilaporkan.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input harus berupa pandas DataFrame.")

    required = (
        CANONICAL_COLUMNS
        if required_columns is None
        else required_columns
    )

    details = {}
    required_missing_columns = []
    optional_missing_columns = []
    extra_missing_columns = []

    total_missing = 0
    total_cells = len(df) * len(df.columns)

    for column in df.columns:
        original_values = df[column]

        missing_mask = original_values.isna()

        marker_mask = (
            original_values.astype("string")
            .str.strip()
            .str.lower()
            .isin(MISSING_VALUE_MARKERS)
            & ~missing_mask
        )

        missing_count = int(missing_mask.sum())
        marker_count = int(marker_mask.sum())
        total_column_missing = missing_count + marker_count

        total_missing += total_column_missing

        completeness = (
            round(
                (len(df) - total_column_missing)
                / len(df) * 100,
                2,
            )
            if len(df) > 0
            else 0.0
        )

        details[column] = {
            "missing_count": missing_count,
            "marker_count": marker_count,
            "total_missing": total_column_missing,
            "completeness_pct": completeness,
        }

        if total_column_missing > 0:
            if column in required:
                required_missing_columns.append(column)
            elif column in OPTIONAL_CANONICAL_COLUMNS:
                optional_missing_columns.append(column)
            else:
                extra_missing_columns.append(column)

    total_completeness = (
        round(
            (total_cells - total_missing)
            / total_cells * 100,
            2,
        )
        if total_cells > 0
        else 0.0
    )

    return {
        "valid": len(required_missing_columns) == 0,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_missing_cells": total_missing,
        "overall_completeness_pct": total_completeness,
        "required_missing_columns": sorted(
            required_missing_columns
        ),
        "optional_missing_columns": sorted(
            optional_missing_columns
        ),
        "extra_missing_columns": sorted(
            extra_missing_columns
        ),
        "details": details,
    }


# ============================================================
# DATA TYPE STANDARDIZATION
# ============================================================

def standardize_data_types(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Mengonversi kolom numerik ke tipe standar tanpa menghapus
    kolom tambahan atau mengubah DataFrame asli.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input harus berupa pandas DataFrame.")

    result = df.copy()
    conversion_errors = {}

    for column in INTEGER_COLUMNS:
        if column not in result.columns:
            continue

        original = result[column]
        converted = pd.to_numeric(original, errors="coerce")

        invalid_mask = original.notna() & converted.isna()
        fractional_mask = (
            converted.notna() & (converted % 1 != 0)
        )

        invalid_count = int(
            (invalid_mask | fractional_mask).sum()
        )

        if invalid_count:
            conversion_errors[column] = invalid_count
            continue

        result[column] = converted.astype("Int64")

    for column in FLOAT_COLUMNS:
        if column not in result.columns:
            continue

        original = result[column]
        converted = pd.to_numeric(original, errors="coerce")

        invalid_mask = original.notna() & converted.isna()
        invalid_count = int(invalid_mask.sum())

        if invalid_count:
            conversion_errors[column] = invalid_count
            continue

        result[column] = converted.astype("Float64")

    report = {
        "valid": len(conversion_errors) == 0,
        "converted_columns": sorted(
            (INTEGER_COLUMNS | FLOAT_COLUMNS) & set(result.columns)
            - set(conversion_errors)
        ),
        "conversion_errors": conversion_errors,
        "total_columns": len(result.columns),
        "original_preserved": True,
    }

    return result, report


# ============================================================
# DATASET STANDARDIZATION PIPELINE
# ============================================================

def standardize_dataset(
    df: pd.DataFrame,
    aliases: Mapping[str, str] | None = None,
    required_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Menjalankan seluruh proses standardisasi dan validasi dataset.

    DataFrame asli tidak diubah.
    Kolom tambahan dan periode usia yang belum didukung
    tetap dipertahankan.
    """

    standardized_df = standardize_column_names(
        df,
        aliases=aliases,
    )

    column_report = validate_required_columns(
        standardized_df,
        required_columns=required_columns,
    )

    age_report = validate_age_months(standardized_df)

    missing_report = validate_missing_values(
        standardized_df,
        required_columns=required_columns,
    )

    type_report = validate_data_types(standardized_df)

    standardized_df, conversion_report = standardize_data_types(
        standardized_df,
    )

    valid = all([
        column_report["valid"],
        age_report["valid"],
        missing_report["valid"],
        type_report["valid"],
        conversion_report["valid"],
    ])

    report = {
        "valid": valid,
        "dashboard_compatible": age_report.get(
            "dashboard_compatible",
            False,
        ),
        "column_validation": column_report,
        "age_validation": age_report,
        "missing_validation": missing_report,
        "data_type_validation": type_report,
        "conversion_validation": conversion_report,
        "total_rows": len(standardized_df),
        "total_columns": len(standardized_df.columns),
    }

    return standardized_df, report

