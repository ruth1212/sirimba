from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# TREE GROWTH ANALYTICS
# STEP 29 — RAW GROWTH LOGIC INSPECTION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "dataset.csv"


print("=" * 80)
print("TREE GROWTH ANALYTICS")
print("STEP 29 — RAW GROWTH LOGIC INSPECTION")
print("=" * 80)

print()
print(f"Dataset : {DATASET_PATH}")
print(f"Exists  : {DATASET_PATH.exists()}")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATASET_PATH}"
    )


# ============================================================
# 1. LOAD DATASET
# ============================================================

print()
print("[1] LOADING DATASET")
print("-" * 80)

df = pd.read_csv(DATASET_PATH)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 2. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
    "inventory_year",
    "age_months",
    "growth_period",
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

print()
print("[2] REQUIRED COLUMN VALIDATION")
print("-" * 80)

if missing:
    print("MISSING COLUMNS:")
    for col in missing:
        print(f"  - {col}")

    raise ValueError(
        "Required columns tidak lengkap."
    )

print("Required columns : PASS")


# ============================================================
# 3. BASIC TARGET VALIDATION
# ============================================================

targets = [
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]

print()
print("[3] TARGET BASIC VALIDATION")
print("-" * 80)

for target in targets:

    values = pd.to_numeric(
        df[target],
        errors="coerce"
    )

    print()
    print(f"{target}")
    print(f"  missing : {values.isna().sum():,}")
    print(f"  min     : {values.min():.10f}")
    print(f"  max     : {values.max():.10f}")
    print(f"  mean    : {values.mean():.10f}")


# ============================================================
# 4. GROWTH PERIOD / AGE STRUCTURE
# ============================================================

print()
print("[4] GROWTH PERIOD STRUCTURE")
print("-" * 80)

period_summary = (
    df.groupby(
        ["growth_period", "age_months"],
        dropna=False
    )
    .agg(
        rows=("growth_period", "size"),
        dbh_mean=("dbh_cm", "mean"),
        height_mean=("height_m", "mean"),
        volume_mean=("volume_m3_tree", "mean"),
        dbh_min=("dbh_cm", "min"),
        dbh_max=("dbh_cm", "max"),
        height_min=("height_m", "min"),
        height_max=("height_m", "max"),
        volume_min=("volume_m3_tree", "min"),
        volume_max=("volume_m3_tree", "max"),
    )
    .reset_index()
    .sort_values("age_months")
)

print(
    period_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 5. UNIQUE GROWTH PERIOD LABELS
# ============================================================

print()
print("[5] GROWTH PERIOD LABELS")
print("-" * 80)

period_labels = (
    df[
        ["growth_period", "age_months"]
    ]
    .drop_duplicates()
    .sort_values("age_months")
)

for _, row in period_labels.iterrows():

    print(
        f"{str(row['growth_period']):>8} "
        f"-> age_months = {int(row['age_months'])}"
    )


# ============================================================
# 6. BIOLOGICAL GROUP DEFINITION
# ============================================================
#
# Kita tidak langsung mengecek seluruh dataset secara global.
#
# Pertumbuhan harus dibandingkan dalam konteks pohon/stand yang
# sama:
#
# regional
# compartment
# species
# clone
# planting_year
#
# Lalu diurutkan berdasarkan age_months.
# ============================================================

group_columns = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]


print()
print("[6] BIOLOGICAL GROWTH GROUP")
print("-" * 80)

print(
    "Growth comparison group:"
)

print(
    " + ".join(group_columns)
)


# ============================================================
# 7. AGGREGATE BY GROWTH GROUP + AGE
# ============================================================
#
# Kita menggunakan mean untuk inspeksi awal.
#
# Tujuan:
# melihat apakah pola rata-rata pertumbuhan actual
# mengalami penurunan.
#
# Ini BELUM melakukan cleaning.
# ============================================================

growth = (
    df.groupby(
        group_columns + ["age_months", "growth_period"],
        dropna=False
    )
    .agg(
        dbh_cm=("dbh_cm", "mean"),
        height_m=("height_m", "mean"),
        volume_m3_tree=("volume_m3_tree", "mean"),
        rows=("dbh_cm", "size"),
    )
    .reset_index()
    .sort_values(
        group_columns + ["age_months"]
    )
)


# ============================================================
# 8. CALCULATE CHANGE FROM PREVIOUS AGE
# ============================================================

for target in targets:

    delta_column = f"{target}_delta"

    growth[delta_column] = (
        growth
        .groupby(group_columns)[target]
        .diff()
    )


# ============================================================
# 9. DBH DECREASE INSPECTION
# ============================================================

print()
print("[7] DBH DECREASE INSPECTION")
print("-" * 80)

dbh_decrease = growth[
    growth["dbh_cm_delta"] < 0
].copy()

print(
    f"DBH decrease records : {len(dbh_decrease):,}"
)

if len(dbh_decrease) > 0:

    print()
    print(
        "Largest DBH decreases:"
    )

    display_columns = (
        group_columns
        + [
            "age_months",
            "growth_period",
            "dbh_cm",
            "dbh_cm_delta",
        ]
    )

    print(
        dbh_decrease
        .sort_values("dbh_cm_delta")
        .head(30)[display_columns]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )


# ============================================================
# 10. HEIGHT DECREASE INSPECTION
# ============================================================

print()
print("[8] HEIGHT DECREASE INSPECTION")
print("-" * 80)

height_decrease = growth[
    growth["height_m_delta"] < 0
].copy()

print(
    f"Height decrease records : {len(height_decrease):,}"
)

if len(height_decrease) > 0:

    print()
    print(
        "Largest height decreases:"
    )

    display_columns = (
        group_columns
        + [
            "age_months",
            "growth_period",
            "height_m",
            "height_m_delta",
        ]
    )

    print(
        height_decrease
        .sort_values("height_m_delta")
        .head(30)[display_columns]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )


# ============================================================
# 11. VOLUME DECREASE INSPECTION
# ============================================================

print()
print("[9] VOLUME DECREASE INSPECTION")
print("-" * 80)

volume_decrease = growth[
    growth["volume_m3_tree_delta"] < 0
].copy()

print(
    f"Volume decrease records : {len(volume_decrease):,}"
)

if len(volume_decrease) > 0:

    print()
    print(
        "Largest volume decreases:"
    )

    display_columns = (
        group_columns
        + [
            "age_months",
            "growth_period",
            "volume_m3_tree",
            "volume_m3_tree_delta",
        ]
    )

    print(
        volume_decrease
        .sort_values("volume_m3_tree_delta")
        .head(30)[display_columns]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}"
        )
    )


# ============================================================
# 12. LATE-AGE INSPECTION
# ============================================================
#
# Mentor menemukan masalah setelah P54.
#
# Kita secara khusus inspeksi:
#
# P54
# P60
# P66
# PHI
#
# ============================================================

print()
print("[10] LATE-AGE INSPECTION")
print("-" * 80)

late_age = (
    growth[
        growth["age_months"] >= 54
    ]
    .groupby(
        ["age_months", "growth_period"]
    )
    .agg(
        groups=("dbh_cm", "size"),
        dbh_mean=("dbh_cm", "mean"),
        height_mean=("height_m", "mean"),
        volume_mean=("volume_m3_tree", "mean"),
        dbh_min=("dbh_cm", "min"),
        dbh_max=("dbh_cm", "max"),
        height_min=("height_m", "min"),
        height_max=("height_m", "max"),
        volume_min=("volume_m3_tree", "min"),
        volume_max=("volume_m3_tree", "max"),
    )
    .reset_index()
    .sort_values("age_months")
)

print(
    late_age.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)


# ============================================================
# 13. PHI INSPECTION
# ============================================================

print()
print("[11] PHI INSPECTION")
print("-" * 80)

phi = df[
    df["growth_period"].astype(str).str.upper() == "PHI"
].copy()

print(
    f"PHI rows : {len(phi):,}"
)

if len(phi) > 0:

    print()
    print(
        "PHI age distribution:"
    )

    print(
        phi["age_months"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "PHI age statistics:"
    )

    print(
        phi["age_months"]
        .describe()
        .to_string()
    )

    print()
    print(
        "PHI target means:"
    )

    print(
        phi[
            [
                "dbh_cm",
                "height_m",
                "volume_m3_tree",
            ]
        ]
        .mean()
        .to_string()
    )


# ============================================================
# 14. P66 INSPECTION
# ============================================================

print()
print("[12] P66 INSPECTION")
print("-" * 80)

p66 = df[
    df["growth_period"].astype(str).str.upper() == "P66"
].copy()

print(
    f"P66 rows : {len(p66):,}"
)

if len(p66) > 0:

    print(
        p66["age_months"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    print(
        p66[
            [
                "dbh_cm",
                "height_m",
                "volume_m3_tree",
            ]
        ]
        .describe()
        .to_string()
    )


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("INSPECTION SUMMARY")
print("=" * 80)

print()
print(
    f"DBH decrease records    : {len(dbh_decrease):,}"
)

print(
    f"Height decrease records : {len(height_decrease):,}"
)

print(
    f"Volume decrease records : {len(volume_decrease):,}"
)

print(
    f"PHI rows                : {len(phi):,}"
)

print(
    f"P66 rows                : {len(p66):,}"
)

print()
print(
    "IMPORTANT:"
)

print(
    "Dataset BELUM diubah."
)

print(
    "Dataset BELUM dilakukan monotonic correction."
)

print(
    "Dataset BELUM dilakukan PHI transformation."
)

print(
    "Dataset BELUM dilakukan volume adjustment."
)

print(
    "Hasil step ini hanya untuk INSPEKSI."
)

print()
print("=" * 80)
print("STEP 29 FINISHED — INSPECTION ONLY")
print("=" * 80)