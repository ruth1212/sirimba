from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 34
# VALIDATE CORRECTED DBH DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = BASE_DIR / "data" / "dataset.csv"
CORRECTED_PATH = BASE_DIR / "data" / "dataset_corrected.csv"

OUTPUT_DIR = BASE_DIR / "results" / "dbH_correction_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_OUTPUT = OUTPUT_DIR / "dbh_trajectory_comparison.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "dbh_correction_summary.csv"


GROUP_COLS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]

AGE_COL = "age_months"
DBH_COL = "dbh_cm"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 34 — VALIDATE CORRECTED DBH DATASET")
print("=" * 70)

print("\n[1/8] Loading datasets...")

if not RAW_PATH.exists():
    raise FileNotFoundError(f"Raw dataset tidak ditemukan: {RAW_PATH}")

if not CORRECTED_PATH.exists():
    raise FileNotFoundError(
        f"Corrected dataset tidak ditemukan: {CORRECTED_PATH}"
    )

raw = pd.read_csv(RAW_PATH)
corrected = pd.read_csv(CORRECTED_PATH)

print(f"Raw rows       : {len(raw):,}")
print(f"Corrected rows : {len(corrected):,}")

print(f"Raw columns       : {len(raw.columns)}")
print(f"Corrected columns : {len(corrected.columns)}")


# ============================================================
# 2. STRUCTURE VALIDATION
# ============================================================

print("\n[2/8] Validating structure...")

if list(raw.columns) != list(corrected.columns):
    raise RuntimeError(
        "ERROR: Struktur kolom raw dan corrected berbeda."
    )

if len(raw) != len(corrected):
    raise RuntimeError(
        "ERROR: Jumlah baris raw dan corrected berbeda."
    )

required = GROUP_COLS + [AGE_COL, DBH_COL]

missing_raw = [c for c in required if c not in raw.columns]
missing_corrected = [c for c in required if c not in corrected.columns]

if missing_raw:
    raise RuntimeError(
        f"Kolom wajib tidak ada di raw: {missing_raw}"
    )

if missing_corrected:
    raise RuntimeError(
        f"Kolom wajib tidak ada di corrected: {missing_corrected}"
    )

print("Structure validation : PASS")


# ============================================================
# 3. ALIGN ROWS
# ============================================================

print("\n[3/8] Aligning raw and corrected rows...")

# Correction script mempertahankan urutan baris asli.
# Kita tetap menggunakan key + age untuk memastikan perbandingan valid.

key_cols = GROUP_COLS + [AGE_COL]

raw_check = raw[key_cols].copy()
corrected_check = corrected[key_cols].copy()

if not raw_check.equals(corrected_check):
    print(
        "WARNING: Urutan/key row berbeda. "
        "Melakukan merge berdasarkan group + age."
    )

    comparison = raw[key_cols + [DBH_COL]].merge(
        corrected[key_cols + [DBH_COL]],
        on=key_cols,
        how="inner",
        suffixes=("_raw", "_corrected"),
    )

else:
    comparison = pd.DataFrame({
        **{
            col: raw[col].values
            for col in key_cols
        },
        "dbh_raw": raw[DBH_COL].values,
        "dbh_corrected": corrected[DBH_COL].values,
    })

print(f"Comparison rows : {len(comparison):,}")


# ============================================================
# 4. CORRECTION STATISTICS
# ============================================================

print("\n[4/8] Calculating correction statistics...")

comparison["correction_cm"] = (
    comparison["dbh_corrected"] -
    comparison["dbh_raw"]
)

comparison["was_corrected"] = (
    comparison["correction_cm"] > 1e-9
)

print("\nDBH correction statistics")
print("-" * 70)

print(
    f"Rows corrected       : "
    f"{comparison['was_corrected'].sum():,}"
)

print(
    f"Rows unchanged       : "
    f"{(~comparison['was_corrected']).sum():,}"
)

print(
    f"Mean correction      : "
    f"{comparison['correction_cm'].mean():.6f} cm"
)

print(
    f"Median correction    : "
    f"{comparison['correction_cm'].median():.6f} cm"
)

print(
    f"Maximum correction   : "
    f"{comparison['correction_cm'].max():.6f} cm"
)

print(
    f"Minimum correction   : "
    f"{comparison['correction_cm'].min():.6f} cm"
)


# ============================================================
# 5. CHECK MONOTONICITY
# ============================================================

print("\n[5/8] Checking DBH monotonicity...")

corrected_sorted = corrected.sort_values(
    GROUP_COLS + [AGE_COL]
).copy()

corrected_sorted["dbh_previous"] = (
    corrected_sorted
    .groupby(GROUP_COLS)[DBH_COL]
    .shift(1)
)

corrected_sorted["dbh_delta"] = (
    corrected_sorted[DBH_COL] -
    corrected_sorted["dbh_previous"]
)

transitions = corrected_sorted[
    corrected_sorted["dbh_previous"].notna()
].copy()

decreases = transitions[
    transitions["dbh_delta"] < -1e-9
]

stagnations = transitions[
    transitions["dbh_delta"].abs() <= 1e-9
]

increases = transitions[
    transitions["dbh_delta"] > 1e-9
]

print("\nCorrected chronological transitions")
print("-" * 70)

print(f"Total transitions : {len(transitions):,}")
print(f"Increases         : {len(increases):,}")
print(f"Stagnations       : {len(stagnations):,}")
print(f"Decreases         : {len(decreases):,}")

if len(decreases) == 0:
    print("\nDBH monotonicity : PASS")
else:
    print("\nDBH monotonicity : FAIL")
    print("\nLargest remaining decreases:")
    print(
        decreases
        .sort_values("dbh_delta")
        [
            GROUP_COLS
            + ["age_months", "dbh_previous", DBH_COL, "dbh_delta"]
        ]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# 6. AGE-BY-AGE VALIDATION
# ============================================================

print("\n[6/8] Checking every age transition...")

age_summary = (
    transitions
    .groupby(AGE_COL)
    .agg(
        transitions=("dbh_delta", "size"),
        increases=("dbh_delta", lambda x: (x > 1e-9).sum()),
        stagnations=("dbh_delta", lambda x: (x.abs() <= 1e-9).sum()),
        decreases=("dbh_delta", lambda x: (x < -1e-9).sum()),
        mean_delta_cm=("dbh_delta", "mean"),
        min_delta_cm=("dbh_delta", "min"),
        max_delta_cm=("dbh_delta", "max"),
    )
    .reset_index()
    .sort_values(AGE_COL)
)

print("\nAge transition summary:")
print(age_summary.to_string(index=False))


# ============================================================
# 7. PHI = 66 VALIDATION
# ============================================================

print("\n[7/8] Validating PHI = 66 months...")

phi = corrected[
    corrected[AGE_COL] == 66
].copy()

if len(phi) == 0:
    raise RuntimeError(
        "ERROR: Tidak ditemukan data age_months = 66."
    )

print("\nPHI = 66 months")
print("-" * 70)
print(f"Rows        : {len(phi):,}")
print(f"DBH mean    : {phi[DBH_COL].mean():.6f} cm")
print(f"DBH min     : {phi[DBH_COL].min():.6f} cm")
print(f"DBH max     : {phi[DBH_COL].max():.6f} cm")

# Pastikan tidak ada nilai corrected DBH di PHI yang
# lebih kecil daripada DBH pada umur sebelumnya.

phi_previous = (
    corrected_sorted[
        corrected_sorted[AGE_COL] < 66
    ]
    .groupby(GROUP_COLS)[DBH_COL]
    .max()
    .reset_index(name="max_dbh_before_phi")
)

phi_check = phi.merge(
    phi_previous,
    on=GROUP_COLS,
    how="left",
)

phi_check["phi_delta_from_previous_max"] = (
    phi_check[DBH_COL] -
    phi_check["max_dbh_before_phi"]
)

phi_decrease = phi_check[
    phi_check["phi_delta_from_previous_max"] < -1e-9
]

print(
    f"PHI DBH decreases vs previous history : "
    f"{len(phi_decrease):,}"
)

if len(phi_decrease) == 0:
    print("PHI DBH validation : PASS")
else:
    print("PHI DBH validation : FAIL")


# ============================================================
# 8. REPRESENTATIVE TRAJECTORIES
# ============================================================

print("\n[8/8] Selecting representative trajectories...")

# Ambil beberapa group deterministik agar hasil dapat direproduksi.
groups = (
    corrected_sorted[GROUP_COLS]
    .drop_duplicates()
    .sort_values(GROUP_COLS)
    .head(5)
)

representative = corrected_sorted.merge(
    groups,
    on=GROUP_COLS,
    how="inner",
)

representative = representative[
    GROUP_COLS
    + [
        AGE_COL,
        DBH_COL,
        "dbh_previous",
        "dbh_delta",
    ]
].copy()

representative.to_csv(
    DETAIL_OUTPUT,
    index=False,
)

print("\nRepresentative trajectories:")
print(
    representative.to_string(index=False)
)


# ============================================================
# SUMMARY FILE
# ============================================================

summary = pd.DataFrame({
    "metric": [
        "raw_rows",
        "corrected_rows",
        "raw_columns",
        "corrected_columns",
        "rows_corrected",
        "rows_unchanged",
        "mean_correction_cm",
        "median_correction_cm",
        "maximum_correction_cm",
        "minimum_correction_cm",
        "total_transitions",
        "increases",
        "stagnations",
        "decreases",
        "phi_rows",
        "phi_mean_dbh_cm",
        "phi_min_dbh_cm",
        "phi_max_dbh_cm",
        "phi_decrease_count",
    ],
    "value": [
        len(raw),
        len(corrected),
        len(raw.columns),
        len(corrected.columns),
        int(comparison["was_corrected"].sum()),
        int((~comparison["was_corrected"]).sum()),
        comparison["correction_cm"].mean(),
        comparison["correction_cm"].median(),
        comparison["correction_cm"].max(),
        comparison["correction_cm"].min(),
        len(transitions),
        len(increases),
        len(stagnations),
        len(decreases),
        len(phi),
        phi[DBH_COL].mean(),
        phi[DBH_COL].min(),
        phi[DBH_COL].max(),
        len(phi_decrease),
    ],
})

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)

age_summary.to_csv(
    OUTPUT_DIR / "dbh_age_transition_summary.csv",
    index=False,
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)

if len(decreases) == 0 and len(phi_decrease) == 0:
    print("STEP 34 RESULT : PASS")
    print("Corrected DBH tidak mengalami penurunan antar-age.")
else:
    print("STEP 34 RESULT : REVIEW REQUIRED")

print("=" * 70)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print(f"\nDetail:")
print(DETAIL_OUTPUT)

print(f"\nSummary:")
print(SUMMARY_OUTPUT)