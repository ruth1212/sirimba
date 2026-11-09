from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 35
# INSPEKSI LOGIKA PERTUMBUHAN HEIGHT
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "dataset_corrected.csv"

OUTPUT_DIR = BASE_DIR / "results" / "height_growth_inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_AGE = OUTPUT_DIR / "height_age_summary.csv"
OUTPUT_TRANSITION = OUTPUT_DIR / "height_transition_summary.csv"
OUTPUT_DECREASE = OUTPUT_DIR / "height_decrease_records.csv"


GROUP_COLS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]

AGE_COL = "age_months"
HEIGHT_COL = "height_m"

PHI_AGE = 66


# ============================================================
# 1. LOAD
# ============================================================

print("=" * 75)
print("STEP 35 — INSPEKSI HEIGHT GROWTH LOGIC")
print("=" * 75)

print("\n[1/7] Loading corrected DBH dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

required = GROUP_COLS + [
    AGE_COL,
    HEIGHT_COL,
    "disease_score",
]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise RuntimeError(
        f"Kolom wajib tidak ditemukan: {missing}"
    )

print("Required columns : PASS")


# ============================================================
# 2. BASIC HEIGHT STATISTICS
# ============================================================

print("\n[2/7] Basic height statistics...")

print("-" * 75)
print(f"Height minimum : {df[HEIGHT_COL].min():.6f} m")
print(f"Height maximum : {df[HEIGHT_COL].max():.6f} m")
print(f"Height mean    : {df[HEIGHT_COL].mean():.6f} m")
print(f"Height median  : {df[HEIGHT_COL].median():.6f} m")


# ============================================================
# 3. AGE PROFILE
# ============================================================

print("\n[3/7] Building age profile...")

age_summary = (
    df
    .groupby(AGE_COL)
    .agg(
        rows=(HEIGHT_COL, "size"),
        mean_height_m=(HEIGHT_COL, "mean"),
        median_height_m=(HEIGHT_COL, "median"),
        min_height_m=(HEIGHT_COL, "min"),
        max_height_m=(HEIGHT_COL, "max"),
        std_height_m=(HEIGHT_COL, "std"),
        mean_disease_score=("disease_score", "mean"),
    )
    .reset_index()
    .sort_values(AGE_COL)
)

# Growth period label
age_summary["growth_period"] = np.where(
    age_summary[AGE_COL] == PHI_AGE,
    "PHI",
    "P" + age_summary[AGE_COL].astype(str)
)

age_summary["height_delta_from_previous_age"] = (
    age_summary["mean_height_m"].diff()
)

age_summary["height_growth_rate_per_month"] = (
    age_summary["height_delta_from_previous_age"]
    /
    age_summary[AGE_COL].diff()
)

print("\nAge-level height profile:")
print(
    age_summary[
        [
            AGE_COL,
            "growth_period",
            "rows",
            "mean_height_m",
            "height_delta_from_previous_age",
            "height_growth_rate_per_month",
        ]
    ].to_string(index=False)
)


# ============================================================
# 4. GROUP-LEVEL TRANSITIONS
# ============================================================

print("\n[4/7] Inspecting group-level height transitions...")

df_sorted = (
    df
    .sort_values(GROUP_COLS + [AGE_COL])
    .copy()
)

df_sorted["height_previous"] = (
    df_sorted
    .groupby(GROUP_COLS)[HEIGHT_COL]
    .shift(1)
)

df_sorted["age_previous"] = (
    df_sorted
    .groupby(GROUP_COLS)[AGE_COL]
    .shift(1)
)

df_sorted["height_delta"] = (
    df_sorted[HEIGHT_COL] -
    df_sorted["height_previous"]
)

df_sorted["age_delta"] = (
    df_sorted[AGE_COL] -
    df_sorted["age_previous"]
)

transitions = df_sorted[
    df_sorted["height_previous"].notna()
].copy()

transitions["growth_rate_per_month"] = (
    transitions["height_delta"] /
    transitions["age_delta"]
)


# ============================================================
# 5. DECREASE ANALYSIS
# ============================================================

print("\n[5/7] Analysing height decreases...")

decreases = transitions[
    transitions["height_delta"] < -1e-9
].copy()

increases = transitions[
    transitions["height_delta"] > 1e-9
].copy()

stagnations = transitions[
    transitions["height_delta"].abs() <= 1e-9
].copy()

before_phi = transitions[
    transitions[AGE_COL] <= PHI_AGE
]

after_phi = transitions[
    transitions[AGE_COL] > PHI_AGE
]

before_phi_decreases = before_phi[
    before_phi["height_delta"] < -1e-9
]

after_phi_decreases = after_phi[
    after_phi["height_delta"] < -1e-9
]


print("\nHEIGHT TRANSITION SUMMARY")
print("-" * 75)

print(f"Total transitions          : {len(transitions):,}")
print(f"Increases                  : {len(increases):,}")
print(f"Stagnations                : {len(stagnations):,}")
print(f"Decreases                  : {len(decreases):,}")

print("\nPHI boundary = 66 months")

print(
    f"Decreases P6-PHI          : "
    f"{len(before_phi_decreases):,}"
)

print(
    f"Decreases AFTER PHI       : "
    f"{len(after_phi_decreases):,}"
)


# ============================================================
# 6. AGE-SPECIFIC TRANSITIONS
# ============================================================

print("\n[6/7] Analysing transitions by age...")

transition_summary = (
    transitions
    .groupby(AGE_COL)
    .agg(
        transitions=("height_delta", "size"),

        increases=(
            "height_delta",
            lambda x: (x > 1e-9).sum()
        ),

        stagnations=(
            "height_delta",
            lambda x: (x.abs() <= 1e-9).sum()
        ),

        decreases=(
            "height_delta",
            lambda x: (x < -1e-9).sum()
        ),

        mean_delta_m=(
            "height_delta",
            "mean"
        ),

        median_delta_m=(
            "height_delta",
            "median"
        ),

        min_delta_m=(
            "height_delta",
            "min"
        ),

        max_delta_m=(
            "height_delta",
            "max"
        ),

        mean_growth_rate_per_month=(
            "growth_rate_per_month",
            "mean"
        ),

        mean_disease_score=(
            "disease_score",
            "mean"
        ),
    )
    .reset_index()
    .sort_values(AGE_COL)
)

print("\nAge transition summary:")
print(
    transition_summary.to_string(index=False)
)


# ============================================================
# 7. LARGEST DECREASES
# ============================================================

print("\n[7/7] Finding largest height decreases...")

if len(decreases) > 0:

    largest_decreases = (
        decreases
        .sort_values("height_delta")
        .head(30)
        [
            GROUP_COLS
            + [
                "age_previous",
                AGE_COL,
                "height_previous",
                HEIGHT_COL,
                "height_delta",
                "growth_rate_per_month",
                "disease_score",
            ]
        ]
    )

    print("\nLargest height decreases:")
    print(
        largest_decreases.to_string(index=False)
    )

    largest_decreases.to_csv(
        OUTPUT_DECREASE,
        index=False
    )

else:

    print("\nTidak ditemukan height decrease.")


# ============================================================
# SAVE OUTPUTS
# ============================================================

age_summary.to_csv(
    OUTPUT_AGE,
    index=False
)

transition_summary.to_csv(
    OUTPUT_TRANSITION,
    index=False
)


# ============================================================
# PHI PROFILE
# ============================================================

print("\n" + "=" * 75)
print("PHI = 66 MONTHS PROFILE")
print("=" * 75)

phi = df[
    df[AGE_COL] == PHI_AGE
]

if len(phi) == 0:
    print("WARNING: Tidak ada data age 66.")
else:

    print(f"Rows             : {len(phi):,}")
    print(
        f"Mean height      : "
        f"{phi[HEIGHT_COL].mean():.6f} m"
    )
    print(
        f"Median height    : "
        f"{phi[HEIGHT_COL].median():.6f} m"
    )
    print(
        f"Minimum height   : "
        f"{phi[HEIGHT_COL].min():.6f} m"
    )
    print(
        f"Maximum height   : "
        f"{phi[HEIGHT_COL].max():.6f} m"
    )


# ============================================================
# GROWTH LOGIC DIAGNOSTIC
# ============================================================

print("\n" + "=" * 75)
print("GROWTH LOGIC DIAGNOSTIC")
print("=" * 75)

# Rata-rata growth rate berdasarkan fase umur.
early = transition_summary[
    transition_summary[AGE_COL].between(12, 30)
]

middle = transition_summary[
    transition_summary[AGE_COL].between(36, 54)
]

late = transition_summary[
    transition_summary[AGE_COL] >= 60
]

def safe_mean(series):
    if len(series) == 0:
        return np.nan
    return series.mean()


early_rate = safe_mean(
    early["mean_growth_rate_per_month"]
)

middle_rate = safe_mean(
    middle["mean_growth_rate_per_month"]
)

late_rate = safe_mean(
    late["mean_growth_rate_per_month"]
)

print(
    f"Early growth rate   : "
    f"{early_rate:.8f} m/month"
)

print(
    f"Middle growth rate  : "
    f"{middle_rate:.8f} m/month"
)

print(
    f"Late growth rate    : "
    f"{late_rate:.8f} m/month"
)

print("\nInterpretasi:")
print(
    "Early > Middle > Late menunjukkan pola "
    "pertumbuhan cepat → melambat."
)

print(
    "Late mendekati 0 menunjukkan stagnasi "
    "pada umur lanjut."
)

print(
    "Nilai negatif pada transition rate menunjukkan "
    "penurunan aktual yang perlu diperiksa."
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("STEP 35 INSPECTION COMPLETE")
print("=" * 75)

print("\nOutput:")
print(OUTPUT_AGE)
print(OUTPUT_TRANSITION)

if len(decreases) > 0:
    print(OUTPUT_DECREASE)

print("\nSTATUS: INSPECTION ONLY")
print("Belum ada perubahan data Height.")
print("Belum dilakukan koreksi Height.")
print("Belum dilakukan training model.")