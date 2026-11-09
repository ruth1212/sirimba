from pathlib import Path

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ============================================================
# TREE GROWTH FEATURE SELECTION
# STEP 5 - REDUNDANCY & MULTICOLLINEARITY
# ============================================================

print("=" * 70)
print("TREE GROWTH FEATURE SELECTION")
print("STEP 5 - REDUNDANCY & MULTICOLLINEARITY")
print("=" * 70)


# ============================================================
# 1. PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n[1] Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(
    f"[OK] Dataset loaded: "
    f"{df.shape[0]:,} rows × {df.shape[1]} columns"
)


# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

duplicate_count = df.duplicated().sum()

print(
    f"\n[2] Duplicate rows: "
    f"{duplicate_count:,}"
)

if duplicate_count > 0:

    df = df.drop_duplicates().reset_index(drop=True)

    print(
        f"[OK] Duplicates removed."
    )


# ============================================================
# 4. FEATURES
# ============================================================

FEATURES = [
    "area_ha",
    "stand_area_total_ha",
    "spacing_x_m",
    "spacing_y_m",
    "trees_per_ha",
    "estimated_trees_compartment",
    "clone_factor",
    "planting_year",
    "planting_cohort",
    "inventory_year",
    "age_months",
    "condition_score",
    "disease_score",
]


# ============================================================
# 5. SAMPLE
# ============================================================

SAMPLE_SIZE = 100_000

if len(df) > SAMPLE_SIZE:

    sample_df = df.sample(
        n=SAMPLE_SIZE,
        random_state=42
    ).reset_index(drop=True)

else:

    sample_df = df.copy()

print(
    f"\n[3] Sample size: "
    f"{len(sample_df):,}"
)


# ============================================================
# 6. CORRELATION MATRIX
# ============================================================

print("\n[4] Calculating Pearson correlation...")

X = sample_df[FEATURES]

correlation_matrix = X.corr(
    method="pearson"
)

correlation_output = (
    RESULTS_DIR /
    "feature_correlation_matrix.csv"
)

correlation_matrix.to_csv(
    correlation_output
)


# ============================================================
# 7. HIGH CORRELATION PAIRS
# ============================================================

print("\n" + "=" * 70)
print("HIGH CORRELATION PAIRS")
print("=" * 70)

THRESHOLD = 0.80

pairs = []

for i in range(len(FEATURES)):

    for j in range(i + 1, len(FEATURES)):

        feature_a = FEATURES[i]
        feature_b = FEATURES[j]

        correlation = correlation_matrix.loc[
            feature_a,
            feature_b
        ]

        if abs(correlation) >= THRESHOLD:

            pairs.append({
                "feature_1": feature_a,
                "feature_2": feature_b,
                "correlation": correlation,
                "absolute_correlation": abs(correlation),
            })


pairs_df = pd.DataFrame(pairs)

if len(pairs_df) > 0:

    pairs_df = pairs_df.sort_values(
        by="absolute_correlation",
        ascending=False
    )

    print(
        pairs_df.to_string(
            index=False
        )
    )

else:

    print(
        "Tidak ada pasangan dengan "
        f"|correlation| >= {THRESHOLD}"
    )


# ============================================================
# 8. SAVE HIGH CORRELATION PAIRS
# ============================================================

pairs_output = (
    RESULTS_DIR /
    "high_correlation_pairs.csv"
)

pairs_df.to_csv(
    pairs_output,
    index=False
)


# ============================================================
# 9. VIF
# ============================================================

print("\n" + "=" * 70)
print("VARIANCE INFLATION FACTOR (VIF)")
print("=" * 70)

print(
    "\nMenghitung VIF..."
)

# VIF dapat menjadi sangat besar apabila terdapat
# fitur yang identik. Oleh karena itu kita menangani
# hasil error/inf secara eksplisit.

vif_results = []

for i, feature in enumerate(FEATURES):

    try:

        vif_value = variance_inflation_factor(
            X.values,
            i
        )

    except Exception:

        vif_value = float("inf")

    vif_results.append({
        "feature": feature,
        "VIF": vif_value,
    })


vif_df = pd.DataFrame(vif_results)

vif_df = vif_df.sort_values(
    by="VIF",
    ascending=False
)


# ============================================================
# 10. DISPLAY VIF
# ============================================================

print(
    vif_df.to_string(
        index=False
    )
)


# ============================================================
# 11. SAVE VIF
# ============================================================

vif_output = (
    RESULTS_DIR /
    "vif_results.csv"
)

vif_df.to_csv(
    vif_output,
    index=False
)


# ============================================================
# 12. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 SELESAI")
print("=" * 70)

print("\nFiles saved:")

print(correlation_output)
print(pairs_output)
print(vif_output)