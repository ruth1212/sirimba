from pathlib import Path
import pandas as pd


# ============================================================
# TREE GROWTH FEATURE SELECTION
# STEP 2 - SPEARMAN CORRELATION
# ============================================================

print("=" * 70)
print("TREE GROWTH FEATURE SELECTION")
print("STEP 2 - SPEARMAN CORRELATION")
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
    f"\n[2] Duplicate rows found: "
    f"{duplicate_count:,}"
)

if duplicate_count > 0:

    df = df.drop_duplicates().reset_index(drop=True)

    print(
        f"[OK] Duplicates removed."
    )

    print(
        f"Remaining rows: "
        f"{len(df):,}"
    )


# ============================================================
# 4. TARGETS
# ============================================================

TARGETS = [
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]


# ============================================================
# 5. CANDIDATE NUMERIC FEATURES
# ============================================================

CANDIDATE_FEATURES = [
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
# 6. CHECK COLUMNS
# ============================================================

print("\n[3] Checking candidate features...")

available_features = []

for feature in CANDIDATE_FEATURES:

    if feature in df.columns:

        available_features.append(feature)

        print(
            f"[OK] {feature}"
        )

    else:

        print(
            f"[WARNING] Missing: {feature}"
        )


# ============================================================
# 7. SPEARMAN CORRELATION
# ============================================================

print("\n[4] Calculating Spearman correlation...")

results = []


for target in TARGETS:

    print(
        f"\nTarget: {target}"
    )

    for feature in available_features:

        correlation = df[
            [feature, target]
        ].corr(
            method="spearman"
        ).iloc[0, 1]

        results.append({
            "target": target,
            "feature": feature,
            "spearman_rho": correlation,
            "absolute_rho": abs(correlation),
        })


# ============================================================
# 8. RESULT DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=[
        "target",
        "absolute_rho"
    ],
    ascending=[
        True,
        False
    ]
)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

output_path = (
    RESULTS_DIR /
    "spearman_correlation.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SPEARMAN RESULTS")
print("=" * 70)

for target in TARGETS:

    print(
        f"\n--- TARGET: {target} ---"
    )

    target_results = results_df[
        results_df["target"] == target
    ]

    print(
        target_results[
            [
                "feature",
                "spearman_rho",
                "absolute_rho"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 11. FINISH
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 SELESAI")
print("=" * 70)

print(
    f"\nHasil disimpan di:"
)

print(output_path)