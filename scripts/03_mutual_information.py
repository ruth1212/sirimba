from pathlib import Path

import pandas as pd
from sklearn.feature_selection import mutual_info_regression


# ============================================================
# TREE GROWTH FEATURE SELECTION
# STEP 3 - MUTUAL INFORMATION
# ============================================================

print("=" * 70)
print("TREE GROWTH FEATURE SELECTION")
print("STEP 3 - MUTUAL INFORMATION")
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

    print(
        f"Remaining rows: "
        f"{len(df):,}"
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
# 5. TARGETS
# ============================================================

TARGETS = [
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]


# ============================================================
# 6. CHECK FEATURES
# ============================================================

available_features = [
    feature
    for feature in FEATURES
    if feature in df.columns
]

print("\n[3] Features used:")

for feature in available_features:
    print(f"- {feature}")


# ============================================================
# 7. SAMPLE DATA
# ============================================================

# Mutual Information on the full 607k rows is unnecessary
# and can be slow. We use a reproducible random sample.
#
# This sample is only used for feature selection.
# The final models will use the complete modeling dataset.

SAMPLE_SIZE = 100_000

if len(df) > SAMPLE_SIZE:

    sample_df = df.sample(
        n=SAMPLE_SIZE,
        random_state=42
    ).reset_index(drop=True)

else:

    sample_df = df.copy()

print(
    f"\n[4] MI sample size: "
    f"{len(sample_df):,}"
)


# ============================================================
# 8. CALCULATE MUTUAL INFORMATION
# ============================================================

results = []


for target in TARGETS:

    print(
        f"\nCalculating MI for target: {target}"
    )

    X = sample_df[
        available_features
    ]

    y = sample_df[target]

    mi_values = mutual_info_regression(
        X,
        y,
        random_state=42
    )

    for feature, mi in zip(
        available_features,
        mi_values
    ):

        results.append({
            "target": target,
            "feature": feature,
            "mutual_information": mi,
        })


# ============================================================
# 9. RESULT DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=[
        "target",
        "mutual_information"
    ],
    ascending=[
        True,
        False
    ]
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

output_path = (
    RESULTS_DIR /
    "mutual_information.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MUTUAL INFORMATION RESULTS")
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
                "mutual_information"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 12. FINISH
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 SELESAI")
print("=" * 70)

print(
    f"\nHasil disimpan di:"
)

print(output_path)