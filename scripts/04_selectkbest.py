from pathlib import Path

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression


# ============================================================
# TREE GROWTH FEATURE SELECTION
# STEP 4 - SELECTKBEST
# ============================================================

print("=" * 70)
print("TREE GROWTH FEATURE SELECTION")
print("STEP 4 - SELECTKBEST")
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
# 6. SAMPLE
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
# 7. CHECK FEATURES
# ============================================================

available_features = [
    feature
    for feature in FEATURES
    if feature in sample_df.columns
]


# ============================================================
# 8. RUN SELECTKBEST
# ============================================================

K = 8

results = []


for target in TARGETS:

    print(
        f"\nCalculating SelectKBest for: {target}"
    )

    X = sample_df[
        available_features
    ]

    y = sample_df[target]

    selector = SelectKBest(
        score_func=f_regression,
        k=K
    )

    selector.fit(X, y)

    scores = selector.scores_
    p_values = selector.pvalues_

    selected_mask = selector.get_support()

    for feature, score, p_value, selected in zip(
        available_features,
        scores,
        p_values,
        selected_mask
    ):

        results.append({
            "target": target,
            "feature": feature,
            "f_score": score,
            "p_value": p_value,
            "selected_top_k": selected,
        })


# ============================================================
# 9. RESULT DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=[
        "target",
        "f_score"
    ],
    ascending=[
        True,
        False
    ]
)


# ============================================================
# 10. SAVE COMPLETE RESULTS
# ============================================================

output_path = (
    RESULTS_DIR /
    "selectkbest_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# 11. DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("SELECTKBEST RESULTS")
print("=" * 70)


for target in TARGETS:

    print(
        f"\n--- TARGET: {target} ---"
    )

    target_results = results_df[
        results_df["target"] == target
    ].copy()

    print(
        target_results[
            [
                "feature",
                "f_score",
                "p_value",
                "selected_top_k"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 12. FINISH
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 SELESAI")
print("=" * 70)

print(
    f"\nHasil disimpan di:"
)

print(output_path)