from pathlib import Path

import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# TREE GROWTH FEATURE SELECTION
# STEP 6 - MODEL FEATURE IMPORTANCE
# ============================================================

print("=" * 70)
print("TREE GROWTH FEATURE SELECTION")
print("STEP 6 - MODEL-BASED FEATURE IMPORTANCE")
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
        "[OK] Duplicates removed."
    )


# ============================================================
# 4. FEATURES
# ============================================================

FEATURES = [
    "area_ha",
    "spacing_x_m",
    "spacing_y_m",
    "trees_per_ha",
    "clone_factor",
    "planting_year",
    "inventory_year",
    "age_months",
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
# 7. SAMPLE
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
    f"\n[4] Sample size: "
    f"{len(sample_df):,}"
)


# ============================================================
# 8. GROUP-AWARE SPLIT
# ============================================================

print(
    "\n[5] Creating group-aware train/test split..."
)

X = sample_df[
    available_features
]

groups = sample_df[
    "compartment_id"
]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_index, test_index = next(
    splitter.split(
        X,
        groups=groups
    )
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

print(
    f"Train rows: {len(X_train):,}"
)

print(
    f"Test rows : {len(X_test):,}"
)

print(
    f"Train groups: "
    f"{groups.iloc[train_index].nunique()}"
)

print(
    f"Test groups : "
    f"{groups.iloc[test_index].nunique()}"
)


# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================

all_importance_results = []
validation_results = []


for target in TARGETS:

    print("\n" + "-" * 70)

    print(
        f"Training Extra Trees for target: "
        f"{target}"
    )

    y = sample_df[target]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    model = ExtraTreesRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        max_features=1.0,
        min_samples_leaf=2
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print(
        f"Test MAE : {mae:.6f}"
    )

    print(
        f"Test R²  : {r2:.6f}"
    )

    validation_results.append({
        "target": target,
        "test_mae": mae,
        "test_r2": r2,
    })


    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    for feature, importance in zip(
        available_features,
        model.feature_importances_
    ):

        all_importance_results.append({
            "target": target,
            "feature": feature,
            "importance": importance,
        })


# ============================================================
# 10. SAVE FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame(
    all_importance_results
)

importance_df = importance_df.sort_values(
    by=[
        "target",
        "importance"
    ],
    ascending=[
        True,
        False
    ]
)

importance_output = (
    RESULTS_DIR /
    "extra_trees_feature_importance.csv"
)

importance_df.to_csv(
    importance_output,
    index=False
)


# ============================================================
# 11. SAVE VALIDATION
# ============================================================

validation_df = pd.DataFrame(
    validation_results
)

validation_output = (
    RESULTS_DIR /
    "model_feature_importance_validation.csv"
)

validation_df.to_csv(
    validation_output,
    index=False
)


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE RESULTS")
print("=" * 70)


for target in TARGETS:

    print(
        f"\n--- TARGET: {target} ---"
    )

    target_results = importance_df[
        importance_df["target"] == target
    ]

    print(
        target_results.to_string(
            index=False
        )
    )


# ============================================================
# 13. VALIDATION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL VALIDATION")
print("=" * 70)

print(
    validation_df.to_string(
        index=False
    )
)


# ============================================================
# 14. FINISH
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 SELESAI")
print("=" * 70)

print("\nFiles saved:")
print(importance_output)
print(validation_output)