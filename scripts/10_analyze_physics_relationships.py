from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression


# ============================================================
# STEP 7D-1
# AUDIT PHYSICS RELATIONSHIPS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "results" / "model_train.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
SAMPLE_SIZE = 100_000

COLUMNS = [
    "age_months",
    "clone_factor",
    "disease_score",
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]


# ============================================================
# HELPER
# ============================================================

def spearman_report(df, x, y):
    rho, p_value = spearmanr(
        df[x],
        df[y],
    )

    return {
        "feature_x": x,
        "feature_y": y,
        "spearman_rho": rho,
        "p_value": p_value,
    }


# ============================================================
# START
# ============================================================

print("=" * 75)
print("TREE GROWTH ANALYTICS")
print("STEP 7D-1 - PHYSICS RELATIONSHIP AUDIT")
print("=" * 75)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/6] Loading training data...")

df = pd.read_csv(DATA_PATH)

print(f"Training rows : {len(df):,}")


# ============================================================
# 2. SAMPLE FOR FAST ANALYSIS
# ============================================================

print("\n[2/6] Creating analysis sample...")

sample_size = min(
    SAMPLE_SIZE,
    len(df),
)

sample = df[
    COLUMNS
].sample(
    n=sample_size,
    random_state=RANDOM_STATE,
)

print(f"Analysis sample : {len(sample):,}")


# ============================================================
# 3. AGE → GROWTH RELATIONSHIPS
# ============================================================

print("\n[3/6] Analyzing age-growth relationships...")

age_relationships = []

for target in [
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]:

    result = spearman_report(
        sample,
        "age_months",
        target,
    )

    age_relationships.append(result)

    print(
        f"age_months -> {target}: "
        f"rho = {result['spearman_rho']:.6f}, "
        f"p = {result['p_value']:.6e}"
    )


# ============================================================
# 4. DISEASE / CLONE RELATIONSHIPS
# ============================================================

print("\n[4/6] Analyzing disease and clone relationships...")

factor_relationships = []

for feature in [
    "disease_score",
    "clone_factor",
]:

    for target in [
        "dbh_cm",
        "height_m",
        "volume_m3_tree",
    ]:

        result = spearman_report(
            sample,
            feature,
            target,
        )

        factor_relationships.append(result)

        print(
            f"{feature} -> {target}: "
            f"rho = {result['spearman_rho']:.6f}, "
            f"p = {result['p_value']:.6e}"
        )


# ============================================================
# 5. VOLUME CONSISTENCY RELATIONSHIP
# ============================================================

print("\n[5/6] Analyzing volume consistency...")

# A geometric-style proxy:
#
# DBH^2 * Height
#
# This is NOT declared as the true biological volume equation.
# It is only used to determine whether volume has a strong
# empirical relationship with diameter and height.

sample["dbh_height_proxy"] = (
    sample["dbh_cm"] ** 2
) * sample["height_m"]


# Spearman correlation

volume_proxy_rho, volume_proxy_p = spearmanr(
    sample["dbh_height_proxy"],
    sample["volume_m3_tree"],
)

print(
    "\nDBH² × Height -> Volume:"
)

print(
    f"Spearman rho = {volume_proxy_rho:.6f}"
)

print(
    f"p-value      = {volume_proxy_p:.6e}"
)


# ============================================================
# LOG-SPACE EMPIRICAL RELATIONSHIP
# ============================================================

positive_mask = (
    (sample["dbh_height_proxy"] > 0)
    &
    (sample["volume_m3_tree"] > 0)
)

volume_model_data = sample.loc[
    positive_mask
].copy()

X_log = np.log(
    volume_model_data[
        ["dbh_height_proxy"]
    ]
)

y_log = np.log(
    volume_model_data[
        "volume_m3_tree"
    ]
)

log_model = LinearRegression()

log_model.fit(
    X_log,
    y_log,
)

log_r2 = log_model.score(
    X_log,
    y_log,
)

log_slope = float(
    log_model.coef_[0]
)

log_intercept = float(
    log_model.intercept_
)

print("\nEmpirical log-volume relationship:")

print(
    "log(volume) = "
    f"{log_intercept:.6f} + "
    f"{log_slope:.6f} × "
    "log(DBH² × Height)"
)

print(
    f"Log-space R² = {log_r2:.6f}"
)


# ============================================================
# VOLUME / GEOMETRIC PROXY RATIO
# ============================================================

sample["volume_proxy_ratio"] = (
    sample["volume_m3_tree"]
    / sample["dbh_height_proxy"]
)

ratio_summary = (
    sample["volume_proxy_ratio"]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
    .dropna()
    .describe()
)

print("\nVolume / (DBH² × Height) summary:")

print(
    ratio_summary.to_string()
)


# ============================================================
# AGE BIN ANALYSIS
# ============================================================

print("\nAnalyzing growth behavior by age...")

age_bins = [
    0,
    12,
    24,
    36,
    48,
    60,
    72,
    84,
    96,
    108,
    120,
    132,
    144,
    156,
    168,
    180,
    192,
    204,
    216,
    228,
    240,
    np.inf,
]

age_bins = [
    0,
    12,
    24,
    36,
    48,
    60,
    72,
    84,
    96,
    108,
    120,
    132,
    144,
    156,
    168,
    180,
    192,
    204,
    216,
    228,
    240,
    np.inf,
]

age_labels = [
    "P0-P12",
    "P12-P24",
    "P24-P36",
    "P36-P48",
    "P48-P60",
    "P60-P72",
    "P72-P84",
    "P84-P96",
    "P96-P108",
    "P108-P120",
    "P120-P132",
    "P132-P144",
    "P144-P156",
    "P156-P168",
    "P168-P180",
    "P180-P192",
    "P192-P204",
    "P204-P216",
    "P216-P228",
    "P228-P240",
    "P240+",
]

sample["age_bin"] = pd.cut(
    sample["age_months"],
    bins=age_bins,
    labels=age_labels,
    include_lowest=True,
)

age_summary = (
    sample
    .groupby(
        "age_bin",
        observed=True,
    )
    [
        [
            "age_months",
            "dbh_cm",
            "height_m",
            "volume_m3_tree",
        ]
    ]
    .agg(
        {
            "age_months": "mean",
            "dbh_cm": "mean",
            "height_m": "mean",
            "volume_m3_tree": "mean",
        }
    )
    .reset_index()
)

print("\nAge-growth summary:")
print(
    age_summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

print("\n[6/6] Saving audit results...")


age_relationships_df = pd.DataFrame(
    age_relationships
)

factor_relationships_df = pd.DataFrame(
    factor_relationships
)

volume_relationship_df = pd.DataFrame(
    [
        {
            "relationship": "DBH^2 x Height -> Volume",
            "spearman_rho": volume_proxy_rho,
            "p_value": volume_proxy_p,
            "log_slope": log_slope,
            "log_intercept": log_intercept,
            "log_r2": log_r2,
        }
    ]
)


age_relationships_df.to_csv(
    RESULTS_DIR
    / "pinns_age_growth_relationships.csv",
    index=False,
)

factor_relationships_df.to_csv(
    RESULTS_DIR
    / "pinns_factor_relationships.csv",
    index=False,
)

volume_relationship_df.to_csv(
    RESULTS_DIR
    / "pinns_volume_relationship.csv",
    index=False,
)

age_summary.to_csv(
    RESULTS_DIR
    / "pinns_age_growth_summary.csv",
    index=False,
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("STEP 7D-1 COMPLETED")
print("=" * 75)

print("\nSaved files:")

print(
    "  - results/pinns_age_growth_relationships.csv"
)

print(
    "  - results/pinns_factor_relationships.csv"
)

print(
    "  - results/pinns_volume_relationship.csv"
)

print(
    "  - results/pinns_age_growth_summary.csv"
)

print("\nIMPORTANT:")
print(
    "The DBH² × Height relationship is an empirical audit only."
)

print(
    "It is NOT yet declared as the biological volume equation."
)

print(
    "Physics loss will be designed only after reviewing these results."
)

print("=" * 75)