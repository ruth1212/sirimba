from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# TREE GROWTH ANALYTICS
# STEP 21G — ANALYZE P6 TARGET VOLUME
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_ROOT
    / "results"
    / "model_train.csv"
)

TRACE_PATH = (
    PROJECT_ROOT
    / "results"
    / "pinns_positive_penalty_analysis.csv"
)


print("=" * 75)
print("STEP 21G — ANALYZE P6 TARGET VOLUME")
print("=" * 75)


# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

print("\n" + "=" * 75)
print("1. LOAD TRAINING DATA")
print("=" * 75)

df = pd.read_csv(TRAIN_PATH)

print(f"Training rows : {len(df):,}")


# ============================================================
# 2. FILTER P6
# ============================================================

print("\n" + "=" * 75)
print("2. FILTER AGE = 6 MONTHS")
print("=" * 75)

p6 = df[
    df["age_months"] == 6
].copy()

print(f"P6 rows : {len(p6):,}")


if len(p6) == 0:
    print("No P6 rows found.")
    raise SystemExit(0)


# ============================================================
# 3. TARGET VOLUME STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("3. P6 TARGET VOLUME STATISTICS")
print("=" * 75)

volume = p6[
    "volume_m3_tree"
].astype(float)

print(
    f"Minimum volume       : {volume.min():.12f}"
)

print(
    f"Maximum volume       : {volume.max():.12f}"
)

print(
    f"Mean volume          : {volume.mean():.12f}"
)

print(
    f"Median volume        : {volume.median():.12f}"
)

print(
    f"Std volume           : {volume.std():.12f}"
)


# ============================================================
# 4. NEGATIVE TARGET CHECK
# ============================================================

print("\n" + "=" * 75)
print("4. NEGATIVE TARGET CHECK")
print("=" * 75)

negative_targets = (
    volume < 0
)

print(
    f"Negative target rows : "
    f"{negative_targets.sum():,}"
)

print(
    f"Zero target rows     : "
    f"{(volume == 0).sum():,}"
)


# ============================================================
# 5. NEAR-ZERO TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("5. NEAR-ZERO TARGET DISTRIBUTION")
print("=" * 75)

thresholds = [
    0.0001,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
]

for threshold in thresholds:

    count = (
        volume < threshold
    ).sum()

    percentage = (
        count / len(volume) * 100
    )

    print(
        f"Volume < {threshold:<8g} : "
        f"{count:>6,} rows "
        f"({percentage:>7.3f}%)"
    )


# ============================================================
# 6. P6 TARGET DISTRIBUTION BY AVAILABLE COLUMNS
# ============================================================

print("\n" + "=" * 75)
print("6. AVAILABLE P6 COLUMNS")
print("=" * 75)

print("Columns available in model_train.csv:")

for column in p6.columns:
    print(f"  - {column}")

print("\nSpecies/clone grouping skipped.")
print(
    "Reason: model_train.csv does not contain "
    "'species' or 'clone' columns."
)


# ============================================================
# 7. COMPARE WITH NEGATIVE PINNs CASES
# ============================================================

print("\n" + "=" * 75)
print("7. COMPARE NEGATIVE PINNs CASES")
print("=" * 75)

trace = pd.read_csv(
    TRACE_PATH
)

print(
    f"Negative PINNs rows in trace : "
    f"{len(trace):,}"
)

print(
    f"Mean PINNs negative volume   : "
    f"{trace['pred_volume_m3_tree'].mean():.12f}"
)

print(
    f"Minimum PINNs volume         : "
    f"{trace['pred_volume_m3_tree'].min():.12f}"
)

print(
    f"Mean physics volume          : "
    f"{trace['physics_volume_m3_tree'].mean():.12f}"
)


# ============================================================
# 8. FINAL
# ============================================================

print("\n" + "=" * 75)
print("STEP 21G COMPLETED")
print("=" * 75)

print("NO TRAINING WAS PERFORMED.")
print("NO MODEL FILE WAS MODIFIED.")
print("NO DASHBOARD DATA WAS MODIFIED.")

print("=" * 75)


# ============================================================
# 7. COMPARE WITH NEGATIVE PINNs CASES
# ============================================================

print("\n" + "=" * 75)
print("7. COMPARE NEGATIVE PINNs CASES")
print("=" * 75)

trace = pd.read_csv(
    TRACE_PATH
)

print(
    f"Negative PINNs rows in trace : "
    f"{len(trace):,}"
)

print(
    f"Mean PINNs negative volume   : "
    f"{trace['pred_volume_m3_tree'].mean():.12f}"
)

print(
    f"Minimum PINNs volume         : "
    f"{trace['pred_volume_m3_tree'].min():.12f}"
)

print(
    f"Mean physics volume          : "
    f"{trace['physics_volume_m3_tree'].mean():.12f}"
)


# ============================================================
# 8. FINAL
# ============================================================

print("\n" + "=" * 75)
print("STEP 21G COMPLETED")
print("=" * 75)

print("NO TRAINING WAS PERFORMED.")
print("NO MODEL FILE WAS MODIFIED.")
print("NO DASHBOARD DATA WAS MODIFIED.")

print("=" * 75)