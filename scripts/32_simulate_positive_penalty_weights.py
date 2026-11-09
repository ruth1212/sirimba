from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# TREE GROWTH ANALYTICS
# STEP 21H — SIMULATE POSITIVITY PENALTY WEIGHTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRACE_PATH = (
    PROJECT_ROOT
    / "results"
    / "pinns_positive_penalty_analysis.csv"
)


print("=" * 75)
print("STEP 21H — SIMULATE POSITIVITY PENALTY WEIGHTS")
print("=" * 75)


# ============================================================
# 1. LOAD ANALYSIS
# ============================================================

df = pd.read_csv(TRACE_PATH)

penalty = (
    df["positivity_penalty_normalized"]
    .astype(float)
)

physics_loss = (
    df["physics_loss_per_row"]
    .astype(float)
)


print("\nNegative rows :", len(df))

print(
    f"Mean positivity penalty : "
    f"{penalty.mean():.12f}"
)

print(
    f"Mean physics loss       : "
    f"{physics_loss.mean():.12f}"
)


# ============================================================
# 2. SIMULATE DIFFERENT WEIGHTS
# ============================================================

weights = [
    0.10,
    0.25,
    0.50,
    1.00,
    2.00,
    5.00,
    10.00,
]


print("\n" + "=" * 75)
print("2. PENALTY WEIGHT SIMULATION")
print("=" * 75)

print(
    f"{'lambda_positive':>18} | "
    f"{'penalty contribution':>22} | "
    f"{'physics contribution':>22} | "
    f"{'positive/physics':>18}"
)

print("-" * 90)


for weight in weights:

    positive_contribution = (
        weight
        * penalty.mean()
    )

    physics_contribution = (
        0.10
        * physics_loss.mean()
    )

    ratio = (
        positive_contribution
        / physics_contribution
        if physics_contribution != 0
        else np.nan
    )

    print(
        f"{weight:>18.2f} | "
        f"{positive_contribution:>22.12f} | "
        f"{physics_contribution:>22.12f} | "
        f"{ratio:>18.6f}"
    )


# ============================================================
# 3. FINAL
# ============================================================

print("\n" + "=" * 75)
print("STEP 21H COMPLETED")
print("=" * 75)

print("This was a simulation only.")
print("NO TRAINING WAS PERFORMED.")
print("NO MODEL FILE WAS MODIFIED.")
print("=" * 75)