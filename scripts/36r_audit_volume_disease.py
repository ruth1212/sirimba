from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 36R
# AUDIT VOLUME DECLINE vs DISEASE SCORE
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "dataset_phi66_dbh_v2.csv"

GROUP_COLS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]

AGE_COL = "age_months"
VOLUME_COL = "volume_m3_tree"
DISEASE_COL = "disease_score"

START_AGE = 18
END_AGE = 66


print("=" * 75)
print("STEP 36R - VOLUME DECLINE vs DISEASE SCORE")
print("=" * 75)


# ============================================================
# 1. LOAD
# ============================================================

print("\n[1/8] Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# ============================================================
# 2. VALIDATE
# ============================================================

print("\n[2/8] Validating required columns...")

required = (
    GROUP_COLS
    + [
        AGE_COL,
        VOLUME_COL,
        DISEASE_COL,
    ]
)

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

print("Schema: PASS")


# ============================================================
# 3. GROUP-AGE AGGREGATION
# ============================================================

print("\n[3/8] Creating group-age observations...")

group_age = (
    df.groupby(
        GROUP_COLS + [AGE_COL],
        as_index=False
    )
    .agg(
        volume_mean=(VOLUME_COL, "mean"),
        disease_mean=(DISEASE_COL, "mean"),
        disease_median=(DISEASE_COL, "median"),
        disease_std=(DISEASE_COL, "std"),
        observation_count=(VOLUME_COL, "size"),
    )
    .sort_values(
        GROUP_COLS + [AGE_COL]
    )
)

print(
    f"Group-age rows: "
    f"{len(group_age):,}"
)


# ============================================================
# 4. CREATE VOLUME TRANSITIONS
# ============================================================

print("\n[4/8] Creating P18-P66 volume transitions...")

group_age["previous_age"] = (
    group_age
    .groupby(GROUP_COLS)[AGE_COL]
    .shift(1)
)

group_age["previous_volume"] = (
    group_age
    .groupby(GROUP_COLS)["volume_mean"]
    .shift(1)
)

group_age["previous_disease"] = (
    group_age
    .groupby(GROUP_COLS)["disease_mean"]
    .shift(1)
)

group_age["volume_change"] = (
    group_age["volume_mean"]
    - group_age["previous_volume"]
)

group_age["disease_change"] = (
    group_age["disease_mean"]
    - group_age["previous_disease"]
)


# Only evaluate transitions beginning at P18.
transitions = group_age[
    (group_age[AGE_COL] >= START_AGE)
    & group_age["previous_volume"].notna()
].copy()


# ============================================================
# 5. CLASSIFY TRANSITIONS
# ============================================================

print("\n[5/8] Classifying volume transitions...")

transitions["volume_direction"] = np.select(
    [
        transitions["volume_change"] < -1e-9,
        transitions["volume_change"] > 1e-9,
    ],
    [
        "DECLINE",
        "INCREASE",
    ],
    default="STAGNATION",
)


declines = transitions[
    transitions["volume_direction"] == "DECLINE"
].copy()

increases = transitions[
    transitions["volume_direction"] == "INCREASE"
].copy()

stagnations = transitions[
    transitions["volume_direction"] == "STAGNATION"
].copy()


print(
    f"Total transitions : {len(transitions):,}"
)

print(
    f"Declines          : {len(declines):,}"
)

print(
    f"Increases         : {len(increases):,}"
)

print(
    f"Stagnations       : {len(stagnations):,}"
)


# ============================================================
# 6. DISEASE COMPARISON
# ============================================================

print("\n[6/8] Comparing disease score...")

print("\nDISEASE SCORE AT CURRENT AGE")
print("-" * 75)

def describe(label, data):

    if data.empty:
        print(f"\n{label}: no data")
        return

    print(f"\n{label}")

    print(
        f"  count          : {len(data):,}"
    )

    print(
        f"  disease mean   : "
        f"{data['disease_mean'].mean():.8f}"
    )

    print(
        f"  disease median : "
        f"{data['disease_mean'].median():.8f}"
    )

    print(
        f"  disease Q25    : "
        f"{data['disease_mean'].quantile(.25):.8f}"
    )

    print(
        f"  disease Q75    : "
        f"{data['disease_mean'].quantile(.75):.8f}"
    )

    print(
        f"  volume change  : "
        f"{data['volume_change'].mean():+.8f}"
    )


describe(
    "VOLUME DECLINES",
    declines
)

describe(
    "VOLUME INCREASES",
    increases
)

describe(
    "VOLUME STAGNATIONS",
    stagnations
)


# ============================================================
# DISEASE CHANGE
# ============================================================

print("\nDISEASE SCORE CHANGE")
print("-" * 75)

for label, data in [
    ("DECLINES", declines),
    ("INCREASES", increases),
]:

    if data.empty:
        continue

    print(f"\n{label}")

    print(
        f"  mean disease change   : "
        f"{data['disease_change'].mean():+.8f}"
    )

    print(
        f"  median disease change : "
        f"{data['disease_change'].median():+.8f}"
    )

    print(
        f"  disease increased    : "
        f"{(data['disease_change'] > 1e-9).sum():,}"
    )

    print(
        f"  disease unchanged    : "
        f"{(data['disease_change'].abs() <= 1e-9).sum():,}"
    )

    print(
        f"  disease decreased    : "
        f"{(data['disease_change'] < -1e-9).sum():,}"
    )


# ============================================================
# 7. AGE-BAND ANALYSIS
# ============================================================

print("\n[7/8] Comparing disease by age band...")

bands = [
    (18, 30, "P18-P30"),
    (36, 48, "P36-P48"),
    (54, 66, "P54-P66"),
]

for start, end, label in bands:

    subset = transitions[
        (transitions[AGE_COL] >= start)
        & (transitions[AGE_COL] <= end)
    ]

    print(
        f"\n{label}"
    )

    for direction in [
        "INCREASE",
        "DECLINE",
    ]:

        data = subset[
            subset["volume_direction"] == direction
        ]

        if data.empty:
            print(
                f"  {direction:<10}: none"
            )
            continue

        print(
            f"  {direction:<10}: "
            f"n={len(data):,}, "
            f"disease={data['disease_mean'].mean():.8f}, "
            f"disease Δ={data['disease_change'].mean():+.8f}, "
            f"volume Δ={data['volume_change'].mean():+.8f}"
        )


# ============================================================
# 8. CORRELATION AUDIT
# ============================================================

print("\n[8/8] Correlation audit...")

valid_corr = transitions[
    [
        "volume_change",
        "disease_change",
        "disease_mean",
    ]
].dropna()

if len(valid_corr) >= 3:

    corr_change = (
        valid_corr[
            "volume_change"
        ].corr(
            valid_corr[
                "disease_change"
            ],
            method="spearman"
        )
    )

    corr_level = (
        valid_corr[
            "volume_change"
        ].corr(
            valid_corr[
                "disease_mean"
            ],
            method="spearman"
        )
    )

    print(
        f"Spearman(volume change, "
        f"disease change): "
        f"{corr_change:.8f}"
    )

    print(
        f"Spearman(volume change, "
        f"disease level): "
        f"{corr_level:.8f}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("FINAL AUDIT")
print("=" * 75)

print(
    "Dataset modified: NO"
)

print(
    "Volume modified : NO"
)

print(
    "\nIMPORTANT:"
)

print(
    "Correlation is descriptive only."
)

print(
    "It does NOT establish that disease causes volume decline."
)

print(
    "\nNext decision will be based on the observed "
    "distribution of volume declines and disease patterns."
)

print("=" * 75)