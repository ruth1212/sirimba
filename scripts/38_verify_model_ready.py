from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# ============================================================
# STEP 38 - FINAL MODEL-READY VERIFICATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "model_ready" / "train.csv"
TEST_PATH = ROOT / "data" / "model_ready" / "test.csv"
METADATA_PATH = ROOT / "data" / "model_ready" / "split_metadata.json"


FEATURES = [
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha",
]

TARGETS = [
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]

GROUP_COL = "compartment_id"

GROUP_KEY_COLS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]

EXPECTED_TOTAL_ROWS = 218_700
EXPECTED_TRAIN_ROWS = 174_960
EXPECTED_TEST_ROWS = 43_740

EXPECTED_TRAIN_COMPARTMENTS = 12
EXPECTED_TEST_COMPARTMENTS = 3

EXPECTED_AGES = [
    6,
    12,
    18,
    24,
    30,
    36,
    42,
    48,
    54,
    60,
    66,
]


def fail(message):
    print()
    print("=" * 80)
    print("STEP 38 STATUS: FAIL")
    print("=" * 80)
    print(message)
    sys.exit(1)


def pass_check(message):
    print(message)


print("=" * 80)
print("STEP 38 - FINAL MODEL-READY VERIFICATION")
print("=" * 80)


# ============================================================
# [1/10] FILE EXISTENCE
# ============================================================

print("\n[1/10] Checking model-ready files...")

for path in [TRAIN_PATH, TEST_PATH, METADATA_PATH]:
    if not path.exists():
        fail(f"Required file not found:\n{path}")

pass_check("Train file: PASS")
pass_check("Test file: PASS")
pass_check("Metadata file: PASS")


# ============================================================
# [2/10] LOAD DATA
# ============================================================

print("\n[2/10] Loading train/test data...")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(f"Train rows : {len(train):,}")
print(f"Test rows  : {len(test):,}")

if len(train) != EXPECTED_TRAIN_ROWS:
    fail(
        f"Unexpected train row count.\n"
        f"Expected: {EXPECTED_TRAIN_ROWS:,}\n"
        f"Actual  : {len(train):,}"
    )

if len(test) != EXPECTED_TEST_ROWS:
    fail(
        f"Unexpected test row count.\n"
        f"Expected: {EXPECTED_TEST_ROWS:,}\n"
        f"Actual  : {len(test):,}"
    )

pass_check("Train/test row counts: PASS")


# ============================================================
# [3/10] FEATURE/TARGET COLUMNS
# ============================================================

print("\n[3/10] Checking official model columns...")

required_columns = FEATURES + TARGETS + GROUP_KEY_COLS

for df_name, df in [("train", train), ("test", test)]:
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        fail(
            f"{df_name} is missing required columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )

pass_check("Official features: PASS")
pass_check("Official targets: PASS")
pass_check("Group columns: PASS")


# ============================================================
# [4/10] NO MISSING / INFINITE MODEL VALUES
# ============================================================

print("\n[4/10] Checking model values...")

model_columns = FEATURES + TARGETS

for df_name, df in [("train", train), ("test", test)]:

    missing = df[model_columns].isna().sum()

    if missing.sum() > 0:
        print(f"\nMissing values in {df_name}:")
        print(missing[missing > 0])
        fail(f"{df_name} contains missing model values.")

    numeric_values = df[model_columns].select_dtypes(
        include=[np.number]
    )

    if not np.isfinite(numeric_values.to_numpy()).all():
        fail(f"{df_name} contains infinite model values.")

pass_check("No missing model values: PASS")
pass_check("No infinite model values: PASS")


# ============================================================
# [5/10] AGE STRUCTURE
# ============================================================

print("\n[5/10] Checking age structure...")

for df_name, df in [("train", train), ("test", test)]:

    ages = sorted(df["age_months"].unique().tolist())

    print(f"{df_name} ages: {ages}")

    unexpected = sorted(set(ages) - set(EXPECTED_AGES))

    if unexpected:
        fail(
            f"{df_name} contains unexpected ages: {unexpected}"
        )

    if any(age not in EXPECTED_AGES for age in ages):
        fail(f"{df_name} contains invalid age values.")

pass_check("Age range P6-P66: PASS")
pass_check("No age above PHI/P66: PASS")


# ============================================================
# [6/10] AGE DISTRIBUTION
# ============================================================

print("\n[6/10] Checking age distribution...")

print("\nTrain age counts:")
print(
    train["age_months"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nTest age counts:")
print(
    test["age_months"]
    .value_counts()
    .sort_index()
    .to_string()
)

if train["age_months"].min() != 6:
    fail("Train does not start at P6.")

if test["age_months"].min() != 6:
    fail("Test does not start at P6.")

if train["age_months"].max() != 66:
    fail("Train does not end at P66/PHI.")

if test["age_months"].max() != 66:
    fail("Test does not end at P66/PHI.")

pass_check("P6-P66 age boundaries: PASS")


# ============================================================
# [7/10] COMPARTMENT GROUP SPLIT
# ============================================================

print("\n[7/10] Checking compartment separation...")

train_groups = set(train[GROUP_COL].unique())
test_groups = set(test[GROUP_COL].unique())

overlap = train_groups.intersection(test_groups)

print(f"Train compartments: {len(train_groups)}")
print(f"Test compartments : {len(test_groups)}")
print(f"Overlap            : {len(overlap)}")

if len(train_groups) != EXPECTED_TRAIN_COMPARTMENTS:
    fail(
        f"Unexpected train compartment count.\n"
        f"Expected: {EXPECTED_TRAIN_COMPARTMENTS}\n"
        f"Actual  : {len(train_groups)}"
    )

if len(test_groups) != EXPECTED_TEST_COMPARTMENTS:
    fail(
        f"Unexpected test compartment count.\n"
        f"Expected: {EXPECTED_TEST_COMPARTMENTS}\n"
        f"Actual  : {len(test_groups)}"
    )

if overlap:
    fail(
        f"COMPARTMENT LEAKAGE DETECTED:\n"
        f"{sorted(overlap)}"
    )

pass_check("Compartment separation: PASS")


# ============================================================
# [8/10] GROWTH-GROUP LEAKAGE
# ============================================================

print("\n[8/10] Checking growth-group separation...")

train_growth_groups = set(
    map(
        tuple,
        train[GROUP_KEY_COLS].drop_duplicates().to_numpy()
    )
)

test_growth_groups = set(
    map(
        tuple,
        test[GROUP_KEY_COLS].drop_duplicates().to_numpy()
    )
)

growth_overlap = train_growth_groups.intersection(
    test_growth_groups
)

print(f"Train growth groups: {len(train_growth_groups)}")
print(f"Test growth groups : {len(test_growth_groups)}")
print(f"Growth overlap     : {len(growth_overlap)}")

if growth_overlap:
    fail(
        "GROWTH-GROUP LEAKAGE DETECTED."
    )

pass_check("Growth-group separation: PASS")


# ============================================================
# [9/10] TARGET SANITY
# ============================================================

print("\n[9/10] Checking target sanity...")

for df_name, df in [("train", train), ("test", test)]:

    dbh_min = df["dbh_cm"].min()
    height_min = df["height_m"].min()
    volume_min = df["volume_m3_tree"].min()

    print(
        f"{df_name}: "
        f"DBH min={dbh_min:.6f}, "
        f"Height min={height_min:.6f}, "
        f"Volume min={volume_min:.6f}"
    )

    if dbh_min <= 0:
        fail(f"{df_name} contains non-positive DBH.")

    if height_min <= 0:
        fail(f"{df_name} contains non-positive Height.")

    if volume_min < 0:
        fail(f"{df_name} contains negative Volume.")

pass_check("DBH positive: PASS")
pass_check("Height positive: PASS")
pass_check("Volume non-negative: PASS")


# ============================================================
# [10/10] METADATA CONSISTENCY
# ============================================================

print("\n[10/10] Checking split metadata...")

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Metadata loaded successfully.")

metadata_text = json.dumps(metadata, indent=2)

required_metadata_terms = [
    "train",
    "test",
]

for term in required_metadata_terms:
    if term not in metadata_text.lower():
        fail(
            f"Metadata does not contain expected section: {term}"
        )

pass_check("Metadata readable: PASS")


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("STEP 38 STATUS: PASS")
print("=" * 80)

print()
print("MODEL-READY DATA IS VERIFIED.")
print()
print("Official training source:")
print("  data/model_ready/train.csv")
print()
print("Official testing source:")
print("  data/model_ready/test.csv")
print()
print("Official split metadata:")
print("  data/model_ready/split_metadata.json")
print()
print("Features:")
for feature in FEATURES:
    print(f"  - {feature}")

print()
print("Targets:")
for target in TARGETS:
    print(f"  - {target}")

print()
print("Prediction rule:")
print("  - DBH       : P6-P66")
print("  - Height    : P6-P66")
print("  - Volume    : P18-P66")
print("  - PHI       : P66 = 66 months")
print("  - MAI       : calculated from predicted volume")
print()
print("Random Forest and PINNs may proceed only after this PASS.")
print("=" * 80)