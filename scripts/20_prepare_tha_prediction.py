from pathlib import Path

import pandas as pd


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"
TRAIN_PATH = PROJECT_ROOT / "results" / "model_train.csv"
TEST_PATH = PROJECT_ROOT / "results" / "model_test.csv"

OUTPUT_TRAIN = PROJECT_ROOT / "results" / "tha_model_train.csv"
OUTPUT_TEST = PROJECT_ROOT / "results" / "tha_model_test.csv"
OUTPUT_INFO = PROJECT_ROOT / "results" / "tha_prediction_config.txt"


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES = [
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha",
]

TARGET = "tha_m"

GROUP_COLUMN = "compartment_id"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 13A — PREPARE THA PREDICTION DATA")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH,
    usecols=FEATURES + [TARGET, GROUP_COLUMN]
)

print(f"Dataset rows : {len(df):,}")
print(f"Dataset cols : {len(df.columns)}")


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = FEATURES + [TARGET, GROUP_COLUMN]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nRequired columns: OK")


# ============================================================
# MISSING VALUES
# ============================================================

print("\nMissing values:")

missing = df[required_columns].isna().sum()

print(missing.to_string())

if missing.sum() > 0:
    raise ValueError(
        "Missing values detected. "
        "Do not continue until the dataset is cleaned."
    )


# ============================================================
# DUPLICATES
# ============================================================

duplicate_count = df.duplicated().sum()

print(f"\nDuplicate rows: {duplicate_count:,}")

if duplicate_count > 0:
    print("Removing duplicate rows...")
    df = df.drop_duplicates().reset_index(drop=True)

print(f"Rows after duplicate removal: {len(df):,}")


# ============================================================
# TARGET INSPECTION
# ============================================================

print("\n" + "=" * 70)
print("THA TARGET INSPECTION")
print("=" * 70)

print(f"\nTarget: {TARGET}")

print("\nTHA statistics:")
print(df[TARGET].describe().to_string())

print("\nUnique THA values:")
print(df[TARGET].nunique())


# ============================================================
# GROUP INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("GROUP INFORMATION")
print("=" * 70)

unique_groups = df[GROUP_COLUMN].nunique()

print(f"\nUnique compartments: {unique_groups}")

group_sizes = (
    df.groupby(GROUP_COLUMN)
    .size()
    .describe()
)

print("\nRows per compartment:")
print(group_sizes.to_string())


# ============================================================
# CHECK THA WITHIN COMPARTMENT + AGE
# ============================================================

print("\n" + "=" * 70)
print("THA CONSISTENCY CHECK")
print("=" * 70)

grouped = (
    df.groupby(
        [GROUP_COLUMN, "age_months"],
        as_index=False
    )
    .agg(
        tha_min=(TARGET, "min"),
        tha_max=(TARGET, "max"),
        tha_mean=(TARGET, "mean"),
        rows=(TARGET, "size"),
    )
)

print(
    f"\nCompartment-age groups: "
    f"{len(grouped):,}"
)

max_internal_difference = (
    grouped["tha_max"] - grouped["tha_min"]
).max()

print(
    f"Maximum THA difference "
    f"inside one compartment-age group: "
    f"{max_internal_difference:.10f}"
)

if max_internal_difference == 0:
    print(
        "\nRESULT: THA is constant within each "
        "compartment-age group."
    )
else:
    print(
        "\nRESULT: THA varies within some "
        "compartment-age groups."
    )


# ============================================================
# BUILD TRAIN / TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("BUILDING THA TRAIN / TEST DATA")
print("=" * 70)

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(f"\nExisting model train rows: {len(train):,}")
print(f"Existing model test rows : {len(test):,}")


# ------------------------------------------------------------
# Make sure THA is attached using the original row identity
# ------------------------------------------------------------

merge_columns = [
    GROUP_COLUMN,
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha",
]

tha_lookup = df[
    merge_columns + [TARGET]
].copy()


# Check whether the merge key is unique enough
key_duplicates = tha_lookup.duplicated(
    subset=merge_columns
).sum()

print(
    f"\nDuplicate feature-key rows: "
    f"{key_duplicates:,}"
)

if key_duplicates > 0:
    print(
        "\nWARNING:"
        "\nThe selected feature key is not unique."
        "\nTHA may have multiple records for the same "
        "feature combination."
    )


# ------------------------------------------------------------
# Merge THA into existing train/test
# ------------------------------------------------------------

train_tha = train.merge(
    tha_lookup,
    on=merge_columns,
    how="left",
    validate="many_to_one",
)

test_tha = test.merge(
    tha_lookup,
    on=merge_columns,
    how="left",
    validate="many_to_one",
)


# ============================================================
# MERGE VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("MERGE VALIDATION")
print("=" * 70)

print(f"\nTHA train rows: {len(train_tha):,}")
print(f"THA test rows : {len(test_tha):,}")

train_missing = train_tha[TARGET].isna().sum()
test_missing = test_tha[TARGET].isna().sum()

print(f"\nMissing THA in train: {train_missing:,}")
print(f"Missing THA in test : {test_missing:,}")

if train_missing > 0 or test_missing > 0:
    raise ValueError(
        "THA could not be matched to every train/test row."
    )


# ============================================================
# GROUP OVERLAP CHECK
# ============================================================

train_groups = set(
    train_tha[GROUP_COLUMN].unique()
)

test_groups = set(
    test_tha[GROUP_COLUMN].unique()
)

overlap = train_groups.intersection(test_groups)

print(
    f"\nTrain compartments: {len(train_groups)}"
)

print(
    f"Test compartments : {len(test_groups)}"
)

print(
    f"Overlapping compartments: {len(overlap)}"
)

if overlap:
    raise ValueError(
        f"Group leakage detected: {sorted(overlap)}"
    )


# ============================================================
# SAVE
# ============================================================

train_tha.to_csv(
    OUTPUT_TRAIN,
    index=False
)

test_tha.to_csv(
    OUTPUT_TEST,
    index=False
)


# ============================================================
# SAVE CONFIG
# ============================================================

config_text = f"""
THA PREDICTION DATA CONFIGURATION
=================================

Target:
{TARGET}

Definition:
THA (Top Height) = tinggi dominan tegakan,
yaitu ukuran tinggi kelompok pohon-pohon
dominan/tertinggi.

Numerical THA method:
NOT DETERMINED FROM DATASET.

Features:
{chr(10).join("- " + x for x in FEATURES)}

Group column:
{GROUP_COLUMN}

Train rows:
{len(train_tha):,}

Test rows:
{len(test_tha):,}

Train groups:
{len(train_groups)}

Test groups:
{len(test_groups)}

Group overlap:
{len(overlap)}

THA missing train:
{train_missing}

THA missing test:
{test_missing}
""".strip()

OUTPUT_INFO.write_text(
    config_text,
    encoding="utf-8"
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("STEP 13A COMPLETED")
print("=" * 70)

print("\nSaved:")
print(f"- {OUTPUT_TRAIN}")
print(f"- {OUTPUT_TEST}")
print(f"- {OUTPUT_INFO}")

print("\nTHA is prepared as a prediction target.")
print("THA is NOT used as a model input.")

print("\nNEXT STEP: THA Random Forest baseline.")
print("=" * 70)