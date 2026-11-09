from pathlib import Path
import pandas as pd


# ============================================================
# TREE GROWTH FEATURE SELECTION
# STEP 1 - INSPECT DATASET
# ============================================================

print("=" * 70)
print("TREE GROWTH FEATURE SELECTION")
print("STEP 1 - DATASET INSPECTION")
print("=" * 70)


# ------------------------------------------------------------
# 1. DATASET PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"

print("\n[1] Dataset path:")
print(DATA_PATH)


# ------------------------------------------------------------
# 2. CHECK FILE
# ------------------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset tidak ditemukan:\n{DATA_PATH}"
    )

print("\n[OK] dataset.csv ditemukan.")


# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------

print("\n[2] Membaca dataset...")

df = pd.read_csv(DATA_PATH)

print("[OK] Dataset berhasil dibaca.")


# ------------------------------------------------------------
# 4. DATASET SIZE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET SIZE")
print("=" * 70)

print(f"Jumlah baris   : {df.shape[0]:,}")
print(f"Jumlah kolom   : {df.shape[1]:,}")


# ------------------------------------------------------------
# 5. COLUMN LIST
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DAFTAR KOLOM")
print("=" * 70)

for number, column in enumerate(df.columns, start=1):
    print(f"{number:02d}. {column}")


# ------------------------------------------------------------
# 6. DATA TYPES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TIPE DATA")
print("=" * 70)

print(df.dtypes.to_string())


# ------------------------------------------------------------
# 7. MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isna().sum()

for column, value in missing.items():
    if value > 0:
        print(
            f"{column:<30} : "
            f"{value:,} "
            f"({value / len(df) * 100:.2f}%)"
        )

if missing.sum() == 0:
    print("[OK] Tidak ada missing value.")


# ------------------------------------------------------------
# 8. DUPLICATES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE ROWS")
print("=" * 70)

duplicates = df.duplicated().sum()

print(f"Duplicate rows : {duplicates:,}")


# ------------------------------------------------------------
# 9. NUMERIC COLUMNS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NUMERIC COLUMNS")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()

for column in numeric_columns:
    print(f"- {column}")


# ------------------------------------------------------------
# 10. CATEGORICAL COLUMNS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CATEGORICAL COLUMNS")
print("=" * 70)

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

for column in categorical_columns:
    print(f"- {column}")


# ------------------------------------------------------------
# 11. CHECK IMPORTANT FIELDS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("IMPORTANT FIELDS")
print("=" * 70)

important_fields = [
    "disease_score",
    "disease_status",
    "tha_m",
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]

for field in important_fields:

    if field in df.columns:

        print(
            f"[OK] {field:<20} "
            f"dtype={df[field].dtype}"
        )

    else:

        print(
            f"[MISSING] {field}"
        )


# ------------------------------------------------------------
# 12. TARGET CANDIDATES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TARGET CANDIDATES")
print("=" * 70)

targets = [
    "dbh_cm",
    "height_m",
    "volume_m3_tree",
]

for target in targets:

    if target in df.columns:

        print(f"\n--- {target} ---")

        print(
            df[target]
            .describe()
            .to_string()
        )


# ------------------------------------------------------------
# 13. FINISH
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 1 SELESAI")
print("=" * 70)