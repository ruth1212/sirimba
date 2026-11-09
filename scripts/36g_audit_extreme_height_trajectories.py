from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "height_isotonic_test"
    / "height_isotonic_group_age.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "height_isotonic_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GROUP_COLS = [
    "regional_id",
    "compartment_id",
    "species",
    "clone",
    "planting_year",
]

AGE_COL = "age_months"
ACTUAL_COL = "actual_height_m"
CORRECTED_COL = "corrected_height_m"
CORRECTION_COL = "absolute_correction_m"


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("STEP 36G : AUDIT EXTREME HEIGHT TRAJECTORIES")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

required = (
    GROUP_COLS
    + [
        AGE_COL,
        ACTUAL_COL,
        CORRECTED_COL,
        CORRECTION_COL,
    ]
)

missing = [
    col
    for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

df = df.sort_values(
    GROUP_COLS + [AGE_COL]
).reset_index(drop=True)

df["group_key"] = (
    df[GROUP_COLS]
    .astype(str)
    .agg("|".join, axis=1)
)

print(f"Rows         : {len(df):,}")
print(f"Growth groups: {df['group_key'].nunique():,}")


# ============================================================
# HELPER
# ============================================================

def show_trajectory(
    group_filter,
    title
):

    mask = pd.Series(
        True,
        index=df.index
    )

    for col, value in group_filter.items():
        mask &= df[col].astype(str) == str(value)

    g = df[mask].copy()

    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)

    if g.empty:
        print("GROUP NOT FOUND")
        return None

    print(
        " | ".join(
            f"{col}={g.iloc[0][col]}"
            for col in GROUP_COLS
        )
    )

    result = g[
        [
            AGE_COL,
            ACTUAL_COL,
            CORRECTED_COL,
            CORRECTION_COL,
        ]
    ].copy()

    result["actual_change_m"] = (
        result[ACTUAL_COL].diff()
    )

    result["corrected_change_m"] = (
        result[CORRECTED_COL].diff()
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    return result


# ============================================================
# 1. KNOWN EXTREME GROUP
# ============================================================

known_extreme = show_trajectory(
    {
        "regional_id": "R02",
        "compartment_id": 202,
        "species": "Eucalyptus grandis",
        "clone": "EG-02",
        "planting_year": 2000,
    },
    "1. KNOWN EXTREME GROUP"
)


# ============================================================
# 2. FIND GROUP WITH LONGEST PLATEAU
# ============================================================

df["corrected_change_m"] = (
    df.groupby("group_key")[CORRECTED_COL]
    .diff()
)

df["stagnant"] = (
    df["corrected_change_m"].abs()
    <= 1e-10
)

plateau_records = []

for group_key, g in df.groupby("group_key"):

    g = g.sort_values(AGE_COL).reset_index(drop=True)

    current = 0
    start_age = None

    for i in range(1, len(g)):

        stagnant = (
            abs(
                g.loc[i, CORRECTED_COL]
                - g.loc[i - 1, CORRECTED_COL]
            )
            <= 1e-10
        )

        if stagnant:

            if current == 0:
                start_age = g.loc[i - 1, AGE_COL]

            current += 1

        else:

            if current > 0:

                plateau_records.append(
                    {
                        "group_key": group_key,
                        "start_age": start_age,
                        "end_age": g.loc[i - 1, AGE_COL],
                        "stagnant_transitions": current,
                        "duration_months": current * 6,
                    }
                )

            current = 0
            start_age = None

    if current > 0:

        plateau_records.append(
            {
                "group_key": group_key,
                "start_age": start_age,
                "end_age": g.loc[len(g) - 1, AGE_COL],
                "stagnant_transitions": current,
                "duration_months": current * 6,
            }
        )

plateaus = pd.DataFrame(
    plateau_records
)

longest = (
    plateaus
    .sort_values(
        [
            "stagnant_transitions",
            "duration_months",
        ],
        ascending=False
    )
    .iloc[0]
)

print("\n" + "-" * 70)
print("2. LONGEST PLATEAU")
print("-" * 70)

print(
    longest.to_string()
)


# Extract group key
longest_group = longest["group_key"]

longest_group_df = df[
    df["group_key"] == longest_group
].copy()

show_trajectory(
    {
        "regional_id": longest_group_df.iloc[0]["regional_id"],
        "compartment_id": longest_group_df.iloc[0]["compartment_id"],
        "species": longest_group_df.iloc[0]["species"],
        "clone": longest_group_df.iloc[0]["clone"],
        "planting_year": longest_group_df.iloc[0]["planting_year"],
    },
    "LONGEST PLATEAU TRAJECTORY"
)


# ============================================================
# 3. GLOBAL MAXIMUM CORRECTION
# ============================================================

max_row = df.loc[
    df[CORRECTION_COL].idxmax()
]

print("\n" + "-" * 70)
print("3. GLOBAL MAXIMUM CORRECTION")
print("-" * 70)

print(
    max_row[
        GROUP_COLS
        + [
            AGE_COL,
            ACTUAL_COL,
            CORRECTED_COL,
            CORRECTION_COL,
        ]
    ].to_string()
)

show_trajectory(
    {
        "regional_id": max_row["regional_id"],
        "compartment_id": max_row["compartment_id"],
        "species": max_row["species"],
        "clone": max_row["clone"],
        "planting_year": max_row["planting_year"],
    },
    "MAXIMUM CORRECTION TRAJECTORY"
)


# ============================================================
# 4. NORMAL / LOW-CORRECTION GROUP
# ============================================================

group_summary = (
    df.groupby("group_key")
    .agg(
        mean_correction=(
            CORRECTION_COL,
            "mean"
        ),
        max_correction=(
            CORRECTION_COL,
            "max"
        ),
    )
    .reset_index()
)

normal_group_key = (
    group_summary
    .sort_values(
        [
            "mean_correction",
            "max_correction",
        ]
    )
    .iloc[0]["group_key"]
)

normal_group_df = df[
    df["group_key"] == normal_group_key
].copy()

show_trajectory(
    {
        "regional_id": normal_group_df.iloc[0]["regional_id"],
        "compartment_id": normal_group_df.iloc[0]["compartment_id"],
        "species": normal_group_df.iloc[0]["species"],
        "clone": normal_group_df.iloc[0]["clone"],
        "planting_year": normal_group_df.iloc[0]["planting_year"],
    },
    "4. LOW-CORRECTION / NORMAL GROUP"
)


# ============================================================
# 5. SAVE SELECTED TRAJECTORIES
# ============================================================

selected_keys = {
    "known_extreme": (
        known_extreme
    ),
    "longest_plateau": (
        longest_group_df
    ),
    "maximum_correction": (
        df[
            df["group_key"]
            == max_row["group_key"]
        ]
    ),
    "normal": (
        normal_group_df
    ),
}

for name, data in selected_keys.items():

    if data is not None and len(data) > 0:

        data.to_csv(
            OUTPUT_DIR
            / f"trajectory_{name}.csv",
            index=False
        )


print("\n" + "=" * 70)
print("STEP 36G FINISHED")
print("=" * 70)
print("Original dataset modified: NO")
print(
    f"Output directory:\n{OUTPUT_DIR}"
)