import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# TREE GROWTH ANALYTICS
# STEP 21P - DIAGNOSTIC COMPARISON 3 PINNs
# ============================================================

print("=" * 80)
print("TREE GROWTH ANALYTICS")
print("STEP 21P - DIAGNOSTIC COMPARISON 3 PINNs")
print("=" * 80)


# ============================================================
# FILES
# ============================================================

BASELINE_FILE = (
    "results/pinns_predictions.csv"
)

POSITIVE_FILE = (
    "results/pinns_positive_test_predictions.csv"
)

HARD_POSITIVE_FILE = (
    "results/pinns_hard_positive_predictions.csv"
)


OUTPUT_GLOBAL = (
    "results/"
    "pinns_three_model_comparison_global.csv"
)

OUTPUT_GROUP = (
    "results/"
    "pinns_three_model_comparison_by_group.csv"
)

OUTPUT_AGE = (
    "results/"
    "pinns_three_model_comparison_by_age.csv"
)

OUTPUT_SUMMARY = (
    "results/"
    "pinns_three_model_comparison_summary.json"
)


TARGETS = [
    "dbh_cm",
    "height_m",
    "volume_m3_tree"
]


# ============================================================
# LOAD
# ============================================================

print("\n[1/8] Loading prediction files...")

baseline = pd.read_csv(
    BASELINE_FILE
)

positive = pd.read_csv(
    POSITIVE_FILE
)

hard_positive = pd.read_csv(
    HARD_POSITIVE_FILE
)

print(
    "Baseline rows      :",
    len(baseline)
)

print(
    "Positive rows      :",
    len(positive)
)

print(
    "Hard-positive rows :",
    len(hard_positive)
)


# ============================================================
# STANDARDIZE ACTUAL TARGET NAMES
# ============================================================

print(
    "\n[2/8] Standardizing actual target columns..."
)


# ------------------------------------------------------------
# Baseline
# ------------------------------------------------------------

baseline_std = pd.DataFrame()

baseline_std["compartment_id"] = (
    baseline["compartment_id"]
)

baseline_std["age_months"] = (
    baseline["age_months"]
)

baseline_std["inventory_year"] = (
    baseline["inventory_year"]
)

baseline_std["clone_factor"] = (
    baseline["clone_factor"]
)

baseline_std["planting_year"] = (
    baseline["planting_year"]
)

baseline_std["disease_score"] = (
    baseline["disease_score"]
)

baseline_std["trees_per_ha"] = (
    baseline["trees_per_ha"]
)

baseline_std["dbh_cm"] = (
    baseline["actual_dbh_cm"]
)

baseline_std["height_m"] = (
    baseline["actual_height_m"]
)

baseline_std["volume_m3_tree"] = (
    baseline["actual_volume_m3_tree"]
)

baseline_std[
    "predicted_pinns_dbh_cm"
] = (
    baseline["predicted_pinns_dbh_cm"]
)

baseline_std[
    "predicted_pinns_height_m"
] = (
    baseline["predicted_pinns_height_m"]
)

baseline_std[
    "predicted_pinns_volume_m3_tree"
] = (
    baseline["predicted_pinns_volume_m3_tree"]
)


# ------------------------------------------------------------
# Positive
# ------------------------------------------------------------

positive_std = pd.DataFrame()

for column in [
    "compartment_id",
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha",
    "dbh_cm",
    "height_m",
    "volume_m3_tree"
]:

    positive_std[column] = (
        positive[column]
    )

positive_std[
    "pinns_positive_dbh_cm"
] = (
    positive["pinns_positive_dbh_cm"]
)

positive_std[
    "pinns_positive_height_m"
] = (
    positive["pinns_positive_height_m"]
)

positive_std[
    "pinns_positive_volume_m3_tree"
] = (
    positive["pinns_positive_volume_m3_tree"]
)


# ------------------------------------------------------------
# Hard-positive
# ------------------------------------------------------------

hard_std = pd.DataFrame()

for column in [
    "compartment_id",
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha",
    "dbh_cm",
    "height_m",
    "volume_m3_tree"
]:

    hard_std[column] = (
        hard_positive[column]
    )

hard_std[
    "predicted_pinns_hard_positive_dbh_cm"
] = (
    hard_positive[
        "predicted_pinns_hard_positive_dbh_cm"
    ]
)

hard_std[
    "predicted_pinns_hard_positive_height_m"
] = (
    hard_positive[
        "predicted_pinns_hard_positive_height_m"
    ]
)

hard_std[
    "predicted_pinns_hard_positive_volume_m3_tree"
] = (
    hard_positive[
        "predicted_pinns_hard_positive_volume_m3_tree"
    ]
)


# ============================================================
# VALIDATION
# ============================================================

print(
    "\n[3/8] Validating row alignment..."
)

if not (
    len(baseline_std)
    == len(positive_std)
    == len(hard_std)
):

    raise ValueError(
        "Prediction files have different row counts."
    )


IDENTITY_COLUMNS = [
    "compartment_id",
    "age_months",
    "inventory_year",
    "clone_factor",
    "planting_year",
    "disease_score",
    "trees_per_ha"
]


for column in IDENTITY_COLUMNS:

    if not baseline_std[column].equals(
        positive_std[column]
    ):

        raise ValueError(
            f"Row alignment mismatch: {column} "
            "baseline vs positive."
        )

    if not baseline_std[column].equals(
        hard_std[column]
    ):

        raise ValueError(
            f"Row alignment mismatch: {column} "
            "baseline vs hard-positive."
        )


for target in TARGETS:

    if not np.allclose(
        baseline_std[target].to_numpy(),
        positive_std[target].to_numpy(),
        rtol=0,
        atol=1e-10
    ):

        raise ValueError(
            f"Actual target mismatch: {target} "
            "baseline vs positive."
        )

    if not np.allclose(
        baseline_std[target].to_numpy(),
        hard_std[target].to_numpy(),
        rtol=0,
        atol=1e-10
    ):

        raise ValueError(
            f"Actual target mismatch: {target} "
            "baseline vs hard-positive."
        )


print(
    "Rows aligned : YES"
)

print(
    "Actual targets aligned : YES"
)


# ============================================================
# MODEL DEFINITION
# ============================================================

models = {

    "Baseline PINNs": {
        "frame": baseline_std,
        "predictions": {
            "dbh_cm":
                "predicted_pinns_dbh_cm",

            "height_m":
                "predicted_pinns_height_m",

            "volume_m3_tree":
                "predicted_pinns_volume_m3_tree"
        }
    },

    "Positive-Penalty PINNs": {
        "frame": positive_std,
        "predictions": {
            "dbh_cm":
                "pinns_positive_dbh_cm",

            "height_m":
                "pinns_positive_height_m",

            "volume_m3_tree":
                "pinns_positive_volume_m3_tree"
        }
    },

    "Hard-Positive PINNs": {
        "frame": hard_std,
        "predictions": {
            "dbh_cm":
                "predicted_pinns_hard_positive_dbh_cm",

            "height_m":
                "predicted_pinns_hard_positive_height_m",

            "volume_m3_tree":
                "predicted_pinns_hard_positive_volume_m3_tree"
        }
    }
}


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    nonzero = (
        np.abs(y_true) > 1e-8
    )

    if nonzero.any():

        mape = (
            np.mean(
                np.abs(
                    (
                        y_true[nonzero]
                        - y_pred[nonzero]
                    )
                    / y_true[nonzero]
                )
            )
            * 100
        )

    else:

        mape = np.nan

    return (
        mae,
        rmse,
        r2,
        mape
    )


# ============================================================
# GLOBAL COMPARISON
# ============================================================

print(
    "\n[4/8] Calculating global metrics..."
)

global_rows = []


for model_name, model_info in models.items():

    frame = model_info["frame"]

    for target in TARGETS:

        prediction_column = (
            model_info[
                "predictions"
            ][target]
        )

        y_true = (
            frame[target]
            .to_numpy(
                dtype=float
            )
        )

        y_pred = (
            frame[prediction_column]
            .to_numpy(
                dtype=float
            )
        )

        mae, rmse, r2, mape = (
            calculate_metrics(
                y_true,
                y_pred
            )
        )

        negative_count = np.nan

        if target == "volume_m3_tree":

            negative_count = int(
                (
                    y_pred < 0
                ).sum()
            )

        global_rows.append({

            "model":
                model_name,

            "target":
                target,

            "n_test":
                len(frame),

            "mae":
                mae,

            "rmse":
                rmse,

            "r2":
                r2,

            "mape_percent":
                mape,

            "negative_volume_count":
                negative_count,

            "minimum_prediction":
                float(
                    y_pred.min()
                ),

            "maximum_prediction":
                float(
                    y_pred.max()
                )
        })


global_df = pd.DataFrame(
    global_rows
)


# ============================================================
# AGE GROUPS
# ============================================================

print(
    "\n[5/8] Calculating age-group diagnostics..."
)


def age_group(age):

    if age == 6:
        return "P6"

    if age == 12:
        return "P12"

    return "AGE_GT_12"


group_rows = []


for model_name, model_info in models.items():

    frame = model_info["frame"]

    for group_name, mask in [

        (
            "P6",
            frame["age_months"] == 6
        ),

        (
            "P12",
            frame["age_months"] == 12
        ),

        (
            "AGE_GT_12",
            frame["age_months"] > 12
        )
    ]:

        for target in TARGETS:

            prediction_column = (
                model_info[
                    "predictions"
                ][target]
            )

            y_true = (
                frame.loc[
                    mask,
                    target
                ]
                .to_numpy(
                    dtype=float
                )
            )

            y_pred = (
                frame.loc[
                    mask,
                    prediction_column
                ]
                .to_numpy(
                    dtype=float
                )
            )

            mae, rmse, r2, mape = (
                calculate_metrics(
                    y_true,
                    y_pred
                )
            )

            negative_count = np.nan

            if target == "volume_m3_tree":

                negative_count = int(
                    (
                        y_pred < 0
                    ).sum()
                )

            group_rows.append({

                "model":
                    model_name,

                "period_group":
                    group_name,

                "target":
                    target,

                "n":
                    len(y_true),

                "mae":
                    mae,

                "rmse":
                    rmse,

                "r2":
                    r2,

                "mape_percent":
                    mape,

                "negative_volume_count":
                    negative_count,

                "mean_actual":
                    float(
                        y_true.mean()
                    ),

                "mean_prediction":
                    float(
                        y_pred.mean()
                    ),

                "minimum_prediction":
                    float(
                        y_pred.min()
                    ),

                "maximum_prediction":
                    float(
                        y_pred.max()
                    )
            })


group_df = pd.DataFrame(
    group_rows
)


# ============================================================
# AGE-BY-AGE VOLUME
# ============================================================

print(
    "\n[6/8] Calculating age-by-age volume diagnostics..."
)

age_rows = []

unique_ages = sorted(
    baseline_std[
        "age_months"
    ].unique()
)


for model_name, model_info in models.items():

    frame = model_info["frame"]

    prediction_column = (
        model_info[
            "predictions"
        ]["volume_m3_tree"]
    )

    for age in unique_ages:

        mask = (
            frame["age_months"]
            == age
        )

        y_true = (
            frame.loc[
                mask,
                "volume_m3_tree"
            ]
            .to_numpy(
                dtype=float
            )
        )

        y_pred = (
            frame.loc[
                mask,
                prediction_column
            ]
            .to_numpy(
                dtype=float
            )
        )

        mae, rmse, r2, mape = (
            calculate_metrics(
                y_true,
                y_pred
            )
        )

        age_rows.append({

            "model":
                model_name,

            "age_months":
                int(age),

            "n":
                len(y_true),

            "mae":
                mae,

            "rmse":
                rmse,

            "r2":
                r2,

            "mape_percent":
                mape,

            "negative_volume_count":
                int(
                    (
                        y_pred < 0
                    ).sum()
                ),

            "mean_actual":
                float(
                    y_true.mean()
                ),

            "mean_prediction":
                float(
                    y_pred.mean()
                ),

            "minimum_prediction":
                float(
                    y_pred.min()
                ),

            "maximum_prediction":
                float(
                    y_pred.max()
                )
        })


age_df = pd.DataFrame(
    age_rows
)


# ============================================================
# SAVE
# ============================================================

print(
    "\n[7/8] Saving diagnostic results..."
)

global_df.to_csv(
    OUTPUT_GLOBAL,
    index=False
)

group_df.to_csv(
    OUTPUT_GROUP,
    index=False
)

age_df.to_csv(
    OUTPUT_AGE,
    index=False
)


summary = {

    "test_rows":
        int(len(baseline_std)),

    "models":
        list(models.keys()),

    "global_volume":
        global_df[
            global_df["target"]
            == "volume_m3_tree"
        ]
        .replace(
            {np.nan: None}
        )
        .to_dict(
            orient="records"
        ),

    "period_group_volume":
        group_df[
            group_df["target"]
            == "volume_m3_tree"
        ]
        .replace(
            {np.nan: None}
        )
        .to_dict(
            orient="records"
        )
}


with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# DISPLAY GLOBAL
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "GLOBAL COMPARISON"
)

print(
    "=" * 80
)

print(
    global_df[
        [
            "model",
            "target",
            "mae",
            "rmse",
            "r2",
            "mape_percent",
            "negative_volume_count"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# DISPLAY VOLUME GROUP
# ============================================================

volume_group = group_df[
    group_df["target"]
    == "volume_m3_tree"
]

print(
    "\n" + "=" * 80
)

print(
    "VOLUME BY PERIOD GROUP"
)

print(
    "=" * 80
)

print(
    volume_group[
        [
            "model",
            "period_group",
            "n",
            "mae",
            "rmse",
            "r2",
            "mape_percent",
            "negative_volume_count",
            "mean_actual",
            "mean_prediction"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# P6
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "P6 VOLUME DIAGNOSTIC"
)

print(
    "=" * 80
)

print(
    volume_group[
        volume_group["period_group"] == "P6"
    ][
        [
            "model",
            "n",
            "mae",
            "rmse",
            "r2",
            "mape_percent",
            "negative_volume_count",
            "mean_actual",
            "mean_prediction",
            "minimum_prediction",
            "maximum_prediction"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# P12
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "P12 VOLUME DIAGNOSTIC"
)

print(
    "=" * 80
)

print(
    volume_group[
        volume_group["period_group"] == "P12"
    ][
        [
            "model",
            "n",
            "mae",
            "rmse",
            "r2",
            "mape_percent",
            "negative_volume_count",
            "mean_actual",
            "mean_prediction",
            "minimum_prediction",
            "maximum_prediction"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# AGE > 12
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "AGE > 12 MONTHS VOLUME DIAGNOSTIC"
)

print(
    "=" * 80
)

print(
    volume_group[
        volume_group["period_group"]
        == "AGE_GT_12"
    ][
        [
            "model",
            "n",
            "mae",
            "rmse",
            "r2",
            "mape_percent",
            "negative_volume_count",
            "mean_actual",
            "mean_prediction"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# COMPLETION
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "STEP 21P COMPLETED"
)

print(
    "=" * 80
)

print(
    "Global :",
    OUTPUT_GLOBAL
)

print(
    "Groups :",
    OUTPUT_GROUP
)

print(
    "Age    :",
    OUTPUT_AGE
)

print(
    "Summary:",
    OUTPUT_SUMMARY
)

print(
    "=" * 80
)