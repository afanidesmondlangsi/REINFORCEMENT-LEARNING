# ============================================================
# RQ2 CROSS-TASK GENERALIZATION ANALYSIS
#
# Research Question 2:
# "What can be generalizable across tasks?"
#
# Tasks:
#   1. Hopper
#   2. HalfCheetah
#   3. Walker2d
#   4. Ant
#
# Algorithms:
#   SAC_TARGET_ONLY
#   BC_SAC
#   RLPD
#
# Friction shifts:
#   0.5
#   2.0
#
# Seeds:
#   0, 1, 2
#
# Primary metric:
#   Average performance = AUC / 4000
#
# Transfer metric:
#   Delta = method performance - target-only performance
#
# IMPORTANT:
# We use absolute transfer difference (Delta) as the main
# cross-task transfer measure because percentage improvement
# is unstable when the target-only baseline is near zero
# or negative.
# ============================================================


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. INPUT FILES
# ============================================================

files = {
    "Hopper": "rq2_auc_by_seed.csv",
    "HalfCheetah": "rq2_halfcheetah_auc_by_seed.csv",
    "Walker2d": "rq2_walker2d_auc_by_seed.csv",
    "Ant": "rq2_ant_auc_by_seed.csv",
}


# ============================================================
# 2. CHECK THAT ALL INPUT FILES EXIST
# ============================================================

print("\n" + "=" * 70)
print("CHECKING INPUT FILES")
print("=" * 70)

for task, filename in files.items():

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Missing file for {task}: {filename}"
        )

    print(f"{task:<15} -> {filename}  [FOUND]")


# ============================================================
# 3. LOAD ALL FOUR TASKS
# ============================================================

all_data = []

for task, filename in files.items():

    df = pd.read_csv(filename)

    # Add task name
    df["task"] = task

    all_data.append(df)


# Combine all data
data = pd.concat(all_data, ignore_index=True)


# ============================================================
# 4. INSPECT COLUMN NAMES
# ============================================================

print("\n" + "=" * 70)
print("COLUMNS FOUND")
print("=" * 70)

print(data.columns.tolist())


# ============================================================
# 5. STANDARDIZE IMPORTANT COLUMN NAMES
# ============================================================

# The previous analysis scripts should already use these names.
# These checks make the script slightly more robust.

if "average_performance" not in data.columns:

    if "avg_performance" in data.columns:

        data = data.rename(
            columns={"avg_performance": "average_performance"}
        )

    elif "auc" in data.columns:

        # Evaluations are from step 1000 to 5000.
        # Width = 4000 environment steps.
        data["average_performance"] = data["auc"] / 4000.0

    else:

        raise ValueError(
            "Could not find average_performance or auc column."
        )


# ============================================================
# 6. BASIC DATA CHECK
# ============================================================

print("\n" + "=" * 70)
print("RQ2 CROSS-TASK DATA CHECK")
print("=" * 70)

run_counts = (
    data.groupby(
        ["task", "algorithm", "shift"]
    )["seed"]
    .nunique()
)

print(run_counts)

print("\nTotal seed-level rows:", len(data))


# Expected:
#
# 4 tasks
# x 3 algorithms
# x 2 shifts
# x 3 seeds
#
# = 72 rows

expected_rows = 72

if len(data) != expected_rows:

    print(
        f"\nWARNING: Expected {expected_rows} rows "
        f"but found {len(data)}."
    )

else:

    print("\nCorrect number of seed-level results: 72")


# Check every condition has exactly 3 seeds

bad_counts = run_counts[run_counts != 3]

if len(bad_counts) > 0:

    print("\nWARNING: Some conditions do not have 3 seeds:")
    print(bad_counts)

else:

    print("Every task/algorithm/shift condition has 3 seeds.")


# ============================================================
# 7. SAVE COMBINED SEED-LEVEL DATA
# ============================================================

# Put task first for readability

preferred_columns = [
    "task",
    "algorithm",
    "shift",
    "seed",
    "auc",
    "average_performance",
    "final_score",
]

existing_columns = [
    col for col in preferred_columns
    if col in data.columns
]

remaining_columns = [
    col for col in data.columns
    if col not in existing_columns
]

data = data[
    existing_columns + remaining_columns
]


data.to_csv(
    "rq2_cross_task_all_seeds.csv",
    index=False
)


# ============================================================
# 8. CROSS-TASK SUMMARY
# ============================================================

summary = (
    data.groupby(
        ["task", "algorithm", "shift"]
    )
    .agg(
        n=("seed", "nunique"),

        auc_mean=("auc", "mean"),
        auc_sd=("auc", "std"),

        average_performance_mean=(
            "average_performance",
            "mean"
        ),

        average_performance_sd=(
            "average_performance",
            "std"
        ),

        final_score_mean=(
            "final_score",
            "mean"
        ),

        final_score_sd=(
            "final_score",
            "std"
        ),
    )
    .reset_index()
)


summary.to_csv(
    "rq2_cross_task_summary.csv",
    index=False
)


print("\n" + "=" * 70)
print("CROSS-TASK SUMMARY")
print("=" * 70)

print(summary.to_string(index=False))


# ============================================================
# 9. CALCULATE TRANSFER RELATIVE TO TARGET-ONLY
# ============================================================

transfer_rows = []


tasks = [
    "Hopper",
    "HalfCheetah",
    "Walker2d",
    "Ant",
]

shifts = [0.5, 2.0]

transfer_methods = [
    "BC_SAC",
    "RLPD",
]


for task in tasks:

    for shift in shifts:

        # ----------------------------------------------------
        # Get target-only baseline
        # ----------------------------------------------------

        baseline_row = summary[
            (summary["task"] == task)
            &
            (summary["algorithm"] == "SAC_TARGET_ONLY")
            &
            (np.isclose(summary["shift"], shift))
        ]

        if len(baseline_row) != 1:

            raise ValueError(
                f"Could not uniquely identify target-only "
                f"baseline for {task}, shift={shift}"
            )


        baseline = baseline_row[
            "average_performance_mean"
        ].iloc[0]


        # ----------------------------------------------------
        # Compare BC_SAC and RLPD with baseline
        # ----------------------------------------------------

        for method in transfer_methods:

            method_row = summary[
                (summary["task"] == task)
                &
                (summary["algorithm"] == method)
                &
                (np.isclose(summary["shift"], shift))
            ]

            if len(method_row) != 1:

                raise ValueError(
                    f"Could not uniquely identify {method} "
                    f"for {task}, shift={shift}"
                )


            method_performance = method_row[
                "average_performance_mean"
            ].iloc[0]


            # Absolute transfer difference
            delta = (
                method_performance
                - baseline
            )


            # ------------------------------------------------
            # Percentage gain
            #
            # Only calculate when baseline is clearly positive.
            #
            # If baseline <= 0, percentage improvement can be
            # misleading, so return NaN.
            # ------------------------------------------------

            if baseline > 0:

                percent_gain = (
                    delta / baseline
                ) * 100.0

            else:

                percent_gain = np.nan


            if delta > 0:

                direction = "Positive"

            elif delta < 0:

                direction = "Negative"

            else:

                direction = "No difference"


            transfer_rows.append(
                {
                    "task": task,
                    "shift": shift,
                    "algorithm": method,

                    "method_performance":
                        method_performance,

                    "target_only_performance":
                        baseline,

                    "delta_vs_target_only":
                        delta,

                    "transfer_gain_percent":
                        percent_gain,

                    "transfer_direction":
                        direction,
                }
            )


transfer = pd.DataFrame(transfer_rows)


transfer.to_csv(
    "rq2_cross_task_transfer.csv",
    index=False
)


print("\n" + "=" * 70)
print("TRANSFER RELATIVE TO SAC_TARGET_ONLY")
print("=" * 70)

print(
    transfer.to_string(
        index=False
    )
)


# ============================================================
# 10. CONSISTENCY ACROSS TASK x SHIFT CONDITIONS
# ============================================================

consistency_rows = []


for method in transfer_methods:

    method_transfer = transfer[
        transfer["algorithm"] == method
    ]


    positive_count = (
        method_transfer[
            "delta_vs_target_only"
        ] > 0
    ).sum()


    negative_count = (
        method_transfer[
            "delta_vs_target_only"
        ] < 0
    ).sum()


    zero_count = (
        method_transfer[
            "delta_vs_target_only"
        ] == 0
    ).sum()


    total_conditions = len(
        method_transfer
    )


    mean_delta = method_transfer[
        "delta_vs_target_only"
    ].mean()


    median_delta = method_transfer[
        "delta_vs_target_only"
    ].median()


    consistency_rows.append(
        {
            "algorithm": method,

            "positive_conditions":
                positive_count,

            "negative_conditions":
                negative_count,

            "zero_conditions":
                zero_count,

            "total_conditions":
                total_conditions,

            "positive_fraction":
                positive_count / total_conditions,

            "mean_delta":
                mean_delta,

            "median_delta":
                median_delta,
        }
    )


consistency = pd.DataFrame(
    consistency_rows
)


consistency.to_csv(
    "rq2_cross_task_consistency.csv",
    index=False
)


print("\n" + "=" * 70)
print("CROSS-TASK TRANSFER CONSISTENCY")
print("=" * 70)

print(
    consistency.to_string(
        index=False
    )
)


# ============================================================
# 11. DISPLAY DELTA TABLE
# ============================================================

delta_table = transfer.pivot_table(
    index=[
        "task",
        "shift"
    ],

    columns="algorithm",

    values="delta_vs_target_only"
)


print("\n" + "=" * 70)
print("ABSOLUTE TRANSFER DIFFERENCE")
print("Delta = Method - SAC_TARGET_ONLY")
print("=" * 70)

print(delta_table)


# ============================================================
# 12. CROSS-TASK TRANSFER FIGURE
# ============================================================

# We plot Delta rather than raw task performance.
#
# Positive values:
#     transfer method exceeded target-only.
#
# Negative values:
#     transfer method was below target-only.


plot_data = transfer.copy()


plot_data["condition"] = (
    plot_data["task"]
    + "\nshift="
    + plot_data["shift"].astype(str)
)


conditions = (
    plot_data["condition"]
    .drop_duplicates()
    .tolist()
)


x = np.arange(
    len(conditions)
)


width = 0.35


bc_values = []

rlpd_values = []


for condition in conditions:

    bc_value = plot_data[
        (plot_data["condition"] == condition)
        &
        (plot_data["algorithm"] == "BC_SAC")
    ]["delta_vs_target_only"].iloc[0]


    rlpd_value = plot_data[
        (plot_data["condition"] == condition)
        &
        (plot_data["algorithm"] == "RLPD")
    ]["delta_vs_target_only"].iloc[0]


    bc_values.append(
        bc_value
    )

    rlpd_values.append(
        rlpd_value
    )


plt.figure(
    figsize=(14, 7)
)


plt.bar(
    x - width / 2,
    bc_values,
    width,
    label="BC_SAC"
)


plt.bar(
    x + width / 2,
    rlpd_values,
    width,
    label="RLPD"
)


plt.axhline(
    y=0,
    linewidth=1
)


plt.xticks(
    x,
    conditions,
    rotation=45,
    ha="right"
)


plt.ylabel(
    "Average Performance Difference vs SAC_TARGET_ONLY"
)


plt.xlabel(
    "Task and Friction Shift"
)


plt.title(
    "RQ2 Cross-Task Transfer Relative to Target-Only Learning"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    "rq2_cross_task_transfer.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 13. OPTIONAL: SHIFT-SPECIFIC CONSISTENCY
# ============================================================

shift_consistency = (
    transfer.groupby(
        ["algorithm", "shift"]
    )
    .agg(
        mean_delta=(
            "delta_vs_target_only",
            "mean"
        ),

        median_delta=(
            "delta_vs_target_only",
            "median"
        ),

        positive_conditions=(
            "delta_vs_target_only",
            lambda x: (x > 0).sum()
        ),

        total_conditions=(
            "delta_vs_target_only",
            "size"
        ),
    )
    .reset_index()
)


shift_consistency[
    "positive_fraction"
] = (
    shift_consistency[
        "positive_conditions"
    ]
    /
    shift_consistency[
        "total_conditions"
    ]
)


shift_consistency.to_csv(
    "rq2_cross_task_shift_consistency.csv",
    index=False
)


print("\n" + "=" * 70)
print("CONSISTENCY BY SHIFT")
print("=" * 70)

print(
    shift_consistency.to_string(
        index=False
    )
)


# ============================================================
# 14. FINAL FILE LIST
# ============================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

output_files = [
    "rq2_cross_task_all_seeds.csv",
    "rq2_cross_task_summary.csv",
    "rq2_cross_task_transfer.csv",
    "rq2_cross_task_consistency.csv",
    "rq2_cross_task_shift_consistency.csv",
    "rq2_cross_task_transfer.png",
]


for filename in output_files:

    print(filename)


print("\n" + "=" * 70)
print("RQ2 CROSS-TASK ANALYSIS COMPLETE")
print("=" * 70)