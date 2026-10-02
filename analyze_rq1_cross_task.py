import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# RQ1 CROSS-TASK ANALYSIS
#
# Research Question:
# How can offline data from source tasks reduce the exploration
# needed in a target task?
#
# Environments:
#   Hopper
#   HalfCheetah
#   Walker2d
#   Ant
#
# Methods:
#   SAC_TARGET_ONLY
#   BC_SAC
#   RLPD
#
# Friction shift:
#   0.1
#
# Seeds:
#   0, 1, 2
# ============================================================


# ============================================================
# 1. INPUT FILES
# ============================================================

FILES = {
    "Hopper": {
        "summary": "rq1_hopper_summary.csv",
        "transfer": "rq1_hopper_transfer.csv",
    },

    "HalfCheetah": {
        "summary": "rq1_halfcheetah_summary.csv",
        "transfer": "rq1_halfcheetah_transfer.csv",
    },

    "Walker2d": {
        "summary": "rq1_walker2d_summary.csv",
        "transfer": "rq1_walker2d_transfer.csv",
    },

    "Ant": {
        "summary": "rq1_ant_summary.csv",
        "transfer": "rq1_ant_transfer.csv",
    },
}


# ============================================================
# 2. COMBINE SUMMARY FILES
# ============================================================

summary_frames = []

for environment, paths in FILES.items():

    df = pd.read_csv(paths["summary"])

    df.insert(
        0,
        "environment",
        environment
    )

    summary_frames.append(df)


cross_task_summary = pd.concat(
    summary_frames,
    ignore_index=True
)


# ============================================================
# 3. COMBINE TRANSFER FILES
# ============================================================

transfer_frames = []

for environment, paths in FILES.items():

    df = pd.read_csv(paths["transfer"])

    df.insert(
        0,
        "environment",
        environment
    )

    transfer_frames.append(df)


cross_task_transfer = pd.concat(
    transfer_frames,
    ignore_index=True
)


# ============================================================
# 4. ADD DIRECTION OF TRANSFER
#
# Positive delta:
# source-data method performed better than target-only SAC.
#
# Negative delta:
# source-data method performed worse than target-only SAC.
#
# We examine 2,000 interactions separately from AUC because
# RQ1 is specifically concerned with reducing target
# exploration.
# ============================================================

cross_task_transfer["early_2k_direction"] = (
    cross_task_transfer["delta_score_2000"]
    .apply(
        lambda x:
        "positive" if x > 0
        else "negative" if x < 0
        else "equal"
    )
)


cross_task_transfer["auc_direction"] = (
    cross_task_transfer["delta_auc"]
    .apply(
        lambda x:
        "positive" if x > 0
        else "negative" if x < 0
        else "equal"
    )
)


cross_task_transfer["average_direction"] = (
    cross_task_transfer["delta_average_performance"]
    .apply(
        lambda x:
        "positive" if x > 0
        else "negative" if x < 0
        else "equal"
    )
)


# ============================================================
# 5. CONSISTENCY ACROSS TASKS
# ============================================================

consistency_rows = []

for algorithm in ["BC_SAC", "RLPD"]:

    group = cross_task_transfer[
        cross_task_transfer["algorithm"] == algorithm
    ]

    consistency_rows.append(
        {
            "algorithm": algorithm,

            "tasks": len(group),

            "positive_at_1000":
                int(
                    (
                        group["delta_score_1000"] > 0
                    ).sum()
                ),

            "positive_at_2000":
                int(
                    (
                        group["delta_score_2000"] > 0
                    ).sum()
                ),

            "positive_auc":
                int(
                    (
                        group["delta_auc"] > 0
                    ).sum()
                ),

            "positive_average_performance":
                int(
                    (
                        group[
                            "delta_average_performance"
                        ] > 0
                    ).sum()
                ),

            "mean_delta_1000":
                group[
                    "delta_score_1000"
                ].mean(),

            "mean_delta_2000":
                group[
                    "delta_score_2000"
                ].mean(),

            "mean_delta_auc":
                group[
                    "delta_auc"
                ].mean(),

            "mean_delta_average_performance":
                group[
                    "delta_average_performance"
                ].mean(),
        }
    )


consistency = pd.DataFrame(
    consistency_rows
)


# ============================================================
# 6. PRINT COMBINED SUMMARY
# ============================================================

print()
print("=" * 80)
print("RQ1 CROSS-TASK SUMMARY")
print("=" * 80)
print()

print(
    cross_task_summary.to_string(
        index=False
    )
)


# ============================================================
# 7. PRINT TRANSFER RESULTS
# ============================================================

print()
print("=" * 80)
print("RQ1 TRANSFER RELATIVE TO SAC_TARGET_ONLY")
print("=" * 80)
print()

print(
    cross_task_transfer.to_string(
        index=False
    )
)


# ============================================================
# 8. PRINT CROSS-TASK CONSISTENCY
# ============================================================

print()
print("=" * 80)
print("RQ1 CROSS-TASK CONSISTENCY")
print("=" * 80)
print()

print(
    consistency.to_string(
        index=False
    )
)


# ============================================================
# 9. SAVE CSV FILES
# ============================================================

cross_task_summary.to_csv(
    "rq1_cross_task_summary.csv",
    index=False
)

cross_task_transfer.to_csv(
    "rq1_cross_task_transfer.csv",
    index=False
)

consistency.to_csv(
    "rq1_cross_task_consistency.csv",
    index=False
)


# ============================================================
# 10. GRAPH 1
#
# Difference in performance at 2,000 target interactions.
#
# This is especially important for RQ1 because a positive
# value means the source-data method achieved a higher score
# after the same small target-interaction budget.
# ============================================================

plot_2k = cross_task_transfer.pivot(
    index="environment",
    columns="algorithm",
    values="delta_score_2000"
)

ax = plot_2k.plot(
    kind="bar",
    figsize=(9, 6)
)

ax.axhline(
    0,
    linewidth=1
)

ax.set_title(
    "RQ1: Transfer Effect After 2,000 Target Interactions"
)

ax.set_xlabel(
    "Target Environment"
)

ax.set_ylabel(
    "Score Difference vs SAC_TARGET_ONLY"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "rq1_cross_task_early_2000.png",
    dpi=300
)

plt.close()


# ============================================================
# 11. GRAPH 2
#
# Difference in average performance across 1,000-5,000
# target interactions.
# ============================================================

plot_average = cross_task_transfer.pivot(
    index="environment",
    columns="algorithm",
    values="delta_average_performance"
)

ax = plot_average.plot(
    kind="bar",
    figsize=(9, 6)
)

ax.axhline(
    0,
    linewidth=1
)

ax.set_title(
    "RQ1: Transfer Effect Across 1,000-5,000 Target Interactions"
)

ax.set_xlabel(
    "Target Environment"
)

ax.set_ylabel(
    "Average Performance Difference vs SAC_TARGET_ONLY"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "rq1_cross_task_average_performance.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. FINISHED
# ============================================================

print()
print("=" * 80)
print("FILES CREATED")
print("=" * 80)

print("rq1_cross_task_summary.csv")
print("rq1_cross_task_transfer.csv")
print("rq1_cross_task_consistency.csv")
print("rq1_cross_task_early_2000.png")
print("rq1_cross_task_average_performance.png")

print()
print("RQ1 cross-task analysis complete.")