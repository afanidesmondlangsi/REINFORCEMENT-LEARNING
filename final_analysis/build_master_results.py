import os
import pandas as pd
import numpy as np

# ============================================================
# FINAL CONSOLIDATION
#
# RQ1:
#   Does offline source data reduce target exploration?
#
# RQ2:
#   What generalizes across tasks?
#
# RQ3:
#   Does relatedness-aware adaptation improve target transfer?
# ============================================================

OUTPUT_DIR = "final_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD VERIFIED RESULT TABLES
# ============================================================

rq1_summary = pd.read_csv(
    "rq1_cross_task_summary.csv"
)

rq1_transfer = pd.read_csv(
    "rq1_cross_task_transfer.csv"
)

rq1_consistency = pd.read_csv(
    "rq1_cross_task_consistency.csv"
)

rq2_summary = pd.read_csv(
    "rq2_cross_task_summary.csv"
)

rq2_transfer = pd.read_csv(
    "rq2_cross_task_transfer.csv"
)

rq2_consistency = pd.read_csv(
    "rq2_cross_task_consistency.csv"
)

rq2_shift_consistency = pd.read_csv(
    "rq2_cross_task_shift_consistency.csv"
)

rq3_auc = pd.read_csv(
    "rq3_analysis/rq3_auc_summary_with_ci.csv"
)

rq3_relatedness = pd.read_csv(
    "rq3_analysis/"
    "rq3_within_environment_z_correlations.csv"
)


# ============================================================
# 2. STANDARDIZE ENVIRONMENT NAMES
# ============================================================

environment_map = {
    "Hopper": "Hopper",
    "HalfCheetah": "HalfCheetah",
    "Walker2d": "Walker2d",
    "Ant": "Ant",
    "hopper-friction": "Hopper",
    "halfcheetah-friction": "HalfCheetah",
    "walker2d-friction": "Walker2d",
    "ant-friction": "Ant"
}


def standardize_environment(series):
    return series.map(environment_map).fillna(series)


rq1_summary["environment"] = standardize_environment(
    rq1_summary["environment"]
)

rq1_transfer["environment"] = standardize_environment(
    rq1_transfer["environment"]
)

rq2_summary["task"] = standardize_environment(
    rq2_summary["task"]
)

rq2_transfer["task"] = standardize_environment(
    rq2_transfer["task"]
)

rq3_auc["environment"] = standardize_environment(
    rq3_auc["environment"]
)


# ============================================================
# 3. RQ1 MASTER TABLE
# ============================================================

rq1_master = rq1_transfer.copy()

rq1_master = rq1_master[
    [
        "environment",
        "algorithm",
        "delta_score_1000",
        "delta_score_2000",
        "delta_auc",
        "delta_average_performance",
        "early_2k_direction",
        "auc_direction",
        "average_direction"
    ]
]

for column in [
    "delta_score_1000",
    "delta_score_2000",
    "delta_auc",
    "delta_average_performance"
]:
    rq1_master[column] = pd.to_numeric(
        rq1_master[column]
    )

rq1_master.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "master_rq1.csv"
    ),
    index=False
)


# ============================================================
# 4. RQ2 MASTER TABLE
# ============================================================

rq2_master = rq2_transfer.copy()

rq2_master = rq2_master.rename(
    columns={
        "task": "environment"
    }
)

for column in [
    "shift",
    "method_performance",
    "target_only_performance",
    "delta_vs_target_only"
]:
    rq2_master[column] = pd.to_numeric(
        rq2_master[column]
    )

rq2_master.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "master_rq2.csv"
    ),
    index=False
)


# ============================================================
# 5. RQ3 MASTER TABLE
# ============================================================

rq3_master = rq3_auc.copy()

rq3_master["shift"] = pd.to_numeric(
    rq3_master["shift"]
)

# Identify the mean AUC column safely.
possible_mean_columns = [
    "mean",
    "auc_mean",
    "normalized_auc_mean",
    "normalized_auc_5000_10000_mean"
]

rq3_mean_column = None

for column in possible_mean_columns:
    if column in rq3_master.columns:
        rq3_mean_column = column
        break

if rq3_mean_column is None:
    raise ValueError(
        "Could not identify RQ3 AUC mean column.\n"
        f"Available columns: {list(rq3_master.columns)}"
    )

print(
    "\nRQ3 mean column:",
    rq3_mean_column
)


# ============================================================
# 6. DETERMINE RQ3 WINNER FOR EACH CONDITION
# ============================================================

winner_indices = (
    rq3_master.groupby(
        ["environment", "shift"]
    )[rq3_mean_column]
    .idxmax()
)

rq3_winners = rq3_master.loc[
    winner_indices,
    [
        "environment",
        "shift",
        "algorithm",
        rq3_mean_column
    ]
].copy()

rq3_winners = rq3_winners.rename(
    columns={
        "algorithm": "winner",
        rq3_mean_column: "winning_auc"
    }
)

rq3_master = rq3_master.merge(
    rq3_winners[
        [
            "environment",
            "shift",
            "winner"
        ]
    ],
    on=[
        "environment",
        "shift"
    ],
    how="left"
)

rq3_master.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "master_rq3.csv"
    ),
    index=False
)


# ============================================================
# 7. RQ3 WIN COUNTS
# ============================================================

rq3_win_counts = (
    rq3_winners["winner"]
    .value_counts()
    .rename_axis("algorithm")
    .reset_index(name="wins")
)

rq3_win_counts["total_conditions"] = len(
    rq3_winners
)

rq3_win_counts["win_percent"] = (
    100
    * rq3_win_counts["wins"]
    / rq3_win_counts["total_conditions"]
)

rq3_win_counts.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rq3_final_win_counts.csv"
    ),
    index=False
)


# ============================================================
# 8. RQ2 CONSISTENCY TABLE
# ============================================================

rq2_consistency.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rq2_final_consistency.csv"
    ),
    index=False
)

rq2_shift_consistency.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rq2_final_shift_consistency.csv"
    ),
    index=False
)


# ============================================================
# 9. RQ1 CONSISTENCY TABLE
# ============================================================

rq1_consistency.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rq1_final_consistency.csv"
    ),
    index=False
)


# ============================================================
# 10. CREATE CROSS-RQ TABLE FOR COMMON SHIFTS
#
# RQ2 and RQ3 overlap at shifts 0.5 and 2.0.
# This table allows direct comparison without pretending
# RQ2 tested RQ3's additional shifts 0.1 and 5.0.
# ============================================================

rq2_bc = rq2_master[
    rq2_master["algorithm"] == "BC_SAC"
][
    [
        "environment",
        "shift",
        "delta_vs_target_only"
    ]
].copy()

rq2_bc = rq2_bc.rename(
    columns={
        "delta_vs_target_only":
        "rq2_bcsac_delta_vs_target_only"
    }
)


rq3_pivot = rq3_master.pivot_table(
    index=[
        "environment",
        "shift"
    ],
    columns="algorithm",
    values=rq3_mean_column,
    aggfunc="first"
).reset_index()

rq3_pivot.columns.name = None

rename_algorithms = {
    "BC_SAC": "rq3_bcsac_auc",
    "RLPD": "rq3_rlpd_auc",
    "BC_PAR": "rq3_bcpar_auc"
}

rq3_pivot = rq3_pivot.rename(
    columns=rename_algorithms
)

rq3_pivot = rq3_pivot.merge(
    rq3_winners[
        [
            "environment",
            "shift",
            "winner"
        ]
    ],
    on=[
        "environment",
        "shift"
    ],
    how="left"
)

common_master = rq2_bc.merge(
    rq3_pivot,
    on=[
        "environment",
        "shift"
    ],
    how="inner"
)

common_master = common_master[
    common_master["shift"].isin(
        [0.5, 2.0]
    )
].copy()

common_master.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "master_common_conditions.csv"
    ),
    index=False
)


# ============================================================
# 11. OVERALL RESEARCH-QUESTION SUMMARY
# ============================================================

rq1_bcsac = rq1_master[
    rq1_master["algorithm"] == "BC_SAC"
]

rq1_positive_auc = (
    rq1_bcsac["delta_auc"] > 0
).sum()

rq1_total = len(rq1_bcsac)

rq1_mean_auc_gain = (
    rq1_bcsac["delta_auc"].mean()
)


rq2_bcsac = rq2_master[
    rq2_master["algorithm"] == "BC_SAC"
]

rq2_positive = (
    rq2_bcsac["delta_vs_target_only"] > 0
).sum()

rq2_total = len(rq2_bcsac)

rq2_mean_gain = (
    rq2_bcsac["delta_vs_target_only"].mean()
)


rq3_bcpar_wins = (
    rq3_winners["winner"] == "BC_PAR"
).sum()

rq3_bcsac_wins = (
    rq3_winners["winner"] == "BC_SAC"
).sum()

rq3_rlpd_wins = (
    rq3_winners["winner"] == "RLPD"
).sum()


# ============================================================
# 12. FIND FINAL RQ3 RELATEDNESS CORRELATION
# ============================================================

relatedness_row = rq3_relatedness[
    (
        rq3_relatedness["x"]
        == "mean_distance_mean_z"
    )
    &
    (
        rq3_relatedness["y"]
        == "bcpar_advantage_z"
    )
]

if len(relatedness_row) != 1:
    raise ValueError(
        "Could not uniquely identify the final "
        "RQ3 distance-vs-advantage correlation."
    )

rq3_pearson_r = float(
    relatedness_row["pearson_r"].iloc[0]
)

rq3_pearson_p = float(
    relatedness_row["pearson_p"].iloc[0]
)

rq3_spearman_rho = float(
    relatedness_row["spearman_rho"].iloc[0]
)

rq3_spearman_p = float(
    relatedness_row["spearman_p"].iloc[0]
)


overall_summary = pd.DataFrame(
    [
        {
            "research_question": "RQ1",
            "main_test":
                "Offline source data vs target-only learning",
            "primary_result":
                (
                    f"BC-SAC positive AUC transfer in "
                    f"{rq1_positive_auc}/{rq1_total} environments"
                ),
            "key_value":
                rq1_mean_auc_gain,
            "interpretation":
                (
                    "Offline source data were useful overall "
                    "under BC-SAC, while early benefits were "
                    "environment-dependent."
                )
        },
        {
            "research_question": "RQ2",
            "main_test":
                (
                    "Cross-environment transfer at "
                    "friction shifts 0.5 and 2.0"
                ),
            "primary_result":
                (
                    f"BC-SAC positive transfer in "
                    f"{rq2_positive}/{rq2_total} conditions"
                ),
            "key_value":
                rq2_mean_gain,
            "interpretation":
                (
                    "BC-SAC showed the most consistent "
                    "cross-task generalization."
                )
        },
        {
            "research_question": "RQ3",
            "main_test":
                (
                    "Relatedness-aware adaptation across "
                    "four environments and four nominal shifts"
                ),
            "primary_result":
                (
                    f"BC-SAC {rq3_bcsac_wins} wins; "
                    f"BC-PAR {rq3_bcpar_wins}; "
                    f"RLPD {rq3_rlpd_wins}"
                ),
            "key_value":
                rq3_pearson_r,
            "interpretation":
                (
                    "BC-PAR was explicitly relatedness-aware "
                    "but its learned distance did not reliably "
                    "predict performance advantage."
                )
        }
    ]
)

overall_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "overall_research_summary.csv"
    ),
    index=False
)


# ============================================================
# 13. PRINT FINAL RESULTS
# ============================================================

print("\n================================================")
print("RQ1 FINAL")
print("================================================")

print(
    f"BC-SAC positive AUC transfer: "
    f"{rq1_positive_auc}/{rq1_total}"
)

print(
    f"Mean BC-SAC AUC gain: "
    f"{rq1_mean_auc_gain:.6f}"
)


print("\n================================================")
print("RQ2 FINAL")
print("================================================")

print(
    f"BC-SAC positive conditions: "
    f"{rq2_positive}/{rq2_total}"
)

print(
    f"Mean BC-SAC improvement: "
    f"{rq2_mean_gain:.6f}"
)


print("\n================================================")
print("RQ3 FINAL")
print("================================================")

print(
    f"BC-SAC wins: {rq3_bcsac_wins}"
)

print(
    f"BC-PAR wins: {rq3_bcpar_wins}"
)

print(
    f"RLPD wins: {rq3_rlpd_wins}"
)

print(
    f"Distance vs advantage Pearson: "
    f"r={rq3_pearson_r:.6f}, "
    f"p={rq3_pearson_p:.6f}"
)

print(
    f"Distance vs advantage Spearman: "
    f"rho={rq3_spearman_rho:.6f}, "
    f"p={rq3_spearman_p:.6f}"
)


print("\n================================================")
print("COMMON RQ2/RQ3 CONDITIONS")
print("================================================")

print(
    common_master.to_string(
        index=False
    )
)


print("\n================================================")
print("OVERALL SUMMARY")
print("================================================")

print(
    overall_summary.to_string(
        index=False
    )
)


print("\nSaved files:")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):
    if filename.endswith(".csv"):
        print(
            os.path.join(
                OUTPUT_DIR,
                filename
            )
        )