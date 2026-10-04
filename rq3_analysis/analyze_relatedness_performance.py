import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# ============================================================
# 1. LOAD RESULTS
# ============================================================

auc = pd.read_csv(
    "rq3_analysis/rq3_auc_summary_with_ci.csv"
)

diag = pd.read_csv(
    "rq3_analysis/bcpar_diagnostics_summary.csv"
)

# Make sure shift is numeric
auc["shift"] = pd.to_numeric(auc["shift"])
diag["shift"] = pd.to_numeric(diag["shift"])


# ============================================================
# 2. CREATE PERFORMANCE TABLE
# ============================================================

performance = (
    auc.pivot(
        index=["environment", "shift"],
        columns="algorithm",
        values="mean"
    )
    .reset_index()
)

# Rename for clarity
performance = performance.rename(
    columns={
        "BC_PAR": "bcpar_auc",
        "BC_SAC": "bcsac_auc",
        "RLPD": "rlpd_auc"
    }
)

# BC-PAR improvement relative to each baseline
performance["bcpar_minus_bcsac"] = (
    performance["bcpar_auc"]
    - performance["bcsac_auc"]
)

performance["bcpar_minus_rlpd"] = (
    performance["bcpar_auc"]
    - performance["rlpd_auc"]
)

# Improvement relative to strongest baseline
performance["best_baseline_auc"] = performance[
    ["bcsac_auc", "rlpd_auc"]
].max(axis=1)

performance["bcpar_advantage"] = (
    performance["bcpar_auc"]
    - performance["best_baseline_auc"]
)


# ============================================================
# 3. MERGE WITH BC-PAR DIAGNOSTICS
# ============================================================

data = pd.merge(
    performance,
    diag,
    on=["environment", "shift"],
    how="inner"
)


# ============================================================
# 4. HOPPER DUPLICATE HANDLING
# ============================================================

# Hopper 0.1 and 0.5 were experimentally verified to have
# identical effective floor-foot friction and identical results.
#
# Keep both in the full descriptive table, but create a second
# analysis dataset where Hopper 0.5 is removed so that the same
# effective condition is not counted twice.

deduplicated = data[
    ~(
        (data["environment"] == "hopper-friction")
        & (np.isclose(data["shift"], 0.5))
    )
].copy()


# ============================================================
# 5. CORRELATION FUNCTION
# ============================================================

def correlation_report(df, label):

    print("\n================================================")
    print(label)
    print("================================================")

    relationships = [
        ("mean_distance_mean", "bcpar_auc"),
        ("mean_distance_mean", "bcpar_minus_bcsac"),
        ("mean_distance_mean", "bcpar_minus_rlpd"),
        ("mean_distance_mean", "bcpar_advantage"),
        ("final_distance_mean", "bcpar_auc"),
        ("final_distance_mean", "bcpar_advantage"),
        ("mean_src_reward_mean", "bcpar_auc"),
        ("mean_src_reward_mean", "bcpar_advantage")
    ]

    rows = []

    for x, y in relationships:

        temp = df[[x, y]].dropna()

        pearson_r, pearson_p = pearsonr(
            temp[x],
            temp[y]
        )

        spearman_r, spearman_p = spearmanr(
            temp[x],
            temp[y]
        )

        rows.append({
            "x": x,
            "y": y,
            "n": len(temp),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_r,
            "spearman_p": spearman_p
        })

    result = pd.DataFrame(rows)

    print(result.to_string(index=False))

    return result


# ============================================================
# 6. RUN BOTH ANALYSES
# ============================================================

full_corr = correlation_report(
    data,
    "ALL 16 NOMINAL CONDITIONS"
)

dedup_corr = correlation_report(
    deduplicated,
    "15 EFFECTIVE CONDITIONS - HOPPER 0.5 REMOVED"
)


# ============================================================
# 7. SAVE CONDITION TABLE
# ============================================================

columns_to_show = [
    "environment",
    "shift",
    "mean_distance_mean",
    "final_distance_mean",
    "mean_src_reward_mean",
    "bcpar_auc",
    "bcsac_auc",
    "rlpd_auc",
    "bcpar_minus_bcsac",
    "bcpar_minus_rlpd",
    "bcpar_advantage"
]

condition_table = data[
    columns_to_show
].copy()

condition_table.to_csv(
    "rq3_analysis/"
    "rq3_relatedness_performance.csv",
    index=False
)

full_corr.to_csv(
    "rq3_analysis/"
    "rq3_relatedness_correlations_all16.csv",
    index=False
)

dedup_corr.to_csv(
    "rq3_analysis/"
    "rq3_relatedness_correlations_effective15.csv",
    index=False
)


# ============================================================
# 8. PRINT CONDITION TABLE
# ============================================================

print("\n================================================")
print("CONDITION-LEVEL RESULTS")
print("================================================")

print(
    condition_table.to_string(
        index=False
    )
)

print("\nSaved:")
print("rq3_analysis/rq3_relatedness_performance.csv")
print("rq3_analysis/rq3_relatedness_correlations_all16.csv")
print("rq3_analysis/rq3_relatedness_correlations_effective15.csv")